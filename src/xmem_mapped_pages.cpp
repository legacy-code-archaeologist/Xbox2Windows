#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "era_physical.h"
#include "xmem_slab_slots.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "startup_stubs/local_log.h"

#include <vector>

#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>

namespace {
struct Backing {
    HANDLE section;
    ULONG_PTR views = 0;
    ULONG64 offset = 0;  // the page is one 64 KiB slot of a shared section (see section_for)
    std::size_t slab = 0;  // index of that section among the slabs
};

// Keep grant transitions and section views serialized. A grant can outlive all its
// views so its bytes survive unmapping. Its backing is released before reuse.
struct PhysicalMemory {
    std::mutex mutex;
    std::map<ULONG_PTR, Backing> backings;
    std::map<ULONG_PTR, ULONG_PTR> views;
    std::map<ULONG_PTR, SIZE_T> reservations;
    // A no-access graphics reservation (protection 0x801) made right after a
    // read-write one of the same size is that memory's GPU view on the console. Pages mapped into
    // the CPU view are mapped at the same offset into the GPU view too (the same section).
    ULONG_PTR last_cpu_base = 0;
    SIZE_T last_cpu_bytes = 0;
    ULONG_PTR mirror_cpu = 0, mirror_gpu = 0;
    SIZE_T mirror_bytes = 0;
    std::map<ULONG_PTR, SIZE_T> runs;  // OS views: base -> bytes
    // The slabs and their 64 KiB slots; released slots are reused lowest first, so that pages reused together
    // are adjacent slots again.
    x2w_xmem::SlabSlots<HANDLE> slots{era_physical::page_bytes};

    ~PhysicalMemory()
    {
        for (const auto& run : runs)
            UnmapViewOfFile(reinterpret_cast<void*>(run.first));
        for (const HANDLE slab : slots.slabs())
            CloseHandle(slab);
    }
};

PhysicalMemory memory;
constexpr SIZE_T page_bytes = era_physical::page_bytes;

BOOL physical_result(era_physical::Status status) noexcept
{
    switch (status) {
    case era_physical::Status::ok:
        SetLastError(ERROR_SUCCESS);
        return TRUE;
    case era_physical::Status::exhausted:
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return FALSE;
    default:
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
}

// Pages are slots of 256 MiB pagefile-backed sections rather than one section each: one handle and
// one view per page is what makes large batches slow. A slab is created with SEC_RESERVE, so it costs
// no commit until a slot is used; a slot is committed here, when its page is first mapped or written,
// and stays committed after it is freed so that reuse cannot fail for lack of commit. A freed slot is
// zeroed when it is reused, as a fresh section was. Fails with the system error (for instance
// ERROR_COMMITMENT_LIMIT) when the slot cannot be committed, leaving no state changed.
bool commit_slot(HANDLE slab, ULONG64 offset, bool zero)
{
#ifdef X2W_XMEM_TEST_CONTROLS
    // Refuse the first commit in a dedicated test process without consuming system commit.
    static bool refuse_commit = x2w_local::env_on("X2W_XMEM_TEST_COMMIT_FAILURE");
    if (refuse_commit) {
        refuse_commit = false;
        SetLastError(ERROR_COMMITMENT_LIMIT);
        return false;
    }
#endif
    void* view = MapViewOfFile(slab, FILE_MAP_WRITE, static_cast<DWORD>(offset >> 32), static_cast<DWORD>(offset), page_bytes);
    if (!view) return false;
    bool ok = VirtualAlloc(view, page_bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
    const DWORD error = ok ? ERROR_SUCCESS : GetLastError();
    if (ok && zero) std::memset(view, 0, page_bytes);  // a reused slot starts zero-filled
    UnmapViewOfFile(view);
    if (!ok) SetLastError(error);
    return ok;
}

HANDLE section_for(ULONG_PTR page)
{
    const auto existing = memory.backings.find(page);
    if (existing != memory.backings.end())
        return existing->second.section;
    x2w_xmem::SlabSlots<HANDLE>::Slot slot;
    if (memory.slots.lowest_free(slot)) {
        if (!commit_slot(slot.slab, slot.offset, true)) return nullptr;
        memory.backings.emplace(page, Backing{slot.slab, 0, slot.offset, slot.slab_index});
        memory.slots.take_lowest_free();
        return slot.slab;
    }
    const ULONG64 slab_bytes = memory.slots.slab_bytes();
    bool first_slot_committed = false;
    const auto make_slab = [slab_bytes, &first_slot_committed](HANDLE& slab) {
        slab = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE,
                                  static_cast<DWORD>(slab_bytes >> 32), static_cast<DWORD>(slab_bytes), nullptr);
        if (!slab) return false;
        first_slot_committed = commit_slot(slab, 0, false);
        if (!first_slot_committed) {
            const DWORD error = GetLastError();
            CloseHandle(slab);
            SetLastError(error);
        }
        return first_slot_committed;
    };
    if (!memory.slots.take_fresh(make_slab, [](HANDLE slab) { CloseHandle(slab); }, slot)) return nullptr;
    if (!first_slot_committed && !commit_slot(slot.slab, slot.offset, false)) {
        const DWORD error = GetLastError();
        memory.slots.unwind_fresh(slot);
        SetLastError(error);
        return nullptr;
    }
    try {
        memory.backings.emplace(page, Backing{slot.slab, 0, slot.offset, slot.slab_index});
    } catch (...) {
        memory.slots.unwind_fresh(slot);
        throw;
    }
    return slot.slab;
}

auto owned_reservation(ULONG_PTR address)
{
    auto it = memory.reservations.upper_bound(address);
    if (it == memory.reservations.begin()) return memory.reservations.end();
    --it;
    return address - it->first < it->second ? it : memory.reservations.end();
}

// Only split a reservation owned by the shim. Caller-owned placeholders must already match the view.
// Once a split succeeds, report the retry's result even if it fails for a different reason.
template <class Map>
bool map_split(void* target, SIZE_T bytes, Map map)
{
    if (map()) return true;
    const DWORD error = GetLastError();
    const auto address = reinterpret_cast<ULONG_PTR>(target);
    const auto reservation = owned_reservation(address);
    if (reservation != memory.reservations.end() && bytes <= reservation->second - (address - reservation->first) &&
        VirtualFree(target, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
        return map();
    SetLastError(error);
    return false;
}

bool map_view(ULONG_PTR address, const Backing& backing)
{
    const HANDLE section = backing.section;
    const ULONG64 section_offset = backing.offset;
    auto* target = reinterpret_cast<void*>(address);
    auto map = [&] {
        return MapViewOfFile3(section, GetCurrentProcess(), target, section_offset,
                              page_bytes, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0) != nullptr;
    };
    return map_split(target, page_bytes, map);
}

// Maps bytes of one section at address as a single view; falls back to one view per page when the
// range is not inside a single placeholder.
bool map_run(ULONG_PTR address, const Backing& backing, SIZE_T bytes, std::vector<std::pair<ULONG_PTR, SIZE_T>>& made)
{
    auto* target = reinterpret_cast<void*>(address);
    auto map = [&] {
        return MapViewOfFile3(backing.section, GetCurrentProcess(), target, backing.offset, bytes,
                              MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0) != nullptr;
    };
    if (map_split(target, bytes, map)) {
        memory.runs[address] = bytes;
        made.emplace_back(address, bytes);
        return true;
    }
    if (bytes == page_bytes) return false;
    for (SIZE_T done = 0; done < bytes; done += page_bytes) {
        Backing one = backing;
        one.offset += done;
        if (!map_view(address + done, one)) return false;
        memory.runs[address + done] = page_bytes;
        made.emplace_back(address + done, page_bytes);
    }
    return true;
}

bool unmap_view(ULONG_PTR address)
{
    return UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(address),
                           MEM_PRESERVE_PLACEHOLDER) != FALSE;
}

void coalesce_unused(const std::map<ULONG_PTR, ULONG_PTR>& batch)
{
    auto previous = memory.reservations.end();
    for (const auto& entry : batch) {
        const auto reservation = owned_reservation(entry.first);
        if (reservation == memory.reservations.end() || reservation == previous) continue;
        previous = reservation;
        const auto [base, size] = *reservation;
        const auto view = memory.views.lower_bound(base);
        if (view == memory.views.end() || view->first - base >= size)
            VirtualFree(reinterpret_cast<void*>(base), size, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
    }
}

// Unmapping a page inside a run unmaps the whole run and maps its other pages again (map_pages).
// A thread that touches one of those pages in between faults. The break records the address
// ranges of the runs it takes apart and a generation that is odd while it is in progress.
struct RemapWindow {
    SRWLOCK lock = SRWLOCK_INIT;
    std::vector<std::pair<ULONG_PTR, SIZE_T>> ranges;  // the runs of the break in progress, or of the last one
    ULONG64 generation = 0;
};
RemapWindow remap_window;

void begin_break(const std::map<ULONG_PTR, SIZE_T>& broken)
{
    AcquireSRWLockExclusive(&remap_window.lock);
    remap_window.ranges.assign(broken.begin(), broken.end());
    ++remap_window.generation;
    ReleaseSRWLockExclusive(&remap_window.lock);
}
void end_break()
{
    AcquireSRWLockExclusive(&remap_window.lock);
    ++remap_window.generation;
    ReleaseSRWLockExclusive(&remap_window.lock);
}

// Retries an access violation only when all of this holds: the address is inside a run that the
// break in progress, or the last finished one, took apart; after that break has finished (the
// mutex) the page is recorded and covered by a view; and this thread has not already retried this
// page for this break. Everything else is passed on, so a fault that is not the transient one is
// reported after at most one retry.
thread_local bool mapping_thread = false;
LONG CALLBACK retry_remapped_page(EXCEPTION_POINTERS* e)
{
    const EXCEPTION_RECORD* record = e->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || mapping_thread)
        return EXCEPTION_CONTINUE_SEARCH;
    const ULONG_PTR address = record->ExceptionInformation[1];
    const ULONG_PTR page = address & ~(ULONG_PTR{page_bytes} - 1);
    bool inside = false;
    AcquireSRWLockShared(&remap_window.lock);
    const ULONG64 seen = remap_window.generation;
    for (const auto& [base, size] : remap_window.ranges)
        inside |= address - base < size;
    ReleaseSRWLockShared(&remap_window.lock);
    if (!inside || seen == 0) return EXCEPTION_CONTINUE_SEARCH;
    const ULONG64 break_id = seen | 1;  // the break in progress, or the one that just finished
    thread_local ULONG_PTR retried_page = 0;
    thread_local ULONG64 retried_break = 0;
    if (retried_page == page && retried_break == break_id) return EXCEPTION_CONTINUE_SEARCH;
    bool mapped = false;
    {
        const std::lock_guard guard(memory.mutex);  // held by the mapping thread until its break is over
        auto run = memory.runs.upper_bound(page);
        mapped = memory.views.count(page) && run != memory.runs.begin() && page - std::prev(run)->first < std::prev(run)->second;
    }
    if (!mapped) return EXCEPTION_CONTINUE_SEARCH;
    retried_page = page;
    retried_break = break_id;
    return EXCEPTION_CONTINUE_EXECUTION;
}
struct MappingThread {
    MappingThread() { mapping_thread = true; }
    ~MappingThread() { mapping_thread = false; }
};

// Shared views are opt-in: X2W_XMEM_SHARED_VIEWS=1, read once by the first mapping call. Without
// it every page has a view of its own, as before: unmapping a page never takes a neighbour away,
// no handler is registered, and the module unloads like any DLL.
//
// With it, the first mapping call registers the handler and pins the module. The handler is
// process-wide and Windows offers no way to remove one that also waits for every dispatch already
// under way, so it is never removed; the pin keeps xmem.dll, the handler's code and this state in
// place until the process ends, also across a host's FreeLibrary and LoadLibrary. Nothing is
// waited for or torn down at DLL detach. If the handler cannot be registered or the module cannot
// be pinned, the opt-in is refused and pages get a view each.
struct FaultRetry {
    SRWLOCK lock = SRWLOCK_INIT;
    bool tried = false;
    bool guaranteed = false;  // handler registered and module pinned: shared views may be created

    bool ready()
    {
        AcquireSRWLockExclusive(&lock);
        if (!tried) {
            tried = true;
#ifdef X2W_XMEM_TEST_CONTROLS
            // Test build only: shared views without the handler, to show that a reader then faults.
            if (x2w_local::env_on("X2W_XMEM_NO_FAULT_RETRY")) {
                guaranteed = true;
                ReleaseSRWLockExclusive(&lock);
                return true;
            }
#endif
            HMODULE pinned = nullptr;
            if (!x2w_local::env_on("X2W_XMEM_SHARED_VIEWS")) {
                // The default: one view per page, nothing process-wide, unloadable.
            } else if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                   reinterpret_cast<LPCWSTR>(&retry_remapped_page), &pinned) &&
                AddVectoredExceptionHandler(1, retry_remapped_page)) {
                guaranteed = true;
                x2w_local::logf("x2w-xmem shared views are on (X2W_XMEM_SHARED_VIEWS): xmem.dll stays loaded until the process ends");
            } else
                x2w_local::logf("x2w-xmem shared views are off: the fault handler could not be set up (error %lu)", GetLastError());
        }
        const bool result = guaranteed;
        ReleaseSRWLockExclusive(&lock);
        return result;
    }
};
FaultRetry fault_retry;

// Test build only (X2W_XMEM_TEST_CONTROLS; tests/xmem_page_runs_check.py): X2W_XMEM_TEST_FAULT=<stage>:<skip> makes one step of
// the next unmap fail as if Windows had refused it, after skipping <skip> such steps. Stage 1 is
// unmapping a run, stage 2 mapping surviving pages again, stage 3 both the unmap and the restore.
#ifdef X2W_XMEM_TEST_CONTROLS
struct TestFault {
    unsigned stage = 0, skip = 0;
    TestFault()
    {
        char text[32];
        if (GetEnvironmentVariableA("X2W_XMEM_TEST_FAULT", text, sizeof(text)) && text[0] >= '1' && text[0] <= '3' && text[1] == ':') {
            stage = static_cast<unsigned>(text[0] - '0');
            skip = static_cast<unsigned>(std::strtoul(text + 2, nullptr, 10));
        }
    }
    bool hit(unsigned at)
    {
        if (stage != at && !(stage == 3 && at == 1)) return false;
        if (skip) {
            --skip;
            return false;
        }
        if (stage != 3) stage = 0;      // one failure
        else if (at == 1) return true;  // stage 3: the unmap now, the restore next
        return true;
    }
};
#else
struct TestFault {
    unsigned stage = 0;
    bool hit(unsigned) { return false; }
};
#endif

// Test build only: X2W_XMEM_TEST_HOLD_WINDOW=<name> keeps an unmap in the window where the run is
// unmapped and not yet mapped again. It sets the event "<name>.unmapped", then waits up to
// five seconds for "<name>.release", so that a reader thread can reach the window every time.
#ifdef X2W_XMEM_TEST_CONTROLS
void hold_window()
{
    char name[96];
    const DWORD length = GetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", name, sizeof(name) - 16);
    if (!length || length >= sizeof(name) - 16) return;
    char event_name[128];
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", name);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", name);
    HANDLE release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    if (unmapped && release) {
        SetEvent(unmapped);
        if (WaitForSingleObject(release, 5000) != WAIT_OBJECT_0)
            x2w_local::logf("x2w-xmem test hold: nobody released the unmapped window within 5 s");
    }
    if (unmapped) CloseHandle(unmapped);
    if (release) CloseHandle(release);
}
#else
void hold_window() {}
#endif

// Drops a page that has no view any more from the registry, so that the registry never names a
// page Windows does not show.
void forget_page(ULONG_PTR address)
{
    const auto live = memory.views.find(address);
    if (live == memory.views.end()) return;
    --memory.backings.at(live->second).views;
    memory.views.erase(live);
}

// Maps every recorded page of [base, base + size) that has no view: neighbours on adjacent slots
// together, one page at a time when a group is refused. A page that cannot be mapped is forgotten
// and counted. fail is true when the test control refuses this range.
ULONG_PTR map_recorded(ULONG_PTR base, SIZE_T size, bool fail)
{
    ULONG_PTR lost = 0;
    ULONG_PTR at = base;
    while (at < base + size) {
        const auto live = memory.views.find(at);
        if (live == memory.views.end()) {
            at += page_bytes;
            continue;
        }
        const Backing head = memory.backings.at(live->second);
        SIZE_T bytes = page_bytes;
        for (; at + bytes < base + size; bytes += page_bytes) {
            const auto next = memory.views.find(at + bytes);
            if (next == memory.views.end()) break;
            const Backing& b = memory.backings.at(next->second);
            if (b.section != head.section || b.offset != head.offset + bytes) break;
        }
        std::vector<std::pair<ULONG_PTR, SIZE_T>> made;
        if (fail || !map_run(at, head, bytes, made)) {
            // map_run may have mapped some pages one at a time before it stopped; keep those.
            for (SIZE_T done = 0; done < bytes; done += page_bytes) {
                if (memory.runs.count(at + done)) continue;
                Backing one = head;
                one.offset += done;
                if (!fail && map_view(at + done, one)) {
                    memory.runs[at + done] = page_bytes;
                } else {
                    forget_page(at + done);
                    ++lost;
                }
            }
        }
        at += bytes;
    }
    return lost;
}

// Destinations must be caller-owned, page-sized Windows placeholders. Live views
// are owned by this shim until it unmaps them. The caller must not replace them.
template <typename AddressAt>
BOOL map_pages(ULONG_PTR count, const ULONG_PTR* tokens, const AddressAt& address_at) noexcept
{
    if (!count || count > era_physical::capacity_pages)
        return physical_result(era_physical::Status::invalid);
    try {
        const bool share_views = fault_retry.ready();
        const MappingThread marker;
        const std::lock_guard guard(memory.mutex);
        std::map<ULONG_PTR, ULONG_PTR> batch;
        for (ULONG_PTR i = 0; i < count; ++i) {
            const auto address = address_at(i);
            const auto live = memory.views.find(address);
            if (!address || address % page_bytes ||
                (tokens ? live != memory.views.end() || !era_physical::owns(tokens[i])
                        : live == memory.views.end()))
                return physical_result(era_physical::Status::invalid);
            if (!batch.emplace(address, tokens ? tokens[i] : live->second).second)
                return physical_result(era_physical::Status::invalid);
        }
        // Adjacent pages that are adjacent slots of one section share a single view (a "run"): a
        // multi-gigabyte batch then takes a few views instead of one per 64 KiB page.
        if (tokens) {
            for (const auto& [address, page] : batch) {
                if (!section_for(page)) return FALSE;
            }
            std::vector<std::pair<ULONG_PTR, SIZE_T>> made;
            auto first = batch.begin();
            while (first != batch.end()) {
                const Backing& head = memory.backings.at(first->second);
                SIZE_T bytes = page_bytes;
                auto last = std::next(first);
                for (; share_views && last != batch.end(); ++last, bytes += page_bytes) {
                    const Backing& b = memory.backings.at(last->second);
                    if (last->first != first->first + bytes || b.section != head.section || b.offset != head.offset + bytes) break;
                }
                if (!map_run(first->first, head, bytes, made)) {
                    const DWORD error = GetLastError();
                    for (const auto& [base, size] : made) {
                        UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(base), MEM_PRESERVE_PLACEHOLDER);
                        memory.runs.erase(base);
                    }
                    coalesce_unused(batch);
                    SetLastError(error);
                    return FALSE;
                }
                first = last;
            }
            for (const auto& [address, page] : batch)
                ++memory.backings.at(page).views;
            memory.views.merge(batch);
        } else {
            // Unmap every run that holds a page of the batch, then map the surviving pages of those
            // runs again. The registry (views, runs) matches what Windows shows on every way out:
            // - an unmap is refused: the runs already unmapped are mapped again whole, nothing is
            //   removed, and the call fails with that error;
            // - a surviving page cannot be mapped again: it is forgotten, the batch is unmapped, and
            //   the call fails.
            std::map<ULONG_PTR, SIZE_T> broken;
            for (const auto& [address, page] : batch) {
                auto run = memory.runs.upper_bound(address);
                if (run == memory.runs.begin()) continue;
                --run;
                if (address - run->first < run->second) broken.insert(*run);
            }
            TestFault test;
            begin_break(broken);
            struct EndBreak {
                ~EndBreak() { end_break(); }
            } const end;
            DWORD error = ERROR_SUCCESS;
            std::vector<std::pair<ULONG_PTR, SIZE_T>> unmapped;
            for (const auto& [base, size] : broken) {
                if (test.hit(1)) {
                    error = ERROR_INVALID_ADDRESS;
                    break;
                }
                if (!UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(base), MEM_PRESERVE_PLACEHOLDER)) {
                    error = GetLastError();
                    break;
                }
                memory.runs.erase(base);
                unmapped.emplace_back(base, size);
            }
            const bool refused = error != ERROR_SUCCESS;
            if (!refused) {
                for (const auto& [address, page] : batch) {
                    --memory.backings.at(page).views;
                    memory.views.erase(address);
                }
            }
            if (!refused && !unmapped.empty()) hold_window();
            ULONG_PTR lost = 0;
            for (const auto& [base, size] : unmapped)
                lost += map_recorded(base, size, refused ? test.stage == 3 && test.hit(3) : test.hit(2));
            if (!refused) coalesce_unused(batch);
            if (refused || lost) {
                x2w_local::logf("x2w-xmem unmap of %llu page(s) failed (error %lu): %s; %llu surviving page(s) could not be mapped "
                                "again and were forgotten", static_cast<unsigned long long>(batch.size()),
                                static_cast<unsigned long>(error), refused ? "an unmap was refused and no page of the batch was removed" : "the pages were unmapped",
                                static_cast<unsigned long long>(lost));
                SetLastError(refused ? error : ERROR_NOT_ENOUGH_MEMORY);
                return FALSE;
            }
        }
    } catch (const std::bad_alloc&) {
        return physical_result(era_physical::Status::exhausted);
    }
    return physical_result(era_physical::Status::ok);
}
}

// ---------------------------------------------------------------------------------------------
// Experimental (docs/startup-runtime.md). With X2W_XMEM_PERMISSIVE=1 the exports
// below accept the call shapes of a startup run on top of the section-backed pages above.
// Observations behind it:
// - Page numbers: the caller rebuilds them as (21-bit entry << 4), so it expects 4 KiB frame numbers
//   with the low four bits clear. Callers get page << 4 and pass it back.
// - XMemAllocatePhysicalPages flags other than MEM_64K_PAGES occur (0x280) and are served the same.
// - XMemVirtualAlloc: commits (type 0x20003000/0x20403000) become plain commits; reservations become
//   placeholders that pages can be mapped into; a "replace placeholder" call (0x20406000) on a piece
//   the caller split from its own placeholder keeps the placeholder and records it for mapping.
// - Scatter mapping passes page 0 to unmap and may name pages that are already mapped elsewhere
//   (a second view; the sections give real sharing).
namespace {
bool local_permissive() noexcept { return x2w_local::env_on("X2W_XMEM_PERMISSIVE"); }

void record_reservation(void* base, SIZE_T bytes)
{
    const std::lock_guard guard(memory.mutex);
    memory.reservations.insert_or_assign(reinterpret_cast<ULONG_PTR>(base), bytes);
}

void* local_virtual_alloc(void* address, SIZE_T bytes, DWORD type, ULONGLONG attributes, DWORD protection)
{
    const DWORD low = protection & 0xFF;
    const bool accessible = low == PAGE_READONLY || low == PAGE_READWRITE || low == PAGE_EXECUTE_READ ||
                            low == PAGE_EXECUTE_READWRITE;
    const DWORD protect = accessible ? low : PAGE_READWRITE;
    void* result = nullptr;
    const char* how = "";
    if ((type & MEM_REPLACE_PLACEHOLDER) && address) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(address, &info, sizeof(info)) && info.State == MEM_RESERVE) {
            record_reservation(address, bytes);
            result = address;
        }
        how = "placeholder kept for page mapping";
    } else if (type & MEM_COMMIT) {
        if (!address) {
            result = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, protect);
            how = "commit";
        } else {
            result = VirtualAlloc(address, bytes, MEM_COMMIT, protect);
            if (!result) {
                VirtualFree(address, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                result = VirtualAlloc2(GetCurrentProcess(), address, bytes,
                                       MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, protect, nullptr, 0);
            }
            how = "commit at address";
        }
    } else {
        result = VirtualAlloc2(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
                               nullptr, 0);
        // A caller that asks for a placeholder splits it itself; other reservations are split here on mapping.
        if (result && !(type & MEM_RESERVE_PLACEHOLDER)) record_reservation(result, bytes);
        how = "placeholder reserve";
    }
    const DWORD error = result ? 0 : GetLastError();
    x2w_local::logf("x2w-xmem XMemVirtualAlloc addr=%p size=0x%llx type=0x%lx attr=0x%llx prot=0x%lx -> %p (%s, err %lu)",
                    address, static_cast<unsigned long long>(bytes), static_cast<unsigned long>(type),
                    static_cast<unsigned long long>(attributes), static_cast<unsigned long>(protection), result, how,
                    static_cast<unsigned long>(error));
    SetLastError(error);
    return result;
}

// Applies (address, page) pairs; page 0 unmaps. An address that already shows the requested page is
// left alone; one that shows another page is unmapped first.
BOOL local_apply(ULONG_PTR count, const ULONG_PTR* addresses, const ULONG_PTR* pages, const char* what)
{
    std::map<ULONG_PTR, ULONG_PTR> want;
    for (ULONG_PTR i = 0; i < count; ++i) want[addresses[i]] = pages[i];
    std::vector<ULONG_PTR> unmap, map_at, map_page;
    {
        const std::lock_guard guard(memory.mutex);
        for (const auto& [address, page] : want) {
            const auto live = memory.views.find(address);
            if (!page) {
                if (live != memory.views.end()) unmap.push_back(address);
                continue;
            }
            if (live != memory.views.end()) {
                if (live->second == page) continue;
                unmap.push_back(address);
            }
            map_at.push_back(address);
            map_page.push_back(page);
        }
    }
    BOOL ok = TRUE;
    DWORD error = 0;
    if (!unmap.empty() && !map_pages(unmap.size(), nullptr, [&](ULONG_PTR i) { return unmap[i]; })) {
        ok = FALSE;
        error = GetLastError();
    }
    if (ok && !map_at.empty() &&
        !map_pages(map_at.size(), map_page.data(), [&](ULONG_PTR i) { return map_at[i]; })) {
        ok = FALSE;
        error = GetLastError();
    }
    static LONG calls;
    if (!ok || x2w_local::first_calls(calls, 16))
        x2w_local::logf("x2w-xmem %s count=0x%llx first=%p map=%llu unmap=%llu -> %d (err %lu)", what,
                        static_cast<unsigned long long>(count), count ? reinterpret_cast<void*>(addresses[0]) : nullptr,
                        static_cast<unsigned long long>(map_at.size()), static_cast<unsigned long long>(unmap.size()), ok,
                        static_cast<unsigned long>(error));
    SetLastError(error);
    return ok;
}

// Applies the same (address, page) pairs again at the GPU view for those addresses in the CPU view.
void local_mirror(ULONG_PTR count, const ULONG_PTR* addresses, const ULONG_PTR* pages)
{
    ULONG_PTR cpu, gpu;
    SIZE_T bytes;
    {
        const std::lock_guard guard(memory.mutex);
        cpu = memory.mirror_cpu, gpu = memory.mirror_gpu, bytes = memory.mirror_bytes;
    }
    if (!cpu) return;
    std::vector<ULONG_PTR> at, page;
    for (ULONG_PTR i = 0; i < count; ++i)
        if (addresses[i] >= cpu && addresses[i] - cpu < bytes) {
            at.push_back(addresses[i] - cpu + gpu);
            page.push_back(pages[i]);
        }
    if (!at.empty()) local_apply(at.size(), at.data(), page.data(), "GPU view mirror");
}

bool to_pages(ULONG_PTR count, const ULONG_PTR* tokens, std::vector<ULONG_PTR>& pages)
{
    if (count > era_physical::capacity_pages) return false;  // as the strict path: no grant is larger
    pages.resize(count);
    for (ULONG_PTR i = 0; i < count; ++i) {
        if (tokens[i] & 0xF) return false;
        pages[i] = tokens[i] >> 4;
    }
    return true;
}
}  // namespace

namespace {
BOOL free_pages(ULONG_PTR count, PULONG_PTR pages);
}

extern "C" BOOL WINAPI XMemAllocatePhysicalPages(ULONG_PTR flags, PULONG_PTR count, PULONG_PTR pages)
{
    static_assert(sizeof(ULONG_PTR) == sizeof(std::uint64_t));
    if (local_permissive()) {
        const ULONG_PTR requested = count ? *count : 0;
        era_physical::Status status;
        {
            const std::lock_guard guard(memory.mutex);
            status = era_physical::allocate(era_physical::mem_64k_pages, reinterpret_cast<std::uint64_t*>(count),
                                            reinterpret_cast<std::uint64_t*>(pages));
        }
        if (status == era_physical::Status::ok)
            for (ULONG_PTR i = 0; i < *count; ++i) pages[i] <<= 4;
        x2w_local::logf("x2w-xmem XMemAllocatePhysicalPages flags=0x%llx pages=0x%llx -> %d",
                        static_cast<unsigned long long>(flags), static_cast<unsigned long long>(requested),
                        status == era_physical::Status::ok);
        return physical_result(status);
    }
    const std::lock_guard guard(memory.mutex);
    return physical_result(era_physical::allocate(flags, reinterpret_cast<std::uint64_t*>(count),
                                                  reinterpret_cast<std::uint64_t*>(pages)));
}

extern "C" BOOL WINAPI XMemFreePhysicalPages(ULONG_PTR count, PULONG_PTR pages)
{
    if (local_permissive() && pages && count && count <= era_physical::capacity_pages) {
        std::vector<ULONG_PTR> converted;
        if (!to_pages(count, pages, converted)) return physical_result(era_physical::Status::invalid);
        return free_pages(count, converted.data());
    }
    return free_pages(count, pages);
}

namespace {
BOOL free_pages(ULONG_PTR count, PULONG_PTR pages)
{
    if (!pages || !count || count > era_physical::capacity_pages)
        return physical_result(era_physical::Status::invalid);
    const std::lock_guard guard(memory.mutex);
    for (ULONG_PTR i = 0; i < count; ++i) {
        const auto backing = memory.backings.find(pages[i]);
        if (backing != memory.backings.end() && backing->second.views)
            return physical_result(era_physical::Status::invalid);
    }
    const auto status = era_physical::release(count, reinterpret_cast<const std::uint64_t*>(pages));
    if (status == era_physical::Status::ok) {
        for (ULONG_PTR i = 0; i < count; ++i) {
            const auto backing = memory.backings.find(pages[i]);
            if (backing != memory.backings.end()) {
                memory.slots.release(backing->second.slab, backing->second.offset);
                memory.backings.erase(backing);
            }
        }
    }
    return physical_result(status);
}
}  // namespace

// Experimental export: one snapshot of the physical-page accounting for the startup heartbeat
// (#1199). A page gets a 64 KiB slot of a shared slab on first use, by a mapping or by a page-token write, and
// loses it when the grant is released; the number of pages holding a slot is reported so runs can be compared.
extern "C" BOOL WINAPI x2w_xmem_page_stats(ULONGLONG* live_sections)
{
    if (!live_sections) return FALSE;
    const std::lock_guard guard(memory.mutex);
    *live_sections = static_cast<ULONGLONG>(memory.backings.size());
    return TRUE;
}

// Experimental export: the slab accounting behind those pages (#1199): slabs open, slots in use and slots free
// in the open slabs. A slab is reserved when it is created, its slots are committed on first use, and it is closed only at process end, so
// a slot freed by a grant stays committed and these numbers do not show the commit held.
extern "C" BOOL WINAPI x2w_xmem_slab_stats(ULONGLONG* slabs, ULONGLONG* slots_in_use, ULONGLONG* slots_free)
{
    if (!slabs || !slots_in_use || !slots_free) return FALSE;
    const std::lock_guard guard(memory.mutex);
    *slabs = memory.slots.slab_count();
    *slots_in_use = memory.slots.slots_in_use();
    *slots_free = memory.slots.slots_free();
    return TRUE;
}

// Experimental export: writes bytes into physical pages named by the title's
// page tokens (page number << 4, as XMemMapPhysicalPages takes them), starting offset bytes into the
// first page. DirectStorage requests can name such pages as their destination.
extern "C" BOOL WINAPI x2w_xmem_write_physical(const ULONG_PTR* tokens, ULONG_PTR count, SIZE_T offset, const void* data, SIZE_T bytes)
{
    if (!tokens || !data || offset >= page_bytes) return FALSE;
    const auto* source = static_cast<const unsigned char*>(data);
    ULONG_PTR index = 0;
    while (bytes) {
        if (index >= count || (tokens[index] & 0xF)) return FALSE;
        HANDLE section = nullptr;
        ULONG64 section_offset = 0;
        {
            const std::lock_guard guard(memory.mutex);
            section = section_for(tokens[index] >> 4);
            if (section) section_offset = memory.backings.at(tokens[index] >> 4).offset;
        }
        if (!section) return FALSE;
        void* view = MapViewOfFile(section, FILE_MAP_WRITE, static_cast<DWORD>(section_offset >> 32),
                                   static_cast<DWORD>(section_offset), page_bytes);
        if (!view) return FALSE;
        const SIZE_T chunk = bytes < page_bytes - offset ? bytes : page_bytes - offset;
        std::memcpy(static_cast<unsigned char*>(view) + offset, source, chunk);
        UnmapViewOfFile(view);
        source += chunk;
        bytes -= chunk;
        offset = 0;
        ++index;
    }
    return TRUE;
}

extern "C" void* WINAPI XMemMapPhysicalPages(void* base, ULONG_PTR count, const ULONG_PTR* tokens)
{
    const auto address = reinterpret_cast<ULONG_PTR>(base);
    if (!address || count > (std::numeric_limits<ULONG_PTR>::max() - address) / page_bytes) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (local_permissive()) {
        if (count > era_physical::capacity_pages) {  // as the strict path; sizes the vectors below
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        // A null array unmaps (seen on a cleanup path, also with a zero count).
        std::vector<ULONG_PTR> addresses(count), pages(count, 0);
        for (ULONG_PTR i = 0; i < count; ++i) addresses[i] = address + i * page_bytes;
        if (tokens && !to_pages(count, tokens, pages)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        if (!count) return base;
        if (!local_apply(count, addresses.data(), pages.data(), "XMemMapPhysicalPages")) return nullptr;
        local_mirror(count, addresses.data(), pages.data());
        return base;
    }
    return map_pages(count, tokens, [=](ULONG_PTR i) { return address + i * page_bytes; }) ? base : nullptr;
}

extern "C" BOOL WINAPI XMemMapPhysicalPagesScatter(void* const* addresses, ULONG_PTR count,
                                                   const ULONG_PTR* tokens)
{
    if (!addresses) return physical_result(era_physical::Status::invalid);
    if (local_permissive() && tokens) {
        std::vector<ULONG_PTR> pages;
        if (!to_pages(count, tokens, pages)) return physical_result(era_physical::Status::invalid);
        if (!local_apply(count, reinterpret_cast<const ULONG_PTR*>(addresses), pages.data(), "XMemMapPhysicalPagesScatter"))
            return FALSE;
        local_mirror(count, reinterpret_cast<const ULONG_PTR*>(addresses), pages.data());
        return TRUE;
    }
    return map_pages(count, tokens, [=](ULONG_PTR i) { return reinterpret_cast<ULONG_PTR>(addresses[i]); });
}

// Startup compatibility configuration (#1758, under #874): X2W_XMEM_STARTUP_BELOW_40=1 places the
// large-page pool blocks with attribute 0x283 (protection 0x2004 or 0x2404) so that the whole block,
// last byte included, lies below 2^40. The title passes the CPU address of such a block straight
// through as a GPU address, and a package acceleration-structure builder refuses destinations with
// more than 40 address bits. Off by default; every other shape keeps its placement. The low range is
// a requirement, never a preference: an allocation that cannot be placed there fails with the host
// error, and nothing is truncated, masked or retried higher.
constexpr ULONG_PTR startup_low_limit = ULONG_PTR{1} << 40;  // exclusive
constexpr SIZE_T startup_large_page = SIZE_T{2} << 20;

#ifdef X2W_XMEM_TEST_CONTROLS
// Test build only (tests/xmem_startup_below_40_check.cpp): X2W_XMEM_TEST_BELOW_40 names mutants
// that drop the address limit ("no_limit"), the check after the allocation ("no_check") or the
// release of a refused block ("no_release").
static bool startup_below_40_mutant(const char* name) noexcept
{
    char text[64]{};
    return GetEnvironmentVariableA("X2W_XMEM_TEST_BELOW_40", text, sizeof(text)) && std::strstr(text, name);
}
#else
static bool startup_below_40_mutant(const char*) noexcept { return false; }
#endif

// Checks a block allocated with the 2^40 requirement: 2 MiB aligned and base + bytes - 1 < 2^40,
// computed without overflow. A block outside that range is released and refused.
static void* keep_startup_block_below_40(void* block, SIZE_T bytes, ULONGLONG attributes, DWORD protection) noexcept
{
    if (!block) {
        const DWORD error = GetLastError();
        x2w_local::logf("x2w-xmem startup block below 2^40 refused: size=0x%llx attr=0x%llx prot=0x%lx, no 2 MiB-aligned "
                        "range ending below 2^40 (host error %lu)",
                        static_cast<unsigned long long>(bytes), attributes, static_cast<unsigned long>(protection),
                        static_cast<unsigned long>(error));
        SetLastError(error);
        return nullptr;
    }
    const auto base = reinterpret_cast<ULONG_PTR>(block);
    const bool inside = base % startup_large_page == 0 && bytes <= startup_low_limit && base <= startup_low_limit - bytes;
    if (!inside && !startup_below_40_mutant("no_check")) {
        const BOOL released = startup_below_40_mutant("no_release") ? FALSE : VirtualFree(block, 0, MEM_RELEASE);
        x2w_local::logf("x2w-xmem startup block below 2^40 refused: size=0x%llx attr=0x%llx prot=0x%lx, placed at %p "
                        "outside the 2 MiB-aligned range below 2^40 (%s)",
                        static_cast<unsigned long long>(bytes), attributes, static_cast<unsigned long>(protection), block,
                        released ? "released" : "not released");
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    x2w_local::logf("x2w-xmem startup block below 2^40: %p size=0x%llx last=%p attr=0x%llx prot=0x%lx", block,
                    static_cast<unsigned long long>(bytes), reinterpret_cast<void*>(base + (bytes - 1)), attributes,
                    static_cast<unsigned long>(protection));
    return block;
}

// Reserve+commit desktop profile for the startup blocks observed on #870: the pool loop asks
// for type 0x20003000 with 0x183/0x22004, 0x283/0x2404 or 0x283/0x2004, and later startup code for type
// 0x20403000 (64 KiB pages) with 0x181/PAGE_READWRITE, each with a null base. Proposed
// desktop policy: commit a fresh block of ordinary zero-filled read/write pages, whose size
// must be a nonzero multiple of 64 KiB and is committed exactly, without rounding. The
// MEM_LARGE_PAGES type gets a 2 MiB-aligned base (the x64 large-page size); the
// MEM_64K_PAGES type (MEM_LARGE_PAGES | MEM_PHYSICAL) keeps the 64 KiB allocation
// granularity. Large pages themselves and the console protection bits above PAGE_READWRITE
// are not reproduced. A host failure returns NULL with the host error. X2W_XMEM_STARTUP_BELOW_40
// adds an address limit for the 0x283 blocks (keep_startup_block_below_40).
static void* commit_startup_block(SIZE_T bytes, DWORD allocation_type, ULONGLONG attributes,
                                  DWORD protection) noexcept
{
    constexpr SIZE_T granularity = SIZE_T{64} << 10;
    const bool large_pages = allocation_type == 0x20003000;
    const bool supported = large_pages
                               ? (attributes == 0x183 && protection == 0x22004)
                                     || (attributes == 0x283 && (protection == 0x2404 || protection == 0x2004))
                               : attributes == 0x181 && protection == PAGE_READWRITE;
    if (!bytes || bytes % granularity || !supported) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (!large_pages) return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    MEM_ADDRESS_REQUIREMENTS requirements{};
    requirements.Alignment = SIZE_T{2} << 20;
    // supported already limits attribute 0x283 to protection 0x2004 or 0x2404.
    const bool below_40 = attributes == 0x283 && x2w_local::env_on("X2W_XMEM_STARTUP_BELOW_40");
    if (below_40 && !startup_below_40_mutant("no_limit"))
        requirements.HighestEndingAddress = reinterpret_cast<void*>(startup_low_limit - 1);
    MEM_EXTENDED_PARAMETER parameter{};
    parameter.Type = MemExtendedParameterAddressRequirements;
    parameter.Pointer = &requirements;
    void* const block = VirtualAlloc2(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE,
                                      &parameter, 1);
    return below_40 ? keep_startup_block_below_40(block, bytes, attributes, protection) : block;
}

// The startup commit profile above, and the reserve-only desktop profile from the source-built
// #888 comparison.
extern "C" void* WINAPI XMemVirtualAlloc(void* address, SIZE_T bytes, DWORD allocation_type,
                                        ULONGLONG attributes, DWORD protection)
{
    constexpr DWORD startup_reserve = 0x20402000;
    constexpr SIZE_T gib = SIZE_T{1} << 30;
    if (!address && (allocation_type == 0x20003000 || allocation_type == (startup_reserve | MEM_COMMIT)))
        return commit_startup_block(bytes, allocation_type, attributes, protection);
    const bool supported = (attributes == 0x21 && protection == PAGE_READWRITE)
                           || (attributes == 3 && protection == (0x800 | PAGE_NOACCESS));
    if (address || (bytes != 16 * gib && bytes != 24 * gib) || allocation_type != startup_reserve || !supported) {
        if (local_permissive()) return local_virtual_alloc(address, bytes, allocation_type, attributes, protection);
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    const std::lock_guard guard(memory.mutex);
    // Callers release unused reservations with Win32. Drop entries for released
    // ranges before another allocation can reuse those virtual addresses.
    for (auto it = memory.reservations.begin(); it != memory.reservations.end();) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<void*>(it->first), &info, sizeof(info)) && info.State == MEM_FREE)
            it = memory.reservations.erase(it);
        else
            ++it;
    }
    auto* reservation = VirtualAlloc2(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                      PAGE_NOACCESS, nullptr, 0);
    if (!reservation) return nullptr;
    try {
        memory.reservations.insert_or_assign(reinterpret_cast<ULONG_PTR>(reservation), bytes);
        const auto base = reinterpret_cast<ULONG_PTR>(reservation);
        if (protection == PAGE_READWRITE) {
            memory.last_cpu_base = base;
            memory.last_cpu_bytes = bytes;
        } else if (memory.last_cpu_base && memory.last_cpu_bytes == bytes && x2w_local::env_on("X2W_XMEM_GPU_MIRROR")) {
            memory.mirror_cpu = memory.last_cpu_base;
            memory.mirror_gpu = base;
            memory.mirror_bytes = bytes;
            x2w_local::logf("x2w-xmem GPU view %p (+0x%llx) mirrors CPU view %p, 0x%llx bytes", reservation,
                            static_cast<unsigned long long>(base - memory.last_cpu_base), reinterpret_cast<void*>(memory.last_cpu_base),
                            static_cast<unsigned long long>(bytes));
        }
    } catch (const std::bad_alloc&) {
        VirtualFree(reservation, 0, MEM_RELEASE);
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    return reservation;
}
