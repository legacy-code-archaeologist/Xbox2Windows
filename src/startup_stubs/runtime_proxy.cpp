// Experimental (docs/startup-runtime.md): proxy between the title and the system gaming runtime.
//
// The title's static runtime library loads the system xgameruntime.dll and asks it for API objects
// with QueryApiImpl(provider GUID, interface GUID, &object). With X2W_RUNTIME_PROXY=1 this file (built
// into the GameInput stand-in, which the title imports) patches the title's import of GetProcAddress
// so that the lookup of "QueryApiImpl" returns a wrapper. The wrapper hands out proxy objects whose
// methods log their first calls ("x2w-runtime") and forward to the real object.
//
// X2W_FAKE_USER=1 additionally answers the user API for one local user (see fake_user below)
// and, while the proxy is on, XStoreCreateContext for that user (see area_call).
#include "startup_stubs/api_trace.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/save_blob_pack.h"
#include "startup_stubs/save_store.h"
#include "startup_stubs/storage_root.h"
#include "startup_stubs/storage_root_utf8.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <intrin.h>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {
using u64 = std::uint64_t;
using Method = u64(__stdcall*)(void*, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);
constexpr int max_providers = 24, max_slots = 96;

struct Proxy {
    void** vtable;
    void* real;
    int provider;
};
GUID provider_ids[max_providers];
Proxy* proxies[max_providers];
int provider_count;
LONG call_counts[max_providers][max_slots];
SRWLOCK proxy_lock = SRWLOCK_INIT;
void* slot_table[max_slots];

using QueryApi = HRESULT(__stdcall*)(const GUID*, const GUID*, void**);
QueryApi real_query;

// ---- local user (X2W_FAKE_USER=1) ----------------------------------------------------------
// The user API object is the provider whose GUID starts 01acd177 (seen: called at start-up and when
// the sign-in button is pressed). Slots are filled in from observed calls; see user_call below.
constexpr unsigned long user_provider_data1 = 0x01acd177;
constexpr u64 local_user_handle = 0x58325755'00000001ull;  // not a pointer the runtime knows
constexpr u64 local_user_id = 0x0009000012345678ull;
// XStoreImpl. Method 3 is XStoreCreateContext; the handle below is the offline context it writes.
constexpr GUID store_provider_id{0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}};
constexpr u64 store_context_handle = 0x58325753'00000001ull;  // not a pointer the runtime knows
bool fake_user() noexcept { return x2w_local::env_on("X2W_FAKE_USER"); }
int user_provider = -1, save_provider = -1;
// The user provider also hands out a second interface (iid cef4fac0...) whose slot 3 is
// GetGamertag(handle, component, size, buffer, &used). It is proxied as an entry of its own (shown as
// provider cef4fac0), not as the main user object, whose slot 3 duplicates a handle.
int gamertag_provider = -1;
constexpr unsigned long gamertag_iid_data1 = 0xcef4fac0;
// Async blocks completed on behalf of a local answer (see complete_async): calls that mention them
// are always logged, and the borrowed operation's own failure code is not passed on for them.
u64 tracked_blocks[64];
volatile LONG tracked_count, tracked_logged;
constexpr u64 borrowed_failure = 0x89245110;
bool tracked(u64 value)
{
    if (!value) return false;
    for (LONG i = 0; i < tracked_count && i < 64; ++i)
        if (tracked_blocks[i] == value) return true;
    return false;
}
bool user_call(int slot, u64* args, u64* result);
bool save_call(int slot, u64* args, u64* result);
bool area_call(const GUID& provider_id, int slot, u64* args, u64* result);
u64 complete_async(u64 async_block);
void* storage_object();
const char* storage_path();
constexpr unsigned long save_provider_data1 = 0x704c3f58;  // hypothesis: the game-save API (first seen: slot 4 with the user handle)

// X2W_TRACE_API=1: one line per provider call with the return code, whether the answer came from the
// stand-in or the real runtime, and which declared output argument it wrote. Only declared outputs are
// read: arguments are not probed as pointers, because a handle or size that happens to look like an
// address would be read as if it were one. A slot with no declared contract reports "out=?" instead.
bool trace_api() noexcept
{
    static const bool on = x2w_local::env_on("X2W_TRACE_API");
    return on;
}

// The contract table and the decision about what was observed live in api_trace.h so the decision can
// be checked by a source-built control without a running title.
using x2w_api_trace::declared;
using x2w_api_trace::DeclaredOut;
using x2w_api_trace::max_outputs;
using x2w_api_trace::max_snapshot;
using x2w_api_trace::no_argument;


template <int N>
__declspec(noinline) u64 forward(Proxy* self, u64 a, u64 b, u64 c, u64 s0, u64 s1, u64 s2, u64 s3, u64 s4, u64 s5, u64 s6, u64 s7)
{
    // call_counts also feeds the periodic call-rate summary, so it keeps counting; a count that has
    // passed 2^31 is negative and must not log again.
    const LONG call = N >= 3 ? InterlockedIncrement(&call_counts[self->provider][N]) : 0;
    const bool log_it = call > 0 && call <= 6;
    u64 args[11] = {a, b, c, s0, s1, s2, s3, s4, s5, s6, s7};
    // Declared outputs are snapshotted so the write is seen rather than assumed; nothing else is read. The
    // snapshot is bounded by the declared width, so a one-byte flag, a DWORD and a caller-sized buffer
    // Declared outputs are snapshotted so a write can be observed rather than assumed; nothing else is
    // read. Each snapshot is bounded by the declared width, so a one-byte flag, a DWORD and a
    // caller-sized buffer are each compared within their own extent.
    bool compared[2] = {};
    std::size_t compared_bytes[2] = {};
    int compared_index[2] = {no_argument, no_argument};
    unsigned char before[2][max_snapshot] = {};
    const DeclaredOut* contract = trace_api() ? declared(provider_ids[self->provider].Data1, N) : nullptr;
    if (contract) {
        const auto take = [&](int slot, unsigned char index, unsigned char fixed_bytes, unsigned char size_index) {
            compared_index[slot] = index;
            if (index == no_argument || !args[index]) return;
            std::size_t bytes = fixed_bytes;
            if (!bytes) {
                if (size_index == no_argument) return;
                const u64 capacity = args[size_index];
                bytes = capacity < max_snapshot ? static_cast<std::size_t>(capacity) : max_snapshot;
            }
            if (!bytes || bytes > max_snapshot) return;
            if (!x2w_local::readable(reinterpret_cast<const void*>(args[index]), bytes)) return;
            std::memcpy(before[slot], reinterpret_cast<const void*>(args[index]), bytes);
            compared_bytes[slot] = bytes;
            compared[slot] = true;
        };
        take(0, contract->first, contract->first_bytes, contract->size_argument);
        take(1, contract->second, contract->second_bytes, no_argument);
    }
    u64 result = 0;
    bool handled = self->provider == user_provider && fake_user() && user_call(N, args, &result);
    if (!handled && self->provider == gamertag_provider && N == 3 && fake_user()) {
        // GetGamertag: the local user's name, NUL-terminated UTF-8; used = bytes including the NUL.
        static const char name[] = "Player";
        const u64 size = c, buffer = s0, used = s1;
        handled = true;
        if (a != local_user_handle) {
            result = 0x89245104;  // no such user
        } else if (size < sizeof(name) || !buffer || !x2w_local::writable(reinterpret_cast<void*>(buffer), sizeof(name))) {
            result = 0x8007007A;  // buffer too small
        } else {
            std::memcpy(reinterpret_cast<void*>(buffer), name, sizeof(name));
            if (used && x2w_local::writable(reinterpret_cast<void*>(used), 8)) *reinterpret_cast<u64*>(used) = sizeof(name);
            result = 0;
        }
    }
    if (!handled && self->provider == save_provider && N >= 3 && fake_user()) handled = save_call(N, args, &result);
    if (!handled && N >= 3 && fake_user()) handled = area_call(provider_ids[self->provider], N, args, &result);
    if (!handled && N >= 3 && fake_user()) {
        // The offline context is our token, never a desktop runtime object. Unimplemented
        // methods must fail here rather than let the real runtime dereference it.
        for (int i = 0; i < 4 && !handled; ++i) {
            if (args[i] == store_context_handle) {
                handled = true;
                result = 0x80004005;
            }
        }
    }
    if (!handled && self->provider != user_provider && N >= 3 && fake_user()) {
        // Other API families cannot resolve the local user's handle: fail the call instead of passing it on.
        for (int i = 0; i < 4 && !handled; ++i) {
            if (args[i] == local_user_handle) {
                handled = true;
                result = 0x80004005;
            }
        }
    }
    if (!handled) {
        const Method method = reinterpret_cast<Method>((*reinterpret_cast<void***>(self->real))[N]);
        result = method(self->real, a, b, c, s0, s1, s2, s3, s4, s5, s6, s7);
    }
    const bool about_tracked = N >= 3 && (tracked(a) || tracked(b) || tracked(c));
    const u64 raw_result = result;
    if (about_tracked && !handled && fake_user() && (result & 0xffffffffu) == borrowed_failure) result = 0;
    if (trace_api()) {
        unsigned char after[2][max_snapshot] = {};
        bool current[2] = {};
        for (int i = 0; i < 2; ++i) {
            if (!compared[i] || !compared_bytes[i]) continue;
            if (!x2w_local::readable(reinterpret_cast<const void*>(args[compared_index[i]]), compared_bytes[i])) continue;
            std::memcpy(after[i], reinterpret_cast<const void*>(args[compared_index[i]]), compared_bytes[i]);
            current[i] = true;
        }
        const x2w_api_trace::Report report = x2w_api_trace::decide(contract, current,
                                                                  const_cast<const unsigned char (*)[max_snapshot]>(before),
                                                                  const_cast<const unsigned char (*)[max_snapshot]>(after),
                                                                  compared_bytes);
        char out[48] = "?";
        x2w_api_trace::format(report, contract, out, sizeof(out));
        x2w_local::logf("x2w-api %08lx::slot%d -> 0x%08x %s out=%s", provider_ids[self->provider].Data1, N,
                        static_cast<unsigned>(result & 0xffffffffu), handled ? "local" : "real", out);
    }
    if (log_it || (about_tracked && x2w_local::first_calls(tracked_logged, 120)))
        x2w_local::logf("x2w-runtime %08lx::slot%d(0x%llx, 0x%llx, 0x%llx | 0x%llx, 0x%llx) -> 0x%llx%s%s from +0x%llx", provider_ids[self->provider].Data1, N,
                        a, b, c, s0, s1, result, handled ? " (local user)" : "",
                        about_tracked ? (raw_result != result ? " (tracked async, failure hidden)" : " (tracked async)") : "",
                        reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr)));
    return result;
}
template <std::size_t... I>
void fill_slots(std::index_sequence<I...>)
{
    ((slot_table[I] = reinterpret_cast<void*>(&forward<static_cast<int>(I)>)), ...);
}

HRESULT __stdcall proxy_query(const GUID* provider, const GUID* iid, void** out)
{
    const HRESULT hr = real_query(provider, iid, out);
    if (FAILED(hr) && out && provider && provider->Data1 == 0xf4faf4d4 && fake_user()) {
        // Persistent local storage: the system runtime has none for an unpackaged process.
        static LONG logged;
        if (x2w_local::first_calls(logged, 2)) x2w_local::logf("x2w-runtime persistent local storage: local object instead of 0x%08lx", static_cast<unsigned long>(hr));
        *out = storage_object();
        return S_OK;
    }
    if (FAILED(hr) || !out || !*out || !provider) return hr;
    GUID key = *provider;
    const bool gamertag = provider->Data1 == user_provider_data1 && iid && iid->Data1 == gamertag_iid_data1;
    if (gamertag) key.Data1 = gamertag_iid_data1;
    provider = &key;
    AcquireSRWLockExclusive(&proxy_lock);
    int index = -1;
    for (int i = 0; i < provider_count; ++i)
        if (std::memcmp(&provider_ids[i], provider, sizeof(GUID)) == 0) index = i;
    if (index < 0 && provider_count < max_providers) {
        index = provider_count++;
        provider_ids[index] = *provider;
        if (provider->Data1 == user_provider_data1) user_provider = index;
        if (gamertag) gamertag_provider = index;
        if (provider->Data1 == save_provider_data1) save_provider = index;
        x2w_local::logf("x2w-runtime provider %08lx-%04x-... -> object %p (proxied as #%d)", provider->Data1, provider->Data2, *out, index);
    }
    Proxy* proxy = nullptr;
    if (index >= 0) {
        if (!proxies[index] || proxies[index]->real != *out) proxies[index] = new Proxy{slot_table, *out, index};
        proxy = proxies[index];
    }
    ReleaseSRWLockExclusive(&proxy_lock);
    if (proxy) *out = proxy;
    return hr;
}

FARPROC WINAPI hooked_get_proc_address(HMODULE module, LPCSTR name)
{
    const FARPROC real = GetProcAddress(module, name);
    if (real && reinterpret_cast<u64>(name) > 0xffff) {
        static LONG logged;
        if (x2w_local::first_calls(logged, 40)) x2w_local::logf("x2w-runtime GetProcAddress(%p, \"%s\") -> %p", static_cast<void*>(module), name, static_cast<void*>(real));
        if (std::strcmp(name, "QueryApiImpl") == 0) {
            real_query = reinterpret_cast<QueryApi>(real);
            return reinterpret_cast<FARPROC>(&proxy_query);
        }
    }
    return real;
}

// Replaces the main module's import of GetProcAddress (by name) with the hook.
void patch_imports()
{
    auto* base = reinterpret_cast<unsigned char*>(GetModuleHandleW(nullptr));
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!directory.VirtualAddress) return;
    int patched = 0;
    for (auto* import = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress); import->Name; ++import) {
        if (!import->OriginalFirstThunk) continue;
        auto* names = reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + import->OriginalFirstThunk);
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + import->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (names->u1.Ordinal & IMAGE_ORDINAL_FLAG64) continue;
            const auto* by_name = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (std::strcmp(by_name->Name, "GetProcAddress") != 0) continue;
            DWORD old = 0;
            if (VirtualProtect(&slots->u1.Function, 8, PAGE_READWRITE, &old)) {
                slots->u1.Function = reinterpret_cast<ULONGLONG>(&hooked_get_proc_address);
                VirtualProtect(&slots->u1.Function, 8, old, &old);
                ++patched;
            }
        }
    }
    x2w_local::logf("x2w-runtime proxy: %d GetProcAddress import(s) of the main module patched", patched);
}

// ---- the local user -------------------------------------------------------------------------
// Filled in from observed calls (slot numbers are positions in the user API object's vtable).
// Returning false forwards the call to the real runtime.
bool user_call(int slot, u64* args, u64* result)
{
    const auto out = [](u64 pointer, std::size_t bytes) { return pointer && x2w_local::writable(reinterpret_cast<void*>(pointer), bytes); };
    constexpr u64 ok = 0, fail = 0x80004005, not_found = 0x80070490;
    switch (slot) {
    case 3:  // XUserDuplicateHandle(handle, &duplicate)
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_handle;
        *result = ok;
        return true;
    case 4:  // XUserCloseHandle(handle)
        if (args[0] != local_user_handle) return false;
        *result = 0;
        return true;
    case 5:  // XUserCompare(a, b)
        if (args[0] != local_user_handle && args[1] != local_user_handle) return false;
        *result = args[0] == args[1] ? 0 : 1;
        return true;
    case 8:  // XUserAddResult(async, &handle): the real operation ends with "no default user"; answer with the local user
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_handle;
        *result = ok;
        return true;
    case 9:  // XUserGetLocalId(handle, &local id)
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = 1;
        *result = ok;
        return true;
    case 10:  // XUserFindUserByLocalId(local id, &handle)
    case 12:  // XUserFindUserById(id, &handle)
        if (args[0] != (slot == 10 ? 1 : local_user_id)) return false;
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_handle;
        *result = ok;
        return true;
    case 11:  // XUserGetId(handle, &id)
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_id;
        *result = ok;
        return true;
    case 13:  // XUserGetIsGuest(handle, &is guest)
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 1)) *reinterpret_cast<bool*>(args[1]) = false;
        *result = ok;
        return true;
    case 14:  // XUserGetState(handle, &state): 0 = signed in
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 4)) *reinterpret_cast<std::uint32_t*>(args[1]) = 0;
        *result = ok;
        return true;
    case 15: {  // XUserGetGamertag(handle, component, size, buffer, &used)
        if (args[0] != local_user_handle) return false;
        static const char name[] = "Player";
        const u64 size = args[2];
        if (size >= sizeof(name) && out(args[3], sizeof(name))) std::memcpy(reinterpret_cast<void*>(args[3]), name, sizeof(name));
        if (out(args[4], 8)) *reinterpret_cast<u64*>(args[4]) = sizeof(name);
        *result = size >= sizeof(name) ? ok : 0x8007007A;  // ERROR_INSUFFICIENT_BUFFER
        return true;
    }
    case 19:  // XUserGetAgeGroup(handle, &group): 3 = adult
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 4)) *reinterpret_cast<std::uint32_t*>(args[1]) = 3;
        *result = ok;
        return true;
    case 20:  // XUserCheckPrivilege(handle, options, privilege, &has, &deny reason)
        if (args[0] != local_user_handle) return false;
        if (out(args[3], 1)) *reinterpret_cast<bool*>(args[3]) = true;
        if (out(args[4], 4)) *reinterpret_cast<std::uint32_t*>(args[4]) = 0;
        *result = ok;
        return true;
    case 40:  // XUserFindForDevice(device id, &handle): the virtual gamepad belongs to the local user
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_handle;
        *result = ok;
        return true;
    default:
        break;
    }
    // Anything else that names the local user cannot go to the real runtime (the handle is not its own).
    for (int i = 0; i < 4; ++i) {
        if (args[i] == local_user_handle) {
            *result = slot == 35 ? not_found : fail;
            return true;
        }
    }
    return false;
}

// ---- local game saves -----------------------------------------------------------------------
// Slot numbers follow the order of the public XGameSave.h (hypothesis, checked against the logged
// argument shapes). Containers and blob writes are kept on disk (save_store.h, below
// storage_path()/x2w-gamesave). Enumerations still report nothing, and the synchronous read (20)
// still reports no blob. Slots 21 and 22 read through the system XAsync service (see async_read).
// The other asynchronous calls are completed by starting the real "add default user silently"
// operation on the caller's async block: it fails quickly inside the runtime and runs the caller's
// completion callback on its task queue; the matching Result call is then answered here.
constexpr u64 save_provider_handle = 0x58325753'00000001ull, save_container_handle = 0x58325743'00000001ull,
              save_update_handle = 0x58325755'50000001ull;
// Container and update handles keep their fixed base; the low bits count from 1 and index these tables.
// Reopening a container name returns the same handle, so the table stays small.
constexpr std::size_t max_save_handles = x2w_save::ContainerTable::capacity;
constexpr u64 e_invalidarg = 0x80070057, e_outofmemory = 0x8007000E, e_fail = 0x80004005;
x2w_save::ContainerTable save_table;  // guarded by save_lock; closed containers give their entry back
SRWLOCK save_lock = SRWLOCK_INIT;
x2w_save::Store save_store() { return x2w_save::Store(std::filesystem::path(storage_path()) / "x2w-gamesave"); }
// A NUL-terminated name of at most x2w_save::max_name_bytes in readable memory.
bool read_save_name(u64 pointer, std::string& name)
{
    const std::size_t readable = x2w_local::probe_readable(static_cast<std::uintptr_t>(pointer), x2w_save::max_name_bytes + 1);
    const char* text = reinterpret_cast<const char*>(pointer);
    const std::size_t length = readable ? strnlen(text, readable) : 0;
    if (!readable || length == readable) return false;
    name.assign(text, length);
    return true;
}
u64 save_status(x2w_save::Status status)
{
    using x2w_save::Status;
    switch (status) {
    case Status::ok: return 0;
    case Status::bad_name: case Status::too_large: return e_invalidarg;
    case Status::not_found: return 0x80830008;
    default: return e_fail;
    }
}
u64 open_container(const std::string& name, u64* handle)
{
    const x2w_save::Status status = save_store().create_container(name);
    if (status != x2w_save::Status::ok) return save_status(status);
    AcquireSRWLockExclusive(&save_lock);
    const long index = save_table.open(name);
    ReleaseSRWLockExclusive(&save_lock);
    if (index < 0) return e_outofmemory;
    *handle = save_container_handle + static_cast<u64>(index);
    return 0;
}
// The container name behind an update handle, or empty when it is not an open update.
std::string update_container(u64 update)
{
    std::string name;
    AcquireSRWLockShared(&save_lock);
    if (update >= save_update_handle) name = save_table.update_container(static_cast<std::size_t>(update - save_update_handle));
    ReleaseSRWLockShared(&save_lock);
    return name;
}
// Publishes what an update has staged, all or nothing. The update itself stays open.
u64 submit_update(u64 update)
{
    if (update < save_update_handle || update - save_update_handle >= max_save_handles) return 0;
    const std::size_t index = static_cast<std::size_t>(update - save_update_handle);
    try {
        std::string container;
        std::vector<x2w_save::Action> actions;
        {
            AcquireSRWLockExclusive(&save_lock);
            struct Unlock { ~Unlock() { ReleaseSRWLockExclusive(&save_lock); } } guard;
            container = save_table.update_container(index);
            actions = save_table.take(index);
        }
        if (container.empty() || actions.empty()) return 0;
        const auto status = save_store().commit(container, actions);
        if (status != x2w_save::Status::ok) x2w_local::logf("x2w-runtime game save: update refused (status %d)", static_cast<int>(status));
        return save_status(status);
    } catch (const std::bad_alloc&) {
        return e_outofmemory;
    } catch (const std::filesystem::filesystem_error&) {
        return e_fail;
    }
}
// XAsyncBegin reserves the caller's block and completion queue. Disk publication
// starts only after acceptance. Complete carries the result for this block;
// it can invoke user code immediately, so no save lock may span that call.
// The verified public PC GDK async interface uses slots 7, 10 and 11.
constexpr GUID submit_async_family{0x073b7dcb,0x1fcf,0x4030,{0x94,0xbe,0xe3,0xc9,0xeb,0x62,0x34,0x28}};
constexpr char submit_identity[] = "local XGameSaveSubmitUpdate";
HRESULT WINAPI submit_provider(unsigned, const void*) { return S_OK; }
u64 async_submit(int slot, u64 update, u64 block)
{
    if (!block || !real_query) return e_fail;
    void* service = nullptr;
    const auto query = real_query(&submit_async_family, &submit_async_family, &service);
    if (FAILED(query)) return static_cast<unsigned>(query);
    if (!service) return e_fail;
    auto** methods = *static_cast<void***>(service);
    struct ApiRef {
        void* object;
        ULONG (WINAPI* release)(void*);
        ~ApiRef() { release(object); }
    } api{service,reinterpret_cast<ULONG(WINAPI*)(void*)>(methods[2])};
    auto* async = reinterpret_cast<void*>(block);
    if (slot == 29) {
        using Result = HRESULT(WINAPI*)(void*,void*,const void*,size_t,void*,size_t*);
        return static_cast<unsigned>(reinterpret_cast<Result>(methods[11])(service, async, submit_identity, 0, nullptr, nullptr));
    }
    using Begin = HRESULT(WINAPI*)(void*,void*,void*,const void*,const char*,decltype(&submit_provider));
    const auto hr = reinterpret_cast<Begin>(methods[7])(service, async, nullptr, submit_identity, submit_identity, submit_provider);
    if (FAILED(hr)) return static_cast<unsigned>(hr);
    const auto status = static_cast<HRESULT>(submit_update(update));
    using Complete = void(WINAPI*)(void*,void*,HRESULT,size_t);
    reinterpret_cast<Complete>(methods[10])(service, async, status, 0);
    return S_OK;
}

// XGameSaveReadBlobDataAsync / Result. Same XAsync service as submit, different identity, so a
// read block is not a submit block. Begin is attempted before any read: a rejection leaves the
// store untouched. The snapshot is packed only after acceptance. Complete can run the caller's
// callback before it returns, so no save lock is held across it. A failed read completes with no
// payload (the runtime then drops the context); the bytes are copied into the caller's buffer only
// when slot 22 asks, and only if that buffer is large enough.
constexpr char read_identity[] = "local XGameSaveReadBlobData";
constexpr u64 small_buffer = 0x80830007;
struct AsyncProviderData {
    void* async;
    std::size_t buffer_size;
    void* buffer;
    void* context;
};
static_assert(sizeof(AsyncProviderData) == 32 && offsetof(AsyncProviderData, context) == 24, "XAsyncProviderData");
struct ReadAsync {
    void* block = nullptr;
    std::uint32_t count = 0;
    std::size_t bytes = 0;
    std::vector<x2w_save_pack::Piece> pieces;
};
struct ReadRecord { void* block; std::uint32_t count; std::size_t bytes; };
std::vector<ReadRecord> read_records;  // accepted reads whose payload is still outstanding
SRWLOCK read_records_lock = SRWLOCK_INIT;
void remember_read(const ReadAsync& state)
{
    AcquireSRWLockExclusive(&read_records_lock);
    struct Unlock { ~Unlock() { ReleaseSRWLockExclusive(&read_records_lock); } } unlock;
    read_records.push_back({state.block, state.count, state.bytes});
}
void forget_read(void* block)
{
    AcquireSRWLockExclusive(&read_records_lock);
    struct Unlock { ~Unlock() { ReleaseSRWLockExclusive(&read_records_lock); } } unlock;
    for (std::size_t i = 0; i < read_records.size(); ++i)
        if (read_records[i].block == block) {
            read_records.erase(read_records.begin() + static_cast<std::ptrdiff_t>(i));
            break;
        }
}
bool pending_read(void* block, std::uint32_t& count, std::size_t& bytes)
{
    AcquireSRWLockShared(&read_records_lock);
    struct Unlock { ~Unlock() { ReleaseSRWLockShared(&read_records_lock); } } unlock;
    for (const ReadRecord& record : read_records)
        if (record.block == block) {
            count = record.count;
            bytes = record.bytes;
            return true;
        }
    return false;
}
HRESULT WINAPI read_provider(unsigned op, const void* raw)
{
    if (op == 0) return S_OK;  // Begin: the read runs after XAsyncBegin accepts the block
    if (!raw) return S_OK;
    const auto* data = static_cast<const AsyncProviderData*>(raw);
    auto* state = static_cast<ReadAsync*>(data->context);
    if (op == 2 && state && data->buffer && state->bytes <= data->buffer_size)
        x2w_save_pack::write(state->pieces, data->buffer);  // GetResult
    if (op == 4 && state) {  // Cleanup, including the runtime's cleanup of a failed read
        forget_read(state->block);
        delete state;
    }
    return S_OK;
}
// Names and bytes of one accepted read. The save lock is not held on return.
u64 snapshot_read(ReadAsync& state, u64 container, u64 names_pointer, std::uint32_t count)
{
    std::string container_name;
    {
        AcquireSRWLockShared(&save_lock);
        struct Unlock { ~Unlock() { ReleaseSRWLockShared(&save_lock); } } unlock;
        if (container >= save_container_handle && container - save_container_handle < max_save_handles)
            container_name = save_table.container_name(static_cast<std::size_t>(container - save_container_handle));
    }
    if (container_name.empty()) return e_invalidarg;
    if (count > SIZE_MAX / sizeof(const char*)) return e_invalidarg;
    if (count && x2w_local::probe_readable(static_cast<std::uintptr_t>(names_pointer), sizeof(const char*) * count) != sizeof(const char*) * count)
        return e_invalidarg;
    const auto* names = reinterpret_cast<const char* const*>(names_pointer);
    std::vector<std::string> wanted;
    wanted.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::string name;
        if (!names[i] || !read_save_name(reinterpret_cast<u64>(names[i]), name) || !x2w_save::valid_name(name))
            return e_invalidarg;
        wanted.push_back(std::move(name));
    }
    std::vector<std::vector<unsigned char>> data;
    const x2w_save::Status status = save_store().read_blobs(container_name, wanted, data);
    if (status != x2w_save::Status::ok) return save_status(status);
    if (data.size() != wanted.size() || data.size() > 0xffffffffu) return e_fail;
    state.pieces.resize(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        state.pieces[i].name = std::move(wanted[i]);
        state.pieces[i].bytes = std::move(data[i]);
    }
    state.count = static_cast<std::uint32_t>(state.pieces.size());
    if (!x2w_save_pack::bytes_needed(state.pieces, state.bytes)) return e_fail;
    return 0;
}
u64 async_read(int slot, u64 container, u64 names, u64 count_or_size, u64 block_or_buffer, u64 count_pointer)
{
    if (!real_query) return e_fail;
    void* service = nullptr;
    const auto query = real_query(&submit_async_family, &submit_async_family, &service);
    if (FAILED(query)) return static_cast<unsigned>(query);
    if (!service) return e_fail;
    auto** methods = *static_cast<void***>(service);
    struct ApiRef {
        void* object;
        ULONG (WINAPI* release)(void*);
        ~ApiRef() { release(object); }
    } api{service, reinterpret_cast<ULONG (WINAPI*)(void*)>(methods[2])};
    if (slot == 22) {
        auto* async = reinterpret_cast<void*>(container);  // XGameSaveReadBlobDataResult(async, size, data, &count)
        if (!async) return e_fail;
        const auto out = [](u64 pointer, std::size_t bytes) {
            return pointer && x2w_local::writable(reinterpret_cast<void*>(pointer), bytes);
        };
        std::uint32_t count = 0;
        std::size_t bytes = 0;
        if (pending_read(async, count, bytes)) {
            // A short or unusable buffer must not consume the result or change the caller's memory.
            if (count_or_size < bytes) return small_buffer;
            if (!out(count_pointer, sizeof(std::uint32_t)) || !out(block_or_buffer, bytes) ||
                block_or_buffer % alignof(x2w_save_pack::Blob)) return e_invalidarg;
            using Result = HRESULT(WINAPI*)(void*, void*, const void*, size_t, void*, size_t*);
            const auto hr = reinterpret_cast<Result>(methods[11])(
                service, async, read_identity, static_cast<std::size_t>(count_or_size),
                reinterpret_cast<void*>(block_or_buffer), nullptr);
            if (FAILED(hr)) return static_cast<unsigned>(hr);
            *reinterpret_cast<std::uint32_t*>(count_pointer) = count;
            return 0;
        }
        // No payload was kept: a missing blob or other failure, or a read of no blobs. The caller's
        // buffer is not passed through, so a failure cannot change it. A success with no blobs
        // reports a count of zero.
        using Result = HRESULT(WINAPI*)(void*, void*, const void*, size_t, void*, size_t*);
        const auto hr = reinterpret_cast<Result>(methods[11])(service, async, read_identity, 0, nullptr, nullptr);
        if (FAILED(hr)) return static_cast<unsigned>(hr);
        if (!out(count_pointer, sizeof(std::uint32_t))) return e_invalidarg;
        *reinterpret_cast<std::uint32_t*>(count_pointer) = 0;
        return 0;
    }
    auto* async = reinterpret_cast<void*>(block_or_buffer);
    if (!async) return e_fail;
    ReadAsync* state;
    try {
        state = new ReadAsync;
    } catch (const std::bad_alloc&) {
        return e_outofmemory;
    }
    using Begin = HRESULT(WINAPI*)(void*, void*, void*, const void*, const char*, decltype(&read_provider));
    const auto begun = reinterpret_cast<Begin>(methods[7])(service, async, state, read_identity, read_identity, read_provider);
    if (FAILED(begun)) {
        delete state;  // rejection: the runtime has not taken the context, and nothing was read
        return static_cast<unsigned>(begun);
    }
    using Complete = void(WINAPI*)(void*, void*, HRESULT, size_t);
    const auto complete = reinterpret_cast<Complete>(methods[10]);
    try {
        state->block = async;
        const u64 status = snapshot_read(*state, container, names, static_cast<std::uint32_t>(count_or_size));
        if (status) {
            complete(service, async, static_cast<HRESULT>(status), 0);  // no payload; context is released by Cleanup
            return S_OK;
        }
        if (state->bytes) remember_read(*state);
        complete(service, async, S_OK, state->bytes);
        return S_OK;
    } catch (const std::bad_alloc&) {
        complete(service, async, static_cast<HRESULT>(e_outofmemory), 0);
        return S_OK;
    } catch (const std::filesystem::filesystem_error&) {
        complete(service, async, static_cast<HRESULT>(e_fail), 0);
        return S_OK;
    }
}
u64 complete_async(u64 async_block)
{
    Proxy* user = user_provider >= 0 ? proxies[user_provider] : nullptr;
    if (!user || !async_block) return 0x80004005;
    const LONG index = InterlockedIncrement(&tracked_count) - 1;
    if (index < 64) tracked_blocks[index] = async_block;
    const Method add = reinterpret_cast<Method>((*reinterpret_cast<void***>(user->real))[7]);
    return add(user->real, 1 /* add default user silently */, async_block, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}
bool save_call(int slot, u64* args, u64* result)
{
    const auto out = [](u64 pointer, std::size_t bytes) { return pointer && x2w_local::writable(reinterpret_cast<void*>(pointer), bytes); };
    constexpr u64 ok = 0, blob_not_found = 0x80830008;
    *result = ok;
    switch (slot) {
    case 3:  // XGameSaveInitializeProvider(user, configuration id, sync on demand, &provider)
        if (out(args[3], 8)) *reinterpret_cast<u64*>(args[3]) = save_provider_handle;
        return true;
    case 4:  // XGameSaveInitializeProviderAsync(user, configuration id, sync on demand, async)
        *result = complete_async(args[3]);
        return true;
    case 5:  // XGameSaveInitializeProviderResult(async, &provider)
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = save_provider_handle;
        return true;
    case 6:  // close provider
        return true;
    case 17:  // XGameSaveCloseContainer(container): the table entry is reused once nothing needs it
        if (args[0] >= save_container_handle && args[0] - save_container_handle < max_save_handles) {
            AcquireSRWLockExclusive(&save_lock);
            save_table.close(static_cast<std::size_t>(args[0] - save_container_handle));
            ReleaseSRWLockExclusive(&save_lock);
        }
        return true;
    case 24:  // XGameSaveCloseUpdate(update)
        if (args[0] >= save_update_handle && args[0] - save_update_handle < max_save_handles) {
            AcquireSRWLockExclusive(&save_lock);
            save_table.end_update(static_cast<std::size_t>(args[0] - save_update_handle));
            ReleaseSRWLockExclusive(&save_lock);
        }
        return true;
    case 7:  // XGameSaveGetRemainingQuota(provider, &bytes)
        if (out(args[1], 8)) *reinterpret_cast<std::int64_t*>(args[1]) = 256ll << 20;
        return true;
    case 8:  // XGameSaveGetRemainingQuotaAsync(provider, async)
        *result = complete_async(args[1]);
        return true;
    case 9:  // XGameSaveGetRemainingQuotaResult(async, &bytes)
        if (out(args[1], 8)) *reinterpret_cast<std::int64_t*>(args[1]) = 256ll << 20;
        return true;
    case 12: case 13: case 14: case 15: case 18: case 19:  // container/blob queries: layout unknown, nothing reported
        return true;
    case 10: case 11: {  // XGameSaveDeleteContainer(provider, name), ...Async(provider, name, async)
        std::string name;
        if (read_save_name(args[1], name)) save_store().delete_container(name);
        if (slot == 11) *result = complete_async(args[2]);
        return true;
    }
    case 16: {  // XGameSaveCreateContainer(provider, name, &container)
        std::string name;
        u64 handle = 0;
        if (!read_save_name(args[1], name)) *result = e_invalidarg;
        else *result = open_container(name, &handle);
        if (!*result && out(args[2], 8)) *reinterpret_cast<u64*>(args[2]) = handle;
        return true;
    }
    case 20:  // XGameSaveReadBlobData(container, blobs, &count, size, data)
        *result = blob_not_found;
        return true;
    case 21:  // XGameSaveReadBlobDataAsync(container, names, count, async)
        *result = async_read(slot, args[0], args[1], args[2], args[3], 0);
        return true;
    case 22:  // XGameSaveReadBlobDataResult(async, size, data, &count)
        *result = async_read(slot, args[0], 0, args[1], args[2], args[3]);
        return true;
    case 23: {  // XGameSaveCreateUpdate(container, display name, &update)
        u64 handle = 0;
        if (args[0] >= save_container_handle && args[0] - save_container_handle < max_save_handles) {
            AcquireSRWLockExclusive(&save_lock);
            const long update = save_table.begin_update(static_cast<std::size_t>(args[0] - save_container_handle));
            ReleaseSRWLockExclusive(&save_lock);
            if (update >= 0) handle = save_update_handle + static_cast<u64>(update);
        }
        if (!handle) *result = e_outofmemory;
        else if (out(args[2], 8)) *reinterpret_cast<u64*>(args[2]) = handle;
        return true;
    }
    case 25: case 26: {
        // Hypothesis (public header order): SubmitBlobWrite(update, blob name, data, size) and
        // SubmitBlobDelete(update, blob name). The action is only kept with the update; nothing
        // reaches the disk before the update is submitted, and an update closed without a submit
        // leaves the save as it was. Only an open update is acted on; any other argument is
        // accepted and dropped as before.
        std::string blob;
        if (update_container(args[0]).empty() || !read_save_name(args[1], blob)) return true;
        x2w_save::Action action;
        action.remove = slot == 26;
        action.blob = blob;
        if (!x2w_save::valid_name(blob)) {
            *result = e_invalidarg;
            return true;
        }
        if (!action.remove) {
            const std::size_t size = static_cast<std::size_t>(args[3]);
            if (size > x2w_save::max_blob_bytes || (size && x2w_local::probe_readable(static_cast<std::uintptr_t>(args[2]), size) != size)) {
                *result = e_invalidarg;
                return true;
            }
            const auto* bytes = reinterpret_cast<const unsigned char*>(args[2]);
            if (size) action.data.assign(bytes, bytes + size);
        }
        AcquireSRWLockExclusive(&save_lock);
        const bool staged = args[0] >= save_update_handle && save_table.stage(static_cast<std::size_t>(args[0] - save_update_handle), std::move(action));
        ReleaseSRWLockExclusive(&save_lock);
        if (!staged) *result = e_outofmemory;
        return true;
    }
    case 27:  // XGameSaveSubmitUpdate(update): publishes the staged actions, all or none
        *result = submit_update(args[0]);
        return true;
    case 28:  // XGameSaveSubmitUpdateAsync(update, async)
        *result = async_submit(slot, args[0], args[1]);
        return true;
    case 29:  // XGameSaveSubmitUpdateResult(async): consume this block's completion
        *result = async_submit(slot, 0, args[0]);
        return true;
    default:
        *result = 0x80004001;  // E_NOTIMPL
        return true;
    }
}

// ---- other areas, answered as an offline console with one installed package --------------------
// Which provider is which area, and what an offline console answers, follow the public VIctory
// project's runtime notes (github.com/Marvisak/VIctory, used as a reference for facts only).
unsigned env_number(const char* name, unsigned fallback, int base)
{
    char text[32]{};
    return GetEnvironmentVariableA(name, text, sizeof(text)) ? static_cast<unsigned>(std::strtoul(text, nullptr, base)) : fallback;
}
bool area_call(const GUID& provider_id, int slot, u64* args, u64* result)
{
    const unsigned long provider = provider_id.Data1;
    const auto out = [](u64 pointer, std::size_t bytes) { return pointer && x2w_local::writable(reinterpret_cast<void*>(pointer), bytes); };
    if (provider == 0x973a344e && slot == 3) {  // title id (a 32-bit value from the title's manifest: X2W_TITLE_ID, hex)
        const unsigned id = env_number("X2W_TITLE_ID", 0, 16);
        if (!id) return false;
        if (out(args[0], 4)) *reinterpret_cast<std::uint32_t*>(args[0]) = id;
        *result = 0;
        return true;
    }
    if (provider == 0xaf406016 && slot == 3) {  // the running package's identifier (size, buffer): the title id in hex
        char text[16];
        const int length = std::snprintf(text, sizeof(text), "%08X", env_number("X2W_TITLE_ID", 0, 16));
        if (args[0] <= static_cast<u64>(length) || !out(args[1], static_cast<std::size_t>(length) + 1)) {
            *result = 0x8007007A;
            return true;
        }
        std::memcpy(reinterpret_cast<void*>(args[1]), text, static_cast<std::size_t>(length) + 1);
        *result = 0;
        return true;
    }
    if (provider == 0xaf406016 && slot >= 5 && slot <= 38) {
        // The rest of the package area, for one fully installed package (the title itself).
        static const unsigned char monitor = 0;
        const u64 monitor_handle = reinterpret_cast<u64>(&monitor);
        constexpr u64 no_package = 0x89245208;
        *result = 0;
        switch (slot) {
        case 5:  // create an installation monitor (id, selector count, selectors, interval, queue, &monitor)
            if (out(args[5], 8)) *reinterpret_cast<u64*>(args[5]) = monitor_handle;
            return true;
        case 6: case 14: case 19:  // close monitor, change install order, uninstall chunks: accepted
            return true;
        case 7:  // progress (monitor, &progress): everything installed
            if (out(args[1], 32)) {
                const u64 bytes = u64{200} << 30;
                u64 words[4] = {bytes, bytes, bytes, 0x0101 /* launchable, completed */};
                std::memcpy(reinterpret_cast<void*>(args[1]), words, sizeof(words));
            }
            return true;
        case 9:  // register for progress changes (monitor, context, callback, &token): never changes
            if (out(args[3], 8)) *reinterpret_cast<u64*>(args[3]) = 1;
            return true;
        case 10:  // unregister
            *result = 1;
            return true;
        case 12:  // chunk availability (id, count, selectors, &availability): 0 = ready
            if (out(args[3], 4)) *reinterpret_cast<std::uint32_t*>(args[3]) = 0;
            return true;
        case 16:  // install chunks (id, count, selectors, interval, suppress confirmation, async)
            *result = complete_async(args[5]);
            return true;
        case 17:  // result of that: a monitor
            if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = monitor_handle;
            return true;
        case 18:  // download estimate (id, count, selectors, &bytes, &needs confirmation): nothing to download
            if (out(args[3], 8)) *reinterpret_cast<u64*>(args[3]) = 0;
            if (out(args[4], 1)) *reinterpret_cast<bool*>(args[4]) = false;
            return true;
        case 34:  // uninstall another package: refused
            return true;
        case 37:  // mount another package (id, async): none is installed
            *result = complete_async(args[1]);
            return true;
        case 38:
            *result = no_package;
            return true;
        default:
            return false;
        }
    }
    if (provider == 0xaf406016 && slot == 4) {  // is the title a package: yes
        *result = 1;
        return true;
    }
    if (provider == 0xaf406016 && slot == 39) {  // enumerate packages: none besides the title
        *result = 0;
        return true;
    }
    if (provider == 0x37e56907 && slot == 15) {  // connectivity hint: stack initialized, no connectivity
        if (out(args[0], 16)) {
            const std::uint32_t words[4] = {1 /* level: none */, 0 /* cost: unknown */, 0, 1 /* initialized; other flags clear */};
            std::memcpy(reinterpret_cast<void*>(args[0]), words, sizeof(words));
        }
        *result = 0;
        return true;
    }
    if (slot == 3 && IsEqualGUID(provider_id, store_provider_id)) {
        // XStoreCreateContext(user, &context). Vtable slot 3 of XStoreImpl
        // (0dd112ac-7c24-448c-b92b-3960fb5bd30c): QueryInterface, AddRef, Release, then this.
        // The public GDK takes the console user and writes an XStoreContextHandle. The VIctory
        // runtime notes (github.com/Marvisak/VIctory, facts only) answer it offline by writing one
        // context and returning success; catalog, licence and purchase methods stay unanswered.
        // Only the local user is answered. A different user is refused and the output is left
        // unchanged. A buffer that cannot be written is E_INVALIDARG.
        if (args[0] != local_user_handle) {
            *result = 0x80004005;
            return true;
        }
        if (!out(args[1], 8)) {
            *result = 0x80070057;
            return true;
        }
        *reinterpret_cast<u64*>(args[1]) = store_context_handle;
        *result = 0;
        return true;
    }
    return false;
}

// Source-built check (tests/startup_stub_consumer.cpp). Both switches are required, as they are
// for a title: the proxy is installed only for X2W_RUNTIME_PROXY, and this answer only for the
// fake user. Returns 0 when the local user receives the context, a mutated provider still
// fails with E_FAIL without writing, and a non-local user is refused the same way.
int store_method3_check()
{
    if (!x2w_local::env_on("X2W_RUNTIME_PROXY") || !fake_user()) return 2;
    if (provider_count >= max_providers) return 3;
    const int index = provider_count++;
    provider_ids[index] = store_provider_id;
    Proxy proxy{nullptr, nullptr, index};

    u64 context = 0;
    const u64 answered = forward<3>(&proxy, local_user_handle, reinterpret_cast<u64>(&context), 0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (answered != 0 || context != store_context_handle) return 4;

    u64 untouched = 0x1111111111111111ull;
    provider_ids[index].Data4[7] ^= 1;  // same first word, different provider
    const u64 refused = forward<3>(&proxy, local_user_handle, reinterpret_cast<u64>(&untouched), 0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (refused != 0x80004005 || untouched != 0x1111111111111111ull) return 5;
    provider_ids[index] = store_provider_id;

    u64 foreign_out = 0x1111111111111111ull;
    const u64 foreign_user = local_user_handle ^ 1ull;
    const u64 foreign = forward<3>(&proxy, foreign_user, reinterpret_cast<u64>(&foreign_out), 0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (foreign != 0x80004005 || foreign_out != 0x1111111111111111ull) return 7;

    const u64 missing = forward<3>(&proxy, local_user_handle, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (missing != 0x80070057) return 6;

    const auto desktop = +[](void*, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 {
        return 0x1234;
    };
    void* methods[5] = {};
    methods[4] = reinterpret_cast<void*>(desktop);
    void** real = methods;
    proxy.real = &real;
    for (int position = 0; position < 4; ++position) {
        u64 arguments[4] = {};
        arguments[position] = context;
        const u64 blocked = forward<4>(&proxy, arguments[0], arguments[1], arguments[2], arguments[3], 0, 0, 0, 0, 0, 0, 0);
        if (blocked != 0x80004005) return 8;
    }
    // A foreign value still reaches the desktop implementation: the guard is token-specific.
    if (forward<4>(&proxy, context ^ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) != 0x1234) return 9;

    std::printf("store context %016llx\n", static_cast<unsigned long long>(context));
    std::printf("store refused %08llx\n", static_cast<unsigned long long>(refused & 0xffffffffu));
    std::printf("store user refused %08llx\n", static_cast<unsigned long long>(foreign & 0xffffffffu));
    return 0;
}

// Persistent local storage: the root of a drive (X2W_STORAGE_PATH, e.g. "S:\\", mapped by the run
// script onto a folder outside the game) with the quota from the manifest (X2W_STORAGE_MB).
struct StorageObject { void* const* vtable; };
// The value is read once. It gets a trailing separator when it has none (storage_root.h). A value
// that does not fit is refused with a log line and the default is used: handing the executable a
// cut-off or empty root would send its files somewhere else without a word.
const char* storage_path()
{
    static char path[x2w_startup::storage_root_capacity];
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, [](PINIT_ONCE, PVOID, PVOID*) -> BOOL {
        wchar_t value[x2w_startup::storage_root_capacity + 1];
        const DWORD capacity = static_cast<DWORD>(sizeof(value) / sizeof(*value));
        const DWORD length = GetEnvironmentVariableW(L"X2W_STORAGE_PATH", value, capacity);
        if (length >= capacity)
            x2w_local::logf("x2w-runtime X2W_STORAGE_PATH has %lu characters, more than the %zu supported: using S:\\",
                            static_cast<unsigned long>(length - 1), x2w_startup::storage_root_capacity - 2);
        else if (length && !x2w_startup::storage_root_utf8(value, path))
            x2w_local::logf("x2w-runtime X2W_STORAGE_PATH is invalid UTF-16 or does not fit as UTF-8 with its trailing separator: using S:\\");
        if (!path[0]) std::strcpy(path, "S:\\");
        return TRUE;
    }, nullptr, nullptr);
    return path;
}
HRESULT __stdcall storage_query(StorageObject* self, const GUID*, void** out) { if (out) *out = self; return S_OK; }
u64 __stdcall storage_ref(StorageObject*) { return 1; }
HRESULT __stdcall storage_path_size(StorageObject*, std::size_t* out)
{
    if (out) *out = std::strlen(storage_path()) + 1;
    x2w_local::logf("x2w-runtime storage path size -> %zu", std::strlen(storage_path()) + 1);
    return S_OK;
}
HRESULT __stdcall storage_get_path(StorageObject*, std::size_t size, char* buffer, std::size_t* used)
{
    const std::size_t need = std::strlen(storage_path()) + 1;
    if (used) *used = need;
    if (!buffer || size < need) return static_cast<HRESULT>(0x8007007A);
    std::memcpy(buffer, storage_path(), need);
    x2w_local::logf("x2w-runtime storage path -> %s", storage_path());
    return S_OK;
}
HRESULT __stdcall storage_space(StorageObject*, u64* out)
{
    const u64 total = u64{env_number("X2W_STORAGE_MB", 1024, 10)} << 20;
    if (out) {
        out[0] = total;  // writable now
        out[1] = total;  // left in the quota
        out[2] = 0;      // used
        out[3] = total;  // quota
    }
    return S_OK;
}
HRESULT __stdcall storage_prompt(StorageObject*, u64, u64 block) { return static_cast<HRESULT>(complete_async(block)); }
HRESULT __stdcall storage_prompt_result(StorageObject*, u64) { return S_OK; }
void* storage_object()
{
    static void* const table[8] = {
        reinterpret_cast<void*>(&storage_query), reinterpret_cast<void*>(&storage_ref), reinterpret_cast<void*>(&storage_ref),
        reinterpret_cast<void*>(&storage_path_size), reinterpret_cast<void*>(&storage_get_path), reinterpret_cast<void*>(&storage_space),
        reinterpret_cast<void*>(&storage_prompt), reinterpret_cast<void*>(&storage_prompt_result)};
    static StorageObject object{table};
    return &object;
}
}  // namespace

int x2w_store_method3_check() { return store_method3_check(); }

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, void*)
{
    if (reason == DLL_PROCESS_ATTACH && x2w_local::env_on("X2W_RUNTIME_PROXY")) {
        fill_slots(std::make_index_sequence<max_slots>{});
        patch_imports();
        // Heartbeat: every 10 s, the runtime methods called since the last report and how often, so
        // a title that keeps polling one call (waiting for something) shows it after the first-call
        // logs have gone quiet.
        CreateThread(nullptr, 0, [](void*) -> DWORD {
            static LONG seen[max_providers][max_slots];
            for (int second = 10;; second += 10) {
                Sleep(10000);
                char line[1024];
                int n = std::snprintf(line, sizeof(line), "x2w-runtime calls %d s:", second);
                bool any = false;
                for (int p = 0; p < provider_count && p < max_providers; ++p)
                    for (int slot = 3; slot < max_slots; ++slot) {
                        const LONG now = call_counts[p][slot], delta = now - seen[p][slot];
                        seen[p][slot] = now;
                        if (delta > 0 && n < static_cast<int>(sizeof(line)) - 40) {
                            n += std::snprintf(line + n, sizeof(line) - n, " %08lx:%d x%ld", provider_ids[p].Data1, slot, delta);
                            any = true;
                        }
                    }
                if (any) x2w_local::logf("%s", line);
            }
        }, nullptr, 0, nullptr);
    }
    return TRUE;
}
