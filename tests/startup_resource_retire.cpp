// X2W_D3D12_RELEASE_RETIRED=1: a committed 2D texture of the retire class (docs/startup-runtime.md#d3d12-resource-budget)
// gives up its desktop object only when no recorded list and no submission still refers to it and the release-time
// fence of every queue has passed. The retained column of the budget snapshot is the oracle for "the desktop object
// is destroyed": a ticket leaves it when its object dies. Readbacks go through the fixture's own desktop queue.
//
//   startup_resource_retire <warp|hardware> [off]
//     (X2W_D3D12_BUDGET=1 X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1, and
//      X2W_D3D12_RELEASE_RETIRED=1 unless the argument is "off")
//
// Cases with the switch on (each failure prints one "FAIL: ..." line and exits 5):
//   1 two lists record a copy of R, both closed and not submitted; R is released, every queue's release-time fence
//     passes: R stays; the lists are submitted one by one and both readbacks are right.
//   2 a list that is Reset, or destroyed, without a submission drops its hold and invents nothing.
//   3 a second submission waits on its own fence; its list is Reset; R stays until the fence has passed.
//   4 a destination released after recording (slots 17, 16, 19), a list destroyed while its submission is held, a copy queue.
//   5 an open recording: D3D12 refuses its Reset (the holds stay); a list destroyed while open drops its state first.
//   6 a release that no list names waits for the release-time fence of a held queue.
//   7 buffer, render target, title memory, a view, MakeResident and Present keep the old lifetime next to ones that retire.
// "off" runs the same steps without the switch: nothing is ever freed, and the probe export reports nothing.
// Present opens the output window of the stand-in, as the other present fixtures do; without a desktop session the
// window presenter fails quietly and the case still holds, since the resource leaves the class before any window work.
#include "startup_stubs/resource_budget.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <deque>
#include <vector>

using Microsoft::WRL::ComPtr;

struct RetireProbe {
    unsigned cache_entries;  // texture_uploads, pending_gpu_writes and view_targets entries naming the resource
    unsigned list_states;    // resources the open recording of the list moved out of COMMON
    unsigned list_holds;     // resources the list's recordings hold
};
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_budget_snapshot(x2w_budget::Snapshot*, SIZE_T);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_retire_probe(void*, void*, RetireProbe*);

namespace {
using namespace x2w_budget;

constexpr UINT side = 64, pitch = 256, image_bytes = side * pitch;
constexpr DWORD wait_ms = 10000;
constexpr int settle_limit = 200;  // bounded drains, no sleeping
constexpr int seed_source = 3, seed_sentinel = 9;

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

template<class T>
ComPtr<T> desktop_object(IUnknown* guest, ID3D12Device* device, IUnknown* object)
{
    // Same temporary private-data bridge as startup_resource_budget. The bridge reference is dropped on return.
    constexpr GUID key{0x0e34d52f, 0xd5a5, 0x4790, {0x91, 0xd4, 0x11, 0xa6, 0x7c, 0xbc, 0x2b, 0xc1}};
    ComPtr<T> output;
    if (FAILED(invoke(guest, 5, &key, object))) return output;
    ComPtr<IUnknown> reference;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT hr = device->GetPrivateData(key, &bytes, reference.GetAddressOf());
    invoke(guest, 5, &key, static_cast<IUnknown*>(nullptr));
    if (SUCCEEDED(hr)) reference.As(&output);
    return output;
}

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
    return SUCCEEDED(hr);
}

std::uint8_t pattern_byte(int seed, UINT i) { return static_cast<std::uint8_t>(i * 7 + seed * 13 + 1); }

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, bool warp)
{
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    wchar_t name[128]{};
    GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!name[0] || std::wcsstr(desc.Description, name)) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    return nullptr;
}

// A guest resource and what the budget must say about it.
struct Res {
    ComPtr<IUnknown> guest;
    const char* name = "";
    Category category = texture;
    UINT64 bytes = 0;
    bool released = false;  // the guest dropped its last reference: the ticket sits in the retained column
    bool freed = false;     // the desktop object is destroyed: the ticket is gone
};

struct GuestList {
    ComPtr<IUnknown> allocator, list;
};

// A guest queue with a completion fence the host can wait on.
struct GuestQueue {
    ComPtr<IUnknown> queue, done;
    ComPtr<ID3D12Fence> done_real;
    UINT64 value = 0;
};

struct Context {
    IUnknown* guest;
    ID3D12Device* device;
    ID3D12InfoQueue* debug;
    ComPtr<ID3D12CommandQueue> own_queue;
    ComPtr<ID3D12CommandAllocator> own_allocator;
    ComPtr<ID3D12GraphicsCommandList> own_list;
    ComPtr<ID3D12Fence> own_fence;
    UINT64 own_value = 0;
    HANDLE event = nullptr;
    ComPtr<ID3D12Resource> upload, readback;
    Snapshot base{};
    std::deque<Res> pool;
    GuestQueue direct, other, gated, copy;  // submissions, drains, held submissions, a copy queue
    GuestList drain_list;
    D3D12_RESOURCE_DESC desc{};
    UINT64 texture_bytes = 0;
    std::array<GUID, 1> iid_resource{__uuidof(ID3D12Resource)};

    bool wait(ID3D12Fence* fence, UINT64 value)
    {
        if (fence->GetCompletedValue() >= value) return true;
        if (FAILED(fence->SetEventOnCompletion(value, event))) return false;
        return WaitForSingleObject(event, wait_ms) == WAIT_OBJECT_0;
    }
    bool own_run()
    {
        if (!require(own_list->Close(), "own list close")) return false;
        ID3D12CommandList* lists[] = {own_list.Get()};
        own_queue->ExecuteCommandLists(1, lists);
        if (!require(own_queue->Signal(own_fence.Get(), ++own_value), "own signal")) return false;
        if (wait(own_fence.Get(), own_value)) return true;
        std::puts("FAIL: the fixture's own queue did not finish");
        return false;
    }
    void own_barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
        own_list->ResourceBarrier(1, &barrier);
    }
    D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* buffer) const
    {
        D3D12_TEXTURE_COPY_LOCATION location{buffer, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        location.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, side, side, 1, pitch}};
        return location;
    }
    // Writes the pattern of `seed` into a guest texture, through the desktop object that stands behind it.
    bool fill(IUnknown* texture_object, int seed)
    {
        const auto real = desktop_object<ID3D12Resource>(guest, device, texture_object);
        if (!real) return false;
        void* mapped = nullptr;
        if (!require(upload->Map(0, nullptr, &mapped), "upload map")) return false;
        for (UINT i = 0; i < image_bytes; ++i) static_cast<std::uint8_t*>(mapped)[i] = pattern_byte(seed, i);
        upload->Unmap(0, nullptr);
        if (!require(own_allocator->Reset(), "own allocator") || !require(own_list->Reset(own_allocator.Get(), nullptr), "own list")) return false;
        own_barrier(real.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        const D3D12_TEXTURE_COPY_LOCATION to{real.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        const auto from = footprint(upload.Get());
        own_list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        own_barrier(real.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        return own_run();
    }
    // True when the guest texture holds the pattern of `seed`.
    bool holds(IUnknown* texture_object, int seed, const char* what)
    {
        const auto real = desktop_object<ID3D12Resource>(guest, device, texture_object);
        if (!real) return false;
        if (!require(own_allocator->Reset(), "own allocator") || !require(own_list->Reset(own_allocator.Get(), nullptr), "own list")) return false;
        own_barrier(real.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        const D3D12_TEXTURE_COPY_LOCATION from{real.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        const auto to = footprint(readback.Get());
        own_list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        own_barrier(real.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        if (!own_run()) return false;
        void* mapped = nullptr;
        const D3D12_RANGE all{0, image_bytes};
        if (!require(readback->Map(0, &all, &mapped), "readback map")) return false;
        UINT wrong = image_bytes;
        for (UINT i = 0; i < image_bytes; ++i)
            if (static_cast<std::uint8_t*>(mapped)[i] != pattern_byte(seed, i)) { wrong = i; break; }
        const std::uint8_t got = wrong < image_bytes ? static_cast<std::uint8_t*>(mapped)[wrong] : 0;
        readback->Unmap(0, nullptr);
        if (wrong < image_bytes) std::printf("FAIL: readback of %s differs at byte %u: got %02x want %02x\n", what, wrong, got, pattern_byte(seed, wrong));
        return wrong == image_bytes;
    }

    // ---- guest objects ----
    Res& make_texture(const char* name, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, bool through_slot_27 = false)
    {
        D3D12_RESOURCE_DESC d = desc;
        d.Flags = flags;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        Res& r = pool.emplace_back();
        r.name = name;
        r.bytes = device->GetResourceAllocationInfo(0, 1, &d).SizeInBytes;
        const HRESULT hr = through_slot_27
            ? invoke(guest, 27, &heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, &iid_resource[0],
                     r.guest.GetAddressOf())
            : invoke(guest, 127, &heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0u, nullptr,
                     &iid_resource[0], r.guest.GetAddressOf());
        if (!require(hr, name)) pool.pop_back(), throw 4;
        return r;
    }
    Res& make_buffer(const char* name)
    {
        D3D12_RESOURCE_DESC d{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                              D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        Res& r = pool.emplace_back();
        r.name = name;
        r.category = buffer;
        r.bytes = device->GetResourceAllocationInfo(0, 1, &d).SizeInBytes;
        if (!require(invoke(guest, 27, &heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, &iid_resource[0],
                            r.guest.GetAddressOf()), name)) pool.pop_back(), throw 4;
        return r;
    }
    void release(Res& r)
    {
        r.guest.Reset();
        r.released = true;
    }
    bool make_queue(GuestQueue& q, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT)
    {
        const D3D12_COMMAND_QUEUE_DESC d{type};
        ComPtr<IUnknown> fence;
        if (!require(invoke(guest, 8, &d, &__uuidof(ID3D12CommandQueue), q.queue.GetAddressOf()), "queue") ||
            !require(invoke(guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), q.done.GetAddressOf()), "done fence"))
            return false;
        q.done_real = desktop_object<ID3D12Fence>(guest, device, q.done.Get());
        return q.done_real != nullptr;
    }
    // Everything the queue was given so far, including the release-time markers, has passed once this returns.
    bool sync(GuestQueue& q)
    {
        if (!require(invoke(q.queue.Get(), 14, q.done.Get(), ++q.value), "queue signal")) return false;
        if (wait(q.done_real.Get(), q.value)) return true;
        std::puts("FAIL: a queue did not reach its completion fence");
        return false;
    }
    bool make_list(GuestList& l, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT)
    {
        const unsigned list_desc[] = {static_cast<unsigned>(type), 2, 1, 0};
        return require(invoke(guest, 9, type, &__uuidof(ID3D12CommandAllocator), l.allocator.GetAddressOf()), "allocator") &&
               require(invoke(guest, 73, list_desc, l.allocator.Get(), nullptr, &__uuidof(ID3D12GraphicsCommandList),
                              l.list.GetAddressOf()), "list") &&
               require(invoke(l.list.Get(), 139, l.allocator.Get(), nullptr), "list reset");
    }
    // Reset onto a new allocator, as a frame loop does. Returns the HRESULT of the Reset.
    HRESULT reset_list(GuestList& l, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT)
    {
        ComPtr<IUnknown> next;
        if (!require(invoke(guest, 9, type, &__uuidof(ID3D12CommandAllocator), next.GetAddressOf()), "allocator")) return E_FAIL;
        const HRESULT hr = invoke(l.list.Get(), 139, next.Get(), nullptr);
        if (SUCCEEDED(hr)) l.allocator = next;
        return hr;
    }
    bool close(GuestList& l) { return require(invoke(l.list.Get(), 9), "list close"); }
    void execute(GuestQueue& q, GuestList& l)
    {
        IUnknown* lists[] = {l.list.Get()};
        invoke<void>(q.queue.Get(), 10, 1u, lists);
    }
    // ExecuteCommandLists on the second queue runs the retire drain; its empty list does nothing else.
    void drain() { execute(other, drain_list); }

    // ---- the oracle ----
    Snapshot expected() const
    {
        Snapshot e = base;
        for (const Res& r : pool) {
            if (r.freed) continue;
            Counts& c = r.released ? e.retained[r.category] : e.live[r.category];
            c.bytes += r.bytes;
            ++c.objects;
        }
        return e;
    }
    static bool same(Counts a, Counts b) { return a.bytes == b.bytes && a.objects == b.objects; }
    bool counts_match(Snapshot* actual_out = nullptr) const
    {
        Snapshot actual{};
        if (!x2w_d3d12_budget_snapshot(&actual, sizeof(actual))) return false;
        if (actual_out) *actual_out = actual;
        const Snapshot e = expected();
        bool pass = e.unmeasured == actual.unmeasured;
        for (unsigned i = 0; i < category_count; ++i) pass = pass && same(e.live[i], actual.live[i]) && same(e.retained[i], actual.retained[i]);
        return pass;
    }
    // The stage's expectation holds right now. `fail` is the one line a missing protection prints.
    bool expect(const char* stage, const char* fail)
    {
        Snapshot actual{};
        if (counts_match(&actual)) return true;
        const Snapshot e = expected();
        std::printf("FAIL: %s\n", fail);
        for (unsigned i = 0; i < category_count; ++i)
            if (!same(e.live[i], actual.live[i]) || !same(e.retained[i], actual.retained[i]))
                std::printf("  stage '%s' %s live=%llu/%llu expected=%llu/%llu retained=%llu/%llu expected=%llu/%llu\n", stage, names[i],
                            actual.live[i].bytes, actual.live[i].objects, e.live[i].bytes, e.live[i].objects,
                            actual.retained[i].bytes, actual.retained[i].objects, e.retained[i].bytes, e.retained[i].objects);
        if (e.unmeasured != actual.unmeasured) std::printf("  unmeasured=%llu expected=%llu\n", actual.unmeasured, e.unmeasured);
        return false;
    }
    // Drains until the counters equal the expectation, at most settle_limit times, without sleeping.
    bool settle(const char* stage, const char* fail)
    {
        for (int i = 0; i < settle_limit; ++i) {
            drain();
            if (counts_match()) return true;
        }
        return expect(stage, fail);
    }
    // A few drains in a row never change the counters: nothing may go early.
    bool stays(const char* stage, const char* fail)
    {
        for (int i = 0; i < 4; ++i) drain();
        return expect(stage, fail);
    }
    // Without the switch the export reports nothing (FALSE, zeros).
    mutable bool probe_available = false;
    RetireProbe probe(const Res* r, const GuestList* l) const
    {
        RetireProbe p{};
        probe_available = x2w_d3d12_retire_probe(r ? r->guest.Get() : nullptr, l ? l->list.Get() : nullptr, &p) != FALSE;
        return p;
    }
};

// ---- stages ----

// 1: two closed lists on R, R released while both are unsubmitted, then submitted one by one.
bool closed_lists(Context& c, bool on, Res& R, Res& D1, Res& D2, GuestList& L1, GuestList& L2)
{
    if (!c.make_list(L1) || !c.make_list(L2)) return false;
    invoke<void>(L1.list.Get(), 17, D1.guest.Get(), R.guest.Get());
    invoke<void>(L2.list.Get(), 17, D2.guest.Get(), R.guest.Get());
    if (!c.close(L1) || !c.close(L2)) return false;
    c.release(R);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
    if (on) {
        // The release-time fence of every queue has passed. The lists still name R.
        if (!c.stays("closed lists", "early release (closed lists)")) return false;
        const RetireProbe p = c.probe(&D1, &L1);
        if (p.cache_entries) {
            std::printf("FAIL: %u cache entries on a copy destination\n", p.cache_entries);
            return false;
        }
    } else if (!c.stays("closed lists, switch off", "switch off changed the lifetime")) return false;
    c.execute(c.direct, L1);
    if (!c.sync(c.direct) || !c.holds(D1.guest.Get(), seed_source, "D1 after the first submission")) return false;
    if (!c.holds(D2.guest.Get(), seed_sentinel, "D2 before its submission")) return false;
    if (!c.stays("first list submitted", "early release (first submission)")) return false;
    // A Reset on a fresh allocator drops L1's hold; L2 still names R.
    if (!SUCCEEDED(c.reset_list(L1))) return false;
    return c.stays("first list reset", "early release (second list)");
}

// 2: discard by Reset, discard by destroying the list; neither invents a submission.
bool discards(Context& c, bool on, Res& Rb, Res& D3, Res& Rc, Res& D4, GuestList& L3, GuestList& L4)
{
    if (!c.make_list(L3) || !c.make_list(L4)) return false;
    invoke<void>(L3.list.Get(), 17, D3.guest.Get(), Rb.guest.Get());
    invoke<void>(L4.list.Get(), 17, D4.guest.Get(), Rc.guest.Get());
    if (!c.close(L3) || !c.close(L4)) return false;
    c.release(Rb);
    c.release(Rc);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
    if (!c.stays("discard lists recorded", "early release (discard lists)")) return false;
    if (!SUCCEEDED(c.reset_list(L3))) return false;  // Reset of a closed, never submitted list
    if (on) Rb.freed = true;
    if (!c.settle("Reset discard", "a Reset list kept its resource")) return false;
    L4.list.Reset();  // the guest destroys a closed, never submitted list
    if (on) Rc.freed = true;
    if (!c.settle("destroy discard", "a destroyed list kept its resource")) return false;
    // Nothing was submitted for either list: the destinations still hold the sentinel.
    return c.holds(D3.guest.Get(), seed_sentinel, "D3 after a discarded Reset") &&
           c.holds(D4.guest.Get(), seed_sentinel, "D4 after a discarded destroy");
}

// 3: a second submission behind its own fence.
bool second_submission(Context& c, bool on, Res& R, Res& D2, GuestList& L2)
{
    ComPtr<IUnknown> gate;
    if (!require(invoke(c.guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), gate.GetAddressOf()), "gate fence")) return false;
    if (!require(invoke(c.gated.queue.Get(), 17, gate.Get(), UINT64{1}), "queue wait")) return false;
    c.execute(c.gated, L2);
    if (!SUCCEEDED(c.reset_list(L2))) return false;  // the list's hold falls; only the submission names R
    if (!c.stays("held submission", "early release (submission fence)")) return false;
    if (!require(invoke(gate.Get(), 10, UINT64{1}), "gate signal") || !c.sync(c.gated)) return false;
    if (on) R.freed = true;
    if (!c.settle("submission passed", "R kept after its submission passed")) return false;
    return c.holds(D2.guest.Get(), seed_source, "D2 after the second submission");
}

// 4a: destination released after recording (CopyResource), the destination only; the second copy checks the run.
bool destination_released(Context& c, bool on, Res& Rs, Res& D5, Res& Dout5, GuestList& L5)
{
    if (!c.make_list(L5)) return false;
    invoke<void>(L5.list.Get(), 17, D5.guest.Get(), Rs.guest.Get());
    invoke<void>(L5.list.Get(), 17, Dout5.guest.Get(), Rs.guest.Get());
    if (!c.close(L5)) return false;
    c.release(D5);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
    if (!c.stays("destination released", "early release (copy destination)")) return false;
    c.execute(c.direct, L5);
    if (!c.sync(c.direct) || !c.holds(Dout5.guest.Get(), seed_source, "Dout5")) return false;
    if (!c.stays("destination submitted", "early release (copy destination, submitted)")) return false;
    if (!SUCCEEDED(c.reset_list(L5))) return false;
    if (on) D5.freed = true;
    return c.settle("destination list reset", "a Reset list kept its destination");
}

// 4b: CopyTextureRegion (slot 16): both ends released after recording.
bool texture_region(Context& c, bool on, Res& R10, Res& D10, Res& Dout10, GuestList& L)
{
    if (!c.make_list(L)) return false;
    D3D12_TEXTURE_COPY_LOCATION to{static_cast<ID3D12Resource*>(static_cast<void*>(D10.guest.Get())), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    D3D12_TEXTURE_COPY_LOCATION out{static_cast<ID3D12Resource*>(static_cast<void*>(Dout10.guest.Get())), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    D3D12_TEXTURE_COPY_LOCATION from{static_cast<ID3D12Resource*>(static_cast<void*>(R10.guest.Get())), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    invoke<void>(L.list.Get(), 16, &to, 0u, 0u, 0u, &from, static_cast<const D3D12_BOX*>(nullptr));
    invoke<void>(L.list.Get(), 16, &out, 0u, 0u, 0u, &from, static_cast<const D3D12_BOX*>(nullptr));
    if (!c.close(L)) return false;
    // The recording must not leave cache entries behind that name these resources by their raw pointer.
    unsigned entries = 0;
    for (const Res* r : {&R10, &D10, &Dout10}) entries += c.probe(r, &L).cache_entries;
    if (entries != 0 || c.probe_available != on) {
        std::printf("FAIL: %u cache entries name the resources of a texture copy (switch %s, probe %s)\n", entries,
                    on ? "on" : "off", c.probe_available ? "available" : "off");
        return false;
    }
    c.release(R10);
    c.release(D10);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
    if (!c.stays("texture region released", "early release (texture copy)")) return false;
    c.execute(c.direct, L);
    if (!c.sync(c.direct) || !c.holds(Dout10.guest.Get(), seed_source, "Dout10")) return false;
    if (!SUCCEEDED(c.reset_list(L))) return false;
    if (on) R10.freed = D10.freed = true;
    return c.settle("texture region list reset", "a Reset list kept its texture copy resources");
}

// 4c: ResolveSubresource (slot 19), destination released after recording.
bool resolve_destination(Context& c, bool on, GuestList& L)
{
    D3D12_RESOURCE_DESC msaa = c.desc;
    msaa.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    msaa.SampleDesc = {4, 0};
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    Res& source = c.pool.emplace_back();
    source.name = "resolve source";
    source.bytes = c.device->GetResourceAllocationInfo(0, 1, &msaa).SizeInBytes;
    if (!require(invoke(c.guest, 127, &heap, D3D12_HEAP_FLAG_NONE, &msaa, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0u,
                        nullptr, &c.iid_resource[0], source.guest.GetAddressOf()), "msaa source")) return false;
    Res& Dr = c.make_texture("resolve destination");
    Res& Dkeep = c.make_texture("resolve destination, kept");
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } rtv_heap{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    ComPtr<IUnknown> heap_object;
    if (!require(invoke(c.guest, 14, &rtv_heap, &__uuidof(ID3D12DescriptorHeap), heap_object.GetAddressOf()), "rtv heap")) return false;
    const UINT64 handle = invoke<UINT64>(heap_object.Get(), 9);
    const D3D12_RENDER_TARGET_VIEW_DESC view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2DMS, {}};
    invoke<void>(c.guest, 20, source.guest.Get(), &view, handle);
    constexpr float magenta[]{1.0f, 0.0f, 1.0f, 1.0f};
    if (!c.make_list(L)) return false;
    invoke<void>(L.list.Get(), 52, handle, magenta, 0u, static_cast<const D3D12_RECT*>(nullptr));
    invoke<void>(L.list.Get(), 19, Dr.guest.Get(), 0u, source.guest.Get(), 0u, static_cast<UINT>(DXGI_FORMAT_R8G8B8A8_UNORM));
    invoke<void>(L.list.Get(), 19, Dkeep.guest.Get(), 0u, source.guest.Get(), 0u, static_cast<UINT>(DXGI_FORMAT_R8G8B8A8_UNORM));
    if (!c.close(L)) return false;
    {
        const RetireProbe p = c.probe(&Dr, &L);
        if (p.cache_entries) {
            std::printf("FAIL: the resolve destination has %u cache entries\n", p.cache_entries);
            return false;
        }
    }
    c.release(Dr);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
    if (!c.stays("resolve destination released", "early release (resolve destination)")) return false;
    c.execute(c.direct, L);
    if (!c.sync(c.direct)) return false;
    // Every pixel of the kept destination is the clear colour: the resolve ran.
    const auto real = desktop_object<ID3D12Resource>(c.guest, c.device, Dkeep.guest.Get());
    if (!real) return false;
    if (!require(c.own_allocator->Reset(), "own allocator") || !require(c.own_list->Reset(c.own_allocator.Get(), nullptr), "own list")) return false;
    c.own_barrier(real.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const D3D12_TEXTURE_COPY_LOCATION from{real.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    const auto to = c.footprint(c.readback.Get());
    c.own_list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    c.own_barrier(real.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (!c.own_run()) return false;
    void* mapped = nullptr;
    const D3D12_RANGE all{0, image_bytes};
    if (!require(c.readback->Map(0, &all, &mapped), "readback map")) return false;
    bool uniform = true;
    for (UINT i = 0; i < image_bytes; i += 4) {
        const auto* p = static_cast<const std::uint8_t*>(mapped) + i;
        uniform = uniform && p[0] == 255 && p[1] == 0 && p[2] == 255 && p[3] == 255;
    }
    c.readback->Unmap(0, nullptr);
    if (!uniform) {
        std::puts("FAIL: the resolved destination is not the clear colour");
        return false;
    }
    if (!SUCCEEDED(c.reset_list(L))) return false;
    if (on) Dr.freed = true;
    return c.settle("resolve list reset", "a Reset list kept its resolve destination");
}

// 5: a list destroyed while its submission is still held, and a copy queue.
bool in_flight(Context& c, bool on, Res& R, Res& D, Res& Dout, GuestList& L, GuestQueue& queue, D3D12_COMMAND_LIST_TYPE type,
               bool destroy_list, const char* fail)
{
    if (!c.make_list(L, type)) return false;
    invoke<void>(L.list.Get(), 17, D.guest.Get(), R.guest.Get());
    invoke<void>(L.list.Get(), 17, Dout.guest.Get(), R.guest.Get());
    if (!c.close(L)) return false;
    ComPtr<IUnknown> gate;
    if (!require(invoke(c.guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), gate.GetAddressOf()), "gate fence") ||
        !require(invoke(queue.queue.Get(), 17, gate.Get(), UINT64{1}), "queue wait"))
        return false;
    c.execute(queue, L);
    c.release(R);
    c.release(D);
    if (destroy_list) L.list.Reset();
    else if (!SUCCEEDED(c.reset_list(L, type))) return false;
    if (!c.stays("in flight", fail)) return false;
    if (!require(invoke(gate.Get(), 10, UINT64{1}), "gate signal") || !c.sync(queue)) return false;
    if (on) R.freed = D.freed = true;
    if (!c.settle("in flight passed", "resources kept after their submission passed")) return false;
    return c.holds(Dout.guest.Get(), seed_source, "output of the held submission");
}

// 6: a recording that is still open. D3D12 refuses a Reset of an open list, so that Reset must leave its holds; a list
// destroyed while open drops the recording state that names the resources before their holds fall.
bool open_list(Context& c, bool on, Res& R7, Res& D7, Res& R11, Res& D11, GuestList& L, GuestList& L11)
{
    if (!c.make_list(L)) return false;
    invoke<void>(L.list.Get(), 17, D7.guest.Get(), R7.guest.Get());
    {
        const RetireProbe p = c.probe(nullptr, &L);
        if (on && p.list_states != 2) {
            std::printf("FAIL: the open recording moved %u resources out of COMMON, expected 2\n", p.list_states);
            return false;
        }
    }
    c.release(R7);
    c.release(D7);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
    if (!c.stays("open list", "early release (open list)")) return false;
    const HRESULT hr = c.reset_list(L);
    std::printf("Reset of an open list: 0x%08lx\n", static_cast<unsigned long>(hr));
    if (SUCCEEDED(hr)) {
        const RetireProbe p = c.probe(nullptr, &L);
        if (on && (p.list_states != 0 || p.list_holds != 0)) {
            std::printf("FAIL: reset kept render state of a dropped recording (%u states, %u holds)\n", p.list_states, p.list_holds);
            return false;
        }
        if (on) R7.freed = D7.freed = true;
        if (!c.settle("open list reset", "a Reset open list kept its resources")) return false;
    } else if (!c.stays("open list, reset refused", "early release (refused Reset)")) return false;
    // The recording closes and runs without touching anything freed.
    if (!c.close(L)) return false;
    c.execute(c.direct, L);
    if (!c.sync(c.direct)) return false;

    if (!c.make_list(L11)) return false;
    void* const destroyed = L11.list.Get();  // the wrapper outlives its Release
    invoke<void>(L11.list.Get(), 17, D11.guest.Get(), R11.guest.Get());
    {
        const RetireProbe p = c.probe(nullptr, &L11);
        if (on && (p.list_states != 2 || p.list_holds != 2)) {
            std::printf("FAIL: the open recording has %u states and %u holds, expected 2 and 2\n", p.list_states, p.list_holds);
            return false;
        }
    }
    c.release(R11);
    c.release(D11);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
    if (!c.stays("open list destroyed", "early release (open list, before destroy)")) return false;
    L11.list.Reset();
    RetireProbe after{};
    x2w_d3d12_retire_probe(nullptr, destroyed, &after);
    if (on && (after.list_states != 0 || after.list_holds != 0)) {
        std::printf("FAIL: reset kept render state of a dropped recording (%u states, %u holds)\n", after.list_states, after.list_holds);
        return false;
    }
    if (on) R11.freed = D11.freed = true;
    return c.settle("open list destroyed", "a destroyed open list kept its resources");
}

// 7: a release with no list at all waits for the release-time fence of a held queue.
bool release_time_fence(Context& c, bool on)
{
    Res& Zg = c.make_texture("Zg");
    ComPtr<IUnknown> gate;
    if (!require(invoke(c.guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), gate.GetAddressOf()), "gate fence") ||
        !require(invoke(c.gated.queue.Get(), 17, gate.Get(), UINT64{1}), "queue wait"))
        return false;
    c.release(Zg);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.copy)) return false;
    if (!c.stays("release-time fence held", "early release (release-time fence)")) return false;
    if (!require(invoke(gate.Get(), 10, UINT64{1}), "gate signal") || !c.sync(c.gated)) return false;
    if (on) Zg.freed = true;
    return c.settle("release-time fence passed", "a resource kept after the release-time fence passed");
}

// 8: what keeps the old lifetime, next to what retires.
bool classes(Context& c, bool on)
{
    constexpr SIZE_T page = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    auto* title_memory = static_cast<char*>(VirtualAlloc(nullptr, page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!title_memory) return false;
    Res& Z127 = c.make_texture("Z127");
    Res& Z27 = c.make_texture("Z27", D3D12_RESOURCE_FLAG_NONE, true);
    Res& B = c.make_buffer("B (buffer)");
    Res& G = c.make_texture("G (render target)", D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    Res& V = c.make_texture("V (view)");
    Res& M = c.make_texture("M (MakeResident)");
    Res& P = c.make_texture("P (presented)");
    Res& T = c.pool.emplace_back();
    T.name = "T (title memory)";
    T.bytes = c.texture_bytes = c.device->GetResourceAllocationInfo(0, 1, &c.desc).SizeInBytes;
    if (!require(invoke(c.guest, 133, title_memory, &c.desc, D3D12_RESOURCE_STATE_COMMON, nullptr, &c.iid_resource[0],
                        T.guest.GetAddressOf()), "slot 133 texture")) return false;
    if (!c.expect("classes created", "the control resources are not all live")) return false;

    // A view, MakeResident and Present each take their resource out of the class.
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } view_heap{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1}, 0, 0};
    ComPtr<IUnknown> heap_object;
    if (!require(invoke(c.guest, 14, &view_heap, &__uuidof(ID3D12DescriptorHeap), heap_object.GetAddressOf()), "view heap")) return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    invoke<void>(c.guest, 18, V.guest.Get(), &srv, invoke<UINT64>(heap_object.Get(), 9));
    IUnknown* residents[] = {M.guest.Get()};
    if (!require(invoke(c.guest, 34, 1u, residents), "MakeResident")) return false;
    void* surface = P.guest.Get();
    const UINT64 plane[10]{0, 1, reinterpret_cast<UINT64>(&surface)};
    const UINT64 present_parameters[8]{};
    if (!require(invoke(c.direct.queue.Get(), 32, 1u, plane, present_parameters), "Present")) return false;

    struct Control { Res* resource; const char* fail; };
    const Control controls[] = {
        {&B, "a buffer was freed"}, {&G, "a render target was freed"},
        {&V, "a resource with a view was freed"}, {&M, "a resource passed to MakeResident was freed"},
        {&P, "the presented surface was freed"}};
    for (const Control& control : controls) {
        c.release(*control.resource);
        if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
        if (!c.stays(control.resource->name, control.fail)) return false;
    }
    c.release(Z127);
    c.release(Z27);
    c.release(T);
    if (!c.sync(c.direct) || !c.sync(c.other) || !c.sync(c.gated) || !c.sync(c.copy)) return false;
    if (on) Z127.freed = Z27.freed = T.freed = true;
    if (!c.settle("class released", "a resource of the class kept its desktop object")) return false;
    VirtualFree(title_memory, 0, MEM_RELEASE);
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware"))
        || (argc == 3 && std::strcmp(argv[2], "off"))) return 2;
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    char debug_env[2]{};
    const bool debug_on = GetEnvironmentVariableA("X2W_D3D12_DEBUG", debug_env, sizeof(debug_env)) && debug_env[0] == '1';
    if (debug_on) {
        ComPtr<ID3D12Debug> debug;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) return 77;
        debug->EnableDebugLayer();
    }
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(guest.GetAddressOf())), "Xbox device")
        || !require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "desktop device")) return 3;
    ComPtr<ID3D12InfoQueue> debug;
    if (debug_on && FAILED(device.As(&debug))) return 3;
    if (debug) {
        D3D12_MESSAGE_SEVERITY severities[] = {
            D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR, D3D12_MESSAGE_SEVERITY_WARNING};
        D3D12_INFO_QUEUE_FILTER filter{};
        filter.AllowList.NumSeverities = static_cast<UINT>(std::size(severities));
        filter.AllowList.pSeverityList = severities;
        if (FAILED(debug->AddStorageFilterEntries(&filter))) return 3;
        debug->ClearStoredMessages();
    }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter->GetDesc1(&adapter_desc);
    char switch_value[2]{};
    const bool switch_on = GetEnvironmentVariableA("X2W_D3D12_RELEASE_RETIRED", switch_value, sizeof(switch_value)) == 1 &&
        switch_value[0] == '1';
    const bool off_mode = argc == 3;
    if (switch_on == off_mode) {
        std::puts("FAIL: the X2W_D3D12_RELEASE_RETIRED environment does not match the requested mode");
        return 2;
    }
    std::printf("adapter=%ls debug_layer=%s mode=%s\n", adapter_desc.Description, debug ? "enabled" : "off", off_mode ? "off" : "on");
    const bool on = !off_mode;

    Context c{guest.Get(), device.Get(), debug.Get()};
    try {
        c.desc = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
                  D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        c.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const D3D12_COMMAND_QUEUE_DESC direct{};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, readback_heap{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, image_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                         D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        if (!c.event || !require(device->CreateCommandQueue(&direct, IID_PPV_ARGS(&c.own_queue)), "own queue") ||
            !require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&c.own_allocator)), "own allocator") ||
            !require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c.own_allocator.Get(), nullptr, IID_PPV_ARGS(&c.own_list)), "own list") ||
            !require(c.own_list->Close(), "own list close") ||
            !require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c.own_fence)), "own fence") ||
            !require(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                     nullptr, IID_PPV_ARGS(&c.upload)), "upload") ||
            !require(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, IID_PPV_ARGS(&c.readback)), "readback")) return 4;
        if (!c.make_queue(c.direct) || !c.make_queue(c.other) || !c.make_queue(c.gated) ||
            !c.make_queue(c.copy, D3D12_COMMAND_LIST_TYPE_COPY)) return 4;
        if (!c.make_list(c.drain_list) || !c.close(c.drain_list)) return 4;
        if (!x2w_d3d12_budget_snapshot(&c.base, sizeof(c.base))) {
            std::puts("FAIL: budget snapshot unavailable");
            return 4;
        }

        // Resources. Sources carry the source pattern, destinations the sentinel.
        Res& R = c.make_texture("R");
        Res& Rb = c.make_texture("Rb");
        Res& Rc = c.make_texture("Rc");
        Res& Rs = c.make_texture("Rs");
        Res& R7 = c.make_texture("R7");
        Res& R8 = c.make_texture("R8");
        Res& R9 = c.make_texture("R9");
        Res& R10 = c.make_texture("R10");
        Res& R11 = c.make_texture("R11");
        Res* sources[] = {&R, &Rb, &Rc, &Rs, &R7, &R8, &R9, &R10, &R11};
        Res& D1 = c.make_texture("D1");
        Res& D2 = c.make_texture("D2");
        Res& D3 = c.make_texture("D3");
        Res& D4 = c.make_texture("D4");
        Res& D5 = c.make_texture("D5");
        Res& Dout5 = c.make_texture("Dout5");
        Res& D7 = c.make_texture("D7");
        Res& D8 = c.make_texture("D8");
        Res& Dout8 = c.make_texture("Dout8");
        Res& D9 = c.make_texture("D9");
        Res& Dout9 = c.make_texture("Dout9");
        Res& D10 = c.make_texture("D10");
        Res& Dout10 = c.make_texture("Dout10");
        Res& D11 = c.make_texture("D11");
        Res* destinations[] = {&D1, &D2, &D3, &D4, &D5, &Dout5, &D7, &D8, &Dout8, &D9, &Dout9, &D10, &Dout10, &D11};
        for (Res* r : sources) if (!c.fill(r->guest.Get(), seed_source)) return 4;
        for (Res* r : destinations) if (!c.fill(r->guest.Get(), seed_sentinel)) return 4;
        if (!c.expect("allocated", "the resources are not all live")) return 5;

        GuestList L1, L2, L3, L4, L5, L6, L8, L9, L10, L11, L12;
        if (!closed_lists(c, on, R, D1, D2, L1, L2)) return 5;
        std::puts("1 closed lists: R stays while both lists are unsubmitted, then both readbacks");
        if (!discards(c, on, Rb, D3, Rc, D4, L3, L4)) return 5;
        std::puts("2 discard: a Reset list and a destroyed list free their resource without a submission");
        if (!second_submission(c, on, R, D2, L2)) return 5;
        std::puts("3 second submission: held behind its own fence, then freed");
        if (!destination_released(c, on, Rs, D5, Dout5, L5)) return 5;
        std::puts("4a copy destination released after recording stays until its list drops it");
        if (!texture_region(c, on, R10, D10, Dout10, L10)) return 5;
        std::puts("4b CopyTextureRegion: both ends held");
        if (!resolve_destination(c, on, L12)) return 5;
        std::puts("4c ResolveSubresource: destination held");
        if (!in_flight(c, on, R8, D8, Dout8, L8, c.gated, D3D12_COMMAND_LIST_TYPE_DIRECT, true, "early release (list destroyed in flight)")) return 5;
        std::puts("5a list destroyed while its submission is held");
        if (!in_flight(c, on, R9, D9, Dout9, L9, c.copy, D3D12_COMMAND_LIST_TYPE_COPY, false, "early release (copy queue)")) return 5;
        std::puts("5b copy queue");
        if (!open_list(c, on, R7, D7, R11, D11, L6, L11)) return 5;
        std::puts("6 open recording: a refused Reset keeps its holds, a destroyed open list drops them");
        if (!release_time_fence(c, on)) return 5;
        std::puts("7 release-time fence: a release with no list waits for the held queue");
        if (!classes(c, on)) return 5;
        std::puts("8 classes: buffer, render target, title memory, view, MakeResident and Present keep the old lifetime");
    } catch (int code) {
        return code;
    }
    const UINT64 errors = debug ? debug->GetNumStoredMessages() : 0;
    std::printf("debug_errors_or_warnings=%llu\n", errors);
    if (errors || FAILED(device->GetDeviceRemovedReason())) return 4;
    std::puts(on ? "release-retired checks passed" : "switch-off checks passed");
    return 0;
}
