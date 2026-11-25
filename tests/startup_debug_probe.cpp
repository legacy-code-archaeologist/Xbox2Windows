// Input for the startup debugger's diagnostics check (tests/startup_debug_check.py). It reaches two
// breakpoints in its own image, the way a retail image keeps soft asserts, reserves one large
// range, reports whether the loader gave it the debug heap, stays alive for a moment and exits 0.
// With STARTUP_DEBUG_PROBE_DLL=<file> and STARTUP_DEBUG_PROBE_EXPORT=<name> it also loads that DLL
// and calls the export kCalls times (the call-hook case; tests/startup_debug_calls.cpp).
// No GPU and no other input is involved.
#include <windows.h>
#include <intrin.h>

#include <cstdio>
#include <cstdlib>

namespace {
volatile int reached;

__declspec(noinline) void first_assert()
{
    reached = 1;
    __debugbreak();
}

__declspec(noinline) void second_assert()
{
    reached = 2;
    __debugbreak();
}

__declspec(noinline) void* reserve_large()
{
    // 256 MiB, reserved only: nothing is committed.
    return VirtualAlloc(nullptr, SIZE_T{256} << 20, MEM_RESERVE, PAGE_NOACCESS);
}

// The heap bits of the PEB's NtGlobalFlag (x64 offset 0xBC): tail check, free check and parameter
// validation (0x70). The loader sets them for a process created under a debugger unless the
// environment has _NO_DEBUG_HEAP=1.
unsigned long debug_heap_flags()
{
    const auto* const peb = reinterpret_cast<const unsigned char*>(__readgsqword(0x60));
    return *reinterpret_cast<const unsigned long*>(peb + 0xBC) & 0x70;
}

constexpr int kCalls = 7;

// Returns the number of export calls made, or -1 when the DLL or the export is missing.
int call_export(const char* dll, const char* name)
{
    HMODULE const module = LoadLibraryA(dll);
    using Export = LONG (*)(void*);
    auto const function = module ? reinterpret_cast<Export>(GetProcAddress(module, name)) : nullptr;
    if (!function) {
        return -1;
    }
    int made = 0;
    for (int i = 0; i < kCalls; ++i) {
        made += function(nullptr) > 0 ? 1 : 0;
    }
    return made;
}
}  // namespace

int main()
{
    first_assert();
    second_assert();
    void* const range = reserve_large();
    std::printf("debug-probe reserved %s\n", range ? "ok" : "failed");
    std::printf("debug-probe heap flags 0x%lx\n", debug_heap_flags());
    const char* const dll = std::getenv("STARTUP_DEBUG_PROBE_DLL");
    const char* const name = std::getenv("STARTUP_DEBUG_PROBE_EXPORT");
    if (dll && name) {
        std::printf("debug-probe calls %d\n", call_export(dll, name));
    }
    std::fflush(stdout);
    Sleep(3000);
    return 0;
}
