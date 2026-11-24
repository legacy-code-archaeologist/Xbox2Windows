// Lifecycle, queue retirement, and refusal checks for the xapu.dll startup stand-in.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

using ConnectFn = long(__cdecl*)(const void* input, void* output, void** handle);
using GetResultCountFn = std::uint32_t(__cdecl*)(void* handle, std::uint64_t unused);
using DequeueResultFn = long(__cdecl*)(void* handle, void* result);
using EnqueueCommandFn = long(__cdecl*)(void* handle, const void* command, std::uint64_t r8, std::uint64_t r9);
using DisconnectFn = long(__cdecl*)(void* handle);

constexpr long hr_ok = 0;
constexpr long hr_pointer = static_cast<long>(0x80004003);
constexpr long hr_handle = static_cast<long>(0x80070006);
constexpr long hr_pending_results = static_cast<long>(0x8AC81001);
constexpr DWORD trap_connect_exit = 0xE2570300u;
constexpr std::size_t result_bytes = 0x58;

int failures = 0;

void check(bool condition, const char* label)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL: %s\n", label);
}

bool all_zero(const std::array<unsigned char, result_bytes>& bytes)
{
    for (unsigned char b : bytes) {
        if (b != 0) return false;
    }
    return true;
}

int run_trap_control(const char* dll_path)
{
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", nullptr);
    HMODULE module = LoadLibraryA(dll_path);
    if (!module) return 2;
    auto connect = reinterpret_cast<ConnectFn>(GetProcAddress(module, "XApuConnect"));
    if (!connect) return 3;
    std::array<unsigned char, 0x18> input{};
    std::array<unsigned char, 0x10> output{};
    void* handle = nullptr;
    connect(input.data(), output.data(), &handle);
    return 4;
}

void check_non_permissive_trap(const char* self_exe, const char* dll_path)
{
    std::string cmd = std::string("\"") + self_exe + "\" --trap-control \"" + dll_path + "\"";
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        check(false, "CreateProcessA for non-permissive trap control");
        return;
    }
    const DWORD wait_rc = WaitForSingleObject(pi.hProcess, 10000);
    DWORD exit_code = 0;
    const BOOL got_code = GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    check(wait_rc == WAIT_OBJECT_0, "non-permissive trap child exited within timeout");
    check(got_code && exit_code == trap_connect_exit, "non-permissive XApuConnect traps with 0xE2570300");
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--trap-control") == 0) return run_trap_control(argv[2]);
    if (argc != 2) return std::printf("usage: startup_xapu_queue_check <xapu.dll>\n"), 2;

    check_non_permissive_trap(argv[0], argv[1]);

    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", "1");
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return std::printf("FAIL: cannot load %s\n", argv[1]), 2;

    auto connect = reinterpret_cast<ConnectFn>(GetProcAddress(module, "XApuConnect"));
    auto get_count = reinterpret_cast<GetResultCountFn>(GetProcAddress(module, "XApuGetResultCount"));
    auto dequeue = reinterpret_cast<DequeueResultFn>(GetProcAddress(module, "XApuDequeueResult"));
    auto enqueue = reinterpret_cast<EnqueueCommandFn>(GetProcAddress(module, "XApuEnqueueCommand"));
    auto disconnect = reinterpret_cast<DisconnectFn>(GetProcAddress(module, "XApuDisconnect"));
    if (!connect || !get_count || !dequeue || !enqueue || !disconnect) {
        FreeLibrary(module);
        return std::printf("FAIL: missing export\n"), 2;
    }

    const std::array<unsigned char, 0x18> input{};
    std::array<unsigned char, 0x10> output{};
    const std::array<unsigned char, 0x60> cmd{};
    std::array<unsigned char, result_bytes> result{};
    void* bogus = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0xDEADBEEFULL));

    // Invalid-handle and null-pointer negative controls before any connection exists.
    check(connect(input.data(), output.data(), nullptr) == hr_pointer, "XApuConnect(null handle*) -> E_POINTER");
    check(get_count(nullptr, 0) == 0, "XApuGetResultCount(null) -> 0");
    check(get_count(bogus, 0) == 0, "XApuGetResultCount(bogus) -> 0");
    check(enqueue(nullptr, cmd.data(), 0, 0) == hr_handle, "XApuEnqueueCommand(null) -> E_HANDLE");
    check(enqueue(bogus, cmd.data(), 0, 0) == hr_handle, "XApuEnqueueCommand(bogus) -> E_HANDLE");
    check(dequeue(nullptr, result.data()) == hr_handle, "XApuDequeueResult(null handle) -> E_HANDLE");
    check(dequeue(bogus, result.data()) == hr_handle, "XApuDequeueResult(bogus handle) -> E_HANDLE");
    check(disconnect(nullptr) == hr_handle, "XApuDisconnect(null) -> E_HANDLE");
    check(disconnect(bogus) == hr_handle, "XApuDisconnect(bogus) -> E_HANDLE");

    // Positive lifecycle: connect, enqueue, count, and dequeue, including empty queues.
    std::array<unsigned char, result_bytes> sentinel;
    sentinel.fill(0xA5);
    void* h1 = nullptr;
    check(connect(input.data(), output.data(), &h1) == hr_ok && h1 != nullptr, "XApuConnect h1 succeeds");
    check(get_count(h1, 0) == 0, "initial XApuGetResultCount == 0");
    result = sentinel;
    check(dequeue(h1, result.data()) == hr_pending_results, "empty queue XApuDequeueResult -> XAPU_E_PENDING_RESULTS");
    check(result == sentinel, "empty dequeue preserves the result buffer");

    check(enqueue(h1, cmd.data(), 0x11, 0x22) == hr_ok, "enqueue #1 on h1");
    check(enqueue(h1, cmd.data(), 0x33, 0x44) == hr_ok, "enqueue #2 on h1");
    check(get_count(h1, 0) == 2, "XApuGetResultCount == 2 after 2 enqueues");

    check(dequeue(h1, nullptr) == hr_pointer, "XApuDequeueResult(null result*) -> E_POINTER");
    check(get_count(h1, 0) == 2, "failed dequeue(null) does not consume queued entry");

    result.fill(0xAA);
    check(dequeue(h1, result.data()) == hr_ok, "dequeue #1 on h1 succeeds");
    check(all_zero(result), "dequeue #1 zeroes 0x58-byte result buffer");
    check(get_count(h1, 0) == 1, "XApuGetResultCount == 1 after first dequeue");

    result.fill(0x55);
    check(dequeue(h1, result.data()) == hr_ok, "dequeue #2 on h1 succeeds");
    check(all_zero(result), "dequeue #2 zeroes 0x58-byte result buffer");
    check(get_count(h1, 0) == 0, "XApuGetResultCount == 0 after draining queue");
    result = sentinel;
    check(dequeue(h1, result.data()) == hr_pending_results, "drained queue XApuDequeueResult -> XAPU_E_PENDING_RESULTS");
    check(result == sentinel, "dequeue after draining preserves the result buffer");

    // Second independent handle + disconnect invalidation.
    void* h2 = nullptr;
    check(connect(input.data(), output.data(), &h2) == hr_ok && h2 != nullptr && h2 != h1, "XApuConnect h2 distinct");
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "enqueue on h1 before disconnect");
    check(enqueue(h2, cmd.data(), 0, 0) == hr_ok, "enqueue #1 on h2");
    check(enqueue(h2, cmd.data(), 0, 0) == hr_ok, "enqueue #2 on h2");

    check(disconnect(h1) == hr_ok, "XApuDisconnect(h1) succeeds");
    check(disconnect(h1) == hr_handle, "repeat XApuDisconnect(h1) -> E_HANDLE");
    check(enqueue(h1, cmd.data(), 0, 0) == hr_handle, "enqueue after disconnect(h1) -> E_HANDLE");
    check(get_count(h1, 0) == 0, "get_count after disconnect(h1) -> 0");
    check(dequeue(h1, result.data()) == hr_handle, "dequeue after disconnect(h1) -> E_HANDLE");

    check(get_count(h2, 0) == 2, "disconnect(h1) preserves h2 queue");
    check(dequeue(h2, result.data()) == hr_ok, "h2 dequeue #1 succeeds");
    check(dequeue(h2, result.data()) == hr_ok, "h2 dequeue #2 succeeds");
    check(dequeue(h2, result.data()) == hr_pending_results, "h2 drained -> XAPU_E_PENDING_RESULTS");
    check(disconnect(h2) == hr_ok, "XApuDisconnect(h2) succeeds");

    FreeLibrary(module);
    std::printf(failures ? "startup_xapu_queue_check: %d failure(s)\n" : "startup_xapu_queue_check: ok\n", failures);
    return failures ? 1 : 0;
}
