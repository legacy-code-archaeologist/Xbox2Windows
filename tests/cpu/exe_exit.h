// End a CRT-free fixture EXE with its entry result (#281, #1375).
//
// When the entry thread of a /NODEFAULTLIB EXE returns, nothing ends the process while another thread is
// alive. An idle loader or thread-pool worker can then keep it running for about 30 s, and its exit code 0
// becomes the process status. exe_finish(code) ends the process with code when Windows started this image as
// the process EXE. In a host mapping (pe_map) it returns code to the caller, like finish_exe in exe_io.cpp.
//
// The helper adds no imports: it reads the process image from the PEB and resolves ntdll!RtlExitUserProcess
// from ntdll's export table, so fixtures whose import tables are asserted keep them unchanged.
#pragma once

#include <intrin.h>

#include <cstdint>

// Debug /RTC would make these inline functions call CRT initialization that a /NODEFAULTLIB EXE cannot serve.
#pragma runtime_checks("", off)

extern "C" const unsigned char __ImageBase;

namespace exe_exit {

inline const unsigned char* peb() { return reinterpret_cast<const unsigned char*>(__readgsqword(0x60)); }

template <typename T>
inline T at(const unsigned char* base, std::uint32_t offset) {
    return *reinterpret_cast<const T*>(base + offset);
}

// True only when Windows started this image as the process EXE. PEB::ImageBaseAddress alone is not enough:
// pe_map_run_main reports the mapped EXE there (#1151), so under pe_run_package it also equals &__ImageBase.
// pe_map leaves the PEB loader list alone, so there the first InLoadOrderModuleList entry is the host EXE.
// Natively the process EXE is the first entry. Both must name this image.
inline bool started_as_process() {
    if (at<const void*>(peb(), 0x10) != &__ImageBase) return false;
    const unsigned char* head = at<const unsigned char*>(peb(), 0x18) + 0x10;
    const unsigned char* first = at<const unsigned char*>(head, 0);
    return first != head && at<const void*>(first, 0x30) == &__ImageBase;  // DllBase
}

inline bool same_ascii(const wchar_t* name, std::uint16_t bytes, const char* expected) {
    std::uint16_t i = 0;
    for (; i < bytes / 2 && expected[i]; ++i) {
        wchar_t c = name[i];
        if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c - L'A' + L'a');
        if (c != static_cast<unsigned char>(expected[i])) return false;
    }
    return i == bytes / 2 && !expected[i];
}

inline bool same_name(const char* name, const char* expected) {
    for (; *name && *name == *expected; ++name, ++expected) {}
    return *name == *expected;
}

// ntdll's base from PEB::Ldr->InLoadOrderModuleList (LDR_DATA_TABLE_ENTRY: DllBase +0x30, BaseDllName +0x58).
inline const unsigned char* ntdll_base() {
    const unsigned char* head = at<const unsigned char*>(peb(), 0x18) + 0x10;
    for (const unsigned char* entry = at<const unsigned char*>(head, 0); entry != head;
         entry = at<const unsigned char*>(entry, 0)) {
        if (same_ascii(at<const wchar_t*>(entry, 0x60), at<std::uint16_t>(entry, 0x58), "ntdll.dll"))
            return at<const unsigned char*>(entry, 0x30);
    }
    return nullptr;
}

// A named, non-forwarded export of a mapped image, or nullptr.
inline const void* export_named(const unsigned char* base, const char* wanted) {
    const unsigned char* nt = base + at<std::uint32_t>(base, 0x3C);
    const std::uint32_t rva = at<std::uint32_t>(nt, 0x88), size = at<std::uint32_t>(nt, 0x8C);  // export directory
    if (!rva) return nullptr;
    const unsigned char* directory = base + rva;
    const std::uint32_t count = at<std::uint32_t>(directory, 0x18);
    const auto* functions = reinterpret_cast<const std::uint32_t*>(base + at<std::uint32_t>(directory, 0x1C));
    const auto* names = reinterpret_cast<const std::uint32_t*>(base + at<std::uint32_t>(directory, 0x20));
    const auto* ordinals = reinterpret_cast<const std::uint16_t*>(base + at<std::uint32_t>(directory, 0x24));
    for (std::uint32_t i = 0; i < count; ++i) {
        if (!same_name(reinterpret_cast<const char*>(base + names[i]), wanted)) continue;
        const std::uint32_t function = functions[ordinals[i]];
        return function - rva < size ? nullptr : base + function;  // a forwarder string is not code
    }
    return nullptr;
}

}  // namespace exe_exit

// Returns only in a host mapping, or if ntdll!RtlExitUserProcess cannot be resolved.
inline std::uint32_t exe_finish(std::uint32_t code) {
#ifndef RETURN_FROM_ENTRY  // control build: return from the entry point as before #1375
    if (exe_exit::started_as_process()) {
        if (const unsigned char* ntdll = exe_exit::ntdll_base()) {
            using ExitUserProcess = void(__stdcall*)(long);
            if (const void* exit = exe_exit::export_named(ntdll, "RtlExitUserProcess"))
                reinterpret_cast<ExitUserProcess>(const_cast<void*>(exit))(static_cast<long>(code));
        }
    }
#endif
    return code;
}
