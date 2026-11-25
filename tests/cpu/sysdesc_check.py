"""Differential, /O2, mutant, UMIP-aware, and rejection tests for cpu_sysdesc_aot."""

import ctypes
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from iced_x86 import (
    Code,
    Decoder,
    Encoder,
    Formatter,
    FormatterSyntax,
    Instruction,
    MemoryOperand,
    Register as R,
)
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_sysdesc_aot import lift_code, translate  # noqa: E402


TRANSLATED_EXPORTS = (
    "op_sgdt",
    "op_sgdt_disp",
    "op_sidt",
    "op_sidt_disp",
    "op_sldt_r16",
    "op_sldt_r32",
    "op_sldt_r64",
    "op_sldt_m16",
    "op_sldt_m16_disp",
    "op_str_r16",
    "op_str_r32",
    "op_str_r64",
    "op_str_m16",
    "op_str_m16_disp",
    "op_smsw_r16",
    "op_smsw_r32",
    "op_smsw_r64",
    "op_smsw_m16",
    "op_smsw_m16_disp",
    "op_sysdesc_flags_preserve",
    "op_sysdesc_discarded_stores",
    "op_sysdesc_unused_reg",
)

SENTINELS = (
    0x1122334455667788,
    0xA5A5A5A5DEADBEEF,
    0x00000000FFFFFFFF,
    0xFFFFFFFF00000000,
)

EXPECTED_SLDT = 0x0000
EXPECTED_STR = 0x0040
EXPECTED_SMSW32 = 0x80050033
EXPECTED_LIFTED_GDT_LIMIT = 0x007F
EXPECTED_LIFTED_GDT_BASE = 0xFFFFF80000000000


def bind_lib(path, prefix=""):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    fns = {}
    for name in TRANSLATED_EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]
        fn.restype = ctypes.c_uint64
        fns[name] = fn
    if prefix:
        umip = lib.host_has_umip
        umip.argtypes = []
        umip.restype = ctypes.c_uint32
        fns["host_has_umip"] = umip
    return lib, fns


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def logical_cpus():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
    kernel32.GetCurrentProcessorNumber.restype = ctypes.c_uint32
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask")
    cpus = [k for k in range(process.value.bit_length()) if (process.value >> k) & 1]
    return kernel32, process.value, cpus


def pin_to_cpu(kernel32, cpu):
    prev = kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), 1 << cpu)
    if not prev:
        raise OSError(ctypes.get_last_error(), "SetThreadAffinityMask")
    for _ in range(200):
        running = kernel32.GetCurrentProcessorNumber()
        if running == cpu:
            return prev
        kernel32.Sleep(1)
    raise AssertionError(f"pinned to logical processor {cpu}, still running on {running}")


def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    ret_ins = Instruction.create(Code.RETNQ)
    ret_bytes = encode_seq(ret_ins)
    ld_rax = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.RDX)

    # 1. Default profile (host_cpu=False) MUST reject all 5 sysdesc instructions.
    for export_name in ("op_sgdt", "op_sidt", "op_sldt_r32", "op_str_r32", "op_smsw_r32"):
        try:
            translate(original_path, [export_name])
        except ValueError as exc:
            if "requires --host-cpu" not in str(exc):
                raise AssertionError(f"default profile {export_name}: unexpected error {exc!r}") from exc
        else:
            raise AssertionError(f"default profile unexpectedly accepted {export_name}")

    default_unit_cases = [
        ("default_sgdt", encode_seq(Instruction.create_mem(Code.SGDT_M1664, MemoryOperand(R.R8)), ld_rax, ret_ins)),
        ("default_sidt", encode_seq(Instruction.create_mem(Code.SIDT_M1664, MemoryOperand(R.R8)), ld_rax, ret_ins)),
        ("default_sldt", encode_seq(Instruction.create_reg(Code.SLDT_R32M16, R.EAX), ret_ins)),
        ("default_str", encode_seq(Instruction.create_reg(Code.STR_R32M16, R.EAX), ret_ins)),
        ("default_smsw", encode_seq(Instruction.create_reg(Code.SMSW_R32M16, R.EAX), ret_ins)),
    ]
    for label, raw in default_unit_cases:
        try:
            lift_code(label, 0x1000, raw, host_cpu=False)
        except ValueError as exc:
            if "requires --host-cpu" not in str(exc):
                raise AssertionError(f"{label}: expected 'requires --host-cpu', got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: default profile unexpectedly accepted sysdesc instruction")

    # 2. Raw encodings checked via Decoder+Formatter before lift_code(..., host_cpu=True).
    formatter = Formatter(FormatterSyntax.INTEL)
    raw_checks = [
        ("vex_vpaddw", bytes.fromhex("c5f1fdc2") + ret_bytes, "vpaddw xmm0,xmm1,xmm2", "unsupported VEX instruction"),
        ("xop_vpcmov", bytes.fromhex("8fe878a2c230") + ret_bytes, "vpcmov xmm0,xmm0,xmm2,xmm3", "unsupported XOP instruction"),
        ("seg_gs_sgdt", bytes.fromhex("65410f0100") + ret_bytes, "sgdt gs:[r8]", "unsupported instruction prefix"),
        ("lock_add_mem", bytes.fromhex("f00101") + ret_bytes, "lock add [rcx],eax", "unsupported instruction prefix"),
        ("rep_movsb", bytes.fromhex("f3a4") + ret_bytes, "rep movsb [rdi],[rsi]", "unsupported instruction prefix"),
        ("repne_sldt", bytes.fromhex("f20f00c0") + ret_bytes, "sldt eax", "unsupported instruction prefix"),
    ]
    for label, raw, expected_asm, expected_reason in raw_checks:
        decoded = formatter.format(next(iter(Decoder(64, raw, ip=0x1000))))
        if decoded != expected_asm:
            raise AssertionError(f"{label}: decoded {decoded!r}, expected {expected_asm!r}")
        try:
            lift_code(label, 0x1000, raw, host_cpu=True)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: unexpectedly accepted excluded encoding")

    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDQ_XMM_K1Z_XMM_XMMM128B64, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1

    # 3. IR-level rejections under host_cpu=True.
    unit_cases = [
        ("evex_vpaddq", encode_seq(evex_ins, ret_ins), "unsupported EVEX instruction"),
        ("priv_lgdt", encode_seq(Instruction.create_mem(Code.LGDT_M1664, MemoryOperand(R.R8)), ld_rax, ret_ins),
         "unsupported instruction"),
        ("priv_lidt", encode_seq(Instruction.create_mem(Code.LIDT_M1664, MemoryOperand(R.R8)), ld_rax, ret_ins),
         "unsupported instruction"),
        ("priv_lldt", encode_seq(Instruction.create_reg(Code.LLDT_RM16, R.CX), ld_rax, ret_ins),
         "unsupported instruction"),
        ("priv_lmsw", encode_seq(Instruction.create_reg(Code.LMSW_RM16, R.CX), ld_rax, ret_ins),
         "unsupported instruction"),
        ("priv_ltr", encode_seq(Instruction.create_reg(Code.LTR_RM16, R.CX), ld_rax, ret_ins),
         "unsupported instruction"),
        ("rip_rel_sgdt", encode_seq(Instruction.create_mem(
            Code.SGDT_M1664, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ld_rax, ret_ins),
         "RIP-relative memory operand"),
        ("bad_base_rax", encode_seq(ld_rax, Instruction.create_mem(
            Code.SIDT_M1664, MemoryOperand(R.RAX)), ret_ins),
         "bad base register"),
        ("bad_base_rsp", encode_seq(ld_rax, Instruction.create_mem(
            Code.SLDT_RM16, MemoryOperand(R.RSP)), ret_ins),
         "bad base register"),
        ("sib_index_sgdt", encode_seq(Instruction.create_mem(
            Code.SGDT_M1664, MemoryOperand(R.R8, R.RCX, 2)), ld_rax, ret_ins),
         "SIB index memory operand"),
        ("sib_index_smsw", encode_seq(Instruction.create_mem(
            Code.SMSW_RM16, MemoryOperand(R.R8, R.RDX, 1)), ld_rax, ret_ins),
         "SIB index memory operand"),
        ("nonvolatile_sldt_rbx", encode_seq(ld_rax, Instruction.create_reg(Code.SLDT_R64M16, R.RBX), ret_ins),
         "nonvolatile GPR"),
        ("nonvolatile_str_r12", encode_seq(ld_rax, Instruction.create_reg(Code.STR_R64M16, R.R12), ret_ins),
         "nonvolatile GPR"),
        ("nonvolatile_smsw_rbp", encode_seq(ld_rax, Instruction.create_reg(Code.SMSW_R32M16, R.EBP), ret_ins),
         "nonvolatile GPR"),
        ("high_byte_ah", encode_seq(ld_rax, Instruction.create_reg_i32(Code.MOV_R8_IMM8, R.AH, 1), ret_ins),
         "unsupported high-byte GPR"),
        ("nonleaf_jmp", encode_seq(ld_rax, Instruction.create_branch(Code.JMP_REL8_64, 0x1008), ret_ins),
         "non-leaf branch or call"),
        ("nonleaf_call", encode_seq(ld_rax, Instruction.create_branch(Code.CALL_REL32_64, 0x1020), ret_ins),
         "non-leaf branch or call"),
        ("nonleaf_je", encode_seq(ld_rax, Instruction.create_branch(Code.JE_REL8_64, 0x1008), ret_ins),
         "non-leaf branch or call"),
        ("undef_sldt_r16", encode_seq(Instruction.create_reg(Code.SLDT_RM16, R.AX), ret_ins),
         "undefined GPR"),
        ("undef_ret_rax", encode_seq(Instruction.create_mem(Code.SGDT_M1664, MemoryOperand(R.R8)), ret_ins),
         "return with undefined RAX"),
        ("undef_flags_sete", encode_seq(ld_rax, Instruction.create_mem(Code.SGDT_M1664, MemoryOperand(R.R8)),
                                        Instruction.create_reg(Code.SETE_RM8, R.AL), ret_ins),
         "undefined flags"),
        ("xor_mem_dst", encode_seq(ld_rax, Instruction.create_mem_reg(
            Code.XOR_RM32_R32, MemoryOperand(R.R8), R.EAX), ret_ins),
         "unsupported xor destination"),
        ("add_mem_dst", encode_seq(ld_rax, Instruction.create_mem_reg(
            Code.ADD_RM32_R32, MemoryOperand(R.R8), R.EAX), ret_ins),
         "unsupported add destination"),
        ("clobbered_r8_ptr", encode_seq(
            ld_rax,
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R8, R.RDX),
            Instruction.create_mem(Code.SGDT_M1664, MemoryOperand(R.R8)),
            ret_ins),
         "clobbered pointer"),
        ("missing_ret", encode_seq(ld_rax), "missing terminal ret"),
        ("after_ret", encode_seq(ld_rax, ret_ins, ld_rax), "unreachable instruction after ret"),
    ]
    for label, raw, expected_reason in unit_cases:
        try:
            lift_code(label, 0x1000, raw, host_cpu=True)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: unexpectedly accepted invalid sequence")

    # 4. PE-patched rejections + CLI rejections without --host-cpu and with bad inputs.
    pe_bytes = original_path.read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_sysdesc_discarded_stores")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(
            i for i in range(0, len(records), 12)
            if struct.unpack_from("<I", records, i)[0] == entry
        )
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    pe_patches = [
        ("pe_vex", bytes.fromhex("c5f1fdc2") + ret_bytes, "unsupported VEX instruction"),
        ("pe_evex", encode_seq(evex_ins, ret_ins), "unsupported EVEX instruction"),
        ("pe_xop", bytes.fromhex("8fe878a2c230") + ret_bytes, "unsupported XOP instruction"),
        ("pe_seg_sgdt", bytes.fromhex("65410f0100") + ret_bytes, "unsupported instruction prefix"),
        ("pe_sib_sgdt", encode_seq(Instruction.create_mem(
            Code.SGDT_M1664, MemoryOperand(R.R8, R.RCX, 2)), ld_rax, ret_ins), "SIB index memory operand"),
        ("pe_nonvol_str", encode_seq(ld_rax, Instruction.create_reg(Code.STR_R64M16, R.R12), ret_ins),
         "nonvolatile GPR"),
        ("pe_priv_lgdt", encode_seq(Instruction.create_mem(Code.LGDT_M1664, MemoryOperand(R.R8)), ld_rax, ret_ins),
         "unsupported instruction"),
        ("pe_branch", encode_seq(ld_rax, Instruction.create_branch(Code.JMP_REL8_64, 0x1008), ret_ins),
         "non-leaf branch or call"),
    ]

    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_sysdesc_aot.py"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        patched_path = tmp / "patched.dll"
        out_cpp = tmp / "rejected.cpp"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds max_len ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off : file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_sysdesc_discarded_stores"], host_cpu=True)
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: translate() accepted rejected PE patch")

        # CLI without --host-cpu must exit 1 without traceback or output file.
        proc_default = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(original_path), "op_sgdt", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc_default.returncode != 1 or "Traceback" in proc_default.stderr or "requires --host-cpu" not in proc_default.stderr:
            raise AssertionError(f"CLI default profile unexpected: rc={proc_default.returncode}, stderr={proc_default.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure without --host-cpu left behind output file")

        # CLI with --host-cpu on patched PE must exit 1 without traceback or output file.
        proc_patch = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched_path), "op_sysdesc_discarded_stores",
             str(out_cpp), "--host-cpu"],
            capture_output=True,
            text=True,
        )
        if proc_patch.returncode != 1 or "Traceback" in proc_patch.stderr or "non-leaf branch or call" not in proc_patch.stderr:
            raise AssertionError(f"CLI patched PE unexpected: rc={proc_patch.returncode}, stderr={proc_patch.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure on patched PE left behind output file")

        # CLI on missing PE must exit 1 without traceback.
        proc_missing = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(tmp / "missing.dll"), "op_sgdt",
             str(out_cpp), "--host-cpu"],
            capture_output=True,
            text=True,
        )
        if proc_missing.returncode != 1 or "Traceback" in proc_missing.stderr:
            raise AssertionError(f"CLI missing file unexpected: rc={proc_missing.returncode}, stderr={proc_missing.stderr!r}")

    print(
        f"rejections passed: 5 default-profile PE + {len(default_unit_cases)} default IR + "
        f"{len(raw_checks)} raw + {len(unit_cases)} IR + {len(pe_patches)} PE-patched + 3 CLI"
    )


def _is_umip_or_priv_trap(exc):
    msg = str(exc).lower()
    return (
        "0xc0000096" in msg
        or "0xc000001d" in msg
        or "privileged instruction" in msg
        or "illegal instruction" in msg
        or "access violation reading 0xffffffffffffffff" in msg
    )


def probe_native_host(native_fns, has_umip):
    """Probe native sysdesc execution; exit 77 cleanly if UMIP traps or host state is non-standard."""
    buf = (ctypes.c_uint8 * 64)()
    ctypes.memset(buf, 0xCC, 64)
    try:
        native_fns["op_sgdt"](0, SENTINELS[0], ctypes.addressof(buf))
        native_fns["op_sidt"](0, SENTINELS[0], ctypes.addressof(buf))
        sldt_val = native_fns["op_sldt_r32"](0, SENTINELS[0], ctypes.addressof(buf))
        str_val = native_fns["op_str_r32"](0, SENTINELS[0], ctypes.addressof(buf))
        smsw_val = native_fns["op_smsw_r32"](0, SENTINELS[0], ctypes.addressof(buf))
    except OSError as exc:
        if _is_umip_or_priv_trap(exc):
            print(f"native sysdesc trapped (CPUID.07H:ECX[2] UMIP={has_umip}, error={exc}); skipping")
            raise SystemExit(77)
        raise

    if sldt_val != EXPECTED_SLDT or str_val != EXPECTED_STR or smsw_val != EXPECTED_SMSW32:
        print(
            f"non-standard Windows x64 sysdesc host state "
            f"(sldt={sldt_val:#06x}, str={str_val:#06x}, smsw={smsw_val:#010x}, UMIP={has_umip}); skipping"
        )
        raise SystemExit(77)


def assert_canonical_gdtr(label, raw10, is_lifted):
    limit, base = struct.unpack_from("<HQ", raw10, 0)
    if is_lifted:
        if limit != EXPECTED_LIFTED_GDT_LIMIT or base != EXPECTED_LIFTED_GDT_BASE:
            raise AssertionError(
                f"{label} SGDT mismatch: got (limit={limit:#06x}, base={base:#018x}) "
                f"!= exp ({EXPECTED_LIFTED_GDT_LIMIT:#06x}, {EXPECTED_LIFTED_GDT_BASE:#018x})"
            )
    else:
        if limit < 0x0037 or (limit & 7) != 7 or (base >> 47) != 0x1FFFF or (base & 7) != 0:
            raise AssertionError(
                f"{label} native SGDT architectural invariant failed: limit={limit:#06x} base={base:#018x}"
            )


def verify_suite(native_fns, target_fns, label, cpus):
    kernel32, full_mask, _ = logical_cpus()
    saved_affinity = None
    buf_n = (ctypes.c_uint8 * 64)()
    buf_t = (ctypes.c_uint8 * 64)()
    cases = 0
    try:
        for cpu in cpus:
            prev = pin_to_cpu(kernel32, cpu)
            if saved_affinity is None:
                saved_affinity = prev

            # 1. SIDT (0 and +16 displacement): exact 10-byte match against pinned native + guard preservation.
            for name, offset in (("op_sidt", 0), ("op_sidt_disp", 16)):
                ctypes.memset(buf_n, 0xCC, 64)
                ctypes.memset(buf_t, 0xCC, 64)
                ret_n = native_fns[name](0, SENTINELS[0], ctypes.addressof(buf_n))
                ret_t = target_fns[name](0, SENTINELS[0], ctypes.addressof(buf_t))
                raw_n, raw_t = bytes(buf_n), bytes(buf_t)
                if ret_n != SENTINELS[0] or ret_t != SENTINELS[0] or raw_n != raw_t:
                    raise AssertionError(
                        f"{label} {name} cpu {cpu}: mismatch native={raw_n[offset:offset+16].hex()} "
                        f"!= target={raw_t[offset:offset+16].hex()}"
                    )
                if raw_t[:offset] != b"\xCC" * offset or raw_t[offset + 10 :] != b"\xCC" * (64 - offset - 10):
                    raise AssertionError(f"{label} {name} cpu {cpu}: corrupted guard bytes around 10-byte IDTR")
                lim, base = struct.unpack_from("<HQ", raw_t, offset)
                if lim == 0 or (base >> 47) != 0x1FFFF:
                    raise AssertionError(f"{label} {name} cpu {cpu}: unexpected IDTR lim={lim:#x} base={base:#x}")
                cases += 1

            # 2. SGDT (0 and +16 displacement): canonical GDTR invariants + exact 10-byte guard preservation.
            for name, offset in (("op_sgdt", 0), ("op_sgdt_disp", 16)):
                ctypes.memset(buf_n, 0xCC, 64)
                ctypes.memset(buf_t, 0xCC, 64)
                ret_n = native_fns[name](0, SENTINELS[1], ctypes.addressof(buf_n))
                ret_t = target_fns[name](0, SENTINELS[1], ctypes.addressof(buf_t))
                raw_n, raw_t = bytes(buf_n), bytes(buf_t)
                if ret_n != SENTINELS[1] or ret_t != SENTINELS[1]:
                    raise AssertionError(f"{label} {name}: return sentinel corrupted")
                for tag, raw, is_lifted in (("native", raw_n, False), (label, raw_t, True)):
                    if raw[:offset] != b"\xCC" * offset or raw[offset + 10 :] != b"\xCC" * (64 - offset - 10):
                        raise AssertionError(f"{tag} {name}: corrupted guard bytes around 10-byte GDTR")
                    assert_canonical_gdtr(tag, raw[offset : offset + 10], is_lifted)
                cases += 1

            # 3. SLDT, STR, SMSW across r16 (preserve [63:16]), r32/r64 (zero-extend), and m16 (+disp).
            reg_specs = (
                ("op_sldt_r16", 16, EXPECTED_SLDT),
                ("op_sldt_r32", 32, EXPECTED_SLDT),
                ("op_sldt_r64", 64, EXPECTED_SLDT),
                ("op_str_r16", 16, EXPECTED_STR),
                ("op_str_r32", 32, EXPECTED_STR),
                ("op_str_r64", 64, EXPECTED_STR),
                ("op_smsw_r16", 16, EXPECTED_SMSW32 & 0xFFFF),
                ("op_smsw_r32", 32, EXPECTED_SMSW32),
                ("op_smsw_r64", 64, EXPECTED_SMSW32),
            )
            for name, width, expected_val in reg_specs:
                for sentinel in SENTINELS:
                    exp = ((sentinel & ~0xFFFF) | expected_val) if width == 16 else expected_val
                    got_n = native_fns[name](0, sentinel, ctypes.addressof(buf_n))
                    got_t = target_fns[name](0, sentinel, ctypes.addressof(buf_t))
                    if got_n != exp or got_t != exp:
                        raise AssertionError(
                            f"{label} {name} sent={sentinel:#018x}: "
                            f"native={got_n:#018x} target={got_t:#018x} exp={exp:#018x}"
                        )
                    cases += 1

            mem_specs = (
                ("op_sldt_m16", 0, EXPECTED_SLDT),
                ("op_sldt_m16_disp", 4, EXPECTED_SLDT),
                ("op_str_m16", 0, EXPECTED_STR),
                ("op_str_m16_disp", 6, EXPECTED_STR),
                ("op_smsw_m16", 0, EXPECTED_SMSW32 & 0xFFFF),
                ("op_smsw_m16_disp", 8, EXPECTED_SMSW32 & 0xFFFF),
            )
            for name, offset, expected_val in mem_specs:
                ctypes.memset(buf_n, 0xCC, 64)
                ctypes.memset(buf_t, 0xCC, 64)
                ret_n = native_fns[name](0, SENTINELS[0], ctypes.addressof(buf_n))
                ret_t = target_fns[name](0, SENTINELS[0], ctypes.addressof(buf_t))
                raw_n, raw_t = bytes(buf_n), bytes(buf_t)
                exp_buf = bytearray(b"\xCC" * 64)
                struct.pack_into("<H", exp_buf, offset, expected_val)
                if ret_n != SENTINELS[0] or ret_t != SENTINELS[0] or raw_n != exp_buf or raw_t != exp_buf:
                    raise AssertionError(
                        f"{label} {name}: native={raw_n[:16].hex()} target={raw_t[:16].hex()} exp={exp_buf[:16].hex()}"
                    )
                cases += 1

            # 4. Flag preservation across all 5 sysdesc instructions + live SIDT match.
            ctypes.memset(buf_n, 0xCC, 64)
            ctypes.memset(buf_t, 0xCC, 64)
            ret_n = native_fns["op_sysdesc_flags_preserve"](0, SENTINELS[0], ctypes.addressof(buf_n))
            ret_t = target_fns["op_sysdesc_flags_preserve"](0, SENTINELS[0], ctypes.addressof(buf_t))
            raw_n, raw_t = bytes(buf_n), bytes(buf_t)
            # 0x80000001 + 0x80000002 = 3 => ZF=0, CF=1, SF=0, OF=1, PF=1
            exp_flags = bytes((0, 1, 0, 1, 1))
            if (
                ret_n != EXPECTED_SMSW32
                or ret_t != EXPECTED_SMSW32
                or raw_n[:5] != exp_flags
                or raw_t[:5] != exp_flags
                or raw_n[32:42] != raw_t[32:42]
                or raw_t[26:32] != b"\xCC" * 6
                or raw_t[42:64] != b"\xCC" * 22
            ):
                raise AssertionError(
                    f"{label} op_sysdesc_flags_preserve cpu {cpu}: "
                    f"ret=({ret_n:#x},{ret_t:#x}) flags=({raw_n[:5].hex()},{raw_t[:5].hex()})"
                )
            assert_canonical_gdtr("native", raw_n[16:26], False)
            assert_canonical_gdtr(label, raw_t[16:26], True)
            cases += 1

            # 5. Discarded volatile stores: valid pointer writes all 5; NULL faults even under /O2.
            ctypes.memset(buf_n, 0xCC, 64)
            ctypes.memset(buf_t, 0xCC, 64)
            ret_n = native_fns["op_sysdesc_discarded_stores"](0, SENTINELS[2], ctypes.addressof(buf_n))
            ret_t = target_fns["op_sysdesc_discarded_stores"](0, SENTINELS[2], ctypes.addressof(buf_t))
            raw_n, raw_t = bytes(buf_n), bytes(buf_t)
            if (
                ret_n != SENTINELS[2]
                or ret_t != SENTINELS[2]
                or raw_n[16:26] != raw_t[16:26]
                or raw_n[32:38] != raw_t[32:38]
                or raw_t[10:16] != b"\xCC" * 6
                or raw_t[26:32] != b"\xCC" * 6
                or raw_t[38:64] != b"\xCC" * 26
            ):
                raise AssertionError(f"{label} op_sysdesc_discarded_stores(valid) mismatch on cpu {cpu}")
            assert_canonical_gdtr("native", raw_n[0:10], False)
            assert_canonical_gdtr(label, raw_t[0:10], True)
            for fault_name in ("op_sysdesc_discarded_stores", "op_sidt", "op_sldt_m16"):
                for tag, fn in (("native", native_fns[fault_name]), (label, target_fns[fault_name])):
                    try:
                        fn(0, SENTINELS[2], 0)
                    except OSError as exc:
                        if "access violation writing" not in str(exc).lower():
                            raise AssertionError(f"{tag} {fault_name}(NULL) unexpected error: {exc}") from exc
                    else:
                        raise AssertionError(f"{tag} {fault_name}(NULL) dropped faulting store")
            cases += 4

            # 6. Unused register writes return sentinel cleanly.
            for sentinel in SENTINELS[:2]:
                if (native_fns["op_sysdesc_unused_reg"](0, sentinel, 0) != sentinel
                        or target_fns["op_sysdesc_unused_reg"](0, sentinel, 0) != sentinel):
                    raise AssertionError(f"{label} op_sysdesc_unused_reg corrupted sentinel")
                cases += 1
    finally:
        if saved_affinity is not None:
            kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), full_mask)
    return cases


def check_altered_and_mutants(original_lifted_fns, altered_native_fns, altered_lifted_fns, cpu):
    kernel32, full_mask, _ = logical_cpus()
    saved_affinity = None
    buf_o = (ctypes.c_uint8 * 64)()
    buf_an = (ctypes.c_uint8 * 64)()
    buf_al = (ctypes.c_uint8 * 64)()
    diverged = []
    try:
        saved_affinity = pin_to_cpu(kernel32, cpu)
        # Altered native and altered lifted agree on every mutated export.
        mutated_exports = (
            "op_sgdt",
            "op_sidt",
            "op_sldt_r16",
            "op_sldt_m16",
            "op_str_r16",
            "op_smsw_r16",
            "op_sysdesc_flags_preserve",
        )
        for name in mutated_exports:
            ctypes.memset(buf_o, 0xCC, 64)
            ctypes.memset(buf_an, 0xCC, 64)
            ctypes.memset(buf_al, 0xCC, 64)
            ret_o = original_lifted_fns[name](0, SENTINELS[0], ctypes.addressof(buf_o))
            ret_an = altered_native_fns[name](0, SENTINELS[0], ctypes.addressof(buf_an))
            ret_al = altered_lifted_fns[name](0, SENTINELS[0], ctypes.addressof(buf_al))
            raw_o, raw_an, raw_al = bytes(buf_o), bytes(buf_an), bytes(buf_al)
            if name == "op_sidt":
                # ALTER_IMM turns op_sidt into sgdt: verify canonical GDTR on both altered native and altered lifted.
                assert_canonical_gdtr("altered native op_sidt->sgdt", raw_an[:10], False)
                assert_canonical_gdtr("altered lifted op_sidt->sgdt", raw_al[:10], True)
            elif name == "op_sysdesc_flags_preserve":
                if ret_an != ret_al or raw_an[:5] != raw_al[:5] or raw_an[32:42] != raw_al[32:42]:
                    raise AssertionError(f"altered mismatch in {name}")
            else:
                if ret_an != ret_al or raw_an != raw_al:
                    raise AssertionError(
                        f"altered mismatch in {name}: native=({ret_an:#x}, {raw_an[:16].hex()}) "
                        f"!= lifted=({ret_al:#x}, {raw_al[:16].hex()})"
                    )
            if ret_o == ret_al and raw_o == raw_al:
                raise AssertionError(f"altered control {name} failed to diverge from original")
            diverged.append(name)
    finally:
        if saved_affinity is not None:
            kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), full_mask)
    return diverged


def compile_o2_and_verify(original_path, native_fns, cxx_path, cpus):
    source_text, _ = translate(original_path, list(TRANSLATED_EXPORTS), host_cpu=True)
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "sysdesc_o2.cpp"
        dll_path = tmp / "sysdesc_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'sysdesc_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_lib(dll_path, "lifted_")
        try:
            return verify_suite(native_fns, o2_fns, "lifted_o2", cpus)
        finally:
            o2_fns.clear()
            unload_lib(o2_lib)


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in TRANSLATED_EXPORTS:
            entry = export_rva(pe, name)
            raw = bounded_code(pe, entry)
            for ins in Decoder(64, raw, ip=entry):
                forms.add(formatter.format_mnemonic(ins).lower())
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def main():
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_path, altered_lifted_path, cxx_path = (
        Path(p) for p in sys.argv[1:6]
    )

    check_rejections(original_path)

    orig_lib, orig_fns = bind_lib(original_path)
    lifted_lib, lifted_fns = bind_lib(lifted_path, "lifted_")
    alt_lib, alt_fns = bind_lib(altered_path)
    alt_lifted_lib, alt_lifted_fns = bind_lib(altered_lifted_path, "lifted_")
    try:
        has_umip = lifted_fns["host_has_umip"]()
        probe_native_host(orig_fns, has_umip)
        _, _, all_cpus = logical_cpus()
        cpus = all_cpus[:4] if len(all_cpus) > 4 else all_cpus
        lifted_cases = verify_suite(orig_fns, lifted_fns, "lifted", cpus)
        diverged = check_altered_and_mutants(lifted_fns, alt_fns, alt_lifted_fns, cpus[0])
        o2_cases = compile_o2_and_verify(original_path, orig_fns, cxx_path, cpus)
    finally:
        for fn_map, lib in (
            (orig_fns, orig_lib),
            (lifted_fns, lifted_lib),
            (alt_fns, alt_lib),
            (alt_lifted_fns, alt_lifted_lib),
        ):
            fn_map.clear()
            unload_lib(lib)

    record_covered_forms(original_path)
    print(
        f"sysdesc passed (UMIP_CPUID={has_umip}, cpus={cpus}): "
        f"{lifted_cases} lifted + {o2_cases} /O2 cases across {len(TRANSLATED_EXPORTS)} exports; "
        f"ALTER_IMM diverged on {diverged}"
    )


if __name__ == "__main__":
    main()
