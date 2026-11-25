"""Differential, /O2, flag-modeling, mutant, and rejection checks for cpu_bmi1_aot (BLSI, BLSMSK, BLSR)."""

import ctypes
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Code, Decoder, Encoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_bmi1_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_blsi32", "op_blsi64",
    "op_blsmsk32", "op_blsmsk64",
    "op_blsr32", "op_blsr64",
    "op_blsi32_mem", "op_blsi64_mem",
    "op_blsmsk32_mem", "op_blsmsk64_mem",
    "op_blsr32_mem", "op_blsr64_mem",
    "op_blsi32_flags", "op_blsi64_flags",
    "op_blsmsk32_flags", "op_blsmsk64_flags",
    "op_blsr32_flags", "op_blsr64_flags",
    "op_blsr64_mem_flags",
    "op_r8_source_then_store",
    "op_bmi1_discarded_read",
)

ALTERED_REG_EXPORTS = (
    "op_blsi32", "op_blsi64",
    "op_blsmsk32", "op_blsmsk64",
    "op_blsr32", "op_blsr64",
)
ALTERED_MEM_EXPORTS = (
    "op_blsi32_mem", "op_blsi64_mem",
    "op_blsmsk32_mem", "op_blsmsk64_mem",
    "op_blsr32_mem", "op_blsr64_mem",
)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))


# --- Independent Python reference model ---

def ref_bmi1(op, val, bits, *, quirk=None):
    """Return (dst, packed_flags) where packed_flags = (CF<<0)|(ZF<<1)|(SF<<2)|(OF<<3)."""
    mask = MASK32 if bits == 32 else MASK64
    sign = 1 << (bits - 1)
    s = val & mask
    if op == "blsi":
        d = ((-s) & s) & mask
        cf = (s == 0) if quirk == "invert_cf" else (s != 0)
        zf = (d == 0)
    elif op == "blsmsk":
        d = ((s - 1) ^ s) & mask
        cf = (s != 0) if quirk == "invert_cf" else (s == 0)
        zf = (s == 0) if quirk == "blsmsk_zf_on_zero" else False
    elif op == "blsr":
        d = ((s - 1) & s) & mask
        cf = (s != 0) if quirk == "invert_cf" else (s == 0)
        zf = (s == 0) if quirk == "blsr_zf_only_zero" else (d == 0)
    else:
        raise ValueError(op)
    sf = False if quirk == "drop_sf" else ((d & sign) != 0)
    of = True if quirk == "set_of" else False
    if quirk == "no_zext32" and bits == 32:
        d |= 0xFFFFFFFF00000000
    packed = (int(cf) << 0) | (int(zf) << 1) | (int(sf) << 2) | (int(of) << 3)
    return d, packed


ALTERED_OP = {"blsi": "blsr", "blsmsk": "blsi", "blsr": "blsmsk"}


def test_values():
    vals = [
        0x0000000000000000,
        0x0000000000000001,
        0x0000000000000002,
        0x0000000000000010,
        0x000000007FFFFFFF,
        0x0000000080000000,  # INT32_MIN (single high bit in 32-bit!)
        0x0000000080000001,  # INT32_MIN | 1 -> blsr leaves INT32_MIN (SF=1)
        0x00000000FFFFFFFE,
        0x00000000FFFFFFFF,
        0xDEADBEEF00000000,  # zero in 32-bit, non-zero in 64-bit!
        0xDEADBEEF80000000,  # INT32_MIN in 32-bit with dirty high 32 bits
        0x0000000100000000,  # single bit 32
        0x7FFFFFFFFFFFFFFF,
        0x8000000000000000,  # INT64_MIN (single high bit in 64-bit!)
        0x8000000000000001,  # INT64_MIN | 1 -> blsr leaves INT64_MIN (SF=1)
        0xFFFFFFFFFFFFFFFE,
        0xFFFFFFFFFFFFFFFF,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
    ]
    # Every single-bit mask across 0..63.
    vals.extend(1 << b for b in range(64))
    rng = random.Random(0xB111A07)
    vals.extend(rng.getrandbits(64) for _ in range(32))
    return vals


# --- Guarded memory & DLL helpers ---

def alloc_guarded(payload):
    raw = ctypes.create_string_buffer(GUARD_PRE + payload + GUARD_POST, len(GUARD_PRE) + len(payload) + len(GUARD_POST))
    base = ctypes.addressof(raw)
    return raw, base + len(GUARD_PRE), len(GUARD_PRE), len(payload)


def read_guarded(buf, off, length, label):
    raw = buf.raw
    if raw[:off] != GUARD_PRE or raw[off + length:] != GUARD_POST:
        raise AssertionError(f"guard bytes corrupted around {label}")
    return raw[off:off + length]


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = ctypes.c_uint64
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def host_has_bmi1(lifted_lib):
    if hasattr(lifted_lib, "host_has_bmi1"):
        lifted_lib.host_has_bmi1.argtypes = []
        lifted_lib.host_has_bmi1.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_bmi1())
    return False


# --- Rejection & alternate-encoding controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    ret = Instruction.create(Code.RETNQ)
    mov_rax_rcx = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.RCX)
    xor_r9 = Instruction.create_reg_reg(Code.XOR_R32_RM32, R.R9D, R.R9D)
    blsi_eax_ecx = Instruction.create_reg_reg(Code.VEX_BLSI_R32_RM32, R.EAX, R.ECX)
    blsmsk_rax_rcx = Instruction.create_reg_reg(Code.VEX_BLSMSK_R64_RM64, R.RAX, R.RCX)
    blsr_rax_rcx = Instruction.create_reg_reg(Code.VEX_BLSR_R64_RM64, R.RAX, R.RCX)

    # Alternate encodings accepted cleanly.
    alt_accept = [
        ("blsi32_r11_r8_mem", (Instruction.create_reg_mem(Code.VEX_BLSI_R32_RM32, R.R11D, MemoryOperand(R.R8)),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R11), ret),
         "r11 = op_blsi32(ld_1000_1, flags);"),
        ("blsmsk64_r10_rdx", (Instruction.create_reg_reg(Code.VEX_BLSMSK_R64_RM64, R.R10, R.RDX),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R10), ret),
         "r10 = op_blsmsk64(rdx, flags);"),
        ("blsr32_r9_r8d", (Instruction.create_reg_reg(Code.VEX_BLSR_R32_RM32, R.R9D, R.R8D),
            Instruction.create_reg_reg(Code.MOV_R32_RM32, R.EAX, R.R9D), ret),
         "r9 = op_blsr32((r8 & 0xFFFFFFFFull), flags);"),
    ]
    for label, seq, expected in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alt {label}: expected {expected!r}")

    seg_blsi = Instruction.create_reg_mem(Code.VEX_BLSI_R64_RM64, R.RAX, MemoryOperand(R.RCX))
    seg_blsi.segment_prefix = R.GS
    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1

    unit_cases = [
        # BMI2 rejections
        ("bmi2_rorx", (Instruction.create_reg_reg_u32(Code.VEX_RORX_R64_RM64_IMM8, R.RAX, R.RCX, 5), ret),
         "unsupported VEX instruction"),
        ("bmi2_sarx", (Instruction.create_reg_reg_reg(Code.VEX_SARX_R64_RM64_R64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_shlx", (Instruction.create_reg_reg_reg(Code.VEX_SHLX_R32_RM32_R32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_shrx", (Instruction.create_reg_reg_reg(Code.VEX_SHRX_R64_RM64_R64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_bzhi", (Instruction.create_reg_reg_reg(Code.VEX_BZHI_R64_RM64_R64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        # Other BMI1 instructions
        ("bmi1_tzcnt", (Instruction.create_reg_reg(Code.TZCNT_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported instruction"),
        ("bmi1_andn", (Instruction.create_reg_reg_reg(Code.VEX_ANDN_R64_R64_RM64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        ("bmi1_bextr", (Instruction.create_reg_reg_reg(Code.VEX_BEXTR_R64_RM64_R64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        # EVEX / XOP
        ("evex_vpaddd", (evex_ins, mov_rax_rcx, ret), "unsupported EVEX instruction"),
        ("xop_blsic", (Instruction.create_reg_reg(Code.XOP_BLSIC_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported XOP instruction"),
        # Undefined PF / AF reads after BLSI / BLSMSK / BLSR (including after prior XOR defining PF!)
        ("undef_pf_after_blsi", (xor_r9, blsi_eax_ecx, Instruction.create_reg(Code.SETP_RM8, R.R9L), ret),
         "undefined flag pf read"),
        ("undef_pf_after_blsmsk", (xor_r9, blsmsk_rax_rcx, Instruction.create_mem(Code.SETNP_RM8, MemoryOperand(R.R8)), ret),
         "undefined flag pf read"),
        ("undef_pf_after_blsr", (xor_r9, blsr_rax_rcx, Instruction.create_reg(Code.SETP_RM8, R.R9L), ret),
         "undefined flag pf read"),
        ("undef_af_lahf_after_blsi", (blsi_eax_ecx, Instruction.create(Code.LAHF), ret),
         "unsupported instruction"),
        ("undef_flags_pushfq", (blsr_rax_rcx, Instruction.create(Code.PUSHFQ), ret),
         "unsupported instruction"),
        ("undef_flags_before_bmi1", (mov_rax_rcx, Instruction.create_mem(Code.SETB_RM8, MemoryOperand(R.R8)), ret),
         "undefined flag cf read"),
        # Nonvolatile GPRs & arg GPR writes & high-byte GPRs
        ("nonvol_rbx_dst", (Instruction.create_reg_reg(Code.VEX_BLSI_R64_RM64, R.RBX, R.RCX), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("nonvol_rsi_src", (Instruction.create_reg_reg(Code.VEX_BLSR_R64_RM64, R.RAX, R.RSI), ret),
         "nonvolatile GPR"),
        ("nonvol_r12_dst", (Instruction.create_reg_reg(Code.VEX_BLSMSK_R32_RM32, R.R12D, R.ECX), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("write_arg_rcx", (Instruction.create_reg_reg(Code.VEX_BLSI_R64_RM64, R.RCX, R.RDX), mov_rax_rcx, ret),
         "cannot write argument GPR rcx"),
        ("write_arg_edx", (Instruction.create_reg_reg(Code.VEX_BLSR_R32_RM32, R.EDX, R.ECX), mov_rax_rcx, ret),
         "cannot write argument GPR rdx"),
        ("write_arg_r8", (Instruction.create_reg_reg(Code.VEX_BLSMSK_R64_RM64, R.R8, R.RCX), mov_rax_rcx, ret),
         "cannot write argument GPR r8"),
        ("high_byte_ah", (mov_rax_rcx, Instruction.create_reg_u32(Code.MOV_R8_IMM8, R.AH, 1), ret),
         "unsupported high-byte GPR"),
        # Undefined GPRs
        ("undef_gpr_r9_src", (Instruction.create_reg_reg(Code.VEX_BLSI_R64_RM64, R.RAX, R.R9), ret),
         "undefined GPR r9"),
        ("undef_rax_ret", (Instruction.create_reg_reg(Code.VEX_BLSR_R64_RM64, R.R9, R.RCX), ret),
         "return with undefined RAX"),
        # Displaced / indexed / non-flat memory & non-r8 stores
        ("displaced_blsi_rcx", (Instruction.create_reg_mem(
            Code.VEX_BLSI_R64_RM64, R.RAX, MemoryOperand(R.RCX, displ=8, displ_size=1)), ret),
         "displaced memory operand"),
        ("displaced_store_r8", (mov_rax_rcx, Instruction.create_mem_reg(
            Code.MOV_RM64_R64, MemoryOperand(R.R8, displ=8, displ_size=1), R.RAX), ret),
         "displaced memory operand"),
        ("indexed_blsmsk", (Instruction.create_reg_mem(
            Code.VEX_BLSMSK_R64_RM64, R.RAX, MemoryOperand(R.RCX, R.RDX, 8)), ret),
         "SIB index memory operand"),
        ("rip_rel_blsr", (Instruction.create_reg_mem(
            Code.VEX_BLSR_R64_RM64, R.RAX, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret),
         "RIP-relative memory operand"),
        ("bad_base_rbx", (Instruction.create_reg_mem(Code.VEX_BLSI_R64_RM64, R.RAX, MemoryOperand(R.RBX)), ret),
         "bad base register in memory operand"),
        ("bad_base_rsp", (Instruction.create_reg_mem(Code.VEX_BLSR_R32_RM32, R.EAX, MemoryOperand(R.RSP)), ret),
         "bad base register in memory operand"),
        ("non_r8_store_rcx", (mov_rax_rcx, Instruction.create_mem_reg(
            Code.MOV_RM64_R64, MemoryOperand(R.RCX), R.RAX), ret), "non-r8 store memory operand"),
        ("non_r8_store_setb", (blsi_eax_ecx, Instruction.create_mem(Code.SETB_RM8, MemoryOperand(R.RDX)), ret),
         "non-r8 store memory operand"),
        # Prefixes, branches, missing/trailing ret
        ("seg_prefix_blsi", (seg_blsi, ret), "unsupported instruction prefix"),
        ("branch_jmp", (mov_rax_rcx, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret),
         "non-leaf branch or call"),
        ("missing_ret", (blsi_eax_ecx,), "missing terminal ret"),
        ("trailing_after_ret", (blsi_eax_ecx, ret, mov_rax_rcx), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("lock_prefix", bytes.fromhex("f00fb109") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("rep_prefix", bytes.fromhex("f3a4") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("empty_stream", b"", "truncated instruction stream"),
        ("invalid_bytes", bytes.fromhex("fffe") + encode_seq(ret), "invalid instruction"),
    ]

    for label, seq, reason in unit_cases:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected control")

    for label, raw, reason in raw_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted raw rejected control")

    # End-to-end PE-patched rejections & CLI non-traceback rejection.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_blsi64_flags")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {label: (encode_seq(*seq), reason) for label, seq, reason in unit_cases}
    pe_labels = (
        "bmi2_rorx",
        "bmi1_tzcnt",
        "evex_vpaddd",
        "undef_pf_after_blsi",
        "nonvol_rbx_dst",
        "write_arg_rcx",
        "displaced_blsi_rcx",
        "non_r8_store_rcx",
        "branch_jmp",
        "seg_prefix_blsi",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_bmi1_aot.py"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        patched = tmp / "patched.dll"
        out_cpp = tmp / "rejected.cpp"
        for label in pe_labels:
            patch, reason = by_label[label]
            if len(patch) > max_len:
                raise AssertionError(f"pe_{label}: patch ({len(patch)} bytes) exceeds max_len ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched.write_bytes(damaged)
            try:
                translate(patched, ["op_blsi64_flags"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched), "op_blsi64_flags", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

    print(
        f"rejections passed: {len(alt_accept)} alt + {len(unit_cases)} IR + "
        f"{len(raw_cases)} raw + {len(pe_labels)} PE-patched + CLI"
    )


# --- Positive, altered, mutant, and /O2 verification ---

def verify_suite(fns, label, vals):
    total = 0

    # 1) Register forms (32-bit zero-ext + 64-bit).
    for op in ("blsi", "blsmsk", "blsr"):
        for bits in (32, 64):
            fn = fns[f"op_{op}{bits}"]
            for v in vals:
                want, _ = ref_bmi1(op, v, bits)
                got = fn(v, 0, 0)
                if got != want:
                    raise AssertionError(f"{label} op_{op}{bits}({v:#x}): got {got:#x} != exp {want:#x}")
                total += 1

    # 2) Flat [rcx|rdx] memory forms.
    mem_specs = (
        ("op_blsi32_mem", "blsi", 32, "rcx"),
        ("op_blsi64_mem", "blsi", 64, "rdx"),
        ("op_blsmsk32_mem", "blsmsk", 32, "rdx"),
        ("op_blsmsk64_mem", "blsmsk", 64, "rcx"),
        ("op_blsr32_mem", "blsr", 32, "rcx"),
        ("op_blsr64_mem", "blsr", 64, "rdx"),
    )
    for name, op, bits, reg in mem_specs:
        fn = fns[name]
        fmt = "<I" if bits == 32 else "<Q"
        mask = MASK32 if bits == 32 else MASK64
        for v in vals:
            buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, v & mask))
            want, _ = ref_bmi1(op, v, bits)
            got = fn(ptr, 0, 0) if reg == "rcx" else fn(0, ptr, 0)
            read_guarded(buf, off, ln, name)
            if got != want:
                raise AssertionError(f"{label} {name}({v:#x}): got {got:#x} != exp {want:#x}")
            total += 1

    # 3) Flag-packing exports (CF, ZF, SF, OF).
    for op in ("blsi", "blsmsk", "blsr"):
        for bits in (32, 64):
            name = f"op_{op}{bits}_flags"
            fn = fns[name]
            for v in vals:
                want_val, want_flags = ref_bmi1(op, v, bits)
                bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
                got_val = fn(v, 0, po)
                got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, name))[0]
                if got_val != want_val or got_flags != want_flags:
                    raise AssertionError(
                        f"{label} {name}({v:#x}): val={got_val:#x}/{want_val:#x} flags={got_flags:#x}/{want_flags:#x}"
                    )
                total += 1

    # Memory flag-packing export: op_blsr64_mem_flags.
    for v in vals:
        want_val, want_flags = ref_bmi1("blsr", v, 64)
        ba, pa, oa, la = alloc_guarded(struct.pack("<Q", v))
        bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
        got_val = fns["op_blsr64_mem_flags"](pa, 0, po)
        read_guarded(ba, oa, la, "op_blsr64_mem_flags_a")
        got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, "op_blsr64_mem_flags_out"))[0]
        if got_val != want_val or got_flags != want_flags:
            raise AssertionError(
                f"{label} op_blsr64_mem_flags({v:#x}): val={got_val:#x}/{want_val:#x} flags={got_flags:#x}/{want_flags:#x}"
            )
        total += 1

    # 4) [r8] source-then-store: reads [r8] via blsi, [rcx] via blsmsk, [rdx] via blsr, stores to [r8].
    for i in range(0, len(vals) - 2, 3):
        a, b, c = vals[i], vals[i + 1], vals[i + 2]
        ba, pa, oa, la = alloc_guarded(struct.pack("<Q", a))
        bb, pb, ob, lb = alloc_guarded(struct.pack("<Q", b))
        br, pr, or_, lr = alloc_guarded(struct.pack("<Q", c))
        got_rax = fns["op_r8_source_then_store"](pa, pb, pr)
        read_guarded(ba, oa, la, "r8_src_a")
        read_guarded(bb, ob, lb, "r8_src_b")
        got_mem = struct.unpack("<Q", read_guarded(br, or_, lr, "r8_src_out"))[0]
        want = (ref_bmi1("blsi", c, 64)[0] | ref_bmi1("blsmsk", a, 64)[0]) ^ ref_bmi1("blsr", b, 64)[0]
        if got_rax != want or got_mem != want:
            raise AssertionError(f"{label} op_r8_source_then_store: rax={got_rax:#x} mem={got_mem:#x} exp={want:#x}")
        total += 1

    # 5) Discarded faulting m64 read must fault on NULL rdx even under /O2.
    bd, pd, od, ld = alloc_guarded(struct.pack("<Q", 0x123456789ABCDEF0))
    assert fns["op_bmi1_discarded_read"](0xCAFEBABE, pd, 0) == 0xCAFEBABE
    read_guarded(bd, od, ld, "discarded_valid")
    try:
        fns["op_bmi1_discarded_read"](0xCAFEBABE, 0, 0)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_bmi1_discarded_read(NULL) dropped faulting m64 read")

    return total


def check_altered_and_mutants(lifted_fns, alt_lifted_fns, vals):
    diverged = 0
    for name in ALTERED_REG_EXPORTS:
        op = name[3:-2]
        bits = int(name[-2:])
        alt_op = ALTERED_OP[op]
        diffs = 0
        for v in vals[:32]:
            want_alt, _ = ref_bmi1(alt_op, v, bits)
            got_al = alt_lifted_fns[name](v, 0, 0)
            if got_al != want_alt:
                raise AssertionError(f"altered_lifted {name}({v:#x}): {got_al:#x} != {want_alt:#x}")
            diffs += got_al != lifted_fns[name](v, 0, 0)
        if not diffs:
            raise AssertionError(f"ALTER_BMI1 failed to diverge on {name}")
        diverged += 1

    mem_reg = {
        "op_blsi32_mem": "rcx", "op_blsi64_mem": "rdx",
        "op_blsmsk32_mem": "rdx", "op_blsmsk64_mem": "rcx",
        "op_blsr32_mem": "rcx", "op_blsr64_mem": "rdx",
    }
    for name in ALTERED_MEM_EXPORTS:
        stem = name[3:-4]
        op, bits = stem[:-2], int(stem[-2:])
        alt_op = ALTERED_OP[op]
        fmt = "<I" if bits == 32 else "<Q"
        mask = MASK32 if bits == 32 else MASK64
        diffs = 0
        for v in vals[:32]:
            buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, v & mask))
            want_alt, _ = ref_bmi1(alt_op, v, bits)
            got_al = alt_lifted_fns[name](ptr, 0, 0) if mem_reg[name] == "rcx" else alt_lifted_fns[name](0, ptr, 0)
            got_ol = lifted_fns[name](ptr, 0, 0) if mem_reg[name] == "rcx" else lifted_fns[name](0, ptr, 0)
            read_guarded(buf, off, ln, name)
            if got_al != want_alt:
                raise AssertionError(f"altered_lifted {name}({v:#x}): {got_al:#x} != {want_alt:#x}")
            diffs += got_al != got_ol
        if not diffs:
            raise AssertionError(f"ALTER_BMI1 failed to diverge on {name}")
        diverged += 1

    # Mutant models (inverted CF, wrong ZF for BLSMSK/BLSR, dropped SF, set OF, missing 32-bit zero-ext).
    mutant_specs = (
        ("blsi", 32, "invert_cf"),
        ("blsmsk", 64, "invert_cf"),
        ("blsr", 64, "invert_cf"),
        ("blsmsk", 32, "blsmsk_zf_on_zero"),
        ("blsr", 64, "blsr_zf_only_zero"),
        ("blsi", 32, "drop_sf"),
        ("blsmsk", 64, "drop_sf"),
        ("blsr", 64, "drop_sf"),
        ("blsi", 64, "set_of"),
        ("blsi", 32, "no_zext32"),
    )
    for op, bits, quirk in mutant_specs:
        fn = lifted_fns[f"op_{op}{bits}_flags"]
        caught = 0
        for v in vals[:32]:
            bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
            got_val = fn(v, 0, po)
            got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, f"mut_{op}"))[0]
            mut_val, mut_flags = ref_bmi1(op, v, bits, quirk=quirk)
            if got_val != mut_val or got_flags != mut_flags:
                caught += 1
        if not caught:
            raise AssertionError(f"mutant {quirk} on {op}{bits} survived")

    return diverged, len(mutant_specs)


def compile_o2_and_verify(original_path, cxx_path, vals):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "bmi1_aot_o2.cpp"
        dll_path = tmp / "bmi1_aot_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'bmi1_aot_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            return verify_suite(o2_fns, "lifted_o2", vals)
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
        for name in EXPORTS:
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

    # 1) Lift + rejections + portable lifted + mutants + /O2 run unconditionally BEFORE CPUID skip!
    check_rejections(original_path)

    vals = test_values()
    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_suite(lifted_fns, "lifted", vals)
        diverged, mutants = check_altered_and_mutants(lifted_fns, alt_lifted_fns, vals)
        o2_cases = compile_o2_and_verify(original_path, cxx_path, vals)

        if not host_has_bmi1(lifted_lib):
            print(
                f"portable lifted passed ({lifted_cases} lifted + {o2_cases} /O2 cases, "
                f"{diverged} ALTER_BMI1 + {mutants} mutants); "
                "Host lacks CPUID.07H.00H:EBX[3] (BMI1), skipping native differential check"
            )
            sys.exit(77)

        # 2) Native BMI1 hardware differential checks.
        orig_lib, orig_fns = bind_exports(original_path)
        alt_lib, alt_fns = bind_exports(altered_path)
        try:
            native_cases = verify_suite(orig_fns, "native", vals)
            for name in ALTERED_REG_EXPORTS:
                for v in vals[:32]:
                    got_an = alt_fns[name](v, 0, 0)
                    got_al = alt_lifted_fns[name](v, 0, 0)
                    if got_an != got_al:
                        raise AssertionError(f"altered native vs lifted {name}({v:#x}): {got_an:#x} != {got_al:#x}")
        finally:
            orig_fns.clear()
            alt_fns.clear()
            unload_lib(orig_lib)
            unload_lib(alt_lib)
    finally:
        lifted_fns.clear()
        alt_lifted_fns.clear()
        unload_lib(lifted_lib)
        unload_lib(alt_lifted_lib)

    record_covered_forms(original_path)
    print(
        f"bmi1_aot passed: {native_cases} native + {lifted_cases} lifted + {o2_cases} /O2 cases "
        f"across {len(EXPORTS)} exports; ALTER_BMI1 diverged on {diverged} exports + {mutants} mutants caught"
    )


if __name__ == "__main__":
    main()
