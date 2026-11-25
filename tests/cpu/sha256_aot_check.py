"""Differential, NIST FIPS 180-4, aliasing, /O2, mutation, and rejection checks for SHA-256 NI AOT lifting."""

import ctypes
import hashlib
import os
from pathlib import Path
import random
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
from cpu_sha256_aot import lift_code, translate  # noqa: E402

EXPORTS = (
    "op_sha256rnds2_reg",
    "op_sha256rnds2_mem",
    "op_sha256rnds2_mem_r8",
    "op_sha256rnds2_self_src",
    "op_sha256rnds2_dst_xmm0",
    "op_sha256rnds2_src_xmm0",
    "op_sha256rnds2_all_xmm0",
    "op_sha256msg1_reg",
    "op_sha256msg1_mem",
    "op_sha256msg1_self",
    "op_sha256msg2_reg",
    "op_sha256msg2_mem",
    "op_sha256msg2_self",
    "op_sha256_pipeline",
    "op_sha256_discarded_read",
)

UNALIGNED_SAFE_EXPORTS = (
    "op_sha256rnds2_reg",
    "op_sha256rnds2_self_src",
    "op_sha256rnds2_dst_xmm0",
    "op_sha256rnds2_src_xmm0",
    "op_sha256rnds2_all_xmm0",
    "op_sha256msg1_reg",
    "op_sha256msg1_self",
    "op_sha256msg2_reg",
    "op_sha256msg2_self",
)

ALTERED_EXPORTS = (
    "op_sha256rnds2_reg",
    "op_sha256rnds2_mem",
    "op_sha256msg1_reg",
    "op_sha256msg2_reg",
    "op_sha256_pipeline",
)

GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))
MASK32 = 0xFFFFFFFF

SHA256_K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)


# --- Independent Python reference for SHA-256 NI instructions ---

def _ror32(x, n):
    x &= MASK32
    return ((x >> n) | (x << (32 - n))) & MASK32


def _unpack4(buf):
    return struct.unpack("<4I", buf)


def _pack4(w0, w1, w2, w3):
    return struct.pack("<4I", w0 & MASK32, w1 & MASK32, w2 & MASK32, w3 & MASK32)


def _ch(e, f, g):
    return ((e & f) ^ ((~e & MASK32) & g)) & MASK32


def _maj(a, b, c):
    return ((a & b) ^ (a & c) ^ (b & c)) & MASK32


def _cap_s0(a):
    return _ror32(a, 2) ^ _ror32(a, 13) ^ _ror32(a, 22)


def _cap_s1(e):
    return _ror32(e, 6) ^ _ror32(e, 11) ^ _ror32(e, 25)


def _s0(x):
    return _ror32(x, 7) ^ _ror32(x, 18) ^ ((x & MASK32) >> 3)


def _s1(x):
    return _ror32(x, 17) ^ _ror32(x, 19) ^ ((x & MASK32) >> 10)


def ref_sha256rnds2(dst_bytes, src_bytes, wk_bytes):
    h, g, d, c = _unpack4(dst_bytes)
    f, e, b, a = _unpack4(src_bytes)
    wk0, wk1, _, _ = _unpack4(wk_bytes)

    t1_0 = (_ch(e, f, g) + _cap_s1(e) + wk0 + h) & MASK32
    t2_0 = (_maj(a, b, c) + _cap_s0(a)) & MASK32
    a1 = (t1_0 + t2_0) & MASK32
    b1, c1, d1 = a, b, c
    e1 = (t1_0 + d) & MASK32
    f1, g1, h1 = e, f, g

    t1_1 = (_ch(e1, f1, g1) + _cap_s1(e1) + wk1 + h1) & MASK32
    t2_1 = (_maj(a1, b1, c1) + _cap_s0(a1)) & MASK32
    a2 = (t1_1 + t2_1) & MASK32
    e2 = (t1_1 + d1) & MASK32
    return _pack4(e1, e2, a1, a2)


def ref_sha256msg1(dst_bytes, src_bytes):
    w0, w1, w2, w3 = _unpack4(dst_bytes)
    w4, _, _, _ = _unpack4(src_bytes)
    return _pack4(
        (w0 + _s0(w1)) & MASK32,
        (w1 + _s0(w2)) & MASK32,
        (w2 + _s0(w3)) & MASK32,
        (w3 + _s0(w4)) & MASK32,
    )


def ref_sha256msg2(dst_bytes, src_bytes):
    d0, d1, d2, d3 = _unpack4(dst_bytes)
    _, _, s2, s3 = _unpack4(src_bytes)
    w0 = (d0 + _s1(s2)) & MASK32
    w1 = (d1 + _s1(s3)) & MASK32
    w2 = (d2 + _s1(w0)) & MASK32
    w3 = (d3 + _s1(w1)) & MASK32
    return _pack4(w0, w1, w2, w3)


def reference(name, a, b, c, *, altered=False):
    if name == "op_sha256rnds2_reg":
        return ref_sha256rnds2(b, a, c) if altered else ref_sha256rnds2(a, b, c)
    if name == "op_sha256rnds2_mem":
        return ref_sha256rnds2(c, b, c) if altered else ref_sha256rnds2(a, b, c)
    if name == "op_sha256rnds2_mem_r8":
        return ref_sha256rnds2(a, c, b)
    if name == "op_sha256rnds2_self_src":
        return ref_sha256rnds2(a, a, b)
    if name == "op_sha256rnds2_dst_xmm0":
        return ref_sha256rnds2(a, b, a)
    if name == "op_sha256rnds2_src_xmm0":
        return ref_sha256rnds2(a, b, b)
    if name == "op_sha256rnds2_all_xmm0":
        return ref_sha256rnds2(a, a, a)
    if name == "op_sha256msg1_reg":
        return ref_sha256msg2(a, b) if altered else ref_sha256msg1(a, b)
    if name == "op_sha256msg1_mem":
        return ref_sha256msg1(a, b)
    if name == "op_sha256msg1_self":
        return ref_sha256msg1(a, a)
    if name == "op_sha256msg2_reg":
        return ref_sha256msg1(a, b) if altered else ref_sha256msg2(a, b)
    if name == "op_sha256msg2_mem":
        return ref_sha256msg2(a, b)
    if name == "op_sha256msg2_self":
        return ref_sha256msg2(a, a)
    if name == "op_sha256_pipeline":
        x2 = ref_sha256msg1(a, b)
        x2 = ref_sha256msg1(x2, b) if altered else ref_sha256msg2(x2, b)
        x3 = ref_sha256rnds2(a, b, x2)
        return ref_sha256rnds2(b, x3, x2)
    if name == "op_sha256_discarded_read":
        return a
    raise KeyError(name)


def all_triples():
    """Edge, carry-wrap, lane-isolation, FIPS IV, dirty-XMM0-upper, and deterministic random 16B triples."""
    z16 = b"\x00" * 16
    f16 = b"\xff" * 16
    fips_abef = _pack4(0x9B05688C, 0x510E527F, 0xBB67AE85, 0x6A09E667)
    fips_cdgh = _pack4(0x5BE0CD19, 0x1F83D9AB, 0xA54FF53A, 0x3C6EF372)
    triples = [
        (z16, z16, z16),
        (f16, f16, f16),
        (z16, f16, z16),
        (f16, z16, f16),
        (fips_cdgh, fips_abef, _pack4((0x61626380 + SHA256_K[0]) & MASK32, SHA256_K[1], 0xDEADBEEF, 0xCAFEBABE)),
        # Same WK[63:0] with completely different XMM0[127:64] to prove upper 64 bits are ignored.
        (fips_cdgh, fips_abef, _pack4((0x61626380 + SHA256_K[0]) & MASK32, SHA256_K[1], 0x00000000, 0x00000000)),
        # Carry wraps and rotation boundaries across all lanes.
        (_pack4(0xFFFFFFFF, 0x80000000, 0x00000001, 0x7FFFFFFF),
         _pack4(0x00000001, 0x7FFFFFFF, 0xFFFFFFFF, 0x80000000),
         _pack4(0xFFFFFFFF, 0x00000001, 0x55555555, 0xAAAAAAAA)),
        *[
            (_pack4(*(0xA5A55A5A if j == i else 0 for j in range(4))),
             _pack4(*(0x3C3CC3C3 if j == k else 0 for j in range(4))),
             _pack4(*(0x96966969 if j == ((i + k) & 3) else 0 for j in range(4))))
            for i in range(4) for k in range(4)
        ],
    ]
    rng = random.Random(0x5A256A07)
    for _ in range(72):
        triples.append((rng.randbytes(16), rng.randbytes(16), rng.randbytes(16)))
    return triples


def alloc_guarded(payload, *, misalign=0):
    size = 48 + len(GUARD_PRE) + len(payload) + len(GUARD_POST)
    raw = ctypes.create_string_buffer(size)
    base = ctypes.addressof(raw)
    aligned = (base + len(GUARD_PRE) + 15) & ~15
    data_addr = aligned + misalign
    pre_off = (data_addr - len(GUARD_PRE)) - base
    data_off = data_addr - base
    post_off = data_off + len(payload)
    raw[pre_off:data_off] = GUARD_PRE
    raw[data_off:post_off] = payload
    raw[post_off:post_off + len(GUARD_POST)] = GUARD_POST
    return raw, pre_off, data_off, post_off, data_addr


def invoke(fn, a, b, c=b"\x5a" * 16, *, a_misalign=0, b_misalign=0, out_misalign=0):
    buf_a, a_pre, a_off, a_post, a_addr = alloc_guarded(a, misalign=a_misalign)
    buf_b, b_pre, b_off, b_post, b_addr = alloc_guarded(b, misalign=b_misalign)
    buf_out, o_pre, o_off, o_post, o_addr = alloc_guarded(c, misalign=out_misalign)
    fn(a_addr, b_addr, o_addr)
    assert buf_a.raw[a_pre:a_off] == GUARD_PRE and buf_a.raw[a_post:a_post + 16] == GUARD_POST, "input A guards corrupted"
    assert buf_b.raw[b_pre:b_off] == GUARD_PRE and buf_b.raw[b_post:b_post + 16] == GUARD_POST, "input B guards corrupted"
    assert buf_out.raw[o_pre:o_off] == GUARD_PRE and buf_out.raw[o_post:o_post + 16] == GUARD_POST, "output guards corrupted"
    assert buf_a.raw[a_off:a_post] == a, "input A mutated unexpectedly"
    assert buf_b.raw[b_off:b_post] == b, "input B mutated unexpectedly"
    return buf_out.raw[o_off:o_post]


def invoke_aliased(fn, a, b, c, mode):
    buf_a, a_pre, a_off, a_post, a_addr = alloc_guarded(a)
    buf_b, b_pre, b_off, b_post, b_addr = alloc_guarded(b)
    buf_out, o_pre, o_off, o_post, o_addr = alloc_guarded(c)
    if mode == "rcx==rdx":
        fn(a_addr, a_addr, o_addr)
        res = buf_out.raw[o_off:o_post]
    elif mode == "r8==rcx":
        fn(a_addr, b_addr, a_addr)
        res = buf_a.raw[a_off:a_post]
    elif mode == "r8==rdx":
        fn(a_addr, b_addr, b_addr)
        res = buf_b.raw[b_off:b_post]
    elif mode == "all":
        fn(a_addr, a_addr, a_addr)
        res = buf_a.raw[a_off:a_post]
    else:
        raise ValueError(f"unknown alias mode {mode}")
    assert buf_a.raw[a_pre:a_off] == GUARD_PRE and buf_a.raw[a_post:a_post + 16] == GUARD_POST, f"{mode}: A guards corrupted"
    assert buf_b.raw[b_pre:b_off] == GUARD_PRE and buf_b.raw[b_post:b_post + 16] == GUARD_POST, f"{mode}: B guards corrupted"
    assert buf_out.raw[o_pre:o_off] == GUARD_PRE and buf_out.raw[o_post:o_post + 16] == GUARD_POST, f"{mode}: out guards corrupted"
    return res


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def host_has_sha(lifted_lib):
    if hasattr(lifted_lib, "host_has_sha"):
        lifted_lib.host_has_sha.argtypes = []
        lifted_lib.host_has_sha.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_sha())
    try:
        return " sha_ni" in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False


def encode_seq(*instructions):
    enc = Encoder(64)
    rip = 0x1000
    for ins in instructions:
        rip += enc.encode(ins, rip)
    return enc.take_buffer()


def check_translator(original_path):
    ret = Instruction.create(Code.RETNQ)
    ld0 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld1 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld2 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM2, MemoryOperand(R.R8))
    ld0_a = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld1_a = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM1, MemoryOperand(R.R8))
    st0 = Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    st1 = Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM1)
    st1_a = Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM1)

    # Positive checks for all 3 SHA-256 NI instructions (reg & mem) + no <immintrin.h>.
    accepted = 0
    src_reg, _ = lift_code(
        "sha256rnds2_reg", 0x1000,
        encode_seq(ld0, ld1, ld2, Instruction.create_reg_reg(Code.SHA256RNDS2_XMM_XMMM128, R.XMM1, R.XMM2), st1, ret),
    )
    assert "xmm1 = op_sha256rnds2(xmm1, xmm2, xmm0);" in src_reg
    assert "<immintrin.h>" not in src_reg
    src_mem, _ = lift_code(
        "sha256rnds2_mem", 0x1000,
        encode_seq(ld0_a, ld1_a, Instruction.create_reg_mem(Code.SHA256RNDS2_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX)), st1_a, ret),
    )
    assert "xmm1 = op_sha256rnds2(xmm1, load128(rdx), xmm0);" in src_mem
    accepted += 1

    for code, helper in (
        (Code.SHA256MSG1_XMM_XMMM128, "op_sha256msg1"),
        (Code.SHA256MSG2_XMM_XMMM128, "op_sha256msg2"),
    ):
        s_reg, _ = lift_code(f"{helper}_reg", 0x1000, encode_seq(ld0, ld1, Instruction.create_reg_reg(code, R.XMM0, R.XMM1), st0, ret))
        assert f"xmm0 = {helper}(xmm0, xmm1);" in s_reg
        s_mem, _ = lift_code(f"{helper}_mem", 0x1000, encode_seq(ld0_a, Instruction.create_reg_mem(code, R.XMM0, MemoryOperand(R.RDX)), st0, ret))
        assert f"xmm0 = {helper}(xmm0, load128(rdx));" in s_mem
        accepted += 1

    # Negative rejection controls.
    rep_ret = Instruction.create(Code.RETNQ)
    rep_ret.has_rep_prefix = True
    fs_ld = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    fs_ld.segment_prefix = R.FS

    cases = [
        # SHA-1 rejections
        ("sha1rnds4", (ld0, ld1, Instruction.create_reg_reg_u32(Code.SHA1RNDS4_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), st0, ret),
         "unsupported SHA-1 instruction"),
        ("sha1nexte", (ld0, ld1, Instruction.create_reg_reg(Code.SHA1NEXTE_XMM_XMMM128, R.XMM0, R.XMM1), st0, ret),
         "unsupported SHA-1 instruction"),
        ("sha1msg1", (ld0, ld1, Instruction.create_reg_reg(Code.SHA1MSG1_XMM_XMMM128, R.XMM0, R.XMM1), st0, ret),
         "unsupported SHA-1 instruction"),
        ("sha1msg2", (ld0, ld1, Instruction.create_reg_reg(Code.SHA1MSG2_XMM_XMMM128, R.XMM0, R.XMM1), st0, ret),
         "unsupported SHA-1 instruction"),
        # AES / PCLMUL rejections
        ("aesenc", (ld0, ld1, Instruction.create_reg_reg(Code.AESENC_XMM_XMMM128, R.XMM0, R.XMM1), st0, ret),
         "unsupported AES/PCLMUL instruction"),
        ("aesdec", (ld0, ld1, Instruction.create_reg_reg(Code.AESDEC_XMM_XMMM128, R.XMM0, R.XMM1), st0, ret),
         "unsupported AES/PCLMUL instruction"),
        ("aeskeygenassist", (ld0, ld1, Instruction.create_reg_reg_u32(Code.AESKEYGENASSIST_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 1), st0, ret),
         "unsupported AES/PCLMUL instruction"),
        ("pclmulqdq", (ld0, ld1, Instruction.create_reg_reg_u32(Code.PCLMULQDQ_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), st0, ret),
         "unsupported AES/PCLMUL instruction"),
        # VEX / EVEX rejections
        ("vex_vmovdqu", (Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), st0, ret),
         "unsupported VEX encoding"),
        ("vex_vpxor", (ld0, ld1, Instruction.create_reg_reg_reg(Code.VEX_VPXOR_XMM_XMM_XMMM128, R.XMM0, R.XMM0, R.XMM1), st0, ret),
         "unsupported VEX encoding"),
        ("evex_vmovdqu32", (Instruction.create_reg_mem(Code.EVEX_VMOVDQU32_XMM_K1Z_XMMM128, R.XMM0, MemoryOperand(R.RCX)), st0, ret),
         "unsupported EVEX encoding"),
        # Nonvolatile XMM6..15
        ("nonvol_xmm6_dst", (ld0, ld1, Instruction.create_reg_reg(Code.SHA256RNDS2_XMM_XMMM128, R.XMM6, R.XMM1), ret),
         "nonvolatile vector register"),
        ("nonvol_xmm15_src", (ld0, Instruction.create_reg_reg(Code.SHA256MSG1_XMM_XMMM128, R.XMM0, R.XMM15), st0, ret),
         "nonvolatile vector register"),
        # Undefined XMM reads — including undefined implicit XMM0 on SHA256RNDS2!
        ("undef_implicit_xmm0_rnds2", (ld1, ld2, Instruction.create_reg_reg(Code.SHA256RNDS2_XMM_XMMM128, R.XMM1, R.XMM2), st1, ret),
         "undefined implicit vector register xmm0"),
        ("undef_dst_rnds2", (ld0, ld2, Instruction.create_reg_reg(Code.SHA256RNDS2_XMM_XMMM128, R.XMM1, R.XMM2), st1, ret),
         "undefined vector register xmm1"),
        ("undef_src_rnds2", (ld0, ld1, Instruction.create_reg_reg(Code.SHA256RNDS2_XMM_XMMM128, R.XMM1, R.XMM3), st1, ret),
         "undefined vector register xmm3"),
        ("undef_dst_msg1", (ld1, Instruction.create_reg_reg(Code.SHA256MSG1_XMM_XMMM128, R.XMM3, R.XMM1), ret),
         "undefined vector register xmm3"),
        ("undef_src_msg2", (ld0, Instruction.create_reg_reg(Code.SHA256MSG2_XMM_XMMM128, R.XMM0, R.XMM4), st0, ret),
         "undefined vector register xmm4"),
        ("undef_store", (st0, ret), "undefined vector register xmm0"),
        # Alignment / displacement / addressing / store-base / prefix / control-flow / ret checks
        ("unaligned_movdqa_load", (Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=8, displ_size=1)), st0, ret),
         "unaligned 16-byte memory operand"),
        ("unaligned_movdqa_store", (ld0, Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8, displ=4, displ_size=1), R.XMM0), ret),
         "unaligned 16-byte memory operand"),
        ("unaligned_sha256_mem", (ld0, ld1, Instruction.create_reg_mem(Code.SHA256RNDS2_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX, displ=8, displ_size=1)), st1, ret),
         "unaligned 16-byte memory operand"),
        ("displaced_movdqu", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=16, displ_size=1)), st0, ret),
         "unsupported displaced/out-of-buffer memory operand"),
        ("displaced_sha256msg2", (ld0, Instruction.create_reg_mem(Code.SHA256MSG2_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=16, displ_size=1)), st0, ret),
         "unsupported displaced/out-of-buffer memory operand"),
        ("indexed_mem", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, index=R.RDX)), st0, ret),
         "unsupported memory operand"),
        ("rip_rel_mem", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RIP, displ=0x1100, displ_size=4)), st0, ret),
         "unsupported memory operand"),
        ("non_arg_base_rax", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RAX)), st0, ret),
         "unsupported memory base"),
        ("nonvol_base_rbx", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RBX)), st0, ret),
         "nonvolatile GPR base"),
        ("store_to_rcx", (ld0, Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.RCX), R.XMM0), ret),
         "store outside output buffer r8"),
        ("store_to_rdx_movdqa", (ld0_a, Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.RDX), R.XMM0), ret),
         "store outside output buffer r8"),
        ("prefix_segment_fs", (fs_ld, st0, ret), "unsupported instruction prefix"),
        ("prefix_rep_ret", (ld0, st0, rep_ret), "unsupported instruction prefix"),
        ("branch_jmp", (ld0, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), st0, ret),
         "unsupported control-flow instruction"),
        ("missing_ret", (ld0, ld1, Instruction.create_reg_reg(Code.SHA256MSG1_XMM_XMMM128, R.XMM0, R.XMM1), st0),
         "missing terminal ret"),
        ("trailing_after_ret", (ld0, st0, ret, ld0), "unreachable instruction after ret"),
    ]
    for label, seq, reason in cases:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected control")

    # End-to-end PE-patched rejections + CLI non-traceback rejection.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_sha256_pipeline")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {label: (encode_seq(*seq), reason) for label, seq, reason in cases}
    pe_labels = (
        "sha1rnds4",
        "aesenc",
        "vex_vmovdqu",
        "undef_implicit_xmm0_rnds2",
        "nonvol_xmm6_dst",
        "unaligned_sha256_mem",
        "store_to_rcx",
        "branch_jmp",
        "prefix_segment_fs",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_sha256_aot.py"
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
                translate(patched, ["op_sha256_pipeline"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched), "op_sha256_pipeline", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

    return accepted, len(cases), len(pe_labels)


# --- Full NIST FIPS 180-4 SHA-256 64-round compression driven through lifted DLL exports ---

def _add4(a_bytes, b_bytes):
    return _pack4(*((x + y) & MASK32 for x, y in zip(_unpack4(a_bytes), _unpack4(b_bytes))))


def sha256_digest_via_funcs(funcs, message, *, use_mem=False):
    """Compute a full FIPS 180-4 SHA-256 digest using only the lifted SHA-256 NI exports."""
    suffix = "mem" if use_mem else "reg"
    rnds2 = funcs[f"op_sha256rnds2_{suffix}"]
    msg1 = funcs[f"op_sha256msg1_{suffix}"]
    msg2 = funcs[f"op_sha256msg2_{suffix}"]

    bit_len = len(message) * 8
    padded = message + b"\x80"
    padded += b"\x00" * ((56 - len(padded) % 64) % 64)
    padded += struct.pack(">Q", bit_len)

    h = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ]
    for off in range(0, len(padded), 64):
        words = list(struct.unpack(">16I", padded[off:off + 64]))
        msgs = [_pack4(*words[4 * i:4 * i + 4]) for i in range(4)]
        for k in range(4, 16):
            t1 = invoke(msg1, msgs[k - 4], msgs[k - 3])
            # W[4k-7 .. 4k-4] = (msgs[k-2].w[1..3], msgs[k-1].w[0]).
            prev2 = _unpack4(msgs[k - 2])
            prev1 = _unpack4(msgs[k - 1])
            w_m7 = _pack4(prev2[1], prev2[2], prev2[3], prev1[0])
            t2 = _add4(t1, w_m7)
            msgs.append(invoke(msg2, t2, msgs[k - 1]))
        sched = [w for m in msgs for w in _unpack4(m)]

        abef = _pack4(h[5], h[4], h[1], h[0])
        cdgh = _pack4(h[7], h[6], h[3], h[2])
        for r in range(0, 64, 2):
            # Put non-zero dirty words in XMM0[127:64] to prove SHA256RNDS2 ignores them.
            wk = _pack4(
                (sched[r] + SHA256_K[r]) & MASK32,
                (sched[r + 1] + SHA256_K[r + 1]) & MASK32,
                0xDEADBEEF ^ r,
                0xCAFEBABE ^ (r << 8),
            )
            next_abef = invoke(rnds2, cdgh, abef, wk)
            cdgh, abef = abef, next_abef

        f, e, b, a = _unpack4(abef)
        hh, g, d, c = _unpack4(cdgh)
        for i, delta in enumerate((a, b, c, d, e, f, g, hh)):
            h[i] = (h[i] + delta) & MASK32

    return struct.pack(">8I", *h).hex()


FIPS_180_4_VECTORS = (
    (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    (
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    ),
    (
        b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
        b"ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
    ),
    (b"a" * 1000, hashlib.sha256(b"a" * 1000).hexdigest()),
)


def check_discarded_read(funcs, label):
    payload = all_triples()[4][0]
    ba, a_pre, a_off, a_post, a_addr = alloc_guarded(payload)
    bo_ok, o_pre, o_off, o_post, o_addr = alloc_guarded(b"\x5a" * 16)
    funcs["op_sha256_discarded_read"](a_addr, a_addr, o_addr)
    assert bo_ok.raw[o_off:o_post] == payload, f"{label} discarded_read(valid) mismatch"

    bo_null, o2_pre, o2_off, o2_post, o2_addr = alloc_guarded(b"\x5a" * 16)
    try:
        funcs["op_sha256_discarded_read"](a_addr, 0, o2_addr)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_sha256_discarded_read(NULL) dropped faulting m128 read")
    assert bo_null.raw[o2_off:o2_post] == b"\x5a" * 16, f"{label} wrote output after fault"
    del ba


def check_lifted_unconditional(lifted, altered_lifted, triples):
    total = 0
    for name in EXPORTS:
        for index, (a, b, c) in enumerate(triples):
            want = reference(name, a, b, c)
            got = invoke(lifted[name], a, b, c)
            if got != want:
                raise AssertionError(f"lifted {name} case {index}: got {got.hex()} != exp {want.hex()}")
            total += 1

    check_discarded_read(lifted, "lifted")

    # Verify dirty XMM0[127:64] produces identical SHA256RNDS2 results.
    a4, b4, c4 = triples[4]
    a5, b5, c5 = triples[5]
    assert invoke(lifted["op_sha256rnds2_reg"], a4, b4, c4) == invoke(lifted["op_sha256rnds2_reg"], a5, b5, c5)

    # Unaligned buffers on movdqu-only exports.
    a, b, c = triples[4]
    for name in UNALIGNED_SAFE_EXPORTS:
        want = reference(name, a, b, c)
        for mis in ((1, 0, 0), (0, 3, 0), (0, 0, 5), (7, 5, 3)):
            got = invoke(lifted[name], a, b, c, a_misalign=mis[0], b_misalign=mis[1], out_misalign=mis[2])
            if got != want:
                raise AssertionError(f"lifted {name} misalign {mis}: got {got.hex()} != exp {want.hex()}")
            total += 1

    # Pointer aliasing checks (rcx==rdx, r8==rcx, r8==rdx, all).
    for name in (
        "op_sha256rnds2_reg", "op_sha256rnds2_mem", "op_sha256rnds2_mem_r8",
        "op_sha256msg1_reg", "op_sha256msg1_mem",
        "op_sha256msg2_reg", "op_sha256msg2_mem",
        "op_sha256_pipeline",
    ):
        for mode in ("rcx==rdx", "r8==rcx", "r8==rdx", "all"):
            ea = a
            eb = a if mode in ("rcx==rdx", "all") else b
            ec = a if mode in ("r8==rcx", "all") else (b if mode == "r8==rdx" else c)
            want = reference(name, ea, eb, ec)
            got = invoke_aliased(lifted[name], a, b, c, mode)
            if got != want:
                raise AssertionError(f"lifted {name} alias {mode}: got {got.hex()} != exp {want.hex()}")
            total += 1

    # Full NIST FIPS 180-4 SHA-256 digests (single-block & multi-block) through reg and mem exports.
    for use_mem in (False, True):
        for msg, expected_hex in FIPS_180_4_VECTORS:
            assert hashlib.sha256(msg).hexdigest() == expected_hex
            got_hex = sha256_digest_via_funcs(lifted, msg, use_mem=use_mem)
            if got_hex != expected_hex:
                raise AssertionError(f"FIPS 180-4 SHA-256 (mem={use_mem}) on len={len(msg)}: {got_hex} != {expected_hex}")
            total += 1

    # Altered lifted build matches altered reference and diverges from original lifted.
    for name in ALTERED_EXPORTS:
        changed = 0
        for a_in, b_in, c_in in triples[:24]:
            orig = invoke(lifted[name], a_in, b_in, c_in)
            alt = invoke(altered_lifted[name], a_in, b_in, c_in)
            want_alt = reference(name, a_in, b_in, c_in, altered=True)
            if alt != want_alt:
                raise AssertionError(f"altered_lifted {name}: got {alt.hex()} != exp {want_alt.hex()}")
            changed += alt != orig
        if not changed:
            raise AssertionError(f"altered_lifted did not diverge on {name}")

    # Faulty mutant models (including parallel SHA256MSG2 missing sequential w0/w1 dependency!).
    def _mut_msg2_parallel(x, y, _z):
        d0, d1, d2, d3 = _unpack4(x)
        _, _, s2, s3 = _unpack4(y)
        return _pack4(
            (d0 + _s1(s2)) & MASK32,
            (d1 + _s1(s3)) & MASK32,
            (d2 + _s1(d0)) & MASK32,  # bug: uses un-updated d0 instead of w0
            (d3 + _s1(d1)) & MASK32,  # bug: uses un-updated d1 instead of w1
        )

    def _mut_msg1_bad_src_lane(x, y, _z):
        w0, w1, w2, w3 = _unpack4(x)
        _, w5, _, _ = _unpack4(y)
        return _pack4((w0 + _s0(w1)) & MASK32, (w1 + _s0(w2)) & MASK32, (w2 + _s0(w3)) & MASK32, (w3 + _s0(w5)) & MASK32)

    def _mut_rnds2_swap_wk(x, y, z):
        wk0, wk1, wk2, wk3 = _unpack4(z)
        return ref_sha256rnds2(x, y, _pack4(wk1, wk0, wk2, wk3))

    def _mut_rnds2_swap_state(x, y, z):
        return ref_sha256rnds2(y, x, z)

    def _mut_rnds2_wrong_sigma(x, y, z):
        h, g, d, c = _unpack4(x)
        f, e, b, a = _unpack4(y)
        wk0, wk1, _, _ = _unpack4(z)
        t1_0 = (_ch(e, f, g) + _cap_s0(e) + wk0 + h) & MASK32  # bug: Sigma0 instead of Sigma1
        t2_0 = (_maj(a, b, c) + _cap_s0(a)) & MASK32
        a1 = (t1_0 + t2_0) & MASK32
        e1 = (t1_0 + d) & MASK32
        t1_1 = (_ch(e1, e, f) + _cap_s1(e1) + wk1 + g) & MASK32
        t2_1 = (_maj(a1, a, b) + _cap_s0(a1)) & MASK32
        return _pack4(e1, (t1_1 + c) & MASK32, a1, (t1_1 + t2_1) & MASK32)

    mutants = (
        ("op_sha256rnds2_reg", _mut_rnds2_swap_wk),
        ("op_sha256rnds2_reg", _mut_rnds2_swap_state),
        ("op_sha256rnds2_reg", _mut_rnds2_wrong_sigma),
        ("op_sha256msg1_reg", lambda x, y, z: ref_sha256msg2(x, y)),
        ("op_sha256msg1_reg", _mut_msg1_bad_src_lane),
        ("op_sha256msg2_reg", lambda x, y, z: ref_sha256msg1(x, y)),
        ("op_sha256msg2_reg", _mut_msg2_parallel),
        ("op_sha256_pipeline", lambda x, y, z: ref_sha256rnds2(x, y, z)),
    )
    for name, wrong in mutants:
        if not any(invoke(lifted[name], x, y, z) != wrong(x, y, z) for x, y, z in triples[:32]):
            raise AssertionError(f"mutant on {name} agreed with lifted on every triple")

    return total, len(mutants)


def compile_o2_and_verify(original_path, cxx_path, triples):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "sha256_aot_o2.cpp"
        dll_path = tmp / "sha256_aot_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'sha256_aot_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind(dll_path, "lifted_")
        try:
            cases = 0
            for name in EXPORTS:
                for idx, (a, b, c) in enumerate(triples[:32]):
                    want = reference(name, a, b, c)
                    got = invoke(o2_fns[name], a, b, c)
                    if got != want:
                        raise AssertionError(f"lifted_o2 {name} case {idx}: {got.hex()} != {want.hex()}")
                    cases += 1
            check_discarded_read(o2_fns, "lifted_o2")
            assert sha256_digest_via_funcs(o2_fns, FIPS_180_4_VECTORS[2][0], use_mem=False) == FIPS_180_4_VECTORS[2][1]
            return cases
        finally:
            o2_fns.clear()
            unload_lib(o2_lib)


def check_native_differential(native, lifted, altered, altered_lifted, triples):
    total = 0
    for name in EXPORTS:
        for index, (a, b, c) in enumerate(triples):
            nat = invoke(native[name], a, b, c)
            lif = invoke(lifted[name], a, b, c)
            if nat != lif:
                raise AssertionError(f"native vs lifted {name} case {index}: native={nat.hex()} lifted={lif.hex()}")
            total += 1

    check_discarded_read(native, "native")

    a, b, c = triples[4]
    for name in UNALIGNED_SAFE_EXPORTS:
        for mis in ((1, 0, 0), (0, 3, 0), (0, 0, 5), (7, 5, 3)):
            nat = invoke(native[name], a, b, c, a_misalign=mis[0], b_misalign=mis[1], out_misalign=mis[2])
            lif = invoke(lifted[name], a, b, c, a_misalign=mis[0], b_misalign=mis[1], out_misalign=mis[2])
            if nat != lif:
                raise AssertionError(f"native vs lifted {name} misalign {mis}: native={nat.hex()} lifted={lif.hex()}")
            total += 1

    for name in (
        "op_sha256rnds2_reg", "op_sha256rnds2_mem", "op_sha256rnds2_mem_r8",
        "op_sha256msg1_reg", "op_sha256msg1_mem",
        "op_sha256msg2_reg", "op_sha256msg2_mem",
        "op_sha256_pipeline",
    ):
        for mode in ("rcx==rdx", "r8==rcx", "r8==rdx", "all"):
            nat = invoke_aliased(native[name], a, b, c, mode)
            lif = invoke_aliased(lifted[name], a, b, c, mode)
            if nat != lif:
                raise AssertionError(f"native vs lifted {name} alias {mode}: native={nat.hex()} lifted={lif.hex()}")
            total += 1

    for name in ALTERED_EXPORTS:
        changed = 0
        for a_in, b_in, c_in in triples[:24]:
            alt_nat = invoke(altered[name], a_in, b_in, c_in)
            alt_lif = invoke(altered_lifted[name], a_in, b_in, c_in)
            if alt_lif != alt_nat:
                raise AssertionError(f"altered {name}: lifted={alt_lif.hex()} != native={alt_nat.hex()}")
            orig_nat = invoke(native[name], a_in, b_in, c_in)
            changed += alt_nat != orig_nat
        if not changed:
            raise AssertionError(f"altered native build did not diverge on {name}")

    return total


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

    original_path, lifted_path, altered_path, altered_lifted_path, cxx_path = (Path(p) for p in sys.argv[1:6])

    accepted, rejected, pe_rejected = check_translator(original_path)
    print(f"{accepted} SHA-256 NI forms accepted; {rejected} IR + {pe_rejected} PE + CLI rejection controls passed")

    lifted_lib, lifted = bind(lifted_path, "lifted_")
    alt_lifted_lib, altered_lifted = bind(altered_lifted_path, "lifted_")
    triples = all_triples()
    try:
        lifted_total, mutant_count = check_lifted_unconditional(lifted, altered_lifted, triples)
        o2_total = compile_o2_and_verify(original_path, cxx_path, triples)
        print(
            f"{lifted_total} portable lifted + {o2_total} /O2 cases (incl. NIST FIPS 180-4 single/multi-block & aliasing) "
            f"passed; {mutant_count} mutants + {len(ALTERED_EXPORTS)} altered-build controls caught"
        )

        if not host_has_sha(lifted_lib):
            print("Host lacks CPUID.07H.00H:EBX[29] (SHA-NI); skipping native differential check")
            sys.exit(77)

        orig_lib, native = bind(original_path, "")
        alt_lib, altered = bind(altered_path, "")
        try:
            native_total = check_native_differential(native, lifted, altered, altered_lifted, triples)
            print(f"{native_total} native vs lifted SHA-NI differential cases matched bit for bit")
        finally:
            native.clear()
            altered.clear()
            unload_lib(orig_lib)
            unload_lib(alt_lib)
    finally:
        lifted.clear()
        altered_lifted.clear()
        unload_lib(lifted_lib)
        unload_lib(alt_lifted_lib)

    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
