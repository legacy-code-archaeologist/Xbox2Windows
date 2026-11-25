"""Packed VOP3P math in tools/startup/gfx2hlsl.py (Part of #1723).

usage: startup_gfx2hlsl_vop3p_check.py

v_pk_{add,mul,max,min,fma}_f16, v_pk_{add,sub,max,min}_{i16,u16}, v_dot2_f32_f16 and v_dot4_u32_u8 had no
handler, so a shader that used one was refused. Each now lowers to two independent 16-bit lanes (or the dot
sum), honouring op_sel / op_sel_hi / neg_lo / neg_hi. Unmodelled modifiers and operands stay refusals.

The instruction bytes are constructed here from the layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3P branch):
  w0: [31:23] = 0b110011000, [22:16] opcode, [15] clamp, [14] op_sel_hi[2], [13:11] op_sel, [10:8] neg_hi, [7:0] vdst
  w1: [8:0] src0, [17:9] src1, [26:18] src2, [28:27] op_sel_hi[1:0], [31:29] neg_lo
"""
import os
from pathlib import Path
import re
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


V1, V2, V3 = 0x101, 0x102, 0x103
OPCODES = {
    'v_pk_add_i16': 0x02, 'v_pk_sub_i16': 0x03, 'v_pk_max_i16': 0x07, 'v_pk_min_i16': 0x08,
    'v_pk_add_u16': 0x0a, 'v_pk_sub_u16': 0x0b, 'v_pk_max_u16': 0x0c, 'v_pk_min_u16': 0x0d,
    'v_pk_fma_f16': 0x0e, 'v_pk_add_f16': 0x0f, 'v_pk_mul_f16': 0x10, 'v_pk_min_f16': 0x11,
    'v_pk_max_f16': 0x12, 'v_dot2_f32_f16': 0x13, 'v_dot4_u32_u8': 0x17,
}


def encode(name, vdst, srcs, *, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1), neg_lo=(0, 0, 0), neg_hi=(0, 0, 0), clamp=0):
    w0 = (0b110011000 << 23) | (OPCODES[name] << 16) | (clamp << 15) | (op_sel_hi[2] << 14) | vdst
    w0 |= sum(bit << (11 + i) for i, bit in enumerate(op_sel)) | sum(bit << (8 + i) for i, bit in enumerate(neg_hi))
    w1 = sum((s & 0x1FF) << (9 * i) for i, s in enumerate(srcs))
    w1 |= (op_sel_hi[0] << 27) | (op_sel_hi[1] << 28)
    w1 |= sum(bit << (29 + i) for i, bit in enumerate(neg_lo))
    return struct.pack('<II', w0, w1)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return decoded[0][3], ' '.join(stage.lines)


def refused(code):
    try:
        translate(code)
    except G.Unsupported:
        return True
    return False


def normal(text):
    return re.sub(r'\s+', ' ', text)


# Default selection: the low lane reads the low halves, the high lane the high halves.
asm, hlsl = translate(encode('v_pk_add_f16', 0, (V1, V2)))
check('pk_add_f16 asm', asm == 'v_pk_add_f16 v0, v1, v2', asm)
check('pk_add_f16 lanes', hlsl == 'v0 = f32tof16(f16tof32((v1 & 0xffffu)) + f16tof32((v2 & 0xffffu))) | '
      '(f32tof16(f16tof32((v1 >> 16)) + f16tof32((v2 >> 16))) << 16);', hlsl)

# op_sel picks the half the low lane reads, op_sel_hi the half the high lane reads.
asm, hlsl = translate(encode('v_pk_mul_f16', 0, (V1, V2), op_sel=(1, 0, 0), op_sel_hi=(0, 1, 1)))
check('pk_mul_f16 op_sel asm', 'op_sel:[1,0]' in asm and 'op_sel_hi:[0,1]' in asm, asm)
check('pk_mul_f16 op_sel lanes', hlsl == 'v0 = f32tof16(f16tof32((v1 >> 16)) * f16tof32((v2 & 0xffffu))) | '
      '(f32tof16(f16tof32((v1 & 0xffffu)) * f16tof32((v2 >> 16))) << 16);', hlsl)

# neg_lo negates the low lane of a source, neg_hi the high lane.
asm, hlsl = translate(encode('v_pk_add_f16', 0, (V1, V2), neg_lo=(0, 1, 0), neg_hi=(1, 0, 0)))
check('pk_add_f16 neg asm', 'neg_lo:[0,1]' in asm and 'neg_hi:[1,0]' in asm, asm)
check('pk_add_f16 neg lanes', '(f16tof32((v2 & 0xffffu)))' not in hlsl and '(-f16tof32((v2 & 0xffffu)))' in hlsl
      and '(-f16tof32((v1 >> 16)))' in hlsl and '(-f16tof32((v1 & 0xffffu)))' not in hlsl, hlsl)

# clamp saturates each float lane.
_, hlsl = translate(encode('v_pk_max_f16', 0, (V1, V2), clamp=1))
check('pk_max_f16 clamp', hlsl.count('saturate(max(') == 2, hlsl)

_, hlsl = translate(encode('v_pk_fma_f16', 0, (V1, V2, V3)))
check('pk_fma_f16 single rounding', 'fma16_rne((v1 & 0xffffu), (v2 & 0xffffu), (v3 & 0xffffu))' in hlsl
      and 'fma16_rne((v1 >> 16), (v2 >> 16), (v3 >> 16))' in hlsl, hlsl)
_, hlsl = translate(encode('v_pk_min_f16', 0, (V1, V2)))
check('pk_min_f16', hlsl.count('min(') == 2, hlsl)

# Integer lanes wrap to 16 bits; the signed forms compare after sign extension.
_, hlsl = translate(encode('v_pk_sub_u16', 0, (V1, V2)))
check('pk_sub_u16', hlsl == 'v0 = (((v1 & 0xffffu) - (v2 & 0xffffu)) & 0xffffu) | '
      '((((v1 >> 16) - (v2 >> 16)) & 0xffffu) << 16);', hlsl)
_, hlsl = translate(encode('v_pk_max_i16', 0, (V1, V2)))
check('pk_max_i16 sign extends', hlsl.count('>> 16') >= 4 and '(int((((v1 & 0xffffu)) << 16)) >> 16)' in hlsl, hlsl)
for name in ('v_pk_add_i16', 'v_pk_add_u16', 'v_pk_sub_i16', 'v_pk_max_u16', 'v_pk_min_i16', 'v_pk_min_u16'):
    check(name + ' lowers', not refused(encode(name, 0, (V1, V2))))

# v_dot2_f32_f16: low product plus high product plus the binary32 accumulator.
_, hlsl = translate(encode('v_dot2_f32_f16', 0, (V1, V2, V3)))
check('dot2_f32_f16', hlsl == 'v0 = U(f16tof32((v1 & 0xffffu)) * f16tof32((v2 & 0xffffu)) + '
      'f16tof32((v1 >> 16)) * f16tof32((v2 >> 16)) + F(v3));', hlsl)
_, hlsl = translate(encode('v_dot2_f32_f16', 0, (V1, V2, V3), clamp=1))
check('dot2_f32_f16 clamp', 'saturate(' in hlsl, hlsl)

# v_dot4_u32_u8: four byte products plus the accumulator, wrapping.
_, hlsl = translate(encode('v_dot4_u32_u8', 0, (V1, V2, V3)))
check('dot4_u32_u8', normal(hlsl) == normal(
    'v0 = ((v1 >> 0) & 0xffu) * ((v2 >> 0) & 0xffu) + ((v1 >> 8) & 0xffu) * ((v2 >> 8) & 0xffu) + '
    '((v1 >> 16) & 0xffu) * ((v2 >> 16) & 0xffu) + ((v1 >> 24) & 0xffu) * ((v2 >> 24) & 0xffu) + v3;'), hlsl)

# The signed 16-bit forms must also compute the right values (operator precedence of the sign extension).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402


def s16(x):
    return x - 0x10000 if x & 0x8000 else x


SIGNED_REFERENCE = {'v_pk_add_i16': lambda a, b: a + b, 'v_pk_sub_i16': lambda a, b: a - b,
                    }
PACKED = [0x00010001, 0x7fff8000, 0x8000ffff, 0x12345678, 0xfffe0002, 0x80007fff, 0, 0xffffffff]
for name, ref in SIGNED_REFERENCE.items():
    _, hlsl = translate(encode(name, 0, (V1, V2)))
    expr = Expr(hlsl[len('v0 = '):-1])
    bad = None
    for a in PACKED:
        for b in PACKED:
            want = sum(((ref(s16(a >> sh & 0xffff), s16(b >> sh & 0xffff)) & 0xffff) << sh) for sh in (0, 16))
            got = expr.run({'v1': a, 'v2': b})[0]
            if got != want:
                bad = (hex(a), hex(b), hex(got), hex(want))
                break
        if bad:
            break
    check(name + ' values', bad is None, str(bad))

# What is not modelled stays a refusal.
check('int op refuses neg_lo', refused(encode('v_pk_add_u16', 0, (V1, V2), neg_lo=(1, 0, 0))))
check('int op refuses clamp', refused(encode('v_pk_add_u16', 0, (V1, V2), clamp=1)))
check('dot4 refuses clamp', refused(encode('v_dot4_u32_u8', 0, (V1, V2, V3), clamp=1)))
check('dot4 refuses op_sel_hi', refused(encode('v_dot4_u32_u8', 0, (V1, V2, V3), op_sel_hi=(0, 1, 1))))
check('dot2 refuses accumulator neg_lo', refused(encode('v_dot2_f32_f16', 0, (V1, V2, V3), neg_lo=(0, 0, 1))))
check('dot2 refuses accumulator op_sel', refused(encode('v_dot2_f32_f16', 0, (V1, V2, V3), op_sel=(0, 0, 1))))
check('float inline constant refuses', refused(encode('v_pk_add_f16', 0, (V1, 0xF2))))  # 1.0 is not an f16 pair

if failures:
    print('FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('startup_gfx2hlsl_vop3p: ok')
