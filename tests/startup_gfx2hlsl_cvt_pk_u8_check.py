"""v_cvt_pk_u8_f32 of tools/startup/gfx2hlsl.py (item K of #2023).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

The RDNA 1.0 and RDNA 2 ISA guides give (VOP3A opcode 350)
  D = (S2 & ~(0xff << 8 * S1[1:0])) | ((flt32_to_uint8(S0) & 0xff) << 8 * S1[1:0])
and leave flt32_to_uint8 undefined. MEASURED holds rows of the probe run on an RX 6900 XT (gfx1030, Windows, ROCm 7.2,
Adrenalin 26.11.2, MODE at the launch default; tools/rdna_cvt_pk_u8 of the research repository): round to nearest
even, saturate to [0, 255], NaN and negatives give 0. The reference model below must reproduce every measured word,
and the translated text is evaluated on the same inputs. S1 keeps its bits above [1:0] in every draw set, and every S2
holds four distinct bytes (the measured rows use the sentinel 0xa5c3e17b), so a wrong byte or an overwritten S2
changes the word.
"""
import math
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import G, MASK, Probe, check, check_draw, main, refused, words  # noqa: E402

SENTINEL = 0xa5c3e17b


def f32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def as_float(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def to_byte(bits):
    """Measured flt32_to_uint8: round half to even, saturate, NaN and negatives (-0.0, -inf) give 0."""
    x = as_float(bits)
    if math.isnan(x) or x <= 0.0:
        return 0
    return 255 if x >= 255.0 else round(x)   # Python round() is half to even


def cvt_pk_u8(s0, s1, s2):
    shift = 8 * (s1 & 3)
    return (s2 & ~(0xff << shift) & MASK) | (to_byte(s0) << shift)


def dropped_src2(s0, s1, s2):
    return cvt_pk_u8(s0, s1, 0)


# Measured words, S1 = 0 and S2 = SENTINEL, one row per input of the probe run (gfx1030, default MODE):
# (S0 bits, D word), then the label of the row.
MEASURED = [
    (0x00000000, 0xa5c3e100),  # +0
    (0x80000000, 0xa5c3e100),  # -0
    (0x3f800000, 0xa5c3e101),  # 1
    (0x40000000, 0xa5c3e102),  # 2
    (0x42fe0000, 0xa5c3e17f),  # 127
    (0x43000000, 0xa5c3e180),  # 128
    (0x43480000, 0xa5c3e1c8),  # 200
    (0x437e0000, 0xa5c3e1fe),  # 254
    (0x437f0000, 0xa5c3e1ff),  # 255
    (0x3f000000, 0xa5c3e100),  # 0.5
    (0x3effffff, 0xa5c3e100),  # 0.5-ulp
    (0x3f000001, 0xa5c3e101),  # 0.5+ulp
    (0x3fc00000, 0xa5c3e102),  # 1.5
    (0x3fbfffff, 0xa5c3e101),  # 1.5-ulp
    (0x3fc00001, 0xa5c3e102),  # 1.5+ulp
    (0x40200000, 0xa5c3e102),  # 2.5
    (0x401fffff, 0xa5c3e102),  # 2.5-ulp
    (0x40200001, 0xa5c3e103),  # 2.5+ulp
    (0x40600000, 0xa5c3e104),  # 3.5
    (0x405fffff, 0xa5c3e103),  # 3.5-ulp
    (0x40600001, 0xa5c3e104),  # 3.5+ulp
    (0x42ff0000, 0xa5c3e180),  # 127.5
    (0x42feffff, 0xa5c3e17f),  # 127.5-ulp
    (0x42ff0001, 0xa5c3e180),  # 127.5+ulp
    (0x437e8000, 0xa5c3e1fe),  # 254.5
    (0x437e7fff, 0xa5c3e1fe),  # 254.5-ulp
    (0x437e8001, 0xa5c3e1ff),  # 254.5+ulp
    (0x437f8000, 0xa5c3e1ff),  # 255.5
    (0x437f7fff, 0xa5c3e1ff),  # 255.5-ulp
    (0x437f8001, 0xa5c3e1ff),  # 255.5+ulp
    (0x3e800000, 0xa5c3e100),  # 0.25
    (0x3f400000, 0xa5c3e101),  # 0.75
    (0x42c88000, 0xa5c3e164),  # 100.25
    (0x42c98000, 0xa5c3e165),  # 100.75
    (0x437e4000, 0xa5c3e1fe),  # 254.25
    (0x437ec000, 0xa5c3e1ff),  # 254.75
    (0x3efffffd, 0xa5c3e100),  # 0.4999999
    (0x3f000002, 0xa5c3e101),  # 0.5000001
    (0xbdcccccd, 0xa5c3e100),  # -0.1
    (0xbf000000, 0xa5c3e100),  # -0.5
    (0xbf19999a, 0xa5c3e100),  # -0.6
    (0xbf800000, 0xa5c3e100),  # -1
    (0xc37f0000, 0xa5c3e100),  # -255
    (0xbf000001, 0xa5c3e100),  # -0.5-ulp
    (0xbeffffff, 0xa5c3e100),  # -0.5+ulp
    (0x437f6666, 0xa5c3e1ff),  # 255.4
    (0x437fe666, 0xa5c3e1ff),  # 255.9
    (0x43800000, 0xa5c3e1ff),  # 256
    (0x43804000, 0xa5c3e1ff),  # 256.5
    (0x43ff8000, 0xa5c3e1ff),  # 511
    (0x477fff00, 0xa5c3e1ff),  # 65535
    (0x4b800000, 0xa5c3e1ff),  # 1.67772e+07
    (0x4f800000, 0xa5c3e1ff),  # 4.29497e+09
    (0x501502f9, 0xa5c3e1ff),  # 1e+10
    (0x7f800000, 0xa5c3e1ff),  # +inf
    (0xff800000, 0xa5c3e100),  # -inf
    (0x7fc00000, 0xa5c3e100),  # qNaN
    (0xffc00000, 0xa5c3e100),  # -qNaN
    (0x7fa00000, 0xa5c3e100),  # sNaN
    (0xffa00000, 0xa5c3e100),  # -sNaN
    (0x7fc00001, 0xa5c3e100),  # qNaN+payload
    (0x7f800001, 0xa5c3e100),  # sNaN+1
    (0x7fffffff, 0xa5c3e100),  # NaN-all-ones
    (0xffffffff, 0xa5c3e100),  # -NaN-all-ones
    (0x7fc000ff, 0xa5c3e100),  # qNaN-low-payload-ff
    (0x00000001, 0xa5c3e100),  # min-subnormal
    (0x80000001, 0xa5c3e100),  # -min-subnormal
    (0x007fffff, 0xa5c3e100),  # max-subnormal
    (0x00800000, 0xa5c3e100),  # min-normal
    (0x7f7fffff, 0xa5c3e1ff),  # max-float
    (0xff7fffff, 0xa5c3e100),  # -max-float
    (0x422c0000, 0xa5c3e12b),  # (43/255)*255
    (0x42ca0000, 0xa5c3e165),  # (101/255)*255
    (0x432a0000, 0xa5c3e1aa),  # (170/255)*255
    (0x43700000, 0xa5c3e1f0),  # (240/255)*255
    (0x411f6000, 0xa5c3e10a),  # (10/256)*255
    (0x42c73800, 0xa5c3e164),  # (100/256)*255
    (0x42fd0200, 0xa5c3e17f),  # (127/256)*255
    (0x43007f00, 0xa5c3e180),  # (129/256)*255
    (0x43473800, 0xa5c3e1c7),  # (200/256)*255
    (0x437c0300, 0xa5c3e1fc),  # (253/256)*255
    (0x437e0100, 0xa5c3e1fe),  # (255/256)*255
    (0x40233333, 0xa5c3e103),  # (1/100)*255
]
# The same inputs under the other selects: (S0 bits, S1, D word).
MEASURED_SELECT = [
    (0x3fc00000, 0x00000001, 0xa5c3027b),  # 1.5
    (0x3fc00000, 0x00000002, 0xa502e17b),  # 1.5
    (0x3fc00000, 0x00000003, 0x02c3e17b),  # 1.5
    (0x3fc00000, 0xfffffffd, 0xa5c3027b),  # 1.5
    (0x3fc00000, 0xfffffffc, 0xa5c3e102),  # 1.5
    (0x3fc00000, 0x80000002, 0xa502e17b),  # 1.5
    (0x3fc00000, 0x00000103, 0x02c3e17b),  # 1.5
    (0x437e8000, 0x00000001, 0xa5c3fe7b),  # 254.5
    (0x437e8000, 0x00000002, 0xa5fee17b),  # 254.5
    (0x437e8000, 0x00000003, 0xfec3e17b),  # 254.5
    (0x437e8000, 0xfffffffd, 0xa5c3fe7b),  # 254.5
    (0x437e8000, 0xfffffffc, 0xa5c3e1fe),  # 254.5
    (0x437e8000, 0x80000002, 0xa5fee17b),  # 254.5
    (0x437e8000, 0x00000103, 0xfec3e17b),  # 254.5
    (0xbf000000, 0x00000001, 0xa5c3007b),  # -0.5
    (0xbf000000, 0x00000002, 0xa500e17b),  # -0.5
    (0xbf000000, 0x00000003, 0x00c3e17b),  # -0.5
    (0xbf000000, 0xfffffffd, 0xa5c3007b),  # -0.5
    (0xbf000000, 0xfffffffc, 0xa5c3e100),  # -0.5
    (0xbf000000, 0x80000002, 0xa500e17b),  # -0.5
    (0xbf000000, 0x00000103, 0x00c3e17b),  # -0.5
    (0x7fc00000, 0x00000001, 0xa5c3007b),  # qNaN
    (0x7fc00000, 0x00000002, 0xa500e17b),  # qNaN
    (0x7fc00000, 0x00000003, 0x00c3e17b),  # qNaN
    (0x7fc00000, 0xfffffffd, 0xa5c3007b),  # qNaN
    (0x7fc00000, 0xfffffffc, 0xa5c3e100),  # qNaN
    (0x7fc00000, 0x80000002, 0xa500e17b),  # qNaN
    (0x7fc00000, 0x00000103, 0x00c3e17b),  # qNaN
    (0x7f800000, 0x00000001, 0xa5c3ff7b),  # +inf
    (0x7f800000, 0x00000002, 0xa5ffe17b),  # +inf
    (0x7f800000, 0x00000003, 0xffc3e17b),  # +inf
    (0x7f800000, 0xfffffffd, 0xa5c3ff7b),  # +inf
    (0x7f800000, 0xfffffffc, 0xa5c3e1ff),  # +inf
    (0x7f800000, 0x80000002, 0xa5ffe17b),  # +inf
    (0x7f800000, 0x00000103, 0xffc3e17b),  # +inf
]


OPS = {'v_cvt_pk_u8_f32': (3, cvt_pk_u8, False)}
# (S0 bits, S1, S2): the four byte selectors twice, S1 high bits set or clear, distinct bytes in every S2. The S0 words
# cover ties and their neighbours, negatives, above 255, infinities, NaN payloads and a subnormal.
DRAW_SETS = [
    (f32(0.0), 0x00000000, 0x11223344), (f32(1.0), 0xfffffff5, 0xa1b2c3d4), (f32(127.0), 0x12345676, 0x01fe7f80),
    (f32(255.0), 0x80000003, 0xdeadbeef), (f32(255.0), 0x00000000, 0x55aa33cc), (f32(2.0), 0x00000001, 0x0f1e2d3c),
    (f32(200.0), 0xabcdef02, 0x89abcdef), (f32(16.0), 0x7ffffffb, 0xc0ffee11),
    (0x3f000000, 0x00000000, 0x11223344), (0x3f000001, 0x00000001, 0xa1b2c3d4),
    (0x3fc00000, 0x00000002, 0x01fe7f80), (0x40200000, 0x00000003, 0xdeadbeef),
    (0x437e8000, 0x00000000, 0x55aa33cc), (0x437f8000, 0x00000001, 0x0f1e2d3c),
    (0x437f7fff, 0x00000002, 0x89abcdef), (0x437f8001, 0x00000003, 0xc0ffee11),
    (0xbf000000, 0x00000000, 0x11223344), (0x80000000, 0x00000001, 0xa1b2c3d4),
    (0xc3800000, 0x00000002, 0x01fe7f80), (0xff800000, 0x00000003, 0xdeadbeef),
    (0x43800000, 0x00000000, 0x55aa33cc), (0x7f800000, 0x00000001, 0x0f1e2d3c),
    (0x7fc00000, 0x00000002, 0x89abcdef), (0xffc00000, 0x00000003, 0xc0ffee11),
    (0x7f800001, 0x00000000, 0x11223344), (0xffffffff, 0x00000001, 0xa1b2c3d4),
    (0x00000001, 0x00000002, 0x01fe7f80), (0x007fffff, 0x00000003, 0xdeadbeef),
    (0x437e0000, 0x00000000, 0x55aa33cc), (0x437dffff, 0x00000001, 0x0f1e2d3c),
    (0x3ecccccd, 0x00000002, 0x89abcdef), (0x42c9999a, 0x00000003, 0xc0ffee11),
]
FAMILY = {
    'ops': OPS, 'draw': {'v_cvt_pk_u8_f32': DRAW_SETS},
    # A lowering that loses S2: the reference of the mutation keeps only the inserted byte.
    'mutations': {'v_cvt_pk_u8_f32': (r'\((v\d+) & ~\(0xffu << \(\(v\d+ & 3u\) << 3u\)\)\)', '0u', dropped_src2)},
    'body': {'v_cvt_pk_u8_f32': words('0c005ed500092204' '0d005ed5010b2604' '0e005ed5020d2a04' '0f005ed5030f2e04')},
    'accepted': ['v_cvt_pk_u8_f32 v1, 1.0, v3, v4', 'v_cvt_pk_u8_f32 v1, 0, 2, 0xa1b2c3d4',
                 'v_cvt_pk_u8_f32 v1, 0x437f0000, v3, v4', 'v_cvt_pk_u8_f32 v1, v2, v3, v4',
                 'v_cvt_pk_u8_f32_e64 v1, v2, 3, s4', 'v_cvt_pk_u8_f32 v1, s2, v3, v4',
                 'v_cvt_pk_u8_f32 v1, 0.5, v3, v4', 'v_cvt_pk_u8_f32 v1, -1.0, v3, v4',
                 'v_cvt_pk_u8_f32 v1, 0x43800000, v3, v4', 'v_cvt_pk_u8_f32 v1, 0x7fc00000, v3, v4',
                 'v_cvt_pk_u8_f32 v1, 0x3fc00000, v3, v4', 'v_cvt_pk_u8_f32 v1, 0x80000000, v3, v4',
                 # the value is shifted by the offset: (2 & 3) << 1 lands on bit 2, off MODE[1:0]
                 's_setreg_imm32_b32 hwreg(HW_REG_MODE, 1, 2), 2', 's_setreg_imm32_b32 hwreg(HW_REG_MODE, 1, 1), 0'],
    'refused': ['v_cvt_pk_u8_f32 v1, v2, v3', 'v_cvt_pk_u8_f32 v1, v2, v3, v4, v5',
                'v_cvt_pk_u8_f32 v1, v2, v3, v4 clamp', 'v_cvt_pk_u8_f32 v1, v2, v3, v4 mul:2',
                'v_cvt_pk_u8_f32 v1, -v2, v3, v4', 'v_cvt_pk_u8_f32 v1, |v2|, v3, v4',
                'v_cvt_pk_u8_f32 s1, 1.0, v3, v4', 'v_cvt_pk_u8_f32 v1, v[2:3], v3, v4', 'v_cvt_pk_u8_f32 v[1:2], v2, v3, v4',
                # a write of the binary32 rounding mode MODE[1:0] changes the conversion
                's_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 3', 's_setreg_imm32_b32 hwreg(HW_REG_MODE), 3', 's_setreg_imm32_b32 hwreg(HW_REG_MODE), 0xf1',
                's_setreg_imm32_b32 hwreg(HW_REG_MODE, 1, 3), 3', 's_setreg_imm32_b32 hwreg(1, 0, 4), 0xf',
                's_setreg_imm32_b32 hwreg(HW_REG_MODE, 1, 1), 1',   # offset 1 writes bit 1
                's_setreg_imm32_b32 hwreg(HW_REG_MODE), 2',         # field omitted: the whole register, width 32
                's_setreg_imm32_b32 0x1801, 3',                     # a raw SIMM16, not hwreg(): not decoded
                's_round_mode 0xf', 's_setreg_b32 hwreg(HW_REG_MODE, 0, 2), s0'],
}


def run(lines):
    probe = Probe()
    probe.has_branches = False
    for asm in lines:
        probe.one(asm.split(' ', 1)[0], asm)
    return probe


def emitted(asm):
    return run([asm]).lines[-1]


def refuses(lines):
    try:
        run(lines)
    except G.Unsupported as error:
        return str(error)
    return None


def evaluate(line, regs, rounder=round, extra=None):
    """Run the emitted assignment as Python: v-registers hold uints, F() reads binary32, the HLSL ?: becomes if/else."""
    expr = line.split(' = ', 1)[1].rstrip(';')
    expr = re.sub(r'\b0x([0-9a-f]+)u\b', r'0x\1', expr)
    expr = re.sub(r'\b(\d+)u\b', r'\1', expr)
    expr = re.sub(r'\((F\(\w+\) > 0\.0) \? (uint\(round\(min\(F\(\w+\), 255\.0\)\)\)) : 0\)', r'(\2 if \1 else 0)', expr)
    env = {'F': as_float, 'uint': lambda x: int(x), 'min': min, 'round': rounder, 'trunc': math.trunc, **regs}
    env.update(extra or {})
    return eval(expr, {'__builtins__': {}}, env) & MASK  # the expression is the translator's own text


def half_away(x):
    return math.floor(x + 0.5)


def measured_controls():
    # The reference model reproduces every measured word; the table is not read from the research repository.
    for bits, word in MEASURED:
        check('model reads measured %08x' % bits, cvt_pk_u8(bits, 0, SENTINEL) == word)
    for bits, select, word in MEASURED_SELECT:
        check('model reads measured %08x select %d' % (bits, select), cvt_pk_u8(bits, select, SENTINEL) == word)
    line = emitted('v_cvt_pk_u8_f32 v12, v0, v4, v8')
    for bits, word in MEASURED:
        check('translated text reads measured %08x' % bits,
              evaluate(line, {'v0': bits, 'v4': 0, 'v8': SENTINEL}) == word)
    for bits, select, word in MEASURED_SELECT:
        check('translated text reads measured %08x select %d' % (bits, select),
              evaluate(line, {'v0': bits, 'v4': select, 'v8': SENTINEL}) == word)
    # Negative controls: each wrong lowering must disagree with a measured row (NaN raising counts as a difference).
    def differs(text, rounder=round, extra=None, only=lambda bits: True):
        for bits, word in MEASURED:
            if only(bits):
                try:
                    if evaluate(text, {'v0': bits, 'v4': 0, 'v8': SENTINEL}, rounder, extra) != word:
                        return True
                except ValueError:
                    return True
        return False
    nan = lambda bits: math.isnan(as_float(bits))
    negative = lambda bits: as_float(bits) < 0.0
    check('truncation is detected', differs(line, extra={'round': math.trunc}) and 'round(' in line)
    check('round half away is detected', differs(line, half_away))
    no_guard = re.sub(r'\(F\((v\d+)\) > 0\.0 \? (.*?) : 0u\)', r'\2', line)
    check('the guard is in the text', no_guard != line)
    check('NaN not mapped to 0 is detected', differs(no_guard, only=nan))
    check('negatives wrapped is detected', differs(no_guard, only=negative))
    check('guard keeps NaN at 0', evaluate(line, {'v0': 0x7fc00000, 'v4': 0, 'v8': 0}) == 0)


def selector_controls():
    # All four byte selectors, high S1 bits, and an S2 of four distinct bytes, through the translated text.
    line = emitted('v_cvt_pk_u8_f32 v12, v0, v4, v8')
    shifted = line.replace('<< 3u', '<< 2u')
    for s0, s1, s2 in DRAW_SETS:
        regs = {'v0': s0, 'v4': s1, 'v8': s2}
        check('translated text reads %08x %08x %08x' % (s0, s1, s2), evaluate(line, regs) == cvt_pk_u8(s0, s1, s2))
    # Negative controls on the text: a wrong byte position and a lost S2 must disagree with the reference.
    check('wrong byte position is detected',
          any(evaluate(shifted, {'v0': a, 'v4': b, 'v8': c}) != cvt_pk_u8(a, b, c) for a, b, c in DRAW_SETS))
    check('overwritten S2 is detected',
          any(dropped_src2(a, b, c) != cvt_pk_u8(a, b, c) for a, b, c in DRAW_SETS))
    check('all four selectors drawn', {s1 & 3 for _, s1, _ in DRAW_SETS} == {0, 1, 2, 3})
    # The reference itself, worked by hand: byte 2 of 0x11223344 replaced by 0xff, and a tie to the even byte.
    check('hand-worked word', cvt_pk_u8(f32(255.0), 2, 0x11223344) == 0x11ff3344)
    check('hand-worked ties', [to_byte(f32(x)) for x in (0.5, 1.5, 2.5, 254.5, 255.5)] == [0, 2, 2, 254, 255])
    shifted_high = evaluate(line, {'v0': f32(9.0), 'v4': 0x100, 'v8': 0x01020304})
    check('S1 bits above [1:0] are ignored', shifted_high == evaluate(line, {'v0': f32(9.0), 'v4': 0, 'v8': 0x01020304}))


def run_controls():
    selector_controls()
    measured_controls()
    check('constant is lowered', 'F(0x3f800000u)' in emitted('v_cvt_pk_u8_f32 v1, 1.0, v3, v4'))
    check('S1 and S2 constants are lowered', not refused('v_cvt_pk_u8_f32 v1, 1.0, 7, 0xffffffff'))
    check('a register S0 needs no proof', refuses(['v_mul_f32 v2, v5, v6', 'v_cvt_pk_u8_f32 v1, v2, v3, v4']) is None)
    # MODE writes that leave bits 1:0 alone stay accepted (the scratch base setup, the f16/f64 rounding field).
    for asm in ('s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0', 's_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3',
                's_setreg_imm32_b32 hwreg(HW_REG_MODE), 0xf0', 's_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 0'):
        check('accepted: ' + asm, refuses([asm]) is None)


if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == 'draw':
        _, _, dxc, renderer, adapter, fxc = sys.argv
        check_draw(FAMILY, dxc, renderer, adapter, fxc)
    else:
        run_controls()
        main(FAMILY, __doc__)
