"""The `src_scc` scalar source in tools/startup/gfx2hlsl.py (issue #1820).

usage: startup_gfx2hlsl_src_scc_check.py

The census contained `s_mul_i32 s28, 1.0, src_scc`, refused as `operand 'src_scc'` (#870, issue #1820).
Source code 253 reads the SCC bit as the integer value 0 or 1; the generated shader already carries SCC as a
`bool scc`, so the source lowers to `(scc ? 1u : 0u)` instead of being refused.

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, SOP2 branch): `[31:30] == 0b10`, opcode in `w0[29:23]`, destination in `w0[22:16]`,
src1 in `w0[15:8]` and src0 in `w0[7:0]`.
"""
import os
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


SOP2_MUL_I32 = 0x26
SOP2_AND_B32 = 0x0E
S_SCC = 253         # the src_scc scalar source code
S_1_0 = 242         # the inline float constant 1.0 (its raw bits are 0x3f800000)
S5 = 5
S28 = 28


def encode_sop2(op, sdst, ssrc0, ssrc1):
    w0 = ((0b10 << 30) | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16)
          | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF))
    return struct.pack('<I', w0)


class Pipeline:
    """Just enough pipeline state for one scalar operate."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.masks = set()
        self.pipe = Pipeline()


def translate(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines), stage


# 1. The encoder produces the form the decoder reads; a layout change shows up here first.
asm, emitted, _stage = translate(encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC))
check('encodes and decodes the census form', asm == ['s_mul_i32 s28, 1.0, src_scc'], repr(asm))

# 2. The census form lowers: the raw bits of the inline 1.0 times the SCC bit as 0 or 1.
check('the census form lowers to the SCC bit', emitted == 's28 = 0x3f800000u * (scc ? 1u : 0u);', emitted)

# 3. `src_scc` is a source in either position, and a bitwise op still updates SCC afterwards.
asm, emitted, _stage = translate(encode_sop2(SOP2_AND_B32, S28, S_SCC, S5))
check('encodes and decodes src_scc in the first position', asm == ['s_and_b32 s28, src_scc, s5'], repr(asm))
check('src_scc in the first position lowers',
      emitted == 's28 = (scc ? 1u : 0u) & s5; scc = s28 != 0;', emitted)

# 4. A definition followed by a use: the first instruction writes SCC, the second reads it back as 0/1. Both
#    statements have to be in the same straight-line block, or the read would see a stale bit.
asm, emitted, _stage = translate(encode_sop2(SOP2_AND_B32, S28, S_SCC, S5)
                                 + encode_sop2(SOP2_MUL_I32, 29, S_1_0, S_SCC))
check('a defined SCC is read back by the next instruction',
      'scc = s28 != 0;' in emitted and 's29 = 0x3f800000u * (scc ? 1u : 0u);' in emitted, emitted)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')