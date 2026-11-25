"""s_abs_i32 in tools/startup/gfx2hlsl.py (Part of #1723; local branch local/intro-shaders).

usage: startup_gfx2hlsl_s_abs_check.py

RDNA 2 ISA, SOP1 S_ABS_I32 (opcode 0x34): D.i = abs(S0.i), SCC = (D != 0). The most negative word has no positive
counterpart and stays 0x80000000. The words below were assembled with LLVM 15 (llvm-mc -triple=amdgcn--amdpal
-mcpu=gfx1013) and are checked against the SOP1 layout the checked-in decoder reads:
  [31:23] = 0b101111101, [22:16] sdst, [15:8] opcode, [7:0] ssrc0
The emitted HLSL statements are evaluated with Python integers and compared with the ISA reference. A lane mask is
one bit per lane in this translator, not a word, so s_abs_i32 on exec/vcc stays refused. Control: with the table
entry removed, the same words are refused again.
CPU only; no GPU or WARP result is claimed.
"""
import os
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


S_ABS_I32 = 0x34
EXEC_LO, VCC_LO = 126, 106
# LLVM 15 output for `s_abs_i32 s3, s2`, `s_abs_i32 s3, -5` and `s_abs_i32 exec_lo, s2`.
LLVM_WORDS = {'s_abs_i32 s3, s2': '023483be', 's_abs_i32 s3, -5': 'c53483be', 's_abs_i32 exec_lo, s2': '0234febe'}


def encode_sop1(op, sdst, ssrc0):
    return struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (op << 8) | ssrc0)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    def __init__(self, masks=()):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports, self.masks = {}, set(masks)
        self.pipe = Pipeline()


def translate(code, masks=()):
    decoded = G.decode(code)
    stage = Stage(masks)
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], stage.lines


def refused(code, masks=()):
    try:
        translate(code, masks)
    except G.Unsupported:
        return True
    return False


def to_python(expr, env):
    """HLSL uint expression -> Python value for the forms emitted here (int(), ternary, unsigned literals)."""
    expr = re.sub(r'\b(0x[0-9a-f]+|\d+)u\b', r'\1', expr)
    for name, value in env.items():
        expr = re.sub(r'\b%s\b' % name, str(value), expr)
    ternary = re.fullmatch(r'\(\((.*)\) \? \((.*)\) : (.*)\)', expr)
    if ternary:
        expr = '((%s) if (%s) else (%s))' % (ternary.group(2), ternary.group(1), ternary.group(3))
    signed = lambda v: (v & 0xffffffff) - ((v & 0x80000000) << 1)
    return eval(expr, {'int': signed}) & 0xffffffff


def run(lines, s2):
    env = {'s2': s2}
    out = {}
    for line in lines:
        m = re.fullmatch(r'\s*(s\d+|scc) = (.*);', line)
        assert m, line
        value = to_python(m.group(2), dict(env, **{k: v for k, v in out.items() if k != 'scc'}))
        out[m.group(1)] = value
    return out


def reference(value):
    signed = value - (1 << 32) if value & 0x80000000 else value
    d = abs(signed) & 0xffffffff
    return d, int(d != 0)


# 1. The constructed words are LLVM 15's, and the decoder reads them back.
for text, word in LLVM_WORDS.items():
    dst = {'s3': 3, 'exec_lo': EXEC_LO}[text.split()[1].rstrip(',')]
    src = 0xc5 if text.endswith('-5') else 2
    built = encode_sop1(S_ABS_I32, dst, src)
    check('encoding of %s' % text, built.hex() == word, built.hex())
    decoded = G.decode(built)
    check('decodes %s' % text, decoded[0][3] == text, repr(decoded[0][3]))

# 2. D = |S0| as a signed word, SCC = D != 0, over the edge cases.
_, lines = translate(encode_sop1(S_ABS_I32, 3, 2))
for value in (0, 1, 5, 0x7fffffff, 0xffffffff, 0xfffffffb, 0x80000001, 0x80000000, 0x12345678, 0xedcba988):
    got = run(lines, value)
    want_d, want_scc = reference(value)
    check('abs of 0x%08x' % value, (got.get('s3'), got.get('scc')) == (want_d, want_scc),
          '%s -> %s, want %s' % (lines, got, (want_d, want_scc)))

# 3. An inline negative constant folds to the same value.
_, lines = translate(encode_sop1(S_ABS_I32, 3, 0xc5))
got = run(lines, 0)
check('abs of inline -5', (got.get('s3'), got.get('scc')) == (5, 1), '%s -> %s' % (lines, got))

# 4. Refusals that must not widen: a lane mask has no word to take the absolute value of.
check('exec_lo source refuses', refused(encode_sop1(S_ABS_I32, 3, EXEC_LO)))
check('vcc_lo source refuses', refused(encode_sop1(S_ABS_I32, 3, VCC_LO)))
check('mask SGPR source refuses', refused(encode_sop1(S_ABS_I32, 3, 2), masks={2}))
check('exec_lo destination refuses', refused(encode_sop1(S_ABS_I32, EXEC_LO, 2)))

# 5. Control: without the table entry the instruction is refused again, so step 2 measured this lowering.
saved = G.SCA.pop('s_abs_i32')
try:
    check('control: refused without the SCA entry', refused(encode_sop1(S_ABS_I32, 3, 2)))
finally:
    G.SCA['s_abs_i32'] = saved

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
