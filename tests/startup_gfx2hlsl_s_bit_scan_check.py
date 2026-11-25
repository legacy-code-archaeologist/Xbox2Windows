"""Scalar bit counts and scans in tools/startup/gfx2hlsl.py (Frontier next batch 3 #1, #1723).

usage: startup_gfx2hlsl_s_bit_scan_check.py

s_bcnt0/1_i32_b32/b64 count zero/one bits (SCC = result != 0). s_ff0/ff1_i32_b64 return the index of the lowest
zero/one bit of a 64-bit source, s_flbit_i32_b64 the number of leading zeros, s_flbit_i32_i64 the number of leading
bits equal to the sign bit; the scans return -1 when nothing qualifies and leave SCC alone. The translator refused
all of them with `Unsupported`. The instruction words are built from the SOP1 layout the decoder reads, the emitted
HLSL runs on the integer evaluator of startup_gfx2hlsl_bit_alu_check.py, and every result is compared with a plain
integer reference written from the ISA description (string and loop based, not the lowering's expressions).
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
failures = []
OPCODES = {'s_bcnt0_i32_b32': 0x0d, 's_bcnt0_i32_b64': 0x0e, 's_bcnt1_i32_b32': 0x0f, 's_bcnt1_i32_b64': 0x10,
           's_ff0_i32_b64': 0x12, 's_ff1_i32_b64': 0x14, 's_flbit_i32_b64': 0x16, 's_flbit_i32_i64': 0x18}
EXEC_LO, M0 = 126, 124


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def encode(name, sdst, ssrc0):
    return struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (OPCODES[name] << 8) | ssrc0)


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


def translate(name, sdst, ssrc0, masks=()):
    stage = Stage(masks)
    decoded = G.decode(encode(name, sdst, ssrc0))
    for _offset, _size, op, asm in decoded:
        stage.one(op, asm)
    return [asm for _o, _s, _n, asm in decoded], stage.lines


def refused(name, sdst, ssrc0, masks=()):
    try:
        translate(name, sdst, ssrc0, masks)
    except G.Unsupported:
        return True
    return False


def run(lines, regs):
    """Evaluate the emitted `sN = expr;` and `scc = expr;` lines over the register dict."""
    env = dict(regs)
    for line in lines:
        target, expr = line.rstrip(';').split(' = ', 1)
        env[target] = Expr(expr).run(env)[0]
    return env


def bits(value, width):
    return [(value >> i) & 1 for i in range(width)]


def ref(name, value):
    """(result, scc or None) from the ISA text, with the source as a 32- or 64-bit integer."""
    width = 32 if name.endswith('b32') else 64
    b = bits(value, width)
    if 'bcnt0' in name:
        return b.count(0), True
    if 'bcnt1' in name:
        return b.count(1), True
    if 'ff0' in name or 'ff1' in name:
        want = 0 if 'ff0' in name else 1
        return next((i for i, x in enumerate(b) if x == want), MASK), False
    sign = b[63] if name.endswith('i64') else 0
    n = 0
    for x in reversed(b):
        if x != sign:
            return n, False
        n += 1
    return MASK, False


VALUES = [0, 1, 2, 0x80, 0xff, 0x7fffffff, 0x80000000, 0xffffffff, 0x12345678, 0xfffffffe, 0xdeadbeef]
WIDE = [0, 1, 1 << 31, 1 << 32, 1 << 33, 1 << 62, 1 << 63, (1 << 64) - 1, (1 << 63) - 1, (1 << 64) - 2,
        0x00000000ffffffff, 0xffffffff00000000, 0x0123456789abcdef, 0xfedcba9876543210, 0x8000000000000001,
        0xffffffff7fffffff] + [1 << k for k in range(64)] + [((1 << 64) - 1) ^ (1 << k) for k in range(64)]

for name in OPCODES:
    wide = not name.endswith('b32')
    asm, lines = translate(name, 3, 4 if wide else 2)
    src = 's[4:5]' if wide else 's2'
    check('%s decodes' % name, asm == ['%s s3, %s' % (name, src)], repr(asm))
    bad = []
    for value in (WIDE if wide else VALUES):
        regs = {'scc': 0, 's4': value & MASK, 's5': value >> 32, 's2': value & MASK}
        env = run(lines, regs)
        want, scc = ref(name, value)
        want_scc = int(want != 0) if scc else 0
        if env['s3'] != want or env['scc'] != want_scc:
            bad.append((hex(value), env['s3'], env['scc'], want, want_scc))
            if len(bad) == 3:
                break
    check('%s values' % name, not bad, repr(bad))
    scc_emitted = any(line.startswith('scc') for line in lines)
    check('%s SCC %s' % (name, 'follows result' if 'bcnt' in name else 'untouched'), scc_emitted == ('bcnt' in name),
          repr(lines))

# Refusals that must not widen: a 64-bit scan needs a register pair, 32-bit one register, and no lane-mask data.
check('b64 refuses an inline constant source', refused('s_ff1_i32_b64', 3, 0x80))
check('b32 refuses exec destination', refused('s_bcnt1_i32_b32', EXEC_LO, 2))
check('b32 refuses m0 destination', refused('s_bcnt1_i32_b32', M0, 2))
check('b32 refuses exec source', refused('s_bcnt1_i32_b32', 3, EXEC_LO))
check('b64 refuses a lane-mask source', refused('s_bcnt1_i32_b64', 3, 4, masks=(5,)))
check('b32 refuses a lane-mask source', refused('s_bcnt0_i32_b32', 3, 2, masks=(2,)))

if failures:
    print('FAIL: %s' % ', '.join(failures))
    sys.exit(1)
print('PASS')
