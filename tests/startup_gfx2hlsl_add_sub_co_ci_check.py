"""v_add_co_ci_u32, v_sub_co_ci_u32 and v_subrev_co_ci_u32 in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_add_sub_co_ci_check.py

Synthetic VOP2 (_e32) and VOP3b (_e64) words only; the spellings are asserted from the translator's decoder.
AMD RDNA 2 ISA: V_ADD_CO_CI_U32 D = S0 + S1 + S2[lane] with carry out to SDST[lane]; V_SUB_CO_CI_U32
D = S0 - S1 - S2[lane] with borrow out to SDST[lane]; V_SUBREV_CO_CI_U32 swaps S0 and S1.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
GRID = [0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, MASK, 0x12345678]
failures = 0


def check(what, ok, detail=''):
    global failures
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', what, '' if ok or not detail else ': %s' % (detail,)))
    if not ok:
        failures += 1


class Pipe:
    root = {'params': [], 'flags': 0}

    def ud(self, source):
        return 0


def vop2(op, vdst, src0, vsrc1):
    return [op << 25 | vdst << 17 | vsrc1 << 9 | src0]


def vop3b(op, vdst, sdst, src0, src1, src2, clamp=False):
    return [0xd4000000 | op << 16 | (1 << 15 if clamp else 0) | sdst << 8 | vdst, src0 | src1 << 9 | src2 << 18]


V2, V3, V4, VCC_LO, EXEC_LO = 256 + 2, 256 + 3, 256 + 4, 106, 126


def stage(exec_known=True, masks=()):
    s = G.Stage(Pipe(), 'ps', struct.pack('<I', 0xbf810000))
    s.lines = []
    s.pc = 0
    s.exec_known = exec_known
    s.masks.update(masks)
    return s


def translate(words, exec_known=True, masks=()):
    s = stage(exec_known, masks)
    _, _, name, text = G.decode(struct.pack('<%dI' % len(words), *words))[0]
    s.lines = []
    s.one(name, text)
    return s, text, s.lines


def evaluate(lines, env):
    env = dict(env)
    for line in lines:
        guard = None
        if line.startswith('if (exec & 1) '):
            guard, line = 'exec', line[len('if (exec & 1) '):]
        target, expr = line.rstrip(';').split(' = ', 1)
        value = Expr(expr).run(env)[0] & MASK
        if guard is None or env.get('exec', 1) & 1:
            env[target] = value
    return env


def ref(kind, a, b, ci):
    if kind == 'subrev':
        a, b = b, a
    if kind == 'add':
        total = a + b + ci
        return total & MASK, total >> 32
    diff = a - b - ci
    return diff & MASK, int(diff < 0)


FORMS = (
    ('add', 0x28, 0x128),
    ('sub', 0x29, 0x129),
    ('subrev', 0x2a, 0x12a),
)


def main():
    count = 0
    for kind, op2, op3 in FORMS:
        # VOP2 (_e32): implicit vcc_lo carry-out and carry-in.
        s, text, lines = translate(vop2(op2, 2, V4, 3))
        expected = 'v_%s_co_ci_u32_e32 v2, vcc_lo, v4, v3, vcc_lo' % kind
        check('decodes as %s' % expected, text == expected, text)
        bad = None
        for a in GRID:
            for b in GRID:
                for ci in (0, 1):
                    env = evaluate(lines, {'v4': a, 'v3': b, 'vcc': ci | 0xdeadbee0, 'v2': 0xdeadbeef})
                    got = env['v2'], env['vcc']
                    if got != ref(kind, a, b, ci) and bad is None:
                        bad = (a, b, ci, got, ref(kind, a, b, ci), lines)
                    count += 1
        check('%s matches the ISA reference on the grid' % text, bad is None, bad)

        # VOP3b (_e64): overlapping destination/carry-out with sources (v4, s18 <- v4, v3, s18).
        s, text, lines = translate(vop3b(op3, 4, 18, V4, V3, 18), masks=(18,))
        expected = 'v_%s_co_ci_u32_e64 v4, s18, v4, v3, s18' % kind
        check('decodes as %s' % expected, text == expected, text)
        check('%s: carry-out SGPR stays a lane mask' % text, 18 in s.masks, s.masks)
        bad = None
        for a in GRID:
            for b in GRID:
                for ci in (0, 1):
                    env = evaluate(lines, {'v4': a, 'v3': b, 's18': ci | 0x12345670})
                    got = env['v4'], env['s18']
                    if got != ref(kind, a, b, ci) and bad is None:
                        bad = (a, b, ci, got, ref(kind, a, b, ci), lines)
                    count += 1
        check('%s overlapping dst/src matches the ISA reference on the grid' % text, bad is None, bad)

    # Constant carry-in (0 and -1) and inactive lane behavior when exec is not known.
    s, text, lines = translate(vop3b(0x128, 2, 18, V4, V3, 128))  # src2 = inline 0
    env = evaluate(lines, {'v4': MASK, 'v3': 1, 'v2': 0xdeadbeef, 's18': 0xdeadbeef})
    check('constant 0 carry-in works', (env['v2'], env['s18']) == (0, 1), (env['v2'], env['s18']))

    s, text, lines = translate(vop3b(0x128, 2, 18, V4, V3, 193))  # src2 = inline -1
    env = evaluate(lines, {'v4': MASK, 'v3': 0, 'v2': 0xdeadbeef, 's18': 0xdeadbeef})
    check('constant -1 carry-in works', (env['v2'], env['s18']) == (0, 1), (env['v2'], env['s18']))

    s, text, lines = translate(vop2(0x28, 2, V4, 3), exec_known=False)
    env = evaluate(lines, {'v4': MASK, 'v3': 1, 'vcc': 1, 'v2': 0x55555555, 'exec': 0})
    check('inactive lane keeps D and gets 0 carry', (env['v2'], env['vcc']) == (0x55555555, 0), (env['v2'], env['vcc']))

    # Refusals: clamp, non-mask SGPR carry-in, vcc_hi carry-in, EXEC carry-out, wrong operand count.
    for label, fn in (
        ('clamp refused', lambda: translate(vop3b(0x128, 2, 18, V4, V3, VCC_LO, clamp=True))),
        ('non-mask SGPR carry-in refused', lambda: translate(vop3b(0x128, 2, 18, V4, V3, 20))),
        ('vcc_hi carry-in refused', lambda: translate(vop3b(0x128, 2, 18, V4, V3, 107))),
        ('EXEC carry-out refused', lambda: translate(vop3b(0x128, 2, EXEC_LO, V4, V3, VCC_LO))),
        ('wrong operand count refused', lambda: stage().one('v_add_co_ci_u32', 'v_add_co_ci_u32 v2, s18, v4, v3')),
    ):
        try:
            fn()
        except G.Unsupported:
            check(label, True)
        else:
            check(label, False, 'accepted')

    # Negative control: omitting carry-in from the reference must disagree on the grid.
    _, _, lines = translate(vop2(0x28, 2, V4, 3))
    env = evaluate(lines, {'v4': MASK, 'v3': 0, 'vcc': 1, 'v2': 0})
    check('negative control: carry-in 1 changes result and carry-out',
          (env['v2'], env['vcc']) == (0, 1) and (env['v2'], env['vcc']) != ref('add', MASK, 0, 0))

    print('checked %d grid points, %d failures' % (count, failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
