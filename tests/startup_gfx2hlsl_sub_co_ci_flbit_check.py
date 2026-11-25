"""v_sub_co_ci_u32, v_subrev_co_ci_u32 and s_flbit_i32_b32 in tools/startup/gfx2hlsl.py (#1723).

Each form is translated and the emitted lines are run on the CPU by the integer evaluator of
startup_gfx2hlsl_bit_alu_check.py over edge values. The reference is plain integer arithmetic from the ISA meaning:
  v_sub_co_ci_u32:    D = S0 - S1 - carry_in, carry out = (S0 < S1 + carry_in)
  v_subrev_co_ci_u32: D = S1 - S0 - carry_in, carry out = (S1 < S0 + carry_in)
  s_flbit_i32_b32:    D = leading zero count of S0, 0xffffffff when S0 = 0, SCC unchanged
Refused: modifiers, a wrong operand count, a non-VGPR destination, an unknown carry-in mask, and lane-mask
operands for s_flbit_i32_b32.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
VALUES = [0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, MASK, 0x12345678, 0xdeadbeef]
SCC_UNSET = 2


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for instructions to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def translate(name, asm, masks=()):
    stage = Probe()
    stage.masks = set(masks)
    stage.one(name, '%s %s' % (name, asm))
    return stage


def run(lines, env):
    """Run emitted lines: plain `x = e;` and the `{ uint a = e, b = e; x = e; y = e; }` block."""
    env = dict(env)
    for line in lines:
        line = line.strip()
        if line.startswith('{'):
            body = line[1:-1].strip()
            head, _, rest = body.partition(';')
            for part in head.replace('uint ', '').split(', '):
                name, _, expr = part.partition(' = ')
                env[name.strip()] = Expr(expr).run(env)[0]
            statements = [s for s in rest.split(';') if s.strip()]
        else:
            statements = [line.rstrip(';')]
        for statement in statements:
            m = re.fullmatch(r'\s*(\w+) = (.*)', statement)
            assert m, statement
            env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sub_reference(a, b, carry_in):
    return (a - b - carry_in) & MASK, int(a < b + carry_in)


def check_sub(name, swap):
    for out_mask in ('vcc', 's[4:5]'):
        asm = 'v1, %s, v2, v3, vcc' % out_mask
        stage = translate(name + '_e32', asm)
        out_name = 'vcc' if out_mask == 'vcc' else 's4'
        bad = None
        for a in VALUES:
            for b in VALUES:
                for carry_in in (0, 1, 3):  # bit 0 is the carry; the other bits are ignored
                    env = run(stage.lines, {'v2': a, 'v3': b, 'vcc': carry_in})
                    want = sub_reference(*((b, a) if swap else (a, b)), carry_in & 1)
                    got = (env['v1'], env[out_name])
                    if got != want and bad is None:
                        bad = (hex(a), hex(b), carry_in, got, want)
        check('%s -> %s' % (name, out_mask), bad is None, str(bad))
    # carry in from a known SGPR mask
    stage = translate(name + '_e64', 'v1, vcc, v2, v3, s6', masks=(6,))
    env = run(stage.lines, {'v2': 5, 'v3': 5, 's6': 1, 'vcc': 0})
    check('%s carry in s6' % name, (env['v1'], env['vcc']) == (MASK, 1))


def refuses(label, name, asm, masks=()):
    try:
        translate(name, asm, masks)
    except G.Unsupported:
        check(label, True)
        return
    check(label, False, 'translated without a refusal')


def check_flbit():
    def reference(x):
        return MASK if x == 0 else 32 - x.bit_length()
    for src, regs in (('s1', None), ('0', None)):
        for x in VALUES + [1 << k for k in range(32)]:
            stage = translate('s_flbit_i32_b32', 's3, s1' if regs is None and src == 's1' else 's3, 0')
            env = run(stage.lines, {'s1': x, 'scc': SCC_UNSET})
            want = reference(x if src == 's1' else 0)
            if (env['s3'], env['scc']) != (want, SCC_UNSET):
                check('s_flbit_i32_b32 %s' % hex(x), False, str((env['s3'], env['scc'], want)))
                return
    check('s_flbit_i32_b32 values and SCC kept', True)
    refuses('s_flbit_i32_b32 vcc source', 's_flbit_i32_b32', 's3, vcc_lo')
    refuses('s_flbit_i32_b32 exec destination', 's_flbit_i32_b32', 'exec_lo, s1')
    refuses('s_flbit_i32_b32 mask source', 's_flbit_i32_b32', 's3, s6', masks=(6,))


def main():
    check_sub('v_sub_co_ci_u32', False)
    check_sub('v_subrev_co_ci_u32', True)
    refuses('sub_co_ci modifier', 'v_sub_co_ci_u32_e64', 'v1, vcc, v2, v3, vcc clamp')
    refuses('sub_co_ci operand count', 'v_sub_co_ci_u32_e32', 'v1, v2, v3')
    refuses('sub_co_ci SGPR destination', 'v_sub_co_ci_u32_e32', 's1, vcc, v2, v3, vcc')
    refuses('sub_co_ci unknown carry-in', 'v_subrev_co_ci_u32_e64', 'v1, vcc, v2, v3, s6')
    check_flbit()
    if failures:
        print('FAIL: %s' % ', '.join(failures))
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
