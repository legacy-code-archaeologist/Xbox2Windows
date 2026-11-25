"""24-bit multiplies, v_xnor_b32 and the funnel shifts in tools/startup/gfx2hlsl.py: v_mul_i32_i24,
v_mul_hi_i32_i24, v_mad_i32_i24, v_xnor_b32, v_alignbit_b32 and v_alignbyte_b32.

Each instruction is translated by the real translator, the emitted 32-bit HLSL text is evaluated on the CPU with the
integer evaluator of startup_gfx2hlsl_bit_alu_check.py, and the result is compared with a reference written from the
ISA description with Python's unbounded integers (a 48-bit product, a 64-bit concatenation), not with the lowering's
own expressions. Run with no arguments; exits non-zero on the first mismatch.
"""
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr, Probe, refused  # noqa: E402

MASK = 0xffffffff


def s24(x):
    x &= 0xffffff
    return x - (1 << 24) if x >> 23 else x


REFERENCE = {
    'v_mul_i32_i24': (2, lambda a, b, c: (s24(a) * s24(b)) & MASK),
    'v_mad_i32_i24': (3, lambda a, b, c: (s24(a) * s24(b) + c) & MASK),
    'v_mul_hi_i32_i24': (2, lambda a, b, c: ((s24(a) * s24(b)) >> 32) & MASK),
    'v_xnor_b32': (2, lambda a, b, c: ~(a ^ b) & MASK),
    'v_alignbit_b32': (3, lambda a, b, c: (((a << 32) | b) >> (c & 31)) & MASK),
    'v_alignbyte_b32': (3, lambda a, b, c: (((a << 32) | b) >> (8 * (c & 3))) & MASK),
}

FIELD = [0, 1, 2, 0xfff, 0x1000, 0x7fffff, 0x800000, 0x800001, 0xffffff, 0xfffffe, 0x555555, 0xaaaaaa, 0x123456]
WORDS = [0, 1, 0x7fffffff, 0x80000000, MASK, 0x12345678, 0xdeadbeef, 0xff000001, 0x01ffffff, 0xff800000,
         0x00ffffff, 0xfffffffe, 0x89abcdef]
INPUTS = {
    'mul24': [(a | h, b | k, c) for a, b in itertools.product(FIELD, FIELD) for h, k in ((0, 0), (0xab000000, 0x5c000000))
              for c in (0, 1, MASK, 0x80000000, 0x12345678)],
    'bits': [(a, b, c) for a in WORDS for b in WORDS for c in list(range(0, 36)) + [0x80000003, MASK]],
}


def lowered(op, arity):
    sources = ['v%d' % (2 + i) for i in range(arity)]
    probe = Probe()
    probe.one(op, '%s v1, %s' % (op, ', '.join(sources)))
    assert len(probe.lines) == 1 and probe.lines[0].startswith('v1 = '), probe.lines
    return Expr(probe.lines[0][len('v1 = '):-1]), sources


def main():
    failures = 0
    for op, (arity, reference) in REFERENCE.items():
        expr, sources = lowered(op, arity)
        grid = INPUTS['mul24' if '24' in op else 'bits']
        bad = None
        for a, b, c in grid:
            got = expr.run(dict(zip(sources, (a, b, c))))[0]
            if got != reference(a, b, c):
                bad = (a, b, c, got, reference(a, b, c))
                break
        print('%s: %d inputs %s' % (op, len(grid), 'ok' if bad is None else 'MISMATCH %s' % (bad,)))
        failures += bad is not None
    # the _e64 spelling of a VOP2 operation reaches the same lowering
    for op, e64 in (('v_mul_i32_i24', 'v_mul_i32_i24_e64'), ('v_xnor_b32', 'v_xnor_b32_e64')):
        a, b = lowered(op, 2), lowered(e64, 2)
        same = a[0].tokens == b[0].tokens
        print('%s: e64 spelling lowers the same %s' % (op, 'ok' if same else 'MISMATCH'))
        failures += not same
    # v_mul_hi_u32_u24 is refused in both spellings (FXC folds every unsigned lowering to zero; see gfx2hlsl.py)
    for assembly in ('v_mul_hi_u32_u24 v1, v2, v3', 'v_mul_hi_u32_u24_e64 v1, v2, v3'):
        ok = refused(assembly)
        print('%s: refused %s' % (assembly.split(' ', 1)[0], 'ok' if ok else 'MISMATCH'))
        failures += not ok
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
