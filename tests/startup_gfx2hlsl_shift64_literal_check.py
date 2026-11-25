"""32-bit hex literal as the 64-bit source of v_lshlrev_b64, v_lshrrev_b64 and v_ashrrev_i64 (Refs #1723).

A literal fills the low word. The high word is zero for the unsigned shifts and copies bit 31 for the signed
shift (the same rule the scalar 64-bit operations use). Pure Python; run with no arguments.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
from startup_gfx2hlsl_bit_alu_check import Probe, check, refused  # noqa: E402

# (operation, helper, literal, expected low word, expected high word)
CASES = (
    ('v_lshlrev_b64', 'shl64', '0x41', '65u', '0u'),
    ('v_lshrrev_b64', 'shr64u', '0x80000000', '2147483648u', '0u'),
    ('v_lshrrev_b64', 'shr64u', '0xffffffff', '4294967295u', '0u'),
    ('v_ashrrev_i64', 'shr64i', '0x7fffffff', '2147483647u', '0u'),
    ('v_ashrrev_i64', 'shr64i', '0x80000000', '2147483648u', '4294967295u'),
    ('v_ashrrev_i64', 'shr64i', '0xffffffff', '4294967295u', '4294967295u'),
)


def lowered(op, literal):
    probe = Probe()
    probe.one(op, '%s v[0:1], v2, %s' % (op, literal))
    return probe.lines


def main():
    for op, helper, literal, low, high in CASES:
        want = ['{ uint2 t = %s(%s, %s, v2); v0 = t.x; v1 = t.y; }' % (helper, low, high)]
        check('%s with literal %s' % (op, literal), lowered(op, literal) == want)
    check('literals wider than 32 bits stay refused',
          all(refused('%s v[0:1], v2, 0x100000000' % op) for op in ('v_lshlrev_b64', 'v_ashrrev_i64')))
    check('a literal is not accepted as a destination or in the wrong operand count',
          refused('v_lshrrev_b64 0x41, v2, v[4:5]') and refused('v_lshrrev_b64 v[0:1], 0x41'))


if __name__ == '__main__':
    main()
