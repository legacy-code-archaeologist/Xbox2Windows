"""Translation control for v_bfm_b32 in gfx2hlsl: a mask of S0[4:0] ones shifted left by S1[4:0].

  (no arguments)                       translation control, pure Python
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402

MASK = 0xffffffff
EXPECTED = 'v1 = (((1u << (v2 & 31)) - 1) << (v3 & 31));'


def translate(asm):
    stage = gfx2hlsl.Stage.__new__(gfx2hlsl.Stage)
    stage.lines = []
    stage.used_v = set()
    stage.uniform_v = set()
    stage.exec_known = True
    stage.one(asm.split()[0], asm)
    return stage.lines


def reference(width, offset):
    return (((1 << (width & 31)) - 1) << (offset & 31)) & MASK


def emitted(line, width, offset):
    """Run the emitted expression over 32-bit words, accepting only digits, shifts and masks."""
    expr = re.fullmatch(r'v1 = (.*);', line).group(1)
    assert re.fullmatch(r'[()\s&<>\-0-9u]*', expr.replace('v2', '').replace('v3', '')), expr
    expr = re.sub(r'(\d+)u', r'\1', expr).replace('v2', str(width)).replace('v3', str(offset))
    return eval(expr, {'__builtins__': {}}) & MASK


def main():
    for suffix in ('', '_e64'):
        lines = translate('v_bfm_b32%s v1, v2, v3' % suffix)
        assert lines == [EXPECTED], lines
        for width, offset in ((0, 0), (1, 0), (8, 4), (31, 1), (32, 5), (33, 31), (MASK, MASK)):
            assert emitted(lines[0], width, offset) == reference(width, offset), (width, offset)
    lines = translate('v_bfm_b32 v1, 8, 4')
    assert lines == ['v1 = (((1u << (0x00000008u & 31)) - 1) << (0x00000004u & 31));'], lines
    print('PASS: startup_gfx2hlsl_v_bfm_check')


if __name__ == '__main__':
    main()
