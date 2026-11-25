"""Scalar s_bitset0_b32 and s_ff0_i32_b32 in tools/startup/gfx2hlsl.py (#1723).

Each instruction word is built from the RDNA 2 SOP1/SOP2 layouts that tools/disasm_rdna2.py reads, decoded,
translated, and the emitted lines are run in order on the CPU by the evaluator of
startup_gfx2hlsl_bit_alu_check.py over edge values. The references use plain integer arithmetic from the ISA text:

  s_bitset0_b32      D = D with bit S0[4:0] cleared, SCC unchanged
  s_ff0_i32_b32      D = index of the lowest zero bit of S0, -1 when none, SCC unchanged

SCC starts at a value that is neither 0 nor 1, so a form that writes it when it must not (or the reverse) fails.
Lane-mask and descriptor operands are refused.
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402
from startup_gfx2hlsl_scalar_m0_check import ProbeStage  # noqa: E402

MASK = 0xffffffff
SCC_START = 7
M0 = 124
SOP1 = {name: op for op, (name, _) in D.SOP1_OPS.items()}
SOP2 = {name: op for op, (name, _) in D.SOP2_OPS.items()}
VALUES = [0, 1, 2, 3, 0x7fffffff, 0x80000000, 0xffffffff, 0x55555555, 0xaaaaaaaa, 0x1fffffff, 0x20000000,
          0x3fffffff, 0x40000000, 0xdeadbeef, 0xfffffff0, 0x0000ffff, 0xfffeffff, 0x12345678]


def sop1(name, dst, src):
    return struct.pack('<I', 0xbe800000 | (dst << 16) | (SOP1[name] << 8) | src)


def sop2(name, dst, a, b):
    return struct.pack('<I', 0x80000000 | (SOP2[name] << 23) | (dst << 16) | (b << 8) | a)


def translate(word):
    stage = ProbeStage()
    _, name, asm, _ = D.decode_instruction(word, 0)
    stage.one(name, asm)
    return stage.lines


def run(lines, env):
    env = dict(env, scc=SCC_START)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


def refused(word, text):
    try:
        translate(word)
    except G.Unsupported as exc:
        assert text in str(exc), (text, str(exc))
        return
    raise AssertionError('expected a refusal containing %r' % text)


def check_bitset0():
    for dst in (4, 6):
        lines = translate(sop1('s_bitset0_b32', dst, 2))
        for old in VALUES:
            for bit in range(0, 64, 3):
                env = run(lines, {'s2': bit, 's%d' % dst: old})
                assert env['s%d' % dst] == old & ~(1 << (bit & 31)) & MASK, (hex(old), bit, lines)
                assert env['scc'] == SCC_START, lines


def check_ff0():
    lines = translate(sop1('s_ff0_i32_b32', 4, 2))
    for a in VALUES:
        env = run(lines, {'s2': a})
        zeros = [i for i in range(32) if not a >> i & 1]
        assert env['s4'] == (zeros[0] if zeros else MASK), (hex(a), lines)
        assert env['scc'] == SCC_START, lines


def check_m0_and_refusals():
    lines = translate(sop1('s_bitset0_b32', M0, 2))
    assert any(l.startswith('m0 = ') for l in lines) and not any(l.startswith('scc') for l in lines), lines
    refused(sop1('s_bitset0_b32', 4, 126), 'lane mask')   # exec_lo
    refused(sop1('s_ff0_i32_b32', 4, 126), 'lane mask')   # exec_lo
    refused(sop1('s_ff0_i32_b32', 4, 106), 'lane mask')   # vcc_lo


def check_numeric_symbols():
    for tag in (('ud', 0), ('fixed_buffer_word',)):
        stage = ProbeStage()
        stage.sym[2] = tag
        _, name, asm, _ = D.decode_instruction(sop1('s_bitset0_b32', 4, 2), 0)
        stage.one(name, asm)
        assert stage.lines, tag


def main():
    check_bitset0()
    check_ff0()
    check_m0_and_refusals()
    check_numeric_symbols()
    print('startup_gfx2hlsl_scalar_twins_check: ok')


if __name__ == '__main__':
    main()
