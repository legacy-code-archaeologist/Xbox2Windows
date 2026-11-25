"""`v_fract_f32` capped at 0x3f7fffff in a drawn pixel, with FXC and DXC (#1741).

  (no arguments)                       translation and arithmetic controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate the fixture, compile with FXC vs_5_1/ps_5_1 (the --pending
                                       path) when given and with DXC vs_6_0/ps_6_0, draw one pixel and compare
                                       the four words

The fixture is a source-built gfx1013 vertex and pixel pair. The vertex stage places a full-screen triangle
from the vertex index (v0) and exports four words to param0; the pixel stage reads them flat with
v_interp_mov_f32 p0, applies v_fract_f32 to each and exports them to mrt0. The words are 0xb0800000
(-2^-30, the input the two NVIDIA reports fail on), 0x3fc00000 (1.5), 0x00000000 and 0xc0200000 (-2.5),
so the capped reading of the four lanes is 0x3f7fffff, 0x3f000000, 0x00000000 and 0x3f000000. Nothing here
reads game data: the words come from the vertex stage, so no compiler can fold them.

The control that counts: the same pair with the lowering left uncapped, `U(frac(F(v)))`, must read a
different word in the first lane. For -2^-30 the unclamped x - floor(x) rounds up to exactly 1.0 in
binary32, `0x3f800000`, while the instruction returns at most `0x3f7fffff`. An adapter whose bare `frc`
already caps reads the capped word there: the test records that as a finding on that adapter instead of
failing, because on it the two forms cannot be separated by this draw. The other three lanes must read the
same word with and without the cap, which is what says the cap adds no rounding of its own.
"""
import math
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import mov_literal  # noqa: E402
from startup_flat_param_check import VS_WORDS, compile_hlsl, draw, root_blob  # noqa: E402

CAP = 0x3f7fffff
ONE = 0x3f800000
INPUTS = (0xb0800000, 0x3fc00000, 0x00000000, 0xc0200000)
VS_EXPECTED = ('v_mov_b32_e32 v3, 0x40400000', 'v_mov_b32_e32 v10, 0xc0400000',
               'v_cmp_eq_u32_e32 vcc_lo, 1, v0', 'v_cndmask_b32_e32 v1, -1.0, v3, vcc_lo',
               'v_cmp_eq_u32_e32 vcc_lo, 2, v0', 'v_cndmask_b32_e32 v2, 1.0, v10, vcc_lo',
               'v_mov_b32_e32 v4, 1.0', 'v_mov_b32_e32 v5, 0',
               'v_mov_b32_e32 v6, 0xb0800000', 'v_mov_b32_e32 v7, 0x3fc00000',
               'v_mov_b32_e32 v8, 0', 'v_mov_b32_e32 v9, 0xc0200000',
               'exp pos0 v1, v2, v5, v4 done', 'exp param0 v6, v7, v8, v9', 's_endpgm')
PS_EXPECTED = ('s_mov_b32 m0, s2',
               'v_interp_mov_f32_e32 v0, p0, attr0.x', 'v_interp_mov_f32_e32 v1, p0, attr0.y',
               'v_interp_mov_f32_e32 v2, p0, attr0.z', 'v_interp_mov_f32_e32 v3, p0, attr0.w',
               'v_fract_f32_e32 v0, v0', 'v_fract_f32_e32 v1, v1',
               'v_fract_f32_e32 v2, v2', 'v_fract_f32_e32 v3, v3',
               'exp mrt0 v0, v1, v2, v3 done vm', 's_endpgm')
failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


# The recorded words are built here rather than assembled: llvm-mc is not needed, and the controls read the
# decoded text back through gfx2hlsl.decode, which is the decoder the translator itself uses.
VS = (VS_WORDS[:40]                                                     # full-screen triangle, v3..v5
      + b''.join(mov_literal(6 + lane, word) for lane, word in enumerate(INPUTS))
      + bytes.fromhex('cf0800f801020504')                               # exp pos0 v1, v2, v5, v4 done
      + struct.pack('<2I', 0xf800000f | 32 << 4, 0x09080706)             # exp param0 v6, v7, v8, v9
      + bytes.fromhex('000081bf'))                                      # s_endpgm


def vop1_e32(op, vdst, src0):
    """VOP1 in the e32 encoding: src0 0..127 is an SGPR, 256+n is a VGPR (VGPR 0 is 0x100, not 0)."""
    return struct.pack('<I', 0x7e000000 | vdst << 17 | op << 9 | src0)


# VOP1 opcode 0x20 is V_FRACT_F32 (tools/disasm_rdna2.py, VOP1_OPS).
PS = (bytes.fromhex('0203fcbe'                                          # s_mov_b32 m0, s2
                    '020002c8' '020106c8' '02020ac8' '02030ec8')         # v_interp_mov_f32 v0..v3, p0, attr0.x..w
      + b''.join(vop1_e32(0x20, lane, 0x100 + lane) for lane in range(4))  # v_fract_f32 v0..v3, v0..v3
      + bytes.fromhex('0f1800f800010203' '000081bf'))                    # exp mrt0 v0..v3 done vm; s_endpgm
WORDS = re.compile(r'(?:actual=|words=)([0-9a-f]{8},[0-9a-f]{8},[0-9a-f]{8},[0-9a-f]{8})')


def frac_uncapped():
    """x - floor(x) for each input word, rounded to binary32: what a D3D frac without the cap returns."""
    out = []
    for word in INPUTS:
        value = struct.unpack('<f', struct.pack('<I', word))[0]
        out.append(struct.unpack('<I', struct.pack('<f', value - math.floor(value)))[0])
    return tuple(out)


def capped(words):
    """The helper's rule: a word that is exactly 1.0 reads the cap, every other word passes."""
    return tuple(CAP if word == ONE else word for word in words)


UNCAPPED = frac_uncapped()
EXPECTED = capped(UNCAPPED)
EXPECTED_TEXT = ','.join('%08x' % word for word in EXPECTED)
UNCAPPED_TEXT = ','.join('%08x' % word for word in UNCAPPED)


def translate():
    return G.Pipeline(VS, PS, root_blob())


def uncapped_ps(text):
    """The pixel shader with the lowering uncapped. The helper keeps its name: only the call sites change."""
    mutated = text.replace('U(fract32(', 'U(frac(')
    assert mutated.count('U(frac(') == 4 and 'U(fract32(' not in mutated, mutated
    assert 'float fract32(float value) {' in mutated, mutated
    return mutated


def controls():
    for name, words, expected in (('vertex', VS, VS_EXPECTED), ('pixel', PS, PS_EXPECTED)):
        decoded = tuple(asm for _off, _n, _name, asm in G.decode(words))
        check('the %s fixture decodes to the program the docstring describes' % name, decoded == expected,
              repr([line for line in decoded if line not in expected]))
    check('the fixture covers the reported input and ordinary fractions, not only the cap',
          len(set(INPUTS)) == 4 and INPUTS[0] == 0xb0800000 and 0x3fc00000 in INPUTS, repr(INPUTS))

    pipe = translate()
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    check('the vertex stage exports the words as they are, so nothing folds them',
          '    o.param0 = float4(F(v6), F(v7), F(v8), F(v9));\n    o.fparam0 = uint4(v6, v7, v8, v9);\n' in vs, vs)
    for lane in range(4):
        check('the pixel stage applies the helper to lane %d' % lane,
              '    v%d = U(fract32(F(v%d)));' % (lane, lane) in ps, ps)
    check('the uncapped control differs only at the call sites', 'U(frac(F(v0)))' in uncapped_ps(ps), ps)

    # The arithmetic the draw is meant to observe, computed independently of the lowering.
    check('-2^-30 rounds x - floor(x) up to exactly 1.0 without the cap: %s' % UNCAPPED_TEXT,
          UNCAPPED[0] == ONE, '0x%08x' % UNCAPPED[0])
    check('the cap turns that one word into 0x3f7fffff and leaves the other three alone',
          EXPECTED[0] == CAP and EXPECTED[1:] == UNCAPPED[1:], EXPECTED_TEXT)


def check_draw(dxc, renderer, adapter, fxc=None):
    pipe = translate()
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    mutant = uncapped_ps(ps)
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    if not fxc:
        print('  no FXC given: the pair is drawn with DXC only, and FXC, the compiler the cap has to survive, '
              'is not covered by this run')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            def build(name, source, profile):
                return compile_hlsl(compiler, kind, source, '%s_%s' % (name, kind), profile, directory)

            vs_cso = build('vs', vs, vs_profile)
            ps_cso = build('ps', ps, ps_profile)

            # The translated pair: the capped words.
            run = draw(renderer, adapter, vs_cso, ps_cso, EXPECTED_TEXT)
            if run.returncode:
                raise AssertionError('%s capped pair: %s\n%s' % (kind, run.stdout, run.stderr))
            check('%s %s: the capped pair reads %s' % (kind, adapter, EXPECTED_TEXT), True)

            # The negative control: the same pair, uncapped. The first lane must differ.
            run = draw(renderer, adapter, vs_cso, build('ps_uncapped', mutant, ps_profile), EXPECTED_TEXT)
            actual = WORDS.search(run.stderr + run.stdout)
            read = tuple(int(word, 16) for word in actual.group(1).split(',')) if actual else None
            if run.returncode == 0:
                # 0x3f800000 -> the cap: this adapter's own frac already stays below 1.0, so the draw cannot
                # separate the two forms here. Recorded, not failed.
                print('  recorded: %s %s uncapped pair reads the capped word %s, so this adapter clamps a bare '
                      'frc itself and the control cannot separate the two forms on it' % (kind, adapter,
                                                                                        EXPECTED_TEXT))
            else:
                check('%s %s: the uncapped pair reads %s, not %s' % (kind, adapter, UNCAPPED_TEXT, EXPECTED_TEXT),
                      read == UNCAPPED, run.stdout + run.stderr)


def main():
    if len(sys.argv) == 1:
        controls()
        if failures:
            print('FAIL')
            for text in failures:
                print('  ' + text)
            sys.exit(1)
        print('PASS')
        return
    mode, *args = sys.argv[1:]
    if mode != 'draw':
        raise SystemExit(__doc__)
    controls()
    check_draw(*args)
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
