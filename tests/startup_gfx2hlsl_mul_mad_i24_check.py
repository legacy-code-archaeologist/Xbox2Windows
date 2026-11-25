"""Signed 24-bit vector multiply and multiply-add (v_mul_i32_i24, v_mad_i32_i24) in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation, evaluator, refusal and mutation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pairs with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input sets and compare the four result words

Each pixel fixture reads three flat parameters (v0..v3, v4..v7, v8..v11), applies v_mul_i32_i24 (alternating VOP2 e32
and VOP3A e64 across the four lanes) or v_mad_i32_i24 (VOP3A) into v12..v15, and exports v12..v15 to mrt0.

Every result is compared with an independent Python signed 24-bit reference (signext24(a) * signext24(b) [+ c]) &
0xffffffff. Controls verify that bits[31:24] of S0/S1 are ignored, bit 23 sign-extends to 32 bits, exec-masked
emission guards the destination, modifiers (including integer clamp) and wrong operand counts stay refused, and
unsigned-24-bit mutations ((a & 0xffffffu) * (b & 0xffffffu) [+ c]) fail with exact expected wrong words on both CPU
evaluation and desktop GPU draws.
"""
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    Expr,
    MASK,
    POST_SOURCE,
    POST_WORDS,
    PRE_SOURCE,
    PRE_WORDS,
    Probe,
    check,
    refused,
    vs_source,
    vs_words,
    words_text,
)

OPS = ('v_mul_i32_i24', 'v_mad_i32_i24')

BODY_SOURCE = {
    'v_mul_i32_i24': ''.join(
        'v_mul_i32_i24_%s v%d, v%d, v%d\n' % ('e64' if lane % 2 else 'e32', 12 + lane, lane, 4 + lane)
        for lane in range(4)
    ),
    'v_mad_i32_i24': ''.join(
        'v_mad_i32_i24 v%d, v%d, v%d, v%d\n' % (12 + lane, lane, 4 + lane, 8 + lane)
        for lane in range(4)
    ),
}
BODY_WORDS = {
    'v_mul_i32_i24': bytes.fromhex('000918120d0009d5010b0200020d1c120f0009d5030f0200'),
    'v_mad_i32_i24': bytes.fromhex('0c0042d5000922040d0042d5010b26040e0042d5020d2a040f0042d5030f2e04'),
}

# Edge values covering 24-bit sign boundary (0x7fffff, 0x800000), -1 (0xffffff), upper-byte noise (0xab..., 0xff...),
# and 32-bit wraparound.
SPECIAL = (
    0x00000000, 0x00000001, 0x00000002, 0x007fffff, 0x00800000, 0x00800001, 0x00ffffff,
    0x01000000, 0x557fffff, 0xaa800000, 0xdeadbeef, 0xcafebabe, 0x7fffffff, 0x80000000,
    0xff7fffff, 0xffffffff,
)
ADDENDS = (0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0x12345678)

# Four (src0, src1, src2) tuples per draw. Set 0 separates signed 24-bit from unsigned 24-bit on every lane.
DRAW = {
    'v_mul_i32_i24': [
        (0x00ffffff, 0x00000002, 0),  # -1 * 2 = -2 (unsigned24: 0x01fffffe)
        (0x00800000, 0x00000003, 0),  # -8388608 * 3 = -25165824 (unsigned24: 0x01800000)
        (0xaa800000, 0x55ffffff, 0),  # upper bytes ignored: -8388608 * -1 = +8388608
        (0x00800000, 0x00800000, 0),  # INT24_MIN * INT24_MIN = 0 (mod 2^32) -> wait: 2^46 mod 2^32 = 0
        (0x007fffff, 0x007fffff, 0),  # +8388607^2
        (0xab000005, 0xcdffffff, 0),  # upper bytes ignored: 5 * -1 = -5
        (0x00ffffff, 0x00ffffff, 0),  # -1 * -1 = 1 (unsigned24: 0xfe000001)
        (0x01000000, 0xdeadbeef, 0),  # low 24 bits of src0 are 0 -> 0
    ],
    'v_mad_i32_i24': [
        (0x00ffffff, 0x00000002, 0x00000005),  # -1 * 2 + 5 = 3 (unsigned24: 0x02000003)
        (0x00800000, 0x00000003, 0x80000000),  # -25165824 + INT32_MIN
        (0xaa800000, 0x55ffffff, 0xffffffff),  # +8388608 + (-1) = 0x007fffff
        (0x00ffffff, 0x00ffffff, 0x7fffffff),  # 1 + INT32_MAX wraps to 0x80000000
        (0x007fffff, 0x007fffff, 0x12345678),  # positive * positive + addend
        (0xab000005, 0xcdffffff, 0x00000005),  # -5 + 5 = 0
        (0x01000000, 0xdeadbeef, 0xcafebabe),  # 0 * x + addend = addend
        (0x00800001, 0x00ffffff, 0x00000001),  # (-8388607) * (-1) + 1 = 0x00800000
    ],
}


def s24(v):
    v &= 0xffffff
    return v - (1 << 24) if v & 0x800000 else v


def ref_op(op, a, b, c):
    prod = s24(a) * s24(b)
    return (prod + (c & MASK) if op == 'v_mad_i32_i24' else prod) & MASK


def mut_unsigned24(op, a, b, c):
    prod = (a & 0xffffff) * (b & 0xffffff)
    return (prod + (c & MASK) if op == 'v_mad_i32_i24' else prod) & MASK


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


def translate(op, mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob())
    text = pipe.hlsl_ps()
    if mutate:
        mutated = re.sub(r'asuint\(asint\((v\d+) << 8\) >> 8\)', r'(\1 & 0xffffffu)', text)
        assert mutated != text, text
        text = mutated
    return pipe, text


def controls():
    for op in OPS:
        _, text = translate(op)
        lanes = re.findall(r'^\s*v(1[2-5]) = (.*);$', text, re.M)
        check('%s: four lane assignments emitted' % op, [r for r, _ in lanes] == ['12', '13', '14', '15'])
        exprs = [Expr(rhs) for _, rhs in lanes]

        # Evaluate emitted HLSL text on CPU across edge grid.
        if op == 'v_mul_i32_i24':
            cases = [(a, b, 0) for a in SPECIAL for b in SPECIAL]
        else:
            cases = [(a, b, c) for a in SPECIAL[:8] for b in SPECIAL[:8] for c in ADDENDS]
        mismatch = None
        for lane, expr in enumerate(exprs):
            for a, b, c in cases:
                got = expr.run({'v%d' % lane: a, 'v%d' % (4 + lane): b, 'v%d' % (8 + lane): c})[0]
                want = ref_op(op, a, b, c)
                if got != want:
                    mismatch = (lane, a, b, c, got, want)
                    break
            if mismatch:
                break
        check('%s: emitted expressions match reference on %d cases x 4 lanes' % (op, len(cases)), mismatch is None)

        # Unsigned-24-bit mutation control on draw set 0.
        mut_text = translate(op, mutate=True)[1]
        mut_lanes = re.findall(r'^\s*v(1[2-5]) = (.*);$', mut_text, re.M)
        mut_exprs = [Expr(rhs) for _, rhs in mut_lanes]
        first = DRAW[op][:4]
        ref_first = [ref_op(op, a, b, c) for a, b, c in first]
        mut_first = [mut_unsigned24(op, a, b, c) for a, b, c in first]
        eval_mut = [expr.run({'v%d' % lane: a, 'v%d' % (4 + lane): b, 'v%d' % (8 + lane): c})[0]
                    for lane, (expr, (a, b, c)) in enumerate(zip(mut_exprs, first))]
        check('%s: draw set 0 unsigned24 mutation reads %s (differs from %s)'
              % (op, words_text(mut_first), words_text(ref_first)),
              mut_first != ref_first and eval_mut == mut_first)

    # Accepted operand forms (VGPR, SGPR, inline constant, literal, _e32/_e64) and exec-masked emission.
    check('accepted operand forms translate', not any(refused(a) for a in (
        'v_mul_i32_i24 v1, v2, v3', 'v_mul_i32_i24_e32 v1, v2, v3', 'v_mul_i32_i24_e64 v1, s0, v3',
        'v_mul_i32_i24 v1, -1, v3', 'v_mul_i32_i24 v1, 0xaa800000, v3',
        'v_mad_i32_i24 v1, v2, v3, v4', 'v_mad_i32_i24 v1, s0, v3, -1',
        'v_mad_i32_i24 v1, v2, 0x00ffffff, v4')))
    for op, line, suffix in (
        ('v_mul_i32_i24', 'v_mul_i32_i24 v1, v2, v3', ''),
        ('v_mad_i32_i24', 'v_mad_i32_i24 v1, v2, v3, v4', ' + v4'),
    ):
        masked = Probe()
        masked.exec_known = False
        masked.one(op, line)
        check('exec-masked %s guards assignment' % op,
              masked.lines == ['if (exec & 1) v1 = asuint(asint(v2 << 8) >> 8) * asuint(asint(v3 << 8) >> 8)%s;' % suffix])

    # Unmodeled modifiers (including integer clamp) and wrong operand counts stay refused.
    check('modifiers and operand modifiers refused', all(refused(a) for a in (
        'v_mul_i32_i24 v1, v2, v3 clamp', 'v_mul_i32_i24 v1, v2, v3 mul:2',
        'v_mul_i32_i24 v1, v2, v3 div:2', 'v_mul_i32_i24 v1, v2, v3 op_sel:[1,0,0,0]',
        'v_mul_i32_i24 v1, -v2, v3', 'v_mul_i32_i24 v1, |v2|, v3',
        'v_mad_i32_i24 v1, v2, v3, v4 clamp', 'v_mad_i32_i24 v1, v2, v3, v4 mul:2',
        'v_mad_i32_i24 v1, v2, v3, v4 op_sel:[1,0,0,0]', 'v_mad_i32_i24 v1, -v2, v3, v4',
        'v_mad_i32_i24 v1, v2, |v3|, v4', 'v_mad_i32_i24 v1, v2, v3, -v4')))
    check('wrong operand counts refused', all(refused(a) for a in (
        'v_mul_i32_i24 v1', 'v_mul_i32_i24 v1, v2', 'v_mul_i32_i24 v1, v2, v3, v4',
        'v_mad_i32_i24 v1', 'v_mad_i32_i24 v1, v2, v3', 'v_mad_i32_i24 v1, v2, v3, v4, v5')))


def check_binary(llvm_mc):
    for op in OPS:
        built = assemble(llvm_mc, BODY_SOURCE[op])
        if built != BODY_WORDS[op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op].hex()))
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for op in OPS:
        for index in range(0, len(DRAW[op]), 4):
            chunk = DRAW[op][index:index + 4]
            if assemble(llvm_mc, vs_source(chunk)) != vs_words(chunk):
                raise AssertionError('vertex fixture for %s %s differs from llvm-mc' % (op, chunk))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op in OPS:
                good_ps = compile_hlsl(compiler, kind, translate(op)[1], 'ps_%s_%s' % (op, kind), ps_profile, directory)
                for index in range(0, len(DRAW[op]), 4):
                    chunk = DRAW[op][index:index + 4]
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), ps_words(op), root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    expected = words_text(ref_op(op, a, b, c) for a, b, c in chunk)
                    run = draw(renderer, adapter, vs, good_ps, expected)
                    if run.returncode:
                        raise AssertionError('%s %s inputs %s\n%s\n%s' % (kind, op, chunk, run.stdout, run.stderr))
                    print('%s %s draw %d: %s' % (kind, op, index // 4, expected), flush=True)
                    if index:
                        continue
                    wrong = words_text(mut_unsigned24(op, a, b, c) for a, b, c in chunk)
                    bad_ps = compile_hlsl(compiler, kind, translate(op, mutate=True)[1],
                                          'mutated_%s_%s' % (op, kind), ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad_ps, expected)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s %s mutation did not read %s\n%s\n%s'
                                             % (kind, op, wrong, run.stdout, run.stderr))
                    print('%s %s mutation: fails as expected, reads %s' % (kind, op, wrong), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
