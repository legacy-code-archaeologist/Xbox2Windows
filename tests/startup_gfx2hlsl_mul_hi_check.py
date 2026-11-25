"""Unsigned 32x32->high-32 vector multiply (v_mul_hi_u32) in tools/startup/gfx2hlsl.py (Refs #870).

  (no arguments)                       translation, helper-math, refusal and mutation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input pairs and compare the four result words

The pixel fixture reads two flat parameters (v0..v3 and v4..v7), computes v_mul_hi_u32 across all four lanes into
v12..v15, and exports v12..v15 to mrt0. The vertex stage supplies the input words through flat parameters so the
pixel stage cannot constant-fold them.

Every result is compared with an independent Python reference ((a * b) >> 32). The emitted mul_hi_u32 HLSL helper is
parsed and evaluated in 32-bit unsigned arithmetic over a grid of edge cases. A mutation control replaces mul_hi_u32
with the low 32-bit product (a * b) while keeping the same inputs read, and must produce the mutated low-word readback
on both CPU evaluation and desktop GPU draws.
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

BODY_SOURCE = ''.join('v_mul_hi_u32 v%d, v%d, v%d\n' % (12 + lane, lane, 4 + lane) for lane in range(4))
BODY_WORDS = bytes.fromhex(
    '0c006ad500090200'
    '0d006ad5010b0200'
    '0e006ad5020d0200'
    '0f006ad5030f0200'
)
PS_WORDS = PRE_WORDS + BODY_WORDS + POST_WORDS

# VOP3A uses source selector 255 followed by the literal DWORD (#1792).
LITERAL_SOURCE = 'v_mul_hi_u32 v12, v0, 0xaaaaaaab\n'
LITERAL_WORDS = bytes.fromhex('0c006ad500ff0100abaaaaaa')

SPECIAL = (
    0, 1, 2, 3, 0xffff, 0x10000, 0x10001, 0x0001ffff, 0x7fffffff, 0x80000000, 0x80000001,
    0xdeadbeef, 0xcafebabe, 0x9e3779b9, 0x85ebca6b, 0xffff0000, 0xffff0001, 0xfffffffe, MASK,
)
GRID = [(a, b) for a in SPECIAL for b in SPECIAL]

# Four (src0, src1, unused_src2) tuples per draw. Set 0 separates high-word from low-word and from missing-carry bugs.
DRAW = [
    (0xffffffff, 0xffffffff, 0),
    (0x80000000, 0x00000002, 0),
    (0xdeadbeef, 0xcafebabe, 0),
    (0x00010001, 0x00010001, 0),
    (0x00000000, 0xffffffff, 0),
    (0x00000001, 0xffffffff, 0),
    (0x0000ffff, 0xffffffff, 0),
    (0xffff0000, 0x00010001, 0),
    (0x7fffffff, 0x7fffffff, 0),
    (0x80000001, 0x80000001, 0),
    (0x0001ffff, 0xffff0001, 0),
    (0x9e3779b9, 0x85ebca6b, 0),
]


def ref_mul_hi_u32(a, b):
    return ((a & MASK) * (b & MASK)) >> 32


def mut_mul_lo_u32(a, b):
    return ((a & MASK) * (b & MASK)) & MASK


def translate(mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), PS_WORDS, root_blob())
    text = pipe.hlsl_ps()
    if mutate:
        mutated = re.sub(r'mul_hi_u32\((v\d+), (v\d+)\)', r'(\1 * \2)', text)
        assert mutated != text, text
        text = mutated
    return pipe, text


def eval_emitted_helper(ps_text):
    """Parse the emitted mul_hi_u32 helper body and return a 32-bit evaluator for it."""
    m = re.search(r'uint mul_hi_u32\(uint a, uint b\) \{ (.*?) \}', ps_text)
    assert m, 'mul_hi_u32 helper missing from generated HLSL'
    body = m.group(1)
    m_vars = re.match(r'uint la = ([^,]+), lb = ([^;]+); uint middle = ([^;]+); return ([^;]+);$', body)
    assert m_vars, body
    la_e, lb_e, middle_e, ret_e = (Expr(part) for part in m_vars.groups())

    def run(a, b):
        env = {'a': a & MASK, 'b': b & MASK}
        env['la'] = la_e.run(env)[0]
        env['lb'] = lb_e.run(env)[0]
        env['middle'] = middle_e.run(env)[0]
        return ret_e.run(env)[0]

    return run

def controls():
    _, text = translate()
    found = re.findall(r'^\s*v(1[2-5]) = mul_hi_u32\(v(\d+), v(\d+)\);$', text, re.M)
    check('v_mul_hi_u32: four lane calls emitted',
          found == [('12', '0', '4'), ('13', '1', '5'), ('14', '2', '6'), ('15', '3', '7')])

    literal_ps = PRE_WORDS + LITERAL_WORDS + BODY_WORDS[8:] + POST_WORDS
    literal_text = G.Pipeline(vs_words([(0, 0, 0)] * 4), literal_ps, root_blob()).hlsl_ps()
    literal_match = re.search(r'v12 = mul_hi_u32\(v0, (0[xX][0-9a-fA-F]+|[0-9]+)u\);', literal_text)
    check('v_mul_hi_u32: encoded literal reaches the helper',
          literal_match is not None and int(literal_match[1], 0) == 0xaaaaaaab)

    helper = eval_emitted_helper(text)
    mismatch = next(((a, b, helper(a, b), ref_mul_hi_u32(a, b)) for a, b in GRID
                     if helper(a, b) != ref_mul_hi_u32(a, b)), None)
    check('v_mul_hi_u32: emitted helper matches reference on %d edge pairs' % len(GRID), mismatch is None)

    # Carry-drop negative control on the helper itself: dropping ((la * lb) >> 16) must fail on 0xffffffff * 0xffffffff.
    dropped = text.replace('+ ((la * lb) >> 16)', '+ 0u')
    assert dropped != text
    bad_helper = eval_emitted_helper(dropped)
    check('v_mul_hi_u32: dropping low-half carry mismatches 0xffffffff * 0xffffffff',
          bad_helper(0xffffffff, 0xffffffff) != ref_mul_hi_u32(0xffffffff, 0xffffffff))

    # Lane mutation control: replacing mul_hi_u32(vX, vY) with (vX * vY) yields the low word on draw set 0.
    mutated = translate(mutate=True)[1]
    mut_lanes = re.findall(r'^\s*v(1[2-5]) = (.*);$', mutated, re.M)
    assert [r for r, _ in mut_lanes] == ['12', '13', '14', '15'], mut_lanes
    mut_exprs = [Expr(rhs) for _, rhs in mut_lanes]
    first = DRAW[:4]
    ref_first = [ref_mul_hi_u32(a, b) for a, b, _ in first]
    mut_first = [mut_mul_lo_u32(a, b) for a, b, _ in first]
    eval_mut = [expr.run({'v%d' % lane: a, 'v%d' % (4 + lane): b})[0]
                for lane, (expr, (a, b, _)) in enumerate(zip(mut_exprs, first))]
    check('v_mul_hi_u32: draw set 0 mutation reads %s (differs from %s)' % (words_text(mut_first), words_text(ref_first)),
          mut_first != ref_first and eval_mut == mut_first)

    # Accepted operand forms (VGPR, SGPR, inline constant, literal) and exec-masked emission.
    check('accepted operand forms translate', not any(refused(a) for a in (
        'v_mul_hi_u32 v1, v2, v3', 'v_mul_hi_u32 v1, s0, v3',
        'v_mul_hi_u32 v1, v2, 7', 'v_mul_hi_u32 v1, v2, 0x12345678')))
    masked = Probe()
    masked.exec_known = False
    masked.one('v_mul_hi_u32', 'v_mul_hi_u32 v1, v2, v3')
    check('exec-masked v_mul_hi_u32 guards assignment',
          masked.lines == ['if (exec & 1) v1 = mul_hi_u32(v2, v3);'])

    # Unmodeled modifiers and wrong operand counts must stay refused.
    check('modifiers and operand modifiers refused', all(refused(a) for a in (
        'v_mul_hi_u32 v1, v2, v3 clamp', 'v_mul_hi_u32 v1, v2, v3 mul:2',
        'v_mul_hi_u32 v1, v2, v3 div:2', 'v_mul_hi_u32 v1, v2, v3 op_sel:[1,0,0,0]',
        'v_mul_hi_u32 v1, -v2, v3', 'v_mul_hi_u32 v1, |v2|, v3',
        'v_mul_hi_u32 v1, v2, -v3', 'v_mul_hi_u32 v1, v2, |v3|')))
    check('wrong operand counts refused', all(refused(a) for a in (
        'v_mul_hi_u32 v1', 'v_mul_hi_u32 v1, v2', 'v_mul_hi_u32 v1, v2, v3, v4')))


def check_binary(llvm_mc):
    built = assemble(llvm_mc, BODY_SOURCE)
    if built != BODY_WORDS:
        raise AssertionError('v_mul_hi_u32: assembled %s, recorded %s' % (built.hex(), BODY_WORDS.hex()))
    if assemble(llvm_mc, LITERAL_SOURCE) != LITERAL_WORDS:
        raise AssertionError('literal v_mul_hi_u32 fixture differs from llvm-mc')
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for index in range(0, len(DRAW), 4):
        chunk = DRAW[index:index + 4]
        if assemble(llvm_mc, vs_source(chunk)) != vs_words(chunk):
            raise AssertionError('vertex fixture for %s differs from llvm-mc' % (chunk,))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            good_ps = compile_hlsl(compiler, kind, translate()[1], 'ps_mul_hi_%s' % kind, ps_profile, directory)
            for index in range(0, len(DRAW), 4):
                chunk = DRAW[index:index + 4]
                vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), PS_WORDS, root_blob()).hlsl_vs(),
                                  'vs_mul_hi_%d_%s' % (index, kind), vs_profile, directory)
                expected = words_text(ref_mul_hi_u32(a, b) for a, b, _ in chunk)
                run = draw(renderer, adapter, vs, good_ps, expected)
                if run.returncode:
                    raise AssertionError('%s v_mul_hi_u32 inputs %s\n%s\n%s' % (kind, chunk, run.stdout, run.stderr))
                print('%s v_mul_hi_u32 %s: %s' % (kind, [('%x' % a, '%x' % b) for a, b, _ in chunk], expected),
                      flush=True)
                if index:
                    continue
                wrong = words_text(mut_mul_lo_u32(a, b) for a, b, _ in chunk)
                bad_ps = compile_hlsl(compiler, kind, translate(mutate=True)[1],
                                      'mutated_mul_hi_%s' % kind, ps_profile, directory)
                run = draw(renderer, adapter, vs, bad_ps, expected)
                if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                    raise AssertionError('%s v_mul_hi_u32 mutation did not read %s\n%s\n%s'
                                         % (kind, wrong, run.stdout, run.stderr))
                print('%s v_mul_hi_u32 mutation: fails as expected, reads %s' % (kind, wrong), flush=True)


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
