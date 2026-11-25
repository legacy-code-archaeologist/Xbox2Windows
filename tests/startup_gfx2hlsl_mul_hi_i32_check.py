"""Signed 32x32->high-32 multiply (v_mul_hi_i32, s_mul_hi_i32) in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation, helper-math, refusal and mutation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input pairs and compare the four result words

The pixel fixture reads two flat parameters (v0..v3 and v4..v7), computes v_mul_hi_i32 across all four lanes into
v12..v15, and exports v12..v15 to mrt0. The vertex stage supplies the input words through flat parameters so the
pixel stage cannot constant-fold them.

Every result is compared with an independent Python signed 64-bit reference ((s32(a) * s32(b)) >> 32) & 0xffffffff.
The emitted mul_hi_i32 helper (built on the existing mul_hi_u32 helper in Pipeline.common()) is parsed and evaluated
in 32-bit unsigned arithmetic over a 361-pair edge grid including INT_MIN * INT_MIN, INT_MIN * -1, -1 * -1, and mixed
signs. Controls verify s_mul_hi_i32 decoding/lowering (SCC unchanged, lane-mask/descriptor/bounds refusals), exec-masked
emission, modifier/arity refusals, and mutations replacing mul_hi_i32 with mul_hi_u32 (unsigned high product) or
dropping one sign-correction term on both CPU evaluation and desktop GPU draws.
"""
import re
import struct
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

BODY_SOURCE = ''.join('v_mul_hi_i32 v%d, v%d, v%d\n' % (12 + lane, lane, 4 + lane) for lane in range(4))
BODY_WORDS = bytes.fromhex(
    '0c006cd500090200'
    '0d006cd5010b0200'
    '0e006cd5020d0200'
    '0f006cd5030f0200'
)
PS_WORDS = PRE_WORDS + BODY_WORDS + POST_WORDS

SCALAR_SOURCE = 's_mul_hi_i32 s12, s10, s11\n'
SCALAR_WORDS = struct.pack('<I', (0b10 << 30) | (0x36 << 23) | (12 << 16) | (11 << 8) | 10)

SPECIAL = (
    0, 1, 2, 3, 0xffff, 0x10000, 0x10001, 0x0001ffff, 0x7fffffff, 0x80000000, 0x80000001,
    0xdeadbeef, 0xcafebabe, 0x9e3779b9, 0x85ebca6b, 0xffff0000, 0xffff0001, 0xfffffffe, MASK,
)
GRID = [(a, b) for a in SPECIAL for b in SPECIAL]

# Four (src0, src1, unused_src2) tuples per draw. Set 0 separates signed mul_hi_i32 from unsigned mul_hi_u32
# and from dropping either sign-correction term.
DRAW = [
    (0xffffffff, 0xffffffff, 0),  # -1 * -1 -> hi 0x00000000 (unsigned: 0xfffffffe)
    (0x80000000, 0x80000000, 0),  # INT_MIN * INT_MIN -> hi 0x40000000 (unsigned: 0x40000000, 1-term drop: 0xc0000000)
    (0x80000000, 0x00000002, 0),  # INT_MIN * 2 -> hi 0xffffffff (unsigned: 0x00000001)
    (0xdeadbeef, 0x7fffffff, 0),  # neg * INT_MAX -> hi 0xef56df77 (unsigned: 0x6f56df77)
    (0x80000000, 0xffffffff, 0),  # INT_MIN * -1 -> hi 0x00000000
    (0x00000000, 0xffffffff, 0),  # 0 * -1 -> hi 0x00000000
    (0x7fffffff, 0x7fffffff, 0),  # INT_MAX * INT_MAX -> hi 0x3fffffff
    (0x00000001, 0x80000000, 0),  # 1 * INT_MIN -> hi 0xffffffff
    (0xdeadbeef, 0xcafebabe, 0),  # neg * neg -> hi 0x0b164486
    (0x9e3779b9, 0x85ebca6b, 0),  # neg * neg -> hi 0x2ec3a0c2
    (0xffff0000, 0x00010001, 0),  # -65536 * 65537 -> hi 0xffffffff
    (0x00010001, 0x00010001, 0),  # pos * pos -> hi 0x00000001
]


def s32(v):
    v &= MASK
    return v - (1 << 32) if v & 0x80000000 else v


def ref_mul_hi_i32(a, b):
    return ((s32(a) * s32(b)) >> 32) & MASK


def ref_mul_hi_u32(a, b):
    return ((a & MASK) * (b & MASK)) >> 32


def translate(mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), PS_WORDS, root_blob())
    text = pipe.hlsl_ps()
    if mutate:
        mutated = re.sub(r'mul_hi_i32\((v\d+), (v\d+)\)', r'mul_hi_u32(\1, \2)', text)
        assert mutated != text, text
        text = mutated
    return pipe, text


def eval_emitted_helpers(ps_text):
    """Parse the emitted mul_hi_u32 and mul_hi_i32 helpers and return a 32-bit evaluator for mul_hi_i32."""
    mu = re.search(r'uint mul_hi_u32\(uint a, uint b\) \{ (.*?) \}', ps_text)
    assert mu, 'mul_hi_u32 helper missing from generated HLSL'
    u_stmts = [part.strip() for part in mu.group(1).split(';') if part.strip()]
    u_decls = []
    for stmt in u_stmts[:-1]:
        assert stmt.startswith('uint '), stmt
        for item in stmt[5:].split(','):
            name, rhs = [x.strip() for x in item.split('=', 1)]
            u_decls.append((name, Expr(rhs)))
    assert u_stmts[-1].startswith('return '), u_stmts[-1]
    u_ret = Expr(u_stmts[-1][7:])

    def run_u(a, b):
        env = {'a': a & MASK, 'b': b & MASK}
        for name, expr in u_decls:
            env[name] = expr.run(env)[0]
        return u_ret.run(env)[0]

    mi = re.search(r'uint mul_hi_i32\(uint a, uint b\) \{ return ([^;]+); \}', ps_text)
    assert mi, 'mul_hi_i32 helper missing from generated HLSL'
    i_expr_text = mi.group(1)
    assert 'mul_hi_u32(a, b)' in i_expr_text, i_expr_text
    i_expr = Expr(i_expr_text.replace('mul_hi_u32(a, b)', 'u'))

    def run_i(a, b):
        return i_expr.run({'a': a & MASK, 'b': b & MASK, 'u': run_u(a, b)})[0]

    return run_i


class ScalarProbe(Probe):
    def __init__(self):
        super().__init__()
        self.masks = set()
        self.scc_effect = None


def scalar_refused(line):
    try:
        ScalarProbe().one(line.split(' ', 1)[0], line)
    except G.Unsupported:
        return True
    return False


def controls():
    _, text = translate()
    found = re.findall(r'^\s*v(1[2-5]) = mul_hi_i32\(v(\d+), v(\d+)\);$', text, re.M)
    check('v_mul_hi_i32: four lane calls emitted',
          found == [('12', '0', '4'), ('13', '1', '5'), ('14', '2', '6'), ('15', '3', '7')])

    helper = eval_emitted_helpers(text)
    mismatch = next(((a, b, helper(a, b), ref_mul_hi_i32(a, b)) for a, b in GRID
                     if helper(a, b) != ref_mul_hi_i32(a, b)), None)
    check('v_mul_hi_i32: emitted helper matches reference on %d edge pairs' % len(GRID), mismatch is None)

    # Sign-correction negative controls: dropping either sign term must mismatch.
    drop_a = text.replace('((a >> 31) != 0u ? b : 0u)', '0u')
    drop_b = text.replace('((b >> 31) != 0u ? a : 0u)', '0u')
    assert drop_a != text and drop_b != text
    check('mul_hi_i32: dropping a-sign correction mismatches INT_MIN * 2',
          eval_emitted_helpers(drop_a)(0x80000000, 2) != ref_mul_hi_i32(0x80000000, 2))
    check('mul_hi_i32: dropping b-sign correction mismatches 2 * INT_MIN',
          eval_emitted_helpers(drop_b)(2, 0x80000000) != ref_mul_hi_i32(2, 0x80000000))

    # Draw set 0 mutation: substituting mul_hi_u32 for mul_hi_i32 produces distinct words.
    mutated = translate(mutate=True)[1]
    check('v_mul_hi_i32: mutated text calls mul_hi_u32 on all four lanes',
          re.findall(r'^\s*v(1[2-5]) = mul_hi_u32\(v(\d+), v(\d+)\);$', mutated, re.M)
          == [('12', '0', '4'), ('13', '1', '5'), ('14', '2', '6'), ('15', '3', '7')])
    first = DRAW[:4]
    ref_first = [ref_mul_hi_i32(a, b) for a, b, _ in first]
    mut_first = [ref_mul_hi_u32(a, b) for a, b, _ in first]
    check('v_mul_hi_i32: draw set 0 unsigned mutation reads %s (differs from %s)'
          % (words_text(mut_first), words_text(ref_first)), mut_first != ref_first)

    # Accepted vector operand forms (VGPR, SGPR, inline constant, literal) and exec-masked emission.
    check('v_mul_hi_i32: accepted operand forms translate', not any(refused(a) for a in (
        'v_mul_hi_i32 v1, v2, v3', 'v_mul_hi_i32 v1, s0, v3',
        'v_mul_hi_i32 v1, v2, -1', 'v_mul_hi_i32 v1, v2, 0x80000000')))
    masked = Probe()
    masked.exec_known = False
    masked.one('v_mul_hi_i32', 'v_mul_hi_i32 v1, v2, v3')
    check('exec-masked v_mul_hi_i32 guards assignment',
          masked.lines == ['if (exec & 1) v1 = mul_hi_i32(v2, v3);'])

    # Vector refusals: modifiers and wrong operand counts.
    check('v_mul_hi_i32: modifiers and operand modifiers refused', all(refused(a) for a in (
        'v_mul_hi_i32 v1, v2, v3 clamp', 'v_mul_hi_i32 v1, v2, v3 mul:2',
        'v_mul_hi_i32 v1, v2, v3 div:2', 'v_mul_hi_i32 v1, v2, v3 op_sel:[1,0,0,0]',
        'v_mul_hi_i32 v1, -v2, v3', 'v_mul_hi_i32 v1, |v2|, v3',
        'v_mul_hi_i32 v1, v2, -v3', 'v_mul_hi_i32 v1, v2, |v3|')))
    check('v_mul_hi_i32: wrong operand counts refused', all(refused(a) for a in (
        'v_mul_hi_i32 v1', 'v_mul_hi_i32 v1, v2', 'v_mul_hi_i32 v1, v2, v3, v4')))

    # Scalar s_mul_hi_i32: decode, lowering (SCC unchanged), SGPR-over-mask overwrite, and refusals.
    decoded = G.decode(SCALAR_WORDS)
    check('s_mul_hi_i32: SOP2 encoding decodes',
          len(decoded) == 1 and decoded[0][2] == 's_mul_hi_i32' and decoded[0][3] == 's_mul_hi_i32 s12, s10, s11')
    sp = ScalarProbe()
    sp.one('s_mul_hi_i32', 's_mul_hi_i32 s12, s10, s11')
    check('s_mul_hi_i32: lowers to mul_hi_i32 without touching scc',
          sp.lines == ['s12 = mul_hi_i32(s10, s11);'])
    sp_lit = ScalarProbe()
    sp_lit.one('s_mul_hi_i32', 's_mul_hi_i32 s12, s10, 0x80000001')
    check('s_mul_hi_i32: literal source translates',
          sp_lit.lines == ['s12 = mul_hi_i32(s10, 0x80000001u);'])
    # Overwriting an SGPR that previously held a compare mask succeeds and drops its mask mark.
    sp_over = ScalarProbe()
    sp_over.masks.add(12)
    sp_over.one('s_mul_hi_i32', 's_mul_hi_i32 s12, s10, s11')
    check('s_mul_hi_i32: overwriting mask-marked SGPR drops mask mark', 12 not in sp_over.masks)
    # Reading a mask-marked SGPR as source must be refused.
    sp_mask_src = ScalarProbe()
    sp_mask_src.masks.add(10)
    mask_src_refused = False
    try:
        sp_mask_src.one('s_mul_hi_i32', 's_mul_hi_i32 s12, s10, s11')
    except G.Unsupported:
        mask_src_refused = True
    check('s_mul_hi_i32: mask-marked SGPR source refused', mask_src_refused)
    check('s_mul_hi_i32: lane-mask, vcc/exec, out-of-range and wrong-count forms refused', all(scalar_refused(a) for a in (
        's_mul_hi_i32 vcc_lo, s10, s11', 's_mul_hi_i32 exec_lo, s10, s11',
        's_mul_hi_i32 s12, vcc_lo, s11', 's_mul_hi_i32 s12, s10, exec_lo',
        's_mul_hi_i32 s106, s10, s11', 's_mul_hi_i32 s12, s106, s11',
        's_mul_hi_i32 s12, s10', 's_mul_hi_i32 s12, s10, s11, s13')))
    sym_p = ScalarProbe()
    sym_p.sym[10] = ('ptr', ('vbtab',))
    sym_refused = False
    try:
        sym_p.one('s_mul_hi_i32', 's_mul_hi_i32 s12, s10, s11')
    except G.Unsupported:
        sym_refused = True
    check('s_mul_hi_i32: symbolic pointer source refused', sym_refused)


def check_binary(llvm_mc):
    for label, source, words in (
        ('v_mul_hi_i32 body', BODY_SOURCE, BODY_WORDS),
        ('s_mul_hi_i32', SCALAR_SOURCE, SCALAR_WORDS),
        ('prologue', PRE_SOURCE, PRE_WORDS),
        ('export', POST_SOURCE, POST_WORDS),
    ):
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s: assembled %s, recorded %s' % (label, built.hex(), words.hex()))
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
            good_ps = compile_hlsl(compiler, kind, translate()[1], 'ps_mul_hi_i32_%s' % kind, ps_profile, directory)
            for index in range(0, len(DRAW), 4):
                chunk = DRAW[index:index + 4]
                vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), PS_WORDS, root_blob()).hlsl_vs(),
                                  'vs_mul_hi_i32_%d_%s' % (index, kind), vs_profile, directory)
                expected = words_text(ref_mul_hi_i32(a, b) for a, b, _ in chunk)
                run = draw(renderer, adapter, vs, good_ps, expected)
                if run.returncode:
                    raise AssertionError('%s v_mul_hi_i32 inputs %s\n%s\n%s' % (kind, chunk, run.stdout, run.stderr))
                print('%s v_mul_hi_i32 %s: %s' % (kind, [('%x' % a, '%x' % b) for a, b, _ in chunk], expected),
                      flush=True)
                if index:
                    continue
                wrong = words_text(ref_mul_hi_u32(a, b) for a, b, _ in chunk)
                bad_ps = compile_hlsl(compiler, kind, translate(mutate=True)[1],
                                      'mutated_mul_hi_i32_%s' % kind, ps_profile, directory)
                run = draw(renderer, adapter, vs, bad_ps, expected)
                if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                    raise AssertionError('%s v_mul_hi_i32 mutation did not read %s\n%s\n%s'
                                         % (kind, wrong, run.stdout, run.stderr))
                print('%s v_mul_hi_i32 mutation: fails as expected, reads %s' % (kind, wrong), flush=True)


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
