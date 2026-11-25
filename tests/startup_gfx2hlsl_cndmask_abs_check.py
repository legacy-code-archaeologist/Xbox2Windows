"""Absolute-value (`|src|` and `-|src|`) modifiers on `v_cndmask_b32_e64` in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per input set and compare the four exported words

Per RDNA2 ISA VOP3A, `abs` clears bit 31 (`& 0x7fffffffu`) FIRST and `neg` flips bit 31 (`^ 0x80000000u`)
SECOND, so `-|x|` forces bit 31 to 1 (`((x & 0x7fffffffu) ^ 0x80000000u)`), whereas `-x` flips bit 31.
All instruction bytes are **constructed here** from the RDNA2 field layout decoded by `G.decode()` (no game data):
  VOP3A w0: [31:26] = 0b110101, [25:16] opcode (VOP2 e64 = 0x100 + op, VOPC e64 = op),
            [15] clamp, [10:8] abs for src2..src0, [7:0] vdst
        w1: [8:0] src0, [17:9] src1, [26:18] src2, [28:27] omod, [31:29] neg for src2..src0
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
from startup_flat_param_check import VS_WORDS, compile_hlsl, draw  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr, mov_literal  # noqa: E402

MASK = 0xffffffff
SIGN_BIT = 0x80000000
MAG_MASK = 0x7fffffff

VOP2_E64_BASE = 0x100
CNDMASK = 0x01
AND_B32 = 0x1b
V_CMP_NE_U32 = 0xc5

# RDNA2 9-bit VOP3 source encoding: 0..105 = s0..s105, 256..511 = v0..v255.
S4 = 4
S8 = 8
V0 = 256 + 0
V1 = 256 + 1
V2 = 256 + 2
V30 = 256 + 30

# 32-bit bit patterns covering signed zero, normal floats, qNaNs, infinities, negative denormal, and bit payloads.
PATTERNS = (
    0x00000000,  # +0.0
    0x80000000,  # -0.0
    0x3f800000,  # +1.0
    0xbf800000,  # -1.0
    0x7fc00001,  # +qNaN
    0xffc00001,  # -qNaN
    0x7f800000,  # +inf
    0xff800000,  # -inf
    0x80000001,  # -denormal
    0x12345678,  # +arbitrary
    0x92345678,  # -arbitrary
)


def encode_vop3a(opcode, vdst, src0, src1, src2=0, *,
                 neg0=0, neg1=0, neg2=0, abs0=0, abs1=0, abs2=0, clamp=0, omod=0):
    """Encode a VOP3A instruction (VOP2 e64 or VOPC e64)."""
    w0 = ((0b110101 << 26) | ((opcode & 0x3ff) << 16) | ((clamp & 1) << 15)
          | ((abs2 & 1) << 10) | ((abs1 & 1) << 9) | ((abs0 & 1) << 8) | (vdst & 0xff))
    w1 = ((src0 & 0x1ff) | ((src1 & 0x1ff) << 9) | ((src2 & 0x1ff) << 18) | ((omod & 0x3) << 27)
          | ((neg0 & 1) << 29) | ((neg1 & 1) << 30) | ((neg2 & 1) << 31))
    return struct.pack('<II', w0, w1)


def encode_vop2_e64(op, vdst, src0, src1, src2=0, **kw):
    return encode_vop3a(VOP2_E64_BASE + op, vdst, src0, src1, src2, **kw)


def encode_sop2_literal(op, sdst, ssrc0, literal):
    """Encode an SOP2 instruction with ssrc1=255 (literal dword)."""
    w0 = (0b10 << 30) | ((op & 0x7f) << 23) | ((sdst & 0x7f) << 16) | (255 << 8) | (ssrc0 & 0xff)
    return struct.pack('<II', w0, literal & MASK)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate_code(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


def refused_code(code):
    try:
        translate_code(code)
    except G.Unsupported:
        return True
    return False


def isa_abs(x):
    return x & MAG_MASK


def isa_neg(x):
    return x ^ SIGN_BIT


def isa_neg_abs(x):
    return (x & MAG_MASK) ^ SIGN_BIT


def wrong_neg_before_abs(x):
    return (x ^ SIGN_BIT) & MAG_MASK


# ---- Draw fixture: synthetic VS + PS decoded and lowered through G.Pipeline ----------------------------
def root_blob_with_const():
    """DXBC container with one RTS0 part declaring one 32-bit root constant at b0 (ud1 = 0 in offscreen_check)."""
    # RTS0 header (24 bytes): ver=2, nparam=1, poff=24, nsamp=0, soff=36, flags=0
    # param0 entry (12 bytes): type=1 (32-bit constants), vis=0 (all), off=36
    # constants payload (12 bytes): reg=0, space=0, count=1
    rts0 = struct.pack('<6I3I3I', 2, 1, 24, 0, 36, 0, 1, 0, 36, 0, 0, 1)
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return (b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part)


def vs_words(x_word, y_word, mask_word):
    """Full-screen triangle VS exporting (x_word, y_word, mask_word, 0) to param0."""
    out = VS_WORDS[:40]
    for idx, val in enumerate((x_word, y_word, mask_word, 0)):
        out += mov_literal(11 + idx, val)
    out += bytes.fromhex('cf0800f801020504')  # exp pos0 v1, v2, v5, v4 done
    out += struct.pack('<2I', 0xf800000f | (32 << 4), (11) | (12 << 8) | (13 << 16) | (14 << 24))  # exp param0
    return out + bytes.fromhex('000081bf')  # s_endpgm


S4_SEED = 0x92345678
PRE_WORDS = bytes.fromhex('0203fcbe020002c8020106c802020ac8')  # s_mov_b32 m0, s2; v_interp_mov_f32 v0..v2, p0, attr0.x..z
POST_WORDS = bytes.fromhex('0f1800f80c0d0e0f000081bf')  # exp mrt0 v12..v15 done vm; s_endpgm

PS_BODY_WORDS = (
    encode_vop3a(V_CMP_NE_U32, 30, 128, V2)                         # v_cmp_ne_u32_e64 s30, 0, v2
    + encode_sop2_literal(0x12, S4, S8, S4_SEED)                    # s_xor_b32 s4, s8, 0x92345678 (runtime ud1 ^ seed)
    + encode_vop2_e64(CNDMASK, 12, V0, V0, 30, abs1=1)              # v12 = (s30 & 1) ? |v0| : v0
    + encode_vop2_e64(CNDMASK, 13, V0, V0, 30, abs0=1, abs1=1, neg1=1)  # v13 = (s30 & 1) ? -|v0| : |v0|
    + encode_vop2_e64(CNDMASK, 14, V1, V1, 30, abs0=1, neg0=1, neg1=1)  # v14 = (s30 & 1) ? -v1 : -|v1|
    + encode_vop2_e64(CNDMASK, 15, S4, S4, 30, abs0=1, abs1=1, neg1=1)  # v15 = (s30 & 1) ? -|s4| : |s4|
)
PS_WORDS = PRE_WORDS + PS_BODY_WORDS + POST_WORDS


def expected_outputs(x_word, y_word, mask_bit):
    """ISA-level reference words for (v12, v13, v14, v15)."""
    v12 = isa_abs(x_word) if mask_bit else x_word
    v13 = isa_neg_abs(x_word) if mask_bit else isa_abs(x_word)
    v14 = isa_neg(y_word) if mask_bit else isa_neg_abs(y_word)
    v15 = isa_neg_abs(S4_SEED) if mask_bit else isa_abs(S4_SEED)
    return (v12, v13, v14, v15)


def words_text(values):
    return ','.join('%08x' % v for v in values)


def mutate_reverse_order(hlsl_text):
    """Mutate `((reg & 0x7fffffffu) ^ 0x80000000u)` -> `((reg ^ 0x80000000u) & 0x7fffffffu)` (wrong neg-then-abs)."""
    mutated = re.sub(r'\(\((\w+) & 0x7fffffffu\) \^ 0x80000000u\)',
                     r'((\1 ^ 0x80000000u) & 0x7fffffffu)', hlsl_text)
    assert mutated != hlsl_text
    return mutated


def mutate_drop_abs(hlsl_text):
    r"""Mutate `(\w+ & 0x7fffffffu)` -> `\1` (drops abs from both `|x|` and `-|x|`)."""
    mutated = re.sub(r'\((\w+) & 0x7fffffffu\)', r'\1', hlsl_text)
    assert mutated != hlsl_text
    return mutated


def reverse_order_outputs(x_word, y_word, mask_bit):
    v12 = isa_abs(x_word) if mask_bit else x_word
    v13 = wrong_neg_before_abs(x_word) if mask_bit else isa_abs(x_word)
    v14 = isa_neg(y_word) if mask_bit else wrong_neg_before_abs(y_word)
    v15 = wrong_neg_before_abs(S4_SEED) if mask_bit else isa_abs(S4_SEED)
    return (v12, v13, v14, v15)


def drop_abs_outputs(x_word, y_word, mask_bit):
    v12 = x_word
    v13 = isa_neg(x_word) if mask_bit else x_word
    v14 = isa_neg(y_word)
    v15 = isa_neg(S4_SEED) if mask_bit else S4_SEED
    return (v12, v13, v14, v15)


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        raise AssertionError(f"{label}: {detail}" if detail else label)


def check_controls():
    # 1. Exact off_diag.py shape: v_cndmask_b32_e64 v0, v0, |v0|, s30.
    asm, emitted = translate_code(encode_vop2_e64(CNDMASK, 0, V0, V0, 30, abs1=1))
    check('decodes exact off_diag shape',
          asm == ['v_cndmask_b32_e64 v0, v0, |v0|, s30'], repr(asm))
    check('lowers |v0| on src1 with s30 mask',
          emitted == 'v0 = (s30 & 1) ? (v0 & 0x7fffffffu) : v0;', emitted)

    # 2. |v0| and -|v0| on src0 and src1, and mixed combinations.
    asm, emitted = translate_code(encode_vop2_e64(CNDMASK, 0, V0, V1, 30, abs0=1))
    check('|v0| on src0',
          asm == ['v_cndmask_b32_e64 v0, |v0|, v1, s30']
          and emitted == 'v0 = (s30 & 1) ? v1 : (v0 & 0x7fffffffu);', emitted)

    asm, emitted = translate_code(encode_vop2_e64(CNDMASK, 0, V0, V1, 30, abs0=1, neg0=1))
    check('-|v0| on src0 applies abs before neg',
          asm == ['v_cndmask_b32_e64 v0, -|v0|, v1, s30']
          and emitted == 'v0 = (s30 & 1) ? v1 : ((v0 & 0x7fffffffu) ^ 0x80000000u);', emitted)

    asm, emitted = translate_code(encode_vop2_e64(CNDMASK, 0, V0, V1, 30, abs1=1, neg1=1))
    check('-|v1| on src1 applies abs before neg',
          asm == ['v_cndmask_b32_e64 v0, v0, -|v1|, s30']
          and emitted == 'v0 = (s30 & 1) ? ((v1 & 0x7fffffffu) ^ 0x80000000u) : v0;', emitted)

    asm, emitted = translate_code(encode_vop2_e64(CNDMASK, 0, V0, V1, 30, abs0=1, neg0=1, abs1=1))
    check('-|v0| on src0 and |v1| on src1 together',
          asm == ['v_cndmask_b32_e64 v0, -|v0|, |v1|, s30']
          and emitted == 'v0 = (s30 & 1) ? (v1 & 0x7fffffffu) : ((v0 & 0x7fffffffu) ^ 0x80000000u);', emitted)

    # 3. SGPR |s4| and -|s4|.
    asm, emitted = translate_code(encode_vop2_e64(CNDMASK, 0, S4, S4, 30, abs0=1, abs1=1, neg1=1))
    check('|s4| and -|s4| on SGPR sources',
          asm == ['v_cndmask_b32_e64 v0, |s4|, -|s4|, s30']
          and emitted == 'v0 = (s30 & 1) ? ((s4 & 0x7fffffffu) ^ 0x80000000u) : (s4 & 0x7fffffffu);', emitted)

    # 4. Existing -v30 behavior unchanged, and neg(|v0|) spelling also lowers identically.
    asm, emitted = translate_code(encode_vop2_e64(CNDMASK, 31, V30, V30, 0, neg1=1))
    check('-v30 keeps existing bit-31 flip',
          emitted == 'v31 = (s0 & 1) ? (v30 ^ 0x80000000u) : v30;', emitted)
    s_neg_paren = Stage()
    s_neg_paren.one('v_cndmask_b32_e64', 'v_cndmask_b32_e64 v0, v0, neg(|v0|), s30')
    check('neg(|v0|) spelling applies abs before neg',
          ' '.join(s_neg_paren.lines) == 'v0 = (s30 & 1) ? ((v0 & 0x7fffffffu) ^ 0x80000000u) : v0;')

    # 5. Refusals preserved: ABS on immediates, ABS/NEG on mask, ABS on other integer ops, low16_only_v.
    literal = struct.pack('<I', 0x12345678)
    for src_enc, desc in ((128, 'inline 0'), (129, 'inline 1'), (193, 'inline -1'),
                          (240, 'inline 0.5'), (243, 'inline -1.0')):
        check(f'ABS on {desc} src0 refused', refused_code(encode_vop2_e64(CNDMASK, 0, src_enc, V0, 30, abs0=1)))
        check(f'ABS+NEG on {desc} src1 refused',
              refused_code(encode_vop2_e64(CNDMASK, 0, V0, src_enc, 30, abs1=1, neg1=1)))
    check('ABS on literal src1 refused',
          refused_code(encode_vop2_e64(CNDMASK, 0, V0, 255, 30, abs1=1) + literal))
    check('ABS+NEG on literal src0 refused',
          refused_code(encode_vop2_e64(CNDMASK, 0, 255, V0, 30, abs0=1, neg0=1) + literal))
    check('ABS on mask (src2) refused', refused_code(encode_vop2_e64(CNDMASK, 0, V0, V1, 30, abs2=1)))
    check('NEG on mask (src2) refused', refused_code(encode_vop2_e64(CNDMASK, 0, V0, V1, 30, neg2=1)))
    check('ABS on v_and_b32_e64 refused', refused_code(encode_vop2_e64(AND_B32, 0, V0, V1, abs1=1)))
    check('v_cndmask_b32 is not in UNIFORM_VALU', 'v_cndmask_b32' not in G.UNIFORM_VALU)

    s_low16 = Stage()
    s_low16.low16_only_v = {0}
    try:
        s_low16.one('v_cndmask_b32_e64', 'v_cndmask_b32_e64 v1, v1, |v0|, s30')
        low16_refused = False
    except G.Unsupported:
        low16_refused = True
    check('|v0| on low16_only_v refused (reads all 32 bits)', low16_refused)

    # 6. Value-level evaluation across all 11 32-bit patterns distinguishing abs, neg, -|x|, and wrong orders.
    #    Construct a full Pipeline from synthetic PS_WORDS and evaluate its emitted HLSL expressions on CPU.
    pipe = G.Pipeline(vs_words(0, 0, 0), PS_WORDS, root_blob_with_const())
    ps_hlsl = pipe.hlsl_ps()
    check('PS pipeline translates cleanly', 'v12 = ' in ps_hlsl and 'v15 = ' in ps_hlsl, ps_hlsl)
    decoded_ps = [asm for _o, _s, _n, asm in G.decode(PS_WORDS)]
    check('PS decodes expected v_cndmask_b32_e64 instructions',
          any('v_cndmask_b32_e64 v12, v0, |v0|, s30' == a for a in decoded_ps)
          and any('v_cndmask_b32_e64 v15, |s4|, -|s4|, s30' == a for a in decoded_ps), repr(decoded_ps))

    def eval_ps(text, x_word, y_word, mask_bit):
        s30_rhs = re.search(r'^\s*s30 = (.*);$', text, re.M).group(1)
        s4_rhs = re.search(r'^\s*s4 = (.*);$', text, re.M).group(1)
        env = {'v0': x_word, 'v1': y_word, 'v2': mask_bit, 'ud1': 0}
        env['s30'] = Expr(s30_rhs).run(env)[0]
        env['s4'] = Expr(s4_rhs).run(env)[0]
        out = []
        for reg in range(12, 16):
            rhs = re.search(r'^\s*v%d = (.*);$' % reg, text, re.M).group(1)
            out.append(Expr(rhs).run(env)[0])
        return tuple(out)

    for x in PATTERNS:
        for mask_bit in (0, 1):
            y = x ^ SIGN_BIT
            got = eval_ps(ps_hlsl, x, y, mask_bit)
            want = expected_outputs(x, y, mask_bit)
            if got != want:
                raise AssertionError(f'CPU eval mismatch for x=0x{x:08x} y=0x{y:08x} mask={mask_bit}: '
                                     f'got={got} want={want}')
    check('CPU HLSL evaluator matches ISA reference on all 22 (11 patterns x 2 masks) combinations', True)

    # Verify the two mutation models fail against the ISA reference and match their own HLSL mutations.
    rev_hlsl = mutate_reverse_order(ps_hlsl)
    drop_hlsl = mutate_drop_abs(ps_hlsl)
    rev_failures = 0
    drop_failures = 0
    for x in PATTERNS:
        for mask_bit in (0, 1):
            y = x ^ SIGN_BIT
            want = expected_outputs(x, y, mask_bit)
            rev_got = eval_ps(rev_hlsl, x, y, mask_bit)
            if rev_got != reverse_order_outputs(x, y, mask_bit):
                raise AssertionError(f'reverse-order evaluator disagreed with model at x=0x{x:08x}')
            if rev_got != want:
                rev_failures += 1
            drop_got = eval_ps(drop_hlsl, x, y, mask_bit)
            if drop_got != drop_abs_outputs(x, y, mask_bit):
                raise AssertionError(f'drop-abs evaluator disagreed with model at x=0x{x:08x}')
            if drop_got != want:
                drop_failures += 1
    check(f'reverse-order mutation differs from ISA on all {rev_failures}/22 cases', rev_failures == 22)
    check(f'drop-abs mutation differs from ISA on {drop_failures}/22 cases', drop_failures == 22)


def run_draw_case(compiler, kind, vs_profile, ps_profile, renderer, adapter, directory,
                  tag, x_word, y_word, mask_bit, ps_cso, expected, *, expect_wrong=None):
    vs_src = G.Pipeline(vs_words(x_word, y_word, mask_bit), PS_WORDS, root_blob_with_const()).hlsl_vs()
    vs_cso = compile_hlsl(compiler, kind, vs_src, f'vs_{tag}_{kind}', vs_profile, directory)
    run = draw(renderer, adapter, vs_cso, ps_cso, expected)
    if expect_wrong is None:
        if run.returncode:
            raise AssertionError(f'{kind} {tag} failed\n{run.stdout}\n{run.stderr}')
        print(f'{kind} {tag}: {expected}', flush=True)
    else:
        if run.returncode == 0 or f'actual={expect_wrong}' not in run.stderr:
            raise AssertionError(f'{kind} {tag} mutation did not report actual={expect_wrong}\n'
                                 f'{run.stdout}\n{run.stderr}')
        print(f'{kind} {tag} mutation: fails as expected, reads {expect_wrong}', flush=True)


def check_draw(dxc, renderer, adapter, fxc=None):
    check_controls()
    ps_hlsl = G.Pipeline(vs_words(0, 0, 0), PS_WORDS, root_blob_with_const()).hlsl_ps()
    rev_ps_hlsl = mutate_reverse_order(ps_hlsl)
    drop_ps_hlsl = mutate_drop_abs(ps_hlsl)

    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    # Representative draw patterns covering +0/-0, +/-1, +/-qNaN, +/-inf, -denorm, +/-normal on mask=0 and mask=1.
    draw_cases = [(x, x ^ SIGN_BIT, m) for x in PATTERNS for m in (0, 1)]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            good_ps = compile_hlsl(compiler, kind, ps_hlsl, f'ps_good_{kind}', ps_profile, directory)
            rev_ps = compile_hlsl(compiler, kind, rev_ps_hlsl, f'ps_rev_{kind}', ps_profile, directory)
            drop_ps = compile_hlsl(compiler, kind, drop_ps_hlsl, f'ps_drop_{kind}', ps_profile, directory)
            for idx, (x_word, y_word, mask_bit) in enumerate(draw_cases):
                expected = words_text(expected_outputs(x_word, y_word, mask_bit))
                run_draw_case(compiler, kind, vs_profile, ps_profile, renderer, adapter, directory,
                              f'case{idx}_x{x_word:08x}_m{mask_bit}', x_word, y_word, mask_bit, good_ps, expected)
            # Negative / mutated draws: reversing neg/abs order and dropping abs must fail with exact wrong words.
            for m_x, m_y, m_bit, label in ((0xbf800000, 0x3f800000, 1, 'neg_one_m1'),
                                           (0x80000000, 0x00000000, 0, 'neg_zero_m0')):
                expected = words_text(expected_outputs(m_x, m_y, m_bit))
                rev_wrong = words_text(reverse_order_outputs(m_x, m_y, m_bit))
                drop_wrong = words_text(drop_abs_outputs(m_x, m_y, m_bit))
                run_draw_case(compiler, kind, vs_profile, ps_profile, renderer, adapter, directory,
                              f'rev_{label}', m_x, m_y, m_bit, rev_ps, expected, expect_wrong=rev_wrong)
                run_draw_case(compiler, kind, vs_profile, ps_profile, renderer, adapter, directory,
                              f'drop_{label}', m_x, m_y, m_bit, drop_ps, expected, expect_wrong=drop_wrong)


def main():
    if len(sys.argv) == 1:
        check_controls()
        print('PASS')
        return
    mode, *args = sys.argv[1:]
    if mode == 'draw':
        check_draw(*args)
        print('PASS')
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
