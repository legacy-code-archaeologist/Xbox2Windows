"""Binary16 rounding and transcendental operations of tools/startup/gfx2hlsl.py drawn on a desktop adapter.

  (no arguments)                       reference controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per three input sets and compare the four result words

Same fixture as startup_gfx2hlsl_bit_alu_check.py: three flat parameters, one operation on each of the four lanes
(VOP1 and VOP3 spellings alternate), v12..v15 exported to mrt0. Each lane source is a binary16 value in the low half
with a non-zero high half, which the operation must ignore. The expected word is written from the ISA description
with float64 arithmetic and the standard binary16 rounding of struct, not with the lowering's expressions: the low
half holds the rounded result and the high half is clear. The inputs keep every result far from a rounding tie, so a
binary32 intermediate and an exact one agree.
"""
import math
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_bit_alu_check as B  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402

FRACT_MAX = 0.99951171875  # the largest binary16 below one: v_fract_f16 never returns 1.0
TAU = 2 * math.pi


def half(bits):
    return struct.unpack('<e', struct.pack('<H', bits))[0]


def bits(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def ref_fract(x):
    return min(x - math.floor(x), FRACT_MAX)


# mnemonic -> reference on the float value, in float64; a rounding that reaches zero keeps the sign of the input
REF = {
    'v_ceil_f16': lambda x: math.copysign(math.ceil(x), x), 'v_floor_f16': math.floor,
    'v_trunc_f16': lambda x: math.copysign(math.trunc(x), x), 'v_rndne_f16': lambda x: math.copysign(round(x), x),
    'v_fract_f16': ref_fract, 'v_rcp_f16': lambda x: 1.0 / x, 'v_rsq_f16': lambda x: 1.0 / math.sqrt(x),
    'v_sqrt_f16': math.sqrt, 'v_sin_f16': lambda x: math.sin(TAU * x), 'v_cos_f16': lambda x: math.cos(TAU * x),
    'v_exp_f16': lambda x: 2.0 ** x, 'v_log_f16': math.log2,
}
ROUND_INPUTS = [-2.5, -1.5, -0.3, 0.3, 0.5, 1.5, 2.5, 2.7, -2.7, 1000.5, 0.0, -0.0001]
POSITIVE = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 9.0, 0.3, 7.5, 100.0, 0.01, 1000.0]
INPUTS = {name: ROUND_INPUTS for name in ('v_ceil_f16', 'v_floor_f16', 'v_trunc_f16', 'v_rndne_f16')}
INPUTS.update({
    'v_fract_f16': [-2.5, -1.25, -0.25, 0.3, 0.5, 1.5, 2.5, 2.75, -2.75, 1000.5, 0.0, -0.0001],
    'v_rcp_f16': [0.25, 0.5, 1.0, -2.0, 3.0, -4.0, 10.0, 0.3, -7.5, 100.0, 0.01, 1000.0],
    'v_rsq_f16': POSITIVE, 'v_sqrt_f16': POSITIVE, 'v_log_f16': POSITIVE,
    'v_sin_f16': [0.125, 0.0625, 0.3, 0.4375, -0.1, 0.9, 1.1, 0.2, 0.6, -0.35, 0.05, 0.8],
    'v_cos_f16': [0.125, 0.0625, 0.3, 0.4375, -0.1, 0.9, 1.1, 0.2, 0.6, -0.35, 0.05, 0.8],
    'v_exp_f16': [0.0, 1.0, -1.0, 0.5, -0.5, 2.5, 3.0, -3.0, 0.3, 7.0, -7.0, 10.0],
})
GARBAGE = 0xdead0000
# A binary16 result below 0x8000 is a binary32 denormal and the render target flushes it. The fixture sets bit 30
# of the word (v_or_b32 with this literal) so the target keeps all 32 bits; the expected word has it set too.
MARK = 0x40000000


def source_words(op):
    return [GARBAGE | bits(x) for x in INPUTS[op]]


def expected_word(op, word):
    return MARK | bits(REF[op](half(word & 0xffff)))


def words_text(values):
    return ','.join('%08x' % v for v in values)


BODY_WORDS = {
    'v_ceil_f16': bytes.fromhex(
        '00b9187ef41818380d00dcd501010000f41a1a3802b91c7ef41c1c380f00dcd5'
        '03010000f41e1e38'),
    'v_floor_f16': bytes.fromhex(
        '00b7187ef41818380d00dbd501010000f41a1a3802b71c7ef41c1c380f00dbd5'
        '03010000f41e1e38'),
    'v_trunc_f16': bytes.fromhex(
        '00bb187ef41818380d00ddd501010000f41a1a3802bb1c7ef41c1c380f00ddd5'
        '03010000f41e1e38'),
    'v_rndne_f16': bytes.fromhex(
        '00bd187ef41818380d00ded501010000f41a1a3802bd1c7ef41c1c380f00ded5'
        '03010000f41e1e38'),
    'v_fract_f16': bytes.fromhex(
        '00bf187ef41818380d00dfd501010000f41a1a3802bf1c7ef41c1c380f00dfd5'
        '03010000f41e1e38'),
    'v_rcp_f16': bytes.fromhex(
        '00a9187ef41818380d00d4d501010000f41a1a3802a91c7ef41c1c380f00d4d5'
        '03010000f41e1e38'),
    'v_rsq_f16': bytes.fromhex(
        '00ad187ef41818380d00d6d501010000f41a1a3802ad1c7ef41c1c380f00d6d5'
        '03010000f41e1e38'),
    'v_sqrt_f16': bytes.fromhex(
        '00ab187ef41818380d00d5d501010000f41a1a3802ab1c7ef41c1c380f00d5d5'
        '03010000f41e1e38'),
    'v_sin_f16': bytes.fromhex(
        '00c1187ef41818380d00e0d501010000f41a1a3802c11c7ef41c1c380f00e0d5'
        '03010000f41e1e38'),
    'v_cos_f16': bytes.fromhex(
        '00c3187ef41818380d00e1d501010000f41a1a3802c31c7ef41c1c380f00e1d5'
        '03010000f41e1e38'),
    'v_exp_f16': bytes.fromhex(
        '00b1187ef41818380d00d8d501010000f41a1a3802b11c7ef41c1c380f00d8d5'
        '03010000f41e1e38'),
    'v_log_f16': bytes.fromhex(
        '00af187ef41818380d00d7d501010000f41a1a3802af1c7ef41c1c380f00d7d5'
        '03010000f41e1e38'),
}


def body_source(op):
    return ''.join('%s%s v%d, v%d\nv_or_b32 v%d, 0x%x, v%d\n' % (op, '_e64' if lane % 2 else '', 12 + lane, lane,
                                                                  12 + lane, MARK, 12 + lane) for lane in range(4))


def chunks(op):
    words = source_words(op)
    return [words[i:i + 4] for i in range(0, len(words), 4)]


def translate(op, chunk):
    sets = [(w, 0, 0) for w in chunk]
    pipe = G.Pipeline(B.vs_words(sets), B.PRE_WORDS + BODY_WORDS[op] + B.POST_WORDS, root_blob())
    return pipe


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


def far_from_tie(op, word):
    """True when the float64 result is not within a binary32 error of a binary16 rounding tie."""
    x = half(word & 0xffff)
    exact = float(REF[op](x))
    rounded = half(bits(exact))
    if exact == rounded:
        return True
    below, above = sorted((rounded, half(bits(exact) + (1 if (exact > rounded) == (rounded > 0) else -1))))
    return abs(exact - (below + above) / 2) > 1e-5 * abs(exact)


def controls():
    for op in BODY_WORDS:
        words = source_words(op)
        check('%s: %d inputs are finite binary16 and far from a rounding tie' % (op, len(words)),
              len(words) == 12 and all(math.isfinite(half(w & 0xffff)) and far_from_tie(op, w) for w in words))
        check('%s: the reference results are finite' % op,
              all(math.isfinite(half(expected_word(op, w) & 0xffff)) for w in words))
        for chunk in chunks(op):
            text = translate(op, chunk).hlsl_ps()
            check('%s: the pixel stage translates and rounds to binary16' % op, 'f32tof16' in text)
    # Deliberate deviation from the guide formula (it gives 1.0 here); this asserts the lowering choice, not hardware.
    check('fract stops below one', expected_word('v_fract_f16', GARBAGE | bits(-0.0001)) == MARK | bits(FRACT_MAX))


def check_binary(llvm_mc):
    for op in BODY_WORDS:
        built = assemble(llvm_mc, body_source(op))
        if built != BODY_WORDS[op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op].hex()))
    for chunk in (c for op in BODY_WORDS for c in chunks(op)):
        sets = [(w, 0, 0) for w in chunk]
        if assemble(llvm_mc, B.vs_source(sets)) != B.vs_words(sets):
            raise AssertionError('vertex fixture for %s differs from llvm-mc' % (chunk,))
    for name, source, words in (('prologue', B.PRE_SOURCE, B.PRE_WORDS), ('export', B.POST_SOURCE, B.POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op in BODY_WORDS:
                for index, chunk in enumerate(chunks(op)):
                    pipe = translate(op, chunk)
                    name = '%s_%d_%s' % (op, index, kind)
                    vs = compile_hlsl(compiler, kind, pipe.hlsl_vs(), 'vs_' + name, vs_profile, directory)
                    ps = compile_hlsl(compiler, kind, pipe.hlsl_ps(), 'ps_' + name, ps_profile, directory)
                    expected = words_text(expected_word(op, w) for w in chunk)
                    run = draw(renderer, adapter, vs, ps, expected)
                    if run.returncode:
                        raise AssertionError('%s %s inputs %s\n%s\n%s' % (kind, op, [hex(w) for w in chunk],
                                                                          run.stdout, run.stderr))
                    print('%s %s %s: %s' % (kind, op, [INPUTS[op][4 * index + i] for i in range(4)], expected),
                          flush=True)


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
