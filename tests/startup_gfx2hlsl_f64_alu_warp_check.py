"""D3D12 draw check for the 64-bit float ALU slice in tools/startup/gfx2hlsl.py: v_add_f64, v_mul_f64,
v_fma_f64, v_cvt_f32_f64 and v_cvt_f64_f32 (Part of #1723).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture bodies and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per input chunk and compare the four result words

The fixture reuses the flat-parameter pixel prologue of startup_gfx2hlsl_bit_alu_check.py: twelve words arrive in
v0..v11 (six double pairs) and v12..v15 are exported. A pair operation runs twice per pixel (result pairs v[12:13]
and v[14:15]); v_cvt_f32_f64 runs on four pairs; v_cvt_f64_f32 on two words.

Every expected word comes from a reference written with Python's IEEE doubles, exact rational arithmetic for the
fused multiply-add and struct for the narrowing conversion, not from the lowering. Each lowering has a mutation
(swapped result words, negated narrowing result) whose draw must read the mutated words, so a draw that cannot see
the difference fails the check.
"""
import re
import struct
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (POST_SOURCE, POST_WORDS, PRE_SOURCE, PRE_WORDS,  # noqa: E402
                                            vs_source, vs_words)

MASK = 0xffffffff


def bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def pair_words(x):
    value = bits(x)
    return [value & MASK, value >> 32]


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def ref_fma(a, b, c):
    return float(Fraction(a) * Fraction(b) + Fraction(c))


# mnemonic -> (source count, results per pixel, body source, body words, reference over doubles)
BODY_SOURCE = {
    'v_add_f64': 'v_add_f64 v[12:13], v[0:1], v[2:3]\nv_add_f64 v[14:15], v[4:5], v[6:7]\n',
    'v_mul_f64': 'v_mul_f64 v[12:13], v[0:1], v[2:3]\nv_mul_f64 v[14:15], v[4:5], v[6:7]\n',
    'v_fma_f64': 'v_fma_f64 v[12:13], v[0:1], v[2:3], v[4:5]\nv_fma_f64 v[14:15], v[6:7], v[8:9], v[10:11]\n',
    'v_cvt_f32_f64': ''.join('v_cvt_f32_f64 v%d, v[%d:%d]\n' % (12 + i, 2 * i, 2 * i + 1) for i in range(4)),
    'v_cvt_f64_f32': 'v_cvt_f64_f32 v[12:13], v0\nv_cvt_f64_f32 v[14:15], v1\n',
}
BODY_WORDS = {
    'v_add_f64': bytes.fromhex('0c0064d5000502000e0064d5040d0200'),
    'v_mul_f64': bytes.fromhex('0c0065d5000502000e0065d5040d0200'),
    'v_fma_f64': bytes.fromhex('0c004cd5000512040e004cd506112a04'),
    'v_cvt_f32_f64': bytes.fromhex('001f187e021f1a7e041f1c7e061f1e7e'),
    'v_cvt_f64_f32': bytes.fromhex('0021187e01211c7e'),
}
ARITY = {'v_add_f64': 2, 'v_mul_f64': 2, 'v_fma_f64': 3}
REFERENCE = {
    'v_add_f64': lambda a, b: a + b,
    'v_mul_f64': lambda a, b: a * b,
    'v_fma_f64': ref_fma,
}
TWO = 2.0 ** -30
# Each entry is one pixel's worth of inputs. Pair operations: two (a, b[, c]) sets per pixel.
# The values need more than 24 mantissa bits, mix signs, and cancel so that a float or an unfused result differs.
PAIR_DRAW = {
    'v_add_f64': [((1e16, 1.0), (0.1, 0.2)), ((-3.5, 3.5000000000000004), (1.5, -1e-300)),
                  ((123456789.123456789, 0.000000001), (-0.0, 0.0))],
    'v_mul_f64': [((1.1, 1.1), (3.0, 0.3333333333333333)), ((-2.5e150, 4.0e-150), (1e200, -1.5)),
                  ((1.0 + TWO, 1.0 - TWO), (123456789.0, 987654321.0))],
    'v_fma_f64': [((1.0 + TWO, 1.0 - TWO, -1.0), (0.1, 10.0, -1.0)),
                  ((1e8 + 0.5, 1e8 - 0.5, -1e16), (-2.0, 3.0, 6.0)),
                  ((1.1, 1.1, 0.0), (1e300, 10.0, -1e301))],
}
NARROW_DRAW = [(1.0, 0.1, -3.141592653589793, 1e10), (16777217.0, 1.0 + 2.0 ** -24, -1e-30, 65504.5),
               (0.0, -0.0, 1.0000001192092896, 3.4e38)]
WIDEN_DRAW = [(0x3f800000, 0xbf000000), (0x3dcccccd, 0x4b800001), (0x00800000, 0x7f7fffff)]


def chunks(op):
    """(12 parameter words, 4 expected result words) per pixel."""
    out = []
    if op in PAIR_DRAW:
        reference = REFERENCE[op]
        for sets in PAIR_DRAW[op]:
            words, expected = [], []
            for values in sets:
                for value in values:
                    words += pair_words(value)
                expected += pair_words(reference(*values))
            out.append((words + [0] * (12 - len(words)), expected))
    elif op == 'v_cvt_f32_f64':
        for values in NARROW_DRAW:
            out.append(([w for v in values for w in pair_words(v)] + [0] * 4, [f32_bits(v) for v in values]))
    else:
        for pair in WIDEN_DRAW:
            expected = [w for word in pair for w in pair_words(struct.unpack('<f', struct.pack('<I', word))[0])]
            out.append((list(pair) + [0] * 10, expected))
    return out


def mutate_text(op, text):
    if op == 'v_cvt_f32_f64':
        mutated = text.replace('asuint((float)', 'asuint(-(float)')
    else:
        mutated = re.sub(r'(v\d+) = (f64lo\d+);\s*(v\d+) = (f64hi\d+);', r'\1 = \4; \3 = \2;', text)
    assert mutated != text, op
    return mutated


def mutated_words(op, expected):
    if op == 'v_cvt_f32_f64':
        return [w ^ 0x80000000 for w in expected]
    return [w for i in range(0, 4, 2) for w in (expected[i + 1], expected[i])]


def sets_for(words):
    """The four (src0, src1, src2) tuples vs_words() turns into parameter words 0..11 in order."""
    return [(words[lane], words[4 + lane], words[8 + lane]) for lane in range(4)]


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


def translate(op, mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob())
    text = pipe.hlsl_ps()
    return pipe, mutate_text(op, text) if mutate else text


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


def controls():
    for op in BODY_SOURCE:
        text = translate(op)[1]
        want = {'v_add_f64': 'asdouble(v0, v1) + asdouble(v2, v3)',
                'v_mul_f64': 'asdouble(v0, v1) * asdouble(v2, v3)',
                'v_fma_f64': 'fma(asdouble(v0, v1), asdouble(v2, v3), asdouble(v4, v5))',
                'v_cvt_f32_f64': 'asuint((float)asdouble(v0, v1))',
                'v_cvt_f64_f32': 'double('}[op]
        check('%s translates to %s' % (op, want), want in text)
        words = [w for pixel in chunks(op) for w in pixel[0]]
        check('%s fixture inputs are 12 words per pixel' % op, len(words) == 12 * len(chunks(op)))
        check('%s mutation changes the translation' % op, translate(op, True)[1] != text)
    # The references must separate the lowering from float and unfused arithmetic.
    check('fma reference is fused', ref_fma(1.0 + TWO, 1.0 - TWO, -1.0) != (1.0 + TWO) * (1.0 - TWO) - 1.0)
    for op in BODY_SOURCE:
        for words, expected in chunks(op):
            check('%s mutation reads other words' % op, mutated_words(op, expected) != expected)


def check_binary(llvm_mc):
    for op, source in BODY_SOURCE.items():
        built = assemble(llvm_mc, source)
        if built != BODY_WORDS[op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op].hex()))
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for op in BODY_SOURCE:
        for words, _ in chunks(op):
            sets = sets_for(words)
            if assemble(llvm_mc, vs_source(sets)) != vs_words(sets):
                raise AssertionError('vertex fixture for %s differs from llvm-mc' % (sets,))
    print('fixture words match llvm-mc')


def words_text(values):
    return ','.join('%08x' % v for v in values)


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op in BODY_SOURCE:
                good_ps = compile_hlsl(compiler, kind, translate(op)[1], 'ps_%s_%s' % (op, kind), ps_profile, directory)
                bad_ps = compile_hlsl(compiler, kind, translate(op, True)[1], 'mutated_%s_%s' % (op, kind),
                                      ps_profile, directory)
                for index, (words, expected) in enumerate(chunks(op)):
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(sets_for(words)), ps_words(op),
                                                                 root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    want = words_text(expected)
                    run = draw(renderer, adapter, vs, good_ps, want)
                    if run.returncode:
                        raise AssertionError('%s %s pixel %d expected %s\n%s\n%s' % (kind, op, index, want, run.stdout,
                                                                                    run.stderr))
                    print('%s %s pixel %d: %s' % (kind, op, index, want), flush=True)
                    # The same draw with the mutated lowering must read the mutated words, not the reference.
                    wrong = words_text(mutated_words(op, expected))
                    run = draw(renderer, adapter, vs, bad_ps, want)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s %s mutation did not read %s\n%s\n%s' % (kind, op, wrong, run.stdout,
                                                                                         run.stderr))
                    print('%s %s mutation: fails as expected, reads %s' % (kind, op, wrong), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        print('gfx2hlsl f64 alu draw fixture: controls passed')
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
