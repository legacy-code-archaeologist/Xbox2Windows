"""16-bit vector ALU lowering in tools/startup/gfx2hlsl.py, drawn on a real D3D12 adapter (Part of #1723).

  (no arguments)                       translation controls and reference self-checks, pure Python
  binary <llvm-mc>                     reassemble the fixture bodies and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair (FXC ps_5_1 when given, DXC ps_6_0), draw one pixel
                                       per four input sets and compare the four result words with the CPU reference

The pixel fixture is the one of startup_gfx2hlsl_bit_alu_check.py: three flat parameters (v0..v3, v4..v7, v8..v11),
one operation per lane, results in v12..v15 exported to mrt0. The vertex stage places the inputs in the parameters, so
the pixel stage cannot fold them. Every source word carries garbage in its upper half: the 16-bit operations read the
low half only and write a result whose upper half is zero.

The expected words come from an exact rational model of binary16 arithmetic (Fraction, then one round to binary16),
not from the lowering's expressions.
"""
import math
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
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    POST_WORDS, PRE_WORDS, vs_source, vs_words, words_text)

K = 0x4900  # literal operand of v_fmamk_f16 / v_fmaak_f16: 10.0


def h2f(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def f2h(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def exact(bits):
    return Fraction(h2f(bits))


def round_half(value):
    """Nearest binary16 (ties to even) of an exact rational; overflow gives infinity."""
    try:
        return f2h(float(value))
    except OverflowError:
        return 0xfc00 if value < 0 else 0x7c00


def trunc_clamp(bits, lo, hi):
    value = h2f(bits)
    if math.isnan(value):
        return 0
    return int(max(lo, min(hi, math.trunc(value)))) if not math.isinf(value) else (hi if value > 0 else lo)


def signed16(v):
    v &= 0xffff
    return v - 0x10000 if v & 0x8000 else v


def f_min(a, b):
    return a if exact(a) < exact(b) else b


def f_max(a, b):
    return a if exact(a) > exact(b) else b


# mnemonic -> (source count, VOP1 spelling, reference over (s0, s1, s2) words)
OPS = {
    'v_add_f16': (2, False, lambda a, b, c: round_half(exact(a) + exact(b))),
    'v_sub_f16': (2, False, lambda a, b, c: round_half(exact(a) - exact(b))),
    'v_subrev_f16': (2, False, lambda a, b, c: round_half(exact(b) - exact(a))),
    'v_mul_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b))),
    'v_min_f16': (2, False, lambda a, b, c: f_min(a & 0xffff, b & 0xffff)),
    'v_max_f16': (2, False, lambda a, b, c: f_max(a & 0xffff, b & 0xffff)),
    'v_fmac_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b) + exact(c))),
    'v_fmamk_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(K) + exact(b))),
    'v_fmaak_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b) + exact(K))),
    'v_cvt_f16_i16': (1, True, lambda a, b, c: round_half(Fraction(signed16(a)))),
    'v_cvt_f16_u16': (1, True, lambda a, b, c: round_half(Fraction(a & 0xffff))),
    'v_cvt_i16_f16': (1, True, lambda a, b, c: trunc_clamp(a, -32768, 32767) & 0xffff),
    'v_cvt_u16_f16': (1, True, lambda a, b, c: trunc_clamp(a, 0, 65535)),
}

G_ = {name: f2h(value) for name, value in (
    ('1', 1.0), ('2', 2.0), ('3', 3.0), ('half', 0.5), ('m1', -1.0), ('m2.5', -2.5), ('100', 100.0), ('0.1', 0.1),
    ('1000', 1000.0), ('m3', -3.0), ('7', 7.0), ('2.75', 2.75), ('m2.75', -2.75), ('60000', 60000.0),
    ('m60000', -60000.0), ('3.25', 3.25), ('300', 300.0), ('0', 0.0), ('1.5', 1.5), ('0.3', 0.3))}
HIGH = (0xabcd0000, 0x5a5a0000, 0xdead0000, 0x12340000)


def sets(pairs, accumulators=None):
    """Eight lanes of (s0, s1, s2) with garbage in the upper half of every word."""
    out = []
    for i, (a, b) in enumerate(pairs):
        c = accumulators[i] if accumulators else 0
        out.append((a | HIGH[i % 4], b | HIGH[(i + 1) % 4], c | HIGH[(i + 2) % 4]))
    return out


F = G_
ARITH = [(F['1'], F['2']), (F['3'], F['m1']), (F['0.1'], F['0.3']), (F['100'], F['1000']),
         (F['m2.5'], F['half']), (F['1.5'], F['7']), (F['300'], F['m3']), (F['3.25'], F['2.75'])]
ACC = [F['1'], F['m1'], F['half'], F['100'], F['2'], F['0'], F['3'], F['7']]
# (a, b, c) binary16 triples where rounding the binary32 sum and then binary16 differs from one rounding (#2073).
# Measured on AMD RX 9070 XT and WARP: the earlier f16rn(a*b + c) lowering was one binary16 unit low on each.
FMAC_DOUBLE_ROUNDING = [(0x4a54, 0x1ef4, 0x4e2d), (0x51ea, 0xbf3e, 0x1d5f), (0x2bd5, 0xbe83, 0x4528),
                        (0x3915, 0x387a, 0x4b71)]
INTS = [(0x0003, 0), (0xfffd, 0), (0x8403, 0), (0x7fff, 0), (0x8000, 0), (0, 0), (0xffff, 0), (0x00ff, 0)]
FLOATS = [(F['2.75'], 0), (F['m2.75'], 0), (F['60000'], 0), (F['m60000'], 0), (F['half'], 0), (F['m1'], 0),
          (F['100'], 0), (f2h(0.99951171875), 0)]
DRAW = {}
for _name, (_arity, _vop1, _ref) in OPS.items():
    if _name.startswith('v_cvt_f16'):
        DRAW[_name] = sets(INTS)
    elif _name.startswith('v_cvt_'):
        DRAW[_name] = sets(FLOATS)
    else:
        DRAW[_name] = sets(ARITH, ACC if _name == 'v_fmac_f16' else None)
# Four double-rounding lanes replace the first four fixture lanes of v_fmac_f16 (eight lanes stay two draws).
DRAW['v_fmac_f16'] = [tuple(w | HIGH[(i + k) % 4] for k, w in enumerate(t)) for i, t in
                      enumerate(FMAC_DOUBLE_ROUNDING)] + DRAW['v_fmac_f16'][4:]


def body_source(op):
    arity, vop1, _ = OPS[op]
    lines = []
    for lane in range(4):
        dst, a, b = 12 + lane, lane, 4 + lane
        if op == 'v_fmac_f16':
            lines.append('v_mov_b32 v%d, v%d\n' % (dst, 8 + lane))
        if op == 'v_fmamk_f16':
            lines.append('%s v%d, v%d, 0x%x, v%d\n' % (op, dst, a, K, b))
        elif op == 'v_fmaak_f16':
            lines.append('%s v%d, v%d, v%d, 0x%x\n' % (op, dst, a, b, K))
        elif vop1:
            lines.append('%s%s v%d, v%d\n' % (op, '_e64' if lane % 2 else '', dst, a))
        else:
            lines.append('%s v%d, v%d, v%d\n' % (op, dst, a, b))
    return ''.join(lines)


# Recorded llvm-mc (gfx1013) encodings of body_source(op); `binary <llvm-mc>` regenerates and compares them.
BODY_WORDS = {
    'v_add_f16': bytes.fromhex(
        '00091864010b1a64020d1c64030f1e64'),
    'v_sub_f16': bytes.fromhex(
        '00091866010b1a66020d1c66030f1e66'),
    'v_subrev_f16': bytes.fromhex(
        '00091868010b1a68020d1c68030f1e68'),
    'v_mul_f16': bytes.fromhex(
        '0009186a010b1a6a020d1c6a030f1e6a'),
    'v_min_f16': bytes.fromhex(
        '00091874010b1a74020d1c74030f1e74'),
    'v_max_f16': bytes.fromhex(
        '00091872010b1a72020d1c72030f1e72'),
    'v_fmac_f16': bytes.fromhex(
        '0803187e0009186c09031a7e010b1a6c0a031c7e020d1c6c0b031e7e030f1e6c'),
    'v_fmamk_f16': bytes.fromhex(
        '0009186e00490000010b1a6e00490000020d1c6e00490000030f1e6e00490000'),
    'v_fmaak_f16': bytes.fromhex(
        '0009187000490000010b1a7000490000020d1c7000490000030f1e7000490000'),
    'v_cvt_f16_i16': bytes.fromhex(
        '00a3187e0d00d1d50101000002a31c7e0f00d1d503010000'),
    'v_cvt_f16_u16': bytes.fromhex(
        '00a1187e0d00d0d50101000002a11c7e0f00d0d503010000'),
    'v_cvt_i16_f16': bytes.fromhex(
        '00a7187e0d00d3d50101000002a71c7e0f00d3d503010000'),
    'v_cvt_u16_f16': bytes.fromhex(
        '00a5187e0d00d2d50101000002a51c7e0f00d2d503010000'),
}


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


def reference(op, x):
    return OPS[op][2](*x) & 0xffffffff


def translate(op):
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob()).hlsl_ps()


# The target is R32G32B32A32_FLOAT, so a small integer result is a binary32 denormal and FXC drops it when it comes
# from a float to integer conversion. The integer-result operations therefore set bit 30 on the way out of the pixel
# stage and in the expected word; the translated statements themselves are unchanged.
INTEGER_RESULT = ('v_cvt_i16_f16', 'v_cvt_u16_f16')
TAG = 0x40000000


def draw_text(op):
    text = translate(op)
    return re.sub(r'F\(v(1[2-5])\)', r'F(v\1 | 0x%xu)' % TAG, text) if op in INTEGER_RESULT else text


def draw_word(op, x):
    return reference(op, x) | (TAG if op in INTEGER_RESULT else 0)


def controls():
    failures = []

    def check(label, ok):
        print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
        if not ok:
            failures.append(label)

    # The model itself, by hand: 0.1 + 0.3 in binary16, saturation, truncation toward zero, NaN reads 0.
    check('model: 1 + 2 = 3', OPS['v_add_f16'][2](F['1'], F['2'], 0) == F['3'])
    check('model: 60000 truncates to the i16 limit', OPS['v_cvt_i16_f16'][2](F['60000'], 0, 0) == 0x7fff)
    check('model: -60000 saturates to -32768', OPS['v_cvt_i16_f16'][2](F['m60000'], 0, 0) == 0x8000)
    check('model: a negative value converts to u16 zero', OPS['v_cvt_u16_f16'][2](F['m1'], 0, 0) == 0)
    check('model: NaN converts to zero', OPS['v_cvt_i16_f16'][2](0x7e00, 0, 0) == 0)
    check('model: 0xfffd is -3 as i16', OPS['v_cvt_f16_i16'][2](0xfffd, 0, 0) == F['m3'])
    for a, b, c, want in ((0x4a54, 0x1ef4, 0x4e2d, 0x4e33), (0x51ea, 0xbf3e, 0x1d5f, 0xd55b),
                          (0x2bd5, 0xbe83, 0x4528, 0x450f), (0x3915, 0x387a, 0x4b71, 0x4b9f)):
        check('model: fmac %04x*%04x+%04x is one rounding' % (a, b, c), OPS['v_fmac_f16'][2](a, b, c) == want)
    check('draw: fmac covers the double-rounding triples',
          all(tuple(w & 0xffff for w in x) in FMAC_DOUBLE_ROUNDING
                                            for x in DRAW['v_fmac_f16'][:4]))
    for op in OPS:
        check('%s: every draw reference has a zero upper half' % op,
              all(reference(op, x) >> 16 == 0 for x in DRAW[op]))
        check('%s: draw sets cover two draws of four lanes' % op, len(DRAW[op]) == 8)
    if BODY_WORDS:
        for op in OPS:
            try:
                text = translate(op)
            except G.Unsupported as e:
                check('%s: translates (%s)' % (op, e), False)
                continue
            check('%s: translates to four lane results' % op,
                  all(('v%d = ' % r) in text for r in range(12, 16)))
    if failures:
        raise SystemExit('FAILED: ' + ', '.join(failures))


def check_binary(llvm_mc):
    for op in OPS:
        built = assemble(llvm_mc, body_source(op))
        if built != BODY_WORDS[op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op].hex()))
    for op in OPS:
        for i in range(0, 8, 4):
            if assemble(llvm_mc, vs_source(DRAW[op][i:i + 4])) != vs_words(DRAW[op][i:i + 4]):
                raise AssertionError('vertex fixture for %s differs from llvm-mc' % op)
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, lanes in DRAW.items():
                ps = compile_hlsl(compiler, kind, draw_text(op), 'ps_%s_%s' % (op, kind), ps_profile, directory)
                for index in range(0, len(lanes), 4):
                    chunk = lanes[index:index + 4]
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), ps_words(op), root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    expected = words_text(draw_word(op, x) for x in chunk)
                    run = draw(renderer, adapter, vs, ps, expected)
                    if run.returncode:
                        failures.append('%s %s set %d' % (kind, op, index // 4))
                        print('%s %s set %d MISMATCH inputs %s\n%s' % (kind, op, index // 4, chunk, run.stderr.strip()),
                              flush=True)
                        continue
                    print('%s %s set %d: %s' % (kind, op, index // 4, expected), flush=True)
    if failures:
        raise AssertionError('draw mismatches: ' + ', '.join(failures))


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
