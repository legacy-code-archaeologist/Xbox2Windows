"""Packed VOP3P math in tools/startup/gfx2hlsl.py drawn on a desktop D3D12 adapter (Part of #1723).

usage: startup_gfx2hlsl_vop3p_draw_check.py draw <dxc> <offscreen_check> <warp|hardware> [fxc]

The pixel fixture reads three flat parameters (v0..v3, v4..v7, v8..v11), applies one packed operation to each of the
four lanes and exports v12..v15 to mrt0 (the layout of startup_gfx2hlsl_bit_alu_check.py). The vertex stage places the
inputs in the parameters, so the pixel stage cannot fold them. The translated text is compiled with FXC ps_5_1 when
given and with DXC ps_6_0, one pixel per four input sets, and the four words read back are compared with a reference
written from the ISA description with plain integer arithmetic and the struct half-float codec, not with the
lowering's own expressions. Float inputs are chosen so every result is exactly representable in binary16.
"""
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import POST_WORDS, PRE_WORDS, vs_words  # noqa: E402

MASK = 0xffffffff
OPCODES = {
    'v_pk_add_i16': 0x02, 'v_pk_sub_i16': 0x03, 'v_pk_max_i16': 0x07, 'v_pk_min_i16': 0x08,
    'v_pk_add_u16': 0x0a, 'v_pk_sub_u16': 0x0b, 'v_pk_max_u16': 0x0c, 'v_pk_min_u16': 0x0d,
    'v_pk_fma_f16': 0x0e, 'v_pk_add_f16': 0x0f, 'v_pk_mul_f16': 0x10, 'v_pk_min_f16': 0x11,
    'v_pk_max_f16': 0x12, 'v_dot2_f32_f16': 0x13, 'v_dot4_u32_u8': 0x17,
}


def encode(name, vdst, srcs, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1), neg_lo=(0, 0, 0), neg_hi=(0, 0, 0)):
    """VOP3P words from the layout tools/disasm_rdna2.py reads (see startup_gfx2hlsl_vop3p_check.py)."""
    w0 = (0b110011000 << 23) | (OPCODES[name] << 16) | (op_sel_hi[2] << 14) | vdst
    w0 |= sum(bit << (11 + i) for i, bit in enumerate(op_sel)) | sum(bit << (8 + i) for i, bit in enumerate(neg_hi))
    w1 = sum((s & 0x1FF) << (9 * i) for i, s in enumerate(srcs))
    w1 |= (op_sel_hi[0] << 27) | (op_sel_hi[1] << 28) | sum(bit << (29 + i) for i, bit in enumerate(neg_lo))
    return struct.pack('<II', w0, w1)


def h(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def f(half):
    return struct.unpack('<e', struct.pack('<H', half))[0]


def pair(lo, hi):
    return h(lo) | h(hi) << 16


def halves(word):
    return f(word & 0xffff), f(word >> 16)


def s16(x):
    return x - 0x10000 if x & 0x8000 else x


def sel(word, high):
    return word >> 16 if high else word & 0xffff


def f16_lanes(op, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1), neg_lo=(0, 0, 0), neg_hi=(0, 0, 0)):
    """Reference for a 2-lane half-float operation: lane 0 reads op_sel, lane 1 reads op_sel_hi, neg_lo/neg_hi."""
    def ref(*words):
        out = 0
        for lane in range(2):
            sels = op_sel if lane == 0 else op_sel_hi
            neg = neg_lo if lane == 0 else neg_hi
            args = [-f(sel(w, sels[i])) if neg[i] else f(sel(w, sels[i])) for i, w in enumerate(words)]
            out |= h(op(*args)) << (16 * lane)
        return out
    return ref


def int_lanes(op, signed):
    def ref(a, b):
        out = 0
        for lane in (0, 16):
            x, y = (a >> lane) & 0xffff, (b >> lane) & 0xffff
            if signed:
                x, y = s16(x), s16(y)
            out |= (op(x, y) & 0xffff) << lane
        return out
    return ref


def dot2(a, b, c):
    (a0, a1), (b0, b1) = halves(a), halves(b)
    return struct.unpack('<I', struct.pack('<f', a0 * b0 + a1 * b1 + struct.unpack('<f', struct.pack('<I', c))[0]))[0]


def dot4(a, b, c):
    return (sum(((a >> s) & 0xff) * ((b >> s) & 0xff) for s in (0, 8, 16, 24)) + c) & MASK


F16_A = [pair(1.0, 2.0), pair(-3.0, 0.5), pair(4.0, -8.0), pair(0.25, 1.5)]
F16_B = [pair(2.0, -1.0), pair(1.0, 0.5), pair(-0.5, 2.0), pair(3.0, 3.0)]
F16_C = [pair(0.5, 1.0), pair(2.0, -2.0), pair(1.0, 0.25), pair(-1.0, 4.0)]
INT_A = [0x00010001, 0x7fff8000, 0x8000ffff, 0x12345678]
INT_B = [0x00020003, 0x00018000, 0x7fff0001, 0xfffe0002]
INT_C = [0, MASK, 0x80000000, 0x00ff00ff]
# a * b + c = 3075.0000..1 lands near a binary16 midpoint: one rounding gives 0x6a01, binary32 first gives 0x6a02.
F16_MID_A, F16_MID_B, F16_MID_C = [0x6a006a00] * 4, [0x3c013c01] * 4, [0x84008400] * 4
BYTES_A = [0x01020304, 0xffffffff, 0x80ff017f, 0]
BYTES_B = [0x04030201, 0xffffffff, 0x02020202, 0xffffffff]
BYTES_C = [0, 5, MASK, 0x12345678]

# label -> (mnemonic, modifiers, arity, input sets, reference)
CASES = {
    'pk_add_f16': ('v_pk_add_f16', {}, 2, (F16_A, F16_B), f16_lanes(lambda a, b: a + b)),
    'pk_mul_f16': ('v_pk_mul_f16', {}, 2, (F16_A, F16_B), f16_lanes(lambda a, b: a * b)),
    'pk_fma_f16': ('v_pk_fma_f16', {}, 3, (F16_A, F16_B, F16_C), f16_lanes(lambda a, b, c: a * b + c)),
    'pk_fma_f16_midpoint': ('v_pk_fma_f16', {}, 3, (F16_MID_A, F16_MID_B, F16_MID_C), lambda a, b, c: 0x6a016a01),
    'pk_min_f16': ('v_pk_min_f16', {}, 2, (F16_A, F16_B), f16_lanes(min)),
    'pk_max_f16': ('v_pk_max_f16', {}, 2, (F16_A, F16_B), f16_lanes(max)),
    'pk_mul_f16_op_sel': ('v_pk_mul_f16', {'op_sel': (1, 0, 0), 'op_sel_hi': (0, 1, 1)}, 2, (F16_A, F16_B),
                          f16_lanes(lambda a, b: a * b, (1, 0, 0), (0, 1, 1))),
    'pk_add_f16_neg': ('v_pk_add_f16', {'neg_lo': (0, 1, 0), 'neg_hi': (1, 0, 0)}, 2, (F16_A, F16_B),
                       f16_lanes(lambda a, b: a + b, neg_lo=(0, 1, 0), neg_hi=(1, 0, 0))),
    'dot2_f32_f16': ('v_dot2_f32_f16', {}, 3, (F16_A, F16_B, F16_C), dot2),
    'dot4_u32_u8': ('v_dot4_u32_u8', {}, 3, (BYTES_A, BYTES_B, BYTES_C), dot4),
}
for _sign, _suffix in ((True, 'i16'), (False, 'u16')):
    for _op, _fn in (('add', lambda a, b: a + b), ('sub', lambda a, b: a - b), ('max', max), ('min', min)):
        CASES['pk_%s_%s' % (_op, _suffix)] = ('v_pk_%s_%s' % (_op, _suffix), {}, 2, (INT_A, INT_B),
                                              int_lanes(_fn, _sign))


def ps_words(name, mods, arity):
    body = b''.join(encode(name, 12 + lane, [0x100 + 4 * s + lane for s in range(arity)], **mods) for lane in range(4))
    return PRE_WORDS + body + POST_WORDS


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for label, (name, mods, arity, inputs, reference) in CASES.items():
                words = ps_words(name, mods, arity)
                text = G.Pipeline(vs_words([(0, 0, 0)] * 4), words, root_blob()).hlsl_ps()
                ps = compile_hlsl(compiler, kind, text, 'ps_%s_%s' % (label, kind), ps_profile, directory)
                sets = [tuple(inputs[s][i] if s < arity else 0 for s in range(3)) for i in range(4)]
                vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(sets), words, root_blob()).hlsl_vs(),
                                  'vs_%s_%s' % (label, kind), vs_profile, directory)
                expected = ','.join('%08x' % reference(*x[:arity]) for x in sets)
                run = draw(renderer, adapter, vs, ps, expected)
                if run.returncode:
                    raise AssertionError('%s %s inputs %s\n%s\n%s' % (kind, label, sets, run.stdout, run.stderr))
                print('%s %s: %s' % (kind, label, expected), flush=True)
    print('startup_gfx2hlsl_vop3p_draw: ok')


if __name__ == '__main__':
    if len(sys.argv) < 5 or sys.argv[1] != 'draw':
        raise SystemExit(__doc__)
    check_draw(*sys.argv[2:])
