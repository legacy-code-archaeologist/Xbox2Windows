"""v_interp_p1ll_f16 / v_interp_p2_f16 in tools/startup/gfx2hlsl.py (pixel stage, issue #1996 item C).

  (no arguments)                       translation controls and mutations, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  compile <dxc>                        compile the translated pair with DXC vs_6_0 / ps_6_0 (no rendering)
  draw <dxc> <offscreen_check> <warp|hardware>
                                       draw one pixel and compare the four result words (Windows, D3D12)

Semantics (LLVM 15 AMDGPU, VOP3Instructions.td and IntrinsicsAMDGPU.td; Mesa aco emit_interp_instr):
  v_interp_p1ll_f16 vdst, vI, attrN.c [high]   vdst(f32) = P0 + I * P10 of the binary16 half `high` of channel c
  v_interp_p2_f16   vdst, vJ, attrN.c, vP1 [high]   vdst(f16) = P1 + J * P20 (same attribute half)
The pair gives the interpolated binary16 value. The vertex stage exports it with `exp paramN ... compr`: two binary16
values per dword, channel x = (v6.lo, v6.hi), channel y = (v7.lo, v7.hi). gfx2hlsl splits the pair into four float
channels, so the half h of channel c is float channel 2 * c + h.

The fixture is a source-built gfx1013 vertex and pixel pair. The vertex stage places a full-screen triangle and exports
the constant binary16 values 1.5, -2.0 (channel x), 0.5 and 3.0 (channel y) to param0; the pixel stage interpolates
the four halves, converts each to f32 and exports them to mrt0. The reference is written from the ISA layout above
with struct ('e' format) conversions, not from the lowering: the translated text is evaluated on the CPU by a small
interpreter of exactly the statements the fixture produces. Negative controls: the lowering removed (the pair is
refused), the high bit inverted on both instructions (the evaluated words differ from the reference), and every
modifier or form the fixture does not cover (refused),
including every op_sel bit (ACO's v_interp_p2_hi_f16 writes the high half of the destination).

The binary mode compares with llvm-mc after clearing the unused src2 field of v_interp_p1ll_f16: LLVM 15 encodes it
as 0, newer llvm-mc encodes the inline constant 0 (128), and the words below are the LLVM 15 ones.
"""
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
from startup_flat_param_check import VS_WORDS as F32_VS_WORDS, assemble, compile_hlsl, draw, root_blob  # noqa: E402

VS_SOURCE = """\
v_mov_b32 v3, 0x40400000
v_mov_b32 v10, 0xc0400000
v_cmp_eq_u32 vcc_lo, 1, v0
v_cndmask_b32 v1, -1.0, v3, vcc_lo
v_cmp_eq_u32 vcc_lo, 2, v0
v_cndmask_b32 v2, 1.0, v10, vcc_lo
v_mov_b32 v4, 1.0
v_mov_b32 v5, 0
v_mov_b32 v6, 0xc0003e00
v_mov_b32 v7, 0x42003800
exp pos0 v1, v2, v5, v4 done
exp param0 v6, v6, v7, v7 compr
s_endpgm
"""
VS_WORDS = bytes.fromhex(
    'ff02067e00004040ff02147e000040c08100847df30602028200847df2140402f202087e80020a7eff020c7e003e00c0'
    'ff020e7e00380042cf0800f8010205040f0600f806070000000081bf')
PS_SOURCE = """\
s_mov_b32 m0, s2
v_interp_p1ll_f16 v4, v0, attr0.x
v_interp_p2_f16 v4, v1, attr0.x, v4
v_interp_p1ll_f16 v5, v0, attr0.x high
v_interp_p2_f16 v5, v1, attr0.x, v5 high
v_interp_p1ll_f16 v6, v0, attr0.y
v_interp_p2_f16 v6, v1, attr0.y, v6
v_interp_p1ll_f16 v7, v0, attr0.y high
v_interp_p2_f16 v7, v1, attr0.y, v7 high
v_cvt_f32_f16 v4, v4
v_cvt_f32_f16 v5, v5
v_cvt_f32_f16 v6, v6
v_cvt_f32_f16 v7, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
PS_WORDS = bytes.fromhex(
    '0203fcbe040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04'
    '070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf')
# Variants of PS_SOURCE for the barycentric, omitted-result and upper-half refusals. The words are llvm-mc output
# (src2 of p1ll cleared as for PS_WORDS), verified by `binary`; the sources are in PS_VARIANT_SOURCES.
V_BODY = PS_SOURCE.split('\n', 1)[1]
V_FIRST = 'v_interp_p1ll_f16 v4, v0, attr0.x\nv_interp_p2_f16 v4, v1, attr0.x, v4\n'
V_TAIL = 'exp mrt0 v4, v4, v4, v4 done vm\ns_endpgm\n'
PS_VARIANT_SOURCES = {
    'I clobbered': 's_mov_b32 m0, s2\nv_mov_b32 v0, 1.0\n' + V_BODY,
    'J clobbered': 's_mov_b32 m0, s2\nv_mov_b32 v1, 1.0\n' + V_BODY,
    'I and J swapped': ('s_mov_b32 m0, s2\n' + V_BODY).replace('v0,', 'vI,').replace('v1,', 'v0,').replace('vI,', 'v1,'),
    'p1ll result read between the pair':
        's_mov_b32 m0, s2\nv_interp_p1ll_f16 v4, v0, attr0.x\nv_mov_b32 v8, v4\n'
        'v_interp_p2_f16 v4, v1, attr0.x, v4\n' + V_BODY.split('\n', 2)[2],
    'p1ll result read after p2 wrote elsewhere':
        's_mov_b32 m0, s2\nv_interp_p1ll_f16 v4, v0, attr0.x\nv_interp_p2_f16 v8, v1, attr0.x, v4\n'
        'v_cvt_f32_f16 v8, v8\nv_mov_b32 v9, v4\nexp mrt0 v8, v9, v9, v9 done vm\ns_endpgm\n',
    'upper half read by a shift': 's_mov_b32 m0, s2\n' + V_FIRST + 'v_lshrrev_b32 v8, 16, v4\n'
        'v_cvt_f32_f16 v4, v4\nexp mrt0 v4, v8, v8, v8 done vm\ns_endpgm\n',
    'upper half exported': 's_mov_b32 m0, s2\n' + V_FIRST + V_TAIL,
    'branch into p2': 's_mov_b32 m0, s2\ns_cbranch_vccz L0\nv_interp_p1ll_f16 v4, v0, attr0.x\nL0:\n'
        'v_interp_p2_f16 v4, v1, attr0.x, v4\nv_cvt_f32_f16 v4, v4\n' + V_TAIL,
    'unrelated read of v2': 's_mov_b32 m0, s2\nv_mov_b32 v8, v2\n' + V_BODY,
    'unrelated write of v9': 's_mov_b32 m0, s2\nv_mov_b32 v9, 1.0\n' + V_BODY,
}
PS_VARIANTS = {name: bytes.fromhex(h) for name, h in (
    ('I clobbered', '0203fcbef202007e040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('J clobbered', '0203fcbef202027e040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('I and J swapped', '0203fcbe040042d70002020004005ad700001204050042d70003020005005ad700011604060042d74002020006005ad740001a04070042d74003020007005ad740011e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('p1ll result read between the pair', '0203fcbe040042d7000002000403107e04005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('p1ll result read after p2 wrote elsewhere', '0203fcbe040042d70000020008005ad7000212040817107e0403127e0f1800f808090909000081bf'),
    ('upper half read by a shift', '0203fcbe040042d70000020004005ad7000212049008102c0417087e0f1800f804080808000081bf'),
    ('upper half exported', '0203fcbe040042d70000020004005ad7000212040f1800f804040404000081bf'),
    ('branch into p2', '0203fcbe020086bf040042d70000020004005ad7000212040417087e0f1800f804040404000081bf'),
    ('unrelated read of v2', '0203fcbe0203107e040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('unrelated write of v9', '0203fcbef202127e040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
)}
P1LL_SRC2_MASK = 0x07fc0000  # w1 bits 26:18, the unused third source

# The reference, from the ISA layout: two binary16 values per dword of the compr export.
EXPORT = (0xc0003e00, 0x42003800)  # v6, v7 of VS_SOURCE
HALVES = ((1.5, -2.0), (0.5, 3.0))


def f16_bits(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def reference():
    """The four words the pixel stage exports: half h of channel c of the attribute, as f32."""
    for c in range(2):
        assert EXPORT[c] == f16_bits(HALVES[c][0]) | f16_bits(HALVES[c][1]) << 16
    words = []
    for c, h in ((0, 0), (0, 1), (1, 0), (1, 1)):
        half = EXPORT[c] >> (16 * h) & 0xffff
        words.append(f32_bits(struct.unpack('<e', struct.pack('<H', half))[0]))
    return tuple(words)


def f16_to_f32_bits(h):
    return f32_bits(struct.unpack('<e', struct.pack('<H', h))[0])


def evaluate(vs, ps, channels=None):
    """Evaluate the translated text: the vertex compr export, then the pixel conversions. Only the statements
    the fixture produces are understood; anything else fails the check. `channels` replaces the vertex export with
    four interpolated f32 words (what the rasterizer hands the pixel stage)."""
    if channels is None:
        regs = {'v6': EXPORT[0], 'v7': EXPORT[1]}
        m = re.search(r'o\.param0 = float4\(([^;]*)\);\n', vs)
        channels = []
        for expr in re.findall(r'f16tof32\((v\d+)( >> 16)?\)', m.group(1)):
            channels.append(f16_to_f32_bits((regs[expr[0]] >> (16 if expr[1] else 0)) & 0xffff))
        assert len(channels) == 4, m.group(0)
    state = {}
    for line in ps.splitlines():
        stmt = re.fullmatch(r'    (v\d+) = \((v\d+) & 0xffff0000u\) \| f32tof16\(i\.param0\.([xyzw])\);', line)
        if stmt:
            assert stmt.group(1) == stmt.group(2), line
            # f32tof16 of the interpolated value; a constant triangle makes it the exported value
            f32 = channels['xyzw'.index(stmt.group(3))]
            state[stmt.group(1)] = f16_bits(struct.unpack('<f', struct.pack('<I', f32))[0])
            continue
        stmt = re.fullmatch(r'    (v\d+) = U\(f16tof32\((v\d+)\)\);', line)
        if stmt:
            state[stmt.group(1)] = f16_to_f32_bits(state[stmt.group(2)] & 0xffff)
    return tuple(state['v%d' % n] for n in (4, 5, 6, 7))


def translate(ps_words=PS_WORDS, vs_words=VS_WORDS, **kwargs):
    return G.Pipeline(vs_words, ps_words, root_blob(), **kwargs)


def patch(words, index, w0=0, w1=0):
    """Instruction `index` of PS_WORDS (after the 4-byte s_mov_b32): w0 / w1 are XORed into its two dwords."""
    out = bytearray(words)
    off = 4 + 8 * index
    a, b = struct.unpack_from('<II', out, off)
    struct.pack_into('<II', out, off, a ^ w0, b ^ w1)
    return bytes(out)


def refused(label, words, needle, vs_words=VS_WORDS):
    try:
        translate(words, vs_words)
    except G.Unsupported as error:
        assert needle in str(error), (label, str(error))
        print('refused: %s (%s)' % (label, error))
        return
    except Exception as error:
        raise AssertionError('%s crashed instead of being refused: %r' % (label, error)) from error
    raise AssertionError('%s was translated' % label)


def controls():
    expected = reference()
    assert expected == (0x3fc00000, 0xc0000000, 0x3f000000, 0x40400000), expected
    pipe = translate()
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    got = evaluate(vs, ps)
    assert got == expected, ('translated words', [hex(w) for w in got], [hex(w) for w in expected])
    print('translated: %s' % ','.join('%08x' % w for w in got))

    # The p2 lowering keeps the upper half of the destination with this mask; dropping it must fail the check.
    assert ps.count('& 0xffff0000u) | f32tof16(') == 4, ps
    # The whole-program scan runs once per translated program, not once per p1ll / p2 statement.
    calls = []
    scan = G.Stage.interp_f16_scan

    def counting(self):
        calls.append(self.interp_scanned)
        return scan(self)

    G.Stage.interp_f16_scan = counting
    try:
        translate()
    finally:
        G.Stage.interp_f16_scan = scan
    assert calls.count(False) == 1 and len(calls) == 8, calls
    print('scan: %d calls, 1 full run' % len(calls))

    # A decoded instruction must be the source line (LLVM's spelling).
    lines = PS_SOURCE.splitlines()
    for (_, size, name, asm), line in zip(G.decode(G.strip_header(PS_WORDS)), lines):
        assert asm.replace('_e32', '') == line, (asm, line)

    # Negative control 1: the lowering removed. The pair must be refused, not translated by some fallback.
    saved = {n: getattr(G.Stage, n) for n in ('op_v_interp_p1ll_f16', 'op_v_interp_p2_f16')}
    try:
        for n in saved:
            delattr(G.Stage, n)
        refused('lowering disabled', PS_WORDS, 'v_interp_p1ll_f16')
    finally:
        for n, f in saved.items():
            setattr(G.Stage, n, f)

    # Negative control 2: the high bit inverted on every instruction of the pair. The pair stays consistent, so it
    # is translated, but it reads the other half: the evaluated words differ from the reference.
    flipped = PS_WORDS
    for index in range(8):
        flipped = patch(flipped, index, w1=0x100)  # w1 bit 8: attribute half select
    pipe = translate(flipped)
    swapped = evaluate(pipe.hlsl_vs(), pipe.hlsl_ps())
    assert swapped == (expected[1], expected[0], expected[3], expected[2]), [hex(w) for w in swapped]
    assert swapped != expected
    print('negative control: high inverted reads %s' % ','.join('%08x' % w for w in swapped))
    # ... and inverting only one of the pair is refused.
    refused('p2 high differs from p1ll', patch(PS_WORDS, 1, w1=0x100), 'is not the v_interp_p1ll_f16 result')

    # What the fixture does not cover is refused.
    refused('p1ll clamp', patch(PS_WORDS, 0, w0=0x8000), 'clamp')
    refused('p1ll div:2', patch(PS_WORDS, 0, w1=3 << 27), 'div:2')
    refused('p1ll mul:2', patch(PS_WORDS, 0, w1=1 << 27), 'mul:2')
    for bit in range(11, 15):
        refused('p1ll op_sel bit %d' % (bit - 11), patch(PS_WORDS, 0, w0=1 << bit), 'op_sel')
        refused('p2 op_sel bit %d' % (bit - 11), patch(PS_WORDS, 1, w0=1 << bit), 'op_sel')
    refused('p2 clamp', patch(PS_WORDS, 1, w0=0x8000), 'clamp')
    refused('p2 abs on j', patch(PS_WORDS, 1, w0=0x200), 'plain VGPRs')
    refused('p2 neg on p1', patch(PS_WORDS, 1, w1=4 << 29), 'plain VGPRs')
    refused('p1ll abs on i', patch(PS_WORDS, 0, w0=0x200), 'plain VGPRs')
    # attr0.z: change channel bits 7:6 of the p1ll and p2 from x (0) to z (2)
    z = patch(patch(PS_WORDS, 0, w1=2 << 6), 1, w1=2 << 6)
    refused('attribute channel z', z, 'channel z')
    # An attribute spelled without a channel is refused, not an AttributeError (decode always prints a channel,
    # so the instruction goes straight to a Stage probe).
    for op, operands in (('v_interp_p1ll_f16', 'v5, v6, attr0'), ('v_interp_p2_f16', 'v5, v6, attr0, v4')):
        try:
            G.Stage.__new__(G.Stage).one(op, '%s %s' % (op, operands))
        except G.Unsupported as error:
            assert 'expected attrN.x/.y/.z/.w' in str(error), (op, str(error))
            print('refused: %s without a channel (%s)' % (op, error))
        else:
            raise AssertionError('%s without a channel was lowered' % op)
    # An attribute the vertex stage does not export (attr5; the vertex stage exports only param0): the whole pair
    # moves to attr5 (w1 bits 5:0), and the pipeline refuses it later, so interp_f16 needs no bound of its own.
    out_of_range = PS_WORDS
    for index in range(8):
        out_of_range = patch(out_of_range, index, w1=5)
    refused('attribute not exported by the vertex stage', out_of_range, 'attribute 5')
    # p2 whose src2 is not the p1ll result: v4 -> v8 in the third source
    other = patch(PS_WORDS, 1, w1=(0x104 ^ 0x108) << 18)
    refused('p2 source is not the p1ll result', other, 'is not the v_interp_p1ll_f16 result')
    # p1ll result overwritten before p2 reads it
    # (an overwrite between the pair: v_mov_b32 v4, v0 = 7e080300 inserted after the p1ll)
    clobbered = PS_WORDS[:12] + bytes.fromhex('0003087e') + PS_WORDS[12:]
    refused('p1ll result overwritten', clobbered, 'before the matching v_interp_p2_f16')
    # The attribute is a plain float export, not a compr pair.
    refused('attribute not exported with compr', PS_WORDS, 'compr', vs_words=F32_VS_WORDS)
    barycentric_controls(expected)
    print('controls ok')


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def barycentric_controls(expected):
    # 1. The operand must be the entry PERSP_CENTER pair, still holding the entry values.
    refused('I clobbered before use', PS_VARIANTS['I clobbered'], 'writes v0')
    refused('J clobbered before use', PS_VARIANTS['J clobbered'], 'writes v1')
    refused('I and J swapped', PS_VARIANTS['I and J swapped'], 'barycentric operand v1')
    # One operand on another register: p1ll reads v2 as I, p2 reads v0 as J (the VGPR is src1, w1 bits 17:9).
    for label, index, field in (('p1ll reads v2', 0, 2), ('p2 reads v0', 1, 0)):
        words = bytearray(PS_WORDS)
        off = 4 + 8 * index
        w1 = struct.unpack_from('<I', words, off + 4)[0]
        w1 = (w1 & ~(0x1ff << 9)) | ((256 + field) << 9)
        struct.pack_into('<I', words, off + 4, w1)
        refused(label, bytes(words), 'barycentric operand')
    # Entry layout: the pair follows the PERSP_SAMPLE pair when that one is enabled, and PERSP_CENTER must be enabled.
    translate(ps_hw_inputs=(2, 2))
    print('accepted: PERSP_CENTER alone (v0, v1)')
    try:
        translate(ps_hw_inputs=(3, 3))
    except G.Unsupported as error:
        assert 'barycentric operand v0' in str(error), str(error)
        print('refused: PERSP_CENTER at v2, v3 (%s)' % error)
    else:
        raise AssertionError('v0, v1 accepted although they hold PERSP_SAMPLE')
    try:
        translate(ps_hw_inputs=(1, 1))
    except G.Unsupported as error:
        assert 'PERSP_CENTER' in str(error), str(error)
        print('refused: PERSP_CENTER not enabled (%s)' % error)
    else:
        raise AssertionError('accepted without PERSP_CENTER')
    # Positive control, varying triangle: the three vertices carry different binary16 pairs and the pixel reads an
    # interior point. ISA arithmetic: p1ll gives P0 + I * (P1 - P0) in f32, p2 adds J * (P2 - P0) and rounds to
    # binary16. The lowering's input is the rasterizer's value of the same interpolation.
    triangle = ((1.5, -2.0, 0.5, 3.0), (2.5, 1.0, -1.5, 0.25), (-0.5, 4.0, 2.0, -3.0))
    for i, j in ((0.25, 0.5), (0.5, 0.25), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.125, 0.375)):
        isa = []
        for c in range(4):
            p0, p1, p2 = (v[c] for v in triangle)
            partial = f32(p0 + i * (p1 - p0))
            isa.append(f16_to_f32_bits(f16_bits(f32(partial + j * (p2 - p0)))))
        interpolated = [f32_bits(f32(p0 + i * (p1 - p0) + j * (p2 - p0)))
                        for p0, p1, p2 in zip(*triangle)]
        pipe = translate()
        got = evaluate(pipe.hlsl_vs(), pipe.hlsl_ps(), interpolated)
        assert got == tuple(isa), ((i, j), [hex(w) for w in got], [hex(w) for w in isa])
    assert tuple(isa) != expected
    print('varying triangle: translated words equal the ISA arithmetic at 6 interior and corner points')
    # Registers other than the pair may be read or written freely.
    for label in ('unrelated read of v2', 'unrelated write of v9'):
        pipe = translate(PS_VARIANTS[label])
        assert evaluate(pipe.hlsl_vs(), pipe.hlsl_ps()) == expected, label
        print('accepted: %s' % label)

    # 2. The omitted p1ll result and the upper half of the p2 result.
    refused('p1ll result read between the pair', PS_VARIANTS['p1ll result read between the pair'],
            'reads it before the matching v_interp_p2_f16')
    refused('p1ll result read after p2 wrote elsewhere', PS_VARIANTS['p1ll result read after p2 wrote elsewhere'],
            'after a v_interp_p2_f16 that wrote another register')
    refused('upper half read by a shift', PS_VARIANTS['upper half read by a shift'], 'upper 16 bits')
    refused('upper half exported', PS_VARIANTS['upper half exported'], 'upper 16 bits')
    refused('branch into p2 without the matching p1ll', PS_VARIANTS['branch into p2'], 'a branch between')
    pair_and_export = PS_VARIANTS['upper half exported']
    # s_cbranch_vccz skips the one-word full-width conversion and reaches the export.
    skip_conversion = pair_and_export[:20] + bytes.fromhex('010086bf0417087e') + pair_and_export[20:]
    refused('branch skips the full-width conversion', skip_conversion, 'result across a branch')


def normalise(words):
    """Clear the unused src2 field of v_interp_p1ll_f16, which newer llvm-mc encodes as 128 and LLVM 15 as 0."""
    out = bytearray(words)
    off = 0
    for _, size, name, _ in G.decode(G.strip_header(words)):
        if name.startswith('v_interp_p1ll_f16'):
            w1 = struct.unpack_from('<I', out, off + 4)[0] & ~P1LL_SRC2_MASK
            struct.pack_into('<I', out, off + 4, w1)
        off += size
    return bytes(out)


def check_binary(llvm_mc):
    for name, source, words in (('vertex', VS_SOURCE, VS_WORDS), ('pixel', PS_SOURCE, PS_WORDS)):
        built = normalise(assemble(llvm_mc, source))
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    for name, source in PS_VARIANT_SOURCES.items():
        built = normalise(assemble(llvm_mc, source))
        # a forward branch is an unresolved fixup in llvm-mc's listing (placeholder 0x0a): the offset is 2 dwords
        built = built.replace(bytes.fromhex('0a0a86bf'), bytes.fromhex('020086bf'))
        if built != PS_VARIANTS[name]:
            raise AssertionError('%s variant: assembled %s, recorded %s' % (name, built.hex(), PS_VARIANTS[name].hex()))
    print('fixture words match llvm-mc')


def pair():
    pipe = translate()
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def check_compile(dxc):
    vs, ps = pair()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        compile_hlsl(dxc, 'dxc', vs, 'vs', 'vs_6_0', directory)
        compile_hlsl(dxc, 'dxc', ps, 'ps', 'ps_6_0', directory)
    print('dxc vs_6_0 / ps_6_0: compiled')


def check_draw(dxc, renderer, adapter):
    expected = ','.join('%08x' % w for w in reference())
    vs, ps = pair()
    flipped_words = PS_WORDS
    for index in range(8):
        flipped_words = patch(flipped_words, index, w1=0x100)
    flipped = translate(flipped_words)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs_cso = compile_hlsl(dxc, 'dxc', vs, 'vs', 'vs_6_0', directory)
        ps_cso = compile_hlsl(dxc, 'dxc', ps, 'ps', 'ps_6_0', directory)
        run = draw(renderer, adapter, vs_cso, ps_cso, expected)
        if run.returncode:
            raise AssertionError('translated pair\n%s\n%s' % (run.stdout, run.stderr))
        print('translated pair: %s' % expected, flush=True)
        bad_ps = compile_hlsl(dxc, 'dxc', flipped.hlsl_ps(), 'ps_high', 'ps_6_0', directory)
        run = draw(renderer, adapter, vs_cso, bad_ps, expected)
        if run.returncode == 0:
            raise AssertionError('high inverted: control read the reference words')
        print('high inverted: fails as expected', flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'compile':
        controls()
        check_compile(*args)
    elif mode == 'draw':
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
