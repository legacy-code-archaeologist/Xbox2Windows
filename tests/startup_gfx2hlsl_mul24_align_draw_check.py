"""D3D12 draw of the translated 24-bit multiplies, v_xnor_b32 and the funnel shifts (see
startup_gfx2hlsl_mul24_align_check.py for the CPU evaluation of the same lowerings).

  binary <llvm-mc>                     reassemble the fixture bodies below and compare them with llvm-mc
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input sets and compare the four result words with the
                                       reference built from unbounded integers

The pixel fixture and the vertex stage are the ones of startup_gfx2hlsl_bit_alu_check.py: three flat parameters
(v0..v3, v4..v7, v8..v11) hold one operation's sources for four lanes, the operation (VOP3 spelling) writes v12..v15
and the pixel stage exports them to mrt0. The vertex stage places the inputs in the parameters, so the pixel stage
cannot fold them.
"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import POST_WORDS, PRE_WORDS, vs_source, vs_words, words_text  # noqa: E402
from startup_gfx2hlsl_mul24_align_check import MASK, REFERENCE  # noqa: E402

BODY_WORDS = {
    'v_mul_i32_i24': '0c0009d5000902000d0009d5010b02000e0009d5020d02000f0009d5030f0200',
    'v_mad_i32_i24': '0c0042d5000922040d0042d5010b26040e0042d5020d2a040f0042d5030f2e04',
    'v_mul_hi_i32_i24': '0c000ad5000902000d000ad5010b02000e000ad5020d02000f000ad5030f0200',
    'v_xnor_b32': '0c001ed5000902000d001ed5010b02000e001ed5020d02000f001ed5030f0200',
    'v_alignbit_b32': '0c004ed5000922040d004ed5010b26040e004ed5020d2a040f004ed5030f2e04',
    'v_alignbyte_b32': '0c004fd5000922040d004fd5010b26040e004fd5020d2a040f004fd5030f2e04',
}
MUL = [(0x7fffff, 0x7fffff, 0), (0x800000, 0x800000, 1), (0xffffff, 0xffffff, MASK), (0x800001, 0x7fffff, 5),
       (0x123456, 0xfedcba, 0x80000000), (0xab123456, 0x5cfedcba, 7), (0x1000, 0xfff, 0), (0xfff, 0x1000, 2),
       (0x555555, 0xaaaaaa, 0x12345678), (1, 0xffffff, 0), (0xfffffe, 0x800000, 3), (0x7fffff, 0x800000, 0)]
BITS = [(0x12345678, 0x9abcdef0, 0), (0x12345678, 0x9abcdef0, 4), (0xdeadbeef, 0x01234567, 31), (MASK, 0, 17),
        (0x80000000, 1, 1), (0x80000001, MASK, 0x80000003), (0xff000001, 0x00ffffff, 8), (0, MASK, 24),
        (0xdeadbeef, 0x89abcdef, 3), (1, 0x80000000, 32), (0x7fffffff, 0x80000000, 2), (MASK, MASK, 19)]
DRAW = {op: BITS if op[2:].startswith(('xnor', 'align')) else MUL for op in REFERENCE}


def source_count(op):
    return REFERENCE[op][0]


def body_source(op):
    return ''.join('%s_e64 v%d, %s\n' % (op, 12 + lane, ', '.join('v%d' % (4 * s + lane) for s in range(source_count(op))))
                   for lane in range(4))


def ps_words(op):
    return PRE_WORDS + bytes.fromhex(BODY_WORDS[op]) + POST_WORDS


def check_binary(llvm_mc):
    for op in REFERENCE:
        built = assemble(llvm_mc, body_source(op))
        if built != bytes.fromhex(BODY_WORDS[op]):
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op]))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, sets in DRAW.items():
                reference = REFERENCE[op][1]
                text = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob()).hlsl_ps()
                ps = compile_hlsl(compiler, kind, text, 'ps_%s_%s' % (op, kind), ps_profile, directory)
                for index in range(0, len(sets), 4):
                    chunk = sets[index:index + 4]
                    vs_text = G.Pipeline(vs_words(chunk), ps_words(op), root_blob()).hlsl_vs()
                    vs = compile_hlsl(compiler, kind, vs_text, 'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    expected = words_text(reference(*x) for x in chunk)
                    run = draw(renderer, adapter, vs, ps, expected)
                    if run.returncode:
                        # Keep going so one compiler's failure does not hide the other compiler's result.
                        failures.append('%s %s inputs %s\n%s\n%s' % (kind, op, chunk, run.stdout, run.stderr))
                        break
                    print('%s %s %s: %s' % (kind, op, [tuple('%x' % v for v in x[:source_count(op)]) for x in chunk],
                                            expected), flush=True)
    if failures:
        raise AssertionError('\n'.join(failures))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
