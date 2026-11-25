"""Cube and cube-array images in tools/startup/gfx2hlsl.py (dim:SQ_RSRC_IMG_CUBE), drawn on D3D12 (#1723).

  (no arguments)                       translation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw and compare the texel of the slice the cube address selects

The stage is the 2D-array sample fixture of startup_image_2d_array_check.py with only the dimension field and the
two in-face coordinates changed: s = 1.75 and t = 1.25 (the ISA's biased range [1, 2], so the lowering must remove
the integer bias) and the third address register, root constant 1, as the cube slice. The ISA's slice is
layer * 8 + face (eight face slots per cube layer); the array view the runtime binds has six slices per cube, so
the lowering maps it to layer * 6 + face. --startup-image-cube-array binds twelve 2x2 slices (two cubes) whose texel
(x, y, slice) holds ((1 + x + 2y + 4 * slice) / 8, x, y, slice); s = 0.75, t = 0.25 reads texel (1, 0).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_image_2d_array_check import SAMPLE_ARRAY, SAMPLE_WORDS, translate  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl, texel  # noqa: E402

CUBE_IMAGE = bytes.fromhex('180f80f00000c400')
CUBE_WORDS = (SAMPLE_WORDS.replace(SAMPLE_ARRAY, CUBE_IMAGE)
              .replace(bytes.fromhex('ff02007e0000403f'), bytes.fromhex('ff02007e0000e03f'))   # v0 = 1.75
              .replace(bytes.fromhex('ff02027e0000803e'), bytes.fromhex('ff02027e0000a03f')))  # v1 = 1.25
assert CUBE_WORDS.count(CUBE_IMAGE) == 1 and bytes.fromhex('0000e03f') in CUBE_WORDS and bytes.fromhex('0000a03f') in CUBE_WORDS
CALL = 'r0.Sample(q0, float3(F(v0) - 1.0, F(v1) - 1.0, cubeslice(F(v2))))'
# The wrapping point sampler maps s = 1.75 to 0.75 even unbiased, so only the slice is mutated.
# (layer * 8 + face as the address, the array slice it must read): layer * 6 + face.
CASES = [('0', 0), ('5', 5), ('8', 6), ('9', 7), ('13', 11)]
MUTANTS = [
    ('slice left as layer * 8 + face', CALL.replace('cubeslice(F(v2))', 'F(v2)'), '9', 9),
]


def controls():
    pipe = translate(CUBE_WORDS)
    ps = pipe.hlsl_ps()
    assert 'Texture2DArray<float4> r0 : register(t0);' in ps and 't = %s;' % CALL in ps, ps
    assert 'srv 0 tex2da table 0 0' in pipe.manifest(), pipe.manifest()
    print('image_sample dim:SQ_RSRC_IMG_CUBE: %s' % CALL)


def run(renderer, adapter, vs, shader, z, expected):
    return subprocess.run([renderer, adapter, '--startup-image-cube-array', str(vs), str(shader), z, expected],
                          capture_output=True, text=True)


def check_draw(dxc, renderer, adapter, fxc=None):
    translated = translate(CUBE_WORDS).hlsl_ps()
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_%s' % kind, vs_profile, directory)
            shader = compile_hlsl(compiler, kind, translated, 'ps_cube_%s' % kind, ps_profile, directory)
            for z, slice_index in CASES:
                result = run(renderer, adapter, vs, shader, z, texel(1, 0, slice_index))
                sys.stdout.write(result.stdout)
                if result.returncode == 77:
                    print(result.stdout + result.stderr)
                    sys.exit(77)
                if result.returncode:
                    raise AssertionError('%s cube address=%s\n%s\n%s' % (kind, z, result.stdout, result.stderr))
                if 'debug_layer=enabled' in result.stdout:
                    assert 'debug_errors=0 debug_warnings=0' in result.stdout, (result.stdout, result.stderr)
                print('%s cube address=%s reads slice %d' % (kind, z, slice_index), flush=True)
            for label, mutated, z, slice_read in MUTANTS:
                source = translated.replace(CALL, mutated)
                assert source != translated, label
                mutant = compile_hlsl(compiler, kind, source, 'ps_cube_mutant_%s' % kind, ps_profile, directory)
                expected_slice = dict(CASES)[z]
                result = run(renderer, adapter, vs, mutant, z, texel(1, 0, expected_slice))
                sys.stdout.write(result.stdout)
                if result.returncode == 0 or 'actual=%s' % texel(1, 0, slice_read) not in result.stderr:
                    raise AssertionError('%s %s: control did not read slice %d\n%s\n%s'
                                         % (kind, label, slice_read, result.stdout, result.stderr))
                print('%s %s: fails as expected, reads slice %d' % (kind, label, slice_read), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
    elif sys.argv[1] == 'draw':
        controls()
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
