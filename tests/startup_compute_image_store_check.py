"""Check 32-bit float compute image stores (with and without unorm) from the startup translator."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

import compute_mimg_float_store_check as fixture
from startup_gfx2hlsl_compute_check import G, TGID_X, TGID_Y, assemble, refused, root_blob, tidig


# Reuse #2105's source and CPU reference unchanged. Three constants put the descriptor table at s7.
RSRC2 = (10 << 1) | TGID_X | TGID_Y | tidig(2)
PREFIX, STORE, END = fixture.WORDS[:-12], fixture.WORDS[-12:-4], fixture.WORDS[-4:]
STORE_UNORM = bytes.fromhex('081f20f0080e0600')  # image_store v[14:17], v[8:9], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
MASK = bytes.fromhex('8002a47d')  # v_cmpx_eq_u32_e32 0, v1
LINE = 'w0[uint2(v8, v9)] = asfloat(uint4(v14, v15, v16, v17));'


def pipeline(body):
    return G.ComputePipeline(PREFIX + body + END, root_blob(3), RSRC2, fixture.GROUP)


def controls(llvm_mc):
    def encoded(line, hex_words):
        words = bytes.fromhex(hex_words)
        if llvm_mc:
            assert assemble(llvm_mc, line + '\n') == words, line
        return words

    if llvm_mc:
        assert assemble(llvm_mc, '\n'.join(fixture.LINES) + '\n') == fixture.WORDS
        assert assemble(llvm_mc, fixture.STORE + ' unorm\n') == STORE_UNORM
        assert assemble(llvm_mc, 'v_cmpx_eq_u32_e32 0, v1\n') == MASK

    pipe = pipeline(STORE)
    assert pipe.manifest().splitlines()[-2:] == [
        'srv 0 tex2d table 1 96',
        'uav 0 tex2d float table 1 64 channels 4',
    ]
    assert LINE in pipe.hlsl_cs() and 'RWTexture2D<float4> w0 : register(u0);' in pipe.hlsl_cs()
    # unorm is a no-op on image_store (integer texel coordinates) and emits identical HLSL and manifest.
    unorm_pipe = pipeline(STORE_UNORM)
    assert unorm_pipe.hlsl_cs() == pipe.hlsl_cs()
    assert unorm_pipe.manifest() == pipe.manifest()
    assert 'if (exec & 1) ' + LINE in pipeline(MASK + STORE_UNORM).hlsl_cs()

    mods = ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D', 'unorm']
    for data in ('v14', 'v[14:15]', 'v[14:16]', 'v[253:256]'):
        refused('four VGPRs expected', lambda: pipe.cs.op_image_store([data, 'v[8:9]', 's[24:31]'], mods))
    refused('fixed table image descriptor',
            lambda: pipe.cs.op_image_store(['v[14:17]', 'v[8:9]', 's[28:35]'], mods))

    for asm, code, reason in (
        (fixture.STORE.replace('v[14:17]', 'v[14:15]').replace('dmask:0xf', 'dmask:0x3'),
         '080320f0080e0600', 'typed UAV store writes all four channels'),
        (fixture.STORE.replace('v[8:9]', 'v8') + ' a16', '080f20f0080e0640', 'modifier a16'),
        (fixture.STORE.replace('v[8:9]', 'v[8:10]').replace('IMG_2D', 'IMG_3D'),
         '100f20f0080e0600', 'only 2D images'),
    ):
        instruction = encoded(asm, code)
        refused(reason, lambda: pipeline(instruction))

    wait = encoded('s_waitcnt lgkmcnt(0)', '7fc08cbf')
    changed = encoded('s_mov_b32 s26, 0', '80039abe')
    refused('modified image descriptor', lambda: pipeline(changed + STORE))
    duplicate = encoded('s_load_dwordx8 s[40:47], s[12:13], 0x40', '060a0cf4400000fa')
    changed_alias = encoded('s_or_b32 s46, s46, 0x40000', '2eff2e8800000400')
    refused('modified image descriptor', lambda: pipeline(duplicate + wait + changed_alias + STORE))

    raw = encoded('buffer_store_dword v14, v0, s[32:35], 0 offen', '001070e0000e0880')
    read = encoded('buffer_load_dword v5, v0, s[32:35], 0 offen', '001030e000050880')
    for offset in (64, 128):
        load = encoded(f's_load_dwordx4 s[32:35], s[12:13], {offset}', f'060808f4{offset:02x}0000fa')
        for body in (raw + STORE, STORE + raw):
            if offset == 64:
                refused('used as', lambda: pipeline(load + wait + body))
                continue
            mixed = pipeline(load + wait + body)
            raw_slot, image_slot = (0, 1) if body.startswith(raw) else (1, 0)
            assert f'uav {raw_slot} table 1 128' in mixed.manifest().splitlines()
            assert f'uav {image_slot} tex2d float table 1 64 channels 4' in mixed.manifest().splitlines()
            assert f'RWByteAddressBuffer u{raw_slot} : register(u{raw_slot});' in mixed.hlsl_cs()
            assert f'RWTexture2D<float4> w{image_slot} : register(u{image_slot});' in mixed.hlsl_cs()
        if offset == 64:
            for body in (read + STORE, STORE + read):
                refused('used for a load and a store', lambda: pipeline(load + wait + body))
    print('Compute 32-bit float image_store, unorm, EXEC, raw/texture binding order, alias and refusal checks passed')


def draw_cases(directory, compiler, renderer, adapter, payload, cases):
    hlsl, cso, case = (directory / f'image.{ext}' for ext in ('hlsl', 'cso', 'case'))
    case.write_bytes(struct.pack(f'<2I{len(payload)}I', fixture.GROUPS | fixture.FLOAT_TEXTURE_FLAG, 0, *payload))
    for name, source, expected_mismatches, needle in cases:
        hlsl.write_text(source, encoding='ascii')
        subprocess.run([compiler, '-T', 'cs_6_0', '-E', 'main', '-Fo', cso, hlsl], check=True)
        result = subprocess.run([renderer, adapter, '--compute-texture-store-case', cso, case],
                                capture_output=True, text=True, timeout=15)
        print(f'{name}:\n{result.stdout}{result.stderr}', end='', flush=True)
        if result.returncode == 77:
            return 77
        if 'debug_layer=enabled' in result.stdout:
            assert 'debug_errors=0 debug_warnings=0' in result.stdout, name
        assert f'compared_words=1344 mismatches={expected_mismatches}' in result.stdout, name
        if expected_mismatches == 0:
            assert result.returncode == 0, name
        else:
            assert result.returncode == 1 and needle in result.stderr, name
    return 0


def dispatch(directory, compiler, renderer, adapter):
    source = pipeline(STORE_UNORM).hlsl_cs()
    assert source.count(LINE) == 1
    cases = (
        ('full EXEC (unorm)', source, 0, None),
        ('omit store', source.replace(LINE, ''), 64, 'mip=0 word=0 expected=3f100000 actual=bf000000'),
        ('swap R/B channels',
         source.replace('uint4(v14, v15, v16, v17)', 'uint4(v15, v14, v16, v17)'),
         32, 'mip=0 word=0 expected=3f100000 actual=3f100004'),
        ('swap RG/BA pairs',
         source.replace('uint4(v14, v15, v16, v17)', 'uint4(v16, v17, v14, v15)'),
         64, 'mip=0 word=0 expected=3f100000 actual=40810000'),
    )
    payload = fixture.reference()
    if draw_cases(directory, compiler, renderer, adapter, payload, cases) == 77:
        return 77
    # v1 is the local Y. Only row zero of the two dispatched rows may change.
    words_per_image = len(payload) // 3
    row = 32 * 4
    payload[words_per_image + row:words_per_image + 2 * row] = payload[row:2 * row]
    masked = pipeline(MASK + STORE_UNORM).hlsl_cs()
    guard = 'if (exec & 1) ' + LINE
    assert masked.count(guard) == 1
    cases = (
        ('partial EXEC', masked, 0, None),
        ('drop EXEC guard', masked.replace(guard, LINE), 32, 'mip=0 word=128 expected=bf001000 actual=3f101000'),
    )
    return draw_cases(directory, compiler, renderer, adapter, payload, cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llvm-mc')
    mode.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RENDERER', 'ADAPTER'))
    args = parser.parse_args()
    if args.dispatch:
        out = Path(__file__).resolve().parents[1] / 'out'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            raise SystemExit(dispatch(Path(tmp), *args.dispatch))
    controls(args.llvm_mc)
