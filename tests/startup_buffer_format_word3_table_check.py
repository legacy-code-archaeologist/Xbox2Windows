"""The stand-in's DXGI -> FORMAT table (src/startup_stubs/buffer_format_word3.h) against tools/startup/gfx2hlsl.py.

A typed buffer view the title created has no hardware descriptor, so the draw builds word 3 for the translated
buffer_load_format_* switch from the view's DXGI format. Each mapped format must name a FORMAT code the switch
decodes (DESC_FORMAT_CODES), agree with the vertex-fetch table FETCH_DXGI where that has the format, and the
decoded-code list must equal the switch's cases, so a code is never accepted that would read zeros.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402

# Public DXGI_FORMAT values (dxgiformat.h) for the names the header uses.
DXGI = {'R32G32B32A32_FLOAT': 2, 'R32G32B32_FLOAT': 6, 'R16G16B16A16_FLOAT': 10, 'R32G32_FLOAT': 16,
        'R8G8B8A8_UNORM': 28, 'R16G16_FLOAT': 34, 'R16G16_UNORM': 35, 'R16G16_UINT': 36, 'R16G16_SNORM': 37, 'R8G8B8A8_UINT': 30, 'R32_FLOAT': 41,
        'R8G8_UNORM': 49, 'R8_UNORM': 61, 'R8_UINT': 62}
# The buffer format each DXGI format reads as: FETCH_DXGI, plus the three formats it does not list.
EXPECTED = {**{dxgi: name for dxgi, (name, _) in gfx2hlsl.FETCH_DXGI.items()},
            10: 'BUF_FMT_16_16_16_16_FLOAT', 62: 'BUF_FMT_8_UINT', 36: 'BUF_FMT_16_16_UINT'}


def main(header=ROOT / 'src' / 'startup_stubs' / 'buffer_format_word3.h'):
    text = header.read_text()
    code_fn = text[text.index('buffer_format_code('):text.index('view_word3(')]
    mapped = {DXGI[n]: int(c) for n, c in re.findall(r'case DXGI_FORMAT_(\w+): return (\d+);', code_fn)}
    supported = (ROOT / 'src' / 'startup_stubs' / 'formatted_buffer.h').read_text()
    decoded_fn = supported[supported.index('formatted_buffer_supported('):]
    decoded = {int(c) for c in re.findall(r'case (\d+):', decoded_fn)}
    codes = gfx2hlsl.DESC_FORMAT_CODES
    assert decoded == set(codes.values()), sorted(decoded ^ set(codes.values()))
    assert set(mapped) == set(EXPECTED), sorted(set(mapped) ^ set(EXPECTED))
    for dxgi, code in mapped.items():
        assert code == codes[EXPECTED[dxgi]], (dxgi, code, EXPECTED[dxgi])
    assert set(EXPECTED.values()) == set(codes), 'every decoded format has a DXGI view format'
    return len(mapped)


if __name__ == '__main__':
    # Negative control: a header that maps R32G32B32A32_FLOAT to 74 (the three-channel code) must fail.
    original = (ROOT / 'src' / 'startup_stubs' / 'buffer_format_word3.h').read_text()
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        wrong = Path(temp) / 'buffer_format_word3.h'
        wrong.write_text(original.replace('case DXGI_FORMAT_R32G32B32A32_FLOAT: return 77;',
                                          'case DXGI_FORMAT_R32G32B32A32_FLOAT: return 74;'))
        try:
            main(wrong)
        except AssertionError:
            pass
        else:
            raise SystemExit('a wrong FORMAT code for R32G32B32A32_FLOAT passed')
    print('buffer format word 3 table: %d DXGI formats match the translation' % main())
