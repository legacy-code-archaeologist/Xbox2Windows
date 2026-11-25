"""s_pack_ll_b32_b16, s_pack_lh_b32_b16 and s_absdiff_i32 in tools/startup/gfx2hlsl.py (Part of #1723).

Pure Python. Each form is built as a SOP2 word, decoded, translated, and the emitted lines run on the integer
evaluator of startup_gfx2hlsl_bit_alu_check.py over edge values (the encoder, forms and value set come from
startup_gfx2hlsl_sop2_complement_check.py). The references use plain integer arithmetic from the ISA meaning:

  s_pack_ll_b32_b16: D = (S0 & 0xffff) | (S1 << 16)             SCC unchanged
  s_pack_lh_b32_b16: D = (S0 & 0xffff) | (S1 & 0xffff0000)      SCC unchanged
  s_pack_hh_b32_b16: D = (S0 >> 16)    | (S1 & 0xffff0000)      SCC unchanged
  s_absdiff_i32:     D = |wrap32(S0 - S1)| (INT_MIN stays INT_MIN); SCC = (D != 0)

Lane-mask operands of the pack forms and s_absdiff_i32 are refused (the per-lane projection has no whole word to read).

dispatch <dxc> <offscreen_check> <adapter> also checks translated HLSL by D3D12 UAV readback.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_sop2_complement_check as C  # noqa: E402

MASK = C.MASK


def signed(x):
    return x - (1 << 32) if x & 0x80000000 else x


REFERENCES = {
    's_pack_ll_b32_b16': (lambda a, b: (a & 0xffff) | ((b << 16) & MASK), False),
    's_pack_lh_b32_b16': (lambda a, b: (a & 0xffff) | (b & 0xffff0000), False),
    's_pack_hh_b32_b16': (lambda a, b: (a >> 16) | (b & 0xffff0000), False),
    's_absdiff_i32': (lambda a, b: abs(signed((a - b) & MASK)) & MASK, True),
}

ABSDIFF_CASES = (
    (0x80000000, 1), (1, 0x80000000),
    (0x7fffffff, 0x80000000), (0x80000000, 0x7fffffff),
    (5, 5), (0, 0), (0xffffffff, 0), (3, 10),
)
PACK_CASES = (
    (0xffffffff, 0xffffffff), (0x0000ffff, 0x0000ffff),
    (0xffff0000, 0xffff0000), (0x00010000, 0x0000ffff),
    (0x12345678, 0x9abcdef0), (0, 0xffffffff),
)
PACK_CASES += tuple((b, a) for a, b in PACK_CASES if a != b)


def dispatch(dxc, runner, adapter, name, cases, updates_scc, directory):
    _, stage = C.translate(C.code(name, (3, 1, 2, None)))
    # The runner fills the source words. XOR at load time to recover the inputs.
    fill = tuple(((i + 1) * 0x9e3779b1 & MASK) | 1 for i in range(2 * len(cases)))
    result_width = 1 + int(updates_scc)
    lines = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, (a, b) in enumerate(cases):
        src_word = 2 * i
        dst_byte = 4 * (len(fill) + result_width * i)
        lines.extend((
            '    {',
            f'        uint s1 = data.Load({4 * src_word}) ^ {fill[src_word] ^ a:#010x}u;',
            f'        uint s2 = data.Load({4 * (src_word + 1)}) ^ {fill[src_word + 1] ^ b:#010x}u;',
            '        uint s3;',
            '        bool scc = true;',
        ))
        lines.extend('        ' + line for line in stage.lines)
        lines.append(f'        data.Store({dst_byte}, s3);')
        if updates_scc:
            lines.append(f'        data.Store({dst_byte + 4}, uint(scc));')
        lines.append('    }')
    lines.append('}')

    src, cso, output = (directory / (name + suffix) for suffix in ('.hlsl', '.cso', '.bin'))
    src.write_text('\n'.join(lines) + '\n', encoding='ascii')
    compile_result = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                                    stderr=subprocess.PIPE, text=True)
    if compile_result.returncode or compile_result.stderr:
        raise AssertionError(compile_result.stderr or f'DXC exited with {compile_result.returncode}')
    word_count = len(fill) + result_width * len(cases)
    # The raw UAV dump takes a size in whole 16-byte rows; the padding words are not read back.
    dump_words = (word_count + 3) // 4 * 4
    run = subprocess.run([runner, adapter, '--raw-uav-dump', str(cso), '1', str(4 * dump_words), str(output)],
                         capture_output=True, text=True)
    print(run.stdout, end='')
    if run.returncode == 77:
        sys.exit(77)
    if run.returncode or run.stderr:
        raise AssertionError(run.stderr or f'raw UAV dump exited with {run.returncode}')
    actual = struct.unpack(f'<{dump_words}I', output.read_bytes())
    assert actual[:len(fill)] == fill, f'{name}: operand region changed'
    return actual[len(fill):word_count]


def check_dispatch(dxc, runner, adapter):
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for name, (reference, updates_scc) in REFERENCES.items():
            cases = ABSDIFF_CASES if name == 's_absdiff_i32' else PACK_CASES
            actual = dispatch(dxc, runner, adapter, name, cases, updates_scc, Path(tmp))
            width = 1 + int(updates_scc)
            for i, (a, b) in enumerate(cases):
                result = reference(a, b)
                want = (result, int(result != 0)) if updates_scc else (result,)
                got = actual[i * width:(i + 1) * width]
                label = f'{name} {a:#010x}, {b:#010x}'
                detail = f'got {tuple(map(hex, got))}, want {tuple(map(hex, want))}'
                C.check(f'{label}: {detail}', got == want)
    if C.failures:
        sys.exit(1)


def main():
    for name, (reference, scc) in REFERENCES.items():
        C.OPCODE[name] = next(op for op, (n, _) in C.D.SOP2_OPS.items() if n == name)
        bad = C.first_mismatch(name, reference, updates_scc=scc)
        C.check('%s value and SCC' % name, bad is None, str(bad))
        if name.startswith('s_pack') or name == 's_absdiff_i32':
            for form in C.MASK_FORMS:
                _, stage = C.attempt(C.code(name, form))
                C.check('%s refuses lane mask form %r' % (name, form), stage.lines[0].startswith('refused: '))
    # the wrapped difference of 0x80000000 and 1 is 0x7fffffff; a compare-first lowering gives 0x80000001
    C.check('absdiff overflow reference', REFERENCES['s_absdiff_i32'][0](0x80000000, 1) == 0x7fffffff)
    # a symbolic descriptor source must be refused before SCC is left stale
    for name in REFERENCES:
        stage = C.Probe()
        stage.sym[1] = ('ptr', 0, 0)
        try:
            stage.one(name, '%s s3, s1, s2' % name)
        except C.G.Unsupported:
            C.check('%s refuses a descriptor source' % name, not stage.lines)
        else:
            C.check('%s refuses a descriptor source' % name, False, str(stage.lines))
    # a wrong reference must be caught: swapped sources
    swapped = C.first_mismatch('s_pack_lh_b32_b16', lambda a, b: REFERENCES['s_pack_lh_b32_b16'][0](b, a), False)
    C.check('swapped pack_lh reference is caught', swapped is not None)
    if C.failures:
        print('FAILED: %s' % ', '.join(C.failures))
        sys.exit(1)
    print('startup_gfx2hlsl_pack_absdiff_check: ok')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        check_dispatch(*sys.argv[2:])
    else:
        sys.exit(__doc__)
