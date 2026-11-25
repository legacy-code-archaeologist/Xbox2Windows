"""s_pack_ll_b32_b16, s_pack_lh_b32_b16 and s_absdiff_i32 in tools/startup/gfx2hlsl.py (Part of #1723).

Pure Python. Each form is built as a SOP2 word, decoded, translated, and the emitted lines run on the integer
evaluator of startup_gfx2hlsl_bit_alu_check.py over edge values (the encoder, forms and value set come from
startup_gfx2hlsl_sop2_complement_check.py). The references use plain integer arithmetic from the ISA meaning:

  s_pack_ll_b32_b16: D = (S0 & 0xffff) | (S1 << 16)             SCC unchanged
  s_pack_lh_b32_b16: D = (S0 & 0xffff) | (S1 & 0xffff0000)      SCC unchanged
  s_pack_hh_b32_b16: D = (S0 >> 16)    | (S1 & 0xffff0000)      SCC unchanged
  s_absdiff_i32:     D = |wrap32(S0 - S1)| (INT_MIN stays INT_MIN); SCC = (D != 0)

Lane-mask operands of the pack forms and s_absdiff_i32 are refused (the per-lane projection has no whole word to read).
"""
import sys
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
    main()
