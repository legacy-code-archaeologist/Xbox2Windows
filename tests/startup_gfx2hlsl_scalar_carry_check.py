"""s_add_u32, s_sub_u32, s_addc_u32 and s_subb_u32 in tools/startup/gfx2hlsl.py (Part of #1723).

Pure Python, no arguments. Each form is decoded and translated, and the emitted lines are run on the CPU with the
integer evaluator of startup_gfx2hlsl_bit_alu_check.py. The references are written from the ISA text with plain
integers, not with the lowering's expressions:

  s_add_u32   D = S0 + S1             SCC = carry out of bit 31
  s_sub_u32   D = S0 - S1             SCC = borrow (S1 > S0)
  s_addc_u32  D = S0 + S1 + SCC       SCC = carry out of bit 31
  s_subb_u32  D = S0 - S1 - SCC       SCC = borrow (S1 + SCC > S0)

Every pair of boundary values is run with SCC clear and set, and with the destination also a source, because the
carry has to be read before the destination is written. The carry-in forms on a lane mask stay refused.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_sop2_complement_check as C  # noqa: E402

G = C.G
MASK = C.MASK


def ref_add(a, b, cin):
    total = a + b + cin
    return total & MASK, int(total >> 32 != 0)


def ref_sub(a, b, cin):
    return (a - b - cin) & MASK, int(b + cin > a)


# mnemonic -> (reference, reads SCC as an input)
OPS = {
    's_add_u32': (ref_add, False),
    's_sub_u32': (ref_sub, False),
    's_addc_u32': (ref_add, True),
    's_subb_u32': (ref_sub, True),
}


def first_mismatch(name):
    reference, carry_in = OPS[name]
    for form in C.VALUE_FORMS:
        sdst, ssrc0, ssrc1, literal = form
        asm, stage = C.attempt(C.code(name, form))
        if stage.lines[0].startswith('refused: '):
            return asm, stage.lines[0]
        destination = 'm0' if sdst == C.M0 else 's%d' % sdst
        for a, b in C.PAIRS:
            for scc in (0, 1):
                env = {'s1': a, 's2': b, 'scc': scc, 'w': 0}
                want, carry = reference(C.source_value(ssrc0, literal, env), C.source_value(ssrc1, literal, env),
                                        scc if carry_in else 0)
                got = C.run_lines(stage.lines, env)
                if (got.get(destination), got['scc']) != (want, carry):
                    return asm, hex(a), hex(b), scc, (got.get(destination), got['scc']), (want, carry)
    return None


def main():
    for name in OPS:
        C.check('%s: result and SCC equal the reference' % name, first_mismatch(name) is None,
                str(first_mismatch(name)))

    for name in ('s_addc_u32', 's_subb_u32'):
        for form in C.MASK_FORMS:
            refused = C.attempt(C.code(name, form))[1].lines
            C.check('%s on a lane mask is refused (%s)' % (name, form[:3]),
                    refused == ['refused: %s on a lane mask' % name], str(refused))

    if C.failures:
        print('FAIL')
        for text in C.failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
