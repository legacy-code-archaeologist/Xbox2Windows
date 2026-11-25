"""s_nor_b32 and s_nand_b32 on a lane mask in tools/startup/gfx2hlsl.py (Part of #1723).

Pure Python, no arguments. A lane mask is the per-lane bit the translator keeps for exec, vcc and the SGPRs that
hold a compare result. Each form is translated, and the emitted lines are run on the CPU with the integer
evaluator of startup_gfx2hlsl_bit_alu_check.py over every pair of lane bits. The reference is the ISA meaning on one
lane: D = ~(S0 op S1), kept to that lane's bit. The destination must also be recorded as a mask (or be exec/vcc), so
that a later mask operation reads it as one.

Refused, as before: s_xnor_b32 on a lane mask; the high halves exec_hi/vcc_hi (not modelled); and any form that mixes a
lane mask with a raw scalar word (inline constant or plain SGPR), because such a word has no per-lane bit.
`s_nor_b32 vcc_lo, 1, 0` is ~(1|0) = 0xfffffffe in the ISA, so lane 1 and up must read 1, which one bit cannot say.

Reproduce: python3 -B tests/startup_gfx2hlsl_s_nor_nand_check.py (CPU only; no GPU or WARP result is claimed).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_sop2_complement_check as C  # noqa: E402

G = C.G
REFERENCE = {
    's_nand_b32': lambda a, b: ~(a & b) & 1,
    's_nor_b32': lambda a, b: ~(a | b) & 1,
}
# (destination operand, source operands, mask SGPRs, destination variable, source variables)
FORMS = [
    ('s3', ['vcc_lo', 's2'], {2}, 's3', ['vcc', 's2']),
    ('s3', ['s1', 'exec_lo'], {1}, 's3', ['s1', 'exec']),
    ('vcc_lo', ['s1', 's2'], {1, 2}, 'vcc', ['s1', 's2']),
    ('exec_lo', ['s1', 's2'], {1, 2}, 'exec', ['s1', 's2']),
]
# Each must raise Unsupported: unprojected scalar words and the unmodelled high halves.
REFUSED = [
    ('vcc_lo', ['1', '0'], set()),
    ('vcc_lo', ['1', '-1'], set()),
    ('exec_lo', ['s1', 's2'], set()),
    ('s3', ['vcc_lo', '0'], set()),
    ('s3', ['s1', '-1'], {1}),
    ('vcc_hi', ['0', '0'], set()),
    ('exec_hi', ['0', '0'], set()),
    ('vcc_hi', ['vcc_lo', 'exec_lo'], set()),
    ('s3', ['vcc_hi', 's2'], set()),
]


def translate(name, dest, srcs, masks):
    stage = C.Probe()
    stage.masks.update(masks)
    stage.one(name, '%s %s, %s, %s' % (name, dest, srcs[0], srcs[1]))
    return stage


def lane_value(stage, dest_var, src_vars, a, b):
    env = {src_vars[0]: a, src_vars[1]: b, 'scc': C.SCC_UNSET}
    for source in ('s1', 's2', 'vcc', 'exec'):
        env.setdefault(source, 0)
    return C.run_lines(stage.lines, env)[dest_var]


def main():
    for name, reference in REFERENCE.items():
        wrong, refused = [], []
        for dest, srcs, masks, dest_var, src_vars in FORMS:
            try:
                stage = translate(name, dest, srcs, masks)
            except G.Unsupported as error:
                refused.append((dest, srcs, str(error)))
                continue
            if dest.startswith('s') and int(dest[1:]) not in stage.masks:
                wrong.append((dest, 'destination not recorded as a mask', stage.masks))
            for a in (0, 1):
                for b in (0, 1):
                    got = lane_value(stage, dest_var, src_vars, a, b)
                    if got != reference(a, b):
                        wrong.append((dest, srcs, a, b, got, reference(a, b)))
        C.check('%s: mask forms are lowered' % name, not refused, str(refused[:2]))
        C.check('%s: lane-bit results equal the reference' % name, not wrong, str(wrong[:2]))

        accepted = []
        for dest, srcs, masks in REFUSED:
            try:
                translate(name, dest, srcs, masks)
                accepted.append((dest, srcs))
            except G.Unsupported:
                pass
        C.check('%s: raw scalar words and high halves stay refused (%d forms)' % (name, len(REFUSED)),
                not accepted, str(accepted[:2]))

    for dest, srcs, masks, _, _ in FORMS[:1]:
        try:
            translate('s_xnor_b32', dest, srcs, masks)
            C.check('s_xnor_b32 on a lane mask is still refused', False, 'accepted')
        except G.Unsupported as error:
            C.check('s_xnor_b32 on a lane mask is still refused', str(error) == 's_xnor_b32 on a lane mask', str(error))

    if C.failures:
        print('FAIL')
        for text in C.failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
