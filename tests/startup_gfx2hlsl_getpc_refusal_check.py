"""`s_getpc_b64` names why it is refused instead of reporting the mnemonic alone.

usage: startup_gfx2hlsl_getpc_refusal_check.py

RDNA 2 GETPC returns the address of its successor, i.e. the shader's load address plus a known byte offset, which
the HLSL model cannot represent as an integer. `calculated_pc_proof()` elides the pair when the program also
calculates a branch target; a program that reaches the dispatcher without such a proof was refused with the bare
mnemonic, which the refusal census cannot tell apart from an instruction the translator has no handler for.

This checks the named refusal for the forms a pixel or compute stage can carry, and the negative control: the same
instruction inside a program whose `s_setpc_b64` target is proven is elided, so the message only covers the unproven
case.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_calculated_target_check import POSITIVE, assert_result, translate  # noqa: E402

REASON = ('s_getpc_b64: the shader load address is only elided for a program that also calculates a branch target')
failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH  ' + detail))
    if not condition:
        failures.append(label)


def probe(kind):
    class Probe(G.Stage):
        """A Stage with no pipeline: enough state for one program to be translated."""

        def __init__(self):
            self.lines, self.sym, self.attrs = [], {}, {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.masks = set()
            self.exec_known, self.kind = True, kind

    return Probe()


def refusal(lines, kind='ps'):
    stage = probe(kind)
    stage.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(lines)]
    try:
        stage.translate()
    except G.Unsupported as reason:
        return str(reason), stage.lines
    return None, stage.lines


# The forms a captured request can carry: the instruction alone, with the low/high word arithmetic that makes the
# pair a pointer, and with the pair used directly as a descriptor.
FORMS = (
    ['s_getpc_b64 s[4:5]'],
    ['s_getpc_b64 s[4:5]', 's_add_u32 s4, s4, 0x10', 's_addc_u32 s5, s5, 0'],
    ['s_getpc_b64 s[4:5]', 's_load_dword s0, s[4:5], 0x0'],
)
for kind in ('ps', 'cs'):
    for lines in FORMS:
        reason, _ = refusal(lines, kind)
        check('%s: %s names the reason' % (kind, ' ; '.join(lines)[:52]), reason == REASON, repr(reason))

# The refusal is not the bare mnemonic: that is the regression this check exists for.
reason, _ = refusal(['s_getpc_b64 s[4:5]'])
check('the refusal is longer than the mnemonic', reason != 's_getpc_b64', repr(reason))
check('the refusal mentions the load address', reason is not None and 'load address' in reason, repr(reason))

# A proven control must translate and skip the decoy.
assert_result(translate(POSITIVE), (0, 20, 24), 24)
print('control: the proven calculated jump translates and skips the decoy: ok')

print('getpc refusal checks: %d failed' % len(failures))
if failures:
    raise SystemExit(1)
print('PASS')
