"""s_cmp_eq_u64, s_cmp_lg_u64, s_bitcmp0_b64, s_bitcmp1_b64, s_quadmask_b32/b64 and s_wqm_b32/b64
in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_s_cmp64_wqm_check.py

RDNA 2 ISA, SOPC/SOP1: the 64-bit compares set SCC from the whole pair; S_BITCMPx_B64 tests bit S1[5:0];
S_QUADMASK packs one bit per group of four bits (SCC = result != 0); S_WQM sets every bit of a group that has
any bit set (SCC = result != 0). The references are plain Python integer arithmetic; the translated text is
evaluated on the CPU.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402

failures = []


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for single instructions to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.masks = set()


def run(line, regs):
    """Translate one line, evaluate its assignments in order, return the final register/scc values."""
    stage = Probe()
    stage.one(line.split(' ', 1)[0], line)
    env = dict(regs)
    for text in stage.lines:
        target, _, expression = text.rstrip(';').partition(' = ')
        expression = re.sub(r'\b(0x[0-9a-f]+|\d+)u\b', r'\1', expression)
        expression = expression.replace('&&', ' and ').replace('||', ' or ')
        value = eval(expression, {'uint': int}, env)
        env[target] = int(value) & 0xffffffff if not isinstance(value, bool) else value
    return env


def check(label, line, regs, want):
    try:
        got = run(line, regs)
        got = {key: got.get(key) for key in want}
    except Exception as exc:  # the translator refused or emitted something unexpected
        got = '%s: %s' % (type(exc).__name__, exc)
    if got != want:
        failures.append(label)
        print('MISMATCH %s: got=%s want=%s' % (line, got, want))


def pair(value):
    return {'s4': value & 0xffffffff, 's5': value >> 32}


QWORDS = (0, 1, 0xffffffffffffffff, 0x8000000000000001, 0x00000000ffffffff, 0xffffffff00000000,
          0x0123456789abcdef, 0x0000100000020000, 0x1111111111111111, 0x8888888888888888)

for a in QWORDS:
    for b in QWORDS:
        regs = {**pair(a), 's6': b & 0xffffffff, 's7': b >> 32}
        check('eq', 's_cmp_eq_u64 s[4:5], s[6:7]', regs, {'scc': a == b})
        check('lg', 's_cmp_lg_u64 s[4:5], s[6:7]', regs, {'scc': a != b})
for a in QWORDS:
    check('eq-zero', 's_cmp_eq_u64 s[4:5], 0', pair(a), {'scc': a == 0})
    check('lg-neg1', 's_cmp_lg_u64 s[4:5], -1', pair(a), {'scc': a != 0xffffffffffffffff})
    check('eq-lit', 's_cmp_eq_u64 s[4:5], 0x12345678', pair(a), {'scc': a == 0x12345678})
    for index in (0, 1, 31, 32, 33, 63, 64, 0x7f, 0xffffffc1):
        for mnemonic, want in (('s_bitcmp1_b64', 1), ('s_bitcmp0_b64', 0)):
            expected = ((a >> (index & 63)) & 1) == want
            check('bitcmp-imm', '%s s[4:5], %d' % (mnemonic, index), pair(a), {'scc': expected})
            check('bitcmp-reg', '%s s[4:5], s8' % mnemonic, {**pair(a), 's8': index}, {'scc': expected})

for word in (0, 1, 0xf, 0x10, 0x80000000, 0xffffffff, 0x01020304, 0x00f0000f, 0x12345678, 0xa0a0a0a0):
    quad = sum(1 << i for i in range(8) if (word >> 4 * i) & 0xf)
    spread = sum(0xf << 4 * i for i in range(8) if (word >> 4 * i) & 0xf)
    check('quadmask32', 's_quadmask_b32 s10, s4', {'s4': word}, {'s10': quad, 'scc': quad != 0})
    check('wqm32', 's_wqm_b32 s10, s4', {'s4': word}, {'s10': spread, 'scc': spread != 0})
for value in QWORDS:
    lo, hi = value & 0xffffffff, value >> 32
    quad = sum(1 << i for i in range(16) if (value >> 4 * i) & 0xf)
    spread = sum(0xf << 4 * i for i in range(16) if (value >> 4 * i) & 0xf)
    check('quadmask64', 's_quadmask_b64 s[10:11], s[4:5]', pair(value),
          {'s10': quad, 's11': 0, 'scc': quad != 0})
    check('wqm64', 's_wqm_b64 s[10:11], s[4:5]', pair(value),
          {'s10': spread & 0xffffffff, 's11': spread >> 32, 'scc': spread != 0})

# A lane mask has no per-lane word to compute on: those forms are refused (s_wqm of a mask stays a no-op).
for text in ('s_cmp_eq_u64 exec, 0', 's_cmp_lg_u64 vcc, s[4:5]', 's_bitcmp1_b64 exec, 3', 's_bitcmp0_b64 s[4:5], vcc_lo',
             's_quadmask_b64 s[10:11], exec', 's_quadmask_b32 s10, vcc_lo', 's_quadmask_b32 vcc_lo, s4',
             's_cmp_eq_u64 s[5:6], 0', 's_wqm_b64 s[11:12], s[4:5]'):
    stage = Probe()
    try:
        stage.one(text.split(' ', 1)[0], text)
    except G.Unsupported:
        pass
    else:
        failures.append(text)
        print('ACCEPTED %s' % text)
    assert not stage.lines, (text, stage.lines)
# A projected lane mask in EITHER word of a source pair is refused (the high word used to slip through).
for mask_reg in (4, 5):
    for text in ('s_bitcmp1_b64 s[4:5], 33', 's_bitcmp0_b64 s[4:5], 1', 's_cmp_eq_u64 s[4:5], 0',
                 's_cmp_lg_u64 s[6:7], s[4:5]', 's_quadmask_b64 s[10:11], s[4:5]'):
        stage = Probe()
        stage.masks = {mask_reg}
        try:
            stage.one(text.split(' ', 1)[0], text)
        except G.Unsupported:
            pass
        else:
            failures.append('%s masks=%d' % (text, mask_reg))
            print('ACCEPTED %s with mask s%d' % (text, mask_reg))

stage = Probe()
stage.one('s_wqm_b64', 's_wqm_b64 exec, exec')
assert not stage.lines, stage.lines

print('s_cmp64_wqm: %s' % ('ok' if not failures else '%d failures' % len(failures)))
sys.exit(1 if failures else 0)
