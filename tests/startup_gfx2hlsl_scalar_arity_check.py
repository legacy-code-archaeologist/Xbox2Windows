"""Operand counts of the scalar ALU table (SCA) in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_scalar_arity_check.py

Every SALU form in `SCA` is written as destination, source, so its body reads `ops[0]` and `ops[1]`.
A short list indexed past the end and left `translator crash in <op>: IndexError` in the refusal census,
which is the marker the batch translator uses for a translator defect rather than for an instruction it
does not lower. This checks the accepted shape and the refusal of the two short forms for every binary
and unary entry, then loads the translator with the guard removed and requires `IndexError` to come back,
so the check cannot pass without it.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402

GUARD = """        if base != 's_movk_i32' and len(ops) < 2:
            raise Unsupported('%s with %d operands, expected 2' % (base, len(ops)))
        if base == 's_movk_i32' and len(ops) != 2:"""
failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH  ' + detail))
    if not condition:
        failures.append(label)


def probe(module):
    class Probe(module.Stage):
        """A Stage with no pipeline: enough state for one scalar instruction to be translated."""

        def __init__(self):
            self.lines, self.sym = [], {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.masks, self.attrs = set(), {}
            self.exec_known = True

    return Probe


def outcome(module, line):
    """'ok', 'refused' or the name of another exception for one instruction text."""
    try:
        probe(module)().one(line.split()[0], line)
    except module.Unsupported:
        return 'refused'
    except Exception as error:  # noqa: BLE001 - the name is the observation
        return type(error).__name__
    return 'ok'


# One accepted shape per entry: the guard must not refuse a complete instruction.
ACCEPTED = {
    's_add_u32': 's_add_u32 s0, s1, s2',
    's_add_i32': 's_add_i32 s0, s1, s2',
    's_sub_u32': 's_sub_u32 s0, s1, s2',
    's_mul_i32': 's_mul_i32 s0, s1, s2',
    's_and_b32': 's_and_b32 s0, s1, s2',
    's_or_b32': 's_or_b32 s0, s1, s2',
    's_xor_b32': 's_xor_b32 s0, s1, s2',
    's_lshl_b32': 's_lshl_b32 s0, s1, 4',
    's_lshr_b32': 's_lshr_b32 s0, s1, 4',
    's_ashr_i32': 's_ashr_i32 s0, s1, 4',
    's_min_i32': 's_min_i32 s0, s1, s2',
    's_max_u32': 's_max_u32 s0, s1, s2',
    's_mul_hi_u32': 's_mul_hi_u32 s0, s1, s2',
    's_bfm_b32': 's_bfm_b32 s0, s1, s2',
    's_pack_hh_b32_b16': 's_pack_hh_b32_b16 s0, s1, s2',
    's_and_b64': 's_and_b64 s[0:1], s[2:3], s[4:5]',
    's_or_b64': 's_or_b64 s[0:1], s[2:3], s[4:5]',
    's_mov_b32': 's_mov_b32 s0, s1',
    's_not_b32': 's_not_b32 s0, s1',
    's_brev_b32': 's_brev_b32 s0, s1',
    's_sext_i32_i8': 's_sext_i32_i8 s0, s1',
    's_mov_b64': 's_mov_b64 s[0:1], s[2:3]',
    's_not_b64': 's_not_b64 s[0:1], s[2:3]',
    's_movk_i32': 's_movk_i32 s0, 0x1',
}

for name, text in sorted(ACCEPTED.items()):
    got = outcome(G, text)
    check('translates: %s' % text, got == 'ok', got)

# Every SCA entry must refuse the two short forms and name the instruction and both counts.
for name in sorted(G.SCA):
    for text, count in ((name, 0), (name + ' s0', 1)):
        got = outcome(G, text)
        check('%s: refuses %d operands' % (name, count), got == 'refused', got)

# The refusal text names the instruction and both counts, so the census groups it by instruction.
for name in ('s_add_u32', 's_mov_b32', 's_and_b64'):
    try:
        probe(G)().one(name, name + ' s0')
    except G.Unsupported as reason:
        check('%s: the refusal names the instruction and both counts' % name,
              str(reason) == '%s with 1 operands, expected 2' % name, repr(str(reason)))

# Negative control: without the guard the same input raises IndexError and a complete form still translates.
source = Path(G.__file__).read_text(encoding='utf-8')
check('control: the guard is in the translator once', source.count(GUARD) == 1,
      'count=%d' % source.count(GUARD))
unguarded_source = source.replace(GUARD, "        if base == 's_movk_i32' and len(ops) != 2:")
spec = importlib.util.spec_from_loader('gfx2hlsl_unguarded_scalar', loader=None)
unguarded = importlib.util.module_from_spec(spec)
unguarded.__dict__['__file__'] = G.__file__
exec(compile(unguarded_source, G.__file__, 'exec'), unguarded.__dict__)
for name in ('s_add_u32', 's_mov_b32', 's_mul_hi_u32', 's_not_b64'):
    got = outcome(unguarded, name)
    check('control: without its guard %s with 0 operands raises IndexError' % name,
          got == 'IndexError', got)
check('control: without its guard a complete s_add_u32 still translates',
      outcome(unguarded, ACCEPTED['s_add_u32']) == 'ok')

print('scalar ALU operand count checks: %d failed' % len(failures))
if failures:
    raise SystemExit(1)
print('PASS')
