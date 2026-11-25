"""Check descriptor pointer propagation through s_add_u32 / s_addc_u32 in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_desc_ptr_prop_check.py

A shader advances a descriptor table or root pointer with `s_add_u32 sD, sP, <literal>` and
`s_addc_u32 sD+1, sP+1, 0`, into the same or another register pair, then loads descriptors through
the result. The load must read at the advanced byte offset. A non-literal addend must be refused,
never silently treated as the unchanged pointer.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


class Pipe:
    sampler_heap = False

    def __init__(self):
        self.keys = []
        self.modified = set()

    def resource(self, key, kind):
        if (key, kind) not in self.keys:
            self.keys.append((key, kind))
        return 'r%d' % self.keys.index((key, kind))


def stage(symbols):
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.sym, s.lines = Pipe(), dict(symbols), []
    s.used_s, s.used_v, s.uniform_v, s.exec_known = set(), set(), set(), True
    s.masks, s.varying_s = set(), set()
    s.scc_uniform = True
    return s


def run(symbols, program, branches=False):
    """Symbols after the (op, operands) steps, or the Unsupported message."""
    s = stage(symbols)
    s.has_branches = branches
    s.ins = [(4 * i, 4, op, '') for i, (op, _operands) in enumerate(program)]
    try:
        for i, (op, operands) in enumerate(program):
            s.pc = 4 * i
            if op == 's_load_dwordx4':
                s.op_s_load_dwordx4(operands.split(', '), [])
            else:
                s.lower(op, '%s %s' % (op, operands), operands.split(', '), [])
    except G.Unsupported as error:
        return 'refused: %s' % error
    return s


TABLE = {8: ('ptr', ('table', 0)), 9: ('ptrhi', ('table', 0))}
ROOT = {8: ('ptr', ('root', 1)), 9: ('ptrhi', ('root', 1))}
ADVANCE = [('s_add_u32', 's20, s8, 0x40'), ('s_addc_u32', 's21, s9, 0')]
LOAD = [('s_load_dwordx4', 's[4:7], s[20:21], 0x10')]

s = run(TABLE, ADVANCE + LOAD)
check('table pointer advanced into another pair, descriptor read at pointer + 0x40 + 0x10',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x50, 0), str(s if isinstance(s, str) else s.sym))

same = [('s_add_u32', 's8, s8, 0x20'), ('s_addc_u32', 's9, s9, 0')]
s = run(TABLE, same + [('s_load_dwordx4', 's[4:7], s[8:9], 0')])
check('table pointer advanced in place',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x20, 0), str(s if isinstance(s, str) else s.sym))

twice = [('s_add_u32', 's20, s8, 0x40'), ('s_addc_u32', 's21, s9, 0'),
         ('s_add_u32', 's22, s20, 0x10'), ('s_addc_u32', 's23, s21, 0')]
s = run(TABLE, twice + [('s_load_dwordx4', 's[4:7], s[22:23], 0x4')])
check('two advances add up',
      not isinstance(s, str) and s.sym.get(4) == ('desc', ('table', 0), 0x54, 0), str(s if isinstance(s, str) else s.sym))

s = run(ROOT, ADVANCE + LOAD)
check('root constant-buffer pointer advanced, buffer read at the advanced offset',
      not isinstance(s, str) and any('Load(80)' in line for line in s.lines), str(s if isinstance(s, str) else s.lines))

s = run(TABLE, [('s_add_u32', 's8, s8, s2')])
check('a register addend on a pointer is refused', isinstance(s, str) and 'non-literal' in s, str(s))

s = run(TABLE, ADVANCE + [('s_load_dwordx4', 's[4:7], s[20:21], s2')])
check('a register offset through an advanced pointer is still refused', isinstance(s, str), str(s))

s = run(TABLE, ADVANCE)
try:
    s.descriptor('s[20:21]')
    check('an advanced pointer used whole as a descriptor is refused', False, 'accepted')
except G.Unsupported:
    check('an advanced pointer used whole as a descriptor is refused', True)

s = run(TABLE, ADVANCE, branches=True)
check('a pointer advance in a program with branches is refused', isinstance(s, str), str(s))

# The high-word add is the carry of the low-word add right before it, never of an unrelated carry in SCC.
s = run(TABLE, [('s_cmp_eq_u32', '0, 0'), ('s_addc_u32', 's9, s9, 0')])
check('s_addc_u32 without the low-word add is refused', isinstance(s, str), str(s))

s = run(TABLE, [ADVANCE[0], ('s_cmp_eq_u32', '0, 0'), ADVANCE[1]])
check('s_addc_u32 separated from its low-word add is refused', isinstance(s, str), str(s))

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('startup_gfx2hlsl_desc_ptr_prop: ok')
