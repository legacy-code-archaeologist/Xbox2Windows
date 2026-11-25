"""Check s_load with a register offset that holds a known constant (#1723).

usage: startup_gfx2hlsl_sload_const_offset_check.py

A title loads a descriptor from the resource heap with the byte offset in a scalar register that an earlier
s_movk filled. The translator must read that register as the literal offset (plus any offset: modifier), so
the result equals the literal form. An unknown register, m0, and the sampler heap keep their old handling.
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

    def resource(self, key, kind):
        return 'r0'


def run(symbols, op, operands, mods=(), branches=False):
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.sym, s.lines = Pipe(), dict(symbols), []
    s.used_s, s.used_v, s.uniform_v, s.exec_known = set(), set(), set(), True
    s.has_branches, s.masks = branches, set()
    try:
        if hasattr(s, 'op_' + op):
            getattr(s, 'op_' + op)(operands.split(', '), list(mods))
        else:
            s.one(op, '%s %s' % (op, operands))
    except G.Unsupported as error:
        return str(error)
    return s.sym


HEAP = {4: ('ptr', ('resheap',)), 9: ('const', 0x40), 10: ('ud', 0)}
want = [('desc', ('resheap',), 0x40, i) for i in range(8)]

got = run(HEAP, 's_load_dwordx8', 's[16:23], s[4:5], s9')
check('constant register offset reaches the heap descriptor',
      isinstance(got, dict) and [got[16 + i] for i in range(8)] == want, repr(got))
literal = run(HEAP, 's_load_dwordx8', 's[16:23], s[4:5], 0x40')
check('same as the literal offset', isinstance(got, dict) and isinstance(literal, dict)
      and [got[16 + i] for i in range(8)] == [literal[16 + i] for i in range(8)])
got = run(HEAP, 's_load_dwordx8', 's[16:23], s[4:5], s9', ['offset:0x20'])
check('offset: modifier adds to the register', isinstance(got, dict) and got[16] == ('desc', ('resheap',), 0x60, 0), repr(got))
got = run(HEAP, 's_load_dwordx8', 's[16:23], s[4:5], s10')
check('unknown register still refused', got == "s_load with a register offset through ('resheap',)", repr(got))
got = run(HEAP, 's_load_dwordx8', 's[16:23], s[4:5], m0')
check('m0 still refused', got == "s_load with a register offset through ('resheap',)", repr(got))

got = run(HEAP, 's_load_dwordx8', 's[16:23], s[4:5], s9', branches=True)
check('constant offset in a shader with branches is refused', got == "s_load with a register offset through ('resheap',)", repr(got))
got = run({}, 's_movk_i32', 's9, 0x8000')
check('s_movk_i32 sign-extends its 16-bit immediate', isinstance(got, dict) and got[9] == ('const', 0xffff8000), repr(got))
got = run({4: ('ptr', ('resheap',)), 9: ('const', 0xffff8000)}, 's_load_dwordx8', 's[16:23], s[4:5], s9')
check('a negative movk offset is refused', got == "s_load with a register offset through ('resheap',)", repr(got))
got = run({}, 's_movk_i32', 's9, 0x7fff')
check('positive movk immediate unchanged', isinstance(got, dict) and got[9] == ('const', 0x7fff), repr(got))
got = run({}, 's_mov_b32', 's9, 0x8000')
check('s_mov_b32 0x8000 is not sign-extended', isinstance(got, dict) and got[9] == ('const', 0x8000), repr(got))

got = run(HEAP, 's_load_dwordx8', 's[16:23], s[4:5], s9', ['offset:0xfffe0'])
check('a constant plus modifier above 0xfffff is refused', got == 'SMEM register offset 0x100020', repr(got))
got = run({4: ('ptr', ('resheap',)), 9: ('const', 0, 'unassigned')}, 's_load_dwordx8', 's[16:23], s[4:5], s9')
check('the unassigned placeholder is not a constant offset',
      got == "s_load with a register offset through ('resheap',)", repr(got))
got = run({4: ('ptr', ('smpheap',)), 9: ('const', 0x40)}, 's_load_dwordx8', 's[16:23], s[4:5], s9')
check('a sampler heap register offset keeps its old handling', not (isinstance(got, dict) and got.get(16, (None,))[0] == 'desc'
      and got[16][2] == 0x40), repr(got))

print('s_load constant offset checks: %d failed' % len(failures))
sys.exit(1 if failures else 0)
