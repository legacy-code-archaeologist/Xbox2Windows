"""Check s_load into s[104:107] in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_sload_s104_check.py

s106 and s107 are vcc. A descriptor load (a table pointer plus an immediate offset) emits no code and only
records which table bytes the destination registers name, so it may end in s[104:107]. A load that must
write the registers (a root constant buffer) or that starts above s105 is still refused.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402


class Pipe:
    sampler_heap = False

    def resource(self, key, kind):
        return 'r0'


def run(symbols, op, operands):
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.sym, s.lines = Pipe(), dict(symbols), []
    s.used_s, s.used_v, s.uniform_v, s.exec_known = set(), set(), set(), True
    try:
        getattr(s, 'op_' + op)(operands.split(', '), [])
    except G.Unsupported as error:
        return str(error)
    return s


def translate(*instructions):
    stage = G.Stage.__new__(G.Stage)
    stage.pipe, stage.sym, stage.lines = Pipe(), dict(TABLE), []
    stage.kind, stage.attrs = 'ps', {}
    stage.used_s, stage.used_v, stage.masks = set(), set(), set()
    stage.exec_known = True
    stage.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
    stage.translate()
    return stage


TABLE = {4: ('ptr', ('table', 0))}
ROOT = {4: ('ptr', ('root', 1))}


def main():
    s = run(TABLE, 's_load_dwordx4', 's[104:107], s[4:5], 0x20')
    assert not isinstance(s, str), s
    assert [s.sym[104 + i] for i in range(4)] == [('desc', ('table', 0), 0x20, i) for i in range(4)], s.sym
    assert s.lines == [], s.lines
    s = run(TABLE, 's_load_dwordx8', 's[100:107], s[4:5], 0x20')
    assert not isinstance(s, str), s
    for bad, symbols in (('s_load_dwordx4 s[104:107], s[4:5], 0x20', ROOT),
                         ('s_load_dwordx2 s[106:107], s[4:5], 0x20', TABLE),
                         ('s_load_dwordx2 s[105:106], s[4:5], 0x20', TABLE),
                         ('s_load_dwordx4 s[105:108], s[4:5], 0x20', TABLE)):
        op, operands = bad.split(' ', 1)
        message = run(symbols, op, operands)
        assert isinstance(message, str) and 's_load destination' in message, (bad, message)
    # The loaded descriptor replaced vcc: reads are refused until vcc is written again.
    s = run(TABLE, 's_load_dwordx4', 's[104:107], s[4:5], 0x20')
    for read in (lambda: s.u('vcc_lo'), lambda: s.emit('v0 = (vcc & 1) ? 1u : 0u;')):
        try:
            read()
        except G.Unsupported as error:
            assert 'vcc read' in str(error), error
        else:
            raise AssertionError('vcc read accepted after s_load into s[104:107]')
    assert s.descriptor('s[104:107]') == (('table', 0), 0x20)
    s.emit('vcc = (v1 == 0u) & 1;')
    assert s.u('vcc_lo') == 'vcc'
    try:
        s.descriptor('s[104:107]')
    except G.Unsupported as error:
        assert 'overlaps a redefined vcc' in str(error), error
    else:
        raise AssertionError('descriptor still accepted after its VCC word was clobbered')
    # The branch jumps over the compare, so its target can read the descriptor word as VCC.
    load = 's_load_dwordx4 s[104:107], s[4:5], 0x20'
    compare = 'v_cmp_eq_u32_e32 0, v1'
    read = 'v_cndmask_b32_e32 v0, 0, 1'
    assert translate(load, compare, read, 's_endpgm').u('vcc_lo') == 'vcc'
    try:
        translate(load, 's_cbranch_scc0 label_000C', compare, read, 's_endpgm')
    except G.Unsupported as error:
        assert 'vcc read after a descriptor load into s106/s107' in str(error), error
    else:
        raise AssertionError('branch skipped the only VCC redefinition')
    translate(load, compare, 's_cbranch_scc0 label_0010', 's_nop 0', read, 's_endpgm')
    # The load may be bypassed on one path; the resource would then have no matching descriptor.
    fetch = 'buffer_load_dword v0, off, s[104:107], 0'
    translate(load, 's_cbranch_scc0 label_000C', 's_nop 0', fetch, 's_endpgm')
    try:
        translate('s_cbranch_scc0 label_000C', load, 's_nop 0', fetch, 's_endpgm')
    except G.Unsupported as error:
        assert ('without a matching load on every path' in str(error)
                or 'joined at a branch target' in str(error)), error
    else:
        raise AssertionError('branch skipped the only descriptor load')
    s = run(TABLE, 's_load_dwordx4', 's[100:103], s[4:5], 0x20')
    assert s.u('vcc_lo') == 'vcc'
    print('ok')


main()
