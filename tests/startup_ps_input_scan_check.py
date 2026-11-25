"""Check tools/startup/ps_input_scan.py on synthetic pixel shaders and synthetic request directories.

Every hand-encoded word sequence is disassembled and compared with the text it is meant to be, so a
wrong encoding cannot pass silently. The corpora are built here; no capture or title data is used.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from unittest import mock

from startup_capture_check import chunk, mem
from startup_image_3d_check import VS_WORDS, G, root_blob

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import ps_input_scan as S  # noqa: E402

TOOL = HERE.parent / 'tools' / 'startup' / 'ps_input_scan.py'
END = 0xbf810000


def w(*words):
    return struct.pack('<%dI' % len(words), *words)


# gfx1013 encodings (VOP1, VOP2, VOPC, VINTRP, SDWA, VOP3 and exp).
def vop2(op, vdst, vsrc1, src0):
    return (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0


def vop1(op, vdst, src0):
    return 0x7e000000 | (vdst << 17) | (op << 9) | src0


def vintrp(op, vdst, vsrc, attr):
    return 0xc8000000 | (vdst << 18) | (op << 16) | (attr << 10) | vsrc


def v(n):
    return 256 + n


def and_low(vdst, src):          # v_and_b32 vdst, 0xffff, src
    return w(vop2(0x1b, vdst, src, 255), 0xffff), 'v_and_b32_e32 v%d, 0xffff, v%d' % (vdst, src)


def shr_high(vdst, src):         # v_lshrrev_b32 vdst, 16, src
    return w(vop2(0x16, vdst, src, 144)), 'v_lshrrev_b32_e32 v%d, 16, v%d' % (vdst, src)


def sdwa_low(vdst, src):         # v_and_b32_sdwa, WORD_0 of src0
    return (w(vop2(0x1b, vdst, src, 249), src | (6 << 8) | (4 << 16) | (6 << 24)),
            'v_and_b32_sdwa v%d, v%d, v%d dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 '
            'src1_sel:DWORD' % (vdst, src, src))


def sdwa_high(vdst, src):        # v_mov_b32_sdwa, WORD_1 of src0
    return (w(vop1(1, vdst, 249), src | (6 << 8) | (5 << 16)),
            'v_mov_b32_sdwa v%d, v%d dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1' % (vdst, src))


def bfe(vdst, src, offset, width):
    return (w(0xd4000000 | (0x148 << 16) | vdst, v(src) | ((128 + offset) << 9) | ((128 + width) << 18)),
            'v_bfe_u32 v%d, v%d, %d, %d' % (vdst, src, offset, width))


def mov(vdst, src):
    return w(vop1(1, vdst, v(src))), 'v_mov_b32_e32 v%d, v%d' % (vdst, src)


def mov_zero(vdst):
    return w(vop1(1, vdst, 128)), 'v_mov_b32_e32 v%d, 0' % vdst


def add(vdst, a, b):
    return w(vop2(0x03, vdst, b, v(a))), 'v_add_f32_e32 v%d, v%d, v%d' % (vdst, a, b)


def cmp_lt(a, b):
    return w(0x7c000000 | (1 << 17) | (b << 9) | v(a)), 'v_cmp_lt_f32_e32 vcc_lo, v%d, v%d' % (a, b)


def interp(vdst, src, attr, second):
    return (w(vintrp(1 if second else 0, vdst, src, attr)),
            'v_interp_p%d_f32_e32 v%d, v%d, attr%d.x' % (2 if second else 1, vdst, src, attr))


def exp(*regs):
    return (w(0xf800180f, *[sum(r << (8 * i) for i, r in enumerate(regs))]),
            'exp mrt0 v%d, v%d, v%d, v%d done vm' % regs)


def br(name, here, target):
    """s_branch or s_cbranch_scc1 at byte offset `here` to byte offset `target`."""
    opcode = {'s_branch': 0xbf820000, 's_cbranch_scc1': 0xbf850000}[name]
    return w(opcode | (((target - (here + 4)) // 4) & 0xffff)), '%s label_0x%x' % (name, target)


def shader(*items):
    """Assemble (words, expected disassembly) items, then an end; assert the text."""
    code = b''.join(item[0] for item in items) + w(END)
    decoded = [x[3] for x in G.decode(code)]
    expected = [item[1] for item in items] + ['s_endpgm']
    assert decoded == expected, (decoded, expected)
    return code


def contract_of(code):
    return S.contract(code)


def sets(contract):
    return sorted(contract.live_in), sorted(contract.ij_pairs), sorted(contract.split16)


SHADERS = {
    'packed_only@v1': shader(and_low(4, 1), shr_high(5, 1), exp(4, 5, 4, 5)),
    'packed_sdwa@v1': shader(sdwa_low(4, 1), sdwa_high(5, 1), exp(4, 5, 4, 5)),
    'packed_bfe@v1': shader(bfe(4, 1, 0, 16), bfe(5, 1, 16, 16), exp(4, 5, 4, 5)),
    'packed_alias@v1': shader(mov(6, 1), and_low(4, 6), shr_high(5, 6), exp(4, 5, 4, 5)),
    'pair_then_packed@v2': shader(interp(4, 0, 0, False), interp(4, 1, 0, True),
                                  and_low(5, 2), shr_high(6, 2), exp(4, 5, 6, 4)),
    'two_pairs': shader(interp(4, 0, 0, False), interp(4, 1, 0, True),
                        interp(5, 2, 1, False), interp(5, 3, 1, True), exp(4, 5, 4, 5)),
    'no_inputs': shader(mov_zero(0), exp(0, 0, 0, 0)),
    'write_first': shader(mov_zero(1), add(2, 1, 1), exp(2, 2, 2, 2)),
    'cmp_and_exp': shader(cmp_lt(1, 2), exp(3, 4, 3, 4)),
    'packed_misfire': shader(and_low(4, 0), shr_high(5, 0), add(6, 1, 1), exp(4, 5, 6, 6)),
    'overwritten': shader(and_low(4, 1), mov_zero(1), shr_high(5, 1), exp(4, 5, 4, 5)),
    # Control flow; branch targets are byte offsets inside the shader.
    # entry -> B (writes v5, branches back) -> A (reads v5): v5 is loop-carried, not live-in.
    'loop_ambiguous': shader(br('s_branch', 0, 20), add(6, 5, 5), exp(6, 6, 6, 6), br('s_branch', 16, 28),
                             mov_zero(5), br('s_branch', 24, 4)),
    'loop_definite': shader(add(6, 1, 1), br('s_cbranch_scc1', 4, 0), exp(6, 6, 6, 6)),
    'loop_rmw': shader(add(5, 5, 1), br('s_cbranch_scc1', 4, 0), exp(5, 5, 5, 5)),
    # A move inside a skipped block, an interpolation after the join: the alias must not survive.
    'alias_block': shader(br('s_cbranch_scc1', 0, 8), mov(4, 0), interp(7, 4, 0, False),
                          interp(7, 1, 0, True), exp(7, 7, 7, 7)),
    'alias_straight': shader(mov(4, 0), interp(7, 4, 0, False), interp(7, 1, 0, True), exp(7, 7, 7, 7)),
    'unsat_v40': shader(add(6, 40, 40), exp(6, 6, 6, 6)),
}

# The address mask each informative shader needs, and one that satisfies it.
MASKS = {'packed_only@v1': 0x8100, 'packed_sdwa@v1': 0x8080, 'packed_bfe@v1': 0x8200,
         'packed_alias@v1': 0x8100, 'pair_then_packed@v2': 0x8002, 'two_pairs': 0x0012,
         'no_inputs': 0x0004, 'write_first': 0x0001, 'packed_misfire': 0x8100,
         'loop_ambiguous': 0x0001, 'unsat_v40': 0x0001}


def empty_contract():
    return S.Contract(frozenset(), frozenset(), frozenset())


def unit_checks():
    expected = {
        'packed_only@v1': ([1], [], [1]), 'packed_sdwa@v1': ([1], [], [1]),
        'packed_bfe@v1': ([1], [], [1]), 'packed_alias@v1': ([1], [], [1]),
        'pair_then_packed@v2': ([0, 1, 2], [0], [2]), 'two_pairs': ([0, 1, 2, 3], [0, 2], []),
        'no_inputs': ([], [], []), 'write_first': ([], [], []),
        'cmp_and_exp': ([1, 2, 3, 4], [], []), 'packed_misfire': ([0, 1], [], [0]), 'overwritten': ([1], [], []),
        'loop_ambiguous': ([], [], []), 'loop_definite': ([1], [], []), 'loop_rmw': ([1, 5], [], []),
        'alias_block': ([0, 1], [], []), 'alias_straight': ([0, 1], [0], []), 'unsat_v40': ([40], [], []),
    }
    for name, code in SHADERS.items():
        assert sets(contract_of(code)) == expected[name], (name, sets(contract_of(code)))
    assert sorted(contract_of(SHADERS['loop_ambiguous']).ambiguous) == [5]
    assert all(not contract_of(code).ambiguous for name, code in SHADERS.items() if name != 'loop_ambiguous')
    print('loop rule: loop-carried v5 is ambiguous, not live-in; unwritten and read-modify-write '
          'loop registers stay live-in; no alias across a block boundary')
    with mock.patch.object(G, 'decode', side_effect=G.Unsupported('undecodable word at +0x4: fixture')):
        try:
            S.make_record('0', b'', {})
        except S.Skip as skip:
            assert str(skip).startswith('undecodable: undecodable word at +0x4'), skip
        else:
            raise AssertionError('an undecodable shader was not skipped')
    print('contract: %d shapes (and/shift, SDWA, bfe, alias, interp pairs, no inputs, write-first, '
          'cmp/exp sources, overwrite)' % len(SHADERS))

    # Destination rule on decoded text: all-source forms and the read-before-write order.
    def lines(*asm):
        return [(4 * i, 4, a.split()[0], a) for i, a in enumerate(asm)]
    cases = (
        (['v_cmpx_lt_f32_e64 v1, v2, v3'], [1, 2, 3]),
        (['global_store_dword v[1:2], v3, off'], [1, 2, 3]),
        (['buffer_store_dword v4, v5, s[0:3], 0 offen'], [4, 5]),
        (['ds_write_b32 v4, v5'], [4, 5]),
        (['v_swap_b32 v7, v8', 'v_add_f32_e32 v9, v7, v8'], [7, 8]),
        (['v_add_f32_e32 v9, v10, v9'], [9, 10]),
        (['v_mad_f32 v1, |v2|, -v3, 0', 'v_add_f32_e32 v4, v1, v1'], [2, 3]),
        (['v_mov_b32_e32 v1, 0', 'image_sample v[4:7], v[1:2], s[8:15], s[16:19] dmask:0xf'], [2]),
        (['image_sample v[4:7], v[0:1], s[8:15], s[16:19] dmask:0xf'], [0, 1]),
        (['v_and_b32_e32 v2, 0xffff, sext(v1)'], [1]),
        (['exp mrt0 off, off, v6, v7 done vm'], [6, 7]),
        (['exp mrt0 v5, v6, v7, v8 done vm'], [5, 6, 7, 8]),
        (['v_cmpx_lt_f32_sdwa v1, v2 src0_sel:WORD_0 src1_sel:WORD_1'], [1, 2]),
    )
    for asm, live in cases:
        got = sorted(S.contract_of_decoded(lines(*asm)).live_in)
        assert got == live, (asm, got, live)
    print('destination rule: %d decoded-text cases' % len(cases))

    # A flow the text cannot resolve makes every written-later first read ambiguous.
    for flow in ('s_setpc_b64 s[2:3]', 's_branch 0x4', 's_cbranch_join s2'):
        got = S.contract_of_decoded(lines('v_add_f32_e32 v6, v5, v5', flow, 'v_mov_b32_e32 v5, 0'))
        assert 5 in got.ambiguous and 5 not in got.live_in, (flow, got)
    got = S.contract_of_decoded(lines('v_add_f32_e32 v6, v5, v5', 's_branch label_0x8', 'v_mov_b32_e32 v5, 0'))
    assert 5 in got.live_in and not got.ambiguous, got   # forward flow only: the linear rule holds
    print('control flow: unresolved or indirect branches fall back to ambiguous')

    # Compare forms give no SDWA half evidence: with the second half taken from a shift, the
    # register must not count as split.
    for compare in ('v_cmp_lt_f32_sdwa vcc_lo, v1, v2 src0_sel:WORD_0 src1_sel:WORD_1',
                    'v_cmpx_lt_f32_sdwa v1, v2 src0_sel:WORD_0 src1_sel:WORD_1'):
        got = S.contract_of_decoded(lines(compare, 'v_lshrrev_b32_e32 v3, 16, v1'))
        assert sorted(got.live_in) == [1, 2] and not got.split16, (compare, got)
    print('soundness: compare forms record no SDWA half evidence')

    # The VOP3 interpolation spelling takes the same operands.
    e64 = (w(0xd4000000 | (0x200 << 16) | 4, 0 | (256 << 9)) + w(0xd4000000 | (0x201 << 16) | 4, 0 | (257 << 9))
           + w(END))
    asm = [x[3] for x in G.decode(e64)]
    assert asm[:2] == ['v_interp_p1_f32_e64 v4, v0, attr0.x', 'v_interp_p2_f32_e64 v4, v1, attr0.x'], asm
    assert sets(S.contract(e64)) == ([0, 1], [0], []), sets(S.contract(e64))
    print('interpolation: VOP3 spelling gives the same pair')

    # Layout against the translator's own packed-position register, for several masks.
    assert S.layout(0x8100) == (2, {8: 0, 15: 1}) and S.layout(0x8002)[1][15] == 2
    assert S.layout(0x800F)[1][15] == 2 + 2 + 2 + 3 == 9
    assert S.layout(0x0000) == (0, {}) and S.layout(0xffff)[0] == 6 + 3 + 6 + 9
    probe = struct.pack('<III', 0xf800180f, 0, END)
    for addr in (0x8000, 0x8100, 0x8002, 0x800F, 0x8ff1, 0xfff0, 0xffff, 0x8010, 0xc08a):
        translated = G.Pipeline(VS_WORDS, probe, root_blob(), ps_hw_inputs=(addr, addr)).packed_position_vgpr
        assert S.layout(addr)[1][15] == translated, (hex(addr), translated)
    print('layout: bit-15 register equals Pipeline.packed_position_vgpr for 9 masks')

    # consistent(): one pass and one failure per rule.
    c = S.Contract
    ok = S.consistent(c(frozenset({0, 1, 2}), frozenset({0}), frozenset({2})), 0x8002)
    assert ok == (True, ''), ok
    assert S.consistent(c(frozenset(), frozenset(), frozenset()), 0x10000) == (False, 'C0 range')
    assert S.consistent(c(frozenset(), frozenset(), frozenset()), -1)[1] == 'C0 range'
    assert S.consistent(c(frozenset({5}), frozenset(), frozenset()), 0x0001) == (False, 'C1 coverage')
    assert S.consistent(c(frozenset({1}), frozenset({1}), frozenset()), 0x0001) == (False, 'C2 pairs')
    assert S.consistent(c(frozenset({1}), frozenset({0}), frozenset()), 0x0008)[1] == 'C2 pairs'  # pull model
    # The packed (16-bit split) evidence never votes: the same contracts stay consistent.
    split = c(frozenset({0, 1}), frozenset(), frozenset({1}))
    for addr in (0x8000, 0x0100, 0x8100, 0x8200):
        assert S.consistent(c(frozenset({0}), frozenset(), frozenset({1})), addr) == (True, ''), hex(addr)
    assert S.packed_agrees(split, 0x8100) is True and S.packed_agrees(split, 0x8200) is True
    assert S.packed_agrees(split, 0x0100) is False and S.packed_agrees(split, 0x8002) is False
    assert S.packed_agrees(split, 0x10000) is False
    assert S.packed_agrees(c(frozenset({0}), frozenset(), frozenset()), 0x8100) is None
    # The scan's cached verdict codes agree with the plain functions.
    for contract in (split, c(frozenset({0, 1, 2}), frozenset({0}), frozenset({2})), empty_contract(),
                     c(frozenset({5}), frozenset(), frozenset()), c(frozenset({3}), frozenset({2}), frozenset())):
        judge = S.Judge(contract)
        for addr in (0, 1, 2, 0x12, 0x8002, 0x8100, 0x8200, 0xffff, 0x10000, 0xffff0000):
            code = judge.code(addr)
            ok, reason = S.consistent(contract, addr)
            assert (code & S.RULE_MASK == 0) == ok and S.RULES[code & S.RULE_MASK] == reason, (addr, code, reason)
            agrees = S.packed_agrees(contract, addr)
            assert bool(code & S.PACKED_APPLIES) == (agrees is not None)
            assert bool(code & S.PACKED_OK) == bool(agrees)
    print('consistent: pass plus one failure for each of C0, C1, C2; packed evidence does not vote')

    empty = c(frozenset(), frozenset(), frozenset())
    assert S.selectivity(empty) == 1.0 and not S.informative(empty) and S.contract_bits(empty) == 0
    tight = S.contract(SHADERS['two_pairs'])
    assert 0 < S.selectivity(tight) < 0.75 and S.contract_bits(tight) > 0.3
    assert S.selectivity(S.contract(SHADERS['packed_only@v1'])) == S.selectivity(S.Contract(frozenset({1}), frozenset(), frozenset()))
    for name, code in SHADERS.items():
        if name in MASKS and name != 'unsat_v40':
            assert S.consistent(S.contract(code), MASKS[name])[0], (name, hex(MASKS[name]))
    print('selectivity: empty contract 1.0; two-pair contract %.4f (%.2f bits)'
          % (S.selectivity(tight), S.contract_bits(tight)))


# ---- end to end ----------------------------------------------------------------------------------

def blob_with_header(header, code):
    assert len(header) % 4 == 0 and len(header) <= 0x7c and G.PREFETCH not in header
    blob = header + G.PREFETCH + code
    assert G.strip_header(blob) == G.PREFETCH + code
    return blob


def header_words(index, ena, addr, at=2, ones=False):
    words = [0] * 12
    words[0] = 2                       # constant decoy (a version number)
    words[1] = index + 1               # per-shader decoy (a counter)
    words[at], words[at + 1] = ena, addr
    words[8] = 0x0001                  # constant decoy that looks like a small mask
    if ones:
        words[9], words[10] = 0x0000ffff, 0xffff0000   # all-ones decoys: a u32 and a u16 0xffff
    return w(*words)


def vs_blob():
    return VS_WORDS


def write_request(folder, index, ps_blob, swap=False):
    key = '%016x' % (0x1000 + index)
    a, b = (ps_blob, vs_blob()) if swap else (vs_blob(), ps_blob)
    (folder / ('req_%s.a' % key)).write_bytes(a)
    (folder / ('req_%s.b' % key)).write_bytes(b)
    (folder / ('req_%s.root' % key)).write_bytes(root_blob())
    return key


ORDER = ('packed_only@v1', 'pair_then_packed@v2', 'two_pairs', 'packed_sdwa@v1', 'packed_bfe@v1',
         'packed_alias@v1', 'no_inputs', 'write_first')


def build(folder, maker, names=ORDER, repeat=1):
    folder.mkdir()
    index = 0
    for _ in range(repeat):
        for name in names:
            write_request(folder, index, maker(index, name), swap=index % 2 == 1)
            index += 1
    return index


def run_tool(*args, env=None):
    run = subprocess.run([sys.executable, '-B', str(TOOL), *map(str, args)], capture_output=True, text=True,
                         env=env)
    return run


def scan(folder, tmp, label):
    out = tmp / (label + '.json')
    run = run_tool(folder, '--json', out, '--top', 12)
    assert run.returncode == 0, (run.stdout, run.stderr)
    return run.stdout, json.loads(out.read_text())


def by_name(data, region, name):
    for entry in data['candidates']:
        if entry['region'] == region and name in [entry['candidate']] + entry['equivalent']:
            return entry
    raise AssertionError('candidate %s %s not in the JSON' % (region, name))


def simple_maker(index, name, at=2, extra=0):
    mask = MASKS[name]
    ena = mask | (extra if index == 1 else 0)
    return blob_with_header(header_words(index, ena, mask, at), SHADERS[name])


def end_to_end(tmp):
    # Corpus A: masks at header +8 and +12, a constant decoy and a counter decoy.
    text, data = scan(tmp / 'a', tmp, 'a') if build(tmp / 'a', simple_maker) else None
    assert 'pipelines: 8 scanned, 0 skipped' in text, text
    assert 'informative contracts: 6 of 8; distinct contracts' in text, text
    assert 'verdict: unique full match ps.header pair32@8' in text, text
    assert data['verdict'] == 'unique full match ps.header pair32@8'
    top = data['candidates'][0]
    assert (top['region'], top['candidate'], top['inconsistent'], top['full_match']) == \
        ('ps.header', 'pair32@8', 0, True), top
    assert top['relations'] == {'equal': 8, 'ena_subset': 0, 'addr_subset': 0, 'other': 0}, top
    assert top['tight'] == 6 and top['packed_total'] == 5 and top['packed_agree'] == 5, top
    for decoy in ('addr32@0', 'addr32@4', 'addr32@32'):
        entry = by_name(data, 'ps.header', decoy)
        assert entry['rank'] > top['rank'] and entry['inconsistent'] >= 1 and not entry['full_match'], entry
    assert 'mask agreement: 8 of 8 pipelines consistent (coverage and pair rules), 6 tight, '         'failures C0=0 C1=0 C2=0' in text, text
    assert 'packed-position agreement: 5 of 5 pipelines that split a live-in register into 16-bit halves' in text
    assert text.rstrip().endswith('does not prove hardware behaviour'), text
    assert data['top_candidate_pipelines'][0]['addr'] in (0x8100, 0x8002, 0x0012)
    assert len(data['top_candidate_pipelines']) == 8
    print('positive control (corpus A): unique full match ps.header pair32@8; decoy controls +0, +4, +32 rank lower and fail')
    corpus_a = text

    # Corpus B: one enable mask differs from its address mask (an extra bit below bit 15).
    build(tmp / 'b', lambda i, n: simple_maker(i, n, extra=0x0080))
    text, data = scan(tmp / 'b', tmp, 'b')
    assert data['verdict'] == 'unique full match ps.header pair32@8', text
    top = data['candidates'][0]
    assert top['relations'] == {'equal': 7, 'ena_subset': 0, 'addr_subset': 1, 'other': 0}, top
    print('enable/address disagreement (corpus B): same unique full match; relations show 1 enable mask that differs from its address')

    # Corpus C: masks at +8/+12 in half of the shaders, at +16/+20 in the other half.
    build(tmp / 'c', lambda i, n: blob_with_header(
        header_words(i, MASKS[n], MASKS[n], at=2 if i % 2 == 0 else 4), SHADERS[n]), repeat=2)
    text, data = scan(tmp / 'c', tmp, 'c')
    assert data['verdict'] == 'no location is consistent with all pipelines', text
    assert not any(entry['full_match'] for entry in data['candidates'])
    print('wrong-location control (corpus C): masks at two different offsets give no location consistent with all pipelines')

    # Corpus D: tagged byte addresses at different offsets per shader.
    def tagged(index, name):
        words = [0] * 28
        words[0] = 2
        words[1] = index + 1
        first, second = 2 + 2 * (index % 4), 14 + 2 * ((index // 2) % 4)
        words[first:first + 2] = [0x000286CC, MASKS[name]]
        words[second:second + 2] = [0x000286D0, MASKS[name]]
        return blob_with_header(w(*words), SHADERS[name])
    build(tmp / 'd', tagged, repeat=2)
    text, data = scan(tmp / 'd', tmp, 'd')
    assert data['verdict'] == 'unique full match ps.header tag:byteaddr', text
    print('corpus D: unique full match ps.header tag:byteaddr with varying word offsets')

    # Corpus E: only shaders without live-in registers.
    build(tmp / 'e', simple_maker, names=('no_inputs', 'write_first'), repeat=4)
    text, data = scan(tmp / 'e', tmp, 'e')
    assert 'corpus is uninformative' in text and 'unique full match' not in text, text
    assert data['verdict'].startswith('uninformative corpus') and data['informative'] == 0
    print('corpus E: uninformative corpus reported, no match claimed')

    # Corpus F: skipped entries are counted with their reason and do not stop the scan.
    build(tmp / 'f', simple_maker)
    (tmp / 'f' / 'req_00000000000fffff.a').write_bytes(vs_blob())     # two vertex shaders
    (tmp / 'f' / 'req_00000000000fffff.b').write_bytes(vs_blob())
    (tmp / 'f' / 'req_00000000000ffffe.a').write_bytes(vs_blob())     # no .b file
    (tmp / 'f' / 'req_00000000000ffffd.a').write_bytes(vs_blob())     # a broken container
    (tmp / 'f' / 'req_00000000000ffffd.b').write_bytes(b'DXBC' + bytes(60))
    text, data = scan(tmp / 'f', tmp, 'f')
    assert 'pipelines: 8 scanned, 3 skipped' in text, text
    for category in ('stages', 'unreadable', 'container'):
        assert 'skipped %s: 1' % category in text, text
    assert data['verdict'] == 'unique full match ps.header pair32@8'
    print('corpus F: stage, unreadable and container failures skipped with reasons, scan continues')

    # All-ones decoy: 0x0000ffff (u32) and 0xffff (u16) in every header fit every contract, so they
    # are a full match too; they must lose to the planted location on tight.
    build(tmp / 'g', lambda i, n: blob_with_header(
        header_words(i, MASKS[n], MASKS[n], ones=True), SHADERS[n]))
    text, data = scan(tmp / 'g', tmp, 'g')
    assert data['verdict'] == 'unique full match ps.header pair32@8', text
    top = data['candidates'][0]
    ones = by_name(data, 'ps.header', 'addr32@36')
    assert 'addr16@42' in [ones['candidate']] + ones['equivalent'], ones
    assert ones['full_match'] and ones['tight'] == 0 and ones['rank'] > top['rank'], ones
    assert top['tight'] == 6 and top['full_match'], top
    print('decoy control (all-ones u32 and u16 words): also a full match with tight 0; the planted '
          'location wins on tight')

    # Packed misfire: one shader splits v0 into halves while its mask puts bit 15 at v1. Packed
    # evidence does not vote, so the planted location stays the unique full match; the packed line
    # shows the one disagreement.
    build(tmp / 'h', simple_maker, names=ORDER + ('packed_misfire',))
    text, data = scan(tmp / 'h', tmp, 'h')
    assert data['verdict'] == 'unique full match ps.header pair32@8', text
    top = data['candidates'][0]
    assert (top['packed_total'], top['packed_agree']) == (6, 5), top
    assert 'packed-position agreement: 5 of 6 pipelines' in text, text
    assert 'mask agreement: 9 of 9 pipelines consistent' in text, text
    print('packed misfire control: planted location stays the unique full match; packed line 5 of 6')

    # Loop-carried register and an unsatisfiable contract: neither may break the planted match.
    build(tmp / 'i', simple_maker, names=ORDER + ('loop_ambiguous', 'unsat_v40'))
    text, data = scan(tmp / 'i', tmp, 'i')
    assert data['verdict'] == 'unique full match ps.header pair32@8', text
    assert 'pipelines: 10 scanned, 0 skipped' in text and 'informative contracts: 6 of 9' in text, text
    assert ('ambiguous live-in reads (inside a loop, written later): 1 pipelines, excluded registers '
            'do not vote') in text, text
    assert 'contracts no mask can satisfy: 1 pipelines (excluded)' in text, text
    assert len(data['excluded_unsatisfiable']) == 1 and len(data['ambiguous_pipelines']) == 1, data
    print('loop control: a loop-carried register is ambiguous and the planted location stays the unique '
          'full match; an unsatisfiable contract (v40) is excluded and counted')

    # Packed lens: two full matches tie on tight; only the real one agrees on every packed split.
    alternatives = {'pair_then_packed@v2': 0x0102, 'packed_only@v1': 0x0300, 'two_pairs': 0x0012,
                    'no_inputs': 0x0004}

    def lens_maker(index, name):
        words = list(struct.unpack('<12I', header_words(index, MASKS[name], MASKS[name], ones=True)))
        words[4] = words[5] = alternatives[name]
        return blob_with_header(w(*words), SHADERS[name])
    build(tmp / 'j', lens_maker, names=tuple(alternatives))
    text, data = scan(tmp / 'j', tmp, 'j')
    assert data['verdict'].startswith('2 full matches (ambiguous): ps.header pair32@8, ps.header pair32@16'), text
    assert 'full matches that also agree on every packed split: ps.header pair32@8\n' in text + '\n', text
    print('packed lens: 2 full matches tie on tight; only the real location agrees on every packed split')

    # Sibling request files become regions req.<ext>.
    folder = tmp / 'k'
    build(folder, lambda i, n: blob_with_header(header_words(i, 0, 0), SHADERS[n]))
    for index, name in enumerate(ORDER):
        key = '%016x' % (0x1000 + index)
        (folder / ('req_%s.desc' % key)).write_bytes(w(0, 0, MASKS[name], MASKS[name], 0x0001, index))
    (folder / ('req_%016x.big' % 0x1000)).write_bytes(bytes(S.SIBLING_LIMIT + 1))
    (folder / ('req_%016x.rtv' % 0x1000)).write_bytes(b'rtv1 1 2')
    records, skipped = [], []
    S.scan_requests(str(folder), records, skipped)
    assert not skipped and sorted(records[0].regions) == ['ps.header', 'req.desc'], sorted(records[0].regions)
    text, data = scan(folder, tmp, 'k')
    assert data['verdict'] == 'unique full match req.desc pair32@8', text
    print('sibling files: masks in req_<key>.desc give a unique full match in region req.desc; '
          'oversized and reserved extensions are not regions')

    pixel_dir_cases(tmp)

    container_case(tmp)
    capture_case(tmp)

    determinism(tmp)
    count = reference_check(tmp / 'a') + reference_check(tmp / 'd') + reference_check(tmp / 'h')
    print('reference: %d candidate summaries equal a direct per-candidate evaluation' % count)

    # Usage errors exit 2; an empty directory still completes.
    assert run_tool().returncode == 2
    assert run_tool(tmp / 'missing').returncode == 2
    assert run_tool(tmp / 'a', '--top', 0).returncode == 2
    (tmp / 'empty').mkdir()
    assert run_tool(tmp / 'empty').returncode == 0
    print('usage: missing input, missing directory and bad --top exit 2; empty directory exits 0')
    return corpus_a


def reference_check(folder, limit=0x1000):
    """Compare the two-pass tallies with a direct per-candidate evaluation of every pipeline."""
    records, skipped = [], []
    S.scan_requests(str(folder), records, skipped)
    assert records and not skipped
    reference = {}
    for record in records:
        shader = record.contract
        for region, data in record.regions.items():
            words, halves = S.unpack(data[:limit])
            tags = S.tag_values(words)
            names = [(k, o) for k in ('addr32', 'pair32', 'pair32r') for o in range(0, len(words) * 4, 4)]
            names += [(k, o) for k in ('addr16', 'pair16', 'pair16r') for o in range(0, len(halves) * 2, 2)]
            names += [(k, -1) for k in tags]
            for kind, offset in names:
                value = S.candidate_value(words, halves, tags, kind, offset)
                if value is None:
                    continue
                addr = value[1]
                ok, reason = S.consistent(shader, addr)
                item = reference.setdefault((region, kind, offset), dict(
                    present=0, consistent=0, tight=0, packed_total=0, packed_agree=0, fails=[0, 0, 0], bits=0.0))
                item['present'] += 1
                item['consistent'] += ok
                if ok and S.informative(shader):
                    item['bits'] += S.contract_bits(shader)
                    if S.layout(addr)[0] == max(shader.live_in) + 1:
                        item['tight'] += 1
                if not ok:
                    item['fails'][{'C0': 0, 'C1': 1, 'C2': 2}[reason[:2]]] += 1
                agrees = S.packed_agrees(shader, addr)
                if agrees is not None:
                    item['packed_total'] += 1
                    item['packed_agree'] += agrees
    stats = S.candidate_stats(*S.tally(records, limit))
    assert len(stats) == len(reference), (len(stats), len(reference))
    for stat in stats:
        item = reference[(stat.region, stat.kind, stat.offset)]
        got = dict(present=stat.present, consistent=stat.consistent, tight=stat.tight,
                   packed_total=stat.packed_total, packed_agree=stat.packed_agree, fails=stat.fails)
        assert got == {k: item[k] for k in got} and abs(stat.bits - item['bits']) < 1e-9, (stat.name, got, item)
    return len(stats)


def pixel_dir_cases(tmp):
    """--pixel-dir: raw blobs and containers; ids are positions, file names never printed."""
    secret = 'localname'
    raw = tmp / 'pd_raw'
    raw.mkdir()
    for index, name in enumerate(ORDER):
        (raw / ('%s_%02d.bin' % (secret, index))).write_bytes(simple_maker(index, name))
    (raw / ('%s_99.bin' % secret)).write_bytes(vs_blob())             # a vertex shader: skipped
    (raw / 'subdirectory').mkdir()                                      # not a regular file: ignored
    out = tmp / 'pd_raw.json'
    run = run_tool('--pixel-dir', raw, '--json', out)
    assert run.returncode == 0, run.stderr
    data = json.loads(out.read_text())
    assert data['verdict'] == 'unique full match ps.header pair32@8', run.stdout
    assert 'pipelines: 8 scanned, 1 skipped' in run.stdout and 'skipped stages: 1' in run.stdout, run.stdout
    assert [c['id'] for c in data['contracts']] == ['p%06d' % n for n in range(1, 9)], data['contracts']
    assert data['skipped'][0]['id'] == 'p000009', data['skipped']
    assert secret not in run.stdout and secret not in out.read_text()
    assert 'p0000' not in run.stdout, run.stdout       # aggregates only without --list
    listed = run_tool('--pixel-dir', raw, '--list')
    assert 'p000001 0x8100 0x8100 1 v1 v1' in listed.stdout and secret not in listed.stdout, listed.stdout
    assert 'id ena addr packed_register live_in split16' in listed.stdout
    print('pixel-dir: raw blobs give ids p000001..p000008, no file name in stdout or JSON; --list '
          'adds the per-pipeline table, default output has no ids')

    # Containers with the masks in the PSV0 part; the image head holds all-ones decoys that sort first.
    boxes = tmp / 'pd_box'
    boxes.mkdir()
    for index, name in enumerate(ORDER):
        mask = MASKS[name]
        blob = container(SHADERS[name], (0xffff, 0xffff), index + 1,
                         extra=[(b'PSV0', struct.pack('<2I', mask, mask) + bytes(8)),
                                (b'ISG1', struct.pack('<4I', index, 0x0001, 7, 0))])
        (boxes / ('%s_%02d.dxbc' % (secret, index))).write_bytes(blob)
    run = run_tool('--pixel-dir', boxes, '--json', tmp / 'pd_box.json')
    data = json.loads((tmp / 'pd_box.json').read_text())
    assert data['verdict'] == 'unique full match ps.part.PSV0 pair32@0', run.stdout
    first = data['candidates'][0]
    assert (first['region'], first['tight']) == ('ps.part.PSV0', 6), first
    ones = by_name(data, 'ps.fssm.head', 'addr32@64')           # all-ones decoy in an earlier region
    assert ones['full_match'] and ones['tight'] == 0 and ones['rank'] > first['rank'], ones
    regions = {c['region'] for c in data['candidates']}
    assert {'ps.part.PSV0', 'ps.part.ISG1', 'ps.fssm.head', 'ps.xdat.7'} <= regions, regions
    assert secret not in run.stdout and secret not in (tmp / 'pd_box.json').read_text()
    print('pixel-dir: containers give a ps.part.PSV0 location; ps.part.ISG1, ps.fssm.head and '
          'ps.xdat.7 regions are scanned')
    both = run_tool(tmp / 'a', '--pixel-dir', raw)
    assert both.returncode == 0 and 'pipelines: 16 scanned, 1 skipped' in both.stdout, both.stdout
    print('inputs: a request directory and --pixel-dir combine')


def determinism(tmp):
    outputs = []
    for seed in ('1', '2', '12345'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(run_tool(tmp / 'a', '--top', 50, env=env).stdout)
    assert outputs[0] == outputs[1] == outputs[2] and outputs[0].count(chr(10)) > 10
    print('determinism: identical stdout under three PYTHONHASHSEED values')


def container(code, masks, decoy, extra=()):
    """A DXBC container (DXIL part, XDAT table with a native FSSM entry and one more entry), plus
    any extra (tag, payload) parts after the DXIL part."""
    image = bytearray(256) + code
    image[:4] = b'FSSM'
    struct.pack_into('<I', image, 192, 256 - 40)
    struct.pack_into('<2I', image, 64, *masks)            # ena, addr in the image head
    native = b'XBCH' + bytes(12) + struct.pack('<II', len(image), 0) + zlib.compress(bytes(image))
    other = struct.pack('<2I', 0x0001, decoy)
    table = 24 + 12 + 2 * 12
    entries = struct.pack('<3I', 13, table, len(native)) + struct.pack('<3I', 7, table + len(native), len(other))
    program = bytes(24) + b'XDAT' + struct.pack('<II', 2, 0) + entries + native + other
    sections = [b'DXIL' + struct.pack('<I', len(program)) + program]
    sections += [tag + struct.pack('<I', len(payload)) + payload for tag, payload in extra]
    offset = 32 + 4 * len(sections)
    offsets = b''
    for section in sections:
        offsets += struct.pack('<I', offset)
        offset += len(section)
    return (b'DXBC' + bytes(16) + struct.pack('<3I', 1, offset, len(sections)) + offsets + b''.join(sections))


def container_case(tmp):
    folder = tmp / 'ctr'
    folder.mkdir()
    for index, name in enumerate(ORDER):
        mask = MASKS[name]
        blob = container(SHADERS[name], (mask, mask), index + 1)
        key = '%016x' % (0x2000 + index)
        (folder / ('req_%s.a' % key)).write_bytes(blob)      # the scan reads only the pixel container
        (folder / ('req_%s.b' % key)).write_bytes(blob)
    text, data = scan(folder, tmp, 'ctr')
    assert data['verdict'] == 'unique full match ps.fssm.head pair32@64', text
    regions = {entry['region'] for entry in data['candidates']}
    assert {'ps.fssm.head', 'ps.xdat.7'} <= regions, regions
    print('container: unique full match ps.fssm.head pair32@64 in a synthetic DXBC pixel container')


def capture_case(tmp):
    """A synthetic capture: path-57 pipelines whose first state block holds the two masks."""
    data = chunk(b'FRAM', struct.pack('<II', 1, 0))
    for index, name in enumerate(ORDER):
        base = 0x1000000 * (index + 1)
        ps = G.PREFETCH + SHADERS[name]
        mask = MASKS[name]
        data += mem(base, VS_WORDS, 1) + mem(base + 0x1000, ps, 1) + mem(base + 0x2000, w(mask, mask), 1)
        desc = struct.pack('<12Q', 0, base + 0x2000, 8, 0, 0, base, len(VS_WORDS),
                           0, 0, base + 0x1000, len(ps), 0)
        wrapper = 0x5500 + index
        data += chunk(b'PSO ', struct.pack('<QQII', wrapper, 0, 57, 1), desc)
        data += chunk(b'CALL', struct.pack('<IIIIQ7QQ', 1, 2 + index, 1, 25, 0x1000, wrapper,
                                           0, 0, 0, 0, 0, 0, 0))
    capture = tmp / 'capture.bin'
    capture.write_bytes(data)
    out = tmp / 'capture.json'
    run = run_tool('--capture', capture, '--json', out)
    assert run.returncode == 0, (run.stdout, run.stderr)
    result = json.loads(out.read_text())
    assert result['verdict'] == 'unique full match pso.state_a pair32@0', run.stdout
    assert 'pipelines: 8 scanned, 0 skipped' in run.stdout, run.stdout
    print('capture: unique full match pso.state_a pair32@0 in a synthetic capture of 8 pipelines')


def main():
    unit_checks()
    with tempfile.TemporaryDirectory() as folder:
        report = end_to_end(Path(folder))
    print('corpus A report:')
    print(report)
    print('ps_input_scan checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
