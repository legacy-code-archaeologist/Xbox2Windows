"""Local diagnostic (branch local/intro-shaders only, not for upstream): why do two refusal groups occur?

1. "... joined at a branch target": for the register named in the refusal, how the paths into the join differed.
2. "s_nand_b32/s_nor_b32 mixes a lane mask with an unprojected scalar word": what the non-mask operand was.
3. "v_writelane_b32/v_readlane_b32 needs straight-line code": where the lane writes and reads sit relative to the
   branches (before the first branch, inside a loop, inside an s_cbranch_execz block).

Read-only on the cache, like --refusals. The output holds only aggregate counts, symbol kinds and opcode names:
no keys, paths, offsets, operands, disassembly or generated HLSL.

    python -B tools/startup/local_join_mask_probe.py <copy of shader_cache> > probe.json
"""
import collections
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, 'gfx2hlsl.py')
JOIN_LINE = "self.sym[reg] = ('joined at a branch target',)"


def load_patched():
    text = open(SOURCE, encoding='utf-8').read()
    if text.count(JOIN_LINE) != 2:
        raise SystemExit('probe out of date: expected 2 join refusal sites in gfx2hlsl.py, found %d'
                         % text.count(JOIN_LINE))
    text = text.replace(JOIN_LINE, "self.sym[reg] = _probe_join(self, reg, arriving, paths, force)")
    module = types.ModuleType('gfx2hlsl')
    module.__file__ = SOURCE
    module._probe_join = _probe_join
    sys.modules['gfx2hlsl'] = module
    exec(compile(text, SOURCE, 'exec'), module.__dict__)
    return module


G = None
EVENTS = {}       # reg -> class of the last join that made it 'joined' in the current request
NAND = []         # operand classes of the current request's mixed s_nand/s_nor refusals


def kind(sym):
    if sym is None:
        return 'none'
    k = sym[0]
    if k == 'const':
        v = sym[1] & 0xffffffff
        return 'const:%s%s' % ({0: '0', 0xffffffff: '-1'}.get(v, 'other'), '/unassigned' if len(sym) > 2 else '')
    if k == 'desc':
        src = sym[1][0] if isinstance(sym[1], tuple) and sym[1] else type(sym[1]).__name__
        return 'desc:%s%s' % (src, '/' + str(sym[4]) if len(sym) > 4 else '')
    if k in ('ptr', 'ptrhi'):
        src = sym[1][0] if isinstance(sym[1], tuple) and sym[1] else type(sym[1]).__name__
        return '%s:%s%s' % (k, src, '/added' if len(sym) > 2 else '')
    return k


def _probe_join(self, reg, arriving, paths, force):
    unwritten = sum(1 for _, w in arriving if reg not in w)
    kinds = sorted({kind(p) for p in paths})
    if unwritten:
        cls = 'written on %d of %d paths; written kinds %s' % (len(arriving) - unwritten, len(arriving), kinds)
    elif force == 'refuse':
        cls = 'loop back edge brings a different symbol; kinds %s' % kinds
    elif paths and all(p == paths[0] for p in paths):
        cls = 'equal symbols that do not name one value: %s' % kinds
    else:
        descs = [p for p in paths if p and p[0] == 'desc' and len(p) >= 4]
        if len(descs) == len(paths):
            same_src = len({repr(p[1]) for p in descs}) == 1
            same_off = len({p[2] for p in descs}) == 1
            same_dw = len({p[3] for p in descs}) == 1
            cls = 'different descriptors: source %s, load offset %s, dword %s, modified %s' % (
                'same' if same_src else 'differs', 'same' if same_off else 'differs',
                'same' if same_dw else 'differs', any(len(p) > 4 for p in descs))
        else:
            cls = 'different symbols: kinds %s' % kinds
    EVENTS[reg] = '%s: %s' % (getattr(self, 'kind', '?'), cls)
    return ('joined at a branch target',)


def last_writer(self, reg):
    """Opcode of the last instruction before the current one that writes SGPR reg (or 'entry')."""
    here = [i for i, ins in enumerate(self.ins) if ins[0] == self.pc]
    stop = here[0] if here else len(self.ins)
    for off, _, name, asm in reversed(self.ins[:stop]):
        try:
            dests = G.scalar_destinations(name, asm)
        except Exception:
            continue
        if reg in dests or 'any' in dests:
            return name
    return 'entry'


def operand_class(self, tok):
    if self.is_lane_mask(tok):
        return 'mask'
    r = G.reg_range(tok)
    if r and r[0] == 's':
        sym = self.sym.get(r[1])
        return 'sgpr sym=%s writer=%s' % (kind(sym), last_writer(self, r[1]))
    try:
        v = int(tok, 0) & 0xffffffff
        return 'literal %s' % {0: '0', 0xffffffff: '-1'}.get(v, 'other')
    except ValueError:
        return 'token %s' % re.sub(r'\d+', 'N', tok)


def wrap_scalar(original):
    def scalar(self, base, ops, *rest, **kw):
        if base in ('s_nand_b32', 's_nor_b32'):
            def low(t):
                r = G.reg_range(t)
                return ('s%d' % r[1]) if r and r[0] == 's' and r[1] != r[2] else t
            operands = [low(t) for t in ops]
            if any(self.is_lane_mask(t) for t in operands) and not all(self.is_lane_mask(t) for t in operands[1:]):
                dest = 'dest mask' if self.is_lane_mask(operands[0]) else 'dest word'
                NAND.append('%s %s; %s; srcs %s' % (getattr(self, 'kind', '?'), base, dest,
                                                     ' | '.join(operand_class(self, t) for t in operands[1:])))
        return original(self, base, ops, *rest, **kw)
    return scalar


SPILL = []       # shape classes of the current request's 'needs straight-line code' spill refusals


def spill_shape(self, vreg):
    """Where the lane writes and reads of v<vreg> sit relative to the program's branches (no offsets kept)."""
    ins = getattr(self, 'ins', [])
    branches = []
    for off, _, name, asm in ins:
        if name.startswith(('s_cbranch', 's_branch')):
            try:
                to = int(G.parse(asm)[1][0].split('_')[1], 16)
            except Exception:
                to = None
            branches.append((off, to, name))
    first_ctrl = min([b for b, _, _ in branches] + [t for _, t, _ in branches if t is not None] + [1 << 30])
    writes, reads, vector = [], [], False
    for off, _, name, asm in ins:
        base = re.sub(r'_e(32|64)$', '', name)
        hit = any(int(lo or s) <= vreg <= int(hi or s)
                  for lo, hi, s in re.findall(r'\bv\[(\d+):(\d+)\]|\bv(\d+)\b', asm))
        if not hit:
            continue
        if base == 'v_writelane_b32':
            writes.append(off)
        elif base == 'v_readlane_b32':
            reads.append(off)
        else:
            vector = True
    loops = [(t, b) for b, t, _ in branches if t is not None and t <= b]
    execz = [(b, t) for b, t, n in branches if n == 's_cbranch_execz' and t is not None and t > b]
    flags = [
        'writes %s first branch' % ('all before' if writes and max(writes) < first_ctrl else 'not all before'),
        'reads %s last write' % ('all after' if writes and reads and min(reads) > max(writes) else 'not all after'),
        'write in loop' if any(t <= w <= b for w in writes for t, b in loops) else 'no write in loop',
        'write in execz block' if any(b < w < t for w in writes for b, t in execz) else 'no write in execz block',
        'vreg also vector' if vector else 'spill-only vreg',
    ]
    return '; '.join(flags)


def wrap_spill(original):
    def spill_operands(self, name, ops, mods):
        try:
            return original(self, name, ops, mods)
        except G.Unsupported as error:
            if 'needs straight-line code' in str(error):
                v = G.reg_range(ops[0] if name == 'v_writelane_b32' else ops[1])
                if v:
                    SPILL.append('%s %s: %s' % (getattr(self, 'kind', '?'), name, spill_shape(self, v[1])))
            raise
    return spill_operands


REG = re.compile(r'\bs\[?(\d+)')


def main(argv):
    global G
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        raise SystemExit(__doc__)
    G = load_patched()
    G.Stage.scalar = wrap_scalar(G.Stage.scalar)
    G.Stage.spill_operands = wrap_spill(G.Stage.spill_operands)
    out_dir = argv[0]
    tallies = {'graphics': collections.Counter(), 'compute': collections.Counter()}
    nand = {'graphics': collections.Counter(), 'compute': collections.Counter()}
    spill = {'graphics': collections.Counter(), 'compute': collections.Counter()}
    for compute in (False, True):
        side = 'compute' if compute else 'graphics'
        for key in dict.fromkeys(G.request_keys(out_dir, compute=compute)):
            EVENTS.clear()
            NAND.clear()
            SPILL.clear()
            try:
                sites = G.compute_refusals(out_dir, key) if compute else G.request_refusals(out_dir, key, [])
            except Exception as error:  # keep going; count the crash class only
                tallies[side]['probe error %s' % type(error).__name__] += 1
                continue
            seen = set()
            for reason, _ in sites:
                if 'joined at a branch target' not in reason:
                    continue
                m = REG.search(reason.split('holding')[0] if 'holding' in reason else reason)
                reg = int(m.group(1)) if m else None
                label = 'numeric use' if reason.startswith('numeric use') else 'descriptor operand' \
                    if reason.startswith('descriptor operand') else 'other use'
                cls = EVENTS.get(reg, 'no join event recorded for the named register')
                item = (label, cls)
                if item not in seen:  # count each request once per class
                    seen.add(item)
                    tallies[side]['%s <- %s' % item] += 1
            for item in set(NAND):
                nand[side][item] += 1
            for item in set(SPILL):
                spill[side][item] += 1
    print(json.dumps({side: {'joined (requests per class)': dict(tallies[side].most_common()),
                             'nand/nor mixed (requests per class)': dict(nand[side].most_common()),
                             'lane spill needs straight-line (requests per class)': dict(spill[side].most_common())}
                      for side in tallies}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
