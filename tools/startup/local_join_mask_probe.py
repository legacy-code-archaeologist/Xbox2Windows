"""Local diagnostic (branch local/intro-shaders only, not for upstream): why do two refusal groups occur?

1. "... joined at a branch target": for the register named in the refusal, how the paths into the join differed.
2. "s_nand_b32/s_nor_b32 mixes a lane mask with an unprojected scalar word": what the non-mask operand was.

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
    def scalar(self, base, ops):
        if base in ('s_nand_b32', 's_nor_b32'):
            def low(t):
                r = G.reg_range(t)
                return ('s%d' % r[1]) if r and r[0] == 's' and r[1] != r[2] else t
            operands = [low(t) for t in ops]
            if any(self.is_lane_mask(t) for t in operands) and not all(self.is_lane_mask(t) for t in operands[1:]):
                dest = 'dest mask' if self.is_lane_mask(operands[0]) else 'dest word'
                NAND.append('%s %s; %s; srcs %s' % (getattr(self, 'kind', '?'), base, dest,
                                                     ' | '.join(operand_class(self, t) for t in operands[1:])))
        return original(self, base, ops)
    return scalar


REG = re.compile(r'\bs\[?(\d+)')


def main(argv):
    global G
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        raise SystemExit(__doc__)
    G = load_patched()
    G.Stage.scalar = wrap_scalar(G.Stage.scalar)
    out_dir = argv[0]
    tallies = {'graphics': collections.Counter(), 'compute': collections.Counter()}
    nand = {'graphics': collections.Counter(), 'compute': collections.Counter()}
    for compute in (False, True):
        side = 'compute' if compute else 'graphics'
        for key in dict.fromkeys(G.request_keys(out_dir, compute=compute)):
            EVENTS.clear()
            NAND.clear()
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
    print(json.dumps({side: {'joined (requests per class)': dict(tallies[side].most_common()),
                             'nand/nor mixed (requests per class)': dict(nand[side].most_common())}
                      for side in tallies}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
