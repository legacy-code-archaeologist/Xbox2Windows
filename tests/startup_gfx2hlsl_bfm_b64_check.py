"""s_bfm_b64 in tools/startup/gfx2hlsl.py (census issue #1723).

usage: startup_gfx2hlsl_bfm_b64_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]

  (no arguments)  translation controls and the CPU evaluation of the emitted text, pure Python
  draw            additionally draw the translated pair on a desktop adapter (WARP or hardware), compile the pixel
                  stage with FXC ps_5_1 when given and with DXC ps_6_0, and compare the read-back words

`s_bfm_b64` had no lowering and was refused. RDNA 2 ISA (SOP2 opcode 37, `isa/rdna2_isa.txt` "S_BFM_B64"):
`D.u64 = ((1ULL << S0.u[5:0]) - 1) << S1.u[5:0]`. Both sources are 32-bit and only their low six bits count; a
width of 0 gives 0; the mask is truncated to 64 bits, so bits shifted past bit 63 are dropped. The ISA lists no SCC
write for this opcode (the bitwise ops print `SCC = (D != 0)`), so SCC is left alone.

The 64-bit scalar value is two words (s[d], s[d+1]) written together. The instruction bytes are constructed here
from the SOP2 layout the checked-in decoder reads (`tools/disasm_rdna2.py`: [31:30]=0b10, opcode [29:23], sdst
[22:16], ssrc1 [15:8], ssrc0 [7:0]).

The emitted HLSL, including the real `shl64` helper from the generated preamble, is evaluated by a small integer
evaluator that rejects any 32-bit shift count of 32 or more (undefined in HLSL), and compared with a reference built
from a bit loop over the ISA description, not from the lowering's expressions. Mutations (no width mask, 31 instead of
63, offset masked to five bits, width 0 reading all ones) must read a wrong word and are applied to the translated
text only.

The draw mode (review of #2007) takes the evaluator out of the loop: the compiler's own treatment of `shl64`, the
`bw >= 32u ? ... : ...` selects and `uint2` is read back from the render target. Width and offset are root constants
(s8 and s9 of the pixel stage), so neither compiler can fold them. One draw computes `s_bfm_b64 s[20:21], s8, s9` and
the swapped `s_bfm_b64 s[22:23], s9, s8` and exports the four words. The expected words are the bit-loop reference.
Two negative controls must show in the draw: a reference that wraps the bits past bit 63 instead of dropping them,
and a mutated lowering (width 0 reads all ones). The pixel stage is built from instruction words made with the
encoder below and read back through the decoder, so no assembler is needed.

A source SGPR that holds a lane mask (a compare result) is refused before either destination word or its mask flag changes,
with and without the destination aliasing the source. After #1931 both written words are proven uniform only when both sources
are, and an older proof for s[d] or s[d+1] is dropped, so a later v_readfirstlane_b32 cannot use it.
"""
import os
import re
import struct
import sys
import types

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

MASK = 0xffffffff
failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def reference(width, offset):
    """The ISA mask as a bit loop: bit i is set when offset <= i < offset + width and i < 64."""
    width, offset = width & 63, offset & 63
    return sum(1 << i for i in range(64) if offset <= i < offset + width)


# ---- instruction construction ------------------------------------------------------------------------------
SOP2_BFM_B64 = 37


def encode(dst, src0, src1, literal=None):
    w0 = 0x80000000 | (SOP2_BFM_B64 << 23) | (dst << 16) | (src1 << 8) | src0
    out = struct.pack('<I', w0)
    return out + (struct.pack('<I', literal) if literal is not None else b'')


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    def __init__(self):
        self.lines, self.sym, self.masks = [], {}, set()
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate(asm, sym=None):
    stage = Stage()
    stage.sym.update(sym or {})
    stage.one('s_bfm_b64', asm)
    return stage


def refused(asm, sym=None):
    try:
        translate(asm, sym)
    except G.Unsupported:
        return True
    return False


# ---- a small evaluator for the HLSL the lowering and the shl64 helper use ---------------------------------
TOKEN = re.compile(r'\s*(0x[0-9a-fA-F]+u?|\d+u?|[A-Za-z_]\w*|<<|>>|<=|>=|==|!=|\|\||&=|[-+*&|^~!<>?:(),.=])')
LEVELS = [['||'], ['|'], ['^'], ['&'], ['==', '!='], ['<', '>', '<=', '>='], ['<<', '>>'], ['+', '-'], ['*']]


class Interp:
    def __init__(self, common):
        self.funcs = {}
        m = re.search(r'uint2 shl64\(([^)]*)\) \{(.*?)\}\s*(?=\'|uint2 shr64u|$)', common)
        assert m, 'shl64 not found in the preamble'
        self.funcs['shl64'] = ([p.split()[-1] for p in m.group(1).split(',')], m.group(2))

    def tokens(self, text):
        toks = TOKEN.findall(text)
        assert ''.join(toks) == re.sub(r'\s+', '', text), text
        return toks

    def expr(self, text, env):
        self.t, self.p, self.env = self.tokens(text), 0, env
        value = self.ternary()
        assert self.p == len(self.t), text
        return value

    def peek(self):
        return self.t[self.p] if self.p < len(self.t) else None

    def take(self):
        self.p += 1
        return self.t[self.p - 1]

    def ternary(self):
        cond = self.binary(0)
        if self.peek() != '?':
            return cond
        self.take()
        yes = self.ternary()
        assert self.take() == ':'
        no = self.ternary()
        return yes if cond else no

    def binary(self, level):
        if level == len(LEVELS):
            return self.unary()
        node = self.binary(level + 1)
        while self.peek() in LEVELS[level]:
            op = self.take()
            b = self.binary(level + 1)
            a = node
            if op in ('<<', '>>'):
                assert 0 <= b < 32, 'HLSL shift count %d is not below 32' % b
                node = (a << b) & MASK if op == '<<' else a >> b
            elif op == '||':
                node = int(bool(a or b))
            else:
                node = {'|': a | b, '^': a ^ b, '&': a & b, '+': a + b, '-': a - b, '*': a * b, '==': int(a == b),
                        '!=': int(a != b), '<': int(a < b), '>': int(a > b), '<=': int(a <= b),
                        '>=': int(a >= b)}[op]
                node &= MASK
        return node

    def unary(self):
        tok = self.take()
        if tok == '~':
            return ~self.unary() & MASK
        if tok == '(':
            v = self.ternary()
            assert self.take() == ')'
            return self.member(v)
        if tok[0].isdigit():
            return int(tok.rstrip('u'), 0)
        if self.peek() == '(':
            self.take()
            args = [self.ternary()]
            while self.peek() == ',':
                self.take()
                args.append(self.ternary())
            assert self.take() == ')'
            return self.member(self.call(tok, args))
        return self.member(self.env[tok])

    def member(self, v):
        while self.peek() == '.':
            self.take()
            v = v['xy'.index(self.take())]
        return v

    def call(self, name, args):
        if name == 'uint2':
            return tuple(args)
        params, body = self.funcs[name]
        saved = (self.t, self.p, self.env)
        env = dict(zip(params, args))
        result = self.run(body, env)
        self.t, self.p, self.env = saved
        return result

    def run(self, text, env):
        """Run statements; returns the value of a `return`."""
        for stmt in [s.strip() for s in text.split(';') if s.strip()]:
            stmt = stmt.strip('{} ')
            if not stmt:
                continue
            if stmt.startswith('return '):
                return self.expr(stmt[7:], env)
            m = re.match(r'(uint2|uint)\s+(.*)$', stmt)
            if m:
                for part in re.split(r',\s*(?=\w+\s*=)', m.group(2)):
                    name, _, rhs = part.partition('=')
                    env[name.strip()] = self.expr(rhs, env)
                continue
            m = re.match(r'(\w+)\s*(&=|=)\s*(.*)$', stmt)
            assert m, stmt
            value = self.expr(m.group(3), env)
            env[m.group(1)] = (env[m.group(1)] & value) if m.group(2) == '&=' else value
        return None


def preamble():
    pipe = object.__new__(G.Pipeline)
    pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
    return ' '.join(pipe.common())


COMMON = preamble()


def evaluate(stage, env, common=COMMON):
    """Run the emitted lines of `stage` and return the (s20, s21) words."""
    interp = Interp(common)
    env = dict(env)
    for line in stage.lines:
        # one block of statements: '{ ...; s20 = t.x; s21 = t.y; }'
        interp.run(line, env)
    return env['s20'], env['s21']


def pair(value):
    return value & MASK, value >> 32


# ---- 1. the encoder produces the form the decoder reads -----------------------------------------------------
decoded = G.decode(encode(20, 2, 3))
check('encodes and decodes the register form', [a for _o, _s, _n, a in decoded] == ['s_bfm_b64 s[20:21], s2, s3'],
      repr(decoded))
decoded = G.decode(encode(20, 128 + 33, 128 + 5))
check('encodes and decodes inline constants', [a for _o, _s, _n, a in decoded] == ['s_bfm_b64 s[20:21], 33, 5'],
      repr(decoded))
decoded = G.decode(encode(20, 255, 3, literal=0x12345681))
check('encodes and decodes a literal', [a for _o, _s, _n, a in decoded] == ['s_bfm_b64 s[20:21], 0x12345681, s3'],
      repr(decoded))

# ---- 2. run-time sources: the whole width x offset grid, evaluated against the bit-loop reference -----------
asm = 's_bfm_b64 s[20:21], s2, s3'
stage = translate(asm)
text = ' '.join(stage.lines)
check('run-time form writes both words and no SCC',
      's20 = t.x;' in text and 's21 = t.y;' in text and 'scc' not in text and 'shl64(' in text, text)
bad = []
for w in range(64):
    for o in range(64):
        got = evaluate(stage, {'s2': w, 's3': o})
        if got != pair(reference(w, o)):
            bad.append((w, o, got))
check('all 64 x 64 width/offset pairs match the reference', not bad, repr(bad[:3]))
key = [(0, 0), (0, 63), (1, 0), (1, 63), (31, 0), (31, 31), (31, 32), (32, 0), (32, 31), (32, 32), (33, 0),
       (33, 31), (33, 32), (33, 63), (63, 0), (63, 1), (63, 31), (63, 32), (63, 63), (0, 31), (1, 31), (1, 32)]
bad = [(w, o) for w, o in key if evaluate(stage, {'s2': w, 's3': o}) != pair(reference(w, o))]
check('named widths 0/1/31/32/33/63 against offsets 0/31/32/63', not bad, repr(bad))
# hand-written anchors, so the reference is not the only witness
anchors = {(0, 0): 0, (0, 63): 0, (1, 63): 1 << 63, (63, 1): (1 << 64) - 2, (32, 32): 0xffffffff00000000,
           (33, 31): 0xffffffff80000000, (63, 63): 1 << 63, (31, 32): 0x7fffffff00000000}
check('hand-computed anchors', all(reference(*k) == v and evaluate(stage, {'s2': k[0], 's3': k[1]}) == pair(v)
                                   for k, v in anchors.items()))
# source bits above [5:0] are ignored
junk = [(w | hi, o | hj) for w in (0, 1, 31, 32, 33, 63) for o in (0, 31, 32, 63)
        for hi, hj in ((0x40, 0), (0, 0x40), (0xffffffc0, 0xffffff80), (0x80000000, 0x80000040), (0x12345640, 0xabcdef00))]
bad = [(w, o) for w, o in junk if evaluate(stage, {'s2': w, 's3': o}) != pair(reference(w, o))]
check('bits above [5:0] of either source are ignored (%d inputs)' % len(junk), not bad, repr(bad[:3]))

# ---- 3. inline and literal sources ---------------------------------------------------------------------------
stage = translate('s_bfm_b64 s[20:21], 33, s3')
check('an inline width with a run-time offset reads the constant',
      '0x00000021u' in ' '.join(stage.lines) and all(
          evaluate(stage, {'s3': o}) == pair(reference(33, o)) for o in range(64)))
stage = translate('s_bfm_b64 s[20:21], s2, 32')
check('a run-time width with an inline offset reads the constant',
      all(evaluate(stage, {'s2': w}) == pair(reference(w, 32)) for w in range(64)))
stage = translate('s_bfm_b64 s[20:21], 0x12345681, s3')
check('a literal width keeps only its low six bits (0x81 & 63 = 1)',
      all(evaluate(stage, {'s3': o}) == pair(reference(1, o)) for o in range(64)))

# ---- 4. fully constant sources fold, with the same result ----------------------------------------------------
bad = []
for w, o in [(0, 0), (1, 0), (31, 0), (32, 0), (33, 0), (63, 0), (1, 63), (63, 63), (33, 32), (32, 31), (0, 63), (5, 3)]:
    stage = translate('s_bfm_b64 s[20:21], %d, %d' % (w, o))
    if stage.sym.get(20) != ('const', pair(reference(w, o))[0]) or stage.sym.get(21) != ('const', pair(reference(w, o))[1]):
        bad.append((w, o, stage.sym))
check('inline constants fold to the reference words', not bad, repr(bad[:2]))
stage = translate('s_bfm_b64 s[20:21], 0x12345681, 0x7f')
check('literals fold and mask both sources',
      (stage.sym.get(20), stage.sym.get(21)) == tuple(('const', x) for x in pair(reference(1, 63))))
stage = translate('s_bfm_b64 s[20:21], s2, s3', {2: ('const', 33), 3: ('const', 31)})
check('SGPRs holding known constants fold',
      (stage.sym.get(20), stage.sym.get(21)) == tuple(('const', x) for x in pair(reference(33, 31))))
stage = translate('s_bfm_b64 s[20:21], s2, s3', {2: ('const', 0x1000), 3: ('const', 0)})
check('a known width whose low six bits are 0 folds to a zero pair',
      (stage.sym.get(20), stage.sym.get(21)) == (('const', 0), ('const', 0)))

# ---- 5. an in-place destination reads the original sources ----------------------------------------------------
stage = translate('s_bfm_b64 s[2:3], s2, s3')
env = {'s2': 33, 's3': 31}
interp = Interp(COMMON)
for line in stage.lines:
    interp.run(line, env)
check('the destination may be the source registers', (env['s2'], env['s3']) == pair(reference(33, 31)), repr(env))

# ---- 6. refusals ---------------------------------------------------------------------------------------------
check('odd, single, vector and mask destinations are refused', all(refused(a) for a in (
    's_bfm_b64 s[21:22], s2, s3', 's_bfm_b64 s20, s2, s3', 's_bfm_b64 v[20:21], s2, s3', 's_bfm_b64 exec, s2, s3',
    's_bfm_b64 vcc, s2, s3', 's_bfm_b64 s[20:23], s2, s3', 's_bfm_b64 s[106:107], s2, s3', 's_bfm_b64 m0, s2, s3')))
check('pair, vector and mask sources are refused', all(refused(a) for a in (
    's_bfm_b64 s[20:21], s[2:3], s3', 's_bfm_b64 s[20:21], s2, s[4:5]', 's_bfm_b64 s[20:21], v2, s3',
    's_bfm_b64 s[20:21], s2, vcc', 's_bfm_b64 s[20:21], exec, s3', 's_bfm_b64 s[20:21], vcc_lo, s3')))
check('wrong operand counts and modifiers are refused', all(refused(a) for a in (
    's_bfm_b64 s[20:21], s2', 's_bfm_b64 s[20:21], s2, s3, s4', 's_bfm_b64 s[20:21]', 's_bfm_b64 s[20:21], s2, s3 clamp')))
check('a source holding descriptor data is refused',
      refused('s_bfm_b64 s[20:21], s2, s3', {2: ('desc', 0, 0, 0)}))
check('a well-formed pair is accepted (control for the refusals)', not refused('s_bfm_b64 s[20:21], s2, s3'))

# ---- 7. mutations must read a wrong word -------------------------------------------------------------------------
GRID = [(w, o) for w in (0, 1, 31, 32, 33, 63) for o in (0, 31, 32, 63)] + \
       [(0x40, 0), (0x41, 5), (0xffffffa1, 3), (32, 0x80), (3, 0x7f)]
base = translate('s_bfm_b64 s[20:21], s2, s3')


def mutated_wrong(stage_lines, common):
    class S:
        lines = stage_lines
    return [(w, o) for w, o in GRID if evaluate(S, {'s2': w, 's3': o}, common) != pair(reference(w, o))]


line = base.lines[0]
check('control: the unmutated text has no wrong word on the mutation grid', not mutated_wrong(base.lines, COMMON))
mutations = {
    'no mask on the width': ([re.sub(r'(uint bw = s2) & 63u', r'\1', line)], COMMON),
    'width masked with 31': ([line.replace('s2 & 63u', 's2 & 31u')], COMMON),
    'offset masked to five bits': (base.lines, COMMON.replace('sh &= 63u;', 'sh &= 31u;')),
    'width 0 reads all ones': ([line.replace('bw >= 32u ?', 'bw >= 32u || bw == 0u ?', 1)], COMMON),
}
for name, (lines, common) in mutations.items():
    changed = lines != base.lines or common != COMMON
    wrong = mutated_wrong(lines, common) if changed else []
    check('mutation "%s" changes the text and reads a wrong word %s' % (name, wrong[:1]), changed and bool(wrong))

# ---- 8. lane-mask sources are refused before any destination state changes (maintainer review of #2007) -----------
class TrackedStage(G.Stage):
    """A stage with the #1931 uniformity state, driven instruction by instruction."""

    def __init__(self):
        self.lines, self.sym, self.masks = [], {}, set()
        self.used_s, self.used_v = set(), set()
        self.uniform_v, self.varying_s, self.scc_uniform = set(), set(), True
        self.has_branches, self.exec_known = False, True
        self.exports, self.attrs = {}, {}
        self.pipe = Pipeline()

    def run(self, *instructions):
        for asm in instructions:
            self.one(asm.split()[0], asm)
        return self


def snapshot(stage):
    return (list(stage.lines), dict(stage.sym), set(stage.masks), set(stage.varying_s), set(stage.used_s))


def refused_after(*instructions):
    """True when the last instruction is refused after running the earlier ones on a tracked stage."""
    stage = TrackedStage().run(*instructions[:-1])
    try:
        stage.run(instructions[-1])
    except G.Unsupported:
        return True
    return False


CMP = 'v_cmp_lt_f32_e64 %s, v0, v1'
for label, cmp_dst, bfm in (
        ('compare mask as width', 's2', 's_bfm_b64 s[20:21], s2, s3'),
        ('compare mask as offset', 's3', 's_bfm_b64 s[20:21], s2, s3'),
        ('compare mask as width, destination aliases it', 's20', 's_bfm_b64 s[20:21], s20, s3'),
        ('compare mask as offset, destination aliases it', 's21', 's_bfm_b64 s[20:21], s2, s21'),
        ('compare mask as both, destination aliases it', 's20', 's_bfm_b64 s[20:21], s20, s20')):
    stage = TrackedStage().run(CMP % cmp_dst)
    reg = int(cmp_dst[1:])
    before = snapshot(stage)
    try:
        stage.run(bfm)
        refusal = None
    except G.Unsupported as error:
        refusal = str(error)
    check('%s is refused' % label, refusal is not None and 'lane mask' in refusal, repr(refusal))
    check('%s: no HLSL, no state change' % label, snapshot(stage) == before and reg in stage.masks,
          repr((before, snapshot(stage))))
# Controls: the same shapes with a non-mask scalar are accepted, and the compare destination alone stays a mask.
for asm in ('s_bfm_b64 s[20:21], s2, s3', 's_bfm_b64 s[20:21], s20, s3', 's_bfm_b64 s[20:21], s2, s21'):
    stage = TrackedStage().run(asm)
    check('control: %s accepted without a mask source' % asm, 'shl64(' in ' '.join(stage.lines))
# A mask destination (not a source) loses its flag: the pair is a number afterwards.
stage = TrackedStage().run(CMP % 's20', 's_bfm_b64 s[20:21], s2, s3')
check('a mask destination that is not a source loses its flag', 20 not in stage.masks and 21 not in stage.masks)

# A lane mask in the high word of the destination pair is a number after the write (review of #2007, the survived
# mutation of `self.masks.discard(d + 1)`): s21 is a compare result, s_bfm_b64 overwrites s[20:21], and s21 is then
# read as a source. That read must be accepted and must use the word written by the first instruction.
stage = TrackedStage().run(CMP % 's21')
check('control: the compare result in s21 is a lane mask before the write', 21 in stage.masks)
check('control: reading s21 while it is still a lane mask is refused', refused_after(CMP % 's21', 's_bfm_b64 s[22:23], s2, s21'))
stage = TrackedStage().run(CMP % 's21', 's_bfm_b64 s[20:21], s2, s3')
check('a lane mask in s[d+1] loses its flag when s_bfm_b64 writes the pair', 21 not in stage.masks and 20 not in stage.masks,
      repr(stage.masks))
check('s21 is read as a number after the write', not refused_after(CMP % 's21', 's_bfm_b64 s[20:21], s2, s3',
                                                                'v_mov_b32_e32 v7, s21', 's_bfm_b64 s[22:23], s2, s21'))
nit_stage = TrackedStage().run(CMP % 's21', 's_bfm_b64 s[20:21], s2, s3', 's_bfm_b64 s[22:23], s2, s21')
# The compare line is HLSL over vector registers; run only the two scalar blocks.
env, interp = {'s2': 33, 's3': 31}, Interp(COMMON)
for line in [l for l in nit_stage.lines if 'shl64(' in l]:
    interp.run(line, env)
check('evaluated: s[22:23] = bfm(33, high word of bfm(33, 31))',
      (env['s22'], env['s23']) == pair(reference(33, pair(reference(33, 31))[1])), repr(env))
# The mutation itself: the same lowering without `self.masks.discard(d + 1)` keeps s21 a lane mask, so the read is
# refused. The patched source is executed as a separate module; the check fails if the mutation does not apply.
SOURCE = open(os.path.join(here, '..', 'tools', 'startup', 'gfx2hlsl.py'), encoding='utf-8').read()
DISCARD = '        self.masks.discard(d + 1)\n'
body = SOURCE[SOURCE.index('def op_s_bfm_b64'):]
body = body[:body.index('\n    def ', 10)]
check('mutation applies: op_s_bfm_b64 holds exactly one discard of d + 1', body.count(DISCARD) == 1)
mutant_source = SOURCE.replace(body, body.replace(DISCARD, ''))
check('mutation changes the source', mutant_source != SOURCE)
mutant = types.ModuleType('gfx2hlsl_mutant')
mutant.__file__ = G.__file__
exec(compile(mutant_source, 'gfx2hlsl_mutant.py', 'exec'), mutant.__dict__)


class MutantStage(mutant.Stage):
    __init__ = TrackedStage.__init__
    run = TrackedStage.run


for label, module_stage, expect_refused in (('unmutated', TrackedStage, False), ('without discard(d + 1)', MutantStage, True)):
    try:
        module_stage().run(CMP % 's21', 's_bfm_b64 s[20:21], s2, s3', 's_bfm_b64 s[22:23], s2, s21')
        got_refused = False
    except (G.Unsupported, mutant.Unsupported):
        got_refused = True
    check('%s: the number read of s21 after the write is %s' % (label, 'refused' if expect_refused else 'accepted'),
          got_refused == expect_refused)

# ---- 9. both written words carry a fresh uniformity proof (#1931) ----------------------------------------------------
READ = 'v_readfirstlane_b32 s9, v7'
REFUSAL = 'v_readfirstlane_b32 requires a proven uniform source'


def read_word(stage, word):
    """Copy s<word> into v7 and read it back: accepted only when the word is proven uniform."""
    try:
        stage.run('v_mov_b32_e32 v7, s%d' % word, READ)
    except G.Unsupported as error:
        return str(error)
    return 'accepted'


# `s_mov_b64 s[2:3], exec` leaves s2 a lane mask and s3 a varying number, a non-mask varying source.
VARYING = 's_mov_b64 s[2:3], exec'
PROVE = 's_bfm_b64 s[20:21], s4, s5'
stage = TrackedStage().run(PROVE)
check('uniform sources: both words are proven uniform', read_word(stage, 20) == 'accepted' and
      read_word(TrackedStage().run(PROVE), 21) == 'accepted')
for label, bfm in (('width', 's_bfm_b64 s[20:21], s3, s5'), ('offset', 's_bfm_b64 s[20:21], s4, s3'),
                   ('width and offset', 's_bfm_b64 s[20:21], s3, s3')):
    for word in (20, 21):
        # control: the old proof exists, otherwise a refusal below would prove nothing
        check('old proof for s%d exists before the %s overwrite' % (word, label),
              read_word(TrackedStage().run(VARYING, PROVE), word) == 'accepted')
        got = read_word(TrackedStage().run(VARYING, PROVE, bfm), word)
        check('varying %s: the old proof for s%d is not used' % (label, word), got == REFUSAL, got)
# Destination aliasing a varying source: s21 is the varying high word of a saved exec.
for word in (20, 21):
    got = read_word(TrackedStage().run('s_mov_b64 s[20:21], exec', 's_bfm_b64 s[20:21], s21, s5'), word)
    check('aliased varying source: s%d is not proven uniform' % word, got == REFUSAL, got)
# A later uniform write brings the proof back, so the refusals above come from the writes.
for word in (20, 21):
    got = read_word(TrackedStage().run(VARYING, 's_bfm_b64 s[20:21], s3, s5', PROVE), word)
    check('control: a later uniform write re-proves s%d' % word, got == 'accepted', got)
    stage = TrackedStage().run(VARYING, 's_bfm_b64 s[20:21], s3, s5', 's_bfm_b64 s[20:21], 5, 3')
    check('control: constant sources re-prove s%d' % word, read_word(stage, word) == 'accepted')

# ---- 10. desktop draw: the compiler's own shl64, select and uint2 handling, read back (review of #2007) ----------
def container(parts):
    """DXBC header (hash zeroed) followed by (fourcc, body) parts."""
    base = 32 + 4 * len(parts)
    offsets, body = [], b''
    for fourcc, data in parts:
        offsets.append(base + len(body))
        body += fourcc + struct.pack('<I', len(data)) + data
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, base + len(body), len(parts)) + struct.pack(
        '<%dI' % len(offsets), *offsets) + body


# One root-constant parameter of two words: s8 (width) and s9 (offset) in both stages, run-time values the compiler
# cannot see. Layout as the checked-in parser reads it: version, parameter count, parameter table at byte 24, no
# samplers, then the (type 1, visibility 0, offset 36) entry and (register 0, space 0, count 2).
ROOT = container([(b'RTS0', struct.pack('<6I', 1, 1, 24, 0, 36, 0) + struct.pack('<3I', 1, 0, 36)
                   + struct.pack('<3I', 0, 0, 2))])
VS_WORDS = bytes.fromhex(  # the recorded full-screen triangle that exports param0 (tests/startup_flat_param_check.py)
    'ff02067e00004040' 'ff02147e000040c0' '8100847d' 'f3060202' '8200847d' 'f2140402' 'f202087e' '80020a7e'
    '81020c7e' '82020e7e' 'ff02107e0100c07f' 'ff02127e01000080' 'cf0800f801020504' '0f0200f806070809' '000081bf')
VOP1_MOV = 1


def ps_instructions():
    """s_bfm_b64 s[20:21], s8, s9; s_bfm_b64 s[22:23], s9, s8; v12..v15 = s20..s23; exp mrt0 v12..v15; s_endpgm."""
    words = [encode(20, 8, 9), encode(22, 9, 8)]
    words += [struct.pack('<I', 0x7e000000 | (12 + i) << 17 | VOP1_MOV << 9 | (20 + i)) for i in range(4)]
    words.append(bytes.fromhex('0f1800f80c0d0e0f' '000081bf'))
    return b''.join(words)


DRAW_WIDTHS, DRAW_OFFSETS = (0, 1, 31, 32, 33, 63), (0, 1, 31, 32, 63)
# Source bits above [5:0] are ignored: 64 reads as 0, 0x41 as 1, 0xffffffa1 as 33, 0xffffff9f as 31.
DRAW_CASES = [(w, o) for w in DRAW_WIDTHS for o in DRAW_OFFSETS] + [(64, 0), (0x41, 0xc1), (0xffffffa1, 3),
                                                                    (5, 0xffffff9f)]


def words_of(w, o):
    """Expected export: s[20:21] = bfm(w, o), s[22:23] = bfm(o, w), each as the low word then the high word."""
    return pair(reference(w, o)) + pair(reference(o, w))


def wrapped_reference(w, o):
    """The wrong reference of the negative control: bits shifted past bit 63 wrap into the low word instead of
    being dropped (a missing cap at bit 63)."""
    width, offset = w & 63, o & 63
    full = ((1 << width) - 1) << offset
    return (full & ((1 << 64) - 1)) | (full >> 64)


def wrapped_words(w, o):
    return pair(wrapped_reference(w, o)) + pair(wrapped_reference(o, w))


def text_of(values):
    return ','.join('%08x' % v for v in values)


def draw_bfm(renderer, adapter, vs, ps, w, o, expected):
    import subprocess
    constants = {1: w & MASK, 2: o & MASK}  # ud1, ud2 (ud0 is the rectangle switch and stays 0)
    entry = text_of([constants.get(i, 0) for i in range(4)])
    run = subprocess.run([renderer, adapter, '--startup-flat-param', str(vs), str(ps), text_of(expected), entry],
                         capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    import tempfile
    from pathlib import Path
    sys.path.insert(0, here)
    from startup_flat_param_check import compile_hlsl
    pipe = G.Pipeline(VS_WORDS, ps_instructions(), ROOT)
    check('draw: the entry constants sit at ud0 (rect) and ud1, ud2 (width, offset)',
          pipe.uds[:3] == [('rect',), ('rootconst', 0, 0), ('rootconst', 0, 1)], repr(pipe.uds))
    vs_text, ps_text = pipe.hlsl_vs(), pipe.hlsl_ps()
    check('draw: the pixel stage computes both pairs from run-time constants',
          ps_text.count('= shl64(') == 2 and 'ud1' in ps_text and 'ud2' in ps_text)
    mutated_text = ps_text.replace('bw >= 32u ?', 'bw >= 32u || bw == 0u ?')
    check('draw: the mutated pixel stage differs from the translation', mutated_text != ps_text)
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, vs_text, 'vs_bfm_%s' % kind, vs_profile, directory)
            good = compile_hlsl(compiler, kind, ps_text, 'ps_bfm_%s' % kind, ps_profile, directory)
            bad = compile_hlsl(compiler, kind, mutated_text, 'ps_bfm_mutated_%s' % kind, ps_profile, directory)
            for w, o in DRAW_CASES:
                expected = words_of(w, o)
                run = draw_bfm(renderer, adapter, vs, good, w, o, expected)
                if run.returncode:
                    raise AssertionError('%s width %#x offset %#x expected %s\n%s\n%s' % (
                        kind, w, o, text_of(expected), run.stdout, run.stderr))
                print('%s width %#x offset %#x: %s' % (kind, w, o, text_of(expected)), flush=True)
            # Negative control 1: a reference without the cap at bit 63 must not match what the compiled code returns.
            for w, o in ((33, 63), (63, 63), (33, 32), (63, 33)):
                wrong = wrapped_words(w, o)
                check('%s: the uncapped reference differs from the ISA reference for width %d offset %d' % (kind, w, o),
                      wrong != words_of(w, o))
                run = draw_bfm(renderer, adapter, vs, good, w, o, wrong)
                real = 'actual=%s' % text_of(words_of(w, o))
                if run.returncode == 0 or real not in run.stderr:
                    raise AssertionError('%s uncapped reference was not rejected for width %d offset %d\n%s\n%s' % (
                        kind, w, o, run.stdout, run.stderr))
                print('%s uncapped reference: fails as expected, reads %s' % (kind, text_of(words_of(w, o))), flush=True)
            # Negative control 2: the mutated lowering (width 0 reads all ones) must read different words.
            for w, o, got in ((0, 0, (0xffffffff, 0, 0xffffffff, 0)), (0, 5, (0xffffffe0, 0x1f, 0x1f, 0))):
                run = draw_bfm(renderer, adapter, vs, bad, w, o, words_of(w, o))
                if run.returncode == 0 or 'actual=%s' % text_of(got) not in run.stderr:
                    raise AssertionError('%s mutated lowering did not read %s for width %d offset %d\n%s\n%s' % (
                        kind, text_of(got), w, o, run.stdout, run.stderr))
                print('%s mutation (width 0 reads all ones): fails as expected, reads %s' % (kind, text_of(got)), flush=True)


if len(sys.argv) > 1:
    if sys.argv[1] != 'draw' or len(sys.argv) not in (5, 6):
        raise SystemExit(__doc__)
    check_draw(*sys.argv[2:])

if failures:
    print('FAIL')
    for item in failures:
        print('  ' + item)
    sys.exit(1)
print('PASS')
