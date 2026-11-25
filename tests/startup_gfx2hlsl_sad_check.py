"""Unsigned sum-of-absolute-differences (v_sad_u8, v_sad_u16, v_sad_u32) in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation, on-demand helper, value-evaluation, refusal and mutation
                                       controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input triples and compare the four result words

The pixel fixture decodes synthetic gfx1013 VOP3A bytes via G.decode() ONLY (no game data). It reads three flat
parameters (v0..v3, v4..v7, v8..v11), computes v_sad_u* across all four lanes into v12..v15, and exports v12..v15
to mrt0. The vertex stage supplies the input words through flat parameters so the pixel stage cannot fold them.
"""
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    MASK,
    POST_WORDS,
    PRE_WORDS,
    Probe,
    check,
    refused,
    signed,
    vs_words,
    words_text,
)

OPCODES = {
    'v_sad_u8': 0x15a,
    'v_sad_u16': 0x15c,
    'v_sad_u32': 0x15d,
}


def vop3a(op, vdst, src0, src1, src2, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg=0, lit=None):
    """Encode one synthetic gfx1013 VOP3A 64-bit (or 96-bit with literal) instruction word."""
    w0 = 0xd4000000 | (op << 16) | (clamp << 15) | (op_sel << 11) | (abs_bits << 8) | vdst
    w1 = (neg << 29) | (omod << 27) | (src2 << 18) | (src1 << 9) | src0
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


BODY_WORDS = {
    op: b''.join(vop3a(code, 12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane) for lane in range(4))
    for op, code in OPCODES.items()
}


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


def ref_sad_u8(a, b, c):
    a, b, c = a & MASK, b & MASK, c & MASK
    return (sum(abs(((a >> shift) & 0xff) - ((b >> shift) & 0xff)) for shift in (0, 8, 16, 24)) + c) & MASK


def ref_sad_u16(a, b, c):
    a, b, c = a & MASK, b & MASK, c & MASK
    return (abs((a & 0xffff) - (b & 0xffff)) + abs(((a >> 16) & 0xffff) - ((b >> 16) & 0xffff)) + c) & MASK


def ref_sad_u32(a, b, c):
    a, b, c = a & MASK, b & MASK, c & MASK
    return (abs(a - b) + c) & MASK


REF = {
    'v_sad_u8': ref_sad_u8,
    'v_sad_u16': ref_sad_u16,
    'v_sad_u32': ref_sad_u32,
}


def mut_sad_u8(a, b, c):
    """Mutated reference: drops the top byte (shift 24)."""
    a, b, c = a & MASK, b & MASK, c & MASK
    return (sum(abs(((a >> shift) & 0xff) - ((b >> shift) & 0xff)) for shift in (0, 8, 16)) + c) & MASK


def mut_sad_u16(a, b, c):
    """Mutated reference: drops the top halfword (bits 31:16)."""
    a, b, c = a & MASK, b & MASK, c & MASK
    return (abs((a & 0xffff) - (b & 0xffff)) + c) & MASK


def mut_sad_u32(a, b, c):
    """Mutated reference: uses signed 32-bit subtraction inside abs(int(a) - int(b))."""
    a, b, c = a & MASK, b & MASK, c & MASK
    return (abs(signed(a) - signed(b)) + c) & MASK


MUTATION_REF = {
    'v_sad_u8': mut_sad_u8,
    'v_sad_u16': mut_sad_u16,
    'v_sad_u32': mut_sad_u32,
}

# Grids with > 500 edge triples per instruction.
U8_WORDS = (
    0x00000000, 0x01020304, 0x04030201, 0x7f80ff00, 0x807f00ff,
    0x00010000, 0x0000ffff, 0x7fffffff, 0x80000000, 0xffffffff,
)
U8_ADDENDS = (0, 1, 0x7fffffff, 0x80000000, 0xfffffc04, 0xffffffff)
U16_WORDS = (
    0x00000000, 0x0000ffff, 0xffff0000, 0x0001ffff, 0xffff0001,
    0x7fff8000, 0x80007fff, 0x7fffffff, 0x80000000, 0xffffffff,
)
U16_ADDENDS = (0, 1, 0x7fffffff, 0x80000000, 0xfffe0002, 0xffffffff)
U32_WORDS = (
    0x00000000, 0x00000001, 0x00000002, 0x12345678, 0x7ffffffe, 0x7fffffff,
    0x80000000, 0x80000001, 0xdeadbeef, 0xfffffffe, 0xffffffff,
)
U32_ADDENDS = (0, 1, 0x7fffffff, 0x80000000, 0xffffffff)

GRID = {
    'v_sad_u8': [(a, b, c) for a in U8_WORDS for b in U8_WORDS for c in U8_ADDENDS],
    'v_sad_u16': [(a, b, c) for a in U16_WORDS for b in U16_WORDS for c in U16_ADDENDS],
    'v_sad_u32': [(a, b, c) for a in U32_WORDS for b in U32_WORDS for c in U32_ADDENDS],
}

# 12 input triples (3 draws of 4 lanes) per instruction.
# The first 4 triples in each list separate the mutation from the reference on every single lane.
DRAW = {
    'v_sad_u8': [
        (0xff000000, 0x00000000, 0),
        (0x00123456, 0xff123456, 10),
        (0x807f0102, 0x7f800304, 0xfffffc04),
        (0x10203040, 0xf0203040, 0xffffffff),
        (0x00010000, 0x0000ffff, 0),
        (0x01020304, 0x04030201, 1),
        (0xffffffff, 0x00000000, 0xfffffc04),
        (0x80000000, 0x7fffffff, 0x80000000),
        (0x7fffffff, 0x80000000, 0x7fffffff),
        (0xdeadbeef, 0xcafebabe, 0x12345678),
        (0x00ff00ff, 0xff00ff00, 0),
        (0xffffffff, 0xffffffff, 0xffffffff),
    ],
    'v_sad_u16': [
        (0xffff0000, 0x00000000, 0),
        (0x00001234, 0xffff1234, 10),
        (0x80007fff, 0x7fff8000, 0xfffe0002),
        (0x1000ffff, 0x2000ffff, 0xffffffff),
        (0x00010000, 0x0000ffff, 0),
        (0x0001ffff, 0xffff0001, 1),
        (0xffffffff, 0x00000000, 0xfffe0002),
        (0x80000000, 0x7fffffff, 0x80000000),
        (0x7fffffff, 0x80000000, 0x7fffffff),
        (0xdeadbeef, 0xcafebabe, 0x12345678),
        (0x00000000, 0x00000000, 0),
        (0xffffffff, 0xffffffff, 0xffffffff),
    ],
    'v_sad_u32': [
        (0xffffffff, 0x00000000, 0),
        (0x00000000, 0xffffffff, 0),
        (0x80000001, 0x00000000, 5),
        (0xfffffffe, 0x00000001, 2),
        (0x7fffffff, 0x80000000, 0),
        (0x80000000, 0x7fffffff, 0xffffffff),
        (0x80000000, 0x00000000, 0x80000000),
        (0x00000000, 0x80000000, 0x80000000),
        (0xffffffff, 0x00000000, 1),
        (0xdeadbeef, 0xcafebabe, 0x12345678),
        (0x00000000, 0x00000000, 0xffffffff),
        (0xffffffff, 0xffffffff, 0xffffffff),
    ],
}


# ---- 32-bit HLSL expression evaluator (supports uint/int, abs(), and sad_u* helpers) -------------------------
TOKEN = re.compile(r'\s*(0x[0-9a-fA-F]+u?|\d+u?|[A-Za-z_]\w*|<<|>>|<=|>=|==|!=|\|\||[-+*&|^~!<>?:(),])')
LEVELS = [['||'], ['|'], ['^'], ['&'], ['==', '!='], ['<', '>', '<=', '>='], ['<<', '>>'], ['+', '-'], ['*']]


class HlslExpr:
    def __init__(self, text, helpers=None):
        self.tokens = TOKEN.findall(text)
        assert ''.join(self.tokens) == re.sub(r'\s+', '', text), text
        self.pos = 0
        self.helpers = helpers or {}
        self.tree = self.ternary()
        assert self.pos == len(self.tokens), text

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self):
        self.pos += 1
        return self.tokens[self.pos - 1]

    def ternary(self):
        cond = self.binary(0)
        if self.peek() != '?':
            return cond
        self.take()
        yes = self.ternary()
        assert self.take() == ':'
        return ('?', cond, yes, self.ternary())

    def binary(self, level):
        if level == len(LEVELS):
            return self.unary()
        node = self.binary(level + 1)
        while self.peek() in LEVELS[level]:
            op = self.take()
            node = (op, node, self.binary(level + 1))
        return node

    def unary(self):
        token = self.take()
        if token in ('~', '-', '!'):
            return ('u' + token, self.unary())
        if token == '(':
            node = self.ternary()
            assert self.take() == ')'
            return node
        if token[0].isdigit():
            return ('num', int(token.rstrip('u'), 0), not token.endswith('u'))
        if self.peek() != '(':
            return ('var', token)
        self.take()
        args = [self.ternary()]
        while self.peek() == ',':
            self.take()
            args.append(self.ternary())
        assert self.take() == ')'
        return ('call', token, args)

    def run(self, env, node=None):
        node = node or self.tree
        kind = node[0]
        if kind == 'num':
            return node[1] & MASK, node[2]
        if kind == 'var':
            return env[node[1]] & MASK, False
        if kind == '?':
            return self.run(env, node[2] if self.run(env, node[1])[0] else node[3])
        if kind == 'call':
            return self.call(node[1], [self.run(env, a) for a in node[2]])
        if kind[0] == 'u' and len(node) == 2:
            v, s = self.run(env, node[1])
            return {'u~': ~v, 'u-': -v, 'u!': int(v == 0)}[kind] & MASK, s
        (a, sa), (b, sb) = self.run(env, node[1]), self.run(env, node[2])
        both = sa and sb
        if kind in ('<<', '>>'):
            count = b & 31
            return ((a << count) & MASK if kind == '<<' else (signed(a) >> count if sa else a >> count) & MASK), sa
        if kind in ('==', '!=', '<', '>', '<=', '>='):
            x, y = (signed(a), signed(b)) if both else (a, b)
            return int({'==': x == y, '!=': x != y, '<': x < y, '>': x > y, '<=': x <= y, '>=': x >= y}[kind]), True
        if kind == '||':
            return int(bool(a or b)), True
        return {'|': a | b, '^': a ^ b, '&': a & b, '+': a + b, '-': a - b, '*': a * b}[kind] & MASK, both

    def call(self, name, args):
        if name in self.helpers:
            assert len(args) == 3 and all(not s for _, s in args), (name, args)
            return self.helpers[name](*(v for v, _ in args)), False
        (v, s), = args
        if name in ('asint', 'int'):
            return v, True
        if name in ('asuint', 'uint'):
            return v, False
        if name == 'abs':
            if s:
                val = abs(signed(v))
                assert val <= 0x7fffffff, 'signed abs overflow on %#x' % v
                return val & MASK, True
            return v, False
        raise AssertionError('unexpected call %s' % name)


def mutate_hlsl(op, text):
    if op == 'v_sad_u8':
        pattern = r'\+\s*uint\(abs\(int\(\(a >> 24\) & 0xffu\) - int\(\(b >> 24\) & 0xffu\)\)\)'
        mutated = re.sub(pattern, '+ 0u', text)
    elif op == 'v_sad_u16':
        pattern = r'\+\s*uint\(abs\(int\(a >> 16\) - int\(b >> 16\)\)\)'
        mutated = re.sub(pattern, '+ 0u', text)
    elif op == 'v_sad_u32':
        pattern = r'\(a > b \? a - b : b - a\)'
        mutated = re.sub(pattern, 'uint(abs(int(a) - int(b)))', text)
    else:
        raise AssertionError(op)
    assert mutated != text, op
    return mutated


def translate(op, mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob())
    text = pipe.hlsl_ps()
    if mutate:
        text = mutate_hlsl(op, text)
    return pipe, text


def extract_sad_helpers(hlsl_text):
    """Parse emitted sad_u8 / sad_u16 / sad_u32 helper definitions into 32-bit evaluators."""
    helpers = {}
    for helper_name in ('sad_u8', 'sad_u16', 'sad_u32'):
        m = re.search(r'^uint %s\(uint a, uint b, uint c\) \{ return (.*); \}$' % helper_name, hlsl_text, re.M)
        if m:
            expr = HlslExpr(m.group(1))
            helpers[helper_name] = lambda a, b, c, e=expr: e.run({'a': a, 'b': b, 'c': c})[0]
    return helpers


def lane_evaluators(hlsl_text):
    helpers = extract_sad_helpers(hlsl_text)
    found = re.findall(r'^\s*v(1[2-5]) = (.*);$', hlsl_text, re.M)
    assert [r for r, _ in found] == ['12', '13', '14', '15'], found
    return helpers, [HlslExpr(rhs, helpers) for _, rhs in found]


def eval_lanes(exprs, chunk):
    return [expr.run({'v%d' % lane: a, 'v%d' % (4 + lane): b, 'v%d' % (8 + lane): c})[0]
            for lane, (expr, (a, b, c)) in enumerate(zip(exprs, chunk))]


def decode_single_sad(word_bytes):
    """Decode synthetic VOP3A bytes via G.decode() and lower through a Stage."""
    ins = G.decode(word_bytes)
    assert len(ins) == 1, ins
    _, _, name, asm = ins[0]
    stage = Probe()
    stage.one(name, asm)
    return stage.lines


def controls():
    # 1. On-demand preamble helpers: none emitted without SAD, only the used helper emitted when used.
    base_ps = PRE_WORDS + POST_WORDS
    base_pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), base_ps, root_blob())
    base_hlsl_ps = base_pipe.hlsl_ps()
    base_hlsl_vs = base_pipe.hlsl_vs()
    check('on-demand: baseline pipeline without v_sad_* emits no sad_u* helpers',
          all(h not in base_hlsl_ps and h not in base_hlsl_vs for h in ('sad_u8(', 'sad_u16(', 'sad_u32(')))

    for op in OPCODES:
        pipe, text = translate(op)
        expected_helper = op[2:]  # sad_u8 / sad_u16 / sad_u32
        other_helpers = [other[2:] for other in OPCODES if other != op]
        helpers, exprs = lane_evaluators(text)
        check('%s: on-demand emits only %s helper in PS/VS' % (op, expected_helper),
              list(helpers) == [expected_helper]
              and all(f'{h}(' not in text and f'{h}(' not in pipe.hlsl_vs() for h in other_helpers))

        # Verify lane calls in emitted HLSL.
        found = re.findall(r'^\s*v(1[2-5]) = %s\(v(\d+), v(\d+), v(\d+)\);$' % expected_helper, text, re.M)
        check('%s: four lane calls emitted' % op,
              found == [('12', '0', '4', '8'), ('13', '1', '5', '9'),
                        ('14', '2', '6', '10'), ('15', '3', '7', '11')])

        # Value-level evaluation of emitted HLSL over 500+ edge/grid triples.
        grid = GRID[op]
        assert len(grid) >= 500, (op, len(grid))
        ref_fn = REF[op]
        mismatch = next((
            (i % 4, triple,
             exprs[i % 4].run({'v%d' % (i % 4): triple[0], 'v%d' % (4 + i % 4): triple[1],
                               'v%d' % (8 + i % 4): triple[2]})[0],
             ref_fn(*triple))
            for i, triple in enumerate(grid)
            if exprs[i % 4].run({'v%d' % (i % 4): triple[0], 'v%d' % (4 + i % 4): triple[1],
                                 'v%d' % (8 + i % 4): triple[2]})[0] != ref_fn(*triple)
        ), None)
        check('%s: emitted HLSL matches reference on %d edge/grid triples' % (op, len(grid)),
              mismatch is None)

        # Mutated HLSL evaluation: must match MUTATION_REF on all non-overflowing grid triples and differ on DRAW[op][:4].
        mutated_text = translate(op, mutate=True)[1]
        _, mut_exprs = lane_evaluators(mutated_text)
        mut_fn = MUTATION_REF[op]
        eval_grid = (
            [t for t in grid if -0x7fffffff <= signed(t[0]) - signed(t[1]) <= 0x7fffffff]
            if op == 'v_sad_u32' else grid
        )
        mut_mismatch = next((
            (i % 4, triple)
            for i, triple in enumerate(eval_grid)
            if mut_exprs[i % 4].run({'v%d' % (i % 4): triple[0], 'v%d' % (4 + i % 4): triple[1],
                                     'v%d' % (8 + i % 4): triple[2]})[0] != mut_fn(*triple)
        ), None)
        first = DRAW[op][:4]
        ref_first = [ref_fn(*t) for t in first]
        mut_first = [mut_fn(*t) for t in first]
        check('%s: mutated HLSL matches MUTATION_REF on %d triples and differs on draw set 0 (%s != %s)'
              % (op, len(eval_grid), words_text(mut_first), words_text(ref_first)),
              mut_mismatch is None and eval_lanes(mut_exprs, first) == mut_first
              and all(m != r for m, r in zip(mut_first, ref_first)))

    # 2. ComputePipeline on-demand helper check + SGPR / inline constant / literal / exec-masked encodings.
    cs_words = vop3a(OPCODES['v_sad_u8'], 1, 4, 129, 255, lit=0x12345678) + struct.pack('<I', 0xbf810000)
    cs_pipe = G.ComputePipeline(cs_words, root_blob(), (4 << 1) | (1 << 7), (1, 1, 1))
    cs_hlsl = cs_pipe.hlsl_cs()
    check('ComputePipeline: v_sad_u8 with SGPR/inline/literal emits sad_u8 only',
          'sad_u8(s4, 0x00000001u, 0x12345678u)' in cs_hlsl
          and 'uint sad_u8(' in cs_hlsl
          and 'sad_u16(' not in cs_hlsl and 'sad_u32(' not in cs_hlsl)

    for op, code in OPCODES.items():
        # Genuine inline -1 (src=193, neg=0) and literal (src=255) decoded via G.decode().
        lit_lines = decode_single_sad(vop3a(code, 12, 256, 193, 255, lit=0xabcdef01))
        check('%s: decoded inline -1 and literal lower cleanly' % op,
              lit_lines == ['v12 = %s(v0, 0xffffffffu, 0xabcdef01u);' % op[2:]])

        # Exec-masked emission.
        masked = Probe()
        masked.exec_known = False
        masked.one(op, '%s v1, v2, v3, v4' % op)
        check('%s: exec-masked emission guards assignment' % op,
              masked.lines == ['if (exec & 1) v1 = %s(v2, v3, v4);' % op[2:]])

    # 3. Refusal controls (binary VOP3A decoded via G.decode() + Stage.one assembly/arity/unlowered).
    for op, code in OPCODES.items():
        # clamp, omod and op_sel are refused through the printed modifier text; ABS (VOP3A w0[10:8]) and NEG
        # (w1[31:29]) are refused by check_encoding() itself, one bit per source. Every source is listed for
        # both modifiers and for both VGPR and inline-integer sources, and the refusal is read back from
        # decode() instead of only from G.Pipeline(): with a narrowed mask the instruction is still refused
        # later, by Stage.one() on the printed |v1| or -1 text, so a Pipeline() raise alone cannot show
        # which guard fired.
        modifiers = [
            ('clamp', vop3a(code, 12, 256, 260, 264, clamp=1)),
            ('omod=mul:2', vop3a(code, 12, 256, 260, 264, omod=1)),
            ('omod=mul:4', vop3a(code, 12, 256, 260, 264, omod=2)),
            ('omod=div:2', vop3a(code, 12, 256, 260, 264, omod=3)),
            ('op_sel=[1,0,0]', vop3a(code, 12, 256, 260, 264, op_sel=1)),
            ('op_sel=[0,0,0,1]', vop3a(code, 12, 256, 260, 264, op_sel=8)),
        ]
        abs_neg = [
            ('abs src0 VGPR', vop3a(code, 12, 256, 260, 264, abs_bits=1)),
            ('abs src1 VGPR', vop3a(code, 12, 256, 260, 264, abs_bits=2)),
            ('abs src2 VGPR', vop3a(code, 12, 256, 260, 264, abs_bits=4)),
            ('abs src0 inline 1', vop3a(code, 12, 129, 260, 264, abs_bits=1)),
            ('abs src1 inline 1', vop3a(code, 12, 256, 129, 264, abs_bits=2)),
            ('abs src2 inline 1', vop3a(code, 12, 256, 260, 129, abs_bits=4)),
            ('neg src0 VGPR', vop3a(code, 12, 256, 260, 264, neg=1)),
            ('neg src1 VGPR', vop3a(code, 12, 256, 260, 264, neg=2)),
            ('neg src2 VGPR', vop3a(code, 12, 256, 260, 264, neg=4)),
            ('neg src0 inline 1', vop3a(code, 12, 129, 260, 264, neg=1)),
            ('neg src1 inline 1', vop3a(code, 12, 256, 129, 264, neg=2)),
            ('neg src2 inline 1', vop3a(code, 12, 256, 260, 129, neg=4)),
        ]
        for label, bad_word in modifiers + abs_neg:
            try:
                G.Pipeline(vs_words([(0, 0, 0)] * 4), PRE_WORDS + bad_word + POST_WORDS, root_blob())
            except G.Unsupported:
                pass
            else:
                raise AssertionError('%s: accepted binary VOP3A with %s' % (op, label))
        for label, bad_word in abs_neg:
            reasons = []
            G.decode(bad_word, reasons)
            check('%s: encoding guard refuses %s' % (op, label),
                  [reason for reason, where in reasons] == ['%s with neg/abs' % op])

        # Direct Stage.one refusals: modifiers, abs/neg, arity.
        check('%s: Stage refuses modifiers, abs/neg, and wrong operand counts' % op, all(refused(asm) for asm in (
            '%s v1, v2, v3, v4 clamp' % op,
            '%s v1, v2, v3, v4 mul:2' % op,
            '%s v1, v2, v3, v4 op_sel:[1,0,0]' % op,
            '%s v1, |v2|, v3, v4' % op,
            '%s v1, v2, |v3|, v4' % op,
            '%s v1, v2, v3, |v4|' % op,
            '%s v1, -v2, v3, v4' % op,
            '%s v1, v2, -v3, v4' % op,
            '%s v1, v2, v3, -v4' % op,
            '%s v1' % op,
            '%s v1, v2' % op,
            '%s v1, v2, v3' % op,
            '%s v1, v2, v3, v4, v5' % op,
        )))

    # 4. Unlowered control: removing entries from VEC_U must cause G.Pipeline and Stage.one to refuse.
    saved = {op: G.VEC_U.pop(op) for op in OPCODES}
    try:
        for op in OPCODES:
            try:
                translate(op)
            except G.Unsupported:
                pass
            else:
                raise AssertionError('%s: translated after removal from VEC_U' % op)
            assert refused('%s v1, v2, v3, v4' % op), op
    finally:
        G.VEC_U.update(saved)
    check('unlowered control: removing v_sad_u* from VEC_U restores refusals', True)


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, triples in DRAW.items():
                ref_fn = REF[op]
                good_ps = compile_hlsl(compiler, kind, translate(op)[1], 'ps_%s_%s' % (op, kind), ps_profile, directory)
                for index in range(0, len(triples), 4):
                    chunk = triples[index:index + 4]
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), ps_words(op), root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    expected = words_text(ref_fn(*t) for t in chunk)
                    run = draw(renderer, adapter, vs, good_ps, expected)
                    if run.returncode:
                        raise AssertionError('%s %s inputs %s\n%s\n%s' % (kind, op, chunk, run.stdout, run.stderr))
                    print('%s %s %s: %s' % (kind, op, [tuple('%x' % v for v in t) for t in chunk], expected),
                          flush=True)
                    if index:
                        continue
                    # Mutated draw: must fail the reference expectation and read MUTATION_REF words.
                    wrong = words_text(MUTATION_REF[op](*t) for t in chunk)
                    bad_ps = compile_hlsl(compiler, kind, translate(op, mutate=True)[1],
                                          'mutated_%s_%s' % (op, kind), ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad_ps, expected)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s %s mutation did not read %s\n%s\n%s'
                                             % (kind, op, wrong, run.stdout, run.stderr))
                    print('%s %s mutation: fails as expected, reads %s' % (kind, op, wrong), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'draw' and len(args) in (3, 4):
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
