"""v_pack_b32_f16 (VOP3) in tools/startup/gfx2hlsl.py: D[15:0] = S0.f16, D[31:16] = S1.f16 (AMD RDNA 2 ISA).

Synthetic VOP3 words (gfx10 opcode 0x311; op_sel in word 0 bits 14:11, abs in bits 10:8, clamp in bit 15, neg in
word 1 bits 31:29). The decoder reads each word back, and the test asserts the printed op_sel and modifier forms.
The emitted expression is evaluated in Python for every op_sel[0..1] x abs x neg combination over sample register
pairs. The reference written here selects the half (op_sel[i] = high half of source i), then applies abs (clear
bit 15) and then neg (flip bit 15). An SGPR source is accepted.
Controls (refused): a destination op_sel bit, clamp, an inline float constant and an inline integer source.
Negative controls: an expression with the halves swapped, or with neg applied before abs, is caught by the
reference; without op_v_pack_b32_f16 the instruction is refused, as on main.
"""
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402


class Pipe:
    root = {'params': [], 'flags': 0}

    def ud(self, source):
        return 0


def vop3(vdst, s0, s1, opsel=0, absm=0, neg=0, clamp=0):
    """s0 and s1 are 9-bit source fields: 256 + n is vN, n < 106 is sN, 128..208 and 240..248 inline constants."""
    return [0xd4000000 | 0x311 << 16 | clamp << 15 | opsel << 11 | absm << 8 | vdst, s0 | s1 << 9 | neg << 29]


def one(words):
    s = G.Stage(Pipe(), 'vs', b'')
    s.lines, s.pc = [], 0
    _, _, name, text = G.decode(struct.pack('<2I', *words))[0]
    s.one(name, text)
    return text, s.lines


def ev(expr, env):
    py = re.sub(r'(0x[0-9a-f]+|\d+)u\b', r'\1', expr)
    return eval(py, {}, env) & 0xffffffff


def ref(a, b, sel, absm, neg, neg_first=False):
    out = []
    for i, x in enumerate((a, b)):
        h = (x >> 16) & 0xffff if sel >> i & 1 else x & 0xffff
        if neg_first and neg >> i & 1:
            h ^= 0x8000
        if absm >> i & 1:
            h &= 0x7fff
        if not neg_first and neg >> i & 1:
            h ^= 0x8000
        out.append(h)
    return out[0] | out[1] << 16


VALUES = [0x00000000, 0x3c00bc00, 0x80017fff, 0xfc007e00, 0x12345678, 0xffff0001]


def refused(words, why):
    try:
        one(words)
    except G.Unsupported as error:
        assert why in str(error), str(error)
    else:
        raise AssertionError('accepted: %s' % why)


def main():
    cases = 0
    for sel in range(4):
        for absm in range(4):
            for neg in range(4):
                text, lines = one(vop3(3, 257, 258, sel, absm, neg))
                assert text.startswith('v_pack_b32_f16 v3, '), text
                assert ('op_sel:[%d,%d,0]' % (sel & 1, sel >> 1) in text) == bool(sel), text
                m = re.fullmatch(r'v3 = (.*);', lines[0])
                assert m and len(lines) == 1, lines
                for a in VALUES:
                    for b in VALUES:
                        got = ev(m.group(1), {'v1': a, 'v2': b})
                        assert got == ref(a, b, sel, absm, neg), (text, hex(a), hex(b), hex(got))
                        cases += 1
                if absm and neg & absm:
                    # The modifier order matters only where both act: the wrong order must be caught.
                    assert any(ev(m.group(1), {'v1': a, 'v2': b}) != ref(a, b, sel, absm, neg, neg_first=True)
                               for a in VALUES for b in VALUES), text
    text, lines = one(vop3(3, 5, 258, 1))  # SGPR source, high half
    assert text == 'v_pack_b32_f16 v3, s5, v2 op_sel:[1,0,0]', text
    assert all(ev(lines[0][len('v3 = '):-1], {'s5': a, 'v2': b}) == ref(a, b, 1, 0, 0)
               for a in VALUES for b in VALUES), lines
    _, lines = one(vop3(3, 257, 258, 1))
    swapped = re.fullmatch(r'v3 = (.*) \| \((.*) << 16\);', lines[0])
    wrong = '%s | (%s << 16)' % (swapped.group(2), swapped.group(1))
    assert any(ev(wrong, {'v1': a, 'v2': b}) != ref(a, b, 1, 0, 0) for a in VALUES for b in VALUES)
    refused(vop3(3, 257, 258, 4), 'v_pack_b32_f16 form')
    refused(vop3(3, 257, 258, clamp=1), 'v_pack_b32_f16 modifier clamp')
    refused(vop3(3, 242, 258), 'v_pack_b32_f16 source 1.0')
    refused(vop3(3, 129, 258), 'v_pack_b32_f16 source 1')
    saved = G.Stage.op_v_pack_b32_f16
    del G.Stage.op_v_pack_b32_f16
    try:
        refused(vop3(3, 257, 258), 'v_pack_b32_f16')
    finally:
        G.Stage.op_v_pack_b32_f16 = saved
    print('pack_b32_f16: %d (op_sel, abs, neg, value) cases match the reference; SGPR source accepted; swapped '
          'halves and neg-before-abs caught; destination op_sel, clamp and constants refused; refused without the '
          'handler' % cases)


if __name__ == '__main__':
    main()
