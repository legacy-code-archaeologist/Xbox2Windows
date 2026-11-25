"""`v_add_co_ci_u32` in tools/startup/gfx2hlsl.py (Run 20 census #19, 38 shader pairs; part of #1723).

usage: startup_gfx2hlsl_add_co_ci_check.py

The add-with-carry used to be refused as an unknown instruction. It now lowers to a 32-bit add of the two
sources and the carry-in bit, and writes the carry out to the destination mask. The bytes are built here
from the field layout the checked-in decoder reads (tools/disasm_rdna2.py):
  VOP2 e32: [31]=0, [30:25] opcode (0x28), [24:17] vdst, [16:9] vsrc1 (VGPR), [8:0] src0; carry in/out = vcc
  VOP3B e64: w0 [31:26]=0b110101, [25:16]=0x128, [14:8] sdst, [7:0] vdst; w1 [8:0] src0, [17:9] src1, [26:18] src2
The emitted HLSL is also evaluated by a small Python model of the same statements to check the carry
arithmetic on the 32-bit edges.
"""
import os
import re
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


VCC_LO = 106
VGPR = 256


def encode_e32(vdst, src0, vsrc1):
    return struct.pack('<I', (0x28 << 25) | (vdst << 17) | (vsrc1 << 9) | src0)


def encode_e64(vdst, sdst, src0, src1, src2):
    w0 = (0b110101 << 26) | (0x128 << 16) | (sdst << 8) | vdst
    return struct.pack('<II', w0, src0 | (src1 << 9) | (src2 << 18))


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    def __init__(self, masks=()):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set(masks)
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate(code, masks=(), exec_known=True):
    decoded = G.decode(code)
    stage = Stage(masks)
    stage.exec_known = exec_known
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines), stage


def run(emitted, regs, vcc=0, s=None):
    """Evaluate `{ uint ca = ..., cb = ..., ci = ...; vD = ...; target = ...; }` on 32-bit values."""
    env = dict(regs)
    env['vcc'] = vcc
    env.update(s or {})
    body = emitted.strip()[1:-1]
    for stmt in [t.strip() for t in body.split(';') if t.strip()]:
        if stmt.startswith('uint '):
            decls = stmt[len('uint '):]
            for part in decls.split(', '):
                name, expr = part.split(' = ', 1)
                env[name] = value(expr, env)
        else:
            name, expr = stmt.split(' = ', 1)
            env[name] = value(expr, env)
    return env


def value(expr, env):
    expr = re.sub(r'(0x[0-9a-f]+)u', r'\1', expr)
    expr = expr.replace('||', ' or ').replace('&&', ' and ').replace('ca + cb >', '((ca + cb) & 0xffffffff) >')
    expr = expr.replace('1u', '1').replace('0u', '0')
    expr = re.sub(r'\(([^()]*) \? 1 : 0\)', r'(1 if \1 else 0)', expr)
    expr = re.sub(r'^(.*) \? 1 : 0$', r'(1 if \1 else 0)', expr)
    return eval(expr, {}, env) & 0xffffffff


# 1. The short form: carry in and out are vcc.
asm, emitted, stage = translate(encode_e32(5, VGPR + 2, 3))
check('decodes the e32 form', asm and asm[0].startswith('v_add_co_ci_u32_e32 v5, vcc_lo, v2, v3, vcc_lo'), repr(asm))
check('writes the destination and vcc', 'v5 = ca + cb + ci;' in emitted and 'vcc = ' in emitted, emitted)

# 2. The carry arithmetic on the 32-bit edges.
for a, b, cin, total, cout in ((1, 2, 0, 3, 0), (0xffffffff, 0, 1, 0, 1), (0xffffffff, 1, 0, 0, 1),
                                (0xffffffff, 0xffffffff, 1, 0xffffffff, 1), (0x80000000, 0x7fffffff, 1, 0, 1),
                                (0x80000000, 0x7fffffff, 0, 0xffffffff, 0), (7, 8, 3, 16, 0)):
    env = run(emitted, {'v2': a, 'v3': b}, vcc=cin | 0xfffffffe)  # only bit 0 of the carry in counts
    check('add %#x + %#x + %d' % (a, b, cin), env['v5'] == total and env['vcc'] == cout, repr((env['v5'], env['vcc'])))

# 3. The long form with an SGPR mask out and a known mask in.
asm, emitted, stage = translate(encode_e64(7, 20, VGPR + 4, VGPR + 6, 22), masks=(22,))
check('decodes the e64 form', asm == ['v_add_co_ci_u32_e64 v7, s20, v4, v6, s22'], repr(asm))
check('carry out lands in the mask register', 's20 = ' in emitted and 20 in stage.masks, emitted)
env = run(emitted, {'v4': 0xffffffff, 'v6': 0}, s={'s22': 1})
check('e64 carry chain', env['v7'] == 0 and env['s20'] == 1, repr((env['v7'], env['s20'])))

# 4. Under an unknown exec the write is guarded and the carry is masked by exec.
asm, emitted, stage = translate(encode_e32(5, VGPR + 2, 3), exec_known=False)
check('guarded by exec', 'if (exec & 1) v5 =' in emitted and '(exec & ' in emitted, emitted)

# 5. A carry in from a register that is not a known mask is refused, not read as a number.
try:
    translate(encode_e64(7, 20, VGPR + 4, VGPR + 6, 22))
    refused = False
except G.Unsupported:
    refused = True
check('carry in from an unknown register is refused', refused)

# 6. A lane-mask register (vcc, exec, a known mask SGPR) as a numeric addend is refused (its 0/1 lane bit is not the word).
for label, code, masks in (('vcc_lo as src0', encode_e32(5, 106, 3), ()),
                           ('vcc_lo as src1', encode_e64(5, 106, VGPR + 2, 106, 106), ()),
                           ('exec_lo as src0', encode_e32(5, 126, 3), ()),
                           ('known mask SGPR as src1', encode_e64(7, 20, VGPR + 4, 22, 22), (22,))):
    try:
        translate(code, masks=masks)
        refused = False
    except G.Unsupported:
        refused = True
    check('lane mask addend refused: ' + label, refused)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
