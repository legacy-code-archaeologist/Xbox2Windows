"""The `exp null` export in tools/startup/gfx2hlsl.py (Part of #1723).

usage: startup_gfx2hlsl_export_null_check.py

A pixel shader that only discards (depth or alpha test with no colour output) ends with
`exp null, off, off, off, off done vm`. The target writes nothing: the translator refused it as
`export target null`, and the stage classifier called a shader whose only export is null "other".

The instruction words are constructed here from the EXP layout the checked-in decoder reads
(tools/disasm_rdna2.py): w0 [31:26] = 0b111110, [3:0] en, [9:4] target (9 = null), [11] done, [12] vm; w1 = vsrc.
"""
import os
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


NULL_TARGET = 9
MRT0_TARGET = 0


def encode_exp(target, *, done=1, vm=1, en=0):
    w0 = (0b111110 << 26) | (vm << 12) | (done << 11) | (target << 4) | en
    return struct.pack('<II', w0, 0)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded export to be translated."""

    def __init__(self, exec_known):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = exec_known
        self.exports = {}
        self.kind = 'ps'
        self.pipe = Pipeline()


def translate(code, exec_known=True):
    decoded = G.decode(code)
    stage = Stage(exec_known)
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return decoded, stage


code = encode_exp(NULL_TARGET)
decoded = G.decode(code)
check('decodes as the null export', [x[3] for x in decoded] == ['exp null off, off, off, off done vm'],
      repr([x[3] for x in decoded]))

try:
    _decoded, stage = translate(code)
    refused = False
except G.Unsupported as e:
    refused, stage = True, None
    print('  ' + str(e))
check('the null export is translated, not refused', not refused)

if stage is not None:
    check('it writes no output', not stage.exports and not any(l.startswith('o.') for l in stage.lines),
          repr((stage.exports, stage.lines)))

try:
    _decoded, stage = translate(code, exec_known=False)
    emitted = ' '.join(stage.lines)
except G.Unsupported:
    emitted = ''
check('vm still discards the pixels the mask turned off', 'discard' in emitted, emitted)

check('a null-only shader is a pixel shader', G.classify_decoded(decoded) == 'ps', G.classify_decoded(decoded))
check('a position export still makes a vertex shader',
      G.classify_decoded([(0, 8, 'exp', 'exp pos0 v0, v1, v2, v3 done')]) == 'vs')
try:
    translate(encode_exp(13))  # pos1 is not modelled
    other_refused = False
except G.Unsupported:
    other_refused = True
check('another unmodelled target stays refused', other_refused)
try:
    translate(struct.pack('<II', 0xf8001c91, 0), exec_known=False)  # compr null export, en:0x1 (not a whole pair)
    partial_en_refused = False
except G.Unsupported:
    partial_en_refused = True
check('a null export with a partial compressed enable mask stays refused (vm is not dropped)', partial_en_refused)

if failures:
    print('FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('all ok')
