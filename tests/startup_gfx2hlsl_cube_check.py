"""Cube-face instructions and cube image sampling in tools/startup/gfx2hlsl.py (#1723).

v_cubeid_f32, v_cubesc_f32, v_cubetc_f32 and v_cubema_f32 turn a direction into a face id, two in-face
coordinates and twice the major axis. The sample that follows addresses the cube as six 2D slices
(s, t, face), so dim:SQ_RSRC_IMG_CUBE declares a Texture2DArray like dim:SQ_RSRC_IMG_2D_ARRAY.
Pure Python: the translated text and the preamble helpers are checked, nothing is compiled.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

CUBE_OPS = ('v_cubeid_f32', 'v_cubesc_f32', 'v_cubetc_f32', 'v_cubema_f32')
HELPERS = {'v_cubeid_f32': 'cubeid', 'v_cubesc_f32': 'cubesc', 'v_cubetc_f32': 'cubetc', 'v_cubema_f32': 'cubema'}


class Pipe:
    def __init__(self):
        self.kinds = []

    def resource(self, _descriptor, kind):
        self.kinds.append(kind)
        return 'r%d' % len(self.kinds)

    def sampler(self, _descriptor):
        return 's0'


class Stage(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.masks = set()
        self.pipe = Pipe()


def lower(name, asm):
    stage = Stage()
    stage.one(name, asm)
    return ' '.join(stage.lines), stage


def refused(call):
    try:
        call()
    except G.Unsupported as error:
        return str(error)
    return None


def main():
    failures = []
    for op, helper in HELPERS.items():
        text, _ = lower(op, '%s v3, v0, v1, v2' % op)
        if '%s(F(v0), F(v1), F(v2))' % helper not in text:
            failures.append('%s: %s' % (op, text))
        # Source modifiers reach the helper as operand text, like any other float source.
        text, _ = lower(op, '%s v3, -v0, |v1|, v2' % op)
        if '%s(' % helper not in text or 'abs(F(v1))' not in text:
            failures.append('%s with modifiers: %s' % (op, text))
        err = refused(lambda: lower(op, '%s v3, v0, v1' % op))
        if not err or 'operands' not in err:
            failures.append('%s with two sources was not refused: %s' % (op, err))

    pipe = object.__new__(G.Pipeline)
    pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
    common = ' '.join(pipe.common())
    for helper in HELPERS.values():
        if ' %s(float x, float y, float z)' % helper not in common:
            failures.append('preamble lacks %s' % helper)

    stage = Stage()
    stage.descriptor = lambda _ops: 'd0'
    stage.heap_sampler = lambda _ops: None
    stage.const_words = lambda _ops: None
    stage.kind, stage.blocks = 'ps', []
    stage.image(['v[4:7]', 'v[0:2]', 's[8:15]', 's[16:19]'], ['dim:SQ_RSRC_IMG_CUBE', 'dmask:0xf'], 'lz')
    if stage.pipe.kinds != ['tex2da'] or 'float3(F(v0) - 1.0, F(v1) - 1.0, cubeslice(F(v2)))' not in ' '.join(stage.lines):
        failures.append('cube sample: %s %s' % (stage.pipe.kinds, stage.lines))
    if ' cubeslice(float z)' not in common:
        failures.append('preamble lacks cubeslice')
    if 'abs(z) >= abs(x) && abs(z) >= abs(y)' not in common:
        failures.append('cubeaxis does not compare |z| with |x| and |y| separately (NaN)')

    for failure in failures:
        print('FAIL', failure)
    if failures:
        return 1
    print('gfx2hlsl cube: ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
