"""`off` in a compressed export and `vcc_hi` as a source in tools/startup/gfx2hlsl.py (#1723).

Pure Python translation controls, no arguments.

`exp mrt0 v0, v0, off, off compr` disables the second register of the pair: it exports as zero, as an uncompressed
`off` does. In wave32 `vcc_hi` holds no lane, so a read of it is zero.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


class Pipe:
    def mrt_kind(self, index):
        return 'float'


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for the instructions below to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True
        self.kind = 'ps'
        self.pipe = Pipe()
        self.exports = {}
        self.write_masks = {}


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def translate(name, asm):
    stage = Probe()
    try:
        stage.one(name, asm)
    except G.Unsupported as error:
        return None, str(error)
    return stage, '\n'.join(stage.lines)


def main():
    stage, text = translate('exp', 'exp mrt0 v0, v0, off, off compr')
    check('compressed export with off translates', stage is not None, text)
    if stage is not None:
        halves = re.findall(r'f16tof32\(([^)]*)\)', text)
        check('off pair adds no reads', len(halves) == 2, text)
        check('live pair is still read', halves[0] == 'v0', text)

    if stage is not None:
        # The disabled B/A channels are not written: the pipeline write mask (manifest 'wmask') keeps R and G only.
        check('off pair is left out of the write mask', stage.write_masks == {'mrt0': 0x3}, repr(stage.write_masks))

    stage, text = translate('v_mov_b32', 'v_mov_b32 v3, vcc_hi')
    check('vcc_hi read without a validated wave32 mode is refused', stage is None
          and 'wave32' in text, text)

    class Wave32(Probe):
        wave32 = True

    stage = Wave32()
    stage.one('v_mov_b32', 'v_mov_b32 v3, vcc_hi')
    text = '\n'.join(stage.lines)
    check('vcc_hi reads zero in a validated wave32 stage', re.search(r'v3 = 0u;', text) is not None
          or re.search(r'v3 = 0x00000000u;', text) is not None, text)

    if failures:
        print('FAIL')
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
