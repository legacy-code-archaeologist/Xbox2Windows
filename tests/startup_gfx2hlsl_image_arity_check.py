"""Image instructions refuse a short operand list or a missing dimension/dmask instead of indexing it.

`image()` reads vdata, an address and a descriptor. A list shorter than three used to reach `ops[1]` or
`ops[2]`, and the `dim:`/`dmask:` modifiers are read with `[0]` after filtering, so a form that omitted one
reached that index instead of refusing. Both are recorded by `Stage.translate` as `translator crash in <op>:
IndexError` -- its marker for a translator defect rather than for an instruction it does not lower -- and the
single-shader path does not catch it at all. The accepted forms need a bound descriptor, so this check pins
the refusal side only; the accepted shapes are covered by the per-form fixtures.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

FORMS = ('image_sample', 'image_sample_l', 'image_sample_lz', 'image_sample_b',
         'image_get_lod', 'image_load', 'image_load_mip')
FULL = ('v[8:11]', 'v0', 's[16:19]', 's[20:23]')


def translate(name, operands, mods=''):
    pipe = SimpleNamespace(root={'params': [], 'flags': 0})
    text = ' '.join((name, ', '.join(operands), mods)).strip()
    refusals = []
    stage = G.Stage(pipe, 'ps', b'', refusals=refusals, decoded=[(0, 4, name, text)])
    return refusals, stage.blocks


def refuse(name, operands, expected, mods=''):
    refusals, blocks = translate(name, operands, mods)
    assert len(refusals) == 1, (name, operands, mods, refusals)
    assert not blocks, (name, operands, mods, blocks)
    assert 'translator crash' not in refusals[0][0], (name, operands, mods, refusals)
    assert expected in refusals[0][0], (name, operands, mods, refusals)


def main():
    for form in FORMS:
        refuse(form, ('v[8:11]',), 'image instruction with 1 operands, expected at least 3')
        refuse(form, ('v[8:11]', 'v0'), 'image instruction with 2 operands, expected at least 3')

    # The dimension and dmask decide the emitted descriptor type and the register span, and the disassembler
    # always prints both, so a form without one comes from the text entry rather than from the ISA.
    for form in FORMS:
        refuse(form, FULL, 'without dim:')
        refuse(form, FULL, 'without dmask:', mods='dim:SQ_RSRC_IMG_2D')
        refuse(form, FULL, 'without dim:', mods='dmask:0xf')

    print('startup gfx2hlsl image arity: %d forms refused at 1 and 2 operands and without dim/dmask'
          % len(FORMS))


if __name__ == '__main__':
    main()
