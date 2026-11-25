"""The scalar ALU table refuses a source list shorter than its template uses (#2463).

`Stage.scalar` builds the expression with `text.format(*values)`, where the `SCA` entry spells its sources as
`{0}`, `{1}`, ... A short operand list leaves `values` too short and `str.format` raises IndexError, which the
single-shader path records as `translator crash in <op>: IndexError` rather than as a refusal naming the
instruction. The guard asserts the count the template needs before formatting.

AI-assisted.
"""

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'tools' / 'startup' / 'gfx2hlsl.py'


def load(text=None):
    spec = importlib.util.spec_from_loader('gfx2hlsl_table', loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__['__file__'] = str(SOURCE)
    exec(compile(text if text is not None else SOURCE.read_text(encoding='utf-8'),
                 str(SOURCE), 'exec'), module.__dict__)
    return module


G = load()
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH %s' % detail))
    if not ok:
        failures.append(label)


def probe_for(module):
    """A Stage with no pipeline: enough state for one scalar instruction to be translated."""
    class Probe(module.Stage):
        def __init__(self):
            self.lines, self.sym = [], {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.exec_known = True
            self.masks = set()
    return Probe()


def needed_sources(module, op):
    """How many sources the SCA template for op reads."""
    text, _ = module.SCA[op]
    return max((int(i) for i in re.findall(r'\{(\d+)\}', text)), default=-1) + 1


def refusal(module, op, sources):
    """The Unsupported text for op with that many sources, 'ok', or the name of another exception."""
    line = '%s s1, %s' % (op, ', '.join('s%d' % (2 + i) for i in range(sources))) if sources else '%s s1' % op
    try:
        probe_for(module).one(op, line)
    except module.Unsupported as error:
        return str(error)
    except Exception as error:  # noqa: BLE001  the point is to see which exception a malformed form raises
        return type(error).__name__
    return 'ok'


TWO_SOURCE = sorted(op for op in G.SCA if needed_sources(G, op) >= 2)
check('the SCA table has two-source templates', len(TWO_SOURCE) >= 30, len(TWO_SOURCE))

# The shape that crashed: one source fewer than the template uses.
crashing = [op for op in TWO_SOURCE if refusal(G, op, needed_sources(G, op) - 1) == 'IndexError']
check('no two-source template crashes on a short source list', not crashing, repr(crashing[:6]))

# Each refusal reads as a requirement: the count the instruction carries, not an internal index.
wrong = []
for op in TWO_SOURCE:
    count = needed_sources(G, op)
    message = refusal(G, op, count - 1)
    if not isinstance(message, str) or not message.endswith('expected %d' % (count + 1)):
        wrong.append((op, message))
check('every short form is refused with the operand count it needs', not wrong, repr(wrong[:4]))

# The complete form is untouched by the guard: same verdict as the reference, which is translation or a
# later refusal, never the arity message.
changed = []
for op in TWO_SOURCE:
    count = needed_sources(G, op)
    message = refusal(G, op, count)
    if isinstance(message, str) and message.endswith('expected %d' % (count + 1)):
        changed.append(op)
check('a complete form is never refused for its operand count', not changed, repr(changed[:4]))

# Negative control: with the guard removed the short form raises IndexError again. One template keeps its own
# arity check (s_bfm_b32 spells "expected 3" in its body), so it refuses either way and is excluded here rather
# than counted as a survivor.
stripped = load(SOURCE.read_text(encoding='utf-8').replace('        if len(srcs) < needed:',
                                                          '        if False:'))
control = [op for op in TWO_SOURCE if refusal(stripped, op, needed_sources(stripped, op) - 1) == 'IndexError']
check('control: without the guard the short form raises IndexError', len(control) >= len(TWO_SOURCE) - 1,
      '%d of %d' % (len(control), len(TWO_SOURCE)))
kept = [op for op in TWO_SOURCE if op not in control]
check('the one template with its own arity check still refuses without the guard',
      all(refusal(stripped, op, needed_sources(stripped, op) - 1).startswith(op) for op in kept), repr(kept))

print('scalar table arity checks: %d failed' % len(failures))
sys.exit(1 if failures else 0)
