"""Find where a pipeline stores its pixel-shader input masks, from the shaders' own register use.

A pixel shader's live-in vector registers (v0, v1, ...) are loaded by hardware according to two
16-bit masks: the PS input-enable mask (SPI_PS_INPUT_ENA, register byte address 0x286CC, context
index 0x1B3, "mm" index 0xA1B3) and the PS input-address mask (SPI_PS_INPUT_ADDR, 0x286D0, 0x1B4,
0xA1B4). Bit 0..15 select PERSP_SAMPLE, PERSP_CENTER, PERSP_CENTROID (two registers each),
PERSP_PULL_MODEL (three), LINEAR_SAMPLE, LINEAR_CENTER, LINEAR_CENTROID (two each), LINE_STIPPLE_TEX,
POS_X/Y/Z/W_FLOAT, FRONT_FACE, ANCILLARY, SAMPLE_COVERAGE and POS_FIXED_PT (one each). The register
layout is the set bits of the address mask in bit order, each taking its width.

Where a pipeline stores the two masks is not known. This tool does not assume a location. For every
pixel shader it derives a contract from the instruction stream alone (which vector registers are
read before any write, which of them feed attribute interpolation as an I/J pair, which are split
into 16-bit halves), then asks, for every candidate location in the pipeline's byte regions, whether
the value read there is a mask whose register layout satisfies the contract.

Two kinds of agreement are kept apart. Mask agreement (rules C0 range, C1 coverage, C2 pairs) decides
consistency and the verdict; "tight" counts consistent pipelines whose layout has exactly as many
registers as the shader's highest live-in register needs, which separates a real mask from a mask
with every bit set. Packed-position agreement (the register a 16-bit split live-in occupies against
the register the address mask predicts for bit 15) is reported on its own and never votes: it tests
the assumed register widths, not a location.

A location that stays consistent over a corpus of shaders with different contracts is evidence about
where the masks are stored. It is not a proof of hardware behaviour: a matching location can still
be a coincidence of the corpus, and a corpus in which no shader reads a live-in register (an
uninformative corpus) cannot discriminate between locations at all. Output is aggregate by
default: pipeline ids are printed only with --list and are always kept in the --json file.

Usage:
    python -B tools/startup/ps_input_scan.py [<request_dir>] [--capture <capture.bin>]
        [--pixel-dir <dir>] [--json <file>] [--list] [--top N] [--region-limit BYTES]
    --pixel-dir: every regular file directly in the directory is one pixel shader (a DXBC container
    or a raw blob). Pipelines are numbered p000001... in sorted file-name order; names are never printed.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
from collections import Counter, namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402  (also puts tools/ on sys.path)

# Register count of each address-mask bit 0..15 (the same table as Pipeline.ps_hw_inputs).
WIDTHS = (2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1)
PAIR_BITS = (0, 1, 2, 4, 5, 6)   # the two-register I/J inputs that interpolation can read
PACKED_BIT = 15                  # POS_FIXED_PT: two 16-bit halves in one register

BYTE_ADDRESS = {'ena': 0x286CC, 'addr': 0x286D0}
CONTEXT_INDEX = {'ena': 0x1B3, 'addr': 0x1B4}
MM_INDEX = {'ena': 0xA1B3, 'addr': 0xA1B4}

Contract = namedtuple('Contract', 'live_in ij_pairs split16 ambiguous', defaults=(frozenset(),))

# Forms whose operands are all sources even though the first one may be a vector register.
ALL_SOURCE_PREFIXES = ('v_cmp', 'ds_write', 'v_nop')
READ_WRITE = ('v_swap_b32', 'v_swaprel_b32')
UNKNOWN_FLOW = ('s_setpc_b64', 's_swappc_b64', 's_rfe_b64', 's_call_b64')


# ---- A. contract of one pixel shader -------------------------------------------------------------

def base_name(name):
    while True:
        stripped = re.sub(r'_(e32|e64|sdwa|dpp16|dpp8|dpp)$', '', name)
        if stripped == name:
            return name
        name = stripped


def operand_vregs(token):
    """'v1', '|v1|', '-v1', 'sext(v1)', 'v[4:6]' -> (first, last) VGPR numbers, else None."""
    text = token.strip()
    while True:
        previous = text
        text = text.strip('-~!|')
        m = re.fullmatch(r'(?:sext|abs|neg)\((.*)\)', text)
        if m:
            text = m.group(1)
        if text == previous:
            break
    parsed = G.reg_range(text)
    if parsed and parsed[0] == 'v':
        return parsed[1], parsed[2]
    return None


def constant(token):
    try:
        return int(token.strip(), 0)
    except ValueError:
        return None


def all_source(base):
    return base == 'exp' or 'store' in base or base.startswith(ALL_SOURCE_PREFIXES)


def parse_instruction(asm):
    """-> (mnemonic, base, operands, modifiers, destination span, source tokens, written spans)."""
    mnemonic, ops, mods = G.parse(asm)
    base = base_name(mnemonic)
    if mnemonic == 'exp' and ops:
        ops[0] = ops[0].partition(' ')[2]
    destination = operand_vregs(ops[0]) if ops else None
    has_destination = destination is not None and not all_source(base) and base not in READ_WRITE
    sources = ops[1:] if has_destination else ops
    if base in READ_WRITE:   # both operands are read, then both are rewritten
        writes = [span for span in map(operand_vregs, ops) if span]
    else:
        writes = [destination] if has_destination else []
    return mnemonic, base, ops, mods, destination if has_destination else None, sources, writes


def control_flow(decoded):
    """-> (branch targets, backward spans (target, source), unknown): offsets from the label text.
    `unknown` is set for a branch whose target the text does not give and for indirect forms."""
    targets, spans, unknown = set(), [], False
    for off, _, name, asm in decoded:
        base = base_name(name)
        if base == 's_branch' or base.startswith('s_cbranch'):
            m = re.search(r'label_0x([0-9a-f]+)', asm)
            if m is None or 'fork' in base or 'join' in base:
                unknown = True
            else:
                target = int(m.group(1), 16)
                targets.add(target)
                if target <= off:
                    spans.append((target, off))
        elif base in UNKNOWN_FLOW:
            unknown = True
    return targets, spans, unknown


def contract_of_decoded(decoded):
    """Scan decoded instructions, (offset, length, name, asm) as gfx2hlsl.decode returns them.

    A register first read before any write is live-in, unless the read sits inside a loop (a
    backward branch spans it) and the register is written later in program order: then the value may
    be loop-carried, the register is `ambiguous` and takes no part in any rule. Evidence taken from
    a read that is unsafe in that sense is dropped, and aliases do not cross basic blocks."""
    parsed = [(off,) + parse_instruction(asm) for off, _, _, asm in decoded]
    targets, spans, unknown = control_flow(decoded)
    last_write = {}
    for item in parsed:
        for first, last in item[-1]:
            for register in range(first, last + 1):
                last_write[register] = item[0]
    written, holds = set(), {}
    live_in, ambiguous = set(), set()
    interp_i, interp_j = {}, {}
    low, high = set(), set()

    def entry(register):
        """The live-in register whose entry value `register` still holds, else None."""
        return register if register not in written else holds.get(register)

    def single(token):
        span = operand_vregs(token)
        return span[0] if span and span[0] == span[1] else None

    for off, mnemonic, base, ops, mods, destination, sources, writes in parsed:
        if off in targets:
            holds.clear()
        if not ops:
            continue
        in_loop = unknown or any(t <= off <= s for t, s in spans)

        def unsafe(register):
            later = last_write.get(register)
            return in_loop and later is not None and later > off

        for token in sources:
            span = operand_vregs(token)
            if span:
                for register in range(span[0], span[1] + 1):
                    if register not in written:
                        (ambiguous if unsafe(register) else live_in).add(register)

        fields = {}
        for modifier in mods:
            key, _, value = modifier.partition(':')
            if value:
                fields[key] = value

        def held_by(token):
            register = single(token)
            if register is None or unsafe(register):
                return None
            return entry(register)

        # Interpolation: I feeds p1, J feeds p2 (operands: destination, I or J, attribute).
        if base in ('v_interp_p1_f32', 'v_interp_p2_f32') and len(ops) >= 3:
            held = held_by(ops[1])
            if held is not None:
                (interp_i if base == 'v_interp_p1_f32' else interp_j).setdefault(ops[2], set()).add(held)

        # 16-bit half evidence on a register that still holds a live-in value.
        def note(token, found):
            held = held_by(token)
            if held is not None:
                found.add(held)

        args = ops[1:] if destination is not None else ops
        # No SDWA evidence from compare forms: v_cmp has a scalar destination and v_cmpx none, so
        # the source index of a selection would be ambiguous. Losing evidence is acceptable.
        for index, token in enumerate(args[:3] if not base.startswith('v_cmp') else ()):
            selection = fields.get('src%d_sel' % index)
            if selection == 'WORD_0':
                note(token, low)
            elif selection == 'WORD_1':
                note(token, high)
        if base == 'v_and_b32' and len(args) == 2:
            for a, b in ((0, 1), (1, 0)):
                if constant(args[a]) == 0xffff:
                    note(args[b], low)
        elif base in ('v_lshrrev_b32', 'v_ashrrev_i32') and len(args) == 2:
            if constant(args[0]) == 16:
                note(args[1], high)
        elif base in ('v_bfe_u32', 'v_bfe_i32') and len(args) == 3 and constant(args[2]) == 16:
            if constant(args[1]) == 0:
                note(args[0], low)
            elif constant(args[1]) == 16:
                note(args[0], high)

        moved = None
        if (destination is not None and mnemonic in ('v_mov_b32', 'v_mov_b32_e32') and len(ops) == 2
                and not mods and re.fullmatch(r'v\d+', ops[1].strip())):
            moved = held_by(ops[1])
        for first, last in writes:
            for register in range(first, last + 1):
                written.add(register)
                holds.pop(register, None)
        if moved is not None and destination[0] == destination[1]:
            holds[destination[0]] = moved
        if base == 's_branch' or base.startswith('s_cbranch') or base in UNKNOWN_FLOW:
            holds.clear()

    pairs = set()
    for attribute, registers in interp_i.items():
        pairs.update(r for r in registers if r + 1 in interp_j.get(attribute, ()))
    return Contract(frozenset(live_in), frozenset(pairs), frozenset(low & high),
                    frozenset(ambiguous - live_in))


def contract(code):
    """Contract of a pixel shader's instruction stream. Raises gfx2hlsl.Unsupported if undecodable."""
    return contract_of_decoded(G.decode(code))


# ---- B. layout and consistency -------------------------------------------------------------------

def layout(addr):
    """-> (register count, {bit: first register}) for the set bits of an address mask."""
    registers, position = {}, 0
    for bit in range(16):
        if addr >> bit & 1:
            registers[bit] = position
            position += WIDTHS[bit]
    return position, registers


_INFO = {}


def mask_info(addr):
    """-> (register count, bit mask of pair-start registers, bit-15 register or -1), cached."""
    info = _INFO.get(addr)
    if info is None:
        count, registers = layout(addr)
        info = _INFO[addr] = (count, sum(1 << registers[b] for b in PAIR_BITS if b in registers),
                              registers.get(PACKED_BIT, -1))
    return info


def consistent(shader, addr):
    """Mask agreement: -> (True, '') or (False, first failing rule C0, C1 or C2). Packed does not vote."""
    if not 0 <= addr <= 0xffff:
        return False, 'C0 range'
    count, registers = layout(addr)
    if any(r >= count for r in shader.live_in):
        return False, 'C1 coverage'
    starts = {registers[bit] for bit in PAIR_BITS if bit in registers}
    if any(r not in starts for r in shader.ij_pairs):
        return False, 'C2 pairs'
    return True, ''


def packed_agrees(shader, addr):
    """None without a split register; else whether every split register is the bit-15 register."""
    if not shader.split16:
        return None
    if not 0 <= addr <= 0xffff:
        return False
    register = layout(addr)[1].get(PACKED_BIT)
    return register is not None and all(r == register for r in shader.split16)


_COMBOS = []


def combos():
    """Counter of (register count, pair-start mask) over all 65536 address masks."""
    if not _COMBOS:
        counter = Counter()
        for addr in range(0x10000):
            count, starts, _ = mask_info(addr)
            counter[(count, starts)] += 1
        _COMBOS.append(counter)
    return _COMBOS[0]


_SELECTIVITY = {}


def selectivity(shader):
    """Fraction of all 65536 address masks that satisfy C1 and C2 for the contract."""
    if shader not in _SELECTIVITY:
        need = sum(1 << r for r in shader.ij_pairs)
        top = max(shader.live_in, default=-1)
        hits = sum(n for (count, starts), n in combos().items() if count > top and need & ~starts == 0)
        _SELECTIVITY[shader] = hits / 0x10000
    return _SELECTIVITY[shader]


def contract_bits(shader):
    """-log2 of the selectivity; 16 bits when no mask at all satisfies the contract."""
    fraction = selectivity(shader)
    return -math.log2(fraction) if fraction else 16.0


def informative(shader):
    return bool(shader.live_in)


# Per (contract, address) verdict packed in one small integer for the scan's inner loop.
RULE_MASK, TIGHT, PACKED_APPLIES, PACKED_OK, INFORMATIVE, CONSISTENT_INFORMATIVE = 3, 4, 8, 16, 32, 64
RULES = ('', 'C1 coverage', 'C2 pairs', 'C0 range')   # rule code 0 = consistent, 3 = C0


_IDS = []


def address_ids():
    """-> (ids, infos): ids[addr] indexes infos, the distinct (count, pair starts, bit-15 register)
    triples over all 65536 masks. Index 0 stands for a value above 0xffff (rule C0)."""
    if not _IDS:
        index, infos, ids = {}, [None], []
        for addr in range(0x10000):
            info = mask_info(addr)
            if info not in index:
                index[info] = len(infos)
                infos.append(info)
            ids.append(index[info])
        _IDS.extend((ids, infos))
    return _IDS


class Judge:
    """The verdict code of one contract for every kind of mask value, filled in on demand."""
    __slots__ = ('max_live', 'need', 'split', 'inf', 'weight', 'codes', 'ids', 'infos')

    def __init__(self, shader):
        self.max_live = max(shader.live_in, default=-1)
        self.need = sum(1 << r for r in shader.ij_pairs)
        self.split = tuple(shader.split16)
        self.inf = informative(shader)
        self.weight = contract_bits(shader) if self.inf else 0.0
        self.ids, self.infos = address_ids()
        self.codes = [None] * len(self.infos)

    def code(self, addr):
        index = self.ids[addr] if 0 <= addr <= 0xffff else 0
        code = self.codes[index]
        return code if code is not None else self.code_of(index)

    def code_of(self, index):
        code = INFORMATIVE if self.inf else 0
        if self.split:
            code |= PACKED_APPLIES
        if index == 0:
            code |= 3
        else:
            count, starts, packed = self.infos[index]
            if self.max_live >= count:
                code |= 1
            elif self.need & ~starts:
                code |= 2
            elif self.inf:
                code |= CONSISTENT_INFORMATIVE
                if count == self.max_live + 1:
                    code |= TIGHT
            if self.split and packed >= 0 and all(r == packed for r in self.split):
                code |= PACKED_OK
        self.codes[index] = code
        return code


# ---- C. pipelines and regions --------------------------------------------------------------------

class Skip(Exception):
    pass


class PipelineRecord:
    def __init__(self, ident, shader, regions):
        self.ident = ident
        self.contract = shader
        self.regions = regions


def raw_regions(blob):
    at = blob.find(G.PREFETCH, 0, 0x80)
    return {'ps.header': blob[:at] if at >= 0 and at % 4 == 0 else b''}


def part_name(tag):
    printable = len(tag) == 4 and all(0x20 < c < 0x7f for c in tag)
    return 'ps.part.' + (tag.decode('ascii') if printable else tag.hex())


def container_regions(blob):
    """Mirror pixel_aot.container_code to also keep the bytes around the pixel code."""
    import zlib
    from pixel_aot import container_code, inflate
    from resource_inventory import parts
    try:
        container = parts(blob)
        program = container[b'DXIL']
        code = container_code(program, stage=0)
        table = program.index(b'XDAT', 24)
        count, = struct.unpack_from('<I', program, table + 4)
        entries = list(struct.iter_unpack('<3I', program[table + 12:table + 12 + count * 12]))
        native, = [(o, n) for tag, o, n in entries if tag == 13]
        offset, length = native
        size, = struct.unpack_from('<I', program, offset + 16)
        image = inflate(program[offset + 24:offset + length], size, 15)
        start = struct.unpack_from('<I', image, 192)[0] + 40
        end = start + len(code)
        if image[start:end] != code:
            raise ValueError('code is not a slice of the native image')
        regions = {'ps.fssm.head': image[:start], 'ps.fssm.tail': image[end:]}
        seen = Counter()
        for tag, o, n in entries:
            if tag == 13:
                continue
            if o + n > len(program):
                raise ValueError('XDAT entry outside the program part')
            seen[tag] += 1
            name = 'ps.xdat.%d' % tag + ('' if seen[tag] == 1 else '.%d' % seen[tag])
            regions[name] = program[o:o + n]
        for tag, payload in container.items():
            if tag != b'DXIL':
                regions[part_name(tag)] = payload
    except (KeyError, ValueError, struct.error, zlib.error) as error:
        raise Skip('container: %s' % error)
    return code, regions


def pixel_blob(blob):
    """-> (pixel code, regions) for one pixel shader that is a container or a raw blob."""
    try:
        if blob.startswith(b'DXBC'):
            code, regions = container_regions(blob)
        else:
            code, regions = G.strip_header(blob), raw_regions(blob)
        kind = G.classify(code)
    except G.Unsupported as error:
        raise Skip('undecodable: %s' % error)
    if kind != 'ps':
        raise Skip('stages: [%r]' % kind)
    return code, regions


def request_pixel_blob(a, b):
    """-> (pixel code, regions), choosing the pixel blob the way gfx2hlsl.translate_request does."""
    try:
        if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
            if not b.startswith(b'DXBC'):
                raise Skip('container: pixel blob is not a container')
            code, regions = container_regions(b)
            if G.classify(code) != 'ps':
                raise Skip('stages: container pixel blob does not classify as pixel')
            return code, regions
        kinds = {G.classify(a): a, G.classify(b): b}
    except G.Unsupported as error:
        raise Skip('undecodable: %s' % error)
    if set(kinds) != {'vs', 'ps'}:
        raise Skip('stages: %s' % sorted(kinds))
    blob = kinds['ps']
    return G.strip_header(blob), raw_regions(blob)


def make_record(ident, code, regions):
    try:
        shader = contract(code)
    except G.Unsupported as error:
        raise Skip('undecodable: %s' % error)
    return PipelineRecord(ident, shader, regions)


SIBLING_LIMIT = 1 << 20


def scan_requests(directory, records, skipped):
    listing = sorted(os.listdir(directory))
    siblings = {}
    for name in listing:
        m = re.fullmatch(r'req_([0-9a-f]{16})\.(.+)', name)
        if m and m.group(2) not in ('a', 'b', 'root', 'rtv'):
            siblings.setdefault(m.group(1), []).append((m.group(2), name))
    for name in listing:
        m = re.fullmatch(r'req_([0-9a-f]{16})\.a', name)
        if not m:
            continue
        ident = m.group(1)
        try:
            base = os.path.join(directory, 'req_' + ident)
            try:
                a, b = (open(base + ext, 'rb').read() for ext in ('.a', '.b'))
            except OSError as error:
                raise Skip('unreadable: %s' % error.strerror)
            code, regions = request_pixel_blob(a, b)
            for rest, sibling in siblings.get(ident, ()):
                path = os.path.join(directory, sibling)
                try:
                    if os.path.isfile(path) and os.path.getsize(path) <= SIBLING_LIMIT:
                        with open(path, 'rb') as file:
                            regions['req.' + rest] = file.read()
                except OSError:
                    pass
            records.append(make_record(ident, code, regions))
        except Skip as reason:
            skipped.append((ident, str(reason)))


def scan_pixel_dir(directory, records, skipped):
    """One pixel shader per regular file; ids are 1-based positions, never the file names."""
    names = sorted(n for n in os.listdir(directory) if os.path.isfile(os.path.join(directory, n)))
    for index, name in enumerate(names, 1):
        ident = 'p%06d' % index
        try:
            try:
                with open(os.path.join(directory, name), 'rb') as file:
                    blob = file.read()
            except OSError as error:
                raise Skip('unreadable: %s' % error.strerror)
            code, regions = pixel_blob(blob)
            records.append(make_record(ident, code, regions))
        except Skip as reason:
            skipped.append((ident, str(reason)))


def scan_capture(path, records, skipped):
    import capture
    cap = capture.Capture(path)
    done = set()
    for call in cap.calls:
        w = call['args'][0]
        if call['slot'] != 25 or w in done:
            continue
        done.add(w)
        ident = '%x' % (w & 0xffffff)
        pso = cap.psos.get(w)
        seen = {w}
        while pso and pso['path'] == 55 and pso['base']:
            if pso['base'] in seen:
                pso = None
                break
            seen.add(pso['base'])
            pso = cap.psos.get(pso['base'])
        if not pso or pso['path'] != 57:
            continue
        try:
            desc = pso['desc']
            if len(desc) < 96:
                raise Skip('pso: description is %d bytes, need 96' % len(desc))
            q = struct.unpack_from('<12Q', desc, 0)
            a, b = cap.memory.read(q[5], q[6]), cap.memory.read(q[9], q[10])
            if not a or not b or len(a) != q[6] or len(b) != q[10]:
                raise Skip('pso: code not captured')
            code, regions = request_pixel_blob(a, b)
            regions['pso.desc'] = bytes(desc)
            for label, (pointer, size) in (('pso.state_a', q[1:3]), ('pso.state_b', q[3:5])):
                if 0 < size <= 0x1000:
                    data = cap.memory.read(pointer, size)
                    if data is not None and len(data) == size:
                        regions[label] = bytes(data)
            records.append(make_record(ident, code, regions))
        except Skip as reason:
            skipped.append((ident, str(reason)))


# ---- D. candidate locations ----------------------------------------------------------------------

KIND_RANK = {'tag:byteaddr': 0, 'tag:ctxindex': 0, 'tag:ctxrun': 0, 'tag:mmindex': 0,
             'pair32': 1, 'pair32r': 2, 'pair16': 3, 'pair16r': 4, 'addr32': 5, 'addr16': 6}
TAG_KINDS = ('tag:byteaddr', 'tag:ctxindex', 'tag:ctxrun', 'tag:mmindex')


def candidate_name(kind, offset):
    return kind if offset < 0 else '%s@%d' % (kind, offset)


def find_word(words, value):
    for index, word in enumerate(words):
        if word == value:
            return index
    return None


def unpack(data):
    return (list(struct.unpack('<%dI' % (len(data) // 4), data[:len(data) // 4 * 4])),
            list(struct.unpack('<%dH' % (len(data) // 2), data[:len(data) // 2 * 2])))


def tag_values(words):
    """{tag kind: (ena, addr)} for the tagged candidates found in a region's words."""
    found = {}

    def following(tags):
        values = []
        for key in ('ena', 'addr'):
            at = find_word(words, tags[key])
            if at is None or at + 1 >= len(words):
                return None
            values.append(words[at + 1])
        return tuple(values)

    for kind, tags in (('tag:byteaddr', BYTE_ADDRESS), ('tag:ctxindex', CONTEXT_INDEX),
                       ('tag:mmindex', MM_INDEX)):
        values = following(tags)
        if values:
            found[kind] = values
    at = find_word(words, CONTEXT_INDEX['ena'])
    if at is not None and at + 2 < len(words):
        found['tag:ctxrun'] = (words[at + 1], words[at + 2])
    return found


def candidate_value(words, halves, tags, kind, offset):
    """(ena, addr) a candidate reads in one region, or None when it is absent."""
    if kind in TAG_KINDS:
        return tags.get(kind)
    wide = kind.endswith(('32', '32r'))
    values = words if wide else halves
    i = offset // (4 if wide else 2)
    if kind.startswith('addr'):
        return (None, values[i]) if i < len(values) else None
    if i + 1 >= len(values):
        return None
    return (values[i], values[i + 1]) if not kind.endswith('r') else (values[i + 1], values[i])


def relation(ena, addr):
    if ena == addr:
        return 0
    if ena & ~addr == 0:
        return 1
    if addr & ~ena == 0:
        return 2
    return 3


class Counts:
    """Verdict-code histogram and consistent-informative bit total of one candidate."""
    __slots__ = ('c', 'bits')

    def __init__(self):
        self.c = [0] * 128
        self.bits = 0.0


class Judges(dict):
    def __missing__(self, shader):
        value = self[shader] = Judge(shader)
        return value


def tally(records, limit):
    """Pass 1: per (region, kind) a list of Counts by position. Pair kinds are derived from the
    address-reading kinds: pair32@i reads the address word of addr32@(i+1), and pair32r@i reads that
    of addr32@i except at a region's last position, which is counted separately in `last`."""
    judges = Judges()
    reads = {}      # (region, 'addr32' | 'addr16') -> [Counts per position]
    last = {}       # the same, for the final position of each region only
    tags = {}       # (region, tag kind) -> Counts
    for record in records:
        judge = judges[record.contract]
        codes, weight, ids = judge.codes, judge.weight, judge.ids
        for region, data in record.regions.items():
            words, halves = unpack(data[:limit])
            for kind, values in (('addr32', words), ('addr16', halves)):
                if not values:
                    continue
                table = reads.setdefault((region, kind), [])
                if len(table) < len(values):
                    table.extend(Counts() for _ in range(len(values) - len(table)))
                for index, addr in enumerate(values):
                    code = codes[ids[addr] if addr < 0x10000 else 0]
                    if code is None:
                        code = judge.code(addr)
                    entry = table[index]
                    entry.c[code] += 1
                    if code & CONSISTENT_INFORMATIVE:
                        entry.bits += weight
                end = len(values) - 1
                tail = last.setdefault((region, kind), {})
                entry = tail.get(end)
                if entry is None:
                    entry = tail[end] = Counts()
                code = judge.code(values[end])
                entry.c[code] += 1
                if code & CONSISTENT_INFORMATIVE:
                    entry.bits += weight
            for kind, (_, addr) in tag_values(words).items():
                entry = tags.get((region, kind))
                if entry is None:
                    entry = tags[(region, kind)] = Counts()
                code = judge.code(addr)
                entry.c[code] += 1
                if code & CONSISTENT_INFORMATIVE:
                    entry.bits += weight
    return reads, last, tags


class Stat:
    """Summary of one candidate over the pipelines where it is present."""

    def __init__(self, region, kind, offset, counts):
        self.region, self.kind, self.offset = region, kind, offset
        self.bits = counts.bits
        self.present = self.present_informative = self.consistent = 0
        self.informative_consistent = self.tight = self.packed_total = self.packed_agree = 0
        self.fails = [0, 0, 0]   # C0, C1, C2
        for code, n in enumerate(counts.c):
            if not n:
                continue
            self.present += n
            rule = code & RULE_MASK
            if code & INFORMATIVE:
                self.present_informative += n
            if rule == 0:
                self.consistent += n
                if code & CONSISTENT_INFORMATIVE:
                    self.informative_consistent += n
                    if code & TIGHT:
                        self.tight += n
            else:
                self.fails[{3: 0, 1: 1, 2: 2}[rule]] += n
            if code & PACKED_APPLIES:
                self.packed_total += n
                if code & PACKED_OK:
                    self.packed_agree += n
        self.inconsistent = self.present - self.consistent
        # filled by the detail pass
        self.addrs = set()
        self.relations = [0, 0, 0, 0]   # ena==addr, ena subset of addr, addr subset of ena, other
        self.example = None
        self.signature = 0xcbf29ce484222325

    @property
    def name(self):
        return candidate_name(self.kind, self.offset)

    def key(self):
        return (self.inconsistent, -self.tight, -round(self.bits, 6), -self.informative_consistent)


def subtract(counts, other):
    result = Counts()
    result.c = [a - b for a, b in zip(counts.c, other.c)]
    result.bits = counts.bits - other.bits
    return result


def candidate_stats(reads, last, tags):
    stats = []
    for (region, kind), table in reads.items():
        step = 4 if kind == 'addr32' else 2
        pair = kind.replace('addr', 'pair')
        for index, counts in enumerate(table):
            stats.append(Stat(region, kind, index * step, counts))
            if index:
                stats.append(Stat(region, pair, (index - 1) * step, counts))
            tail = last[(region, kind)].get(index)
            reduced = subtract(counts, tail) if tail else counts
            if any(reduced.c):
                stats.append(Stat(region, pair + 'r', index * step, reduced))
    for (region, kind), counts in tags.items():
        stats.append(Stat(region, kind, -1, counts))
    return stats


SELECT_MIN, SELECT_MAX = 300, 1500
FNV_PRIME, FNV_MASK = 0x100000001b3, (1 << 64) - 1


def detail(records, stats, limit):
    """Pass 2 for the best-ranked candidates: addresses, relations, example failure, signature."""
    stats.sort(key=lambda s: s.key() + (s.region, KIND_RANK[s.kind], s.offset))
    selected, truncated = [], False
    for stat in stats:
        if len(selected) >= SELECT_MIN and stat.key() != selected[-1].key():
            break
        if len(selected) >= SELECT_MAX:
            truncated = True
            break
        selected.append(stat)
    by_region = {}
    for stat in selected:
        by_region.setdefault(stat.region, []).append(stat)
    judges = Judges()
    for index, record in enumerate(records):
        judge = judges[record.contract]
        for region, wanted in by_region.items():
            data = record.regions.get(region)
            if data is None:
                continue
            words, halves = unpack(data[:limit])
            tags = tag_values(words)
            for stat in wanted:
                value = candidate_value(words, halves, tags, stat.kind, stat.offset)
                if value is None:
                    continue
                ena, addr = value
                code = judge.code(addr)
                h = ((stat.signature ^ (index + 1)) * FNV_PRIME) & FNV_MASK
                stat.signature = ((h ^ addr) * FNV_PRIME) & FNV_MASK
                if code & RULE_MASK == 0:
                    stat.addrs.add(addr)
                elif stat.example is None:
                    stat.example = (record.ident, RULES[code & RULE_MASK])
                if ena is not None:
                    stat.relations[relation(ena, addr)] += 1
    return selected, truncated


def group(selected):
    """Group candidates that read the same address value everywhere; rank the groups."""
    classes = {}
    for stat in selected:
        classes.setdefault((stat.region, stat.signature), []).append(stat)
    ranked = []
    for members in classes.values():
        # The member whose enable value agrees best with the address is the representative.
        members.sort(key=lambda s: (s.relations[3], -s.relations[0], KIND_RANK[s.kind], s.offset))
        ranked.append(members)
    ranked.sort(key=lambda m: m[0].key() + (m[0].region, KIND_RANK[m[0].kind], m[0].offset))
    return ranked


def is_full(stat, informative_count):
    return (informative_count > 0 and stat.inconsistent == 0 and stat.informative_consistent >= 1
            and stat.present_informative == informative_count)


def verdict_of(ranked, informative_count):
    if informative_count == 0:
        return 'uninformative corpus (no live-in vector registers)', []
    full = [m for m in ranked if is_full(m[0], informative_count)]
    if not full:
        return 'no location is consistent with all pipelines', []
    best = max(m[0].tight for m in full)
    leaders = [m for m in full if m[0].tight == best]
    if len(leaders) == 1:
        return 'unique full match %s %s' % (leaders[0][0].region, leaders[0][0].name), full
    listing = ', '.join('%s %s' % (m[0].region, m[0].name) for m in leaders)
    return '%d full matches (ambiguous): %s' % (len(leaders), listing), full


# ---- E. report -----------------------------------------------------------------------------------

def row(stat):
    return {'region': stat.region, 'candidate': stat.name, 'present': stat.present,
            'consistent': stat.consistent, 'inconsistent': stat.inconsistent,
            'informative_consistent': stat.informative_consistent, 'tight': stat.tight,
            'bits': round(stat.bits, 1), 'distinct_addr': len(stat.addrs),
            'packed_total': stat.packed_total, 'packed_agree': stat.packed_agree,
            'fail_c0': stat.fails[0], 'fail_c1': stat.fails[1], 'fail_c2': stat.fails[2],
            'relations': dict(zip(('equal', 'ena_subset', 'addr_subset', 'other'), stat.relations)),
            'example_inconsistent': list(stat.example) if stat.example else None}


def hex_or_dash(value):
    return '-' if value is None else '0x%x' % value


FOOTER = 'agreement is evidence about where the masks are stored; it does not prove hardware behaviour'


def run(directory, capture_path, pixel_dir, top, limit, listing):
    records, skipped = [], []
    if directory:
        scan_requests(directory, records, skipped)
    if pixel_dir:
        scan_pixel_dir(pixel_dir, records, skipped)
    if capture_path:
        scan_capture(capture_path, records, skipped)
    records.sort(key=lambda r: r.ident)
    scanned = len(records)
    excluded = [r for r in records if informative(r.contract) and selectivity(r.contract) == 0]
    records = [r for r in records if not (informative(r.contract) and selectivity(r.contract) == 0)]
    informative_count = sum(informative(r.contract) for r in records)
    distinct = {r.contract for r in records}
    stats = candidate_stats(*tally(records, limit))
    selected, truncated = detail(records, stats, limit)
    ranked = group(selected)
    verdict, full = verdict_of(ranked, informative_count)

    lines = ['pipelines: %d scanned, %d skipped' % (scanned, len(skipped))]
    categories = Counter(reason.split(':')[0] for _, reason in skipped)
    for category, count in sorted(categories.items()):
        first = next(r for _, r in skipped if r.split(':')[0] == category)
        lines.append('  skipped %s: %d (first reason: %s)' % (category, count, first))
    lines.append('informative contracts: %d of %d; distinct contracts: %d'
                 % (informative_count, len(records), len(distinct)))
    lines.append('ambiguous live-in reads (inside a loop, written later): %d pipelines, excluded registers '
                 'do not vote' % sum(bool(r.contract.ambiguous) for r in records))
    lines.append('contracts no mask can satisfy: %d pipelines (excluded)' % len(excluded))
    if records and informative_count == 0:
        lines.append('corpus is uninformative: no pixel shader reads a live-in vector register, '
                     'so no location can be told apart')
    lines.append('rank region candidate present consistent inconsistent informative tight bits '
                 'distinct_addr packed(agree/total) relations(eq,ena<addr,addr<ena,other) equivalent')
    for number, members in enumerate(ranked[:top], 1):
        stat = members[0]
        lines.append('%d %s %s %d %d %d %d %d %.1f %d %d/%d %s %s' % (
            number, stat.region, stat.name, stat.present, stat.consistent, stat.inconsistent,
            stat.informative_consistent, stat.tight, stat.bits, len(stat.addrs),
            stat.packed_agree, stat.packed_total, ','.join(map(str, stat.relations)),
            '+%d' % (len(members) - 1) if len(members) > 1 else '-'))
    if truncated:
        lines.append('note: the detail pass stopped at %d candidates inside a tie; equivalent counts '
                     'for tied candidates may be partial' % SELECT_MAX)
    lines.append('verdict: ' + verdict)
    per_pipeline = []
    top_stat = ranked[0][0] if ranked else None
    if top_stat:
        lines.append('mask agreement: %d of %d pipelines consistent (coverage and pair rules), %d tight, '
                     'failures C0=%d C1=%d C2=%d' % (top_stat.consistent, top_stat.present, top_stat.tight,
                                                     *top_stat.fails))
        lines.append('packed-position agreement: %d of %d pipelines that split a live-in register into '
                     '16-bit halves have it at the register the address mask predicts for bit 15 '
                     '(agreement with the assumed register widths, not a proof of the hardware '
                     'register order)' % (top_stat.packed_agree, top_stat.packed_total))
        if any(r.contract.split16 for r in records):
            agreeing = ['%s %s' % (m[0].region, m[0].name) for m in full
                        if m[0].packed_total and m[0].packed_agree == m[0].packed_total]
            lines.append('full matches that also agree on every packed split: %s' % (', '.join(agreeing) or 'none'))
        for record in records:
            data = record.regions.get(top_stat.region)
            value = None
            if data is not None:
                words, halves = unpack(data[:limit])
                value = candidate_value(words, halves, tag_values(words), top_stat.kind, top_stat.offset)
            ena, addr = value if value else (None, None)
            predicted = None
            if addr is not None and 0 <= addr <= 0xffff:
                predicted = layout(addr)[1].get(PACKED_BIT)
            per_pipeline.append({'id': record.ident, 'present': value is not None, 'ena': ena, 'addr': addr,
                                 'packed_register': predicted,
                                 'live_in': sorted(record.contract.live_in),
                                 'split16': sorted(record.contract.split16),
                                 'ambiguous': sorted(record.contract.ambiguous)})
        if listing:
            lines.append('top candidate %s %s per pipeline:' % (top_stat.region, top_stat.name))
            lines.append('id ena addr packed_register live_in split16 ambiguous')
            for item in per_pipeline:
                lines.append('%s %s %s %s %s %s %s' % (
                    item['id'], hex_or_dash(item['ena']),
                    'absent' if not item['present'] else hex_or_dash(item['addr']),
                    '-' if item['packed_register'] is None else item['packed_register'],
                    ','.join('v%d' % r for r in item['live_in']) or '-',
                    ','.join('v%d' % r for r in item['split16']) or '-',
                    ','.join('v%d' % r for r in item['ambiguous']) or '-'))
            if top_stat.example:
                lines.append('first inconsistent pipeline for the top candidate: %s (%s)' % top_stat.example)
    lines.append(FOOTER)
    result = {
        'pipelines': scanned, 'skipped': [{'id': i, 'reason': r} for i, r in skipped],
        'informative': informative_count, 'distinct_contracts': len(distinct),
        'excluded_unsatisfiable': [r.ident for r in excluded],
        'ambiguous_pipelines': [r.ident for r in records if r.contract.ambiguous],
        'contracts': [{'id': r.ident, 'live_in': sorted(r.contract.live_in),
                       'ambiguous': sorted(r.contract.ambiguous),
                       'ij_pairs': sorted(r.contract.ij_pairs), 'split16': sorted(r.contract.split16),
                       'informative': informative(r.contract),
                       'bits': round(contract_bits(r.contract), 3) if informative(r.contract) else 0.0}
                      for r in records],
        'candidates': [dict(row(m[0]), rank=n, full_match=is_full(m[0], informative_count),
                            equivalent=[s.name for s in m[1:]])
                       for n, m in enumerate(ranked[:max(top, 1000)], 1)],
        'detail_truncated': truncated,
        'verdict': verdict, 'top_candidate_pipelines': per_pipeline, 'footer': FOOTER,
    }
    return lines, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('request_dir', nargs='?', help='directory with req_<key>.a/.b/.root files')
    parser.add_argument('--capture', help='capture.bin whose path-57 PSOs are scanned')
    parser.add_argument('--pixel-dir', help='directory in which every regular file is one pixel shader')
    parser.add_argument('--json', help='also write the full result (with pipeline ids) to this file')
    parser.add_argument('--list', action='store_true', help='print pipeline ids and the per-pipeline table')
    parser.add_argument('--top', type=int, default=10, help='ranked rows to print (default 10)')
    parser.add_argument('--region-limit', type=int, default=0x1000,
                        help='bytes of each region searched for candidates (default 0x1000)')
    args = parser.parse_args(argv)
    if not (args.request_dir or args.capture or args.pixel_dir):
        parser.error('give a request directory, --pixel-dir, --capture, or a combination')
    for label, path in (('request directory', args.request_dir), ('--pixel-dir', args.pixel_dir)):
        if path and not os.path.isdir(path):
            parser.error('%s is not a directory' % label)
    if args.capture and not os.path.isfile(args.capture):
        parser.error('--capture is not a file')
    if args.top < 1 or args.region_limit < 8:
        parser.error('--top must be positive and --region-limit at least 8')
    lines, result = run(args.request_dir, args.capture, args.pixel_dir, args.top, args.region_limit, args.list)
    print('\n'.join(lines))
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
