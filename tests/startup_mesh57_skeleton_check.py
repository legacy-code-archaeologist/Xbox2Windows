"""Slot 57 mesh requests of one group-memory shape translate in tools/startup/gfx2hlsl.py (#2413).

usage: startup_mesh57_skeleton_check.py
       startup_mesh57_skeleton_check.py binary <llvm-mc>
       startup_mesh57_skeleton_check.py draw <dxc> <offscreen_check> <warp|hardware>

The fixture (tests/startup_mesh57_skeleton.s, source-built) holds the accepted shape: lane 0 of wave 0 stores a
word and both waves pass a barrier; later the same lane stores the primitive and vertex counts, both waves pass
a barrier, wave 0 sends one allocation request and wave 1 reads the counts back. Each thread then exports the
position and one parameter of vertex `thread` and the packed indices of primitive `thread`, each under its count.
The data of a thread is the buffer slot of its low four bits.

Without arguments: the fixture translates through the slot 57 request path, and each mutation of the shape is
refused with its own message. `binary` compares the embedded words with llvm-mc. `draw` compiles the request
path's output (ms_6_5, ps_6_0), draws it and compares the read-back target with a CPU rasteriser written here;
a reference with one wrong pixel must fail.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

MESH = (
    'bfa00003 befe0481 be84047e 8706ff03 0f000000 bf850007 2c0a0a87 7e0c0280 d8340008 00000506'
    ' bfa3ffe3 bf8a0000 bf820004 bf8a0000 7e0c0280 d8d80008 05000006 befe04c1 9386ff03 00040018'
    ' 8f078606 d7650000 00000ec1 d7660000 000200c1 bf8cc07f 7e180505 f4080405 fa000000 be8d0308'
    ' be8e0309 bf8a0000 bf008006 bf840010 befe0404 bf880007 7e02020d 7e040280 7e06020e d8340000'
    ' 00000102 d8340004 00000302 befe04c1 bfa3ffe3 bf8a0000 8f048c0d 887c0e04 bf900009 bf82000d'
    ' bf8a0000 befe0404 bf880009 7e040280 d8d80000 01000002 d8d80004 03000002 bf8cc17f 7e1a0501'
    ' bf8cc07f 7e1c0503 befe04c1 7d28000e 3608008f 34080884 e0381000 80041004 e03810c0 80041404'
    ' bf8c3f70 f80008cf 13121110 f800020f 17161514 befe04c1 7d28000d 3608008f 34080882 e0301180'
    ' 80041804 bf8c3f70 f8000941 00000018 bf810000').split()
PIXEL_ASM = ('v_interp_mov_f32 v0, p0, attr0.x\nv_interp_mov_f32 v1, p0, attr0.y\nv_interp_mov_f32 v2, p0, attr0.z\n'
             'v_interp_mov_f32 v3, p0, attr0.w\nexp mrt0 v0, v1, v2, v3 done vm\ns_endpgm\n')
PIXEL = 'c8020002 c8060102 c80a0202 c80e0302 f800180f 03020100 bf810000'.split()


def code(words):
    return b''.join(struct.pack('<I', int(word, 16)) for word in words)


def root():
    # Parameter 0: two root constants (primitive count, vertex count). Parameter 1: a descriptor table.
    rts0 = (struct.pack('<6I', 2, 2, 24, 0, 0, 0) + struct.pack('<3I', 1, 0, 48) + struct.pack('<3I', 0, 0, 60)
            + struct.pack('<3I', 0, 0, 2) + struct.pack('<2I', 0, 0))
    return b'DXBC' + bytes(16) + struct.pack('<4I', 1, 44 + len(rts0), 1, 36) + b'RTS0' + struct.pack('<I', len(rts0)) + rts0


def mutate(old, new, nth=0):
    """The fixture with the nth word equal to old replaced."""
    at = [i for i, word in enumerate(MESH) if word == old][nth]
    return MESH[:at] + [new] + MESH[at + 1:]


def request(directory, mesh, pixel=PIXEL):
    ms, ps, blob = code(mesh), code(pixel), root()
    key = G.mesh57_key(ms, ps, blob)
    for ext, data in (('.root', blob), ('.ps57', ps), ('.m57', ms)):
        (Path(directory) / ('req_%016x%s' % (key, ext))).write_bytes(data)
    return key, G.translate_request(str(directory), key)


FAILED = []


def expect(what, got, want):
    if got != want:
        FAILED.append(what)
        print('FAIL %s: got %r, want %r' % (what, got, want))


REFUSALS = (
    # The request must carry the two stored counts, in this order.
    ('a request that publishes another count', mutate('887c0e04', '887c0d04'),
     'slot 57 mesh count exchange: instruction 15 of the accepted shape (s_or_b32) is absent'),
    ('no barrier between the count store and the reads', mutate('bf8a0000', 'bf800000', 3),
     'slot 57 mesh count exchange: instruction 13 of the accepted shape (s_barrier) is absent'),
    ('a count store by every lane', mutate('befe0404', 'befe04c1'),
     'slot 57 mesh count exchange: instruction 4 of the accepted shape (s_mov_b64) is absent'),
    ('no wait before the first barrier', mutate('bfa3ffe3', 'bf800000'),
     'slot 57 mesh group-index exchange: instruction 8 of the accepted shape (s_waitcnt_depctr) is absent'),
    ('a branch that leaves the count exchange', mutate('bf82000d', 'bf82000e'),
     'slot 57 mesh count exchange: a branch leaves the accepted shape'),
    ('a barrier after the count exchange', mutate('bf8c3f70', 'bf8a0000'),
     'slot 57 mesh: s_barrier outside the group-memory exchanges'),
    ('a branch after the count exchange', mutate('bf8c3f70', 'bf820000'),
     'slot 57 mesh: a branch outside the group-memory exchanges is not covered'),
    ('a position export under the primitive count', mutate('7d28000e', '7d28000d'),
     'slot 57 mesh: export pos0 is not under exec = -1 narrowed by its output count > thread index'),
    # v_mov_b32 v7, v1: v1 holds the stored count in wave 0 and the read one elsewhere.
    ('a read of a register the exchange leaves different per wave', mutate('bf8c3f70', '7e0e0301'),
     'slot 57 mesh: v1 is read where it is not defined for every lane (an entry value, or left different per wave '
     'by an exchange)'),
    # v_mov_b32 v7, v10; s_mov_b32 s20, s25; s_mov_b32 s20, s3: entry values the model does not define.
    ('a read of an entry VGPR', mutate('bf8c3f70', '7e0e030a'),
     'slot 57 mesh: v10 is read where it is not defined for every lane (an entry value, or left different per wave '
     'by an exchange)'),
    ('a read of an entry SGPR', mutate('bf8c3f70', 'be940319'),
     'numeric use of s25 holding an entry value the mesh model does not define'),
    ('a read of s3 other than the wave index', mutate('bf8c3f70', 'be940303'),
     'slot 57 mesh: s3 is read other than as the wave index'),
    # s_and_b32 s4, s3, 0xf000000: wave 0 would restore a zero mask and store no count.
    ('a wave test that overwrites the saved lane-0 mask', mutate('8706ff03', '8704ff03'),
     'slot 57 mesh count exchange: registers of the accepted shape overlap'),
    # s_mov_b32 s6, s8 after the thread index is derived: the count compare no longer tells the waves apart.
    ('a count compare of a register rewritten after the wave index', mutate('bf8cc07f', 'be860308'),
     'slot 57 mesh count exchange: the compared register is not the wave index'),
)


def controls():
    os.environ['X2W_FXC'] = ''   # sources and manifest only; the draw mode compiles
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH)
        expect('the fixture translates', result, 'ok')
        names = sorted(p.name[16:] for p in Path(tmp).iterdir() if not p.name.startswith('req_'))
        expect('published files', names, ['.ms57.hlsl', '.ms57.txt', '.ps57.hlsl'])
        expect('a translated request is not pending', G.pending_keys(tmp, False), [])
        text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
        body = [line.strip() for line in text[text.index('void main('):].splitlines()]
        want = ['if (tid == 0u) { lds[0] = s13; lds[1] = s14; }', 'GroupMemoryBarrierWithGroupSync();',
                's13 = min(lds[0], 128u);', 's14 = min(lds[1], 128u);', 'if (s14 == 0u) s13 = 0u;',
                'SetMeshOutputCounts(s14, s13);']
        at = body.index(want[0]) if want[0] in body else 0
        expect('one store, one barrier, clamped counts, then the allocation', body[at:at + len(want)], want)
        expect('one barrier', text.count('GroupMemoryBarrier'), 1)
        expect('one allocation', text.count('SetMeshOutputCounts('), 1)
        expect('group memory is the two count words', re.findall(r'groupshared[^;]*;', text), ['groupshared uint lds[2];'])
        expect('outputs follow the allocation',
               all(body.index(line) > body.index(want[-1]) for line in body if 'verts[tid]' in line or 'tris[tid]' in line), True)
    for what, mesh, reason in REFUSALS:
        with tempfile.TemporaryDirectory() as tmp:
            key, result = request(tmp, mesh)
            expect(what, result, 'unsupported: ' + reason)
            expect(what + ': only the refusal is published',
                   sorted(p.name for p in Path(tmp).iterdir() if not p.name.startswith('req_')), ['%016x.fail' % key])
    with tempfile.TemporaryDirectory() as tmp:
        ms, blob = code(MESH), root()
        key = G.mesh57_key(ms, b'', blob)
        for ext, data in (('.root', blob), ('.ps57', b''), ('.m57', ms)):
            (Path(tmp) / ('req_%016x%s' % (key, ext))).write_bytes(data)
        expect('no pixel stage', G.translate_request(tmp, key), 'unsupported: slot 57 mesh without a pixel stage is not covered')
    # exp null off, off, off, off done vm; s_endpgm: a pixel stage without a colour output.
    with tempfile.TemporaryDirectory() as tmp:
        expect('a depth-only pixel stage', request(tmp, MESH, 'f8001890 00000000 bf810000'.split())[1],
               'unsupported: slot 57 mesh with a depth-only pixel stage is not covered')
    # The request path refuses a missing s_endpgm earlier; the walk takes the last instruction as the terminator
    # unread, so the pipeline refuses it too (s_nop in its place).
    try:
        G.MeshPipeline(code(MESH[:-1] + ['bf800000']), code(PIXEL), root())
        expect('a mesh program without s_endpgm', 'ok', 'refused')
    except G.Unsupported as refusal:
        expect('a mesh program without s_endpgm', str(refusal), 'slot 57 mesh program does not end with s_endpgm')


def assemble(mc, text):
    r = subprocess.run([mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64,-wavefrontsize32', '-show-encoding'],
                       input=text, capture_output=True, text=True, check=True)
    data = b''.join(bytes(int(b, 16) for b in m.split(',')) for m in re.findall(r'encoding: \[([^\]]*)\]', r.stdout))
    return ['%08x' % word for word in struct.unpack('<%dI' % (len(data) // 4), data)]


def binary(mc):
    expect('mesh words match llvm-mc', assemble(mc, (HERE / 'startup_mesh57_skeleton.s').read_text()), MESH)
    expect('pixel words match llvm-mc', assemble(mc, PIXEL_ASM), PIXEL)


# ---- draw: four triangles in a 32x32 target, vertices stored out of order ----------------------
SIDE = 32
TRIANGLES = (((2, 2), (14, 2), (2, 14)), ((18, 3), (30, 5), (21, 14)), ((3, 18), (13, 20), (6, 30)), ((18, 18), (30, 18), (30, 30)))
ORDER = (7, 2, 10, 5, 0, 8, 3, 11, 1, 6, 9, 4)   # ORDER[3 * triangle + corner] = vertex slot


def geometry():
    points, colours, indices = [None] * 12, [None] * 12, []
    for t, triangle in enumerate(TRIANGLES):
        for c, point in enumerate(triangle):
            slot = ORDER[3 * t + c]
            points[slot] = point
            colours[slot] = (40 + 17 * slot, 250 - 19 * slot, (slot * 53) % 256, 255)
        indices.append(tuple(ORDER[3 * t:3 * t + 3]))
    return points, colours, indices


def buffer(points, colours, words):
    data = bytearray(512)
    for slot, (x, y) in enumerate(points):
        struct.pack_into('<4f', data, 16 * slot, x / 16 - 1, 1 - y / 16, 0.5, 1)
        struct.pack_into('<4f', data, 192 + 16 * slot, *(channel / 255 for channel in colours[slot]))
    for t, word in enumerate(words):
        struct.pack_into('<I', data, 384 + 4 * t, word)
    return bytes(data)


def rasterise(points, colours, indices, verts, prims):
    """D3D rules: sample at the pixel centre, top-left fill, flat colour of the first vertex, later primitives on top.

    Counts above the 128 threads of a group are clamped and an index at or past the vertex count names the last
    vertex, as the lowering documents. Vertex `thread` holds slot `thread & 15`, as the fixture reads it.
    Coordinates are doubled so centres are integers."""
    verts, prims = min(verts, 128), min(prims, 128)
    image = [(0, 0, 0, 255)] * (SIDE * SIDE)
    if not verts:
        return image
    for a, b, c in indices[:prims]:
        a, b, c = (min(i, verts - 1) for i in (a, b, c))
        colour = colours[a % 16]
        tri = [(2 * points[i % 16][0], 2 * points[i % 16][1]) for i in (a, b, c)]
        area = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0])
        if not area:
            continue
        if area < 0:
            tri[1], tri[2] = tri[2], tri[1]
        for y in range(SIDE):
            for x in range(SIDE):
                inside = True
                for k in range(3):
                    (ax, ay), (bx, by) = tri[k], tri[(k + 1) % 3]
                    dx, dy = bx - ax, by - ay
                    edge = dx * (2 * y + 1 - ay) - dy * (2 * x + 1 - ax)
                    inside &= edge > 0 or (edge == 0 and (dy < 0 or (dy == 0 and dx > 0)))
                if inside:
                    image[y * SIDE + x] = colour
    return image


def draw(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH)
        expect('the fixture translates and compiles', result, 'ok')
        if result != 'ok':
            return 0
        name = str(Path(tmp) / ('%016x' % key))
        uds = G.MeshPipeline(code(MESH), code(PIXEL), root()).uds
        # (vertex count, primitive count, thread of vertex slot 0, flags of the first primitive words): all, fewer
        # primitives, fewer vertices (indices clamp), no vertex; every thread, with the triangles on vertices
        # 112..123 (wave 1, lanes 48..59), each named by eight primitives and the other 96 primitives on vertex 0
        # alone, the first triangle null (bit 31) and the second with its three edge flags (bits 9, 19, 29).
        # Each case is one process and one device, which takes seconds on a hardware adapter.
        cases = ((12, 4, 0, ()), (12, 1, 0, ()), (7, 4, 0, ()), (0, 4, 0, ()),
                 (128, 128, 112, (1 << 31, 1 << 9 | 1 << 19 | 1 << 29)))
        for case in cases:
            verts, prims, first, flags = case
            named = [tuple(first + i for i in triangle) for triangle in indices]
            words = [a | b << 10 | c << 20 | flag for (a, b, c), flag in zip(named, flags + (0,) * 4)]
            drawn = [(0, 0, 0) if word >> 31 else triangle for triangle, word in zip(named, words)]
            constants = bytearray(1024)
            struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 0)), prims)
            struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 1)), verts)
            image = rasterise(points, colours, drawn, verts, prims)
            head = bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
            path = Path(tmp) / 'draw.bin'
            path.write_bytes(head + b''.join(bytes(pixel) for pixel in image))
            args = [renderer, adapter, '--mesh-indexed', name + '.ms57.cso', name + '.ps57.cso', str(path)]
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode == 77:
                print(r.stdout + r.stderr)
                return 77
            covered = sum(pixel != (0, 0, 0, 255) for pixel in image)
            what = 'vertices=%d primitives=%d first=%d flags=%s' % (verts, prims, first, ['%08x' % flag for flag in flags])
            print('%s covered=%d: %s' % (what, covered, r.stdout.strip().splitlines()[-1:]))
            expect('draw of ' + what,
                   (r.returncode, 'pixels=1024 mismatches=0' in r.stdout), (0, True))
            if case is cases[0]:
                expect('the full case covers pixels', covered > 200, True)
                wrong = list(image)
                at = next(i for i, pixel in enumerate(image) if pixel != (0, 0, 0, 255))
                wrong[at] = (0, 0, 0, 255)
                path.write_bytes(head + b''.join(bytes(pixel) for pixel in wrong))
                r = subprocess.run(args, capture_output=True, text=True)
                expect('a reference with one wrong pixel fails',
                       (r.returncode, 'pixels=1024 mismatches=1' in r.stdout, 'Draw reference mismatch' in r.stderr),
                       (1, True, True))
    return 0


def main(argv):
    status = 0
    if argv[1:2] == ['binary'] and len(argv) == 3:
        binary(argv[2])
    elif argv[1:2] == ['draw'] and len(argv) == 5:
        status = draw(*argv[2:])
    elif len(argv) == 1:
        controls()
    else:
        print(__doc__)
        return 2
    if FAILED:
        return 1
    print('skip: no mesh shader tier' if status == 77 else 'ok')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
