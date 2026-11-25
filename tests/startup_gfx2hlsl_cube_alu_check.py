"""Cube-face operations of tools/startup/gfx2hlsl.py drawn on D3D12 (#1723).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

Covers v_cubeid_f32, v_cubesc_f32, v_cubetc_f32 and v_cubema_f32. The fixture and runner are
startup_gfx2hlsl_lane_alu.py: one operation per lane on flat parameters the compilers cannot fold, results compared
as exact words. The references follow the AMD RDNA 2 ISA reference: the major axis is z when |z| >= |x| and
|z| >= |y|, else y when |y| >= |x|, else x; the face id runs +x, -x, +y, -y, +z, -z; sc and tc are the in-face
coordinates; ma is twice the major-axis component. startup_gfx2hlsl_cube_check.py keeps the pure text controls.

Not covered: NaN and infinite inputs and zero major axes (the hardware picks a face by comparison order there).
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import main, words  # noqa: E402


def f32(value):
    """The binary32 word of a Python number."""
    return struct.unpack('<I', struct.pack('<f', value))[0]


def value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def axis(x, y, z):
    if abs(z) >= max(abs(x), abs(y)):
        return 2, z
    if abs(y) >= abs(x):
        return 1, y
    return 0, x


def cubeid(a, b, c):
    x, y, z = value(a), value(b), value(c)
    major, m = axis(x, y, z)
    return f32(float(major * 2 + (1 if m < 0 else 0)))


def cubesc(a, b, c):
    x, y, z = value(a), value(b), value(c)
    major, _ = axis(x, y, z)
    if major == 2:
        return f32(-x if z < 0 else x)
    if major == 1:
        return f32(x)
    return f32(z if x < 0 else -z)


def cubetc(a, b, c):
    x, y, z = value(a), value(b), value(c)
    major, _ = axis(x, y, z)
    return f32((-z if y < 0 else z) if major == 1 else -y)


def cubema(a, b, c):
    return f32(2.0 * axis(value(a), value(b), value(c))[1])


def swapped(function):
    """The reference of a lowering that reads x where y belongs and y where x belongs."""
    return lambda a, b, c: function(b, a, c)


OPS = {'v_cubeid_f32': (3, cubeid, False), 'v_cubesc_f32': (3, cubesc, False),
       'v_cubetc_f32': (3, cubetc, False), 'v_cubema_f32': (3, cubema, False)}
# Directions for the six faces, a tie between z and x (z wins), and a tie between y and x (y wins). Every component
# is exactly representable, none is zero, so the sign of every result is defined.
DIRECTIONS = [(1.5, -0.5, 0.25), (-2.0, 0.5, 1.0), (0.5, 3.0, -1.0), (0.25, -4.0, 2.0),
              (0.5, -0.25, 2.5), (-1.0, 0.75, -3.0), (-2.0, 2.0, 1.0), (-2.0, 1.0, -2.0)]
SETS = [tuple(f32(c) for c in d) for d in DIRECTIONS]
CALL = r'(cube\w+)\(F\((v\d+)\), F\((v\d+)\), F\((v\d+)\)\)'
MUTATIONS = {op: (CALL, r'\1(F(\3), F(\2), F(\4))', swapped(reference)) for op, (_, reference, _) in OPS.items()}
# Swapping x and y leaves ma unchanged on the first set, so its wrong lowering drops the sign instead.
MUTATIONS['v_cubema_f32'] = (CALL, r'-\1(F(\2), F(\3), F(\4))', lambda a, b, c: cubema(a, b, c) ^ 0x80000000)
BODY = {
    'v_cubeid_f32': words('0c0044d5000922040d0044d5010b26040e0044d5020d2a040f0044d5030f2e04'),
    'v_cubesc_f32': words('0c0045d5000922040d0045d5010b26040e0045d5020d2a040f0045d5030f2e04'),
    'v_cubetc_f32': words('0c0046d5000922040d0046d5010b26040e0046d5020d2a040f0046d5030f2e04'),
    'v_cubema_f32': words('0c0047d5000922040d0047d5010b26040e0047d5020d2a040f0047d5030f2e04'),
}
FAMILY = {
    'ops': OPS, 'draw': {op: SETS for op in OPS}, 'mutations': MUTATIONS, 'body': BODY,
    'accepted': ['v_cubeid_f32 v1, v2, v3, v4', 'v_cubesc_f32 v1, -v2, |v3|, v4',
                 'v_cubetc_f32_e64 v1, v2, v3, v4', 'v_cubema_f32 v1, v2, v3, v4'],
    'refused': ['v_cubeid_f32 v1, v2, v3', 'v_cubema_f32 v1, v2'],
}

if __name__ == '__main__':
    main(FAMILY, __doc__)
