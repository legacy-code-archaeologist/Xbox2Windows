"""Source-built gfx1013 min/max translation and pixel checks.

No arguments checks the emitted HLSL. Binary mode verifies the encodings with
llvm-mc. The before, draw, plain and invert modes take DXC, offscreen_check,
the adapter name and optionally FXC.
"""

import math
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_gfx2hlsl_bit_alu_check import G, POST_SOURCE, POST_WORDS, PRE_SOURCE, PRE_WORDS, vs_source, vs_words
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob

import pixel_aot

NEG_ZERO, POS_ZERO = 0x80000000, 0
NEG_ONE, TWO, NAN = 0xbf800000, 0x40000000, 0x7fc01234
CASE_TEXELS = 8 * 8
DRAW_TEXELS = 64 * 64
OPS = {
    "min": ("v_min_f32_e32", 2), "max": ("v_max_f32_e32", 2),
    "min-e64": ("v_min_f32_e64", 2), "max-e64": ("v_max_f32_e64", 2),
    "min3": ("v_min3_f32", 3), "max3": ("v_max3_f32", 3), "med3": ("v_med3_f32", 3),
    "min-sdwa": ("v_min_f32_sdwa", 2), "max-sdwa": ("v_max_f32_sdwa", 2),
    "min-dpp": ("v_min_f32_dpp", 2), "max-dpp": ("v_max_f32_dpp", 2),
    "min-pack": ("v_pk_min_f16", 2), "max-pack": ("v_pk_max_f16", 2),
}
STARTUP_OPS = ("min", "max", "min-e64", "max-e64", "min3", "max3", "med3")
BODY_WORDS = {
    "min": "0009181e010b1a1e020d1c1e030f1e1e",
    "max": "00091820010b1a20020d1c20030f1e20",
    "min-e64": "0c000fd5000902000d000fd5010b02000e000fd5020d02000f000fd5030f0200",
    "max-e64": "0c0010d5000902000d0010d5010b02000e0010d5020d02000f0010d5030f0200",
    "min3": "0c0051d5000922040d0051d5010b26040e0051d5020d2a040f0051d5030f2e04",
    "max3": "0c0054d5000922040d0054d5010b26040e0054d5020d2a040f0054d5030f2e04",
    "med3": "0c0057d5000922040d0057d5010b26040e0057d5020d2a040f0057d5030f2e04",
    "min-sdwa": "f908181e00060606f90a1a1e01060606f90c1c1e02060606f90e1e1e03060606",
    "max-sdwa": "f908182000060606f90a1a2001060606f90c1c2002060606f90e1e2003060606",
    "min-dpp": "fa08181e00e400fffa0a1a1e01e400fffa0c1c1e02e400fffa0e1e1e03e400ff",
    "max-dpp": "fa08182000e400fffa0a1a2001e400fffa0c1c2002e400fffa0e1e2003e400ff",
    "min-pack": "0c4011cc000902180d4011cc010b02180e4011cc020d02180f4011cc030f0218",
    "max-pack": "0c4012cc000902180d4012cc010b02180e4012cc020d02180f4012cc030f0218",
}

PIXEL_PREFIX = """s_mov_b32 s21, 0
s_load_dwordx4 s[44:47], s[20:21], 0x40
s_waitcnt lgkmcnt(0)
s_buffer_load_dwordx4 s[48:51], s[44:47], 0xc
s_buffer_load_dwordx4 s[52:55], s[44:47], 0x1c
s_buffer_load_dwordx4 s[56:59], s[44:47], 0x2c
s_waitcnt lgkmcnt(0)
""" + "".join(f"v_mov_b32_e32 v{i}, s{48 + i}\n" for i in range(12))
PIXEL_PREFIX_WORDS = (
    "800395be0a0b08f4400000fa7fc08cbf160c28f40c0000fa160d28f41c0000fa"
    "160e28f42c0000fa7fc08cbf3002007e3102027e3202047e3302067e3402087e"
    "35020a7e36020c7e37020e7e3802107e3902127e3a02147e3b02167e"
)


def body_source(op):
    mnemonic, arity = OPS[op]
    mods = ""
    if op.endswith("-sdwa"):
        mods = " dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"
    if op.endswith("-dpp"):
        mods = " quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf"
    return "".join(f"{mnemonic} v{12 + lane}, " +
                   ", ".join(f"v{lane + 4 * arg}" for arg in range(arity)) + mods + "\n"
                   for lane in range(4))


def draw_cases(op):
    nz, pz = NEG_ZERO, POS_ZERO
    if op.endswith("-pack"):
        # Each word has distinct low and high halves. The first two lanes swap the packed operands.
        return (((0x00008000, 0x80000000, 0), (0x80000000, 0x00008000, 0),
                 (0x80000000, 0x80000000, 0), (0x00008000, 0x00008000, 0)),
                ((0x7e003c00, 0x4000fe00, 0), (0x4000fe00, 0x7e003c00, 0),
                 (0x3c00bc00, 0x40004000, 0), (0x40004000, 0x3c00bc00, 0)))
    if OPS[op][1] == 2:
        return (((nz, pz, 0), (pz, nz, 0), (nz, nz, 0), (pz, pz, 0)),
                ((NAN, NEG_ONE, 0), (NEG_ONE, NAN, 0), (NAN, TWO, 0), (NEG_ONE, TWO, 0)))
    return (((nz, pz, NEG_ONE), (pz, nz, NEG_ONE), (nz, pz, TWO), (pz, nz, TWO)),
            ((nz, nz, pz), (pz, pz, nz), (nz, pz, pz), (pz, nz, nz)),
            ((NAN, NEG_ONE, TWO), (NEG_ONE, NAN, TWO), (NEG_ONE, TWO, NAN), (NEG_ONE, pz, TWO)))


def value(bits, size):
    return struct.unpack("<e" if size == 16 else "<f", bits.to_bytes(size // 8, "little"))[0]


def ordered(word, size):
    # Sort zeros by their sign after comparing numerical values.
    return value(word, size), word != 1 << (size - 1)


def reference(op, words, kind):
    if op.endswith("-pack"):
        def lane(shift):
            selected = [(word >> shift) & 0xffff for word in words[:2]]
            return reference_word(op, selected, 16, kind)
        return lane(0) | (lane(16) << 16)
    return reference_word(op, words[:OPS[op][1]], 32, kind)


def reference_word(op, words, size, kind):
    numeric = [word for word in words if not math.isnan(value(word, size))]
    ordered_words = sorted(numeric, key=lambda word: ordered(word, size))
    if op.startswith("min"):
        return ordered_words[0]
    if op.startswith("max"):
        return ordered_words[-1]
    # Keep the startup median's original nested min/max behavior for one NaN.
    # Its result is the larger number if the last operand is NaN and the
    # smaller number if either of the first two operands is NaN.
    if len(numeric) < len(words):
        if kind == "startup" and numeric == list(words[:2]):
            return ordered_words[-1]
        return ordered_words[0]
    return ordered_words[1]


def pixel_hlsl(op):
    code = bytes.fromhex(PIXEL_PREFIX_WORDS + BODY_WORDS[op]) + POST_WORDS
    asm = [line for _, _, _, line in G.decode(code)]
    return pixel_aot.lower(asm, 20, 24, None, constant_descriptor=64, integer_export=True)


def startup_pipeline(op, rows):
    ps = PRE_WORDS + bytes.fromhex(BODY_WORDS[op]) + POST_WORDS
    return G.Pipeline(vs_words(rows), ps, root_blob())


def check_binary(llvm_mc):
    assert assemble(llvm_mc, PIXEL_PREFIX).hex() == PIXEL_PREFIX_WORDS, "pixel constant-load prologue"
    for op in OPS:
        assert assemble(llvm_mc, body_source(op)).hex() == BODY_WORDS[op], op
        if op in STARTUP_OPS:
            for rows in draw_cases(op):
                assert assemble(llvm_mc, vs_source(rows)) == vs_words(rows), op
            assert assemble(llvm_mc, PRE_SOURCE + body_source(op) + POST_SOURCE) == (
                PRE_WORDS + bytes.fromhex(BODY_WORDS[op]) + POST_WORDS), op
    print("All signed-zero fixtures match llvm-mc")


def check_lowering(kind, op, source):
    assert "uint a_bits = asuint(a), b_bits = asuint(b);" in source
    assert "uint either = a_bits | b_bits;" in source
    assert "uint zero = maximum ? (a_bits & b_bits) : either;" in source
    assert "return asfloat((either & 0x7fffffffu) == 0u ? zero :" in source
    body = source[source.index(" main("):]
    polarity = "true" if op.startswith("max") else "false"
    if op.endswith("-pack"):
        assert "isnan(a) ? b : (isnan(b) ? a : minmax_zero32(a, b, maximum))" in source
        assert body.count("minmax16_num(") == body.count(", " + polarity + ")") == 8, (op, body)
    elif op == "med3":
        name = "med3_32" if kind == "pixel" else "med3"
        assert body.count(name + "(") == 4, (kind, body)
        start = source.index("float " + name + "(")
        helper = source[start:source.index("\n}", start)]
        assert helper.count("minmax_zero32(") == (6 if kind == "pixel" else 4), helper
    else:
        count = 8 if OPS[op][1] == 3 else 4
        assert body.count("minmax_zero32(") == body.count(", " + polarity + ")") == count, (op, body)


def mutate(source, mode):
    start = source.index("float minmax_zero32(")
    end = source.index("\n}", start)
    helper = source[start:end]
    if mode == "plain":
        replacement = helper.splitlines()[0] + "\n    return maximum ? max(a, b) : min(a, b);"
    else:
        replacement = helper.replace("maximum ? (a_bits & b_bits) : either",
                                     "maximum ? either : (a_bits & b_bits)")
        assert replacement != helper
    return source[:start] + replacement + source[end:]


def check_translation():
    for kind, ops in (("pixel", OPS), ("startup", STARTUP_OPS)):
        for op in ops:
            source = pixel_hlsl(op) if kind == "pixel" else startup_pipeline(op, draw_cases(op)[0]).hlsl_ps()
            check_lowering(kind, op, source)
            for mode in ("plain", "invert"):
                try:
                    check_lowering(kind, op, mutate(source, mode))
                except AssertionError:
                    continue
                raise AssertionError(f"{kind} {op} accepted the {mode} control")
    print("20 signed-zero lowering checks reject both helper mutations")


def pixel_readback(renderer, backend, directory, shader, rows, expect):
    # Existing integer-target case format, with three runtime uint4 operands from b0.
    constants = [0] * 64
    for arg in range(3):
        constants[3 + 4 * arg:7 + 4 * arg] = [row[arg] for row in rows]
    case, raw = (directory / name for name in ("minmax.case", "minmax.bin"))
    case.write_bytes(struct.pack("<64I", *constants) + struct.pack("<4I", *expect) * CASE_TEXELS + b"\x01")
    raw.unlink(missing_ok=True)
    run = subprocess.run([renderer, backend, "--check-pixel-case", str(shader), str(case), str(raw)],
                         capture_output=True, text=True)
    return run, tuple(struct.iter_unpack("<4I", raw.read_bytes())) if raw.exists() else ()


def startup_readback(renderer, backend, vs, ps, expect):
    run = draw(renderer, backend, vs, ps, ",".join(f"{word:08x}" for word in expect))
    match = re.search(r"startup_flat_param words=([0-9a-f,]+)", run.stdout)
    return run, (tuple(int(word, 16) for word in match[1].split(",")),) if match else ()


def check_draw(mode, dxc, renderer, backend, fxc=None):
    renderer = str(Path(renderer).resolve())
    sites = [("pixel", dxc, "dxc", "ps_6_2"), ("startup", dxc, "dxc", "ps_6_0")]
    if fxc:
        sites.append(("startup", fxc, "fxc", "ps_5_1"))
    out = Path(__file__).resolve().parents[1] / "out"
    out.mkdir(exist_ok=True)
    mismatches = total = 0
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        for kind, compiler, tool, profile in sites:
            for op in OPS if kind == "pixel" else STARTUP_OPS:
                cases = draw_cases(op)
                source = pixel_hlsl(op) if kind == "pixel" else startup_pipeline(op, cases[0]).hlsl_ps()
                if mode in ("plain", "invert"):
                    source = mutate(source, mode)
                if kind == "pixel":
                    hlsl, shader = (directory / name for name in ("pixel.hlsl", "pixel.cso"))
                    hlsl.write_text(source, encoding="ascii")
                    subprocess.run([dxc, "-Gis", "-T", profile, "-E", "main", "-Fo", str(shader), str(hlsl)],
                                   check=True)
                else:
                    shader = compile_hlsl(compiler, tool, source, "pixel", profile, directory)
                for index, rows in enumerate(cases):
                    expected = tuple(reference(op, row, kind) for row in rows)
                    if kind == "pixel":
                        run, raw = pixel_readback(renderer, backend, directory, shader, rows, expected)
                    else:
                        pipeline = startup_pipeline(op, rows)
                        vs = compile_hlsl(compiler, tool, pipeline.hlsl_vs(), "vertex", profile.replace("ps", "vs"), directory)
                        run, raw = startup_readback(renderer, backend, vs, shader, expected)
                    if run.returncode == 77:
                        print(run.stdout + run.stderr)
                        return 77
                    pixel_count = DRAW_TEXELS if kind == "pixel" else 1
                    if len(raw) != pixel_count:
                        raise AssertionError(f"{kind} {op}: expected {pixel_count} readback pixels, got {len(raw)}\n"
                                             f"{run.stdout}\n{run.stderr}")
                    if (any(row != raw[0] for row in raw) or run.returncode not in (0, 1)
                            or ("debug_layer=enabled" in run.stdout and "debug_errors=0 debug_warnings=0" not in run.stdout)):
                        raise AssertionError(f"{kind} {op}: {run.stdout}\n{run.stderr}")
                    got = ",".join(f"{word:08x}" for word in raw[0])
                    want = ",".join(f"{word:08x}" for word in expected)
                    print(f"{backend} {kind} {profile} {op} row {index}: {got} (expected {want})")
                    if bool(run.returncode) != (raw[0] != expected):
                        raise AssertionError(f"{kind} {op}: {run.stdout}\n{run.stderr}")
                    mismatches += sum(got != want for got, want in zip(raw[0], expected))
                    total += len(expected)
    print(f"{backend} {mode}: {mismatches}/{total} result words differ")
    if mode == "invert" and not mismatches:
        raise AssertionError("The inverted sign rule escaped detection")
    return int(mode != "before" and mismatches != 0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "binary":
        check_binary(sys.argv[2])
    elif len(sys.argv) in (5, 6) and sys.argv[1] in ("before", "draw", "plain", "invert"):
        sys.exit(check_draw(*sys.argv[1:]))
    elif len(sys.argv) == 1:
        check_translation()
    else:
        sys.exit(__doc__)
