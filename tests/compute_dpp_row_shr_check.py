"""DPP16 row_shr in tools/compute_store_aot.py for gfx1013 compute (#1543).

  (no arguments)                        translation and refusal controls, pure Python
  binary <llvm-mc> <llvm-objcopy> <libLLVM>
                                        assemble and decode the fixture with LLVM 15, translate both
  draw <dxc> <offscreen_check> <warp|hardware>
                                        translate, dispatch and compare the readback

`v_or_b32_dpp` and `v_add_nc_u32_dpp` with `row_shr:n` (1..15) and full row and bank masks read
src0 from lane - n of the same 16-lane row. A lane whose source is before its row start uses 0 as
src0 when BOUND_CTRL is set and keeps its old destination when it is not. The translation indexes
the guest's logical 1D lanes (SV_GroupThreadID.x) through group-shared memory, as the quad_perm
slice does, so the result does not depend on the desktop wave width.

The fixture is an original program, source-built for gfx1013. One group of 64 threads (four rows)
loads a word that is unique for each lane and each row, and stores four words per thread: OR and
ADD, each with and without BOUND_CTRL. It runs with shifts 1, 8 and 15. A per-lane CPU reference
gives all 320 words, including the 64 sentinel words after the records. Two mutants of the
generated HLSL must fail at a predicted first word: one reads across the row boundary, one writes
0 where the old destination must be kept.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import compute_store_aot as aot  # noqa: E402
from pixel_aot import disassemble  # noqa: E402

SENTINEL = 0xc5c5c5c5
WORDS = 320
THREADS = 64
SHIFTS = (1, 8, 15)
# Each thread keeps these words before the DPP without BOUND_CTRL: its dispatch index XOR a key.
OLD_KEYS = {"or": 0x6b000000, "add": 0x6c000000}
# ADD reads src0 = source word XOR this key, so a step that read another step's staged words would differ.
ADD_SOURCE_KEY = 0x00a50000
# The four stored words of each thread, in store order: (operation, BOUND_CTRL).
STEPS = (("or", True), ("or", False), ("add", True), ("add", False))
BINDING = dict(table_reg=0, descriptor_offset=0, constant_descriptor=32, group_reg=1, stride=16,
               source_descriptor=16)


def program(shift, spelling="bound_ctrl:1"):
    """v1 = source word tid, v7 = v1 ^ key, v0 = tid; OR shifts v1 and ADD shifts v7.

    v5/v6 are written with BOUND_CTRL; v2/v4 keep tid ^ key without it.
    """
    masks = f"row_shr:{shift} row_mask:0xf bank_mask:0xf"
    return [
        "s_mov_b32 s2, s1",
        "s_mov_b32 s1, 0",
        "v_lshl_add_u32 v0, s2, 6, v0",
        "s_load_dwordx4 s[4:7], s[0:1], 0x20",
        "s_waitcnt lgkmcnt(0)",
        "s_load_dwordx8 s[8:15], s[0:1], 0x0",
        "s_waitcnt lgkmcnt(0)",
        "v_lshlrev_b32_e32 v3, 2, v0",
        "buffer_load_dword v1, v3, s[12:15], 0 offen",
        "s_waitcnt vmcnt(0)",
        f"v_xor_b32_e32 v2, 0x{OLD_KEYS['or']:x}, v0",
        f"v_xor_b32_e32 v4, 0x{OLD_KEYS['add']:x}, v0",
        f"v_xor_b32_e32 v7, 0x{ADD_SOURCE_KEY:x}, v1",
        f"v_or_b32_dpp v5, v1, v0 {masks} {spelling}",
        f"v_or_b32_dpp v2, v1, v0 {masks}",
        f"v_add_nc_u32_dpp v6, v7, v0 {masks} {spelling}",
        f"v_add_nc_u32_dpp v4, v7, v0 {masks}",
        "buffer_store_dword v5, v0, s[8:11], 0 idxen",
        "buffer_store_dword v2, v0, s[8:11], 0 idxen offset:4",
        "buffer_store_dword v6, v0, s[8:11], 0 idxen offset:8",
        "buffer_store_dword v4, v0, s[8:11], 0 idxen offset:12",
        "s_endpgm",
    ]


def translate(lines, group=THREADS, **change):
    return aot.lower(lines, group, **(BINDING | change))


def source_word(i):
    """t0 as offscreen_check fills it: 320 known words."""
    return (0x8b02a7f1 + i * 0x413b1) & 0xffffffff


def expected(shift, mutant=None):
    """The readback from a per-lane model of row_shr, independent of the translator.

    mutant "cross_row" bounds the source by the group start instead of the row start;
    mutant "zero" writes 0 where a lane without BOUND_CTRL keeps its old destination.
    """
    words = [SENTINEL] * WORDS
    for tid in range(THREADS):
        inside = (tid if mutant == "cross_row" else tid % 16) >= shift
        for k, (op, bound) in enumerate(STEPS):
            key = ADD_SOURCE_KEY if op == "add" else 0
            src0 = source_word(tid - shift) ^ key if inside else 0
            result = (src0 | tid) if op == "or" else (src0 + tid) & 0xffffffff
            if not (inside or bound):
                result = 0 if mutant == "zero" else tid ^ OLD_KEYS[op]
            words[4 * tid + k] = result
    return words


def mutate(source, mutant):
    """Apply one mutant to every DPP step of the generated HLSL."""
    if mutant == "cross_row":
        pattern, replacement = r"bool (dpp\d+)_inside = \1_row_lane >= ", r"bool \1_inside = local.x >= "
    else:
        pattern, replacement = r"(uint dpp\d+_value = dpp\d+_inside \? .+) : \(.+\);", r"\1 : 0u;"
    mutated, count = re.subn(pattern, replacement, source)
    assert count == (4 if mutant == "cross_row" else 2), (mutant, count)
    return mutated


def first_difference(left, right):
    return next(i for i, (a, b) in enumerate(zip(left, right)) if a != b)


def rejects(message, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert message in str(error), (message, error)
        return
    raise AssertionError(f"accepted; expected {message!r}")


def variant(lines, old, new):
    assert sum(old in line for line in lines) == 1, old
    return [line.replace(old, new) for line in lines]


def controls():
    # The fixture words are unique for every lane of every row.
    words = [source_word(i) for i in range(THREADS)]
    assert len(set(words)) == THREADS and all(len(set(words[r:r + 16])) == 16 for r in range(0, THREADS, 16))

    lines = translate(program(8)).splitlines()
    assert "groupshared uint dpp_stage[64];" in lines, lines
    step = lines.index("    uint dpp0_row_lane = local.x & 15u;")
    assert lines[step:step + 8] == [
        "    uint dpp0_row_lane = local.x & 15u;",
        "    bool dpp0_inside = dpp0_row_lane >= 8u;",
        "    dpp_stage[local.x] = inputData.Load((tid.x << 2u) + 0u);",
        "    GroupMemoryBarrierWithGroupSync();",
        "    uint dpp0_source_thread = dpp0_inside ? local.x - 8u : local.x;",
        "    uint dpp0_source = dpp0_inside ? dpp_stage[dpp0_source_thread] : 0u;",
        "    GroupMemoryBarrierWithGroupSync();",
        "    uint dpp0_value = (dpp0_source | (tid.x));",
    ], lines
    text = "\n".join(lines)
    # Without BOUND_CTRL an out-of-row lane keeps the old destination, tid ^ key.
    assert "    uint dpp1_value = dpp1_inside ? (dpp1_source | (tid.x)) : ((0x6b000000u ^ tid.x));" in lines, text
    assert "    uint dpp2_value = (dpp2_source + (tid.x));" in lines, text
    # ADD stages its own source, different from OR's.
    assert f"    dpp_stage[local.x] = (0x{ADD_SOURCE_KEY:08x}u ^ inputData.Load((tid.x << 2u) + 0u));" in lines, text
    assert "    uint dpp3_value = dpp3_inside ? (dpp3_source + (tid.x)) : ((0x6c000000u ^ tid.x));" in lines, text
    assert text.count("GroupMemoryBarrierWithGroupSync();") == 8, text
    assert "Wave" not in text, "the logical-lane model must not use wave intrinsics"
    # Every shift from 1 to 15 is accepted, and both LLVM 15 spellings of the set bit lower identically.
    for shift in range(1, 16):
        assert f"bool dpp0_inside = dpp0_row_lane >= {shift}u;" in translate(program(shift))
        assert translate(program(shift, "bound_ctrl:0")) == translate(program(shift))
    # A group of 16 (one row) and of 1024 are accepted.
    assert "groupshared uint dpp_stage[16];" in translate(variant(program(1), "s2, 6, v0", "s2, 4, v0"), 16)
    assert "groupshared uint dpp_stage[1024];" in translate(variant(program(1), "s2, 6, v0", "s2, 10, v0"), 1024)
    print("row_shr: shifts 1-15, OR and ADD, BOUND_CTRL set and clear, groups of 16 to 1024")

    # With BOUND_CTRL the old destination is never read, so an unwritten one is accepted.
    no_old = variant(program(1), f"v_xor_b32_e32 v2, 0x{OLD_KEYS['or']:x}, v0", "s_nop 0")
    rejects("Unverified DPP16 operand", translate, no_old)
    bound_only = variant(no_old, "v_or_b32_dpp v2, v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf",
                         "v_or_b32_dpp v2, v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1")
    translate(bound_only)

    dpp = "v_or_b32_dpp v5, v1, v0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1"
    unsupported = "Unsupported compute instruction: v_"
    for form in (dpp.replace("row_shr:8", "row_shl:8"), dpp.replace("row_shr:8", "row_ror:8"),
                 dpp.replace("row_shr:8", "row_share:8"), dpp.replace("row_shr:8", "row_xmask:8"),
                 dpp.replace("row_shr:8", "row_mirror"), dpp.replace("row_shr:8", "row_half_mirror"),
                 dpp.replace("row_shr:8", "row_bcast:15"), dpp.replace("row_shr:8", "wave_shr:1"),
                 dpp + " fi:1", dpp.replace("row_mask:0xf", "row_mask:0x7"),
                 dpp.replace("bank_mask:0xf", "bank_mask:0xe"), dpp.replace("v1, v0 row_shr", "s1, v0 row_shr"),
                 dpp.replace("v_or_b32_dpp", "v_xor_b32_dpp"), dpp.replace("v_or_b32_dpp", "v_add_f32_dpp"),
                 dpp.replace("v_or_b32_dpp", "v_sub_nc_u32_dpp"), dpp.replace("v_or_b32_dpp", "v_mov_b32_dpp")):
        rejects(unsupported, translate, variant(program(8), dpp, form))
    shift = "Unsupported or masked DPP16 row shift"
    for bad in ("row_shr:0", "row_shr:16"):
        rejects(shift, translate, variant(program(8), dpp, dpp.replace("row_shr:8", bad)))
    rejects(shift, translate, variant(program(8), dpp, dpp.replace("v5, v1", "v256, v1")))
    rejects(shift, translate, program(8), cpu="gfx1030")
    # Groups of 8 (part of a row) and two-dimensional groups are refused.
    rejects(shift, translate, variant(program(8), "s2, 6, v0", "s2, 3, v0"), 8)
    rejects(shift, translate, variant(program(8), "s2, 6, v0", "s2, 5, v0"), (32, 2, 1))
    # A lane guard before the DPP is refused (READ_COMPUTE's guard with the DPP after it).
    guarded = [
        "s_mov_b32 s2, s1", "s_mov_b32 s1, 0", "v_lshl_add_u32 v0, s2, 6, v0",
        "s_load_dwordx4 s[4:7], s[0:1], 0x20", "s_waitcnt lgkmcnt(0)", "s_buffer_load_dword s3, s[4:7], 0x0",
        "s_waitcnt lgkmcnt(0)", "v_cmp_gt_u32_e32 vcc_lo, s3, v0", "s_and_saveexec_b32 s2, vcc_lo",
        "s_cbranch_execz .LBB0_2", "s_load_dwordx8 s[8:15], s[0:1], 0x0", "v_lshlrev_b32_e32 v3, 2, v0",
        "s_waitcnt lgkmcnt(0)", "buffer_load_dword v1, v3, s[12:15], 0 offen", "s_waitcnt vmcnt(0)", dpp,
        "buffer_store_dword v5, v0, s[8:11], 0 idxen", ".LBB0_2:", "s_endpgm"]
    rejects(shift, translate, guarded, stride=4)
    print("controls: row_shl, row_ror, row_share, row_xmask, other DPP16 controls, fi:1, partial masks,")
    print("  other opcodes, an SGPR source, shifts 0 and 16, gfx1030, groups of 8, 2D groups, a lane guard")
    print("  and a missing old destination without BOUND_CTRL are refused")

    # The reference and its mutants differ at the predicted first word, for each shift.
    for shift in SHIFTS:
        assert first_difference(expected(shift), expected(shift, "cross_row")) == 64, shift
        assert first_difference(expected(shift), expected(shift, "zero")) == 1, shift
        source = translate(program(shift))
        for mutant in ("cross_row", "zero"):
            assert mutate(source, mutant) != source
    print("mutants: cross_row first differs at word 64 (thread 16), zero at word 1 (thread 0)")


def check_binary(llvm_mc, objcopy, library, directory):
    """Assemble with LLVM 15, decode, and require the same translation as the source text."""
    for shift in SHIFTS:
        encoded = {}
        for spelling in ("bound_ctrl:0", "bound_ctrl:1"):
            src, obj, raw = (directory / f"row_shr_{shift}.{ext}" for ext in ("s", "o", "bin"))
            src.write_text("\n".join(program(shift, spelling)) + "\n", encoding="ascii")
            subprocess.run([llvm_mc, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", str(src),
                            "-o", str(obj)], check=True)
            subprocess.run([objcopy, "-O", "binary", "--only-section=.text", str(obj), str(raw)], check=True)
            encoded[spelling] = raw.read_bytes()
        assert encoded["bound_ctrl:0"] == encoded["bound_ctrl:1"], shift
        code = encoded["bound_ctrl:1"]
        decoded = aot.labelled(disassemble(code, library, offsets=True))
        dpp = [(offset, line) for offset, line in disassemble(code, library, offsets=True) if "_dpp " in line]
        assert [line for _, line in dpp] == [line for line in program(shift) if "_dpp " in line], dpp
        for (offset, line), bound, src0 in zip(dpp, (True, False, True, False), (1, 1, 7, 7)):
            # The DPP16 dword: src0 [7:0], DPP_CTRL [16:8], FI [18], BOUND_CTRL [19], masks [31:24].
            word = struct.unpack_from("<I", code, offset + 4)[0]
            assert word & 0xff == src0 and (word >> 8) & 0x1ff == 0x110 + shift, (line, hex(word))
            assert (word >> 18) & 1 == 0 and (word >> 19) & 1 == bound and word >> 24 == 0xff, (line, hex(word))
        assert translate(decoded) == translate(program(shift)), shift
    print("binary: LLVM 15 encodes bound_ctrl:0 and bound_ctrl:1 alike (bit 19 set) and decodes them as")
    print("  bound_ctrl:1; DPP_CTRL 0x111, 0x118 and 0x11f; the decoded programs translate as the source")


def dispatch(directory, dxc, renderer, backend, label, source, words):
    src, cso, case = (directory / f"{label}.{ext}" for ext in ("hlsl", "cso", "case"))
    src.write_text(source, encoding="ascii")
    subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", str(cso), str(src)], check=True)
    case.write_bytes(struct.pack(f"<{1 + WORDS}I", 1, *words))
    return subprocess.run([renderer, backend, "--compute-store-case", str(cso), str(case)],
                          capture_output=True, text=True)


def check_draw(directory, dxc, renderer, backend):
    controls()
    for shift in SHIFTS:
        source, words = translate(program(shift)), expected(shift)
        run = dispatch(directory, dxc, renderer, backend, f"row_shr_{shift}", source, words)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            return 77
        if run.returncode:
            raise AssertionError(f"row_shr:{shift}\n{run.stdout}\n{run.stderr}")
        output = (run.stdout + run.stderr).splitlines()
        if shift == SHIFTS[0]:
            print(" ".join(line.strip() for line in output if line.startswith(("adapter=", "driver="))))
        debug = next((line for line in output if "debug_errors=" in line), "")
        result = next((line for line in run.stdout.splitlines() if "compared_words=" in line), "")
        print(f"row_shr:{shift}: {result.strip()} {debug.strip()}")
        for mutant in ("cross_row", "zero"):
            bad_words = expected(shift, mutant)
            first = first_difference(words, bad_words)
            run = dispatch(directory, dxc, renderer, backend, f"row_shr_{shift}_{mutant}", mutate(source, mutant),
                           words)
            line = f"word={first} expected={words[first]:08x} actual={bad_words[first]:08x}"
            if run.returncode != 1 or line not in run.stderr:
                raise AssertionError(f"row_shr:{shift} {mutant}: expected first mismatch {line}\n"
                                     f"{run.stdout}\n{run.stderr}")
            print(f"row_shr:{shift} {mutant}: first mismatch {line}, as predicted")
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        opcodes = sorted({line.split()[0] for shift in SHIFTS for line in program(shift)})
        path.write_text("".join(f"{op}\n" for op in opcodes), encoding="utf-8")
    return 0


def main():
    if len(sys.argv) == 1:
        controls()
        return 0
    mode, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "binary":
            controls()
            check_binary(*args, Path(tmp))
            return 0
        if mode == "draw":
            return check_draw(Path(tmp), *args)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
