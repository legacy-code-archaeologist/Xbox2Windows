"""v_readlane_b32 and v_permlanex16_b32 are refused: the selected source lane may be inactive."""
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/startup"))
import gfx2hlsl as gfx


class TestPipeline:
    root = {"params": [{"type": 2}], "flags": 0}

    def resource(self, key, kind):
        return "r0"


def vop1(opcode, dst, src):
    return 0x7e000000 | dst << 17 | opcode << 9 | src


def readlane(sdst, vsrc, select):
    return [0xd7600000 | sdst, 256 + vsrc | select << 9]


def permlanex16(vdst, vsrc, sel0, sel1):
    return [0xd7780000 | vdst, 256 + vsrc | sel0 << 9 | sel1 << 18]


def translate(*extra):
    # s_buffer_load_dword s7, s[8:11], 12; v_cvt_i32_f32 v4, s7 (uniform over the active lanes); extra; s_endpgm
    words = [0xf4000000 | 8 << 18 | 7 << 6 | 4, 125 << 25 | 12, vop1(8, 4, 7), *extra, 0xbf810000]
    return "\n".join(gfx.Stage(TestPipeline(), "ps", struct.pack(f"<{len(words)}I", *words)).blocks[0][1])


# A uniform VGPR over the active lanes does not make an inactive source lane readable: with a sparse EXEC,
# permlanex16 lane 0 can select inactive lane 16 and keep the old destination, so both forms are refused.
for words, text in [
    (readlane(6, 4, 128 + 5), "v_readlane_b32 requires every selected source lane to be proven active"),
    (permlanex16(5, 4, 1, 2), "v_permlanex16_b32 requires every selected source lane to be proven active"),
    (permlanex16(5, 4, 0, 0), "v_permlanex16_b32 requires every selected source lane to be proven active"),
]:
    try:
        translate(*words)
    except gfx.Unsupported as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("accepted " + text)
print("readlane and permlanex16 refused: source lane activity is not provable")
