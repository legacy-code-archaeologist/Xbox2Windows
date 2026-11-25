"""The `src_scc` scalar source in tools/startup/gfx2hlsl.py (issue #1820).

usage: startup_gfx2hlsl_src_scc_check.py

The census contained `s_mul_i32 s28, 1.0, src_scc`, refused as `operand 'src_scc'` (#870, issue #1820).
Source code 253 reads the SCC bit as the integer value 0 or 1; the generated shader carries SCC as a
`bool scc`, so `Stage.u()` lowers `src_scc` to `(scc ? 1u : 0u)` when a modeled writer has set SCC on
every reaching path in the current block, and refuses when the incoming SCC is unverified (at shader entry,
after an unmodeled SCC writer such as `s_add_u32`, after descriptor-elided SCC, or across control-flow
joins).

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`):
- SOP2 (`[31:30] == 0b10`, op in `w0[29:23]`, sdst in `w0[22:16]`, src1 in `w0[15:8]`, src0 in `w0[7:0]`,
  optional literal dword when a source is 255)
- SOPC (`[31:23] == 0b101111110`, op in `w0[22:16]`, src1 in `w0[15:8]`, src0 in `w0[7:0]`)
- SOPP (`[31:23] == 0b101111111`, op in `w0[22:16]`, simm16 in `w0[15:0]`)
"""
import os
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


SOP2_ADD_U32 = 0x00
SOP2_AND_B32 = 0x0E
SOP2_MUL_I32 = 0x26
SOPC_CMP_EQ_U32 = 0x06
SOPP_NOP = 0x00
SOPP_ENDPGM = 0x01
SOPP_CBRANCH_SCC0 = 0x04

S_ZERO = 128        # inline integer 0
S_ONE = 129         # inline integer 1
S_SCC = 253         # the src_scc scalar source code
S_1_0 = 242         # the inline float constant 1.0 (its raw bits are 0x3f800000)
LITERAL = 255
S1, S4, S5, S27, S28, S29 = 1, 4, 5, 27, 28, 29


def encode_sop2(op, sdst, ssrc0, ssrc1, literal=None):
    w0 = ((0b10 << 30) | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16)
          | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF))
    code = struct.pack('<I', w0)
    if literal is not None:
        code += struct.pack('<I', literal)
    return code


def encode_sopc(op, ssrc0, ssrc1, literal=None):
    w0 = (0xBF000000 | ((op & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF))
    code = struct.pack('<I', w0)
    if literal is not None:
        code += struct.pack('<I', literal)
    return code


def encode_sopp(op, simm=0):
    w0 = 0xBF800000 | ((op & 0x7F) << 16) | (simm & 0xFFFF)
    return struct.pack('<I', w0)


class Pipeline:
    """Just enough pipeline state for Stage.translate()."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


def translate_program(code):
    """Decode and run the full Stage.translate() CFG/block pass on synthetic bytes."""
    decoded = G.decode(code)
    stage = G.Stage.__new__(G.Stage)
    stage.pipe = Pipeline()
    stage.kind = 'ps'
    stage.lines, stage.sym = [], {}
    stage.used_s, stage.used_v, stage.masks = set(), set(), set()
    stage.attrs = {}
    stage.exec_known = True
    stage.ins = decoded
    stage.force_joined = {}
    stage.translate()
    body = ' '.join(line for _, lines, _ in stage.blocks for line in lines)
    return [asm for _o, _s, _n, asm in decoded], body, stage


def refused_reason(code):
    try:
        translate_program(code)
    except G.Unsupported as error:
        return str(error)
    return None


# 1. At shader entry, SCC is unverified: reading src_scc before any modeled writer must be refused.
check('entry src_scc in s_mul_i32 is refused',
      refused_reason(encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC) + encode_sopp(SOPP_ENDPGM))
      == 'src_scc with unverified SCC')
check('entry src_scc in s_and_b32 is refused',
      refused_reason(encode_sop2(SOP2_AND_B32, S28, S_SCC, S5) + encode_sopp(SOPP_ENDPGM))
      == 'src_scc with unverified SCC')

# 2. Modeled compare -> census s_mul_i32 s28, 1.0, src_scc: decodes and lowers using the SCC bit.
cmp_census = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)
              + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)
              + encode_sopp(SOPP_ENDPGM))
asm, emitted, stage = translate_program(cmp_census)
check('encodes and decodes compare + census form',
      asm[:2] == ['s_cmp_eq_u32 0, 0', 's_mul_i32 s28, 1.0, src_scc'], repr(asm))
check('compare + census form lowers to the SCC bit',
      'scc = 0x00000000u == 0x00000000u;' in emitted and 's28 = 0x3f800000u * (scc ? 1u : 0u);' in emitted,
      emitted)
check('low16_only=True returns the same 0/1 expression',
      stage.u('src_scc', low16_only=True) == '(scc ? 1u : 0u)')

# 3. Modeled bitwise writer -> src_scc in the first source position -> next instruction reads updated SCC.
bitwise_seq = (encode_sop2(SOP2_AND_B32, S27, S4, S5)
               + encode_sop2(SOP2_AND_B32, S28, S_SCC, S5)
               + encode_sop2(SOP2_MUL_I32, S29, S_1_0, S_SCC)
               + encode_sopp(SOPP_ENDPGM))
asm, emitted, _stage = translate_program(bitwise_seq)
check('encodes and decodes bitwise sequence with src_scc in first position',
      asm[:3] == ['s_and_b32 s27, s4, s5', 's_and_b32 s28, src_scc, s5', 's_mul_i32 s29, 1.0, src_scc'],
      repr(asm))
check('bitwise sequence reads and updates SCC cleanly',
      's28 = (scc ? 1u : 0u) & s5; scc = s28 != 0;' in emitted
      and 's29 = 0x3f800000u * (scc ? 1u : 0u);' in emitted,
      emitted)

# 4. Distinguishing carry case: compare sets SCC=0, then s_add_u32 0xffffffff + 1 wraps and sets hardware
#    carry SCC=1 without an HLSL SCC update. Reading src_scc afterwards must refuse rather than read stale 0.
carry_seq = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ONE)
             + encode_sop2(SOP2_ADD_U32, S1, LITERAL, S_ONE, literal=0xFFFFFFFF)
             + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)
             + encode_sopp(SOPP_ENDPGM))
check('s_add_u32 carry after compare leaves SCC unverified and is refused',
      refused_reason(carry_seq) == 'src_scc with unverified SCC',
      repr(refused_reason(carry_seq)))

# 5. s_add_u32 reading src_scc as an input before overwriting SCC is accepted (input read precedes write),
#    while a second src_scc after s_add_u32 is refused.
add_reads_scc = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)
                 + encode_sop2(SOP2_ADD_U32, S1, S_SCC, S_ONE)
                 + encode_sopp(SOPP_ENDPGM))
asm, emitted, _stage = translate_program(add_reads_scc)
check('s_add_u32 reading incoming verified src_scc is accepted',
      's1 = (scc ? 1u : 0u) + 0x00000001u;' in emitted, emitted)
check('reading src_scc after s_add_u32 is refused',
      refused_reason(encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)
                     + encode_sop2(SOP2_ADD_U32, S1, S_SCC, S_ONE)
                     + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)
                     + encode_sopp(SOPP_ENDPGM)) == 'src_scc with unverified SCC')

# 6. Across a control-flow join: branch skips s_nop 0 to target 0x10; incoming SCC at the join is unverified,
#    so src_scc there is refused unless a modeled writer runs inside the joined block first.
join_refused = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)     # 0x00
                + encode_sopp(SOPP_CBRANCH_SCC0, simm=1)          # 0x04 -> label_0x0c (skip 1 dword at 0x08)
                + encode_sopp(SOPP_NOP, simm=0)                   # 0x08
                + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)    # 0x0c (branch target / join)
                + encode_sopp(SOPP_ENDPGM))                       # 0x10
check('src_scc across a control-flow join is refused',
      refused_reason(join_refused) == 'src_scc with unverified SCC',
      repr(refused_reason(join_refused)))

join_rewritten = (encode_sopc(SOPC_CMP_EQ_U32, S_ZERO, S_ZERO)   # 0x00
                  + encode_sopp(SOPP_CBRANCH_SCC0, simm=1)        # 0x04 -> 0x0c
                  + encode_sopp(SOPP_NOP, simm=0)                 # 0x08
                  + encode_sopc(SOPC_CMP_EQ_U32, S4, S5)          # 0x0c (modeled writer after join)
                  + encode_sop2(SOP2_MUL_I32, S28, S_1_0, S_SCC)  # 0x10
                  + encode_sopp(SOPP_ENDPGM))
asm, emitted, _stage = translate_program(join_rewritten)
check('src_scc after a modeled writer in the joined block is accepted',
      's28 = 0x3f800000u * (scc ? 1u : 0u);' in emitted, emitted)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
