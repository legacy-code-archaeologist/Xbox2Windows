"""Compare native and translated FCOM, FCOMP, FCOMPP, FUCOM, FUCOMP, FUCOMPP and FNSTSW AX of the x87
leaf-function profile, with the readers of the status word that the profile accepts.

Usage:
    python tests/cpu/x87_status.py asm <asm> <def>             regenerate the MASM source and exports
    python tests/cpu/x87_status.py emit <dll> <cpp> [control]   translate (and optionally mutate) the fixture
    python tests/cpu/x87_status.py check <harness> <native> <lifted> [controls...]

The exports, harness and guards follow tests/cpu/x87_add.py, and the load and store steps use the format of
tests/cpu/x87_moves.py. After FNSTSW AX a record step reads AH through each accepted consumer:
- SAHF and JE, JNE, JB, JAE, JA, JBE, JP and JNP (taken bytes);
- TEST AH, imm8 with JNE for C0, C2, C3 and each TOP bit, and with JP for the 45h, 44h, 05h and 41h masks
  that compilers test;
- AND AH, 45h, stored, then JP, then CMP AH with 40h, 01h and 00h and JE.
It then clears EAX, because the other bits of AX are not known. byte_parity reads the PF of the other 8-bit
TEST and AND forms on every byte value. Only flag-neutral MOVs come in between.
emit checks that every decoded x87 instruction is the form its sequence names. The model compares with
Fraction arithmetic (tests/cpu/x87_compare.py) and is independent of tools/x87_soft.h. Two native-only
exports (NATIVE) read status-word bits that the profile does not know, and translation must reject each.
"""
from pathlib import Path
import platform
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)
import x87_compare as compare  # noqa: E402
import x87_moves as moves  # noqa: E402
from x87_add import J, from_binary, m80_bytes, to_binary  # noqa: E402
from x87_div import SIZE  # noqa: E402

INPUT, OUTPUT = base.INPUT, base.OUTPUT
REGISTER = ("fcom", "fcomp", "fucom", "fucomp")  # (op, i): ST(0) with ST(i)
DOUBLE = ("fcompp", "fucompp")                   # (op,): ST(0) with ST(1), then two pops
MEMORY = ("fcom_m", "fcomp_m")                   # (op, kind, offset): ST(0) with an m32 or m64 input
FLAG_COMPARES = ("fcomi", "fucomip")             # (op, i): leave C0, C2 and C3 as they were
POPS = {"fcom": 0, "fcomp": 1, "fucom": 0, "fucomp": 1, "fcompp": 2, "fucompp": 2, "fcom_m": 0, "fcomp_m": 1,
        "fcomi": 0, "fucomip": 1}
SAHF_JUMPS = ("je", "jne", "jb", "jae", "ja", "jbe", "jp", "jnp")
TEST_NE = (0x01, 0x04, 0x40, 0x08, 0x10, 0x20)  # C0, C2, C3 and the three TOP bits, in AH
TEST_P = (0x45, 0x44, 0x05, 0x41)
CMP_EQ = (0x40, 0x01, 0x00)
RECORD = len(SAHF_JUMPS) + len(TEST_NE) + len(TEST_P) + 2 + len(CMP_EQ)  # 23 bytes
PARITY8 = 4  # the PF of TEST AL, AL, TEST m8, 0A5h and AND AL, 3Ch, and the AND result

P80 = [("fld", "m80", 0), ("fld", "m80", 16)]  # ST(0) = the value at 16, ST(1) = the value at 0
T80 = P80 + [("fld", "m80", 32)]
SW = [("fnstsw",), ("record", 0)]
SEQUENCES = {
    **{f"{op}_st1": ("pair80", P80 + [(op, 1)] + SW + [("fstp", "m80", 24)] + ([] if POPS[op] else [("fstp", "m80", 40)]))
       for op in REGISTER},
    **{f"{op}_pair": ("pair80", P80 + [(op,)] + SW) for op in DOUBLE},
    "fcom_st2": ("triple80", T80 + [("fcom", 2)] + SW + [("fstp", "m80", 24), ("fstp", "m80", 40), ("fstp_st", 0)]),
    "fucomp_st2": ("triple80", T80 + [("fucomp", 2)] + SW + [("fstp", "m80", 24), ("fstp", "m80", 40)]),
    "fcom_st0": ("m80", [("fld", "m80", 0), ("fcom", 0)] + SW + [("fstp", "m80", 24)]),
    "fucomp_st0": ("m80", [("fld", "m80", 0), ("fucomp", 0)] + SW),
    "fcom_m32": ("pair32", [("fld", "m32", 0), ("fcom_m", "m32", 4)] + SW + [("fstp_st", 0)]),
    "fcomp_m32": ("pair32", [("fld", "m32", 0), ("fcomp_m", "m32", 4)] + SW),
    "fcom_m64": ("pair64", [("fld", "m64", 0), ("fcom_m", "m64", 8)] + SW + [("fstp_st", 0)]),
    "fcomp_m64": ("pair64", [("fld", "m64", 0), ("fcomp_m", "m64", 8)] + SW),
    # Depth 8, where TOP is 0 again.
    "fucom_st7": ("m64x8", [("fld", "m64", 8 * k) for k in range(8)] + [("fucom", 7)] + SW + [("fstp_st", 0)] * 8),
    # FCOMI and FUCOMIP leave C0, C2 and C3 as the earlier comparison set them; FUCOMIP also pops (TOP 6).
    "fcom_fcomi": ("triple80", T80 + [("fcom", 1), ("fcomi", 2)] + SW + [("fstp", "m80", 24), ("fstp", "m80", 40),
                                                                       ("fstp_st", 0)]),
    "fucom_fucomip": ("triple80", T80 + [("fucom", 2), ("fucomip", 1)] + SW + [("fstp", "m80", 24), ("fstp", "m80", 40)]),
    # Two comparisons, each read through its own FNSTSW.
    "two_records": ("triple80", T80 + [("fucomp", 1), ("fnstsw",), ("record", 0), ("fcomp", 1), ("fnstsw",),
                                       ("record", RECORD), ("fstp", "m80", 48)]),
    # The input dword at 48 selects FCOM ST(1) or FUCOM ST(2); both paths define C0, C2 and C3.
    "joined": ("tripleint", T80 + [("choose", 1, 2)] + SW + [("fstp", "m80", 24), ("fstp", "m80", 40), ("fstp_st", 0)]),
    # Each successor of TEST AH, 40h tests a known bit: C0 when C3 is clear, C2 when it is set.
    "split": ("pair80", P80 + [("fcom", 1), ("fnstsw",), ("split", 0), ("fstp", "m80", 24), ("fstp", "m80", 40)]),
    # SAHF and the record on every AH value loaded by MOV, so all eight bits are known.
    "byte_record": ("byte", [("fld", "m80", 0), ("byte_ah", 48), ("record", 0), ("fstp", "m80", 24)]),
    # PF of the other 8-bit TEST and AND forms, on every byte.
    "byte_parity": ("byte", [("fld", "m80", 0), ("parity8", 0), ("fstp", "m80", 24)]),
}
# Native only: each reads a status-word bit that the profile does not know, and translation must reject it.
# The value is the export's sequence and the needle of the rejection.
NATIVE = {
    # The C3 successor tests IE (bit 0 of AL) instead of C2. FUCOM sets IE for a signaling NaN only.
    "split_sticky": (("pair80", P80 + [("fucom", 1), ("fnstsw",), ("split_al", 0), ("fstp", "m80", 24),
                                       ("fstp", "m80", 40)]),
                     "test al,1 at 0x"),
    # The whole status word, stored.
    "status_word": (("pair80", P80 + [("fcom", 1), ("fnstsw",), ("store_ax", 0), ("fstp", "m80", 24),
                                      ("fstp", "m80", 40)]),
                    "mov [rdx],ax at 0x"),
}
STEPS = {**SEQUENCES, **{name: steps for name, (steps, _) in NATIVE.items()}}

STATUS = r"x87_sw = static_cast<uint16_t>\(r == 0 \? 0x0100 : r == 1 \? 0x4000 : r == 3 \? 0x4500 : 0\);"
CONTROLS = {
    # An unordered comparison leaves C2 clear.
    "c2drop": (re.compile(r"r == 3 \? 0x4500"), "r == 3 ? 0x4100"),
    # Less sets C3 and equal sets C0.
    "c0c3": (re.compile(r"r == 0 \? 0x0100 : r == 1 \? 0x4000"), "r == 0 ? 0x4000 : r == 1 ? 0x0100"),
    # FNSTSW stores TOP as 0.
    "top0": (re.compile(r"x87_sw \| 0x[0-9a-f]{4}\);"), "x87_sw | 0x0000);"),
    # SAHF loads PF from bit 4 (AF) instead of bit 2.
    "sahfpf": (re.compile(r"parity = \(h & 4\) != 0;"), "parity = (h & 0x10) != 0;"),
    # TEST and AND set PF for odd parity.
    "oddparity": (re.compile(r"& 0xfu\)\) & 1u\) == 0;"), "& 0xfu)) & 1u) == 1;"),
    # AND sets PF for odd parity; TEST is unchanged.
    "andparity": (re.compile(r"& 0xfu\)\) & 1u\) == 0;( zero = v == 0;)"), r"& 0xfu)) & 1u) == 1;\1"),
    # FCOMI and FUCOMIP clear C0, C2 and C3.
    "fcomiclear": (re.compile(r"(carry = r == 0 \|\| r == 3; overflow = false;) \}"), r"\1 x87_sw = 0; }"),
}


# ---- MASM generation and decoding ------------------------------------------------------------------

def taken(label, offset, jump, value=(1, 0)):
    """Store value[0] at offset if jump is taken, else value[1]; MOV leaves the flags unchanged."""
    return [f"    mov byte ptr [rdx+{offset}], {value[0]}", f"    {jump} {label}",
            f"    mov byte ptr [rdx+{offset}], {value[1]}", f"{label}:"]


def asm_lines(step, name):
    op, *args = step
    if op in REGISTER:
        return [f"    {op} st({args[0]})"]
    if op in DOUBLE:
        return [f"    {op}"]
    if op in MEMORY:
        return [f"    {op[:-2]} {base.PTR[args[0]]} ptr [rcx+{args[1]}]"]
    if op in FLAG_COMPARES:
        return [f"    {op} st, st({args[0]})"]
    if op == "fnstsw":
        return ["    fnstsw ax"]
    if op == "byte_ah":
        return [f"    mov ah, byte ptr [rcx+{args[0]}]"]
    if op == "choose":
        return ["    mov eax, dword ptr [rcx+48]", "    test eax, eax", f"    je {name}_alt", f"    fcom st({args[0]})",
                f"    jmp {name}_join", f"{name}_alt:", f"    fucom st({args[1]})", f"{name}_join:"]
    if op in ("split", "split_al"):
        o = args[0]
        last = "    test ah, 04h" if op == "split" else "    test al, 01h"
        return (["    test ah, 40h", f"    jne {name}_c3", "    test ah, 01h", *taken(f"{name}_c0", o, "jne")[:2],
                 f"    mov byte ptr [rdx+{o}], 0", f"    jmp {name}_c0", f"{name}_c3:", last,
                 *taken(f"{name}_c0", o, "jne", (3, 2))[:3], f"{name}_c0:", "    mov eax, 0"])
    if op == "parity8":
        o = args[0]
        return (["    mov al, byte ptr [rcx+48]", "    test al, al", *taken(f"{name}_p0", o, "jp"),
                 "    test byte ptr [rcx+48], 0A5h", *taken(f"{name}_p1", o + 1, "jp"),
                 "    and al, 3Ch", *taken(f"{name}_p2", o + 2, "jp"), f"    mov byte ptr [rdx+{o + 3}], al",
                 "    mov eax, 0"])
    if op == "store_ax":
        return [f"    mov word ptr [rdx+{args[0]}], ax", "    mov eax, 0"]
    if op != "record":
        return [moves.asm_line(step)]
    o = args[0]
    lines = ["    sahf"]
    for k, jump in enumerate(SAHF_JUMPS):
        lines += taken(f"{name}_{o}_{jump}", o + k, jump)
    o += len(SAHF_JUMPS)
    for k, mask in enumerate(TEST_NE):
        lines += [f"    test ah, 0{mask:02X}h", *taken(f"{name}_{o}_ne{k}", o + k, "jne")]
    o += len(TEST_NE)
    for k, mask in enumerate(TEST_P):
        lines += [f"    test ah, 0{mask:02X}h", *taken(f"{name}_{o}_p{k}", o + k, "jp")]
    o += len(TEST_P)
    lines += ["    and ah, 45h", f"    mov byte ptr [rdx+{o}], ah", *taken(f"{name}_{o}_andp", o + 1, "jp")]
    o += 2
    for k, value in enumerate(CMP_EQ):
        lines += [f"    cmp ah, 0{value:02X}h", *taken(f"{name}_{o}_eq{k}", o + k, "je")]
    return lines + ["    mov eax, 0"]


def expected_forms(step):
    op, *args = step
    if op in REGISTER or op in FLAG_COMPARES:
        return [(op, f"st0,st{args[0]}")]
    if op in DOUBLE:
        return [(op, "")]
    if op in MEMORY:
        return [(op[:-2], args[0])]
    if op == "fnstsw":
        return [("fnstsw", "ax")]
    if op == "choose":
        return [("fcom", f"st0,st{args[0]}"), ("fucom", f"st0,st{args[1]}")]
    if op in ("record", "byte_ah", "split", "split_al", "store_ax", "parity8"):
        return []
    return [moves.expected_form(step)]


def decoded_form(i):
    from iced_x86 import Mnemonic as M
    import x87_div as forms

    if i.mnemonic == M.FNSTSW:
        return "fnstsw", "ax"
    if i.op_count == 2:
        return forms.decoded_form(i)
    return moves.decoded_form(i)


def write_asm(asm_path, def_path):
    lines = ["; coverage: source cpu_aot_x87_status",
             "; Generated by tests/cpu/x87_status.py asm; edit SEQUENCES there and regenerate.",
             "; Source-owned differential fixture for the comparisons into the status word and FNSTSW AX of the",
             "; x87 leaf-function profile. Each export reads an input block at RCX, writes an output block at RDX",
             "; and returns with the x87 stack empty.", ".code", ""]
    for name, (_, steps) in STEPS.items():
        lines += [f"{name} PROC FRAME", "    .endprolog", *(line for s in steps for line in asm_lines(s, name)),
                  "    ret", f"{name} ENDP", ""]
    lines.append("END")
    Path(asm_path).write_text("\n".join(lines) + "\n", encoding="ascii")
    Path(def_path).write_text("EXPORTS\n" + "".join(f"    {name}\n" for name in STEPS), encoding="ascii")


def lower_export(pe, name):
    from cpu_leaf_aot import export_rva
    from cpu_ranges import function_ranges
    from ffx_version_query import lower, region
    from x87_lower import is_x87

    (bounds,) = function_ranges(pe, export_rva(pe, name))
    body = region(pe, *bounds)
    decoded = [decoded_form(i) for i in body if is_x87(i)]
    wanted = [form for s in STEPS[name][1] for form in expected_forms(s)]
    if decoded != wanted:
        raise AssertionError(f"{name} assembled as {decoded}, expected {wanted}")
    return lower(body, ["rcx = a; rdx = b;"], {}, {}, {}, x87=True)


def emit(image, output, control="lifted"):
    import pefile
    from ffx_version_query import INTEGER_HEADER
    from x87_lower import X87_HEADER

    parts = [INTEGER_HEADER, X87_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in SEQUENCES:
            code = lower_export(pe, name)
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b) {{\n{code}\n}}\n')
    text = "".join(parts)
    if not re.search(STATUS, text):
        raise AssertionError("no comparison into the status word was lowered")
    if control != "lifted":
        expression, replacement = CONTROLS[control]
        text, count = expression.subn(replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
        print(f"control {control}: {count} replacements")
    output.write_text(text, encoding="ascii")


# ---- Independent model -----------------------------------------------------------------------------

def known_ah(result, top):
    """The known bits of AH after FNSTSW AX: C0 (bit 0), C2 (bit 2), TOP (bits 3 to 5) and C3 (bit 6)."""
    c0, c2, c3 = result in (0, 3), result == 3, result in (1, 3)
    return int(c0) | int(c2) << 2 | top << 3 | int(c3) << 6


def even(value):
    return bin(value).count("1") % 2 == 0


def record_bytes(ah):
    """The record of an AH whose bits 1 and 7 (C1 and the busy bit) are not read."""
    zf, cf, pf = bool(ah & 0x40), bool(ah & 0x01), bool(ah & 0x04)
    jumps = {"je": zf, "jne": not zf, "jb": cf, "jae": not cf, "ja": not cf and not zf, "jbe": cf or zf,
             "jp": pf, "jnp": not pf}
    masked = ah & 0x45
    return (bytes(int(jumps[j]) for j in SAHF_JUMPS) + bytes(int(ah & m != 0) for m in TEST_NE)
            + bytes(int(even(ah & m)) for m in TEST_P) + bytes((masked, int(even(masked))))
            + bytes(int(masked == v) for v in CMP_EQ))


def operand(kind, raw):
    return (raw & (2 ** 64 - 1), raw >> 64) if kind == "m80" else from_binary(kind, raw)


def reference(name, data):
    """The output block, and the (known AH, comparison result) at the last FNSTSW."""
    slots, out = [], bytearray(b"\xcc" * OUTPUT)
    result, ah = None, None

    def read(kind, offset):
        return int.from_bytes(data[offset:offset + SIZE[kind]], "little")

    for op, *args in STEPS[name][1]:
        if op == "fld":
            slots.append(operand(args[0], read(*args)))
        elif op == "fstp":
            kind, offset = args
            v = slots.pop()
            encoded = v[0] | v[1] << 64 if kind == "m80" else to_binary(kind, v)
            out[offset:offset + SIZE[kind]] = encoded.to_bytes(SIZE[kind], "little")
        elif op == "fstp_st":
            slots[-1 - args[0]] = slots[-1]
            slots.pop()
        elif op in REGISTER or op in DOUBLE or op in MEMORY or op in FLAG_COMPARES:
            if op in MEMORY:
                other = operand(args[0], read(*args))
            else:
                other = slots[-1 - (args[0] if args else 1)]
            r = compare.compare(slots[-1], other)
            if op not in FLAG_COMPARES:
                result = r
            del slots[len(slots) - POPS[op]:]
        elif op == "choose":
            r = compare.compare(slots[-1], slots[-1 - (args[0] if read("m32", 48) else args[1])])
            result = r
        elif op == "fnstsw":
            ah = known_ah(result, (8 - len(slots)) & 7)
            result = None  # FNSTSW leaves C0, C2 and C3 undefined
        elif op == "byte_ah":
            ah = data[args[0]]
        elif op == "parity8":
            b = data[48]
            out[args[0]:args[0] + PARITY8] = bytes((int(even(b)), int(even(b & 0xA5)), int(even(b & 0x3C)), b & 0x3C))
        elif op == "record":
            out[args[0]:args[0] + RECORD] = record_bytes(ah)
        elif op == "split":  # 0 greater, 1 less, 2 equal, 3 unordered
            c0, c2, c3 = ah & 1, ah >> 2 & 1, ah >> 6 & 1
            out[args[0]] = (2 + c2) if c3 else c0
        elif op in ("split_al", "store_ax"):
            pass  # bits the model does not know; check_native compares the known part
        else:
            raise AssertionError(op)
    assert not slots
    return bytes(out), ah


# ---- Inputs ----------------------------------------------------------------------------------------

def cases():
    rng = random.Random(53)
    specials = base.special_m80()

    def random_m80():
        return J | rng.getrandbits(63), rng.randint(0x3FFF - 70, 0x3FFF + 70) | rng.choice((0, 0x8000))

    near = []  # equal values, neighbours and equal values in other encodings
    for _ in range(100):
        v = random_m80()
        near += [(v, v), (v, (v[0] ^ 1, v[1])), (v, (v[0], v[1] ^ 0x8000)), (v, (v[0], v[1] + 1))]
    near += [((J | x, 0), (J | x, 1)) for x in (0, 1, 0x7FF)]  # pseudo-denormal and normal twins
    pairs = [(a, b) for a in specials for b in specials] + near + [(random_m80(), random_m80()) for _ in range(200)]
    f64, f32 = base.f64_bits(), base.f32_bits()

    def triple():
        return b"".join(m80_bytes(v) for v in rng.sample(specials, 3))

    blocks = {
        "m80": [m80_bytes(v) for v in base.m80_singles()],
        "pair80": [m80_bytes(a) + m80_bytes(b) for a, b in pairs],
        "triple80": [triple() for _ in range(300)] + [b"".join(m80_bytes(v) for v in (a, b, a)) for a, b in near[:200]],
        "tripleint": [triple().ljust(48, b"\0") + rng.choice((0, 1, 0x80000000, rng.getrandbits(32))).to_bytes(4, "little")
                      for _ in range(400)],
        "pair64": [a.to_bytes(8, "little") + b.to_bytes(8, "little") for a in f64[:18] for b in f64[:18]]
                  + [a.to_bytes(8, "little") + b.to_bytes(8, "little") for a, b in zip(f64, reversed(f64))],
        "pair32": [a.to_bytes(4, "little") + b.to_bytes(4, "little") for a in f32[:17] for b in f32[:17]],
        "m64x8": [b"".join(rng.choice(f64).to_bytes(8, "little") for _ in range(8)) for _ in range(200)],
        "byte": [m80_bytes(random_m80()).ljust(48, b"\0") + bytes((ah,)) for ah in range(256)],
    }
    for name, (kind, _) in STEPS.items():
        for block in blocks[kind]:
            yield name, block.ljust(INPUT, b"\0")


# ---- Check -----------------------------------------------------------------------------------------

def check(harness_path, native_path, lifted_path, *control_paths):
    expected = [(name, data, *reference(name, data)) for name, data in cases() if name in SEQUENCES]
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        run = base.Runner(harness_path, path)
        bad = 0
        for name, data, want, _ in expected:
            got, state = run(name, data)
            if not base.entry_state_ok(state):
                raise AssertionError(f"{label} {name} left state {state:#014x}; the profile requires control "
                                     "0x027f and an empty stack at TOP 0")
            if got != want:
                if bad < 3:
                    print(f"{label} mismatch: {name}(input {data[:48].hex()})\n  got      {got[:48].hex()}\n"
                          f"  expected {want[:48].hex()}")
                bad += 1
        if bad:
            raise AssertionError(f"{label}: {bad}/{len(expected)} mismatches")
    print(f"{len(expected)} cases over {len(SEQUENCES)} exports (output block and guards) matched the reference "
          "in native and translated DLLs; each export returned with control 0x027f and an empty stack")
    outcomes = {}
    with_status = {name for name, (_, steps) in SEQUENCES.items() if ("fnstsw",) in steps}
    for name, data, want, ah in expected:
        if name in with_status:
            key = {0x01: "less", 0x40: "equal", 0x00: "greater", 0x45: "unordered"}[ah & 0x45]
            outcomes[key] = outcomes.get(key, 0) + 1
    tops = sorted({ah >> 3 & 7 for name, data, want, ah in expected if name in with_status})
    print("C3, C2 and C0 at FNSTSW: " + ", ".join(f"{key} {outcomes.get(key, 0)}"
                                               for key in ("less", "equal", "greater", "unordered"))
          + f"; TOP values {tops}")
    print(f"processor: {platform.processor()}")
    first = {}
    for name, data, want, _ in expected:
        first.setdefault(name, (data, want))
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        run = base.Runner(harness_path, path)
        failed = [name for name, (data, want) in first.items() if not base.preserved(run, "x87_call", name, data, want)]
        if failed:
            raise AssertionError(f"x87_call lost a live caller value or control word around {label} {failed}")
    print(f"x87_call kept a live caller ST(0) value and control word around all {len(first)} native and "
          "translated exports")
    check_native(harness_path, native_path)
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        run = base.Runner(harness_path, path)
        bad = sum(1 for name, data, want, _ in expected if run(name, data)[0] != want)
        print(f"control {control}: {bad}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")
    check_rejections()


def check_native(harness_path, native_path):
    """The NATIVE exports: the known bits match the model, the unknown ones vary, and translation rejects."""
    import pefile

    run = base.Runner(harness_path, native_path)
    blocks = [data for name, data in cases() if name == "status_word"]
    words, sticky, runs = {}, {}, 0
    for data in blocks:
        result = compare.compare(operand("m80", int.from_bytes(data[16:26], "little")),
                                 operand("m80", int.from_bytes(data[0:10], "little")))
        for name in NATIVE:
            want, ah = reference(name, data)
            got, state = run(name, data)
            runs += 1
            if not base.entry_state_ok(state) or got[2:] != want[2:]:
                raise AssertionError(f"native {name} gave {got[:48].hex()}, expected {want[:48].hex()}")
            if name == "status_word":
                word = int.from_bytes(got[0:2], "little")
                if word >> 8 & 0x7D != ah:
                    raise AssertionError(f"native {name} stored {word:#06x}; the known bits of AH are {ah:#04x}")
                words[word & 0x82FF] = words.get(word & 0x82FF, 0) + 1
            else:
                if result in (0, 2) and got[0] != (result == 0):
                    raise AssertionError(f"native {name} gave {got[0]} for result {result}")
                if result in (1, 3):
                    sticky.setdefault(result, set()).add(got[0])
    if len(words) < 2 or sticky.get(3) != {2, 3}:
        raise AssertionError(f"the unknown bits did not vary: {words}, {sticky}")
    print(f"native status_word: the known bits matched in {len(blocks)} runs; C1, the exception flags, the stack "
          "fault and the busy bit (mask 0x82ff) took " + ", ".join(f"{w:#06x} {n}" for w, n in sorted(words.items())))
    print("native split_sticky: after an unordered FUCOM the IE byte was " + " and ".join(
        str(v) for v in sorted(sticky[3])) + f" (a signaling NaN sets IE, a quiet NaN does not); {runs} native runs")
    with pefile.PE(str(native_path), fast_load=True) as pe:
        for name, (_, needle) in NATIVE.items():
            try:
                lower_export(pe, name)
            except ValueError as exc:
                if needle not in str(exc) or "FNSTSW AX leaves unknown" not in str(exc):
                    raise
            else:
                raise AssertionError(f"translation accepted {name}")
    print("translation rejects split_sticky at TEST AL, 1 and status_word at the store of AX, which read bits "
          "that FNSTSW AX leaves unknown")


def check_rejections():
    from iced_x86 import Decoder
    from ffx_version_query import lower
    from x87_lower import Checked

    reject = "{{ x87_status = {status}; return 0; }}"  # the channel of tests/cpu/x87_fldcw.py

    def attempt(hex_bytes, calls=None, exits=None, checked=False):
        body = list(Decoder(64, bytes.fromhex(hex_bytes), ip=0x1000))
        try:
            lower(body, [], {}, calls or {}, exits or {}, x87=Checked(reject) if checked else True)
        except ValueError as exc:
            return str(exc)
        return None

    fld = "db 29"  # fld tbyte ptr [rcx]
    two, pop2, clear = f"{fld} {fld}", "dd d8 dd d8", "31 c0 c3"  # fstp st(0) twice; xor eax, eax; ret
    fcom, sw = "d8 d1", "df e0"  # fcom st(1); fnstsw ax
    sahf_jb, test_c0 = "9e 72 00", "f6 c4 01 75 00"  # sahf; jb +0 / test ah, 01h; jne +0

    def status(between, reader=test_c0, before=fcom):
        return attempt(f"{two} {before} {between} {sw} {reader} {pop2} {clear}")

    def split(last):  # test ah, 40h; jne C3; test ah, 01h; jmp out; C3: <last>; jne out; out:
        return attempt(f"{two} {fcom} {sw} f6 c4 40 75 07 f6 c4 01 75 07 eb 05 {last} 75 00 {pop2} {clear}")

    unknown = "which depends on a status-word bit that is not known on every path from the entry"
    ax = "which FNSTSW AX leaves unknown on some path from the entry"
    parity = "reads PF, which the integer lowering models only after SAHF, FCOMI, FCOMIP, FUCOMI, FUCOMIP and an 8-bit TEST or AND"
    rejected = {
        # The C0, C2 and C3 bits that a consumer sees must come from a comparison on every path.
        "FNSTSW AX with no comparison, then SAHF and JB": (status("", sahf_jb, before=""), "reads CF, " + unknown),
        "FADD between FCOM and FNSTSW": (status("dc c1"), "reads ZF, " + unknown),
        "FLD ST(0) and FSTP ST(0) between FCOM and FNSTSW": (status("d9 c0 dd d8"), "reads ZF, " + unknown),
        "FNSTCW between FCOM and FNSTSW": (status("d9 3a"), "reads ZF, " + unknown),
        "FSTSW AX (WAIT and FNSTSW AX) after FCOM": (status("9b"), "reads ZF, " + unknown),
        "a second FNSTSW AX": (status(sw), "reads ZF, " + unknown),
        "a call between FCOMPP and FNSTSW": (attempt(f"{two} de d9 e8 00 00 00 00 {sw} {test_c0} {clear}",
                                                     calls={0x1006: ";"}), "reads ZF, " + unknown),
        "a join where one path skips FCOM": (attempt(f"{two} e3 02 {fcom} {sw} {test_c0} {pop2} {clear}"),
                                             "reads ZF, " + unknown),
        # The maintainer's control: one successor tests C2, the other C1, which the profile does not know.
        "a branch whose other successor tests C1": (split("f6 c4 02"), "reads ZF, " + unknown),
        # Readers of AX other than SAHF and TEST or AND of AH with an immediate.
        "TEST AL, 1 (IE)": (status("", "a8 01 75 00"), "reads AX bits 0x00ff, " + ax),
        "TEST AX, 4500h": (status("", "66 a9 00 45 75 00"), "reads AX bits 0x82ff, " + ax),
        "AND AH, 47h, then a store of AH (C1)": (status("", "80 e4 47 88 22"), "reads AX bits 0x0200, " + ax),
        "a store of AX": (status("", "66 89 02"), "reads AX bits 0x82ff, " + ax),
        "a return with the status word in AX": (attempt(f"{two} {fcom} {sw} {pop2} c3"),
                                                "RAX holds unknown status-word bits 0x82ff at ret"),
        "a return with the status word in AX in the checked entry, after FLDCW": (
            attempt(f"d9 29 {two} {fcom} {sw} {pop2} c3", checked=True), "RAX holds unknown status-word bits 0x82ff at ret"),
        "a jump to an external edge with the status word in AX": (
            attempt(f"{fld} {fcom.replace('d1', 'd0')} {sw} dd d8 e9 00 01 00 00", exits={0x110d: "return rax;"}),
            "RAX holds unknown status-word bits 0x82ff at the external edge"),
        "an AH store after a path skips masking AH": (status("", "e3 03 80 e4 45 88 22"),
                                                      "reads AX bits 0x8200, " + ax),
        "TEST EAX, EAX after writing AL": (status("", "b0 00 85 c0"), "reads AX bits 0x8200, " + ax),
        "a direct call with the status word in AX": (
            attempt(f"{two} {fcom} {sw} {pop2} e8 00 00 00 00 {clear}", calls={0x100c: ";"}),
            "RAX holds unknown status-word bits 0x82ff at call"),
        # PF of the integer lowering.
        "FUCOMI and INC EAX, then JP": (attempt(f"{two} db e9 ff c0 7a 00 {pop2} {clear}"), parity),
        "CMP, then JP": (attempt(f"{two} 39 c0 7a 00 {pop2} {clear}"), parity),
        "TEST EAX, EAX, then JP": (attempt(f"{two} 85 c0 7a 00 {pop2} {clear}"), parity),
        "JP before any flag writer": (attempt(f"{two} 7a 00 {pop2} {clear}"), "reads PF, which no instruction defines"),
        "SAHF in a function without x87": (attempt(f"b4 01 9e 72 00 {clear}"), "unsupported query operation"),
        "LAHF": (attempt(f"{two} {fcom} 9f {pop2} {clear}"), "unsupported query operation"),
        # Forms outside the profile.
        "FNSTSW m16": (attempt(f"{two} {fcom} dd 3a {pop2} {clear}"), "stores status-word bits"),
        "FCOM alias DC D1": (attempt(f"{two} dc d1 {pop2} {clear}"), "unsupported x87 encoding"),
        "FCOMP alias DC D9": (attempt(f"{two} dc d9 dd d8 {clear}"), "unsupported x87 encoding"),
        "FCOMP alias DE D1": (attempt(f"{two} de d1 dd d8 {clear}"), "unsupported x87 encoding"),
        "F2XM1": (attempt(f"{fld} d9 f0 dd d8 {clear}"), "unsupported x87 operation"),
        "FYL2X": (attempt(f"{fld} d9 f1 dd d8 {clear}"), "unsupported x87 operation"),
        "FLDENV m28": (attempt(f"{fld} d9 21 dd d8 {clear}"), "unsupported x87 operation"),
        "FCOMPP with one value": (attempt(f"{fld} de d9 {clear}"), "x87 stack underflow"),
        "FUCOM ST(2) with two values": (attempt(f"{two} dd e2 {pop2} {clear}"), "x87 stack underflow"),
        "FCOM m32 on an empty stack": (attempt(f"d8 11 {clear}"), "x87 stack underflow"),
    }
    for case, (message, needle) in rejected.items():
        if message is None or needle not in message:
            raise AssertionError(f"{case}: expected a rejection containing {needle!r}, got {message!r}")
    print(f"rejected {len(rejected)} sequences outside the profile: " + "; ".join(rejected))
    accepted = {
        "FNSTSW AX after FCOM, then SAHF and JB": status("", sahf_jb),
        "FCOMI between FCOM and FNSTSW": status("db f1"),
        "a branch whose successors test C0 and C2": split("f6 c4 04"),
        "TEST AL, AL, then JP": attempt(f"{two} 84 c0 7a 00 {pop2} {clear}"),
        "AND AH, 45h, then a store of AH": status("", "80 e4 45 88 22"),
        "a return after XOR EAX, EAX": attempt(f"{two} {fcom} {sw} {pop2} {clear}"),
        "SAHF on AH from MOV": attempt(f"{two} b4 41 9e 7a 00 {pop2} {clear}"),
        "FNSTSW AX after FCOM in the checked entry, after FLDCW": attempt(f"d9 29 {two} {fcom} {sw} {sahf_jb} {pop2} {clear}",
                                                                          checked=True),
    }
    for case, message in accepted.items():
        if message is not None:
            raise AssertionError(f"{case}: expected translation, got {message!r}")
    print(f"translated {len(accepted)} sequences that read only known bits: " + "; ".join(accepted))


if __name__ == "__main__":
    mode, first, second, *rest = sys.argv[1:]
    if mode == "asm":
        write_asm(first, second)
    elif mode == "emit":
        emit(Path(first), Path(second), *rest)
    elif mode == "check":
        check(first, second, *rest)
    else:
        raise ValueError(mode)
