"""Compare native and translated FCOMI, FCOMIP, FUCOMI and FUCOMIP of the x87 leaf-function profile.

Usage:
    python tests/cpu/x87_compare.py asm <asm> <def>             regenerate the MASM source and exports
    python tests/cpu/x87_compare.py emit <dll> <cpp> [control]   translate (and optionally mutate) the fixture
    python tests/cpu/x87_compare.py check <harness> <native> <lifted> [controls...]

The exports, harness and guards follow tests/cpu/x87_add.py, and the x87 steps use the format of
tests/cpu/x87_moves.py. After each comparison a record step reads the flags through the integer
lowering's conditional jumps (JE, JNE, JA, JAE, JB and JBE as taken bytes), SETE, RCL (CF rotated into a
zeroed register, last because it changes CF) and its conditional moves (CMOVE, CMOVNE and CMOVBE as
dwords), with only flag-neutral MOVs in between. LOOPE and LOOPNE, which also read ZF, are not used.
add_fcomi_of reads OF through SETO and ADOX after an ADD that sets it and a comparison that clears it.
emit checks that every decoded x87 instruction is the form its sequence names. The model compares with Fraction arithmetic and is independent of
tools/x87_soft.h.
"""
import itertools
from pathlib import Path
import platform
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)
import x87_moves as moves  # noqa: E402
from x87_add import J, classify, from_binary, m80_bytes, to_binary  # noqa: E402
from x87_div import SIZE  # noqa: E402

INPUT, OUTPUT = base.INPUT, base.OUTPUT
COMPARES = ("fcomi", "fcomip", "fucomi", "fucomip")
JUMPS = ("je", "jne", "ja", "jae", "jb", "jbe")
MOVES = ("cmove", "cmovne", "cmovbe")
RECORD = 20  # six taken bytes, the SETE byte, the RCL byte and three dwords

P80 = [("fld", "m80", 0), ("fld", "m80", 16)]
T80 = P80 + [("fld", "m80", 32)]
# Comparison steps are (mnemonic, i) for ST(0), ST(i); ("record", offset) captures the flags.
SEQUENCES = {
    **{f"{op}_st1": ("pair80", P80 + [(op, 1), ("record", 0)]
                     + ([("fstp", "m80", 24)] if op.endswith("p") else [("fstp", "m80", 24), ("fstp", "m80", 40)]))
       for op in COMPARES},
    "fcomi_st2": ("triple80", T80 + [("fcomi", 2), ("record", 0), ("fstp", "m80", 24), ("fstp", "m80", 40),
                                     ("fstp_st", 0)]),
    "fucomip_st2": ("triple80", T80 + [("fucomip", 2), ("record", 0), ("fstp", "m80", 24), ("fstp", "m80", 40)]),
    "fucomi_st0": ("m80", [("fld", "m80", 0), ("fucomi", 0), ("record", 0), ("fstp", "m80", 24)]),
    "fcomip_st0": ("m80", [("fld", "m80", 0), ("fcomip", 0), ("record", 0)]),
    "fcomi_m64": ("pair64", [("fld", "m64", 0), ("fld", "m64", 8), ("fcomi", 1), ("record", 0), ("fstp_st", 0),
                             ("fstp_st", 0)]),
    "fucomi_m32": ("pair32", [("fld", "m32", 0), ("fld", "m32", 4), ("fucomi", 1), ("record", 0), ("fstp_st", 0),
                              ("fstp_st", 0)]),
    # Two comparisons in a row; each record sees only the flags of the comparison before it.
    "two_compares": ("triple80", T80 + [("fucomip", 1), ("record", 0), ("fcomip", 1), ("record", RECORD),
                                        ("fstp", "m80", 40)]),
    # ADD of the input dwords at 32 and 36 sets OF (SETO stores it at 21), and FCOMI clears it: SETO stores
    # OF at 20 and ADOX of a zeroed register at 22, before the record, whose RCL changes OF.
    "add_fcomi_of": ("ofpair", [("add_of", 32, 21)] + P80 + [("fcomi", 1), ("seto", 20), ("adox", 22), ("record", 0),
                                                            ("fstp_st", 0), ("fstp_st", 0)]),
}

CONTROLS = {
    # An unordered comparison reported as equal.
    "unordeq": (re.compile(r"return 3;  // unordered"), "return 1;  // unordered"),
    # The operands exchanged.
    "swap": (re.compile(r"x87_compare\(x87_slot\[(\d)\], x87_slot\[(\d)\]\)"), r"x87_compare(x87_slot[\2], x87_slot[\1])"),
    # -0 ordered below +0.
    "zeroorder": (re.compile(r"if \(za && zb\) return 1;  // signed zeros"),
                  "if (za && zb) return x87_sign(a) == x87_sign(b) ? 1 : (x87_sign(a) ? 0 : 2);  // signed zeros"),
    # The low 11 significand bits ignored, as if the values had been rounded to binary64 first.
    "low11": (re.compile(r"const X87Unpacked ua = x87_unpack\(a\), ub = x87_unpack\(b\);"),
              "X87Unpacked ua = x87_unpack(a), ub = x87_unpack(b); ua.sig &= ~0x7ffull; ub.sig &= ~0x7ffull;"),
    # The raw biased exponent, so a pseudo-denormal sorts below the equal value at exponent 1.
    "rawexp": (re.compile(r"const X87Unpacked ua = x87_unpack\(a\), ub = x87_unpack\(b\);"),
               "const X87Unpacked ua{x87_sign(a), static_cast<int>(x87_exp(a)), a.sig}, "
               "ub{x87_sign(b), static_cast<int>(x87_exp(b)), b.sig};"),
    # Unnormals, pseudo-infinities and pseudo-NaNs compared by their bits instead of unordered.
    "unnormal": (re.compile(r"if \(x87_unsupported\(a\) \|\| x87_unsupported\(b\) \|\| "), "if ("),
    # The comparisons leave OF as it was.
    "keepof": (re.compile(r"(carry = r == 0 \|\| r == 3;) overflow = false;"), r"\1"),
}


# ---- MASM generation and decoding ------------------------------------------------------------------

def asm_lines(step, name):
    op, *args = step
    if op in COMPARES:
        return [f"    {op} st, st({args[0]})"]
    if op == "add_of":
        return [f"    mov eax, dword ptr [rcx+{args[0]}]", f"    add eax, dword ptr [rcx+{args[0] + 4}]",
                "    seto al", f"    mov byte ptr [rdx+{args[1]}], al"]
    if op == "seto":
        return ["    seto al", f"    mov byte ptr [rdx+{args[0]}], al"]
    if op == "adox":
        return ["    mov r9d, 0", "    adox r9d, r9d", f"    mov byte ptr [rdx+{args[0]}], r9b"]
    if op != "record":
        return [moves.asm_line(step)]
    offset = args[0]
    lines = []
    for k, jump in enumerate(JUMPS):
        label = f"{name}_{offset}_{jump}"
        lines += [f"    mov byte ptr [rdx+{offset + k}], 1", f"    {jump} {label}",
                  f"    mov byte ptr [rdx+{offset + k}], 0", f"{label}:"]
    lines.append("    mov r8d, 1")
    for k, move in enumerate(MOVES):
        lines += ["    mov eax, 0", f"    {move} eax, r8d", f"    mov dword ptr [rdx+{offset + 8 + 4 * k}], eax"]
    lines += ["    mov eax, 0", "    sete al", f"    mov byte ptr [rdx+{offset + 6}], al",
              "    mov eax, 0", "    rcl eax, 1", f"    mov byte ptr [rdx+{offset + 7}], al"]
    return lines


def expected_forms(step):
    op, *args = step
    if op in COMPARES:
        return [(op, f"st0,st{args[0]}")]
    return [] if op in ("record", "add_of", "seto", "adox") else [moves.expected_form(step)]


def write_asm(asm_path, def_path):
    lines = ["; coverage: source cpu_aot_x87_compare",
             "; Generated by tests/cpu/x87_compare.py asm; edit SEQUENCES there and regenerate.",
             "; Source-owned differential fixture for the comparisons into EFLAGS of the x87 leaf-function",
             "; profile. Each export reads an input block at RCX, writes an output block at RDX and returns",
             "; with the x87 stack empty.", ".code", ""]
    for name, (_, steps) in SEQUENCES.items():
        lines += [f"{name} PROC FRAME", "    .endprolog", *(line for s in steps for line in asm_lines(s, name)),
                  "    ret", f"{name} ENDP", ""]
    lines.append("END")
    Path(asm_path).write_text("\n".join(lines) + "\n", encoding="ascii")
    Path(def_path).write_text("EXPORTS\n" + "".join(f"    {name}\n" for name in SEQUENCES), encoding="ascii")


def emit(image, output, control="lifted"):
    import pefile
    from cpu_leaf_aot import export_rva
    from cpu_ranges import function_ranges
    from ffx_version_query import INTEGER_HEADER, lower, region
    from x87_lower import X87_HEADER, is_x87
    from adx import HOST_HAS_ADX

    parts = ["#include <intrin.h>\n", INTEGER_HEADER, X87_HEADER, HOST_HAS_ADX]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, (_, steps) in SEQUENCES.items():
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            decoded = [moves.decoded_form(i) for i in body if is_x87(i)]
            wanted = [form for s in steps for form in expected_forms(s)]
            if decoded != wanted:
                raise AssertionError(f"{name} assembled as {decoded}, expected {wanted}")
            code = lower(body, ["rcx = a; rdx = b;"], {}, {}, {}, x87=True)
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b) {{\n{code}\n}}\n')
    text = "".join(parts)
    if control != "lifted":
        expression, replacement = CONTROLS[control]
        text, count = expression.subn(replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
        print(f"control {control}: {count} replacements")
    output.write_text(text, encoding="ascii")


# ---- Independent model -----------------------------------------------------------------------------

def compare(a, b):
    """0 if a < b, 1 if equal, 2 if a > b, 3 if unordered (a NaN or an unsupported encoding)."""
    values = []
    for c in (classify(a), classify(b)):
        if c[0] in ("unsupported", "nan"):
            return 3
        magnitude = 0 if c[0] == "zero" else float("inf") if c[0] == "inf" else c[2]
        values.append(-magnitude if c[1] else magnitude)
    return 0 if values[0] < values[1] else 1 if values[0] == values[1] else 2


def record_bytes(result):
    zf, cf = result in (1, 3), result in (0, 3)
    taken = {"je": zf, "jne": not zf, "ja": not cf and not zf, "jae": not cf, "jb": cf, "jbe": cf or zf}
    moved = {"cmove": zf, "cmovne": not zf, "cmovbe": cf or zf}
    return (bytes(int(taken[j]) for j in JUMPS) + bytes((int(zf), int(cf)))  # SETE, then RCL of CF
            + b"".join(int(moved[m]).to_bytes(4, "little") for m in MOVES))


def reference(name, data):
    slots, out = [], bytearray(b"\xcc" * OUTPUT)
    result = None

    def read(kind, offset):
        return int.from_bytes(data[offset:offset + SIZE[kind]], "little")

    for op, *args in SEQUENCES[name][1]:
        if op == "fld":
            raw = read(*args)
            slots.append((raw & (2 ** 64 - 1), raw >> 64) if args[0] == "m80" else from_binary(args[0], raw))
        elif op == "fstp":
            kind, offset = args
            v = slots.pop()
            encoded = v[0] | v[1] << 64 if kind == "m80" else to_binary(kind, v)
            out[offset:offset + SIZE[kind]] = encoded.to_bytes(SIZE[kind], "little")
        elif op == "fstp_st":
            slots[-1 - args[0]] = slots[-1]
            slots.pop()
        elif op in COMPARES:
            result = compare(slots[-1], slots[-1 - args[0]])
            if op.endswith("p"):
                slots.pop()
        elif op == "add_of":  # OF of the signed 32-bit sum
            a, b = (int.from_bytes(data[o:o + 4], "little", signed=True) for o in (args[0], args[0] + 4))
            out[args[1]] = int(not -2 ** 31 <= a + b < 2 ** 31)
        elif op in ("seto", "adox"):  # OF after the comparison, which clears it
            out[args[0]] = 0
        else:  # ("record", offset)
            out[args[0]:args[0] + RECORD] = record_bytes(result)
    assert not slots
    return bytes(out)


# ---- Inputs ----------------------------------------------------------------------------------------

def cases():
    rng = random.Random(41)
    specials = base.special_m80()

    def random_m80():
        return J | rng.getrandbits(63), rng.randint(0x3FFF - 70, 0x3FFF + 70) | rng.choice((0, 0x8000))

    near = []  # equal values, neighbours and equal values in other encodings
    for _ in range(150):
        v = random_m80()
        near += [(v, v), (v, (v[0] ^ 1, v[1])), (v, (v[0] ^ 1 << 11, v[1])), (v, (v[0], v[1] ^ 0x8000)),
                 (v, (v[0], v[1] + 1))]
    near += [((J | x, 0), (J | x, 1)) for x in (0, 1, 0x7FF, 1 << 62)]  # pseudo-denormal and normal twins
    near += [((J | x, 0x8000), (J | x, 0x8001)) for x in (0, 5)]
    near += [((x, 0), (x << 1, 0)) for x in (1, 0x1234)]                 # denormals one bit apart
    pairs = [(a, b) for a in specials for b in specials] + near + [(random_m80(), random_m80()) for _ in range(300)]
    f64, f32 = base.f64_bits(), base.f32_bits()
    blocks = {
        "m80": [m80_bytes(v) for v in base.m80_singles()],
        "pair80": [m80_bytes(a) + m80_bytes(b) for a, b in pairs],
        "triple80": [b"".join(m80_bytes(v) for v in rng.sample(specials, 3)) for _ in range(300)]
                    + [b"".join(m80_bytes(v) for v in (a, b, a)) for a, b in near[:300]],
        "pair64": [a.to_bytes(8, "little") + b.to_bytes(8, "little") for a in f64[:18] for b in f64[:18]]
                  + [a.to_bytes(8, "little") + b.to_bytes(8, "little") for a, b in zip(f64, reversed(f64))],
        "pair32": [a.to_bytes(4, "little") + b.to_bytes(4, "little") for a in f32[:17] for b in f32[:17]],
        # A comparison pair, then two dwords whose sum overflows in most cases.
        "ofpair": [(m80_bytes(a) + m80_bytes(b)).ljust(32, b"\0") + x.to_bytes(4, "little") + y.to_bytes(4, "little")
                   for (a, b), (x, y) in zip(pairs[::7], itertools.cycle(
                       [(0x7FFFFFFF, 1), (0x80000000, 0x80000000), (0x40000000, 0x40000000), (1, 2),
                        (0xFFFFFFFF, 1), (0x80000000, 0xFFFFFFFF)]))],
    }
    for name, (kind, _) in SEQUENCES.items():
        for block in blocks[kind]:
            yield name, block.ljust(INPUT, b"\0")


# ---- Check -----------------------------------------------------------------------------------------

def check(harness_path, native_path, lifted_path, *control_paths):
    import ctypes
    # The native ADOX in add_fcomi_of faults without ADX; the translation does not need it.
    host_has_adx = ctypes.CDLL(str(Path(lifted_path).resolve())).host_has_adx
    host_has_adx.argtypes, host_has_adx.restype = [], ctypes.c_int
    adx_only = set() if host_has_adx() else {
        name for name, (_, steps) in SEQUENCES.items() if any(step[0] == "adox" for step in steps)}
    expected = [(name, data, reference(name, data)) for name, data in cases()]
    outcomes = {}
    for name, data, want in expected:
        outcomes[want[0:6]] = outcomes.get(want[0:6], 0) + 1
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        run = base.Runner(harness_path, path)
        bad = 0
        for name, data, want in expected:
            if label == "native" and name in adx_only:
                continue
            got, state = run(name, data)
            if not base.entry_state_ok(state):
                raise AssertionError(f"{label} {name} left state {state:#014x}; the profile requires control "
                                     "0x027f and an empty stack at TOP 0")
            if got != want:
                if not bad:
                    print(f"{label} first mismatch: {name}(input {data[:48].hex()})\n  got      {got[:48].hex()}\n"
                          f"  expected {want[:48].hex()}")
                bad += 1
        if bad:
            raise AssertionError(f"{label}: {bad}/{len(expected)} mismatches")
    print(f"{len(expected)} cases over {len(SEQUENCES)} exports (output block and guards) matched the reference "
          "in native and translated DLLs; each export returned with control 0x027f and an empty stack")
    names = {record_bytes(r)[0:6]: name for r, name in enumerate(("less", "equal", "greater", "unordered"))}
    print("first-record outcomes: " + ", ".join(f"{names[key]} {outcomes.get(key, 0)}" for key in names))
    of_cases = [want[21] for name, data, want in expected if name == "add_fcomi_of"]
    print(f"add_fcomi_of: ADD set OF before the comparison in {sum(of_cases)} of {len(of_cases)} cases")
    print(f"processor: {platform.processor()}")
    first = {}
    for name, data, want in expected:
        first.setdefault(name, (data, want))
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        run = base.Runner(harness_path, path)
        failed = [name for name, (data, want) in first.items()
                  if not (label == "native" and name in adx_only)
                  and not base.preserved(run, "x87_call", name, data, want)]
        if failed:
            raise AssertionError(f"x87_call lost a live caller value or control word around {label} {failed}")
    print(f"x87_call kept a live caller ST(0) value and control word around all {len(first)} native and "
          "translated exports")
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        run = base.Runner(harness_path, path)
        bad = sum(1 for name, data, want in expected if run(name, data)[0] != want)
        print(f"control {control}: {bad}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")
    check_rejections()
    if adx_only:
        print(f"Host CPU lacks ADX (CPUID leaf 7 EBX bit 19): native {', '.join(sorted(adx_only))} not run; "
              "every other check passed. Skipping.")
        sys.exit(77)


def check_rejections():
    from iced_x86 import Decoder
    from ffx_version_query import lower

    def attempt(hex_bytes):
        body = list(Decoder(64, bytes.fromhex(hex_bytes), ip=0x1000))
        try:
            lower(body, [], {}, {}, {}, x87=True)
        except ValueError as exc:
            return str(exc)
        return None

    fld = "db 29"  # fld tbyte ptr [rcx]
    pop2 = "dd d8 dd d8 c3"  # fstp st(0) twice; ret

    def after_fucomi(consumer):
        return attempt(f"{fld} {fld} db e9 {consumer} {pop2}")

    rejected = {
        # Readers of PF, SF or the whole flags register (the comparison clears OF, which SETO and ADOX read).
        # JP, JNP, SETP and CMOVP are lowered only when FUCOMI, SAHF or an 8-bit TEST or AND wrote PF last, and
        # INC writes it after FUCOMI.
        **{f"fucomi and inc eax, then {name}": (after_fucomi(f"ff c0 {code}"),
                                                "reads PF, which the integer lowering models only after")
           for name, code in (("jp", "7a 00"), ("jnp", "7b 00"), ("setp", "0f 9a c0"), ("cmovp", "0f 4a c1"))},
        **{f"fucomi then {name}": (after_fucomi(code), "unsupported query operation")
           for name, code in (("js", "78 00"), ("jl", "7c 00"), ("lahf", "9f"), ("cmovs", "0f 48 c1"),
                              ("pushfq", "9c"))},
        "fcomi st, st(1) with one value": (attempt(f"{fld} db f1 dd d8 c3"), "x87 stack underflow"),
        "fucomip st, st(2) with two values": (attempt(f"{fld} {fld} df ea dd d8 c3"), "x87 stack underflow"),
        "fcomip on an empty stack": (attempt("df f0 c3"), "x87 stack underflow"),
        # Comparisons into the status word that stay outside the profile.
        "fprem1": (attempt(f"{fld} d9 f5 dd d8 c3"), "unsupported x87 operation"),
        "fcom st(1) as DC D1": (attempt(f"{fld} {fld} dc d1 {pop2}"), "unsupported x87 encoding"),
        "fucompp with one value": (attempt(f"{fld} da e9 c3"), "x87 stack underflow"),
        "f2xm1": (attempt(f"{fld} d9 f0 dd d8 c3"), "unsupported x87 operation"),
        "fyl2x": (attempt(f"{fld} d9 f1 dd d8 c3"), "unsupported x87 operation"),
        "frstor m108": (attempt(f"{fld} dd 21 dd d8 c3"), "unsupported x87 operation"),
        "fnstsw m16": (attempt(f"{fld} dd 3a dd d8 c3"), "stores status-word bits"),
    }
    for case, (message, needle) in rejected.items():
        if message is None or needle not in message:
            raise AssertionError(f"{case}: expected a rejection containing {needle!r}, got {message!r}")
    print(f"rejected {len(rejected)} sequences outside the profile: " + "; ".join(rejected))


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
