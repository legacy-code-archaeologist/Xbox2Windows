"""Compare native and translated FSIN, FCOS, FSINCOS, FPTAN and FPATAN of the x87 leaf-function profile.

Usage:
    python tests/cpu/x87_trig.py asm <asm> <def>             regenerate the MASM source and exports
    python tests/cpu/x87_trig.py emit <dll> <cpp> [control]   translate (and optionally mutate) the fixture
    python tests/cpu/x87_trig.py check <harness> <native> <lifted> [controls...]

The exports, harness and guards follow tests/cpu/x87_add.py, and the other steps use the format of
tests/cpu/x87_moves.py. emit checks that every decoded x87 instruction is the form its sequence names.
The independent model uses 150-digit Decimal arithmetic and shares no code with tools/x87_soft.h.
"""
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
import ctypes
from pathlib import Path
import platform
import random
import re
import sys

getcontext().prec = 150

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)
import x87_moves as moves  # noqa: E402
from x87_add import INDEFINITE, J, Q, classify, from_binary, m80_bytes, to_binary  # noqa: E402
from x87_div import SIZE  # noqa: E402

INPUT, OUTPUT = base.INPUT, base.OUTPUT
PI = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128")
TWO_PI = 2 * PI
SQRT3 = Decimal(3).sqrt()
ATAN_THRESH = Decimal(2) - SQRT3

# name: (input kind, steps).
SEQUENCES = {
    "sin_m80": ("m80", [("fld", "m80", 0), ("fsin",), ("sw_c2", 32), ("fstp", "m80", 0)]),
    "sin_to_m64": ("m80", [("fld", "m80", 0), ("fsin",), ("fstp", "m64", 0)]),
    "sin_to_m32": ("m80", [("fld", "m80", 0), ("fsin",), ("fstp", "m32", 0)]),
    "cos_m80": ("m80", [("fld", "m80", 0), ("fcos",), ("sw_c2", 32), ("fstp", "m80", 0)]),
    "cos_to_m64": ("m80", [("fld", "m80", 0), ("fcos",), ("fstp", "m64", 0)]),
    "cos_to_m32": ("m80", [("fld", "m80", 0), ("fcos",), ("fstp", "m32", 0)]),
    "sin_c2_readers": ("m80", [("fld", "m80", 0), ("fsin",), ("c2_readers", 32), ("fstp", "m80", 0)]),
    # FSINCOS replaces ST(0) with sin and pushes cos: ST(0)=cos -> [rdx+16], ST(1)=sin -> [rdx+0].
    "sincos_m80": ("m80_in", [("fld", "m80", 0), ("fsincos",), ("sw_c2", 32),
                              ("fstp", "m80", 16), ("fstp", "m80", 0)]),
    "sincos_sum": ("m80_in", [("fld", "m80", 0), ("fsincos",), ("fmul_st0",), ("fxch", 1),
                              ("fmul_st0",), ("faddp", "pop", 1), ("fstp", "m80", 0)]),
    "sincos_depth7": ("m64x7_in", [("fld", "m64", 8 * k) for k in range(7)] + [("fsincos",)]
                      + [("fstp", "m64", 8 * k) for k in range(8)]),
    # FPTAN replaces ST(0) with tan and pushes 1.0: ST(0)=1.0 -> [rdx+16], ST(1)=tan -> [rdx+0].
    "ptan_m80": ("m80_in", [("fld", "m80", 0), ("fptan",), ("sw_c2", 32),
                            ("fstp", "m80", 16), ("fstp", "m80", 0)]),
    "ptan_c2_reset": ("m80_in", [("fldz",), ("fldz",), ("fdivp", "pop", 1), ("ftst",), ("pop",),
                                 ("fld", "m80", 0), ("fptan",), ("sw_c2", 32),
                                 ("fstp", "m80", 16), ("fstp", "m80", 0)]),
    "ptan_to_m64": ("m80_in", [("fld", "m80", 0), ("fptan",), ("pop",), ("fstp", "m64", 0)]),
    "ptan_depth7": ("m64x7_in", [("fld", "m64", 8 * k) for k in range(7)] + [("fptan",)]
                    + [("fstp", "m64", 8 * k) for k in range(8)]),
    # FPATAN computes atan2(ST(1), ST(0)) into ST(1) and pops: load y at [rcx+16], x at [rcx+0].
    "patan_m80": ("pair80", [("fld", "m80", 16), ("fld", "m80", 0), ("fpatan",), ("fstp", "m80", 0)]),
    "patan_to_m64": ("pair80", [("fld", "m80", 16), ("fld", "m80", 0), ("fpatan",), ("fstp", "m64", 0)]),
    "patan_to_m32": ("pair80", [("fld", "m80", 16), ("fld", "m80", 0), ("fpatan",), ("fstp", "m32", 0)]),
    "patan_depth8": ("m64x8", [("fld", "m64", 8 * k) for k in range(8)] + [("fpatan",)]
                     + [("fstp", "m64", 8 * k) for k in range(7)]),
}

CONTROLS = {
    # C2 is never set for |x| >= 2^63.
    "noc2": (re.compile(r"x87_sw = static_cast<uint16_t>\(c2 \? 0x0400 : 0\);"), "x87_sw = 0;"),
    # The out-of-range threshold uses 2^62 instead of 2^63.
    "c262": (re.compile(r"if \(u\.e >= 63\) \{([^\n]*// trig range)"), r"if (u.e >= 62) {\1"),
    # FSINCOS writes cos into ST(1) and sin into ST(0).
    "sincosswap": (re.compile(r"x87_slot\[(\d+)\] = x87_sincos\(([^;]+)\); if \(c2\) x87_trap\(\); x87_slot\[(\d+)\] = c;"),
                   r"x87_slot[\3] = x87_sincos(\2); if (c2) x87_trap(); x87_slot[\1] = c;"),
    # FPTAN pushes 0.0 instead of 1.0.
    "ptanno1": (re.compile(r"one_out = \{x87_integer_bit, 0x3fff\};  // ptan one"),
                "one_out = {0, 0};  // ptan one"),
    # FPATAN computes atan2(x, y) instead of atan2(y, x).
    "patanswap": (re.compile(r"x87_patan\(x87_slot\[(\d+)\], x87_slot\[(\d+)\], x87_cw\)"),
                  r"x87_patan(x87_slot[\2], x87_slot[\1], x87_cw)"),
    # sin(-0) returns +0.
    "sinnegzero": (re.compile(r"if \(sin_out\) \*sin_out = a;  // trig zero"),
                   "if (sin_out) *sin_out = {0, 0};  // trig zero"),
    # Signaling NaN is returned unquieted.
    "snan": (re.compile(r"x87_quiet\(a\)"), "a"),
    # Transcendental results round to 53 bits instead of 64.
    "pc53": (re.compile(r"x87_arith64, x87_rounding"), "x87_arith, x87_rounding"),
}


# ---- MASM generation and decoding ------------------------------------------------------------------

def asm_lines(step, name):
    op, *args = step
    if op in ("fsin", "fcos", "fsincos", "fptan", "fpatan", "fldz", "ftst"):
        return [f"    {op}"]
    if op == "fmul_st0":
        return ["    fmul st(0), st(0)"]
    if op == "pop":
        return ["    fstp st(0)"]
    if op == "sw_c2":
        return ["    fnstsw ax", "    and ah, 04h", f"    mov byte ptr [rdx+{args[0]}], ah"]
    if op == "c2_readers":
        o = args[0]
        return ["    fnstsw ax", "    sahf",
                f"    mov byte ptr [rdx+{o}], 1", f"    jp {name}_jp", f"    mov byte ptr [rdx+{o}], 0",
                f"{name}_jp:",
                "    test ah, 04h",
                f"    mov byte ptr [rdx+{o + 1}], 1", f"    jne {name}_jne", f"    mov byte ptr [rdx+{o + 1}], 0",
                f"{name}_jne:"]
    return [moves.asm_line(step)]


def expected_forms(step):
    op, *args = step
    if op in ("fsin", "fcos", "fsincos", "fptan", "fpatan", "fldz", "ftst"):
        return [(op, "")]
    if op == "fmul_st0":
        return [("fmul", "st0,st0")]
    if op == "pop":
        return [("fstp", "st0")]
    if op in ("sw_c2", "c2_readers"):
        return [("fnstsw", "ax")]
    return [moves.expected_form(step)]


def decoded_form(i):
    from iced_x86 import Mnemonic as M, Register as R

    if i.mnemonic == M.FNSTSW and i.op0_register == R.AX:
        return "fnstsw", "ax"
    return moves.decoded_form(i)


def write_asm(asm_path, def_path):
    lines = ["; coverage: source cpu_aot_x87_trig",
             "; Generated by tests/cpu/x87_trig.py asm; edit SEQUENCES there and regenerate.",
             "; Source-owned differential fixture for FSIN, FCOS, FSINCOS, FPTAN and FPATAN in the x87",
             "; leaf-function profile. Each export reads an input block at RCX, writes an output block at",
             "; RDX and returns with the x87 stack empty.", ".code", ""]
    for name, (_, steps) in SEQUENCES.items():
        body = [line for s in steps for line in asm_lines(s, name)]
        lines += [f"{name} PROC FRAME", "    .endprolog", *body, "    mov eax, 0", "    ret", f"{name} ENDP", ""]
    lines.append("END")
    Path(asm_path).write_text("\n".join(lines) + "\n", encoding="ascii")
    Path(def_path).write_text("EXPORTS\n" + "".join(f"    {name}\n" for name in SEQUENCES), encoding="ascii")


def emit(image, output, control="lifted"):
    import pefile
    from cpu_leaf_aot import export_rva
    from cpu_ranges import function_ranges
    from ffx_version_query import INTEGER_HEADER, lower, region
    from x87_lower import X87_HEADER, is_x87

    parts = [INTEGER_HEADER, X87_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, (_, steps) in SEQUENCES.items():
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            decoded = [decoded_form(i) for i in body if is_x87(i)]
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

def m80_to_dec(v):
    c = classify(v)
    assert c[0] in ("zero", "finite"), c
    if c[0] == "zero":
        return Decimal(0)
    sign = -1 if c[1] else 1
    q = c[2]
    return sign * Decimal(q.numerator) / Decimal(q.denominator)


def dec_to_m80(sign, val):
    """Round exact nonnegative Decimal val to 64-bit extended m80 (nearest-even)."""
    if val == 0:
        return 0, sign << 15
    # Estimate binary exponent via Decimal.adjusted() * log2(10).
    e = int(val.adjusted() * 3.321928094887362)
    p2 = Decimal(2)
    while p2 ** (e + 1) <= val:
        e += 1
    while p2 ** e > val:
        e -= 1
    if e > 16383:
        return J, (sign << 15) | 0x7FFF
    if e < -16382:
        shift = -16382 - 63 - e
        if shift > 64:
            return 0, sign << 15
        scaled = val / (p2 ** (-16382 - 63))
        sig = int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))
        if sig >= J:
            return J, (sign << 15) | 1
        return sig, sign << 15
    scaled = val * (p2 ** (63 - e))
    sig = int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))
    if sig >= (1 << 64):
        sig >>= 1
        e += 1
        if e > 16383:
            return J, (sign << 15) | 0x7FFF
    return sig, (sign << 15) | (e + 16383)


def dec_sincos(x):
    """High-precision (sin(x), cos(x)) for Decimal x."""
    k = (x / TWO_PI).to_integral_value()
    r = x - k * TWO_PI
    r2 = r * r
    term = r
    s = r
    n = 1
    while abs(term) > Decimal("1e-115"):
        term = -term * r2 / ((2 * n) * (2 * n + 1))
        s += term
        n += 1
    term = Decimal(1)
    c = Decimal(1)
    n = 1
    while abs(term) > Decimal("1e-115"):
        term = -term * r2 / ((2 * n - 1) * (2 * n))
        c += term
        n += 1
    return s, c


def dec_atan_unit(z):
    if z > ATAN_THRESH:
        return PI / 6 + dec_atan_unit((SQRT3 * z - 1) / (SQRT3 + z))
    term = z
    a = z
    z2 = z * z
    n = 1
    while abs(term) > Decimal("1e-115"):
        term = -term * z2
        a += term / (2 * n + 1)
        n += 1
    return a


def dec_atan2(y, x):
    ay, ax = abs(y), abs(x)
    a = dec_atan_unit(ay / ax) if ay <= ax else (PI / 2 - dec_atan_unit(ax / ay))
    if x < 0:
        a = PI - a
    return -a if y < 0 else a


def trig_op(op, v):
    """Model FSIN/FCOS/FSINCOS/FPTAN: returns (c2, result_list) where result_list has 1 or 2 m80 pairs."""
    c = classify(v)
    if c[0] in ("unsupported", "inf"):
        return 0, [INDEFINITE] if op in ("fsin", "fcos") else [INDEFINITE, INDEFINITE]
    if c[0] == "nan":
        q = (v[0] | Q, v[1])
        return 0, [q] if op in ("fsin", "fcos") else [q, q]
    if c[0] == "zero":
        one = (J, 0x3FFF)
        if op == "fsin":
            return 0, [v]
        if op == "fcos":
            return 0, [one]
        if op == "fsincos":
            return 0, [v, one]
        return 0, [v, one]
    # Finite nonzero
    x = m80_to_dec(v)
    if abs(x) >= Decimal(2) ** 63:
        if op in ("fsin", "fcos"):
            return 1, [v]
        return 1, [v, INDEFINITE]
    if abs(x) < Decimal(2) ** -64:
        # Re-encode pseudo-denormal at biased exp 1, keep denormal/normal unchanged.
        re_v = (v[0], (v[1] & 0x8000) | 1) if (v[1] & 0x7FFF) == 0 and (v[0] & J) else v
        one = (J, 0x3FFF)
        if op == "fsin":
            return 0, [re_v]
        if op == "fcos":
            return 0, [one]
        return 0, [re_v, one]
    s, co = dec_sincos(x)
    if op == "fsin":
        return 0, [dec_to_m80(int(s < 0), abs(s))]
    if op == "fcos":
        return 0, [dec_to_m80(int(co < 0), abs(co))]
    if op == "fsincos":
        return 0, [dec_to_m80(int(s < 0), abs(s)), dec_to_m80(int(co < 0), abs(co))]
    t = s / co
    return 0, [dec_to_m80(int(t < 0), abs(t)), (J, 0x3FFF)]


def patan(y, x):
    """Model FPATAN = atan2(y, x)."""
    cy, cx = classify(y), classify(x)
    if "unsupported" in (cy[0], cx[0]) or "nan" in (cy[0], cx[0]):
        return base.add(y, x)
    sy, sx = y[1] >> 15, x[1] >> 15
    if cy[0] == "zero":
        return dec_to_m80(sy, PI) if sx else (0, sy << 15)
    if cy[0] == "inf":
        if cx[0] == "inf":
            return dec_to_m80(sy, 3 * PI / 4 if sx else PI / 4)
        return dec_to_m80(sy, PI / 2)
    if cx[0] == "inf":
        return dec_to_m80(sy, PI) if sx else (0, sy << 15)
    if cx[0] == "zero":
        return dec_to_m80(sy, PI / 2)
    val = dec_atan2(m80_to_dec(y), m80_to_dec(x))
    return dec_to_m80(sy, abs(val))


def reference(name, data):
    """Returns (output_bytes, c2, exact_slots_mask) for output comparison."""
    slots, out = [], bytearray(b"\xcc" * OUTPUT)
    c2 = 0
    inexact = []  # track whether each output offset may differ by ULP on native hardware

    def read(kind, offset):
        return int.from_bytes(data[offset:offset + SIZE[kind]], "little")

    for op, *args in SEQUENCES[name][1]:
        if op in ("fsin", "fcos", "fsincos", "fptan"):
            v = slots.pop()
            c2, res = trig_op(op, v)
            c = classify(v)
            is_finite_inrange = c[0] == "finite" and abs(m80_to_dec(v)) < Decimal(2) ** 63
            slots.extend(res)
            # Mark transcendental slot(s) as inexact if finite in-range (except pushed 1.0 of fptan).
            if op in ("fsin", "fcos"):
                inexact.append(is_finite_inrange)
            elif op == "fsincos":
                inexact.extend([is_finite_inrange, is_finite_inrange])
            else:
                inexact.extend([is_finite_inrange, False])
        elif op == "fpatan":
            x = slots.pop()
            y = slots.pop()
            cy, cx = classify(y), classify(x)
            is_fin = cy[0] == "finite" and cx[0] == "finite"
            slots.append(patan(y, x))
            inexact = inexact[:-2] + [is_fin]
        elif op == "fldz":
            slots.append((0, 0))
            inexact.append(False)
        elif op == "fdivp":
            b = slots.pop()
            a = slots.pop()
            slots.append(_div(a, b))
            inexact = inexact[:-2] + [False]
        elif op == "ftst":
            c2 = 1 if classify(slots[-1])[0] in ("nan", "unsupported") else 0
        elif op == "fld":
            raw = read(*args)
            slots.append((raw & (2 ** 64 - 1), raw >> 64) if args[0] == "m80" else from_binary(args[0], raw))
            inexact.append(False)
        elif op == "fstp":
            kind, offset = args
            v = slots.pop()
            inexact.pop()
            encoded = v[0] | v[1] << 64 if kind == "m80" else to_binary(kind, v)
            out[offset:offset + SIZE[kind]] = encoded.to_bytes(SIZE[kind], "little")
        elif op == "pop":
            slots.pop()
            inexact.pop()
        elif op == "fxch":
            k = -1 - args[0]
            slots[-1], slots[k] = slots[k], slots[-1]
            inexact[-1], inexact[k] = inexact[k], inexact[-1]
        elif op == "fmul_st0":
            slots[-1] = base.mul(slots[-1], slots[-1]) if hasattr(base, "mul") else _mul(slots[-1], slots[-1])
        elif op == "faddp":
            slots[-2] = base.add(slots[-2], slots[-1])
            slots.pop()
            inexact[-2] = inexact[-2] or inexact[-1]
            inexact.pop()
        elif op == "sw_c2":
            out[args[0]] = 4 if c2 else 0
        elif op == "c2_readers":
            out[args[0]] = 1 if c2 else 0
            out[args[0] + 1] = 1 if c2 else 0
    assert not slots
    return bytes(out)


def _mul(a, b):
    import x87_mul
    return x87_mul.mul(a, b)


def _div(a, b):
    import x87_div
    return x87_div.div(a, b)


# ---- Inputs ----------------------------------------------------------------------------------------

def quadrant_values():
    """Values around quadrant boundaries k * pi / 2 and fractions."""
    vals = []
    for num, den in [(1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (1, 1), (5, 4), (3, 2), (7, 4), (2, 1), (5, 2), (3, 1)]:
        for sign in (0, 1):
            base_v = dec_to_m80(sign, Decimal(num) * PI / Decimal(den))
            vals.append(base_v)
            # Neighbors +-1 ULP around the boundary
            vals.append((base_v[0] - 1, base_v[1]))
            vals.append((base_v[0] + 1 if base_v[0] < (1 << 64) - 1 else base_v[0], base_v[1]))
    for frac in ["0.1", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "7.5"]:
        d = Decimal(frac)
        vals.append(dec_to_m80(0, d))
        vals.append(dec_to_m80(1, d))
    return vals


def cases():
    rng = random.Random(37)
    singles = list(base.m80_singles()) + quadrant_values()
    # Add out-of-range boundaries: 2^63 - 1, 2^63, -2^63, 2^64, -2^64, 2^100, and tiny values.
    singles += [
        (0xFFFFFFFFFFFFFFFF, 16383 + 62),
        (0xFFFFFFFFFFFFFFFF, (16383 + 62) | 0x8000),
        (J, 16383 + 63),
        (J, (16383 + 63) | 0x8000),
        (J, 16383 + 64),
        (J, (16383 + 64) | 0x8000),
        (J, 16383 + 100),
        (J, 16383 - 34),
        (J, 16383 - 65),
        (J, (16383 - 65) | 0x8000),
        (1, 0),
        (1, 0x8000),
        (J, 0),
    ]
    # Small/moderate random values (|x| <= 8) for tight native comparison, plus large |x| < 2^63.
    for _ in range(150):
        singles.append((J | rng.getrandbits(63), rng.randint(16383 - 30, 16383 + 2) | rng.choice((0, 0x8000))))
    for _ in range(80):
        singles.append((J | rng.getrandbits(63), rng.randint(16383 + 3, 16383 + 62) | rng.choice((0, 0x8000))))
    for _ in range(20):
        singles.append((J | 1 + rng.getrandbits(61), 0x7FFF | rng.choice((0, 0x8000))))

    def in_range(v):
        c = classify(v)
        return c[0] != "finite" or abs(m80_to_dec(v)) < Decimal(2) ** 63

    singles_in = [v for v in singles if in_range(v)]
    f64_in = [b for b in base.f64_bits() if in_range(from_binary("m64", b))]
    pairs = [(a, b) for a in base.m80_singles() for b in base.m80_singles()[::3]]
    pairs += [(a, b) for a, b in zip(singles, reversed(singles))]
    # Huge/tiny ratio pairs for FPATAN
    pairs += [((1, 0), (J, 0x7FFE)), ((J, 0x7FFE), (1, 0)), ((J, 0x3FFF), (J, 0x3FFF))]

    blocks = {
        "m80": [m80_bytes(v) for v in singles],
        "m80_in": [m80_bytes(v) for v in singles_in],
        "pair80": [m80_bytes(x) + m80_bytes(y) for x, y in pairs],
        "m64x7_in": [b"".join(rng.choice(f64_in).to_bytes(8, "little") for _ in range(7)) for _ in range(150)],
        "m64x8": [b"".join(rng.choice(base.f64_bits()).to_bytes(8, "little") for _ in range(8)) for _ in range(150)],
    }
    for name, (kind, _) in SEQUENCES.items():
        for block in blocks[kind]:
            yield name, block.ljust(INPUT, b"\0")


# ---- Comparison helpers ----------------------------------------------------------------------------

def unpack_m80(buf):
    return int.from_bytes(buf[:8], "little"), int.from_bytes(buf[8:10], "little")


def m80_ulp_diff(got_v, want_v):
    cg, cw = classify(got_v), classify(want_v)
    if cg[0] != cw[0]:
        return float("inf")
    if cg[0] in ("unsupported", "nan", "inf", "zero"):
        return 0.0 if got_v == want_v else float("inf")
    dg, dw = m80_to_dec(got_v), m80_to_dec(want_v)
    e = (want_v[1] & 0x7FFF) - 16383
    if (want_v[1] & 0x7FFF) == 0:
        e = -16382
    return float(abs(dg - dw) / (Decimal(2) ** (e - 63)))


def binary_ulp_diff(kind, got_bytes, want_bytes):
    g_raw = int.from_bytes(got_bytes, "little")
    w_raw = int.from_bytes(want_bytes, "little")
    if g_raw == w_raw:
        return 0
    g = from_binary(kind, g_raw)
    w = from_binary(kind, w_raw)
    if classify(g)[0] != "finite" or classify(w)[0] != "finite":
        return float("inf")
    bits = 32 if kind == "m32" else 64
    # Signed magnitude to ordered int for ULP distance
    def ordered(raw):
        sign = 1 << (bits - 1)
        return -(raw & (sign - 1)) if (raw & sign) else raw
    return abs(ordered(g_raw) - ordered(w_raw))


def native_matches(name, data, got, want):
    """Compare native hardware output with the model, allowing documented x87 transcendental tolerance."""
    if got == want:
        return True
    kind, steps = SEQUENCES[name]
    if kind == "pair80" or name == "patan_depth8":
        # FPATAN matches <= 2 ULPs in m80 and <= 1 ULP in m64/m32 across the entire range.
        if name == "patan_m80":
            return got[10:] == want[10:] and m80_ulp_diff(unpack_m80(got[:10]), unpack_m80(want[:10])) <= 2.0
        if name == "patan_to_m64":
            return got[8:] == want[8:] and binary_ulp_diff("m64", got[:8], want[:8]) <= 1
        if name == "patan_to_m32":
            return got[4:] == want[4:] and binary_ulp_diff("m32", got[:4], want[:4]) <= 1
        if name == "patan_depth8":
            return got[8:] == want[8:] and binary_ulp_diff("m64", got[:8], want[:8]) <= 1
    # Single-input trig functions: check input magnitude and special status first.
    v = unpack_m80(data[:10]) if kind in ("m80", "m80_in") else from_binary("m64", int.from_bytes(data[48:56], "little"))
    c = classify(v)
    if c[0] in ("unsupported", "nan", "inf", "zero"):
        return False  # Special values must match bit-for-bit.
    xv = m80_to_dec(v)
    if abs(xv) >= Decimal(2) ** 63:
        return False  # Out-of-range must match bit-for-bit (C2=1 and ST(0) unchanged).
    # For |x| > 8 (large pi reduction) or near exact zero of sin/cos where 66-bit pi diverges:
    # check C2/guard/auxiliary bytes match and result is finite; for |x| <= 8 check tight ULP bound.
    large = abs(xv) > 8
    if name in ("sin_m80", "cos_m80"):
        if got[10:32] != want[10:32] or got[32] != want[32] or got[33:] != want[33:]:
            return False
        gv, wv = unpack_m80(got[:10]), unpack_m80(want[:10])
        if large or abs(m80_to_dec(wv)) < Decimal(2) ** -50:
            return classify(gv)[0] in ("zero", "finite")
        return m80_ulp_diff(gv, wv) <= 4.0
    if name == "sin_c2_readers":
        if got[10:34] != want[10:34] or got[34:] != want[34:]:
            return False
        gv, wv = unpack_m80(got[:10]), unpack_m80(want[:10])
        if large or abs(m80_to_dec(wv)) < Decimal(2) ** -50:
            return classify(gv)[0] in ("zero", "finite")
        return m80_ulp_diff(gv, wv) <= 4.0
    if name in ("sin_to_m64", "cos_to_m64", "ptan_to_m64"):
        if got[8:] != want[8:]:
            return False
        if large:
            return classify(from_binary("m64", int.from_bytes(got[:8], "little")))[0] in ("zero", "finite")
        wv = from_binary("m64", int.from_bytes(want[:8], "little"))
        if abs(m80_to_dec(wv)) < Decimal(2) ** -40 or (name == "ptan_to_m64" and abs(m80_to_dec(wv)) > Decimal(2) ** 40):
            return classify(from_binary("m64", int.from_bytes(got[:8], "little")))[0] in ("zero", "finite")
        return binary_ulp_diff("m64", got[:8], want[:8]) <= 1
    if name in ("sin_to_m32", "cos_to_m32"):
        if got[4:] != want[4:]:
            return False
        if large:
            return classify(from_binary("m32", int.from_bytes(got[:4], "little")))[0] in ("zero", "finite")
        wv = from_binary("m32", int.from_bytes(want[:4], "little"))
        if abs(m80_to_dec(wv)) < Decimal(2) ** -20:
            return classify(from_binary("m32", int.from_bytes(got[:4], "little")))[0] in ("zero", "finite")
        return binary_ulp_diff("m32", got[:4], want[:4]) <= 1
    if name == "sincos_m80":
        if got[10:16] != want[10:16] or got[26:32] != want[26:32] or got[32] != want[32] or got[33:] != want[33:]:
            return False
        gs, ws = unpack_m80(got[:10]), unpack_m80(want[:10])
        gc, wc = unpack_m80(got[16:26]), unpack_m80(want[16:26])
        if large:
            return classify(gs)[0] in ("zero", "finite") and classify(gc)[0] in ("zero", "finite")
        ok_s = (classify(gs)[0] in ("zero", "finite")) if abs(m80_to_dec(ws)) < Decimal(2) ** -50 else (m80_ulp_diff(gs, ws) <= 4.0)
        ok_c = (classify(gc)[0] in ("zero", "finite")) if abs(m80_to_dec(wc)) < Decimal(2) ** -50 else (m80_ulp_diff(gc, wc) <= 4.0)
        return ok_s and ok_c
    if name == "sincos_sum":
        if got[10:] != want[10:]:
            return False
        gv, wv = unpack_m80(got[:10]), unpack_m80(want[:10])
        return m80_ulp_diff(gv, wv) <= 4.0
    if name in ("ptan_m80", "ptan_c2_reset"):
        if got[10:16] != want[10:16] or got[16:26] != want[16:26] or got[26:32] != want[26:32] or got[32] != want[32] or got[33:] != want[33:]:
            return False
        gt, wt = unpack_m80(got[:10]), unpack_m80(want[:10])
        if large or abs(m80_to_dec(wt)) < Decimal(2) ** -50 or abs(m80_to_dec(wt)) > Decimal(2) ** 50:
            return classify(gt)[0] in ("zero", "finite")
        return m80_ulp_diff(gt, wt) <= 4.0
    if name in ("sincos_depth7", "ptan_depth7"):
        # St(0) at [0..8] is cos / 1.0, St(1) at [8..16] is sin / tan, rest [16..64] exact.
        if got[16:] != want[16:]:
            return False
        if name == "ptan_depth7" and got[:8] != want[:8]:
            return False
        if large:
            return True
        ok0 = True if name == "ptan_depth7" else binary_ulp_diff("m64", got[:8], want[:8]) <= 1
        ok1 = binary_ulp_diff("m64", got[8:16], want[8:16]) <= 1
        return ok0 and ok1
    return False


def lifted_matches(name, got, want):
    if got == want:
        return True
    kind, _ = SEQUENCES[name]
    if name in ("sin_m80", "cos_m80", "patan_m80", "sincos_sum"):
        off_end = 32 if name in ("sin_m80", "cos_m80") else 10
        if got[10:off_end] != want[10:off_end] or got[off_end:] != want[off_end:]:
            return False
        return m80_ulp_diff(unpack_m80(got[:10]), unpack_m80(want[:10])) <= 1.0
    if name == "sin_c2_readers":
        return got[10:] == want[10:] and m80_ulp_diff(unpack_m80(got[:10]), unpack_m80(want[:10])) <= 1.0
    if name in ("sin_to_m64", "cos_to_m64", "ptan_to_m64", "patan_to_m64", "patan_depth8"):
        return got[8:] == want[8:] and binary_ulp_diff("m64", got[:8], want[:8]) <= 1
    if name in ("sin_to_m32", "cos_to_m32", "patan_to_m32"):
        return got[4:] == want[4:] and binary_ulp_diff("m32", got[:4], want[:4]) <= 1
    if name == "sincos_m80":
        return (got[10:16] == want[10:16] and got[26:] == want[26:]
                and m80_ulp_diff(unpack_m80(got[:10]), unpack_m80(want[:10])) <= 1.0
                and m80_ulp_diff(unpack_m80(got[16:26]), unpack_m80(want[16:26])) <= 1.0)
    if name in ("ptan_m80", "ptan_c2_reset"):
        return (got[10:16] == want[10:16] and got[16:] == want[16:]
                and m80_ulp_diff(unpack_m80(got[:10]), unpack_m80(want[:10])) <= 1.0)
    if name in ("sincos_depth7", "ptan_depth7"):
        return (got[16:] == want[16:]
                and (got[:8] == want[:8] if name == "ptan_depth7" else binary_ulp_diff("m64", got[:8], want[:8]) <= 1)
                and binary_ulp_diff("m64", got[8:16], want[8:16]) <= 1)
    return False


# ---- Check -----------------------------------------------------------------------------------------

def check(harness_path, native_path, lifted_path, *control_paths):
    expected = [(name, data, reference(name, data)) for name, data in cases()]
    for label, path, matcher in (("native", native_path, lambda n, d, g, w: native_matches(n, d, g, w)),
                                 ("lifted", lifted_path, lambda n, d, g, w: lifted_matches(n, g, w))):
        run = base.Runner(harness_path, path)
        bad = 0
        for name, data, want in expected:
            got, state = run(name, data)
            if not base.entry_state_ok(state):
                raise AssertionError(f"{label} {name} left state {state:#014x}; the profile requires control "
                                     "0x027f and an empty stack at TOP 0")
            if not matcher(name, data, got, want):
                if not bad:
                    print(f"{label} first mismatch: {name}(input {data[:32].hex()})\n  got      {got[:40].hex()}\n"
                          f"  expected {want[:40].hex()}")
                bad += 1
        if bad:
            raise AssertionError(f"{label}: {bad}/{len(expected)} mismatches")
    print(f"{len(expected)} cases over {len(SEQUENCES)} exports (output block and guards) matched the reference "
          "in native and translated DLLs; each export returned with control 0x027f and an empty stack")
    # Out-of-range |x| >= 2^63 for FSINCOS/FPTAN is outside the static-depth profile (hardware does not
    # push when C2=1); verify the lifted DLL traps via x87_trap() (STATUS_ILLEGAL_INSTRUCTION) before
    # writing the output buffer instead of continuing with a divergent stack.
    lifted_lib = ctypes.CDLL(str(Path(lifted_path).resolve()))
    in_buf = ctypes.create_string_buffer(INPUT)
    out_buf = ctypes.create_string_buffer(OUTPUT)
    for oor_v in ((J, 16383 + 63), (J, (16383 + 63) | 0x8000), (J, 16383 + 64), (J, 16383 + 100)):
        oor_data = m80_bytes(oor_v).ljust(INPUT, b"\0")
        for seq_name in ("sincos_m80", "ptan_m80", "ptan_c2_reset"):
            fn = getattr(lifted_lib, seq_name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            fn.restype = ctypes.c_uint64
            ctypes.memmove(in_buf, oor_data, INPUT)
            ctypes.memset(out_buf, 0xCC, OUTPUT)
            try:
                fn(ctypes.addressof(in_buf), ctypes.addressof(out_buf))
            except OSError as exc:
                if (getattr(exc, "winerror", 0) or 0) & 0xFFFFFFFF != 0xC000001D:
                    raise AssertionError(f"lifted {seq_name} out-of-range {oor_v} unexpected SEH {exc!r}") from exc
            else:
                raise AssertionError(f"lifted {seq_name} out-of-range {oor_v} returned instead of trapping")
            if out_buf.raw != b"\xcc" * OUTPUT:
                raise AssertionError(f"lifted {seq_name} out-of-range {oor_v} wrote output before trapping")
    print(f"processor: {platform.processor()}")
    first = {}
    for name, data, want in expected:
        first.setdefault(name, (data, want))
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        run = base.Runner(harness_path, path)
        failed = [name for name, (data, want) in first.items() if not base.preserved(run, "x87_call", name, data, want)]
        if failed:
            raise AssertionError(f"x87_call lost a live caller value or control word around {label} {failed}")
    print(f"x87_call kept a live caller ST(0) value and control word around all {len(first)} native and "
          "translated exports")
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        run = base.Runner(harness_path, path)
        bad = 0
        for name, data, want in expected:
            try:
                got, _ = run(name, data)
            except OSError as exc:
                if (getattr(exc, "winerror", 0) or 0) & 0xFFFFFFFF == 0xC000001D:
                    bad += 1
                    continue
                raise
            if not lifted_matches(name, got, want):
                bad += 1
        print(f"control {control}: {bad}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")
    check_rejections()


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

    fld, fstp = "db 29", "db 3a"  # fld tbyte ptr [rcx]; fstp tbyte ptr [rdx]
    sw = "df e0"  # fnstsw ax
    rejected = {
        "fsin on an empty stack": (attempt("d9 fe c3"), "x87 stack underflow"),
        "fcos on an empty stack": (attempt("d9 ff c3"), "x87 stack underflow"),
        "fsincos on an empty stack": (attempt("d9 fb c3"), "x87 stack underflow"),
        "fsincos at depth 8": (attempt(f"{fld} " * 8 + "d9 fb " + f"{fstp} " * 9 + "c3"), "x87 stack overflow"),
        "fptan on an empty stack": (attempt("d9 f2 c3"), "x87 stack underflow"),
        "fptan at depth 8": (attempt(f"{fld} " * 8 + "d9 f2 " + f"{fstp} " * 9 + "c3"), "x87 stack overflow"),
        "fpatan on an empty stack": (attempt("d9 f3 c3"), "x87 stack underflow"),
        "fpatan with one value": (attempt(f"{fld} d9 f3 {fstp} c3"), "x87 stack underflow"),
        "fsincos with one store": (attempt(f"{fld} d9 fb {fstp} c3"), "x87 stack not empty at return"),
        "fptan without popping 1.0": (attempt(f"{fld} d9 f2 {fstp} c3"), "x87 stack not empty at return"),
        "fsin then SAHF and JB (C0 unknown)": (attempt(f"{fld} d9 fe {sw} 9e 72 00 {fstp} c3"),
                                               "reads CF, which depends on a status-word bit"),
        "fcos then TEST AH, 01h (C0 unknown)": (attempt(f"{fld} d9 ff {sw} f6 c4 01 75 00 {fstp} c3"),
                                                "reads ZF, which depends on a status-word bit"),
        "fpatan then SAHF and JP (C2 unknown)": (attempt(f"{fld} {fld} d9 f3 {sw} 9e 7a 00 {fstp} c3"),
                                                 "reads PF, which depends on a status-word bit"),
        "fsin then FNSTSW m16": (attempt(f"{fld} d9 fe dd 3a {fstp} c3"), "stores status-word bits"),
        "f2xm1": (attempt(f"{fld} d9 f0 {fstp} c3"), "unsupported x87 operation"),
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
