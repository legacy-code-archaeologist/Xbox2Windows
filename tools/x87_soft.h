// Extended-precision helpers for the opt-in x87 profile of tools/ffx_version_query.lower().
//
// A value is exact: a sign, a 15-bit biased exponent and a 64-bit significand with an explicit integer
// bit. Every rounding helper takes the control word cw. Its precision control (bits 8 and 9) selects 24,
// 53 or 64 significand bits for arithmetic results, within the extended exponent range, and its rounding
// control (bits 10 and 11) the direction. Stores round again into the destination format. Every
// exception is masked and the status word is not modelled.
#include <intrin.h>

struct X87Value { uint64_t sig; uint16_t se; };  // se = sign << 15 | biased exponent

static const X87Value x87_indefinite{0xc000000000000000ull, 0xffff};
static const uint64_t x87_integer_bit = 0x8000000000000000ull;
// The rounding control: 0 to nearest-even, 1 toward negative infinity, 2 toward positive infinity and 3
// toward zero.
static inline unsigned x87_rounding(uint16_t cw) { return (cw >> 10) & 3u; }

static inline unsigned x87_exp(X87Value v) { return v.se & 0x7fffu; }
static inline bool x87_sign(X87Value v) { return (v.se >> 15) != 0; }
// A nonzero exponent with a clear integer bit (unnormal, pseudo-infinity, pseudo-NaN) is an invalid operand.
static inline bool x87_unsupported(X87Value v) { return x87_exp(v) != 0 && (v.sig & x87_integer_bit) == 0; }
static inline bool x87_is_nan(X87Value v) { return x87_exp(v) == 0x7fff && (v.sig & x87_integer_bit) && (v.sig << 1) != 0; }
static inline bool x87_is_inf(X87Value v) { return x87_exp(v) == 0x7fff && v.sig == x87_integer_bit; }
static inline bool x87_is_zero(X87Value v) { return x87_exp(v) == 0 && v.sig == 0; }
// FXAM's C3, C2 and C0 for a register that holds a value (C1, the sign, is not kept): unsupported 000,
// NaN 001, normal 010, infinity 011, zero 100 and denormal 110, which includes a pseudo-denormal. An empty
// ST(0), 101, is known when the function is translated.
static inline uint16_t x87_examine(X87Value v) {
    if (x87_unsupported(v)) return 0x0000;
    if (x87_is_nan(v)) return 0x0100;
    if (x87_is_inf(v)) return 0x0500;
    if (x87_is_zero(v)) return 0x4000;
    return x87_exp(v) == 0 ? 0x4400 : 0x0400;
}
static inline bool x87_is_quiet(X87Value v) { return (v.sig & 0x4000000000000000ull) != 0; }
static inline X87Value x87_quiet(X87Value v) { v.sig |= 0x4000000000000000ull; return v; }

static inline int x87_clz(uint64_t v) {
    unsigned long index;
    return _BitScanReverse64(&index, v) ? 63 - static_cast<int>(index) : 64;
}

// A finite nonzero value as sig * 2^(e - 63) with the integer bit of sig set. A denormal or
// pseudo-denormal has the exponent of biased exponent 1.
struct X87Unpacked { bool sign; int e; uint64_t sig; };
static inline X87Unpacked x87_unpack(X87Value v) {
    if (x87_exp(v) != 0) return {x87_sign(v), static_cast<int>(x87_exp(v)) - 16383, v.sig};
    const int shift = x87_clz(v.sig);
    return {x87_sign(v), 1 - 16383 - shift, v.sig << shift};
}

// Shift the 128-bit hi:lo right by n, folding every bit shifted out into bit 0 (the sticky bit).
static inline void x87_shr_sticky(uint64_t& hi, uint64_t& lo, int n) {
    if (n <= 0) return;
    if (n >= 128) { lo = (hi | lo) != 0; hi = 0; return; }
    uint64_t sticky;
    if (n >= 64) {
        const int m = n - 64;
        sticky = lo != 0 || (m != 0 && (hi << (64 - m)) != 0);
        lo = m != 0 ? hi >> m : hi;
        hi = 0;
    } else {
        sticky = (lo << (64 - n)) != 0;
        lo = (lo >> n) | (hi << (64 - n));
        hi >>= n;
    }
    lo |= sticky;
}

// A rounding target: significand bits, exponent bias and the largest biased exponent of a finite value.
struct X87Format { int precision, bias, emax; };
static const X87Format x87_f32{24, 127, 0xfe};
static const X87Format x87_f64{53, 1023, 0x7fe};
static const X87Format x87_arith{53, 16383, 0x7ffe};  // precision control 53 bits, extended exponent range
static const X87Format x87_arith24{24, 16383, 0x7ffe};
static const X87Format x87_arith64{64, 16383, 0x7ffe};
// The arithmetic format of a control word: PC 00 gives 24 bits, 10 gives 53 and 11 gives 64. The reserved
// PC 01 never reaches a helper, because the profile rejects a control word that sets it.
static inline X87Format x87_format(uint16_t cw) {
    const unsigned pc = (cw >> 8) & 3u;
    return pc == 0 ? x87_arith24 : pc == 3 ? x87_arith64 : x87_arith;
}

// Round sign * (hi:lo / 2^127) * 2^e, with hi:lo normalized (bit 127 set) and the sticky bit in bit 0 of
// lo. The result significand keeps its precision bits at the top of a 64-bit word. A result below the
// normal range is denormalized first and then rounded at the same bit positions. A result above it
// becomes infinity or the largest finite value, as the masked overflow response gives for the rounding
// direction.
struct X87Rounded { uint64_t sig; int biased; };
static inline X87Rounded x87_round(bool sign, int e, uint64_t hi, uint64_t lo, X87Format format, unsigned rc) {
    int biased = e + format.bias;
    if (biased <= 0) {
        x87_shr_sticky(hi, lo, 1 - biased);
        biased = 0;
    }
    const int drop = 64 - format.precision;
    const uint64_t unit = 1ull << drop, mask = unit - 1;
    bool above, half, inexact;
    if (drop == 0) {
        above = lo > x87_integer_bit;
        half = lo == x87_integer_bit;
        inexact = lo != 0;
    } else {
        const uint64_t rest = hi & mask, middle = unit >> 1;
        above = rest > middle || (rest == middle && lo != 0);
        half = rest == middle && lo == 0;
        inexact = rest != 0 || lo != 0;
    }
    hi &= ~mask;
    bool up = false;
    switch (rc) {
    case 0: up = above || (half && (hi & unit) != 0); break;   // to nearest, ties to even
    case 1: up = sign && inexact; break;                       // toward negative infinity
    case 2: up = !sign && inexact; break;                      // toward positive infinity
    default: break;                                            // toward zero
    }
    if (up) {
        hi += unit;
        if (hi == 0) { hi = x87_integer_bit; ++biased; }
    }
    if (biased == 0 && (hi & x87_integer_bit) != 0) biased = 1;  // a denormal rounded up to the normal range
    if (biased > format.emax) {
        const bool to_infinity = rc == 0 || (rc == 1 && sign) || (rc == 2 && !sign);
        return to_infinity ? X87Rounded{x87_integer_bit, format.emax + 1} : X87Rounded{~0ull << drop, format.emax};
    }
    return {hi, biased};
}

// Loads from m32 and m64 convert exactly; a signaling NaN is quieted (the masked invalid response). An
// arithmetic memory operand converts the same way but keeps a signaling NaN signaling (quiet = false), so
// the NaN rules of the arithmetic see the operand's own kind.
static inline X87Value x87_from_binary(uint64_t bits, int fraction_bits, int bias, int exponent_max, bool quiet = true) {
    const bool sign = (bits >> (fraction_bits + (fraction_bits == 23 ? 8 : 11))) & 1;
    const uint16_t s = sign ? 0x8000 : 0;
    const uint64_t fraction = bits & ((1ull << fraction_bits) - 1);
    const int exponent = static_cast<int>((bits >> fraction_bits) & static_cast<uint64_t>(exponent_max));
    if (exponent == exponent_max) {
        if (fraction == 0) return {x87_integer_bit, static_cast<uint16_t>(s | 0x7fff)};
        const X87Value nan{x87_integer_bit | (fraction << (63 - fraction_bits)), static_cast<uint16_t>(s | 0x7fff)};
        return quiet ? x87_quiet(nan) : nan;
    }
    if (exponent == 0) {
        if (fraction == 0) return {0, s};
        const int shift = x87_clz(fraction);  // value = fraction * 2^(1 - bias - fraction_bits)
        const int e = 1 - bias - fraction_bits + (63 - shift);
        return {fraction << shift, static_cast<uint16_t>(s | (e + 16383))};
    }
    return {x87_integer_bit | (fraction << (63 - fraction_bits)), static_cast<uint16_t>(s | (exponent - bias + 16383))};
}
static inline X87Value x87_from_f32(uint32_t bits) { return x87_from_binary(bits, 23, 127, 0xff); }
static inline X87Value x87_from_f64(uint64_t bits) { return x87_from_binary(bits, 52, 1023, 0x7ff); }
static inline X87Value x87_operand_f32(uint32_t bits) { return x87_from_binary(bits, 23, 127, 0xff, false); }
static inline X87Value x87_operand_f64(uint64_t bits) { return x87_from_binary(bits, 52, 1023, 0x7ff, false); }

// Stores to m32 and m64 round into the destination's precision and range. A NaN is quieted and its
// significand truncated; an invalid operand stores the destination's indefinite.
static inline uint64_t x87_to_binary(X87Value v, X87Format format, int fraction_bits, uint16_t cw) {
    const int sign_bit = fraction_bits == 23 ? 31 : 63;
    const uint64_t sign = static_cast<uint64_t>(x87_sign(v)) << sign_bit;
    const uint64_t exponent_all = static_cast<uint64_t>(format.emax + 1) << fraction_bits;
    const uint64_t quiet_bit = 1ull << (fraction_bits - 1);
    const uint64_t fraction_mask = (1ull << fraction_bits) - 1;
    if (x87_unsupported(v)) return (1ull << sign_bit) | exponent_all | quiet_bit;
    if (x87_is_nan(v)) return sign | exponent_all | quiet_bit | ((v.sig >> (63 - fraction_bits)) & fraction_mask);
    if (x87_is_inf(v)) return sign | exponent_all;
    if (x87_is_zero(v)) return sign;
    const X87Unpacked u = x87_unpack(v);
    const X87Rounded r = x87_round(u.sign, u.e, u.sig, 0, format, x87_rounding(cw));
    return sign | (static_cast<uint64_t>(r.biased) << fraction_bits) | ((r.sig >> (63 - fraction_bits)) & fraction_mask);
}
static inline uint32_t x87_to_f32(X87Value v, uint16_t cw) { return static_cast<uint32_t>(x87_to_binary(v, x87_f32, 23, cw)); }
static inline uint64_t x87_to_f64(X87Value v, uint16_t cw) { return x87_to_binary(v, x87_f64, 52, cw); }

// m80 loads and stores copy all 80 bits, including NaNs and unnormals.
static inline X87Value x87_from_m80(uint64_t sig, uint16_t se) { return {sig, se}; }

// NaN operands of an arithmetic instruction (Intel SDM Vol. 1, Table 4-7, x87 column): a QNaN wins over
// an SNaN, otherwise the larger significand wins, and the result is quieted. Of two NaNs with equal
// significands the positive one wins in either operand order (measured; the table does not say). An
// invalid operand gives the indefinite. Returns false when neither operand is a NaN or invalid.
static inline bool x87_special_operands(X87Value a, X87Value b, X87Value* out) {
    if (x87_unsupported(a) || x87_unsupported(b)) { *out = x87_indefinite; return true; }
    const bool na = x87_is_nan(a), nb = x87_is_nan(b);
    if (!na && !nb) return false;
    if (na && nb) {
        if (x87_is_quiet(a) != x87_is_quiet(b)) *out = x87_is_quiet(a) ? a : b;
        else *out = b.sig > a.sig || (b.sig == a.sig && !x87_sign(b)) ? b : a;
    } else {
        *out = na ? a : b;
    }
    *out = x87_quiet(*out);
    return true;
}

// FADD: the exact sum rounded to the arithmetic format. Addends far apart keep their low bits in the
// sticky bit. An exact zero sum of addends with different signs is +0, or -0 when rounding toward
// negative infinity; two zeros of the same sign keep it.
static inline X87Value x87_add(X87Value a, X87Value b, uint16_t cw) {
    X87Value out;
    if (x87_special_operands(a, b, &out)) return out;
    const bool ia = x87_is_inf(a), ib = x87_is_inf(b);
    if (ia || ib) {
        if (ia && ib && x87_sign(a) != x87_sign(b)) return x87_indefinite;
        return ia ? a : b;
    }
    const bool za = x87_is_zero(a), zb = x87_is_zero(b);
    const uint16_t zero_sign = x87_rounding(cw) == 1 ? 0x8000 : 0;  // x + (-x) and +0 + -0
    if (za && zb) return {0, static_cast<uint16_t>(x87_sign(a) == x87_sign(b) ? a.se & 0x8000 : zero_sign)};
    X87Unpacked x = x87_unpack(za ? b : a), y = x87_unpack(zb ? a : b);
    if (za || zb) {  // x + 0 still rounds to the arithmetic precision
        const X87Rounded r = x87_round(x.sign, x.e, x.sig, 0, x87_format(cw), x87_rounding(cw));
        return {r.sig, static_cast<uint16_t>((x.sign ? 0x8000 : 0) | r.biased)};
    }
    if (x.e < y.e || (x.e == y.e && x.sig < y.sig)) { const X87Unpacked t = x; x = y; y = t; }
    uint64_t hi = x.sig, lo = 0, yhi = y.sig, ylo = 0;
    x87_shr_sticky(yhi, ylo, x.e - y.e);
    int e = x.e;
    if (x.sign == y.sign) {
        unsigned char carry = _addcarry_u64(0, lo, ylo, &lo);
        carry = _addcarry_u64(carry, hi, yhi, &hi);
        if (carry) {
            const uint64_t sticky = lo & 1;
            lo = (lo >> 1) | (hi << 63) | sticky;
            hi = (hi >> 1) | x87_integer_bit;
            ++e;
        }
    } else {
        const unsigned char borrow = _subborrow_u64(0, lo, ylo, &lo);
        _subborrow_u64(borrow, hi, yhi, &hi);
        if (hi == 0 && lo == 0) return {0, zero_sign};  // x + (-x)
        const int shift = hi != 0 ? x87_clz(hi) : 64 + x87_clz(lo);
        if (shift >= 64) { hi = lo << (shift - 64); lo = 0; }
        else if (shift != 0) { hi = (hi << shift) | (lo >> (64 - shift)); lo <<= shift; }
        e -= shift;
    }
    const X87Rounded r = x87_round(x.sign, e, hi, lo, x87_format(cw), x87_rounding(cw));
    return {r.sig, static_cast<uint16_t>((x.sign ? 0x8000 : 0) | r.biased)};
}

// FMUL: the exact product rounded to the arithmetic format. The sign is the exclusive or of the operand
// signs, also for zero and infinite results; infinity times zero is invalid and gives the indefinite.
static inline X87Value x87_mul(X87Value a, X87Value b, uint16_t cw) {
    X87Value out;
    if (x87_special_operands(a, b, &out)) return out;
    const bool sign = x87_sign(a) != x87_sign(b);
    const uint16_t s = sign ? 0x8000 : 0;
    const bool ia = x87_is_inf(a), ib = x87_is_inf(b), za = x87_is_zero(a), zb = x87_is_zero(b);
    if ((ia && zb) || (ib && za)) return x87_indefinite;
    if (ia || ib) return {x87_integer_bit, static_cast<uint16_t>(s | 0x7fff)};
    if (za || zb) return {0, s};
    const X87Unpacked x = x87_unpack(a), y = x87_unpack(b);
    uint64_t hi, lo = _umul128(x.sig, y.sig, &hi);  // x.sig * y.sig in [2^126, 2^128)
    int e = x.e + y.e;
    if (hi & x87_integer_bit) ++e;
    else { hi = (hi << 1) | (lo >> 63); lo <<= 1; }
    const X87Rounded r = x87_round(sign, e, hi, lo, x87_format(cw), x87_rounding(cw));
    return {r.sig, static_cast<uint16_t>(s | r.biased)};
}

// FDIV: the quotient rounded to the arithmetic format. A 64-bit quotient of the significands, the next
// quotient bit (bit 63 of lo) and a sticky bit for any further remainder are enough to round to every
// precision. The sign is the exclusive or of the operand
// signs. Zero by zero and infinity by infinity are invalid and give the indefinite; a finite nonzero value
// by zero gives infinity (the masked divide-by-zero response); a finite value by infinity gives zero.
static inline X87Value x87_div(X87Value a, X87Value b, uint16_t cw) {
    X87Value out;
    if (x87_special_operands(a, b, &out)) return out;
    const bool sign = x87_sign(a) != x87_sign(b);
    const uint16_t s = sign ? 0x8000 : 0;
    const bool ia = x87_is_inf(a), ib = x87_is_inf(b), za = x87_is_zero(a), zb = x87_is_zero(b);
    if ((ia && ib) || (za && zb)) return x87_indefinite;
    if (ia || zb) return {x87_integer_bit, static_cast<uint16_t>(s | 0x7fff)};
    if (ib || za) return {0, s};
    const X87Unpacked x = x87_unpack(a), y = x87_unpack(b);
    // Divide x.sig * 2^64 (or x.sig * 2^63 when x.sig >= y.sig) by y.sig: the quotient lies in [2^63, 2^64).
    const bool shift = x.sig >= y.sig;
    uint64_t remainder;
    const uint64_t q = _udiv128(shift ? x.sig >> 1 : x.sig, shift ? x.sig << 63 : 0, y.sig, &remainder);
    const bool round = remainder >= y.sig - remainder;  // 2 * remainder >= divisor
    const uint64_t lo = (round ? x87_integer_bit : 0) | (round ? remainder != y.sig - remainder : remainder != 0);  // div sticky
    const X87Rounded r = x87_round(sign, x.e - y.e - (shift ? 0 : 1), q, lo, x87_format(cw), x87_rounding(cw));
    return {r.sig, static_cast<uint16_t>(s | r.biased)};
}
// FDIVR: the same division with the operands exchanged.
static inline X87Value x87_divr(X87Value a, X87Value b, uint16_t cw) { return x87_div(b, a, cw); }

// FSUB: a - b as a + (-b). A NaN or invalid operand is selected before the negation, so a NaN result keeps
// its own sign; x - x is +0, as x + (-x) is.
static inline X87Value x87_sub(X87Value a, X87Value b, uint16_t cw) {
    X87Value out;
    if (x87_special_operands(a, b, &out)) return out;
    b.se ^= 0x8000;
    return x87_add(a, b, cw);
}
// FSUBR: the same subtraction with the operands exchanged.
static inline X87Value x87_subr(X87Value a, X87Value b, uint16_t cw) { return x87_sub(b, a, cw); }

// FCHS and FABS change only the sign bit, of every encoding: NaNs stay signaling or quiet, and unnormals
// and pseudo-denormals keep their bits.
static inline X87Value x87_chs(X87Value v) { v.se ^= 0x8000; return v; }
static inline X87Value x87_abs(X87Value v) { v.se &= 0x7fff; return v; }

// FILD: a 16-, 32- or 64-bit two's-complement integer converts exactly, and zero gives +0. The precision
// control does not apply to loads.
static inline X87Value x87_from_integer(int64_t n) {
    if (n == 0) return {0, 0};
    const bool negative = n < 0;
    const uint64_t magnitude = negative ? 0 - static_cast<uint64_t>(n) : static_cast<uint64_t>(n);
    const int shift = x87_clz(magnitude);
    return {magnitude << shift, static_cast<uint16_t>((negative ? 0x8000 : 0) | (16383 + 63 - shift))};
}

// FIST, FISTP and FISTTP: ST(0) rounded to an integer (in the direction of the rounding control, or toward
// zero when chop is set)
// as two's-complement bits in the low `bits` bits. A NaN, an infinity, an unsupported encoding or a
// rounded value outside the destination range gives the integer indefinite (only the sign bit set), the
// masked invalid-operation response. The precision control does not apply.
static inline uint64_t x87_to_integer(X87Value v, int bits, bool chop, uint16_t cw) {
    const uint64_t indefinite = 1ull << (bits - 1);
    const uint64_t mask = bits == 64 ? ~0ull : (1ull << bits) - 1;
    if (x87_unsupported(v) || x87_exp(v) == 0x7fff) return indefinite;
    if (v.sig == 0) return 0;
    const int shift = 16383 + 63 - (x87_exp(v) == 0 ? 1 : static_cast<int>(x87_exp(v)));  // v = sig * 2^-shift
    if (shift < 0) return indefinite;
    uint64_t magnitude = shift >= 64 ? 0 : v.sig >> shift;
    const unsigned rc = chop ? 3u : x87_rounding(cw);
    if (shift > 0) {
        const uint64_t rest = shift >= 64 ? v.sig : v.sig & ((1ull << shift) - 1);  // the fraction bits
        if (rc != 0) {  // up in magnitude when the direction is away from zero
            if (rest != 0 && rc == (x87_sign(v) ? 1u : 2u)) ++magnitude;
        } else if (shift <= 64) {  // to nearest; a value below one half (shift > 64) rounds to 0
            const uint64_t half = 1ull << (shift - 1);
            if (rest > half || (rest == half && (magnitude & 1))) ++magnitude;
        }
    }
    if (magnitude > indefinite - (x87_sign(v) ? 0 : 1)) return indefinite;
    return (x87_sign(v) ? 0 - magnitude : magnitude) & mask;
}

// FBSTP: ST(0) rounded to an integer in the direction of the rounding control, as 18 packed BCD digits
// in bytes 0 to 8 (low and high, two digits a byte, the lower digit in the low nibble) and the sign of
// ST(0) in bit 7 of byte 9, which a negative value that rounds to zero keeps. A NaN, an infinity, an
// unsupported encoding or a rounded magnitude of 10^18 or more gives the packed BCD indefinite (bytes 7 to
// 9 0xc0, 0xff and 0xff), the masked invalid-operation response. The precision control does not apply.
static inline void x87_to_bcd(X87Value v, uint16_t cw, uint64_t& low, uint16_t& high) {
    const uint64_t r = x87_to_integer(v, 64, false, cw);  // two's complement, or 2^63 for the indefinite
    uint64_t n = x87_sign(v) ? 0 - r : r;
    if (r == 1ull << 63 || n >= 1000000000000000000ull) {  // bcd range
        low = 0xc000000000000000ull;
        high = 0xffff;
        return;
    }
    const uint16_t sign = x87_sign(v) ? 0x8000 : 0;  // bcd sign
    low = 0;
    for (int k = 0; k < 16; ++k, n /= 10) low |= (n % 10) << (4 * k);
    high = static_cast<uint16_t>(sign | (n / 10) << 4 | n % 10);
}

// FBLD: the 18 packed BCD digits of bytes 0 to 8 (low and high), converted exactly, with the sign in bit
// 7 of byte 9; the other bits of that byte are ignored, and a zero keeps its sign. False for a digit
// above 9, whose result the SDM leaves undefined.
static inline bool x87_from_bcd(uint64_t low, uint16_t high, X87Value& out) {
    uint64_t n = 0;
    for (int k = 17; k >= 0; --k) {
        const uint64_t digit = ((k >= 16 ? high : low) >> (4 * (k & 15))) & 15;  // bcd digit
        if (digit > 9) return false;
        n = n * 10 + digit;
    }
    const bool negative = (high & 0x8000) != 0;  // bcd sign bit
    if (n == 0) {
        out = {0, static_cast<uint16_t>(negative ? 0x8000 : 0)};
        return true;
    }
    out = x87_from_integer(negative ? -static_cast<int64_t>(n) : static_cast<int64_t>(n));
    return true;
}

// FCOMI, FUCOMI and their P forms: 0 if a < b, 1 if equal, 2 if a > b and 3 if unordered (a NaN or an
// unsupported encoding; with the invalid operation masked, FCOMI and FUCOMI give the same result). The
// comparison uses all 64 significand bits, +0 equals -0, and a pseudo-denormal equals the normal value
// with the same significand at biased exponent 1.
static inline int x87_compare(X87Value a, X87Value b) {
    if (x87_unsupported(a) || x87_unsupported(b) || x87_is_nan(a) || x87_is_nan(b)) return 3;  // unordered
    const bool za = x87_is_zero(a), zb = x87_is_zero(b);
    if (za && zb) return 1;  // signed zeros
    if (za) return x87_sign(b) ? 2 : 0;
    if (zb) return x87_sign(a) ? 0 : 2;
    if (x87_sign(a) != x87_sign(b)) return x87_sign(a) ? 0 : 2;
    const X87Unpacked ua = x87_unpack(a), ub = x87_unpack(b);  // infinities unpack above every finite value
    if (ua.e == ub.e && ua.sig == ub.sig) return 1;
    const bool less = ua.e != ub.e ? ua.e < ub.e : ua.sig < ub.sig;
    return less != x87_sign(a) ? 0 : 2;
}

// FSQRT: the exact square root rounded to the arithmetic format.
// sqrt(-0) is -0 and sqrt(+inf) is +inf; a NaN is quieted; any other negative operand, and an unsupported
// encoding, gives the indefinite. The root of sig * 2^(e - 63) is isqrt(sig * 2^t) * 2^((e - 63 - t) / 2)
// with t = 63 or 64 chosen to make the exponent even, which puts the 64-bit root q in [2^63, 2^64). The
// next root bit is set when the remainder exceeds q (an exact half cannot occur), and any nonzero
// remainder sets the sticky bit. The result never leaves the extended normal range.
static inline X87Value x87_sqrt(X87Value a, uint16_t cw) {
    if (x87_unsupported(a)) return x87_indefinite;
    if (x87_is_nan(a)) return x87_quiet(a);
    if (x87_is_zero(a)) return a;
    if (x87_sign(a)) return x87_indefinite;
    if (x87_is_inf(a)) return a;
    const X87Unpacked x = x87_unpack(a);
    const int t = (x.e & 1) ? 64 : 63;
    const uint64_t r_hi = t == 64 ? x.sig : x.sig >> 1, r_lo = t == 64 ? 0 : x.sig << 63;  // R = sig * 2^t
    uint64_t q = 0, sq_hi = 0, sq_lo = 0;
    for (uint64_t bit = x87_integer_bit; bit != 0; bit >>= 1) {
        const uint64_t c = q | bit;
        uint64_t hi;
        const uint64_t lo = _umul128(c, c, &hi);
        if (hi < r_hi || (hi == r_hi && lo <= r_lo)) { q = c; sq_hi = hi; sq_lo = lo; }
    }
    uint64_t rem_lo, rem_hi;  // R - q * q
    _subborrow_u64(_subborrow_u64(0, r_lo, sq_lo, &rem_lo), r_hi, sq_hi, &rem_hi);
    const uint64_t sticky = ((rem_hi != 0 || rem_lo > q) ? x87_integer_bit : 0) | (sq_hi != r_hi || sq_lo != r_lo);  // sqrt sticky
    const X87Rounded r = x87_round(false, (x.e - 63 - t) / 2 + 63, q, sticky, x87_format(cw), x87_rounding(cw));  // sqrt round
    return {r.sig, static_cast<uint16_t>(r.biased)};
}

// FRNDINT: ST(0) rounded to an integral value in the direction of the rounding control, kept in the
// extended format. The precision
// control does not apply. Zeros, infinities and values of 2^63 and above are already integral; a value
// that rounds to zero keeps its sign; a NaN is quieted and an unsupported encoding gives the indefinite.
static inline X87Value x87_rndint(X87Value v, uint16_t cw) {
    if (x87_unsupported(v)) return x87_indefinite;
    if (x87_is_nan(v)) return x87_quiet(v);
    if (x87_exp(v) == 0x7fff || v.sig == 0) return v;
    const int shift = 16383 + 63 - (x87_exp(v) == 0 ? 1 : static_cast<int>(x87_exp(v)));  // v = sig * 2^-shift
    if (shift <= 0) return v;
    uint64_t magnitude = shift >= 64 ? 0 : v.sig >> shift;
    const unsigned rc = x87_rounding(cw);
    if (rc != 0) {  // up in magnitude when the direction is away from zero and the value is not integral
        const uint64_t rest = shift >= 64 ? v.sig : v.sig & ((1ull << shift) - 1);
        if (rest != 0 && rc == (x87_sign(v) ? 1u : 2u)) ++magnitude;
    } else if (shift <= 64) {  // to nearest; a value below one half (shift > 64) rounds to 0
        const uint64_t rest = shift == 64 ? v.sig : v.sig & ((1ull << shift) - 1);
        const uint64_t half = 1ull << (shift - 1);
        if (rest > half || (rest == half && (magnitude & 1))) ++magnitude;  // rndint round
    }
    const uint16_t sign = v.se & 0x8000;
    if (magnitude == 0) return {0, sign};
    const int lead = x87_clz(magnitude);
    return {magnitude << lead, static_cast<uint16_t>(sign | (16383 + 63 - lead))};
}

// FXTRACT: the significand of v, with v's sign and biased exponent 16383 (a value in [1, 2)), and in
// `exponent` the unbiased exponent as an exact value. A denormal is normalized first. A zero keeps its
// sign and gives an exponent of -inf (the masked divide-by-zero response); an infinity keeps its sign
// and gives +inf. A NaN is quieted into both results, and an unsupported encoding gives the indefinite
// in both. The precision control does not apply.
static inline X87Value x87_xtract(X87Value v, X87Value& exponent) {
    if (x87_unsupported(v)) { exponent = x87_indefinite; return x87_indefinite; }
    if (x87_is_nan(v)) { exponent = x87_quiet(v); return exponent; }
    if (x87_is_inf(v)) { exponent = {x87_integer_bit, 0x7fff}; return v; }
    if (x87_is_zero(v)) { exponent = {x87_integer_bit, 0xffff}; return v; }  // xtract zero
    const X87Unpacked u = x87_unpack(v);
    exponent = x87_from_integer(u.e);
    return {u.sig, static_cast<uint16_t>((u.sign ? 0x8000 : 0) | 16383)};
}

// FSCALE: ST(0) * 2^n with n = ST(1) truncated toward zero. The precision control does not apply: the
// result keeps 64 significand bits and rounds only below the normal range (denormalized at the same
// bit positions) or overflows to infinity. ST(1) = -inf gives a zero and +inf an infinity, both with
// the sign of ST(0); an infinite ST(0) scaled by -inf and a zero scaled by +inf are invalid. A zero or
// infinite ST(0) is otherwise unchanged. NaN and unsupported operands follow x87_special_operands. n is
// clamped to +-2^20, far beyond the exponent range, before it is added to the exponent.
static const X87Format x87_scale_format{64, 16383, 0x7ffe};
static inline X87Value x87_scale(X87Value a, X87Value b, uint16_t cw) {
    X87Value out;
    if (x87_special_operands(a, b, &out)) return out;
    const uint16_t s = a.se & 0x8000;
    if (x87_is_inf(b)) {
        if (x87_sign(b)) return x87_is_inf(a) ? x87_indefinite : X87Value{0, s};
        return x87_is_zero(a) ? x87_indefinite : X87Value{x87_integer_bit, static_cast<uint16_t>(s | 0x7fff)};
    }
    if (x87_is_zero(a) || x87_is_inf(a)) return a;
    int n = 0;
    if (!x87_is_zero(b)) {
        const X87Unpacked y = x87_unpack(b);
        const int magnitude = y.e >= 20 ? 1 << 20 : y.e >= 0 ? static_cast<int>(y.sig >> (63 - y.e)) : 0;
        n = x87_sign(b) ? -magnitude : magnitude;  // scale truncate
    }
    const X87Unpacked x = x87_unpack(a);
    const X87Rounded r = x87_round(x.sign, x.e + n, x.sig, 0, x87_scale_format, x87_rounding(cw));  // scale round
    return {r.sig, static_cast<uint16_t>(s | r.biased)};
}

// FPREM and FPREM1: the partial remainder ST(0) - q * ST(1), where q is |ST(0) / ST(1)| truncated toward
// zero (FPREM) or rounded to nearest-even (FPREM1); out receives it. When the normalized exponent of ST(0)
// exceeds that of ST(1) by 64 or more, the instruction reduces ST(0) only partially, by an amount the SDM
// leaves to the implementation, and the helper returns x87_prem_partial without computing a result. A NaN
// or unsupported operand, an infinite ST(0) or a zero ST(1) gives a NaN as for the arithmetic instructions,
// and the helper returns x87_prem_nan. Otherwise the remainder is exact and has the sign of ST(0) when it
// is zero, and the helper returns the low three bits of q, which the instructions copy to C0 (bit 2), C3
// (bit 1) and C1 (bit 0). A zero ST(0) is kept, and an infinite ST(1) gives ST(0) with q = 0. A
// pseudo-denormal result is re-encoded at biased exponent 1. Neither precision nor rounding control
// applies.
static const int x87_prem_partial = -1;
static const int x87_prem_nan = -2;
static inline int x87_prem(X87Value a, X87Value b, bool nearest, X87Value& out) {
    if (x87_special_operands(a, b, &out)) return x87_prem_nan;
    if (x87_is_inf(a) || x87_is_zero(b)) { out = x87_indefinite; return x87_prem_nan; }
    if (x87_is_zero(a)) { out = a; return 0; }
    const X87Unpacked x = x87_unpack(a);
    uint64_t r = x.sig, q = 0;
    int e = x.e;  // the result is r * 2^(e - 63)
    bool flip = false;
    if (!x87_is_inf(b)) {
        const X87Unpacked y = x87_unpack(b);
        const int d = x.e - y.e;
        if (d >= 64) { out = a; return x87_prem_partial; }  // prem partial
        if (d >= 0) {  // long division of x.sig * 2^d by y.sig; r stays below y.sig
            if (r >= y.sig) { r -= y.sig; q = 1; }
            for (int k = 0; k < d; ++k) {
                const bool high = (r >> 63) != 0;  // 2r reaches 2^64, so it is above y.sig
                r <<= 1;
                q <<= 1;
                if (high || r >= y.sig) { r -= y.sig; q |= 1; }
            }
            e = y.e;
            if (nearest && (r > y.sig - r || (r == y.sig - r && (q & 1)))) { r = y.sig - r; ++q; flip = true; }  // prem nearest
        } else if (nearest && d == -1 && x.sig > y.sig) {  // |ST(0)| above half of |ST(1)|: q rounds to 1
            r = y.sig - (x.sig - y.sig);  // |ST(1)| - |ST(0)| in units of ST(0)
            q = 1;
            flip = true;
        }
    }
    if (r == 0) { out = {0, static_cast<uint16_t>(a.se & 0x8000)}; return static_cast<int>(q & 7); }
    const int shift = x87_clz(r);
    int biased = e - shift + 16383;
    uint64_t sig = r << shift;
    if (biased < 1) { sig >>= 1 - biased; biased = 0; }  // exact: the remainder is a multiple of the denormal step
    out = {sig, static_cast<uint16_t>(((a.se & 0x8000) ^ (flip ? 0x8000 : 0)) | biased)};
    return static_cast<int>(q & 7);
}

// 128-bit unsigned fixed-point helpers for the transcendental instructions (pure integer arithmetic).
struct X87U128 { uint64_t hi, lo; };

static inline int x87_u128_cmp(X87U128 a, X87U128 b) {
    if (a.hi != b.hi) return a.hi < b.hi ? -1 : 1;
    if (a.lo != b.lo) return a.lo < b.lo ? -1 : 1;
    return 0;
}

static inline X87U128 x87_u128_add(X87U128 a, X87U128 b) {
    X87U128 r;
    const unsigned char c = _addcarry_u64(0, a.lo, b.lo, &r.lo);
    _addcarry_u64(c, a.hi, b.hi, &r.hi);
    return r;
}

static inline X87U128 x87_u128_sub(X87U128 a, X87U128 b) {
    X87U128 r;
    const unsigned char c = _subborrow_u64(0, a.lo, b.lo, &r.lo);
    _subborrow_u64(c, a.hi, b.hi, &r.hi);
    return r;
}

// Upper 128 bits of the 256-bit product a * b.
static inline X87U128 x87_u128_mul_hi(X87U128 a, X87U128 b) {
    uint64_t p0_hi;
    _umul128(a.lo, b.lo, &p0_hi);
    uint64_t p1_hi;
    const uint64_t p1_lo = _umul128(a.hi, b.lo, &p1_hi);
    uint64_t p2_hi;
    const uint64_t p2_lo = _umul128(a.lo, b.hi, &p2_hi);
    uint64_t p3_hi;
    uint64_t p3_lo = _umul128(a.hi, b.hi, &p3_hi);
    uint64_t s0 = p0_hi;
    unsigned char c = _addcarry_u64(0, s0, p1_lo, &s0);
    uint64_t c1 = c;
    c = _addcarry_u64(0, s0, p2_lo, &s0);
    c1 += c;
    uint64_t s1 = p1_hi;
    c = _addcarry_u64(0, s1, p2_hi, &s1);
    uint64_t c2 = c;
    c = _addcarry_u64(0, s1, c1, &s1);
    c2 += c;
    c = _addcarry_u64(0, p3_lo, s1, &p3_lo);
    _addcarry_u64(c, p3_hi, c2, &p3_hi);
    return {p3_hi, p3_lo};
}

static inline X87U128 x87_u128_div_u64(X87U128 a, uint64_t d) {
    uint64_t rem = 0;
    const uint64_t qhi = _udiv128(0, a.hi, d, &rem);
    const uint64_t qlo = _udiv128(rem, a.lo, d, &rem);
    return {qhi, qlo};
}

static inline X87U128 x87_u128_shr(X87U128 a, int n) {
    if (n <= 0) return a;
    if (n >= 128) return {0, 0};
    if (n >= 64) return {0, a.hi >> (n - 64)};
    return {a.hi >> n, (a.lo >> n) | (a.hi << (64 - n))};
}

static inline void x87_u128_norm(X87U128& v, int& e) {
    if (v.hi == 0) {
        if (v.lo == 0) return;
        const int s = x87_clz(v.lo);
        v.hi = v.lo << s;
        v.lo = 0;
        e -= 64 + s;
    } else {
        const int s = x87_clz(v.hi);
        if (s != 0) {
            v.hi = (v.hi << s) | (v.lo >> (64 - s));
            v.lo <<= s;
            e -= s;
        }
    }
}

// Normalized 128-bit quotient a / b for normalized nonzero operands, returning value = (q / 2^127) * 2^qe.
static inline X87U128 x87_u128_div_norm(X87U128 a, int ea, X87U128 b, int eb, int& qe) {
    int e = ea - eb;
    X87U128 rem = a;
    if (x87_u128_cmp(rem, b) >= 0) {
        rem = x87_u128_sub(rem, b);
    } else {
        --e;
        const bool top = (rem.hi >> 63) != 0;
        rem.hi = (rem.hi << 1) | (rem.lo >> 63);
        rem.lo <<= 1;
        if (top || x87_u128_cmp(rem, b) >= 0) rem = x87_u128_sub(rem, b);
    }
    uint64_t qhi = x87_integer_bit, qlo = 0;
    for (int bit = 126; bit >= 0; --bit) {
        const bool top = (rem.hi >> 63) != 0;
        rem.hi = (rem.hi << 1) | (rem.lo >> 63);
        rem.lo <<= 1;
        if (top || x87_u128_cmp(rem, b) >= 0) {
            rem = x87_u128_sub(rem, b);
            if (bit >= 64) qhi |= 1ull << (bit - 64);
            else qlo |= 1ull << bit;
        }
    }
    qe = e;
    return {qhi, qlo};
}

static inline X87Value x87_round_u128(bool sign, int e, X87U128 v, uint16_t cw, bool force_inexact = true) {
    if (v.hi == 0 && v.lo == 0) return {0, static_cast<uint16_t>(sign ? 0x8000 : 0)};
    x87_u128_norm(v, e);
    const uint64_t lo = v.lo | (force_inexact ? 1ull : 0ull);
    const X87Rounded r = x87_round(sign, e, v.hi, lo, x87_arith64, x87_rounding(cw));
    return {r.sig, static_cast<uint16_t>((sign ? 0x8000 : 0) | r.biased)};
}

// 2/pi to 256 bits (value * 2^256) and pi/2 to 128 bits (value * 2^127).
static const uint64_t x87_inv_pi2[4] = {
    0xa2f9836e4e441529ull, 0xfc2757d1f534ddc0ull, 0xdb6295993c439041ull, 0xfe5163abdebbc562ull
};
static const X87U128 x87_pi2{0xc90fdaa22168c234ull, 0xc4c6628b80dc1cd1ull};

// Payne-Hanek range reduction of finite |x| = sig * 2^(e - 63) with e < 63 modulo pi/2.
// Sets r_neg when |x| = k * (pi/2) - r, and normalizes r = (r_val / 2^127) * 2^re in [0, pi/4].
static inline int x87_reduce_pi2(uint64_t sig, int e, X87U128& r_val, int& re, bool& r_neg) {
    r_neg = false;
    if (e <= -1 && (e < -1 || sig <= 0xc90fdaa22168c234ull)) {
        r_val = {sig, 0};
        re = e;
        return 0;
    }
    uint64_t p[5]{};
    for (int i = 0; i < 4; ++i) {
        uint64_t hi;
        const uint64_t lo = _umul128(sig, x87_inv_pi2[3 - i], &hi);
        unsigned char c = _addcarry_u64(0, p[i], lo, &p[i]);
        c = _addcarry_u64(c, p[i + 1], hi, &p[i + 1]);
        for (int j = i + 2; c && j < 5; ++j) c = _addcarry_u64(c, p[j], 0, &p[j]);
    }
    const int s = 63 - e;
    uint64_t k = 0;
    uint64_t f[4]{};
    if (s == 64) {
        f[3] = p[4]; f[2] = p[3]; f[1] = p[2]; f[0] = p[1];
    } else {
        const int sh = 64 - s;
        k = p[4] >> s;
        f[3] = (p[4] << sh) | (p[3] >> s);
        f[2] = (p[3] << sh) | (p[2] >> s);
        f[1] = (p[2] << sh) | (p[1] >> s);
        f[0] = (p[1] << sh) | (p[0] >> s);
    }
    if (f[3] & x87_integer_bit) {
        ++k;
        r_neg = true;
        unsigned char b = _subborrow_u64(0, 0, f[0], &f[0]);
        b = _subborrow_u64(b, 0, f[1], &f[1]);
        b = _subborrow_u64(b, 0, f[2], &f[2]);
        _subborrow_u64(b, 0, f[3], &f[3]);
    }
    int top_limb = 3;
    while (top_limb >= 0 && f[top_limb] == 0) --top_limb;
    if (top_limb < 0) { r_val = {0, 0}; re = 0; return static_cast<int>(k & 3); }
    const int lz = x87_clz(f[top_limb]);
    const int fe = -1 - (3 - top_limb) * 64 - lz;
    uint64_t fhi = f[top_limb] << lz;
    uint64_t flo = top_limb >= 1 ? f[top_limb - 1] : 0;
    if (lz != 0 && top_limb >= 1) fhi |= f[top_limb - 1] >> (64 - lz);
    if (lz != 0) {
        flo <<= lz;
        if (top_limb >= 2) flo |= f[top_limb - 2] >> (64 - lz);
    }
    r_val = x87_u128_mul_hi({fhi, flo}, x87_pi2);
    re = fe + 1;
    x87_u128_norm(r_val, re);
    return static_cast<int>(k & 3);
}

// sin(r) and cos(r) for normalized r in [0, pi/4].
static inline void x87_sincos_reduced(X87U128 r_val, int re, X87U128& sin_v, int& sin_e, X87U128& cos_v, int& cos_e) {
    if (r_val.hi == 0 && r_val.lo == 0) {
        sin_v = {0, 0}; sin_e = 0;
        cos_v = {x87_integer_bit, 0}; cos_e = 0;
        return;
    }
    if (re <= -65) {
        sin_v = r_val; sin_e = re;
        cos_v = {0xffffffffffffffffull, 0xffffffffffffffffull}; cos_e = -1;
        return;
    }
    X87U128 r2 = x87_u128_mul_hi(r_val, r_val);
    r2 = x87_u128_shr(r2, -2 * re - 2);
    const X87U128 one_q126{1ull << 62, 0};
    X87U128 s_sum = one_q126, s_term = one_q126;
    X87U128 c_sum = one_q126, c_term = one_q126;
    for (uint64_t n = 1; n <= 19; ++n) {
        s_term = x87_u128_div_u64(x87_u128_mul_hi(s_term, r2), (2 * n) * (2 * n + 1));
        c_term = x87_u128_div_u64(x87_u128_mul_hi(c_term, r2), (2 * n - 1) * (2 * n));
        if (n & 1) {
            s_sum = x87_u128_sub(s_sum, s_term);
            c_sum = x87_u128_sub(c_sum, c_term);
        } else {
            s_sum = x87_u128_add(s_sum, s_term);
            c_sum = x87_u128_add(c_sum, c_term);
        }
        if ((s_term.hi | s_term.lo | c_term.hi | c_term.lo) == 0) break;
    }
    sin_v = x87_u128_mul_hi(r_val, s_sum);
    sin_e = re + 2;
    x87_u128_norm(sin_v, sin_e);
    cos_v = c_sum;
    cos_e = 1;
    x87_u128_norm(cos_v, cos_e);
}

// Trap when an instruction hits a case outside the static-depth profile (such as |ST(0)| >= 2^63 on FSINCOS/FPTAN).
static inline void x87_trap() {
#if defined(_MSC_VER) && !defined(__clang__)
    __ud2();
#else
    __builtin_trap();
#endif
}

// FSIN, FCOS and FSINCOS: 64-bit extended precision (precision control does not apply), C2 set when |ST(0)| >= 2^63.
static inline void x87_sincos_core(X87Value a, uint16_t cw, bool& c2, X87Value* sin_out, X87Value* cos_out) {
    c2 = false;
    if (x87_unsupported(a) || x87_is_inf(a)) {
        if (sin_out) *sin_out = x87_indefinite;
        if (cos_out) *cos_out = x87_indefinite;
        return;
    }
    if (x87_is_nan(a)) {
        const X87Value q = x87_quiet(a);
        if (sin_out) *sin_out = q;
        if (cos_out) *cos_out = q;
        return;
    }
    if (x87_is_zero(a)) {
        if (sin_out) *sin_out = a;  // trig zero
        if (cos_out) *cos_out = {x87_integer_bit, 0x3fff};
        return;
    }
    const X87Unpacked u = x87_unpack(a);
    if (u.e >= 63) {  // trig range
        c2 = true;
        if (sin_out) *sin_out = a;
        if (cos_out) *cos_out = sin_out ? x87_indefinite : a;
        return;
    }
    if (u.e <= -65) {
        if (sin_out) *sin_out = x87_round_u128(u.sign, u.e, {u.sig, 0}, cw, false);
        if (cos_out) *cos_out = x87_round_u128(false, -1, {~0ull, ~0ull}, cw);
        return;
    }
    X87U128 r_val, sv, cv;
    int re, se, ce;
    bool r_neg;
    const int q = x87_reduce_pi2(u.sig, u.e, r_val, re, r_neg);
    x87_sincos_reduced(r_val, re, sv, se, cv, ce);
    if (sin_out) {
        const bool neg = u.sign ^ (q >= 2) ^ ((q & 1) == 0 && r_neg);
        *sin_out = (q & 1) ? x87_round_u128(neg, ce, cv, cw) : x87_round_u128(neg, se, sv, cw);
    }
    if (cos_out) {
        const int qc = (q + 1) & 3;
        const bool neg = (qc >= 2) ^ ((qc & 1) == 0 && r_neg);
        *cos_out = (qc & 1) ? x87_round_u128(neg, ce, cv, cw) : x87_round_u128(neg, se, sv, cw);
    }
}

static inline X87Value x87_sin(X87Value a, uint16_t cw, bool& c2) {
    X87Value s{};
    x87_sincos_core(a, cw, c2, &s, nullptr);
    return s;
}

static inline X87Value x87_cos(X87Value a, uint16_t cw, bool& c2) {
    X87Value c{};
    x87_sincos_core(a, cw, c2, nullptr, &c);
    return c;
}

static inline X87Value x87_sincos(X87Value a, X87Value& cos_out, uint16_t cw, bool& c2) {
    X87Value s{};
    x87_sincos_core(a, cw, c2, &s, &cos_out);
    return s;
}

// FPTAN: replaces ST(0) by tan(ST(0)) and pushes 1.0 (or quiet NaN / indefinite on NaN/Inf/unsupported).
static inline X87Value x87_ptan(X87Value a, X87Value& one_out, uint16_t cw, bool& c2) {
    c2 = false;
    if (x87_unsupported(a) || x87_is_inf(a)) { one_out = x87_indefinite; return x87_indefinite; }
    if (x87_is_nan(a)) { one_out = x87_quiet(a); return one_out; }
    one_out = {x87_integer_bit, 0x3fff};  // ptan one
    if (x87_is_zero(a)) return a;
    const X87Unpacked u = x87_unpack(a);
    if (u.e >= 63) { c2 = true; one_out = x87_indefinite; return a; }  // trig range
    if (u.e <= -65) return x87_round_u128(u.sign, u.e, {u.sig, 0}, cw, false);
    X87U128 r_val, sv, cv;
    int re, se, ce;
    bool r_neg;
    const int q = x87_reduce_pi2(u.sig, u.e, r_val, re, r_neg);
    x87_sincos_reduced(r_val, re, sv, se, cv, ce);
    const bool neg = u.sign ^ ((q & 1) != 0) ^ r_neg;
    int te;
    const X87U128 tv = (q & 1) ? x87_u128_div_norm(cv, ce, sv, se, te)
                               : x87_u128_div_norm(sv, se, cv, ce, te);
    return x87_round_u128(neg, te, tv, cw);
}

// tan(k * pi / 16) in Q0.128 for k = 1, 2, 3, and Q0.64 midpoint thresholds for atan reduction.
static const X87U128 x87_tan_k[3] = {
    {0x32ebebc0aa62b00full, 0x7ae9b3a20ca321fbull},
    {0x6a09e667f3bcc908ull, 0xb2fb1366ea957d3eull},
    {0xab0dc155bfcc82f5ull, 0x18a42e56cea7698cull},
};
static const uint64_t x87_atan_thresh[4] = {
    0x1936bb8c5b2da262ull, 0x4da820d5718f6f14ull, 0x88d5b8c841a75c35ull, 0xd218015721427e59ull
};

static inline void x87_atan_unit(X87U128 t_val, int te, X87U128& res, int& re) {
    if (te <= -65) { res = t_val; re = te; return; }
    const uint64_t t_q64 = te == 0 ? ~0ull : (te >= -64 ? (t_val.hi >> (-te - 1)) : 0);
    int k = 0;
    while (k < 4 && t_q64 > x87_atan_thresh[k]) ++k;
    X87U128 u_val;
    int ue;
    bool u_neg = false;
    if (k == 0) {
        u_val = t_val;
        ue = te;
    } else if (k == 4) {
        const X87U128 one_q126{1ull << 62, 0};
        const X87U128 t_q126 = x87_u128_shr(t_val, 1 - te);
        X87U128 num = x87_u128_sub(one_q126, t_q126);
        X87U128 den = x87_u128_add(one_q126, t_q126);
        if (num.hi == 0 && num.lo == 0) { res = x87_pi2; re = -1; return; }
        int ne = 1, de = 1;
        x87_u128_norm(num, ne);
        x87_u128_norm(den, de);
        u_val = x87_u128_div_norm(num, ne, den, de, ue);
        u_neg = true;
    } else {
        const X87U128 t_q128 = x87_u128_shr(t_val, -1 - te);
        const X87U128 c_q128 = x87_tan_k[k - 1];
        X87U128 num;
        if (x87_u128_cmp(t_q128, c_q128) >= 0) {
            num = x87_u128_sub(t_q128, c_q128);
            u_neg = false;
        } else {
            num = x87_u128_sub(c_q128, t_q128);
            u_neg = true;
        }
        if (num.hi == 0 && num.lo == 0) {
            res = x87_pi2;
            re = (k == 1) ? -3 : (k == 2) ? -2 : -1;
            if (k == 3) {
                res = x87_u128_add(x87_u128_shr(x87_pi2, 1), x87_u128_shr(x87_pi2, 2));
                re = -1;
                x87_u128_norm(res, re);
            }
            return;
        }
        const X87U128 tc_q128 = x87_u128_mul_hi(t_q128, c_q128);
        X87U128 den = x87_u128_add({1ull << 62, 0}, x87_u128_shr(tc_q128, 2));
        int ne = -1, de = 1;
        x87_u128_norm(num, ne);
        x87_u128_norm(den, de);
        u_val = x87_u128_div_norm(num, ne, den, de, ue);
    }
    X87U128 u_atan;
    int ua_e;
    if (ue <= -65) {
        u_atan = u_val; ua_e = ue;
    } else {
        X87U128 u2 = x87_u128_mul_hi(u_val, u_val);
        u2 = x87_u128_shr(u2, -2 * ue - 2);
        const X87U128 one_q126{1ull << 62, 0};
        X87U128 sum = one_q126, pow = one_q126;
        for (uint64_t n = 1; n <= 19; ++n) {
            pow = x87_u128_mul_hi(pow, u2);
            const X87U128 term = x87_u128_div_u64(pow, 2 * n + 1);
            if (n & 1) sum = x87_u128_sub(sum, term);
            else sum = x87_u128_add(sum, term);
            if ((term.hi | term.lo) == 0) break;
        }
        u_atan = x87_u128_mul_hi(u_val, sum);
        ua_e = ue + 2;
        x87_u128_norm(u_atan, ua_e);
    }
    if (k == 0) { res = u_atan; re = ua_e; return; }
    const X87U128 pi16 = x87_u128_shr(x87_pi2, 2);
    X87U128 base{0, 0};
    for (int i = 0; i < k; ++i) base = x87_u128_add(base, pi16);
    const X87U128 u_q128 = x87_u128_shr(u_atan, -1 - ua_e);
    res = u_neg ? x87_u128_sub(base, u_q128) : x87_u128_add(base, u_q128);
    re = -1;
    x87_u128_norm(res, re);
}

// FPATAN: atan2(y, x) = atan2(ST(1), ST(0)) rounded to 64 significand bits (precision control does not apply).
static inline X87Value x87_patan(X87Value y, X87Value x, uint16_t cw) {
    X87Value out;
    if (x87_special_operands(y, x, &out)) return out;
    const bool sy = x87_sign(y), sx = x87_sign(x);
    if (x87_is_zero(y)) return sx ? x87_round_u128(sy, 1, x87_pi2, cw) : X87Value{0, static_cast<uint16_t>(sy ? 0x8000 : 0)};
    if (x87_is_inf(y)) {
        if (x87_is_inf(x)) {
            if (!sx) return x87_round_u128(sy, -1, x87_pi2, cw);
            const X87U128 three = x87_u128_add(x87_u128_shr(x87_pi2, 1), x87_u128_shr(x87_pi2, 2));
            return x87_round_u128(sy, 1, three, cw);
        }
        return x87_round_u128(sy, 0, x87_pi2, cw);
    }
    if (x87_is_inf(x)) return sx ? x87_round_u128(sy, 1, x87_pi2, cw) : X87Value{0, static_cast<uint16_t>(sy ? 0x8000 : 0)};
    if (x87_is_zero(x)) return x87_round_u128(sy, 0, x87_pi2, cw);
    const X87Unpacked uy = x87_unpack(y), ux = x87_unpack(x);
    const bool swapped = uy.e > ux.e || (uy.e == ux.e && uy.sig > ux.sig);
    const X87Unpacked num = swapped ? ux : uy, den = swapped ? uy : ux;
    int te;
    const X87U128 t_val = x87_u128_div_norm({num.sig, 0}, num.e, {den.sig, 0}, den.e, te);
    X87U128 av;
    int ae;
    x87_atan_unit(t_val, te, av, ae);
    if (!swapped && !sx) return x87_round_u128(sy, ae, av, cw);
    const X87U128 base = swapped ? x87_u128_shr(x87_pi2, 1) : x87_pi2;
    const X87U128 av_q126 = x87_u128_shr(av, 1 - ae);
    const bool sub = !swapped || !sx;
    const X87U128 res = sub ? x87_u128_sub(base, av_q126) : x87_u128_add(base, av_q126);
    return x87_round_u128(sy, 1, res, cw);
}
