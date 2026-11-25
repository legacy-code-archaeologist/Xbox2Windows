"""HLSL shared by the gfx1013 pixel and startup translators."""

# RDNA 2 ISA section 12.7 settles both operand orders for signed zeros:
#   V_MIN_F32: S0 = +0, S1 = -0 -> S1;  S0 = -0, S1 = +0 -> S0  (min returns -0 if either operand is -0)
#   V_MAX_F32: S0 = +0, S1 = -0 -> S0;  S0 = -0, S1 = +0 -> S1  (max returns +0 when the zeros differ)
# Equal zeros take the general branch, so max(-0, -0) stays -0. For two zeros, the OR of the bit
# patterns gives the min result and the AND gives the max result, in either order. V_MIN3_F32,
# V_MAX3_F32 and V_MED3_F32 are defined through nested V_MIN_F32/V_MAX_F32, so chaining this helper
# applies the same rule. The emitted HLSL below is unchanged by this note.
FLOAT_MINMAX = """// RDNA 2 ISA section 12.7 orders -0 below +0 for min and max.
float minmax_zero32(float a, float b, bool maximum) {
    uint a_bits = asuint(a), b_bits = asuint(b);
    uint either = a_bits | b_bits;
    uint zero = maximum ? (a_bits & b_bits) : either;
    // Keep the selection integer so the compiler cannot merge the two float zeros.
    return asfloat((either & 0x7fffffffu) == 0u ? zero : asuint(maximum ? max(a, b) : min(a, b)));
}
"""
