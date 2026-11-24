"""HLSL shared by the gfx1013 pixel and startup translators."""

FLOAT_MINMAX = """// RDNA 2 ISA section 12.7 orders -0 below +0 for min and max.
float minmax_zero32(float a, float b, bool maximum) {
    uint a_bits = asuint(a), b_bits = asuint(b);
    uint either = a_bits | b_bits;
    uint zero = maximum ? (a_bits & b_bits) : either;
    // Keep the selection integer so the compiler cannot merge the two float zeros.
    return asfloat((either & 0x7fffffffu) == 0u ? zero : asuint(maximum ? max(a, b) : min(a, b)));
}
"""
