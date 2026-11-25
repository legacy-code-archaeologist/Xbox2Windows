"""Lower a bounded mesh with runtime geometry words to desktop mesh HLSL.

This reuses the mesh entry model and partitions inputs only at comparisons
between a geometry DWORD and a concrete value. Every reachable partition must
have concrete allocation, masks and indices. The caller binds the declared
number of DWORDs as a raw SRV at t0 and keeps the contents stable for the draw.
The entry profile is two wave64s, triangle output, and no task payload.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from mesh_entry_aot import U32, Wave, decode_mesh, float4


MAX_VARIANTS = 32


@dataclass(frozen=True)
class RuntimeWord:
    index: int


class SplitInput(Exception):
    def __init__(self, index, value):
        self.index = index
        self.value = value


class Geometry:
    def __init__(self, word_count, conditions):
        self.word_count = word_count
        self.equal = {}
        self.excluded = {}
        for index, value, equal in conditions:
            if equal:
                self.equal[index] = value
            else:
                self.excluded.setdefault(index, set()).add(value)

    def __len__(self):
        return self.word_count

    def __getitem__(self, key):
        if isinstance(key, slice):
            if not 0 <= key.start <= key.stop <= self.word_count:
                raise ValueError("Geometry read outside binding")
            return [self[index] for index in range(key.start, key.stop)]
        if not 0 <= key < self.word_count:
            raise ValueError("Geometry read outside binding")
        return self.equal.get(key, RuntimeWord(key))


class RuntimeWave(Wave):
    def lane_spill(self, op, args):
        raise ValueError(f"Unsupported runtime mesh instruction {op}")

    def scalar(self, token):
        value = super().scalar(token)
        if isinstance(value, RuntimeWord):
            raise ValueError(f"Unproven runtime geometry word {value.index} in {token}")
        return value

    def lane_value(self, token, lane):
        value = super().lane_value(token, lane)
        if isinstance(value, RuntimeWord):
            raise ValueError(f"Unproven runtime geometry word {value.index} in {token}")
        return value

    def scalar_op(self, op, args):
        if op == "s_cmp_eq_u32":
            left, right = map(super().scalar, args)
            if isinstance(right, RuntimeWord):
                left, right = right, left
            if isinstance(left, RuntimeWord) and type(right) is int:
                value = right & U32
                if value not in self.geometry.excluded.get(left.index, ()):
                    raise SplitInput(left.index, value)
                self.scc = False
                return
        super().scalar_op(op, args)

    def vector_op(self, op, args, expressions):
        if op not in ("v_cndmask_b32_e32", "v_cndmask_b32_e64"):
            return super().vector_op(op, args, expressions)
        dest, low, high, predicate = args
        mask = self.scalar(predicate)
        concrete_model = super()
        for lane in range(64):
            if not self.exec & (1 << lane):
                continue
            # The unused value may be a defined runtime word. Both sources
            # still go through the existing checks for undefined/pending data.
            values = [concrete_model.lane_value(token, lane) if token.startswith("v")
                      else concrete_model.scalar(token) for token in (low, high)]
            value = values[bool(mask & (1 << lane))]
            if isinstance(value, RuntimeWord):
                raise ValueError(f"Unproven runtime geometry word {value.index} in {op}")
            if (lane, dest) in self.pending_v:
                raise ValueError("Pending vector destination")
            self.v[lane][int(dest[1:])] = value & U32 if isinstance(value, int) else value


@dataclass
class Variant:
    conditions: tuple
    counts: tuple
    expressions: dict
    positions: dict
    attributes: dict
    indices: dict


def run_variant(decoded, geometry, limits, conditions):
    expressions = {}
    first, second = RuntimeWave(0, geometry), RuntimeWave(1, geometry)
    for wave in (first, second):
        wave.run(decoded, expressions)
    if first.barriers != second.barriers or len(first.counts) != 1 or second.counts:
        raise ValueError("Inconsistent mesh synchronization or allocation")
    nv, np = first.counts[0]
    pos = first.positions | second.positions
    attrs = first.attributes | second.attributes
    indices = first.indices | second.indices
    if (not 0 <= nv <= limits[0] or not 0 <= np <= limits[1]
            or set(pos) != set(range(nv)) or set(indices) != set(range(np))
            or any(max(triangle) >= nv for triangle in indices.values())):
        raise ValueError("Conflicting mesh counts, vertices or primitive indices")
    if attrs and attrs.keys() != pos.keys():
        raise ValueError("Incomplete vertex attribute")
    return Variant(conditions, (nv, np), expressions, pos, attrs, indices)


def enumerate_variants(decoded, geometry_word_count, limits):
    # ByteAddressBuffer.Load takes a 32-bit byte offset.
    if type(geometry_word_count) is not int or not 0 < geometry_word_count <= 1 << 30:
        raise ValueError("Invalid runtime geometry word count")
    if len(limits) != 2 or any(type(n) is not int or not 0 < n <= 256 for n in limits):
        raise ValueError("Invalid mesh declaration")
    pending, variants = [()], []
    while pending:
        conditions = pending.pop()
        geometry = Geometry(geometry_word_count, conditions)
        try:
            variants.append(run_variant(decoded, geometry, limits, conditions))
        except SplitInput as split:
            # Replay only after a comparison splits the possible inputs into
            # disjoint cases. Neither branch substitutes a sample for an
            # unknown word. A case may discard an unknown word or compare it
            # again, but a relevant arithmetic/export use must be proven.
            if len(variants) + len(pending) + 2 > MAX_VARIANTS:
                raise ValueError(f"Runtime mesh exceeds {MAX_VARIANTS} input cases") from None
            pending.append((*conditions, (split.index, split.value, False)))
            pending.append((*conditions, (split.index, split.value, True)))
    if not any(variant.conditions for variant in variants):
        raise ValueError("No runtime geometry comparisons")
    if len({bool(variant.attributes) for variant in variants if variant.positions}) > 1:
        raise ValueError("Runtime mesh branches disagree on the vertex signature")
    return variants


def emit_hlsl(variants, limits):
    has_attributes = any(variant.attributes for variant in variants)
    input_words = sorted({index for variant in variants for index, _, _ in variant.conditions})
    lines = [
        "ByteAddressBuffer geometryData : register(t0);",
        "cbuffer Points : register(b0) { float4 pointData[32]; };",
        "cbuffer Matrices : register(b1) { float4 matrixData[32]; };",
        "struct Vertex { float4 position : SV_Position; "
        + ("float4 attr : TEXCOORD0; " if has_attributes else "") + "};",
        '[outputtopology("triangle")]',
        "[numthreads(128,1,1)]",
        "void main(uint id : SV_GroupIndex, "
        f"out vertices Vertex vertices[{limits[0]}], out indices uint3 primitives[{limits[1]}]) {{",
        *(f"  uint word{index} = geometryData.Load({index * 4});" for index in input_words),
        "  uint variant, vertexCount, primitiveCount;",
    ]
    for i, variant in enumerate(variants):
        if i + 1 == len(variants):
            condition = "else"
        else:
            predicate = " && ".join(
                f"word{index} {'==' if equal else '!='} 0x{value:08x}u"
                for index, value, equal in variant.conditions
            )
            condition = f"{'else ' if i else ''}if ({predicate})"
        vertices, primitives = variant.counts
        lines.append(
            f"  {condition} {{ variant = {i}; vertexCount = {vertices}; primitiveCount = {primitives}; }}"
        )
    lines.append("  SetMeshOutputCounts(vertexCount, primitiveCount);")
    for i, variant in enumerate(variants):
        if not variant.positions and not variant.indices:
            continue
        lines.append(f"  if (variant == {i}) {{")
        lines.extend(f"    precise float {name} = {expr};" for expr, name in variant.expressions.items())
        for index, position in sorted(variant.positions.items()):
            lines.append(f"    if (id == {index}) {{")
            lines.append(f"      vertices[{index}].position = {float4(position)};")
            if has_attributes:
                lines.append(f"      vertices[{index}].attr = {float4(variant.attributes[index])};")
            lines.append("    }")
        for index, triangle in sorted(variant.indices.items()):
            indices = ", ".join(map(str, triangle))
            lines.append(f"    if (id == {index}) primitives[{index}] = uint3({indices});")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def lower(decoded, geometry_word_count, limits):
    return emit_hlsl(enumerate_variants(decoded, geometry_word_count, limits), limits)


def translate(blob, geometry_word_count, library):
    decoded, limits = decode_mesh(blob, library)
    return lower(decoded, geometry_word_count, limits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path)
    parser.add_argument("--geometry-word-count", type=int, required=True)
    parser.add_argument("--llvm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".hlsl" or args.output.resolve() == args.container.resolve():
        parser.error("Expected a separate HLSL output")
    try:
        hlsl = translate(args.container.read_bytes(), args.geometry_word_count, args.llvm)
    except (ValueError, KeyError) as error:
        parser.exit(1, f"mesh_dynamic_aot: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(hlsl, encoding="ascii")
