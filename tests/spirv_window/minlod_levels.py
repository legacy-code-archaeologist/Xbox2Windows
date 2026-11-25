"""Check which mips a MinLod-clamped sample reads at exact and fractional clamps.

The texture has four mips of distinct solid colors and the coordinate is constant,
so the implicit LOD is the clamp itself. Each clamp runs as a translated SPIR-V
shader and as a direct HLSL control on the same adapter.

Point filtering must read the exact integer mip. At fractional clamps, the
translated and control shaders must agree on the nearest or observed lower mip.
With MIN_MAG_POINT_MIP_LINEAR the sample blends the two mips around the clamp. Adapters quantize
that blend weight to 1/256 steps, rounding or truncating, so the result is checked
against the exact blend within BLEND_TOLERANCE, and the translated shader must match
the control bit for bit on the same adapter. The raw bits are printed per adapter.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

# (clamp, accepted exact mips), listing nearest first.
EXACT = (("0", (0,)), ("1", (1,)), ("2", (2,)), ("3", (3,)))
# Probe one 1/256 step below both observed point-filter transitions (1.5 and 2).
FRACTIONAL = (("1.4375", (1,)), ("1.49609375", (1, 2)), ("1.5", (2, 1)), ("1.5625", (2, 1)),
              ("1.99609375", (2, 1)), ("2", (2,)), ("3.75", (3,)))
WRONG = ("1", 2)  # negative control: a clamp of 1 must not read mip 2
WRONG_FRACTIONAL = ("1.5625", 0)  # neither observed rounding rule selects mip zero
# Clamps for the mip-linear sampler: exact steps, a weight between 1/256 steps (1.3)
# and weights at and below 1/256 near the mip 1 boundary.
BLENDS = ("1", "1.25", "1.5", "1.75", "1.3", "1.00390625", "1.001953125", "2.5", "3.75")
BLEND_TOLERANCE = "0.0078125"  # two 1/256 steps of the blend weight
WRONG_BLEND = ("1.25", "1.5")  # negative control: a 1.25 clamp must not match the 1.5 blend
SKIP = 77


def draw(renderer, adapter, operation, dxil, level):
    result = subprocess.run([renderer, adapter, operation, dxil, str(level)],
                            capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def blend(renderer, adapter, dxil, clamp):
    result = subprocess.run([renderer, adapter, "--check-pixel-mip-blend", dxil, clamp, BLEND_TOLERANCE],
                            capture_output=True, text=True)
    output = result.stdout + result.stderr
    bits = next((line.split("=", 1)[1] for line in output.splitlines() if line.startswith("first_pixel_bits=")), None)
    return result.returncode, output, bits


def pixel_mismatch(code, output):
    return code == 1 and "pixel_mismatch x=0 y=0" in output


def point_mip(renderer, adapter, operation, dxil, clamp, levels, kind):
    for level in levels:
        code, output = draw(renderer, adapter, operation, dxil, level)
        print(f"{kind} clamp={clamp} mip={level}\n{output}", end="", flush=True)
        if code == SKIP:
            return None
        if not code and "pixels=64 channels=256 max_absolute_error=0" in output:
            return level
        if not pixel_mismatch(code, output):
            raise RuntimeError(f"{kind} clamp {clamp} did not produce a valid pixel comparison")
    raise RuntimeError(f"{kind} clamp {clamp} did not read any accepted mip {levels}")


def main(assembler, validator, cross, dxc, renderer, adapter, *, fractional=False,
         strict_rounding=False):
    # Issue #809 records the fractional point-minimum discrepancy on older WARP.
    shader_model, profile = ("66", "ps_6_6") if fractional else ("60", "ps_6_0")
    operation = "--check-pixel-mip-clamp" if fractional else "--check-pixel-mip-exact"
    cases = FRACTIONAL if fractional else EXACT
    wrong_point = WRONG_FRACTIONAL if fractional else WRONG
    print(f"MinLod shader profile {profile}", flush=True)
    here = Path(__file__).parent
    template = (here / "minlod.spvasm").read_text(encoding="ascii")
    anchor = "%one = OpConstant %float 1\n"
    if anchor not in template or "@sample@" not in template:
        raise RuntimeError("minlod.spvasm no longer has the expected constant and sample markers")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)

        def translated(clamp):
            asm, module, hlsl, dxil = (path / f"spirv.{ext}" for ext in ("spvasm", "spv", "hlsl", "dxil"))
            source = template.replace(anchor, f"{anchor}%clamp = OpConstant %float {clamp}\n")
            asm.write_text(source.replace("@sample@", "OpImageSampleImplicitLod %v4 %sampled %uv MinLod %clamp"),
                           encoding="ascii")
            subprocess.run([assembler, "--target-env", "vulkan1.2", asm, "-o", module], check=True)
            subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
            result = subprocess.run([cross, module, "--hlsl", "--shader-model", shader_model, "--output", hlsl],
                                    capture_output=True, text=True)
            if "MinLod texture operand not supported in HLSL." in result.stderr:
                return None
            if result.returncode:
                print(result.stderr, file=sys.stderr, flush=True)
                result.check_returncode()
            subprocess.run([dxc, "-T", profile, "-E", "main", "-Fo", dxil, hlsl], check=True)
            return dxil

        def control(clamp):
            dxil = path / "control.dxil"
            subprocess.run([dxc, "-T", profile, "-E", "main", f"-DCLAMP={clamp}", "-Fo", dxil,
                            here / "minlod_control.hlsl"], check=True)
            return dxil

        selected, blended = {}, {}
        for kind, build in (("translated", translated), ("control", control)):
            for clamp, levels in cases:
                dxil = build(clamp)
                if dxil is None:
                    print("skip: SPIRV-Cross lacks MinLod support; run tools/setup-spirv-tools.ps1")
                    return SKIP
                if strict_rounding and clamp != "1.5":
                    levels = levels[:1]
                level = point_mip(renderer, adapter, operation, dxil, clamp, levels, kind)
                if level is None:
                    return SKIP
                if fractional:
                    print(f"{kind} clamp={clamp} selected_mip={level}", flush=True)
                    if kind == "translated":
                        selected[clamp] = level
                    elif selected[clamp] != level:
                        raise RuntimeError(f"Clamp {clamp} read mip {selected[clamp]} translated, "
                                           f"mip {level} in the HLSL control")
            clamp, level = wrong_point
            code, output = draw(renderer, adapter, operation, build(clamp), level)
            print(f"{kind} negative clamp={clamp} mip={level}\n{output}", end="", flush=True)
            if not pixel_mismatch(code, output):
                raise RuntimeError(f"{kind} negative control did not fail at pixel (0, 0)")
            if fractional:
                continue
            for clamp in BLENDS:
                code, output, bits = blend(renderer, adapter, build(clamp), clamp)
                print(f"{kind} blend clamp={clamp}\n{output}", end="", flush=True)
                if code or bits is None or "pixels=64 channels=256 max_absolute_error=" not in output:
                    raise RuntimeError(f"{kind} clamp {clamp} did not blend within {BLEND_TOLERANCE}")
                blended[kind, clamp] = bits
            clamp, wrong = WRONG_BLEND
            code, output, _ = blend(renderer, adapter, build(clamp), wrong)
            print(f"{kind} negative blend clamp={clamp} expected={wrong}\n{output}", end="", flush=True)
            if not pixel_mismatch(code, output):
                raise RuntimeError(f"{kind} negative blend control did not fail at pixel (0, 0)")
        if not fractional:
            differing = [clamp for clamp in BLENDS if blended["translated", clamp] != blended["control", clamp]]
            if differing:
                raise RuntimeError(f"Translated blend bits differ from the HLSL control at clamps {', '.join(differing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
