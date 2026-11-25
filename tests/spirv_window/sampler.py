"""Validate the original and split sampler module, then optionally run the D3D12 readback."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from spirv_sampler_views import split_sampler_views


def check_sampler_rewrite(source):
    # Use the fragment variant: its ordinary sample is the implicit-LOD form.
    sample = "%raw = OpImageSampleImplicitLod %vec4 %si %uv"
    comparison = "%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod"
    second = comparison.replace("%cmp =", "%cmp2 =")
    depth_gather = "%cmp = OpImageDrefGather %vec4 %si %uv %half"
    ordinary_gather = "%raw = OpImageGather %vec4 %si %uv %zero"
    for variant, ordinary, comparisons in (
        (source.replace(comparison, comparison + "\n" + second), sample, (comparison, second)),
        (source.replace(comparison, depth_gather).replace(
            "%cw = OpBitcast %uint %cmp",
            "%depth = OpCompositeExtract %float %cmp 0\n%cw = OpBitcast %uint %depth"),
         sample, (depth_gather,)),
        (source.replace(sample, ordinary_gather), ordinary_gather, (comparison,)),
    ):
        original_lines = variant.splitlines()
        rewritten_lines = split_sampler_views(variant).splitlines()
        assert ordinary in rewritten_lines, ordinary
        for instruction in comparisons:
            name = f"%bridge_cmp_{original_lines.index(instruction)}"
            assert f"{name}_ptr = OpAccessChain %_ptr_UC_smp %bridge_cmp %i" in rewritten_lines, name
            expected = instruction.replace(" %si ", f" {name}_image ")
            assert expected in rewritten_lines, expected
            assert instruction not in rewritten_lines, instruction

    binding = "OpDecorate %samplers Binding 1"
    table = "%samplers = OpVariable %_ptr_UC_rt_smp UniformConstant"
    foreign = source.replace(' "main" %texture %samplers %out',
                             ' "main" %texture %samplers %out %foreign')
    foreign = foreign.replace(binding, binding + "\nOpDecorate %foreign DescriptorSet 0\n"
                              "OpDecorate %foreign Binding 2")
    foreign = foreign.replace(table, table + "\n%foreign = OpVariable %_ptr_UC_rt_smp UniformConstant")
    foreign = foreign.replace(comparison,
                              "%foreign_p = OpAccessChain %_ptr_UC_smp %foreign %i\n"
                              "%foreign_s = OpLoad %smp %foreign_p\n"
                              "%foreign_si = OpSampledImage %pair %im %foreign_s\n"
                              + comparison.replace(" %si ", " %foreign_si "))
    # A second table used only for comparisons has no ordinary view to separate from: nothing changes.
    assert split_sampler_views(foreign) == foreign
    for unsupported, reason in (
        (source.replace(binding, binding + "\n%group = OpDecorationGroup\n"
                        "OpDecorate %group RelaxedPrecision\nOpGroupDecorate %group %raw"),
         "unsupported indirect descriptor decorations"),
        (source.replace("%out", "%bridge_cmp"),
         "comparison sampler view requires free descriptor set 3 and bridge_cmp IDs"),
        (source.replace(binding, "").replace("OpDecorate %samplers DescriptorSet 0\n", ""),
         "sampler table requires one DescriptorSet and one Binding decoration"),
        (source.replace(binding, binding + "\nOpDecorate %samplers Binding 2"),
         "sampler table requires one DescriptorSet and one Binding decoration"),
        (source.replace(binding, binding + "\nOpDecorate %samplers NonUniformEXT"),
         "unsupported sampler table declaration"),
        (source.replace("%s = OpLoad %smp %p", "%s = OpLoad %img %p"),
         "unsupported depth-reference sampler load"),
        (source.replace("%si = OpSampledImage %pair %im %s", "%si = OpCopyObject %pair %im"),
         "unsupported depth-reference sampled image"),
        (source.replace("%_ptr_UC_smp = OpTypePointer UniformConstant %smp",
                        "%_ptr_UC_smp = OpTypePointer UniformConstant %img"),
         "unsupported depth-reference sampler provenance"),
        (source.replace("OpDecorate %samplers DescriptorSet 0", "OpDecorate %samplers DescriptorSet 3"),
         "comparison sampler view requires free descriptor set 3 and bridge_cmp IDs"),
        (source.replace(binding, binding + "\nOpDecorate %si NonUniformEXT"),
         "unsupported decorated sampler path"),
    ):
        try:
            split_sampler_views(unsupported)
        except ValueError as error:
            assert str(error) == reason, error
        else:
            raise AssertionError(f"accepted invalid sampler profile: {reason}")


def rename(text, names):
    for old, new in names.items():
        text = re.sub(re.escape(old) + r"(?!\w)", new, text)
    return text


def renamed_table(text):
    """The same module with another set of IDs, a bounds-checked chain and a table that is not "%samplers"."""
    names = {"%samplers": "%heap_a", "%_ptr_UC_rt_smp": "%heap_ptr", "%samparr": "%heap_type",
             "%_ptr_UC_smp": "%slot_ptr", "%smp": "%sampler_t", "%p": "%slot"}
    return rename(text, names).replace("OpAccessChain %slot_ptr", "OpInBoundsAccessChain %slot_ptr")


def emit_checked(source, folder, assembler, validator, cross):
    """Assemble, validate and cross-compile a module; return the HLSL text."""
    spvasm, module, hlsl = folder / "t.spvasm", folder / "t.spv", folder / "t.hlsl"
    spvasm.write_text(source)
    subprocess.run([assembler, "--target-env", "vulkan1.2", spvasm, "-o", module], check=True)
    subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
    subprocess.run([cross, module, "--hlsl", "--shader-model", "60", "--output", hlsl], check=True)
    return hlsl.read_text()


def check_sampler_tables(source, folder, assembler, validator, cross):
    """Any sampler table variable, in any set but 3, of fixed or runtime size, can be split."""
    def split_lines(text):
        return split_sampler_views(text).splitlines()

    comparison = "%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod"
    # A renamed table through an in-bounds chain keeps its views at set 0, binding 1 and 3, 1.
    lines = split_lines(renamed_table(source))
    index = renamed_table(source).splitlines().index(comparison)
    name = f"%bridge_cmp_{index}"
    assert f"{name}_ptr = OpInBoundsAccessChain %slot_ptr %bridge_cmp %i" in lines, lines
    assert f"{name}_value = OpLoad %sampler_t {name}_ptr" in lines
    assert "%bridge_cmp = OpVariable %heap_ptr UniformConstant" in lines
    assert "OpDecorate %bridge_cmp DescriptorSet 3" in lines and "OpDecorate %bridge_cmp Binding 1" in lines
    assert "OpDecorate %heap_a Binding 1" in lines and "%heap_a = OpVariable %heap_ptr UniformConstant" in lines
    assert "%bridge_cmp" in next(line for line in lines if line.startswith("OpEntryPoint")).split()

    # A table at another set and binding, of runtime size, moves its comparison view to set 3 at the same
    # binding, and SPIRV-Cross puts the two views in separate register spaces.
    moved = renamed_table(source).replace("DescriptorSet 0\nOpDecorate %heap_a Binding 1",
                                          "DescriptorSet 2\nOpDecorate %heap_a Binding 7")
    moved = moved.replace("%heap_type = OpTypeArray %sampler_t %two", "%heap_type = OpTypeRuntimeArray %sampler_t")
    moved = moved.replace("OpCapability Shader", "OpCapability Shader\nOpCapability RuntimeDescriptorArray")
    assert "OpDecorate %heap_a DescriptorSet 2" in moved
    lines = split_lines(moved)
    assert "OpDecorate %bridge_cmp DescriptorSet 3" in lines and "OpDecorate %bridge_cmp Binding 7" in lines
    hlsl = emit_checked(split_sampler_views(moved), folder, assembler, validator, cross)
    assert re.search(r"SamplerComparisonState \w+\[\] : register\(s7, space3\)", hlsl), hlsl
    assert re.search(r"SamplerState \w+\[\] : register\(s7, space2\)", hlsl), hlsl

    # Two tables, each with both kinds of use, get one bridge each at their own binding. A third that is only
    # sampled in the ordinary way, and a fourth only compared, get none.
    extra = (
        "%p2 = OpAccessChain %_ptr_UC_smp %second %i\n%s2 = OpLoad %smp %p2\n%si2 = OpSampledImage %pair %im %s2\n"
        "%raw2 = OpImageSampleImplicitLod %vec4 %si2 %uv\n"
        "%cmp2 = OpImageSampleDrefExplicitLod %float %si2 %uv %half Lod %lod\n"
        "%p3 = OpAccessChain %_ptr_UC_smp %third %i\n%s3 = OpLoad %smp %p3\n%si3 = OpSampledImage %pair %im %s3\n"
        "%raw3 = OpImageSampleImplicitLod %vec4 %si3 %uv\n"
        "%p4 = OpAccessChain %_ptr_UC_smp %fourth %i\n%s4 = OpLoad %smp %p4\n%si4 = OpSampledImage %pair %im %s4\n"
        "%cmp4 = OpImageSampleDrefExplicitLod %float %si4 %uv %half Lod %lod\n")
    many = source.replace("OpStore %rp %rw", extra + "OpStore %rp %rw")
    many = many.replace("%texture %samplers %out", "%texture %samplers %out %second %third %fourth")
    many = many.replace("OpImageSampleExplicitLod %vec4 %si %uv Lod %lod", "OpImageSampleImplicitLod %vec4 %si %uv")
    many = many.replace("OpDecorate %out DescriptorSet 0", "".join(
        f"OpDecorate %{n} DescriptorSet 0\nOpDecorate %{n} Binding {b}\n"
        for n, b in (("second", 3), ("third", 5), ("fourth", 7))) + "OpDecorate %out DescriptorSet 0")
    many = many.replace("%out = OpVariable %buf StorageBuffer", "%out = OpVariable %buf StorageBuffer\n" + "".join(
        f"%{n} = OpVariable %_ptr_UC_rt_smp UniformConstant\n" for n in ("second", "third", "fourth")))
    many = many.replace("GLCompute", "Fragment").replace(" %out %tid", " %out").replace("LocalSize 1 1 1", "OriginUpperLeft")
    many = many.replace("OpDecorate %tid BuiltIn WorkgroupId", "").replace("%tid = OpVariable %input Input", "")
    many = many.replace("%xyzval = OpLoad %xyz %tid", "").replace("OpCompositeExtract %uint %xyzval 0", "OpCopyObject %uint %zero")
    lines = split_lines(many)
    bridges = [line for line in lines if re.match(r"%bridge_cmp(_table_\d+)? = ", line)]
    assert bridges == ["%bridge_cmp = OpVariable %_ptr_UC_rt_smp UniformConstant",
                       "%bridge_cmp_table_1 = OpVariable %_ptr_UC_rt_smp UniformConstant"], bridges
    for decoration in ("OpDecorate %bridge_cmp Binding 1", "OpDecorate %bridge_cmp_table_1 Binding 3"):
        assert decoration in lines, decoration
    assert not any("Binding 5" in line and "bridge" in line for line in lines)
    assert not any("Binding 7" in line and "bridge" in line for line in lines)
    assert sum("%bridge_cmp_table_1 %i" in line for line in lines) == 1
    assert sum("%bridge_cmp %i" in line for line in lines) == 1
    assert "%cmp4 = OpImageSampleDrefExplicitLod %float %si4 %uv %half Lod %lod" in lines
    emit_checked(split_sampler_views(many), folder, assembler, validator, cross)
    # Every view lands in set 3, so the register ranges of two split tables must not overlap, whatever their
    # original sets: a size-2 table at binding 1 covers s1..s2, and a runtime-sized one has no end.
    def second_at(binding, runtime=False):
        text = many.replace("OpDecorate %second DescriptorSet 0", "OpDecorate %second DescriptorSet 1").replace(
            "OpDecorate %second Binding 3", f"OpDecorate %second Binding {binding}")
        if runtime:
            text = text.replace("%second = OpVariable %_ptr_UC_rt_smp UniformConstant",
                                "%second = OpVariable %_ptr_UC_rtarr UniformConstant").replace(
                "%samplers = OpVariable", "%rtarr = OpTypeRuntimeArray %smp\n%_ptr_UC_rtarr = OpTypePointer UniformConstant %rtarr\n"
                "%samplers = OpVariable").replace("OpCapability Shader", "OpCapability Shader\nOpCapability RuntimeDescriptorArray")
        return text

    for binding, runtime, clash in ((1, False, True), (2, False, True), (3, False, False), (0, False, True),
                                    (4, True, False), (0, True, True)):
        text = second_at(binding, runtime)
        try:
            split_sampler_views(text)
        except ValueError as error:
            assert clash and "overlap in the set 3 register range" in str(error), (binding, runtime, error)
        else:
            assert not clash, (binding, runtime)
    # The reserved set is found from the parsed decoration, whatever the spacing or leading zeros.
    for spelled in ("OpDecorate  %samplers   DescriptorSet 3", "\tOpDecorate %samplers DescriptorSet 03"):
        try:
            split_sampler_views(source.replace("OpDecorate %samplers DescriptorSet 0", spelled))
        except ValueError as error:
            assert "free descriptor set 3" in str(error), error
        else:
            raise AssertionError("accepted set 3 spelled " + repr(spelled))
    # A comment may follow any declaration, and a semicolon in a name is not a comment.
    named = source.replace('"main" %texture', '"main;x" %texture')
    commented = named.replace("%smp = OpTypeSampler", "%smp = OpTypeSampler ; sampler type").replace(
        "%samplers %out\n", "%samplers %out ; interface\n")
    assert commented.count(";") == 3 + source.count(";"), "expected three semicolons added"
    assert split_lines(commented) == split_lines(named), "a comment changed the result"
    assert any(line.startswith("OpEntryPoint") and line.endswith("%out %bridge_cmp") for line in split_lines(commented))
    emit_checked(split_sampler_views(commented), folder, assembler, validator, cross)
    # A string that spans lines keeps its closing quote even when a continuation line holds a semicolon.
    multiline = source.replace("OpDecorate %texture DescriptorSet 0",
                               '%text = OpString "first line\nint tmp = 0; }"\nOpDecorate %texture DescriptorSet 0', 1)
    assert multiline != source
    assert 'int tmp = 0; }"' in split_sampler_views(multiline).splitlines()
    emit_checked(split_sampler_views(multiline), folder, assembler, validator, cross)
    # A sparse ordinary sample is still an ordinary use of the table (text-level: no residency capability here).
    sparse = source.replace("%raw = OpImageSampleImplicitLod %vec4 %si %uv", "%raw = OpImageSparseSampleImplicitLod %vec4 %si %uv")
    assert sparse != source and any(line.startswith("OpDecorate %bridge_cmp ") for line in split_lines(sparse))
    # A variable that is not an array of samplers is not a sampler table: the module comes back unchanged.
    not_samplers = source.replace("%samparr = OpTypeArray %smp %two", "%samparr = OpTypeArray %img %two")
    assert split_sampler_views(not_samplers) == not_samplers


def check_query_lod(source, folder, assembler, validator, cross):
    """A level-of-detail query is an ordinary use: the table keeps a SamplerState for it."""
    sample = "%raw = OpImageSampleImplicitLod %vec4 %si %uv"
    query = "%lodv = OpImageQueryLod %vec2 %si %uv\n%raw = OpVectorShuffle %vec4 %lodv %lodv 0 1 0 1"
    module = source.replace(sample, query).replace("OpCapability Shader", "OpCapability Shader\nOpCapability ImageQuery")
    assert module != source and "OpImageSample" not in module.replace("OpImageSampleDref", "")
    lines = split_sampler_views(module).splitlines()
    assert "%bridge_cmp = OpVariable %_ptr_UC_rt_smp UniformConstant" in lines, lines
    assert "%lodv = OpImageQueryLod %vec2 %si %uv" in lines
    # A query whose sampled image is not a direct OpSampledImage may reach any table, as for a sample.
    copied = module.replace("%lodv = OpImageQueryLod %vec2 %si %uv",
                            "%sc = OpCopyObject %pair %si\n%lodv = OpImageQueryLod %vec2 %sc %uv")
    assert any(line.startswith("%bridge_cmp = ") for line in split_sampler_views(copied).splitlines())
    hlsl = emit_checked(split_sampler_views(module), folder, assembler, validator, cross)
    call = re.search(r"CalculateLevelOfDetail\w*\(\s*(\w+)\[", hlsl)
    assert call, hlsl
    assert re.search(rf"SamplerState {call.group(1)}\[", hlsl), hlsl
    assert re.search(r"SamplerComparisonState \w+\[2\] : register\(s1, space3\)", hlsl), hlsl
    return folder / "query.hlsl", hlsl


def emit(folder, assembler, validator, cross, pixel=False, variant=None):
    original = Path(__file__).with_name("sampler.spvasm")
    if pixel:
        # Same resource flow with implicit LOD in the fragment stage.
        text = original.read_text().replace('GLCompute', 'Fragment').replace(' %out %tid', ' %out')
        text = text.replace('LocalSize 1 1 1', 'OriginUpperLeft')
        text = text.replace('OpDecorate %tid BuiltIn WorkgroupId', '').replace('%tid = OpVariable %input Input', '')
        text = text.replace('%xyzval = OpLoad %xyz %tid', '').replace('OpCompositeExtract %uint %xyzval 0', 'OpCopyObject %uint %zero')
        text = text.replace('OpImageSampleExplicitLod %vec4 %si %uv Lod %lod', 'OpImageSampleImplicitLod %vec4 %si %uv')
        original = folder / "pixel.spvasm"
        original.write_text(text)
    source = original.read_text()
    if variant == "renamed":
        source = renamed_table(source)
        original = folder / "renamed.spvasm"
        original.write_text(source)
    split = folder / "split.spvasm"
    if pixel and variant is None:
        check_sampler_rewrite(source)
        check_sampler_tables(source, folder, assembler, validator, cross)
    for unsupported in (source.replace("DescriptorSet 0", "DescriptorSet 3", 1),
                        source.replace("OpDecorate %out Binding 0",
                                       f"OpDecorate {'%slot' if variant else '%p'} NonUniformEXT")):
        try:
            split_sampler_views(unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an unsupported descriptor layout")
    split.write_text(split_sampler_views(source))
    module = folder / "shader.spv"
    for source in (original, split):
        subprocess.run([assembler, "--target-env", "vulkan1.2", source, "-o", module], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
    hlsl = folder / "shader.hlsl"
    subprocess.run([cross, module, "--hlsl", "--shader-model", "60", "--output", hlsl], check=True)
    text = hlsl.read_text()
    for token in ("SamplerState ", "SamplerComparisonState ", (".Sample(" if pixel else ".SampleLevel("), ".SampleCmpLevelZero("):
        if token not in text:
            raise ValueError(f"missing {token} in emitted HLSL")
    return hlsl


def check_missing_stores(folder, compiler, renderer, adapter):
    shader = Path(__file__).resolve().parents[1] / "sampler_index_control.hlsl"
    dxil = folder / "missing-stores.dxil"
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-D", "OMIT_STORES_ON_ORDINARY_SWAP",
                    "-Fo", dxil, shader], check=True)
    run = subprocess.run([renderer, adapter, "--spirv-sampler-views", dxil], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr, end="")
        return 77
    differences = [line for line in run.stdout.splitlines() if re.match(r"sampler swap \d+ word ", line)]
    expected = ["sampler swap 1 word 2: expected=00000000 actual=c0400000",
                "sampler swap 1 word 3: expected=3f800000 actual=c0800000"]
    if run.returncode != 1 or differences != expected or "Split sampler readback mismatch" not in run.stderr:
        raise AssertionError(f"missing comparison stores: exit {run.returncode}\n{run.stdout}{run.stderr}")
    print("missing comparison stores detected:", *differences, sep="\n")
    return 0


if __name__ == "__main__":
    status = 0
    with tempfile.TemporaryDirectory() as tmp:
        for variant in (None, "renamed"):
            folder = Path(tmp) / (variant or "plain")
            folder.mkdir()
            pixel = emit(folder, *sys.argv[1:4], pixel=True, variant=variant)
            if len(sys.argv) > 4:
                subprocess.run([sys.argv[4], "-T", "ps_6_0", "-E", "main", "-Fo", pixel.with_suffix(".dxil"), pixel], check=True)
            if variant is None:
                query, text = check_query_lod((folder / "pixel.spvasm").read_text(), folder, *sys.argv[1:4])
                if len(sys.argv) > 4:
                    query.write_text(text)
                    subprocess.run([sys.argv[4], "-T", "ps_6_0", "-E", "main", "-Fo", query.with_suffix(".dxil"), query], check=True)
            hlsl = emit(folder, *sys.argv[1:4], variant=variant)
            if len(sys.argv) > 4:
                dxil = hlsl.with_suffix(".dxil")
                subprocess.run([sys.argv[4], "-T", "cs_6_0", "-E", "main", "-Fo", dxil, hlsl], check=True)
                # The renamed table is read back exactly as the original is: same slots, same expected words.
                print(f"sampler table: {variant or 'plain'}", flush=True)
                result = subprocess.run([sys.argv[5], sys.argv[6], "--spirv-sampler-views", dxil]).returncode
                if result and (result != 77 or status == 0):
                    status = result
        if len(sys.argv) > 4 and status == 0:
            status = check_missing_stores(Path(tmp), *sys.argv[4:7])
    print("sampler views: validated plain and renamed SPIR-V tables and the level-of-detail query")
    sys.exit(status)
