"""Tests for tools/startup/triage_run.py using synthetic logs and negative controls."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_STARTUP = REPO_ROOT / "tools" / "startup"
sys.path.insert(0, str(TOOLS_STARTUP))
import triage_run  # noqa: E402


SYNTHETIC_STDERR = """\
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description res{dim=1 fmt=0 size=4096x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0} view{dim=11 fmt=0 map=0x1688 raw=55667788,11223344,00000000,00000000,00000000,00000000 (acceleration structure views are not supported)} dest=0x00007ff712340000 dest_in_view_heap=1 address=0x00000180deadbeef address_readable=0
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description res{dim=1 fmt=0 size=8192x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0} view{dim=11 fmt=0 map=0x1688 raw=55667788,11223344,00000000,00000000,00000000,00000000 (acceleration structure views are not supported)} dest=0x00007ff712340080 dest_in_view_heap=1 address=0x00000180cafebabe address_readable=1
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=destination_outside_view_heaps res{dim=3 fmt=98 size=128x128x2 mips=6 layout=0x105 flags=0x300000 align=65536 samples=1/0} view{dim=5 fmt=98 map=0x1688 mostmip=1 mips=5 first=1 count=1 plane=0 clamp=0.5} dest=0x00007ff712341001 dest_in_view_heap=0 address=0x0000018000007000 address_readable=1
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unreadable_resource res{unreadable} view{unreadable} dest=0x1000 dest_in_view_heap=1 address=0x0 address_readable=0
x2w-noop d3d12_xs!Device::slot44 rcx=0x0000018011112222 rdx=0x0000018033334444 r8=0x0 r9=0x0 ret=sample_app.exe+0x1234 exit=0xe257002c
x2w-trap d3d12_xs!Device::slot99 rcx=0x0000018011112222 rdx=0x0 r8=0x0 r9=0x0 ret=C:\\Users\\alice\\secret\\game.exe+0x4560 exit=0xe2570063
x2w-d3d12 Device::slot10(CreateGraphicsPipelineState) desc=[00 01] riid={1234} -> trap
x2w-d3d12-debug [Device 44] sev=2 id=999 Failed reading C:\\Users\\alice\\secret\\game.exe at 0x00007ff7deadbeef
x2w-dstorage   options=0x1 dest=0x0000018044440000 dest_size=0x1000 source=0x1 offset=0x0 size=0x100 -> zlib failed
x2w-dstorage   options=0x1 dest=0x0000018044441000 dest_size=0x1000 source=0x1 offset=0x200 size=0x100 -> zlib failed
x2w-dstorage   options=0x1 dest=0x0000018044442000 dest_size=0x1000 source=0x1 offset=0x400 size=0x100 -> done
x2w-dstorage Factory::OpenFile path="C:\\Users\\alice\\secret\\data.pak" riid={abcd} -> fail (err 2)
x2w-dstorage StatusArray::GetHResult index=0 -> 0x80004005
x2w-dstorage StatusArray::GetHResult index=1 -> 0x00000000
x2w-runtime 12345678::slot5(0x1, 0x2, 0x3 | 0x4, 0x5) -> 0xffffffff80004005 from +0x100
x2w-runtime 12345678::slot5(0x1, 0x2, 0x3 | 0x4, 0x5) -> 0x80004005 from +0x104
x2w-runtime 12345678::slot6(0x1, 0x2, 0x3 | 0x4, 0x5) -> 0x0 from +0x108
x2w-runtime persistent local storage: local object instead of 0x80070005
"""


def check_aggregation_and_redaction(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logdir = root / "logs"
        logdir.mkdir()
        (logdir / "stderr.txt").write_text(SYNTHETIC_STDERR, encoding="utf-8")

        cache_dir = root / "shader_cache"
        cache_dir.mkdir()
        (cache_dir / "0000000000000001.fail").write_text(
            "image_load_mip tfe\n", encoding="utf-8"
        )
        (cache_dir / "0000000000000002.fail").write_text(
            "image_load_mip tfe\n", encoding="utf-8"
        )
        (cache_dir / "0000000000000003.fail").write_text(
            "fxc ps: /home/alice/secret/shader.hlsl(1,1): error X3000 at 0x00007ff712345678\n",
            encoding="utf-8",
        )

        report = triage_run.triage_run(logdir, shader_cache=cache_dir, max_items=16)

        expected_snippets = [
            "Log files scanned: `1`",
            "| 2 | unsupported_description | 1 | 0 | 11 | 0 | 0x1 | 0x0 |",
            "| 1 | destination_outside_view_heaps | 3 | 98 | 5 | 98 | 0x105 | 0x300000 |",
            "| 1 | unreadable_resource | unreadable | unreadable | unreadable | unreadable | unreadable | unreadable |",
            "| 1 | `x2w-noop Device::slot44` |",
            "| 1 | `Device::slot10(CreateGraphicsPipelineState)` |",
            "| 2 | zlib failed |",
            "| 1 | 2 | 999 |",
            "| 1 | OpenFile: fail (err 2) |",
            "| 1 | `0x80004005` |",
            "| 2 | `slot5` | `0x80004005` |",
            "| 1 | `persistent_local_storage` | `0x80070005` |",
            "| 2 | `image_load_mip tfe` |",
            f"| 1 | `{triage_run.OTHER}` |",
        ]
        for snippet in expected_snippets:
            if snippet not in report:
                failures.append(f"missing expected summary line: {snippet!r}")

        # Negative controls: sensitive paths, executables, and 64-bit addresses must never leak.
        forbidden = [
            "alice",
            "game.exe",
            "sample_app.exe",
            "C:\\Users",
            "/home/alice",
            "0x00007ff712340000",
            "0x00000180deadbeef",
            "0x00007ff7deadbeef",
            "0x00007ff712345678",
            "data.pak",
        ]
        for secret in forbidden:
            if secret in report:
                failures.append(f"negative control failed: leaked {secret!r} in report")


PRIVATE_LINES = """\
x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=ACMEPrivateReason res{dim=PrivateDim fmt=acme_fmt layout=0xZZ flags=acmeflag} view{dim=x fmt=y} dest=0x1
x2w-trap acmegame_private!Device::slot31 rcx=0x1 ret=acmegame+0x10
x2w-noop ACMEOpaqueObject rcx=0x1
x2w-d3d12-debug [AcmeContext 7] sev=2 id=1234 loading AcmeTitle shader [secret](https://acme.example/leak) 0000018012345678 acme_widget_7f3a
x2w-d3d12 Device::slot44 view skipped: acme_widget_7f3a in AcmeModule.dll
x2w-d3d12 DEVICE REMOVED, reason 0x887a0005, first noticed after acme_widget_7f3a
x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1 source=0x1 offset=0x0 size=0x1 -> acme_widget_7f3a exploded
x2w-dstorage entry type 3 failed: AcmeTitle bad [secret](https://acme.example/leak)
x2w-dstorage Factory::OpenFile path="x" riid={1} -> fail acme_widget_7f3a
x2w-runtime 0000018012345678::slot9(0x1) -> 0x80004005 from +0x1
x2w-runtime 0000018012345678::slot5555555555555555(0x1) -> 0x80004005
x2w-runtime 0000018012345678::slot9(0x1) -> 0x00000180DEADBEEF
"""
PRIVATE_SHADER_FAILS = [
    "AcmeModule: 0000018012345678 acme_widget_7f3a [secret](https://acme.example/leak)",
    "v_acme_widget_7f3a_private with AcmeTitle",
    "acmegame_private",
    "undecodable word at +0x10: AcmeTitle shader text",
    "image_load_mip tfe AcmeTitle",
    "image_load_mip tfe",
]
PRIVATE_NEEDLES = [
    "Acme", "acme", "ACME", "Private", "secret", "leak", "widget", "7f3a", "exploded",
    "0000018012345678", "12345678", "5555555555555555", "deadbeef", "[", "](", "http", ".dll", "Opaque",
]


def check_private_text_never_echoed(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logdir = root / "logs"
        logdir.mkdir()
        (logdir / "stderr.txt").write_text(PRIVATE_LINES, encoding="utf-8")
        cache_dir = root / "cache"
        cache_dir.mkdir()
        for index, text in enumerate(PRIVATE_SHADER_FAILS):
            (cache_dir / f"{index:016x}.fail").write_text(text + "\n", encoding="utf-8")
        report = triage_run.triage_run(logdir, shader_cache=cache_dir)
        for needle in PRIVATE_NEEDLES:
            if needle in report:
                failures.append(f"free-form text leaked into report: {needle!r}\n{report}")
        for want in [
            "| 1 | 2 | 1234 |",
            "`DEVICE REMOVED 0x887a0005`",
            "`x2w-trap Device::slot31`",
            "`x2w-noop other (text omitted)`",
            "`Device::slot44 view skipped`",
            "| 1 | OpenFile: fail (other) |",
            "| 1 | entry failed |",
            "`slot9` | `0x80004005`",
            "`undecodable word`",
            "`image_load_mip tfe`",
            "`v_* with modifier`",
            "`other (text omitted)`",
        ]:
            if want not in report:
                failures.append(f"missing grouped allowlist row: {want!r}")
        if "| 1 | other (text omitted) | ? | ? | ? | ? | ? | ? |" not in report:
            failures.append("unknown slot70 reason/values were not grouped as other/?")


def check_review_regressions(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logdir = root / "run"
        logdir.mkdir()
        fail_line = "x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1000 source=0x1 offset=0x0 size=0x100 -> zlib failed\n"
        (logdir / "stderr.txt").write_text(fail_line, encoding="utf-8")
        (logdir / "trap.log").write_text(fail_line, encoding="utf-8")
        report = triage_run.triage_run(logdir)
        if "| 1 | zlib failed |" not in report:
            failures.append("mirrored stderr.txt/trap.log line counted other than once:\n" + report)
        # A genuine repeat (second occurrence only in trap.log) still counts.
        (logdir / "trap.log").write_text(fail_line * 2, encoding="utf-8")
        report = triage_run.triage_run(logdir)
        if "| 2 | zlib failed |" not in report:
            failures.append("repeated trap.log occurrence beyond stderr.txt was dropped:\n" + report)
    # Successful forwarded slots are not "unhandled"; failed ones and untraced calls still are.
    lines = [
        "x2w-d3d12 Device::slot13(0x1, 0x2, 0x3, 0x4, 0x5) -> 0x0\n",
        "x2w-d3d12 Device::slot41(0x1, 0x2, 0x3, 0x4, 0x5) -> 0x0\n",
        "x2w-d3d12 Device::slot14(0x1, 0x2, 0x3, 0x4, 0x5) -> 0x80070057\n",
        "x2w-d3d12 Device::slot15 not implemented\n",
    ]
    unhandled = triage_run.summarize_lines(lines)["d3d12_unhandled"]
    if "Device::slot13" in unhandled or "Device::slot41" in unhandled:
        failures.append(f"successful forwarded slots counted as unhandled: {dict(unhandled)}")
    if "Device::slot14" not in unhandled or "Device::slot15" not in unhandled:
        failures.append(f"failed/untraced slots missing from unhandled: {dict(unhandled)}")


def check_every_slot70_reason_is_reported(failures: list[str]) -> None:
    # Each reason the producer can log keeps its own row; none is folded into "other".
    cases = [
        ("unsupported memory SRV arguments", "null_destination"),
        ("unsupported memory SRV arguments", "unreadable_resource"),
        ("unsupported memory SRV arguments", "unreadable_view"),
        ("unsupported memory SRV arguments", "null_address"),
        ("unsupported memory SRV arguments", "unsupported_description"),
        ("unsupported memory SRV arguments", "destination_outside_view_heaps"),
        ("unavailable buffer at the supplied address", "buffer_unavailable"),
        ("unavailable texture at the supplied base and layout", "texture_unavailable"),
    ]
    reasons = [reason for _, reason in cases]
    lines = [
        f"x2w-d3d12 Device::slot70 {what}: reason={reason} "
        "res{dim=3 fmt=98 size=128x128x2 mips=6 layout=0x105 flags=0x300000 align=65536 samples=1/0} "
        "view{dim=5 fmt=98 map=0x1688} dest=0x1 dest_in_view_heap=1 address=0x2 address_readable=1\n"
        for what, reason in cases
    ]
    refusals = triage_run.summarize_lines(lines)["slot70_refusals"]
    reported = sorted(key[0] for key in refusals)
    if reported != sorted(reasons):
        failures.append(f"slot70 reasons reported as {reported}, expected {sorted(reasons)}")


def check_empty_and_missing_dirs(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        empty_dir = root / "empty"
        empty_dir.mkdir()
        missing_dir = root / "does_not_exist"

        for target in (empty_dir, missing_dir):
            report = triage_run.triage_run(target, shader_cache=missing_dir)
            if "Log files scanned: `0`" not in report:
                failures.append(f"expected 0 scanned files for {target}")
            if "_None observed._" not in report:
                failures.append(f"expected '_None observed._' for {target}")


def check_slot133_outcomes(failures: list[str]) -> None:
    # Device::slot133 logs every handled placement with a trailing "aliased|not aliased gpuva=..." field.
    prefix = "x2w-d3d12 Device::slot133(CreatePlacedResourceX?) "
    lines = [
        prefix + "address=0x1000 base=0x1000 offset=0x0 width=0x100 flags=0x0 state=0x0 clear=0x0 riid=ID3D12Resource -> 0x00000000 aliased gpuva=0x2000\n",
        prefix + "address=0x3000 base=0x3000 offset=0x0 width=0x100 flags=0x0 state=0x0 clear=0x0 riid=ID3D12Resource -> 0x00000000 not aliased gpuva=0x4000\n",
        prefix + "address=0x5000 base=0x5000 offset=0x0 width=0x100 flags=0x0 state=0x0 clear=0x0 riid=ID3D12Resource -> 0x80070057 not aliased gpuva=0x0\n",
        prefix + "texture address=0x6000 dim=3 64x64x1 mips=1 fmt=28 samples=1 flags=0x0 layout=0x0 state=0x0 clear=0x0 -> 0x00000000 not aliased []\n",
        prefix + "address=0x7000 not committed -> E_INVALIDARG\n",
        "x2w-d3d12 Device::slot15 not implemented\n",
    ]
    summary = triage_run.summarize_lines(lines)
    outcomes = dict(summary["slot133_outcomes"])
    if outcomes != {"aliased": 1, "not_aliased": 2, "failed": 2}:
        failures.append(f"slot133 outcomes wrong: {outcomes}")
    unhandled = dict(summary["d3d12_unhandled"])
    if unhandled != {"Device::slot133": 2, "Device::slot15": 1}:
        failures.append(f"slot133 successes counted as unhandled, or failures dropped: {unhandled}")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "stderr.txt").write_text("".join(lines), encoding="utf-8")
        report = triage_run.triage_run(Path(tmp))
    if "| 2 | not_aliased |" not in report or "| 1 | aliased |" not in report or "| 2 | failed |" not in report:
        failures.append("slot133 outcome table missing from the report:\n" + report)
    if "gpuva" in report or "0x4000" in report:
        failures.append("slot133 addresses leaked into the report:\n" + report)
    # Control: with the old classification (no slot133 rule) the same log counts the successes as unhandled.
    saved = (triage_run.RE_D3D12_SLOT133, triage_run.RE_D3D12_SLOT133_UNCOMMITTED)
    never = re.compile(r"(?!)")
    triage_run.RE_D3D12_SLOT133 = triage_run.RE_D3D12_SLOT133_UNCOMMITTED = never
    try:
        old = triage_run.summarize_lines(lines)["d3d12_unhandled"]
    finally:
        triage_run.RE_D3D12_SLOT133, triage_run.RE_D3D12_SLOT133_UNCOMMITTED = saved
    if old.get("Device::slot133") != 5:
        failures.append(f"control did not reproduce the old unhandled noise: {dict(old)}")


def check_cli_invocation(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / "single.log"
        log_file.write_text(
            "x2w-dstorage   options=0x1 dest=0x0000018000001000 dest_size=0x10 "
            "source=0x1 offset=0x0 size=0x10 -> read failed\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-B", str(TOOLS_STARTUP / "triage_run.py"), str(log_file), "--max-items", "4"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            failures.append(f"CLI exited with {proc.returncode}: {proc.stderr}")
        elif "| 1 | read failed |" not in proc.stdout:
            failures.append(f"CLI output missing expected row: {proc.stdout!r}")


def check_every_shader_reason_entry_is_reachable(failures: list[str]) -> None:
    # Each table entry is a distinct report token. A pattern that can never match, or a label whose
    # text changed, folds a known cause into "other" without failing anything else. The samples are
    # matched back to the table so a new entry without a sample fails here.
    samples = (
        ("image_load_mip tfe", "image_load_mip tfe"),
        ("image_load_mip a16", "image_load_mip a16"),
        ("buffer_load_format_xyzw requires descriptor FORMAT and DST_SEL", "buffer_* instruction"),
        ("SDWA UNUSED_SEXT", "SDWA UNUSED_SEXT"),
        ("SDWA dst_unused:UNUSED_PRESERVE needs a tracked destination", "SDWA dst_unused"),
        ("undecodable word at +0x10", "undecodable word"),
        ("numeric use of s0 holding a descriptor", "numeric use of scalar register"),
        ("operand v[0:3] is not a single register", "unsupported operand"),
        ("modifier r128 is not lowered", "unsupported modifier"),
        ("descriptor operand s[0:7]", "descriptor operand"),
        ("sampler operand s[8:11]", "sampler operand"),
        ("mask from descriptor data", "mask from descriptor data"),
        ("s_load of 4 dwords from an unaligned address", "s_load form"),
        ("v_mov_b32 SDWA sel_enc", "v_* SDWA form"),
        ("v_mov_b32 with clamp", "v_* with modifier"),
        ("v_mov_b32", "v_* instruction"),
    )
    covered: set[int] = set()
    for line, expected in samples:
        got = triage_run.classify_shader_reason(line)
        if got != expected:
            failures.append(f"classify_shader_reason({line!r}) -> {got!r}, expected {expected!r}")
        for index, (pattern, _) in enumerate(triage_run.SHADER_REASONS):
            if pattern.match(line):
                covered.add(index)
                break
    missing = [index for index in range(len(triage_run.SHADER_REASONS)) if index not in covered]
    for index in missing:
        failures.append(
            f"SHADER_REASONS[{index}] ({triage_run.SHADER_REASONS[index][0].pattern!r}) has no sample line"
        )
    if triage_run.classify_shader_reason("a reason nobody modelled") != triage_run.OTHER:
        failures.append("an unmodelled reason did not fall back to the other token")

def main() -> int:
    failures: list[str] = []
    check_aggregation_and_redaction(failures)
    check_private_text_never_echoed(failures)
    check_review_regressions(failures)
    check_every_slot70_reason_is_reported(failures)
    check_empty_and_missing_dirs(failures)
    check_cli_invocation(failures)
    check_slot133_outcomes(failures)
    check_every_shader_reason_entry_is_reachable(failures)
    for item in failures:
        print("FAIL", item)
    print(f"startup triage_run checks: {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
