#!/usr/bin/env python3
"""Check the per-chunk layout summary emitted by verify_build.compare_layout."""

import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))

from resource_inventory import main as resource_inventory_main
from verify_build import compare_layout, decode_torrent, main as verify_build_main

LAYOUT = r"""<?xml version="1.0" encoding="utf-8"?>
<Package>
  <Chunk Id="1000">
    <FileGroup SourcePath=".\" DestinationPath="\" Include="MicrosoftGame.Config" />
  </Chunk>
  <Chunk Id="1001">
    <FileGroup SourcePath=".\..\..\content\" DestinationPath="\" Include="game_scarlett_master.exe" />
  </Chunk>
  <Chunk Id="3009" Marker="Launch">
    <FileGroup SourcePath=".\" DestinationPath="\" Include="Update.alignmentchunk" />
  </Chunk>
</Package>
"""


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "layout_test.xml"
        path.write_text(LAYOUT, encoding="utf-8")
        report = compare_layout(
            path,
            {"MicrosoftGame.Config", "Update.alignmentchunk"},
            {"MicrosoftGame.Config", "Update.alignmentchunk"},
        )
    assert report["target_count"] == 3, report
    assert report["launch_marker"] == "3009", report
    assert report["absent_from_disk"] == {"game_scarlett_master.exe": ["1001"]}, report
    assert report["absent_from_torrent"] == {"game_scarlett_master.exe": ["1001"]}, report
    assert report["torrent_paths_outside_layout"] == [], report
    chunks = {chunk["id"]: chunk for chunk in report["chunk_summary"]}
    assert chunks["1001"]["absent_from_disk"] == ["game_scarlett_master.exe"], chunks
    assert chunks["1001"]["absent_from_torrent"] == ["game_scarlett_master.exe"], chunks
    assert chunks["3009"]["marker"] == "Launch", chunks
    assert chunks["3009"]["absent_from_disk"] == [], chunks
    check_layout_destination_paths()
    check_unreadable_torrent_is_a_usage_error()
    check_unreadable_layout_is_a_usage_error()
    check_decode_torrent()
    check_manifest_selection()
    check_inventory_layout_selection()
    print("layout chunk summary ok")


def _run_cli(main_fn, argv, prog):
    stdout = io.StringIO()
    stderr = io.StringIO()
    saved_argv = sys.argv
    sys.argv = [prog, *argv]
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main_fn(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            returncode = 0
        elif isinstance(code, int):
            returncode = int(code)
        else:
            stderr.write(f"{code}\n")
            returncode = 1
    else:
        returncode = 0 if result is None else int(result)
    finally:
        sys.argv = saved_argv
    return SimpleNamespace(returncode=returncode, stdout=stdout.getvalue(), stderr=stderr.getvalue())


def check_unreadable_torrent_is_a_usage_error() -> None:
    single_file = (b"d4:infod6:lengthi1e4:name1:a12:piece lengthi16384e6:pieces20:" + bytes(20) + b"ee")
    for label, torrent in (
        ("single-file torrent", single_file),
        ("info is not a dictionary", b"d4:infoi7ee"),
        ("missing piece hashes", b"d4:infod5:filesld6:lengthi1e4:pathl1:aeee12:piece lengthi16384eee"),
        ("wrong piece count", b"d4:infod5:filesld6:lengthi1e4:pathl1:aeee12:piece lengthi16384e6:pieces0:ee"),
        ("path is not text", b"d4:infod5:filesld6:lengthi1e4:pathl1:\xffeee12:piece lengthi16384e6:pieces20:"
                             + bytes(20) + b"ee"),
        ("negative size", b"d4:infod5:filesld6:lengthi-1e4:pathl1:aeee12:piece lengthi16384e6:pieces20:"
                          + bytes(20) + b"ee"),
    ):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            package.mkdir()
            (package / "layout.xml").write_text("<Package/>", encoding="utf-8")
            (package / "payload.torrent").write_bytes(torrent)
            report = Path(directory) / "report.json"
            run = _run_cli(
                verify_build_main,
                [str(package), "--output", str(report)],
                "verify_build.py",
            )
            assert run.returncode == 2, (label, run.returncode, run.stderr)
            assert "payload.torrent" in run.stderr and "Traceback" not in run.stderr, (label, run.stderr)
            assert not report.exists(), label
    with tempfile.TemporaryDirectory() as directory:
        run = subprocess.run(
            [sys.executable, "-B", str(PROJECT / "tools/verify_build.py"), str(Path(directory) / "missing")],
            capture_output=True, text=True,
        )
        assert run.returncode == 2 and "Traceback" not in run.stderr, run.stderr


def check_unreadable_layout_is_a_usage_error() -> None:
    torrent = (b"d4:infod5:filesld6:lengthi0e4:pathl1:aeee"
               b"12:piece lengthi16384e6:pieces0:ee")
    for layout in ("<Package>", "<Package><Chunk/></Package>",
                   '<Package><Chunk Id="1"><FileGroup/></Chunk></Package>'):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            package.mkdir()
            (package / "layout.xml").write_text(layout, encoding="utf-8")
            (package / "payload.torrent").write_bytes(torrent)
            report = Path(directory) / "report.json"
            run = subprocess.run(
                [sys.executable, "-B", str(PROJECT / "tools/verify_build.py"),
                 str(package), "--output", str(report)],
                capture_output=True, text=True,
            )
            assert run.returncode == 2, (layout, run.returncode, run.stderr)
            assert "layout.xml" in run.stderr and "Traceback" not in run.stderr, run.stderr
            assert not report.exists(), layout


def check_manifest_selection() -> None:
    for pattern, files, count in (
        ("*.torrent", ["layout.xml"], 0),
        ("*.torrent", ["a.torrent", "b.torrent", "layout.xml"], 2),
        ("layout*.xml", ["a.torrent"], 0),
        ("layout*.xml", ["a.torrent", "layout_a.xml", "layout_b.xml"], 2),
    ):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            package.mkdir()
            (package / "directory.torrent").mkdir()
            (package / "layout_directory.xml").mkdir()
            for name in files:
                (package / name).touch()
            report = Path(directory) / "report.json"
            run = _run_cli(
                verify_build_main,
                [str(package), "--output", str(report)],
                "verify_build.py",
            )
            assert run.returncode == 2, run.stderr
            assert pattern in run.stderr and f"found {count}" in run.stderr, run.stderr
            assert "Traceback" not in run.stderr and not report.exists(), run.stderr


def check_inventory_layout_selection() -> None:
    for use_subprocess, files, count in (
        (True, [], 0),
        (False, ["layout_a.xml", "layout_b.xml"], 2),
    ):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "layout_directory.xml").mkdir()
            for name in files:
                (package / name).touch()
            if use_subprocess:
                run = subprocess.run(
                    [sys.executable, "-B", str(PROJECT / "tools/resource_inventory.py"), str(package)],
                    capture_output=True, text=True,
                )
            else:
                run = _run_cli(
                    resource_inventory_main,
                    [str(package)],
                    "resource_inventory.py",
                )
            assert run.returncode == 2, run.stderr
            assert "layout*.xml" in run.stderr and f"found {count}" in run.stderr, run.stderr
            assert "Traceback" not in run.stderr, run.stderr


def check_layout_destination_paths() -> None:
    layout = r"""<Package><Chunk Id="3000">
      <FileGroup SourcePath="." DestinationPath="\fixture\\.\nested\" Include="item.bin" />
      <FileGroup SourcePath="." DestinationPath="." Include="root.bin" />
      <FileGroup SourcePath="." DestinationPath="\fixture\..\unresolved" Include="held.bin" />
    </Chunk></Package>"""
    inventory = {"fixture/nested/item.bin", "root.bin", "unresolved/held.bin"}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "layout_destinations.xml"
        path.write_text(layout, encoding="utf-8")
        report = compare_layout(path, inventory, inventory)
    unresolved = {"fixture/../unresolved/held.bin": ["3000"]}
    assert report["absent_from_disk"] == unresolved, report
    assert report["absent_from_torrent"] == unresolved, report
    assert report["torrent_paths_outside_layout"] == ["unresolved/held.bin"], report


def check_decode_torrent():
    valid = (
        b"d"
        b"7:comment0:"
        b"4:infod"
        b"5:filesld6:lengthi0e4:pathl9:empty.bin"
        b"eed6:lengthi128e4:pathl3:sub8:data.pak"
        b"eee"
        b"12:piece lengthi262144e"
        b"6:pieces20:" + bytes(range(20)) +
        b"e"
        b"6:offseti-7e"
        b"e"
    )
    decoded = decode_torrent(valid)
    assert decoded[b"comment"] == b""
    assert decoded[b"offset"] == -7
    assert decoded[b"info"][b"piece length"] == 262144
    assert len(decoded[b"info"][b"files"]) == 2
    assert decode_torrent(b"d1:Z1:x1:a1:ye") == {b"Z": b"x", b"a": b"y"}
    assert decode_torrent(b"l" * 32 + b"e" * 32) is not None

    for malformed in (
        b"",
        b"ie",
        b"i-0e",
        b"i00e",
        b"i03e",
        b"i-03e",
        b"i+3e",
        b"i 3e",
        b"i3 e",
        b"i1_0e",
        b"i3",
        b"00:",
        b"01:a",
        b"1_0:aaaaaaaaaa",
        b"1 :a",
        b"3:ab",
        b"3abc",
        b"d1:b1:x1:a1:ye",
        b"d1:a1:y1:Z1:xe",
        b"d1:a1:x1:a1:ye",
        b"di1e1:xe",
        b"d1:a1:x",
        b"l1:x",
        b"i0ei1e",
        b"l" * 5000 + b"e" * 5000,
        b"l" * 5000,
        b"d1:a" * 5000,
    ):
        try:
            decode_torrent(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted malformed bencode: {malformed!r}")


if __name__ == "__main__":
    main()
