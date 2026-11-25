"""Pure-Python CLI, atomic output, and mutation control suite for tools/mesh_entry_aot.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "mesh_entry_aot.py"
sys.path.insert(0, str(ROOT / "tools"))

import mesh_entry_aot  # noqa: E402


def _build_dxbc(part_items: list[tuple[bytes, bytes]]) -> bytes:
    encoded = [tag + struct.pack("<I", len(payload)) + payload for tag, payload in part_items]
    table_end = 32 + 4 * len(encoded)
    offsets = []
    cursor = table_end
    for chunk in encoded:
        offsets.append(cursor)
        cursor += len(chunk)
    total = table_end + sum(len(chunk) for chunk in encoded)
    return (
        b"DXBC"
        + bytes(16)
        + struct.pack("<III", 1, total, len(offsets))
        + struct.pack(f"<{len(offsets)}I", *offsets)
        + b"".join(encoded)
    )


def _amplification_dxbc() -> bytes:
    psv0 = bytearray(52)
    psv0[28] = 14
    dxil = bytearray(4)
    struct.pack_into("<H", dxil, 2, 14)
    return _build_dxbc([(b"PSV0", bytes(psv0)), (b"DXIL", bytes(dxil))])


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = mesh_entry_aot.main(argv)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class MeshEntryAotCliPositiveTests(unittest.TestCase):
    def test_main_creates_nested_output_and_forwards_all_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container_bytes = b"synthetic-dxbc-payload"
            container.write_bytes(container_bytes)
            geometry_path = root / "geometry.json"
            geometry_words = [0x1003, 7]
            geometry_path.write_text(json.dumps(geometry_words), encoding="utf-8")
            out_path = root / "nested" / "deep" / "mesh.hlsl"
            self.assertFalse(out_path.parent.exists())

            expected_hlsl = "// translated mesh hlsl\nvoid main() {}\n"
            seen_translate = []
            seen_replaces = []
            real_replace = os.replace

            def fake_translate(blob, geometry, library, vertex_header=None, *, compute_fallback=False, vs_out_cntl=None):
                seen_translate.append((blob, geometry, library, vertex_header, compute_fallback, vs_out_cntl))
                return expected_hlsl

            def tracking_replace(src, dst):
                src_path = Path(src)
                dst_path = Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with (
                mock.patch.object(mesh_entry_aot, "translate", side_effect=fake_translate),
                mock.patch.object(mesh_entry_aot.os, "replace", side_effect=tracking_replace),
            ):
                code, out, err = _run_main(
                    [
                        str(container),
                        "--geometry-words",
                        str(geometry_path),
                        "--llvm",
                        "mock-llvm.dll",
                        "--vertex-header",
                        "0x260",
                        "--compute-fallback",
                        "--pa-cl-vs-out-cntl",
                        "0x00c01003",
                        "--output",
                        str(out_path),
                    ]
                )

            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(
                seen_translate,
                [(container_bytes, geometry_words, "mock-llvm.dll", 0x260, True, 0x00C01003)],
            )
            self.assertEqual(len(seen_replaces), 1)
            self.assertEqual([p.name for p in out_path.parent.iterdir()], ["mesh.hlsl"])
            self.assertEqual(out_path.read_text(encoding="ascii"), expected_hlsl)

    def test_main_defaults_and_atomic_overwrite_of_existing_hlsl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(b"dxbc")
            geometry_path = root / "geometry.json"
            geometry_path.write_text("[4099]", encoding="utf-8")
            out_path = root / "OUT.HLSL"
            out_path.write_text("// stale\n", encoding="ascii")

            seen_translate = []

            def fake_translate(blob, geometry, library, vertex_header=None, *, compute_fallback=False, vs_out_cntl=None):
                seen_translate.append((blob, geometry, library, vertex_header, compute_fallback, vs_out_cntl))
                return "// fresh\n"

            with mock.patch.object(mesh_entry_aot, "translate", side_effect=fake_translate):
                code, _, err = _run_main(
                    [
                        str(container),
                        "--geometry-words",
                        str(geometry_path),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_path),
                    ]
                )

            self.assertEqual(code, 0, err)
            self.assertEqual(seen_translate, [(b"dxbc", [4099], "mock-llvm.dll", None, False, None)])
            self.assertEqual([p.name for p in root.iterdir() if p.is_dir()], [])
            self.assertEqual(out_path.read_text(encoding="ascii"), "// fresh\n")


class MeshEntryAotCliNegativeAndMutationTests(unittest.TestCase):
    def test_non_hlsl_suffix_and_existing_directory_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(b"dxbc")
            geometry_path = root / "geometry.json"
            geometry_path.write_text("[4099]", encoding="utf-8")

            # Existing directory outputs (with and without .hlsl suffix)
            for dir_name in ("out_dir", "dir_named.hlsl"):
                out_dir = root / dir_name
                out_dir.mkdir()
                with self.subTest(dir_name=dir_name):
                    with mock.patch.object(mesh_entry_aot, "translate") as translate_mock:
                        code, out, err = _run_main(
                            [
                                str(container),
                                "--geometry-words",
                                str(geometry_path),
                                "--llvm",
                                "mock-llvm.dll",
                                "--output",
                                str(out_dir),
                            ]
                        )
                    translate_mock.assert_not_called()
                    self.assertEqual(code, 1)
                    self.assertEqual(out, "")
                    self.assertNotIn("Traceback", err)
                    self.assertIn("existing directory", err)

                    proc = _run_cli_subprocess(
                        str(container),
                        "--geometry-words",
                        str(geometry_path),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_dir),
                    )
                    self.assertEqual(proc.returncode, 1, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertIn("existing directory", proc.stderr)

            # Non-.hlsl suffix
            bad_out = root / "nested" / "out.txt"
            with mock.patch.object(mesh_entry_aot, "translate") as translate_mock:
                code, out, err = _run_main(
                    [
                        str(container),
                        "--geometry-words",
                        str(geometry_path),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(bad_out),
                    ]
                )
            translate_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("Expected a separate HLSL output", err)
            self.assertFalse(bad_out.parent.exists())

            proc = _run_cli_subprocess(
                str(container),
                "--geometry-words",
                str(geometry_path),
                "--llvm",
                "mock-llvm.dll",
                "--output",
                str(bad_out),
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("Expected a separate HLSL output", proc.stderr)

    def test_output_collision_with_container_or_geometry_resolve_and_samefile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()

            # 1) Non-existent paths colliding via parent traversal (kills resolve() mutation).
            ghost_container = root / "ghost.hlsl"
            ghost_geom = root / "ghost_words.hlsl"
            for label, argv in (
                (
                    "container_resolve",
                    [
                        str(ghost_container),
                        "--geometry-words",
                        str(ghost_geom),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(sub / ".." / "ghost.hlsl"),
                    ],
                ),
                (
                    "geometry_resolve",
                    [
                        str(ghost_container),
                        "--geometry-words",
                        str(ghost_geom),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(sub / ".." / "ghost_words.hlsl"),
                    ],
                ),
            ):
                with self.subTest(case=label):
                    with mock.patch.object(mesh_entry_aot, "translate") as translate_mock:
                        code, _, err = _run_main(argv)
                    translate_mock.assert_not_called()
                    self.assertEqual(code, 1)
                    self.assertNotIn("Traceback", err)
                    self.assertIn("Expected a separate HLSL output", err)

            # 2) Hardlink collisions where resolve() differs (kills samefile() mutation).
            container = root / "shader.dxbc"
            container.write_bytes(b"keep-dxbc")
            geometry_path = root / "geometry.json"
            geometry_path.write_text("[4099]", encoding="utf-8")

            container_link = sub / "container_alias.hlsl"
            geometry_link = sub / "geometry_alias.hlsl"
            os.link(container, container_link)
            os.link(geometry_path, geometry_link)
            self.assertNotEqual(container.resolve(), container_link.resolve())
            self.assertNotEqual(geometry_path.resolve(), geometry_link.resolve())
            self.assertTrue(container.samefile(container_link))
            self.assertTrue(geometry_path.samefile(geometry_link))

            for label, alias in (("container_samefile", container_link), ("geometry_samefile", geometry_link)):
                with self.subTest(case=label):
                    with mock.patch.object(mesh_entry_aot, "translate") as translate_mock:
                        code, _, err = _run_main(
                            [
                                str(container),
                                "--geometry-words",
                                str(geometry_path),
                                "--llvm",
                                "mock-llvm.dll",
                                "--output",
                                str(alias),
                            ]
                        )
                    translate_mock.assert_not_called()
                    self.assertEqual(code, 1)
                    self.assertNotIn("Traceback", err)
                    self.assertIn("Expected a separate HLSL output", err)
            self.assertEqual(container.read_bytes(), b"keep-dxbc")
            self.assertEqual(geometry_path.read_text(encoding="utf-8"), "[4099]")

    def test_missing_inputs_and_invalid_json_exit_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(b"dxbc")
            geometry_path = root / "geometry.json"
            geometry_path.write_text("[4099]", encoding="utf-8")
            bad_json = root / "bad.json"
            bad_json.write_text("{not-json", encoding="utf-8")
            out_path = root / "never_created" / "out.hlsl"

            cases = (
                ("missing_container", root / "missing.dxbc", geometry_path),
                ("missing_geometry", container, root / "missing.json"),
                ("invalid_json", container, bad_json),
            )
            for label, cont_arg, geom_arg in cases:
                with self.subTest(case=label):
                    argv = [
                        str(cont_arg),
                        "--geometry-words",
                        str(geom_arg),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_path),
                    ]
                    code, out, err = _run_main(argv)
                    self.assertEqual(code, 1)
                    self.assertEqual(out, "")
                    self.assertNotIn("Traceback", err)
                    self.assertTrue(err.strip())
                    self.assertFalse(out_path.parent.exists())

                    proc = _run_cli_subprocess(*argv)
                    self.assertEqual(proc.returncode, 1, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertTrue(proc.stderr.strip())
                    self.assertFalse(out_path.parent.exists())

    def test_translate_exception_guards_exit_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(_amplification_dxbc())
            geometry_path = root / "geometry.json"
            geometry_path.write_text("[4099]", encoding="utf-8")
            out_path = root / "never_created" / "out.hlsl"

            # Real subprocess refusal of amplification stage via ValueError guard.
            proc = _run_cli_subprocess(
                str(container),
                "--geometry-words",
                str(geometry_path),
                "--llvm",
                "mock-llvm.dll",
                "--output",
                str(out_path),
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("ValueError: Unsupported amplification stage", proc.stderr)
            self.assertFalse(out_path.parent.exists())

            # Every guarded exception type in except (...) dies if removed.
            for exc in (
                ValueError("bad mesh profile"),
                KeyError(b"DXIL"),
                struct.error("truncated psv0"),
                zlib.error("corrupt dxil stream"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(mesh_entry_aot, "translate", side_effect=exc):
                        code, out, err = _run_main(
                            [
                                str(container),
                                "--geometry-words",
                                str(geometry_path),
                                "--llvm",
                                "mock-llvm.dll",
                                "--output",
                                str(out_path),
                            ]
                        )
                    self.assertEqual(code, 1)
                    self.assertEqual(out, "")
                    self.assertNotIn("Traceback", err)
                    self.assertIn(f"{type(exc).__name__}: {exc}", err)
                    self.assertFalse(out_path.parent.exists())

    def test_os_replace_and_staging_write_failures_leave_zero_temp_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(b"dxbc")
            geometry_path = root / "geometry.json"
            geometry_path.write_text("[4099]", encoding="utf-8")
            out_dir = root / "atomic_out"
            out_path = out_dir / "mesh.hlsl"

            replace_calls = []

            def failing_replace(src, dst):
                src_path = Path(src)
                replace_calls.append((src_path, Path(dst)))
                self.assertTrue(src_path.is_file())
                self.assertEqual(src_path.parent.parent.resolve(), out_dir.resolve())
                raise OSError("simulated replace failure")

            with (
                mock.patch.object(mesh_entry_aot, "translate", return_value="// ok\n"),
                mock.patch.object(mesh_entry_aot.os, "replace", side_effect=failing_replace),
            ):
                code, _, err = _run_main(
                    [
                        str(container),
                        "--geometry-words",
                        str(geometry_path),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_path),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("simulated replace failure", err)
            self.assertEqual(len(replace_calls), 1)
            self.assertTrue(out_dir.is_dir())
            self.assertEqual(list(out_dir.iterdir()), [])

            # Failure during staging write_text -> os.replace never called, zero temp dirs left.
            with (
                mock.patch.object(mesh_entry_aot, "translate", return_value="// ok\n"),
                mock.patch.object(mesh_entry_aot.os, "replace") as replace_mock,
                mock.patch.object(Path, "write_text", side_effect=OSError("simulated disk full")),
            ):
                code, _, err = _run_main(
                    [
                        str(container),
                        "--geometry-words",
                        str(geometry_path),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_path),
                    ]
                )
            replace_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("simulated disk full", err)
            self.assertEqual(list(out_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
