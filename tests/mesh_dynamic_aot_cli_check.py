"""Pure-Python CLI, atomic output, and mutation control suite for tools/mesh_dynamic_aot.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
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
TOOL = ROOT / "tools" / "mesh_dynamic_aot.py"
sys.path.insert(0, str(ROOT / "tools"))

import mesh_dynamic_aot  # noqa: E402


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
            code = mesh_dynamic_aot.main(argv)
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


class MeshDynamicAotCliPositiveTests(unittest.TestCase):
    def test_main_creates_nested_output_and_forwards_all_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container_bytes = b"synthetic-dxbc-payload"
            container.write_bytes(container_bytes)
            out_path = root / "nested" / "deep" / "dynamic_mesh.hlsl"
            self.assertFalse(out_path.parent.exists())

            expected_hlsl = "// translated dynamic mesh\nvoid main() {}\n"
            seen_translate = []
            seen_replaces = []
            real_replace = os.replace

            def fake_translate(blob, word_count, library, sampler_heap_sgpr=None, vs_out_cntl=None):
                seen_translate.append((blob, word_count, library, sampler_heap_sgpr, vs_out_cntl))
                return expected_hlsl

            def tracking_replace(src, dst):
                src_path = Path(src)
                dst_path = Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with (
                mock.patch.object(mesh_dynamic_aot, "translate", side_effect=fake_translate),
                mock.patch.object(mesh_dynamic_aot.os, "replace", side_effect=tracking_replace),
            ):
                code, out, err = _run_main(
                    [
                        str(container),
                        "--geometry-word-count",
                        "4",
                        "--llvm",
                        "mock-llvm.dll",
                        "--sampler-heap-sgpr",
                        "5",
                        "--pa-cl-vs-out-cntl",
                        "0x00c01003",
                        "--output",
                        str(out_path),
                    ]
                )

            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(seen_translate, [(container_bytes, 4, "mock-llvm.dll", 5, 0x00C01003)])
            self.assertEqual(len(seen_replaces), 1)
            self.assertEqual([p.name for p in out_path.parent.iterdir()], ["dynamic_mesh.hlsl"])
            self.assertEqual(out_path.read_text(encoding="ascii"), expected_hlsl)

    def test_main_defaults_and_atomic_overwrite_of_existing_hlsl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(b"dxbc")
            out_path = root / "OUT.HLSL"
            out_path.write_text("// stale\n", encoding="ascii")

            seen_translate = []

            def fake_translate(blob, word_count, library, sampler_heap_sgpr=None, vs_out_cntl=None):
                seen_translate.append((blob, word_count, library, sampler_heap_sgpr, vs_out_cntl))
                return "// fresh\n"

            with mock.patch.object(mesh_dynamic_aot, "translate", side_effect=fake_translate):
                code, _, err = _run_main(
                    [
                        str(container),
                        "--geometry-word-count",
                        "2",
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_path),
                    ]
                )

            self.assertEqual(code, 0, err)
            self.assertEqual(seen_translate, [(b"dxbc", 2, "mock-llvm.dll", None, None)])
            self.assertEqual([p.name for p in root.iterdir() if p.is_dir()], [])
            self.assertEqual(out_path.read_text(encoding="ascii"), "// fresh\n")


class MeshDynamicAotCliNegativeAndMutationTests(unittest.TestCase):
    def test_invalid_sampler_heap_sgpr_exits_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(b"dxbc")
            out_path = root / "never_created" / "out.hlsl"

            for bad_sgpr in (-1, 106, *mesh_dynamic_aot.ENTRY_SGPRS):
                with self.subTest(sgpr=bad_sgpr):
                    argv = [
                        str(container),
                        "--geometry-word-count",
                        "2",
                        "--llvm",
                        "mock-llvm.dll",
                        "--sampler-heap-sgpr",
                        str(bad_sgpr),
                        "--output",
                        str(out_path),
                    ]
                    with mock.patch.object(mesh_dynamic_aot, "translate") as translate_mock:
                        code, out, err = _run_main(argv)
                    translate_mock.assert_not_called()
                    self.assertEqual(code, 2)
                    self.assertEqual(out, "")
                    self.assertNotIn("Traceback", err)
                    self.assertIn("--sampler-heap-sgpr must name a free SGPR", err)
                    self.assertFalse(out_path.parent.exists())

            proc = _run_cli_subprocess(
                str(container),
                "--geometry-word-count",
                "2",
                "--llvm",
                "mock-llvm.dll",
                "--sampler-heap-sgpr",
                "17",
                "--output",
                str(out_path),
            )
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("--sampler-heap-sgpr must name a free SGPR", proc.stderr)

    def test_non_hlsl_suffix_and_existing_directory_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(b"dxbc")

            for dir_name in ("out_dir", "dir_named.hlsl"):
                out_dir = root / dir_name
                out_dir.mkdir()
                with self.subTest(dir_name=dir_name):
                    argv = [
                        str(container),
                        "--geometry-word-count",
                        "2",
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_dir),
                    ]
                    with mock.patch.object(mesh_dynamic_aot, "translate") as translate_mock:
                        code, out, err = _run_main(argv)
                    translate_mock.assert_not_called()
                    self.assertEqual(code, 1)
                    self.assertEqual(out, "")
                    self.assertNotIn("Traceback", err)
                    self.assertIn("ValueError: Output path is an existing directory:", err)

                    proc = _run_cli_subprocess(*argv)
                    self.assertEqual(proc.returncode, 1, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertIn("ValueError: Output path is an existing directory:", proc.stderr)

            bad_out = root / "nested" / "out.txt"
            argv = [
                str(container),
                "--geometry-word-count",
                "2",
                "--llvm",
                "mock-llvm.dll",
                "--output",
                str(bad_out),
            ]
            with mock.patch.object(mesh_dynamic_aot, "translate") as translate_mock:
                code, out, err = _run_main(argv)
            translate_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("ValueError: Expected a separate HLSL output", err)
            self.assertFalse(bad_out.parent.exists())

            proc = _run_cli_subprocess(*argv)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("ValueError: Expected a separate HLSL output", proc.stderr)

    def test_output_collision_with_container_resolve_and_samefile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()

            # 1) Non-existent container path via parent traversal (kills resolve() mutation).
            ghost_container = root / "ghost.hlsl"
            with mock.patch.object(mesh_dynamic_aot, "translate") as translate_mock:
                code, _, err = _run_main(
                    [
                        str(ghost_container),
                        "--geometry-word-count",
                        "2",
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(sub / ".." / "ghost.hlsl"),
                    ]
                )
            translate_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("ValueError: Expected a separate HLSL output", err)

            # 2) Hardlink collision where resolve() differs (kills samefile() mutation).
            container = root / "shader.dxbc"
            container.write_bytes(b"keep-dxbc")
            hardlink = sub / "shader_alias.hlsl"
            os.link(container, hardlink)
            self.assertNotEqual(container.resolve(), hardlink.resolve())
            self.assertTrue(container.samefile(hardlink))

            with mock.patch.object(mesh_dynamic_aot, "translate") as translate_mock:
                code, _, err = _run_main(
                    [
                        str(container),
                        "--geometry-word-count",
                        "2",
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(hardlink),
                    ]
                )
            translate_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("ValueError: Expected a separate HLSL output", err)
            self.assertEqual(container.read_bytes(), b"keep-dxbc")

    def test_missing_container_and_exception_guards_preserve_type_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.dxbc"
            out_path = root / "never_created" / "out.hlsl"

            # Missing container -> OSError (FileNotFoundError)
            argv_missing = [
                str(missing),
                "--geometry-word-count",
                "2",
                "--llvm",
                "mock-llvm.dll",
                "--output",
                str(out_path),
            ]
            code, out, err = _run_main(argv_missing)
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("FileNotFoundError:", err)
            self.assertFalse(out_path.parent.exists())

            proc_missing = _run_cli_subprocess(*argv_missing)
            self.assertEqual(proc_missing.returncode, 1, proc_missing.stderr)
            self.assertNotIn("Traceback", proc_missing.stderr)
            self.assertIn("FileNotFoundError:", proc_missing.stderr)
            self.assertFalse(out_path.parent.exists())

            # Real subprocess refusal of amplification stage -> ValueError: ...
            amp = root / "amp.dxbc"
            amp.write_bytes(_amplification_dxbc())
            proc_amp = _run_cli_subprocess(
                str(amp),
                "--geometry-word-count",
                "2",
                "--llvm",
                "mock-llvm.dll",
                "--output",
                str(out_path),
            )
            self.assertEqual(proc_amp.returncode, 1, proc_amp.stderr)
            self.assertNotIn("Traceback", proc_amp.stderr)
            self.assertIn("ValueError: Unsupported amplification stage", proc_amp.stderr)
            self.assertFalse(out_path.parent.exists())

            # Every guarded exception type in except (...) dies if removed.
            for exc in (
                ValueError("bad runtime mesh"),
                KeyError(b"DXIL"),
                struct.error("truncated psv0"),
                zlib.error("corrupt dxil stream"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(mesh_dynamic_aot, "translate", side_effect=exc):
                        code, out, err = _run_main(
                            [
                                str(amp),
                                "--geometry-word-count",
                                "2",
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
                mock.patch.object(mesh_dynamic_aot, "translate", return_value="// ok\n"),
                mock.patch.object(mesh_dynamic_aot.os, "replace", side_effect=failing_replace),
            ):
                code, _, err = _run_main(
                    [
                        str(container),
                        "--geometry-word-count",
                        "2",
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_path),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("OSError: simulated replace failure", err)
            self.assertEqual(len(replace_calls), 1)
            self.assertTrue(out_dir.is_dir())
            self.assertEqual(list(out_dir.iterdir()), [])

            with (
                mock.patch.object(mesh_dynamic_aot, "translate", return_value="// ok\n"),
                mock.patch.object(mesh_dynamic_aot.os, "replace") as replace_mock,
                mock.patch.object(Path, "write_text", side_effect=OSError("simulated disk full")),
            ):
                code, _, err = _run_main(
                    [
                        str(container),
                        "--geometry-word-count",
                        "2",
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_path),
                    ]
                )
            replace_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("OSError: simulated disk full", err)
            self.assertEqual(list(out_dir.iterdir()), [])

    def test_non_ascii_hlsl_rejected_before_replace_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(b"dxbc")
            out_dir = root / "ascii_guard"
            out_dir.mkdir()
            out_path = out_dir / "mesh.hlsl"
            out_path.write_text("// preserve-existing\n", encoding="ascii")

            with (
                mock.patch.object(mesh_dynamic_aot, "translate", return_value="// sym_\u00e9\n"),
                mock.patch.object(mesh_dynamic_aot.os, "replace") as replace_mock,
            ):
                code, out, err = _run_main(
                    [
                        str(container),
                        "--geometry-word-count",
                        "2",
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(out_path),
                    ]
                )

            replace_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("UnicodeEncodeError:", err)
            self.assertEqual(out_path.read_text(encoding="ascii"), "// preserve-existing\n")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["mesh.hlsl"])


if __name__ == "__main__":
    unittest.main()
