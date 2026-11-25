"""CLI, atomic output, directory/collision, and mutation control suite for tools/xmem_attribute_scan.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pefile

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "xmem_attribute_scan.py"
sys.path.insert(0, str(ROOT / "tools"))

import xmem_attribute_scan as scanner  # noqa: E402


BINK_WORD = 0x05E50000
PHYSICAL_NULL_WORD = 0xE0000000


def _build_synthetic_x64_pe(
    *,
    word: int = BINK_WORD,
    resolved: bool = True,
    machine: int = 0x8664,
) -> bytes:
    """Build a minimal synthetic x64 PE32+ image importing XMemAlloc from xmem.dll."""
    file_alignment = 0x200
    section_alignment = 0x1000
    headers_size = 0x200
    text_rva = 0x1000
    text_raw = 0x200
    rdata_rva = 0x2000
    rdata_raw = 0x400

    # .rdata layout at rdata_rva (0x2000):
    # +0x00..0x28: 2x IMAGE_IMPORT_DESCRIPTOR (20B each, second is zero terminator)
    # +0x28..0x38: ILT (2x 8B)
    # +0x38..0x48: IAT (2x 8B)
    # +0x48..0x58: b"xmem.dll\0" padded
    # +0x58..0x68: IMAGE_IMPORT_BY_NAME (Hint=0, b"XMemAlloc\0")
    # +0x68..0x78: RUNTIME_FUNCTION (12B) when resolved=True
    # +0x78..0x7c: UNWIND_INFO (4B)
    ilt_rva = rdata_rva + 0x28
    iat_rva = rdata_rva + 0x38
    dll_name_rva = rdata_rva + 0x48
    by_name_rva = rdata_rva + 0x58
    pdata_rva = rdata_rva + 0x68
    unwind_rva = rdata_rva + 0x78

    if resolved:
        # mov edx, word (5B) + call [rip+disp32] (6B) + ret (1B) = 12B inside .pdata function.
        call_next_rva = text_rva + 5 + 6
        disp = iat_rva - call_next_rva
        code = struct.pack("<B I 2B i B", 0xBA, word, 0xFF, 0x15, disp, 0xC3)
    else:
        # Unresolved: `add edx, ecx` (2B) before `call [rip+disp32]` (6B) + `ret` (1B).
        call_next_rva = text_rva + 2 + 6
        disp = iat_rva - call_next_rva
        code = struct.pack("<2B 2B i B", 0x01, 0xCA, 0xFF, 0x15, disp, 0xC3)

    rdata = bytearray(0x200)
    struct.pack_into("<5I", rdata, 0x00, ilt_rva, 0, 0, dll_name_rva, iat_rva)
    struct.pack_into("<Q", rdata, 0x28, by_name_rva)
    struct.pack_into("<Q", rdata, 0x38, by_name_rva)
    rdata[0x48:0x48 + 9] = b"xmem.dll\0"
    struct.pack_into("<H", rdata, 0x58, 0)
    rdata[0x5A:0x5A + 10] = b"XMemAlloc\0"
    if resolved:
        struct.pack_into("<3I", rdata, 0x68, text_rva, text_rva + len(code), unwind_rva)
        rdata[0x78:0x7C] = b"\x01\x00\x00\x00"

    text = bytearray(0x200)
    text[:len(code)] = code

    # DOS header (e_lfanew = 0x40)
    dos = bytearray(0x40)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)

    # PE signature + COFF FileHeader (20B)
    pe_sig = b"PE\0\0"
    num_sections = 2
    opt_header_size = 0xF0
    coff = struct.pack("<2H 3I 2H", machine, num_sections, 0, 0, 0, opt_header_size, 0x0022)

    # Optional Header PE32+ (0xF0 = 240 bytes: 112B fixed + 16*8B data directories)
    opt = bytearray(opt_header_size)
    struct.pack_into("<H 2B 5I Q 2I 6H 4I 2H 4Q 2I", opt, 0,
                     0x020B,          # Magic (PE32+)
                     14, 0,           # Linker version
                     0x200, 0x200, 0, # SizeOfCode, SizeOfInitializedData, SizeOfUninitializedData
                     text_rva,        # AddressOfEntryPoint
                     text_rva,        # BaseOfCode
                     0x140000000,     # ImageBase
                     section_alignment,
                     file_alignment,
                     6, 0, 0, 0, 6, 0, 0,
                     0x3000,          # SizeOfImage
                     headers_size,
                     0,               # CheckSum
                     3,               # Subsystem (CONSOLE)
                     0x8160,          # DllCharacteristics
                     0x100000, 0x1000, 0x100000, 0x1000,
                     0, 16)
    # Data directories at offset 112 in Optional Header
    struct.pack_into("<2I", opt, 112 + 1 * 8, rdata_rva, 40)  # IMPORT
    if resolved:
        struct.pack_into("<2I", opt, 112 + 3 * 8, pdata_rva, 12)  # EXCEPTION

    sec_text = struct.pack("<8s 6I 2H I", b".text\0\0\0", len(code), text_rva, 0x200, text_raw, 0, 0, 0, 0,
                           0x60000020)
    sec_rdata = struct.pack("<8s 6I 2H I", b".rdata\0\0", 0x100, rdata_rva, 0x200, rdata_raw, 0, 0, 0, 0,
                            0x40000040)

    headers = bytes(dos) + pe_sig + coff + bytes(opt) + sec_text + sec_rdata
    assert len(headers) <= headers_size
    headers_padded = headers + b"\0" * (headers_size - len(headers))
    return headers_padded + bytes(text) + bytes(rdata)


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = scanner.main(argv)
        except SystemExit as exc:
            return exc.code, stdout.getvalue(), stderr.getvalue()
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class XmemAttributeScanCliPositiveTests(unittest.TestCase):
    def _assert_report_schema(self, payload: dict, *, image_name: str, expected_sites: int) -> None:
        self.assertEqual(
            set(payload),
            {"format", "image", "imports", "unmatched_xmem_ordinals", "thunks", "sites", "summary", "stats"},
        )
        self.assertEqual(payload["format"], scanner.FORMAT)
        self.assertEqual(payload["image"], image_name)
        self.assertEqual(payload["imports"], [{"import": "XMemAlloc", "kind": "import"}])
        self.assertEqual(len(payload["sites"]), expected_sites)
        self.assertEqual(
            set(payload["summary"]),
            {"sites", "by_source", "words", "null_words", "sites_not_fully_constant"},
        )
        self.assertEqual(set(payload["stats"]), {"pdata_functions", "instructions_decoded", "passes", "seconds"})

    def test_stdout_and_atomic_nested_output_json_schema_and_exit_codes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_pe = root / "clean.exe"
            null_pe = root / "null_word.exe"
            unresolved_pe = root / "unresolved.exe"
            clean_pe.write_bytes(_build_synthetic_x64_pe(word=BINK_WORD, resolved=True))
            null_pe.write_bytes(_build_synthetic_x64_pe(word=PHYSICAL_NULL_WORD, resolved=True))
            unresolved_pe.write_bytes(_build_synthetic_x64_pe(resolved=False))

            # 1) Stdout JSON mode (exit 0) + both --fail-on-* flags stay 0 on clean image.
            for extra in ([], ["--fail-on-null", "--fail-on-unresolved"]):
                code, out, err = _run_main([str(clean_pe), *extra])
                self.assertIs(type(code), int)
                self.assertEqual(code, 0, err)
                self.assertIn("clean.exe: 1 sites", err)
                report = json.loads(out)
                self._assert_report_schema(report, image_name="clean.exe", expected_sites=1)
                self.assertEqual(report["summary"]["null_words"], [])
                self.assertEqual(report["summary"]["sites_not_fully_constant"], [])
                self.assertEqual(report["stats"]["pdata_functions"], 1)

            # 2) Nested --output uses TemporaryDirectory(dir=...) + os.replace() atomically, leaves zero temp dirs.
            nested_out = root / "nested" / "reports" / "scan.json"
            self.assertFalse(nested_out.parent.exists())
            replaces: list[tuple[Path, Path]] = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with mock.patch.object(scanner.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main([str(clean_pe), "--output", str(nested_out)])
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(len(replaces), 1)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["scan.json"])
            self._assert_report_schema(
                json.loads(nested_out.read_text(encoding="utf-8")),
                image_name="clean.exe",
                expected_sites=1,
            )

            # 3) Gate exit 1 separation: --fail-on-null and --fail-on-unresolved strictly return 1 (not 0 or 2).
            for img, flag, expected_code in (
                (null_pe, "--fail-on-null", 1),
                (null_pe, "--fail-on-unresolved", 0),
                (unresolved_pe, "--fail-on-unresolved", 1),
                (unresolved_pe, "--fail-on-null", 0),
            ):
                with self.subTest(image=img.name, flag=flag):
                    proc = _run_subprocess(str(img), flag)
                    self.assertEqual(proc.returncode, expected_code, proc.stderr)
                    self._assert_report_schema(
                        json.loads(proc.stdout),
                        image_name=img.name,
                        expected_sites=1,
                    )

    def test_main_forwards_default_sys_argv_and_script_entrypoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pe_path = root / "entry.exe"
            out_path = root / "entry.json"
            pe_path.write_bytes(_build_synthetic_x64_pe())
            cli_argv = [str(TOOL), str(pe_path), "--output", str(out_path)]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self._assert_report_schema(json.loads(out_path.read_text(encoding="utf-8")),
                                       image_name="entry.exe", expected_sites=1)

            out_path.unlink()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self._assert_report_schema(json.loads(out_path.read_text(encoding="utf-8")),
                                       image_name="entry.exe", expected_sites=1)


class XmemAttributeScanCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_exit_2(
        self,
        argv: list[str],
        *,
        expected_type: str,
        expected_fragment: str,
        check_subprocess: bool = True,
    ) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, 2, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(f"{expected_type}: ", err)
        self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(f"{expected_type}: ", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)

    def test_missing_directory_and_corrupt_pe_exit_2_without_traceback(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.exe"
            self._assert_exit_2([str(missing)], expected_type="FileNotFoundError", expected_fragment=missing.name)

            image_dir = root / "dir.exe"
            image_dir.mkdir()
            with mock.patch.object(scanner, "scan") as scan_mock:
                self._assert_exit_2(
                    [str(image_dir)],
                    expected_type="ValueError",
                    expected_fragment="directory",
                    check_subprocess=False,
                )
            scan_mock.assert_not_called()
            proc = _run_subprocess(str(image_dir))
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("ValueError: ", proc.stderr)
            self.assertIn("directory", proc.stderr)

            corrupt = root / "corrupt.exe"
            corrupt.write_bytes(b"NOT_A_PE_HEADER" * 16)
            self._assert_exit_2([str(corrupt)], expected_type="PEFormatError", expected_fragment="corrupt.exe")

            x86_pe = root / "x86.exe"
            x86_pe.write_bytes(_build_synthetic_x64_pe(machine=0x014C))
            self._assert_exit_2([str(x86_pe)], expected_type="ValueError", expected_fragment="Expected an x64 image")

    def test_output_directory_rejected_upfront_before_scanning(self) -> None:
        # Kills removal of args.output.is_dir() guard (and checks scan is never called).
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pe_path = root / "image.exe"
            pe_path.write_bytes(_build_synthetic_x64_pe())
            out_dir = root / "existing_out_dir"
            out_dir.mkdir()

            argv = [str(pe_path), "--output", str(out_dir)]
            with mock.patch.object(scanner, "scan") as scan_mock:
                self._assert_exit_2(
                    argv,
                    expected_type="ValueError",
                    expected_fragment="existing directory",
                    check_subprocess=False,
                )
            scan_mock.assert_not_called()

            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("ValueError: ", proc.stderr)
            self.assertIn("existing directory", proc.stderr)

    def test_output_collision_via_resolve_and_samefile_rejected_upfront(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()

            # 1) Non-existent image + alias --output path colliding via resolve() (kills resolve() mutation).
            ghost_img = root / "missing.exe"
            alias_out = root / "sub" / ".." / "missing.exe"
            with mock.patch.object(scanner, "scan") as scan_mock:
                self._assert_exit_2(
                    [str(ghost_img), "--output", str(alias_out)],
                    expected_type="ValueError",
                    expected_fragment="collides with image",
                )
            scan_mock.assert_not_called()

            # 2) Existing image + hardlink alias where resolve() differs (kills samefile() mutation).
            real_img = root / "real.exe"
            hardlink_out = sub / "hardlink.exe"
            pe_bytes = _build_synthetic_x64_pe()
            real_img.write_bytes(pe_bytes)
            os.link(real_img, hardlink_out)
            self.assertNotEqual(real_img.resolve(), hardlink_out.resolve())
            self.assertTrue(real_img.samefile(hardlink_out))

            with mock.patch.object(scanner, "scan") as scan_mock:
                self._assert_exit_2(
                    [str(real_img), "--output", str(hardlink_out)],
                    expected_type="ValueError",
                    expected_fragment="collides with image",
                )
            scan_mock.assert_not_called()
            self.assertEqual(real_img.read_bytes(), pe_bytes)

    def test_os_replace_failure_exits_2_and_leaves_zero_temp_dirs_or_partial_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pe_path = root / "image.exe"
            pe_path.write_bytes(_build_synthetic_x64_pe())
            out_dir = root / "out"
            out_path = out_dir / "report.json"

            # 1) os.replace fails on fresh output -> exit 2, zero output files, zero temp dirs.
            def failing_replace(src, dst):
                self.assertTrue(Path(src).is_file())
                raise OSError("simulated replace failure")

            with mock.patch.object(scanner.os, "replace", side_effect=failing_replace):
                self._assert_exit_2(
                    [str(pe_path), "--output", str(out_path)],
                    expected_type="OSError",
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(list(out_dir.iterdir()), [])

            # 2) os.replace fails when replacing existing output -> prior file stays intact, zero temp dirs.
            out_path.write_text("prior-report\n", encoding="utf-8")
            with mock.patch.object(scanner.os, "replace", side_effect=failing_replace):
                self._assert_exit_2(
                    [str(pe_path), "--output", str(out_path)],
                    expected_type="OSError",
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(out_path.read_text(encoding="utf-8"), "prior-report\n")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["report.json"])


if __name__ == "__main__":
    unittest.main()
