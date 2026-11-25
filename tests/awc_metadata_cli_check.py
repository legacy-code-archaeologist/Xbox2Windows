"""CLI, bounds-validation, error-guard, and mutation control suite for tools/awc_metadata.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "awc_metadata.py"
sys.path.insert(0, str(ROOT / "tools"))

import awc_metadata  # noqa: E402


MAGIC = {"<": b"ADAT", ">": b"TADA"}


def _sfx_bank(order: str = "<") -> bytes:
    """Build a synthetic ADAT/TADA SFX bank with one 0xFA and two 0x76 streams."""
    awc = bytearray(512)
    struct.pack_into(
        order + "4sIII3I",
        awc,
        0,
        MAGIC[order],
        0xFF020001,
        3,
        76,
        2 << 29 | 11,
        2 << 29 | 12,
        2 << 29 | 13,
    )
    tags = (
        (0x55, 208, 16),
        (0xFA, 96, 20),
        (0x76, 240, 32),
        (0x55, 304, 16),
        (0x55, 336, 16),
        (0x76, 400, 32),
    )
    for index, (kind, offset, length) in enumerate(tags):
        struct.pack_into(order + "Q", awc, 28 + index * 8, kind << 56 | length << 28 | offset)
        if kind == 0x55:
            awc[offset:offset + length] = b"\xa5" * length

    struct.pack_into(order + "I", awc, 96, 300)
    struct.pack_into(order + "H", awc, 104, 22050)
    awc[96 + 0x13] = 4

    struct.pack_into(order + "I", awc, 240, 225)
    struct.pack_into(order + "H", awc, 248, 22050)
    awc[240 + 28] = 4

    struct.pack_into(order + "I", awc, 400, 4097)
    struct.pack_into(order + "H", awc, 408, 48000)
    awc[400 + 28] = 11
    return bytes(awc)


EXPECTED_SFX = {
    "stream_count": 3,
    "formats": [
        {"sample_rate_hz": 22050, "codec_id": 4, "streams": 2, "samples": 525},
        {"sample_rate_hz": 48000, "codec_id": 11, "streams": 1, "samples": 4097},
    ],
}


def _music_bank(order: str = "<", *, flags: int = 0xFF020001) -> bytes:
    """Build a synthetic ADAT/TADA music bank with 3 streams and 4 channels."""
    awc = bytearray(256)
    stream_words = (2 << 29 | 0, 0 << 29 | 0x101, 1 << 29 | 0x102)
    tags = ((0x55, 64, 160), (0x48, 76, 64), (0xBD, 16, 144))
    records = (
        (0x101, 1000, 48000, 11),
        (0x102, 990, 48000, 11),
        (0x103, 500, 32000, 11),
        (0x104, 10, 48000, 11),
    )
    struct.pack_into(order + "4sIII", awc, 0, MAGIC[order], flags, len(stream_words), 16 + 4 * len(stream_words) + 8 * len(tags))
    for index, word in enumerate(stream_words):
        struct.pack_into(order + "I", awc, 16 + 4 * index, word)
    for index, (kind, size, offset) in enumerate(tags):
        struct.pack_into(order + "Q", awc, 16 + 4 * len(stream_words) + 8 * index, kind << 56 | size << 28 | offset)
        if kind != 0x48:
            awc[offset:offset + size] = b"\xa5" * size

    body = struct.pack(order + "III", 2, 0x800, len(records))
    for hash_, samples, rate, codec in records:
        body += struct.pack(order + "IIHHB3x", hash_, samples, 0, rate, codec)
    awc[64:64 + len(body)] = body
    return bytes(awc)


EXPECTED_MUSIC = {
    "stream_count": 3,
    "layout": "music",
    "music": {"channels": 4, "block_count": 2, "block_chunk": 2048},
    "formats": [
        {"sample_rate_hz": 32000, "codec_id": 11, "streams": 1, "samples": 500},
        {"sample_rate_hz": 48000, "codec_id": 11, "streams": 3, "samples": 1000},
    ],
}


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = awc_metadata.main(argv)
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


class AwcMetadataCliPositiveTests(unittest.TestCase):
    def test_sfx_and_music_banks_return_zero_and_exact_json_schema(self) -> None:
        cases = (
            ("adat_sfx_zero_offset", _sfx_bank("<"), 0, "0", "512", EXPECTED_SFX),
            ("adat_sfx_decimal_and_hex", _sfx_bank("<"), 13, "13", "0x200", EXPECTED_SFX),
            ("tada_sfx_hex_offset_decimal_size", _sfx_bank(">"), 0x20, "0x20", "512", EXPECTED_SFX),
            ("adat_music_decimal_and_hex", _music_bank("<"), 7, "7", "0x100", EXPECTED_MUSIC),
            ("tada_music_hex_offset_and_size", _music_bank(">"), 0x10, "0x10", "0x100", EXPECTED_MUSIC),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, bank_bytes, pad, offset_arg, size_arg, expected in cases:
                with self.subTest(label=label):
                    archive = root / f"{label}.awc"
                    archive.write_bytes(b"X" * pad + bank_bytes + b"TAIL")
                    argv = [str(archive), "--offset", offset_arg, "--size", size_arg]

                    code, out, err = _run_main(argv)
                    self.assertIs(type(code), int)
                    self.assertEqual(code, 0, err)
                    self.assertEqual(err, "")
                    payload = json.loads(out)
                    self.assertEqual(payload, expected)
                    self.assertEqual(set(payload), set(expected))
                    for fmt in payload["formats"]:
                        self.assertEqual(set(fmt), {"sample_rate_hz", "codec_id", "streams", "samples"})
                    if "music" in expected:
                        self.assertEqual(set(payload["music"]), {"channels", "block_count", "block_chunk"})

                    proc = _run_subprocess(*argv)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(proc.stderr, "")
                    self.assertEqual(json.loads(proc.stdout), expected)

    def test_main_forwards_default_sys_argv_and_script_entrypoint_returns_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "sfx.awc"
            archive.write_bytes(b"pad" + _sfx_bank("<"))
            cli_argv = [str(TOOL), str(archive), "--offset", "3", "--size", "0x200"]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(json.loads(out), EXPECTED_SFX)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertIs(type(exited.exception.code), int)
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(stdout.getvalue()), EXPECTED_SFX)


class AwcMetadataCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_type: str,
        expected_fragment: str,
        check_subprocess: bool = True,
    ) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(f"{expected_type}: ", err)
        self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(f"{expected_type}: ", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)

    def test_missing_archive_caught_without_traceback(self) -> None:
        # Kills removal of OSError from the exception handler.
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.awc"
            self._assert_clean_failure(
                [str(missing), "--offset", "0", "--size", "512"],
                expected_type="FileNotFoundError",
                expected_fragment=missing.name,
            )

    def test_directory_archive_rejected_upfront_without_reading(self) -> None:
        # Kills removal of args.archive.is_dir() guard (open() raises PermissionError/IsADirectoryError, not ValueError).
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bank_dir"
            directory.mkdir()
            argv = [str(directory), "--offset", "0", "--size", "512"]
            with mock.patch.object(awc_metadata, "read_bank") as read_bank_mock:
                self._assert_clean_failure(
                    argv,
                    expected_type="ValueError",
                    expected_fragment="directory",
                    check_subprocess=False,
                )
            read_bank_mock.assert_not_called()

            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("ValueError: ", proc.stderr)
            self.assertIn("directory", proc.stderr)

    def test_negative_offset_rejected_before_opening_archive(self) -> None:
        # Testing both existing and missing archives kills removal or reordering of the --offset < 0 check.
        with TemporaryDirectory() as tmp:
            existing = Path(tmp) / "bank.awc"
            existing.write_bytes(_sfx_bank("<"))
            missing = Path(tmp) / "missing.awc"
            for archive in (existing, missing):
                for offset_args in (["--offset", "-1"], ["--offset=-0x10"]):
                    with self.subTest(archive=archive.name, offset_args=offset_args):
                        argv = [str(archive), *offset_args, "--size", "512"]
                        with mock.patch.object(awc_metadata, "read_bank") as read_bank_mock:
                            self._assert_clean_failure(
                                argv,
                                expected_type="ValueError",
                                expected_fragment="--offset",
                            )
                        read_bank_mock.assert_not_called()

    def test_non_positive_size_rejected_before_opening_archive(self) -> None:
        # Testing --size 0 on existing & missing paths kills <= 0 -> < 0 mutation and removal/reordering.
        with TemporaryDirectory() as tmp:
            existing = Path(tmp) / "bank.awc"
            existing.write_bytes(_sfx_bank("<"))
            missing = Path(tmp) / "missing.awc"
            for archive in (existing, missing):
                for size_args in (["--size", "0"], ["--size", "-1"], ["--size=-0x20"]):
                    with self.subTest(archive=archive.name, size_args=size_args):
                        argv = [str(archive), "--offset", "0", *size_args]
                        with mock.patch.object(awc_metadata, "read_bank") as read_bank_mock:
                            self._assert_clean_failure(
                                argv,
                                expected_type="ValueError",
                                expected_fragment="--size",
                            )
                        read_bank_mock.assert_not_called()

    def test_truncated_banks_and_struct_error_caught_cleanly(self) -> None:
        with TemporaryDirectory() as tmp:
            short_file = Path(tmp) / "short_file.awc"
            short_file.write_bytes(_sfx_bank("<")[:32])
            self._assert_clean_failure(
                [str(short_file), "--offset", "0", "--size", "512"],
                expected_type="ValueError",
                expected_fragment="Segment extent disagrees",
            )

            full_sfx = Path(tmp) / "full_sfx.awc"
            full_sfx.write_bytes(_sfx_bank("<"))
            for size_arg, fragment in (
                ("12", "Logical range outside"),
                ("60", "outside declared header"),
                ("430", "outside supplied bank extent"),
            ):
                with self.subTest(size=size_arg):
                    self._assert_clean_failure(
                        [str(full_sfx), "--offset", "0", "--size", size_arg],
                        expected_type="ValueError",
                        expected_fragment=fragment,
                    )

            full_music = Path(tmp) / "full_music.awc"
            full_music.write_bytes(_music_bank(">"))
            self._assert_clean_failure(
                [str(full_music), "--offset", "0", "--size", "220"],
                expected_type="ValueError",
                expected_fragment="outside supplied bank extent",
            )

            # Kills removal of struct.error from except (OSError, ValueError, struct.error).
            with mock.patch.object(
                awc_metadata.struct,
                "unpack",
                side_effect=struct.error("unpack requires a buffer of 12 bytes"),
            ):
                self._assert_clean_failure(
                    [str(full_sfx), "--offset", "0", "--size", "512"],
                    expected_type="error",
                    expected_fragment="unpack requires a buffer of 12 bytes",
                    check_subprocess=False,
                )

    def test_corrupt_adat_and_tada_headers_caught_without_traceback(self) -> None:
        def patch_u32(data: bytes, order: str, offset: int, value: int) -> bytes:
            buf = bytearray(data)
            struct.pack_into(order + "I", buf, offset, value)
            return bytes(buf)

        corrupt_cases = (
            ("bad_magic", b"RIFF" + _sfx_bank("<")[4:], "Expected an ADAT or TADA v1"),
            ("tada_magic_with_le_fields", b"TADA" + _sfx_bank("<")[4:], "Expected an ADAT or TADA v1"),
            ("adat_magic_with_be_fields", b"ADAT" + _music_bank(">")[4:], "Expected an ADAT or TADA v1"),
            ("unsupported_adat_flags", patch_u32(_sfx_bank("<"), "<", 4, 0xFF250001), "Expected an ADAT or TADA v1"),
            ("unsupported_tada_flags", patch_u32(_music_bank(">"), ">", 4, 0xFF010001), "Expected an ADAT or TADA v1"),
            ("zero_stream_count", patch_u32(_sfx_bank("<"), "<", 8, 0), "1 to 4096"),
            ("excess_stream_count", patch_u32(_music_bank(">"), ">", 8, 4097), "1 to 4096"),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, payload, fragment in corrupt_cases:
                with self.subTest(label=label):
                    archive = root / f"{label}.awc"
                    archive.write_bytes(payload)
                    self._assert_clean_failure(
                        [str(archive), "--offset", "0", "--size", str(len(payload))],
                        expected_type="ValueError",
                        expected_fragment=fragment,
                    )


if __name__ == "__main__":
    unittest.main()
