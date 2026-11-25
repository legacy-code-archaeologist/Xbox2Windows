"""Exercise the local compute index CLI with source-built framing controls."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import zlib


TOOL = Path(__file__).resolve().parents[1] / "tools" / "compute_container_index.py"
DIGESTS = [bytes([index]) * 16 for index in range(1, 8)]


def container(hash_part, *, stage=5, magic=b"DXIL"):
    program = struct.pack("<II4sIII", stage << 16, 7, magic, 0, 16, 4) + b"stub"
    parts = [(b"DXIL", program)]
    if hash_part is not None:
        parts.append((b"HASH", hash_part))
    pos = 32 + 4 * len(parts)
    offsets = []
    payload = bytearray()
    for name, part in parts:
        offsets.append(pos + len(payload))
        payload.extend(struct.pack("<4sI", name, len(part)) + part)
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, pos + len(payload), len(parts))
            + struct.pack(f"<{len(offsets)}I", *offsets) + payload)


def archive_record(index, blob):
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(blob) + compressor.flush()
    size = len(compressed)
    return f"fixture-{index}".encode() + bytes(2) + struct.pack("<4I", size, 0, size, len(blob)) + compressed


def slot_data(digest):
    return struct.pack("<III", 0x403, 24, 0) + struct.pack("<I", 0x43) + digest + bytes(12)


def key(index):
    return f"{index:016x}"


class ComputeContainerIndex(unittest.TestCase):
    def test_join_duplicates_conflicts_and_dispatch_selection(self):
        unique = container(bytes(4) + DIGESTS[0])
        conflict_a = container(bytes(4) + DIGESTS[1])
        conflict_b = container(struct.pack("<I", 1) + DIGESTS[1])
        invalid_flags = container(struct.pack("<I", 3) + DIGESTS[2])
        invalid_size = container(bytes(4) + DIGESTS[2][:-1])
        invalid_dxil = container(bytes(4) + DIGESTS[2], magic=b"bad!")
        wrong_stage = container(bytes(4) + DIGESTS[2], stage=0)
        no_hash = container(None)
        invalid_parts = bytearray(container(bytes(4) + DIGESTS[2]))
        struct.pack_into("<I", invalid_parts, 36, 40)  # duplicate DXIL part offset
        blobs = [unique, unique, conflict_a, conflict_b, invalid_flags, invalid_size,
                 invalid_dxil, wrong_stage, no_hash, invalid_parts]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "input.bin"
            archive.write_bytes(b"\0".join(archive_record(i, blob) for i, blob in enumerate(blobs)))
            cache = root / "cache"
            cache.mkdir()
            for index, digest in enumerate(DIGESTS, 1):
                stem = cache / f"req_{key(index)}"
                stem.with_suffix(".c67").write_bytes(b"native fixture marker")
                if index != 6:
                    data = slot_data(digest)
                    stem.with_suffix(".data").write_bytes(data if index != 5 else data[:-1] + b"!")
            trace = root / "dispatch.log"
            trace.write_text("\n".join([
                f"x2w-mesh-compute mode=direct key={key(1)} frame=10 real-pso=0",
                f"x2w-mesh-compute mode=direct key={key(1)} frame=11",
                f"x2w-mesh-compute mode=indirect key={key(2)} frame=12",
                f"x2w-mesh-compute mode=direct key={key(3)} frame=13",
                f"x2w-mesh-compute mode=unknown-signature key={key(4)} frame=14",
                f"x2w-mesh-compute mode=direct key={key(5)} frame=15",
                f"x2w-mesh-compute mode=indirect key={key(6)} frame=16",
                f"x2w-mesh-compute mode=direct key={key(7)}a frame=17",
                f"x2w-mesh-compute mode=direct key={key(8)} frame=18",
            ]), encoding="utf-8")
            original_files = set(root.rglob("*"))
            command = [sys.executable, "-B", str(TOOL), str(archive), str(cache)]
            report = json.loads(subprocess.check_output(command + ["--trace", str(trace)], text=True))
            full = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(set(root.rglob("*")), original_files)

        self.assertEqual(report["selection"], "dispatch-trace")
        self.assertEqual(report["summary"]["cache_requests"], 7)
        self.assertEqual(report["summary"]["selected_keys"], 6)
        self.assertEqual(report["summary"]["archive_records"], {
            "compute_records": 7, "usable_hash_records": 4, "invalid_hash_flags": 1,
            "invalid_hash_size": 1, "missing_hash": 1,
        })
        self.assertEqual(report["summary"]["archive_framing_rejections"], {
            "container_framing": 1, "no_program_header": 1,
        })
        rows = report["requests"]
        self.assertEqual(rows[key(1)], {"status": "unique", "dispatch_rows": {"direct": 2}, "candidates": [
            {"sha256": hashlib.sha256(unique).hexdigest(), "hash_flags": 0, "records": 2},
        ]})
        self.assertEqual(rows[key(2)]["status"], "conflict")
        self.assertEqual(rows[key(2)]["dispatch_rows"], {"indirect": 1})
        self.assertEqual(rows[key(2)]["candidates"], sorted([
            {"sha256": hashlib.sha256(blob).hexdigest(), "hash_flags": flags, "records": 1}
            for flags, blob in enumerate((conflict_a, conflict_b))
        ], key=lambda candidate: candidate["sha256"]))
        for index, status in [(3, "unmatched"), (5, "unrecognized-data"), (6, "missing-data"), (8, "missing-request")]:
            self.assertEqual(rows[key(index)]["status"], status)
            self.assertEqual(rows[key(index)]["candidates"], [])
        self.assertNotIn(key(4), rows)
        self.assertNotIn(key(7), rows)
        self.assertEqual(full["selection"], "cache")
        self.assertEqual(set(full["requests"]), {key(i) for i in range(1, 8)})
        self.assertNotIn("dispatch_rows", full["requests"][key(1)])


if __name__ == "__main__":
    unittest.main()
