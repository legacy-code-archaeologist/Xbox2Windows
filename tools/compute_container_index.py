"""Index local slot 67 compute requests against framed DXIL archive records."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct

from resource_inventory import scene_records


REQUEST = re.compile(r"req_([0-9a-f]{16})\.c67")
DISPATCH = re.compile(r"^x2w-mesh-compute mode=(direct|indirect) key=([0-9a-f]{16})(?=\s|$)")


def request_digest(data):
    """Read the digest from the observed slot 67 data shape."""
    if (len(data) >= 40 and data[:4] == b"\x03\x04\0\0"
            and struct.unpack_from("<I", data, 4)[0] == len(data) - 20
            and data[-32:-28] == b"\x43\0\0\0" and data[-12:] == bytes(12)):
        return data[-28:-12]
    return None


def dispatch_counts(trace):
    counts = defaultdict(Counter)
    with trace.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = DISPATCH.match(line)
            if match:
                mode, key = match.groups()
                counts[key][mode] += 1
    return counts


def build_index(archive, cache, trace):
    observed = dispatch_counts(trace) if trace is not None else None
    request_paths = {match[1]: path for path in cache.iterdir()
                     if path.is_file() and (match := REQUEST.fullmatch(path.name))}
    keys = sorted(observed if observed is not None else request_paths)
    rows = {}
    digests = {}
    for key in keys:
        row = rows[key] = {"status": "missing-request", "candidates": []}
        if observed is not None:
            row["dispatch_rows"] = dict(observed[key])
        if key not in request_paths:
            continue
        try:
            data = request_paths[key].with_suffix(".data").read_bytes()
        except FileNotFoundError:
            row["status"] = "missing-data"
            continue
        digest = request_digest(data)
        row["status"] = "unmatched" if digest is not None else "unrecognized-data"
        if digest is not None:
            digests[key] = digest

    archive_counts = Counter()
    wanted = set(digests.values())
    candidates = defaultdict(dict)

    def observe(kind, blob, sections):
        if kind != 5:
            return
        archive_counts["compute_records"] += 1
        part = sections.get(b"HASH")
        if part is None:
            archive_counts["missing_hash"] += 1
            return
        if len(part) != 20:
            archive_counts["invalid_hash_size"] += 1
            return
        flags, digest = struct.unpack("<I16s", part)
        # DXC defines bit 0 as IncludesSource. No other bits are defined.
        if flags & ~1:
            archive_counts["invalid_hash_flags"] += 1
            return
        archive_counts["usable_hash_records"] += 1
        if digest in wanted:
            sha = hashlib.sha256(blob).hexdigest()
            candidate = candidates[digest].setdefault(sha, {"sha256": sha, "hash_flags": flags, "records": 0})
            candidate["records"] += 1

    with archive.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        inventory = scene_records(data, observe=observe)
    for key, digest in digests.items():
        matches = sorted(candidates[digest].values(), key=lambda candidate: candidate["sha256"])
        if not matches:
            continue
        row = rows[key]
        row["status"] = "unique" if len(matches) == 1 else "conflict"
        row["candidates"] = matches

    return {
        "selection": "dispatch-trace" if observed is not None else "cache",
        "summary": {
            "cache_requests": len(request_paths),
            "selected_keys": len(keys),
            "request_statuses": dict(Counter(row["status"] for row in rows.values())),
            "archive_records": dict(archive_counts),
            "archive_framing_rejections": dict(inventory["candidate_rejections"]),
        },
        "requests": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="local archive with named shader records")
    parser.add_argument("cache", type=Path, help="local runtime cache containing req_<key>.c67 and .data files")
    parser.add_argument("--trace", type=Path, help="select keys from direct and indirect dispatch trace rows")
    args = parser.parse_args(argv)
    try:
        report = build_index(args.archive, args.cache, args.trace)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
