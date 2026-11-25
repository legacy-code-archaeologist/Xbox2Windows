"""Run independent Python check scripts side by side and fail if any of them fails.

Usage:
    python tools/run_checks_parallel.py [-j N] tests/a_check.py tests/b_check.py ...

Each script runs as `python -B <script>` from the current directory. Output is printed one
script at a time, in the order given, so a failure log stays readable. The exit status is 1
when any script exits non-zero.
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time


def run_one(script):
    start = time.monotonic()
    proc = subprocess.run([sys.executable, "-B", script], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace")
    return script, proc.returncode, proc.stdout, time.monotonic() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 2)
    parser.add_argument("scripts", nargs="+")
    args = parser.parse_args(argv)
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for script, code, output, secs in pool.map(run_one, args.scripts):
            print(f"::group::{script} ({'ok' if code == 0 else f'exit {code}'}, {secs:.1f}s)")
            sys.stdout.write(output)
            if output and not output.endswith("\n"):
                print()
            print("::endgroup::")
            if code != 0:
                failed.append(script)
    if failed:
        print("FAILED: " + " ".join(failed))
        return 1
    print(f"all {len(args.scripts)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
