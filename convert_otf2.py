#!/usr/bin/env python3
"""Recursively convert all .otf2 trace files under a directory using ~/otf2csv."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def find_otf2_files(root: Path):
    return sorted(root.rglob("*.otf2"))


def convert(otf2_path: Path, converter: Path, force: bool) -> tuple[Path, bool, str]:
    work_dir = otf2_path.parent
    expected_csv = work_dir / (otf2_path.stem + ".csv")

    if expected_csv.exists() and not force:
        return otf2_path, True, f"skipped (exists: {expected_csv.name})"

    cmd = f"cd {str(work_dir)!r} && {str(converter)!r} ./{otf2_path.name}"
    proc = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        return otf2_path, False, msg
    return otf2_path, True, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path, help="Root directory to search recursively")
    ap.add_argument(
        "--converter",
        type=Path,
        default=Path.home() / "otf2csv",
        help="Path to otf2csv binary (default: ~/otf2csv)",
    )
    ap.add_argument(
        "-f", "--force",
        action="store_true",
        help="Reconvert even if a matching .csv already exists",
    )
    ap.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="List files that would be converted, but don't run anything",
    )
    ap.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Run up to N conversions in parallel (default: 1). Use 0 for os.cpu_count().",
    )
    args = ap.parse_args()

    root = args.directory.expanduser().resolve()
    converter = args.converter.expanduser().resolve()

    if not root.is_dir():
        sys.exit(f"error: {root} is not a directory")
    if not args.dry_run and not os.access(converter, os.X_OK):
        sys.exit(f"error: converter {converter} is not executable")

    files = find_otf2_files(root)
    if not files:
        print(f"No .otf2 files found under {root}")
        return

    print(f"Found {len(files)} .otf2 file(s) under {root}")
    if args.dry_run:
        for f in files:
            print(f"  would convert: {f}")
        return

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = min(jobs, len(files))

    ok_count = 0
    fail_count = 0
    total = len(files)

    if jobs <= 1:
        for i, f in enumerate(files, 1):
            rel = f.relative_to(root)
            print(f"[{i}/{total}] {rel} ... ", end="", flush=True)
            _, ok, msg = convert(f, converter, args.force)
            print(msg)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
    else:
        print(f"Running with {jobs} parallel workers")
        done = 0
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            future_to_file = {
                ex.submit(convert, f, converter, args.force): f for f in files
            }
            for fut in as_completed(future_to_file):
                done += 1
                f = future_to_file[fut]
                rel = f.relative_to(root)
                try:
                    _, ok, msg = fut.result()
                except Exception as e:
                    ok, msg = False, f"exception: {e}"
                print(f"[{done}/{total}] {rel} ... {msg}")
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1

    print(f"\nDone: {ok_count} succeeded, {fail_count} failed")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
