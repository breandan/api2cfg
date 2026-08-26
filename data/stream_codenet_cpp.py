#!/usr/bin/env python3
"""Stream standalone, locally compilable C++ submissions from Project CodeNet."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import BinaryIO


DEFAULT_ARCHIVE = Path(__file__).resolve().with_name("Project_CodeNet.tar.gtar")
CPP_SUBMISSION = re.compile(r"Project_CodeNet/data/p[0-9]{5}/C\+\+/s[0-9]{9}\.cpp").fullmatch


def is_cpp_submission(name: str) -> bool:
    return CPP_SUBMISSION(name) is not None


def stream_submissions(
    archive_path: Path,
    limit: int,
    output: BinaryIO,
) -> tuple[int, int]:
    """Write up to ``limit`` compilable submissions; return emitted and checked counts."""

    emitted = 0
    checked = 0
    if limit == 0:
        return emitted, checked

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not is_cpp_submission(member.name):
                continue

            source = archive.extractfile(member)
            if source is None:
                continue
            with source:
                data = source.read()

            checked += 1
            if subprocess.run(["g++", "-std=c++17", "-x", "c++", "-", "-o", os.devnull], input=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                continue

            output.write(f"// --- {member.name} ---\n".encode())
            output.write(data)
            if not data.endswith(b"\n"):
                output.write(b"\n")
            output.write(b"\n")
            output.flush()
            emitted += 1
            if emitted == limit:
                break

    return emitted, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=100,
        help="number of compilable files to stream (default: 100)",
    )
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE, help="CodeNet tar archive")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if not args.archive.is_file():
        parser.error(f"archive not found: {args.archive}")
    if args.limit == 0:
        return 0

    try:
        emitted, checked = stream_submissions(args.archive, args.limit, sys.stdout.buffer)
    except BrokenPipeError:
        # Avoid another broken-pipe error when Python flushes stdout at exit.
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
        return 0

    if emitted < args.limit:
        print(
            f"warning: found only {emitted} standalone C++ submissions after checking {checked}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
