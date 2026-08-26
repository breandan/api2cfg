#!/usr/bin/env python3
"""Stream the first Project CodeNet Python submissions that pass ty."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO


DEFAULT_ARCHIVE = "Project_CodeNet.tar.gtar"
PYTHON_SUBMISSION = re.compile(r"Project_CodeNet/data/p[0-9]{5}/Python/s[0-9]{9}\.py").fullmatch

def is_python_submission(name: str) -> bool:
    return PYTHON_SUBMISSION(name) is not None


def stream_submissions(archive_path: Path, limit: int, output: BinaryIO) -> int:
    """Write up to ``limit`` Python submissions that pass ty."""

    count = 0
    if limit == 0:
        return count

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not is_python_submission(member.name):
                continue

            source = archive.extractfile(member)
            if source is None:
                continue

            with source:
                data = source.read()

            with tempfile.NamedTemporaryFile(suffix=".py") as candidate:
                candidate.write(data)
                candidate.flush()
                if subprocess.run(["ty", "check", candidate.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                    continue

            output.write(f"# --- {member.name} ---\n".encode())
            output.write(data)
            output.write(b"\n\n")
            output.flush()
            count += 1
            if count == limit:
                break

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--limit", type=int, default=100, help="number of files to stream (default: 100)")
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE, help="path to the CodeNet tar archive")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if not args.archive.is_file():
        parser.error(f"archive not found: {args.archive}")

    try:
        count = stream_submissions(args.archive, args.limit, sys.stdout.buffer)
    except BrokenPipeError:
        # Avoid another broken-pipe error when Python flushes stdout at exit.
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
        return 0
    if count < args.limit:
        print(f"warning: found only {count} Python submissions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
