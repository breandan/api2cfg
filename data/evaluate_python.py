#!/usr/bin/env python3
"""Evaluate cursor-specific Python CFGs on APPS or Project CodeNet.

The evaluator keeps the held-out statement away from the ``ty`` language
server used to construct its grammar.  It supports one-line expression
statements built from names, literals, attributes, calls, arithmetic binary
operators, subscripts, and slices, together with simple
single-name bindings whose right-hand side is in the same fragment.  A live
binding name is inferred from the ablated context; a private binding is
canonicalized to a fresh-name placeholder.

For every eligible statement in every selected source file it independently:

1. requires the original source to have no ty errors and prepares an ablated
   or semantically sealed context without exposing the held-out RHS;
2. queries ty completion, hover, member completion, and signature help at the
   ablated line;
3. lowers the returned semantic information into a lexicalized, type-indexed
   CFG;
4. recognizes the canonicalized ground-truth token sequence; and
5. draws a uniform integer from ``[0, 10^6)``, decodes that global shortlex
   rank through an exact DFA bijection, then reinserts the word and checks it
   with ty.

The evaluator implementation itself uses only the Python standard library.  It
streams APPS solutions from a split JSONL file, or CodeNet submissions from its
tarball without extracting either dataset.  Evaluated sources may use
third-party packages available in ty's pinned semantic environment; compatible
precomputed fragments under ``data/lib`` are loaded for visible imports.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import builtins
import contextlib
import functools
import hashlib
import heapq
import io
import importlib.metadata
import itertools
import json
import keyword
import math
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tokenize
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TextIO, cast
from urllib.parse import quote, unquote


DEFAULT_CODENET_ARCHIVE = Path(__file__).resolve().with_name(
    "Project_CodeNet.tar.gtar"
)
DEFAULT_APPS_DIRECTORY = Path(__file__).resolve().with_name("apps")
DEFAULT_LIBRARY_DIRECTORY = Path(__file__).resolve().with_name("lib")
LIBRARY_CFG_SCHEMA = "2"
SUPPORTED_LIBRARY_CFG_SCHEMAS = frozenset({"1", LIBRARY_CFG_SCHEMA})
ASSIGNABILITY_RELATION_VERSION = 2
SAMPLE_RANK_LIMIT = 1_000_000
DEFAULT_MAX_DFA_STATES = 500_000


def library_cfg_filename(module: str) -> str:
    """Map an absolute module name injectively to its cache filename."""

    # Python identifiers cannot contain ``%``, so escaping dots this way keeps
    # common root-module names readable while distinguishing ``a.b`` from
    # the unrelated root module ``a_b``.
    return f"{module.replace('.', '%2E')}.cfg"
PYTHON_SUBMISSION = re.compile(
    r"Project_CodeNet/data/p[0-9]{5}/Python/s[0-9]{9}\.py"
).fullmatch
FRESH_TOKEN = "@fresh"
DYNAMIC_NONTERMINAL = "E:__contextual_dynamic__"
TRUSTED_DYNAMIC_CALL_NONTERMINAL = "C:__trusted_dynamic_call__"
SLICE_NONTERMINAL = "Q:__slice__"
SLICE_BOUND_NONTERMINAL = "Q:__slice_bound__"
DYNAMIC_COMPOSITION_ATOM_NONTERMINAL = "E:__contextual_dynamic_atom__"
DYNAMIC_BINARY_OPERAND_NONTERMINAL = "E:__contextual_binary_operand__"

LSP_METHOD = 2
LSP_FUNCTION = 3
LSP_CONSTRUCTOR = 4
LSP_CLASS = 7
LSP_MODULE = 9
MAX_PROGRESSIVE_COMPLETION_PASSES = 8

CALLABLE_KINDS = {LSP_METHOD, LSP_FUNCTION, LSP_CONSTRUCTOR, LSP_CLASS}
IGNORED_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.COMMENT,
}
SUPPORTED_CONSTANT_TYPES = (type(None), bool, int, float, complex, str, bytes)
SUPPORTED_BINARY_OPERATOR_TYPES = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
BINARY_OPERATOR_SYMBOLS: Mapping[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
}


def surface_fragment_metadata() -> dict[str, object]:
    """Describe syntax choices that materially change the candidate language."""

    return {
        "binary_operators": list(BINARY_OPERATOR_SYMBOLS.values()),
        "subscripts": True,
        "slices": True,
        "list_displays": False,
    }


BUILTIN_NAMES = frozenset(dir(builtins))
CORE_BUILTINS = frozenset(
    {
        "abs",
        "dict",
        "enumerate",
        "eval",
        "float",
        "input",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "reversed",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)
CORE_MEMBERS = frozenset(
    {
        "add",
        "append",
        "clear",
        "copy",
        "count",
        "discard",
        "endswith",
        "extend",
        "find",
        "get",
        "heapify",
        "index",
        "insert",
        "items",
        "join",
        "keys",
        "lower",
        "linalg",
        "max",
        "mean",
        "min",
        "norm",
        "ones",
        "pop",
        "popleft",
        "read",
        "readline",
        "readlines",
        "remove",
        "replace",
        "reverse",
        "setdefault",
        "setrecursionlimit",
        "shuffle",
        "sort",
        "sqrt",
        "split",
        "startswith",
        "strip",
        "stdin",
        "stdout",
        "buffer",
        "update",
        "upper",
        "values",
        "write",
        "zeros",
        "array",
        "argsort",
        "argmax",
        "argmin",
        "reshape",
        "concatenate",
        "stack",
        "dot",
    }
)


class EvaluationError(RuntimeError):
    """An expected benchmark operation could not be completed."""


class TyTransportError(EvaluationError):
    """The ty language-server transport closed during an evaluation."""


class StatementTimeout(Exception):
    """One complete prepared-target evaluation exceeded its wall deadline."""


class LanguageTooLarge(EvaluationError):
    """Exact bounded-language determinization exceeded its configured cap."""


@contextlib.contextmanager
def wall_deadline(
    seconds: float,
    exception: Callable[[], Exception],
) -> Iterator[None]:
    """Compose a wall deadline with any earlier active ``ITIMER_REAL``."""

    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def expired(_signum: int, _frame: object) -> None:
        raise exception()

    old_handler = signal.getsignal(signal.SIGALRM)
    old_remaining, old_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    outer_expires_first = 0 < old_remaining <= seconds
    if outer_expires_first:
        active_handler = old_handler
        active_seconds = old_remaining
    else:
        active_handler = expired
        active_seconds = seconds
    signal.signal(signal.SIGALRM, active_handler)
    signal.setitimer(signal.ITIMER_REAL, active_seconds)
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        restored_remaining = (
            max(0.0, old_remaining - elapsed) if old_remaining > 0 else 0.0
        )
        if restored_remaining > 0 or old_interval > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                restored_remaining,
                old_interval,
            )


@contextlib.contextmanager
def statement_deadline(seconds: float) -> Iterator[None]:
    """Bound one end-to-end prepared-target attempt."""

    with wall_deadline(
        seconds,
        lambda: StatementTimeout(
            f"statement evaluation exceeded {seconds:g}s end-to-end deadline"
        ),
    ):
        yield


def stable_digest(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def compact_cardinality(value: int) -> str:
    """Render a language cardinality without Python's huge-int digit limit."""

    if value < 0:
        raise ValueError("cardinality must be nonnegative")
    try:
        return str(value)
    except ValueError:
        # Python limits decimal conversion of very large integers.  Preserve
        # exact strings in the ordinary case and use a deterministic six-digit
        # scientific rendering only beyond that runtime safety limit.
        if value == 0:
            return "0"
        bits = value.bit_length()
        shift = max(0, bits - 53)
        leading = value >> shift
        logarithm = math.log10(leading) + shift * math.log10(2)
        exponent = math.floor(logarithm)
        mantissa = 10 ** (logarithm - exponent)
        return f"{mantissa:.6g}e+{exponent}"


def source_encoding(data: bytes) -> str:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    except (LookupError, SyntaxError, UnicodeDecodeError):
        return "utf-8"
    return encoding


def decode_source(data: bytes) -> str | None:
    try:
        return data.decode(source_encoding(data))
    except (LookupError, UnicodeDecodeError):
        return None


def default_dataset_source(dataset: str, split: str) -> Path:
    """Return the default on-disk source for a dataset and APPS split."""

    if dataset == "apps":
        if split not in {"train", "test"}:
            raise ValueError(f"unsupported APPS split: {split}")
        return DEFAULT_APPS_DIRECTORY / f"{split}.jsonl"
    if dataset == "codenet":
        return DEFAULT_CODENET_ARCHIVE
    raise ValueError(f"unsupported dataset: {dataset}")


def resolved_dataset_source(dataset: str, source: Path, split: str) -> Path:
    """Resolve an APPS directory to its selected split JSONL file."""

    if dataset == "apps":
        if source.is_dir():
            return source / f"{split}.jsonl"
        if source.suffix.lower() != ".jsonl":
            raise EvaluationError(
                f"APPS source must be a directory or JSONL file, not {source}; "
                "use --dataset codenet for a CodeNet archive"
            )
        named_split = source.stem if source.stem in {"train", "test"} else None
        if named_split is not None and named_split != split:
            raise EvaluationError(
                f"APPS source {source.name} conflicts with --split {split}; "
                f"use --split {named_split}"
            )
    return source


@dataclass(frozen=True)
class DatasetSource:
    """One independently evaluated Python source from a streamed dataset."""

    member: str
    data: bytes
    dataset: str
    split: str | None = None
    problem_id: str | int | None = None
    solution_index: int | None = None
    difficulty: str | None = None
    url: str | None = None


def decode_apps_solutions(
    value: object,
    *,
    path: Path,
    line_number: int,
) -> list[str]:
    """Decode the APPS ``solutions`` field in either supported representation."""

    if isinstance(value, str):
        # APPS uses an empty string, rather than ``[]``, for the 1,235 test
        # problems that have no reference solutions.
        if not value.strip():
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"{path}:{line_number}: invalid JSON in solutions field: {error}"
            ) from error
    if not isinstance(value, list) or not all(
        isinstance(solution, str) for solution in value
    ):
        raise EvaluationError(
            f"{path}:{line_number}: solutions must decode to a list of strings"
        )
    return value


def apps_member_name(
    split: str,
    problem_id: str | int,
    solution_index: int,
) -> str:
    """Return a stable virtual filename for an APPS solution."""

    encoded_problem = quote(str(problem_id), safe="")
    return (
        f"APPS/{split}/{encoded_problem}/"
        f"solution_{solution_index:04d}.py"
    )


def _iter_apps_sources(
    path: Path,
    split: str,
    shard_count: int,
    shard_index: int,
) -> Iterator[DatasetSource]:
    source_index = 0
    with path.open("r", encoding="utf-8") as rows:
        for line_number, line in enumerate(rows, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(
                    f"{path}:{line_number}: invalid APPS JSONL row: {error}"
                ) from error
            if not isinstance(row, dict):
                raise EvaluationError(
                    f"{path}:{line_number}: APPS row must be a JSON object"
                )
            problem_id = row.get("id")
            if not isinstance(problem_id, (str, int)) or isinstance(
                problem_id, bool
            ):
                raise EvaluationError(
                    f"{path}:{line_number}: APPS row has no scalar id"
                )
            solutions = decode_apps_solutions(
                row.get("solutions"),
                path=path,
                line_number=line_number,
            )
            difficulty = row.get("difficulty")
            if not isinstance(difficulty, str):
                difficulty = None
            url = row.get("url")
            if not isinstance(url, str):
                url = None
            for solution_index, solution in enumerate(solutions):
                owned = source_index % shard_count == shard_index
                source_index += 1
                if not owned:
                    continue
                try:
                    data = solution.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise EvaluationError(
                        f"{path}:{line_number}: solution {solution_index} "
                        "is not valid UTF-8"
                    ) from error
                yield DatasetSource(
                    member=apps_member_name(
                        split,
                        problem_id,
                        solution_index,
                    ),
                    data=data,
                    dataset="apps",
                    split=split,
                    problem_id=problem_id,
                    solution_index=solution_index,
                    difficulty=difficulty,
                    url=url,
                )


def _iter_codenet_sources(
    path: Path,
    shard_count: int,
    shard_index: int,
) -> Iterator[DatasetSource]:
    source_index = 0
    try:
        with tarfile.open(path, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or PYTHON_SUBMISSION(member.name) is None:
                    continue
                owned = source_index % shard_count == shard_index
                source_index += 1
                if not owned:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                with extracted:
                    data = extracted.read()
                yield DatasetSource(
                    member=member.name,
                    data=data,
                    dataset="codenet",
                )
    except (tarfile.ReadError, tarfile.CompressionError) as error:
        raise EvaluationError(f"invalid CodeNet tar archive {path}: {error}") from error


def iter_dataset_sources(
    dataset: str,
    source: Path,
    split: str,
    shard_count: int = 1,
    shard_index: int = 0,
) -> Iterator[DatasetSource]:
    """Stream dataset sources, sharding APPS over solutions rather than rows."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    path = resolved_dataset_source(dataset, source, split)
    if not path.is_file():
        raise EvaluationError(f"dataset source not found: {path}")
    if dataset == "apps":
        yield from _iter_apps_sources(path, split, shard_count, shard_index)
    elif dataset == "codenet":
        yield from _iter_codenet_sources(path, shard_count, shard_index)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")


def uri_for(path: Path) -> str:
    return path.resolve().as_uri()


class TyLspClient:
    """Small synchronous JSON-RPC client for ``ty server``."""

    def __init__(self, executable: str, workspace: Path, *, quiet: bool = True):
        self.executable = executable
        self.workspace = workspace
        self.quiet = quiet
        stderr = subprocess.DEVNULL if quiet else None
        self.process = subprocess.Popen(
            [executable, "server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise EvaluationError("failed to open ty language-server pipes")
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        self.next_id = 0
        self.document_uri: str | None = None
        self.document_version = 0
        try:
            result = self.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": uri_for(workspace),
                    "workspaceFolders": [
                        {"uri": uri_for(workspace), "name": workspace.name}
                    ],
                    "capabilities": {
                        "general": {"positionEncodings": ["utf-8"]},
                        "textDocument": {
                            "completion": {
                                "completionItem": {"labelDetailsSupport": True}
                            },
                            "hover": {
                                "contentFormat": ["plaintext"],
                            },
                            "signatureHelp": {
                                "signatureInformation": {
                                    "parameterInformation": {
                                        "labelOffsetSupport": False
                                    }
                                }
                            },
                        },
                    },
                    "initializationOptions": {
                        "completions": {"autoImport": False},
                    },
                },
            )
        except BaseException:
            self._terminate_process()
            self._close_pipes()
            raise
        result_map = result if isinstance(result, dict) else {}
        capabilities = result_map.get("capabilities", {})
        capabilities_map = capabilities if isinstance(capabilities, dict) else {}
        position_encoding = capabilities_map.get("positionEncoding", "utf-16")
        self.position_encoding = (
            position_encoding if isinstance(position_encoding, str) else "utf-16"
        )
        try:
            self.notify("initialized", {})
        except BaseException:
            self._terminate_process()
            self._close_pipes()
            raise

    def _write(self, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self.stdin.write(
                f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            )
            self.stdin.write(body)
            self.stdin.flush()
        except BrokenPipeError as error:
            raise TyTransportError(
                "ty language server closed its input"
            ) from error

    def notify(self, method: str, params: object | None = None) -> None:
        payload: dict[str, object] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._write(payload)

    def request(self, method: str, params: object | None = None) -> object:
        self.next_id += 1
        request_id = self.next_id
        payload: dict[str, object] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        self._write(payload)
        while True:
            message = self._read_message()
            if "method" in message:
                if "id" in message:
                    self._answer_server_request(message)
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise EvaluationError(
                        f"ty {method} failed: {message['error']}"
                    )
                return message.get("result")

    def _read_message(self) -> dict[str, object]:
        headers: dict[str, str] = {}
        while True:
            line = self.stdout.readline()
            if not line:
                return_code = self.process.poll()
                detail = f" (exit {return_code})" if return_code is not None else ""
                raise TyTransportError(
                    f"ty language server closed its output{detail}"
                )
            if line in {b"\r\n", b"\n"}:
                break
            try:
                name, value = line.decode("ascii").split(":", 1)
            except ValueError as error:
                raise EvaluationError(f"invalid LSP header: {line!r}") from error
            headers[name.lower()] = value.strip()
        try:
            length = int(headers["content-length"])
        except (KeyError, ValueError) as error:
            raise EvaluationError("LSP response omitted Content-Length") from error
        body = self.stdout.read(length)
        if len(body) != length:
            return_code = self.process.poll()
            detail = f" (exit {return_code})" if return_code is not None else ""
            raise TyTransportError(
                "ty language server closed its output while reading an LSP "
                f"message{detail}"
            )
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"invalid LSP JSON: {body[:200]!r}") from error
        if not isinstance(decoded, dict):
            raise EvaluationError("LSP response was not an object")
        return decoded

    def _answer_server_request(self, message: Mapping[str, object]) -> None:
        method = message.get("method")
        if method == "workspace/configuration":
            params = message.get("params")
            items = params.get("items", []) if isinstance(params, dict) else []
            result: object = [None for _ in items]
        else:
            result = None
        self._write({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def open(self, document_uri: str, text: str) -> None:
        if self.document_uri is not None:
            self.notify(
                "textDocument/didClose",
                {"textDocument": {"uri": self.document_uri}},
            )
        self.document_uri = document_uri
        self.document_version = 1
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": document_uri,
                    "languageId": "python",
                    "version": self.document_version,
                    "text": text,
                }
            },
        )

    def change(self, text: str) -> None:
        if self.document_uri is None:
            raise EvaluationError("no open ty document")
        self.document_version += 1
        self.notify(
            "textDocument/didChange",
            {
                "textDocument": {
                    "uri": self.document_uri,
                    "version": self.document_version,
                },
                "contentChanges": [{"text": text}],
            },
        )

    def completion(
        self,
        line: int,
        character: int,
        *,
        trigger: str | None = None,
        retrigger_incomplete: bool = False,
    ) -> tuple[list[dict[str, object]], bool]:
        if self.document_uri is None:
            raise EvaluationError("no open ty document")
        context: dict[str, object]
        if retrigger_incomplete:
            context = {"triggerKind": 3}
        elif trigger is not None:
            context = {"triggerKind": 2, "triggerCharacter": trigger}
        else:
            context = {"triggerKind": 1}
        result = self.request(
            "textDocument/completion",
            {
                "textDocument": {"uri": self.document_uri},
                "position": {"line": line, "character": character},
                "context": context,
            },
        )
        if result is None:
            return [], False
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)], False
        if not isinstance(result, dict):
            return [], False
        items = result.get("items", [])
        return (
            [item for item in items if isinstance(item, dict)],
            bool(result.get("isIncomplete", False)),
        )

    def hover(self, line: int, character: int) -> str | None:
        if self.document_uri is None:
            raise EvaluationError("no open ty document")
        result = self.request(
            "textDocument/hover",
            {
                "textDocument": {"uri": self.document_uri},
                "position": {"line": line, "character": character},
            },
        )
        if not isinstance(result, dict):
            return None
        contents = result.get("contents")
        if isinstance(contents, str):
            return contents.strip()
        if isinstance(contents, dict):
            value = contents.get("value")
            return value.strip() if isinstance(value, str) else None
        if isinstance(contents, list):
            values: list[str] = []
            for item in contents:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict) and isinstance(item.get("value"), str):
                    values.append(item["value"])
            return "\n".join(values).strip() or None
        return None

    def signature_help(self, line: int, character: int) -> list[str]:
        if self.document_uri is None:
            raise EvaluationError("no open ty document")
        result = self.request(
            "textDocument/signatureHelp",
            {
                "textDocument": {"uri": self.document_uri},
                "position": {"line": line, "character": character},
                "context": {"triggerKind": 1, "isRetrigger": False},
            },
        )
        if not isinstance(result, dict):
            return []
        signatures = result.get("signatures", [])
        return [
            signature["label"]
            for signature in signatures
            if isinstance(signature, dict) and isinstance(signature.get("label"), str)
        ]

    def diagnostics(self) -> list[dict[str, object]]:
        if self.document_uri is None:
            raise EvaluationError("no open ty document")
        result = self.request(
            "textDocument/diagnostic",
            {"textDocument": {"uri": self.document_uri}},
        )
        if not isinstance(result, dict):
            return []
        items = result.get("items", [])
        return [item for item in items if isinstance(item, dict)]

    def restart(self) -> None:
        """Replace a failed or timed-out server while preserving this client."""

        # ``restart`` is only used after transport failure or an evaluation
        # deadline.  A graceful shutdown can itself wait forever when ty is
        # still computing the request that timed out, so recovery must abort
        # the old process before starting its replacement.
        try:
            self._terminate_process()
        finally:
            self._close_pipes()
        replacement: TyLspClient | None = None
        adopted = False
        try:
            replacement = type(self)(
                self.executable,
                self.workspace,
                quiet=self.quiet,
            )
            replacement_state = {
                "process": replacement.process,
                "stdin": replacement.stdin,
                "stdout": replacement.stdout,
                "next_id": replacement.next_id,
                "document_uri": replacement.document_uri,
                "document_version": replacement.document_version,
                "position_encoding": replacement.position_encoding,
            }
            self.__dict__.update(replacement_state)
            adopted = True
        except BaseException:
            if replacement is not None and not adopted:
                replacement.close()
            raise

    def _terminate_process(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def _close_pipes(self) -> None:
        for pipe in (self.stdin, self.stdout):
            try:
                pipe.close()
            except OSError:
                pass

    def close(self) -> None:
        try:
            try:
                if self.process.poll() is not None:
                    return
                self.request("shutdown")
                self.notify("exit")
                self.process.wait(timeout=5)
            except (BrokenPipeError, EvaluationError, subprocess.TimeoutExpired):
                self._terminate_process()
            except BaseException:
                self._terminate_process()
                raise
        finally:
            self._close_pipes()

    def __enter__(self) -> TyLspClient:
        return self

    def __exit__(self, *_errors: object) -> None:
        self.close()


def lsp_character(prefix: str, encoding: str) -> int:
    if encoding == "utf-8":
        return len(prefix.encode("utf-8"))
    if encoding == "utf-32":
        return len(prefix)
    return len(prefix.encode("utf-16-le")) // 2


@dataclass(frozen=True)
class Hole:
    """A one-line source span that can be replaced without moving the suffix."""

    before: str
    after: str
    line: int
    indentation: str

    def render(self, statement: str) -> str:
        return f"{self.before}{statement}{self.after}"

    def character_after(self, inserted_prefix: str, encoding: str) -> int:
        return lsp_character(f"{self.indentation}{inserted_prefix}", encoding)


@dataclass(frozen=True)
class Target:
    node: ast.stmt
    hole: Hole
    text: str
    kind: str
    assigned_name: str | None = None
    fresh_name: str | None = None
    bound_before: bool = False
    loaded_after: bool = False


@dataclass(frozen=True)
class PreparedTarget:
    target: Target
    ablated: str
    semantic_source: str
    required_assignment: str | None = None
    expression_prefix: str = ""
    expression_suffix: str = ""
    excluded_names: frozenset[str] = frozenset()


def byte_column_to_character(line: str, byte_column: int) -> int:
    raw = line.encode("utf-8")
    return len(raw[:byte_column].decode("utf-8"))


def hole_for_node(source: str, node: ast.stmt) -> Hole | None:
    if node.end_lineno is None or node.end_col_offset is None:
        return None
    if node.lineno != node.end_lineno:
        return None
    lines = source.splitlines(keepends=True)
    line_index = node.lineno - 1
    if not (0 <= line_index < len(lines)):
        return None
    line = lines[line_index]
    start = byte_column_to_character(line, node.col_offset)
    end = byte_column_to_character(line, node.end_col_offset)
    trailing = line[end:].rstrip("\r\n")
    trailing_content = trailing.lstrip()
    if trailing_content and not trailing_content.startswith("#"):
        return None
    before = "".join(lines[:line_index]) + line[:start]
    after = line[end:] + "".join(lines[line_index + 1 :])
    indentation = line[:start]
    if indentation.strip():
        return None
    return Hole(before=before, after=after, line=line_index, indentation=indentation)


def supported_surface_node(root: ast.AST) -> bool:
    """Check the recursive expression fragment without Python recursion."""

    pending = [root]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Name):
            continue
        if isinstance(node, ast.Constant):
            if type(node.value) not in SUPPORTED_CONSTANT_TYPES:
                return False
            continue
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return False
            pending.append(node.value)
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
                return False
            pending.append(node.operand)
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, SUPPORTED_BINARY_OPERATOR_TYPES):
                return False
            pending.extend((node.right, node.left))
            continue
        if isinstance(node, ast.Subscript):
            pending.extend((node.slice, node.value))
            continue
        if isinstance(node, ast.Slice):
            pending.extend(
                item
                for item in (node.step, node.upper, node.lower)
                if item is not None
            )
            continue
        if isinstance(node, ast.Call):
            if any(isinstance(argument, ast.Starred) for argument in node.args):
                return False
            if any(item.arg is None for item in node.keywords):
                return False
            pending.append(node.func)
            pending.extend(reversed(node.args))
            pending.extend(item.value for item in reversed(node.keywords))
            continue
        return False
    return True


def supported_subscript_slice(node: ast.expr | ast.slice) -> bool:
    """Return whether one ordinary index or slice stays in the surface syntax."""

    return supported_surface_node(node)


def supported_expression(node: ast.expr) -> bool:
    return supported_surface_node(node)


def unsupported_surface_spelling_counts(
    source: str,
    root: ast.expr,
) -> Counter[str]:
    """Count spellings erased by the AST but absent from the surface CFG.

    Python drops a call's trailing comma and folds adjacent string literals
    into one ``Constant`` node.  Canonical tokenization retains both spellings,
    so accepting them here would select words that the grammar cannot derive.
    """

    issues: Counter[str] = Counter()
    pending: list[ast.AST] = [root]
    while pending:
        node = pending.pop()
        # String-valued Constant nodes inside an f-string are formatting
        # segments, not implicit literal concatenation.  The JoinedStr itself
        # is outside the surface fragment, but expressions interpolated into
        # either its value or a nested format specification still need their
        # spelling checked for a complete exclusion audit.
        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                if not isinstance(value, ast.FormattedValue):
                    continue
                pending.append(value.value)
                if value.format_spec is not None:
                    pending.append(value.format_spec)
            continue
        pending.extend(ast.iter_child_nodes(node))
        if not isinstance(node, (ast.Call, ast.Constant)):
            continue
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (str, bytes)
        ):
            continue
        segment = ast.get_source_segment(source, node)
        if segment is None:
            issues["source_segment"] += 1
            continue
        try:
            tokens = [
                item
                for item in tokenize.generate_tokens(io.StringIO(segment).readline)
                if item.type not in IGNORED_TOKEN_TYPES
            ]
        except (IndentationError, SyntaxError, tokenize.TokenError):
            issues["tokenization"] += 1
            continue
        if isinstance(node, ast.Call):
            if (
                len(tokens) >= 2
                and tokens[-1].string == ")"
                and tokens[-2].string == ","
            ):
                issues["trailing_call_comma"] += 1
        elif sum(item.type == tokenize.STRING for item in tokens) != 1:
            issues["adjacent_literal"] += 1
    return issues


def unsupported_surface_spellings(
    source: str,
    root: ast.expr,
) -> frozenset[str]:
    return frozenset(unsupported_surface_spelling_counts(source, root))


def surface_spelling_supported(source: str, root: ast.expr) -> bool:
    return not unsupported_surface_spellings(source, root)


LEXICAL_SCOPES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def enclosing_scope(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> ast.AST | None:
    current = parents.get(node)
    while current is not None and not isinstance(current, LEXICAL_SCOPES):
        current = parents.get(current)
    return current


def name_loaded_after(
    scope: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    name: str,
    line: int,
) -> bool:
    comprehension_scopes = (
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    def resolution_scope(node: ast.AST) -> ast.AST | None:
        current = parents.get(node)
        while current is not None and not isinstance(
            current, (*LEXICAL_SCOPES, *comprehension_scopes)
        ):
            current = parents.get(current)
        return current

    def declarations(
        lexical_scope: ast.AST,
    ) -> tuple[set[str], set[str], set[str]]:
        local: set[str] = set()
        global_names: set[str] = set()
        nonlocal_names: set[str] = set()
        if isinstance(
            lexical_scope,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            arguments = (
                *lexical_scope.args.posonlyargs,
                *lexical_scope.args.args,
                *lexical_scope.args.kwonlyargs,
            )
            if lexical_scope.args.vararg is not None:
                arguments += (lexical_scope.args.vararg,)
            if lexical_scope.args.kwarg is not None:
                arguments += (lexical_scope.args.kwarg,)
            local.update(argument.arg for argument in arguments)
        comprehension_targets = {
            target
            for comprehension in ast.walk(lexical_scope)
            if isinstance(comprehension, ast.comprehension)
            for target in ast.walk(comprehension.target)
            if isinstance(target, ast.Name)
        }
        for item in ast.walk(lexical_scope):
            if enclosing_scope(item, parents) is not lexical_scope:
                continue
            if isinstance(item, ast.Global):
                global_names.update(item.names)
            elif isinstance(item, ast.Nonlocal):
                nonlocal_names.update(item.names)
            elif (
                isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Store)
                and item not in comprehension_targets
            ):
                local.add(item.id)
            elif isinstance(
                item,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                local.add(item.name)
            elif isinstance(item, (ast.Import, ast.ImportFrom)):
                for alias in item.names:
                    local.add(alias.asname or alias.name.split(".", 1)[0])
            elif isinstance(item, ast.ExceptHandler) and item.name is not None:
                local.add(item.name)
            elif isinstance(item, (ast.MatchAs, ast.MatchStar)):
                if item.name is not None:
                    local.add(item.name)
            elif isinstance(item, ast.MatchMapping) and item.rest is not None:
                local.add(item.rest)
        local.difference_update(global_names)
        local.difference_update(nonlocal_names)
        return local, global_names, nonlocal_names

    declaration_cache: dict[int, tuple[set[str], set[str], set[str]]] = {}

    def cached_declarations(
        lexical_scope: ast.AST,
    ) -> tuple[set[str], set[str], set[str]]:
        key = id(lexical_scope)
        result = declaration_cache.get(key)
        if result is None:
            result = declarations(lexical_scope)
            declaration_cache[key] = result
        return result

    for node in ast.walk(scope):
        if not (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == name
            and getattr(node, "lineno", 0) > line
        ):
            continue
        current = resolution_scope(node)
        resolves_to_target = True
        while current is not None and current is not scope:
            if isinstance(current, comprehension_scopes):
                bound = {
                    target.id
                    for generator in current.generators
                    for target in ast.walk(generator.target)
                    if isinstance(target, ast.Name)
                }
                if name in bound:
                    resolves_to_target = False
                    break
            else:
                local, global_names, _nonlocal_names = cached_declarations(current)
                if name in global_names:
                    resolves_to_target = isinstance(scope, ast.Module)
                    break
                if name in local:
                    resolves_to_target = False
                    break
            current = resolution_scope(current)
        if current is None:
            resolves_to_target = False
        if resolves_to_target:
            return True
    return False


def bound_before(
    scope: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    name: str,
    line: int,
) -> bool:
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = (
            *scope.args.posonlyargs,
            *scope.args.args,
            *scope.args.kwonlyargs,
        )
        if scope.args.vararg is not None:
            arguments += (scope.args.vararg,)
        if scope.args.kwarg is not None:
            arguments += (scope.args.kwarg,)
        if any(argument.arg == name for argument in arguments):
            return True

    comprehension_targets = {
        target
        for comprehension in ast.walk(scope)
        if isinstance(comprehension, ast.comprehension)
        for target in ast.walk(comprehension.target)
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(scope):
        if (
            isinstance(node, (ast.Global, ast.Nonlocal))
            and name in node.names
            and enclosing_scope(node, parents) is scope
        ):
            return True
        if getattr(node, "lineno", line + 1) >= line:
            continue
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == name
            and node not in comprehension_targets
            and enclosing_scope(node, parents) is scope
        ):
            return True
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
            and enclosing_scope(node, parents) is scope
        ):
            return True
        if (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and enclosing_scope(node, parents) is scope
        ):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if bound == name:
                    return True
        if (
            isinstance(node, ast.ExceptHandler)
            and node.name == name
            and enclosing_scope(node, parents) is scope
        ):
            return True
        if (
            isinstance(node, (ast.MatchAs, ast.MatchStar))
            and node.name == name
            and enclosing_scope(node, parents) is scope
        ):
            return True
        if (
            isinstance(node, ast.MatchMapping)
            and node.rest == name
            and enclosing_scope(node, parents) is scope
        ):
            return True
    return False


def candidate_targets(
    source: str,
    tree: ast.Module,
    *,
    max_tokens: int | None = None,
) -> list[Target]:
    result: list[Target] = []
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        kind: str
        assigned_name: str | None = None
        fresh_name: str | None = None
        was_bound_before = False
        loaded_later = False
        expression: ast.expr
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.JoinedStr):
                continue
            expression = node.value
            kind = "expression"
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigned_name = node.targets[0].id
            scope = enclosing_scope(node, parents)
            if scope is None:
                continue
            was_bound_before = bound_before(
                scope, parents, assigned_name, node.lineno
            )
            loaded_later = name_loaded_after(
                scope,
                parents,
                assigned_name,
                node.lineno,
            )
            expression = node.value
            kind = "assignment"
        else:
            continue
        if not supported_expression(expression) or not surface_spelling_supported(
            source, expression
        ):
            continue
        hole = hole_for_node(source, node)
        if hole is None:
            continue
        text = ast.get_source_segment(source, node)
        if text is None:
            continue
        target = Target(
            node=node,
            hole=hole,
            text=text,
            kind=kind,
            assigned_name=assigned_name,
            fresh_name=fresh_name,
            bound_before=was_bound_before,
            loaded_after=loaded_later,
        )
        try:
            tokens = canonical_tokens(target)
        except (SyntaxError, ValueError, tokenize.TokenError):
            continue
        if tokens and (max_tokens is None or len(tokens) <= max_tokens):
            result.append(target)
    return result


def canonical_number(value: str) -> str:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, bool) or not isinstance(parsed, (int, float, complex)):
        raise ValueError(f"unsupported numeric token: {value}")
    if isinstance(parsed, complex):
        return "0j"
    if isinstance(parsed, float):
        return "0.0"
    return "0"


def canonical_string(value: str) -> str:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, bytes):
        return 'b""'
    if isinstance(parsed, str):
        return '""'
    raise ValueError(f"unsupported string token: {value}")


def canonical_tokens(target: Target) -> tuple[str, ...]:
    tokens: list[str] = []
    replaced_fresh = False
    reader = io.StringIO(target.text).readline
    for item in tokenize.generate_tokens(reader):
        if item.type in IGNORED_TOKEN_TYPES:
            continue
        value = item.string
        if item.type == tokenize.NUMBER:
            value = canonical_number(value)
        elif item.type == tokenize.STRING:
            value = canonical_string(value)
        elif (
            item.type == tokenize.NAME
            and target.fresh_name is not None
            and not replaced_fresh
            and value == target.fresh_name
        ):
            value = FRESH_TOKEN
            replaced_fresh = True
        elif item.type not in {tokenize.NAME, tokenize.OP}:
            raise ValueError(f"unsupported token {tokenize.tok_name[item.type]}")
        tokens.append(value)
    return tuple(tokens)


def source_identifiers(source: str) -> frozenset[str]:
    result: set[str] = set()
    try:
        stream = tokenize.generate_tokens(io.StringIO(source).readline)
        for item in stream:
            if item.type == tokenize.NAME and not keyword.iskeyword(item.string):
                result.add(item.string)
    except (IndentationError, tokenize.TokenError):
        pass
    return frozenset(result)


@dataclass(frozen=True)
class Completion:
    label: str
    detail: str
    kind: int | None


class SemanticProbe:
    """Issue semantic queries against source revisions derived from one hole."""

    def __init__(
        self,
        client: TyLspClient,
        hole: Hole,
        ablated: str,
        required_assignment: str | None = None,
        expression_prefix: str = "",
        expression_suffix: str = "",
        excluded_names: frozenset[str] = frozenset(),
    ):
        self.client = client
        self.hole = hole
        self.ablated = ablated
        self.current = ablated
        self.required_assignment = required_assignment
        self.expression_prefix = expression_prefix
        self.expression_suffix = expression_suffix
        self.excluded_names = excluded_names

    def _expression_statement(self, expression: str) -> str:
        return (
            f"{self.expression_prefix}{expression}{self.expression_suffix}"
        )

    def _expression_cursor_text(self, expression: str) -> str:
        return f"{self.expression_prefix}{expression}"

    def _change_statement(self, statement: str) -> str:
        source = self.hole.render(statement)
        self.client.change(source)
        self.current = source
        return source

    def _change_expression(self, expression: str) -> str:
        return self._change_statement(self._expression_statement(expression))

    def accepts_expression(self, expression: str) -> bool:
        self._change_expression(expression)
        return not error_diagnostics(self.client.diagnostics())

    def accepts_assignment(self, expression: str) -> bool:
        if self.required_assignment is None:
            return False
        local, downstream = self.assignment_diagnostic_partition(expression)
        return not local and not downstream

    def assignment_diagnostic_partition(
        self, expression: str
    ) -> tuple[
        tuple[Mapping[str, object], ...],
        tuple[Mapping[str, object], ...],
    ]:
        """Return local and continuation errors for an output assignment."""

        if self.required_assignment is None:
            return (), ()
        statement = f"{self.required_assignment} = {expression}"
        self._change_statement(statement)
        diagnostics = error_diagnostics(self.client.diagnostics())
        local: list[Mapping[str, object]] = []
        downstream: list[Mapping[str, object]] = []
        for diagnostic in diagnostics:
            destination = (
                local
                if diagnostic_overlaps_statement(
                    diagnostic,
                    self.hole,
                    statement,
                    self.client.position_encoding,
                )
                else downstream
            )
            destination.append(diagnostic)
        return tuple(local), tuple(downstream)

    def diagnostics(self, statement: str) -> list[dict[str, object]]:
        self._change_statement(statement)
        return self.client.diagnostics()

    def scope(self) -> tuple[list[Completion], bool]:
        self._change_expression("()")
        prefix = self._expression_cursor_text("(")
        character = self.hole.character_after(prefix, self.client.position_encoding)
        return progressive_completions(
            lambda retrigger: self.client.completion(
                self.hole.line,
                character,
                retrigger_incomplete=retrigger,
            )
        )

    def hover_expression(self, expression: str) -> str | None:
        self._change_expression(expression)
        inserted = self._expression_cursor_text(expression)
        character = self.hole.character_after(inserted, self.client.position_encoding)
        value = self.client.hover(self.hole.line, max(0, character - 1))
        return clean_hover(value)

    def members(self, expression: str) -> tuple[list[Completion], bool]:
        expression_inserted = f"{expression}."
        self._change_expression(expression_inserted)
        inserted = self._expression_cursor_text(expression_inserted)
        character = self.hole.character_after(inserted, self.client.position_encoding)
        return progressive_completions(
            lambda retrigger: self.client.completion(
                self.hole.line,
                character,
                trigger=".",
                retrigger_incomplete=retrigger,
            )
        )

    def signatures(self, expression: str) -> list[str]:
        expression_prefix = f"{expression}("
        expression_inserted = f"{expression_prefix})"
        self._change_expression(expression_inserted)
        prefix = self._expression_cursor_text(expression_prefix)
        character = self.hole.character_after(prefix, self.client.position_encoding)
        return self.client.signature_help(self.hole.line, character)


def clean_hover(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    value = value.split("---------------------------------------------", 1)[0].strip()
    return normalize_type(value) if value else None


def simplify_completions(items: Iterable[Mapping[str, object]]) -> list[Completion]:
    result: dict[str, Completion] = {}
    for item in items:
        label = item.get("label")
        detail = item.get("detail")
        kind = item.get("kind")
        if not isinstance(label, str) or not isinstance(detail, str):
            continue
        if item.get("additionalTextEdits"):
            continue
        if not label.isidentifier() or keyword.iskeyword(label):
            continue
        if label.startswith("_"):
            continue
        result[label] = Completion(
            label=label,
            detail=normalize_type(detail),
            kind=kind if isinstance(kind, int) else None,
        )
    return sorted(result.values(), key=lambda item: item.label)


def progressive_completions(
    fetch: Callable[[bool], tuple[list[dict[str, object]], bool]],
) -> tuple[list[Completion], bool]:
    """Union deterministic completion passes until complete or repeated.

    LSP completion lists marked ``isIncomplete`` must be retriggered with
    trigger kind 3.  Servers are permitted to keep returning incomplete lists,
    so a repeated normalized page terminates the loop and a hard pass bound is
    retained as a final guard.
    """

    merged: dict[str, Completion] = {}
    seen_pages: set[tuple[tuple[str, str, int | None], ...]] = set()
    saw_incomplete = False
    for pass_index in range(MAX_PROGRESSIVE_COMPLETION_PASSES):
        items, incomplete = fetch(pass_index > 0)
        saw_incomplete = saw_incomplete or incomplete
        page = simplify_completions(items)
        fingerprint = tuple(
            (item.label, item.detail, item.kind) for item in page
        )
        if fingerprint in seen_pages:
            break
        seen_pages.add(fingerprint)
        for item in page:
            current = merged.get(item.label)
            preference = (
                item.detail in {"Any", "Unknown"},
                item.detail,
                item.kind is None,
                -1 if item.kind is None else item.kind,
            )
            current_preference = (
                current.detail in {"Any", "Unknown"},
                current.detail,
                current.kind is None,
                -1 if current.kind is None else current.kind,
            ) if current is not None else None
            if current_preference is None or preference < current_preference:
                merged[item.label] = item
        if not incomplete:
            break
    return sorted(merged.values(), key=lambda item: item.label), saw_incomplete


def normalize_type(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = value.replace("typing.", "").replace("typing_extensions.", "")
    if value in {"NoneType", "types.NoneType"}:
        return "None"
    literal = literal_base_type(value)
    return literal or value


def split_top_level(value: str, delimiter: str = ",") -> list[str]:
    result: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}" and stack and stack[-1] == pairs[character]:
            stack.pop()
        elif character == delimiter and not stack:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    return result


def split_union(value: str) -> list[str]:
    result: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}" and stack and stack[-1] == pairs[character]:
            stack.pop()
        elif character == "|" and not stack:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    return [item for item in result if item]


def find_top_level(value: str, needle: str) -> int:
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}" and stack and stack[-1] == pairs[character]:
            stack.pop()
        elif character == needle and not stack:
            return index
    return -1


def literal_base_type(value: str) -> str | None:
    if not value.startswith("Literal[") or not value.endswith("]"):
        return None
    payload = value[len("Literal[") : -1]
    classes: set[str] = set()
    for item in split_top_level(payload):
        try:
            parsed = ast.literal_eval(item)
        except (SyntaxError, ValueError):
            return None
        if parsed is None:
            classes.add("None")
        elif isinstance(parsed, bool):
            classes.add("bool")
        elif isinstance(parsed, int):
            classes.add("int")
        elif isinstance(parsed, float):
            classes.add("float")
        elif isinstance(parsed, complex):
            classes.add("complex")
        elif isinstance(parsed, str):
            classes.add("str")
        elif isinstance(parsed, bytes):
            classes.add("bytes")
        else:
            return None
    return next(iter(classes)) if len(classes) == 1 else None


def generic_parts(value: str) -> tuple[str, tuple[str, ...]]:
    value = normalize_type(value)
    bracket = value.find("[")
    if bracket < 0 or not value.endswith("]"):
        return value, ()
    base = value[:bracket].strip().split(".")[-1]
    return base, tuple(split_top_level(value[bracket + 1 : -1]))


def iterable_element_type(value: str) -> str | None:
    """Recover a concrete element type from ty's rendered iterable unions."""

    elements: set[str] = set()
    for branch in split_union(normalize_type(value)):
        base, arguments = generic_parts(branch)
        base = base.split(".")[-1]
        if base == "range":
            elements.add("int")
            continue
        if base == "str":
            elements.add("str")
            continue
        if base in {"bytes", "bytearray", "memoryview"}:
            elements.add("int")
            continue
        if base not in {
            "list",
            "tuple",
            "set",
            "frozenset",
            "deque",
            "map",
            "filter",
            "reversed",
            "Iterable",
            "Iterator",
            "Sequence",
            "MutableSequence",
            "ValuesView",
            "KeysView",
            "ItemsView",
            "dict_values",
            "dict_keys",
            "dict_items",
            "defaultdict",
        } or not arguments:
            return None
        if base in {"ValuesView", "dict_values"} and len(arguments) >= 2:
            candidates = arguments[1:2]
        elif base in {"ItemsView", "dict_items"} and len(arguments) >= 2:
            candidates = (f"tuple[{arguments[0]}, {arguments[1]}]",)
        else:
            candidates = arguments if base == "tuple" else arguments[:1]
        for candidate in candidates:
            for element in split_union(candidate):
                element = normalize_type(element)
                if element not in {"Divergent", "Never", "NoReturn", "..."}:
                    elements.add(element)
    if not elements:
        return None
    if len(elements) == 1:
        return next(iter(elements))
    if all(is_assignable(element, "__numeric__") for element in elements):
        return " | ".join(sorted(elements))
    return None


def strip_negative_flow_refinements(value: str) -> str | None:
    """Discard ty's negative flow facts while preserving the value type.

    Facts such as ``~AlwaysFalsy`` narrow which values reach a program point,
    but do not change the operations supported by the underlying type.  A
    positive intersection may add a protocol and is therefore not erased.
    """

    intersections = split_top_level(normalize_type(value), "&")
    if not intersections or any(
        not refinement.strip().startswith("~")
        for refinement in intersections[1:]
    ):
        return None
    return normalize_type(intersections[0])


def has_gradual_value_branch(value: str) -> bool:
    """Whether a top-level union/intersection branch is ``Any``/``Unknown``.

    Generic arguments may themselves be gradual without making the container
    value gradual.  Only the value-level head matters when deciding whether a
    complete expression may enter a capability-specific grammar family.
    """

    for branch in split_union(normalize_type(value)):
        intersections = split_top_level(normalize_type(branch), "&")
        if intersections and normalize_type(intersections[0]) in {
            "Any",
            "Unknown",
        }:
            return True
    return False


def concrete_heap_list_element_type(value: str) -> str | None:
    """Recover ``T`` from a concrete ``list[T]`` plus flow refinements.

    ``heapq.heappop`` mutates its argument, so accepting an arbitrary iterable
    here would be unsound.  ty may append negative flow refinements such as
    ``& ~AlwaysFalsy`` after a loop guard; those do not change the invariant
    list element type and are safe to discard.
    """

    concrete = strip_negative_flow_refinements(value)
    if concrete is None:
        return None
    base, arguments = generic_parts(concrete)
    if base.split(".")[-1] != "list" or len(arguments) != 1:
        return None
    element = normalize_type(arguments[0])
    if element in {"Any", "Unknown"}:
        return element
    return element if groundable_type(element) else None


def numeric_unary_kinds(value: str) -> tuple[bool, bool]:
    """Return whether ``+/-`` and ``~`` are justified by rendered branches."""

    saw_concrete = False
    integral = True
    for raw_branch in split_union(normalize_type(value)):
        branch = strip_negative_flow_refinements(raw_branch)
        if branch is None:
            return False, False
        if branch in {"Any", "Unknown", "Divergent", "Never", "NoReturn"}:
            continue
        base, _arguments = generic_parts(branch)
        base = base.split(".")[-1]
        if base not in {
            "bool",
            "int",
            "float",
            "complex",
            "integer",
            "signedinteger",
            "unsignedinteger",
            "floating",
        }:
            return False, False
        saw_concrete = True
        if base not in {"int", "integer", "signedinteger", "unsignedinteger"}:
            integral = False
    return saw_concrete, saw_concrete and integral


def numeric_unary_result(value: str) -> str:
    result: list[str] = []
    for raw_branch in split_union(normalize_type(value)):
        branch = strip_negative_flow_refinements(raw_branch)
        if branch is None:
            continue
        base, _arguments = generic_parts(branch)
        mapped = "int" if base.split(".")[-1] == "bool" else branch
        if mapped not in result:
            result.append(mapped)
    return " | ".join(result)


BUILTIN_NUMERIC_RANK: Mapping[str, int] = {
    "bool": 0,
    "int": 0,
    "float": 1,
    "complex": 2,
}
ARITHMETIC_SEQUENCE_BASES = frozenset(
    {"list", "tuple", "str", "bytes", "bytearray"}
)
INDEXED_SEQUENCE_BASES = frozenset(
    {
        "list",
        "tuple",
        "str",
        "bytes",
        "bytearray",
        "memoryview",
        "range",
        "deque",
        "Sequence",
        "MutableSequence",
    }
)
SLICEABLE_SEQUENCE_BASES = INDEXED_SEQUENCE_BASES - {"deque"}
INDEXED_MAPPING_BASES = frozenset(
    {"dict", "defaultdict", "Mapping", "MutableMapping"}
)


def tuple_arithmetic_preserves_type(arguments: tuple[str, ...]) -> bool:
    """Whether concatenation/repetition retains this rendered tuple type."""

    return (
        not arguments
        or arguments == ("()",)
        or (len(arguments) == 2 and arguments[1] == "...")
    )


def arithmetic_binary_result(
    left: str, operator: str, right: str
) -> str | None:
    """Conservatively type one common Python arithmetic operation.

    These rows cover Python's concrete numeric tower and the built-in
    sequence operations whose result is fixed by the displayed operand types.
    User-defined overloads are deliberately left to calls and members learned
    from ``ty`` rather than inferred from a class name.
    """

    concrete_left = strip_negative_flow_refinements(left)
    concrete_right = strip_negative_flow_refinements(right)
    if concrete_left is None or concrete_right is None:
        return None
    left = concrete_left
    right = concrete_right
    if len(split_union(left)) != 1 or len(split_union(right)) != 1:
        return None
    left_base, left_arguments = generic_parts(left)
    right_base, right_arguments = generic_parts(right)
    left_base = left_base.split(".")[-1]
    right_base = right_base.split(".")[-1]

    left_rank = BUILTIN_NUMERIC_RANK.get(left_base)
    right_rank = BUILTIN_NUMERIC_RANK.get(right_base)
    if left_rank is not None and right_rank is not None:
        if operator in {"//", "%"} and max(left_rank, right_rank) == 2:
            return None
        if operator == "/":
            return "complex" if max(left_rank, right_rank) == 2 else "float"
        rank = max(left_rank, right_rank)
        return ("int", "float", "complex")[rank]

    if operator == "+" and left == right and left_base in {
        "str",
        "bytes",
        "bytearray",
        "list",
        "tuple",
    }:
        if left_base == "tuple" and not tuple_arithmetic_preserves_type(
            left_arguments
        ):
            return None
        return left

    if (
        operator == "+"
        and left_base == right_base == "list"
        and len(left_arguments) == len(right_arguments) == 1
    ):
        left_element = normalize_type(left_arguments[0])
        right_element = normalize_type(right_arguments[0])
        left_element_base, _ = generic_parts(left_element)
        right_element_base, _ = generic_parts(right_element)
        left_element_rank = BUILTIN_NUMERIC_RANK.get(
            left_element_base.split(".")[-1]
        )
        right_element_rank = BUILTIN_NUMERIC_RANK.get(
            right_element_base.split(".")[-1]
        )
        if left_element_rank is not None and right_element_rank is not None:
            element = ("int", "float", "complex")[
                max(left_element_rank, right_element_rank)
            ]
            return f"list[{element}]"

    if operator == "*":
        if (
            left_base in ARITHMETIC_SEQUENCE_BASES
            and is_assignable(right, "SupportsIndex")
        ):
            if left_base == "tuple" and not tuple_arithmetic_preserves_type(
                left_arguments
            ):
                return None
            return left
        if (
            right_base in ARITHMETIC_SEQUENCE_BASES
            and is_assignable(left, "SupportsIndex")
        ):
            if right_base == "tuple" and not tuple_arithmetic_preserves_type(
                right_arguments
            ):
                return None
            return right

    return None


def indexed_access_types(value: str) -> tuple[str, str] | None:
    """Return ``(index_type, result_type)`` for a displayed container type."""

    value = normalize_type(value)
    if len(split_union(value)) != 1:
        return None
    container = strip_negative_flow_refinements(value)
    if container is None:
        return None
    base, arguments = generic_parts(container)
    base = base.split(".")[-1]
    if base in INDEXED_MAPPING_BASES and len(arguments) >= 2:
        return normalize_type(arguments[0]), normalize_type(arguments[1])
    if base == "str":
        return "SupportsIndex", "str"
    if base in {"bytes", "bytearray", "memoryview", "range"}:
        return "SupportsIndex", "int"
    if base == "tuple":
        if arguments == ("()",):
            return None
        elements = [
            normalize_type(item)
            for item in arguments
            if item != "..."
        ]
        if not elements:
            return None
        result = " | ".join(dict.fromkeys(elements))
        return "SupportsIndex", result
    if base in INDEXED_SEQUENCE_BASES and arguments:
        return "SupportsIndex", normalize_type(arguments[0])
    return None


def sliced_access_type(value: str) -> str | None:
    """Return the type of an ordinary slice when it is container-preserving."""

    value = normalize_type(value)
    if len(split_union(value)) != 1:
        return None
    container = strip_negative_flow_refinements(value)
    if container is None:
        return None
    base, arguments = generic_parts(container)
    base = base.split(".")[-1]
    if base in {"str", "bytes", "bytearray", "memoryview", "range"}:
        return container
    if base == "tuple":
        if arguments == ("()",) or (
            len(arguments) == 2 and arguments[1] == "..."
        ):
            return container
        if arguments:
            element = " | ".join(dict.fromkeys(map(normalize_type, arguments)))
            return f"tuple[{element}, ...]"
        return None
    if base in SLICEABLE_SEQUENCE_BASES and arguments:
        return container
    return None


def is_callable_type(value: str, kind: int | None = None) -> bool:
    value = value.strip()
    return (
        kind in CALLABLE_KINDS
        or value.startswith("def ")
        or value.startswith("bound method ")
        or value.startswith("Overload[")
        or value.startswith("<class '")
        or value.startswith("Callable[")
        or "_lru_cache_wrapper[" in value
    )


TYPE_VARIABLE = re.compile(
    r"^(?:Self|AnyStr|_?(?:[KVRSP]|[KVRSP]?T[0-9]*|[A-Z][A-Za-z0-9]*T)"
    r"(?:_(?:co|contra))?)(?:@.*)?$|^_?[A-Za-z][A-Za-z0-9_]*@.+$"
)


def has_unresolved_type_variable(value: str) -> bool:
    """Return whether a rendered type still contains a ty type variable."""

    return any(
        TYPE_VARIABLE.fullmatch(identifier) is not None
        for identifier in re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*(?:@[A-Za-z0-9_.:-]+)?",
            normalize_type(value),
        )
    )


def groundable_type(value: str) -> bool:
    """Return whether ``value`` is concrete enough to specialize a call."""

    value = normalize_type(value)
    return (
        value not in {"Any", "Unknown", "Divergent", "Never", "NoReturn"}
        and not has_unresolved_type_variable(value)
    )


SEQUENCE_BASES = frozenset(
    {
        "list",
        "tuple",
        "str",
        "bytes",
        "bytearray",
        "memoryview",
        "range",
        "Sequence",
        "MutableSequence",
    }
)
MUTABLE_SEQUENCE_BASES = frozenset(
    {"list", "bytearray", "MutableSequence"}
)
MAPPING_BASES = frozenset(
    {"dict", "defaultdict", "Mapping", "MutableMapping"}
)
MUTABLE_MAPPING_BASES = frozenset(
    {"dict", "defaultdict", "MutableMapping"}
)
COLLECTION_BASES = frozenset(
    {
        *SEQUENCE_BASES,
        *MAPPING_BASES,
        "set",
        "frozenset",
        "deque",
        "ValuesView",
        "KeysView",
        "ItemsView",
        "dict_values",
        "dict_keys",
        "dict_items",
        "Collection",
    }
)
ITERATOR_BASES = frozenset(
    {
        "Iterator",
        "Generator",
        "map",
        "filter",
        "zip",
        "enumerate",
        "reversed",
    }
)
ITERABLE_BASES = frozenset(
    {
        *COLLECTION_BASES,
        *ITERATOR_BASES,
        "Iterable",
        "Reversible",
    }
)
REVERSIBLE_BASES = frozenset(
    {
        "list",
        "tuple",
        "dict",
        "defaultdict",
        "str",
        "bytes",
        "bytearray",
        "memoryview",
        "range",
        "deque",
        "Sequence",
        "MutableSequence",
        "Reversible",
        "_SupportsReversed",
    }
)
LEN_AND_GETITEM_BASES = frozenset(
    {
        "list",
        "tuple",
        "str",
        "bytes",
        "bytearray",
        "memoryview",
        "range",
        "deque",
        "Sequence",
        "MutableSequence",
        "SupportsLenAndGetItem",
    }
)
PROTOCOL_CAPABILITY_BASES: Mapping[str, frozenset[str]] = {
    "Iterable": ITERABLE_BASES,
    "Iterator": ITERATOR_BASES,
    "Collection": COLLECTION_BASES,
    "Container": COLLECTION_BASES | frozenset({"Container"}),
    "Sequence": SEQUENCE_BASES,
    "MutableSequence": MUTABLE_SEQUENCE_BASES,
    "Mapping": MAPPING_BASES,
    "MutableMapping": MUTABLE_MAPPING_BASES,
    "Reversible": REVERSIBLE_BASES,
    "_SupportsReversed": REVERSIBLE_BASES,
    "SupportsLenAndGetItem": LEN_AND_GETITEM_BASES,
}
SIZED_BASES = COLLECTION_BASES | frozenset({"Sized"})


@functools.lru_cache(maxsize=250_000)
def is_assignable(actual: str, expected: str) -> bool:
    """Conservative relation over ty's rendered display types.

    Member and call signatures still come from ty.  This relation only lowers
    their displayed argument slots to grammar edges; sampled candidates are
    independently checked by ty.
    """

    actual = normalize_type(actual)
    expected = normalize_type(expected)
    if actual == expected:
        return True
    if actual in {"Any", "Unknown"} or expected in {"Any", "Unknown", "object"}:
        return True
    if actual in {"Never", "NoReturn", "Divergent"}:
        return True
    actual_union = split_union(actual)
    expected_union = split_union(expected)
    if len(actual_union) > 1:
        return all(is_assignable(part, expected) for part in actual_union)
    if len(expected_union) > 1:
        return any(is_assignable(actual, part) for part in expected_union)

    if " & " in actual:
        return is_assignable(actual.split(" & ", 1)[0], expected)
    if " & " in expected:
        return all(is_assignable(actual, part) for part in expected.split(" & "))

    actual_base, actual_args = generic_parts(actual)
    expected_base, expected_args = generic_parts(expected)
    actual_base = actual_base.split(".")[-1]
    expected_base = expected_base.split(".")[-1]
    if expected_base == "__dict_source__":
        return actual_base in {
            "dict",
            "defaultdict",
            "Mapping",
            "MutableMapping",
        }
    if expected_base == "__numeric_iterable__":
        if actual_base in {"bytes", "bytearray", "range"}:
            return True
        if actual_base not in {"list", "tuple", "set", "frozenset", "Iterable", "Iterator"}:
            return False
        if not actual_args:
            return True
        return normalize_type(actual_args[0]) in {
            "Any",
            "Unknown",
            "bool",
            "int",
            "float",
            "complex",
        }
    if expected_base == "__numeric__":
        return actual_base in {"Any", "Unknown", "bool", "int", "float", "complex"}
    if expected_base in {
        "_SupportsArray",
        "SupportsArray",
        "_NestedSequence",
        "ArrayLike",
        "NDArray",
    }:
        return actual_base in {
            "Any",
            "Unknown",
            "list",
            "tuple",
            "Sequence",
            "ndarray",
        }
    if "->" in expected and is_callable_type(actual):
        return True
    if expected_base == "SupportsRichComparisonT":
        return actual_base in {
            "bool",
            "int",
            "float",
            "str",
            "bytes",
            "tuple",
        }
    if (
        TYPE_VARIABLE.fullmatch(expected)
        and expected_base.split("@", 1)[0] == "AnyStr"
    ):
        # ``AnyStr`` is constrained to the two string families; treating it
        # like an unconstrained type variable admits calls such as
        # ``re.escape(set())`` and ``re.template("".isnumeric)``.  Literal
        # strings retain their string constraint even when ty renders them as
        # ``Literal[...]`` rather than ``str``.
        return (
            actual_base in {"str", "bytes", "LiteralString"}
            or (
                actual_base == "Literal"
                and bool(actual_args)
                and all(
                    argument.startswith(('"', "'", "b\"", "b'"))
                    for argument in actual_args
                )
            )
        )
    if TYPE_VARIABLE.fullmatch(expected):
        return True
    if actual_base == expected_base:
        if not expected_args or not actual_args:
            return True
        if len(actual_args) != len(expected_args):
            return False
        return all(
            left == right
            or TYPE_VARIABLE.fullmatch(right) is not None
            or (
                generic_parts(right)[0] == "SupportsRichComparisonT"
                and is_assignable(left, right)
            )
            for left, right in zip(actual_args, expected_args)
        )

    numeric_supertypes = {
        "bool": {"int", "float", "complex", "SupportsIndex", "SupportsFloat", "SupportsComplex"},
        "int": {"float", "complex", "SupportsIndex", "SupportsFloat", "SupportsComplex"},
        "float": {"complex", "SupportsFloat", "SupportsComplex"},
        "complex": {"SupportsComplex"},
    }
    if expected_base in numeric_supertypes.get(actual_base, set()):
        return True
    if expected_base == "SupportsIndex" and actual_base in {
        "integer",
        "signedinteger",
        "unsignedinteger",
    }:
        return True
    if expected_base == "SupportsInt" and actual_base in {
        "bool",
        "int",
        "float",
    }:
        return True

    capability_bases = PROTOCOL_CAPABILITY_BASES.get(expected_base)
    if capability_bases is not None:
        if actual_base not in capability_bases:
            return False
        if not expected_args or expected_base == "Container":
            return True
        actual_elements: tuple[str, ...]
        if actual_base == "range":
            actual_elements = ("int",)
        elif actual_base == "str":
            actual_elements = ("str",)
        elif actual_base in {"bytes", "bytearray", "memoryview"}:
            actual_elements = ("int",)
        elif actual_base in {"ValuesView", "dict_values"} and len(actual_args) >= 2:
            actual_elements = (actual_args[1],)
        elif actual_base in {"ItemsView", "dict_items"} and len(actual_args) >= 2:
            actual_elements = (f"tuple[{actual_args[0]}, {actual_args[1]}]",)
        elif actual_base == "tuple" and actual_args == ("()",):
            actual_elements = ()
        elif actual_base == "tuple":
            actual_elements = tuple(
                argument for argument in actual_args if argument != "..."
            )
        elif actual_args:
            actual_elements = (actual_args[0],)
        else:
            return True

        def invariant_argument_matches(left: str, right: str) -> bool:
            left = normalize_type(left)
            right = normalize_type(right)
            return (
                left == right
                or left in {"Any", "Unknown"}
                or right in {"Any", "Unknown"}
                or TYPE_VARIABLE.fullmatch(left) is not None
                or TYPE_VARIABLE.fullmatch(right) is not None
            )

        invariant_elements = expected_base in {
            "MutableSequence",
            "MutableMapping",
        }
        element_matches = all(
            invariant_argument_matches(actual_element, expected_args[0])
            if invariant_elements
            else is_assignable(actual_element, expected_args[0])
            for actual_element in actual_elements
        )
        if not element_matches:
            return False
        if (
            expected_base in {"Mapping", "MutableMapping"}
            and actual_base in MAPPING_BASES
            and len(expected_args) >= 2
            and len(actual_args) >= 2
        ):
            if expected_base == "MutableMapping":
                return invariant_argument_matches(
                    actual_args[1], expected_args[1]
                )
            return is_assignable(actual_args[1], expected_args[1])
        return True
    if expected_base == "Sized":
        return actual_base in SIZED_BASES
    if expected_base in {"Buffer", "ReadableBuffer", "SupportsBytes"} and actual_base in {
        "bytes",
        "bytearray",
        "memoryview",
    }:
        return True
    if expected_base == "Hashable" and actual_base not in {
        "list",
        "dict",
        "defaultdict",
        "set",
    }:
        return True
    if expected_base in {"Callable", "Protocol"} and is_callable_type(actual):
        return True
    if expected_base == "type" and (
        actual.startswith("<class '") or actual_base == "type"
    ):
        return True
    return False


def set_tuple_refinement_candidate(actual: str, expected: str) -> bool:
    """Identify the flow-sensitive tuple shape that ty may refine at a call.

    A displayed ``set[tuple[T, ...]]`` is not globally assignable to
    ``set[tuple[T, T]]``: ``set`` is invariant and the variadic tuple may have
    another length.  At a particular program point ty can nevertheless know
    that a local set contains only fixed-length tuples.  This predicate only
    selects that narrow case for an explicit contextual query; it never adds
    a global assignability edge.
    """

    actual_base, actual_arguments = generic_parts(actual)
    expected_base, expected_arguments = generic_parts(expected)
    if (
        actual_base.split(".")[-1] != "set"
        or expected_base.split(".")[-1] != "set"
        or len(actual_arguments) != 1
        or len(expected_arguments) != 1
    ):
        return False
    actual_tuple, actual_items = generic_parts(actual_arguments[0])
    expected_tuple, expected_items = generic_parts(expected_arguments[0])
    if (
        actual_tuple.split(".")[-1] != "tuple"
        or expected_tuple.split(".")[-1] != "tuple"
        or len(actual_items) != 2
        or actual_items[1] != "..."
        or not expected_items
        or expected_items[-1] == "..."
    ):
        return False
    actual_item = normalize_type(actual_items[0])
    return all(normalize_type(item) == actual_item for item in expected_items)


def dotted_identifier_tokens(expression: str) -> tuple[str, ...] | None:
    """Tokenize a plain dotted identifier without admitting other syntax."""

    parts = expression.split(".")
    if not parts or any(
        not part.isidentifier() or keyword.iskeyword(part) for part in parts
    ):
        return None
    tokens: list[str] = []
    for index, part in enumerate(parts):
        if index:
            tokens.append(".")
        tokens.append(part)
    return tuple(tokens)


def canonical_expression_tokens(expression: str) -> tuple[str, ...] | None:
    """Lexicalize one exact expression using the evaluator's literal policy."""

    try:
        ast.parse(expression, mode="eval")
        stream = tokenize.generate_tokens(io.StringIO(expression).readline)
        tokens: list[str] = []
        for item in stream:
            if item.type in IGNORED_TOKEN_TYPES:
                continue
            value = item.string
            if item.type == tokenize.NUMBER:
                value = canonical_number(value)
            elif item.type == tokenize.STRING:
                value = canonical_string(value)
            elif item.type not in {tokenize.NAME, tokenize.OP}:
                return None
            tokens.append(value)
        return tuple(tokens) or None
    except (SyntaxError, ValueError, tokenize.TokenError):
        return None


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str
    kind: str
    required: bool


@dataclass(frozen=True)
class Signature:
    parameters: tuple[Parameter, ...]
    return_type: str


@dataclass(frozen=True)
class ArgumentLayout:
    positional: tuple[str, ...]
    keywords: tuple[tuple[str, str], ...]


SELF_TYPE_VARIABLE = re.compile(r"^Self(?:@[A-Za-z0-9_.:-]+)?$")
SELF_TYPE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])Self(?:@[A-Za-z0-9_.:-]+)?(?![A-Za-z0-9_])"
)
CLASS_DISPLAY = re.compile(r"^<class '([^']+)'>$")


def class_instance_type(class_display: str) -> str | None:
    """Recover the instance type named by ty's ``<class '...'>`` display."""

    match = CLASS_DISPLAY.fullmatch(normalize_type(class_display))
    if match is None:
        return None
    return normalize_type(match.group(1))


def bind_unbound_self_signature(
    signature: Signature, receiver_instance_type: str
) -> Signature:
    """Bind an erased leading ``self`` slot to its descriptor's owner.

    Signature help for an unbound class member retains ``self``.  When that
    parameter is unannotated, ``parse_signature`` deliberately renders it as
    ``object``; typeshed's implicit descriptor constraint is consequently
    lost.  ``Self`` may likewise remain as an unresolved return variable.
    Replacing both with the class receiver's instance type preserves that
    constraint without changing already-bound methods or constructors.
    """

    if not signature.parameters:
        return signature
    first = signature.parameters[0]
    first_type = normalize_type(first.type)
    if first.name != "self" or not (
        first_type == "object"
        or SELF_TYPE_VARIABLE.fullmatch(first_type) is not None
    ):
        return signature
    receiver_instance_type = normalize_type(receiver_instance_type)
    parameters = (
        replace(first, type=receiver_instance_type),
        *signature.parameters[1:],
    )
    return_type = normalize_type(
        SELF_TYPE_REFERENCE.sub(
            lambda _match: receiver_instance_type,
            signature.return_type,
        )
    )
    return Signature(parameters, return_type)


def matching_paren(value: str, start: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(value)):
        character = value[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def parse_signature(label: str) -> Signature | None:
    start = label.find("(")
    if start < 0:
        return None
    end = matching_paren(label, start)
    if end < 0:
        return None
    body = label[start + 1 : end]
    suffix = label[end + 1 :].strip()
    return_type = suffix[2:].strip() if suffix.startswith("->") else "Unknown"
    chunks = split_top_level(body) if body.strip() else []
    slash = next((index for index, chunk in enumerate(chunks) if chunk == "/"), -1)
    star = next((index for index, chunk in enumerate(chunks) if chunk == "*"), -1)
    parameters: list[Parameter] = []
    keyword_only = False
    for index, chunk in enumerate(chunks):
        if not chunk:
            continue
        if chunk == "/":
            continue
        if chunk == "*":
            keyword_only = True
            continue
        if chunk.startswith("**"):
            raw = chunk[2:]
            kind = "varkw"
            keyword_only = True
        elif chunk.startswith("*"):
            raw = chunk[1:]
            kind = "vararg"
            keyword_only = True
        else:
            raw = chunk
            if slash >= 0 and index < slash:
                kind = "posonly"
            elif keyword_only or (star >= 0 and index > star):
                kind = "kwonly"
            else:
                kind = "poskw"
        equals = find_top_level(raw, "=")
        required = equals < 0 and kind not in {"vararg", "varkw"}
        declaration = raw if equals < 0 else raw[:equals].strip()
        colon = find_top_level(declaration, ":")
        if colon < 0:
            name = declaration.strip() or "arg"
            parameter_type = "object"
        else:
            name = declaration[:colon].strip()
            parameter_type = declaration[colon + 1 :].strip() or "object"
        if not name.isidentifier():
            name = "arg"
        parameters.append(
            Parameter(
                name=name,
                type=normalize_type(parameter_type),
                kind=kind,
                required=required,
            )
        )
    return Signature(tuple(parameters), normalize_type(return_type))


def signatures_from_detail(detail: str) -> list[Signature]:
    candidates: list[str] = []
    detail = detail.strip()
    if detail.startswith("Overload[") and detail.endswith("]"):
        payload = detail[len("Overload[") : -1]
        candidates.extend(split_top_level(payload))
    else:
        candidates.append(detail)
    result: list[Signature] = []
    for candidate in candidates:
        parsed = parse_signature(candidate)
        if parsed is not None:
            result.append(parsed)
    return result


def argument_layouts(
    signature: Signature,
    *,
    max_arity: int,
    max_layouts: int,
) -> list[ArgumentLayout]:
    fixed = [
        parameter
        for parameter in signature.parameters
        if parameter.kind in {"posonly", "poskw"}
    ]
    vararg = next(
        (parameter for parameter in signature.parameters if parameter.kind == "vararg"),
        None,
    )
    keyword_only = [
        parameter for parameter in signature.parameters if parameter.kind == "kwonly"
    ]
    skeletons: list[
        tuple[tuple[str, ...], list[Parameter], list[Parameter], int]
    ] = []
    for positional_count in range(len(fixed) + 1):
        positional = fixed[:positional_count]
        remaining = fixed[positional_count:]
        if any(parameter.kind == "posonly" and parameter.required for parameter in remaining):
            continue
        remaining_poskw = [parameter for parameter in remaining if parameter.kind == "poskw"]
        required_keywords = [parameter for parameter in remaining_poskw if parameter.required]
        optional_keywords = [parameter for parameter in remaining_poskw if not parameter.required]
        required_keywords.extend(parameter for parameter in keyword_only if parameter.required)
        optional_keywords.extend(parameter for parameter in keyword_only if not parameter.required)
        for repeat in range((max_arity - len(positional)) + 1 if vararg else 1):
            positional_types = tuple(parameter.type for parameter in positional)
            if vararg is not None:
                positional_types += (vararg.type,) * repeat
            budget = max_arity - len(positional_types) - len(required_keywords)
            if budget < 0:
                continue
            optional_limit = min(len(optional_keywords), budget)
            skeletons.append(
                (
                    positional_types,
                    required_keywords,
                    optional_keywords,
                    optional_limit,
                )
            )

    parameter_order = {
        id(parameter): index
        for index, parameter in enumerate(signature.parameters)
    }

    def make_layout(
        positional_types: tuple[str, ...],
        required_keywords: Sequence[Parameter],
        chosen: Sequence[Parameter],
    ) -> ArgumentLayout:
        selected = [*required_keywords, *chosen]
        selected.sort(key=lambda parameter: parameter_order[id(parameter)])
        return ArgumentLayout(
            positional=positional_types,
            keywords=tuple(
                (parameter.name, parameter.type) for parameter in selected
            ),
        )

    # Reserve the minimal layout for every positional/vararg skeleton before
    # optional keyword combinations can consume the cap.  This keeps common
    # positional calls reachable even for signatures with many optional
    # parameters (notably numpy.array).
    layouts = {
        make_layout(positional_types, required_keywords, ())
        for positional_types, required_keywords, _optional, _limit in skeletons
    }
    for optional_count in range(1, max_arity + 1):
        for (
            positional_types,
            required_keywords,
            optional_keywords,
            optional_limit,
        ) in skeletons:
            if optional_count > optional_limit:
                continue
            for chosen in itertools.combinations(optional_keywords, optional_count):
                layouts.add(
                    make_layout(positional_types, required_keywords, chosen)
                )
                if len(layouts) >= max_layouts:
                    break
            if len(layouts) >= max_layouts:
                break
        if len(layouts) >= max_layouts:
            break
    ordered_layouts: set[ArgumentLayout] = set()
    for layout in layouts:
        # A signature with twelve keyword-only parameters has 12! orders.
        # We only return ``max_layouts`` rows, and the (max_layouts + 1)-th
        # lexicographic permutation of any one layout already has
        # ``max_layouts`` distinct rows from that same layout before it, so it
        # cannot enter the global prefix.  Bound generation before sorting to
        # avoid factorial time and memory without changing the result.
        keywords = tuple(sorted(layout.keywords))
        for keyword_order in itertools.islice(
            itertools.permutations(keywords), max_layouts
        ):
            ordered_layouts.add(
                ArgumentLayout(layout.positional, keyword_order)
            )
    return sorted(
        ordered_layouts,
        key=lambda layout: (
            len(layout.positional) + len(layout.keywords),
            layout.positional,
            layout.keywords,
        ),
    )[:max_layouts]


@dataclass(frozen=True)
class Terminal:
    value: str


@dataclass(frozen=True)
class Nonterminal:
    value: str


Symbol = Terminal | Nonterminal


@dataclass(frozen=True)
class Production:
    lhs: str
    rhs: tuple[Symbol, ...]


@dataclass
class Grammar:
    start: str
    productions: set[Production] = field(default_factory=set)
    type_labels: dict[str, str] = field(default_factory=dict)

    def add(self, lhs: str, *rhs: Symbol) -> None:
        self.productions.add(Production(lhs, tuple(rhs)))

    @property
    def nonterminals(self) -> frozenset[str]:
        result = {self.start, *(production.lhs for production in self.productions)}
        result.update(
            symbol.value
            for production in self.productions
            for symbol in production.rhs
            if isinstance(symbol, Nonterminal)
        )
        return frozenset(result)

    @property
    def terminals(self) -> frozenset[str]:
        return frozenset(
            symbol.value
            for production in self.productions
            for symbol in production.rhs
            if isinstance(symbol, Terminal)
        )

    @property
    def symbol_count(self) -> int:
        """Return |G|, counting the LHS and every RHS symbol occurrence."""
        return sum(1 + len(production.rhs) for production in self.productions)


def type_nonterminal(type_display: str) -> str:
    return f"E:{type_display}"


def argument_nonterminal(type_display: str) -> str:
    return f"A:{type_display}"


def postfix_nonterminal(expression_nonterminal: str) -> str:
    """Return the primary/postfix layer corresponding to an ``E:`` symbol."""

    if not expression_nonterminal.startswith("E:"):
        raise ValueError(
            f"postfix layer requires an expression nonterminal: "
            f"{expression_nonterminal!r}"
        )
    return f"P:{expression_nonterminal[2:]}"


def expression_layer_nonterminal(
    expression_nonterminal: str, layer: str
) -> str:
    """Return one precedence layer corresponding to an ``E:`` symbol."""

    if not expression_nonterminal.startswith("E:"):
        raise ValueError(
            f"expression layer requires an expression nonterminal: "
            f"{expression_nonterminal!r}"
        )
    return f"{layer}:{expression_nonterminal[2:]}"


def sum_nonterminal(expression_nonterminal: str) -> str:
    return expression_layer_nonterminal(expression_nonterminal, "S")


def term_nonterminal(expression_nonterminal: str) -> str:
    return expression_layer_nonterminal(expression_nonterminal, "M")


def factor_nonterminal(expression_nonterminal: str) -> str:
    return expression_layer_nonterminal(expression_nonterminal, "F")


def power_nonterminal(expression_nonterminal: str) -> str:
    return expression_layer_nonterminal(expression_nonterminal, "W")


@dataclass(frozen=True)
class BuilderOptions:
    max_call_arity: int = 3
    max_dynamic_composition_depth: int = 2
    max_tokens: int = 20
    max_layouts_per_signature: int = 64
    member_depth: int = 2
    max_receiver_types: int = 32
    max_module_members: int = 128
    max_output_producers: int = 2048


@dataclass(frozen=True)
class LibraryArtifact:
    """Alias-independent grammar fragment generated for one Python module."""

    path: Path
    module: str
    module_type: str
    grammar: Grammar
    metadata: Mapping[str, object]
    expected_types: frozenset[str]
    exports: frozenset[str]
    export_types: Mapping[str, str]
    local_actual_types: frozenset[str]
    local_expected_types: frozenset[str]
    local_assignability_complete: bool
    nonlocal_productions_by_length: Mapping[int, frozenset[Production]]
    local_links_by_expected: Mapping[str, frozenset[Production]]


@dataclass(frozen=True)
class LibraryLookup:
    module: str
    artifact: LibraryArtifact | None
    reason: str | None = None


@dataclass(frozen=True)
class FromImportBinding:
    module: str
    member: str
    bound_name: str


def decode_library_nonterminal(atom: str) -> str:
    """Decode the reversible ``E:``/``A:`` atoms used by library CFGs."""

    if not atom.startswith(("E:", "A:")):
        return atom
    prefix, payload = atom[:2], atom[2:]
    malformed = re.search(r"%(?![0-9A-Fa-f]{2})", payload)
    if malformed is not None:
        raise EvaluationError(f"malformed percent escape in CFG atom {atom!r}")
    try:
        decoded = unquote(payload, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise EvaluationError(
            f"invalid UTF-8 percent encoding in CFG atom {atom!r}"
        ) from error
    return f"{prefix}{decoded}"


def encode_library_nonterminal(atom: str) -> str:
    if not atom.startswith(("E:", "A:")):
        return atom
    return f"{atom[:2]}{quote(atom[2:], safe='')}"


def parse_library_metadata(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def is_canonical_library_terminal(atom: str) -> bool:
    """Accept exactly one canonical Python token as a cached terminal."""

    items: list[tokenize.TokenInfo] = []
    try:
        for item in tokenize.generate_tokens(io.StringIO(atom).readline):
            if item.type not in IGNORED_TOKEN_TYPES:
                items.append(item)
    except (IndentationError, tokenize.TokenError):
        # A single opening delimiter is a valid grammar terminal even though
        # it is not a balanced Python input by itself.  Keep tokens emitted
        # before the tokenizer reports the incomplete input.
        pass
    try:
        if len(items) != 1:
            return False
        item = items[0]
        value = item.string
        if item.type == tokenize.NUMBER:
            value = canonical_number(value)
        elif item.type == tokenize.STRING:
            value = canonical_string(value)
        elif item.type not in {tokenize.NAME, tokenize.OP}:
            return False
        return value == atom
    except ValueError:
        return False


def parse_library_cfg_text(text: str, path: Path) -> LibraryArtifact:
    """Parse the line-oriented, alias-independent Python library CFG format."""

    metadata: dict[str, object] = {}
    symbol_table: dict[str, str] = {}
    rows: list[tuple[int, str, tuple[str, ...]]] = []
    saw_header = False
    schema: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, separator, value = line[1:].strip().partition(":")
            if not separator or not key.strip():
                raise EvaluationError(
                    f"{path}:{line_number}: malformed metadata comment"
                )
            key = key.strip()
            if not saw_header:
                parsed_schema = str(parse_library_metadata(value.strip()))
                if (
                    key != "api2cfg-python-library-cfg"
                    or parsed_schema not in SUPPORTED_LIBRARY_CFG_SCHEMAS
                ):
                    raise EvaluationError(
                        f"{path}:{line_number}: expected supported library "
                        f"CFG schema header"
                    )
                schema = parsed_schema
                saw_header = True
                metadata[key] = parse_library_metadata(value.strip())
                continue
            if key.startswith("symbol-"):
                if schema != "2":
                    raise EvaluationError(
                        f"{path}:{line_number}: symbol tables require schema 2"
                    )
                alias = key.removeprefix("symbol-")
                if not re.fullmatch(r"N[0-9]+", alias) or alias in symbol_table:
                    raise EvaluationError(
                        f"{path}:{line_number}: invalid or duplicate symbol alias {alias!r}"
                    )
                decoded = decode_library_nonterminal(value.strip())
                if not decoded.startswith(("E:", "A:")):
                    raise EvaluationError(
                        f"{path}:{line_number}: symbol alias must map to E:/A:"
                    )
                symbol_table[alias] = decoded
                continue
            metadata[key] = parse_library_metadata(value.strip())
            continue
        if not saw_header:
            raise EvaluationError(f"{path}:{line_number}: missing library CFG header")
        lhs_text, separator, rhs_text = line.partition(" -> ")
        if not separator or not lhs_text or not rhs_text:
            raise EvaluationError(
                f"{path}:{line_number}: expected `LHS -> RHS ...`"
            )
        if schema == "2":
            lhs = symbol_table.get(lhs_text, "")
            if not lhs:
                raise EvaluationError(
                    f"{path}:{line_number}: undefined nonterminal alias {lhs_text!r}"
                )
            decoded_rhs: list[str] = []
            for atom in rhs_text.split():
                if re.fullmatch(r"N[0-9]+", atom) and atom not in symbol_table:
                    raise EvaluationError(
                        f"{path}:{line_number}: undefined nonterminal alias {atom!r}"
                    )
                decoded_rhs.append(symbol_table.get(atom, atom))
            rhs = tuple(decoded_rhs)
        else:
            lhs = decode_library_nonterminal(lhs_text)
            rhs = tuple(
                decode_library_nonterminal(atom) for atom in rhs_text.split()
            )
        if any(atom in {"[", "]"} for atom in rhs):
            raise EvaluationError(
                f"{path}:{line_number}: cached library fragments must not "
                "contain bracket terminals; subscripts and slices are built "
                "from the live typing context"
            )
        if lhs == "START":
            raise EvaluationError(
                f"{path}:{line_number}: library fragments must not define START"
            )
        rows.append((line_number, lhs, rhs))
    if not saw_header:
        raise EvaluationError(f"{path}: empty library CFG or missing schema header")

    module_value = metadata.get("module")
    module_type_value = metadata.get("module-type")
    if not isinstance(module_value, str) or not module_value:
        raise EvaluationError(f"{path}: missing string `module` metadata")
    if not isinstance(module_type_value, str) or not module_type_value:
        raise EvaluationError(f"{path}: missing string `module-type` metadata")
    expected_module_type = f"<module '{module_value}'>"
    if normalize_type(module_type_value) != expected_module_type:
        raise EvaluationError(
            f"{path}: module-type {module_type_value!r} does not match "
            f"module {module_value!r}"
        )

    lhs_names = {lhs for _line, lhs, _rhs in rows}
    grammar = Grammar(start="START")
    canonical_names = {name: name for name in lhs_names}
    terminal_symbols: dict[str, Terminal] = {}
    nonterminal_symbols: dict[str, Nonterminal] = {}
    for _line_number, lhs, rhs_atoms in rows:
        symbols: list[Symbol] = []
        for atom in rhs_atoms:
            is_nonterminal = atom in lhs_names or atom.startswith(("E:", "A:"))
            if is_nonterminal:
                name = canonical_names.setdefault(atom, atom)
                nonterminal_symbol = nonterminal_symbols.get(name)
                if nonterminal_symbol is None:
                    nonterminal_symbol = Nonterminal(name)
                    nonterminal_symbols[name] = nonterminal_symbol
                symbol: Symbol = nonterminal_symbol
            else:
                name = canonical_names.setdefault(atom, atom)
                if not is_canonical_library_terminal(name):
                    raise EvaluationError(
                        f"{path}:{_line_number}: cached terminal {name!r} "
                        "must be one canonical Python token"
                    )
                terminal_symbol = terminal_symbols.get(name)
                if terminal_symbol is None:
                    terminal_symbol = Terminal(name)
                    terminal_symbols[name] = terminal_symbol
                symbol = terminal_symbol
            symbols.append(symbol)
            if atom.startswith("E:"):
                grammar.type_labels[name] = name[2:]
        lhs = canonical_names[lhs]
        if lhs.startswith("E:"):
            grammar.type_labels[lhs] = lhs[2:]
        grammar.add(lhs, *symbols)

    module_nonterminal = type_nonterminal(expected_module_type)
    if module_nonterminal not in grammar.nonterminals:
        raise EvaluationError(
            f"{path}: fragment does not reference its module anchor "
            f"{encode_library_nonterminal(module_nonterminal)}"
        )
    if "START" in {
        symbol.value
        for production in grammar.productions
        for symbol in production.rhs
        if isinstance(symbol, Nonterminal)
    }:
        raise EvaluationError(f"{path}: library fragments must not reference START")

    actual_counts = {
        "productions": len(grammar.productions),
        "nonterminals": len(grammar.nonterminals - {grammar.start}),
        "terminals": len(grammar.terminals),
        "grammar-symbols": grammar.symbol_count,
    }
    for key, actual in actual_counts.items():
        declared = metadata.get(key)
        if declared is not None and declared != actual:
            raise EvaluationError(
                f"{path}: metadata {key}={declared!r}, parsed {actual}"
            )
    expected_types = frozenset(
        nonterminal[2:]
        for nonterminal in grammar.nonterminals
        if nonterminal.startswith("A:")
    )
    export_types: dict[str, str] = {}
    for production in grammar.productions:
        if not (
            len(production.rhs) == 3
            and isinstance(production.rhs[0], Nonterminal)
            and production.rhs[0].value.startswith("E:<module '")
            and isinstance(production.rhs[1], Terminal)
            and production.rhs[1].value == "."
            and isinstance(production.rhs[2], Terminal)
        ):
            continue
        parent_type = production.rhs[0].value[2:]
        match = re.fullmatch(r"<module '([^']+)'>", parent_type)
        if match is None:
            continue
        parent_module = match.group(1)
        if parent_module == module_value:
            qualified_name = production.rhs[2].value
        elif parent_module.startswith(f"{module_value}."):
            relative_parent = parent_module[len(module_value) + 1 :]
            qualified_name = f"{relative_parent}.{production.rhs[2].value}"
        else:
            continue
        previous = export_types.setdefault(qualified_name, production.lhs)
        if previous != production.lhs:
            raise EvaluationError(
                f"{path}: export {qualified_name!r} has conflicting E: types"
            )
    exports = frozenset(export_types.values())

    def local_domain(key: str, prefix: str) -> frozenset[str]:
        declared = metadata.get(key)
        if declared is None:
            return frozenset()
        if not isinstance(declared, list) or not all(
            isinstance(item, str) for item in declared
        ):
            raise EvaluationError(f"{path}: metadata {key} must be a string list")
        decoded: list[str] = []
        for item in declared:
            nonterminal = symbol_table.get(item, decode_library_nonterminal(item))
            if not nonterminal.startswith(prefix):
                raise EvaluationError(
                    f"{path}: metadata {key} contains invalid symbol {item!r}"
                )
            decoded.append(nonterminal[2:])
        if len(decoded) != len(set(decoded)):
            raise EvaluationError(f"{path}: metadata {key} contains duplicates")
        return frozenset(decoded)

    local_actual_types = local_domain(
        "local-assignability-actuals", "E:"
    )
    local_expected_types = local_domain(
        "local-assignability-expecteds", "A:"
    )
    local_assignability_complete = (
        metadata.get("local-assignability-complete") is True
    )
    local_link_productions: frozenset[Production] = frozenset()
    if local_assignability_complete:
        if (
            metadata.get("local-assignability-version")
            != ASSIGNABILITY_RELATION_VERSION
        ):
            raise EvaluationError(
                f"{path}: unsupported local assignability relation version"
            )
        structural_actual_types = frozenset(
            production.lhs[2:]
            for production in grammar.productions
            if production.lhs.startswith("E:")
        )
        if local_actual_types != structural_actual_types:
            raise EvaluationError(
                f"{path}: local assignability actual domain does not match "
                "the fragment E: production domain"
            )
        if local_expected_types != expected_types:
            raise EvaluationError(
                f"{path}: local assignability expected domain does not match "
                "the fragment A: domain"
            )
        declared_pairs = metadata.get("local-assignability-pairs")
        expected_pairs = len(local_actual_types) * len(local_expected_types)
        if declared_pairs != expected_pairs:
            raise EvaluationError(
                f"{path}: metadata local-assignability-pairs="
                f"{declared_pairs!r}, parsed {expected_pairs}"
            )
        local_link_productions = frozenset(
            production
            for production in grammar.productions
            if production.lhs.startswith("A:")
            and len(production.rhs) == 1
            and isinstance(production.rhs[0], Nonterminal)
            and production.rhs[0].value.startswith("E:")
        )
        local_links = {
            (production.lhs[2:], production.rhs[0].value[2:])
            for production in local_link_productions
        }
        if any(
            expected not in local_expected_types
            or actual not in local_actual_types
            for expected, actual in local_links
        ):
            raise EvaluationError(
                f"{path}: local assignability link lies outside declared domains"
            )
        if metadata.get("local-assignability-links") != len(local_links):
            raise EvaluationError(
                f"{path}: metadata local-assignability-links="
                f"{metadata.get('local-assignability-links')!r}, parsed "
                f"{len(local_links)}"
            )
    nonlocal_buckets: dict[int, set[Production]] = defaultdict(set)
    local_link_buckets: dict[str, set[Production]] = defaultdict(set)
    for production in grammar.productions:
        if production in local_link_productions:
            local_link_buckets[production.lhs[2:]].add(production)
        else:
            nonlocal_buckets[len(production.rhs)].add(production)
    return LibraryArtifact(
        path=path,
        module=module_value,
        module_type=expected_module_type,
        grammar=grammar,
        metadata=metadata,
        expected_types=expected_types,
        exports=exports,
        export_types=dict(export_types),
        local_actual_types=local_actual_types,
        local_expected_types=local_expected_types,
        local_assignability_complete=local_assignability_complete,
        nonlocal_productions_by_length={
            length: frozenset(productions)
            for length, productions in nonlocal_buckets.items()
        },
        local_links_by_expected={
            expected: frozenset(productions)
            for expected, productions in local_link_buckets.items()
        },
    )


def parse_library_cfg(path: Path) -> LibraryArtifact:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EvaluationError(f"cannot read library CFG {path}: {error}") from error
    return parse_library_cfg_text(text, path)


def imported_library_modules(source: str) -> tuple[str, ...]:
    """Return absolute modules named by syntactic import statements."""

    try:
        tree = ast.parse(source, type_comments=True)
    except (SyntaxError, ValueError):
        return ()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported = [node.module]
        else:
            continue
        for module in imported:
            modules.add(module)
            if "." in module:
                # Root fragments can make a submodule anchor productive, and
                # imported members share their exact ty E:type with cached
                # call rules even when no module spelling is in scope.
                modules.add(module.split(".", 1)[0])
    return tuple(sorted(modules))


def visible_absolute_import_nodes(
    source: str, target_line: int
) -> tuple[ast.Import | ast.ImportFrom, ...]:
    """Find conservative absolute import statements visible at a hole."""

    try:
        tree = ast.parse(source, type_comments=True)
    except (SyntaxError, ValueError):
        return ()
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    statement_line = target_line + 1
    targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.stmt)
        and getattr(node, "lineno", -1) == statement_line
    ]
    if not targets:
        return ()
    target = min(
        targets,
        key=lambda node: (
            (node.end_lineno or node.lineno) - node.lineno,
            (node.end_col_offset or node.col_offset) - node.col_offset,
        ),
    )
    target_scope = enclosing_scope(target, parents)
    if target_scope is None:
        return ()
    visible_scopes: set[ast.AST] = {target_scope}
    current = target_scope
    while current is not tree:
        parent_scope = enclosing_scope(current, parents)
        if parent_scope is None:
            break
        if isinstance(parent_scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            visible_scopes.add(parent_scope)
        current = parent_scope

    result: list[ast.Import | ast.ImportFrom] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and node.lineno < statement_line
            and enclosing_scope(node, parents) in visible_scopes
        ):
            continue
        if isinstance(node, ast.ImportFrom) and (
            node.level != 0 or not node.module
        ):
            continue
        result.append(node)
    return tuple(sorted(result, key=lambda node: (node.lineno, node.col_offset)))


def visible_imported_library_modules(
    source: str, target_line: int
) -> tuple[str, ...]:
    """Return absolute module names whose bindings are visible at a hole."""

    modules: set[str] = set()
    for node in visible_absolute_import_nodes(source, target_line):
        imported: Sequence[str | None] = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module]
        )
        for module in imported:
            if module is None:
                continue
            modules.add(module)
            if "." in module:
                modules.add(module.split(".", 1)[0])
    return tuple(sorted(modules))


def visible_from_import_bindings(
    source: str, target_line: int
) -> tuple[FromImportBinding, ...]:
    """Find conservative absolute from-import bindings visible at a hole."""

    result: set[FromImportBinding] = set()
    for node in visible_absolute_import_nodes(source, target_line):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            result.add(
                FromImportBinding(
                    module=node.module,
                    member=alias.name,
                    bound_name=alias.asname or alias.name,
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda binding: (
                binding.module,
                binding.member,
                binding.bound_name,
            ),
        )
    )


def artifact_covers_builder(
    artifact: LibraryArtifact, options: BuilderOptions
) -> bool:
    metadata = artifact.metadata
    if metadata.get("receiver-policy") != "module-namespaces":
        return False
    if metadata.get("receiver-limit-reached") is True:
        return False
    # ``ty`` currently marks every completion response incomplete, so that
    # advisory bit cannot decide whether the cache can replace live probing.
    # A response that actually reached ty's 1,000-item hard cap can be
    # truncated, however, and must retain the live fallback.
    completion_caps = metadata.get("raw-completion-queries-at-cap")
    if not isinstance(completion_caps, int) or completion_caps != 0:
        return False
    requirements = (
        ("max-call-arity", options.max_call_arity),
        ("max-layouts-per-signature", options.max_layouts_per_signature),
        ("member-depth", options.member_depth),
    )
    for key, required in requirements:
        declared = metadata.get(key)
        if not isinstance(declared, int) or declared < required:
            return False
    return True


def semantic_environment_python(ty: str) -> tuple[Path, str]:
    """Find the interpreter whose installed packages ty most likely resolves."""

    resolved_ty = shutil.which(ty)
    if resolved_ty is None and Path(ty).is_file():
        resolved_ty = str(Path(ty).resolve())
    if resolved_ty is not None:
        parent = Path(resolved_ty).resolve().parent
        for name in ("python", "python3", "python.exe"):
            candidate = parent / name
            if candidate.is_file():
                return candidate, "ty-sibling"
    environment = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if environment:
        root = Path(environment)
        for candidate in (
            root / "bin" / "python",
            root / "bin" / "python3",
            root / "Scripts" / "python.exe",
        ):
            if candidate.is_file():
                return candidate, "activated-environment"
    return Path(sys.executable), "current-interpreter"


def semantic_python_details(interpreter: Path) -> tuple[str, str]:
    result = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import platform;print(platform.python_version());print(platform.platform())",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    lines = result.stdout.splitlines()
    if result.returncode == 0 and len(lines) >= 2:
        return lines[0], lines[1]
    return platform.python_version(), platform.platform()


def semantic_package_versions(module: str, interpreter: Path) -> list[str]:
    """Fingerprint distributions in ty's activated semantic environment."""

    root = module.split(".", 1)[0]
    script = (
        "import importlib.metadata as m,json,sys;"
        "d=m.packages_distributions().get(sys.argv[1],[]);"
        "print(json.dumps(sorted({f'{x}=={m.version(x)}' for x in d},"
        "key=str.casefold)))"
    )
    result = subprocess.run(
        [str(interpreter), "-c", script, root],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        try:
            decoded = json.loads(result.stdout)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list) and all(
            isinstance(item, str) for item in decoded
        ):
            return decoded
    distributions = importlib.metadata.packages_distributions().get(root, [])
    fallback: list[str] = []
    for distribution in sorted(set(distributions), key=str.casefold):
        try:
            fallback.append(
                f"{distribution}=={importlib.metadata.version(distribution)}"
            )
        except importlib.metadata.PackageNotFoundError:
            continue
    return fallback


class LibraryCatalog:
    """Load and compatibility-check immutable module CFGs once per run."""

    def __init__(
        self, directory: Path, ty_executable: str, ty_release: str | None = None
    ):
        self.directory = directory
        self.ty = ty_release or ty_version(ty_executable)
        self.environment_python, self.environment_source = (
            semantic_environment_python(ty_executable)
        )
        self.python, self.platform = semantic_python_details(
            self.environment_python
        )
        self.cache: dict[str, LibraryLookup] = {}
        self.package_cache: dict[str, list[str]] = {}

    def lookup(self, module: str) -> LibraryLookup:
        cached = self.cache.get(module)
        if cached is not None:
            return cached
        path = self.directory / library_cfg_filename(module)
        if not path.is_file():
            result = LibraryLookup(module, None, "missing")
            self.cache[module] = result
            return result
        try:
            artifact = parse_library_cfg(path)
            reason = self.compatibility_error(module, artifact)
        except EvaluationError as error:
            artifact = None
            reason = f"invalid: {error}"
        if reason is not None:
            artifact = None
        result = LibraryLookup(module, artifact, reason)
        self.cache[module] = result
        return result

    def compatibility_error(
        self, module: str, artifact: LibraryArtifact
    ) -> str | None:
        metadata = artifact.metadata
        if artifact.module != module:
            return f"module mismatch ({artifact.module})"
        expected_values = {
            "ty": self.ty,
            "python": self.python,
            "platform": self.platform,
            "python-executable": str(self.environment_python.resolve()),
            "python-environment-source": self.environment_source,
        }
        for key, expected in expected_values.items():
            if metadata.get(key) != expected:
                return f"{key} mismatch"
        packages = metadata.get("package")
        if not isinstance(packages, list) or not all(
            isinstance(item, str) for item in packages
        ):
            return "invalid package fingerprint"
        root = module.split(".", 1)[0]
        installed = self.package_cache.get(root)
        if installed is None:
            installed = semantic_package_versions(root, self.environment_python)
            self.package_cache[root] = installed
        if packages != installed:
            return "package fingerprint mismatch"
        return None


@dataclass
class BuildStats:
    scope_names: int = 0
    expression_types: int = 0
    callables: int = 0
    signatures: int = 0
    receiver_types: int = 0
    member_completions: int = 0
    incomplete_completion_queries: int = 0
    completion_queries_at_cap: int = 0
    dynamic_types: int = 0
    assignment_types_checked: int = 0
    assignment_types_rejected: int = 0
    output_producer_families: int = 0
    output_producers_checked: int = 0
    output_producers_rejected: int = 0
    output_producers_local_fallback: int = 0
    output_producers_unchecked: int = 0
    output_producer_validation_seconds: float = 0.0
    module_member_fallbacks: int = 0
    derived_representatives: int = 0
    invalid_representatives: int = 0
    library_artifacts: int = 0
    library_productions: int = 0
    library_live_fallbacks: int = 0
    library_incomplete_artifacts: int = 0
    assignability_pairs_cached: int = 0
    assignability_pairs_checked: int = 0
    binary_operation_productions: int = 0
    subscript_productions: int = 0
    slice_productions: int = 0


class GrammarBuilder:
    def __init__(
        self,
        probe: SemanticProbe,
        source_ids: frozenset[str],
        options: BuilderOptions,
        signature_cache: dict[tuple[str, str], tuple[Signature, ...]],
        required_assignment: str | None = None,
        library_artifacts: Sequence[LibraryArtifact] = (),
        from_import_bindings: Sequence[FromImportBinding] = (),
    ):
        self.probe = probe
        self.source_ids = source_ids
        self.options = options
        self.signature_cache = signature_cache
        self.required_assignment = required_assignment
        self.library_artifacts = tuple(library_artifacts)
        self.from_import_bindings = tuple(from_import_bindings)
        self.active_library_artifacts: list[LibraryArtifact] = []
        self.cache_eligible_module_types = {
            artifact.module_type
            for artifact in self.library_artifacts
            if artifact_covers_builder(artifact, options)
        }
        self.precomputed_module_types: set[str] = set()
        self.grammar = Grammar(start="START")
        self.stats = BuildStats()
        self.representatives: dict[str, str] = {}
        self.callables: dict[str, str] = {}
        self.exact_callables: set[tuple[str, str]] = set()
        self.processed_callable_entries: set[tuple[str, str]] = set()
        self.member_callable_receivers: dict[tuple[str, str], str] = {}
        self.contextual_dynamic_output_families: set[str] = set()
        self.internal_expression_types: set[str] = set()
        self.receivers: list[tuple[int, int, str, str, str]] = []
        self.receiver_entries: dict[
            str, tuple[int, int, str, str, str]
        ] = {}
        self.processed_receiver_types: set[str] = set()
        self.expected_types: set[str] = set()
        self.contextual_call_results: dict[str, bool] = {}
        self.dynamic_representatives: dict[str, set[str]] = defaultdict(set)
        self.dynamic_scope_representatives: dict[str, set[str]] = defaultdict(set)
        contextual_source = getattr(self.probe, "ablated", None)
        self.has_contextual_source = isinstance(contextual_source, str)
        try:
            contextual_tree = (
                ast.parse(contextual_source)
                if isinstance(contextual_source, str)
                else None
            )
        except SyntaxError:
            contextual_tree = None
        self.source_callable_names = frozenset(
            node.name
            for node in ast.walk(contextual_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ) if contextual_tree is not None else frozenset()

    def expression_nonterminal(self, type_display: str) -> str:
        normalized = normalize_type(type_display)
        name = type_nonterminal(normalized)
        self.grammar.type_labels[name] = normalized
        return name

    def callable_entries(self) -> tuple[tuple[str, str], ...]:
        """Return type/callable spellings, retaining exact dynamic origins."""

        return tuple(
            sorted({*self.callables.items(), *self.exact_callables})
        )

    def callable_symbols(
        self, detail: str, expression: str
    ) -> tuple[Symbol, ...]:
        """Lower a callable without merging unrelated dynamic receivers."""

        if (detail, expression) not in self.exact_callables:
            return (Nonterminal(self.expression_nonterminal(detail)),)
        tokens = canonical_expression_tokens(expression)
        if tokens is None:
            return ()
        return tuple(Terminal(token) for token in tokens)

    def contextual_dynamic_output_nonterminal(
        self,
        return_type: str,
        identity: object,
    ) -> str:
        """Create one producer-specific Any/Unknown result family.

        The family remains usable through its displayed ``Any``/``Unknown``
        expression type, but output assignments root it only after a complete
        producer from this exact family passes in the ablated downstream
        context.
        """

        normalized_return = normalize_type(return_type)
        encoded_identity = repr((normalized_return, identity))
        nonterminal = (
            f"C:__contextual_dynamic_"
            f"{stable_digest(encoded_identity, 16)}__"
        )
        self.contextual_dynamic_output_families.add(nonterminal)
        self.grammar.add(
            self.expression_nonterminal(normalized_return),
            Nonterminal(nonterminal),
        )
        return nonterminal

    def contextual_dynamic_call_nonterminal(
        self,
        return_type: str,
        detail: str,
        expression: str,
        layout: ArgumentLayout,
    ) -> str:
        return self.contextual_dynamic_output_nonterminal(
            return_type,
            ("call", detail, expression, layout),
        )

    def call_result_nonterminal(
        self,
        type_display: str,
        *,
        trusted_dynamic_output: bool = False,
    ) -> str:
        """Return the result symbol for a call justified by a ty signature.

        A displayed ``Any``/``Unknown`` result cannot be validated by testing
        an arbitrary witness of the same broad type in the post-statement
        context.  Keep such calls on a separate trusted path: they remain
        usable as ordinary unknown/dynamic expressions, but output-assignment
        rooting can admit the call itself without admitting every unrelated
        ``E:Unknown`` witness.
        """

        normalized = normalize_type(type_display)
        expression = self.expression_nonterminal(normalized)
        if normalized not in {"Any", "Unknown"}:
            return expression
        if not trusted_dynamic_output:
            return expression
        self.grammar.add(
            expression,
            Nonterminal(TRUSTED_DYNAMIC_CALL_NONTERMINAL),
        )
        self.grammar.add(
            DYNAMIC_NONTERMINAL,
            Nonterminal(TRUSTED_DYNAMIC_CALL_NONTERMINAL),
        )
        return TRUSTED_DYNAMIC_CALL_NONTERMINAL

    def trusts_dynamic_callable(self, detail: str, expression: str) -> bool:
        """Whether an erased result may retain its signature-backed call shape.

        Builtin/generic constructors sometimes expose ``Unknown`` only because
        signature help has erased a type variable.  Treating every such result
        as one trusted output family merged unrelated calls such as ``tuple()``
        with a source-defined ``lcm(...)``.  A source-defined function has a
        declaration in the contextual source; its signature-backed call may
        use the narrow trusted-output path.  Correlated library calls (currently
        ``heappop``) opt in explicitly at their grounding site.
        """

        terminal_name = expression.rsplit(".", 1)[-1]
        if terminal_name in self.source_callable_names:
            return True
        # Minimal semantic fixtures do not carry source text.  Retain the
        # equivalent narrow signal there without trusting Python builtins.
        return (
            not self.has_contextual_source
            and expression.isidentifier()
            and expression in self.source_ids
            and expression not in BUILTIN_NAMES
            and re.search(rf"\bdef\s+{re.escape(expression)}(?:\[|\()", detail)
            is not None
        )

    def add_expression(
        self,
        type_display: str,
        rhs: Sequence[Symbol],
        *,
        representative: str | None = None,
    ) -> str:
        normalized = normalize_type(type_display)
        nonterminal = self.expression_nonterminal(normalized)
        self.grammar.add(nonterminal, *rhs)
        if representative is not None:
            self.representatives.setdefault(normalized, representative)
            if normalized in {"Any", "Unknown"}:
                self.dynamic_representatives[normalized].add(representative)
        if normalized in {"Any", "Unknown"}:
            self.stats.dynamic_types += 1
        return nonterminal

    def add_literals(self) -> None:
        literal_rows = [
            ("None", "None"),
            ("bool", "False"),
            ("bool", "True"),
            ("int", "0"),
            ("float", "0.0"),
            ("complex", "0j"),
            ("str", '""'),
            ("bytes", 'b""'),
        ]
        for type_display, token in literal_rows:
            self.add_expression(
                type_display,
                (Terminal(token),),
                representative=token,
            )
        for type_display, operator in (
            ("int", "+"),
            ("int", "-"),
            ("int", "~"),
            ("float", "+"),
            ("float", "-"),
            ("complex", "+"),
            ("complex", "-"),
        ):
            nonterminal = self.expression_nonterminal(type_display)
            self.grammar.add(
                nonterminal,
                Terminal(operator),
                Nonterminal(nonterminal),
            )

    def add_scope(self) -> None:
        completions, incomplete = self.probe.scope()
        if incomplete:
            self.stats.incomplete_completion_queries += 1
        if len(completions) >= 1000:
            self.stats.completion_queries_at_cap += 1
        completion_by_label = {item.label: item for item in completions}
        for label in sorted(self.source_ids):
            if (
                label in completion_by_label
                or label in self.probe.excluded_names
                or not label.isidentifier()
                or keyword.iskeyword(label)
                or not self.probe.accepts_expression(label)
            ):
                continue
            detail = self.probe.hover_expression(label)
            if detail is not None:
                completion_by_label[label] = Completion(label, detail, None)
        completions = sorted(completion_by_label.values(), key=lambda item: item.label)
        for completion in completions:
            if completion.label in self.probe.excluded_names:
                continue
            if (
                completion.label in BUILTIN_NAMES
                and completion.label not in CORE_BUILTINS
                and completion.label not in self.source_ids
            ):
                continue
            detail = completion.detail
            needs_exact_probe = (
                completion.label in self.source_ids
                or completion.label not in BUILTIN_NAMES
            )
            if needs_exact_probe:
                if not self.probe.accepts_expression(completion.label):
                    continue
                detail = self.probe.hover_expression(completion.label) or detail
            detail = normalize_type(detail)
            if detail in {"Any", "Unknown"}:
                self.grammar.add(DYNAMIC_NONTERMINAL, Terminal(completion.label))
                self.grammar.add(
                    self.expression_nonterminal(detail),
                    Nonterminal(DYNAMIC_NONTERMINAL),
                )
                self.representatives.setdefault(detail, completion.label)
                self.dynamic_representatives[detail].add(completion.label)
                if completion.label in self.source_ids:
                    self.dynamic_scope_representatives[detail].add(
                        completion.label
                    )
                self.stats.dynamic_types += 1
            else:
                self.add_expression(
                    detail,
                    (Terminal(completion.label),),
                    representative=completion.label,
                )
            self.stats.scope_names += 1
            callable_value = is_callable_type(detail, completion.kind)
            if callable_value:
                if detail in {"Any", "Unknown"}:
                    self.exact_callables.add((detail, completion.label))
                else:
                    self.callables.setdefault(detail, completion.label)
            should_receive = not callable_value or (
                completion.kind == LSP_CLASS
                and completion.label in self.source_ids
                and completion.label not in BUILTIN_NAMES
            )
            if should_receive:
                self.queue_receiver(detail, completion.label, 0)

    def queue_receiver(self, type_display: str, expression: str, depth: int) -> None:
        type_display = normalize_type(type_display)
        if type_display in self.precomputed_module_types:
            return
        receiver_key = (
            f"{type_display}\0{expression}"
            if type_display in {"Any", "Unknown"}
            else type_display
        )
        if receiver_key in self.processed_receiver_types:
            return
        if type_display in {"Never", "NoReturn", "None"}:
            return
        contextual = bool(
            self.source_ids.intersection(source_identifiers(expression))
        )
        common_container = type_display in {
            "bytes",
            "dict",
            "list",
            "set",
            "str",
            "tuple",
        }
        priority = 0 if contextual else 1 if common_container else 2
        entry = (priority, depth, expression, type_display, receiver_key)
        current = self.receiver_entries.get(receiver_key)
        if current is not None and current <= entry:
            return
        self.receiver_entries[receiver_key] = entry
        heapq.heappush(self.receivers, entry)

    def add_library_artifacts(self) -> None:
        pending = list(self.library_artifacts)
        while pending:
            generating: set[str] = set()
            changed = True
            while changed:
                changed = False
                for production in self.grammar.productions:
                    if production.lhs in generating:
                        continue
                    if all(
                        isinstance(symbol, Terminal)
                        or symbol.value in generating
                        for symbol in production.rhs
                    ):
                        generating.add(production.lhs)
                        changed = True
            active = [
                artifact
                for artifact in pending
                if self.library_artifact_is_active(artifact, generating)
            ]
            if not active:
                break
            for artifact in active:
                pending.remove(artifact)
                self.add_library_artifact(artifact)

    def library_artifact_is_active(
        self, artifact: LibraryArtifact, generating: set[str]
    ) -> bool:
        module_anchor = type_nonterminal(artifact.module_type)
        if module_anchor in generating:
            return True
        if any(
            export_type.startswith("E:<module '")
            and export_type in generating
            for export_type in artifact.export_types.values()
        ):
            return True
        for binding in self.from_import_bindings:
            if binding.module == artifact.module:
                qualified_name = binding.member
            elif binding.module.startswith(f"{artifact.module}."):
                relative_module = binding.module[len(artifact.module) + 1 :]
                qualified_name = f"{relative_module}.{binding.member}"
            else:
                continue
            export_type = artifact.export_types.get(qualified_name)
            if export_type is None:
                continue
            if Production(
                export_type, (Terminal(binding.bound_name),)
            ) in self.grammar.productions:
                return True
        return False

    def add_library_artifact(self, artifact: LibraryArtifact) -> None:
        """Union one fragment after its exact module anchor is productive."""

        self.active_library_artifacts.append(artifact)

        # Every grammar symbol yields at least one lexical token.  A live
        # assignment spends two root tokens on ``name =``; private/expression
        # holes retain a bare-expression alternative.  Pull only the indexed
        # non-local buckets that can fit this exact expression budget.  Dense
        # cached A->E links are added lazily in finish() for argument types
        # that survived this bounded call-production filter.
        expression_budget = self.options.max_tokens - (
            2 if self.required_assignment is not None else 0
        )
        productions: set[Production] = set()
        for rhs_length, bucket in artifact.nonlocal_productions_by_length.items():
            if rhs_length <= expression_budget:
                productions.update(bucket)
        used_nonterminals = {
            production.lhs for production in productions
        } | {
            symbol.value
            for production in productions
            for symbol in production.rhs
            if isinstance(symbol, Nonterminal)
        }
        for nonterminal in used_nonterminals:
            label = artifact.grammar.type_labels.get(nonterminal)
            if label is None:
                continue
            existing = self.grammar.type_labels.get(nonterminal)
            if existing is not None and existing != label:
                raise EvaluationError(
                    f"library type-label collision for {nonterminal!r}"
                )
            self.grammar.type_labels[nonterminal] = label
        self.grammar.productions.update(productions)
        self.expected_types.update(
            nonterminal[2:]
            for nonterminal in used_nonterminals
            if nonterminal.startswith("A:")
        )
        self.stats.library_artifacts += 1
        self.stats.library_productions += len(productions)
        if artifact.metadata.get("completion-complete") is not True:
            self.stats.library_incomplete_artifacts += 1
        covered = artifact.module_type in self.cache_eligible_module_types
        if not covered:
            # The fragment remains a sound cache of the portion it covers;
            # retain live expansion to fill its declared generation bounds.
            self.stats.library_live_fallbacks += 1
            return

        for export in artifact.exports:
            if export.startswith("E:"):
                self.callables.pop(export[2:], None)

        # A namespace artifact replaces only module completion/signature
        # discovery.  Seed representatives and continue live receiver probing
        # for exported non-module values, preserving contextual member access
        # such as ``sys.stdin.readline`` without re-querying the whole module.
        receiver_nonterminals = {
            production.rhs[0].value
            for production in productions
            if len(production.rhs) >= 3
            and isinstance(production.rhs[0], Nonterminal)
            and isinstance(production.rhs[1], Terminal)
            and production.rhs[1].value == "."
            and production.rhs[0].value.startswith("E:<module '")
        }
        self.precomputed_module_types.update(
            nonterminal[2:]
            for nonterminal in receiver_nonterminals
        )
        member_rows = sorted(
            (
                production
                for production in productions
                if production.lhs.startswith("E:")
                and len(production.rhs) == 3
                and isinstance(production.rhs[0], Nonterminal)
                and production.rhs[0].value.startswith("E:")
                and isinstance(production.rhs[1], Terminal)
                and production.rhs[1].value == "."
                and isinstance(production.rhs[2], Terminal)
            ),
            key=lambda production: (
                production.rhs[0].value,
                production.rhs[2].value,
                production.lhs,
            ),
        )
        pending = member_rows
        while pending:
            deferred: list[Production] = []
            progressed = False
            for production in pending:
                parent_type = production.rhs[0].value.removeprefix("E:")
                parent = self.representatives.get(parent_type)
                if parent is None:
                    deferred.append(production)
                    continue
                member_type = production.lhs.removeprefix("E:")
                expression = f"{parent}.{production.rhs[2].value}"
                if member_type not in self.representatives:
                    self.representatives[member_type] = expression
                    if member_type in {"Any", "Unknown"}:
                        self.dynamic_representatives[member_type].add(expression)
                    progressed = True
                if not is_callable_type(member_type):
                    self.queue_receiver(member_type, expression, 1)
            if not progressed:
                break
            pending = deferred

    def add_members(self) -> None:
        string_representative = self.representatives.get("str")
        if string_representative is not None:
            self.queue_receiver("str", string_representative, 0)
        while (
            self.receivers
            and self.stats.receiver_types < self.options.max_receiver_types
        ):
            entry = heapq.heappop(self.receivers)
            (
                _priority,
                depth,
                expression,
                receiver_type,
                receiver_key,
            ) = entry
            if self.receiver_entries.get(receiver_key) != entry:
                continue
            del self.receiver_entries[receiver_key]
            self.processed_receiver_types.add(receiver_key)
            if receiver_type in self.precomputed_module_types:
                continue
            self.stats.receiver_types += 1
            completions, incomplete = self.probe.members(expression)
            if incomplete:
                self.stats.incomplete_completion_queries += 1
            if len(completions) >= 1000:
                self.stats.completion_queries_at_cap += 1
            module_receiver = receiver_type.startswith("<module '")
            expand_all_module_members = (
                module_receiver
                and len(completions) <= self.options.max_module_members
            )
            # Source-contextual receivers and the small built-in containers are
            # both bounded and especially likely to supply the held-out API.
            # Keeping only a hand-written member allowlist here would discard
            # valid completions such as ``deque.popleft`` or ``str.rfind`` even
            # after the LSP supplied them.
            expand_all_receiver_members = (
                not module_receiver and _priority <= 1
            )
            if module_receiver and not expand_all_module_members:
                self.stats.module_member_fallbacks += 1
            receiver_nonterminal = self.expression_nonterminal(receiver_type)
            for completion in completions:
                if (
                    not expand_all_receiver_members
                    and not expand_all_module_members
                    and completion.label not in CORE_MEMBERS
                    and completion.label not in self.source_ids
                ):
                    continue
                member_type = normalize_type(completion.detail)
                member_expression = f"{expression}.{completion.label}"
                member_rhs: tuple[Symbol, ...]
                if receiver_type in {"Any", "Unknown"}:
                    receiver_tokens = canonical_expression_tokens(expression)
                    if receiver_tokens is None:
                        continue
                    # Completion was queried for this exact dynamic receiver.
                    # A shared E:Any/E:Unknown receiver would cross-product its
                    # members onto every unrelated dynamic expression.
                    member_rhs = (
                        *(Terminal(token) for token in receiver_tokens),
                        Terminal("."),
                        Terminal(completion.label),
                    )
                else:
                    member_rhs = (
                        Nonterminal(receiver_nonterminal),
                        Terminal("."),
                        Terminal(completion.label),
                    )
                if member_type in {"Any", "Unknown"}:
                    self.grammar.add(DYNAMIC_NONTERMINAL, *member_rhs)
                    self.grammar.add(
                        self.expression_nonterminal(member_type),
                        Nonterminal(DYNAMIC_NONTERMINAL),
                    )
                    self.representatives.setdefault(
                        member_type, member_expression
                    )
                    self.dynamic_representatives[member_type].add(
                        member_expression
                    )
                    self.stats.dynamic_types += 1
                else:
                    self.add_expression(
                        member_type,
                        member_rhs,
                        representative=member_expression,
                    )
                self.stats.member_completions += 1
                callable_value = is_callable_type(member_type, completion.kind)
                if callable_value:
                    callable_identity = (member_type, member_expression)
                    receiver_instance = class_instance_type(receiver_type)
                    if receiver_instance is not None:
                        # Class attributes expose an unbound descriptor.  Keep
                        # its lexical receiver separate from otherwise equal
                        # callable displays on other classes, and retain the
                        # owner needed to restore the erased ``self`` bound.
                        self.exact_callables.add(callable_identity)
                        self.member_callable_receivers[
                            callable_identity
                        ] = receiver_instance
                    elif receiver_type in {"Any", "Unknown"} or member_type in {
                        "Any",
                        "Unknown",
                    }:
                        self.exact_callables.add(callable_identity)
                    else:
                        self.callables.setdefault(
                            member_type, member_expression
                        )
                elif depth < self.options.member_depth:
                    self.queue_receiver(member_type, member_expression, depth + 1)

    def signatures_for(self, detail: str, expression: str) -> tuple[Signature, ...]:
        cache_key = (detail, expression)
        cached = self.signature_cache.get(cache_key)
        if cached is not None:
            return cached
        labels = self.probe.signatures(expression)
        signatures = [
            parsed
            for label in labels
            if (parsed := parse_signature(label)) is not None
        ]
        if not signatures:
            signatures = signatures_from_detail(detail)
        unique = tuple(dict.fromkeys(signatures))
        self.signature_cache[cache_key] = unique
        return unique

    def add_context_validated_append(
        self,
        return_nonterminal: str,
        expression: str,
        expected: str,
    ) -> None:
        """Add an exact append word when ty proves a flow-sensitive argument.

        ty can retain a broad displayed type such as ``set[tuple[int, ...]]``
        while its flow graph knows that one local contains only pairs.  A
        global ``A:`` edge would be unsound because mutable containers are
        invariant.  Instead, validate the exact receiver/argument spelling in
        this ablated context and add only that lexical call production.
        """

        if (
            return_nonterminal != self.expression_nonterminal("None")
            or not expression.endswith(".append")
        ):
            return
        callable_tokens = dotted_identifier_tokens(expression)
        if callable_tokens is None:
            return
        for actual, representative in sorted(self.representatives.items()):
            if (
                not representative.isidentifier()
                or keyword.iskeyword(representative)
                or not set_tuple_refinement_candidate(actual, expected)
            ):
                continue
            call = f"{expression}({representative})"
            accepted = self.contextual_call_results.get(call)
            if accepted is None:
                accepted = self.probe.accepts_expression(call)
                self.contextual_call_results[call] = accepted
            if accepted:
                self.grammar.add(
                    return_nonterminal,
                    *(Terminal(token) for token in callable_tokens),
                    Terminal("("),
                    Terminal(representative),
                    Terminal(")"),
                )

    def add_calls(self) -> None:
        callable_entries = tuple(
            entry
            for entry in self.callable_entries()
            if entry not in self.processed_callable_entries
        )
        self.processed_callable_entries.update(callable_entries)
        self.stats.callables = len(self.processed_callable_entries)
        for callable_type, expression in callable_entries:
            signatures = self.signatures_for(callable_type, expression)
            if (
                not signatures
                and callable_type in {"Any", "Unknown"}
                and (callable_type, expression) in self.exact_callables
            ):
                # ty marks a completion as callable even when an Any/Unknown
                # receiver has no inspectable signature.  That exact member
                # accepts arbitrary positional arguments under ty's gradual
                # semantics; retain the bounded arities without sharing the
                # member with any other dynamic receiver.
                signatures = (
                    Signature(
                        (
                            Parameter(
                                "args", "object", "vararg", False
                            ),
                        ),
                        callable_type,
                    ),
                )
            self.stats.signatures += len(signatures)
            callable_symbols = self.callable_symbols(
                callable_type, expression
            )
            if not callable_symbols:
                continue
            for signature in signatures:
                receiver_instance = self.member_callable_receivers.get(
                    (callable_type, expression)
                )
                if receiver_instance is not None:
                    signature = bind_unbound_self_signature(
                        signature, receiver_instance
                    )
                if expression == "map":
                    # ``map`` couples the callback's positional parameters to
                    # the element type of each following iterable.  Lowering
                    # its displayed signature one slot at a time loses that
                    # correlation and admits, for example, ``map(x.bit_length,
                    # numbers)`` or a unary callback with two iterables.  The
                    # correlated rows are added in add_grounded_generic_calls.
                    continue
                normalized_return = normalize_type(signature.return_type)
                if (
                    expression == "tuple"
                    and (
                        normalized_return in {"Any", "Unknown"}
                        or has_unresolved_type_variable(normalized_return)
                    )
                ):
                    # ty's live builtin signature currently erases tuple's
                    # covariant element parameter to Unknown.  A broad
                    # Unknown call row destroys the fact that the result is a
                    # tuple; add_grounded_generic_calls restores its precise
                    # zero/one-argument shapes below.
                    continue
                layouts = argument_layouts(
                    signature,
                    max_arity=self.options.max_call_arity,
                    max_layouts=self.options.max_layouts_per_signature,
                )
                trusted_dynamic_output = (
                    normalized_return == "Any"
                    or self.trusts_dynamic_callable(
                        callable_type, expression
                    )
                )
                for layout in layouts:
                    if expression in {"max", "min"} and len(layout.positional) > 1:
                        # Correlated variadic type variables are not independent
                        # CFG slots.  The iterable overload remains available.
                        continue
                    if (
                        normalized_return in {"Any", "Unknown"}
                        and not trusted_dynamic_output
                    ):
                        return_nonterminal = (
                            self.contextual_dynamic_call_nonterminal(
                                normalized_return,
                                callable_type,
                                expression,
                                layout,
                            )
                        )
                    else:
                        return_nonterminal = self.call_result_nonterminal(
                            normalized_return,
                            trusted_dynamic_output=trusted_dynamic_output,
                        )
                    rhs: list[Symbol] = [
                        *callable_symbols,
                        Terminal("("),
                    ]
                    normalized_positional: list[str] = []
                    first = True
                    for position, expected in enumerate(layout.positional):
                        if not first:
                            rhs.append(Terminal(","))
                        first = False
                        if expression == "dict" and position == 0:
                            expected = "__dict_source__"
                        elif expression == "sum" and position == 0:
                            expected = "__numeric_iterable__"
                        elif expression == "sum" and position == 1:
                            expected = "__numeric__"
                        elif "list[Divergent]" in callable_type:
                            # An empty list's element type widens on mutation;
                            # Divergent here is ty's inference bottom, not a
                            # demand that the argument itself be bottom-typed.
                            expected = re.sub(r"\bDivergent\b", "object", expected)
                        expected = normalize_type(expected)
                        normalized_positional.append(expected)
                        self.expected_types.add(expected)
                        rhs.append(Nonterminal(argument_nonterminal(expected)))
                    for name, expected in layout.keywords:
                        if not first:
                            rhs.append(Terminal(","))
                        first = False
                        if "list[Divergent]" in callable_type:
                            expected = re.sub(r"\bDivergent\b", "object", expected)
                        expected = normalize_type(expected)
                        self.expected_types.add(expected)
                        rhs.extend(
                            (
                                Terminal(name),
                                Terminal("="),
                                Nonterminal(argument_nonterminal(expected)),
                            )
                        )
                    rhs.append(Terminal(")"))
                    self.grammar.add(return_nonterminal, *rhs)
                    if len(normalized_positional) == 1 and not layout.keywords:
                        self.add_context_validated_append(
                            return_nonterminal,
                            expression,
                            normalized_positional[0],
                        )

        expression_types = {
            production.lhs[2:]
            for production in self.grammar.productions
            if production.lhs.startswith("E:")
        }
        comparable_types = {
            actual
            for actual in expression_types
            if is_assignable(actual, "SupportsRichComparisonT")
        }
        for callable_type, expression in callable_entries:
            if expression not in {"max", "min"}:
                continue
            callable_symbols = self.callable_symbols(
                callable_type, expression
            )
            if not callable_symbols:
                continue
            for actual in comparable_types:
                result_nonterminal = self.expression_nonterminal(actual)
                for arity in range(2, self.options.max_call_arity + 1):
                    variadic_rhs: list[Symbol] = [
                        *callable_symbols,
                        Terminal("("),
                    ]
                    for index in range(arity):
                        if index:
                            variadic_rhs.append(Terminal(","))
                        variadic_rhs.append(Nonterminal(type_nonterminal(actual)))
                    variadic_rhs.append(Terminal(")"))
                    self.grammar.add(result_nonterminal, *variadic_rhs)

            # Ground the single-iterable overload's result to the iterable's
            # concrete element type.  LSP signature help otherwise leaves the
            # constrained type variable unresolved (for example min(list[float])).
            for actual in sorted(expression_types):
                element_type = iterable_element_type(actual)
                if (
                    element_type is None
                    or not is_assignable(
                        element_type, "SupportsRichComparisonT"
                    )
                ):
                    continue
                self.grammar.add(
                    self.expression_nonterminal(element_type),
                    *callable_symbols,
                    Terminal("("),
                    Nonterminal(type_nonterminal(actual)),
                    Terminal(")"),
                )

    def add_grounded_generic_calls(self) -> None:
        """Ground iterable APIs whose LSP types retain correlations.

        Ordinary call productions intentionally keep ty's displayed type
        variables symbolic.  That loses safe correlations such as
        ``map(str, ints) -> map[str]`` and consequently prevents a later
        ``str.join`` from seeing an ``Iterable[str]``.  For ``map``, enumerate
        each callable's accepted positional arities and couple every callback
        parameter to the corresponding iterable element type.  The remaining
        helpers enumerate concrete iterable element types already present in
        the contextual grammar.  Each added row is a direct instance of the
        corresponding ty signature or Python container protocol; no
        independent callback/iterable cross-product is added.
        """

        callable_by_expression: dict[str, tuple[str, str]] = {
            expression: (
                callable_type,
                self.expression_nonterminal(callable_type),
            )
            for callable_type, expression in self.callable_entries()
        }
        map_entry = callable_by_expression.get("map")
        if map_entry is not None and self.options.max_call_arity >= 2:
            _map_type, map_nonterminal = map_entry
            max_callback_arity = self.options.max_call_arity - 1
            for callable_type, expression in self.callable_entries():
                callable_symbols = self.callable_symbols(
                    callable_type, expression
                )
                if not callable_symbols:
                    continue
                for signature in self.signatures_for(callable_type, expression):
                    return_type = normalize_type(signature.return_type)
                    if return_type in {"Divergent", "Never", "NoReturn"} or (
                        has_unresolved_type_variable(return_type)
                    ):
                        continue
                    layouts = argument_layouts(
                        signature,
                        max_arity=max_callback_arity,
                        max_layouts=self.options.max_layouts_per_signature,
                    )
                    for layout in layouts:
                        if not layout.positional or layout.keywords:
                            continue
                        rhs: list[Symbol] = [
                            Nonterminal(map_nonterminal),
                            Terminal("("),
                            *callable_symbols,
                        ]
                        for expected_element in layout.positional:
                            expected_iterable = (
                                f"Iterable[{normalize_type(expected_element)}]"
                            )
                            self.expected_types.add(expected_iterable)
                            rhs.extend(
                                (
                                    Terminal(","),
                                    Nonterminal(
                                        argument_nonterminal(expected_iterable)
                                    ),
                                )
                            )
                        rhs.append(Terminal(")"))
                        self.grammar.add(
                            self.expression_nonterminal(f"map[{return_type}]"),
                            *rhs,
                        )

        def concrete_iterable_elements() -> set[str]:
            elements: set[str] = set()
            expression_types = {
                production.lhs[2:]
                for production in self.grammar.productions
                if production.lhs.startswith("E:")
            }
            for actual in expression_types:
                element = iterable_element_type(actual)
                if element is not None and groundable_type(element):
                    elements.add(normalize_type(element))
            return elements

        tuple_entry = callable_by_expression.get("tuple")
        if tuple_entry is not None:
            _tuple_type, tuple_nonterminal = tuple_entry
            expression_types = {
                production.lhs[2:]
                for production in self.grammar.productions
                if production.lhs.startswith("E:")
            }
            self.grammar.add(
                self.expression_nonterminal("tuple[()]"),
                Nonterminal(tuple_nonterminal),
                Terminal("("),
                Terminal(")"),
            )
            self.grammar.add(
                self.expression_nonterminal("tuple[()]"),
                Nonterminal(tuple_nonterminal),
                Terminal("("),
                Nonterminal(type_nonterminal("tuple[()]")),
                Terminal(")"),
            )
            for actual in sorted(expression_types):
                if normalize_type(actual) == "tuple[()]":
                    result_type = "tuple[()]"
                else:
                    element = iterable_element_type(actual)
                    if element is not None:
                        element = normalize_type(element)
                    if element not in {None, "Any", "Unknown"}:
                        result_type = f"tuple[{element}, ...]"
                    elif is_assignable(actual, "Iterable[Unknown]"):
                        # Keep an opaque element only for this exact displayed
                        # actual type.  A shared A:Iterable[Unknown] fallback
                        # overlaps every concrete grounding and doubles parse
                        # counts for ordinary tuple(iterable) words.
                        result_type = "tuple[Unknown, ...]"
                    else:
                        continue
                self.grammar.add(
                    self.expression_nonterminal(result_type),
                    Nonterminal(tuple_nonterminal),
                    Terminal("("),
                    Nonterminal(type_nonterminal(actual)),
                    Terminal(")"),
                )

        list_entry = callable_by_expression.get("list")
        if list_entry is not None and self.options.max_call_arity >= 1:
            _list_type, list_nonterminal = list_entry
            for element in sorted(concrete_iterable_elements()):
                expected = f"Iterable[{element}]"
                self.expected_types.add(expected)
                self.grammar.add(
                    self.expression_nonterminal(f"list[{element}]"),
                    Nonterminal(list_nonterminal),
                    Terminal("("),
                    Nonterminal(argument_nonterminal(expected)),
                    Terminal(")"),
                )

        # Recompute after grounding list(map(...)); sorted can now retain the
        # map/list element type instead of returning list[TypeVariable].
        sorted_entry = callable_by_expression.get("sorted")
        if sorted_entry is not None and self.options.max_call_arity >= 1:
            _sorted_type, sorted_nonterminal = sorted_entry
            for element in sorted(concrete_iterable_elements()):
                if not is_assignable(element, "SupportsRichComparisonT"):
                    continue
                expected = f"Iterable[{element}]"
                self.expected_types.update((expected, "bool"))
                result = self.expression_nonterminal(f"list[{element}]")
                prefix: tuple[Symbol, ...] = (
                    Nonterminal(sorted_nonterminal),
                    Terminal("("),
                    Nonterminal(argument_nonterminal(expected)),
                )
                self.grammar.add(result, *prefix, Terminal(")"))
                self.grammar.add(
                    result,
                    *prefix,
                    Terminal(","),
                    Terminal("reverse"),
                    Terminal("="),
                    Nonterminal(argument_nonterminal("bool")),
                    Terminal(")"),
                )

        reversed_entry = callable_by_expression.get("reversed")
        if reversed_entry is not None and self.options.max_call_arity >= 1:
            _reversed_type, reversed_nonterminal = reversed_entry
            for element in sorted(concrete_iterable_elements()):
                result = self.expression_nonterminal(f"reversed[{element}]")
                for protocol in (
                    f"_SupportsReversed[{element}]",
                    f"SupportsLenAndGetItem[{element}]",
                ):
                    self.expected_types.add(protocol)
                    self.grammar.add(
                        result,
                        Nonterminal(reversed_nonterminal),
                        Terminal("("),
                        Nonterminal(argument_nonterminal(protocol)),
                        Terminal(")"),
                    )

        # typeshed renders heapq.heappop as list[T] -> T, but signature help
        # leaves T unresolved.  Unlike the iterable helpers above, heappop
        # mutates its input, so correlate the result only with an actual
        # concrete list element type already represented in this grammar.
        # Negative flow refinements on that list are accepted by
        # concrete_heap_list_element_type; arbitrary iterable/list-like
        # protocols are deliberately not.
        if self.options.max_call_arity >= 1:
            expression_types = {
                production.lhs[2:]
                for production in self.grammar.productions
                if production.lhs.startswith("E:")
            }
            # Covered library artifacts intentionally remove their exports
            # from self.callables because their ordinary call productions are
            # already cached.  Their exact lexical representatives remain,
            # however; recover just heappop here so this correlation pass is
            # artifact-aware without regenerating every cached call family.
            heappop_callables = dict(self.callables)
            cached_heappop_types: set[str] = set()
            for callable_type, expression in self.representatives.items():
                if (
                    expression.rsplit(".", 1)[-1] == "heappop"
                    and is_callable_type(callable_type)
                ):
                    if callable_type not in heappop_callables:
                        heappop_callables[callable_type] = expression
                        cached_heappop_types.add(callable_type)
            for callable_type, expression in sorted(heappop_callables.items()):
                if expression.rsplit(".", 1)[-1] != "heappop":
                    continue
                callable_nonterminal = self.expression_nonterminal(callable_type)
                signatures = (
                    tuple(signatures_from_detail(callable_type))
                    if callable_type in cached_heappop_types
                    else self.signatures_for(callable_type, expression)
                )
                if not signatures:
                    signatures = self.signatures_for(callable_type, expression)
                for signature in signatures:
                    result_variable = normalize_type(signature.return_type)
                    if TYPE_VARIABLE.fullmatch(result_variable) is None:
                        continue
                    layouts = argument_layouts(
                        signature,
                        max_arity=1,
                        max_layouts=self.options.max_layouts_per_signature,
                    )
                    for layout in layouts:
                        if len(layout.positional) != 1 or layout.keywords:
                            continue
                        expected = normalize_type(layout.positional[0])
                        expected_base, expected_arguments = generic_parts(expected)
                        if (
                            expected_base.split(".")[-1] != "list"
                            or len(expected_arguments) != 1
                            or normalize_type(expected_arguments[0])
                            != result_variable
                        ):
                            continue
                        for actual in sorted(expression_types):
                            element = concrete_heap_list_element_type(actual)
                            if (
                                element is None
                                or (
                                    element not in {"Any", "Unknown"}
                                    and not is_assignable(
                                        element, "SupportsRichComparisonT"
                                    )
                                )
                                or not is_assignable(actual, expected)
                            ):
                                continue
                            self.grammar.add(
                                self.call_result_nonterminal(
                                    element,
                                    trusted_dynamic_output=True,
                                ),
                                Nonterminal(callable_nonterminal),
                                Terminal("("),
                                Nonterminal(type_nonterminal(actual)),
                                Terminal(")"),
                            )

    def add_typed_unary_operations(self) -> None:
        expression_types = {
            production.lhs[2:]
            for production in self.grammar.productions
            if production.lhs.startswith("E:")
        }
        for actual in expression_types:
            supports_sign, supports_invert = numeric_unary_kinds(actual)
            if not supports_sign:
                continue
            operand_nonterminal = self.expression_nonterminal(actual)
            result_nonterminal = self.expression_nonterminal(
                numeric_unary_result(actual)
            )
            for operator in ("+", "-", "~"):
                if operator == "~" and not supports_invert:
                    continue
                self.grammar.add(
                    result_nonterminal,
                    Terminal(operator),
                    Nonterminal(operand_nonterminal),
                )

    def current_expression_types(self) -> set[str]:
        return {
            production.lhs[2:]
            for production in self.grammar.productions
            if production.lhs.startswith("E:")
        }

    def subscript_receiver_representative(
        self,
        receiver_type: str,
        expected_index: str,
    ) -> str | None:
        """Return one bounded, statically typed index expression for probing.

        Member completion needs a concrete spelling even though the resulting
        grammar row remains type-indexed.  Prefer an exact index type, then a
        shortest conservative assignability witness.  Dynamic witnesses are
        deliberately excluded: using one to reopen a concrete result family
        would make its provenance depend on gradual typing rather than the
        typed subscript row being closed here.
        """

        receiver = self.representatives.get(normalize_type(receiver_type))
        if receiver is None:
            return None
        expected = normalize_type(expected_index)
        candidates: list[
            tuple[int, int, tuple[str, ...], str]
        ] = []
        for actual, expression in self.representatives.items():
            normalized_actual = normalize_type(actual)
            if normalized_actual in {"Any", "Unknown"} or not is_assignable(
                normalized_actual, expected
            ):
                continue
            tokens = canonical_expression_tokens(expression)
            if tokens is None:
                continue
            candidates.append(
                (
                    normalized_actual != expected,
                    len(tokens),
                    tokens,
                    expression,
                )
            )
        if not candidates:
            return None
        index = min(candidates)[-1]
        expression = f"{receiver}[{index}]"
        try:
            parsed = ast.parse(expression, mode="eval").body
        except SyntaxError:
            parsed = None
        if not isinstance(parsed, ast.Subscript):
            expression = f"({receiver})[{index}]"
        tokens = canonical_expression_tokens(expression)
        if tokens is None:
            return None
        expression_budget = self.options.max_tokens - (
            2 if self.required_assignment is not None else 0
        )
        # A closure is useful only if at least one following ``.member`` can
        # still fit the grammar's expression budget.
        return expression if len(tokens) + 2 <= expression_budget else None

    def queue_typed_subscript_receiver(
        self,
        receiver_type: str,
        expected_index: str,
        result_type: str,
    ) -> None:
        """Seed one concrete indexed result for a single member/call closure."""

        result_type = normalize_type(result_type)
        if result_type in {"Any", "Unknown", "Never", "NoReturn", "None"}:
            return
        expression = self.subscript_receiver_representative(
            receiver_type, expected_index
        )
        if expression is None:
            return
        self.representatives.setdefault(result_type, expression)
        if is_callable_type(result_type):
            self.callables.setdefault(result_type, expression)
        else:
            self.queue_receiver(result_type, expression, 0)

    def add_typed_subscript_receiver_closure(self) -> None:
        """Expand indexed result members once, under the ordinary query caps.

        ``add_members`` may follow ordinary attributes up to ``member_depth``,
        but this method is invoked only once and does not rerun subscript
        synthesis.  Repeating ``add_calls`` is safe because it consumes only
        callable identities not processed by the first call pass.
        """

        self.add_members()
        self.add_calls()

    def add_typed_subscripts(self) -> None:
        """Add index and ordinary-slice trailers justified by container types."""

        expression_types = self.current_expression_types()
        slice_receivers = {
            receiver_type: result_type
            for receiver_type in expression_types
            if (result_type := sliced_access_type(receiver_type)) is not None
        }
        if slice_receivers:
            self.add_slice_syntax()

        for receiver_type in sorted(expression_types):
            receiver = Nonterminal(type_nonterminal(receiver_type))
            indexed = indexed_access_types(receiver_type)
            if indexed is not None:
                expected, result_type = indexed
                self.expected_types.add(expected)
                normalized_result = normalize_type(result_type)
                if normalized_result in {"Any", "Unknown"}:
                    result_nonterminal = (
                        self.contextual_dynamic_output_nonterminal(
                            normalized_result,
                            ("subscript", receiver_type, expected),
                        )
                    )
                else:
                    result_nonterminal = self.expression_nonterminal(
                        normalized_result
                    )
                before = len(self.grammar.productions)
                self.grammar.add(
                    result_nonterminal,
                    receiver,
                    Terminal("["),
                    Nonterminal(argument_nonterminal(expected)),
                    Terminal("]"),
                )
                self.stats.subscript_productions += (
                    len(self.grammar.productions) - before
                )
                if normalized_result not in {"Any", "Unknown"}:
                    self.queue_typed_subscript_receiver(
                        receiver_type,
                        expected,
                        normalized_result,
                    )

            result_type = slice_receivers.get(receiver_type)
            if result_type is None:
                continue
            before = len(self.grammar.productions)
            self.grammar.add(
                self.expression_nonterminal(result_type),
                receiver,
                Terminal("["),
                Nonterminal(SLICE_NONTERMINAL),
                Terminal("]"),
            )
            self.stats.slice_productions += (
                len(self.grammar.productions) - before
            )

    def add_slice_syntax(self) -> None:
        """Materialize the twelve ordinary one- and two-colon slice forms."""

        if any(
            production.lhs == SLICE_NONTERMINAL
            for production in self.grammar.productions
        ):
            return
        # This is deliberately not ``A:SupportsIndex``.  That broad argument
        # family includes E:Any/E:Unknown, whose contextual postfix results
        # could then recurse through slice bounds and evade the explicit
        # dynamic-composition depth.  ``finish`` links only non-gradual
        # SupportsIndex expressions and exact lexical gradual names here.
        bound = Nonterminal(SLICE_BOUND_NONTERMINAL)
        for include_step in (False, True):
            for lower_present, upper_present, step_present in itertools.product(
                (False, True), repeat=3
            ):
                if not include_step and step_present:
                    continue
                rhs: list[Symbol] = []
                if lower_present:
                    rhs.append(bound)
                rhs.append(Terminal(":"))
                if upper_present:
                    rhs.append(bound)
                if include_step:
                    rhs.append(Terminal(":"))
                    if step_present:
                        rhs.append(bound)
                before = len(self.grammar.productions)
                self.grammar.add(SLICE_NONTERMINAL, *rhs)
                self.stats.slice_productions += (
                    len(self.grammar.productions) - before
                )

    def add_typed_binary_operations(self) -> None:
        """Add the common arithmetic rows supported by concrete built-in types."""

        expression_types = self.current_expression_types()
        numeric_types: list[str] = []
        sequence_types: list[str] = []
        index_types: list[str] = []
        for actual in sorted(expression_types):
            concrete = strip_negative_flow_refinements(actual)
            if concrete is None or len(split_union(concrete)) != 1:
                continue
            base, _arguments = generic_parts(concrete)
            base = base.split(".")[-1]
            if base in BUILTIN_NUMERIC_RANK:
                numeric_types.append(actual)
                if is_assignable(concrete, "SupportsIndex"):
                    index_types.append(actual)
            if base in ARITHMETIC_SEQUENCE_BASES:
                sequence_types.append(actual)

        candidates: set[tuple[str, str, str]] = set()
        for left, right in itertools.product(numeric_types, repeat=2):
            for operator in BINARY_OPERATOR_SYMBOLS.values():
                candidates.add((left, operator, right))
        for left, right in itertools.product(sequence_types, repeat=2):
            candidates.add((left, "+", right))
        for sequence_type in sequence_types:
            for index_type in index_types:
                candidates.add((sequence_type, "*", index_type))
                candidates.add((index_type, "*", sequence_type))

        for left, operator, right in sorted(candidates):
            result_type = arithmetic_binary_result(left, operator, right)
            if result_type is None:
                continue
            before = len(self.grammar.productions)
            self.grammar.add(
                self.expression_nonterminal(result_type),
                Nonterminal(type_nonterminal(left)),
                Terminal(operator),
                Nonterminal(type_nonterminal(right)),
            )
            self.stats.binary_operation_productions += (
                len(self.grammar.productions) - before
            )

    def add_dynamic_operations(self) -> None:
        if not any(
            production.lhs == DYNAMIC_NONTERMINAL
            for production in self.grammar.productions
        ):
            return
        representatives = sorted(
            set().union(*self.dynamic_representatives.values())
            if self.dynamic_representatives
            else set()
        )
        expression_budget = self.options.max_tokens - (
            2 if self.required_assignment is not None else 0
        )
        dynamic_spines: list[tuple[str, str, tuple[str, ...], str]] = []
        if representatives:
            self.expected_types.add("object")
            self.add_slice_syntax()
            self.add_dynamic_composition_operands()
        for dynamic_type, type_representatives in sorted(
            self.dynamic_representatives.items()
        ):
            if dynamic_type not in {"Any", "Unknown"}:
                continue
            for representative in sorted(type_representatives):
                dynamic_spines.extend(
                    self.add_dynamic_postfix_compositions(
                        dynamic_type,
                        representative,
                        expression_budget,
                    )
                )
        self.add_dynamic_binary_operations(dynamic_spines)
        for representative in representatives:
            # Dynamic member rows come from an exact completion query in
            # add_members.  For the two syntactic operations that do not have
            # completion/signature structure, retain only one-step words that
            # ty accepts for this exact contextual expression.  Recursive
            # DYNAMIC -> DYNAMIC op/call/member rules created an unbounded
            # cross-product across unrelated Any/Unknown values.
            for candidate in (
                *(f"{operator}{representative}" for operator in ("+", "-", "~")),
                f"{representative}()",
            ):
                tokens = canonical_expression_tokens(candidate)
                if (
                    tokens is None
                    or len(tokens) > expression_budget
                    or not self.probe.accepts_expression(candidate)
                ):
                    continue
                self.grammar.add(
                    DYNAMIC_NONTERMINAL,
                    *(Terminal(token) for token in tokens),
                )
            if self.required_assignment is not None:
                continue
            receiver_tokens = canonical_expression_tokens(representative)
            if receiver_tokens is None:
                continue
            # ty may return an incomplete, empty completion page for an
            # Any/Unknown receiver.  Its gradual type nevertheless accepts
            # these bounded core member calls.  Anchor every row to this exact
            # contextual receiver spelling; never reintroduce the old shared
            # DYNAMIC.member/call recursion.
            for member in sorted(CORE_MEMBERS):
                for arity in range(self.options.max_call_arity + 1):
                    minimum_tokens = len(receiver_tokens) + 4 + max(
                        0, 2 * arity - 1
                    )
                    if minimum_tokens > expression_budget:
                        continue
                    rhs: list[Symbol] = [
                        *(Terminal(token) for token in receiver_tokens),
                        Terminal("."),
                        Terminal(member),
                        Terminal("("),
                    ]
                    for index in range(arity):
                        if index:
                            rhs.append(Terminal(","))
                        rhs.append(
                            Nonterminal(argument_nonterminal("object"))
                        )
                    rhs.append(Terminal(")"))
                    self.expected_types.add("object")
                    self.grammar.add(
                        self.expression_nonterminal("Unknown"), *rhs
                    )

    def add_dynamic_composition_operands(self) -> None:
        """Materialize a small finite operand language with no gradual values.

        The language contains canonical literals and statically typed lexical
        names.  Keeping it finite lets an origin-specific gradual spine
        compose with an ordinary atom without introducing ``Any op Any`` or
        recursively importing the complete expression grammar.
        """

        atom_type = DYNAMIC_COMPOSITION_ATOM_NONTERMINAL.removeprefix("E:")
        operand_type = DYNAMIC_BINARY_OPERAND_NONTERMINAL.removeprefix("E:")
        self.internal_expression_types.update({atom_type, operand_type})
        canonical_atoms: set[tuple[Symbol, ...]] = {
            (Terminal(token),)
            for token in (
                "None",
                "False",
                "True",
                "0",
                "0.0",
                "0j",
                '""',
                'b""',
            )
        }
        canonical_atoms.update(
            production.rhs
            for production in self.grammar.productions
            if production.lhs.startswith("E:")
            and production.lhs != DYNAMIC_NONTERMINAL
            and production.lhs[2:] not in {
                "Any",
                "Unknown",
                *self.internal_expression_types,
            }
            and len(production.rhs) == 1
            and isinstance(production.rhs[0], Terminal)
            and production.rhs[0].value in self.source_ids
        )
        ordered_atoms = sorted(
            canonical_atoms,
            key=lambda rhs: tuple(symbol.value for symbol in rhs),
        )
        for rhs in ordered_atoms:
            self.grammar.add(DYNAMIC_COMPOSITION_ATOM_NONTERMINAL, *rhs)
            self.grammar.add(DYNAMIC_BINARY_OPERAND_NONTERMINAL, *rhs)
            if rhs in {
                (Terminal("0"),),
                (Terminal("0.0"),),
                (Terminal("0j"),),
            }:
                for operator in ("+", "-"):
                    unary_rhs = (Terminal(operator), *rhs)
                    self.grammar.add(
                        DYNAMIC_COMPOSITION_ATOM_NONTERMINAL, *unary_rhs
                    )
                    self.grammar.add(
                        DYNAMIC_BINARY_OPERAND_NONTERMINAL, *unary_rhs
                    )

    def add_dynamic_postfix_compositions(
        self,
        dynamic_type: str,
        representative: str,
        expression_budget: int,
    ) -> list[tuple[str, str, tuple[str, ...], str]]:
        """Unroll origin-specific index/slice spines to the configured depth."""

        tokens = canonical_expression_tokens(representative)
        if tokens is None or len(tokens) > expression_budget:
            return []
        base_identity = ("dynamic-origin", dynamic_type, representative)
        base_type = (
            "__contextual_dynamic_value_"
            f"{stable_digest(repr(base_identity), 16)}__"
        )
        self.internal_expression_types.add(base_type)
        base = self.expression_nonterminal(base_type)
        self.grammar.add(base, *(Terminal(token) for token in tokens))
        result: list[tuple[str, str, tuple[str, ...], str]] = [
            (dynamic_type, representative, (), base)
        ]
        frontier: list[tuple[tuple[str, ...], str]] = [((), base)]
        token_depth = max(0, (expression_budget - len(tokens)) // 3)
        maximum_depth = max(
            0,
            min(
                self.options.max_dynamic_composition_depth,
                token_depth,
            ),
        )
        for _depth in range(1, maximum_depth + 1):
            next_frontier: list[tuple[tuple[str, ...], str]] = []
            for path, parent in frontier:
                for trailer, body in (
                    (
                        "index",
                        (
                            Terminal("["),
                            Nonterminal(DYNAMIC_BINARY_OPERAND_NONTERMINAL),
                            Terminal("]"),
                        ),
                    ),
                    (
                        "slice",
                        (
                            Terminal("["),
                            Nonterminal(SLICE_NONTERMINAL),
                            Terminal("]"),
                        ),
                    ),
                ):
                    child_path = (*path, trailer)
                    identity = (
                        "dynamic-postfix",
                        dynamic_type,
                        representative,
                        child_path,
                    )
                    child_type = (
                        "__contextual_dynamic_value_"
                        f"{stable_digest(repr(identity), 16)}__"
                    )
                    self.internal_expression_types.add(child_type)
                    child = self.expression_nonterminal(child_type)
                    before = len(self.grammar.productions)
                    self.grammar.add(child, Nonterminal(parent), *body)
                    family = self.contextual_dynamic_output_nonterminal(
                        dynamic_type, identity
                    )
                    self.grammar.add(family, Nonterminal(child))
                    added = len(self.grammar.productions) - before
                    if trailer == "index":
                        self.stats.subscript_productions += added
                    else:
                        self.stats.slice_productions += added
                    next_frontier.append((child_path, child))
                    result.append(
                        (
                            dynamic_type,
                            representative,
                            child_path,
                            child,
                        )
                    )
            frontier = next_frontier
        return result

    def add_dynamic_binary_operations(
        self,
        spines: Sequence[tuple[str, str, tuple[str, ...], str]],
    ) -> None:
        """Add one arithmetic step tied to each gradual producer and spine.

        Under the pinned checker, ``Any`` and ``Unknown`` support the common
        arithmetic operators on either side.  Each result retains its exact
        visible origin and postfix path while the other operand comes from the
        finite non-gradual language above.  Internal expression types are never
        rooted or exported as assignability facts; assignment outputs instead
        pass through their producer-specific contextual family.
        """

        for dynamic_type, representative, path, value in spines:
            for operator in BINARY_OPERATOR_SYMBOLS.values():
                for dynamic_on_left in (True, False):
                    identity = (
                        "dynamic-binary",
                        dynamic_type,
                        representative,
                        path,
                        operator,
                        dynamic_on_left,
                    )
                    result_type = (
                        "__contextual_dynamic_binary_"
                        f"{stable_digest(repr(identity), 16)}__"
                    )
                    self.internal_expression_types.add(result_type)
                    result = self.expression_nonterminal(result_type)
                    left = (
                        value
                        if dynamic_on_left
                        else DYNAMIC_BINARY_OPERAND_NONTERMINAL
                    )
                    right = (
                        DYNAMIC_BINARY_OPERAND_NONTERMINAL
                        if dynamic_on_left
                        else value
                    )
                    before = len(self.grammar.productions)
                    self.grammar.add(
                        result,
                        Nonterminal(left),
                        Terminal(operator),
                        Nonterminal(right),
                    )
                    family = self.contextual_dynamic_output_nonterminal(
                        dynamic_type, identity
                    )
                    self.grammar.add(family, Nonterminal(result))
                    self.stats.binary_operation_productions += (
                        len(self.grammar.productions) - before
                    )

    def add_redundant_grouping(self) -> None:
        """Permit Python's type-preserving parenthesized expression form."""

        dynamic_aliases = {
            production.lhs
            for production in self.grammar.productions
            if production.rhs == (Nonterminal(DYNAMIC_NONTERMINAL),)
        }
        expression_nonterminals = {
            production.lhs
            for production in self.grammar.productions
            if production.lhs.startswith("E:")
            and production.lhs not in dynamic_aliases
            and production.lhs[2:] not in self.internal_expression_types
        }
        for nonterminal in expression_nonterminals:
            self.grammar.add(
                nonterminal,
                Terminal("("),
                Nonterminal(nonterminal),
                Terminal(")"),
            )
        # Shared E:Any/E:Unknown grouping remains disabled: it would merge all
        # gradual producers before later composition.  Group the exact
        # producer families themselves instead, preserving both provenance and
        # the surface rule `(e)` for contextual binary/subscript/call results.
        exact_dynamic_families = {
            *self.contextual_dynamic_output_families,
            DYNAMIC_NONTERMINAL,
            TRUSTED_DYNAMIC_CALL_NONTERMINAL,
        }
        productive_lhs = {
            production.lhs for production in self.grammar.productions
        }
        for nonterminal in sorted(exact_dynamic_families & productive_lhs):
            self.grammar.add(
                nonterminal,
                Terminal("("),
                Nonterminal(nonterminal),
                Terminal(")"),
            )

    def shortest_terminal_words(self) -> dict[str, tuple[str, ...]]:
        """Find a deterministic shortest terminal witness for each nonterminal."""

        best: dict[str, tuple[str, ...]] = {}
        ordered = sorted(
            self.grammar.productions,
            key=lambda production: (
                production.lhs,
                tuple(
                    (0 if isinstance(symbol, Terminal) else 1, symbol.value)
                    for symbol in production.rhs
                ),
            ),
        )
        changed = True
        while changed:
            changed = False
            for production in ordered:
                tokens: list[str] = []
                for symbol in production.rhs:
                    if isinstance(symbol, Terminal):
                        tokens.append(symbol.value)
                        continue
                    child = best.get(symbol.value)
                    if child is None:
                        break
                    tokens.extend(child)
                else:
                    candidate = tuple(tokens)
                    current = best.get(production.lhs)
                    if current is None or (len(candidate), candidate) < (
                        len(current),
                        current,
                    ):
                        best[production.lhs] = candidate
                        changed = True
        return best

    def refine_output_producer_roots(
        self, shortest: Mapping[str, tuple[str, ...]]
    ) -> None:
        """Validate independently rooted expression producers in context.

        An accepted representative of ``E:T`` establishes only that one value
        with type T satisfies the uses after an output assignment.  Rooting the
        whole nonterminal also admits producers whose more specific result
        violates that continuation.  Expand the E/P unit frontier and validate
        one shortest witness for each producer family instead.

        A local error in that witness is not evidence that every word in the
        family is bad, so retain the family conservatively.  Families beyond
        the query budget are retained for the same recall-preserving reason.
        Only a locally valid witness with downstream errors is rejected.
        Exact dynamic roots never pass through ``E:`` and remain untouched.
        """

        if self.required_assignment is None:
            return
        started = time.perf_counter()
        by_lhs: dict[str, list[Production]] = defaultdict(list)
        for production in self.grammar.productions:
            by_lhs[production.lhs].append(production)

        typed_roots: set[Production] = set()
        frontier_names: set[str] = set()
        for production in by_lhs[self.grammar.start]:
            if (
                len(production.rhs) == 3
                and production.rhs[:2]
                == (
                    Terminal(self.required_assignment),
                    Terminal("="),
                )
                and isinstance(production.rhs[2], Nonterminal)
                and production.rhs[2].value.startswith("E:")
            ):
                typed_roots.add(production)
                frontier_names.add(production.rhs[2].value)
        if not typed_roots:
            return

        # Unit productions between precedence layers carry no independently
        # testable syntax.  Traverse them until a concrete producer RHS is
        # reached, while breaking the harmless cycles introduced by aliases.
        producer_rhs: set[tuple[Symbol, ...]] = set()
        pending = list(frontier_names)
        visited: set[str] = set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            for production in by_lhs.get(name, ()):
                if (
                    len(production.rhs) == 1
                    and isinstance(production.rhs[0], Nonterminal)
                    and production.rhs[0].value.startswith(
                        ("E:", "S:", "M:", "F:", "W:", "P:")
                    )
                ):
                    pending.append(production.rhs[0].value)
                else:
                    producer_rhs.add(production.rhs)

        def rhs_sort_key(
            rhs: tuple[Symbol, ...],
        ) -> tuple[tuple[int, str], ...]:
            return tuple(
                (0 if isinstance(symbol, Terminal) else 1, symbol.value)
                for symbol in rhs
            )

        retained: set[tuple[Symbol, ...]] = set()
        ordered = sorted(producer_rhs, key=rhs_sort_key)
        self.stats.output_producer_families += len(ordered)
        for index, rhs in enumerate(ordered):
            if index >= self.options.max_output_producers:
                retained.add(rhs)
                self.stats.output_producers_unchecked += 1
                continue
            tokens: list[str] = []
            for symbol in rhs:
                if isinstance(symbol, Terminal):
                    tokens.append(symbol.value)
                    continue
                child = shortest.get(symbol.value)
                if child is None:
                    self.stats.output_producers_unchecked += 1
                    retained.add(rhs)
                    break
                tokens.extend(child)
            else:
                expression = render_tokens(tokens, self.source_ids)
                local, downstream = (
                    self.probe.assignment_diagnostic_partition(expression)
                )
                self.stats.output_producers_checked += 1
                if local:
                    self.stats.output_producers_local_fallback += 1
                    retained.add(rhs)
                elif downstream:
                    self.stats.output_producers_rejected += 1
                else:
                    retained.add(rhs)

        self.grammar.productions.difference_update(typed_roots)
        for rhs in retained:
            self.grammar.add(
                self.grammar.start,
                Terminal(self.required_assignment),
                Terminal("="),
                *rhs,
            )
        self.stats.output_producer_validation_seconds += (
            time.perf_counter() - started
        )

    def add_slice_bound_links(
        self,
        public_expression_types: Iterable[str],
    ) -> None:
        """Populate slice bounds without importing gradual expression graphs.

        Ordinary statically typed SupportsIndex expressions remain fully
        compositional.  A source-visible Any/Unknown name is admitted only as
        its exact lexical token; linking E:Any or E:Unknown would also import
        every derived contextual family beneath that shared type.
        """

        if not any(
            production.lhs == SLICE_NONTERMINAL
            for production in self.grammar.productions
        ):
            return
        for actual in sorted(public_expression_types):
            if actual in {"Divergent", "Never", "NoReturn"}:
                continue
            if has_gradual_value_branch(actual):
                continue
            self.stats.assignability_pairs_checked += 1
            if is_assignable(actual, "SupportsIndex"):
                self.grammar.add(
                    SLICE_BOUND_NONTERMINAL,
                    Nonterminal(type_nonterminal(actual)),
                )

        lexical_gradual_representatives: set[str] = set()
        for representatives in self.dynamic_scope_representatives.values():
            lexical_gradual_representatives.update(representatives)
        for representative in sorted(lexical_gradual_representatives):
            tokens = canonical_expression_tokens(representative)
            if (
                tokens is None
                or len(tokens) != 1
                or tokens[0] not in self.source_ids
            ):
                continue
            self.grammar.add(
                SLICE_BOUND_NONTERMINAL,
                Terminal(tokens[0]),
            )

    def finish(self) -> tuple[Grammar, BuildStats]:
        # Activate only the precomputed compatibility rows whose A: slot is
        # actually referenced by a bounded library or contextual call.  This
        # avoids copying hundreds of thousands of dead unit rules from large
        # artifacts while preserving every bounded derivation exactly.
        for artifact in self.active_library_artifacts:
            if not artifact.local_assignability_complete:
                continue
            for expected in self.expected_types & artifact.local_expected_types:
                links = artifact.local_links_by_expected.get(expected, ())
                before = len(self.grammar.productions)
                self.grammar.productions.update(links)
                self.stats.library_productions += (
                    len(self.grammar.productions) - before
                )
        self.grammar = enforce_expression_precedence(self.grammar)
        expression_types = {
            production.lhs[2:]
            for production in self.grammar.productions
            if production.lhs.startswith("E:")
        }
        public_expression_types = (
            expression_types - self.internal_expression_types
        )
        self.add_slice_bound_links(public_expression_types)
        cached_actuals_by_expected: dict[str, set[str]] = defaultdict(set)
        for artifact in self.active_library_artifacts:
            if not artifact.local_assignability_complete:
                continue
            for expected in artifact.local_expected_types:
                cached_actuals_by_expected[expected].update(
                    artifact.local_actual_types
                )
        for expected in sorted(self.expected_types):
            argument = argument_nonterminal(expected)
            cached_actuals = cached_actuals_by_expected.get(expected, set())
            for actual in sorted(public_expression_types):
                if actual in cached_actuals:
                    self.stats.assignability_pairs_cached += 1
                    continue
                self.stats.assignability_pairs_checked += 1
                if is_assignable(actual, expected):
                    self.grammar.add(argument, Nonterminal(type_nonterminal(actual)))
        if self.required_assignment is not None:
            shortest = self.shortest_terminal_words()
            for actual in sorted(public_expression_types):
                if actual in {
                    "Any",
                    "Unknown",
                    DYNAMIC_NONTERMINAL.removeprefix("E:"),
                }:
                    # A single accepted witness cannot justify rooting every
                    # expression of a broad dynamic type.  Signature-backed
                    # dynamic calls are rooted separately below.
                    continue
                if actual in self.representatives:
                    continue
                tokens = shortest.get(type_nonterminal(actual))
                if not tokens:
                    continue
                representative = " ".join(tokens)
                if self.probe.accepts_expression(representative):
                    self.representatives[actual] = representative
                    self.stats.derived_representatives += 1
                else:
                    self.stats.invalid_representatives += 1
        for actual in public_expression_types:
            expression = type_nonterminal(actual)
            if self.required_assignment is None:
                self.grammar.add(self.grammar.start, Nonterminal(expression))
                self.grammar.add(
                    self.grammar.start,
                    Terminal(FRESH_TOKEN),
                    Terminal("="),
                    Nonterminal(expression),
                )
            else:
                if actual in {
                    "Any",
                    "Unknown",
                    DYNAMIC_NONTERMINAL.removeprefix("E:"),
                }:
                    # Preserve legacy/source-visible dynamic assignments as
                    # exact, independently context-validated words.  Rooting
                    # E:Unknown after one witness passed would also admit all
                    # rejected Unknown expressions (and trusted call results).
                    for representative in sorted(
                        self.dynamic_representatives.get(actual, ())
                    ):
                        for candidate in (
                            representative,
                            *(
                                f"{operator}{representative}"
                                for operator in ("+", "-", "~")
                            ),
                        ):
                            self.stats.assignment_types_checked += 1
                            if not self.probe.accepts_assignment(candidate):
                                self.stats.assignment_types_rejected += 1
                                continue
                            tokens = canonical_expression_tokens(candidate)
                            if tokens is None:
                                self.stats.assignment_types_rejected += 1
                                continue
                            self.grammar.add(
                                self.grammar.start,
                                Terminal(self.required_assignment),
                                Terminal("="),
                                *(Terminal(token) for token in tokens),
                            )
                    continue
                assignment_representative = self.representatives.get(actual)
                if assignment_representative is not None:
                    self.stats.assignment_types_checked += 1
                    if not self.probe.accepts_assignment(
                        assignment_representative
                    ):
                        self.stats.assignment_types_rejected += 1
                        continue
                self.grammar.add(
                    self.grammar.start,
                    Terminal(self.required_assignment),
                    Terminal("="),
                    Nonterminal(expression),
                )
        if self.required_assignment is not None:
            for call_family in sorted(
                self.contextual_dynamic_output_families
            ):
                tokens = shortest.get(call_family)
                if not tokens:
                    continue
                self.stats.assignment_types_checked += 1
                if not self.probe.accepts_assignment(" ".join(tokens)):
                    self.stats.assignment_types_rejected += 1
                    continue
                self.grammar.add(
                    self.grammar.start,
                    Terminal(self.required_assignment),
                    Terminal("="),
                    Nonterminal(call_family),
                )
        if (
            self.required_assignment is not None
            and any(
                production.lhs == TRUSTED_DYNAMIC_CALL_NONTERMINAL
                for production in self.grammar.productions
            )
        ):
            self.grammar.add(
                self.grammar.start,
                Terminal(self.required_assignment),
                Terminal("="),
                Nonterminal(TRUSTED_DYNAMIC_CALL_NONTERMINAL),
            )
        if self.required_assignment is not None:
            self.refine_output_producer_roots(shortest)
        self.grammar = prune_grammar(self.grammar)
        self.stats.expression_types = len(public_expression_types)
        return self.grammar, self.stats

    def build(self) -> tuple[Grammar, BuildStats]:
        self.add_literals()
        self.add_scope()
        self.add_library_artifacts()
        self.add_members()
        self.add_calls()
        self.add_grounded_generic_calls()
        self.add_typed_subscripts()
        self.add_typed_subscript_receiver_closure()
        self.add_typed_binary_operations()
        self.add_typed_unary_operations()
        self.add_dynamic_operations()
        self.add_redundant_grouping()
        return self.finish()


def enforce_expression_precedence(grammar: Grammar) -> Grammar:
    """Lower typed expressions into Python's postfix and arithmetic layers.

    The semantic grammar indexes expressions by type, but a single ``E:T``
    symbol is not quite enough to encode Python's precedence.  In particular,
    a rule such as ``E:member -> E:int '.' member`` can otherwise treat
    ``~ 0`` as the receiver in ``~ 0 . member``.  Python parses that word as
    ``~(0.member)`` because attribute access binds more tightly than unary
    operators, so the derivation can disagree with the expression ty checks.

    ``P:T`` denotes primary/postfix expressions, ``W:T`` power, ``F:T``
    prefix factor, ``M:T`` multiplicative term, and ``S:T`` additive sum.
    ``E:T`` is the unrestricted expression view.  Typed binary rows are moved
    to their Python layer: sums and terms associate left, while power takes a
    primary on the left and a factor on the right.  Prefix unary rules move to
    F.  Every remaining E producer moves to P, and a leading expression child
    of a primary producer becomes P as well.  This covers attributes, calls,
    subscripts, and type-preserving unit aliases without changing argument
    slots.  Parentheses begin with a terminal, so their inner E remains
    unrestricted and explicitly grouped expressions become safe receivers.

    Non-expression call-result symbols (notably the trusted dynamic-call
    root) retain their LHS, but their callable receiver is changed to P too.
    This keeps the transformation valid for both live and cached-library
    production shapes.
    """

    rewritten: set[Production] = set()
    expression_aliases: set[str] = set()
    unary_operators = {"+", "-", "~"}
    sum_operators = {"+", "-"}
    term_operators = {"*", "/", "//", "%"}
    power_operators = {"**"}
    for production in grammar.productions:
        rhs = production.rhs
        if (
            (
                production.lhs == DYNAMIC_BINARY_OPERAND_NONTERMINAL
                or production.lhs.startswith(
                    "E:__contextual_dynamic_binary_value_"
                )
            )
            and len(rhs) == 1
            and isinstance(rhs[0], Nonterminal)
            and rhs[0].value.startswith("E:")
        ):
            # This internal symbol is a union of complete expressions.  Keep
            # the union at every layer; lowering it as an ordinary primary
            # alias would permit only postfix operands in binary rows.
            for layer in ("S", "M", "F", "W", "P"):
                rewritten.add(
                    Production(
                        expression_layer_nonterminal(
                            production.lhs, layer
                        ),
                        (
                            Nonterminal(
                                expression_layer_nonterminal(
                                    rhs[0].value, layer
                                )
                            ),
                        ),
                    )
                )
            expression_aliases.add(production.lhs)
            continue
        binary_operator = (
            rhs[1].value
            if (
                production.lhs.startswith("E:")
                and len(rhs) == 3
                and isinstance(rhs[0], Nonterminal)
                and rhs[0].value.startswith("E:")
                and isinstance(rhs[1], Terminal)
                and rhs[1].value
                in sum_operators | term_operators | power_operators
                and isinstance(rhs[2], Nonterminal)
                and rhs[2].value.startswith("E:")
            )
            else None
        )
        if binary_operator is not None:
            left = rhs[0].value
            right = rhs[2].value
            if binary_operator in sum_operators:
                lhs = sum_nonterminal(production.lhs)
                rewritten_rhs = (
                    Nonterminal(sum_nonterminal(left)),
                    rhs[1],
                    Nonterminal(term_nonterminal(right)),
                )
            elif binary_operator in term_operators:
                lhs = term_nonterminal(production.lhs)
                rewritten_rhs = (
                    Nonterminal(term_nonterminal(left)),
                    rhs[1],
                    Nonterminal(factor_nonterminal(right)),
                )
            else:
                lhs = power_nonterminal(production.lhs)
                rewritten_rhs = (
                    Nonterminal(postfix_nonterminal(left)),
                    rhs[1],
                    Nonterminal(factor_nonterminal(right)),
                )
            rewritten.add(Production(lhs, rewritten_rhs))
            expression_aliases.add(production.lhs)
            continue

        prefix_unary = (
            production.lhs.startswith("E:")
            and bool(rhs)
            and isinstance(rhs[0], Terminal)
            and rhs[0].value in unary_operators
        )
        if prefix_unary:
            unary_rhs: tuple[Symbol, ...] = rhs
            if (
                len(rhs) >= 2
                and isinstance(rhs[1], Nonterminal)
                and rhs[1].value.startswith("E:")
            ):
                unary_rhs = (
                    rhs[0],
                    Nonterminal(factor_nonterminal(rhs[1].value)),
                    *rhs[2:],
                )
            rewritten.add(
                Production(
                    factor_nonterminal(production.lhs), unary_rhs
                )
            )
            expression_aliases.add(production.lhs)
            continue

        primary_rhs: tuple[Symbol, ...] = rhs
        leading_expression_is_postfix = (
            production.lhs.startswith("E:")
            or (
                len(rhs) >= 2
                and isinstance(rhs[1], Terminal)
                and rhs[1].value in {".", "(", "["}
            )
        )
        if (
            leading_expression_is_postfix
            and rhs
            and isinstance(rhs[0], Nonterminal)
            and rhs[0].value.startswith("E:")
        ):
            primary_rhs = (
                Nonterminal(postfix_nonterminal(rhs[0].value)),
                *rhs[1:],
            )

        if production.lhs.startswith("E:"):
            postfix_lhs = postfix_nonterminal(production.lhs)
            rewritten.add(Production(postfix_lhs, primary_rhs))
            expression_aliases.add(production.lhs)
        else:
            rewritten.add(Production(production.lhs, primary_rhs))

    for expression in expression_aliases:
        rewritten.add(
            Production(
                expression,
                (Nonterminal(sum_nonterminal(expression)),),
            )
        )
        rewritten.add(
            Production(
                sum_nonterminal(expression),
                (Nonterminal(term_nonterminal(expression)),),
            )
        )
        rewritten.add(
            Production(
                term_nonterminal(expression),
                (Nonterminal(factor_nonterminal(expression)),),
            )
        )
        rewritten.add(
            Production(
                factor_nonterminal(expression),
                (Nonterminal(power_nonterminal(expression)),),
            )
        )
        rewritten.add(
            Production(
                power_nonterminal(expression),
                (Nonterminal(postfix_nonterminal(expression)),),
            )
        )
    return Grammar(
        start=grammar.start,
        productions=rewritten,
        type_labels=dict(grammar.type_labels),
    )


def enforce_postfix_precedence(grammar: Grammar) -> Grammar:
    """Backward-compatible name for the full expression transformation."""

    return enforce_expression_precedence(grammar)


def prune_grammar(grammar: Grammar) -> Grammar:
    productions = set(grammar.productions)
    rows = list(productions)
    dependencies: list[set[str]] = [
        {
            symbol.value
            for symbol in production.rhs
            if isinstance(symbol, Nonterminal)
        }
        for production in rows
    ]
    waiting: dict[str, list[int]] = defaultdict(list)
    remaining = [len(items) for items in dependencies]
    for index, items in enumerate(dependencies):
        for dependency in items:
            waiting[dependency].append(index)
    generating: set[str] = set()
    queue: deque[str] = deque(
        dict.fromkeys(
            rows[index].lhs for index, count in enumerate(remaining) if count == 0
        )
    )
    generating.update(queue)
    while queue:
        completed = queue.popleft()
        for index in waiting.get(completed, ()):
            remaining[index] -= 1
            lhs = rows[index].lhs
            if remaining[index] == 0 and lhs not in generating:
                generating.add(lhs)
                queue.append(lhs)
    productions = {
        production
        for production in productions
        if production.lhs in generating
        and all(
            isinstance(symbol, Terminal) or symbol.value in generating
            for symbol in production.rhs
        )
    }
    by_lhs: dict[str, list[Production]] = defaultdict(list)
    for production in productions:
        by_lhs[production.lhs].append(production)
    reachable = {grammar.start}
    queue = deque((grammar.start,))
    while queue:
        lhs = queue.popleft()
        for production in by_lhs.get(lhs, ()):
            for symbol in production.rhs:
                if isinstance(symbol, Nonterminal) and symbol.value not in reachable:
                    reachable.add(symbol.value)
                    queue.append(symbol.value)
    return Grammar(
        start=grammar.start,
        productions={
            production for production in productions if production.lhs in reachable
        },
        type_labels=dict(grammar.type_labels),
    )


def to_cnf(grammar: Grammar) -> Grammar:
    """Return an epsilon-free, unit-free binary/terminal grammar."""

    productions = set(grammar.productions)
    nonterminals = set(grammar.nonterminals)
    unit_edges: dict[str, set[str]] = defaultdict(set)
    nonunit: dict[str, set[Production]] = defaultdict(set)
    for production in productions:
        if len(production.rhs) == 1 and isinstance(production.rhs[0], Nonterminal):
            unit_edges[production.lhs].add(production.rhs[0].value)
        else:
            nonunit[production.lhs].add(production)

    without_units: set[Production] = set()
    for lhs in nonterminals:
        closure = {lhs}
        queue = deque((lhs,))
        while queue:
            current = queue.popleft()
            for target in unit_edges.get(current, ()):
                if target not in closure:
                    closure.add(target)
                    queue.append(target)
        for target in closure:
            for production in nonunit.get(target, ()):
                without_units.add(Production(lhs, production.rhs))

    used_names = set(nonterminals)
    synthetic_ids = itertools.count()

    def synthetic_name(prefix: str) -> str:
        while True:
            name = f"#{prefix}:{next(synthetic_ids)}"
            if name not in used_names:
                used_names.add(name)
                return name

    def production_key(production: Production) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (
            production.lhs,
            tuple(
                (
                    "T" if isinstance(symbol, Terminal) else "N",
                    symbol.value,
                )
                for symbol in production.rhs
            ),
        )

    preterminals: dict[str, str] = {}
    lifted: set[Production] = set()

    def lift(symbol: Symbol) -> Nonterminal:
        if isinstance(symbol, Nonterminal):
            return symbol
        name = preterminals.get(symbol.value)
        if name is None:
            name = synthetic_name("T")
            preterminals[symbol.value] = name
        lifted.add(Production(name, (symbol,)))
        return Nonterminal(name)

    binarized: set[Production] = set()
    suffix_names: dict[tuple[str, ...], str] = {}

    def suffix_nonterminal(symbols: tuple[Nonterminal, ...]) -> Nonterminal:
        key = tuple(symbol.value for symbol in symbols)
        name = suffix_names.get(key)
        if name is None:
            name = synthetic_name("B")
            suffix_names[key] = name
            if len(symbols) == 2:
                binarized.add(Production(name, symbols))
            else:
                binarized.add(
                    Production(
                        name,
                        (symbols[0], suffix_nonterminal(symbols[1:])),
                    )
                )
        return Nonterminal(name)

    for production in sorted(without_units, key=production_key):
        if not production.rhs:
            raise EvaluationError("epsilon production cannot be sampled by exact positive length")
        if len(production.rhs) == 1:
            symbol = production.rhs[0]
            if not isinstance(symbol, Terminal):
                raise EvaluationError("unit elimination left a nonterminal unit production")
            binarized.add(production)
            continue
        symbols = tuple(lift(symbol) for symbol in production.rhs)
        if len(symbols) == 2:
            binarized.add(Production(production.lhs, symbols))
        else:
            binarized.add(
                Production(
                    production.lhs,
                    (symbols[0], suffix_nonterminal(symbols[1:])),
                )
            )
    binarized.update(lifted)
    return prune_grammar(
        Grammar(
            start=grammar.start,
            productions=binarized,
            type_labels=dict(grammar.type_labels),
        )
    )


def binarize_with_units(grammar: Grammar) -> Grammar:
    """Lift terminals and binarize RHSs without expanding unit closure.

    ``to_cnf`` is deliberately retained for the small exact-language DAFSA
    implementation below.  It is a poor fit for contextual API grammars,
    though: copying every non-unit production to every unit ancestor can turn
    a modest type hierarchy into tens of thousands of Python objects.  The
    online recognizer and DFA index use this representation instead and process
    the unit graph directly.
    """

    used_names = set(grammar.nonterminals)
    synthetic_ids = itertools.count()

    def synthetic_name(prefix: str) -> str:
        while True:
            name = f"#{prefix}:{next(synthetic_ids)}"
            if name not in used_names:
                used_names.add(name)
                return name

    def production_key(production: Production) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (
            production.lhs,
            tuple(
                (
                    "T" if isinstance(symbol, Terminal) else "N",
                    symbol.value,
                )
                for symbol in production.rhs
            ),
        )

    preterminals: dict[str, str] = {}
    lifted: set[Production] = set()

    def lift(symbol: Symbol) -> Nonterminal:
        if isinstance(symbol, Nonterminal):
            return symbol
        name = preterminals.get(symbol.value)
        if name is None:
            name = synthetic_name("T")
            preterminals[symbol.value] = name
        lifted.add(Production(name, (symbol,)))
        return Nonterminal(name)

    binarized: set[Production] = set()
    suffix_names: dict[tuple[str, ...], str] = {}

    def suffix_nonterminal(symbols: tuple[Nonterminal, ...]) -> Nonterminal:
        key = tuple(symbol.value for symbol in symbols)
        name = suffix_names.get(key)
        if name is None:
            name = synthetic_name("B")
            suffix_names[key] = name
            if len(symbols) == 2:
                binarized.add(Production(name, symbols))
            else:
                binarized.add(
                    Production(
                        name,
                        (symbols[0], suffix_nonterminal(symbols[1:])),
                    )
                )
        return Nonterminal(name)

    for production in sorted(grammar.productions, key=production_key):
        if not production.rhs:
            raise EvaluationError(
                "epsilon production cannot be sampled by exact positive length"
            )
        if len(production.rhs) == 1:
            binarized.add(production)
            continue
        symbols = tuple(lift(symbol) for symbol in production.rhs)
        if len(symbols) == 2:
            binarized.add(Production(production.lhs, symbols))
        else:
            binarized.add(
                Production(
                    production.lhs,
                    (symbols[0], suffix_nonterminal(symbols[1:])),
                )
            )
    binarized.update(lifted)
    return prune_grammar(
        Grammar(
            start=grammar.start,
            productions=binarized,
            type_labels=dict(grammar.type_labels),
        )
    )


class UnitAwareBinaryGrammar:
    """A compact binary grammar whose unit SCC condensation is a DAG.

    Strongly connected unit components denote the same language, so they are
    represented by one integer.  Unit edges between components remain explicit
    rather than being expanded into copied terminal/binary productions.  A
    diamond therefore remains visible to diagnostic parse counts.  The exact
    DFA views merge those derivations before counting distinct words.
    """

    def __init__(self, grammar: Grammar):
        self.grammar = binarize_with_units(grammar)
        names = sorted(self.grammar.nonterminals)
        name_ids = {name: index for index, name in enumerate(names)}
        unit_children_by_name: list[set[int]] = [set() for _ in names]
        terminal_by_name: list[set[str]] = [set() for _ in names]
        binary_by_name: list[set[tuple[int, int]]] = [set() for _ in names]
        for production in self.grammar.productions:
            lhs = name_ids[production.lhs]
            if len(production.rhs) == 1:
                symbol = production.rhs[0]
                if isinstance(symbol, Terminal):
                    terminal_by_name[lhs].add(symbol.value)
                elif isinstance(symbol, Nonterminal):
                    unit_children_by_name[lhs].add(name_ids[symbol.value])
                else:  # pragma: no cover - Symbol exhaustiveness
                    raise EvaluationError(f"invalid unary production: {production}")
            elif (
                len(production.rhs) == 2
                and isinstance(production.rhs[0], Nonterminal)
                and isinstance(production.rhs[1], Nonterminal)
            ):
                binary_by_name[lhs].add(
                    (
                        name_ids[production.rhs[0].value],
                        name_ids[production.rhs[1].value],
                    )
                )
            else:
                raise EvaluationError(f"non-binary production: {production}")

        # Iterative Kosaraju avoids Python's recursion limit on long subtype
        # chains.  First compute DFS finishing order on the unit graph.
        finish_order: list[int] = []
        visited: set[int] = set()
        for root in range(len(names)):
            if root in visited:
                continue
            visited.add(root)
            stack: list[tuple[int, Iterator[int]]] = [
                (root, iter(sorted(unit_children_by_name[root])))
            ]
            while stack:
                node, child_iterator = stack[-1]
                try:
                    child = next(child_iterator)
                except StopIteration:
                    finish_order.append(node)
                    stack.pop()
                    continue
                if child not in visited:
                    visited.add(child)
                    stack.append(
                        (child, iter(sorted(unit_children_by_name[child])))
                    )

        reverse_edges: list[set[int]] = [set() for _ in names]
        for parent, reverse_children in enumerate(unit_children_by_name):
            for child in reverse_children:
                reverse_edges[child].add(parent)
        name_component = [-1] * len(names)
        components: list[list[int]] = []
        for root in reversed(finish_order):
            if name_component[root] >= 0:
                continue
            component = len(components)
            members: list[int] = []
            name_component[root] = component
            queue = [root]
            while queue:
                node = queue.pop()
                members.append(node)
                for parent in reverse_edges[node]:
                    if name_component[parent] < 0:
                        name_component[parent] = component
                        queue.append(parent)
            components.append(members)

        self.component_names = tuple(
            tuple(names[index] for index in sorted(members))
            for members in components
        )
        self.name_components = {
            name: name_component[index] for index, name in enumerate(names)
        }
        component_count = len(components)
        terminal_rules: list[set[str]] = [set() for _ in components]
        binary_rules: list[set[tuple[int, int]]] = [set() for _ in components]
        unit_children: list[set[int]] = [set() for _ in components]
        for name_id in range(len(names)):
            component = name_component[name_id]
            terminal_rules[component].update(terminal_by_name[name_id])
            binary_rules[component].update(
                (name_component[left], name_component[right])
                for left, right in binary_by_name[name_id]
            )
            unit_children[component].update(
                name_component[child]
                for child in unit_children_by_name[name_id]
                if name_component[child] != component
            )

        self.terminal_rules = tuple(
            tuple(sorted(terminals)) for terminals in terminal_rules
        )
        self.binary_rules = tuple(
            tuple(sorted(rules)) for rules in binary_rules
        )
        self.unit_children = tuple(
            tuple(sorted(children)) for children in unit_children
        )
        unit_parents: list[set[int]] = [set() for _ in components]
        for parent, component_children in enumerate(self.unit_children):
            for child in component_children:
                unit_parents[child].add(parent)
        self.unit_parents = tuple(
            tuple(sorted(parents)) for parents in unit_parents
        )
        self.start = name_component[name_ids[self.grammar.start]]

        indegree = [0] * component_count
        for component_children in self.unit_children:
            for child in component_children:
                indegree[child] += 1
        ready = deque(index for index, degree in enumerate(indegree) if degree == 0)
        topological: list[int] = []
        while ready:
            parent = ready.popleft()
            topological.append(parent)
            for child in self.unit_children[parent]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(topological) != component_count:
            raise EvaluationError("unit SCC condensation unexpectedly contains a cycle")
        self.bottom_up = tuple(reversed(topological))
        bottom_up_rank = [0] * component_count
        for rank, component in enumerate(self.bottom_up):
            bottom_up_rank[component] = rank
        self.bottom_up_rank = tuple(bottom_up_rank)

        terminal_parents: dict[str, list[int]] = defaultdict(list)
        for parent, terminals in enumerate(self.terminal_rules):
            for terminal in terminals:
                terminal_parents[terminal].append(parent)
        self.terminal_parents = {
            terminal: tuple(parents)
            for terminal, parents in terminal_parents.items()
        }
        binary_parents: dict[tuple[int, int], list[int]] = defaultdict(list)
        for parent, rules in enumerate(self.binary_rules):
            for rule in rules:
                binary_parents[rule].append(parent)
        self.binary_parents = {
            rule: tuple(parents) for rule, parents in binary_parents.items()
        }

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    def close_recognition(self, active: set[int]) -> None:
        """Add every unit ancestor of an active component in place."""

        queue = deque(active)
        while queue:
            child = queue.popleft()
            for parent in self.unit_parents[child]:
                if parent not in active:
                    active.add(parent)
                    queue.append(parent)

    def recognizes(self, tokens: Sequence[str]) -> bool:
        if not tokens:
            return False
        length = len(tokens)
        chart: list[list[set[int]]] = [
            [set() for _ in range(length + 1)] for _ in range(length)
        ]
        for index, token in enumerate(tokens):
            cell = chart[index][index + 1]
            cell.update(self.terminal_parents.get(token, ()))
            self.close_recognition(cell)
        for span in range(2, length + 1):
            for start in range(0, length - span + 1):
                end = start + span
                cell = chart[start][end]
                for split in range(start + 1, end):
                    for left in chart[start][split]:
                        for right in chart[split][end]:
                            cell.update(self.binary_parents.get((left, right), ()))
                self.close_recognition(cell)
        return self.start in chart[0][length]


class BoundedLanguage:
    """Exact distinct token languages represented by hash-consed DAFSAs."""

    EMPTY = -1
    FINAL = 0

    def __init__(self, grammar: Grammar, max_length: int, max_states: int):
        self.grammar = to_cnf(grammar)
        self.max_length = max_length
        self.max_states = max_states
        terminals = sorted(self.grammar.terminals)
        self.token_ids = {token: index for index, token in enumerate(terminals)}
        self.tokens = terminals
        self.rows: list[tuple[tuple[int, int], ...]] = [()]
        self.counts: list[int] = [1]
        self.interned: dict[tuple[tuple[int, int], ...], int] = {(): self.FINAL}
        self.union_cache: dict[tuple[int, ...], int] = {}
        self.product_union_cache: dict[tuple[tuple[int, int], ...], int] = {}
        self.singletons = {
            token: self._intern(((token_id, self.FINAL),))
            for token, token_id in self.token_ids.items()
        }
        self.roots: dict[tuple[str, int], int] = {}
        self._build()

    def _intern(self, row: tuple[tuple[int, int], ...]) -> int:
        existing = self.interned.get(row)
        if existing is not None:
            return existing
        if len(self.rows) >= self.max_states:
            raise LanguageTooLarge(
                f"bounded language exceeded {self.max_states:,} DAFSA states"
            )
        state = len(self.rows)
        self.interned[row] = state
        self.rows.append(row)
        self.counts.append(sum(self.counts[child] for _token, child in row))
        return state

    def _union(self, roots: Iterable[int]) -> int:
        key = tuple(sorted(set(root for root in roots if root != self.EMPTY)))
        if not key:
            return self.EMPTY
        if len(key) == 1:
            return key[0]
        cached = self.union_cache.get(key)
        if cached is not None:
            return cached
        if self.FINAL in key:
            raise EvaluationError("attempted to union unequal residual lengths")
        grouped: dict[int, list[int]] = defaultdict(list)
        for root in key:
            for token, child in self.rows[root]:
                grouped[token].append(child)
        row = tuple(
            (token, self._union(children))
            for token, children in sorted(grouped.items())
        )
        result = self._intern(row)
        self.union_cache[key] = result
        return result

    def _union_products(self, products: Iterable[tuple[int, int]]) -> int:
        """Determinize a union of concatenated exact-length DFA languages."""

        normalized: set[tuple[int, int]] = set()
        for prefix, suffix in products:
            if prefix == self.EMPTY or suffix == self.EMPTY:
                continue
            if prefix == self.FINAL:
                prefix, suffix = suffix, self.FINAL
            normalized.add((prefix, suffix))
        if not normalized:
            return self.EMPTY
        key = tuple(sorted(normalized))
        if len(key) == 1 and key[0][1] == self.FINAL:
            return key[0][0]
        cached = self.product_union_cache.get(key)
        if cached is not None:
            return cached
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for prefix, suffix in key:
            if prefix == self.FINAL:
                raise EvaluationError("attempted to determinize unequal residual lengths")
            for token, child in self.rows[prefix]:
                grouped[token].append((child, suffix))
        row = tuple(
            (token, self._union_products(children))
            for token, children in sorted(grouped.items())
        )
        result = self._intern(row)
        self.product_union_cache[key] = result
        return result

    def _build(self) -> None:
        terminal_rules: dict[str, list[str]] = defaultdict(list)
        binary_rules: list[tuple[str, str, str]] = []
        for production in self.grammar.productions:
            if len(production.rhs) == 1 and isinstance(production.rhs[0], Terminal):
                terminal_rules[production.lhs].append(production.rhs[0].value)
            elif (
                len(production.rhs) == 2
                and isinstance(production.rhs[0], Nonterminal)
                and isinstance(production.rhs[1], Nonterminal)
            ):
                binary_rules.append(
                    (
                        production.lhs,
                        production.rhs[0].value,
                        production.rhs[1].value,
                    )
                )
            else:
                raise EvaluationError(f"non-CNF production: {production}")
        for lhs, tokens in terminal_rules.items():
            self.roots[(lhs, 1)] = self._union(
                self.singletons[token] for token in tokens
            )
        by_lhs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for lhs, left, right in binary_rules:
            by_lhs[lhs].append((left, right))
        for length in range(2, self.max_length + 1):
            for lhs, rules in by_lhs.items():
                products: list[tuple[int, int]] = []
                for left, right in rules:
                    for split in range(1, length):
                        left_root = self.roots.get((left, split), self.EMPTY)
                        if left_root == self.EMPTY:
                            continue
                        right_root = self.roots.get(
                            (right, length - split), self.EMPTY
                        )
                        if right_root == self.EMPTY:
                            continue
                        products.append((left_root, right_root))
                root = self._union_products(products)
                if root != self.EMPTY:
                    self.roots[(lhs, length)] = root

    def root(self, length: int) -> int:
        return self.roots.get((self.grammar.start, length), self.EMPTY)

    def language_size(self, length: int) -> int:
        root = self.root(length)
        return 0 if root == self.EMPTY else self.counts[root]

    def recognizes(self, tokens: Sequence[str]) -> bool:
        state = self.root(len(tokens))
        if state == self.EMPTY:
            return False
        for token in tokens:
            token_id = self.token_ids.get(token)
            if token_id is None or state == self.FINAL:
                return False
            row = self.rows[state]
            child = next((child for edge, child in row if edge == token_id), None)
            if child is None:
                return False
            state = child
        return state == self.FINAL

    def unrank(self, length: int, rank: int) -> tuple[str, ...]:
        root = self.root(length)
        size = 0 if root == self.EMPTY else self.counts[root]
        if not (0 <= rank < size):
            raise IndexError(f"rank {rank} outside language of size {size}")
        state = root
        result: list[str] = []
        while state != self.FINAL:
            for token_id, child in self.rows[state]:
                count = self.counts[child]
                if rank < count:
                    result.append(self.tokens[token_id])
                    state = child
                    break
                rank -= count
            else:
                raise EvaluationError("invalid DAFSA suffix count")
        if len(result) != length:
            raise EvaluationError("unrank produced the wrong token length")
        return tuple(result)

    def rank(self, tokens: Sequence[str]) -> int:
        state = self.root(len(tokens))
        if state == self.EMPTY:
            raise ValueError("word is outside the bounded DFA")
        rank = 0
        for token in tokens:
            token_id = self.token_ids.get(token)
            if token_id is None or state == self.FINAL:
                raise ValueError(f"word is outside the bounded DFA at {token!r}")
            selected = False
            for edge, child in self.rows[state]:
                if edge < token_id:
                    rank += self.counts[child]
                elif edge == token_id:
                    state = child
                    selected = True
                    break
                else:
                    break
            if not selected:
                raise ValueError(f"word is outside the bounded DFA at {token!r}")
        if state != self.FINAL:
            raise ValueError("word ended outside the bounded DFA final state")
        return rank

    def sample(self, length: int, random_source: random.Random) -> tuple[str, ...]:
        size = self.language_size(length)
        if size == 0:
            raise EvaluationError(f"grammar has no words of length {length}")
        return self.unrank(length, random_source.randrange(size))


class UnitAwareBoundedLanguage(BoundedLanguage):
    """Exact bounded DAFSAs without materializing the grammar's unit closure."""

    def __init__(
        self,
        grammar: Grammar | UnitAwareBinaryGrammar,
        max_length: int,
        max_states: int,
    ) -> None:
        if max_length < 0:
            raise ValueError("maximum token length must be nonnegative")
        if max_states < 1:
            raise ValueError("maximum DFA states must be positive")
        self.compiled = (
            grammar
            if isinstance(grammar, UnitAwareBinaryGrammar)
            else UnitAwareBinaryGrammar(grammar)
        )
        self.grammar = self.compiled.grammar
        self.max_length = 0
        self.max_states = max_states
        terminals = sorted(self.grammar.terminals)
        self.token_ids = {token: index for index, token in enumerate(terminals)}
        self.tokens = terminals
        self.rows: list[tuple[tuple[int, int], ...]] = [()]
        self.counts: list[int] = [1]
        self.interned: dict[tuple[tuple[int, int], ...], int] = {
            (): self.FINAL
        }
        self.union_cache: dict[tuple[int, ...], int] = {}
        self.product_union_cache: dict[tuple[tuple[int, int], ...], int] = {}
        self.singletons = {
            token: self._intern(((token_id, self.FINAL),))
            for token, token_id in self.token_ids.items()
        }
        self.component_roots: dict[tuple[int, int], int] = {}
        self.ensure_length(max_length)

    def _append_length(self) -> None:
        length = self.max_length + 1
        base: dict[int, int] = {}
        if length == 1:
            for component, terminals in enumerate(self.compiled.terminal_rules):
                root = self._union(
                    self.singletons[token] for token in terminals
                )
                if root != self.EMPTY:
                    base[component] = root
        else:
            for component, rules in enumerate(self.compiled.binary_rules):
                products: list[tuple[int, int]] = []
                for left, right in rules:
                    for split in range(1, length):
                        left_root = self.component_roots.get(
                            (left, split), self.EMPTY
                        )
                        right_root = self.component_roots.get(
                            (right, length - split), self.EMPTY
                        )
                        if left_root != self.EMPTY and right_root != self.EMPTY:
                            products.append((left_root, right_root))
                root = self._union_products(products)
                if root != self.EMPTY:
                    base[component] = root
        # Unit SCCs are condensed into a DAG by UnitAwareBinaryGrammar.
        # Unioning child roots in child-before-parent order merges ambiguity
        # before suffix counts are computed.
        for component in self.compiled.bottom_up:
            root = self._union(
                (
                    base.get(component, self.EMPTY),
                    *(
                        self.component_roots.get((child, length), self.EMPTY)
                        for child in self.compiled.unit_children[component]
                    ),
                )
            )
            if root != self.EMPTY:
                self.component_roots[(component, length)] = root
        self.max_length = length

    def ensure_length(self, length: int) -> None:
        if length < 0:
            raise ValueError("token length must be nonnegative")
        while self.max_length < length:
            self._append_length()

    def root(self, length: int) -> int:
        return self.component_roots.get(
            (self.compiled.start, length), self.EMPTY
        )


class ShortlexDFAIndex:
    """Lazy zero-based bijection for a CFG's nonempty global shortlex language."""

    def __init__(
        self,
        grammar: Grammar | UnitAwareBinaryGrammar,
        max_states: int = DEFAULT_MAX_DFA_STATES,
    ) -> None:
        self.language = UnitAwareBoundedLanguage(grammar, 0, max_states)
        self.slice_ends: list[int] = []
        self.finite_max_length = self._finite_maximum_length()

    def _finite_maximum_length(self) -> int | None:
        """Return the language's maximum length, or None when it is infinite."""

        compiled = self.language.compiled

        def dependencies(component: int) -> tuple[int, ...]:
            return (
                *compiled.unit_children[component],
                *(
                    child
                    for left, right in compiled.binary_rules[component]
                    for child in (left, right)
                ),
            )

        colors = [0] * compiled.component_count
        postorder: list[int] = []
        start = compiled.start
        colors[start] = 1
        stack: list[tuple[int, Iterator[int]]] = [
            (start, iter(dependencies(start)))
        ]
        while stack:
            component, children = stack[-1]
            try:
                child = next(children)
            except StopIteration:
                colors[component] = 2
                postorder.append(component)
                stack.pop()
                continue
            if colors[child] == 1:
                return None
            if colors[child] == 0:
                colors[child] = 1
                stack.append((child, iter(dependencies(child))))

        longest = [0] * compiled.component_count
        for component in postorder:
            maximum = 1 if compiled.terminal_rules[component] else 0
            for child in compiled.unit_children[component]:
                maximum = max(maximum, longest[child])
            for left, right in compiled.binary_rules[component]:
                maximum = max(maximum, longest[left] + longest[right])
            longest[component] = maximum
        return longest[start]

    @property
    def indexed_size(self) -> int:
        return self.slice_ends[-1] if self.slice_ends else 0

    @property
    def indexed_lengths(self) -> int:
        return len(self.slice_ends)

    @property
    def state_count(self) -> int:
        return len(self.language.rows)

    @property
    def transition_count(self) -> int:
        return sum(len(row) for row in self.language.rows)

    def _append_slice(self) -> None:
        length = self.indexed_lengths + 1
        self.language.ensure_length(length)
        self.slice_ends.append(
            self.indexed_size + self.language.language_size(length)
        )

    def ensure_rank(self, rank: int) -> None:
        if rank < 0:
            raise ValueError(f"rank must be nonnegative: {rank}")
        while rank >= self.indexed_size:
            if (
                self.finite_max_length is not None
                and self.indexed_lengths >= self.finite_max_length
            ):
                raise IndexError(
                    f"rank {rank} outside finite DFA language of size "
                    f"{self.indexed_size}"
                )
            self._append_slice()

    def unrank(self, rank: int) -> tuple[str, ...]:
        self.ensure_rank(rank)
        slice_index = bisect.bisect_right(self.slice_ends, rank)
        slice_start = self.slice_ends[slice_index - 1] if slice_index else 0
        return self.language.unrank(slice_index + 1, rank - slice_start)

    def rank(self, tokens: Sequence[str]) -> int:
        if not tokens:
            raise ValueError("the global DFA index excludes the empty word")
        if (
            self.finite_max_length is not None
            and len(tokens) > self.finite_max_length
        ):
            raise ValueError("word is longer than the finite DFA language")
        while self.indexed_lengths < len(tokens):
            self._append_slice()
        slice_start = self.slice_ends[len(tokens) - 2] if len(tokens) > 1 else 0
        return slice_start + self.language.rank(tokens)


class DerivationCounter:
    """Exact inside and parse counts used by grammar regression tests."""

    def __init__(
        self,
        grammar: Grammar | UnitAwareBinaryGrammar,
        minimum_length: int,
        maximum_length: int | None = None,
    ):
        if maximum_length is None:
            maximum_length = minimum_length
        if minimum_length < 1 or maximum_length < minimum_length:
            raise ValueError("invalid positive token-length range")
        self.compiled = (
            grammar
            if isinstance(grammar, UnitAwareBinaryGrammar)
            else UnitAwareBinaryGrammar(grammar)
        )
        self.grammar = self.compiled.grammar
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length
        self.counts: list[list[int]] = [
            [0] * self.compiled.component_count
            for _ in range(self.maximum_length + 1)
        ]
        self._compute_inside_counts()

    def _close_unit_counts(self, counts: list[int]) -> None:
        for parent in self.compiled.bottom_up:
            for child in self.compiled.unit_children[parent]:
                counts[parent] += counts[child]

    def _compute_inside_counts(self) -> None:
        terminals = self.counts[1]
        for parent, terminal_rules in enumerate(self.compiled.terminal_rules):
            terminals[parent] = len(terminal_rules)
        self._close_unit_counts(terminals)
        for length in range(2, self.maximum_length + 1):
            counts = self.counts[length]
            for parent, binary_rules in enumerate(self.compiled.binary_rules):
                count = 0
                for left, right in binary_rules:
                    for split in range(1, length):
                        count += (
                            self.counts[split][left]
                            * self.counts[length - split][right]
                        )
                counts[parent] = count
            self._close_unit_counts(counts)

    @property
    def derivation_count(self) -> int:
        return sum(
            self.counts[length][self.compiled.start]
            for length in range(self.minimum_length, self.maximum_length + 1)
        )

    @property
    def derivation_counts(self) -> dict[int, int]:
        return {
            length: self.counts[length][self.compiled.start]
            for length in range(self.minimum_length, self.maximum_length + 1)
        }

    def _unrank_derivation(
        self,
        nonterminal: str | int,
        length: int,
        rank: int,
    ) -> tuple[str, ...]:
        component = (
            nonterminal
            if isinstance(nonterminal, int)
            else self.compiled.name_components[nonterminal]
        )
        total = self.counts[length][component]
        if not (0 <= rank < total):
            raise IndexError(f"derivation rank {rank} outside [0, {total})")
        if length == 1:
            terminals = self.compiled.terminal_rules[component]
            if rank < len(terminals):
                return (terminals[rank],)
            rank -= len(terminals)
        for left, right in self.compiled.binary_rules[component]:
            for split in range(1, length):
                left_count = self.counts[split][left]
                right_count = self.counts[length - split][right]
                block = left_count * right_count
                if rank >= block:
                    rank -= block
                    continue
                left_rank, right_rank = divmod(rank, right_count)
                return self._unrank_derivation(
                    left, split, left_rank
                ) + self._unrank_derivation(
                    right, length - split, right_rank
                )
        for child in self.compiled.unit_children[component]:
            child_count = self.counts[length][child]
            if rank < child_count:
                return self._unrank_derivation(child, length, rank)
            rank -= child_count
        raise EvaluationError("inside count and derivation unranking disagree")

    def recognizes(self, tokens: Sequence[str]) -> bool:
        return self.compiled.recognizes(tokens)

    def parse_count(self, tokens: Sequence[str]) -> int:
        length = len(tokens)
        if not self.minimum_length <= length <= self.maximum_length:
            return 0
        chart: list[list[dict[int, int]]] = [
            [defaultdict(int) for _ in range(length + 1)]
            for _ in range(length)
        ]

        def close_units(cell: dict[int, int]) -> None:
            # Discover only unit ancestors reachable from this CKY cell, then
            # evaluate that induced DAG in child-before-parent order.  This is
            # the same recurrence as the dense pass and preserves diamond-path
            # ambiguity, without scanning every library component per cell.
            active = set(cell)
            queue = deque(active)
            while queue:
                child = queue.popleft()
                for parent in self.compiled.unit_parents[child]:
                    if parent not in active:
                        active.add(parent)
                        queue.append(parent)
            for parent in sorted(
                active, key=self.compiled.bottom_up_rank.__getitem__
            ):
                for child in self.compiled.unit_children[parent]:
                    child_count = cell.get(child, 0)
                    if child_count:
                        cell[parent] += child_count

        for index, token in enumerate(tokens):
            cell = chart[index][index + 1]
            for parent in self.compiled.terminal_parents.get(token, ()):
                cell[parent] += 1
            close_units(cell)
        for span in range(2, length + 1):
            for start in range(0, length - span + 1):
                end = start + span
                cell = chart[start][end]
                for split in range(start + 1, end):
                    left_cell = chart[start][split]
                    right_cell = chart[split][end]
                    for left, left_count in left_cell.items():
                        for right, right_count in right_cell.items():
                            for parent in self.compiled.binary_parents.get(
                                (left, right), ()
                            ):
                                cell[parent] += left_count * right_count
                close_units(cell)
        return chart[0][length].get(self.compiled.start, 0)

def recognizes(
    grammar: Grammar | UnitAwareBinaryGrammar,
    tokens: Sequence[str],
) -> bool:
    compiled = (
        grammar
        if isinstance(grammar, UnitAwareBinaryGrammar)
        else UnitAwareBinaryGrammar(grammar)
    )
    return compiled.recognizes(tokens)


def instantiate_tokens(tokens: Sequence[str], occupied: frozenset[str]) -> tuple[str, ...]:
    fresh = "__api2cfg_fresh_0"
    counter = 0
    while fresh in occupied:
        counter += 1
        fresh = f"__api2cfg_fresh_{counter}"
    return tuple(fresh if token == FRESH_TOKEN else token for token in tokens)


def render_tokens(tokens: Sequence[str], occupied: frozenset[str]) -> str:
    return " ".join(instantiate_tokens(tokens, occupied))


def maximum_call_arity(maximum_tokens: int, *, assignment: bool) -> int:
    """Upper-bound call arity within a fixed grammar-construction budget."""

    root_tokens = 2 if assignment else 0
    expression_tokens = maximum_tokens - root_tokens
    return max(0, (expression_tokens - 2) // 2)


@dataclass
class RunningMetrics:
    files_evaluated: int = 0
    evaluated: int = 0
    recognized: int = 0
    precision_accepted: int = 0
    precision_checked: int = 0
    precision_requested: int = 0
    sampleable_statements: int = 0
    sampler_failures: int = 0
    total_cfg_intersection_seconds: float = 0.0
    failure_reasons: Counter[str] = field(default_factory=Counter)
    diagnostic_codes: Counter[str] = field(default_factory=Counter)
    sampled_lengths: Counter[int] = field(default_factory=Counter)
    sampled_length_offsets: Counter[int] = field(default_factory=Counter)

    @property
    def recall(self) -> float:
        return self.recognized / self.evaluated if self.evaluated else math.nan

    @property
    def precision(self) -> float:
        if not self.precision_checked:
            return math.nan
        return self.precision_accepted / self.precision_checked

    @property
    def precision_coverage(self) -> float:
        if not self.precision_requested:
            return math.nan
        return self.precision_checked / self.precision_requested

    @property
    def average_cfg_intersection_seconds(self) -> float:
        if not self.evaluated:
            return math.nan
        return self.total_cfg_intersection_seconds / self.evaluated


def copy_running_metrics(metrics: RunningMetrics) -> RunningMetrics:
    """Return an independent checkpoint of all per-run evaluator metrics."""

    return replace(
        metrics,
        failure_reasons=metrics.failure_reasons.copy(),
        diagnostic_codes=metrics.diagnostic_codes.copy(),
        sampled_lengths=metrics.sampled_lengths.copy(),
        sampled_length_offsets=metrics.sampled_length_offsets.copy(),
    )


def commit_running_metrics(
    destination: RunningMetrics,
    source: RunningMetrics,
) -> None:
    """Commit one transactional statement attempt without changing identity."""

    destination.files_evaluated = source.files_evaluated
    destination.evaluated = source.evaluated
    destination.recognized = source.recognized
    destination.precision_accepted = source.precision_accepted
    destination.precision_checked = source.precision_checked
    destination.precision_requested = source.precision_requested
    destination.sampleable_statements = source.sampleable_statements
    destination.sampler_failures = source.sampler_failures
    destination.total_cfg_intersection_seconds = (
        source.total_cfg_intersection_seconds
    )
    destination.failure_reasons.clear()
    destination.failure_reasons.update(source.failure_reasons)
    destination.diagnostic_codes.clear()
    destination.diagnostic_codes.update(source.diagnostic_codes)
    destination.sampled_lengths.clear()
    destination.sampled_lengths.update(source.sampled_lengths)
    destination.sampled_length_offsets.clear()
    destination.sampled_length_offsets.update(source.sampled_length_offsets)


def retry_ty_transport_once(
    semantics: TyLspClient,
    metrics: RunningMetrics,
    funnel: dict[str, int],
    evaluate_once: Callable[[RunningMetrics, dict[str, int]], None],
) -> TyTransportError | None:
    """Retry once, returning a terminal transport error after recovery."""

    metrics_checkpoint = copy_running_metrics(metrics)
    funnel_checkpoint = funnel.copy()
    restarted = False
    for attempt in range(2):
        try:
            evaluate_once(metrics, funnel)
        except TyTransportError as error:
            commit_running_metrics(metrics, metrics_checkpoint)
            funnel.clear()
            funnel.update(funnel_checkpoint)
            semantics.restart()
            if attempt == 0:
                restarted = True
                continue
            return error
        if restarted:
            funnel["ty_transport_restarts"] = (
                funnel.get("ty_transport_restarts", 0) + 1
            )
        return None
    raise AssertionError("transport retry loop did not return or raise")


def percentage(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100.0 * value:.2f}%"


def has_suppression(source: str) -> bool:
    pattern = re.compile(r"#\s*(?:type|ty)\s*:\s*ignore|#\s*noqa\b")
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return any(
            token.type == tokenize.COMMENT and pattern.search(token.string)
            for token in tokens
        )
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return False


def check_source(client: TyLspClient, uri: str, source: str) -> bool:
    return not error_diagnostics(diagnose_source(client, uri, source))


def diagnose_source(
    client: TyLspClient, uri: str, source: str
) -> list[dict[str, object]]:
    if client.document_uri is None:
        client.open(uri, source)
    elif client.document_uri == uri:
        client.change(source)
    else:
        client.open(uri, source)
    return client.diagnostics()


def error_diagnostics(
    diagnostics: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.get("severity", 1) == 1
    ]


def diagnostic_code(diagnostic: Mapping[str, object]) -> str | None:
    code = diagnostic.get("code")
    if isinstance(code, str):
        return code
    if isinstance(code, dict):
        value = code.get("value")
        return value if isinstance(value, str) else None
    return None


def lsp_offset_to_index(text: str, offset: int, encoding: str) -> int | None:
    if offset < 0:
        return None
    try:
        if encoding == "utf-8":
            return len(text.encode("utf-8")[:offset].decode("utf-8"))
        if encoding == "utf-32":
            return len(text.encode("utf-32-le")[: 4 * offset].decode("utf-32-le"))
        if encoding == "utf-16":
            return len(text.encode("utf-16-le")[: 2 * offset].decode("utf-16-le"))
    except UnicodeDecodeError:
        return None
    return min(offset, len(text))


def diagnostic_identifier(
    source: str,
    diagnostic: Mapping[str, object],
    position_encoding: str,
) -> str | None:
    range_value = diagnostic.get("range")
    if not isinstance(range_value, dict):
        return None
    start = range_value.get("start")
    end = range_value.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    start_line = start.get("line")
    end_line = end.get("line")
    start_offset = start.get("character")
    end_offset = end.get("character")
    if not (
        isinstance(start_line, int)
        and isinstance(end_line, int)
        and isinstance(start_offset, int)
        and isinstance(end_offset, int)
    ):
        return None
    if start_line != end_line:
        return None
    lines = source.splitlines()
    if not (0 <= start_line < len(lines)):
        return None
    line = lines[start_line]
    start_index = lsp_offset_to_index(line, start_offset, position_encoding)
    end_index = lsp_offset_to_index(line, end_offset, position_encoding)
    if start_index is None or end_index is None:
        return None
    value = line[start_index:end_index]
    if value.isidentifier() and not keyword.iskeyword(value):
        return value
    return None


def unresolved_names(
    source: str,
    diagnostics: Sequence[Mapping[str, object]],
    position_encoding: str,
) -> frozenset[str]:
    result: set[str] = set()
    for diagnostic in diagnostics:
        if diagnostic_code(diagnostic) not in {
            "unresolved-reference",
            "possibly-unbound-reference",
        }:
            continue
        name = diagnostic_identifier(source, diagnostic, position_encoding)
        if name is not None:
            result.add(name)
    return frozenset(result)


def diagnostic_range(
    diagnostic: Mapping[str, object],
) -> tuple[int, int, int, int] | None:
    value = diagnostic.get("range")
    if not isinstance(value, dict):
        return None
    start = value.get("start")
    end = value.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    start_line = start.get("line")
    start_character = start.get("character")
    end_line = end.get("line")
    end_character = end.get("character")
    if not (
        isinstance(start_line, int)
        and isinstance(start_character, int)
        and isinstance(end_line, int)
        and isinstance(end_character, int)
    ):
        return None
    return start_line, start_character, end_line, end_character


def diagnostic_overlaps_statement(
    diagnostic: Mapping[str, object],
    hole: Hole,
    statement: str,
    position_encoding: str,
) -> bool:
    coordinates = diagnostic_range(diagnostic)
    if coordinates is None:
        return False
    start_line, start_character, end_line, end_character = coordinates
    if not start_line <= hole.line <= end_line:
        return False
    statement_start = lsp_character(hole.indentation, position_encoding)
    statement_end = statement_start + lsp_character(statement, position_encoding)
    overlap_start = start_character if start_line == hole.line else 0
    overlap_end = end_character if end_line == hole.line else statement_end
    return overlap_start <= statement_end and overlap_end >= statement_start


def simplified_diagnostic(
    diagnostic: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {
        "code": diagnostic_code(diagnostic) or "unknown",
        "message": str(diagnostic.get("message", "diagnostic")),
    }
    coordinates = diagnostic_range(diagnostic)
    if coordinates is not None:
        start_line, start_character, end_line, end_character = coordinates
        result["range"] = {
            "start": {"line": start_line, "character": start_character},
            "end": {"line": end_line, "character": end_character},
        }
    return result


def classify_sample_failure(
    kind: str,
    hole: Hole,
    statement: str,
    diagnostics: Sequence[Mapping[str, object]],
    position_encoding: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    local = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic_overlaps_statement(
            diagnostic,
            hole,
            statement,
            position_encoding,
        )
    ]
    local_codes = tuple(
        sorted({diagnostic_code(item) or "unknown" for item in local})
    )
    downstream_codes = tuple(
        sorted(
            {
                diagnostic_code(item) or "unknown"
                for item in diagnostics
                if item not in local
            }
        )
    )
    codes = set(local_codes)
    messages = "\n".join(
        str(item.get("message", "")).lower() for item in local
    )
    if local:
        if any("syntax" in code for code in codes) or "syntax error" in messages:
            reason = "syntax-rendering"
        elif codes & {"unresolved-reference", "possibly-unbound-reference"}:
            reason = "local-name-resolution"
        elif (
            "invalid-argument-type" in codes
            and "argument to constructor `map.__new__`" in messages
        ):
            reason = "local-callable-contract"
        elif "no-matching-overload" in codes:
            reason = "local-overload-generic-correlation"
        elif codes & {
            "missing-argument",
            "unknown-argument",
            "duplicate-argument",
            "too-many-positional-arguments",
        }:
            reason = "local-call-layout"
        elif "invalid-argument-type" in codes:
            reason = "local-argument-assignability"
        elif "unsupported-operator" in codes:
            reason = "local-operator"
        elif codes & {
            "unresolved-attribute",
            "not-subscriptable",
            "call-non-callable",
            "invalid-subscript-assignment",
        }:
            reason = "local-member-subscript-callability"
        elif codes & {"invalid-assignment", "invalid-return-type"}:
            reason = "local-value-flow"
        else:
            reason = "other-local"
    elif kind == "output-assignment":
        reason = "downstream-output-contract"
    elif diagnostics:
        reason = "nonlocal-context-effect"
    else:
        reason = "other"
    return reason, local_codes, downstream_codes


def prepare_target(
    checker: TyLspClient,
    checker_uri: str,
    source: str,
    target: Target,
) -> tuple[PreparedTarget | None, str | None]:
    ablated = target.hole.render("()")
    diagnostics = diagnose_source(checker, checker_uri, ablated)
    if target.assigned_name is None:
        if error_diagnostics(diagnostics):
            return None, "expression_ablation_errors"
        return PreparedTarget(target, ablated, ablated), None

    # Reassignments need a richer post-state analysis.  This first pass handles
    # only a binding introduced by the ablated statement.
    if target.bound_before:
        return None, "assignment_rebinding"

    names = unresolved_names(
        ablated,
        diagnostics,
        checker.position_encoding,
    )
    if len(names) == 1:
        required_name = next(iter(names))
        occupied = set(source_identifiers(ablated))
        seal_alias = "__api2cfg_seal_eval_0"
        counter = 0
        while seal_alias in occupied or seal_alias == required_name:
            counter += 1
            seal_alias = f"__api2cfg_seal_eval_{counter}"
        expression_suffix = (
            f"; from builtins import eval as {seal_alias}; "
            f"{required_name} = {seal_alias}(\"None\")"
        )
        semantic_source = target.hole.render(f"(){expression_suffix}")
        semantic_diagnostics = diagnose_source(
            checker,
            checker_uri,
            semantic_source,
        )
        if error_diagnostics(semantic_diagnostics):
            return None, "assignment_seal_errors"
        exact_target = replace(
            target,
            kind="output-assignment",
            fresh_name=None,
        )
        return (
            PreparedTarget(
                exact_target,
                ablated,
                semantic_source,
                required_assignment=required_name,
                expression_suffix=expression_suffix,
                excluded_names=frozenset({required_name, seal_alias}),
            ),
            None,
        )

    if len(names) > 1:
        return None, "assignment_multiple_unbound_names"
    if error_diagnostics(diagnostics):
        return None, "assignment_ablation_errors"
    if target.loaded_after:
        return None, "assignment_unresolved_name_not_inferred"
    private_target = replace(
        target,
        kind="fresh-assignment",
        fresh_name=target.assigned_name,
    )
    return PreparedTarget(private_target, ablated, ablated), None


def ty_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return result.stdout.strip() or f"{executable} (version unavailable)"


@dataclass(frozen=True)
class EvaluationOptions:
    dataset: str
    source: Path
    split: str
    files: int
    precision_samples: int
    max_samples: int | None
    seed: int
    ty: str
    allow_ignores: bool
    builder: BuilderOptions
    max_dfa_states: int
    statement_timeout: float
    json_lines: bool
    show_samples: bool
    library_directory: Path = DEFAULT_LIBRARY_DIRECTORY
    shard_count: int = 1
    shard_index: int = 0


def emit_record(record: Mapping[str, object], *, json_lines: bool) -> None:
    if json_lines:
        print(json.dumps(record, sort_keys=True), flush=True)
        return
    sample_checked = record["sample_checked"]
    sample_accepted = record["sample_accepted"]
    index = record["index"]
    seconds = record["seconds"]
    cfg_intersection_seconds = record["cfg_intersection_seconds"]
    original_statement = record["original_statement"]
    sampled_ranks = record["sampled_ranks"]
    sampled_words = record["sampled_words"]
    if not isinstance(sample_checked, int) or not isinstance(sample_accepted, int):
        raise EvaluationError("sample counts must be integers")
    if (
        not isinstance(index, int)
        or not isinstance(seconds, (int, float))
        or not isinstance(cfg_intersection_seconds, (int, float))
        or not isinstance(original_statement, str)
        or not isinstance(sampled_ranks, list)
        or not all(isinstance(rank, int) for rank in sampled_ranks)
        or not isinstance(sampled_words, list)
        or not all(isinstance(word, str) for word in sampled_words)
    ):
        raise EvaluationError("record has invalid display fields")
    per_precision = math.nan if sample_checked == 0 else sample_accepted / sample_checked
    status = "hit" if record["recognized"] else "MISS"
    print(
        f"[{index:04d}] {status:4s} "
        f"{record['member']}:{record['line']} "
        f"{record['kind']} tokens={record['tokens']} "
        f"recall={record['running_recall']} "
        f"precision={record['running_precision']} "
        f"coverage={record['running_precision_coverage']} "
        f"(this={percentage(per_precision)} {sample_accepted}/{sample_checked}) "
        f"|V|={record['nonterminals']} |P|={record['productions']} "
        f"|G|={record['grammar_symbols']} "
        f"assignability={record['assignability_pairs_cached']}cached/"
        f"{record['assignability_pairs_checked']}checked "
        f"DFA[0..{record['sample_rank_limit']})="
        f"{record['dfa_indexed_words']}words/"
        f"{record['dfa_states']}states "
        f"cfg+intersection={cfg_intersection_seconds:.3f}s "
        f"total={seconds:.3f}s",
        flush=True,
    )
    print(f"  original: {original_statement}", flush=True)
    if sampled_words:
        print("  sampled:", flush=True)
        for sample_index, (rank, word) in enumerate(
            zip(sampled_ranks[:3], sampled_words[:3]), start=1
        ):
            print(f"    {sample_index}. rank {rank}: {word}", flush=True)
    else:
        print("  sampled: <none>", flush=True)


def statement_common_record(
    metrics: RunningMetrics,
    dataset_source: DatasetSource,
    file_index: int,
    statement_index: int,
    candidate_index: int,
    prepared: PreparedTarget,
    truth: Sequence[str],
) -> dict[str, object]:
    """Return metadata shared by every terminal statement record."""

    selected = prepared.target
    inference_matches: bool | None = None
    if (
        prepared.required_assignment is not None
        and selected.assigned_name is not None
    ):
        inference_matches = prepared.required_assignment == selected.assigned_name
    return {
        "event": "statement",
        "dataset": dataset_source.dataset,
        "split": dataset_source.split,
        "problem_id": dataset_source.problem_id,
        "solution_index": dataset_source.solution_index,
        "difficulty": dataset_source.difficulty,
        "url": dataset_source.url,
        "index": metrics.evaluated,
        "file_index": file_index,
        "statement_index": statement_index,
        "candidate_index": candidate_index,
        "member": dataset_source.member,
        "line": selected.node.lineno,
        "column": selected.node.col_offset,
        "kind": selected.kind,
        "assigned_name": selected.assigned_name,
        "required_assignment": prepared.required_assignment,
        "assignment_inference_matches": inference_matches,
        "tokens": len(truth),
        "original_statement": selected.text,
        "ground_truth": list(truth),
    }


def failed_statement_fields(
    metrics: RunningMetrics,
    *,
    sample_quota: int,
    sample_tokens: int,
    builder_max_tokens: int,
    message: str,
    elapsed: float,
) -> dict[str, object]:
    """Return the schema-complete zero-sample fields for a terminal failure."""

    return {
        "recognized": False,
        "running_recall": percentage(metrics.recall),
        "sample_accepted": 0,
        "sample_checked": 0,
        "sample_requested": sample_quota,
        "sampled_ranks": [],
        "sampled_words": [],
        "sample_failures": [],
        "running_precision": percentage(metrics.precision),
        "running_precision_coverage": percentage(metrics.precision_coverage),
        "sample_tokens": sample_tokens,
        "sample_rank_limit": SAMPLE_RANK_LIMIT,
        "builder_max_tokens": builder_max_tokens,
        "dfa_indexed_words": 0,
        "dfa_indexed_lengths": 0,
        "dfa_states": 0,
        "dfa_transitions": 0,
        "productions": 0,
        "nonterminals": 0,
        "terminals": 0,
        "grammar_symbols": 0,
        "sampler_error": message,
        "scope_names": 0,
        "expression_types": 0,
        "callables": 0,
        "signatures": 0,
        "receiver_types": 0,
        "member_completions": 0,
        "incomplete_completion_queries": 0,
        "completion_queries_at_cap": 0,
        "dynamic_types": 0,
        "assignment_types_checked": 0,
        "assignment_types_rejected": 0,
        "output_producer_families": 0,
        "output_producers_checked": 0,
        "output_producers_rejected": 0,
        "output_producers_local_fallback": 0,
        "output_producers_unchecked": 0,
        "output_producer_validation_seconds": 0.0,
        "module_member_fallbacks": 0,
        "derived_representatives": 0,
        "invalid_representatives": 0,
        "library_artifacts": 0,
        "library_productions": 0,
        "library_live_fallbacks": 0,
        "library_incomplete_artifacts": 0,
        "assignability_pairs_cached": 0,
        "assignability_pairs_checked": 0,
        "binary_operation_productions": 0,
        "subscript_productions": 0,
        "slice_productions": 0,
        "cfg_intersection_seconds": elapsed,
        "seconds": elapsed,
    }


def _evaluate_prepared_statement_once(
    options: EvaluationOptions,
    sample_quota: int,
    metrics: RunningMetrics,
    funnel: dict[str, int],
    semantics: TyLspClient,
    workspace: Path,
    dataset_source: DatasetSource,
    file_index: int,
    statement_index: int,
    candidate_index: int,
    prepared: PreparedTarget,
    library_catalog: LibraryCatalog,
) -> None:
    member_name = dataset_source.member
    selected = prepared.target
    started = time.perf_counter()
    truth = canonical_tokens(selected)
    sample_tokens = len(truth)
    builder_max_tokens = sample_tokens + 2
    builder_options = replace(
        options.builder,
        max_call_arity=maximum_call_arity(
            builder_max_tokens,
            assignment=prepared.required_assignment is not None,
        ),
        max_tokens=builder_max_tokens,
    )
    target_identity = (
        f"{member_name}\0{selected.node.lineno}\0{selected.node.col_offset}"
        f"\0{selected.node.end_lineno}\0{selected.node.end_col_offset}"
    )
    semantic_uri = uri_for(
        workspace / f"semantic_{stable_digest(target_identity, 16)}.py"
    )
    semantics.open(semantic_uri, prepared.semantic_source)

    def common_record() -> dict[str, object]:
        return statement_common_record(
            metrics,
            dataset_source,
            file_index,
            statement_index,
            candidate_index,
            prepared,
            truth,
        )

    def emit_construction_failure(message: str) -> None:
        metrics.evaluated += 1
        metrics.precision_requested += sample_quota
        funnel["evaluated_statements"] += 1
        elapsed = time.perf_counter() - started
        metrics.total_cfg_intersection_seconds += elapsed
        record = common_record()
        record.update(
            failed_statement_fields(
                metrics,
                sample_quota=sample_quota,
                sample_tokens=sample_tokens,
                builder_max_tokens=builder_options.max_tokens,
                message=message,
                elapsed=elapsed,
            )
        )
        emit_record(record, json_lines=options.json_lines)

    semantic_errors = error_diagnostics(semantics.diagnostics())
    if semantic_errors:
        funnel["semantic_ablation_disagreement"] += 1
        emit_construction_failure("semantic scaffold contains ty errors")
        return

    probe = SemanticProbe(
        semantics,
        selected.hole,
        prepared.semantic_source,
        required_assignment=prepared.required_assignment,
        expression_prefix=prepared.expression_prefix,
        expression_suffix=prepared.expression_suffix,
        excluded_names=prepared.excluded_names,
    )
    ablated_ids = source_identifiers(prepared.ablated)
    library_artifacts: list[LibraryArtifact] = []
    for module in visible_imported_library_modules(
        prepared.ablated, selected.hole.line
    ):
        funnel["library_import_mentions"] += 1
        lookup = library_catalog.lookup(module)
        if lookup.artifact is None:
            if lookup.reason == "missing":
                funnel["library_artifact_missing"] += 1
            else:
                funnel["library_artifact_incompatible"] += 1
            continue
        library_artifacts.append(lookup.artifact)
        funnel["library_artifact_uses"] += 1
    builder = GrammarBuilder(
        probe,
        ablated_ids,
        builder_options,
        {},
        required_assignment=prepared.required_assignment,
        library_artifacts=library_artifacts,
        from_import_bindings=visible_from_import_bindings(
            prepared.ablated, selected.hole.line
        ),
    )
    try:
        grammar, build_stats = builder.build()
    except TyTransportError:
        raise
    except EvaluationError as error:
        funnel["grammar_failures"] += 1
        if not options.json_lines:
            print(
                f"warning: {member_name}:{selected.node.lineno}: {error}",
                file=sys.stderr,
                flush=True,
            )
        emit_construction_failure(f"grammar construction failed: {error}")
        return

    cfg_seconds = time.perf_counter() - started
    intersection_started = time.perf_counter()
    compiled_grammar = UnitAwareBinaryGrammar(grammar)
    recognized = compiled_grammar.recognizes(truth)
    sample_accepted = 0
    sample_checked = 0
    sampled_ranks: list[int] = []
    sampled_words: list[str] = []
    sample_failures: list[dict[str, object]] = []
    sampler_error: str | None = None
    intersection_seconds: float | None = None
    dfa_index: ShortlexDFAIndex | None = None
    random_source = random.Random(
        int(stable_digest(f"{options.seed}\0{target_identity}", 16), 16)
    )
    try:
        sampled_ranks = [
            random_source.randrange(SAMPLE_RANK_LIMIT)
            for _ in range(max(sample_quota, 1))
        ]
        dfa_index = ShortlexDFAIndex(
            compiled_grammar,
            max_states=options.max_dfa_states,
        )
        # Sampling is defined over the complete fixed rank domain, not merely
        # the ranks we happened to draw for this statement.  Requiring its last
        # rank makes eligibility independent of the seed and sample quota.
        dfa_index.ensure_rank(SAMPLE_RANK_LIMIT - 1)
        intersection_seconds = time.perf_counter() - intersection_started
        metrics.sampleable_statements += 1
        occupied = ablated_ids
        for sample_index, rank in enumerate(sampled_ranks):
            sampled = dfa_index.unrank(rank)
            statement = render_tokens(sampled, occupied)
            if len(sampled_words) < 3:
                sampled_words.append(statement)
            if sample_index >= sample_quota:
                continue
            sample_checked += 1
            diagnostics = error_diagnostics(probe.diagnostics(statement))
            if not diagnostics:
                sample_accepted += 1
            metrics.sampled_lengths[len(sampled)] += 1
            metrics.sampled_length_offsets[len(sampled) - sample_tokens] += 1
            if diagnostics:
                reason, local_codes, downstream_codes = classify_sample_failure(
                    selected.kind,
                    selected.hole,
                    statement,
                    diagnostics,
                    semantics.position_encoding,
                )
                metrics.failure_reasons[reason] += 1
                codes = {
                    diagnostic_code(item) or "unknown"
                    for item in diagnostics
                }
                metrics.diagnostic_codes.update(codes)
                sample_failures.append(
                    {
                        "ordinal": metrics.precision_checked + sample_checked,
                        "sample_index": sample_index + 1,
                        "rank0": rank,
                        "canonical_tokens": list(sampled),
                        "statement": statement,
                        "tokens": len(sampled),
                        "reason": reason,
                        "local_codes": list(local_codes),
                        "downstream_codes": list(downstream_codes),
                        "diagnostics": [
                            simplified_diagnostic(item)
                            for item in sorted(
                                diagnostics,
                                key=lambda diagnostic: (
                                    diagnostic_range(diagnostic)
                                    or (-1, -1, -1, -1),
                                    diagnostic_code(diagnostic) or "unknown",
                                ),
                            )
                        ],
                    }
                )
    except TyTransportError:
        raise
    except (LanguageTooLarge, EvaluationError, IndexError) as error:
        if intersection_seconds is None:
            intersection_seconds = time.perf_counter() - intersection_started
        sampler_error = str(error)
        metrics.sampler_failures += 1

    metrics.evaluated += 1
    funnel["evaluated_statements"] += 1
    metrics.recognized += int(recognized)
    metrics.precision_accepted += sample_accepted
    metrics.precision_checked += sample_checked
    metrics.precision_requested += sample_quota
    elapsed = time.perf_counter() - started
    if intersection_seconds is None:
        intersection_seconds = time.perf_counter() - intersection_started
    cfg_intersection_seconds = cfg_seconds + intersection_seconds
    metrics.total_cfg_intersection_seconds += cfg_intersection_seconds
    record = common_record()
    record.update(
        {
            "recognized": recognized,
            "running_recall": percentage(metrics.recall),
            "sample_accepted": sample_accepted,
            "sample_checked": sample_checked,
            "sample_requested": sample_quota,
            "sampled_ranks": sampled_ranks,
            "sampled_words": sampled_words,
            "sample_failures": sample_failures,
            "running_precision": percentage(metrics.precision),
            "running_precision_coverage": percentage(metrics.precision_coverage),
            "sample_tokens": sample_tokens,
            "sample_rank_limit": SAMPLE_RANK_LIMIT,
            "builder_max_tokens": builder_options.max_tokens,
            "dfa_indexed_words": (
                0 if dfa_index is None else dfa_index.indexed_size
            ),
            "dfa_indexed_lengths": (
                0 if dfa_index is None else dfa_index.indexed_lengths
            ),
            "dfa_states": 0 if dfa_index is None else dfa_index.state_count,
            "dfa_transitions": (
                0 if dfa_index is None else dfa_index.transition_count
            ),
            "productions": len(grammar.productions),
            "nonterminals": len(grammar.nonterminals),
            "terminals": len(grammar.terminals),
            "grammar_symbols": grammar.symbol_count,
            "sampler_error": sampler_error,
            "scope_names": build_stats.scope_names,
            "expression_types": build_stats.expression_types,
            "callables": build_stats.callables,
            "signatures": build_stats.signatures,
            "receiver_types": build_stats.receiver_types,
            "member_completions": build_stats.member_completions,
            "incomplete_completion_queries": build_stats.incomplete_completion_queries,
            "completion_queries_at_cap": build_stats.completion_queries_at_cap,
            "dynamic_types": build_stats.dynamic_types,
            "assignment_types_checked": build_stats.assignment_types_checked,
            "assignment_types_rejected": build_stats.assignment_types_rejected,
            "output_producer_families": build_stats.output_producer_families,
            "output_producers_checked": build_stats.output_producers_checked,
            "output_producers_rejected": build_stats.output_producers_rejected,
            "output_producers_local_fallback": (
                build_stats.output_producers_local_fallback
            ),
            "output_producers_unchecked": build_stats.output_producers_unchecked,
            "output_producer_validation_seconds": (
                build_stats.output_producer_validation_seconds
            ),
            "module_member_fallbacks": build_stats.module_member_fallbacks,
            "derived_representatives": build_stats.derived_representatives,
            "invalid_representatives": build_stats.invalid_representatives,
            "library_artifacts": build_stats.library_artifacts,
            "library_productions": build_stats.library_productions,
            "library_live_fallbacks": build_stats.library_live_fallbacks,
            "library_incomplete_artifacts": build_stats.library_incomplete_artifacts,
            "assignability_pairs_cached": build_stats.assignability_pairs_cached,
            "assignability_pairs_checked": build_stats.assignability_pairs_checked,
            "binary_operation_productions": (
                build_stats.binary_operation_productions
            ),
            "subscript_productions": build_stats.subscript_productions,
            "slice_productions": build_stats.slice_productions,
            "cfg_intersection_seconds": cfg_intersection_seconds,
            "seconds": elapsed,
        }
    )
    emit_record(record, json_lines=options.json_lines)


def emit_terminal_statement_failure_record(
    options: EvaluationOptions,
    sample_quota: int,
    metrics: RunningMetrics,
    funnel: dict[str, int],
    dataset_source: DatasetSource,
    file_index: int,
    statement_index: int,
    candidate_index: int,
    prepared: PreparedTarget,
    message: str,
    funnel_key: str,
    started: float,
) -> None:
    """Record one schema-complete, denominator-bearing target failure."""

    truth = canonical_tokens(prepared.target)
    sample_tokens = len(truth)
    metrics.evaluated += 1
    metrics.precision_requested += sample_quota
    funnel["evaluated_statements"] = funnel.get("evaluated_statements", 0) + 1
    funnel[funnel_key] = funnel.get(funnel_key, 0) + 1
    elapsed = time.perf_counter() - started
    metrics.total_cfg_intersection_seconds += elapsed
    record = statement_common_record(
        metrics,
        dataset_source,
        file_index,
        statement_index,
        candidate_index,
        prepared,
        truth,
    )
    record.update(
        failed_statement_fields(
            metrics,
            sample_quota=sample_quota,
            sample_tokens=sample_tokens,
            builder_max_tokens=sample_tokens + 2,
            message=message,
            elapsed=elapsed,
        )
    )
    emit_record(record, json_lines=options.json_lines)


def evaluate_prepared_statement(
    options: EvaluationOptions,
    sample_quota: int,
    metrics: RunningMetrics,
    funnel: dict[str, int],
    semantics: TyLspClient,
    workspace: Path,
    dataset_source: DatasetSource,
    file_index: int,
    statement_index: int,
    candidate_index: int,
    prepared: PreparedTarget,
    library_catalog: LibraryCatalog,
) -> None:
    """Evaluate one target, recovering once from a lost semantics transport."""

    started = time.perf_counter()
    metrics_checkpoint = copy_running_metrics(metrics)
    funnel_checkpoint = funnel.copy()

    def evaluate_once(
        attempt_metrics: RunningMetrics,
        attempt_funnel: dict[str, int],
    ) -> None:
        _evaluate_prepared_statement_once(
            options,
            sample_quota,
            attempt_metrics,
            attempt_funnel,
            semantics,
            workspace,
            dataset_source,
            file_index,
            statement_index,
            candidate_index,
            prepared,
            library_catalog,
        )

    try:
        with statement_deadline(options.statement_timeout):
            transport_error = retry_ty_transport_once(
                semantics,
                metrics,
                funnel,
                evaluate_once,
            )
    except StatementTimeout as error:
        commit_running_metrics(metrics, metrics_checkpoint)
        funnel.clear()
        funnel.update(funnel_checkpoint)
        semantics.restart()
        emit_terminal_statement_failure_record(
            options,
            sample_quota,
            metrics,
            funnel,
            dataset_source,
            file_index,
            statement_index,
            candidate_index,
            prepared,
            str(error),
            "statement_timeouts",
            started,
        )
        return
    if transport_error is not None:
        emit_terminal_statement_failure_record(
            options,
            sample_quota,
            metrics,
            funnel,
            dataset_source,
            file_index,
            statement_index,
            candidate_index,
            prepared,
            f"ty transport failed after retry: {transport_error}",
            "ty_transport_failures",
            started,
        )


def evaluate(options: EvaluationOptions) -> int:
    source_path = resolved_dataset_source(
        options.dataset,
        options.source,
        options.split,
    )
    if not source_path.is_file():
        raise EvaluationError(f"dataset source not found: {source_path}")
    metrics = RunningMetrics()
    funnel: dict[str, int] = defaultdict(int)
    version = ty_version(options.ty)
    library_catalog = LibraryCatalog(
        options.library_directory, options.ty, version
    )
    source_stat = source_path.stat()
    population = (
        "all independently ablated eligible statements in the first "
        f"dataset-order ty-clean {options.dataset} source files"
    )
    if options.shard_count > 1:
        population += (
            f" in source shard {options.shard_index}/"
            f"{options.shard_count}"
        )
    if options.max_samples is not None:
        population += f", stopping after {options.max_samples} checked samples"

    def sample_limit_reached() -> bool:
        return (
            options.max_samples is not None
            and metrics.precision_checked >= options.max_samples
        )
    if options.json_lines:
        print(
            json.dumps(
                {
                    "event": "start",
                    "dataset": options.dataset,
                    "split": options.split if options.dataset == "apps" else None,
                    "ty": version,
                    "source": str(source_path),
                    "source_bytes": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "files": options.files,
                    "sample_rank_interval": f"[0, {SAMPLE_RANK_LIMIT})",
                    "sample_order": "global token shortlex DFA bijection",
                    "precision_samples": options.precision_samples,
                    "max_samples": options.max_samples,
                    "shard_count": options.shard_count,
                    "shard_index": options.shard_index,
                    "seed": options.seed,
                    "python": sys.version,
                    "platform": platform.platform(),
                    "library_directory": str(options.library_directory),
                    "allow_ignores": options.allow_ignores,
                    "max_dfa_states": options.max_dfa_states,
                    "statement_timeout": options.statement_timeout,
                    "surface_fragment": surface_fragment_metadata(),
                    "builder": {
                        "max_call_arity": (
                            "floor((ground_truth_tokens-root_tokens)/2)"
                        ),
                        "max_dynamic_composition_depth": (
                            options.builder.max_dynamic_composition_depth
                        ),
                        "max_tokens": "ground_truth_tokens+2",
                        "max_layouts_per_signature": options.builder.max_layouts_per_signature,
                        "member_depth": options.builder.member_depth,
                        "max_receiver_types": options.builder.max_receiver_types,
                        "max_module_members": options.builder.max_module_members,
                        "max_output_producers": options.builder.max_output_producers,
                    },
                    "population": population,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(
            f"ty={version}; dataset={options.dataset}; "
            f"split={options.split if options.dataset == 'apps' else 'n/a'}; "
            f"source={source_path}; target_files={options.files}; "
            f"precision_samples={options.precision_samples}; "
            f"max_samples={options.max_samples or 'unlimited'}; "
            f"statement_timeout={options.statement_timeout:g}s; "
            f"shard={options.shard_index}/{options.shard_count}; "
            "targets=all eligible statements; "
            f"sample_ranks=uniform [0, {SAMPLE_RANK_LIMIT}) via global shortlex DFA",
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="api2cfg-python-") as directory:
        workspace = Path(directory)
        checker_uri = uri_for(workspace / "clean_check.py")
        with (
            TyLspClient(options.ty, workspace) as checker,
            TyLspClient(options.ty, workspace) as semantics,
        ):
            sources = iter_dataset_sources(
                options.dataset,
                options.source,
                options.split,
                options.shard_count,
                options.shard_index,
            )
            while (
                metrics.files_evaluated < options.files
                and not sample_limit_reached()
            ):
                try:
                    dataset_source = next(sources)
                except StopIteration:
                    break
                funnel["submissions"] += 1
                source = decode_source(dataset_source.data)
                if source is None:
                    funnel["decode_failures"] += 1
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        tree = ast.parse(
                            source,
                            filename=dataset_source.member,
                            type_comments=True,
                        )
                except (SyntaxError, ValueError):
                    funnel["parse_failures"] += 1
                    continue
                funnel["parsed"] += 1
                if not options.allow_ignores and has_suppression(source):
                    funnel["suppressed"] += 1
                    continue
                if not check_source(checker, checker_uri, source):
                    funnel["ty_dirty"] += 1
                    continue

                funnel["ty_clean"] += 1
                metrics.files_evaluated += 1
                funnel["evaluated_files"] += 1
                file_index = metrics.files_evaluated
                targets = sorted(
                    candidate_targets(source, tree),
                    key=lambda target: (
                        target.node.lineno,
                        target.node.col_offset,
                        target.node.end_lineno or target.node.lineno,
                        target.node.end_col_offset or target.node.col_offset,
                        target.kind,
                    ),
                )
                funnel["candidate_statements"] += len(targets)
                if not targets:
                    funnel["no_candidate_statement_files"] += 1
                    continue

                prepared_targets: list[tuple[int, PreparedTarget]] = []
                for candidate_index, target in enumerate(targets, start=1):
                    prepared, rejection = prepare_target(
                        checker,
                        checker_uri,
                        source,
                        target,
                    )
                    if prepared is None:
                        reason = rejection or "unknown"
                        funnel[f"ineligible_{reason}"] += 1
                        continue
                    prepared_targets.append((candidate_index, prepared))
                    if prepared.required_assignment is not None:
                        funnel["output_assignment_statements"] += 1
                        if (
                            prepared.target.assigned_name
                            == prepared.required_assignment
                        ):
                            funnel["assignment_name_inference_matches"] += 1
                        else:
                            funnel["assignment_name_inference_mismatches"] += 1
                    elif prepared.target.fresh_name is not None:
                        funnel["private_assignment_statements"] += 1
                    else:
                        funnel["expression_statements"] += 1

                funnel["eligible_statements"] += len(prepared_targets)
                if not prepared_targets:
                    funnel["no_eligible_statement_files"] += 1
                    continue
                for statement_index, (
                    candidate_index,
                    prepared,
                ) in enumerate(prepared_targets, start=1):
                    if sample_limit_reached():
                        break
                    sample_quota = options.precision_samples
                    if options.max_samples is not None:
                        sample_quota = min(
                            sample_quota,
                            options.max_samples - metrics.precision_checked,
                        )
                    evaluate_prepared_statement(
                        options,
                        sample_quota,
                        metrics,
                        funnel,
                        semantics,
                        workspace,
                        dataset_source,
                        file_index,
                        statement_index,
                        candidate_index,
                        prepared,
                        library_catalog,
                    )

    summary = {
        "event": "summary",
        "dataset": options.dataset,
        "split": options.split if options.dataset == "apps" else None,
        "source": str(source_path),
        "population": population,
        "files_requested": options.files,
        "files_evaluated": metrics.files_evaluated,
        "shard_count": options.shard_count,
        "shard_index": options.shard_index,
        "statements_evaluated": metrics.evaluated,
        "evaluated": metrics.evaluated,
        "recognized": metrics.recognized,
        "recall": percentage(metrics.recall),
        "precision_accepted": metrics.precision_accepted,
        "precision_checked": metrics.precision_checked,
        "precision_requested": metrics.precision_requested,
        "precision_target": options.max_samples,
        "precision": percentage(metrics.precision),
        "precision_coverage": percentage(metrics.precision_coverage),
        "sampleable_statements": metrics.sampleable_statements,
        "sampler_failures": metrics.sampler_failures,
        "failure_reasons": dict(sorted(metrics.failure_reasons.items())),
        "diagnostic_codes": dict(sorted(metrics.diagnostic_codes.items())),
        "sampled_lengths": {
            str(length): count
            for length, count in sorted(metrics.sampled_lengths.items())
        },
        "sampled_length_offsets": {
            str(offset): count
            for offset, count in sorted(metrics.sampled_length_offsets.items())
        },
        "average_cfg_intersection_seconds": (
            0.0
            if metrics.evaluated == 0
            else metrics.average_cfg_intersection_seconds
        ),
        "funnel": dict(sorted(funnel.items())),
    }
    if options.json_lines:
        print(json.dumps(summary, sort_keys=True), flush=True)
    else:
        print(
            f"summary: files={metrics.files_evaluated}/{options.files} "
            f"statements={metrics.evaluated} "
            f"recall={summary['recall']} "
            f"({metrics.recognized}/{metrics.evaluated}) "
            f"precision={summary['precision']} "
            f"({metrics.precision_accepted}/{metrics.precision_checked}) "
            f"coverage={summary['precision_coverage']} "
            f"avg_cfg_intersection="
            f"{metrics.average_cfg_intersection_seconds:.3f}s "
            f"sampler_failures={metrics.sampler_failures}",
            flush=True,
        )
        print(f"funnel: {json.dumps(summary['funnel'], sort_keys=True)}", flush=True)
    if options.max_samples is not None:
        return 0 if sample_limit_reached() else 2
    return 0 if metrics.files_evaluated == options.files else 2


def run_self_tests() -> None:
    assert compact_cardinality(0) == "0"
    assert compact_cardinality(123_456) == "123456"
    assert "e+" in compact_cardinality(1 << 20_000)
    assert maximum_call_arity(3, assignment=False) == 0
    assert maximum_call_arity(4, assignment=False) == 1

    if hasattr(signal, "SIGALRM"):
        class OuterFixtureTimeout(Exception):
            pass

        original_alarm_handler = signal.getsignal(signal.SIGALRM)
        with wall_deadline(
            0.20,
            lambda: OuterFixtureTimeout("outer fixture deadline"),
        ):
            outer_before = signal.getitimer(signal.ITIMER_REAL)[0]
            with statement_deadline(0.08):
                time.sleep(0.01)
            outer_after = signal.getitimer(signal.ITIMER_REAL)[0]
            assert 0 < outer_after < outer_before
        assert signal.getsignal(signal.SIGALRM) == original_alarm_handler
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0

        try:
            with wall_deadline(
                0.04,
                lambda: OuterFixtureTimeout("outer fixture deadline"),
            ):
                with statement_deadline(0.20):
                    time.sleep(0.08)
        except OuterFixtureTimeout:
            pass
        else:
            raise AssertionError("a nested deadline paused its outer timer")
        assert signal.getsignal(signal.SIGALRM) == original_alarm_handler
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0

        try:
            with wall_deadline(
                0.20,
                lambda: OuterFixtureTimeout("outer fixture deadline"),
            ):
                with statement_deadline(0.03):
                    time.sleep(0.06)
        except StatementTimeout as error:
            assert "0.03s end-to-end deadline" in str(error)
        else:
            raise AssertionError("the inner statement deadline did not fire")
        assert signal.getsignal(signal.SIGALRM) == original_alarm_handler
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0

    restart_events: list[str] = []

    class RestartFixtureClient(TyLspClient):
        def __init__(
            self,
            executable: str,
            workspace: Path,
            *,
            quiet: bool = True,
        ) -> None:
            self.executable = executable
            self.workspace = workspace
            self.quiet = quiet
            self.process = cast(Any, object())
            self.stdin = cast(Any, object())
            self.stdout = cast(Any, object())
            self.next_id = 0
            self.document_uri = None
            self.document_version = 0
            self.position_encoding = "utf-8"

        def _terminate_process(self) -> None:
            restart_events.append("terminate")

        def _close_pipes(self) -> None:
            restart_events.append("close-pipes")

        def close(self) -> None:
            restart_events.append("graceful-close")

    restart_fixture = RestartFixtureClient("ty", Path.cwd())
    restart_fixture.restart()
    assert restart_events == ["terminate", "close-pipes"]

    class BrokenTyInput:
        def write(self, _value: bytes) -> int:
            raise BrokenPipeError("fixture pipe closed")

        def flush(self) -> None:
            raise AssertionError("a failed write must not be flushed")

    broken_input_client = object.__new__(TyLspClient)
    setattr(broken_input_client, "stdin", BrokenTyInput())
    try:
        broken_input_client._write({"fixture": True})
    except TyTransportError as error:
        assert isinstance(error.__cause__, BrokenPipeError)
        assert "closed its input" in str(error)
    else:
        raise AssertionError("BrokenPipeError was not classified as transport")

    class ClosedTyOutput:
        def readline(self) -> bytes:
            return b""

    class ExitedTyProcess:
        def poll(self) -> int:
            return 101

    closed_output_client = object.__new__(TyLspClient)
    setattr(closed_output_client, "stdout", ClosedTyOutput())
    setattr(closed_output_client, "process", ExitedTyProcess())
    try:
        closed_output_client._read_message()
    except TyTransportError as error:
        assert "closed its output (exit 101)" in str(error)
    else:
        raise AssertionError("server-output EOF was not classified as transport")

    retry_client = object.__new__(TyLspClient)
    restart_count = 0

    def restart_fixture_client() -> None:
        nonlocal restart_count
        restart_count += 1

    setattr(retry_client, "restart", restart_fixture_client)
    retry_metrics = RunningMetrics(
        evaluated=2,
        sampleable_statements=3,
        failure_reasons=Counter({"prior": 4}),
        sampled_lengths=Counter({7: 5}),
    )
    retry_funnel = {"evaluated_statements": 2, "prior": 6}
    retry_calls = 0
    emitted_records: list[int] = []

    def flaky_statement_attempt(
        attempt_metrics: RunningMetrics,
        attempt_funnel: dict[str, int],
    ) -> None:
        nonlocal retry_calls
        retry_calls += 1
        attempt_metrics.sampleable_statements += 1
        attempt_metrics.failure_reasons["attempt"] += 1
        attempt_metrics.sampled_lengths[9] += 1
        attempt_funnel["attempt"] = attempt_funnel.get("attempt", 0) + 1
        if retry_calls == 1:
            raise TyTransportError("fixture transport loss")
        attempt_metrics.evaluated += 1
        attempt_funnel["evaluated_statements"] += 1
        emitted_records.append(attempt_metrics.evaluated)

    retry_error = retry_ty_transport_once(
        retry_client,
        retry_metrics,
        retry_funnel,
        flaky_statement_attempt,
    )
    assert retry_error is None
    assert retry_calls == 2
    assert restart_count == 1
    assert emitted_records == [3]
    assert retry_metrics.evaluated == 3
    assert retry_metrics.sampleable_statements == 4
    assert retry_metrics.failure_reasons == Counter({"prior": 4, "attempt": 1})
    assert retry_metrics.sampled_lengths == Counter({7: 5, 9: 1})
    assert retry_funnel == {
        "evaluated_statements": 3,
        "prior": 6,
        "attempt": 1,
        "ty_transport_restarts": 1,
    }

    ordinary_client = object.__new__(TyLspClient)
    ordinary_restart_count = 0

    def restart_ordinary_client() -> None:
        nonlocal ordinary_restart_count
        ordinary_restart_count += 1

    setattr(ordinary_client, "restart", restart_ordinary_client)
    ordinary_calls = 0

    def ordinary_failure(
        _metrics: RunningMetrics,
        _funnel: dict[str, int],
    ) -> None:
        nonlocal ordinary_calls
        ordinary_calls += 1
        raise EvaluationError("ordinary evaluator failure")

    try:
        retry_ty_transport_once(
            ordinary_client,
            RunningMetrics(),
            {},
            ordinary_failure,
        )
    except EvaluationError as error:
        assert type(error) is EvaluationError
    else:
        raise AssertionError("ordinary EvaluationError was swallowed")
    assert ordinary_calls == 1
    assert ordinary_restart_count == 0

    initialization_failure_client = object.__new__(TyLspClient)

    def fail_restart_initialization() -> None:
        raise TyTransportError("fixture restart initialization failed")

    def trigger_initial_transport(
        attempt_metrics: RunningMetrics,
        attempt_funnel: dict[str, int],
    ) -> None:
        attempt_metrics.evaluated += 1
        attempt_funnel["evaluated_statements"] += 1
        raise TyTransportError("fixture initial transport loss")

    setattr(
        initialization_failure_client,
        "restart",
        fail_restart_initialization,
    )
    initialization_failure_metrics = RunningMetrics(evaluated=4)
    initialization_failure_funnel = {"evaluated_statements": 4}
    try:
        retry_ty_transport_once(
            initialization_failure_client,
            initialization_failure_metrics,
            initialization_failure_funnel,
            trigger_initial_transport,
        )
    except TyTransportError as error:
        assert str(error) == "fixture restart initialization failed"
    else:
        raise AssertionError("restart initialization failure was swallowed")
    assert initialization_failure_metrics.evaluated == 4
    assert initialization_failure_funnel == {"evaluated_statements": 4}

    repeated_client = object.__new__(TyLspClient)
    repeated_restart_count = 0

    def restart_repeated_client() -> None:
        nonlocal repeated_restart_count
        repeated_restart_count += 1

    setattr(repeated_client, "restart", restart_repeated_client)
    repeated_metrics = RunningMetrics(evaluated=8)
    repeated_funnel = {"evaluated_statements": 8}
    repeated_calls = 0

    def repeated_transport_failure(
        attempt_metrics: RunningMetrics,
        attempt_funnel: dict[str, int],
    ) -> None:
        nonlocal repeated_calls
        repeated_calls += 1
        attempt_metrics.evaluated += 1
        attempt_funnel["evaluated_statements"] += 1
        raise TyTransportError("fixture repeated transport loss")

    repeated_error = retry_ty_transport_once(
        repeated_client,
        repeated_metrics,
        repeated_funnel,
        repeated_transport_failure,
    )
    assert isinstance(repeated_error, TyTransportError)
    assert repeated_calls == 2
    assert repeated_restart_count == 2
    assert repeated_metrics.evaluated == 8
    assert repeated_funnel == {"evaluated_statements": 8}

    transport_source = "value\n"
    transport_targets = candidate_targets(
        transport_source,
        ast.parse(transport_source),
    )
    assert len(transport_targets) == 1
    transport_target = transport_targets[0]
    transport_scaffold = transport_target.hole.render("()")
    transport_prepared = PreparedTarget(
        transport_target,
        transport_scaffold,
        transport_scaffold,
    )
    transport_options = EvaluationOptions(
        dataset="apps",
        source=Path("unused.jsonl"),
        split="train",
        files=1,
        precision_samples=2,
        max_samples=None,
        seed=0,
        ty="ty",
        allow_ignores=False,
        builder=BuilderOptions(),
        max_dfa_states=100,
        statement_timeout=60.0,
        json_lines=True,
        show_samples=False,
    )
    transport_dataset_source = DatasetSource(
        member="APPS/train/fixture/solution_0000.py",
        data=transport_source.encode(),
        dataset="apps",
        split="train",
        problem_id="fixture",
        solution_index=0,
    )
    terminal_client = object.__new__(TyLspClient)
    terminal_open_calls = 0
    terminal_restart_count = 0

    def terminal_open(document_uri: str, _text: str) -> None:
        nonlocal terminal_open_calls
        terminal_open_calls += 1
        if terminal_open_calls <= 2:
            raise TyTransportError(f"fixture loss {terminal_open_calls}")
        setattr(terminal_client, "document_uri", document_uri)

    def terminal_diagnostics() -> list[dict[str, object]]:
        return [{"severity": 1, "message": "fixture semantic error"}]

    def restart_terminal_client() -> None:
        nonlocal terminal_restart_count
        terminal_restart_count += 1
        setattr(terminal_client, "document_uri", None)

    setattr(terminal_client, "open", terminal_open)
    setattr(terminal_client, "diagnostics", terminal_diagnostics)
    setattr(terminal_client, "restart", restart_terminal_client)
    terminal_metrics = RunningMetrics()
    terminal_funnel: dict[str, int] = defaultdict(int)
    terminal_output = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = terminal_output
        evaluate_prepared_statement(
            transport_options,
            2,
            terminal_metrics,
            terminal_funnel,
            terminal_client,
            Path.cwd(),
            transport_dataset_source,
            1,
            1,
            1,
            transport_prepared,
            object.__new__(LibraryCatalog),
        )
        # The second target reaches the fresh client created after the
        # terminal failure and completes as an ordinary construction failure.
        evaluate_prepared_statement(
            transport_options,
            1,
            terminal_metrics,
            terminal_funnel,
            terminal_client,
            Path.cwd(),
            transport_dataset_source,
            1,
            2,
            2,
            transport_prepared,
            object.__new__(LibraryCatalog),
        )
    finally:
        sys.stdout = original_stdout
    terminal_records = [
        json.loads(line) for line in terminal_output.getvalue().splitlines()
    ]
    assert len(terminal_records) == 2
    transport_record, continuation_record = terminal_records
    assert sum(
        str(record["sampler_error"]).startswith(
            "ty transport failed after retry:"
        )
        for record in terminal_records
    ) == 1
    assert set(transport_record) == set(continuation_record)
    assert transport_record["event"] == "statement"
    assert transport_record["index"] == 1
    assert transport_record["recognized"] is False
    assert transport_record["sample_requested"] == 2
    assert transport_record["sample_checked"] == 0
    assert transport_record["sample_accepted"] == 0
    assert transport_record["ground_truth"] == ["value"]
    assert str(transport_record["sampler_error"]).startswith(
        "ty transport failed after retry: fixture loss 2"
    )
    assert continuation_record["index"] == 2
    assert continuation_record["sampler_error"] == (
        "semantic scaffold contains ty errors"
    )
    assert terminal_open_calls == 3
    assert terminal_restart_count == 2
    assert terminal_metrics.evaluated == 2
    assert terminal_metrics.recognized == 0
    assert terminal_metrics.precision_accepted == 0
    assert terminal_metrics.precision_requested == 3
    assert terminal_metrics.precision_checked == 0
    assert terminal_funnel["evaluated_statements"] == 2
    assert terminal_funnel["ty_transport_failures"] == 1
    assert terminal_funnel["semantic_ablation_disagreement"] == 1
    assert terminal_funnel.get("ty_transport_restarts", 0) == 0

    if hasattr(signal, "SIGALRM"):
        deadline_options = replace(
            transport_options,
            statement_timeout=0.02,
        )
        deadline_client = object.__new__(TyLspClient)
        deadline_metrics = RunningMetrics()
        deadline_funnel: dict[str, int] = defaultdict(int)
        deadline_open_calls = 0
        deadline_restart_count = 0

        def deadline_open(document_uri: str, _text: str) -> None:
            nonlocal deadline_open_calls
            deadline_open_calls += 1
            if deadline_open_calls == 1:
                raise TyTransportError("fixture transport before slow restart")
            setattr(deadline_client, "document_uri", document_uri)

        def deadline_diagnostics() -> list[dict[str, object]]:
            return [{"severity": 1, "message": "fixture semantic error"}]

        def restart_deadline_client() -> None:
            nonlocal deadline_restart_count
            deadline_restart_count += 1
            setattr(deadline_client, "document_uri", None)
            if deadline_restart_count == 1:
                # Mutations made anywhere in a timed attempt, including its
                # transport recovery, must not leak into the terminal row.
                deadline_metrics.sampleable_statements += 1
                deadline_funnel["attempt_only"] += 1
                time.sleep(0.06)

        setattr(deadline_client, "open", deadline_open)
        setattr(deadline_client, "diagnostics", deadline_diagnostics)
        setattr(deadline_client, "restart", restart_deadline_client)
        deadline_output = io.StringIO()
        original_stdout = sys.stdout
        original_alarm_handler = signal.getsignal(signal.SIGALRM)
        try:
            sys.stdout = deadline_output
            evaluate_prepared_statement(
                deadline_options,
                3,
                deadline_metrics,
                deadline_funnel,
                deadline_client,
                Path.cwd(),
                transport_dataset_source,
                1,
                1,
                1,
                transport_prepared,
                object.__new__(LibraryCatalog),
            )
            evaluate_prepared_statement(
                deadline_options,
                1,
                deadline_metrics,
                deadline_funnel,
                deadline_client,
                Path.cwd(),
                transport_dataset_source,
                1,
                2,
                2,
                transport_prepared,
                object.__new__(LibraryCatalog),
            )
        finally:
            sys.stdout = original_stdout
        deadline_records = [
            json.loads(line) for line in deadline_output.getvalue().splitlines()
        ]
        assert len(deadline_records) == 2
        timeout_record, post_timeout_record = deadline_records
        assert set(timeout_record) == set(post_timeout_record)
        assert timeout_record["index"] == 1
        assert timeout_record["recognized"] is False
        assert timeout_record["sample_requested"] == 3
        assert timeout_record["sample_checked"] == 0
        assert timeout_record["sample_accepted"] == 0
        assert timeout_record["sampler_error"] == (
            "statement evaluation exceeded 0.02s end-to-end deadline"
        )
        assert post_timeout_record["index"] == 2
        assert post_timeout_record["sampler_error"] == (
            "semantic scaffold contains ty errors"
        )
        assert deadline_open_calls == 2
        assert deadline_restart_count == 2
        assert deadline_metrics.evaluated == 2
        assert deadline_metrics.precision_requested == 4
        assert deadline_metrics.precision_checked == 0
        assert deadline_metrics.sampleable_statements == 0
        assert deadline_funnel["evaluated_statements"] == 2
        assert deadline_funnel["statement_timeouts"] == 1
        assert deadline_funnel["semantic_ablation_disagreement"] == 1
        assert deadline_funnel.get("attempt_only", 0) == 0
        assert deadline_funnel.get("ty_transport_restarts", 0) == 0
        assert deadline_funnel.get("ty_transport_failures", 0) == 0
        assert signal.getsignal(signal.SIGALRM) == original_alarm_handler
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0

    assert default_dataset_source("apps", "test") == (
        DEFAULT_APPS_DIRECTORY / "test.jsonl"
    )
    assert default_dataset_source("codenet", "train") == (
        DEFAULT_CODENET_ARCHIVE
    )
    try:
        resolved_dataset_source("apps", Path("train.jsonl"), "test")
    except EvaluationError as error:
        assert "use --split train" in str(error)
    else:
        raise AssertionError("mismatched explicit APPS split was accepted")
    try:
        resolved_dataset_source("apps", Path("archive.tar.gz"), "test")
    except EvaluationError as error:
        assert "--dataset codenet" in str(error)
    else:
        raise AssertionError("CodeNet archive was accepted as APPS JSONL")
    assert apps_member_name("test", "x/y", 3) == (
        "APPS/test/x%2Fy/solution_0003.py"
    )
    with tempfile.TemporaryDirectory(prefix="api2cfg-source-test-") as directory:
        fixture_directory = Path(directory)
        apps_path = fixture_directory / "test.jsonl"
        apps_rows = (
            {
                "id": 7,
                "solutions": json.dumps(["first = 1\n", "second = 2\n"]),
                "difficulty": "introductory",
                "url": "https://example.invalid/7",
            },
            {
                "id": "x/y",
                "solutions": ["third = 3\n"],
                "difficulty": None,
            },
            {
                "id": "no-solutions",
                "solutions": "",
                "difficulty": "competition",
            },
        )
        apps_path.write_text(
            "".join(json.dumps(row) + "\n" for row in apps_rows),
            encoding="utf-8",
        )
        apps_sources = list(
            iter_dataset_sources("apps", fixture_directory, "test")
        )
        assert [source.solution_index for source in apps_sources] == [0, 1, 0]
        assert [source.problem_id for source in apps_sources] == [7, 7, "x/y"]
        assert apps_sources[0].data == b"first = 1\n"
        assert apps_sources[0].difficulty == "introductory"
        assert apps_sources[0].url == "https://example.invalid/7"
        assert apps_sources[2].member == "APPS/test/x%2Fy/solution_0000.py"
        assert [
            source.member
            for source in iter_dataset_sources(
                "apps", apps_path, "test", shard_count=2, shard_index=0
            )
        ] == [apps_sources[0].member, apps_sources[2].member]
        assert [
            source.member
            for source in iter_dataset_sources(
                "apps", apps_path, "test", shard_count=2, shard_index=1
            )
        ] == [apps_sources[1].member]

        archive_path = fixture_directory / "codenet.tar.gz"
        submission_name = "Project_CodeNet/data/p00001/Python/s000000001.py"
        with tarfile.open(archive_path, "w:gz") as archive:
            submission = b"answer = 42\n"
            submission_info = tarfile.TarInfo(submission_name)
            submission_info.size = len(submission)
            archive.addfile(submission_info, io.BytesIO(submission))
            ignored = b"not Python"
            ignored_info = tarfile.TarInfo(
                "Project_CodeNet/data/p00001/Java/s000000001.java"
            )
            ignored_info.size = len(ignored)
            archive.addfile(ignored_info, io.BytesIO(ignored))
        codenet_sources = list(
            iter_dataset_sources("codenet", archive_path, "test")
        )
        assert codenet_sources == [
            DatasetSource(
                member=submission_name,
                data=b"answer = 42\n",
                dataset="codenet",
            )
        ]

    default_args = parse_arguments([])
    assert default_args.dataset == "apps"
    assert default_args.split == "test"
    assert default_args.source is None
    default_options = evaluation_options(default_args)
    assert default_options.source == DEFAULT_APPS_DIRECTORY / "test.jsonl"
    assert default_options.max_dfa_states == DEFAULT_MAX_DFA_STATES

    private_completion_items = simplify_completions(
        (
            {
                "label": "_private_library_name",
                "detail": "int",
                "kind": LSP_FUNCTION,
            },
            {
                "label": "__private_library_name",
                "detail": "int",
                "kind": LSP_FUNCTION,
            },
            {
                "label": "public_library_name",
                "detail": "int",
                "kind": LSP_FUNCTION,
            },
        )
    )
    assert [item.label for item in private_completion_items] == [
        "public_library_name"
    ]

    def raw_completion(
        label: str, detail: str, kind: int | None
    ) -> dict[str, object]:
        return {"label": label, "detail": detail, "kind": kind}

    completion_passes: Iterator[
        tuple[list[dict[str, object]], bool]
    ] = iter(
        (
            (
                [raw_completion("alpha", "Unknown", LSP_FUNCTION)],
                True,
            ),
            (
                [
                    raw_completion("beta", "int", None),
                    raw_completion("alpha", "str", None),
                ],
                True,
            ),
            (
                [
                    raw_completion("beta", "int", None),
                    raw_completion("alpha", "str", LSP_FUNCTION),
                ],
                True,
            ),
            (
                [
                    raw_completion("beta", "int", None),
                    raw_completion("alpha", "str", LSP_FUNCTION),
                ],
                True,
            ),
        )
    )
    completion_retriggers: list[bool] = []

    def fetch_completion_pass(
        retrigger: bool,
    ) -> tuple[list[dict[str, object]], bool]:
        completion_retriggers.append(retrigger)
        return next(completion_passes)

    progressive, progressive_was_incomplete = progressive_completions(
        fetch_completion_pass
    )
    assert progressive_was_incomplete
    assert completion_retriggers == [False, True, True, True]
    assert progressive == [
        Completion("alpha", "str", LSP_FUNCTION),
        Completion("beta", "int", None),
    ]

    class ReceiverPriorityProbe(SemanticProbe):
        def __init__(self) -> None:
            self.queries: list[str] = []

        def members(self, expression: str) -> tuple[list[Completion], bool]:
            self.queries.append(expression)
            if expression == "important":
                return [Completion("child", "list[int]", None)], False
            if expression == "important.child":
                return [Completion("leaf", "int", None)], False
            return [Completion("irrelevant_member", "int", None)], False

    receiver_probe = ReceiverPriorityProbe()
    receiver_builder = GrammarBuilder(
        receiver_probe,
        frozenset({"important"}),
        BuilderOptions(max_receiver_types=2, member_depth=2),
        {},
    )
    # Enqueue the same type through an irrelevant representative first.  The
    # contextual representative must replace it before bounded processing.
    receiver_builder.queue_receiver("SharedType", "aaa", 0)
    receiver_builder.queue_receiver("SharedType", "important", 0)
    receiver_builder.add_members()
    assert receiver_probe.queries == ["important", "important.child"]
    assert receiver_builder.stats.receiver_types == 2
    assert {"child", "leaf"} <= receiver_builder.grammar.terminals
    assert "irrelevant_member" not in receiver_builder.grammar.terminals

    grouping_builder = GrammarBuilder(
        SemanticProbe.__new__(SemanticProbe),
        frozenset(),
        BuilderOptions(),
        {},
    )
    int_nonterminal = grouping_builder.add_expression(
        "int", (Terminal("0"),)
    )
    grouping_builder.grammar.add(
        grouping_builder.grammar.start, Nonterminal(int_nonterminal)
    )
    grouping_builder.add_redundant_grouping()
    grouping_compiled = UnitAwareBinaryGrammar(grouping_builder.grammar)
    assert grouping_compiled.recognizes(("0",))
    assert grouping_compiled.recognizes(("(", "0", ")"))
    assert grouping_compiled.recognizes(("(", "(", "0", ")", ")"))
    assert not grouping_compiled.recognizes(("(", ")"))
    assert not grouping_compiled.recognizes(("(", "0"))

    collection_builder = GrammarBuilder(
        SemanticProbe.__new__(SemanticProbe),
        frozenset(),
        BuilderOptions(max_call_arity=3),
        {},
    )
    collection_builder.add_expression("int", (Terminal("i"),))
    collection_builder.add_expression("float", (Terminal("f"),))
    collection_builder.add_expression("str", (Terminal("s"),))
    collection_builder.add_expression("list[int]", (Terminal("items"),))
    collection_builder.add_expression(
        "tuple[int, str]", (Terminal("pair"),)
    )
    collection_builder.add_expression(
        "dict[str, int]", (Terminal("mapping"),)
    )
    collection_builder.add_typed_subscripts()
    collection_builder.add_typed_binary_operations()
    collection_builder.add_typed_unary_operations()
    collection_builder.add_redundant_grouping()
    collection_grammar, collection_stats = collection_builder.finish()
    collection_compiled = UnitAwareBinaryGrammar(collection_grammar)
    bracket_productions = [
        production
        for production in collection_grammar.productions
        if Terminal("[") in production.rhs
    ]
    assert bracket_productions
    for production in bracket_productions:
        # A bracket may follow a typed receiver as an index or slice trailer;
        # no reachable production may open a list display.
        assert production.rhs.index(Terminal("[")) == 1, production
        assert isinstance(production.rhs[0], Nonterminal), production
    for collection_word in (
        ("items", "[", "i", "]"),
        ("items", "[", ":", "]"),
        ("items", "[", "i", ":", "i", ":", "i", "]"),
        ("mapping", "[", "s", "]"),
        ("items", "+", "items"),
        ("pair", "[", ":", "]"),
        ("i", "+", "i", "*", "i"),
        ("(", "i", "+", "i", ")", "*", "i"),
        ("i", "**", "i", "**", "i"),
        ("-", "i", "**", "i"),
        ("i", "**", "-", "i"),
    ):
        assert collection_compiled.recognizes(collection_word), collection_word
    for rejected_collection_word in (
        ("items", "[", "s", "]"),
        ("items", "[", "f", "]"),
        ("mapping", "[", "i", "]"),
        ("mapping", "[", ":", "]"),
        ("(", "-", "items", ")", "[", "i", "]"),
        ("[", "]"),
        ("[", "i", "]"),
        ("[", "i", ",", "s", "]"),
        (FRESH_TOKEN, "=", "[", "i", "]"),
        ("[", "i", "]", "+", "[", "f", "]"),
    ):
        assert not collection_compiled.recognizes(
            rejected_collection_word
        ), rejected_collection_word
    assert collection_stats.binary_operation_productions > 0
    assert collection_stats.subscript_productions > 0
    assert collection_stats.slice_productions > 0

    class TypedSubscriptReceiverProbe(SemanticProbe):
        """Minimal ty-shaped fixture for members reached through indexing."""

        def __init__(self) -> None:
            self.excluded_names = frozenset()
            self.member_queries: list[str] = []
            self.signature_queries: list[str] = []

        def scope(self) -> tuple[list[Completion], bool]:
            return [Completion("items", "list[Item]", None)], False

        def accepts_expression(self, expression: str) -> bool:
            return expression == "items"

        def hover_expression(self, expression: str) -> str | None:
            return "list[Item]" if expression == "items" else None

        def members(self, expression: str) -> tuple[list[Completion], bool]:
            self.member_queries.append(expression)
            if expression == "items[0]":
                return [
                    Completion(
                        "foo",
                        "bound method Item.foo() -> int",
                        LSP_METHOD,
                    ),
                    Completion("value", "int", None),
                ], False
            return [], False

        def signatures(self, expression: str) -> list[str]:
            self.signature_queries.append(expression)
            return ["() -> int"] if expression == "items[0].foo" else []

    indexed_receiver_probe = TypedSubscriptReceiverProbe()
    indexed_receiver_builder = GrammarBuilder(
        indexed_receiver_probe,
        frozenset({"items"}),
        BuilderOptions(
            max_call_arity=0,
            member_depth=0,
            max_receiver_types=3,
        ),
        {},
    )
    indexed_receiver_grammar, indexed_receiver_stats = (
        indexed_receiver_builder.build()
    )
    indexed_receiver_compiled = UnitAwareBinaryGrammar(
        indexed_receiver_grammar
    )
    assert indexed_receiver_probe.member_queries == [
        "items",
        '""',
        "items[0]",
    ]
    assert indexed_receiver_probe.signature_queries == ["items[0].foo"]
    for indexed_member_word in (
        ("items", "[", "0", "]", ".", "foo", "(", ")"),
        ("items", "[", "0", "]", ".", "value", "+", "0"),
    ):
        assert indexed_receiver_compiled.recognizes(
            indexed_member_word
        ), indexed_member_word
    assert not indexed_receiver_compiled.recognizes(
        ("items", "[", '""', "]", ".", "foo", "(", ")")
    )
    assert indexed_receiver_stats.receiver_types == 3
    assert indexed_receiver_stats.callables == 1

    capped_indexed_probe = TypedSubscriptReceiverProbe()
    capped_indexed_builder = GrammarBuilder(
        capped_indexed_probe,
        frozenset({"items"}),
        BuilderOptions(
            max_call_arity=0,
            member_depth=0,
            max_receiver_types=2,
        ),
        {},
    )
    capped_indexed_grammar, _capped_indexed_stats = (
        capped_indexed_builder.build()
    )
    assert capped_indexed_probe.member_queries == ["items", '""']
    assert "foo" not in capped_indexed_grammar.terminals

    additive_left = Grammar(start="START")
    additive_left.add("E:A", Terminal("a"))
    additive_left.add("E:B", Terminal("b"))
    additive_left.add("E:C", Terminal("c"))
    additive_left.add(
        "E:AB", Nonterminal("E:A"), Terminal("-"), Nonterminal("E:B")
    )
    additive_left.add(
        "E:ABC", Nonterminal("E:AB"), Terminal("-"), Nonterminal("E:C")
    )
    additive_left.add("START", Nonterminal("E:ABC"))
    assert UnitAwareBinaryGrammar(
        enforce_expression_precedence(additive_left)
    ).recognizes(("a", "-", "b", "-", "c"))

    additive_right = Grammar(start="START")
    additive_right.add("E:A", Terminal("a"))
    additive_right.add("E:B", Terminal("b"))
    additive_right.add("E:C", Terminal("c"))
    additive_right.add(
        "E:BC", Nonterminal("E:B"), Terminal("-"), Nonterminal("E:C")
    )
    additive_right.add(
        "E:ABC", Nonterminal("E:A"), Terminal("-"), Nonterminal("E:BC")
    )
    additive_right.add("START", Nonterminal("E:ABC"))
    assert not UnitAwareBinaryGrammar(
        enforce_expression_precedence(additive_right)
    ).recognizes(("a", "-", "b", "-", "c"))

    power_right = Grammar(start="START")
    power_right.add("E:A", Terminal("a"))
    power_right.add("E:B", Terminal("b"))
    power_right.add("E:C", Terminal("c"))
    power_right.add(
        "E:BC", Nonterminal("E:B"), Terminal("**"), Nonterminal("E:C")
    )
    power_right.add(
        "E:ABC", Nonterminal("E:A"), Terminal("**"), Nonterminal("E:BC")
    )
    power_right.add("START", Nonterminal("E:ABC"))
    assert UnitAwareBinaryGrammar(
        enforce_expression_precedence(power_right)
    ).recognizes(("a", "**", "b", "**", "c"))

    power_left = Grammar(start="START")
    power_left.add("E:A", Terminal("a"))
    power_left.add("E:B", Terminal("b"))
    power_left.add("E:C", Terminal("c"))
    power_left.add(
        "E:AB", Nonterminal("E:A"), Terminal("**"), Nonterminal("E:B")
    )
    power_left.add(
        "E:ABC", Nonterminal("E:AB"), Terminal("**"), Nonterminal("E:C")
    )
    power_left.add("START", Nonterminal("E:ABC"))
    assert not UnitAwareBinaryGrammar(
        enforce_expression_precedence(power_left)
    ).recognizes(("a", "**", "b", "**", "c"))

    # Python trailers bind more tightly than prefix unary operators.  Keep a
    # distinct primary layer so a typed member row cannot reinterpret ``~0``
    # as the receiver in the source word ``~0.to_bytes``.  Parentheses make
    # that receiver explicit and must restore the postfix operation.
    precedence_grammar = Grammar(start="START")
    precedence_grammar.add("E:int", Terminal("0"))
    precedence_grammar.add(
        "E:int", Terminal("~"), Nonterminal("E:int")
    )
    precedence_grammar.add(
        "E:int", Terminal("("), Nonterminal("E:int"), Terminal(")")
    )
    precedence_grammar.add("E:complex", Terminal("x"))
    precedence_grammar.add(
        "E:complex", Terminal("-"), Nonterminal("E:complex")
    )
    precedence_grammar.add(
        "E:complex",
        Terminal("("),
        Nonterminal("E:complex"),
        Terminal(")"),
    )
    precedence_grammar.add(
        "E:float",
        Nonterminal("E:complex"),
        Terminal("."),
        Terminal("real"),
    )
    precedence_grammar.add(
        "E:float", Terminal("-"), Nonterminal("E:float")
    )
    precedence_grammar.add(
        "E:to_bytes",
        Nonterminal("E:int"),
        Terminal("."),
        Terminal("to_bytes"),
    )
    precedence_grammar.add(
        "E:bytes",
        Nonterminal("E:to_bytes"),
        Terminal("("),
        Terminal(")"),
    )
    precedence_grammar.add("E:factory", Terminal("factory"))
    precedence_grammar.add(
        "E:object",
        Nonterminal("E:factory"),
        Terminal("("),
        Terminal(")"),
    )
    precedence_grammar.add(
        "E:child",
        Nonterminal("E:object"),
        Terminal("."),
        Terminal("child"),
    )
    precedence_grammar.add(
        "E:method",
        Nonterminal("E:child"),
        Terminal("."),
        Terminal("method"),
    )
    precedence_grammar.add(
        "E:result",
        Nonterminal("E:method"),
        Terminal("("),
        Terminal(")"),
    )
    # A callable that is also invertible is artificial but isolates call
    # precedence: ``~f()`` is unary-of-call, never call-of-unary.
    precedence_grammar.add("E:callable_invertible", Terminal("f"))
    precedence_grammar.add(
        "E:callable_invertible",
        Terminal("~"),
        Nonterminal("E:callable_invertible"),
    )
    precedence_grammar.add(
        "E:callable_invertible",
        Terminal("("),
        Nonterminal("E:callable_invertible"),
        Terminal(")"),
    )
    precedence_grammar.add(
        "E:called",
        Nonterminal("E:callable_invertible"),
        Terminal("("),
        Terminal(")"),
    )
    # Context-validated dynamic operations can be exact terminal rows rather
    # than the usual ``op E`` shape.  They are still unary expressions and
    # must not become primary receivers merely because their RHS is longer.
    precedence_grammar.add(DYNAMIC_NONTERMINAL, Terminal("dynamic"))
    precedence_grammar.add(
        DYNAMIC_NONTERMINAL, Terminal("-"), Terminal("dynamic")
    )
    precedence_grammar.add(
        DYNAMIC_NONTERMINAL,
        Terminal("("),
        Nonterminal(DYNAMIC_NONTERMINAL),
        Terminal(")"),
    )
    precedence_grammar.add(
        "E:Any", Nonterminal(DYNAMIC_NONTERMINAL)
    )
    precedence_grammar.add(
        "E:dynamic_called",
        Nonterminal("E:Any"),
        Terminal("("),
        Terminal(")"),
    )
    # Precomputed library fragments use the same unbinarized E-receiver
    # shapes as live rows; exercise them without relying on a live LSP.
    precedence_grammar.add("E:<module 'lib'>", Terminal("lib"))
    precedence_grammar.add(
        "E:lib.sqrt",
        Nonterminal("E:<module 'lib'>"),
        Terminal("."),
        Terminal("sqrt"),
    )
    precedence_grammar.add(
        "E:lib.result",
        Nonterminal("E:lib.sqrt"),
        Terminal("("),
        Nonterminal("A:int"),
        Terminal(")"),
    )
    precedence_grammar.add("A:int", Nonterminal("E:int"))
    for expression in (
        "E:int",
        "E:float",
        "E:to_bytes",
        "E:bytes",
        "E:result",
        "E:called",
        "E:dynamic_called",
        "E:lib.result",
    ):
        precedence_grammar.add("START", Nonterminal(expression))
    precedence_compiled = UnitAwareBinaryGrammar(
        enforce_postfix_precedence(precedence_grammar)
    )
    assert precedence_compiled.recognizes(("0",))
    assert DerivationCounter(precedence_compiled, 1).parse_count(("0",)) == 1
    assert precedence_compiled.recognizes(("-", "x", ".", "real"))
    assert precedence_compiled.recognizes(
        ("(", "-", "x", ")", ".", "real")
    )
    assert not precedence_compiled.recognizes(
        ("~", "0", ".", "to_bytes")
    )
    assert precedence_compiled.recognizes(
        ("(", "~", "0", ")", ".", "to_bytes")
    )
    assert DerivationCounter(precedence_compiled, 6).parse_count(
        ("(", "~", "0", ")", ".", "to_bytes")
    ) == 1
    assert not precedence_compiled.recognizes(
        ("~", "0", ".", "to_bytes", "(", ")")
    )
    assert precedence_compiled.recognizes(
        ("(", "~", "0", ")", ".", "to_bytes", "(", ")")
    )
    assert precedence_compiled.recognizes(
        (
            "factory", "(", ")", ".", "child", ".", "method", "(", ")",
        )
    )
    assert not precedence_compiled.recognizes(("~", "f", "(", ")"))
    assert precedence_compiled.recognizes(
        ("(", "~", "f", ")", "(", ")")
    )
    assert not precedence_compiled.recognizes(
        ("-", "dynamic", "(", ")")
    )
    assert precedence_compiled.recognizes(
        ("(", "-", "dynamic", ")", "(", ")")
    )
    assert precedence_compiled.recognizes(
        ("lib", ".", "sqrt", "(", "0", ")")
    )

    class DynamicMemberProbe(SemanticProbe):
        def members(self, expression: str) -> tuple[list[Completion], bool]:
            assert expression == "owner"
            return [
                Completion("any_value", "Any", None),
                Completion("known_text", "str", None),
                Completion("unknown_value", "Unknown", None),
                Completion("value", "Any", None),
            ], False

        def accepts_expression(self, expression: str) -> bool:
            return expression in {
                "-owner.any_value",
                "-owner.unknown_value",
            }

    dynamic_member_builder = GrammarBuilder(
        DynamicMemberProbe.__new__(DynamicMemberProbe),
        frozenset({"owner"}),
        BuilderOptions(max_receiver_types=1, member_depth=0),
        {},
    )
    owner_nonterminal = dynamic_member_builder.add_expression(
        "Owner", (Terminal("owner"),)
    )
    dynamic_member_builder.queue_receiver("Owner", "owner", 0)
    dynamic_member_builder.add_members()
    dynamic_member_builder.add_dynamic_operations()
    dynamic_member_builder.add_redundant_grouping()
    dynamic_member_builder.grammar.add(
        dynamic_member_builder.grammar.start,
        Nonterminal(type_nonterminal("Any")),
    )
    dynamic_member_compiled = UnitAwareBinaryGrammar(
        dynamic_member_builder.grammar
    )
    assert dynamic_member_compiled.recognizes(
        ("-", "owner", ".", "any_value")
    )
    assert dynamic_member_compiled.recognizes(
        ("-", "owner", ".", "unknown_value")
    )
    assert not dynamic_member_compiled.recognizes(
        ("-", "owner", ".", "known_text")
    )
    assert not dynamic_member_compiled.recognizes(
        ("-", "-", "owner", ".", "any_value")
    )
    assert not dynamic_member_compiled.recognizes(
        ("owner", ".", "any_value", "(", ")")
    )
    assert not dynamic_member_compiled.recognizes(
        ("owner", ".", "any_value", ".", "known_text")
    )
    assert dynamic_member_compiled.recognizes(
        ("owner", ".", "value", "+", "0")
    )
    dynamic_grouping_sampler = DerivationCounter(
        dynamic_member_compiled, 5
    )
    assert dynamic_grouping_sampler.parse_count(
        ("(", "owner", ".", "any_value", ")")
    ) == 1
    assert owner_nonterminal == type_nonterminal("Owner")

    class GradualBinaryProbe(SemanticProbe):
        def accepts_expression(self, expression: str) -> bool:
            return True

    gradual_binary_builder = GrammarBuilder(
        GradualBinaryProbe.__new__(GradualBinaryProbe),
        frozenset({"mystery", "other"}),
        BuilderOptions(
            max_call_arity=0,
            max_dynamic_composition_depth=1,
        ),
        {},
    )
    gradual_binary_builder.add_literals()
    gradual_binary_builder.add_expression(
        "int & ~AlwaysFalsy", (Terminal("nonzero"),)
    )
    gradual_binary_builder.grammar.add(
        DYNAMIC_NONTERMINAL, Terminal("mystery")
    )
    gradual_binary_builder.grammar.add(
        DYNAMIC_NONTERMINAL, Terminal("other")
    )
    gradual_binary_builder.grammar.add(
        type_nonterminal("Unknown"), Nonterminal(DYNAMIC_NONTERMINAL)
    )
    gradual_binary_builder.representatives["Unknown"] = "mystery"
    gradual_binary_builder.dynamic_representatives["Unknown"].add("mystery")
    gradual_binary_builder.dynamic_representatives["Unknown"].add("other")
    gradual_binary_builder.dynamic_scope_representatives["Unknown"].add(
        "mystery"
    )
    gradual_binary_builder.add_typed_binary_operations()
    gradual_binary_builder.add_typed_unary_operations()
    gradual_binary_builder.add_dynamic_operations()
    gradual_binary_builder.add_redundant_grouping()
    gradual_binary_grammar, gradual_binary_stats = (
        gradual_binary_builder.finish()
    )
    gradual_binary_compiled = UnitAwareBinaryGrammar(
        gradual_binary_grammar
    )
    for gradual_word in (
        ("mystery", "+", "0"),
        ("0", "+", "mystery"),
        ("mystery", "**", "-", "0"),
        ("mystery", "[", "0", "]"),
        ("mystery", "[", ":", "]"),
        ("mystery", "[", "nonzero", ":", "]"),
        ("mystery", "[", "mystery", ":", "]"),
        ("mystery", "[", "0", "]", "+", "0"),
        ("(", "mystery", "+", "0", ")"),
        ("(", "mystery", "[", "0", "]", ")"),
    ):
        assert gradual_binary_compiled.recognizes(gradual_word), gradual_word
    assert not gradual_binary_compiled.recognizes(
        ("mystery", "&", "0")
    )
    assert not gradual_binary_compiled.recognizes(
        ("mystery", "+", "other")
    )
    assert not gradual_binary_compiled.recognizes(
        ("mystery", "+", "[", "0", "]")
    )
    assert not gradual_binary_compiled.recognizes(
        ("mystery", "[", "0", "]", "[", "0", "]")
    )
    nested_dynamic_slice_bound: tuple[str, ...] = (
        "mystery",
        "[",
        ":",
        "]",
    )
    for _nested_depth in range(2, 6):
        nested_dynamic_slice_bound = (
            "mystery",
            "[",
            *nested_dynamic_slice_bound,
            ":",
            "]",
        )
        assert not gradual_binary_compiled.recognizes(
            nested_dynamic_slice_bound
        )
    assert gradual_binary_stats.binary_operation_productions > 0
    gradual_binary_index = ShortlexDFAIndex(
        gradual_binary_compiled, 20_000
    )
    gradual_binary_index.ensure_rank(SAMPLE_RANK_LIMIT - 1)
    assert gradual_binary_index.state_count < 20_000

    class LocalUnderscoreProbe(SemanticProbe):
        """Minimal semantic fixture for source-local underscore bindings."""

        def __init__(self) -> None:
            self.excluded_names = frozenset()

        def scope(self) -> tuple[list[Completion], bool]:
            # LSP completion filtering deliberately omits private names.  The
            # builder must recover only names known to occur in the source.
            return [], False

        def accepts_expression(self, expression: str) -> bool:
            return expression in {"_helper", "__main__", "__starting_point"}

        def hover_expression(self, expression: str) -> str | None:
            if not self.accepts_expression(expression):
                return None
            return f"def {expression}() -> None"

        def members(self, expression: str) -> tuple[list[Completion], bool]:
            return [], False

        def signatures(self, expression: str) -> list[str]:
            if self.accepts_expression(expression):
                return ["() -> None"]
            return []

    underscore_source = (
        "def _helper():\n"
        "    pass\n"
        "def __main__():\n"
        "    pass\n"
        "def __starting_point():\n"
        "    pass\n"
        "__starting_point()\n"
        "__main__()\n"
    )
    underscore_ids = source_identifiers(underscore_source)
    assert {"_helper", "__main__", "__starting_point"} <= underscore_ids
    underscore_builder = GrammarBuilder(
        LocalUnderscoreProbe(),
        underscore_ids,
        BuilderOptions(max_call_arity=0),
        {},
    )
    underscore_grammar, _underscore_stats = underscore_builder.build()
    underscore_compiled = UnitAwareBinaryGrammar(underscore_grammar)
    assert underscore_compiled.recognizes(("__starting_point", "(", ")"))
    assert underscore_compiled.recognizes(("__main__", "(", ")"))
    assert "_private_library_name" not in underscore_grammar.terminals
    assert "__private_library_name" not in underscore_grammar.terminals

    module_type = "<module 'fixture_lib'>"
    encoded_module_type = encode_library_nonterminal(
        type_nonterminal(module_type)
    )
    fixture_cfg = "\n".join(
        (
            "# api2cfg-python-library-cfg: 1",
            "# module: fixture_lib",
            f"# module-type: {module_type}",
            "# local-assignability-complete: true",
            f"# local-assignability-version: {ASSIGNABILITY_RELATION_VERSION}",
            "# local-assignability-actuals: "
            '["E:%3Cbuilt-in%20function%20answer%3E","E:int",'
            '"E:%3Cmodule%20%27fixture_lib.sub%27%3E"]',
            '# local-assignability-expecteds: ["A:int"]',
            "# local-assignability-pairs: 3",
            "# local-assignability-links: 1",
            f"E:%3Cbuilt-in%20function%20answer%3E -> "
            f"{encoded_module_type} . answer",
            "E:int -> E:%3Cbuilt-in%20function%20answer%3E ( A:int )",
            "E:%3Cmodule%20%27fixture_lib.sub%27%3E -> "
            f"{encoded_module_type} . sub",
            "E:int -> E:%3Cmodule%20%27fixture_lib.sub%27%3E . value",
            "A:int -> E:int",
        )
    )
    fixture_artifact = parse_library_cfg_text(
        fixture_cfg, Path("fixture_lib.cfg")
    )
    assert fixture_artifact.module == "fixture_lib"
    assert type_nonterminal(module_type) in fixture_artifact.grammar.nonterminals
    assert fixture_artifact.expected_types == frozenset({"int"})
    assert fixture_artifact.local_assignability_complete
    assert fixture_artifact.local_actual_types == frozenset(
        {"<built-in function answer>", "int", "<module 'fixture_lib.sub'>"}
    )
    assert fixture_artifact.local_expected_types == frozenset({"int"})
    assert fixture_artifact.exports == frozenset(
        {
            "E:<built-in function answer>",
            "E:<module 'fixture_lib.sub'>",
            "E:int",
        }
    )
    assert fixture_artifact.export_types == {
        "answer": "E:<built-in function answer>",
        "sub": "E:<module 'fixture_lib.sub'>",
        "sub.value": "E:int",
    }

    def finish_fixture(
        artifact: LibraryArtifact,
    ) -> tuple[Grammar, BuildStats]:
        probe = SemanticProbe.__new__(SemanticProbe)
        builder = GrammarBuilder(
            probe,
            frozenset(),
            BuilderOptions(),
            {},
            library_artifacts=(artifact,),
        )
        builder.grammar.add(
            type_nonterminal(module_type), Terminal("fixture_lib")
        )
        builder.add_library_artifacts()
        return builder.finish()

    cached_grammar, cached_stats = finish_fixture(fixture_artifact)
    uncached_grammar, uncached_stats = finish_fixture(
        replace(fixture_artifact, local_assignability_complete=False)
    )
    assert cached_grammar.productions == uncached_grammar.productions
    assert cached_stats.assignability_pairs_cached == 3
    assert cached_stats.assignability_pairs_checked == 1
    assert uncached_stats.assignability_pairs_cached == 0
    assert uncached_stats.assignability_pairs_checked == 4

    def activated_fixture(
        root_type: str,
        bound_name: str,
        bindings: Sequence[FromImportBinding] = (),
    ) -> bool:
        probe = SemanticProbe.__new__(SemanticProbe)
        builder = GrammarBuilder(
            probe,
            frozenset(),
            BuilderOptions(),
            {},
            library_artifacts=(fixture_artifact,),
            from_import_bindings=bindings,
        )
        builder.grammar.add(root_type, Terminal(bound_name))
        builder.add_library_artifacts()
        return fixture_artifact in builder.active_library_artifacts

    assert activated_fixture("E:<module 'fixture_lib.sub'>", "sub")
    answer_binding = FromImportBinding("fixture_lib", "answer", "ans")
    assert activated_fixture(
        "E:<built-in function answer>", "ans", (answer_binding,)
    )
    assert not activated_fixture(
        "E:<built-in function answer>", "different", (answer_binding,)
    )
    nested_binding = FromImportBinding("fixture_lib.sub", "value", "v")
    assert activated_fixture("E:int", "v", (nested_binding,))
    assert visible_from_import_bindings(
        "from fixture_lib import answer as ans\nans()\n", 1
    ) == (answer_binding,)
    assert visible_imported_library_modules(
        "import math\nvalue\nimport numba\n", 1
    ) == ("math",)
    assert not visible_from_import_bindings(
        "def f():\n    from fixture_lib import answer as ans\n"
        "def g():\n    ans()\n",
        3,
    )
    assert encode_library_nonterminal("E:list[int | str]") == (
        "E:list%5Bint%20%7C%20str%5D"
    )
    assert decode_library_nonterminal("E:list%5Bint%20%7C%20str%5D") == (
        "E:list[int | str]"
    )
    try:
        decode_library_nonterminal("E:%FF")
    except EvaluationError:
        pass
    else:
        raise AssertionError("invalid UTF-8 library atoms must be rejected")
    assert imported_library_modules(
        "import numpy as np\nimport scipy.linalg\nfrom math import sqrt\n"
    ) == ("math", "numpy", "scipy", "scipy.linalg")
    try:
        parse_library_cfg_text(
            "\n".join(
                (
                    "# api2cfg-python-library-cfg: 1",
                    "# module: fixture_lib",
                    f"# module-type: {module_type}",
                    "START -> E:int",
                )
            ),
            Path("bad.cfg"),
        )
    except EvaluationError:
        pass
    else:
        raise AssertionError("library CFG START definitions must be rejected")
    try:
        parse_library_cfg_text(
            "\n".join(
                (
                    "# api2cfg-python-library-cfg: 1",
                    "# module: fixture_lib",
                    f"# module-type: {module_type}",
                    "E:list%5Bint%5D -> [ E:int ]",
                )
            ),
            Path("list-display.cfg"),
        )
    except EvaluationError as error:
        assert "must not contain bracket terminals" in str(error)
    else:
        raise AssertionError("library CFG list displays must be rejected")
    for composite_terminal in ("[]", "[0]", "f([0])"):
        try:
            parse_library_cfg_text(
                "\n".join(
                    (
                        "# api2cfg-python-library-cfg: 1",
                        "# module: fixture_lib",
                        f"# module-type: {module_type}",
                        f"E:int -> {composite_terminal}",
                    )
                ),
                Path("composite-terminal.cfg"),
            )
        except EvaluationError as error:
            assert "must be one canonical Python token" in str(error)
        else:
            raise AssertionError(
                f"composite cached terminal {composite_terminal!r} must be rejected"
            )

    empty = Grammar("S")
    assert empty.nonterminals == frozenset({"S"})
    assert empty.symbol_count == 0

    ambiguous = Grammar("S")
    ambiguous.add("S", Nonterminal("A"))
    ambiguous.add("S", Nonterminal("B"))
    ambiguous.add("A", Terminal("x"))
    ambiguous.add("B", Terminal("x"))
    assert len(ambiguous.nonterminals) == 3
    assert len(ambiguous.productions) == 4
    assert ambiguous.symbol_count == 8
    bounded = BoundedLanguage(ambiguous, 1, 1_000)
    assert bounded.language_size(1) == 1
    assert bounded.unrank(1, 0) == ("x",)
    assert bounded.recognizes(("x",))
    ambiguous_index = ShortlexDFAIndex(ambiguous, 1_000)
    assert ambiguous_index.unrank(0) == ("x",)
    assert ambiguous_index.rank(("x",)) == 0
    assert ambiguous_index.indexed_size == 1
    try:
        ambiguous_index.unrank(1)
    except IndexError:
        pass
    else:
        raise AssertionError("a finite DFA index must reject its end rank")

    empty_index = ShortlexDFAIndex(empty, 1_000)
    assert empty_index.finite_max_length == 0
    try:
        empty_index.unrank(0)
    except IndexError:
        pass
    else:
        raise AssertionError("an empty DFA index must reject every rank")

    unit_chain = Grammar("U0")
    for index in range(100):
        unit_chain.add(f"U{index}", Nonterminal(f"U{index + 1}"))
    unit_chain.add("U100", Terminal("tail"))
    compiled_chain = UnitAwareBinaryGrammar(unit_chain)
    chain_sampler = DerivationCounter(compiled_chain, 1)
    assert chain_sampler.compiled is compiled_chain
    assert chain_sampler.derivation_count == 1
    assert chain_sampler.parse_count(("tail",)) == 1
    assert chain_sampler.recognizes(("tail",))

    unit_diamond = Grammar("S")
    unit_diamond.add("S", Nonterminal("A"))
    unit_diamond.add("S", Nonterminal("B"))
    unit_diamond.add("A", Nonterminal("C"))
    unit_diamond.add("B", Nonterminal("C"))
    unit_diamond.add("C", Terminal("x"))
    diamond_sampler = DerivationCounter(unit_diamond, 1)
    assert diamond_sampler.derivation_count == 2
    assert diamond_sampler.parse_count(("x",)) == 2
    assert {
        diamond_sampler._unrank_derivation(
            diamond_sampler.grammar.start,
            1,
            rank,
        )
        for rank in range(2)
    } == {("x",)}

    unit_cycle = Grammar("S")
    unit_cycle.add("S", Nonterminal("A"))
    unit_cycle.add("A", Nonterminal("B"))
    unit_cycle.add("B", Nonterminal("A"))
    unit_cycle.add("B", Terminal("cycle"))
    cycle_sampler = DerivationCounter(unit_cycle, 1)
    assert cycle_sampler.derivation_count == 1
    assert cycle_sampler.parse_count(("cycle",)) == 1
    assert cycle_sampler.recognizes(("cycle",))

    def materialized_cnf_recognizes(
        grammar: Grammar, tokens: tuple[str, ...]
    ) -> bool:
        cnf = to_cnf(grammar)
        terminal_parents: dict[str, set[str]] = defaultdict(set)
        binary_parents: dict[tuple[str, str], set[str]] = defaultdict(set)
        for production in cnf.productions:
            if len(production.rhs) == 1 and isinstance(
                production.rhs[0], Terminal
            ):
                terminal_parents[production.rhs[0].value].add(production.lhs)
            elif (
                len(production.rhs) == 2
                and isinstance(production.rhs[0], Nonterminal)
                and isinstance(production.rhs[1], Nonterminal)
            ):
                binary_parents[
                    (production.rhs[0].value, production.rhs[1].value)
                ].add(production.lhs)
        length = len(tokens)
        chart: list[list[set[str]]] = [
            [set() for _ in range(length + 1)] for _ in range(length)
        ]
        for index, token in enumerate(tokens):
            chart[index][index + 1].update(terminal_parents.get(token, ()))
        for span in range(2, length + 1):
            for start in range(length - span + 1):
                end = start + span
                for split in range(start + 1, end):
                    for left in chart[start][split]:
                        for right in chart[split][end]:
                            chart[start][end].update(
                                binary_parents.get((left, right), ())
                            )
        return cnf.start in chart[0][length]

    # Randomized language equivalence catches interactions among unit cycles,
    # diamonds, binary recursion, pruning, and SCC rewriting.  Ambiguity may be
    # represented differently, but recognition must match unit-expanded CNF.
    random_grammar_source = random.Random(9173)
    for _case in range(30):
        names = tuple(f"N{index}" for index in range(5))
        random_grammar = Grammar(names[0])
        for name in names:
            if random_grammar_source.random() < 0.7:
                random_grammar.add(
                    name,
                    Terminal(random_grammar_source.choice(("a", "b"))),
                )
            for target in names:
                if random_grammar_source.random() < 0.12:
                    random_grammar.add(name, Nonterminal(target))
            for _rule in range(random_grammar_source.randrange(3)):
                random_grammar.add(
                    name,
                    Nonterminal(random_grammar_source.choice(names)),
                    Nonterminal(random_grammar_source.choice(names)),
                )
        compact = UnitAwareBinaryGrammar(random_grammar)
        for length in range(1, 4):
            for word in itertools.product(("a", "b"), repeat=length):
                assert compact.recognizes(word) == materialized_cnf_recognizes(
                    random_grammar, word
                )
        randomized_sampler = DerivationCounter(random_grammar, 1, 3)
        if randomized_sampler.derivation_count <= 5_000:
            proposal_multiplicities: Counter[tuple[str, ...]] = Counter()
            for length, count in randomized_sampler.derivation_counts.items():
                for rank in range(count):
                    proposal_multiplicities[
                        randomized_sampler._unrank_derivation(
                            randomized_sampler.grammar.start,
                            length,
                            rank,
                        )
                    ] += 1
            assert all(
                randomized_sampler.parse_count(word) == multiplicity
                for word, multiplicity in proposal_multiplicities.items()
            )

    binary = Grammar("S")
    binary.add("S", Nonterminal("C"), Nonterminal("C"))
    binary.add("C", Terminal("a"))
    binary.add("C", Terminal("b"))
    bounded = BoundedLanguage(binary, 2, 1_000)
    assert bounded.language_size(2) == 4
    assert {bounded.unrank(2, rank) for rank in range(4)} == {
        ("a", "a"),
        ("a", "b"),
        ("b", "a"),
        ("b", "b"),
    }
    binary_index = ShortlexDFAIndex(binary, 1_000)
    assert [binary_index.unrank(rank) for rank in range(4)] == [
        ("a", "a"),
        ("a", "b"),
        ("b", "a"),
        ("b", "b"),
    ]
    assert [binary_index.rank(binary_index.unrank(rank)) for rank in range(4)] == list(
        range(4)
    )
    assert binary_index.slice_ends == [0, 4]

    recursive = Grammar("S")
    recursive.add("S", Terminal("a"))
    recursive.add("S", Nonterminal("S"), Nonterminal("S"))
    bounded = BoundedLanguage(recursive, 4, 10_000)
    assert bounded.language_size(4) == 1
    assert bounded.unrank(4, 0) == ("a", "a", "a", "a")

    branching = Grammar("S")
    branching.add("S", Nonterminal("C"))
    branching.add("S", Nonterminal("S"), Nonterminal("C"))
    branching.add("C", Terminal("a"))
    branching.add("C", Terminal("b"))
    branching_index = ShortlexDFAIndex(branching, 10_000)
    assert branching_index.finite_max_length is None
    millionth_prefix_word = branching_index.unrank(SAMPLE_RANK_LIMIT - 1)
    assert len(millionth_prefix_word) == 19
    assert branching_index.language.recognizes(millionth_prefix_word)
    assert branching_index.rank(millionth_prefix_word) == SAMPLE_RANK_LIMIT - 1

    unequal_ambiguity = Grammar("S")
    unequal_ambiguity.add("S", Nonterminal("A"), Nonterminal("Z"))
    unequal_ambiguity.add("S", Nonterminal("B"), Nonterminal("Z"))
    unequal_ambiguity.add("S", Nonterminal("C"), Nonterminal("Z"))
    unequal_ambiguity.add("A", Terminal("x"))
    unequal_ambiguity.add("B", Terminal("x"))
    unequal_ambiguity.add("C", Terminal("y"))
    unequal_ambiguity.add("Z", Terminal("z"))
    sampler = DerivationCounter(unequal_ambiguity, 2)
    assert sampler.derivation_count == 3
    assert sampler.derivation_counts == {2: 3}
    assert sampler.parse_count(("x", "z")) == 2
    assert sampler.parse_count(("y", "z")) == 1

    mixed_lengths = Grammar("S")
    mixed_lengths.add("S", Nonterminal("A"))
    mixed_lengths.add("S", Nonterminal("A"), Nonterminal("Z"))
    mixed_lengths.add("S", Nonterminal("B"), Nonterminal("Z"))
    mixed_lengths.add("S", Nonterminal("C"), Nonterminal("Z"))
    mixed_lengths.add("A", Terminal("x"))
    mixed_lengths.add("B", Terminal("x"))
    mixed_lengths.add("C", Terminal("y"))
    mixed_lengths.add("Z", Terminal("z"))
    range_sampler = DerivationCounter(mixed_lengths, 1, 2)
    assert range_sampler.derivation_counts == {1: 1, 2: 3}
    assert range_sampler.derivation_count == 4
    proposal_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for length, count in range_sampler.derivation_counts.items():
        for rank in range(count):
            word = range_sampler._unrank_derivation(
                range_sampler.grammar.start,
                length,
                rank,
            )
            proposal_counts[word] += 1
    assert proposal_counts == {
        ("x",): 1,
        ("x", "z"): 2,
        ("y", "z"): 1,
    }
    assert all(
        range_sampler.parse_count(word) == count
        for word, count in proposal_counts.items()
    )
    range_index = ShortlexDFAIndex(mixed_lengths, 1_000)
    assert [range_index.unrank(rank) for rank in range(3)] == [
        ("x",),
        ("x", "z"),
        ("y", "z"),
    ]
    assert [range_index.rank(range_index.unrank(rank)) for rank in range(3)] == [
        0,
        1,
        2,
    ]
    for supported_operator in ("+", "-", "*", "/", "//", "%", "**"):
        parsed_operator_expression = ast.parse(
            f"left {supported_operator} right", mode="eval"
        ).body
        assert isinstance(parsed_operator_expression, ast.expr)
        assert supported_expression(parsed_operator_expression)
    for unsupported_operator in ("&", "|", "^", "<<", ">>", "@"):
        parsed_operator_expression = ast.parse(
            f"left {unsupported_operator} right", mode="eval"
        ).body
        assert isinstance(parsed_operator_expression, ast.expr)
        assert not supported_expression(parsed_operator_expression)
    assert supported_expression(ast.parse("items[index]", mode="eval").body)
    assert supported_expression(ast.parse("items[:stop]", mode="eval").body)
    assert supported_expression(ast.parse("items[start::step]", mode="eval").body)
    for list_display in ("[]", "[left, right]", "f([left])", "left + [right]"):
        assert not supported_expression(ast.parse(list_display, mode="eval").body)
    assert not supported_expression(ast.parse("items[i, j]", mode="eval").body)
    assert not supported_expression(ast.parse("[*items]", mode="eval").body)
    for unsupported_spelling in ('f(0,)', '"left" "right"'):
        parsed_spelling = ast.parse(unsupported_spelling, mode="eval").body
        assert isinstance(parsed_spelling, ast.expr)
        assert supported_expression(parsed_spelling)
        assert not surface_spelling_supported(
            unsupported_spelling, parsed_spelling
        )
    for supported_spelling in ('f(0)', 'f("left", "right")'):
        parsed_spelling = ast.parse(supported_spelling, mode="eval").body
        assert isinstance(parsed_spelling, ast.expr)
        assert surface_spelling_supported(supported_spelling, parsed_spelling)
    f_string_spelling = 'f"{value} suffix"'
    parsed_f_string = ast.parse(f_string_spelling, mode="eval").body
    assert isinstance(parsed_f_string, ast.expr)
    assert not unsupported_surface_spellings(
        f_string_spelling, parsed_f_string
    )
    nested_f_string_spelling = 'f"{g(0,)}"'
    parsed_nested_f_string = ast.parse(
        nested_f_string_spelling, mode="eval"
    ).body
    assert isinstance(parsed_nested_f_string, ast.expr)
    assert unsupported_surface_spelling_counts(
        nested_f_string_spelling, parsed_nested_f_string
    ) == Counter({"trailing_call_comma": 1})
    repeated_spelling = '(f(0,), g(1,), "a" "b", "c" "d")'
    parsed_repeated_spelling = ast.parse(repeated_spelling, mode="eval").body
    assert isinstance(parsed_repeated_spelling, ast.expr)
    assert unsupported_surface_spelling_counts(
        repeated_spelling, parsed_repeated_spelling
    ) == Counter({"trailing_call_comma": 2, "adjacent_literal": 2})
    spelling_source = 'f(0,)\n"left" "right"\nf(0)\n'
    spelling_targets = candidate_targets(
        spelling_source,
        ast.parse(spelling_source),
    )
    assert [target.text for target in spelling_targets] == ["f(0)"]
    list_display_source = "[]\nf([x])\ny = [x]\nz = x + [x]\nitems[x]\n"
    list_display_targets = candidate_targets(
        list_display_source,
        ast.parse(list_display_source),
    )
    assert [target.text for target in list_display_targets] == ["items[x]"]
    deep_binary: ast.expr = ast.Name(id="x", ctx=ast.Load())
    for _ in range(2_000):
        deep_binary = ast.BinOp(
            left=deep_binary,
            op=ast.Add(),
            right=ast.Name(id="x", ctx=ast.Load()),
        )
    assert supported_expression(deep_binary)
    assert not any(isinstance(node, ast.List) for node in ast.walk(deep_binary))

    assert arithmetic_binary_result("int", "+", "float") == "float"
    assert arithmetic_binary_result("int", "/", "int") == "float"
    assert arithmetic_binary_result("int", "//", "float") == "float"
    assert arithmetic_binary_result("str", "+", "str") == "str"
    assert (
        arithmetic_binary_result("list[int]", "+", "list[int]")
        == "list[int]"
    )
    assert (
        arithmetic_binary_result("list[int]", "+", "list[float]")
        == "list[float]"
    )
    assert arithmetic_binary_result("str", "*", "int") == "str"
    assert arithmetic_binary_result("int", "*", "str") == "str"
    assert (
        arithmetic_binary_result(
            "int & ~AlwaysFalsy", "+", "int & ~Literal[0]"
        )
        == "int"
    )
    assert (
        arithmetic_binary_result(
            "list[int] & ~AlwaysFalsy", "+", "list[int]"
        )
        == "list[int]"
    )
    assert (
        arithmetic_binary_result(
            "list[int] & ~AlwaysFalsy", "*", "int & ~AlwaysFalsy"
        )
        == "list[int]"
    )
    assert (
        arithmetic_binary_result(
            "tuple[int, ...] & ~AlwaysFalsy", "+", "tuple[int, ...]"
        )
        == "tuple[int, ...]"
    )
    assert (
        arithmetic_binary_result(
            "tuple[int, str]", "+", "tuple[int, str]"
        )
        is None
    )
    assert arithmetic_binary_result("tuple[int, str]", "*", "int") is None
    assert arithmetic_binary_result("int", "*", "tuple[int, str]") is None
    assert arithmetic_binary_result("str", "-", "str") is None
    assert arithmetic_binary_result("complex", "//", "int") is None
    assert strip_negative_flow_refinements(
        "list[int] & ~AlwaysFalsy"
    ) == "list[int]"
    assert strip_negative_flow_refinements("int & SupportsAbs[int]") is None
    assert numeric_unary_kinds("int & ~AlwaysFalsy") == (True, True)
    assert numeric_unary_result("int & ~AlwaysFalsy") == "int"
    assert indexed_access_types("list[int]") == ("SupportsIndex", "int")
    assert indexed_access_types("dict[str, int]") == ("str", "int")
    assert indexed_access_types("tuple[int, ...]") == (
        "SupportsIndex",
        "int",
    )
    assert indexed_access_types("tuple[()]") is None
    assert indexed_access_types("list[int] & ~AlwaysFalsy") == (
        "SupportsIndex",
        "int",
    )
    assert sliced_access_type("tuple[int, ...]") == "tuple[int, ...]"
    assert (
        sliced_access_type("tuple[int, str]")
        == "tuple[int | str, ...]"
    )
    assert (
        sliced_access_type("list[int] & ~AlwaysFalsy") == "list[int]"
    )

    refined_arithmetic_builder = GrammarBuilder(
        SemanticProbe.__new__(SemanticProbe),
        frozenset(),
        BuilderOptions(),
        {},
    )
    refined_arithmetic_builder.add_expression(
        "int & ~AlwaysFalsy", (Terminal("nonzero"),)
    )
    refined_arithmetic_builder.add_expression(
        "list[int] & ~AlwaysFalsy", (Terminal("nonempty"),)
    )
    refined_arithmetic_builder.add_typed_binary_operations()
    refined_arithmetic_builder.add_typed_unary_operations()
    refined_arithmetic_grammar, _refined_arithmetic_stats = (
        refined_arithmetic_builder.finish()
    )
    refined_arithmetic_compiled = UnitAwareBinaryGrammar(
        refined_arithmetic_grammar
    )
    assert refined_arithmetic_compiled.recognizes(
        ("nonzero", "+", "nonzero")
    )
    assert refined_arithmetic_compiled.recognizes(
        ("nonempty", "*", "nonzero")
    )
    assert refined_arithmetic_compiled.recognizes(("-", "nonzero"))
    many_argument_call = ast.parse("f(a, b, c, d)", mode="eval").body
    assert supported_expression(many_argument_call)

    commented_source = "x = abs(0)  # retain this comment\n"
    commented_targets = candidate_targets(
        commented_source,
        ast.parse(commented_source),
    )
    assert len(commented_targets) == 1
    assert "# retain this comment" in commented_targets[0].hole.render("()")
    literal_source = "0\n"
    literal_targets = candidate_targets(literal_source, ast.parse(literal_source))
    assert len(literal_targets) == 1
    assert canonical_tokens(literal_targets[0]) == ("0",)

    existing_argument = ast.parse("def f(x: int) -> None:\n    x = abs(x)\n")
    existing_argument_targets = candidate_targets(
        ast.unparse(existing_argument),
        existing_argument,
        max_tokens=20,
    )
    assert len(existing_argument_targets) == 1
    assert existing_argument_targets[0].bound_before
    existing_exception = ast.parse(
        "try:\n    pass\nexcept Exception as error:\n    error = Exception()\n"
    )
    existing_exception_targets = candidate_targets(
        ast.unparse(existing_exception),
        existing_exception,
        max_tokens=20,
    )
    assert len(existing_exception_targets) == 1
    assert existing_exception_targets[0].bound_before
    fresh_source = "def f() -> None:\n    unused = abs(0)\n"
    fresh_targets = candidate_targets(
        fresh_source,
        ast.parse(fresh_source),
        max_tokens=20,
    )
    assert len(fresh_targets) == 1 and fresh_targets[0].kind == "assignment"
    assert not fresh_targets[0].bound_before
    assert not fresh_targets[0].loaded_after
    canonical_fresh = replace(
        fresh_targets[0],
        kind="fresh-assignment",
        fresh_name=fresh_targets[0].assigned_name,
    )
    assert canonical_tokens(canonical_fresh)[0] == FRESH_TOKEN
    assert canonical_tokens(fresh_targets[0])[0] == "unused"
    unicode_diagnostic: dict[str, object] = {
        "code": "unresolved-reference",
        "range": {
            "start": {"line": 0, "character": 6},
            "end": {"line": 0, "character": 8},
        },
    }
    assert diagnostic_identifier(
        "print(α)\n", unicode_diagnostic, "utf-8"
    ) == "α"
    closure_source = (
        "x = 0\n"
        "def outer() -> None:\n"
        "    x = abs(1)\n"
        "    def inner() -> int:\n"
        "        return x\n"
    )
    closure_targets = candidate_targets(
        closure_source,
        ast.parse(closure_source),
    )
    assert any(
        target.node.lineno == 3 and target.loaded_after
        for target in closure_targets
    )
    shadow_source = (
        "def outer() -> None:\n"
        "    x = abs(1)\n"
        "    def inner(x: int) -> int:\n"
        "        return x\n"
    )
    shadow_targets = candidate_targets(
        shadow_source,
        ast.parse(shadow_source),
    )
    assert any(
        target.node.lineno == 2 and not target.loaded_after
        for target in shadow_targets
    )
    comprehension_source = (
        "def outer() -> None:\n"
        "    x = abs(1)\n"
        "    print([x for x in range(3)])\n"
    )
    comprehension_targets = candidate_targets(
        comprehension_source,
        ast.parse(comprehension_source),
    )
    assert any(
        target.node.lineno == 2 and not target.loaded_after
        for target in comprehension_targets
    )
    assert has_suppression("message = '# type: ignore'\n") is False
    assert has_suppression("value = unknown  # type: ignore\n") is True

    signature = parse_signature(
        "(x: int, /, y: str = \"\", *, reverse: bool = False) -> float"
    )
    assert signature is not None
    assert signature.return_type == "float"
    layouts = argument_layouts(signature, max_arity=3, max_layouts=64)
    assert ArgumentLayout(("int",), ()) in layouts
    assert ArgumentLayout(("int",), (("reverse", "bool"),)) in layouts
    option_heavy = parse_signature(
        "(object: object, dtype: object = None, copy: object = None, "
        "order: object = None, subok: bool = False, ndmin: int = 0) -> object"
    )
    assert option_heavy is not None
    capped_layouts = argument_layouts(
        option_heavy,
        max_arity=6,
        max_layouts=4,
    )
    assert ArgumentLayout(("object",), ()) in capped_layouts
    assert iterable_element_type("map[int]") == "int"
    assert iterable_element_type("reversed[str]") == "str"
    assert concrete_heap_list_element_type("list[int]") == "int"
    assert (
        concrete_heap_list_element_type("list[Unknown] & ~AlwaysFalsy")
        == "Unknown"
    )
    assert concrete_heap_list_element_type("Sequence[int]") is None
    assert concrete_heap_list_element_type("list[_T]") is None
    assert concrete_heap_list_element_type("list[int] & Sized") is None
    assert groundable_type("int")
    assert groundable_type("int | str")
    assert not groundable_type("_T")
    assert not groundable_type("list[SupportsRichComparisonT]")
    assert is_assignable("list[int]", "Iterable[int]")
    assert is_assignable("map[int]", "Iterable[int]")
    for actual, expected in (
        ("list[int]", "Sequence[int]"),
        ("map[int]", "Iterator[int]"),
        ("reversed[int]", "Iterator[int]"),
        ("set[int]", "Collection[int]"),
        ("dict[int, str]", "Mapping[int, str]"),
        ("MutableMapping[int, str]", "Mapping[int, str]"),
        ("list[int]", "MutableSequence[int]"),
        ("defaultdict[int, str]", "MutableMapping[int, str]"),
        ("range", "Sized"),
        ("tuple[()]", "Iterable[str]"),
    ):
        assert is_assignable(actual, expected), (actual, expected)
    for actual, expected in (
        ("list[int]", "Mapping[int, int]"),
        ("set[int]", "Sequence[int]"),
        ("list[int]", "Iterator[int]"),
        ("map[int]", "Collection[int]"),
        ("map[int]", "Sized"),
        ("reversed[int]", "Sized"),
        ("dict[int, str]", "MutableSequence[int]"),
        ("tuple[int, ...]", "MutableSequence[int]"),
        ("list[int]", "MutableSequence[object]"),
        ("dict[int, str]", "MutableMapping[int, object]"),
    ):
        assert not is_assignable(actual, expected), (actual, expected)
    for reversible in (
        "list[int]",
        "tuple[int, ...]",
        "dict[int, str]",
        "range",
        "deque[int]",
    ):
        assert is_assignable(reversible, "_SupportsReversed[int]")
    for forward_only in (
        "set[int]",
        "map[int]",
        "filter[int]",
        "dict_values[str, int]",
    ):
        assert not is_assignable(
            forward_only, "_SupportsReversed[int]"
        )
        assert not is_assignable(
            forward_only, "SupportsLenAndGetItem[int]"
        )
    assert is_assignable("defaultdict[Unknown, int]", "Sized")
    assert not is_assignable("defaultdict[Unknown, int]", "Hashable")
    assert not is_assignable(
        "set[tuple[int, ...]]", "set[tuple[int, int]]"
    )
    assert not is_assignable(
        "set[tuple[str, ...]]", "set[tuple[int, int]]"
    )
    assert set_tuple_refinement_candidate(
        "set[tuple[int, ...]]", "set[tuple[int, int]]"
    )
    assert not set_tuple_refinement_candidate(
        "set[tuple[str, ...]]", "set[tuple[int, int]]"
    )
    assert is_assignable("float", "SupportsInt")
    assert is_assignable(
        "float", "str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc"
    )
    assert not is_assignable("complex", "SupportsInt")
    assert is_assignable("str", "AnyStr")
    assert is_assignable("bytes", "AnyStr@re.escape")
    assert is_assignable('Literal["x"]', "AnyStr")
    assert not is_assignable("set[str]", "AnyStr")
    assert not is_assignable("int", "AnyStr@fixture")
    assert is_assignable("<class 'int'>", "(_T1, /) -> _S")
    assert not is_assignable("str", "int")

    def parsed_signature(label: str) -> Signature:
        parsed = parse_signature(label)
        assert parsed is not None
        return parsed

    assert class_instance_type("<class 'defaultdict'>") == "defaultdict"
    assert class_instance_type("bound method defaultdict.copy") is None
    erased_self = parsed_signature("[Self](self) -> Self | None")
    bound_self = bind_unbound_self_signature(erased_self, "Widget")
    assert bound_self.parameters[0].type == "Widget"
    assert bound_self.return_type == "Widget | None"
    already_bound = parsed_signature("(value: str, /) -> int")
    assert bind_unbound_self_signature(already_bound, "Widget") == already_bound

    class SelfBindingProbe(SemanticProbe):
        """Exercise unbound descriptors alongside bound methods/classes."""

        member_rows: Mapping[str, tuple[Completion, ...]] = {
            "str": (
                Completion(
                    "split",
                    "Overload[(self: LiteralString, sep: LiteralString | None = None) "
                    "-> list[LiteralString], (self, sep: str | None = None) "
                    "-> list[str]]",
                    LSP_FUNCTION,
                ),
            ),
            "defaultdict": (
                Completion(
                    "copy", "def copy[Self](self) -> Self", LSP_FUNCTION
                ),
            ),
            "deque": (
                Completion(
                    "index",
                    "def index(self, x: Unknown, start: int = 0, /) -> int",
                    LSP_FUNCTION,
                ),
            ),
            "dq": (
                Completion(
                    "index",
                    "bound method deque[str].index(x: str, start: int = 0, /) -> int",
                    LSP_METHOD,
                ),
            ),
            "Widget": (
                Completion(
                    "clone", "def clone[Self](self: Self) -> Self", LSP_FUNCTION
                ),
            ),
            "widget": (
                Completion(
                    "clone", "bound method Widget.clone() -> Widget", LSP_METHOD
                ),
            ),
        }
        signature_rows: Mapping[str, tuple[str, ...]] = {
            "str.split": (
                "(self: LiteralString, sep: LiteralString | None = None) "
                "-> list[LiteralString]",
                "(self, sep: str | None = None) -> list[str]",
            ),
            "defaultdict.copy": ("[Self](self) -> Self",),
            "deque.index": (
                "(self, x: Unknown, start: int = 0, /) -> int",
            ),
            "dq.index": ("(x: str, start: int = 0, /) -> int",),
            "Widget.clone": ("[Self](self: Self) -> Self",),
            "widget.clone": ("() -> Widget",),
            "deque": ("(iterable: Iterable[str], /) -> deque[str]",),
        }

        def members(self, expression: str) -> tuple[list[Completion], bool]:
            return list(self.member_rows.get(expression, ())), False

        def signatures(self, expression: str) -> list[str]:
            return list(self.signature_rows.get(expression, ()))

    self_binding_probe = SelfBindingProbe.__new__(SelfBindingProbe)
    self_binding_builder = GrammarBuilder(
        self_binding_probe,
        frozenset(
            {
                "str",
                "defaultdict",
                "deque",
                "Widget",
                "s",
                "dd",
                "dq",
                "widget",
                "split",
                "copy",
                "index",
                "clone",
            }
        ),
        BuilderOptions(max_call_arity=3, max_receiver_types=8),
        {},
    )
    self_binding_builder.add_literals()
    for type_display, expression in (
        ("<class 'str'>", "str"),
        ("<class 'defaultdict'>", "defaultdict"),
        ("<class 'deque'>", "deque"),
        ("<class 'Widget'>", "Widget"),
        ("<class 'enumerate'>", "enumerate"),
        ("str", "s"),
        ("defaultdict[str, int]", "dd"),
        ("deque[str]", "dq"),
        ("Widget", "widget"),
    ):
        self_binding_builder.add_expression(
            type_display,
            (Terminal(expression),),
            representative=expression,
        )
    self_binding_builder.callables["<class 'deque'>"] = "deque"
    for type_display, expression in (
        ("<class 'str'>", "str"),
        ("<class 'defaultdict'>", "defaultdict"),
        ("<class 'deque'>", "deque"),
        ("<class 'Widget'>", "Widget"),
        ("deque[str]", "dq"),
        ("Widget", "widget"),
    ):
        self_binding_builder.queue_receiver(type_display, expression, 0)
    self_binding_builder.add_members()
    self_binding_builder.add_calls()
    self_binding_builder.add_redundant_grouping()
    self_binding_grammar, _self_binding_stats = self_binding_builder.finish()
    self_binding_compiled = UnitAwareBinaryGrammar(self_binding_grammar)
    for word in (
        ("str", ".", "split", "(", "s", ")"),
        ("defaultdict", ".", "copy", "(", "dd", ")"),
        ("deque", ".", "index", "(", "dq", ",", "s", ")"),
        ("Widget", ".", "clone", "(", "widget", ")"),
        (
            "Widget", ".", "clone", "(", "widget", ")", ".",
            "clone", "(", ")",
        ),
        ("dq", ".", "index", "(", "s", ")"),
        ("widget", ".", "clone", "(", ")"),
        ("deque", "(", "s", ")"),
    ):
        assert self_binding_compiled.recognizes(word), word
    for word in (
        ("str", ".", "split", "(", "enumerate", ")"),
        ("defaultdict", ".", "copy", "(", "enumerate", ")"),
        ("deque", ".", "index", "(", "str", ",", "s", ")"),
        ("Widget", ".", "clone", "(", "enumerate", ")"),
    ):
        assert not self_binding_compiled.recognizes(word), word
    assert type_nonterminal("Self") not in self_binding_grammar.nonterminals
    assert (
        postfix_nonterminal(type_nonterminal("Self"))
        not in self_binding_grammar.nonterminals
    )

    generic_probe = SemanticProbe.__new__(SemanticProbe)
    generic_signatures: dict[tuple[str, str], tuple[Signature, ...]] = {
        ("<class 'map'>", "map"): (
            parsed_signature(
                "[_T1, _S](func: (_T1, /) -> _S, "
                "iterable: Iterable[_T1], /) -> map[_S]"
            ),
        ),
        ("<class 'list'>", "list"): (
            parsed_signature(
                "[_T](iterable: Iterable[_T], /) -> list[_T]"
            ),
        ),
        ("<class 'tuple'>", "tuple"): (
            parsed_signature(
                "[_T_co](iterable: Iterable[Unknown] = ..., /) -> Unknown"
            ),
        ),
        ("def sorted", "sorted"): (
            parsed_signature(
                "[SupportsRichComparisonT](iterable: "
                "Iterable[SupportsRichComparisonT], /, *, "
                "reverse: bool = False) -> list[SupportsRichComparisonT]"
            ),
        ),
        ("def max", "max"): (
            parsed_signature(
                "[SupportsRichComparisonT](iterable: "
                "Iterable[SupportsRichComparisonT], /) -> "
                "SupportsRichComparisonT"
            ),
        ),
        ("<class 'reversed'>", "reversed"): (
            parsed_signature(
                "[_T](sequence: _SupportsReversed[_T], /) -> reversed[_T]"
            ),
        ),
        ("<class 'range'>", "range"): (
            parsed_signature("(stop: SupportsIndex, /) -> range"),
        ),
        ("<class 'int'>", "int"): (
            parsed_signature("(x: str, /) -> int"),
        ),
        ("<class 'str'>", "str"): (
            parsed_signature("(object: object, /) -> str"),
        ),
        ("def len", "len"): (
            parsed_signature("(obj: Sized, /) -> int"),
        ),
        ("def add", "add"): (
            parsed_signature("(left: int, right: int, /) -> int"),
        ),
        ("def unknown_result", "unknown_result"): (
            parsed_signature("(value: int, /) -> Unknown"),
        ),
        ("bound method int.conjugate", "number.conjugate"): (
            parsed_signature("() -> int"),
        ),
        ("bound method str.zfill", '"".zfill'): (
            parsed_signature("(width: SupportsIndex, /) -> str"),
        ),
        ("bound method str.join", '\"\".join'): (
            parsed_signature("(iterable: Iterable[str], /) -> str"),
        ),
        ("bound method list[int].extend", "target.extend"): (
            parsed_signature("(iterable: Iterable[int], /) -> None"),
        ),
    }
    generic_builder = GrammarBuilder(
        generic_probe,
        frozenset(),
        BuilderOptions(max_call_arity=3),
        generic_signatures,
    )
    generic_builder.add_literals()
    for callable_type, expression in (
        ("<class 'map'>", "map"),
        ("<class 'list'>", "list"),
        ("<class 'tuple'>", "tuple"),
        ("def sorted", "sorted"),
        ("def max", "max"),
        ("<class 'reversed'>", "reversed"),
        ("<class 'range'>", "range"),
        ("<class 'int'>", "int"),
        ("<class 'str'>", "str"),
        ("def len", "len"),
        ("def add", "add"),
        ("def unknown_result", "unknown_result"),
    ):
        generic_builder.add_expression(
            callable_type,
            (Terminal(expression),),
            representative=expression,
        )
        generic_builder.callables[callable_type] = expression
    generic_builder.add_expression(
        "list[str]", (Terminal("words"),), representative="words"
    )
    generic_builder.add_expression(
        "list[int]", (Terminal("numbers"),), representative="numbers"
    )
    generic_builder.add_expression(
        "list[complex]",
        (Terminal("complexes"),),
        representative="complexes",
    )
    generic_builder.add_expression(
        "set[int]",
        (Terminal("unique_numbers"),),
        representative="unique_numbers",
    )
    generic_builder.add_expression(
        "bound method int.conjugate",
        (
            Nonterminal(type_nonterminal("int")),
            Terminal("."),
            Terminal("conjugate"),
        ),
        representative="number.conjugate",
    )
    generic_builder.callables[
        "bound method int.conjugate"
    ] = "number.conjugate"
    generic_builder.add_expression(
        "bound method str.zfill",
        (
            Nonterminal(type_nonterminal("str")),
            Terminal("."),
            Terminal("zfill"),
        ),
        representative='"".zfill',
    )
    generic_builder.callables["bound method str.zfill"] = '"".zfill'
    generic_builder.add_expression("list[int]", (Terminal("target"),))
    generic_builder.add_expression(
        "bound method str.join",
        (
            Nonterminal(type_nonterminal("str")),
            Terminal("."),
            Terminal("join"),
        ),
        representative='\"\".join',
    )
    generic_builder.callables["bound method str.join"] = '\"\".join'
    generic_builder.add_expression(
        "bound method list[int].extend",
        (
            Nonterminal(type_nonterminal("list[int]")),
            Terminal("."),
            Terminal("extend"),
        ),
        representative="target.extend",
    )
    generic_builder.callables[
        "bound method list[int].extend"
    ] = "target.extend"
    generic_builder.add_calls()
    generic_builder.add_grounded_generic_calls()
    generic_grammar, _generic_stats = generic_builder.finish()
    generic_compiled = UnitAwareBinaryGrammar(generic_grammar)
    assert generic_compiled.recognizes(
        (
            "sorted", "(", "list", "(", "map", "(", "int", ",",
            "words", ")", ")", ")",
        )
    )
    assert generic_compiled.recognizes(
        ("sorted", "(", "map", "(", "int", ",", "words", ")", ")")
    )
    assert generic_compiled.recognizes(("max", "(", "numbers", ")"))
    assert generic_compiled.recognizes(("tuple", "(", ")"))
    assert generic_compiled.recognizes(("tuple", "(", "numbers", ")"))
    assert generic_compiled.recognizes(
        ("tuple", "(", "tuple", "(", ")", ")")
    )
    assert DerivationCounter(generic_compiled, 4).parse_count(
        ("tuple", "(", "numbers", ")")
    ) == 1
    assert DerivationCounter(generic_compiled, 6).parse_count(
        ("tuple", "(", "tuple", "(", ")", ")")
    ) == 1
    assert not any(
        production.lhs == TRUSTED_DYNAMIC_CALL_NONTERMINAL
        and any(
            isinstance(symbol, Nonterminal)
            and symbol.value == type_nonterminal("<class 'tuple'>")
            for symbol in production.rhs
        )
        for production in generic_grammar.productions
    )
    assert generic_compiled.recognizes(
        ('\"\"', ".", "join", "(", "map", "(", "str", ",", "numbers", ")", ")")
    )
    assert generic_compiled.recognizes(
        ("map", "(", "int", ",", "words", ")")
    )
    assert generic_compiled.recognizes(
        ("map", "(", "str", ",", "numbers", ")")
    )
    assert generic_compiled.recognizes(
        ("map", "(", "len", ",", "words", ")")
    )
    assert generic_compiled.recognizes(
        (
            "map", "(", "add", ",", "numbers", ",", "numbers", ")",
        )
    )
    assert generic_compiled.recognizes(
        ("map", "(", "unknown_result", ",", "numbers", ")")
    )
    assert generic_compiled.recognizes(
        (
            "target", ".", "extend", "(", "reversed", "(", "range",
            "(", "0", ")", ")", ")",
        )
    )
    assert not generic_compiled.recognizes(
        ('\"\"', ".", "join", "(", "map", "(", "int", ",", "words", ")", ")")
    )
    assert not generic_compiled.recognizes(
        ("map", "(", "add", ",", "numbers", ")")
    )
    assert not generic_compiled.recognizes(
        (
            "map", "(", "int", ",", "words", ",", "numbers", ")",
        )
    )
    assert not generic_compiled.recognizes(
        (
            "map", "(", "0", ".", "conjugate", ",", "numbers", ")",
        )
    )
    assert not generic_compiled.recognizes(
        (
            "map", "(", '\"\"', ".", "zfill", ",", "words", ")",
        )
    )
    assert not generic_compiled.recognizes(
        ("target", ".", "extend", "(", "reversed", "(", "words", ")", ")")
    )
    assert not generic_compiled.recognizes(
        ("max", "(", "complexes", ")")
    )
    assert not generic_compiled.recognizes(
        ("reversed", "(", "unique_numbers", ")")
    )
    assert not generic_compiled.recognizes(
        (
            "reversed", "(", "map", "(", "int", ",", "words", ")", ")",
        )
    )

    class RejectingOutputAssignmentProbe(SemanticProbe):
        """Accept expressions, but reject every ordinary output witness."""

        def __init__(
            self,
            accepted: Iterable[str] = (),
            local_rejections: Iterable[str] = (),
        ) -> None:
            self.assignment_queries: list[str] = []
            self.accepted = frozenset(accepted)
            self.local_rejections = frozenset(local_rejections)

        def accepts_expression(self, expression: str) -> bool:
            return True

        def accepts_assignment(self, expression: str) -> bool:
            self.assignment_queries.append(expression)
            return expression in self.accepted

        def assignment_diagnostic_partition(
            self, expression: str
        ) -> tuple[
            tuple[Mapping[str, object], ...],
            tuple[Mapping[str, object], ...],
        ]:
            self.assignment_queries.append(expression)
            if expression in self.accepted:
                return (), ()
            diagnostic: dict[str, object] = {
                "code": "fixture-error",
                "message": "synthetic output-assignment rejection",
            }
            if expression in self.local_rejections:
                return (diagnostic,), ()
            return (), (diagnostic,)

    producer_probe = RejectingOutputAssignmentProbe(
        {"good"}, local_rejections={"local_bad"}
    )
    producer_builder = GrammarBuilder(
        producer_probe,
        frozenset({"good", "bad", "local_bad"}),
        BuilderOptions(max_output_producers=2048),
        {},
        required_assignment="result",
    )
    producer_builder.add_expression(
        "int", (Terminal("good"),), representative="good"
    )
    producer_builder.add_expression("int", (Terminal("bad"),))
    producer_builder.add_expression("int", (Terminal("local_bad"),))
    producer_grammar, producer_stats = producer_builder.finish()
    producer_compiled = UnitAwareBinaryGrammar(producer_grammar)
    assert producer_compiled.recognizes(("result", "=", "good"))
    assert producer_compiled.recognizes(("result", "=", "local_bad"))
    assert not producer_compiled.recognizes(("result", "=", "bad"))
    assert producer_stats.output_producer_families == 3
    assert producer_stats.output_producers_checked == 3
    assert producer_stats.output_producers_rejected == 1
    assert producer_stats.output_producers_local_fallback == 1
    assert producer_stats.output_producers_unchecked == 0

    capped_probe = RejectingOutputAssignmentProbe({"gate"})
    capped_builder = GrammarBuilder(
        capped_probe,
        frozenset({"a_bad", "b_bad", "gate"}),
        BuilderOptions(max_output_producers=1),
        {},
        required_assignment="result",
    )
    capped_builder.add_expression(
        "int", (Terminal("gate"),), representative="gate"
    )
    capped_builder.add_expression("int", (Terminal("a_bad"),))
    capped_builder.add_expression("int", (Terminal("b_bad"),))
    capped_grammar, capped_stats = capped_builder.finish()
    capped_compiled = UnitAwareBinaryGrammar(capped_grammar)
    assert not capped_compiled.recognizes(("result", "=", "a_bad"))
    assert capped_compiled.recognizes(("result", "=", "b_bad"))
    assert capped_compiled.recognizes(("result", "=", "gate"))
    assert capped_stats.output_producer_families == 3
    assert capped_stats.output_producers_checked == 1
    assert capped_stats.output_producers_rejected == 1
    assert capped_stats.output_producers_unchecked == 2

    dynamic_probe = RejectingOutputAssignmentProbe()
    dynamic_type = "def lcm(x: int, y: int) -> Unknown"
    dynamic_builder = GrammarBuilder(
        dynamic_probe,
        frozenset({"lcm", "n", "m", "mystery", "tuple"}),
        BuilderOptions(max_call_arity=2),
        {
            (dynamic_type, "lcm"): (
                parsed_signature("(x: int, y: int) -> Unknown"),
            ),
            ("<class 'tuple'>", "tuple"): (
                parsed_signature(
                    "(iterable: Iterable[Unknown] = ..., /) -> Unknown"
                ),
            ),
        },
        required_assignment="fpb",
    )
    dynamic_builder.add_literals()
    dynamic_builder.add_expression(
        dynamic_type,
        (Terminal("lcm"),),
        representative="lcm",
    )
    dynamic_builder.callables[dynamic_type] = "lcm"
    dynamic_builder.add_expression(
        "<class 'tuple'>",
        (Terminal("tuple"),),
        representative="tuple",
    )
    dynamic_builder.callables["<class 'tuple'>"] = "tuple"
    dynamic_builder.add_expression("int", (Terminal("n"),))
    dynamic_builder.add_expression("int", (Terminal("m"),))
    dynamic_builder.add_expression(
        "Unknown", (Terminal("mystery"),), representative="mystery"
    )
    dynamic_builder.add_calls()
    dynamic_builder.add_grounded_generic_calls()
    dynamic_builder.add_dynamic_operations()
    dynamic_grammar, _dynamic_stats = dynamic_builder.finish()
    dynamic_compiled = UnitAwareBinaryGrammar(dynamic_grammar)
    assert dynamic_compiled.recognizes(
        ("fpb", "=", "lcm", "(", "n", ",", "m", ")")
    )
    assert not dynamic_compiled.recognizes(("fpb", "=", "mystery"))
    assert not dynamic_compiled.recognizes(("fpb", "=", "tuple", "(", ")"))
    assert not dynamic_compiled.recognizes(
        ("fpb", "=", "lcm", "(", "n", ",", "m", ")", ".", "real")
    )
    assert all(
        Nonterminal(type_nonterminal("Unknown")) not in production.rhs
        for production in dynamic_grammar.productions
        if production.lhs == dynamic_grammar.start
    )

    contextual_output_probe = RejectingOutputAssignmentProbe(
        {"stack . pop ( )"}
    )
    stack_pop_type = (
        "bound method list[Unknown].pop"
        "(index: SupportsIndex = -1, /) -> Unknown"
    )
    queue_pop_type = "bound method deque[Unknown].pop() -> Unknown"
    contextual_output_builder = GrammarBuilder(
        contextual_output_probe,
        frozenset({"stack", "queue"}),
        BuilderOptions(max_call_arity=0),
        {
            (stack_pop_type, "stack.pop"): (
                parsed_signature("() -> Unknown"),
            ),
            (queue_pop_type, "queue.pop"): (
                parsed_signature("() -> Unknown"),
            ),
        },
        required_assignment="value",
    )
    for callable_type, expression in (
        (stack_pop_type, "stack.pop"),
        (queue_pop_type, "queue.pop"),
    ):
        expression_tokens = canonical_expression_tokens(expression)
        assert expression_tokens is not None
        contextual_output_builder.add_expression(
            callable_type,
            tuple(
                Terminal(token) for token in expression_tokens
            ),
            representative=expression,
        )
        contextual_output_builder.callables[callable_type] = expression
    contextual_output_builder.add_calls()
    contextual_output_grammar, _contextual_output_stats = (
        contextual_output_builder.finish()
    )
    contextual_output_compiled = UnitAwareBinaryGrammar(
        contextual_output_grammar
    )
    assert contextual_output_compiled.recognizes(
        ("value", "=", "stack", ".", "pop", "(", ")")
    )
    assert not contextual_output_compiled.recognizes(
        ("value", "=", "queue", ".", "pop", "(", ")")
    )
    assert set(contextual_output_probe.assignment_queries) == {
        "stack.pop",
        "queue.pop",
        "stack . pop ( )",
        "queue . pop ( )",
    }

    legacy_dynamic_probe = RejectingOutputAssignmentProbe(
        {"visible_unknown", "-self.y"}
    )
    legacy_dynamic_builder = GrammarBuilder(
        legacy_dynamic_probe,
        frozenset({"visible_unknown", "rejected_unknown"}),
        BuilderOptions(),
        {},
        required_assignment="result",
    )
    legacy_dynamic_builder.add_expression(
        "Unknown",
        (Terminal("visible_unknown"),),
        representative="visible_unknown",
    )
    legacy_dynamic_builder.add_expression(
        "Unknown",
        (Terminal("rejected_unknown"),),
        representative="rejected_unknown",
    )
    legacy_dynamic_builder.add_expression(
        "Unknown",
        (Terminal("self"), Terminal("."), Terminal("y")),
        representative="self.y",
    )
    legacy_dynamic_grammar, _legacy_dynamic_stats = (
        legacy_dynamic_builder.finish()
    )
    legacy_dynamic_compiled = UnitAwareBinaryGrammar(legacy_dynamic_grammar)
    assert legacy_dynamic_compiled.recognizes(
        ("result", "=", "visible_unknown")
    )
    assert not legacy_dynamic_compiled.recognizes(
        ("result", "=", "rejected_unknown")
    )
    assert legacy_dynamic_compiled.recognizes(
        ("result", "=", "-", "self", ".", "y")
    )
    assert not legacy_dynamic_compiled.recognizes(
        ("result", "=", "+", "self", ".", "y")
    )

    heap_probe = RejectingOutputAssignmentProbe()
    heappop_type = (
        "def heappop[SupportsRichComparisonT]"
        "(heap: list[SupportsRichComparisonT], /) -> SupportsRichComparisonT"
    )
    heap_builder = GrammarBuilder(
        heap_probe,
        frozenset({"heapq", "heappop", "w", "words"}),
        BuilderOptions(max_call_arity=1),
        {
            (heappop_type, "heapq.heappop"): (
                parsed_signature(
                    "[SupportsRichComparisonT]"
                    "(heap: list[SupportsRichComparisonT], /) "
                    "-> SupportsRichComparisonT"
                ),
            )
        },
        required_assignment="head",
    )
    heap_builder.add_literals()
    heap_builder.add_expression(
        heappop_type,
        (Terminal("heapq"), Terminal("."), Terminal("heappop")),
        representative="heapq.heappop",
    )
    heap_builder.callables[heappop_type] = "heapq.heappop"
    heap_builder.add_expression(
        "list[Unknown] & ~AlwaysFalsy",
        (Terminal("w"),),
        representative="w",
    )
    heap_builder.add_expression(
        "list[str]", (Terminal("words"),), representative="words"
    )
    for rejected_type, token in (
        ("list[object]", "objects"),
        ("list[dict[str, int]]", "dictionaries"),
        ("list[complex]", "complexes"),
    ):
        heap_builder.add_expression(
            rejected_type, (Terminal(token),), representative=token
        )
    heap_builder.add_calls()
    # Model a covered heapq artifact: its ordinary call row remains in the
    # grammar, while add_library_artifact removes the cached export from the
    # live callable worklist.
    heap_builder.callables.pop(heappop_type)
    heap_builder.add_grounded_generic_calls()
    heappop_nonterminal = Nonterminal(type_nonterminal(heappop_type))
    for rejected_type, result_type in (
        ("list[object]", "object"),
        ("list[dict[str, int]]", "dict[str, int]"),
        ("list[complex]", "complex"),
    ):
        assert Production(
            type_nonterminal(result_type),
            (
                heappop_nonterminal,
                Terminal("("),
                Nonterminal(type_nonterminal(rejected_type)),
                Terminal(")"),
            ),
        ) not in heap_builder.grammar.productions
    heap_grammar, _heap_stats = heap_builder.finish()
    heap_compiled = UnitAwareBinaryGrammar(heap_grammar)
    assert heap_compiled.recognizes(
        ("head", "=", "heapq", ".", "heappop", "(", "w", ")")
    )
    assert not heap_compiled.recognizes(
        ("head", "=", "heapq", ".", "heappop", "(", "words", ")")
    )

    class ContextualAppendProbe(SemanticProbe):
        def __init__(self) -> None:
            self.queries: list[str] = []

        def accepts_expression(self, expression: str) -> bool:
            self.queries.append(expression)
            return expression == "l.append(s)"

    append_probe = ContextualAppendProbe()
    append_type = (
        "bound method list[set[tuple[int, int]]].append"
        "(object: set[tuple[int, int]], /) -> None"
    )
    append_signatures: dict[tuple[str, str], tuple[Signature, ...]] = {
        (append_type, "l.append"): (
            parsed_signature("(object: set[tuple[int, int]], /) -> None"),
        )
    }
    append_builder = GrammarBuilder(
        append_probe,
        frozenset({"l", "s", "y"}),
        BuilderOptions(max_call_arity=1),
        append_signatures,
    )
    append_builder.add_literals()
    append_builder.add_expression(
        "set[tuple[int, ...]]", (Terminal("s"),), representative="s"
    )
    append_builder.add_expression(
        "set[tuple[int, ...]]", (Terminal("y"),), representative="y"
    )
    append_builder.add_expression(
        append_type,
        (Terminal("l"), Terminal("."), Terminal("append")),
        representative="l.append",
    )
    append_builder.callables[append_type] = "l.append"
    append_builder.add_calls()
    append_grammar, _append_stats = append_builder.finish()
    append_compiled = UnitAwareBinaryGrammar(append_grammar)
    assert append_probe.queries == ["l.append(s)"]
    assert append_compiled.recognizes(("l", ".", "append", "(", "s", ")"))
    assert not append_compiled.recognizes(
        ("l", ".", "append", "(", "y", ")")
    )
    print("self-test passed")


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "dataset source (APPS directory/split JSONL or CodeNet gzip tar); "
            "defaults according to --dataset and --split"
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("apps", "codenet"),
        default="apps",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="APPS split to read when its source is a directory",
    )
    parser.add_argument(
        "-n",
        "--files",
        type=int,
        default=1000,
        help="number of dataset-order ty-clean source files to evaluate",
    )
    parser.add_argument("--precision-samples", type=int, default=10)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="stop after this many generated candidates have been checked by ty",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ty", default="ty")
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIRECTORY,
        help="directory containing precomputed Python library .cfg fragments",
    )
    parser.add_argument("--allow-ignores", action="store_true")
    parser.add_argument("--max-layouts-per-signature", type=int, default=64)
    parser.add_argument("--member-depth", type=int, default=2)
    parser.add_argument("--max-receiver-types", type=int, default=32)
    parser.add_argument("--max-module-members", type=int, default=128)
    parser.add_argument("--max-output-producers", type=int, default=2048)
    parser.add_argument(
        "--max-dfa-states",
        type=int,
        default=DEFAULT_MAX_DFA_STATES,
        help="maximum hash-consed states in each sampling DFA index",
    )
    parser.add_argument(
        "--statement-timeout",
        type=float,
        default=60.0,
        help="end-to-end wall deadline in seconds per prepared target",
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--show-samples", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true")
    parsed = parser.parse_args(arguments)
    if parsed.max_samples is not None and parsed.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if parsed.max_samples is not None and parsed.precision_samples < 1:
        parser.error("--precision-samples must be positive with --max-samples")
    if (
        not math.isfinite(parsed.statement_timeout)
        or parsed.statement_timeout <= 0
    ):
        parser.error("--statement-timeout must be a positive finite number")
    for name in (
        "files",
        "precision_samples",
        "max_layouts_per_signature",
        "member_depth",
        "max_receiver_types",
        "max_module_members",
        "max_output_producers",
        "max_dfa_states",
    ):
        value = getattr(parsed, name)
        minimum = (
            0
            if name
            in {
                "files",
                "precision_samples",
                "member_depth",
                "max_output_producers",
            }
            else 1
        )
        if value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be at least {minimum}")
    if parsed.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if not 0 <= parsed.shard_index < parsed.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    return parsed


def evaluation_options(args: argparse.Namespace) -> EvaluationOptions:
    source = args.source
    if source is None:
        source = default_dataset_source(args.dataset, args.split)
    return EvaluationOptions(
        dataset=args.dataset,
        source=source,
        split=args.split,
        files=args.files,
        precision_samples=args.precision_samples,
        max_samples=args.max_samples,
        seed=args.seed,
        ty=args.ty,
        allow_ignores=args.allow_ignores,
        builder=BuilderOptions(
            max_layouts_per_signature=args.max_layouts_per_signature,
            member_depth=args.member_depth,
            max_receiver_types=args.max_receiver_types,
            max_module_members=args.max_module_members,
            max_output_producers=args.max_output_producers,
        ),
        max_dfa_states=args.max_dfa_states,
        statement_timeout=args.statement_timeout,
        json_lines=args.jsonl,
        show_samples=args.show_samples,
        library_directory=args.library_dir,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    if args.self_test:
        run_self_tests()
        return 0
    options = evaluation_options(args)
    try:
        return evaluate(options)
    except (EvaluationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
