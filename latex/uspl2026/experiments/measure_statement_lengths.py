#!/usr/bin/env python3
"""
Measure statement lengths, in lexical-ish Tree-sitter tokens, from sampled GitHub files.

Typical use:

    python -m pip install requests tree-sitter tree-sitter-language-pack
    export GITHUB_TOKEN=ghp_...   # strongly recommended; unauthenticated limits are too low
    python measure_statement_lengths_cache_recovery.py \
        --out-dir stmtlen-study \
        --languages java,kotlin,python,cpp,c,javascript,typescript,rust,go,csharp \
        --min-files 100 \
        --min-bytes 20000 \
        --strategy repo-tree \
        --seed 20260807

Generated output:
    stmtlen-study/statement_lengths_slide.tex

The existing ``stmtlen-study`` directory remains the canonical source and parser cache. The script
uses ``sampled_files.jsonl``, ``src/``, ``parse_cache/``, ``parse_failures/``, and ``parse_current/``
in place. The physical files under ``src/<language>`` are authoritative: missing or truncated
manifest entries are reconstructed from parser checkpoints and, as a last resort, from the cached
filename plus a content hash. If all requested files are available locally, no GitHub client is
constructed and no network request is made. Pass ``--offline`` to prohibit all new sampling even
when a cached file is genuinely missing.

The only presentation artifact generated is a self-contained Beamer source file. Its bar chart is
written directly with LaTeX's built-in ``picture`` environment. All bar heights,
confidence-interval endpoints, labels, and sample sizes are literal values in the TeX; there is no
external PDF/PNG, CSV import, Matplotlib output, pgfplots data file, or TeX build step.

Sampling note:
    ``repo-tree`` first samples repositories by GitHub repository-search strata, then samples
    at most one qualifying file from each repository's default-branch tree. This is usually
    less biased than raw code-search ranking and gives a variety of projects. ``code-search``
    is included for runs where recursive tree access is too slow or blocked; it is more biased
    because GitHub's code-search API requires a search term and returns ranked results.

Parser-safety note:
    Parser workers disable CPython's cyclic garbage collector before importing Tree-sitter,
    traverse syntax trees with TreeCursor, derive source positions from byte offsets, and restart
    after a small batch of files. These measures contain a known py-tree-sitter native crash while
    ordinary reference counting remains active. Each language still runs in disposable subprocesses.

Measurement note:
    A "statement" is a language-specific Tree-sitter statement/declaration node. For compound
    statements (if/for/while/try/etc.), tokens inside nested statement/block bodies are excluded
    from the compound node's own length, and nested statements are counted separately. Comments
    and whitespace are ignored. String/template literals are coalesced as a single token when
    the grammar exposes them as multi-token subtrees. Only statements with more than
    ``--min-tokens`` tokens are included.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import faulthandler
import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import random
import re
import signal
import statistics as stats
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote

# requests is imported lazily by GitHub.__init__. Parser workers never need it, and keeping
# unrelated native extensions out of those processes makes crash isolation more reliable.

def percentile(values: Iterable[float], percent: float) -> float:
    """NumPy-compatible linear percentile, implemented with the standard library."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

# Tree-sitter and its grammar pack are imported lazily inside parser workers. This keeps
# native modules out of the sampling/aggregation process and lets the worker disable cyclic GC
# before loading them.


GITHUB_API = "https://api.github.com"
DEFAULT_API_VERSION = os.environ.get("GITHUB_API_VERSION", "2022-11-28")

# Small enough for parse speed; large enough to include the requested >20KB files.
DEFAULT_MAX_BYTES = 1_000_000

# Increment this when the statement-extraction or cache format changes.
SCRIPT_VERSION = "2026-08-08-cache-inventory-v1"
PARSE_CACHE_SCHEMA = 5
LOCAL_INVENTORY_SCHEMA = 1

# Used only by --strategy code-search because GitHub code search requires a non-qualifier term.
# Each anchor is intentionally common and low-specificity, but this strategy is still biased.
CODE_SEARCH_ANCHORS: dict[str, list[str]] = {
    "java": ["class", "return", "public", "if", "for", "new"],
    "kotlin": ["fun", "val", "var", "return", "if", "class"],
    "python": ["def", "return", "import", "if", "for", "self"],
    "cpp": ["return", "include", "class", "if", "for", "std"],
    "c": ["return", "include", "if", "for", "struct", "static"],
    "javascript": ["function", "const", "return", "import", "if", "async"],
    "typescript": ["function", "const", "return", "import", "interface", "async"],
    "rust": ["fn", "let", "return", "impl", "match", "pub"],
    "go": ["func", "return", "if", "for", "type", "var"],
    "csharp": ["class", "return", "public", "if", "using", "var"],
}

# Star strata keep the repository sample from being only top-star projects or only tiny repos.
STAR_BUCKETS = [
    "stars:0..0",
    "stars:1..3",
    "stars:4..10",
    "stars:11..50",
    "stars:51..200",
    "stars:201..1000",
    "stars:>1000",
]

VENDORISH_PARTS = {
    ".git",
    ".gradle",
    ".idea",
    ".mvn",
    "bazel-bin",
    "bazel-out",
    "bower_components",
    "build",
    "coverage",
    "debug",
    "deps",
    "dist",
    "docs",
    "examples",
    "generated",
    "node_modules",
    "out",
    "release",
    "target",
    "testdata",
    "third_party",
    "third-party",
    "vendor",
    "vendors",
}

GENERATED_RE = re.compile(
    r"(^|/)(gen|generated|autogen|protobuf|proto|schema|schemas)(/|$)|"
    r"(\.g\.|\.gen\.|\.generated\.|\.pb\.|\.min\.)",
    re.IGNORECASE,
)

EXTRA_OR_COMMENT_TYPES = {
    "comment",
    "line_comment",
    "block_comment",
    "documentation_comment",
    "doc_comment",
    "multiline_comment",
}

# Coalesce grammar-level sub-tokenization of literals/templates into one lexical token.
LITERAL_ATOMS = {
    "string",
    "string_literal",
    "raw_string_literal",
    "interpreted_string_literal",
    "template_string",
    "template_literal",
    "verbatim_string_literal",
    "character_literal",
    "char_literal",
    "rune_literal",
    "integer_literal",
    "float_literal",
    "number_literal",
    "real_literal",
    "imaginary_literal",
    "boolean_literal",
    "null_literal",
    "none",
    "true",
    "false",
}

# We include local declaration forms, but intentionally exclude function/class/type declarations.
EXCLUDED_DECLARATION_TYPES = {
    "annotation_type_declaration",
    "class_declaration",
    "class_definition",
    "class_item",
    "concept_definition",
    "constructor_declaration",
    "enum_declaration",
    "enum_item",
    "field_declaration",
    "function_declaration",
    "function_definition",
    "function_item",
    "impl_item",
    "interface_declaration",
    "method_declaration",
    "method_definition",
    "mod_item",
    "namespace_definition",
    "package_declaration",
    "protocol_declaration",
    "struct_item",
    "struct_specifier",
    "trait_item",
    "type_declaration",  # top-level in Go/C#; local aliases are rare, keep out by default
    "union_specifier",
}

# Bodies/blocks skipped when counting the header of a compound statement.
BODY_OR_BLOCK_TYPES = {
    "block",
    "compound_statement",
    "declaration_list",
    "statement_block",
    "class_body",
    "function_body",
    "method_body",
    "switch_block",
    "switch_body",
    "enum_body",
    "initializer_list",  # avoids huge initializer declarations dominating
}


@dataclass(frozen=True)
class LangSpec:
    key: str
    label: str
    github_language: str
    parser_key: str
    extensions: tuple[str, ...]
    statement_types: frozenset[str]


LANGS: dict[str, LangSpec] = {
    "java": LangSpec(
        "java",
        "Java",
        "Java",
        "java",
        (".java",),
        frozenset(
            {
                "assert_statement",
                "break_statement",
                "continue_statement",
                "do_statement",
                "enhanced_for_statement",
                "expression_statement",
                "for_statement",
                "if_statement",
                "labeled_statement",
                "local_variable_declaration",
                "return_statement",
                "switch_expression",
                "switch_rule",
                "synchronized_statement",
                "throw_statement",
                "try_statement",
                "try_with_resources_statement",
                "while_statement",
                "yield_statement",
            }
        ),
    ),
    "kotlin": LangSpec(
        "kotlin",
        "Kotlin",
        "Kotlin",
        "kotlin",
        (".kt", ".kts"),
        frozenset(
            {
                "statement",
                "property_declaration",
                "control_statement",
                "loop_statement",
                "jump_expression",
            }
        ),
    ),
    "python": LangSpec(
        "python",
        "Python",
        "Python",
        "python",
        (".py",),
        frozenset(
            {
                "assert_statement",
                "assignment",
                "augmented_assignment",
                "break_statement",
                "continue_statement",
                "delete_statement",
                "expression_statement",
                "for_statement",
                "future_import_statement",
                "global_statement",
                "if_statement",
                "import_from_statement",
                "import_statement",
                "match_statement",
                "nonlocal_statement",
                "pass_statement",
                "raise_statement",
                "return_statement",
                "try_statement",
                "while_statement",
                "with_statement",
                "yield",
            }
        ),
    ),
    "cpp": LangSpec(
        "cpp",
        "C++",
        "C++",
        "cpp",
        (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"),
        frozenset(
            {
                "break_statement",
                "case_statement",
                "co_return_statement",
                "compound_literal_expression",
                "continue_statement",
                "declaration",
                "do_statement",
                "expression_statement",
                "for_statement",
                "goto_statement",
                "if_statement",
                "labeled_statement",
                "range_based_for_statement",
                "return_statement",
                "switch_statement",
                "throw_statement",
                "try_statement",
                "while_statement",
            }
        ),
    ),
    "c": LangSpec(
        "c",
        "C",
        "C",
        "c",
        (".c", ".h"),
        frozenset(
            {
                "break_statement",
                "case_statement",
                "continue_statement",
                "declaration",
                "do_statement",
                "expression_statement",
                "for_statement",
                "goto_statement",
                "if_statement",
                "labeled_statement",
                "return_statement",
                "switch_statement",
                "while_statement",
            }
        ),
    ),
    "javascript": LangSpec(
        "javascript",
        "JavaScript",
        "JavaScript",
        "javascript",
        (".js", ".mjs", ".cjs", ".jsx"),
        frozenset(
            {
                "break_statement",
                "continue_statement",
                "debugger_statement",
                "do_statement",
                "export_statement",
                "expression_statement",
                "for_in_statement",
                "for_of_statement",
                "for_statement",
                "if_statement",
                "import_statement",
                "labeled_statement",
                "lexical_declaration",
                "return_statement",
                "switch_case",
                "switch_statement",
                "throw_statement",
                "try_statement",
                "variable_declaration",
                "while_statement",
                "with_statement",
            }
        ),
    ),
    "typescript": LangSpec(
        "typescript",
        "TypeScript",
        "TypeScript",
        "typescript",
        (".ts", ".tsx"),
        frozenset(
            {
                "break_statement",
                "continue_statement",
                "debugger_statement",
                "do_statement",
                "export_statement",
                "expression_statement",
                "for_in_statement",
                "for_of_statement",
                "for_statement",
                "if_statement",
                "import_statement",
                "labeled_statement",
                "lexical_declaration",
                "return_statement",
                "switch_case",
                "switch_statement",
                "throw_statement",
                "try_statement",
                "variable_declaration",
                "while_statement",
                "with_statement",
            }
        ),
    ),
    "rust": LangSpec(
        "rust",
        "Rust",
        "Rust",
        "rust",
        (".rs",),
        frozenset(
            {
                "break_expression",
                "continue_expression",
                "expression_statement",
                "for_expression",
                "if_expression",
                "let_declaration",
                "loop_expression",
                "match_expression",
                "return_expression",
                "while_expression",
            }
        ),
    ),
    "go": LangSpec(
        "go",
        "Go",
        "Go",
        "go",
        (".go",),
        frozenset(
            {
                "assignment_statement",
                "break_statement",
                "communication_case",
                "const_declaration",
                "continue_statement",
                "defer_statement",
                "expression_statement",
                "fallthrough_statement",
                "for_statement",
                "go_statement",
                "goto_statement",
                "if_statement",
                "import_declaration",
                "inc_statement",
                "labeled_statement",
                "range_clause",
                "return_statement",
                "select_statement",
                "send_statement",
                "short_var_declaration",
                "switch_statement",
                "type_switch_statement",
                "var_declaration",
            }
        ),
    ),
    "csharp": LangSpec(
        "csharp",
        "C#",
        "C#",
        "csharp",
        (".cs",),
        frozenset(
            {
                "break_statement",
                "checked_statement",
                "continue_statement",
                "declaration_expression",
                "do_statement",
                "empty_statement",
                "expression_statement",
                "fixed_statement",
                "for_each_statement",
                "for_statement",
                "if_statement",
                "labeled_statement",
                "local_declaration_statement",
                "lock_statement",
                "return_statement",
                "switch_expression",
                "switch_statement",
                "throw_statement",
                "try_statement",
                "unsafe_statement",
                "using_statement",
                "while_statement",
                "yield_statement",
            }
        ),
    ),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=SCRIPT_VERSION)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stmtlen-study"),
        help="Canonical cache directory and destination for statement_lengths_slide.tex",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional separate cache directory; by default the cache remains in --out-dir",
    )
    p.add_argument("--languages", default=",".join(LANGS.keys()), help="Comma-separated language keys")
    p.add_argument("--min-files", type=int, default=100)
    p.add_argument("--min-bytes", type=int, default=20_000)
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    p.add_argument("--min-tokens", type=int, default=2, help="Keep statements with token length > this")
    p.add_argument("--max-files-per-repo", type=int, default=1)
    p.add_argument("--strategy", choices=("repo-tree", "code-search"), default="repo-tree")
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--sleep", type=float, default=0.25, help="Seconds between GitHub API calls")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-tests", action="store_true", help="Include tests/specs/examples paths")
    p.add_argument(
        "--compile-tex",
        action="store_true",
        help="Deprecated compatibility flag; ignored because this version emits TeX only",
    )
    p.add_argument(
        "--offline",
        "--dry-run-sampling",
        dest="offline",
        action="store_true",
        help="Never contact GitHub; analyze only source files already present in the cache",
    )
    p.add_argument(
        "--parse-mode",
        choices=("isolated", "in-process"),
        default="isolated",
        help="Run each Tree-sitter grammar in a child process so a native crash cannot kill the study",
    )
    p.add_argument(
        "--max-parser-crashes",
        type=int,
        default=20,
        help="Maximum native parser crashes to quarantine before aborting a language",
    )
    p.add_argument(
        "--parser-batch-size",
        type=int,
        default=25,
        help="Files parsed by each disposable worker before the parser is reloaded",
    )
    p.add_argument(
        "--max-replacement-rounds",
        type=int,
        default=5,
        help="Maximum rounds of sampling replacements for files that crash or yield no statements",
    )
    # Internal worker switches. They are intentionally omitted from --help.
    p.add_argument("--_parse-worker-language", help=argparse.SUPPRESS)
    p.add_argument(
        "--_parser-backend",
        choices=("get-parser", "language"),
        default="get-parser",
        help=argparse.SUPPRESS,
    )
    return p.parse_args(argv)


class GitHub:
    def __init__(self, sleep_s: float):
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Missing dependency: requests. Install with: pip install requests") from exc

        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.sleep_s = sleep_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": DEFAULT_API_VERSION,
                "User-Agent": "stmtlen-tree-sitter-study/1.0",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def get(self, path_or_url: str, **params: Any) -> Any:
        url = path_or_url if path_or_url.startswith("http") else f"{GITHUB_API}{path_or_url}"
        for attempt in range(8):
            if self.sleep_s:
                time.sleep(self.sleep_s)
            r = self.session.get(url, params={k: v for k, v in params.items() if v is not None}, timeout=60)
            if r.status_code == 403 and "rate limit" in r.text.lower():
                reset = r.headers.get("X-RateLimit-Reset")
                if reset and reset.isdigit():
                    wait = max(1, int(reset) - int(time.time()) + 2)
                else:
                    wait = min(300, 2 ** attempt * 10)
                print(f"GitHub rate-limited; sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code in {502, 503, 504}:
                time.sleep(min(120, 2 ** attempt))
                continue
            if r.status_code == 404:
                return None
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"GitHub GET failed: {r.status_code} {url} {r.text[:500]}") from e
            return r.json()
        raise RuntimeError(f"GitHub GET retried too many times: {url}")


def path_is_probably_source(path: str, spec: LangSpec, allow_tests: bool) -> bool:
    low = path.lower()
    if not low.endswith(spec.extensions):
        return False
    parts = {p for p in low.split("/") if p}
    if parts & VENDORISH_PARTS:
        return False
    if GENERATED_RE.search(low):
        return False
    if not allow_tests and any(p in parts for p in {"test", "tests", "testing", "spec", "specs", "fixtures", "example", "examples"}):
        return False
    return True


def safe_cache_name(repo: str, path: str, sha: str, ext: str) -> str:
    h = hashlib.sha1(f"{repo}\0{path}\0{sha}".encode()).hexdigest()[:20]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{repo.replace('/', '__')}__{Path(path).stem}")[:80]
    return f"{base}__{h}{ext}"


def decode_blob(blob: dict[str, Any]) -> bytes | None:
    if not blob or blob.get("encoding") != "base64":
        return None
    data = base64.b64decode(blob.get("content", ""), validate=False)
    if b"\x00" in data[:4096]:
        return None
    return data


def already_sampled(manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the manifest, tolerating blank, malformed, or duplicate lines."""
    by_lang: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    if not manifest_path.exists():
        return by_lang
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: ignoring malformed manifest line {line_number} in {manifest_path}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(rec, dict) or not rec.get("language"):
                continue
            identity = (
                str(rec.get("language", "")),
                str(rec.get("repo", "")),
                str(rec.get("sha", "")),
                str(rec.get("path", "")),
            )
            if identity in seen:
                continue
            seen.add(identity)
            by_lang.setdefault(str(rec["language"]), []).append(rec)
    return by_lang


def normalized_record_cache_path(rec: dict[str, Any], cache_dir: Path) -> Path:
    """Resolve a manifest record to its existing local source file when possible.

    The old scripts stored absolute paths.  When the experiment directory is moved, the
    absolute path can become stale even though the source is still present under
    ``<cache_dir>/src/<language>/``.  Prefer that in-tree copy by basename, then fall back to
    the recorded path.  Returning the canonical in-tree path for a missing file lets the
    caller report or replace only the genuinely absent entry.
    """
    raw_text = str(rec.get("cache_path", "")).strip()
    raw = Path(raw_text).expanduser() if raw_text else None
    language = str(rec.get("language", ""))
    basename = raw.name if raw is not None else ""

    if basename and language:
        canonical = (cache_dir / "src" / language / basename).resolve()
        if canonical.is_file():
            return canonical
    else:
        canonical = (cache_dir / "src" / language / basename).resolve()

    if raw is not None:
        candidates = [raw]
        if not raw.is_absolute():
            candidates.insert(0, cache_dir / raw)
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate

    return canonical


def locally_cached_samples(
        manifest_path: Path,
        cache_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Return only manifest records whose source bytes already exist locally."""
    result: dict[str, list[dict[str, Any]]] = {}
    for language, records in already_sampled(manifest_path).items():
        for original in records:
            rec = dict(original)
            cache_path = normalized_record_cache_path(rec, cache_dir)
            rec["cache_path"] = str(cache_path)
            if cache_path.is_file():
                result.setdefault(language, []).append(rec)
    return result


def _canonical_path(path: Path) -> str:
    """Return a stable absolute path string without requiring the target to exist."""
    return str(path.expanduser().resolve())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_records(cache_dir: Path, language: str) -> list[dict[str, Any]]:
    """Recover source metadata embedded in successful, failed, or in-flight checkpoints."""
    records: list[dict[str, Any]] = []
    for root_name in ("parse_cache", "parse_failures"):
        root = cache_dir / root_name / language
        if not root.is_dir():
            continue
        for checkpoint in sorted(root.glob("*.json")):
            payload = read_json(checkpoint)
            record = payload.get("record") if isinstance(payload, dict) else None
            if isinstance(record, dict) and record.get("language") == language:
                records.append(dict(record))

    current = read_json(cache_dir / "parse_current" / f"{language}.json")
    record = current.get("record") if isinstance(current, dict) else None
    if isinstance(record, dict) and record.get("language") == language:
        records.append(dict(record))
    return records


def _infer_repo_from_cache_filename(path: Path, language: str) -> str:
    """Best-effort recovery of ``owner/repository`` from ``safe_cache_name`` output."""
    name = path.name
    core = name
    for suffix in sorted(path.suffixes, key=len, reverse=True):
        if core.lower().endswith(suffix.lower()):
            core = core[: -len(suffix)]
            break
    core = re.sub(r"__[0-9a-fA-F]{20}$", "", core)
    parts = core.split("__")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return f"{parts[0]}/{parts[1]}"
    return f"local-cache/{language}"


def _synthesized_local_record(path: Path, spec: LangSpec) -> dict[str, Any]:
    """Create sufficient metadata for a source file whose manifest entry was lost."""
    content_sha = _file_sha256(path)
    return {
        "language": spec.key,
        "label": spec.label,
        "repo": _infer_repo_from_cache_filename(path, spec.key),
        "repo_id": None,
        "stars": None,
        "default_branch": None,
        "path": f"recovered-cache/{path.name}",
        "sha": f"local-sha256:{content_sha}",
        "bytes": path.stat().st_size,
        "strategy": "local-cache-recovery",
        "query": None,
        "cache_path": _canonical_path(path),
        "html_url": None,
        "local_inventory_schema": LOCAL_INVENTORY_SCHEMA,
    }


def _valid_local_source_paths(
        cache_dir: Path,
        spec: LangSpec,
        min_bytes: int,
        max_bytes: int,
) -> tuple[list[Path], int]:
    """Inventory regular source files on disk, independently of the JSONL manifest."""
    root = cache_dir / "src" / spec.key
    if not root.is_dir():
        return [], 0
    valid: list[Path] = []
    ignored = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.lower().endswith(spec.extensions):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            ignored += 1
            continue
        if min_bytes <= size <= max_bytes:
            valid.append(path.resolve())
        else:
            ignored += 1
    return valid, ignored


def reconcile_manifest_with_local_sources(
        cache_dir: Path,
        specs: list[LangSpec],
        min_bytes: int,
        max_bytes: int,
) -> None:
    """Merge the manifest, parser checkpoints, and the physical ``src`` inventory.

    Earlier versions treated ``sampled_files.jsonl`` as the sole authority.  Consequently a
    complete cache could be reported as incomplete after a manifest was truncated or rewritten.
    This routine makes local source bytes authoritative: every qualifying file already under
    ``src/<language>`` is restored to the manifest before any decision about GitHub is made.
    """
    manifest_path = cache_dir / "sampled_files.jsonl"
    existing_by_language = already_sampled(manifest_path)
    all_records: list[dict[str, Any]] = [
        dict(record)
        for records in existing_by_language.values()
        for record in records
    ]

    # Preserve records for unselected languages exactly as they are, while normalizing selected
    # records and recovering additional metadata from parser checkpoints.
    selected = {spec.key: spec for spec in specs}
    by_path: dict[str, dict[str, Any]] = {}
    identity_to_index: dict[tuple[str, str, str, str], int] = {}
    merged: list[dict[str, Any]] = []

    def add_record(record: dict[str, Any], *, require_local_file: bool = False) -> bool:
        language = str(record.get("language", ""))
        if not language:
            return False
        rec = dict(record)
        local_path: Path | None = None
        local_key: str | None = None
        if language in selected:
            local_path = normalized_record_cache_path(rec, cache_dir)
            rec["cache_path"] = _canonical_path(local_path)
            if require_local_file and not local_path.is_file():
                return False
            if local_path.is_file():
                local_key = _canonical_path(local_path)

        identity = (
            language,
            str(rec.get("repo", "")),
            str(rec.get("sha", "")),
            str(rec.get("path", "")),
        )
        existing_index = identity_to_index.get(identity)
        if existing_index is not None:
            # A checkpoint may contain the same logical record as a stale manifest entry but
            # with the only working cache_path. Prefer the locally backed version in place.
            if local_key is not None:
                previous = merged[existing_index]
                previous_path = normalized_record_cache_path(previous, cache_dir)
                if not previous_path.is_file():
                    merged[existing_index] = rec
                    by_path[local_key] = rec
                    return True
                by_path.setdefault(local_key, previous)
            return False

        if local_key is not None and local_key in by_path:
            return False

        identity_to_index[identity] = len(merged)
        merged.append(rec)
        if local_key is not None:
            by_path[local_key] = rec
        return True
    for record in all_records:
        add_record(record)

    for spec in specs:
        recovered_from_checkpoints = 0
        for record in _checkpoint_records(cache_dir, spec.key):
            if add_record(record, require_local_file=True):
                recovered_from_checkpoints += 1

        physical, ignored = _valid_local_source_paths(
            cache_dir,
            spec,
            min_bytes,
            max_bytes,
        )
        synthesized = 0
        for path in physical:
            key = _canonical_path(path)
            if key in by_path:
                continue
            if add_record(_synthesized_local_record(path, spec), require_local_file=True):
                synthesized += 1

        indexed = sum(
            1
            for path_key, record in by_path.items()
            if record.get("language") == spec.key and Path(path_key).is_file()
        )
        details: list[str] = []
        if recovered_from_checkpoints:
            details.append(f"{recovered_from_checkpoints} restored from parser checkpoints")
        if synthesized:
            details.append(f"{synthesized} reconstructed from src/{spec.key}")
        if ignored:
            details.append(f"{ignored} nonqualifying/unreadable source entries ignored")
        suffix = f" ({'; '.join(details)})" if details else ""
        print(
            f"[{spec.label}] physical cache inventory: {len(physical)} qualifying files; "
            f"{indexed} indexed{suffix}",
            file=sys.stderr,
            flush=True,
        )

    # Keep deterministic ordering, which also makes later runs byte-for-byte stable.
    merged.sort(
        key=lambda record: (
            str(record.get("language", "")),
            str(record.get("repo", "")),
            str(record.get("path", "")),
            str(record.get("sha", "")),
        )
    )
    write_manifest(manifest_path, merged)


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def random_repo_candidate(gh: GitHub, spec: LangSpec, rng: random.Random) -> dict[str, Any] | None:
    bucket = rng.choice(STAR_BUCKETS)
    q = f"language:{spec.github_language} fork:false archived:false mirror:false {bucket}"
    first = gh.get("/search/repositories", q=q, per_page=1, page=1)
    if not first or first.get("total_count", 0) == 0:
        return None
    cap = min(first["total_count"], 1000)
    page = rng.randint(1, max(1, math.ceil(cap / 100)))
    result = gh.get("/search/repositories", q=q, per_page=100, page=page)
    items = result.get("items", []) if result else []
    if not items:
        return None
    return rng.choice(items)


def sample_via_repo_tree(
        gh: GitHub,
        spec: LangSpec,
        rng: random.Random,
        out_dir: Path,
        manifest_path: Path,
        have_repo_counts: dict[str, int],
        seen_file_keys: set[str],
        args: argparse.Namespace,
) -> bool:
    repo = random_repo_candidate(gh, spec, rng)
    if not repo:
        return False
    full_name = repo["full_name"]
    if have_repo_counts.get(full_name, 0) >= args.max_files_per_repo:
        return False
    branch = repo.get("default_branch") or "HEAD"
    tree = gh.get(f"/repos/{full_name}/git/trees/{quote(branch, safe='')}", recursive="1")
    if not tree or tree.get("truncated"):
        # Large repos may be truncated. Skipping avoids partial-tree bias.
        return False
    candidates = []
    for ent in tree.get("tree", []):
        path = ent.get("path", "")
        size = ent.get("size")
        if ent.get("type") != "blob" or size is None:
            continue
        if not (args.min_bytes <= int(size) <= args.max_bytes):
            continue
        if not path_is_probably_source(path, spec, args.allow_tests):
            continue
        key = f"{full_name}:{ent.get('sha')}:{path}"
        if key in seen_file_keys:
            continue
        candidates.append(ent)
    if not candidates:
        return False
    ent = rng.choice(candidates)
    blob = gh.get(ent["url"])
    data = decode_blob(blob)
    if not data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    if len(text.encode("utf-8")) < args.min_bytes:
        return False
    ext = next((e for e in spec.extensions if ent["path"].lower().endswith(e)), spec.extensions[0])
    cache_dir = out_dir / "src" / spec.key
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / safe_cache_name(full_name, ent["path"], ent.get("sha", ""), ext)
    cache_path.write_text(text, encoding="utf-8")
    rec = {
        "language": spec.key,
        "label": spec.label,
        "repo": full_name,
        "repo_id": repo.get("id"),
        "stars": repo.get("stargazers_count"),
        "default_branch": branch,
        "path": ent["path"],
        "sha": ent.get("sha"),
        "bytes": len(text.encode("utf-8")),
        "strategy": "repo-tree",
        "cache_path": str(cache_path),
        "html_url": f"https://github.com/{full_name}/blob/{branch}/{ent['path']}",
    }
    append_jsonl(manifest_path, rec)
    seen_file_keys.add(f"{full_name}:{ent.get('sha')}:{ent['path']}")
    have_repo_counts[full_name] = have_repo_counts.get(full_name, 0) + 1
    return True


def code_search_query(spec: LangSpec, rng: random.Random, args: argparse.Namespace) -> str:
    anchor = rng.choice(CODE_SEARCH_ANCHORS.get(spec.key, ["return", "if", "for"]))
    ext = rng.choice(spec.extensions).lstrip(".")
    # The legacy code-search API supports language/extension/size qualifiers, but still
    # requires a non-qualifier search term.
    return f"{anchor} language:{spec.github_language} extension:{ext} size:>{args.min_bytes} fork:false"


def sample_via_code_search(
        gh: GitHub,
        spec: LangSpec,
        rng: random.Random,
        out_dir: Path,
        manifest_path: Path,
        have_repo_counts: dict[str, int],
        seen_file_keys: set[str],
        args: argparse.Namespace,
) -> bool:
    q = code_search_query(spec, rng, args)
    first = gh.get("/search/code", q=q, per_page=1, page=1)
    if not first or first.get("total_count", 0) == 0:
        return False
    cap = min(first["total_count"], 1000)
    page = rng.randint(1, max(1, math.ceil(cap / 100)))
    result = gh.get("/search/code", q=q, per_page=100, page=page)
    items = result.get("items", []) if result else []
    rng.shuffle(items)
    for item in items:
        repo = item.get("repository", {})
        full_name = repo.get("full_name")
        path = item.get("path", "")
        sha = item.get("sha", "")
        if not full_name or not sha or not path_is_probably_source(path, spec, args.allow_tests):
            continue
        if have_repo_counts.get(full_name, 0) >= args.max_files_per_repo:
            continue
        key = f"{full_name}:{sha}:{path}"
        if key in seen_file_keys:
            continue
        git_url = item.get("git_url")
        if not git_url:
            continue
        blob = gh.get(git_url)
        if not blob:
            continue
        size = int(blob.get("size") or 0)
        if not (args.min_bytes <= size <= args.max_bytes):
            continue
        data = decode_blob(blob)
        if not data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        ext = next((e for e in spec.extensions if path.lower().endswith(e)), spec.extensions[0])
        cache_dir = out_dir / "src" / spec.key
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / safe_cache_name(full_name, path, sha, ext)
        cache_path.write_text(text, encoding="utf-8")
        rec = {
            "language": spec.key,
            "label": spec.label,
            "repo": full_name,
            "repo_id": repo.get("id"),
            "stars": repo.get("stargazers_count"),
            "default_branch": repo.get("default_branch"),
            "path": path,
            "sha": sha,
            "bytes": len(text.encode("utf-8")),
            "strategy": "code-search",
            "query": q,
            "cache_path": str(cache_path),
            "html_url": item.get("html_url"),
        }
        append_jsonl(manifest_path, rec)
        seen_file_keys.add(key)
        have_repo_counts[full_name] = have_repo_counts.get(full_name, 0) + 1
        return True
    return False


def sample_files(
        args: argparse.Namespace,
        specs: list[LangSpec],
        target_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Ensure the cache contains the requested number of local files per language.

    This function is deliberately cache-first.  It does not even construct ``GitHub`` when
    every requested source is already present under ``src/``.  Manifest entries whose source
    bytes are missing do not count toward the target.
    """
    cache_dir = args.work_dir
    manifest_path = cache_dir / "sampled_files.jsonl"
    targets = target_counts or {spec.key: args.min_files for spec in specs}
    selected_keys = {spec.key for spec in specs}

    manifest_records = already_sampled(manifest_path) if args.resume else {}
    if not args.resume and manifest_path.exists():
        manifest_path.unlink()
        manifest_records = {}
    cached = locally_cached_samples(manifest_path, cache_dir) if args.resume else {}

    deficits: dict[str, int] = {}
    for spec in specs:
        target = int(targets.get(spec.key, args.min_files))
        present = len(cached.get(spec.key, []))
        manifest_count = len(manifest_records.get(spec.key, []))
        suffix = ""
        if manifest_count > present:
            suffix = f" ({manifest_count - present} manifest entries have no local source)"
        print(
            f"[{spec.label}] have {present}/{target} locally cached files{suffix}",
            file=sys.stderr,
            flush=True,
        )
        if present < target:
            deficits[spec.key] = target - present

    if not deficits:
        print(
            "All requested source files are already in the local cache; skipping GitHub entirely.",
            file=sys.stderr,
            flush=True,
        )
        return [rec for key in selected_keys for rec in cached.get(key, [])]

    if args.offline:
        return [rec for key in selected_keys for rec in cached.get(key, [])]

    # Network dependencies are imported only after a genuine local-cache deficit is known.
    gh = GitHub(args.sleep)
    rng = random.Random(args.seed)
    seen_file_keys = {
        f"{rec.get('repo')}:{rec.get('sha')}:{rec.get('path')}"
        for rows in manifest_records.values()
        for rec in rows
    }
    have_repo_counts: dict[str, int] = {}
    for rows in manifest_records.values():
        for rec in rows:
            repo = str(rec.get("repo", ""))
            if repo:
                have_repo_counts[repo] = have_repo_counts.get(repo, 0) + 1

    for spec in specs:
        target = int(targets.get(spec.key, args.min_files))
        attempts = 0
        while len(cached.get(spec.key, [])) < target:
            attempts += 1
            if attempts > max(target, 1) * 400:
                raise RuntimeError(
                    f"Too many failed sampling attempts for {spec.key}; "
                    f"only got {len(cached.get(spec.key, []))}/{target} local files"
                )
            ok = (
                sample_via_repo_tree(
                    gh,
                    spec,
                    rng,
                    cache_dir,
                    manifest_path,
                    have_repo_counts,
                    seen_file_keys,
                    args,
                )
                if args.strategy == "repo-tree"
                else sample_via_code_search(
                    gh,
                    spec,
                    rng,
                    cache_dir,
                    manifest_path,
                    have_repo_counts,
                    seen_file_keys,
                    args,
                )
            )
            if ok:
                manifest_records = already_sampled(manifest_path)
                cached = locally_cached_samples(manifest_path, cache_dir)
                print(
                    f"[{spec.label}] sampled {len(cached.get(spec.key, []))}/{target}",
                    file=sys.stderr,
                    flush=True,
                )

    final = locally_cached_samples(manifest_path, cache_dir)
    return [rec for key in selected_keys for rec in final.get(key, [])]


def is_statement_node(node_type: str, spec: LangSpec) -> bool:
    if node_type in EXCLUDED_DECLARATION_TYPES:
        return False
    if node_type in spec.statement_types:
        return True
    # Conservative fallback for grammars with explicit *_statement nodes.
    return node_type.endswith("_statement") and node_type not in BODY_OR_BLOCK_TYPES


def child_count(node: Any) -> int:
    return int(getattr(node, "child_count", 0))


def _advance_cursor_after_subtree(cursor: Any) -> bool:
    """Move to the next node after the current subtree in depth-first order."""
    if cursor.goto_next_sibling():
        return True
    while cursor.goto_parent():
        if cursor.goto_next_sibling():
            return True
    return False


def token_count_for_node(node: Any, spec: LangSpec, *, root_statement: Any | None = None) -> int:
    """Count lexical Tree-sitter leaves using a cursor and no Python recursion.

    For a compound statement, nested statement/block bodies are pruned so the count describes
    the statement itself rather than recursively including every statement in its body. Comments
    and ERROR subtrees are ignored, and compound literal subtrees count as one lexical token.

    ``root_statement`` is retained for call-site compatibility; ``node`` is always the root.
    """
    del root_statement
    total = 0
    cursor = node.walk()
    is_root = True

    while True:
        cur = cursor.node
        node_type = cur.type
        prune = False

        if node_type in EXTRA_OR_COMMENT_TYPES or node_type == "ERROR":
            prune = True
        elif not is_root and (node_type in BODY_OR_BLOCK_TYPES or is_statement_node(node_type, spec)):
            prune = True
        elif node_type in LITERAL_ATOMS:
            total += 1
            prune = True
        elif child_count(cur) == 0:
            # Tree-sitter omits whitespace leaves. Anonymous leaves are punctuation/keywords.
            total += 1
            prune = True

        if not prune and cursor.goto_first_child():
            is_root = False
            continue

        is_root = False
        if not _advance_cursor_after_subtree(cursor):
            return total


def iter_statement_nodes(node: Any, spec: LangSpec) -> Iterator[Any]:
    """Yield statement nodes in depth-first order using TreeCursor.

    TreeCursor avoids materializing ``node.children`` lists for the entire frontier, sharply
    reducing the number and lifetime of Python Node wrappers in the native-parser worker.
    """
    cursor = node.walk()
    while True:
        cur = cursor.node
        if is_statement_node(cur.type, spec):
            yield cur

        if cursor.goto_first_child():
            continue
        if not _advance_cursor_after_subtree(cursor):
            return


def line_number_for_byte(line_starts: list[int], byte_offset: int) -> int:
    """Return a 1-based line number without constructing Tree-sitter Point objects."""
    return bisect.bisect_right(line_starts, max(0, int(byte_offset)))


def analyze_file(rec: dict[str, Any], spec: LangSpec, parser: Any, min_tokens: int) -> list[dict[str, Any]]:
    path = Path(rec["cache_path"])
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cached source file is missing: {path}") from exc

    # Compute these once. The previous version used
    #     rec.get("bytes", path.stat().st_size)
    # inside the statement loop. Python eagerly evaluates dict.get's default, so that
    # issued one stat() syscall per statement even when rec["bytes"] was present. Besides
    # being expensive, those allocations repeatedly triggered the py-tree-sitter/GC crash.
    file_size = len(data)
    line_starts = [0]
    line_starts.extend(i + 1 for i, byte in enumerate(data) if byte == 0x0A)

    tree = parser.parse(data)
    if tree is None:
        raise RuntimeError(f"Tree-sitter returned no tree for {path}")

    rows: list[dict[str, Any]] = []
    root = tree.root_node
    for i, node in enumerate(iter_statement_nodes(root, spec)):
        n = token_count_for_node(node, spec, root_statement=node)
        if n <= min_tokens:
            continue

        start_byte = int(node.start_byte)
        end_byte = int(node.end_byte)
        # end_byte is exclusive; report the line containing the statement's final byte.
        final_byte = start_byte if end_byte <= start_byte else end_byte - 1
        rows.append(
            {
                "language": spec.key,
                "label": spec.label,
                "repo": rec.get("repo", ""),
                "path": rec.get("path", ""),
                "sha": rec.get("sha", ""),
                "file_cache": str(path),
                "statement_index": i,
                "statement_type": node.type,
                "tokens": n,
                "start_line": line_number_for_byte(line_starts, start_byte),
                "end_line": line_number_for_byte(line_starts, final_byte),
                "bytes": file_size,
            }
        )

    # Rows contain only ordinary Python values; no Tree/Node object escapes this function.
    del root, tree
    return rows




def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON checkpoint atomically so a SIGSEGV cannot leave a partial result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for dist, key in (
            ("tree-sitter", "tree_sitter"),
            ("tree-sitter-language-pack", "tree_sitter_language_pack"),
    ):
        try:
            versions[key] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def record_cache_id(rec: dict[str, Any], min_tokens: int) -> str:
    identity = "\0".join(
        [
            str(PARSE_CACHE_SCHEMA),
            str(min_tokens),
            str(rec.get("language", "")),
            str(rec.get("repo", "")),
            str(rec.get("sha", "")),
            str(rec.get("path", "")),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()[:32]


def parse_result_path(out_dir: Path, rec: dict[str, Any], min_tokens: int) -> Path:
    rid = record_cache_id(rec, min_tokens)
    return out_dir / "parse_cache" / rec["language"] / f"{rid}.json"


def parse_failure_path(out_dir: Path, rec: dict[str, Any], min_tokens: int) -> Path:
    rid = record_cache_id(rec, min_tokens)
    return out_dir / "parse_failures" / rec["language"] / f"{rid}.json"


def parse_current_path(out_dir: Path, language: str) -> Path:
    return out_dir / "parse_current" / f"{language}.json"


def build_parser(parser_key: str, backend: str) -> Any:
    """Construct a parser using either language-pack convenience or its raw Language."""
    try:
        if backend == "get-parser":
            from tree_sitter_language_pack import get_parser as language_pack_get_parser

            return language_pack_get_parser(parser_key)

        # This alternate construction avoids the package's get_parser wrapper while still using
        # the same grammar binary. It is useful for diagnosing native-wrapper compatibility.
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing parser dependency. Install tree-sitter and tree-sitter-language-pack."
        ) from exc

    language = get_language(parser_key)
    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        parser.language = language
        return parser


def worker_records(args: argparse.Namespace, language: str) -> list[dict[str, Any]]:
    rows = locally_cached_samples(args.work_dir / "sampled_files.jsonl", args.work_dir).get(language, [])
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("repo", "")),
            str(r.get("path", "")),
            str(r.get("sha", "")),
        ),
    )


def parse_worker_main(args: argparse.Namespace) -> int:
    """Parse one language in a disposable child process with per-file checkpoints.

    py-tree-sitter currently has an open native crash whose stack can surface while
    CPython's cyclic collector is running. Reference counting remains active, and the
    traversal below creates no intentional cycles, so disabling only cyclic GC in this
    short-lived worker is a safe containment measure.
    """
    # Collect anything created during Python startup, then disable cyclic GC before importing
    # either tree_sitter or tree_sitter_language_pack. Reference counting remains enabled.
    if gc.isenabled():
        gc.collect()
        gc.disable()

    language = str(args._parse_worker_language)
    spec = LANGS.get(language)
    if spec is None:
        print(f"Unknown parse-worker language: {language}", file=sys.stderr)
        return 2

    current = parse_current_path(args.work_dir, language)
    atomic_write_json(
        current,
        {
            "stage": "parser_init",
            "language": language,
            "label": spec.label,
            "parser_key": spec.parser_key,
            "backend": args._parser_backend,
            "versions": package_versions(),
        },
    )
    print(
        f"[{spec.label}] loading Tree-sitter parser ({args._parser_backend}; cyclic GC disabled)",
        file=sys.stderr,
        flush=True,
    )
    parser = build_parser(spec.parser_key, args._parser_backend)
    records = worker_records(args, language)
    max_files = max(1, int(args.parser_batch_size))
    attempted = 0

    for index, rec in enumerate(records, start=1):
        result_path = parse_result_path(args.work_dir, rec, args.min_tokens)
        failure_path = parse_failure_path(args.work_dir, rec, args.min_tokens)
        if result_path.exists() or failure_path.exists():
            continue
        attempted += 1
        rid = record_cache_id(rec, args.min_tokens)
        atomic_write_json(
            current,
            {
                "stage": "parse",
                "language": language,
                "label": spec.label,
                "record_id": rid,
                "backend": args._parser_backend,
                "index": index,
                "total": len(records),
                "record": rec,
                "versions": package_versions(),
            },
        )
        print(
            f"[{spec.label}] parsing {index}/{len(records)} {rec.get('repo', '')}:{rec.get('path', '')}",
            file=sys.stderr,
            flush=True,
        )
        try:
            rows = analyze_file(rec, spec, parser, args.min_tokens)
            atomic_write_json(
                result_path,
                {
                    "schema": PARSE_CACHE_SCHEMA,
                    "status": "ok",
                    "record_id": rid,
                    "record": rec,
                    "backend": args._parser_backend,
                    "rows": rows,
                },
            )
        except Exception as exc:
            atomic_write_json(
                failure_path,
                {
                    "schema": PARSE_CACHE_SCHEMA,
                    "status": "python_exception",
                    "record_id": rid,
                    "record": rec,
                    "backend": args._parser_backend,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "versions": package_versions(),
                },
            )
            print(
                f"[{spec.label}] skipped after {type(exc).__name__}: {rec.get('repo', '')}:{rec.get('path', '')}",
                file=sys.stderr,
                flush=True,
            )
        current.unlink(missing_ok=True)
        if attempted >= max_files:
            print(
                f"[{spec.label}] completed parser batch of {attempted} files; recycling worker",
                file=sys.stderr,
                flush=True,
            )
            return 0

    current.unlink(missing_ok=True)
    return 0


def record_is_finished(args: argparse.Namespace, rec: dict[str, Any]) -> bool:
    return parse_result_path(args.work_dir, rec, args.min_tokens).exists() or parse_failure_path(
        args.work_dir, rec, args.min_tokens
    ).exists()


def all_language_records_finished(args: argparse.Namespace, language: str) -> bool:
    return all(record_is_finished(args, rec) for rec in worker_records(args, language))


def worker_command(args: argparse.Namespace, spec: LangSpec, backend: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--work-dir",
        str(args.work_dir),
        "--min-tokens",
        str(args.min_tokens),
        "--parser-batch-size",
        str(args.parser_batch_size),
        "--_parse-worker-language",
        spec.key,
        "--_parser-backend",
        backend,
    ]


def signal_name(returncode: int) -> str:
    if returncode >= 0:
        return f"exit status {returncode}"
    try:
        return signal.Signals(-returncode).name
    except (ValueError, AttributeError):
        return f"signal {-returncode}"


def run_isolated_language(args: argparse.Namespace, spec: LangSpec) -> None:
    """Run/restart a language worker, quarantining only files that crash native code."""
    backend = "get-parser"
    tried_init_backends: set[str] = set()
    crash_attempts: dict[tuple[str, str], int] = {}
    quarantined = 0

    while not all_language_records_finished(args, spec.key):
        current_path = parse_current_path(args.work_dir, spec.key)
        current_path.unlink(missing_ok=True)
        proc = subprocess.run(worker_command(args, spec, backend), check=False)
        if proc.returncode == 0:
            if all_language_records_finished(args, spec.key):
                return
            # A healthy worker intentionally exits after --parser-batch-size files.
            continue
        if all_language_records_finished(args, spec.key):
            print(
                f"[{spec.label}] worker ended with {signal_name(proc.returncode)} after writing all checkpoints; accepting cached results",
                file=sys.stderr,
                flush=True,
            )
            return

        current = read_json(current_path) or {}
        stage = current.get("stage")
        if stage == "parser_init":
            tried_init_backends.add(backend)
            alternate = "language" if backend == "get-parser" else "get-parser"
            if alternate not in tried_init_backends:
                print(
                    f"[{spec.label}] parser initialization ended with {signal_name(proc.returncode)}; retrying alternate construction",
                    file=sys.stderr,
                    flush=True,
                )
                backend = alternate
                continue
            versions = current.get("versions", package_versions())
            raise RuntimeError(
                f"Tree-sitter parser initialization failed for {spec.label} with both construction paths "
                f"({signal_name(proc.returncode)}). Parser stack: {versions}."
            )

        if stage != "parse" or not isinstance(current.get("record"), dict):
            raise RuntimeError(
                f"{spec.label} parser worker failed with {signal_name(proc.returncode)} without a recoverable file checkpoint."
            )

        rec = current["record"]
        rid = str(current.get("record_id") or record_cache_id(rec, args.min_tokens))
        result_path = parse_result_path(args.work_dir, rec, args.min_tokens)
        if result_path.exists():
            # The worker may have crashed during native cleanup after committing this file.
            continue

        key = (rid, backend)
        crash_attempts[key] = crash_attempts.get(key, 0) + 1
        if backend == "get-parser" and crash_attempts[key] == 1:
            print(
                f"[{spec.label}] native crash in {rec.get('repo', '')}:{rec.get('path', '')}; retrying via raw Language construction",
                file=sys.stderr,
                flush=True,
            )
            backend = "language"
            continue

        failure_path = parse_failure_path(args.work_dir, rec, args.min_tokens)
        atomic_write_json(
            failure_path,
            {
                "schema": PARSE_CACHE_SCHEMA,
                "status": "native_crash",
                "record_id": rid,
                "record": rec,
                "backend": backend,
                "returncode": proc.returncode,
                "signal": signal_name(proc.returncode),
                "versions": current.get("versions", package_versions()),
            },
        )
        quarantined += 1
        print(
            f"[{spec.label}] quarantined crashing file ({signal_name(proc.returncode)}): "
            f"{rec.get('repo', '')}:{rec.get('path', '')}",
            file=sys.stderr,
            flush=True,
        )
        if quarantined > args.max_parser_crashes:
            raise RuntimeError(
                f"Aborting {spec.label}: more than {args.max_parser_crashes} files crashed the native parser."
            )


def analyze_records_in_process(
        args: argparse.Namespace,
        specs: list[LangSpec],
        records: list[dict[str, Any]],
) -> None:
    """Compatibility/debug mode. Isolated mode is safer and remains the default."""
    if gc.isenabled():
        gc.collect()
        gc.disable()
    print(
        "In-process parser mode: cyclic GC disabled; isolated mode remains safer",
        file=sys.stderr,
        flush=True,
    )
    by_language = groupby(records, lambda r: r["language"])
    for spec in specs:
        parser = build_parser(spec.parser_key, "get-parser")
        lang_records = by_language.get(spec.key, [])
        for index, rec in enumerate(lang_records, start=1):
            result_path = parse_result_path(args.work_dir, rec, args.min_tokens)
            failure_path = parse_failure_path(args.work_dir, rec, args.min_tokens)
            if result_path.exists() or failure_path.exists():
                continue
            print(
                f"[{spec.label}] parsing {index}/{len(lang_records)} {rec.get('repo', '')}:{rec.get('path', '')}",
                file=sys.stderr,
                flush=True,
            )
            try:
                rows = analyze_file(rec, spec, parser, args.min_tokens)
                atomic_write_json(
                    result_path,
                    {
                        "schema": PARSE_CACHE_SCHEMA,
                        "status": "ok",
                        "record_id": record_cache_id(rec, args.min_tokens),
                        "record": rec,
                        "backend": "in-process",
                        "rows": rows,
                    },
                )
            except Exception as exc:
                atomic_write_json(
                    failure_path,
                    {
                        "schema": PARSE_CACHE_SCHEMA,
                        "status": "python_exception",
                        "record_id": record_cache_id(rec, args.min_tokens),
                        "record": rec,
                        "backend": "in-process",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "versions": package_versions(),
                    },
                )
        # Drop each native parser before loading the next grammar.
        del parser


def collect_parse_rows(
        args: argparse.Namespace,
        specs: list[LangSpec],
        records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = {spec.key for spec in specs}
    all_rows: list[dict[str, Any]] = []
    usable_files: dict[str, int] = {spec.key: 0 for spec in specs}
    for rec in records:
        if rec.get("language") not in selected:
            continue
        result = read_json(parse_result_path(args.work_dir, rec, args.min_tokens))
        if not result or result.get("status") != "ok":
            continue
        rows = result.get("rows")
        if not isinstance(rows, list):
            continue
        valid_rows = [row for row in rows if isinstance(row, dict)]
        all_rows.extend(valid_rows)
        if valid_rows:
            usable_files[rec["language"]] += 1
    return all_rows, usable_files


def analyze_records(
        args: argparse.Namespace,
        specs: list[LangSpec],
        records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if args.parse_mode == "isolated":
        for spec in specs:
            run_isolated_language(args, spec)
    else:
        analyze_records_in_process(args, specs, records)
    return collect_parse_rows(args, specs, records)


def write_failure_ledger(args: argparse.Namespace, specs: list[LangSpec], records: list[dict[str, Any]]) -> None:
    failures: list[dict[str, Any]] = []
    selected = {spec.key for spec in specs}
    for rec in records:
        if rec.get("language") not in selected:
            continue
        failure = read_json(parse_failure_path(args.work_dir, rec, args.min_tokens))
        if failure:
            failures.append(failure)
    path = args.work_dir / "parse_failures" / "_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, sort_keys=True, ensure_ascii=False) + "\n")

def summarize(
        rows: list[dict[str, Any]], specs: list[LangSpec]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_file: dict[tuple[str, str], list[int]] = {}
    file_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["language"], row["file_cache"])
        by_file.setdefault(key, []).append(int(row["tokens"]))
        file_meta[key] = row

    file_rows: list[dict[str, Any]] = []
    for key, values in sorted(by_file.items()):
        meta = file_meta[key]
        numeric = [float(value) for value in values]
        file_rows.append(
            {
                "language": key[0],
                "label": meta["label"],
                "repo": meta["repo"],
                "path": meta["path"],
                "file_cache": key[1],
                "n_statements": len(values),
                "mean_tokens": stats.fmean(numeric),
                "median_tokens": stats.median(numeric),
                "p25_tokens": percentile(numeric, 25),
                "p75_tokens": percentile(numeric, 75),
                "max_tokens": max(values),
            }
        )

    lang_rows: list[dict[str, Any]] = []
    order = {spec.key: index for index, spec in enumerate(specs)}
    grouped = groupby(file_rows, lambda row: row["language"])
    for language, group in sorted(grouped.items(), key=lambda item: order.get(item[0], 999)):
        file_means = [
            float(row["mean_tokens"])
            for row in group
            if int(row["n_statements"]) > 0
        ]
        statement_lengths = [
            int(row["tokens"]) for row in rows if row["language"] == language
        ]
        if not file_means:
            continue
        mean = stats.fmean(file_means)
        sd = stats.stdev(file_means) if len(file_means) > 1 else 0.0
        sem = sd / math.sqrt(len(file_means)) if len(file_means) > 1 else 0.0
        ci95 = 1.96 * sem
        label = next((spec.label for spec in specs if spec.key == language), language)
        lang_rows.append(
            {
                "language": language,
                "label": label,
                "n_files": len(file_means),
                "n_statements": len(statement_lengths),
                "mean_file_mean_tokens": mean,
                "sd_file_mean_tokens": sd,
                "sem_file_mean_tokens": sem,
                "ci95_file_mean_tokens": ci95,
                "median_statement_tokens": (
                    float(stats.median(statement_lengths)) if statement_lengths else math.nan
                ),
                "p25_statement_tokens": percentile(statement_lengths, 25),
                "p75_statement_tokens": percentile(statement_lengths, 75),
            }
        )
    return file_rows, lang_rows

def groupby(rows: Iterable[dict[str, Any]], keyfn: Any) -> dict[Any, list[dict[str, Any]]]:
    d: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        d.setdefault(keyfn(r), []).append(r)
    return d


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def format_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def nice_step(value: float) -> float:
    """Choose a conventional 1/2/2.5/5/10 axis step at or above value."""
    if not math.isfinite(value) or value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    scale = 10.0**exponent
    fraction = value / scale
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= candidate:
            return candidate * scale
    return 10.0 * scale


def render_inline_picture(lang_rows: list[dict[str, Any]]) -> str:
    """Return a package-free LaTeX picture chart with literal result coordinates."""
    if not lang_rows:
        raise ValueError("Cannot render an empty language summary")

    canvas_width = 148.0
    canvas_height = 63.0
    left = 14.0
    bottom = 14.0
    plot_width = 132.0
    plot_height = 43.0
    top_value = max(
        float(row["mean_file_mean_tokens"]) + float(row["ci95_file_mean_tokens"])
        for row in lang_rows
    )
    step = nice_step(max(top_value * 1.08, 1.0) / 5.0)
    axis_max = max(step, math.ceil(top_value / step) * step)
    tick_count = int(round(axis_max / step))

    lines: list[str] = [
        r"\setlength{\unitlength}{1mm}",
        rf"\begin{{picture}}({canvas_width:.1f},{canvas_height:.1f})",
        rf"  \put({left + plot_width / 2:.3f},{canvas_height - 1.2:.3f})"
        r"{\makebox(0,0){\small Mean lexical tokens per statement}}",
    ]

    for tick_index in range(tick_count + 1):
        tick_value = tick_index * step
        y = bottom + plot_height * tick_value / axis_max
        lines.extend(
            [
                rf"  \put({left:.3f},{y:.3f}){{\color{{black!12}}\line(1,0){{{plot_width:.3f}}}}}",
                rf"  \put({left - 1.0:.3f},{y:.3f}){{\color{{black}}\line(1,0){{1.0}}}}",
                rf"  \put({left - 1.6:.3f},{y:.3f})"
                rf"{{\makebox(0,0)[r]{{\tiny {format_tick(tick_value)}}}}}",
            ]
        )

    lines.extend(
        [
            rf"  \put({left:.3f},{bottom:.3f}){{\color{{black}}\line(1,0){{{plot_width:.3f}}}}}",
            rf"  \put({left:.3f},{bottom:.3f}){{\color{{black}}\line(0,1){{{plot_height:.3f}}}}}",
        ]
    )

    slot = plot_width / len(lang_rows)
    bar_width = min(8.0, slot * 0.58)
    cap_width = min(3.8, bar_width * 0.55)
    for index, row in enumerate(lang_rows):
        label = latex_escape(str(row["label"]))
        mean = float(row["mean_file_mean_tokens"])
        ci95 = float(row["ci95_file_mean_tokens"])
        n_files = int(row["n_files"])
        n_statements = int(row["n_statements"])
        center = left + (index + 0.5) * slot
        bar_x = center - bar_width / 2.0
        bar_height = plot_height * mean / axis_max
        lower = max(0.0, mean - ci95)
        upper = min(axis_max, mean + ci95)
        lower_y = bottom + plot_height * lower / axis_max
        upper_y = bottom + plot_height * upper / axis_max
        error_height = max(0.0, upper_y - lower_y)
        value_y = min(canvas_height - 3.4, upper_y + 1.2)
        lines.extend(
            [
                rf"  % {label}: mean={mean:.6f}, ci95={ci95:.6f}, "
                rf"n_files={n_files}, n_statements={n_statements}",
                rf"  \put({bar_x:.3f},{bottom:.3f})"
                rf"{{\usebeamercolor[fg]{{structure}}\rule{{{bar_width:.3f}\unitlength}}{{{bar_height:.3f}\unitlength}}}}",
                rf"  \put({center:.3f},{lower_y:.3f}){{\color{{black}}\line(0,1){{{error_height:.3f}}}}}",
                rf"  \put({center - cap_width / 2.0:.3f},{lower_y:.3f})"
                rf"{{\color{{black}}\line(1,0){{{cap_width:.3f}}}}}",
                rf"  \put({center - cap_width / 2.0:.3f},{upper_y:.3f})"
                rf"{{\color{{black}}\line(1,0){{{cap_width:.3f}}}}}",
                rf"  \put({center:.3f},{value_y:.3f})"
                rf"{{\makebox(0,0)[b]{{\tiny {mean:.1f}}}}}",
                rf"  \put({center:.3f},{bottom - 2.2:.3f})"
                rf"{{\makebox(0,0)[t]{{\tiny {label}}}}}",
                rf"  \put({center:.3f},{bottom - 6.0:.3f})"
                rf"{{\makebox(0,0)[t]{{\tiny $n={n_files}$}}}}",
            ]
        )

    lines.append(r"\end{picture}")
    return "\n".join(lines)


def write_beamer(lang_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    chart = render_inline_picture(lang_rows)
    result_comments = [
        "% Embedded numerical results (mean of per-file means, 95% CI over files):"
    ]
    for row in lang_rows:
        result_comments.append(
            "% "
            f"{row['label']}: mean={float(row['mean_file_mean_tokens']):.8f}, "
            f"ci95={float(row['ci95_file_mean_tokens']):.8f}, "
            f"n_files={int(row['n_files'])}, "
            f"n_statements={int(row['n_statements'])}"
        )

    tex = "\n".join(
        [
            r"\documentclass[aspectratio=169]{beamer}",
            r"\usetheme{default}",
            r"\setbeamertemplate{navigation symbols}{}",
            r"\setbeamerfont{frametitle}{size=\Large,series=\bfseries}",
            *result_comments,
            r"\begin{document}",
            r"\begin{frame}{How long is a typical source-code statement?}",
            r"\centering",
            chart,
            r"\vspace{-0.8ex}",
            r"{\scriptsize Bars are means of per-file statement means; whiskers are 95\% CIs over sampled files. "
            r"Statements are Tree-sitter statement/declaration nodes with comments and whitespace ignored; "
            r"only statements longer than two lexical tokens are included.}",
            r"\end{frame}",
            r"\end{document}",
            "",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "statement_lengths_slide.tex"
    path.write_text(tex, encoding="utf-8")
    return path


GENERATED_FILES_TO_REMOVE = (
    "statement_lengths_barplot.pdf",
    "statement_lengths_barplot.png",
    "statement_lengths_slide.aux",
    "statement_lengths_slide.fdb_latexmk",
    "statement_lengths_slide.fls",
    "statement_lengths_slide.log",
    "statement_lengths_slide.nav",
    "statement_lengths_slide.out",
    "statement_lengths_slide.pdf",
    "statement_lengths_slide.snm",
    "statement_lengths_slide.toc",
    "statements.csv",
    "file_summary.csv",
    "language_summary.csv",
    "parse_failures.jsonl",  # obsolete root-level aggregate
)


def write_manifest(manifest_path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically rewrite the manifest after normalizing local cache paths."""
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.parent.mkdir(parents=True, exist_ok=True)
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(manifest_tmp, manifest_path)


def normalize_manifest_and_checkpoints(cache_dir: Path, min_tokens: int) -> None:
    """Normalize paths in place while retaining sources and successful parse results.

    Successful results from earlier cache schemas are reusable because the statement extraction
    is unchanged. Failure checkpoints are handled more conservatively: old versions could
    quarantine valid Java files because of the native-GC bug, so only failures written by this
    schema are retained. Older failures are removed and their already-downloaded sources are
    reparsed locally.
    """
    manifest_path = cache_dir / "sampled_files.jsonl"
    if not manifest_path.exists():
        return

    records: list[dict[str, Any]] = []
    for language_records in already_sampled(manifest_path).values():
        for original in language_records:
            rec = dict(original)
            rec["cache_path"] = str(normalized_record_cache_path(rec, cache_dir))
            records.append(rec)
    write_manifest(manifest_path, records)

    reused_results = 0
    parse_root = cache_dir / "parse_cache"
    if parse_root.exists():
        for old_path in list(parse_root.glob("*/*.json")):
            payload = read_json(old_path)
            if (
                    not payload
                    or payload.get("status") != "ok"
                    or not isinstance(payload.get("rows"), list)
            ):
                continue
            original_rec = payload.get("record")
            if not isinstance(original_rec, dict):
                continue
            rec = dict(original_rec)
            rec["cache_path"] = str(normalized_record_cache_path(rec, cache_dir))
            new_id = record_cache_id(rec, min_tokens)
            payload["schema"] = PARSE_CACHE_SCHEMA
            payload["record_id"] = new_id
            payload["record"] = rec
            for row in payload["rows"]:
                if isinstance(row, dict):
                    row["file_cache"] = rec["cache_path"]
            new_path = (
                    parse_root
                    / str(rec.get("language", old_path.parent.name))
                    / f"{new_id}.json"
            )
            if new_path != old_path and new_path.exists():
                existing = read_json(new_path)
                if (
                        existing
                        and existing.get("status") == "ok"
                        and existing.get("schema") == PARSE_CACHE_SCHEMA
                ):
                    old_path.unlink(missing_ok=True)
                    reused_results += 1
                    continue
            atomic_write_json(new_path, payload)
            if old_path != new_path:
                old_path.unlink(missing_ok=True)
            reused_results += 1

    discarded_failures = 0
    failure_root = cache_dir / "parse_failures"
    if failure_root.exists():
        for old_path in list(failure_root.glob("*/*.json")):
            payload = read_json(old_path)
            if not payload or payload.get("schema") != PARSE_CACHE_SCHEMA:
                old_path.unlink(missing_ok=True)
                discarded_failures += 1
                continue
            original_rec = payload.get("record")
            if not isinstance(original_rec, dict):
                old_path.unlink(missing_ok=True)
                discarded_failures += 1
                continue
            rec = dict(original_rec)
            rec["cache_path"] = str(normalized_record_cache_path(rec, cache_dir))
            new_id = record_cache_id(rec, min_tokens)
            payload["record_id"] = new_id
            payload["record"] = rec
            new_path = (
                    failure_root
                    / str(rec.get("language", old_path.parent.name))
                    / f"{new_id}.json"
            )
            atomic_write_json(new_path, payload)
            if old_path != new_path:
                old_path.unlink(missing_ok=True)

    # These are transient crash-recovery cursors, not measurements.
    current_root = cache_dir / "parse_current"
    if current_root.exists():
        for marker in current_root.glob("*.json"):
            marker.unlink(missing_ok=True)

    if reused_results:
        print(
            f"Reused {reused_results} successful parse checkpoints in place.",
            file=sys.stderr,
            flush=True,
        )
    if discarded_failures:
        print(
            f"Discarded {discarded_failures} stale failure checkpoints; "
            "their local source files will be retried.",
            file=sys.stderr,
            flush=True,
        )


def prepare_output_layout(out_dir: Path, cache_dir: Path, min_tokens: int) -> None:
    """Keep cache state in place and remove only obsolete presentation by-products."""
    out_dir = out_dir.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in GENERATED_FILES_TO_REMOVE:
        path = out_dir / name
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.exists():
                # Refuse to recursively remove an unexpected directory under a generated name.
                print(
                    f"Warning: not removing unexpected directory {path}",
                    file=sys.stderr,
                    flush=True,
                )
        except OSError as exc:
            print(f"Warning: could not remove old artifact {path}: {exc}", file=sys.stderr)

    normalize_manifest_and_checkpoints(cache_dir, min_tokens)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.parser_batch_size < 1:
        raise SystemExit("--parser-batch-size must be at least 1")

    # Internal workers are launched with an explicit --work-dir and never touch presentation files.
    if args._parse_worker_language:
        if args.work_dir is None:
            raise SystemExit("Internal parser worker requires --work-dir")
        args.work_dir = args.work_dir.expanduser().resolve()
        args.work_dir.mkdir(parents=True, exist_ok=True)
        return parse_worker_main(args)

    args.out_dir = args.out_dir.expanduser().resolve()
    args.work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else args.out_dir
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_layout(args.out_dir, args.work_dir, args.min_tokens)

    langs = [part.strip().lower() for part in args.languages.split(",") if part.strip()]
    unknown = [language for language in langs if language not in LANGS]
    if unknown:
        raise SystemExit(
            f"Unknown language key(s): {unknown}; available: {', '.join(LANGS)}"
        )
    specs = [LANGS[language] for language in langs]

    # Source files on disk are authoritative. Recover manifest records from checkpoints and,
    # when necessary, synthesize local-only records before deciding whether GitHub is needed.
    reconcile_manifest_with_local_sources(
        args.work_dir,
        specs,
        args.min_bytes,
        args.max_bytes,
    )

    versions = package_versions()
    print(f"Statement-length study script: {SCRIPT_VERSION}", file=sys.stderr, flush=True)
    print(
        "Parser stack: "
        f"Python {versions['python']}, tree-sitter {versions['tree_sitter']}, "
        f"tree-sitter-language-pack {versions['tree_sitter_language_pack']} "
        f"(cyclic GC off in workers; batch size {args.parser_batch_size})",
        file=sys.stderr,
        flush=True,
    )
    print(f"Cache directory: {args.work_dir}", file=sys.stderr, flush=True)

    target_counts = {spec.key: args.min_files for spec in specs}
    records = sample_files(args, specs, target_counts)
    if not records:
        if args.offline:
            raise SystemExit(
                f"No locally cached source files were found under {args.work_dir}."
            )
        raise SystemExit(
            "No sampled files. Set GITHUB_TOKEN, or use an existing cache containing "
            "sampled_files.jsonl and src/."
        )

    all_rows: list[dict[str, Any]] = []
    usable_files: dict[str, int] = {}
    for replacement_round in range(args.max_replacement_rounds + 1):
        all_rows, usable_files = analyze_records(args, specs, records)
        deficits = {
            spec.key: max(0, args.min_files - usable_files.get(spec.key, 0))
            for spec in specs
            if usable_files.get(spec.key, 0) < args.min_files
        }
        if not deficits:
            break

        details = ", ".join(
            f"{LANGS[key].label}: {usable_files.get(key, 0)}/{args.min_files}"
            for key in deficits
        )
        if args.offline:
            raise SystemExit(
                "Cached parsing did not leave enough usable files ("
                + details
                + "). Remove --offline to permit sampling genuinely new replacements."
            )
        if replacement_round >= args.max_replacement_rounds:
            raise SystemExit(
                f"Could not reach {args.min_files} usable files per language after "
                f"{args.max_replacement_rounds} replacement rounds ({details})."
            )

        print(
            f"Sampling replacements for unusable/crashing files ({details})",
            file=sys.stderr,
            flush=True,
        )
        existing = locally_cached_samples(
            args.work_dir / "sampled_files.jsonl",
            args.work_dir,
            )
        target_counts = {
            spec.key: len(existing.get(spec.key, [])) + deficits.get(spec.key, 0)
            for spec in specs
        }
        replacement_args = argparse.Namespace(**vars(args))
        replacement_args.resume = True
        replacement_args.seed = args.seed + (replacement_round + 1) * 1_000_003
        records = sample_files(replacement_args, specs, target_counts)

    print("Aggregating statement measurements", file=sys.stderr, flush=True)
    write_failure_ledger(args, specs, records)
    _, lang_rows = summarize(all_rows, specs)
    if not lang_rows:
        raise SystemExit(
            "No statement rows found; check parser support and statement type sets."
        )

    print("Writing one self-contained Beamer source file", file=sys.stderr, flush=True)
    tex_path = write_beamer(lang_rows, args.out_dir)
    if args.compile_tex:
        print(
            "--compile-tex is ignored: this version intentionally emits no PDF or TeX auxiliaries.",
            file=sys.stderr,
            flush=True,
        )

    print(f"Wrote: {tex_path}")
    if args.work_dir == args.out_dir:
        print(
            "Retained cached sources, manifest, and parser checkpoints in place.",
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    raise SystemExit(main(sys.argv[1:]))
