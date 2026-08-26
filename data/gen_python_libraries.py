#!/usr/bin/env python3
"""Precompute ty-derived CFG fragments for commonly imported Python modules.

The script first scans Project CodeNet without extracting it and counts the
number of Python submissions that import each top-level module.  It then asks
``ty server`` for the public semantic surface of selected modules and writes
deterministic, evaluator-native fragments to ``data/lib/*.cfg``.

A fragment intentionally has no ``START`` production and no production for a
literal module spelling.  For example, ``numpy.cfg`` contains member rules
rooted at ``E:<module 'numpy'>``.  At evaluation time the contextual grammar
supplies the only binding rule, such as ``E:<module 'numpy'> -> np`` for
``import numpy as np``.  Consequently the same artifact works for aliases and
cannot introduce an unbound ``numpy`` token.

The file format is line-oriented.  The part following an ``E:`` or ``A:``
nonterminal prefix is percent encoded; terminals are ordinary lexical tokens::

    # api2cfg-python-library-cfg: 2
    # module: numpy
    # symbol-N00000: E:%3Cmodule%20%27numpy%27%3E
    N00001 -> N00000 . array

The short ``Nxxxxx`` aliases keep repeated, often very long ty display types
from making large-library artifacts hundreds of megabytes.  Their comment
table is reversible, while the production rows remain ordinary ``.cfg``
syntax.

Completion responses marked incomplete by ty are retained but explicitly
marked ``completion-complete: false``.  Such artifacts are useful caches, but
must not be described as complete semantic models.

Each fragment also materializes the complete ``A:expected -> E:actual``
assignability relation over its own finite type domains.  Versioned domain
metadata lets the evaluator reuse those edges while continuing to check every
contextual and cross-fragment pair online.

Member traversal follows the root module and exported module namespaces up to
``member-depth``; it does not recursively crawl arbitrary exported classes or
values.  The generator also deliberately does not invent arguments in order
to call APIs and inspect their returned values: doing so would make one guessed
overload witness stand in for the whole return surface.  Artifacts record these
bounds as ``receiver-policy: module-namespaces`` and
``callable-return-members: false``; contextual receiver completion remains the
evaluator's fallback for nonmodule values.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import keyword
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import evaluate_codenet_python as evaluator


CFG_SCHEMA_VERSION = 2
DEFAULT_ARCHIVE = evaluator.DEFAULT_ARCHIVE
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().with_name("lib")
LIBRARY_ALIAS = "__api2cfg_library"
FRAGMENT_START = "__LIBRARY_FRAGMENT__"
MODULE_NAME = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
).fullmatch
AUTOMATIC_MODULE_EXCLUSIONS = frozenset({"__future__"})


class GenerationError(RuntimeError):
    """A requested module could not be represented by a cache fragment."""


@dataclass
class ImportScan:
    """Document frequencies collected from CodeNet Python submissions."""

    frequencies: Counter[str] = field(default_factory=Counter)
    python_files: int = 0
    decoded_files: int = 0
    parsed_files: int = 0
    decode_failures: int = 0
    parse_failures: int = 0


@dataclass
class FragmentStats:
    receiver_types: int = 0
    member_completions: int = 0
    callables: int = 0
    signatures: int = 0
    incomplete_completion_queries: int = 0
    completion_queries_at_cap: int = 0
    raw_member_completion_items: int = 0
    max_raw_member_completion_items: int = 0
    receiver_limit_reached: bool = False
    local_assignability_actual_types: frozenset[str] = frozenset()
    local_assignability_expected_types: frozenset[str] = frozenset()
    local_assignability_links: int = 0
    local_assignability_complete: bool = False


@dataclass(frozen=True)
class GeneratorOptions:
    archive: Path
    output_directory: Path
    modules: tuple[str, ...]
    top: int
    min_files: int
    scan_files: int | None
    include_observed_stdlib: bool
    include_all_stdlib: bool
    scan_only: bool
    ty: str
    max_call_arity: int
    max_layouts_per_signature: int
    member_depth: int
    max_receiver_types: int
    require_complete: bool
    json_lines: bool


def imported_top_level_modules(tree: ast.AST) -> frozenset[str]:
    """Return absolute top-level imports, counted at most once per file."""

    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
        ):
            result.add(node.module.split(".", 1)[0])
    return frozenset(result)


def scan_archive_imports(
    archive_path: Path,
    *,
    max_python_files: int | None = None,
) -> ImportScan:
    """Stream a CodeNet archive and count importing files per root module."""

    if not archive_path.is_file():
        raise GenerationError(f"archive not found: {archive_path}")
    result = ImportScan()
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if (
                max_python_files is not None
                and result.python_files >= max_python_files
            ):
                break
            if (
                not member.isfile()
                or evaluator.PYTHON_SUBMISSION(member.name) is None
            ):
                continue
            result.python_files += 1
            extracted = archive.extractfile(member)
            if extracted is None:
                result.decode_failures += 1
                continue
            with extracted:
                source = evaluator.decode_source(extracted.read())
            if source is None:
                result.decode_failures += 1
                continue
            result.decoded_files += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(
                        source,
                        filename=member.name,
                        type_comments=True,
                    )
            except (SyntaxError, ValueError):
                result.parse_failures += 1
                continue
            result.parsed_files += 1
            result.frequencies.update(imported_top_level_modules(tree))
    return result


def validate_module_name(module: str) -> str:
    if MODULE_NAME(module) is None or any(
        keyword.iskeyword(part) for part in module.split(".")
    ):
        raise GenerationError(f"invalid absolute module name: {module!r}")
    return module


def selected_modules(options: GeneratorOptions, scan: ImportScan) -> list[str]:
    """Choose modules deterministically from explicit and scanned inputs."""

    explicit = {validate_module_name(module) for module in options.modules}
    ranked = [
        module
        for module, count in sorted(
            scan.frequencies.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count >= options.min_files
        and module not in AUTOMATIC_MODULE_EXCLUSIONS
    ]
    if options.top:
        ranked = ranked[: options.top]
    selected = explicit | set(ranked)
    observed_stdlib = (
        set(scan.frequencies)
        & set(sys.stdlib_module_names)
        - AUTOMATIC_MODULE_EXCLUSIONS
    )
    if options.include_observed_stdlib:
        selected.update(observed_stdlib)
    if options.include_all_stdlib:
        selected.update(set(sys.stdlib_module_names) - AUTOMATIC_MODULE_EXCLUSIONS)
    return sorted(selected, key=lambda module: (-scan.frequencies[module], module))


def synthetic_module_probe(
    client: evaluator.TyLspClient,
    workspace: Path,
    module: str,
) -> tuple[evaluator.SemanticProbe, str]:
    """Open a clean synthetic import and return its expression-hole probe."""

    source = f"import {module} as {LIBRARY_ALIAS}\n{LIBRARY_ALIAS}\n"
    tree = ast.parse(source)
    statement = tree.body[1]
    if not isinstance(statement, ast.Expr):
        raise AssertionError("synthetic module expression was not an Expr")
    hole = evaluator.hole_for_node(source, statement)
    if hole is None:
        raise AssertionError("synthetic module expression did not form a hole")
    ablated = hole.render("()")
    document_uri = evaluator.uri_for(
        workspace
        / f"library_{evaluator.stable_digest(module, length=20)}.py"
    )
    client.open(document_uri, ablated)
    diagnostics = evaluator.error_diagnostics(client.diagnostics())
    if diagnostics:
        messages = "; ".join(
            str(item.get("message", "diagnostic")) for item in diagnostics[:3]
        )
        raise GenerationError(f"ty cannot import {module}: {messages}")
    probe = evaluator.SemanticProbe(client, hole, ablated)
    module_type = probe.hover_expression(LIBRARY_ALIAS)
    expected = f"<module '{module}'>"
    if module_type != expected:
        raise GenerationError(
            f"ty resolved {module!r} as {module_type!r}; expected {expected!r}"
        )
    return probe, module_type


def _receiver_key(type_display: str, expression: str) -> str:
    normalized = evaluator.normalize_type(type_display)
    if normalized in {"Any", "Unknown"}:
        return f"{normalized}\0{expression}"
    return normalized


def _member_completions(
    probe: evaluator.SemanticProbe,
    expression: str,
) -> tuple[list[evaluator.Completion], bool, int]:
    """Query members while retaining the raw LSP size before simplification."""

    expression_inserted = f"{expression}."
    statement = probe._expression_statement(expression_inserted)
    probe.client.change(probe.hole.render(statement))
    inserted = probe._expression_cursor_text(expression_inserted)
    character = probe.hole.character_after(
        inserted,
        probe.client.position_encoding,
    )
    items, incomplete = probe.client.completion(
        probe.hole.line,
        character,
        trigger=".",
    )
    return evaluator.simplify_completions(items), incomplete, len(items)


def _add_public_members(
    builder: evaluator.GrammarBuilder,
    module_type: str,
    stats: FragmentStats,
    *,
    member_depth: int,
    max_receiver_types: int,
) -> list[tuple[str, str]]:
    """Add all returned public completions without contextual whitelisting."""

    receivers: deque[tuple[str, str, int]] = deque(
        ((module_type, LIBRARY_ALIAS, 0),)
    )
    queued = {_receiver_key(module_type, LIBRARY_ALIAS)}
    callables: list[tuple[str, str]] = []
    while receivers:
        receiver_type, expression, depth = receivers.popleft()
        stats.receiver_types += 1
        completions, incomplete, raw_items = _member_completions(
            builder.probe,
            expression,
        )
        stats.raw_member_completion_items += raw_items
        stats.max_raw_member_completion_items = max(
            stats.max_raw_member_completion_items,
            raw_items,
        )
        if incomplete:
            stats.incomplete_completion_queries += 1
        if raw_items >= 1000:
            stats.completion_queries_at_cap += 1
        receiver_nonterminal = builder.expression_nonterminal(receiver_type)
        for completion in completions:
            member_type = evaluator.normalize_type(completion.detail)
            member_expression = f"{expression}.{completion.label}"
            builder.add_expression(
                member_type,
                (
                    evaluator.Nonterminal(receiver_nonterminal),
                    evaluator.Terminal("."),
                    evaluator.Terminal(completion.label),
                ),
                representative=member_expression,
            )
            stats.member_completions += 1
            callable_value = evaluator.is_callable_type(
                member_type,
                completion.kind,
            )
            if callable_value:
                callables.append((member_type, member_expression))
            should_receive = member_type.startswith("<module '")
            if not should_receive or depth >= member_depth:
                continue
            key = _receiver_key(member_type, member_expression)
            if key in queued:
                continue
            if max_receiver_types and len(queued) >= max_receiver_types:
                stats.receiver_limit_reached = True
                continue
            queued.add(key)
            receivers.append((member_type, member_expression, depth + 1))
    return callables


def _add_call_productions(
    builder: evaluator.GrammarBuilder,
    callables: Iterable[tuple[str, str]],
    stats: FragmentStats,
    *,
    max_call_arity: int,
    max_layouts_per_signature: int,
) -> None:
    """Lower every discovered callable overload into evaluator-native rules."""

    unique_callables = sorted(set(callables))
    stats.callables = len(unique_callables)
    for callable_type, expression in unique_callables:
        signatures = builder.signatures_for(callable_type, expression)
        stats.signatures += len(signatures)
        callable_nonterminal = builder.expression_nonterminal(callable_type)
        for signature in signatures:
            layouts = evaluator.argument_layouts(
                signature,
                max_arity=max_call_arity,
                max_layouts=max_layouts_per_signature,
            )
            return_nonterminal = builder.expression_nonterminal(
                signature.return_type
            )
            for layout in layouts:
                rhs: list[evaluator.Symbol] = [
                    evaluator.Nonterminal(callable_nonterminal),
                    evaluator.Terminal("("),
                ]
                first = True
                for expected in layout.positional:
                    if not first:
                        rhs.append(evaluator.Terminal(","))
                    first = False
                    expected = evaluator.normalize_type(expected)
                    rhs.append(
                        evaluator.Nonterminal(
                            evaluator.argument_nonterminal(expected)
                        )
                    )
                for name, expected in layout.keywords:
                    if not first:
                        rhs.append(evaluator.Terminal(","))
                    first = False
                    expected = evaluator.normalize_type(expected)
                    rhs.extend(
                        (
                            evaluator.Terminal(name),
                            evaluator.Terminal("="),
                            evaluator.Nonterminal(
                                evaluator.argument_nonterminal(expected)
                            ),
                        )
                    )
                rhs.append(evaluator.Terminal(")"))
                builder.grammar.add(return_nonterminal, *rhs)


def _add_fragment_argument_links(
    grammar: evaluator.Grammar,
    stats: FragmentStats,
) -> None:
    """Materialize the complete assignability relation over local E/A types."""

    actual_types = frozenset(
        production.lhs[2:]
        for production in grammar.productions
        if production.lhs.startswith("E:")
    )
    nonterminals = {
        production.lhs for production in grammar.productions
    } | {
        symbol.value
        for production in grammar.productions
        for symbol in production.rhs
        if isinstance(symbol, evaluator.Nonterminal)
    }
    expected_types = frozenset(
        nonterminal[2:]
        for nonterminal in nonterminals
        if nonterminal.startswith("A:")
    )
    for expected in sorted(expected_types):
        argument = evaluator.argument_nonterminal(expected)
        for actual in sorted(actual_types):
            if evaluator.is_assignable(actual, expected):
                grammar.add(
                    argument,
                    evaluator.Nonterminal(evaluator.type_nonterminal(actual)),
                )
    stats.local_assignability_actual_types = actual_types
    stats.local_assignability_expected_types = expected_types
    stats.local_assignability_links = sum(
        1
        for production in grammar.productions
        if production.lhs.startswith("A:")
        and len(production.rhs) == 1
        and isinstance(production.rhs[0], evaluator.Nonterminal)
        and production.rhs[0].value.startswith("E:")
    )
    stats.local_assignability_complete = True


def build_module_fragment(
    client: evaluator.TyLspClient,
    workspace: Path,
    module: str,
    options: GeneratorOptions,
) -> tuple[evaluator.Grammar, FragmentStats, str]:
    """Construct a raw, context-free module fragment with no START rule."""

    probe, module_type = synthetic_module_probe(client, workspace, module)
    builder_options = evaluator.BuilderOptions(
        max_call_arity=options.max_call_arity,
        max_layouts_per_signature=options.max_layouts_per_signature,
        member_depth=options.member_depth,
        max_receiver_types=max(1, options.max_receiver_types or 1_000_000_000),
        max_module_members=1_000_000_000,
    )
    builder = evaluator.GrammarBuilder(
        probe,
        frozenset(),
        builder_options,
        {},
    )
    builder.grammar = evaluator.Grammar(start=FRAGMENT_START)
    builder.expression_nonterminal(module_type)
    stats = FragmentStats()
    callables = _add_public_members(
        builder,
        module_type,
        stats,
        member_depth=options.member_depth,
        max_receiver_types=options.max_receiver_types,
    )
    _add_call_productions(
        builder,
        callables,
        stats,
        max_call_arity=options.max_call_arity,
        max_layouts_per_signature=options.max_layouts_per_signature,
    )
    _add_fragment_argument_links(builder.grammar, stats)
    if any(production.lhs == "START" for production in builder.grammar.productions):
        raise AssertionError("library fragment unexpectedly contains START")
    if stats.receiver_limit_reached and options.require_complete:
        raise GenerationError(
            f"{module}: receiver limit {options.max_receiver_types} was reached"
        )
    if stats.incomplete_completion_queries and options.require_complete:
        raise GenerationError(
            f"{module}: ty marked {stats.incomplete_completion_queries} "
            "completion queries incomplete"
        )
    if stats.completion_queries_at_cap and options.require_complete:
        raise GenerationError(
            f"{module}: {stats.completion_queries_at_cap} raw completion "
            "queries reached the 1000-item hard-cap sentinel"
        )
    return builder.grammar, stats, module_type


def encode_nonterminal(value: str) -> str:
    """Render an evaluator E:/A: name as one whitespace-free CFG atom."""

    if not value.startswith(("E:", "A:")):
        raise GenerationError(f"unsupported fragment nonterminal: {value!r}")
    return f"{value[:2]}{urllib.parse.quote(value[2:], safe='')}"


def decode_nonterminal(value: str) -> str:
    if not value.startswith(("E:", "A:")):
        raise GenerationError(f"invalid encoded nonterminal: {value!r}")
    return f"{value[:2]}{urllib.parse.unquote(value[2:])}"


def _production_line(
    production: evaluator.Production,
    aliases: Mapping[str, str],
) -> str:
    lhs = aliases[production.lhs]
    rhs: list[str] = []
    for symbol in production.rhs:
        if isinstance(symbol, evaluator.Nonterminal):
            rhs.append(aliases[symbol.value])
        else:
            if not symbol.value or any(character.isspace() for character in symbol.value):
                raise GenerationError(
                    f"terminal is not a single CFG atom: {symbol.value!r}"
                )
            if (
                symbol.value.startswith(("E:", "A:", "#"))
                or symbol.value == "->"
                or re.fullmatch(r"N[0-9]+", symbol.value)
            ):
                raise GenerationError(
                    f"terminal collides with CFG syntax: {symbol.value!r}"
                )
            rhs.append(symbol.value)
    if not rhs:
        raise GenerationError("epsilon productions are not supported")
    return f"{lhs} -> {' '.join(rhs)}"


def fragment_nonterminals(grammar: evaluator.Grammar) -> frozenset[str]:
    result = {production.lhs for production in grammar.productions}
    result.update(
        symbol.value
        for production in grammar.productions
        for symbol in production.rhs
        if isinstance(symbol, evaluator.Nonterminal)
    )
    return frozenset(result)


def semantic_environment_python(ty: str) -> tuple[Path, str]:
    """Find the interpreter whose installed packages ty most likely resolves.

    A Python interpreter beside the resolved ty executable is the strongest
    available signal for pip/Conda installations.  Activated virtual/Conda
    environments come next, followed by the generator's own interpreter.  Ty
    does not currently expose its resolved environment through standard LSP.
    """

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
        candidates = (
            root / "bin" / "python",
            root / "bin" / "python3",
            root / "Scripts" / "python.exe",
        )
        for candidate in candidates:
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


def package_versions(module: str, interpreter: Path) -> list[str]:
    """Fingerprint packages in ty's activated environment without importing them."""

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
    # A broken environment fingerprint should not prevent stdlib generation.
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


def render_fragment(
    grammar: evaluator.Grammar,
    *,
    module: str,
    module_type: str,
    stats: FragmentStats,
    options: GeneratorOptions,
    scan: ImportScan,
    ty: str,
    environment_python: Path,
    environment_source: str,
) -> str:
    """Serialize a fragment and its reproducibility metadata."""

    nonterminals = fragment_nonterminals(grammar)
    aliases = {
        nonterminal: f"N{index:05d}"
        for index, nonterminal in enumerate(sorted(nonterminals))
    }
    production_lines = sorted(
        {
            _production_line(production, aliases)
            for production in grammar.productions
        }
    )
    terminals = {
        symbol.value
        for production in grammar.productions
        for symbol in production.rhs
        if isinstance(symbol, evaluator.Terminal)
    }
    completion_complete = (
        stats.incomplete_completion_queries == 0
        and stats.completion_queries_at_cap == 0
        and not stats.receiver_limit_reached
    )
    environment_version, environment_platform = semantic_python_details(
        environment_python
    )
    local_actual_symbols = sorted(
        aliases[evaluator.type_nonterminal(actual)]
        for actual in stats.local_assignability_actual_types
    )
    local_expected_symbols = sorted(
        aliases[evaluator.argument_nonterminal(expected)]
        for expected in stats.local_assignability_expected_types
    )
    metadata: list[tuple[str, str]] = [
        ("api2cfg-python-library-cfg", str(CFG_SCHEMA_VERSION)),
        ("module", module),
        ("module-type", module_type),
        ("ty", ty),
        ("python", environment_version),
        ("platform", environment_platform),
        ("python-executable", str(environment_python.resolve())),
        ("python-environment-source", environment_source),
        (
            "package",
            json.dumps(
                package_versions(module, environment_python),
                separators=(",", ":"),
            ),
        ),
        ("import-files", str(scan.frequencies[module])),
        ("scanned-python-files", str(scan.python_files)),
        ("max-call-arity", str(options.max_call_arity)),
        (
            "max-layouts-per-signature",
            str(options.max_layouts_per_signature),
        ),
        ("member-depth", str(options.member_depth)),
        ("receiver-policy", "module-namespaces"),
        ("callable-return-members", "false"),
        (
            "local-assignability-complete",
            str(stats.local_assignability_complete).lower(),
        ),
        (
            "local-assignability-version",
            str(evaluator.ASSIGNABILITY_RELATION_VERSION),
        ),
        (
            "local-assignability-actuals",
            json.dumps(local_actual_symbols, separators=(",", ":")),
        ),
        (
            "local-assignability-expecteds",
            json.dumps(local_expected_symbols, separators=(",", ":")),
        ),
        (
            "local-assignability-pairs",
            str(len(local_actual_symbols) * len(local_expected_symbols)),
        ),
        (
            "local-assignability-links",
            str(stats.local_assignability_links),
        ),
        ("receiver-types", str(stats.receiver_types)),
        ("member-completions", str(stats.member_completions)),
        ("callables", str(stats.callables)),
        ("signatures", str(stats.signatures)),
        (
            "incomplete-completion-queries",
            str(stats.incomplete_completion_queries),
        ),
        (
            "raw-member-completion-items",
            str(stats.raw_member_completion_items),
        ),
        (
            "max-raw-member-completion-items",
            str(stats.max_raw_member_completion_items),
        ),
        (
            "raw-completion-queries-at-cap",
            str(stats.completion_queries_at_cap),
        ),
        (
            "receiver-limit-reached",
            str(stats.receiver_limit_reached).lower(),
        ),
        ("completion-complete", str(completion_complete).lower()),
        ("productions", str(len(production_lines))),
        ("nonterminals", str(len(nonterminals))),
        ("terminals", str(len(terminals))),
        (
            "grammar-symbols",
            str(sum(1 + len(production.rhs) for production in grammar.productions)),
        ),
    ]
    lines = [f"# {key}: {value}" for key, value in metadata]
    lines.extend(
        f"# symbol-{alias}: {encode_nonterminal(nonterminal)}"
        for nonterminal, alias in sorted(aliases.items(), key=lambda item: item[1])
    )
    lines.extend(production_lines)
    return "\n".join(lines) + "\n"


def write_fragment(output_directory: Path, module: str, text: str) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    # Match the evaluator catalog's flat, injective module-to-file mapping.
    destination = output_directory / evaluator.library_cfg_filename(module)
    temporary = destination.with_suffix(".cfg.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)
    return destination


def emit(value: Mapping[str, object], *, json_lines: bool) -> None:
    if json_lines:
        print(json.dumps(value, sort_keys=True), flush=True)
        return
    event = value.get("event")
    if event == "scan":
        print(
            f"scanned {value['python_files']} Python files; "
            f"parsed={value['parsed_files']} modules={value['modules']}",
            flush=True,
        )
    elif event == "library":
        print(
            f"wrote {value['module']} |V|={value['nonterminals']} "
            f"|P|={value['productions']} |G|={value['grammar_symbols']} "
            f"assignability={value['local_assignability_links']}/"
            f"{value['local_assignability_pairs']} "
            f"complete={value['completion_complete']} "
            f"seconds={value['seconds']:.3f} -> {value['path']}",
            flush=True,
        )
    else:
        print(json.dumps(value, sort_keys=True), flush=True)


def generate(options: GeneratorOptions) -> int:
    needs_scan = (
        not options.modules
        or options.scan_files is not None
        or options.include_observed_stdlib
        or options.scan_only
    )
    if needs_scan:
        scan = scan_archive_imports(
            options.archive,
            max_python_files=options.scan_files,
        )
    else:
        scan = ImportScan()
    modules = selected_modules(options, scan)
    emit(
        {
            "event": "scan",
            "archive": str(options.archive),
            "python_files": scan.python_files,
            "decoded_files": scan.decoded_files,
            "parsed_files": scan.parsed_files,
            "decode_failures": scan.decode_failures,
            "parse_failures": scan.parse_failures,
            "modules": len(scan.frequencies),
            "selected": modules,
            "frequencies": {
                module: scan.frequencies[module] for module in modules
            },
        },
        json_lines=options.json_lines,
    )
    if options.scan_only:
        return 0
    if not modules:
        raise GenerationError("no modules selected")
    ty = evaluator.ty_version(options.ty)
    environment_python, environment_source = semantic_environment_python(
        options.ty
    )
    failures = 0
    with tempfile.TemporaryDirectory(prefix="api2cfg-python-libraries-") as directory:
        workspace = Path(directory)
        with evaluator.TyLspClient(options.ty, workspace) as client:
            for module in modules:
                started = time.perf_counter()
                try:
                    grammar, stats, module_type = build_module_fragment(
                        client,
                        workspace,
                        module,
                        options,
                    )
                    text = render_fragment(
                        grammar,
                        module=module,
                        module_type=module_type,
                        stats=stats,
                        options=options,
                        scan=scan,
                        ty=ty,
                        environment_python=environment_python,
                        environment_source=environment_source,
                    )
                    path = write_fragment(options.output_directory, module, text)
                except (GenerationError, evaluator.EvaluationError) as error:
                    failures += 1
                    print(f"warning: {module}: {error}", file=sys.stderr, flush=True)
                    continue
                emit(
                    {
                        "event": "library",
                        "module": module,
                        "path": str(path),
                        "import_files": scan.frequencies[module],
                        "productions": len(grammar.productions),
                        "nonterminals": len(fragment_nonterminals(grammar)),
                        "terminals": len(
                            {
                                symbol.value
                                for production in grammar.productions
                                for symbol in production.rhs
                                if isinstance(symbol, evaluator.Terminal)
                            }
                        ),
                        "grammar_symbols": grammar.symbol_count,
                        "local_assignability_pairs": (
                            len(stats.local_assignability_actual_types)
                            * len(stats.local_assignability_expected_types)
                        ),
                        "local_assignability_links": (
                            stats.local_assignability_links
                        ),
                        "completion_complete": (
                            stats.incomplete_completion_queries == 0
                            and stats.completion_queries_at_cap == 0
                            and not stats.receiver_limit_reached
                        ),
                        "seconds": time.perf_counter() - started,
                    },
                    json_lines=options.json_lines,
                )
    return 0 if failures == 0 else 2


def run_self_tests() -> None:
    source = """
import numpy as np
from collections import Counter
import os.path
from . import local
"""
    modules = imported_top_level_modules(ast.parse(source))
    assert modules == frozenset({"numpy", "collections", "os"})
    assert encode_nonterminal("E:<module 'numpy'>") == (
        "E:%3Cmodule%20%27numpy%27%3E"
    )
    assert decode_nonterminal(encode_nonterminal("A:list[int] | None")) == (
        "A:list[int] | None"
    )
    grammar = evaluator.Grammar(start=FRAGMENT_START)
    grammar.add(
        "E:<built-in function sqrt>",
        evaluator.Nonterminal("E:<module 'math'>"),
        evaluator.Terminal("."),
        evaluator.Terminal("sqrt"),
    )
    grammar.add(
        "E:float",
        evaluator.Nonterminal("E:<built-in function sqrt>"),
        evaluator.Terminal("("),
        evaluator.Nonterminal("A:SupportsFloat"),
        evaluator.Terminal(")"),
    )
    stats = FragmentStats(member_completions=1, callables=1, signatures=1)
    _add_fragment_argument_links(grammar, stats)
    assert evaluator.Production(
        "A:SupportsFloat",
        (evaluator.Nonterminal("E:float"),),
    ) in grammar.productions
    options = GeneratorOptions(
        archive=Path("unused"),
        output_directory=Path("unused"),
        modules=("math",),
        top=0,
        min_files=1,
        scan_files=None,
        include_observed_stdlib=False,
        include_all_stdlib=False,
        scan_only=False,
        ty="ty",
        max_call_arity=3,
        max_layouts_per_signature=64,
        member_depth=1,
        max_receiver_types=0,
        require_complete=False,
        json_lines=False,
    )
    scan = ImportScan(Counter({"math": 7}), python_files=10)
    text = render_fragment(
        grammar,
        module="math",
        module_type="<module 'math'>",
        stats=stats,
        options=options,
        scan=scan,
        ty="ty test",
        environment_python=Path(sys.executable),
        environment_source="current-interpreter",
    )
    assert text.startswith("# api2cfg-python-library-cfg: 2\n# module: math\n")
    assert "# receiver-policy: module-namespaces\n" in text
    assert "# callable-return-members: false\n" in text
    assert "# local-assignability-complete: true\n" in text
    assert "\nSTART -> " not in text
    assert "# symbol-N" in text
    parsed = evaluator.parse_library_cfg_text(text, Path("math.cfg"))
    assert parsed.grammar.productions == grammar.productions
    assert parsed.local_assignability_complete
    assert parsed.local_actual_types == frozenset(
        {"<built-in function sqrt>", "float"}
    )
    assert parsed.local_expected_types == frozenset({"SupportsFloat"})
    with tempfile.TemporaryDirectory() as directory:
        path = write_fragment(Path(directory), "xml.etree", text)
        assert path.name == "xml%2Eetree.cfg"
        assert path.read_text(encoding="utf-8") == text
    try:
        encode_nonterminal("START")
    except GenerationError:
        pass
    else:
        raise AssertionError("START should not be serializable in a fragment")
    defaults = parse_arguments([])
    assert defaults.max_call_arity == 12
    assert defaults.max_layouts_per_signature == 64
    assert defaults.member_depth == 2
    print("self-test passed")


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Project CodeNet gzip tar archive",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="generate this absolute module (repeatable); explicit-only mode skips scanning",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="generate at most this many frequent scanned modules; 0 means all",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=10,
        help="minimum number of importing submissions",
    )
    parser.add_argument(
        "--scan-files",
        type=int,
        help="stop the import scan after this many Python submissions",
    )
    parser.add_argument(
        "--include-observed-stdlib",
        action="store_true",
        help="include every observed stdlib root regardless of frequency/top limits",
    )
    parser.add_argument(
        "--include-all-stdlib",
        action="store_true",
        help="also generate every stdlib root available to this Python",
    )
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--ty", default="ty")
    parser.add_argument(
        "--max-call-arity",
        type=int,
        default=12,
        help="largest flattened call arity cached in each fragment",
    )
    parser.add_argument("--max-layouts-per-signature", type=int, default=64)
    parser.add_argument("--member-depth", type=int, default=2)
    parser.add_argument(
        "--max-receiver-types",
        type=int,
        default=0,
        help="exported module namespaces to inspect; 0 means unlimited",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail modules when ty marks completion incomplete or a receiver cap is reached",
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parsed = parser.parse_args(arguments)
    for name, minimum in (
        ("top", 0),
        ("min_files", 1),
        ("max_call_arity", 0),
        ("max_layouts_per_signature", 1),
        ("member_depth", 0),
        ("max_receiver_types", 0),
    ):
        if getattr(parsed, name) < minimum:
            parser.error(f"--{name.replace('_', '-')} must be at least {minimum}")
    if parsed.scan_files is not None and parsed.scan_files < 1:
        parser.error("--scan-files must be at least 1")
    for module in parsed.module:
        try:
            validate_module_name(module)
        except GenerationError as error:
            parser.error(str(error))
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    if args.self_test:
        run_self_tests()
        return 0
    options = GeneratorOptions(
        archive=args.archive,
        output_directory=args.output_dir,
        modules=tuple(args.module),
        top=args.top,
        min_files=args.min_files,
        scan_files=args.scan_files,
        include_observed_stdlib=args.include_observed_stdlib,
        include_all_stdlib=args.include_all_stdlib,
        scan_only=args.scan_only,
        ty=args.ty,
        max_call_arity=args.max_call_arity,
        max_layouts_per_signature=args.max_layouts_per_signature,
        member_depth=args.member_depth,
        max_receiver_types=args.max_receiver_types,
        require_complete=args.require_complete,
        json_lines=args.jsonl,
    )
    try:
        return generate(options)
    except (GenerationError, evaluator.EvaluationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
