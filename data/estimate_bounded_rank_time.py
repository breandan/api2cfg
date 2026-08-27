#!/usr/bin/env python3
"""Time exact shortlex ranking in bounded contextual Python CFGs.

For one independently ablated statement per selected source, this script builds
the contextual grammar, determinizes the distinct language at every length
from one through the ground-truth token length, and ranks the ground truth in
shortlex order.  Ambiguous derivations are merged by the bounded DAFSA.
The first one million distinct shortlex words are also streamed once through
a Laplace-smoothed lexical 4-gram model and Tidyparse's Python WDFA.  They are
reranked by decreasing 4-gram probability and increasing WDFA cost.
Ground truths outside that finite prefix are reported as censored.

The selected APPS split or CodeNet archive is sampled once, then independent
sources are evaluated by a worker pool.  Results are consumed in deterministic
sample order (rather than worker completion order), written incrementally to
JSONL, and used to refresh rank and rank-time histograms every 100 successful
statements by default.
The online plots also include a paired shortlex/4-gram/WDFA comparison.

Without ``--manifest``, distinct sources are sampled uniformly by keeping the
smallest seeded hashes during a full streaming pass.  A JSONL evaluator result
can be supplied as an eligibility manifest for a much faster, but
manifest-population-specific, timing run.
"""

from __future__ import annotations

import argparse
import atexit
import ast
import contextlib
import gc
import hashlib
import heapq
import json
import keyword
import math
import multiprocessing
import os
import platform
import signal
import statistics
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO

import evaluate_python as evaluator
from python_wdfa import PythonWDFA, WDFAFormatError, WDFA_INF


@dataclass(frozen=True)
class TargetHint:
    line: int
    column: int
    candidate_index: int | None = None
    kind: str | None = None
    ground_truth: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ManifestSelection:
    """Dataset-compatible manifest members and filtering diagnostics."""

    members: tuple[tuple[str, TargetHint], ...]
    compatible_records: int
    compatible_members: int
    dataset_mismatch_records: int
    split_mismatch_records: int

    @property
    def selected_members(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class SourceSample:
    member: str
    data: bytes
    hint: TargetHint | None = None
    dataset: str = "codenet"
    split: str | None = None
    problem_id: int | str | None = None
    solution_index: int | None = None
    difficulty: str | None = None
    url: str | None = None


def source_record_metadata(sample: SourceSample) -> dict[str, object]:
    """Return stable dataset provenance shared by rank and failure records."""

    return {
        "dataset": sample.dataset,
        "split": sample.split,
        "problem_id": sample.problem_id,
        "solution_index": sample.solution_index,
        "difficulty": sample.difficulty,
        "url": sample.url,
        "source_sha256": hashlib.sha256(sample.data).hexdigest(),
    }


@dataclass(frozen=True)
class RankWorkerConfig:
    seed: int
    allow_ignores: bool
    ty: str
    ty_release: str
    library_dir: Path
    max_dfa_states: int
    max_grammar_productions: int
    rank_timeout: float
    sample_timeout: float
    fourgram_counts: Path
    wdfa_path: Path
    fourgram_candidate_limit: int
    builder_options: evaluator.BuilderOptions


@dataclass(frozen=True)
class RankOutcome:
    member: str
    record: dict[str, object] | None = None
    rank1_exact: int | None = None
    failure: str | None = None
    failure_message: str | None = None
    elapsed_seconds: float | None = None
    censored_rank_seconds: float | None = None


@dataclass
class RankWorkerState:
    temporary: tempfile.TemporaryDirectory[str]
    workspace: Path
    checker: evaluator.TyLspClient
    semantics: evaluator.TyLspClient
    checker_uri: str
    library_catalog: evaluator.LibraryCatalog
    fourgram_model: LexicalFourGramModel
    wdfa_model: PythonWDFA


_WORKER_CONFIG: RankWorkerConfig | None = None
_WORKER_STATE: RankWorkerState | None = None


class RankTimeout(Exception):
    pass


class SampleTimeout(Exception):
    pass


@dataclass
class LexicalFourGramModel:
    """Conditional add-one lexical 4-gram model from raw count rows."""

    counts: dict[tuple[str, str, str, str], int]
    context_totals: dict[tuple[str, str, str], int]
    vocabulary: frozenset[str]
    transition_cache: dict[tuple[str, str, str, str], float]

    @classmethod
    def from_path(cls, path: Path) -> LexicalFourGramModel:
        counts: dict[tuple[str, str, str, str], int] = {}
        symbols: set[str] = set()
        with path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.rstrip("\n")
                gram_text, separator, count_text = line.rpartition(" ::: ")
                gram = tuple(gram_text.split())
                if not separator or len(gram) != 4:
                    raise evaluator.EvaluationError(
                        f"invalid lexical 4-gram at {path}:{line_number}"
                    )
                try:
                    count = int(count_text)
                except ValueError as error:
                    raise evaluator.EvaluationError(
                        f"invalid lexical 4-gram count at "
                        f"{path}:{line_number}"
                    ) from error
                if count < 0:
                    raise evaluator.EvaluationError(
                        f"negative lexical 4-gram count at "
                        f"{path}:{line_number}"
                    )
                typed_gram = (gram[0], gram[1], gram[2], gram[3])
                counts[typed_gram] = counts.get(typed_gram, 0) + count
                symbols.update(typed_gram)
        if not counts:
            raise evaluator.EvaluationError(
                f"lexical 4-gram count file is empty: {path}"
            )
        context_totals: dict[tuple[str, str, str], int] = defaultdict(int)
        for gram, count in counts.items():
            context_totals[gram[:3]] += count
        # BOS is a context marker rather than a possible prediction.  UNK
        # reserves probability mass for canonical terminals absent from the
        # frequency-truncated count table.
        vocabulary = frozenset((symbols - {"BOS"}) | {"UNK"})
        return cls(counts, dict(context_totals), vocabulary, {})

    def lexicalize(self, token: str) -> str:
        """Map a canonical CFG terminal to the count file's token alphabet."""

        if token == evaluator.FRESH_TOKEN:
            lexical = "NAME"
        elif token in {"0", "0.0", "0j"}:
            lexical = "NUMBER"
        elif token in {'""', 'b""'}:
            lexical = "STRING"
        elif token.isidentifier() and not keyword.iskeyword(token):
            lexical = "NAME"
        else:
            lexical = token
        return lexical if lexical in self.vocabulary else "UNK"

    def transition_log_probability(
        self, context: tuple[str, str, str], token: str
    ) -> float:
        """Return log P(token | context) with conditional add-one smoothing."""

        gram = (context[0], context[1], context[2], token)
        cached = self.transition_cache.get(gram)
        if cached is not None:
            return cached
        value = math.log(self.counts.get(gram, 0) + 1) - math.log(
            self.context_totals.get(context, 0) + len(self.vocabulary)
        )
        self.transition_cache[gram] = value
        return value

    def advance(
        self,
        history: tuple[str, ...],
        token: str,
        log_probability: float,
    ) -> tuple[tuple[str, ...], float]:
        """Advance one already-lexicalized token through the 4-gram model."""

        if len(history) == 3:
            log_probability += self.transition_log_probability(
                (history[0], history[1], history[2]), token
            )
        return (*history, token)[-3:], log_probability

    def score_lexicalized(self, tokens: Sequence[str]) -> float:
        history: tuple[str, ...] = ("BOS", "NEWLINE")
        result = 0.0
        for token in tokens:
            history, result = self.advance(history, token, result)
        for token in ("NEWLINE", "EOS"):
            history, result = self.advance(history, token, result)
        return result

    def score(self, tokens: Sequence[str]) -> float:
        return self.score_lexicalized(
            tuple(self.lexicalize(token) for token in tokens)
        )


@dataclass(frozen=True)
class LexicalRerankResult:
    """Ground-truth position within a shortlex candidate prefix."""

    rank1: int | None
    candidate_count: int
    ground_truth_log_probability: float
    censored: bool
    strictly_better: int | None = None
    tied_before: int | None = None
    tied_total: int | None = None


@dataclass(frozen=True)
class WDFARerankResult:
    """Ground-truth position under WDFA cost in a shortlex prefix."""

    rank1: int | None
    candidate_count: int
    ground_truth_cost: int | None
    ground_truth_valid: bool
    censored: bool
    strictly_better: int | None = None
    tied_before: int | None = None
    tied_total: int | None = None
    valid_candidates: int | None = None


@dataclass(frozen=True)
class CombinedRerankResult:
    """Paired reranks obtained during one shortlex-prefix traversal."""

    fourgram: LexicalRerankResult
    wdfa: WDFARerankResult


class UnitAwareBoundedLanguage(evaluator.BoundedLanguage):
    """Exact bounded DAFSAs without materializing the grammar's unit closure."""

    def __init__(
        self,
        grammar: evaluator.Grammar,
        max_length: int,
        max_states: int,
    ) -> None:
        self.grammar = grammar
        self.compiled = evaluator.UnitAwareBinaryGrammar(grammar)
        self.max_length = max_length
        self.max_states = max_states
        terminals = sorted(grammar.terminals)
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
        self.roots: dict[tuple[int, int], int] = {}
        self._build_unit_aware()

    def _build_unit_aware(self) -> None:
        for length in range(1, self.max_length + 1):
            base: dict[int, int] = {}
            if length == 1:
                for component, terminals in enumerate(
                    self.compiled.terminal_rules
                ):
                    root = self._union(
                        self.singletons[token] for token in terminals
                    )
                    if root != self.EMPTY:
                        base[component] = root
            if length >= 2:
                for component, rules in enumerate(self.compiled.binary_rules):
                    products: list[tuple[int, int]] = []
                    for left, right in rules:
                        for split in range(1, length):
                            left_root = self.roots.get(
                                (left, split), self.EMPTY
                            )
                            right_root = self.roots.get(
                                (right, length - split), self.EMPTY
                            )
                            if left_root != self.EMPTY and right_root != self.EMPTY:
                                products.append((left_root, right_root))
                    root = self._union_products(products)
                    if root != self.EMPTY:
                        base[component] = root
            # Unit SCCs are already condensed into a DAG.  Children precede
            # parents in bottom_up, so this unions distinct child languages
            # exactly once without expanding every A ->* B production.
            for component in self.compiled.bottom_up:
                root = self._union(
                    (
                        base.get(component, self.EMPTY),
                        *(
                            self.roots.get((child, length), self.EMPTY)
                            for child in self.compiled.unit_children[component]
                        ),
                    )
                )
                if root != self.EMPTY:
                    self.roots[(component, length)] = root

    def root(self, length: int) -> int:
        return self.roots.get((self.compiled.start, length), self.EMPTY)


def sample_hash(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{value}".encode()).digest()
    return int.from_bytes(digest, "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_members(
    path: Path,
    seed: int,
    pool_files: int,
    dataset: str = "codenet",
    split: str = "test",
) -> ManifestSelection:
    """Select only manifest items belonging to the requested population."""

    rows: dict[str, list[TargetHint]] = defaultdict(list)
    compatible_records = 0
    dataset_mismatch_records = 0
    split_mismatch_records = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "statement" and not record.get("recognized"):
                continue
            if event not in {"statement", "rank"}:
                continue
            member = record.get("member")
            if not isinstance(member, str):
                continue
            member_dataset: str | None = None
            member_split: str | None = None
            if member.startswith("Project_CodeNet/"):
                member_dataset = "codenet"
            elif member.startswith("APPS/"):
                member_dataset = "apps"
                parts = member.split("/", 2)
                if len(parts) >= 2:
                    member_split = parts[1]
            declared_dataset = record.get("dataset")
            if declared_dataset is None:
                record_dataset = member_dataset
            elif isinstance(declared_dataset, str):
                record_dataset = declared_dataset
            else:
                record_dataset = None
            if (
                record_dataset != dataset
                or (
                    member_dataset is not None
                    and record_dataset != member_dataset
                )
            ):
                dataset_mismatch_records += 1
                continue
            if dataset == "apps":
                declared_split = record.get("split")
                if declared_split is None:
                    record_split = member_split
                elif isinstance(declared_split, str):
                    record_split = declared_split
                else:
                    record_split = None
                if record_split != split or (
                    member_split is not None and record_split != member_split
                ):
                    split_mismatch_records += 1
                    continue
            line_number = record.get("line")
            column = record.get("column")
            candidate_index = record.get("candidate_index")
            kind = record.get("kind")
            ground_truth = record.get("ground_truth")
            if (
                not isinstance(line_number, int)
                or not isinstance(column, int)
                or (
                    candidate_index is not None
                    and not isinstance(candidate_index, int)
                )
                or (kind is not None and not isinstance(kind, str))
                or (
                    ground_truth is not None
                    and not (
                        isinstance(ground_truth, list)
                        and all(isinstance(token, str) for token in ground_truth)
                    )
                )
            ):
                continue
            compatible_records += 1
            rows[member].append(
                TargetHint(
                    line_number,
                    column,
                    candidate_index,
                    kind,
                    (
                        tuple(ground_truth)
                        if isinstance(ground_truth, list)
                        else None
                    ),
                )
            )
    selected: list[tuple[str, TargetHint]] = []
    for member in sorted(rows, key=lambda item: sample_hash(seed, item)):
        hints = rows[member]
        hint = min(
            hints,
            key=lambda item: sample_hash(
                seed,
                f"{member}\0{item.line}\0{item.column}\0"
                f"{item.candidate_index}\0{item.kind}",
            ),
        )
        selected.append((member, hint))
        if len(selected) >= pool_files:
            break
    return ManifestSelection(
        members=tuple(selected),
        compatible_records=compatible_records,
        compatible_members=len(rows),
        dataset_mismatch_records=dataset_mismatch_records,
        split_mismatch_records=split_mismatch_records,
    )


def source_sample(
    dataset_source: evaluator.DatasetSource,
    hint: TargetHint | None = None,
) -> SourceSample:
    """Copy the evaluator's streamed source into the picklable worker shape."""

    return SourceSample(
        member=dataset_source.member,
        data=dataset_source.data,
        hint=hint,
        dataset=dataset_source.dataset,
        split=dataset_source.split,
        problem_id=dataset_source.problem_id,
        solution_index=dataset_source.solution_index,
        difficulty=getattr(dataset_source, "difficulty", None),
        url=getattr(dataset_source, "url", None),
    )


def extract_manifest_samples(
    dataset: str,
    source_path: Path,
    split: str,
    selected: Sequence[tuple[str, TargetHint]],
) -> list[SourceSample]:
    """Resolve stable manifest members through the shared dataset streamer."""

    hints = dict(selected)
    remaining = set(hints)
    samples: dict[str, SourceSample] = {}
    for dataset_source in evaluator.iter_dataset_sources(
        dataset, source_path, split
    ):
        member = dataset_source.member
        if member not in remaining:
            continue
        samples[member] = source_sample(dataset_source, hints[member])
        remaining.remove(member)
        if not remaining:
            break
    if remaining:
        examples = ", ".join(sorted(remaining)[:3])
        suffix = "" if len(remaining) <= 3 else ", ..."
        raise evaluator.EvaluationError(
            f"manifest selected {len(selected):,} {dataset} members, but "
            f"{len(remaining):,} were not found in {source_path}: "
            f"{examples}{suffix}"
        )
    return [samples[name] for name, _hint in selected if name in samples]


def minhash_dataset_samples(
    dataset: str,
    source_path: Path,
    split: str,
    seed: int,
    pool_files: int,
    scan_limit: int | None,
) -> tuple[list[SourceSample], int]:
    """Take a deterministic bottom-k sample over stable source identities."""

    # Python's heap is a min-heap; negative hashes keep the largest retained
    # hash at the root, allowing an exact fixed-size bottom-k sample.
    retained: list[tuple[int, str, SourceSample]] = []
    seen_members: set[str] = set()
    scanned = 0
    for dataset_source in evaluator.iter_dataset_sources(
        dataset, source_path, split
    ):
        if dataset_source.member in seen_members:
            continue
        seen_members.add(dataset_source.member)
        scanned += 1
        score = sample_hash(seed, dataset_source.member)
        competitive = len(retained) < pool_files or score < -retained[0][0]
        if competitive:
            sample = source_sample(dataset_source)
            entry = (-score, sample.member, sample)
            if len(retained) < pool_files:
                heapq.heappush(retained, entry)
            else:
                heapq.heapreplace(retained, entry)
        if scanned % 100_000 == 0:
            print(
                f"scanned {scanned:,} Python sources; retained "
                f"{len(retained):,}",
                file=sys.stderr,
                flush=True,
            )
        if scan_limit is not None and scanned >= scan_limit:
            break
    ordered = sorted(retained, key=lambda entry: (-entry[0], entry[1]))
    return [sample for _score, _name, sample in ordered], scanned


def target_order_key(
    seed: int, member: str, target: evaluator.Target
) -> int:
    node = target.node
    return sample_hash(
        seed,
        f"{member}\0{node.lineno}\0{node.col_offset}\0"
        f"{node.end_lineno}\0{node.end_col_offset}\0{target.kind}",
    )


def evaluator_target_order_key(
    target: evaluator.Target,
) -> tuple[int, int, int, int, str]:
    """Mirror the evaluator's raw-candidate indexing order exactly."""

    node = target.node
    return (
        node.lineno,
        node.col_offset,
        node.end_lineno or node.lineno,
        node.end_col_offset or node.col_offset,
        target.kind,
    )


def manifest_raw_targets(
    targets: Sequence[evaluator.Target], hint: TargetHint
) -> list[evaluator.Target]:
    """Select a manifest item before preparation changes assignment kinds."""

    return [
        target
        for index, target in enumerate(
            sorted(targets, key=evaluator_target_order_key), start=1
        )
        if target.node.lineno == hint.line
        and target.node.col_offset == hint.column
        and (hint.candidate_index is None or index == hint.candidate_index)
    ]


def prepared_target_matches_hint(
    prepared: evaluator.PreparedTarget, hint: TargetHint
) -> bool:
    """Validate fields emitted only after the evaluator prepares a target."""

    return (
        (hint.kind is None or prepared.target.kind == hint.kind)
        and (
            hint.ground_truth is None
            or evaluator.canonical_tokens(prepared.target)
            == hint.ground_truth
        )
    )


def _assert_manifest_target_matching_contract() -> None:
    """Cover post-preparation assignment kinds and nested RHS tokenization."""

    source = (
        "def f() -> None:\n"
        '    result = abs(len("x"))\n'
        "    print(result)\n"
        '    unused = abs(len("y"))\n'
    )
    raw_targets = sorted(
        evaluator.candidate_targets(source, ast.parse(source)),
        key=evaluator_target_order_key,
    )
    assert len(raw_targets) == 3
    output_raw, expression_raw, fresh_raw = raw_targets
    assert output_raw.kind == fresh_raw.kind == "assignment"
    assert expression_raw.kind == "expression"

    output_target = replace(output_raw, kind="output-assignment")
    output_truth = evaluator.canonical_tokens(output_target)
    assert output_truth == (
        "result",
        "=",
        "abs",
        "(",
        "len",
        "(",
        '""',
        ")",
        ")",
    )
    output_hint = TargetHint(
        output_raw.node.lineno,
        output_raw.node.col_offset,
        1,
        "output-assignment",
        output_truth,
    )
    assert manifest_raw_targets(raw_targets, output_hint) == [output_raw]
    output_prepared = evaluator.PreparedTarget(
        output_target,
        "",
        "",
        required_assignment="result",
    )
    assert prepared_target_matches_hint(output_prepared, output_hint)

    fresh_target = replace(
        fresh_raw,
        kind="fresh-assignment",
        fresh_name=fresh_raw.assigned_name,
    )
    fresh_truth = evaluator.canonical_tokens(fresh_target)
    assert fresh_truth[0] == evaluator.FRESH_TOKEN
    fresh_hint = TargetHint(
        fresh_raw.node.lineno,
        fresh_raw.node.col_offset,
        3,
        "fresh-assignment",
        fresh_truth,
    )
    assert manifest_raw_targets(raw_targets, fresh_hint) == [fresh_raw]
    fresh_prepared = evaluator.PreparedTarget(fresh_target, "", "")
    assert prepared_target_matches_hint(fresh_prepared, fresh_hint)
    assert not prepared_target_matches_hint(
        fresh_prepared, replace(fresh_hint, kind="output-assignment")
    )
    assert not prepared_target_matches_hint(
        fresh_prepared, replace(fresh_hint, ground_truth=("wrong",))
    )


if __debug__:
    _assert_manifest_target_matching_contract()


def prepare_one_target(
    checker: evaluator.TyLspClient,
    checker_uri: str,
    sample: SourceSample,
    seed: int,
    allow_ignores: bool,
) -> tuple[str, evaluator.PreparedTarget] | None:
    source = evaluator.decode_source(sample.data)
    if source is None or (not allow_ignores and evaluator.has_suppression(source)):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=sample.member, type_comments=True)
    except (SyntaxError, ValueError):
        return None
    if not evaluator.check_source(checker, checker_uri, source):
        return None
    targets = sorted(
        evaluator.candidate_targets(source, tree),
        key=evaluator_target_order_key,
    )
    if sample.hint is not None:
        preferred = manifest_raw_targets(targets, sample.hint)
    else:
        preferred = []
    # A manifest identifies a particular benchmark item.  If that target is
    # stale or no longer eligible, reject the item instead of silently timing
    # a different statement from the same source.
    remaining = (
        []
        if sample.hint is not None
        else sorted(
            targets,
            key=lambda target: target_order_key(seed, sample.member, target),
        )
    )
    for target in [*preferred, *remaining]:
        prepared, _rejection = evaluator.prepare_target(
            checker, checker_uri, source, target
        )
        if prepared is not None and (
            sample.hint is None
            or prepared_target_matches_hint(prepared, sample.hint)
        ):
            return source, prepared
    return None


def contextual_grammar(
    semantics: evaluator.TyLspClient,
    workspace: Path,
    member: str,
    prepared: evaluator.PreparedTarget,
    base_builder: evaluator.BuilderOptions,
    library_catalog: evaluator.LibraryCatalog,
) -> tuple[evaluator.Grammar, tuple[str, ...], float]:
    started = time.perf_counter()
    selected = prepared.target
    truth = evaluator.canonical_tokens(selected)
    token_length = len(truth)
    identity = (
        f"{member}\0{selected.node.lineno}\0{selected.node.col_offset}\0rank"
    )
    semantic_uri = evaluator.uri_for(
        workspace / f"rank_{evaluator.stable_digest(identity, 16)}.py"
    )
    semantics.open(semantic_uri, prepared.semantic_source)
    if evaluator.error_diagnostics(semantics.diagnostics()):
        raise evaluator.EvaluationError("semantic scaffold contains ty errors")
    probe = evaluator.SemanticProbe(
        semantics,
        selected.hole,
        prepared.semantic_source,
        required_assignment=prepared.required_assignment,
        expression_prefix=prepared.expression_prefix,
        expression_suffix=prepared.expression_suffix,
        excluded_names=prepared.excluded_names,
    )
    options = replace(
        base_builder,
        max_call_arity=evaluator.maximum_call_arity(
            token_length,
            assignment=prepared.required_assignment is not None,
        ),
        max_tokens=token_length,
    )
    artifacts: list[evaluator.LibraryArtifact] = []
    for module in evaluator.visible_imported_library_modules(
        prepared.ablated, selected.hole.line
    ):
        lookup = library_catalog.lookup(module)
        if lookup.artifact is not None:
            artifacts.append(lookup.artifact)
    builder = evaluator.GrammarBuilder(
        probe,
        evaluator.source_identifiers(prepared.ablated),
        options,
        {},
        required_assignment=prepared.required_assignment,
        library_artifacts=artifacts,
        from_import_bindings=evaluator.visible_from_import_bindings(
            prepared.ablated, selected.hole.line
        ),
    )
    grammar, _stats = builder.build()
    return grammar, truth, time.perf_counter() - started


@contextlib.contextmanager
def rank_deadline(seconds: float) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def expired(_signum: int, _frame: object) -> None:
        raise RankTimeout(f"rank construction exceeded {seconds:g}s")

    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


@contextlib.contextmanager
def sample_deadline(seconds: float) -> Iterator[None]:
    """Bound an entire worker attempt, including ty semantic queries."""

    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def expired(_signum: int, _frame: object) -> None:
        raise SampleTimeout(f"sample evaluation exceeded {seconds:g}s")

    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def shortlex_rank(
    bounded: evaluator.BoundedLanguage,
    tokens: Sequence[str],
) -> tuple[int, int, list[int]]:
    length_sizes = [
        bounded.language_size(length) for length in range(1, len(tokens) + 1)
    ]
    rank = sum(length_sizes[:-1])
    state = bounded.root(len(tokens))
    if state == bounded.EMPTY:
        raise evaluator.EvaluationError("ground truth length slice is empty")
    for token in tokens:
        token_id = bounded.token_ids.get(token)
        if token_id is None or state == bounded.FINAL:
            raise evaluator.EvaluationError("ground truth is outside bounded DFA")
        next_state: int | None = None
        for edge, child in bounded.rows[state]:
            if edge < token_id:
                rank += bounded.counts[child]
            elif edge == token_id:
                next_state = child
                break
            else:
                break
        if next_state is None:
            raise evaluator.EvaluationError("ground truth is outside bounded DFA")
        state = next_state
    if state != bounded.FINAL:
        raise evaluator.EvaluationError("ground truth did not end at a DFA final state")
    return rank, sum(length_sizes), length_sizes


_WDFA_COMPOUND_TOKENS = {
    ("not", "in"): "not_in",
    ("is", "not"): "is_not",
}
_WDFA_PENDING_TOKENS = frozenset({"not", "is"})


def _advance_wdfa_canonical_token(
    model: Any,
    state: int,
    cost: int,
    pending: str | None,
    token: str,
    token_id: int | None,
) -> tuple[int, int, str | None]:
    """Advance the WDFA's tiny canonical-to-model-token transducer."""

    if pending is not None:
        compound = _WDFA_COMPOUND_TOKENS.get((pending, token))
        if compound is not None:
            state, cost, _present = model.advance_id(
                state, cost, model.token_id(compound)
            )
            return state, cost, None
        state, cost, _present = model.advance_id(
            state, cost, model.canonical_token_id(pending)
        )
    if token in _WDFA_PENDING_TOKENS:
        return state, cost, token
    state, cost, _present = model.advance_id(
        state, cost, token_id
    )
    return state, cost, None


def _finalize_wdfa_statement(
    model: Any,
    state: int,
    cost: int,
    pending: str | None,
) -> int:
    if pending is not None:
        state, cost, _present = model.advance_id(
            state, cost, model.canonical_token_id(pending)
        )
    final_cost, _final_state, _newline_present = model.finalize_statement(
        state, cost
    )
    return final_cost


def _score_wdfa_tokens(model: Any, tokens: Sequence[str]) -> int:
    """Score canonical statement tokens with tidyparse's WDFA convention."""

    state, cost = model.start_state, model.start_cost
    pending: str | None = None
    for token in tokens:
        state, cost, pending = _advance_wdfa_canonical_token(
            model,
            state,
            cost,
            pending,
            token,
            model.canonical_token_id(token),
        )
    return _finalize_wdfa_statement(model, state, cost, pending)


def _wdfa_cost_key(cost: int) -> tuple[bool, int]:
    """Put unsupported WDFA words after every finite-cost word."""

    valid = cost < WDFA_INF
    return not valid, cost if valid else 0


def _wdfa_cost_is_valid(cost: int) -> bool:
    return cost < WDFA_INF


def _rerank_shortlex_prefix(
    bounded: evaluator.BoundedLanguage,
    ground_truth: Sequence[str],
    ground_truth_rank0: int,
    model: LexicalFourGramModel,
    max_candidates: int,
    wdfa_model: Any | None,
) -> tuple[LexicalRerankResult, WDFARerankResult | None]:
    """Rerank a global shortlex prefix in one streamed traversal.

    The lexical model sorts by decreasing log probability and the WDFA sorts
    by increasing finalized cost.  Both retain shortlex order on ties.  A
    ground truth outside the finite prefix is censored for both models.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if ground_truth_rank0 < 0:
        raise ValueError("ground_truth_rank0 must be nonnegative")
    language_size = sum(
        bounded.language_size(length)
        for length in range(1, len(ground_truth) + 1)
    )
    if ground_truth_rank0 >= language_size:
        raise evaluator.EvaluationError(
            "ground-truth shortlex rank is outside the bounded language"
        )
    candidate_count = min(max_candidates, language_size)
    ground_truth_lexical = tuple(
        model.lexicalize(token) for token in ground_truth
    )
    ground_truth_score = model.score_lexicalized(ground_truth_lexical)
    ground_truth_wdfa_cost = (
        _score_wdfa_tokens(wdfa_model, ground_truth)
        if wdfa_model is not None
        else None
    )
    ground_truth_wdfa_key = (
        _wdfa_cost_key(ground_truth_wdfa_cost)
        if ground_truth_wdfa_cost is not None
        else None
    )
    if ground_truth_rank0 >= max_candidates:
        return (
            LexicalRerankResult(
                rank1=None,
                candidate_count=candidate_count,
                ground_truth_log_probability=ground_truth_score,
                censored=True,
            ),
            (
                WDFARerankResult(
                    rank1=None,
                    candidate_count=candidate_count,
                    ground_truth_cost=ground_truth_wdfa_cost,
                    ground_truth_valid=(
                        ground_truth_wdfa_cost is not None
                        and _wdfa_cost_is_valid(ground_truth_wdfa_cost)
                    ),
                    censored=True,
                )
                if wdfa_model is not None
                else None
            ),
        )

    lexical_terminals = tuple(
        model.lexicalize(token) for token in bounded.tokens
    )
    wdfa_token_ids = (
        tuple(
            wdfa_model.canonical_token_id(token)
            for token in bounded.tokens
        )
        if wdfa_model is not None
        else ()
    )
    enumerated = 0
    lexical_strictly_better = 0
    lexical_tied_before = 0
    lexical_tied_total = 0
    wdfa_strictly_better = 0
    wdfa_tied_before = 0
    wdfa_tied_total = 0
    wdfa_valid_candidates = 0
    saw_ground_truth = False

    def visit(
        state: int,
        history: tuple[str, ...],
        log_probability: float,
        wdfa_state: Any,
        wdfa_cost: Any,
        wdfa_pending: str | None,
    ) -> None:
        nonlocal enumerated, lexical_strictly_better
        nonlocal lexical_tied_before, lexical_tied_total
        nonlocal wdfa_strictly_better, wdfa_tied_before, wdfa_tied_total
        nonlocal wdfa_valid_candidates, saw_ground_truth
        if enumerated >= candidate_count:
            return
        if state == bounded.FINAL:
            final_history, candidate_score = model.advance(
                history, "NEWLINE", log_probability
            )
            _final_history, candidate_score = model.advance(
                final_history, "EOS", candidate_score
            )
            ordinal = enumerated
            if candidate_score > ground_truth_score:
                lexical_strictly_better += 1
            elif candidate_score == ground_truth_score:
                lexical_tied_total += 1
                if ordinal < ground_truth_rank0:
                    lexical_tied_before += 1
            candidate_wdfa_cost: int | None = None
            if wdfa_model is not None:
                candidate_wdfa_cost = _finalize_wdfa_statement(
                    wdfa_model,
                    wdfa_state,
                    wdfa_cost,
                    wdfa_pending,
                )
                if _wdfa_cost_is_valid(candidate_wdfa_cost):
                    wdfa_valid_candidates += 1
                candidate_wdfa_key = _wdfa_cost_key(candidate_wdfa_cost)
                if ground_truth_wdfa_key is None:
                    raise RuntimeError("WDFA ground-truth score is unavailable")
                if candidate_wdfa_key < ground_truth_wdfa_key:
                    wdfa_strictly_better += 1
                elif candidate_wdfa_key == ground_truth_wdfa_key:
                    wdfa_tied_total += 1
                    if ordinal < ground_truth_rank0:
                        wdfa_tied_before += 1
            if ordinal == ground_truth_rank0:
                saw_ground_truth = True
                if candidate_score != ground_truth_score:
                    raise evaluator.EvaluationError(
                        "streamed ground-truth lexical score disagrees"
                    )
                if (
                    wdfa_model is not None
                    and candidate_wdfa_cost != ground_truth_wdfa_cost
                ):
                    raise evaluator.EvaluationError(
                        "streamed ground-truth WDFA score disagrees"
                    )
            enumerated += 1
            return
        for token_id, child in bounded.rows[state]:
            token = lexical_terminals[token_id]
            next_history, next_probability = model.advance(
                history, token, log_probability
            )
            if wdfa_model is None:
                next_wdfa_state, next_wdfa_cost, next_wdfa_pending = (
                    wdfa_state,
                    wdfa_cost,
                    wdfa_pending,
                )
            else:
                (
                    next_wdfa_state,
                    next_wdfa_cost,
                    next_wdfa_pending,
                ) = _advance_wdfa_canonical_token(
                    wdfa_model,
                    wdfa_state,
                    wdfa_cost,
                    wdfa_pending,
                    bounded.tokens[token_id],
                    wdfa_token_ids[token_id],
                )
            visit(
                child,
                next_history,
                next_probability,
                next_wdfa_state,
                next_wdfa_cost,
                next_wdfa_pending,
            )
            if enumerated >= candidate_count:
                return

    for length in range(1, len(ground_truth) + 1):
        if enumerated >= candidate_count:
            break
        root = bounded.root(length)
        if root != bounded.EMPTY:
            if wdfa_model is None:
                initial_wdfa_state, initial_wdfa_cost = None, None
            else:
                initial_wdfa_state = wdfa_model.start_state
                initial_wdfa_cost = wdfa_model.start_cost
            visit(
                root,
                ("BOS", "NEWLINE"),
                0.0,
                initial_wdfa_state,
                initial_wdfa_cost,
                None,
            )
    if enumerated != candidate_count:
        raise evaluator.EvaluationError(
            "shortlex enumeration ended before the requested candidate prefix"
        )
    if not saw_ground_truth:
        raise evaluator.EvaluationError(
            "ground truth was not encountered at its shortlex rank"
        )
    lexical_result = LexicalRerankResult(
        rank1=lexical_strictly_better + lexical_tied_before + 1,
        candidate_count=candidate_count,
        ground_truth_log_probability=ground_truth_score,
        censored=False,
        strictly_better=lexical_strictly_better,
        tied_before=lexical_tied_before,
        tied_total=lexical_tied_total,
    )
    wdfa_result = (
        WDFARerankResult(
            rank1=wdfa_strictly_better + wdfa_tied_before + 1,
            candidate_count=candidate_count,
            ground_truth_cost=ground_truth_wdfa_cost,
            ground_truth_valid=(
                ground_truth_wdfa_cost is not None
                and _wdfa_cost_is_valid(ground_truth_wdfa_cost)
            ),
            censored=False,
            strictly_better=wdfa_strictly_better,
            tied_before=wdfa_tied_before,
            tied_total=wdfa_tied_total,
            valid_candidates=wdfa_valid_candidates,
        )
        if wdfa_model is not None
        else None
    )
    return lexical_result, wdfa_result


def rerank_shortlex_prefix_by_fourgrams(
    bounded: evaluator.BoundedLanguage,
    ground_truth: Sequence[str],
    ground_truth_rank0: int,
    model: LexicalFourGramModel,
    max_candidates: int = 1_000_000,
) -> LexicalRerankResult:
    """Backward-compatible lexical-only prefix reranker."""

    result, _wdfa = _rerank_shortlex_prefix(
        bounded,
        ground_truth,
        ground_truth_rank0,
        model,
        max_candidates,
        None,
    )
    return result


def rerank_shortlex_prefix_by_fourgrams_and_wdfa(
    bounded: evaluator.BoundedLanguage,
    ground_truth: Sequence[str],
    ground_truth_rank0: int,
    fourgram_model: LexicalFourGramModel,
    wdfa_model: Any,
    max_candidates: int = 1_000_000,
) -> CombinedRerankResult:
    """Compute lexical and WDFA ranks while enumerating the prefix once."""

    fourgram, wdfa = _rerank_shortlex_prefix(
        bounded,
        ground_truth,
        ground_truth_rank0,
        fourgram_model,
        max_candidates,
        wdfa_model,
    )
    if wdfa is None:  # Defensive: this public entry point requires a model.
        raise RuntimeError("combined reranking unexpectedly omitted WDFA output")
    return CombinedRerankResult(fourgram=fourgram, wdfa=wdfa)


def compact_integer(value: int) -> str:
    return evaluator.compact_cardinality(value)


def _create_rank_worker_state(config: RankWorkerConfig) -> RankWorkerState:
    temporary = tempfile.TemporaryDirectory(prefix="api2cfg-rank-worker-")
    workspace = Path(temporary.name)
    checker: evaluator.TyLspClient | None = None
    semantics: evaluator.TyLspClient | None = None
    try:
        checker = evaluator.TyLspClient(config.ty, workspace)
        semantics = evaluator.TyLspClient(config.ty, workspace)
        return RankWorkerState(
            temporary=temporary,
            workspace=workspace,
            checker=checker,
            semantics=semantics,
            checker_uri=evaluator.uri_for(workspace / "clean_check.py"),
            library_catalog=evaluator.LibraryCatalog(
                config.library_dir, config.ty, config.ty_release
            ),
            fourgram_model=LexicalFourGramModel.from_path(
                config.fourgram_counts
            ),
            wdfa_model=PythonWDFA.from_path(config.wdfa_path),
        )
    except BaseException:
        if semantics is not None:
            semantics.close()
        if checker is not None:
            checker.close()
        temporary.cleanup()
        raise


def _destroy_rank_worker(*, force: bool) -> None:
    global _WORKER_STATE
    state = _WORKER_STATE
    _WORKER_STATE = None
    if state is None:
        return
    if force:
        for client in (state.semantics, state.checker):
            if client.process.poll() is None:
                client.process.kill()
                client.process.wait()
    else:
        state.semantics.close()
        state.checker.close()
    state.temporary.cleanup()


def _close_rank_worker() -> None:
    _destroy_rank_worker(force=False)


def _restart_rank_worker() -> None:
    global _WORKER_STATE
    config = _WORKER_CONFIG
    if config is None:
        raise RuntimeError("rank worker configuration is unavailable")
    _destroy_rank_worker(force=True)
    _WORKER_STATE = _create_rank_worker_state(config)


def _initialize_rank_worker(config: RankWorkerConfig) -> None:
    global _WORKER_CONFIG, _WORKER_STATE
    _WORKER_CONFIG = config
    _WORKER_STATE = _create_rank_worker_state(config)
    atexit.register(_close_rank_worker)


def _rank_sample_impl(sample: SourceSample) -> RankOutcome:
    config = _WORKER_CONFIG
    state = _WORKER_STATE
    if config is None or state is None:
        raise RuntimeError("rank worker was not initialized")
    worker_started = time.perf_counter()
    prepared_pair = prepare_one_target(
        state.checker,
        state.checker_uri,
        sample,
        config.seed,
        config.allow_ignores,
    )
    if prepared_pair is None:
        return RankOutcome(
            member=sample.member,
            failure="not_clean_or_eligible",
            elapsed_seconds=time.perf_counter() - worker_started,
        )
    _source, prepared = prepared_pair
    attempt_started = time.perf_counter()
    rank_started: float | None = None
    bounded: UnitAwareBoundedLanguage | None = None
    try:
        grammar, truth, cfg_seconds = contextual_grammar(
            state.semantics,
            state.workspace,
            sample.member,
            prepared,
            config.builder_options,
            state.library_catalog,
        )
        if len(grammar.productions) > config.max_grammar_productions:
            return RankOutcome(
                member=sample.member,
                failure="grammar_production_cap",
                failure_message=(
                    f"{len(grammar.productions):,} productions exceeds "
                    f"{config.max_grammar_productions:,}"
                ),
                elapsed_seconds=time.perf_counter() - worker_started,
            )
        rank_started = time.perf_counter()
        with rank_deadline(config.rank_timeout):
            bounded = UnitAwareBoundedLanguage(
                grammar,
                len(truth),
                config.max_dfa_states,
            )
            rank_walk_started = time.perf_counter()
            rank0, language_size, length_sizes = shortlex_rank(bounded, truth)
            rank_within_length = rank0 - sum(length_sizes[:-1])
            if bounded.unrank(
                len(truth), rank_within_length
            ) != tuple(truth):
                raise evaluator.EvaluationError(
                    "rank/unrank round trip disagrees"
                )
            rank_walk_seconds = time.perf_counter() - rank_walk_started
        rank_seconds = time.perf_counter() - rank_started
        rank1 = rank0 + 1
        rerank_started = time.perf_counter()
        reranked = rerank_shortlex_prefix_by_fourgrams_and_wdfa(
            bounded,
            truth,
            rank0,
            state.fourgram_model,
            state.wdfa_model,
            config.fourgram_candidate_limit,
        )
        rerank_seconds = time.perf_counter() - rerank_started
        fourgram = reranked.fourgram
        wdfa = reranked.wdfa
        total_seconds = time.perf_counter() - attempt_started
        prefix_truncated = language_size > config.fourgram_candidate_limit
        record: dict[str, object] = {
            "event": "rank",
            "index": 0,
            "member": sample.member,
            **source_record_metadata(sample),
            "line": prepared.target.node.lineno,
            "column": prepared.target.node.col_offset,
            "kind": prepared.target.kind,
            "tokens": len(truth),
            "ground_truth": list(truth),
            "productions": len(grammar.productions),
            "nonterminals": len(grammar.nonterminals),
            "dfa_states": len(bounded.rows),
            "length_sizes": [compact_integer(size) for size in length_sizes],
            "length_sizes_exact": [str(size) for size in length_sizes],
            "bounded_language_size": compact_integer(language_size),
            "bounded_language_size_exact": str(language_size),
            "rank0": compact_integer(rank0),
            "rank1": compact_integer(rank1),
            "rank0_exact": str(rank0),
            "rank1_exact": str(rank1),
            "rank1_log2": math.log2(rank1),
            "cfg_seconds": cfg_seconds,
            "dfa_seconds": rank_seconds - rank_walk_seconds,
            "rank_walk_seconds": rank_walk_seconds,
            "rank_seconds": rank_seconds,
            "fourgram_rank1": fourgram.rank1,
            "fourgram_rank1_exact": (
                str(fourgram.rank1) if fourgram.rank1 is not None else None
            ),
            "fourgram_ground_truth_log_probability": (
                fourgram.ground_truth_log_probability
            ),
            "fourgram_candidate_count": fourgram.candidate_count,
            "fourgram_candidate_limit": config.fourgram_candidate_limit,
            "fourgram_truth_in_prefix": not fourgram.censored,
            "fourgram_prefix_truncated": prefix_truncated,
            "fourgram_rank_is_global": not prefix_truncated,
            "fourgram_rank_scope": (
                "censored_outside_shortlex_prefix"
                if fourgram.censored
                else (
                    "full_bounded_language"
                    if not prefix_truncated
                    else "first_shortlex_prefix"
                )
            ),
            "fourgram_censored_reason": (
                "ground_truth_outside_shortlex_prefix"
                if fourgram.censored
                else None
            ),
            "fourgram_strictly_better": fourgram.strictly_better,
            "fourgram_tied_before": fourgram.tied_before,
            "fourgram_tied_total": fourgram.tied_total,
            "wdfa_rank1": wdfa.rank1,
            "wdfa_rank1_exact": (
                str(wdfa.rank1) if wdfa.rank1 is not None else None
            ),
            "wdfa_ground_truth_cost": wdfa.ground_truth_cost,
            "wdfa_ground_truth_negative_log_probability": (
                wdfa.ground_truth_cost / state.wdfa_model.scale
                if wdfa.ground_truth_valid
                and wdfa.ground_truth_cost is not None
                else None
            ),
            "wdfa_ground_truth_valid": wdfa.ground_truth_valid,
            "wdfa_candidate_count": wdfa.candidate_count,
            "wdfa_candidate_limit": config.fourgram_candidate_limit,
            "wdfa_truth_in_prefix": not wdfa.censored,
            "wdfa_prefix_truncated": prefix_truncated,
            "wdfa_rank_is_global": not prefix_truncated,
            "wdfa_rank_scope": (
                "censored_outside_shortlex_prefix"
                if wdfa.censored
                else (
                    "full_bounded_language"
                    if not prefix_truncated
                    else "first_shortlex_prefix"
                )
            ),
            "wdfa_censored_reason": (
                "ground_truth_outside_shortlex_prefix"
                if wdfa.censored
                else None
            ),
            "wdfa_strictly_better": wdfa.strictly_better,
            "wdfa_tied_before": wdfa.tied_before,
            "wdfa_tied_total": wdfa.tied_total,
            "wdfa_valid_candidates": wdfa.valid_candidates,
            "wdfa_invalid_candidates": (
                wdfa.candidate_count - wdfa.valid_candidates
                if wdfa.valid_candidates is not None
                else None
            ),
            "rerank_seconds": rerank_seconds,
            "total_seconds": total_seconds,
        }
        return RankOutcome(
            member=sample.member,
            record=record,
            rank1_exact=rank1,
            elapsed_seconds=time.perf_counter() - worker_started,
        )
    except evaluator.LanguageTooLarge as error:
        censored = (
            time.perf_counter() - rank_started
            if rank_started is not None
            else None
        )
        return RankOutcome(
            member=sample.member,
            failure="dfa_state_cap",
            failure_message=str(error),
            elapsed_seconds=time.perf_counter() - worker_started,
            censored_rank_seconds=censored,
        )
    except RankTimeout as error:
        censored = (
            time.perf_counter() - rank_started
            if rank_started is not None
            else None
        )
        return RankOutcome(
            member=sample.member,
            failure="rank_timeout",
            failure_message=str(error),
            elapsed_seconds=time.perf_counter() - worker_started,
            censored_rank_seconds=censored,
        )
    except (evaluator.EvaluationError, MemoryError, RecursionError) as error:
        return RankOutcome(
            member=sample.member,
            failure=type(error).__name__,
            failure_message=str(error),
            elapsed_seconds=time.perf_counter() - worker_started,
        )
    finally:
        if bounded is not None:
            del bounded
        gc.collect()


def _rank_sample(sample: SourceSample) -> RankOutcome:
    config = _WORKER_CONFIG
    if config is None:
        raise RuntimeError("rank worker was not initialized")
    started = time.perf_counter()
    try:
        with sample_deadline(config.sample_timeout):
            return _rank_sample_impl(sample)
    except SampleTimeout as error:
        # The alarm may have interrupted a blocking JSON-RPC read.  Those LSP
        # streams cannot be safely reused, so replace both servers before the
        # worker accepts another source.
        try:
            _restart_rank_worker()
        except BaseException as restart_error:
            raise RuntimeError(
                f"failed to restart worker after timeout: {restart_error}"
            ) from restart_error
        return RankOutcome(
            member=sample.member,
            failure="sample_timeout",
            failure_message=str(error),
            elapsed_seconds=time.perf_counter() - started,
        )


def logarithmic_edges(
    minimum: float, maximum: float, bins: int
) -> list[float]:
    if minimum <= 0:
        raise ValueError("logarithmic histogram values must be positive")
    if minimum == maximum:
        return [minimum / 2, maximum * 2]
    low = math.log10(minimum)
    high = math.log10(maximum)
    return [
        10 ** (low + (high - low) * index / bins)
        for index in range(bins + 1)
    ]


def integer_log10(value: int) -> float:
    """Return log10(value) without converting an enormous int to float."""

    if value <= 0:
        raise ValueError("rank must be positive")
    bits = value.bit_length()
    if bits <= 53:
        return math.log10(value)
    shift = bits - 53
    return math.log10(value >> shift) + shift * math.log10(2)


def _load_plotting() -> tuple[Any, Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, MaxNLocator

    return matplotlib, plt, LogLocator, MaxNLocator


def _save_figure_atomically(
    figure: Any, path: Path, *, dpi: int | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        savefig = getattr(figure, "savefig")
        keywords: dict[str, object] = {
            "format": path.suffix.lstrip("."),
            "bbox_inches": "tight",
        }
        if dpi is not None:
            keywords["dpi"] = dpi
        savefig(temporary_path, **keywords)
        temporary_path.chmod(0o644)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _problem_count(members: Sequence[str]) -> int:
    problems: set[tuple[str, ...]] = set()
    for member in members:
        parts = member.split("/")
        if (
            len(parts) >= 3
            and len(parts[2]) == 6
            and parts[2].startswith("p")
            and parts[2][1:].isdigit()
        ):
            problems.add(("codenet", parts[2]))
        elif len(parts) >= 4 and parts[0] == "APPS":
            problems.add(("apps", parts[1], parts[2]))
    return len(problems)


def update_histograms(
    rank1_values: Sequence[int],
    rank_times: Sequence[float],
    members: Sequence[str],
    paired_shortlex_rank1_values: Sequence[int],
    fourgram_rank1_values: Sequence[int],
    wdfa_rank1_values: Sequence[int],
    fourgram_censored: int,
    wdfa_censored: int,
    fourgram_prefix_truncated: int,
    wdfa_prefix_truncated: int,
    fourgram_candidate_limit: int,
    output_directory: Path,
    output_stem: str,
    rank_bins: int,
    time_bins: int,
    dpi: int,
    workers: int,
    write_pdf: bool,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if not rank1_values or len(rank1_values) != len(rank_times):
        raise ValueError("rank and timing histogram inputs must be nonempty")
    if not (
        len(paired_shortlex_rank1_values)
        == len(fourgram_rank1_values)
        == len(wdfa_rank1_values)
    ):
        raise ValueError(
            "paired shortlex, 4-gram, and WDFA ranks must have equal length"
        )
    if any(rank <= 0 for rank in paired_shortlex_rank1_values):
        raise ValueError("paired shortlex ranks must be positive")
    if any(rank <= 0 for rank in fourgram_rank1_values):
        raise ValueError("4-gram ranks must be positive")
    if any(rank <= 0 for rank in wdfa_rank1_values):
        raise ValueError("WDFA ranks must be positive")
    if any(
        value < 0
        for value in (
            fourgram_censored,
            wdfa_censored,
            fourgram_prefix_truncated,
            wdfa_prefix_truncated,
        )
    ):
        raise ValueError("reranking censoring counts must be nonnegative")
    if fourgram_candidate_limit < 1:
        raise ValueError("reranking candidate limit must be positive")
    _matplotlib, plt, LogLocator, MaxNLocator = _load_plotting()
    problem_count = _problem_count(members)
    sorted_ranks = sorted(rank1_values)
    lower_median_rank = sorted_ranks[(len(sorted_ranks) - 1) // 2]

    rank_figure, rank_axis = plt.subplots(figsize=(8.2, 4.8))
    # Leave headroom below float overflow: Matplotlib's logarithmic tick
    # locator may exponentiate one tick beyond the observed maximum.
    if max(rank1_values).bit_length() <= 900:
        rank_floats = [float(rank) for rank in rank1_values]
        rank_axis.hist(
            rank_floats,
            bins=logarithmic_edges(
                min(rank_floats), max(rank_floats), rank_bins
            ),
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.8,
        )
        rank_axis.set_xscale("log")
        rank_axis.xaxis.set_major_locator(LogLocator(base=10))
        rank_axis.axvline(
            float(lower_median_rank),
            color="#E45756",
            linestyle="--",
            linewidth=1.4,
            label=f"lower median = {compact_integer(lower_median_rank)}",
        )
        rank_axis.set_xlabel(
            "Ground-truth shortlex rank (one-based, log scale)"
        )
    else:
        rank_logs = [integer_log10(rank) for rank in rank1_values]
        low, high = min(rank_logs), max(rank_logs)
        edges = (
            [low - 0.5, high + 0.5]
            if low == high
            else [
                low + (high - low) * index / rank_bins
                for index in range(rank_bins + 1)
            ]
        )
        rank_axis.hist(
            rank_logs,
            bins=edges,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.8,
        )
        rank_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        rank_axis.axvline(
            integer_log10(lower_median_rank),
            color="#E45756",
            linestyle="--",
            linewidth=1.4,
            label=f"lower median = {compact_integer(lower_median_rank)}",
        )
        rank_axis.set_xlabel(
            "Ground-truth shortlex rank (one-based, log10 coordinate)"
        )
    rank_axis.set_ylabel("Statements")
    rank_axis.set_title("Contextual CFG Ground-Truth Rank Distribution")
    rank_axis.grid(axis="y", alpha=0.25)
    rank_axis.legend(frameon=False)
    rank_figure.text(
        0.99,
        0.01,
        f"n={len(rank1_values):,} statements; {problem_count:,} "
        f"problems; range={compact_integer(min(rank1_values))}–"
        f"{compact_integer(max(rank1_values))}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    rank_figure.tight_layout(rect=(0, 0.035, 1, 1))

    positive_times = [value for value in rank_times if value > 0]
    if not positive_times:
        plt.close(rank_figure)
        raise ValueError("rank timings must be positive")
    time_figure, time_axis = plt.subplots(figsize=(8.2, 4.8))
    time_axis.hist(
        positive_times,
        bins=logarithmic_edges(
            min(positive_times), max(positive_times), time_bins
        ),
        color="#72B7B2",
        edgecolor="white",
        linewidth=0.8,
    )
    time_axis.set_xscale("log")
    time_axis.xaxis.set_major_locator(LogLocator(base=10))
    median_time = statistics.median(positive_times)
    time_axis.axvline(
        median_time,
        color="#E45756",
        linestyle="--",
        linewidth=1.4,
        label=f"median = {median_time:.4g}s",
    )
    time_axis.set_xlabel(
        "Per-worker DAFSA construction + shortlex rank wall time "
        "(seconds, log scale)"
    )
    time_axis.set_ylabel("Statements")
    time_axis.set_title("Contextual CFG Ground-Truth Rank Timing")
    time_axis.grid(axis="y", alpha=0.25)
    time_axis.legend(frameon=False)
    time_figure.text(
        0.99,
        0.01,
        f"n={len(positive_times):,}; workers={workers:,}; "
        f"range={min(positive_times):.4g}–"
        f"{max(positive_times):.4g}s",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    time_figure.tight_layout(rect=(0, 0.035, 1, 1))

    comparison_figure = plt.figure(figsize=(13.2, 7.4))
    comparison_grid = comparison_figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.13,
        hspace=0.31,
    )
    comparison_shortlex_axis = comparison_figure.add_subplot(
        comparison_grid[:, 0]
    )
    comparison_fourgram_axis = comparison_figure.add_subplot(
        comparison_grid[0, 1],
        sharex=comparison_shortlex_axis,
        sharey=comparison_shortlex_axis,
    )
    comparison_wdfa_axis = comparison_figure.add_subplot(
        comparison_grid[1, 1],
        sharex=comparison_shortlex_axis,
        sharey=comparison_shortlex_axis,
    )
    comparison_axes = (
        comparison_shortlex_axis,
        comparison_fourgram_axis,
        comparison_wdfa_axis,
    )
    paired_count = len(fourgram_rank1_values)
    if paired_count:
        comparison_lower_medians = (
            sorted(paired_shortlex_rank1_values)[(paired_count - 1) // 2],
            sorted(fourgram_rank1_values)[(paired_count - 1) // 2],
            sorted(wdfa_rank1_values)[(paired_count - 1) // 2],
        )
        comparison_values = [
            *paired_shortlex_rank1_values,
            *fourgram_rank1_values,
            *wdfa_rank1_values,
        ]
        if any(rank > fourgram_candidate_limit for rank in comparison_values):
            raise ValueError(
                "paired reranking ranks exceed the shared candidate limit"
            )
        # Fix the shared domain to the full reranking prefix rather than each
        # sample's observed extrema, so successive online plots and all three
        # ranking strategies remain directly comparable.
        if fourgram_candidate_limit.bit_length() <= 900:
            common_edges = logarithmic_edges(
                1.0, float(fourgram_candidate_limit), rank_bins
            )
            plot_values = (
                [float(rank) for rank in paired_shortlex_rank1_values],
                [float(rank) for rank in fourgram_rank1_values],
                [float(rank) for rank in wdfa_rank1_values],
            )
            median_positions = (
                float(comparison_lower_medians[0]),
                float(comparison_lower_medians[1]),
                float(comparison_lower_medians[2]),
            )
            for axis in comparison_axes:
                axis.set_xscale("log")
                axis.xaxis.set_major_locator(LogLocator(base=10))
                axis.set_xlim(common_edges[0], common_edges[-1])
            x_label = "Ground-truth rank (one-based, log scale)"
        else:
            high = integer_log10(fourgram_candidate_limit)
            common_edges = [
                high * index / rank_bins for index in range(rank_bins + 1)
            ]
            plot_values = (
                [
                    integer_log10(rank)
                    for rank in paired_shortlex_rank1_values
                ],
                [integer_log10(rank) for rank in fourgram_rank1_values],
                [integer_log10(rank) for rank in wdfa_rank1_values],
            )
            median_positions = (
                integer_log10(comparison_lower_medians[0]),
                integer_log10(comparison_lower_medians[1]),
                integer_log10(comparison_lower_medians[2]),
            )
            for axis in comparison_axes:
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.set_xlim(common_edges[0], common_edges[-1])
            x_label = "Ground-truth rank (one-based, log10 coordinate)"

        panel_specs = (
            (
                comparison_axes[0],
                plot_values[0],
                median_positions[0],
                paired_shortlex_rank1_values,
                "#4C78A8",
                "Shortlex rank",
            ),
            (
                comparison_axes[1],
                plot_values[1],
                median_positions[1],
                fourgram_rank1_values,
                "#F58518",
                "Lexical 4-gram rank",
            ),
            (
                comparison_axes[2],
                plot_values[2],
                median_positions[2],
                wdfa_rank1_values,
                "#54A24B",
                "WDFA rank",
            ),
        )
        for axis, values, median_position, exact_values, color, title in panel_specs:
            axis.hist(
                values,
                bins=common_edges,
                color=color,
                edgecolor="white",
                linewidth=0.65,
            )
            lower_median = sorted(exact_values)[(paired_count - 1) // 2]
            axis.axvline(
                median_position,
                color="#E45756",
                linestyle="--",
                linewidth=1.3,
                label=f"lower median = {compact_integer(lower_median)}",
            )
            axis.set_title(title)
            axis.set_xlabel(x_label)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(frameon=False)
    else:
        for axis, title in zip(
            comparison_axes,
            ("Shortlex rank", "Lexical 4-gram rank", "WDFA rank"),
            strict=True,
        ):
            axis.set_title(title)
            axis.set_xscale("log")
            axis.set_xlim(0.5, 2.0)
            axis.set_xlabel("Ground-truth rank (one-based, log scale)")
            axis.text(
                0.5,
                0.5,
                "No measurable paired ranks yet",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
            axis.grid(axis="y", alpha=0.25)
    comparison_shortlex_axis.set_ylabel("Statements")
    comparison_figure.suptitle(
        "Ground-Truth Rank: Shortlex, Lexical 4-Gram, and WDFA Reranking"
    )
    comparison_figure.text(
        0.99,
        0.01,
        f"joint n={paired_count:,}; each reranker scores up to the first "
        f"{fourgram_candidate_limit:,} shortlex statements; censored outside "
        f"prefix: 4-gram={fourgram_censored:,}, WDFA={wdfa_censored:,}; "
        f"measurable from truncated languages: "
        f"4-gram={fourgram_prefix_truncated:,}, "
        f"WDFA={wdfa_prefix_truncated:,}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    comparison_figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.10,
        top=0.90,
    )

    rank_png = output_directory / f"{output_stem}_rank_histogram.png"
    rank_pdf = output_directory / f"{output_stem}_rank_histogram.pdf"
    time_png = output_directory / f"{output_stem}_rank_time_histogram.png"
    time_pdf = output_directory / f"{output_stem}_rank_time_histogram.pdf"
    comparison_png = (
        output_directory
        / f"{output_stem}_shortlex_vs_4gram_vs_wdfa_rank_histogram.png"
    )
    comparison_pdf = (
        output_directory
        / f"{output_stem}_shortlex_vs_4gram_vs_wdfa_rank_histogram.pdf"
    )
    try:
        _save_figure_atomically(rank_figure, rank_png, dpi=dpi)
        _save_figure_atomically(time_figure, time_png, dpi=dpi)
        _save_figure_atomically(comparison_figure, comparison_png, dpi=dpi)
        if write_pdf:
            _save_figure_atomically(rank_figure, rank_pdf)
            _save_figure_atomically(time_figure, time_pdf)
            _save_figure_atomically(comparison_figure, comparison_pdf)
    finally:
        plt.close(rank_figure)
        plt.close(time_figure)
        plt.close(comparison_figure)
    return (
        rank_png,
        rank_pdf,
        time_png,
        time_pdf,
        comparison_png,
        comparison_pdf,
    )


def emit(
    record: Mapping[str, object],
    json_lines: bool,
    output: TextIO | None = None,
) -> None:
    serialized = json.dumps(record, sort_keys=True)
    if output is not None:
        print(serialized, file=output, flush=True)
    if json_lines:
        print(serialized, flush=True)
        return
    if record.get("event") == "start":
        print(
            f"population={record['population']}; pool={record['pool_files']} "
            f"sources; target={record['requested_statements']} exact ranks; "
            f"workers={record['workers']}",
            flush=True,
        )
        return
    if record.get("event") == "rank":
        print(
            f"[{record['index']:05d}] {record['member']}:{record['line']} "
            f"tokens={record['tokens']} states={record['dfa_states']:,} "
            f"rank={record['rank0']} dfa+rank={record['rank_seconds']:.4f}s "
            f"cfg={record['cfg_seconds']:.4f}s",
            flush=True,
        )
    elif record.get("event") == "failure":
        return
    elif record.get("event") == "summary":
        print(serialized, flush=True)


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("apps", "codenet"),
        default="apps",
        help="source dataset (default: apps)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="APPS split (default: test; ignored for CodeNet)",
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        help=(
            "dataset source (default: evaluator's APPS split JSONL or "
            "CodeNet archive)"
        ),
    )
    parser.add_argument("-n", "--statements", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pool-files",
        type=int,
        help="sampled-source pool (default: max(400, 2 * statements))",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional evaluator JSONL used as a fast eligible-source manifest",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        help="test-only cap on dataset sources scanned without a manifest",
    )
    parser.add_argument("--max-dfa-states", type=int, default=250_000)
    parser.add_argument("--max-grammar-productions", type=int, default=250_000)
    parser.add_argument("--rank-timeout", type=float, default=60.0)
    parser.add_argument(
        "--sample-timeout",
        type=float,
        default=300.0,
        help="wall-time cap for a whole source, including ty queries; 0 disables",
    )
    parser.add_argument(
        "--fourgram-counts",
        type=Path,
        default=Path(__file__).resolve().with_name("python_4grams.txt"),
        help="raw lexical 4-gram counts used for Laplace-smoothed reranking",
    )
    parser.add_argument(
        "--wdfa",
        type=Path,
        default=Path(__file__).resolve().with_name("wdfa.bin"),
        help="serialized Tidyparse Python WDFA used for cost reranking",
    )
    parser.add_argument(
        "--fourgram-candidate-limit",
        "--rerank-candidate-limit",
        dest="fourgram_candidate_limit",
        type=int,
        default=1_000_000,
        help="shortlex-prefix size reranked by both the 4-gram and WDFA",
    )
    parser.add_argument("--ty", default="ty")
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="rank workers (default: all available CPU cores)",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        help="maximum queued sources (default: 8 * workers)",
    )
    parser.add_argument(
        "--library-dir", type=Path, default=evaluator.DEFAULT_LIBRARY_DIRECTORY
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="incremental JSONL output (default name includes dataset and split)",
    )
    parser.add_argument(
        "--plot-directory",
        type=Path,
        help="plot directory (default: beside --output-jsonl)",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=100,
        help="refresh plots after this many successful ranks; 0 disables",
    )
    parser.add_argument("--rank-plot-bins", type=int, default=100)
    parser.add_argument("--time-plot-bins", type=int, default=20)
    parser.add_argument("--plot-dpi", type=int, default=240)
    parser.add_argument("--allow-ignores", action="store_true")
    parser.add_argument("--jsonl", action="store_true")
    parsed = parser.parse_args(arguments)
    if parsed.source is None:
        parsed.source = evaluator.default_dataset_source(
            parsed.dataset, parsed.split
        )
    if parsed.pool_files is None:
        parsed.pool_files = max(400, 2 * parsed.statements)
    if parsed.lookahead is None:
        parsed.lookahead = 8 * parsed.workers
    if parsed.output_jsonl is None:
        dataset_label = (
            f"{parsed.dataset}_{parsed.split}"
            if parsed.dataset == "apps"
            else parsed.dataset
        )
        parsed.output_jsonl = Path(__file__).resolve().with_name(
            f"api2cfg_rank_{dataset_label}_{parsed.statements}_full.jsonl"
        )
    if parsed.plot_directory is None:
        parsed.plot_directory = parsed.output_jsonl.parent
    if parsed.statements < 1:
        parser.error("--statements must be at least 1")
    if parsed.pool_files < parsed.statements:
        parser.error("--pool-files must be at least --statements")
    if parsed.max_dfa_states < 1:
        parser.error("--max-dfa-states must be at least 1")
    if parsed.max_grammar_productions < 1:
        parser.error("--max-grammar-productions must be at least 1")
    if parsed.rank_timeout < 0:
        parser.error("--rank-timeout must be nonnegative")
    if parsed.sample_timeout < 0:
        parser.error("--sample-timeout must be nonnegative")
    if parsed.fourgram_candidate_limit < 1:
        parser.error("--fourgram-candidate-limit must be at least 1")
    if not parsed.fourgram_counts.is_file():
        parser.error(f"4-gram count file not found: {parsed.fourgram_counts}")
    if not parsed.wdfa.is_file():
        parser.error(f"WDFA file not found: {parsed.wdfa}")
    if parsed.workers < 1:
        parser.error("--workers must be at least 1")
    if parsed.lookahead < parsed.workers:
        parser.error("--lookahead must be at least --workers")
    if parsed.plot_every < 0:
        parser.error("--plot-every must be nonnegative")
    if parsed.rank_plot_bins < 1:
        parser.error("--rank-plot-bins must be at least 1")
    if parsed.time_plot_bins < 1:
        parser.error("--time-plot-bins must be at least 1")
    if parsed.plot_dpi < 1:
        parser.error("--plot-dpi must be at least 1")
    if parsed.scan_limit is not None and parsed.scan_limit < parsed.pool_files:
        parser.error("--scan-limit must be at least --pool-files")
    return parsed


def evaluate(arguments: argparse.Namespace) -> int:
    benchmark_started = time.perf_counter()
    resolved_source = evaluator.resolved_dataset_source(
        arguments.dataset, arguments.source, arguments.split
    )
    if not resolved_source.is_file():
        raise evaluator.EvaluationError(
            f"dataset source not found: {resolved_source}"
        )
    fourgram_model = LexicalFourGramModel.from_path(arguments.fourgram_counts)
    fourgram_digest = hashlib.sha256(
        arguments.fourgram_counts.read_bytes()
    ).hexdigest()
    try:
        wdfa_model = PythonWDFA.from_path(arguments.wdfa)
    except WDFAFormatError as error:
        raise evaluator.EvaluationError(f"invalid Python WDFA: {error}") from error
    if arguments.plot_every:
        try:
            _load_plotting()
        except ImportError as error:
            raise evaluator.EvaluationError(
                "online plotting requires matplotlib; use a Python environment "
                "that provides it or pass --plot-every 0"
            ) from error

    manifest_selection: ManifestSelection | None = None
    manifest_digest: str | None = None
    if arguments.manifest is not None:
        manifest_selection = manifest_members(
            arguments.manifest,
            arguments.seed,
            arguments.pool_files,
            arguments.dataset,
            arguments.split,
        )
        if not manifest_selection.members:
            split_description = (
                f", split={arguments.split}" if arguments.dataset == "apps" else ""
            )
            raise evaluator.EvaluationError(
                "manifest has no compatible recognized statement/rank rows "
                f"for dataset={arguments.dataset}{split_description}; "
                f"dataset mismatches={manifest_selection.dataset_mismatch_records:,}, "
                f"split mismatches={manifest_selection.split_mismatch_records:,}"
            )
        samples = extract_manifest_samples(
            arguments.dataset,
            arguments.source,
            arguments.split,
            manifest_selection.members,
        )
        manifest_digest = file_sha256(arguments.manifest)
        scanned: int | str = "manifest"
        population = f"recognized {arguments.dataset} sources in {arguments.manifest}"
    else:
        samples, scanned = minhash_dataset_samples(
            arguments.dataset,
            arguments.source,
            arguments.split,
            arguments.seed,
            arguments.pool_files,
            arguments.scan_limit,
        )
        dataset_population = (
            f"APPS {arguments.split} solutions"
            if arguments.dataset == "apps"
            else "CodeNet Python files"
        )
        population = (
            f"bottom-k hash sample of all distinct {dataset_population}"
            if arguments.scan_limit is None
            else (
                "bottom-k hash sample of first "
                f"{arguments.scan_limit} distinct {dataset_population}"
            )
        )

    worker_count = min(arguments.workers, max(1, len(samples)))
    output_stem = arguments.output_jsonl.stem
    source_stat = resolved_source.stat()
    version = evaluator.ty_version(arguments.ty)
    builder_options = evaluator.BuilderOptions()
    start: dict[str, object] = {
        "event": "start",
        "dataset": arguments.dataset,
        "split": arguments.split if arguments.dataset == "apps" else None,
        "source": str(arguments.source),
        "resolved_source": str(resolved_source),
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "population": population,
        "seed": arguments.seed,
        "ty": version,
        "python": sys.version,
        "platform": platform.platform(),
        "library_directory": str(arguments.library_dir),
        "allow_ignores": arguments.allow_ignores,
        "builder": {
            "max_layouts_per_signature": (
                builder_options.max_layouts_per_signature
            ),
            "member_depth": builder_options.member_depth,
            "max_receiver_types": builder_options.max_receiver_types,
            "max_module_members": builder_options.max_module_members,
        },
        "requested_statements": arguments.statements,
        "pool_files": len(samples),
        "scanned_python_sources": scanned,
        "max_dfa_states": arguments.max_dfa_states,
        "max_grammar_productions": arguments.max_grammar_productions,
        "rank_timeout": arguments.rank_timeout,
        "sample_timeout": arguments.sample_timeout,
        "fourgram_counts": str(arguments.fourgram_counts),
        "fourgram_counts_sha256": fourgram_digest,
        "fourgram_count_rows": len(fourgram_model.counts),
        "fourgram_contexts": len(fourgram_model.context_totals),
        "fourgram_vocabulary_size": len(fourgram_model.vocabulary),
        "fourgram_smoothing": "conditional add-one (Laplace alpha=1)",
        "fourgram_boundaries": "BOS NEWLINE <statement> NEWLINE EOS",
        "fourgram_initial_token_prior": "uniform omitted constant",
        "fourgram_candidate_limit": arguments.fourgram_candidate_limit,
        "fourgram_candidate_scope": (
            "first N distinct bounded-language words in global shortlex order"
        ),
        "fourgram_tie_break": "original shortlex order",
        "fourgram_count_table_note": (
            "frequency-truncated raw counts; absent grams treated as zero"
        ),
        "wdfa": str(arguments.wdfa),
        "wdfa_metadata": wdfa_model.metadata.as_dict(),
        "wdfa_scoring": (
            "quantized negative-log transition costs plus final cost; "
            "lower is better"
        ),
        "wdfa_boundaries": "append NEWLINE; start/final weights encode boundaries",
        "wdfa_missing_transition": (
            "stay in current state and add missing_cost"
        ),
        "wdfa_candidate_limit": arguments.fourgram_candidate_limit,
        "wdfa_candidate_scope": (
            "same first N distinct bounded-language words as 4-gram reranking"
        ),
        "wdfa_tie_break": "original shortlex order",
        "workers": worker_count,
        "lookahead": arguments.lookahead,
        "output_jsonl": str(arguments.output_jsonl),
        "plot_every": arguments.plot_every,
        "rank_plot_bins": arguments.rank_plot_bins,
        "time_plot_bins": arguments.time_plot_bins,
        "plot_directory": str(arguments.plot_directory),
        "terminal_order": "sorted canonical token strings",
        "result_order": "deterministic bottom-k sample order",
    }
    if manifest_selection is not None:
        start.update(
            {
                "manifest": str(arguments.manifest),
                "manifest_sha256": manifest_digest,
                "manifest_compatible_records": (
                    manifest_selection.compatible_records
                ),
                "manifest_compatible_members": (
                    manifest_selection.compatible_members
                ),
                "manifest_selected_members": (
                    manifest_selection.selected_members
                ),
                "manifest_matched_members": len(samples),
                "manifest_missing_members": (
                    manifest_selection.selected_members - len(samples)
                ),
                "manifest_dataset_mismatch_records": (
                    manifest_selection.dataset_mismatch_records
                ),
                "manifest_split_mismatch_records": (
                    manifest_selection.split_mismatch_records
                ),
            }
        )
    arguments.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    failures: Counter[str] = Counter()
    rank_times: list[float] = []
    censored_rank_times: list[float] = []
    total_times: list[float] = []
    rank1_values: list[int] = []
    rerank_times: list[float] = []
    rerank_censored_times: list[float] = []
    fourgram_rank1_values: list[int] = []
    wdfa_rank1_values: list[int] = []
    wdfa_ground_truth_costs: list[int] = []
    paired_shortlex_rank1_values: list[int] = []
    fourgram_censored = 0
    wdfa_censored = 0
    fourgram_prefix_truncated = 0
    wdfa_prefix_truncated = 0
    fourgram_global = 0
    wdfa_global = 0
    wdfa_invalid_ground_truths = 0
    wdfa_valid_candidates = 0
    wdfa_invalid_candidates = 0
    ranked_members: list[str] = []
    ranked = 0
    attempted = 0
    plot_failures = 0
    config = RankWorkerConfig(
        seed=arguments.seed,
        allow_ignores=arguments.allow_ignores,
        ty=arguments.ty,
        ty_release=version,
        library_dir=arguments.library_dir,
        max_dfa_states=arguments.max_dfa_states,
        max_grammar_productions=arguments.max_grammar_productions,
        rank_timeout=arguments.rank_timeout,
        sample_timeout=arguments.sample_timeout,
        fourgram_counts=arguments.fourgram_counts,
        wdfa_path=arguments.wdfa,
        fourgram_candidate_limit=arguments.fourgram_candidate_limit,
        builder_options=builder_options,
    )
    context = multiprocessing.get_context("spawn")
    with arguments.output_jsonl.open("w", encoding="utf-8", buffering=1) as output:
        emit(start, arguments.jsonl, output)
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_rank_worker,
            initargs=(config,),
        )
        pending: dict[int, Future[RankOutcome]] = {}
        next_submit = 0
        next_consume = 0

        def fill_queue() -> None:
            nonlocal next_submit
            while (
                len(pending) < arguments.lookahead
                and next_submit < len(samples)
                and ranked < arguments.statements
            ):
                pending[next_submit] = executor.submit(
                    _rank_sample, samples[next_submit]
                )
                next_submit += 1

        def refresh_plots(*, final: bool) -> None:
            nonlocal plot_failures
            if not arguments.plot_every or not rank1_values:
                return
            try:
                paths = update_histograms(
                    rank1_values,
                    rank_times,
                    ranked_members,
                    paired_shortlex_rank1_values,
                    fourgram_rank1_values,
                    wdfa_rank1_values,
                    fourgram_censored,
                    wdfa_censored,
                    fourgram_prefix_truncated,
                    wdfa_prefix_truncated,
                    arguments.fourgram_candidate_limit,
                    arguments.plot_directory,
                    output_stem,
                    arguments.rank_plot_bins,
                    arguments.time_plot_bins,
                    arguments.plot_dpi,
                    worker_count,
                    final,
                )
            except Exception as error:
                plot_failures += 1
                print(f"plot update failed: {error}", file=sys.stderr, flush=True)
                return
            print(
                f"updated rank/timing histograms at n={ranked:,}: "
                f"{paths[0].name}, {paths[2].name}, {paths[4].name}",
                file=sys.stderr,
                flush=True,
            )

        try:
            fill_queue()
            while (
                next_consume < len(samples)
                and ranked < arguments.statements
            ):
                future = pending.pop(next_consume, None)
                if future is None:
                    break
                sample = samples[next_consume]
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = RankOutcome(
                        member=sample.member,
                        failure="worker_exception",
                        failure_message=f"{type(error).__name__}: {error}",
                    )
                next_consume += 1
                attempted += 1
                if outcome.record is None:
                    failure = outcome.failure or "unknown_failure"
                    failures[failure] += 1
                    if outcome.censored_rank_seconds is not None:
                        censored_rank_times.append(
                            outcome.censored_rank_seconds
                        )
                    emit(
                        {
                            "event": "failure",
                            "attempt": attempted,
                            "member": outcome.member,
                            **source_record_metadata(sample),
                            "reason": failure,
                            "message": outcome.failure_message,
                            "elapsed_seconds": outcome.elapsed_seconds,
                            "censored_rank_seconds": (
                                outcome.censored_rank_seconds
                            ),
                        },
                        arguments.jsonl,
                        output,
                    )
                else:
                    ranked += 1
                    record = outcome.record
                    record["index"] = ranked
                    emit(record, arguments.jsonl, output)
                    rank_seconds_value = record.get("rank_seconds")
                    total_seconds_value = record.get("total_seconds")
                    if not isinstance(rank_seconds_value, (int, float)) or not isinstance(
                        total_seconds_value, (int, float)
                    ):
                        raise evaluator.EvaluationError(
                            "worker omitted numeric timing fields"
                        )
                    rank_times.append(float(rank_seconds_value))
                    total_times.append(float(total_seconds_value))
                    if outcome.rank1_exact is None:
                        raise evaluator.EvaluationError(
                            "worker omitted exact one-based rank"
                        )
                    rank1_values.append(outcome.rank1_exact)
                    ranked_members.append(outcome.member)
                    rerank_seconds_value = record.get("rerank_seconds")
                    if not isinstance(rerank_seconds_value, (int, float)):
                        raise evaluator.EvaluationError(
                            "worker omitted numeric joint-reranking timing field"
                        )
                    fourgram_rank_value = record.get("fourgram_rank1")
                    wdfa_rank_value = record.get("wdfa_rank1")
                    fourgram_truth_in_prefix = record.get(
                        "fourgram_truth_in_prefix"
                    )
                    wdfa_truth_in_prefix = record.get("wdfa_truth_in_prefix")
                    fourgram_truncated = record.get(
                        "fourgram_prefix_truncated"
                    )
                    wdfa_truncated = record.get("wdfa_prefix_truncated")
                    wdfa_ground_truth_valid = record.get(
                        "wdfa_ground_truth_valid"
                    )
                    if not all(
                        isinstance(value, bool)
                        for value in (
                            fourgram_truth_in_prefix,
                            wdfa_truth_in_prefix,
                            fourgram_truncated,
                            wdfa_truncated,
                            wdfa_ground_truth_valid,
                        )
                    ):
                        raise evaluator.EvaluationError(
                            "worker omitted reranking prefix metadata"
                        )
                    if (
                        fourgram_truth_in_prefix != wdfa_truth_in_prefix
                        or fourgram_truncated != wdfa_truncated
                    ):
                        raise evaluator.EvaluationError(
                            "4-gram and WDFA candidate cohorts disagree"
                        )
                    if not wdfa_ground_truth_valid:
                        wdfa_invalid_ground_truths += 1
                    else:
                        ground_truth_cost = record.get(
                            "wdfa_ground_truth_cost"
                        )
                        if not isinstance(ground_truth_cost, int):
                            raise evaluator.EvaluationError(
                                "worker omitted finite WDFA ground-truth cost"
                            )
                        wdfa_ground_truth_costs.append(ground_truth_cost)
                    if fourgram_truth_in_prefix:
                        rerank_times.append(float(rerank_seconds_value))
                        if not isinstance(fourgram_rank_value, int) or not isinstance(
                            wdfa_rank_value, int
                        ):
                            raise evaluator.EvaluationError(
                                "worker omitted a measurable reranking rank"
                            )
                        fourgram_rank1_values.append(fourgram_rank_value)
                        wdfa_rank1_values.append(wdfa_rank_value)
                        paired_shortlex_rank1_values.append(
                            outcome.rank1_exact
                        )
                        valid_candidates_value = record.get(
                            "wdfa_valid_candidates"
                        )
                        invalid_candidates_value = record.get(
                            "wdfa_invalid_candidates"
                        )
                        if not isinstance(
                            valid_candidates_value, int
                        ) or not isinstance(invalid_candidates_value, int):
                            raise evaluator.EvaluationError(
                                "worker omitted WDFA candidate support counts"
                            )
                        wdfa_valid_candidates += valid_candidates_value
                        wdfa_invalid_candidates += invalid_candidates_value
                        if fourgram_truncated:
                            fourgram_prefix_truncated += 1
                            wdfa_prefix_truncated += 1
                        else:
                            fourgram_global += 1
                            wdfa_global += 1
                    else:
                        rerank_censored_times.append(
                            float(rerank_seconds_value)
                        )
                        if fourgram_rank_value is not None or wdfa_rank_value is not None:
                            raise evaluator.EvaluationError(
                                "censored reranking ranks must be null"
                            )
                        fourgram_censored += 1
                        wdfa_censored += 1
                    if (
                        arguments.plot_every
                        and ranked % arguments.plot_every == 0
                    ):
                        refresh_plots(final=False)
                fill_queue()
        finally:
            for future in pending.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

        if arguments.plot_every and rank1_values:
            # Always produce final PDFs.  Checkpoints update only the PNGs to
            # avoid hundreds of unnecessary vector-rendering passes.
            refresh_plots(final=True)

        summary: dict[str, object] = {
            "event": "summary",
            "dataset": arguments.dataset,
            "split": arguments.split if arguments.dataset == "apps" else None,
            "source": str(arguments.source),
            "resolved_source": str(resolved_source),
            "population": population,
            "requested_statements": arguments.statements,
            "ranked_statements": ranked,
            "distinct_sources": len(set(ranked_members)),
            "distinct_files": len(set(ranked_members)),
            "distinct_problems": _problem_count(ranked_members),
            "attempted_sources": attempted,
            "attempted_files": attempted,
            "sample_pool_exhausted": next_consume >= len(samples),
            "failures": dict(sorted(failures.items())),
            "plot_failures": plot_failures,
            "min_rank": str(min(rank1_values)) if rank1_values else None,
            "max_rank": str(max(rank1_values)) if rank1_values else None,
            "joint_rerank_measured_statements": len(
                paired_shortlex_rank1_values
            ),
            "fourgram_measured_statements": len(fourgram_rank1_values),
            "fourgram_censored_statements": fourgram_censored,
            "fourgram_global_rank_statements": fourgram_global,
            "fourgram_measured_prefix_truncated_statements": (
                fourgram_prefix_truncated
            ),
            "min_fourgram_rank": (
                str(min(fourgram_rank1_values))
                if fourgram_rank1_values
                else None
            ),
            "max_fourgram_rank": (
                str(max(fourgram_rank1_values))
                if fourgram_rank1_values
                else None
            ),
            "median_fourgram_rank": (
                str(statistics.median_low(fourgram_rank1_values))
                if fourgram_rank1_values
                else None
            ),
            "wdfa_measured_statements": len(wdfa_rank1_values),
            "wdfa_censored_statements": wdfa_censored,
            "wdfa_global_rank_statements": wdfa_global,
            "wdfa_measured_prefix_truncated_statements": (
                wdfa_prefix_truncated
            ),
            "min_wdfa_rank": (
                str(min(wdfa_rank1_values)) if wdfa_rank1_values else None
            ),
            "max_wdfa_rank": (
                str(max(wdfa_rank1_values)) if wdfa_rank1_values else None
            ),
            "median_wdfa_rank": (
                str(statistics.median_low(wdfa_rank1_values))
                if wdfa_rank1_values
                else None
            ),
            "wdfa_invalid_ground_truths": wdfa_invalid_ground_truths,
            "min_wdfa_ground_truth_cost": (
                min(wdfa_ground_truth_costs)
                if wdfa_ground_truth_costs
                else None
            ),
            "max_wdfa_ground_truth_cost": (
                max(wdfa_ground_truth_costs)
                if wdfa_ground_truth_costs
                else None
            ),
            "median_wdfa_ground_truth_cost": (
                statistics.median_low(wdfa_ground_truth_costs)
                if wdfa_ground_truth_costs
                else None
            ),
            "wdfa_valid_candidates": wdfa_valid_candidates,
            "wdfa_invalid_candidates": wdfa_invalid_candidates,
            "wdfa_invalid_candidate_rate": (
                wdfa_invalid_candidates
                / (wdfa_valid_candidates + wdfa_invalid_candidates)
                if wdfa_valid_candidates + wdfa_invalid_candidates
                else None
            ),
            "fourgram_improved_over_shortlex": sum(
                reranked_rank < shortlex_rank
                for shortlex_rank, reranked_rank in zip(
                    paired_shortlex_rank1_values,
                    fourgram_rank1_values,
                    strict=True,
                )
            ),
            "wdfa_improved_over_shortlex": sum(
                reranked_rank < shortlex_rank
                for shortlex_rank, reranked_rank in zip(
                    paired_shortlex_rank1_values,
                    wdfa_rank1_values,
                    strict=True,
                )
            ),
            "wdfa_improved_over_fourgram": sum(
                wdfa_rank < fourgram_rank
                for fourgram_rank, wdfa_rank in zip(
                    fourgram_rank1_values,
                    wdfa_rank1_values,
                    strict=True,
                )
            ),
            "min_joint_rerank_seconds": (
                min(rerank_times) if rerank_times else None
            ),
            "max_joint_rerank_seconds": (
                max(rerank_times) if rerank_times else None
            ),
            "mean_joint_rerank_seconds": (
                sum(rerank_times) / len(rerank_times)
                if rerank_times
                else None
            ),
            "min_censored_rerank_seconds": (
                min(rerank_censored_times)
                if rerank_censored_times
                else None
            ),
            "max_censored_rerank_seconds": (
                max(rerank_censored_times)
                if rerank_censored_times
                else None
            ),
            "mean_censored_rerank_seconds": (
                sum(rerank_censored_times) / len(rerank_censored_times)
                if rerank_censored_times
                else None
            ),
            "min_rank_seconds": min(rank_times) if rank_times else None,
            "max_rank_seconds": max(rank_times) if rank_times else None,
            "median_rank_seconds": (
                statistics.median(rank_times) if rank_times else None
            ),
            "mean_rank_seconds": (
                sum(rank_times) / len(rank_times) if rank_times else None
            ),
            "min_censored_rank_seconds": (
                min(censored_rank_times) if censored_rank_times else None
            ),
            "max_censored_rank_seconds": (
                max(censored_rank_times) if censored_rank_times else None
            ),
            "min_total_seconds": min(total_times) if total_times else None,
            "max_total_seconds": max(total_times) if total_times else None,
            "mean_total_seconds": (
                sum(total_times) / len(total_times) if total_times else None
            ),
            "benchmark_wall_seconds": time.perf_counter() - benchmark_started,
            "output_jsonl": str(arguments.output_jsonl),
            "rank_histogram": str(
                arguments.plot_directory / f"{output_stem}_rank_histogram.png"
            ),
            "rank_time_histogram": str(
                arguments.plot_directory
                / f"{output_stem}_rank_time_histogram.png"
            ),
            "shortlex_vs_fourgram_vs_wdfa_rank_histogram": str(
                arguments.plot_directory
                / (
                    f"{output_stem}_shortlex_vs_4gram_vs_wdfa_"
                    "rank_histogram.png"
                )
            ),
        }
        emit(summary, arguments.jsonl, output)
    return 0 if ranked == arguments.statements else 2


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    try:
        return evaluate(parsed)
    except (evaluator.EvaluationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
