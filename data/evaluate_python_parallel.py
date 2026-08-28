#!/usr/bin/env python3
"""Run evaluate_python.py over disjoint dataset shards in parallel.

Each worker owns one modulo shard of the selected dataset's Python sources.
The requested clean-file count is divided across the workers, so no source
file is evaluated twice.  This intentionally trades the evaluator's exact
global "first N clean files" boundary for a simple per-worker partition.  If
the source is omitted, ``evaluate_python.py`` selects the default APPS or
CodeNet source for ``--dataset`` and ``--split``.  Statement records are
emitted in worker-arrival order; stable dataset/member metadata should be used
when reproducible ordering is required.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import queue
import sys
import traceback
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import Manager
from typing import Mapping, Protocol, Sequence, TextIO, cast

import evaluate_python as evaluator


def split_quota(total: int, workers: int) -> list[int]:
    quotient, remainder = divmod(total, workers)
    return [quotient + int(index < remainder) for index in range(workers)]


class OutputQueue(Protocol):
    def put(self, value: tuple[int, str, str], /) -> None: ...


def integer_field(record: Mapping[str, object], name: str) -> int:
    value = record[name]
    if not isinstance(value, int):
        raise evaluator.EvaluationError(f"record field {name!r} is not an integer")
    return value


def number_field(record: Mapping[str, object], name: str) -> float:
    value = record[name]
    if not isinstance(value, (int, float)):
        raise evaluator.EvaluationError(f"record field {name!r} is not numeric")
    return float(value)


class QueueWriter:
    """A small line-buffered TextIO replacement backed by a process queue."""

    def __init__(self, output: OutputQueue, worker: int, stream: str) -> None:
        self.output = output
        self.worker = worker
        self.stream = stream
        self.buffer = ""

    def write(self, value: str) -> int:
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.output.put((self.worker, self.stream, line))
        return len(value)

    def flush(self) -> None:
        if self.buffer:
            self.output.put((self.worker, self.stream, self.buffer))
            self.buffer = ""


def run_worker(
    options: evaluator.EvaluationOptions,
    worker: int,
    output: OutputQueue,
) -> tuple[int, int]:
    stdout = QueueWriter(output, worker, "stdout")
    stderr = QueueWriter(output, worker, "stderr")
    return_code = 1
    try:
        with contextlib.redirect_stdout(
            cast(TextIO, stdout)
        ), contextlib.redirect_stderr(cast(TextIO, stderr)):
            try:
                return_code = evaluator.evaluate(options)
            except (evaluator.EvaluationError, FileNotFoundError) as error:
                print(f"error: {error}", file=sys.stderr)
            except Exception:  # Keep a failed worker from hiding its traceback.
                traceback.print_exc()
    finally:
        stdout.flush()
        stderr.flush()
        output.put((worker, "done", ""))
    return worker, return_code


@dataclass
class Aggregate:
    json_lines: bool
    dataset: str
    split: str | None
    source: str
    files_requested: int
    max_samples: int | None
    workers: int
    evaluated: int = 0
    recognized: int = 0
    precision_accepted: int = 0
    precision_checked: int = 0
    precision_requested: int = 0
    total_cfg_intersection_seconds: float = 0.0
    next_file_index: int = 0
    file_indices: dict[tuple[int, int], int] = field(default_factory=dict)
    worker_summaries: dict[int, Mapping[str, object]] = field(
        default_factory=dict
    )

    def consume(self, worker: int, record: dict[str, object]) -> None:
        event = record.get("event")
        if event == "statement":
            self.consume_statement(worker, record)
        elif event == "summary":
            self.worker_summaries[worker] = record

    def consume_statement(self, worker: int, record: dict[str, object]) -> None:
        local_file_index = integer_field(record, "file_index")
        file_key = (worker, local_file_index)
        if file_key not in self.file_indices:
            self.next_file_index += 1
            self.file_indices[file_key] = self.next_file_index

        self.evaluated += 1
        self.recognized += int(bool(record["recognized"]))
        self.precision_accepted += integer_field(record, "sample_accepted")
        self.precision_checked += integer_field(record, "sample_checked")
        self.precision_requested += integer_field(record, "sample_requested")
        self.total_cfg_intersection_seconds += number_field(
            record, "cfg_intersection_seconds"
        )

        record["worker"] = worker
        record["worker_file_index"] = local_file_index
        record["file_index"] = self.file_indices[file_key]
        record["index"] = self.evaluated
        record["running_recall"] = evaluator.percentage(
            self.recognized / self.evaluated
        )
        precision = (
            float("nan")
            if self.precision_checked == 0
            else self.precision_accepted / self.precision_checked
        )
        coverage = (
            float("nan")
            if self.precision_requested == 0
            else self.precision_checked / self.precision_requested
        )
        record["running_precision"] = evaluator.percentage(precision)
        record["running_precision_coverage"] = evaluator.percentage(coverage)
        evaluator.emit_record(record, json_lines=self.json_lines)

    def summed_int(self, name: str) -> int:
        total = 0
        for summary in self.worker_summaries.values():
            value = summary.get(name, 0)
            if isinstance(value, int):
                total += value
        return total

    def summed_counter(self, name: str) -> dict[str, int]:
        combined: Counter[str] = Counter()
        for summary in self.worker_summaries.values():
            value = summary.get(name, {})
            if isinstance(value, Mapping):
                combined.update(
                    {
                        str(key): count
                        for key, count in value.items()
                        if isinstance(count, int)
                    }
                )
        return dict(sorted(combined.items()))

    def summary(self, return_codes: Mapping[int, int]) -> dict[str, object]:
        recall = (
            float("nan") if self.evaluated == 0 else self.recognized / self.evaluated
        )
        precision = (
            float("nan")
            if self.precision_checked == 0
            else self.precision_accepted / self.precision_checked
        )
        coverage = (
            float("nan")
            if self.precision_requested == 0
            else self.precision_checked / self.precision_requested
        )
        average_seconds = (
            0.0
            if self.evaluated == 0
            else self.total_cfg_intersection_seconds / self.evaluated
        )
        return {
            "event": "summary",
            "parallel": True,
            "dataset": self.dataset,
            "split": self.split,
            "source": self.source,
            "workers": self.workers,
            "worker_return_codes": {
                str(index): code for index, code in sorted(return_codes.items())
            },
            "population": (
                "all eligible statements in statically sharded source-order "
                f"ty-clean {self.dataset}"
                f"{' ' + self.split if self.split is not None else ''} files"
            ),
            "files_requested": self.files_requested,
            "files_evaluated": self.summed_int("files_evaluated"),
            "statements_evaluated": self.evaluated,
            "evaluated": self.evaluated,
            "recognized": self.recognized,
            "recall": evaluator.percentage(recall),
            "precision_accepted": self.precision_accepted,
            "precision_checked": self.precision_checked,
            "precision_requested": self.precision_requested,
            "precision_target": self.max_samples,
            "precision": evaluator.percentage(precision),
            "precision_coverage": evaluator.percentage(coverage),
            "sampleable_statements": self.summed_int("sampleable_statements"),
            "sampler_failures": self.summed_int("sampler_failures"),
            "failure_reasons": self.summed_counter("failure_reasons"),
            "diagnostic_codes": self.summed_counter("diagnostic_codes"),
            "sampled_lengths": self.summed_counter("sampled_lengths"),
            "sampled_length_offsets": self.summed_counter(
                "sampled_length_offsets"
            ),
            "average_cfg_intersection_seconds": average_seconds,
            "funnel": self.summed_counter("funnel"),
        }


def parse_parallel_arguments(arguments: Sequence[str]) -> tuple[argparse.Namespace, argparse.Namespace]:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument(
        "-j",
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="worker processes (default: all detected CPUs)",
    )
    parallel, remaining = wrapper.parse_known_args(arguments)
    if parallel.workers < 1:
        wrapper.error("--workers must be at least 1")
    if "-h" in remaining or "--help" in remaining:
        print("Parallel-only option: -j/--workers N (default: all detected CPUs)\n")
    serial = evaluator.parse_arguments(remaining)
    return parallel, serial


def run_self_tests() -> None:
    assert split_quota(10, 3) == [4, 3, 3]
    assert split_quota(2, 2) == [1, 1]
    assert split_quota(0, 1) == [0]

    parallel, serial = parse_parallel_arguments(["-j", "2", "--files", "0"])
    defaults = evaluator.evaluation_options(serial)
    assert parallel.workers == 2
    assert defaults.dataset == "apps"
    assert defaults.split == "test"
    assert defaults.statement_timeout == 60.0
    assert defaults.source == evaluator.default_dataset_source("apps", "test")

    _parallel, serial = parse_parallel_arguments(
        [
            "--dataset",
            "codenet",
            "--split",
            "train",
            "--files",
            "0",
            "--statement-timeout",
            "0.25",
        ]
    )
    codenet = evaluator.evaluation_options(serial)
    assert codenet.dataset == "codenet"
    assert codenet.split == "train"
    assert codenet.statement_timeout == 0.25
    assert codenet.source == evaluator.default_dataset_source(
        "codenet", "train"
    )
    evaluator.run_self_tests()
    print("parallel wrapper self-test passed")


def _main(arguments: Sequence[str] | None = None) -> int:
    parallel, serial = parse_parallel_arguments(
        sys.argv[1:] if arguments is None else arguments
    )
    if serial.self_test:
        run_self_tests()
        return 0

    worker_count = min(parallel.workers, max(serial.files, 1))
    if serial.max_samples is not None:
        worker_count = min(worker_count, serial.max_samples)
    file_quotas = split_quota(serial.files, worker_count)
    sample_quotas: Sequence[int | None]
    if serial.max_samples is None:
        sample_quotas = [None] * worker_count
    else:
        sample_quotas = split_quota(serial.max_samples, worker_count)
    base_options = evaluator.evaluation_options(serial)
    source_path = evaluator.resolved_dataset_source(
        base_options.dataset,
        base_options.source,
        base_options.split,
    )
    display_split = (
        base_options.split if base_options.dataset == "apps" else None
    )
    source_stat = source_path.stat()
    version = evaluator.ty_version(base_options.ty)
    aggregate = Aggregate(
        json_lines=serial.jsonl,
        dataset=base_options.dataset,
        split=display_split,
        source=str(source_path),
        files_requested=serial.files,
        max_samples=serial.max_samples,
        workers=worker_count,
    )
    start = {
        "event": "start",
        "parallel": True,
        "workers": worker_count,
        "available_cpus": os.cpu_count() or 1,
        "dataset": base_options.dataset,
        "split": display_split,
        "source": str(source_path),
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "files": serial.files,
        "precision_samples": serial.precision_samples,
        "max_samples": serial.max_samples,
        "seed": base_options.seed,
        "ty": version,
        "python": sys.version,
        "platform": platform.platform(),
        "library_directory": str(base_options.library_directory),
        "allow_ignores": base_options.allow_ignores,
        "sample_rank_interval": f"[0, {evaluator.SAMPLE_RANK_LIMIT})",
        "sample_order": "global token shortlex DFA bijection",
        "max_dfa_states": base_options.max_dfa_states,
        "statement_timeout": base_options.statement_timeout,
        "surface_fragment": evaluator.surface_fragment_metadata(),
        "builder": {
            "max_call_arity": "floor((ground_truth_tokens-root_tokens)/2)",
            "max_dynamic_composition_depth": (
                base_options.builder.max_dynamic_composition_depth
            ),
            "max_tokens": "ground_truth_tokens+2",
            "max_layouts_per_signature": (
                base_options.builder.max_layouts_per_signature
            ),
            "member_depth": base_options.builder.member_depth,
            "max_receiver_types": base_options.builder.max_receiver_types,
            "max_module_members": base_options.builder.max_module_members,
            "max_output_producers": base_options.builder.max_output_producers,
        },
        "partition": "source-index modulo workers",
        "shard_count": worker_count,
    }
    if serial.jsonl:
        print(json.dumps(start, sort_keys=True), flush=True)
    else:
        print(
            f"parallel workers={worker_count}/{os.cpu_count() or 1}; "
            f"dataset={base_options.dataset}; "
            f"split={display_split or 'n/a'}; "
            f"source={source_path}; target_files={serial.files}; "
            f"statement_timeout={base_options.statement_timeout:g}s; "
            "partition=source-index-modulo",
            flush=True,
        )

    return_codes: dict[int, int] = {}
    futures: list[Future[tuple[int, int]]] = []
    malformed_output = False
    with Manager() as manager:
        output = manager.Queue()
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            for worker in range(worker_count):
                options = replace(
                    base_options,
                    files=file_quotas[worker],
                    max_samples=sample_quotas[worker],
                    json_lines=True,
                    shard_count=worker_count,
                    shard_index=worker,
                )
                futures.append(pool.submit(run_worker, options, worker, output))

            completed_streams: set[int] = set()
            while len(completed_streams) < worker_count:
                try:
                    worker, stream, line = output.get(timeout=0.25)
                except queue.Empty:
                    if all(future.done() for future in futures):
                        break
                    continue
                if stream == "done":
                    completed_streams.add(worker)
                elif stream == "stderr":
                    print(f"[worker {worker}] {line}", file=sys.stderr, flush=True)
                else:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        malformed_output = True
                        print(
                            f"[worker {worker}] malformed output: {line}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if isinstance(record, dict):
                        aggregate.consume(worker, record)

            for future in futures:
                try:
                    worker, return_code = future.result()
                except Exception:
                    malformed_output = True
                    traceback.print_exc()
                    continue
                return_codes[worker] = return_code

    summary = aggregate.summary(return_codes)
    if serial.jsonl:
        print(json.dumps(summary, sort_keys=True), flush=True)
    else:
        print(
            f"summary: files={summary['files_evaluated']}/{serial.files} "
            f"statements={aggregate.evaluated} "
            f"recall={summary['recall']} "
            f"({aggregate.recognized}/{aggregate.evaluated}) "
            f"precision={summary['precision']} "
            f"({aggregate.precision_accepted}/{aggregate.precision_checked}) "
            f"coverage={summary['precision_coverage']} "
            f"avg_cfg_intersection="
            f"{summary['average_cfg_intersection_seconds']:.3f}s",
            flush=True,
        )
        print(f"funnel: {json.dumps(summary['funnel'], sort_keys=True)}", flush=True)

    if malformed_output or len(return_codes) != worker_count:
        return 1
    if any(code == 1 for code in return_codes.values()):
        return 1
    if any(code != 0 for code in return_codes.values()):
        return 2
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        return _main(arguments)
    except (evaluator.EvaluationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
