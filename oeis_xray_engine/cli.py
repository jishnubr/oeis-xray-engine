"""Command-line interface for ingesting and running OEIS X-Ray sweeps.

The CLI deliberately keeps mathematical configuration explicit and bounded:
``plan`` creates versioned inverse-Euler and exact-prefix fingerprint tasks,
while the coordinator remains the sole owner of catalog writes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence, cast
from uuid import uuid4

from .checkpoint import CheckpointStore
from .errors import XRayError
from .exporter import write_gems
from .indexing.hash_index import FingerprintProfile
from .indexing.sequence_index import SequenceIndex
from .models import TransformSpec
from .pipeline import PipelineCoordinator, PipelineSpec, PipelineStage, RunReport
from .storage.catalog import Catalog
from .storage.term_store import TermStore
from .transformers.ramanujan.search import CFSearchSpec

_LOGGER = logging.getLogger("oeis_xray_engine.cli")
_DEFAULT_STAGES = ("transform", "fingerprint", "match")
_VALID_STAGES = frozenset(("transform", "fingerprint", "match"))
_DEFAULT_CATALOG = Path("oeis_xray_catalog.sqlite")
_DEFAULT_TERM_STORE = Path("oeis_xray_terms")


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 1) - 1)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_stages(values: Sequence[str]) -> tuple[PipelineStage, ...]:
    stages: list[str] = []
    for value in values:
        for stage in value.split(","):
            normalized = stage.strip().lower()
            if normalized not in _VALID_STAGES:
                raise ValueError(
                    f"unsupported stage {stage!r}; choose from "
                    f"{', '.join(sorted(_VALID_STAGES))}"
                )
            if normalized not in stages:
                stages.append(normalized)
    if not stages:
        raise ValueError("at least one stage is required")
    ordered = tuple(
        cast(PipelineStage, stage) for stage in _DEFAULT_STAGES if stage in stages
    )
    if "match" in ordered and "fingerprint" not in ordered:
        raise ValueError("'match' requires the 'fingerprint' stage")
    return ordered


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oeis-xray",
        description="Ingest OEIS sequence files and run bounded X-Ray sweeps.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=_DEFAULT_CATALOG,
        help=f"SQLite catalog path (default: {_DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--term-store",
        type=Path,
        default=_DEFAULT_TERM_STORE,
        help=f"content-addressed term store directory (default: {_DEFAULT_TERM_STORE})",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO"),
        default="INFO",
        help="logging verbosity written to stdout (default: INFO)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="ingest a local OEIS seq directory")
    _add_command_options(ingest)
    ingest.add_argument(
        "--source",
        required=True,
        type=Path,
        help="path to the local oeisdata/seq directory",
    )
    ingest.add_argument(
        "--snapshot",
        required=True,
        help="immutable corpus snapshot identifier, e.g. pilot-v1",
    )

    plan = commands.add_parser("plan", help="create a resumable pipeline run")
    _add_command_options(plan)
    plan.add_argument("--snapshot", required=True, help="ready snapshot identifier")
    plan.add_argument(
        "--max-terms",
        type=_positive_integer,
        default=32,
        help="bounded transform/fingerprint term count (default: 32)",
    )
    plan.add_argument(
        "--sweep",
        choices=("inverse-euler", "denumerant", "ramanujan"),
        default="inverse-euler",
        help=(
            "select exact inverse-Euler extraction, denumerant extraction, "
            "or a bounded numerical Ramanujan CF search"
        ),
    )
    plan.add_argument(
        "--ramanujan-spec",
        type=Path,
        help="JSON file containing one validated Ramanujan CFSearchSpec",
    )
    plan.add_argument(
        "--stages",
        nargs="+",
        default=None,
        metavar="STAGE",
        help=(
            "one or more of transform, fingerprint, match; may also be comma "
            "separated (defaults depend on --sweep)"
        ),
    )
    plan.add_argument(
        "--run-id",
        help="run identifier (default: generated UUID)",
    )

    run = commands.add_parser("run", help="execute pending planned tasks")
    _add_command_options(run)
    run.add_argument("--run-id", required=True, help="planned run identifier")
    run.add_argument(
        "--workers",
        type=_positive_integer,
        default=_default_workers(),
        help=f"worker process count (default: {_default_workers()})",
    )

    resume = commands.add_parser("resume", help="recover and continue a halted run")
    _add_command_options(resume)
    resume.add_argument("--run-id", required=True, help="paused run identifier")
    resume.add_argument(
        "--workers",
        type=_positive_integer,
        default=_default_workers(),
        help=f"worker process count (default: {_default_workers()})",
    )

    report = commands.add_parser("report", help="summarize a pipeline run")
    _add_command_options(report)
    report.add_argument("--run-id", required=True, help="run identifier")

    gems = commands.add_parser(
        "export-gems",
        help="export E1-or-higher relation claims as JSON or Markdown",
    )
    _add_command_options(gems)
    gems.add_argument("--run-id", help="limit the export to one pipeline run")
    gems.add_argument(
        "--format",
        dest="export_format",
        choices=("json", "markdown"),
        default="markdown",
        help="export format (default: markdown)",
    )
    return parser


def _add_command_options(parser: argparse.ArgumentParser) -> None:
    """Allow storage and logging options both before and after subcommands."""

    parser.add_argument(
        "--catalog",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--term-store",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO"),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def _make_pipeline_spec(
    run_id: str,
    snapshot_id: str,
    max_terms: int,
    stages: tuple[PipelineStage, ...],
    *,
    sweep: str = "inverse-euler",
    ramanujan_search_specs: tuple[CFSearchSpec, ...] = (),
) -> PipelineSpec:
    transform_specs: tuple[TransformSpec, ...] = ()
    if "transform" in stages:
        if sweep == "ramanujan":
            raise ValueError(
                "the ramanujan sweep is a numerical candidate search, "
                "not a sequence transform"
            )
        if sweep == "denumerant":
            if max_terms < 15:
                raise ValueError(
                    "the denumerant sweep requires --max-terms of at least 15"
                )
            transform_specs = (
                TransformSpec(
                    name="denumerant",
                    version="1",
                    parameters={
                        "input_term_count": max_terms,
                        "term_cap": min(max_terms, 100),
                        "confidence_window": 5,
                    },
                    input_domain="finite_integer_sequence_pure_bosonic_euler_product",
                    output_bound=100_000,
                ),
            )
        else:
            transform_specs = (
                TransformSpec(
                    name="inverse_euler",
                    version="1",
                    parameters={
                        "input_start_index": 0,
                        "input_term_count": max_terms + 1,
                        "output_start_index": 1,
                        "output_term_count": max_terms,
                    },
                    input_domain="formal_series_constant_term_one",
                    output_bound=max_terms,
                ),
            )
    profiles: tuple[FingerprintProfile, ...] = ()
    if "fingerprint" in stages:
        profiles = (
            FingerprintProfile(
                name="exact-prefix",
                version="1",
                mode="exact",
                term_count=max_terms,
            ),
        )
    return PipelineSpec(
        run_id=run_id,
        snapshot_id=snapshot_id,
        stages=stages,
        transform_specs=transform_specs,
        fingerprint_profiles=profiles,
        ramanujan_search_specs=ramanujan_search_specs,
    )


def _default_stages_for_sweep(sweep: str) -> tuple[PipelineStage, ...]:
    if sweep == "denumerant":
        return ("transform",)
    if sweep == "ramanujan":
        return ("fingerprint", "match")
    return _DEFAULT_STAGES


def _load_ramanujan_spec(path: Path) -> CFSearchSpec:
    """Load and validate one local JSON specification for numerical search."""

    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
        return CFSearchSpec.from_json_value(value)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Ramanujan search specification: {exc}") from exc


def _format_report(run_id: str, run_state: str, catalog: Catalog) -> str:
    tasks = catalog.fetch_one(
        """
        SELECT
          SUM(state = 'complete') AS completed,
          SUM(state IN ('retryable_error','fatal_error')) AS errors
        FROM task WHERE run_id = ?
        """,
        (run_id,),
    )
    claims = catalog.fetch_one(
        """
        SELECT
          SUM(evidence_level = 'E0') AS e0,
          SUM(evidence_level = 'E1') AS e1,
          SUM(evidence_level = 'E2') AS e2,
          SUM(evidence_level = 'E3') AS e3,
          SUM(evidence_level = 'E4') AS e4
        FROM relation_claim WHERE created_by_run = ?
        """,
        (run_id,),
    )
    counts = {
        "Tasks completed": tasks["completed"] or 0,
        "E0 candidates found": claims["e0"] or 0,
        "E1 matches verified": claims["e1"] or 0,
        "E2 reconstructions verified": claims["e2"] or 0,
        "E3 symbolic identities verified": claims["e3"] or 0,
        "E4 human reviews recorded": claims["e4"] or 0,
        "Errors encountered": tasks["errors"] or 0,
    }
    label_width = max(len(label) for label in counts)
    value_width = max(len(str(value)) for value in counts.values())
    inner_width = max(
        len(f" Run: {run_id}"),
        len(f" State: {run_state}"),
        max(label_width + value_width + 5, 20),
    )
    border = "+" + "-" * (inner_width + 2) + "+"
    lines = [border, f"| {'OEIS X-Ray Run Report'.ljust(inner_width)} |", border]
    lines.append(f"| {('Run: ' + run_id).ljust(inner_width)} |")
    lines.append(f"| {('State: ' + run_state).ljust(inner_width)} |")
    lines.append(border)
    for label, value in counts.items():
        row = f" {label.ljust(label_width)} : {str(value).rjust(value_width)}"
        lines.append(f"| {row.ljust(inner_width)} |")
    lines.append(border)
    return "\n".join(lines)


def _print_report(run_id: str, catalog: Catalog) -> None:
    run = CheckpointStore(catalog).get_run(run_id)
    print(_format_report(run_id, str(run["state"]), catalog))


def _pause_interrupted_run(coordinator: PipelineCoordinator, run_id: str) -> None:
    """Pause a run and expire this process's leases after its pool has shut down."""

    owner = f"{socket.gethostname()}:{os.getpid()}"
    now = datetime.now(timezone.utc)
    expired_at = (now - timedelta(seconds=1)).isoformat()
    with coordinator.catalog.transaction():
        coordinator.catalog.execute(
            """
            UPDATE task SET lease_until = ?, updated_at = ?
            WHERE run_id = ? AND state = 'leased' AND lease_owner = ?
            """,
            (expired_at, now.isoformat(), run_id, owner),
        )
    run = coordinator.checkpoints.get_run(run_id)
    manifest = run["manifest"]
    spec = manifest.get("pipeline_spec", {})
    max_attempts = int(spec.get("max_attempts", 3))
    coordinator.checkpoints.recover_expired_leases(
        run_id,
        now=now.isoformat(),
        max_attempts=max_attempts,
    )
    coordinator.checkpoints.set_run_state(run_id, "paused")


def _run_pipeline(
    coordinator: PipelineCoordinator,
    run_id: str,
    workers: int,
    *,
    resume: bool,
) -> RunReport:
    if resume:
        run = coordinator.checkpoints.get_run(run_id)
        spec_data = run["manifest"].get("pipeline_spec", {})
        max_attempts = int(spec_data.get("max_attempts", 3))
        recovered = coordinator.checkpoints.recover_expired_leases(
            run_id,
            max_attempts=max_attempts,
        )
        _LOGGER.info("Recovered %d expired task lease(s).", recovered)
    _LOGGER.info(
        "%s pipeline run %s using %d worker(s).",
        "Resuming" if resume else "Starting",
        run_id,
        workers,
    )
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_monitor_run,
        args=(coordinator, run_id, stop_monitor),
        name=f"xray-progress-{run_id}",
        daemon=True,
    )
    monitor.start()
    try:
        try:
            result = (
                coordinator.resume(run_id, workers=workers)
                if resume
                else coordinator.execute(run_id, workers=workers)
            )
        except KeyboardInterrupt:
            _pause_interrupted_run(coordinator, run_id)
            _LOGGER.warning("Run %s paused cleanly after interruption.", run_id)
            raise
    finally:
        stop_monitor.set()
        monitor.join()
    _LOGGER.info(
        "Run %s finished in state %s (%d newly completed task(s)).",
        run_id,
        result.state,
        result.newly_completed,
    )
    return result


def _monitor_run(
    coordinator: PipelineCoordinator,
    run_id: str,
    stop: threading.Event,
) -> None:
    """Log changed task totals while the coordinator executes worker batches."""

    previous: tuple[int, int, int, int, int] | None = None
    while not stop.wait(5.0):
        try:
            counts = coordinator.checkpoints.task_counts(run_id)
        except XRayError:
            _LOGGER.exception("Unable to read progress for run %s.", run_id)
            return
        current = (
            counts.pending,
            counts.leased,
            counts.complete,
            counts.retryable_error,
            counts.fatal_error,
        )
        if current != previous:
            _LOGGER.info(
                "Progress %s: %d complete, %d leased, %d pending, "
                "%d retryable error(s), %d fatal error(s).",
                run_id,
                counts.complete,
                counts.leased,
                counts.pending,
                counts.retryable_error,
                counts.fatal_error,
            )
            previous = current


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    try:
        with Catalog(args.catalog) as catalog:
            if args.command == "export-gems":
                write_gems(
                    catalog,
                    sys.stdout,
                    run_id=args.run_id,
                    format=args.export_format,
                )
                return 0

            term_store = TermStore(args.term_store, catalog)
            if args.command == "ingest":
                _LOGGER.info(
                    "Ingesting source directory %s into snapshot %s.",
                    args.source,
                    args.snapshot,
                )
                report = SequenceIndex(catalog, term_store).ingest_directory(
                    args.source,
                    snapshot_id=args.snapshot,
                    source_revision="local-source",
                    source_name=str(args.source.expanduser().resolve()),
                )
                _LOGGER.info(
                    "Ingestion complete: %d records, %d parsed, %d partial, "
                    "%d invalid, %d terms, %d chunks.",
                    report.records_seen,
                    report.parsed,
                    report.partial,
                    report.invalid,
                    report.terms_ingested,
                    report.chunks_written,
                )
                _LOGGER.debug(
                    "Snapshot %s source digest: %s; diagnostics digest: %s",
                    report.snapshot_id,
                    report.source_digest,
                    report.diagnostics_digest,
                )
                return 0

            coordinator = PipelineCoordinator(catalog, term_store)
            if args.command == "plan":
                if args.sweep == "ramanujan":
                    if args.ramanujan_spec is None:
                        raise ValueError(
                            "--sweep ramanujan requires --ramanujan-spec"
                        )
                    ramanujan_search_specs = (
                        _load_ramanujan_spec(args.ramanujan_spec),
                    )
                else:
                    if args.ramanujan_spec is not None:
                        raise ValueError(
                            "--ramanujan-spec is only valid with "
                            "--sweep ramanujan"
                        )
                    ramanujan_search_specs = ()
                requested_stages = (
                    args.stages
                    if args.stages is not None
                    else _default_stages_for_sweep(args.sweep)
                )
                try:
                    stages = _parse_stages(requested_stages)
                except ValueError as exc:
                    parser.error(str(exc))
                run_id = args.run_id or f"xray-{uuid4().hex}"
                spec = _make_pipeline_spec(
                    run_id,
                    args.snapshot,
                    args.max_terms,
                    stages,
                    sweep=args.sweep,
                    ramanujan_search_specs=ramanujan_search_specs,
                )
                planned = coordinator.plan(spec)
                _LOGGER.info("Planned run %s for snapshot %s.", planned, args.snapshot)
                _LOGGER.info(
                    "Sweep: %s; stages: %s; maximum terms: %d.",
                    args.sweep,
                    ", ".join(stages),
                    args.max_terms,
                )
                if args.sweep == "ramanujan":
                    _LOGGER.warning(
                        "Ramanujan search results are numerical candidates, "
                        "not exact relation claims or proofs."
                    )
                print(planned)
                return 0

            if args.command in ("run", "resume"):
                try:
                    result = _run_pipeline(
                        coordinator,
                        args.run_id,
                        args.workers,
                        resume=args.command == "resume",
                    )
                except KeyboardInterrupt:
                    return 130
                _LOGGER.info(
                    "Completed tasks: %d; retryable errors: %d; fatal errors: %d.",
                    result.counts.complete,
                    result.counts.retryable_error,
                    result.counts.fatal_error,
                )
                return 0 if result.state == "complete" else 1

            if args.command == "report":
                _print_report(args.run_id, catalog)
                return 0

            parser.error(f"unsupported command: {args.command}")
    except (XRayError, OSError, ValueError) as exc:
        _LOGGER.error("%s", exc)
        _LOGGER.debug("Command failed.", exc_info=True)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
