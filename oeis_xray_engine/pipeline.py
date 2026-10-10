"""Resumable, bounded, single-writer parallel pipeline coordinator."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from queue import Empty
from typing import TYPE_CHECKING, Generator, Iterable, Literal, Mapping, Protocol, Sequence

from .checkpoint import (
    CheckpointStore,
    TaskCounts,
    TaskLease,
    TaskManifest,
    canonical_json,
    digest_json,
)
from .bfile_validation import (
    BFileValidationKind,
    BFileFetcher,
    BFileFetchResult,
    BFileValidationResult,
    BFileValidationService,
)
from .errors import (
    CatalogError,
    DomainError,
    InsufficientTermsError,
    ValidationError,
    WorkerError,
)
from .indexing.hash_index import (
    Candidate,
    Fingerprint,
    FingerprintProfile,
    HashIndex,
)
from .indexing.sequence_index import Partition, SequenceIndex
from .matching.collision_detector import CollisionDetector, RelationSpec
from .models import ChunkRef, RelationClaim, SequenceRef, TermSeries, TransformSpec
from .storage.catalog import Catalog
from .storage.term_store import TermStore
from .transformers import (
    TransformError,
    TransformerExecutionContext,
    TransformerRegistry,
    default_transformer_registry,
)

if TYPE_CHECKING:
    from .transformers.ramanujan.search import CFSearchSpec

PipelineStage = Literal["transform", "fingerprint", "match"]
_LOGGER = logging.getLogger(__name__)
_STAGE_ORDER = {"transform": 0, "fingerprint": 1, "match": 2}
_RESULT_COMMIT_BATCH_SIZE = 200
_MAX_MATCH_BUCKET_SIZE = 500
_WORKER_HEARTBEAT_SECONDS = 60
_COORDINATOR_POLL_SECONDS = 5
_LEASE_REAPER_INTERVAL_SECONDS = 30


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Immutable configuration for one snapshot-bound pipeline run."""

    run_id: str
    snapshot_id: str
    stages: tuple[PipelineStage, ...]
    transform_specs: tuple[TransformSpec, ...] = ()
    fingerprint_profiles: tuple[FingerprintProfile, ...] = ()
    ramanujan_search_specs: tuple[CFSearchSpec, ...] = ()
    page_size: int = 500
    lease_seconds: int = 900
    max_attempts: int = 3
    code_revision: str = "working-tree"

    def __post_init__(self) -> None:
        if not self.run_id or not self.snapshot_id:
            raise ValueError("run_id and snapshot_id must be non-empty")
        if not self.stages:
            raise ValueError("at least one pipeline stage is required")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("pipeline stages must not be repeated")
        for stage in self.stages:
            if stage not in _STAGE_ORDER:
                raise ValueError(f"unsupported pipeline stage: {stage!r}")
        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        if type(self.lease_seconds) is not int or self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if "transform" in self.stages and not self.transform_specs:
            raise ValueError("transform stage requires at least one TransformSpec")
        if "fingerprint" in self.stages and not self.fingerprint_profiles:
            raise ValueError("fingerprint stage requires at least one profile")
        if "match" in self.stages and "fingerprint" not in self.stages:
            raise ValueError("match stage requires fingerprint stage")
        if self.ramanujan_search_specs and "match" not in self.stages:
            raise ValueError("Ramanujan candidate searches require the match stage")
        if self.ramanujan_search_specs:
            from .transformers.ramanujan.search import CFSearchSpec

            if any(
                not isinstance(search_spec, CFSearchSpec)
                for search_spec in self.ramanujan_search_specs
            ):
                raise TypeError(
                    "ramanujan_search_specs must contain CFSearchSpec values"
                )
        if any(profile.mode != "exact" for profile in self.fingerprint_profiles) and (
            "match" in self.stages
        ):
            raise ValueError(
                "match stage currently verifies only exact fingerprint profiles"
            )


@dataclass(frozen=True, slots=True)
class RunReport:
    """Stable report for a planned, paused, completed, or failed run."""

    run_id: str
    state: str
    counts: TaskCounts
    result_manifest: str
    newly_completed: int


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    task_id: str
    task_key: str
    output_digest: str
    payload_json: str


class _HeartbeatQueue(Protocol):
    def get(self, *, timeout: float) -> tuple[str, str]: ...

    def put(self, message: tuple[str, str]) -> None: ...

    def close(self) -> None: ...

    def join_thread(self) -> None: ...

_WORKER_HEARTBEAT_QUEUE: _HeartbeatQueue | None = None
_WORKER_HEARTBEAT_LOCK = threading.Lock()
_WORKER_HEARTBEAT_STATE: tuple[str, str] | None = None
_WORKER_HEARTBEAT_STOP: threading.Event | None = None
_WORKER_HEARTBEAT_FAILURES: list[Exception] = []


@contextmanager
def _close_heartbeat_queue(
    heartbeat_queue: _HeartbeatQueue,
) -> Generator[None, None, None]:
    try:
        yield
    finally:
        heartbeat_queue.close()
        heartbeat_queue.join_thread()


def _initialize_worker_process(
    heartbeat_queue: _HeartbeatQueue,
    lease_seconds: int,
) -> None:
    global _WORKER_HEARTBEAT_QUEUE, _WORKER_HEARTBEAT_STOP
    _WORKER_HEARTBEAT_QUEUE = heartbeat_queue
    _WORKER_HEARTBEAT_STOP = threading.Event()
    heartbeat_interval = min(
        _WORKER_HEARTBEAT_SECONDS,
        max(1, lease_seconds // 3),
    )
    thread = threading.Thread(
        target=_worker_heartbeat_loop,
        args=(heartbeat_queue, heartbeat_interval, _WORKER_HEARTBEAT_STOP),
        name="xray-worker-heartbeat-sender",
        daemon=True,
    )
    thread.start()


def _worker_heartbeat_loop(
    heartbeat_queue: _HeartbeatQueue,
    interval: int,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(interval):
        with _WORKER_HEARTBEAT_LOCK:
            heartbeat = _WORKER_HEARTBEAT_STATE
        if heartbeat is None:
            continue
        try:
            heartbeat_queue.put(heartbeat)
        except Exception as exc:
            _WORKER_HEARTBEAT_FAILURES.append(exc)
            stop_event.set()
            return


class PipelineCoordinator:
    """Execute deterministic task manifests with coordinator-only DB writes.

    Workers are separate processes. Each receives only JSON task input and
    filesystem paths, opens its own read-only SQLite connection for chunk
    manifests, and writes only immutable content-addressed transform files.
    The coordinator alone leases tasks and commits all catalog artifacts and
    task completion in one SQLite transaction.
    """

    def __init__(
        self,
        catalog: Catalog,
        term_store: TermStore,
        *,
        task_batch_size: int = 500,
        transformer_registry: TransformerRegistry | None = None,
    ) -> None:
        if not isinstance(catalog, Catalog) or catalog.read_only:
            raise TypeError("coordinator requires a writable Catalog")
        if not isinstance(term_store, TermStore) or term_store.catalog is not catalog:
            raise ValueError("coordinator requires a TermStore using its Catalog")
        if type(task_batch_size) is not int or task_batch_size <= 0:
            raise ValueError("task_batch_size must be positive")
        self.catalog = catalog
        self.term_store = term_store
        self.checkpoints = CheckpointStore(catalog)
        self.task_batch_size = task_batch_size
        self.sequence_index = SequenceIndex(catalog, term_store)
        if transformer_registry is not None and not isinstance(
            transformer_registry, TransformerRegistry
        ):
            raise TypeError("transformer_registry must be a TransformerRegistry")
        self.transformer_registry = (
            transformer_registry or default_transformer_registry()
        )

    def plan(self, spec: PipelineSpec) -> str:
        """Create a run and insert bounded, deterministic per-sequence tasks."""

        if not isinstance(spec, PipelineSpec):
            raise TypeError("spec must be a PipelineSpec")
        for transform in spec.transform_specs:
            self.transformer_registry.worker_reference(
                transform.name,
                transform.version,
            )
        run_manifest = {
            "pipeline_spec": _spec_to_json(spec),
            "transformer_references": {
                f"{transform.name}@{transform.version}":
                    self.transformer_registry.worker_reference(
                        transform.name,
                        transform.version,
                    )
                for transform in spec.transform_specs
            },
            "match_tasks_planned": False,
            "derived_fingerprints_planned": not (
                "transform" in spec.stages and "fingerprint" in spec.stages
            ),
            "manifest_version": 1,
        }
        config = {
            "stages": list(spec.stages),
            "transform_specs": [_transform_to_json(x) for x in spec.transform_specs],
            "transformer_references": {
                f"{transform.name}@{transform.version}":
                    self.transformer_registry.worker_reference(
                        transform.name,
                        transform.version,
                    )
                for transform in spec.transform_specs
            },
            "fingerprint_profiles": [
                _profile_to_json(x) for x in spec.fingerprint_profiles
            ],
            "ramanujan_search_specs": [
                search_spec.to_json_value()
                for search_spec in spec.ramanujan_search_specs
            ],
            "page_size": spec.page_size,
            "lease_seconds": spec.lease_seconds,
            "max_attempts": spec.max_attempts,
        }
        self.checkpoints.create_run(
            run_id=spec.run_id,
            snapshot_id=spec.snapshot_id,
            run_kind="xray-pipeline",
            config=config,
            code_revision=spec.code_revision,
            manifest=run_manifest,
        )
        self.checkpoints.add_tasks(
            self._sequence_tasks(spec),
            batch_size=self.task_batch_size,
        )
        return spec.run_id

    def execute(
        self,
        run_id: str,
        *,
        workers: int = 1,
        max_completed_tasks: int | None = None,
    ) -> RunReport:
        """Execute pending work or pause after a bounded number of completions.

        ``max_completed_tasks`` is a deterministic interruption hook suitable
        for controlled maintenance windows and recovery testing. In-flight
        workers in the current bounded batch are allowed to finish before the
        coordinator pauses.
        """

        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer")
        if max_completed_tasks is not None and (
            type(max_completed_tasks) is not int or max_completed_tasks < 0
        ):
            raise ValueError("max_completed_tasks must be non-negative or None")

        run = self.checkpoints.get_run(run_id)
        manifest = dict(run["manifest"])
        spec = _spec_from_json(manifest["pipeline_spec"])
        if run["snapshot_id"] != spec.snapshot_id:
            raise CatalogError("run snapshot and pipeline spec disagree")
        if run["state"] in {"complete", "cancelled"}:
            return self._report(run_id, run["state"], 0)

        self.checkpoints.set_run_state(run_id, "running")
        owner = f"{socket.gethostname()}:{os.getpid()}"
        newly_completed = 0
        completed_this_call = 0

        heartbeat_queue = get_context().Queue()
        with (
            _close_heartbeat_queue(heartbeat_queue),
            _heartbeat_listener(self, heartbeat_queue, spec.lease_seconds),
            ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker_process,
                initargs=(heartbeat_queue, spec.lease_seconds),
            ) as executor,
        ):
            while True:
                remaining_budget = None
                if max_completed_tasks is not None:
                    remaining_budget = max_completed_tasks - completed_this_call
                    if remaining_budget <= 0:
                        self.checkpoints.set_run_state(run_id, "paused")
                        return self._report(run_id, "paused", newly_completed)

                request_count = self.task_batch_size
                if remaining_budget is not None:
                    request_count = min(request_count, remaining_budget)
                leases = self.checkpoints.lease_tasks(
                    run_id,
                    owner,
                    limit=request_count,
                    lease_seconds=spec.lease_seconds,
                    max_attempts=spec.max_attempts,
                )
                if not leases:
                    counts = self.checkpoints.task_counts(run_id)
                    if counts.pending or counts.retryable_error or counts.leased:
                        # Pending leases may have expired between batches. Re-run
                        # lease recovery with the same bounded loop.
                        recovered = self.checkpoints.recover_expired_leases(
                            run_id, max_attempts=spec.max_attempts
                        )
                        if recovered:
                            continue
                        self.checkpoints.set_run_state(run_id, "paused")
                        return self._report(run_id, "paused", newly_completed)

                    current_manifest = dict(self.checkpoints.get_run(run_id)["manifest"])
                    if (
                        "transform" in spec.stages
                        and "fingerprint" in spec.stages
                        and not current_manifest.get(
                            "derived_fingerprints_planned", False
                        )
                    ):
                        _LOGGER.info(
                            "Starting derived fingerprint task planning for snapshot %s",
                            spec.snapshot_id,
                        )
                        self._plan_derived_fingerprint_tasks(spec)
                        _LOGGER.info(
                            "Finished derived fingerprint task planning for snapshot %s",
                            spec.snapshot_id,
                        )
                        current_manifest["derived_fingerprints_planned"] = True
                        self.checkpoints.update_manifest(run_id, current_manifest)
                        continue
                    if (
                        "match" in spec.stages
                        and not current_manifest.get("match_tasks_planned", False)
                    ):
                        _LOGGER.info(
                            "Starting match task planning for snapshot %s",
                            spec.snapshot_id,
                        )
                        self._plan_match_tasks(spec)
                        _LOGGER.info(
                            "Finished match task planning for snapshot %s",
                            spec.snapshot_id,
                        )
                        current_manifest["match_tasks_planned"] = True
                        self.checkpoints.update_manifest(run_id, current_manifest)
                        continue

                    final_counts = self.checkpoints.task_counts(run_id)
                    if final_counts.unfinished == 0:
                        self.checkpoints.set_run_state(run_id, "complete")
                        return self._report(run_id, "complete", newly_completed)
                    self.checkpoints.set_run_state(run_id, "paused")
                    return self._report(run_id, "paused", newly_completed)

                futures = {
                    executor.submit(
                        _execute_worker_task,
                        lease.payload_json,
                        self.catalog.path,
                        str(self.term_store.root),
                        lease.task_id,
                        lease.task_key,
                        lease.lease_owner,
                    ): lease
                    for lease in leases
                }
                completed_results: list[tuple[TaskLease, _WorkerResult]] = []
                pending = set(futures)
                last_lease_reap = time.monotonic()
                while pending:
                    current_time = time.monotonic()
                    if (
                        current_time - last_lease_reap
                        >= _LEASE_REAPER_INTERVAL_SECONDS
                    ):
                        self.checkpoints.recover_expired_leases(
                            run_id,
                            max_attempts=spec.max_attempts,
                        )
                        last_lease_reap = current_time
                    done, pending = wait(
                        pending,
                        timeout=_COORDINATOR_POLL_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        lease = futures[future]
                        try:
                            result = future.result()
                        except BaseException as exc:
                            try:
                                self.checkpoints.fail_task(
                                    lease.task_id,
                                    owner,
                                    exc,
                                    max_attempts=(
                                        1 if isinstance(exc, DomainError)
                                        else spec.max_attempts
                                    ),
                                )
                            except CatalogError:
                                # If the lease expired and was reclaimed, this
                                # worker no longer owns it; the new owner will finish.
                                pass
                            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                                self.checkpoints.set_run_state(run_id, "paused")
                                raise
                        else:
                            completed_results.append((lease, result))
                            if len(completed_results) >= _RESULT_COMMIT_BATCH_SIZE:
                                try:
                                    committed = self._commit_worker_results(
                                        completed_results,
                                        max_attempts=spec.max_attempts,
                                    )
                                except (KeyboardInterrupt, SystemExit):
                                    self.checkpoints.set_run_state(run_id, "paused")
                                    raise
                                newly_completed += committed
                                completed_this_call += committed
                                completed_results.clear()
                if completed_results:
                    try:
                        committed = self._commit_worker_results(
                            completed_results,
                            max_attempts=spec.max_attempts,
                        )
                    except (KeyboardInterrupt, SystemExit):
                        self.checkpoints.set_run_state(run_id, "paused")
                        raise
                    newly_completed += committed
                    completed_this_call += committed
                # Loop to lease another bounded task batch or finalize stages.

    def resume(
        self,
        run_id: str,
        *,
        workers: int = 1,
        max_completed_tasks: int | None = None,
    ) -> RunReport:
        """Resume a planned or paused run without changing its task manifest."""

        return self.execute(
            run_id,
            workers=workers,
            max_completed_tasks=max_completed_tasks,
        )

    def result_manifest(self, run_id: str) -> str:
        """Return canonical, byte-stable output manifest for completed tasks."""

        return canonical_json(
            [
                {"task_key": task_key, "output_digest": output_digest}
                for task_key, output_digest in self.checkpoints.iter_completed_tasks(
                    run_id, page_size=self.task_batch_size
                )
            ]
        )

    def validate_bfile(
        self,
        ref: SequenceRef,
        fetcher: BFileFetcher,
        *,
        max_terms: int = 20_000,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> BFileValidationResult:
        """Fetch and validate B-file data outside transformers, then persist it.

        The injected fetcher owns all acquisition I/O. B-file corroboration is
        stored in its own status table and does not create or promote an
        E0--E4 relation claim.
        """

        if not isinstance(ref, SequenceRef):
            raise TypeError("ref must be a SequenceRef")
        if type(max_terms) is not int or not 0 < max_terms <= 100_000:
            raise ValueError("max_terms must be between 1 and 100000")
        revision = self.sequence_index.get_revision(ref)
        series: TermSeries | None = None
        if revision.offset is not None and revision.term_count:
            count = min(revision.term_count, max_terms)
            series = self.term_store.read_range(ref, revision.offset, count)
        return self._validate_bfile_series(
            ref,
            fetcher,
            series,
            validation_kind=BFileValidationKind.SEQUENCE,
            max_bytes=max_bytes,
        )

    def validate_denumerant_bfile(
        self,
        ref: SequenceRef,
        weights: Sequence[int],
        source_unit_index: int,
        fetcher: BFileFetcher,
        *,
        max_terms: int = 1_000,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> BFileValidationResult:
        """Validate a reconstructed denumerant against an indexed B-file.

        The first source coefficient equal to one establishes that generating
        function degree zero maps to ``source_unit_index``. That source term is
        checked against the immutable revision before reconstruction; the
        remaining comparison is performed by actual B-file indices.
        """

        if not isinstance(ref, SequenceRef):
            raise TypeError("ref must be a SequenceRef")
        if type(source_unit_index) is not int:
            raise TypeError("source_unit_index must be an integer")
        if type(max_terms) is not int or not 0 < max_terms <= 100_000:
            raise ValueError("max_terms must be between 1 and 100000")
        revision = self.sequence_index.get_revision(ref)
        if (
            revision.offset is None
            or revision.known_end_index is None
            or not revision.offset <= source_unit_index <= revision.known_end_index
        ):
            raise InsufficientTermsError(
                "denumerant unit index is outside the stored source range",
                context={"source_unit_index": source_unit_index},
            )
        source_unit = self.term_store.read_range(ref, source_unit_index, 1)
        if source_unit.value_at(source_unit_index) != 1:
            raise ValidationError(
                "denumerant source_unit_index does not identify a stored unit term",
                context={"source_unit_index": source_unit_index},
            )
        from .transformers.denumerant import reconstruct_denominations

        expected_values = reconstruct_denominations(weights, max_terms)
        expected_series = TermSeries(
            ref=ref,
            start_index=source_unit_index,
            values=expected_values,
            complete_through=source_unit_index + max_terms - 1,
        )
        return self._validate_bfile_series(
            ref,
            fetcher,
            expected_series,
            validation_kind=BFileValidationKind.DENUMERANT,
            max_bytes=max_bytes,
        )

    def _validate_bfile_series(
        self,
        ref: SequenceRef,
        fetcher: BFileFetcher,
        series: TermSeries | None,
        *,
        validation_kind: BFileValidationKind,
        max_bytes: int,
    ) -> BFileValidationResult:
        fetch_result = fetcher.fetch(ref.sequence_id)
        if not isinstance(fetch_result, BFileFetchResult):
            raise TypeError("fetcher.fetch must return a BFileFetchResult")
        result = BFileValidationService(max_bytes=max_bytes).validate(
            fetch_result,
            series,
            validation_kind=validation_kind,
        )
        validation_id = hashlib.sha256(
            json.dumps(
                [
                    ref.revision_id,
                    result.validation_kind.value,
                    result.validator_version,
                    result.status.value,
                    result.source_digest,
                    result.comparison_digest,
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.catalog.transaction():
            self.catalog.execute(
                """
                INSERT INTO bfile_validation (
                  validation_id, revision_id, validation_kind, validation_status,
                  source_url, source_digest, comparison_digest, compared_start,
                  compared_end, terms_checked, first_mismatch_index,
                  error_message, validator_version, validated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(validation_id) DO UPDATE SET
                  validation_status = excluded.validation_status,
                  source_url = excluded.source_url,
                  compared_start = excluded.compared_start,
                  compared_end = excluded.compared_end,
                  terms_checked = excluded.terms_checked,
                  first_mismatch_index = excluded.first_mismatch_index,
                  error_message = excluded.error_message,
                  validated_at = excluded.validated_at
                """,
                (
                    validation_id,
                    ref.revision_id,
                    result.validation_kind.value,
                    result.status.value,
                    result.source_url,
                    result.source_digest,
                    result.comparison_digest,
                    result.compared_start,
                    result.compared_end,
                    result.terms_checked,
                    result.first_mismatch_index,
                    result.error_message,
                    result.validator_version,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return result

    def _sequence_tasks(self, spec: PipelineSpec) -> Iterable[TaskManifest]:
        for descriptor in self.sequence_index.iter_sequences(
            spec.snapshot_id, Partition(page_size=spec.page_size)
        ):
            if descriptor.parse_status == "invalid":
                continue
            ref_json = _ref_to_json(descriptor.ref)
            inverse_euler_eligible: bool | None = None
            if "transform" in spec.stages and any(
                transform.name == "inverse_euler"
                for transform in spec.transform_specs
            ):
                if descriptor.offset is not None and descriptor.offset > 0:
                    inverse_euler_eligible = False
                else:
                    first_term = self.term_store.read_range(
                        descriptor.ref,
                        0,
                        1,
                    ).value_at(0)
                    inverse_euler_eligible = first_term == 1
            if "transform" in spec.stages:
                for transform in spec.transform_specs:
                    if (
                        transform.name == "inverse_euler"
                        and not inverse_euler_eligible
                    ):
                        continue
                    transformer_reference = self.transformer_registry.worker_reference(
                        transform.name,
                        transform.version,
                    )
                    transform_key = _transform_key(
                        transform,
                        transformer_reference,
                    )
                    task_key = (
                        f"{_STAGE_ORDER['transform']:02d}:transform:"
                        f"{descriptor.ref.sequence_id}:{transform_key}"
                    )
                    payload = {
                        "stage": "transform",
                        "ref": ref_json,
                        "transform": _transform_to_json(transform),
                        "transformer_reference": transformer_reference,
                    }
                    yield TaskManifest.create(
                        spec.run_id, task_key, descriptor.ref.sequence_id, payload
                    )
            if "fingerprint" in spec.stages:
                for profile in spec.fingerprint_profiles:
                    task_key = (
                        f"{_STAGE_ORDER['fingerprint']:02d}:fingerprint:"
                        f"{descriptor.ref.sequence_id}:{profile.profile_key}"
                    )
                    payload = {
                        "stage": "fingerprint",
                        "ref": ref_json,
                        "profile": _profile_to_json(profile),
                    }
                    yield TaskManifest.create(
                        spec.run_id, task_key, descriptor.ref.sequence_id, payload
                    )
        for search_spec in spec.ramanujan_search_specs:
            search_payload = search_spec.to_json_value()
            search_digest = digest_json(search_payload)
            task_key = f"{_STAGE_ORDER['match']:02d}:match:ramanujan-cf:{search_digest}"
            yield TaskManifest.create(
                spec.run_id,
                task_key,
                f"ramanujan-cf:{search_digest[:24]}",
                {
                    "stage": "match",
                    "match_kind": "ramanujan_cf_search",
                    "search_spec": search_payload,
                },
            )

    def _plan_derived_fingerprint_tasks(self, spec: PipelineSpec) -> None:
        """Plan bounded fingerprints for completed transform artifacts."""

        _LOGGER.info(
            "Starting derived fingerprint planning for snapshot %s",
            spec.snapshot_id,
        )
        transform_keys = self._spec_transform_keys(spec)
        if not transform_keys:
            _LOGGER.info(
                "No derived fingerprints to plan for snapshot %s",
                spec.snapshot_id,
            )
            return
        transform_placeholders = ",".join("?" for _ in transform_keys)
        last_artifact_id = ""
        pages = 0
        artifacts_seen = 0
        while True:
            artifacts = self.catalog.fetch_all(
                f"""
                SELECT a.artifact_id, a.revision_id, a.term_manifest_json
                FROM derived_artifact AS a
                JOIN sequence_revision AS r ON r.revision_id = a.revision_id
                WHERE r.snapshot_id = ?
                  AND a.transform_key IN ({transform_placeholders})
                  AND a.artifact_id > ?
                ORDER BY a.artifact_id
                LIMIT ?
                """,
                (
                    spec.snapshot_id,
                    *transform_keys,
                    last_artifact_id,
                    self.task_batch_size,
                ),
            )
            if not artifacts:
                _LOGGER.info(
                    "Finished derived fingerprint planning for snapshot %s "
                    "(%d artifact pages, %d artifacts)",
                    spec.snapshot_id,
                    pages,
                    artifacts_seen,
                )
                return
            pages += 1
            artifacts_seen += len(artifacts)
            tasks: list[TaskManifest] = []
            for artifact in artifacts:
                artifact_id = str(artifact["artifact_id"])
                last_artifact_id = artifact_id
                manifest = json.loads(artifact["term_manifest_json"])
                chunk = _chunk_from_json(manifest)
                source_ref = self.catalog.get_sequence_ref(artifact["revision_id"])
                for profile in spec.fingerprint_profiles:
                    start = (
                        profile.start_index
                        if profile.start_index is not None
                        else chunk.first_index
                    )
                    end = start + profile.term_count - 1
                    if start < chunk.first_index or end > chunk.last_index:
                        continue
                    task_key = (
                        f"{_STAGE_ORDER['fingerprint']:02d}:fingerprint:derived:"
                        f"{artifact_id}:{profile.profile_key}"
                    )
                    tasks.append(
                        TaskManifest.create(
                            spec.run_id,
                            task_key,
                            source_ref.sequence_id,
                            {
                                "stage": "fingerprint",
                                "artifact_id": artifact_id,
                                "source_ref": _ref_to_json(source_ref),
                                "profile": _profile_to_json(profile),
                                "range_start": start,
                            },
                        )
                    )
            if tasks:
                self.checkpoints.add_tasks(
                    tasks,
                    batch_size=self.task_batch_size,
                )
            _LOGGER.info(
                "Processed derived fingerprint page %d for snapshot %s "
                "(%d artifacts, %d tasks)",
                pages,
                spec.snapshot_id,
                len(artifacts),
                len(tasks),
            )

    def _spec_transform_keys(self, spec: PipelineSpec) -> tuple[str, ...]:
        keys = []
        for transform in spec.transform_specs:
            reference = self.transformer_registry.worker_reference(
                transform.name,
                transform.version,
            )
            keys.append(_transform_key(transform, reference))
        return tuple(sorted(set(keys)))

    def _persist_match_task_batch(
        self,
        items: Sequence[tuple[RelationClaim, str | None, TaskManifest]],
        run_id: str,
    ) -> None:
        if not items:
            return
        with self.catalog.transaction():
            for claim, transform_key, _task in items:
                self._persist_e0_claim(
                    claim,
                    run_id,
                    transform_key=transform_key,
                )
            self.checkpoints.add_tasks(
                (task for _claim, _transform_key, task in items),
                batch_size=len(items),
            )

    def _plan_match_tasks(self, spec: PipelineSpec) -> None:
        """Generate bounded exact-match work for source and derived fingerprints."""

        _LOGGER.info(
            "Starting candidate match planning for snapshot %s",
            spec.snapshot_id,
        )
        transform_keys = self._spec_transform_keys(spec)
        transform_placeholders = ",".join("?" for _ in transform_keys)
        for profile in spec.fingerprint_profiles:
            if profile.mode != "exact":
                continue
            _LOGGER.info(
                "Preparing match buckets for profile %s (maximum bucket size %d)",
                profile.profile_key,
                _MAX_MATCH_BUCKET_SIZE,
            )
            self.catalog.execute(
                "DROP TABLE IF EXISTS temp.xray_eligible_match_bucket"
            )
            self.catalog.execute(
                """
                CREATE TEMP TABLE xray_eligible_match_bucket (
                  range_start INTEGER NOT NULL,
                  range_end INTEGER NOT NULL,
                  canonicalization_version TEXT NOT NULL,
                  digest TEXT NOT NULL,
                  PRIMARY KEY (
                    range_start, range_end, canonicalization_version, digest
                  )
                ) WITHOUT ROWID
                """
            )
            self.catalog.execute(
                """
                INSERT INTO temp.xray_eligible_match_bucket
                SELECT f.range_start, f.range_end,
                       f.canonicalization_version, f.digest
                FROM fingerprint AS f
                JOIN sequence_revision AS r
                  ON r.revision_id = f.revision_id
                WHERE r.snapshot_id = ? AND f.profile_key = ?
                GROUP BY f.range_start, f.range_end,
                         f.canonicalization_version, f.digest
                HAVING COUNT(*) <= ?
                """,
                (
                    spec.snapshot_id,
                    profile.profile_key,
                    _MAX_MATCH_BUCKET_SIZE,
                ),
            )
            eligible_bucket_count = self.catalog.fetch_one(
                "SELECT COUNT(*) AS bucket_count "
                "FROM temp.xray_eligible_match_bucket"
            )
            _LOGGER.info(
                    "Prepared %d eligible fingerprint buckets for profile %s",
                    eligible_bucket_count["bucket_count"]
                    if eligible_bucket_count
                    else 0,
                    profile.profile_key,
                )
            last: tuple[str, str] | None = None
            source_pages = 0
            source_candidates = 0
            source_batch: list[
                tuple[RelationClaim, str | None, TaskManifest]
            ] = []
            while True:
                cursor_clause = ""
                if last is not None:
                    cursor_clause = "AND (left_r.revision_id, right_r.revision_id) > (?, ?)"
                rows = self.catalog.fetch_all(
                    f"""
                    SELECT left_r.revision_id AS left_revision,
                           right_r.revision_id AS right_revision,
                           left_s.sequence_id AS left_sequence,
                           right_s.sequence_id AS right_sequence,
                           f1.range_start, f1.range_end, f1.digest
                    FROM fingerprint AS f1
                    JOIN temp.xray_eligible_match_bucket AS eb
                      ON eb.range_start = f1.range_start
                     AND eb.range_end = f1.range_end
                     AND eb.canonicalization_version =
                         f1.canonicalization_version
                     AND eb.digest = f1.digest
                    JOIN fingerprint AS f2
                      ON f1.profile_key = f2.profile_key
                     AND f1.range_start = f2.range_start
                     AND f1.range_end = f2.range_end
                     AND f1.canonicalization_version = f2.canonicalization_version
                     AND f1.digest = f2.digest
                     AND f1.revision_id < f2.revision_id
                    JOIN sequence_revision AS left_r
                      ON left_r.revision_id = f1.revision_id
                    JOIN sequence_revision AS right_r
                      ON right_r.revision_id = f2.revision_id
                    JOIN sequence AS left_s ON left_s.sequence_id = left_r.sequence_id
                    JOIN sequence AS right_s ON right_s.sequence_id = right_r.sequence_id
                    WHERE left_r.snapshot_id = ? AND right_r.snapshot_id = ?
                      AND f1.profile_key = ?
                      {cursor_clause}
                    ORDER BY left_r.revision_id, right_r.revision_id
                    LIMIT ?
                    """,
                    [
                        spec.snapshot_id,
                        spec.snapshot_id,
                        profile.profile_key,
                        *(last if last is not None else ()),
                        self.task_batch_size,
                    ],
                )
                if not rows:
                    break
                source_pages += 1
                for row in rows:
                    last = (row["left_revision"], row["right_revision"])
                    source = SequenceRef(
                        row["left_sequence"], row["left_revision"], spec.snapshot_id
                    )
                    target = SequenceRef(
                        row["right_sequence"], row["right_revision"], spec.snapshot_id
                    )
                    candidate = Candidate(
                        source=source,
                        target=target,
                        profile=profile,
                        range_start=row["range_start"],
                        range_end=row["range_end"],
                        digest=row["digest"],
                    )
                    task_key = (
                        f"{_STAGE_ORDER['match']:02d}:match:{profile.profile_key}:"
                        f"{source.revision_id}:{target.revision_id}"
                    )
                    task = TaskManifest.create(
                        spec.run_id,
                        task_key,
                        f"{source.sequence_id}:{target.sequence_id}",
                        {
                            "stage": "match",
                            "source": _ref_to_json(source),
                            "target": _ref_to_json(target),
                            "profile": _profile_to_json(profile),
                            "range_start": candidate.range_start,
                            "range_end": candidate.range_end,
                            "digest": candidate.digest,
                        },
                    )
                    source_batch.append(
                        (self._e0_claim(candidate), None, task)
                    )
                    source_candidates += 1
                    if len(source_batch) >= self.task_batch_size:
                        self._persist_match_task_batch(source_batch, spec.run_id)
                        source_batch.clear()
                if source_batch:
                    self._persist_match_task_batch(source_batch, spec.run_id)
                    source_batch.clear()
                _LOGGER.info(
                    "Processed source match candidate page %d for profile %s "
                    "(%d candidates planned)",
                    source_pages,
                    profile.profile_key,
                    source_candidates,
                )

            if not transform_keys:
                continue
            last: tuple[str, str] | None = None
            derived_pages = 0
            derived_candidates = 0
            while True:
                cursor_clause = ""
                parameters: list[object] = [
                    profile.profile_key,
                    spec.snapshot_id,
                    spec.snapshot_id,
                    *transform_keys,
                ]
                if last is not None:
                    cursor_clause = "AND (af.artifact_id, target_r.revision_id) > (?, ?)"
                    parameters.extend(last)
                parameters.append(self.task_batch_size)
                rows = self.catalog.fetch_all(
                    f"""
                    SELECT af.artifact_id, af.range_start, af.range_end, af.digest,
                           a.revision_id AS source_revision,
                           a.transform_key, a.output_bound, a.term_manifest_json,
                           t.name AS transform_name,
                           t.implementation_version AS transform_version,
                           t.canonical_parameters, t.domain_spec,
                           source_s.sequence_id AS source_sequence,
                           target_r.revision_id AS target_revision,
                           target_s.sequence_id AS target_sequence
                    FROM artifact_fingerprint AS af
                    JOIN derived_artifact AS a ON a.artifact_id = af.artifact_id
                    JOIN transform_spec AS t ON t.transform_key = a.transform_key
                    JOIN sequence_revision AS source_r
                      ON source_r.revision_id = a.revision_id
                    JOIN sequence AS source_s
                      ON source_s.sequence_id = source_r.sequence_id
                    JOIN fingerprint AS f
                      ON f.profile_key = af.profile_key
                     AND f.range_start = af.range_start
                     AND f.range_end = af.range_end
                     AND f.canonicalization_version = af.canonicalization_version
                     AND f.digest = af.digest
                    JOIN temp.xray_eligible_match_bucket AS eb
                      ON eb.range_start = f.range_start
                     AND eb.range_end = f.range_end
                     AND eb.canonicalization_version =
                         f.canonicalization_version
                     AND eb.digest = f.digest
                    JOIN sequence_revision AS target_r
                      ON target_r.revision_id = f.revision_id
                    JOIN sequence AS target_s
                      ON target_s.sequence_id = target_r.sequence_id
                    WHERE af.profile_key = ?
                      AND source_r.snapshot_id = ?
                      AND target_r.snapshot_id = ?
                      AND a.transform_key IN ({transform_placeholders})
                      AND source_r.revision_id != target_r.revision_id
                      {cursor_clause}
                    ORDER BY af.artifact_id, target_r.revision_id
                    LIMIT ?
                    """,
                    parameters,
                )
                if not rows:
                    break
                derived_pages += 1
                tasks: list[TaskManifest] = []
                derived_batch: list[
                    tuple[RelationClaim, str | None, TaskManifest]
                ] = []
                for row in rows:
                    last = (row["artifact_id"], row["target_revision"])
                    source = SequenceRef(
                        row["source_sequence"],
                        row["source_revision"],
                        spec.snapshot_id,
                    )
                    target = SequenceRef(
                        row["target_sequence"],
                        row["target_revision"],
                        spec.snapshot_id,
                    )
                    candidate = Candidate(
                        source=source,
                        target=target,
                        profile=profile,
                        range_start=row["range_start"],
                        range_end=row["range_end"],
                        digest=row["digest"],
                    )
                    canonical_parameters = json.loads(row["canonical_parameters"])
                    transform = TransformSpec(
                        name=row["transform_name"],
                        version=row["transform_version"],
                        parameters=canonical_parameters["parameters"],
                        input_domain=row["domain_spec"],
                        output_bound=row["output_bound"],
                    )
                    manifest = json.loads(row["term_manifest_json"])
                    source_revision = self.catalog.fetch_one(
                        "SELECT offset FROM sequence_revision WHERE revision_id = ?",
                        (source.revision_id,),
                    )
                    input_start = transform.parameters.get(
                        "input_start_index",
                        source_revision["offset"] if source_revision else None,
                    )
                    input_count = transform.parameters.get(
                        "input_term_count",
                        transform.output_bound
                        + (1 if transform.name == "inverse_euler" else 0),
                    )
                    if type(input_start) is not int or type(input_count) is not int:
                        raise CatalogError(
                            "derived artifact lacks a reconstructible transform input range",
                            context={"artifact_id": row["artifact_id"]},
                        )
                    task_key = (
                        f"{_STAGE_ORDER['match']:02d}:match:derived:"
                        f"{row['artifact_id']}:{target.revision_id}:"
                        f"{profile.profile_key}"
                    )
                    task_payload = {
                        "stage": "match",
                        "artifact_id": row["artifact_id"],
                        "transform_key": row["transform_key"],
                        "transform": _transform_to_json(transform),
                        "input_start_index": input_start,
                        "input_term_count": input_count,
                        "output_start_index": _chunk_from_json(manifest).first_index,
                        "source": _ref_to_json(source),
                        "target": _ref_to_json(target),
                        "profile": _profile_to_json(profile),
                        "range_start": candidate.range_start,
                        "range_end": candidate.range_end,
                        "digest": candidate.digest,
                    }
                    tasks.append(
                        TaskManifest.create(
                            spec.run_id,
                            task_key,
                            f"{source.sequence_id}:{target.sequence_id}",
                            task_payload,
                        )
                    )
                    derived_batch.append(
                        (
                            self._e0_claim(candidate, row["transform_key"]),
                            row["transform_key"],
                            tasks[-1],
                        )
                    )
                    derived_candidates += 1
                    if len(derived_batch) >= self.task_batch_size:
                        self._persist_match_task_batch(
                            derived_batch,
                            spec.run_id,
                        )
                        derived_batch.clear()
                if derived_batch:
                    self._persist_match_task_batch(
                        derived_batch,
                        spec.run_id,
                    )
                _LOGGER.info(
                    "Processed derived match candidate page %d for profile %s "
                    "(%d candidates planned)",
                    derived_pages,
                    profile.profile_key,
                    derived_candidates,
                )

    def _commit_worker_result(
        self,
        lease: TaskLease,
        result: _WorkerResult,
    ) -> None:
        if result.task_id != lease.task_id or result.task_key != lease.task_key:
            raise ValidationError("worker result identity does not match its task lease")
        payload = json.loads(lease.payload_json)
        output = json.loads(result.payload_json)

        def commit(catalog: Catalog) -> None:
            stage = payload["stage"]
            if stage == "transform":
                self._commit_transform(catalog, payload, output, lease.run_id)
            elif stage == "fingerprint":
                self._commit_fingerprint(catalog, payload, output, lease.run_id)
            elif stage == "match":
                self._commit_match(
                    catalog,
                    payload,
                    output,
                    lease.run_id,
                    lease.task_id,
                )
            else:
                raise ValidationError("unknown worker result stage", context={"stage": stage})

        self.checkpoints.complete_task(
            lease.task_id,
            lease.lease_owner,
            result.output_digest,
            commit_result=commit,
        )

    def _commit_worker_results(
        self,
        results: Sequence[tuple[TaskLease, _WorkerResult]],
        *,
        max_attempts: int,
    ) -> int:
        """Commit results in bounded groups with per-task savepoints."""

        committed = 0
        for offset in range(0, len(results), _RESULT_COMMIT_BATCH_SIZE):
            with self.catalog.transaction():
                for lease, result in results[
                    offset : offset + _RESULT_COMMIT_BATCH_SIZE
                ]:
                    try:
                        self._commit_worker_result(lease, result)
                    except BaseException as exc:
                        try:
                            self.checkpoints.fail_task(
                                lease.task_id,
                                lease.lease_owner,
                                exc,
                                max_attempts=(
                                    1 if isinstance(exc, DomainError) else max_attempts
                                ),
                            )
                        except CatalogError:
                            # A recovered lease belongs to another worker now.
                            pass
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise
                    else:
                        committed += 1
        return committed

    def _record_worker_heartbeats(
        self,
        heartbeat_queue: _HeartbeatQueue,
        lease_seconds: int,
        stop_event: threading.Event,
        failures: list[Exception],
    ) -> None:
        while not stop_event.is_set():
            try:
                task_id, lease_owner = heartbeat_queue.get(timeout=1)
            except Empty:
                continue
            except Exception as exc:
                failures.append(exc)
                return
            try:
                self.checkpoints.heartbeat_task(
                    task_id,
                    lease_owner,
                    lease_seconds=lease_seconds,
                )
            except Exception as exc:
                failures.append(exc)
                return

    def _commit_transform(
        self,
        catalog: Catalog,
        task: Mapping[str, object],
        result: Mapping[str, object],
        run_id: str,
    ) -> None:
        spec = _transform_from_json(task["transform"])
        transformer_reference = str(task["transformer_reference"])
        transform_key = _transform_key(spec, transformer_reference)
        self._insert_transform_spec(
            catalog,
            spec,
            transform_key,
            transformer_reference,
        )
        artifact_id = str(result["artifact_id"])
        artifact_digest = str(result["artifact_digest"])
        manifest = dict(result["chunk"])
        manifest["_transformer_reference"] = transformer_reference
        manifest["_transformer_metadata"] = result["transformer_metadata"]
        manifest_json = canonical_json(manifest)
        catalog.execute(
            """
            INSERT INTO derived_artifact (
              artifact_id, revision_id, transform_key, output_bound,
              artifact_digest, term_manifest_json, created_by_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO NOTHING
            """,
            (
                artifact_id,
                _ref_from_json(task["ref"]).revision_id,
                transform_key,
                spec.output_bound,
                artifact_digest,
                manifest_json,
                run_id,
            ),
        )
        row = catalog.fetch_one(
            """
            SELECT revision_id, transform_key, output_bound, artifact_digest,
                   term_manifest_json
            FROM derived_artifact WHERE artifact_id = ?
            """,
            (artifact_id,),
        )
        if row is None or tuple(row[:4]) != (
            _ref_from_json(task["ref"]).revision_id,
            transform_key,
            spec.output_bound,
            artifact_digest,
        ):
            raise CatalogError("derived artifact identity conflicts with stored data")
        stored_manifest = json.loads(row["term_manifest_json"])
        storage_location_fields = {
            "content_path",
            "bundle_id",
            "byte_offset",
            "byte_length",
        }
        stored_identity = {
            key: value
            for key, value in stored_manifest.items()
            if key not in storage_location_fields
        }
        new_identity = {
            key: value
            for key, value in manifest.items()
            if key not in storage_location_fields
        }
        if stored_identity != new_identity:
            raise CatalogError("derived artifact identity conflicts with stored data")

    def _commit_fingerprint(
        self,
        catalog: Catalog,
        task: Mapping[str, object],
        result: Mapping[str, object],
        run_id: str,
    ) -> None:
        profile = _profile_from_json(task["profile"])
        if "artifact_id" in task:
            artifact_id = str(task["artifact_id"])
            start = int(result["range_start"])
            end = int(result["range_end"])
            fingerprint_id = hashlib.sha256(
                json.dumps(
                    [artifact_id, profile.profile_key, start, end],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            catalog.execute(
                """
                INSERT INTO artifact_fingerprint (
                  fingerprint_id, artifact_id, profile_key, range_start,
                  range_end, digest, canonicalization_version, created_by_run
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint_id) DO NOTHING
                """,
                (
                    fingerprint_id,
                    artifact_id,
                    profile.profile_key,
                    start,
                    end,
                    str(result["digest"]),
                    str(result["canonicalization_version"]),
                    run_id,
                ),
            )
            stored = catalog.fetch_one(
                """
                SELECT artifact_id, profile_key, range_start, range_end, digest,
                       canonicalization_version
                FROM artifact_fingerprint WHERE fingerprint_id = ?
                """,
                (fingerprint_id,),
            )
            expected = (
                artifact_id,
                profile.profile_key,
                start,
                end,
                str(result["digest"]),
                str(result["canonicalization_version"]),
            )
            if stored is None or tuple(stored) != expected:
                raise CatalogError("derived fingerprint identity conflicts with stored data")
            return

        ref = _ref_from_json(task["ref"])
        fingerprint = Fingerprint(
            ref=ref,
            profile=profile,
            range_start=int(result["range_start"]),
            range_end=int(result["range_end"]),
            digest=str(result["digest"]),
        )
        fingerprint_id = hashlib.sha256(
            json.dumps(
                [
                    ref.revision_id,
                    profile.profile_key,
                    fingerprint.range_start,
                    fingerprint.range_end,
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        catalog.execute(
            """
            INSERT INTO fingerprint (
              fingerprint_id, revision_id, profile_key, range_start, range_end,
              digest, canonicalization_version, created_by_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint_id) DO NOTHING
            """,
            (
                fingerprint_id,
                ref.revision_id,
                profile.profile_key,
                fingerprint.range_start,
                fingerprint.range_end,
                fingerprint.digest,
                fingerprint.canonicalization_version,
                run_id,
            ),
        )

    def _commit_match(
        self,
        catalog: Catalog,
        task: Mapping[str, object],
        result: Mapping[str, object],
        run_id: str,
        task_id: str,
    ) -> None:
        if task.get("match_kind") == "ramanujan_cf_search":
            self._commit_ramanujan_candidates(
                catalog,
                result,
                run_id,
                task_id,
            )
            return
        if result["status"] != "exact_match":
            return
        source = _ref_from_json(task["source"])
        target = _ref_from_json(task["target"])
        profile = _profile_from_json(task["profile"])
        candidate = Candidate(
            source=source,
            target=target,
            profile=profile,
            range_start=int(task["range_start"]),
            range_end=int(task["range_end"]),
            digest=str(task["digest"]),
        )
        detector = CollisionDetector(
            self.term_store,
            catalog,
            run_id=run_id,
        )
        if "artifact_id" in task:
            transform = _transform_from_json(task["transform"])
            relation = RelationSpec(
                relation_kind="transform-fingerprint-range",
                source_start_index=int(task["input_start_index"]),
                target_start_index=candidate.range_start,
                term_count=candidate.range_end - candidate.range_start + 1,
                transform_spec=transform,
                input_term_count=int(task["input_term_count"]),
                output_start_index=int(task["output_start_index"]),
            )
        else:
            relation = RelationSpec(
                relation_kind="exact-fingerprint-range",
                source_start_index=candidate.range_start,
                target_start_index=candidate.range_start,
                term_count=candidate.range_end - candidate.range_start + 1,
            )
        verified = detector.verify(candidate, relation, persist=False)
        if verified.status != "exact_match" or verified.claim is None:
            raise ValidationError(
                "worker match result did not reproduce in coordinator verification",
                context={"status": verified.status},
            )
        detector.persist_claim(verified.claim)

    @staticmethod
    def _commit_ramanujan_candidates(
        catalog: Catalog,
        result: Mapping[str, object],
        run_id: str,
        task_id: str,
    ) -> None:
        """Persist numerical conjecture candidates separately from relation claims."""

        for candidate in result["candidates"]:
            candidate_digest = digest_json(candidate)
            candidate_id = hashlib.sha256(
                f"{run_id}\0{candidate_digest}".encode("utf-8")
            ).hexdigest()
            catalog.execute(
                """
                INSERT INTO cf_search_candidate (
                  candidate_id, run_id, task_id, candidate_digest, lhs_key,
                  lhs_numerator_json, lhs_denominator_json, rhs_a_terms_json,
                  rhs_b_terms_json, lhs_value, rhs_value, compared_digits
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO NOTHING
                """,
                (
                    candidate_id,
                    run_id,
                    task_id,
                    candidate_digest,
                    str(candidate["lhs_key"]),
                    canonical_json(candidate["lhs_numerator"]),
                    canonical_json(candidate["lhs_denominator"]),
                    canonical_json(candidate["rhs_a_terms"]),
                    canonical_json(candidate["rhs_b_terms"]),
                    candidate["lhs_value"],
                    candidate["rhs_value"],
                    candidate["compared_digits"],
                ),
            )

    def _insert_transform_spec(
        self,
        catalog: Catalog,
        spec: TransformSpec,
        transform_key: str,
        transformer_reference: str,
    ) -> None:
        parameters_json = canonical_json(
            {
                "parameters": dict(spec.parameters),
                "transformer_reference": transformer_reference,
            }
        )
        catalog.execute(
            """
            INSERT INTO transform_spec (
              transform_key, name, implementation_version,
              canonical_parameters, domain_spec
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(transform_key) DO NOTHING
            """,
            (transform_key, spec.name, spec.version, parameters_json, spec.input_domain),
        )
        stored = catalog.fetch_one(
            """
            SELECT name, implementation_version, canonical_parameters, domain_spec
            FROM transform_spec WHERE transform_key = ?
            """,
            (transform_key,),
        )
        if stored is None or tuple(stored) != (
            spec.name,
            spec.version,
            parameters_json,
            spec.input_domain,
        ):
            raise CatalogError("transform key conflicts with stored transform metadata")

    def _e0_claim(
        self,
        candidate: Candidate,
        transform_key: str | None = None,
    ):
        evidence = {
            "kind": (
                "derived-fingerprint-candidate"
                if transform_key is not None
                else "fingerprint-candidate"
            ),
            "source": candidate.source.revision_id,
            "target": candidate.target.revision_id,
            "profile": candidate.profile.profile_key,
            "range": [candidate.range_start, candidate.range_end],
            "digest": candidate.digest,
        }
        if transform_key is not None:
            evidence["transform_key"] = transform_key
        evidence_digest = digest_json(evidence)
        from .models import RelationClaim

        return RelationClaim(
            relation_kind=(
                "transform-fingerprint-range"
                if transform_key is not None
                else f"fingerprint:{candidate.profile.mode}"
            ),
            source=candidate.source,
            target=candidate.target,
            index_map={
                "range_start": candidate.range_start,
                "range_end": candidate.range_end,
            },
            parameters={
                "profile_key": candidate.profile.profile_key,
                "digest": candidate.digest,
                **(
                    {"transform_key": transform_key}
                    if transform_key is not None
                    else {}
                ),
            },
            compared_start=candidate.range_start,
            compared_end=candidate.range_end,
            evidence_level="E0",
            evidence_digest=evidence_digest,
        )

    def _persist_e0_claim(
        self,
        claim,
        run_id: str,
        *,
        transform_key: str | None = None,
    ) -> None:
        claim_id = hashlib.sha256(
            json.dumps(
                [
                    "E0",
                    claim.source.revision_id,
                    claim.target.revision_id,
                    claim.evidence_digest,
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.catalog.execute(
            """
            INSERT INTO relation_claim (
              claim_id, relation_kind, source_revision_id, target_revision_id,
              transform_key, index_map_json, parameters_json, compared_start,
              compared_end, evidence_level, status, evidence_digest,
              created_by_run, supersedes_claim_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'E0', 'candidate',
                      ?, ?, NULL, ?)
            ON CONFLICT(claim_id) DO NOTHING
            """,
            (
                claim_id,
                claim.relation_kind,
                claim.source.revision_id,
                claim.target.revision_id,
                transform_key,
                canonical_json(dict(claim.index_map)),
                canonical_json(dict(claim.parameters)),
                claim.compared_start,
                claim.compared_end,
                claim.evidence_digest,
                run_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def _report(self, run_id: str, state: str, newly_completed: int) -> RunReport:
        return RunReport(
            run_id=run_id,
            state=state,
            counts=self.checkpoints.task_counts(run_id),
            result_manifest=self.result_manifest(run_id),
            newly_completed=newly_completed,
        )


@contextmanager
def _heartbeat_listener(
    coordinator: PipelineCoordinator,
    heartbeat_queue: _HeartbeatQueue,
    lease_seconds: int,
) -> Generator[None, None, None]:
    stop_event = threading.Event()
    failures: list[Exception] = []
    thread = threading.Thread(
        target=coordinator._record_worker_heartbeats,
        args=(heartbeat_queue, lease_seconds, stop_event, failures),
        name="xray-worker-heartbeats",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        if failures:
            raise WorkerError(
                "coordinator could not process worker heartbeat",
                context={"error": str(failures[0])},
            ) from failures[0]


def _execute_worker_task(
    payload_json: str,
    catalog_path: str,
    term_root: str,
    task_id: str,
    task_key: str,
    lease_owner: str,
) -> _WorkerResult:
    """Process-pool entry point: read-only catalog plus immutable file writes."""
    global _WORKER_HEARTBEAT_STATE

    from .storage.catalog import Catalog as WorkerCatalog

    payload = json.loads(payload_json)
    stage = payload["stage"]
    if stage not in {"transform", "fingerprint", "match"}:
        raise ValidationError("unsupported worker stage", context={"stage": stage})
    heartbeat_queue = _WORKER_HEARTBEAT_QUEUE
    if heartbeat_queue is None or _WORKER_HEARTBEAT_STOP is None:
        raise WorkerError("worker heartbeat queue was not initialized")

    with _WORKER_HEARTBEAT_LOCK:
        if _WORKER_HEARTBEAT_STATE is not None:
            raise WorkerError("worker already has an active task heartbeat")
        _WORKER_HEARTBEAT_STATE = (task_id, lease_owner)
    try:
        with WorkerCatalog(
            catalog_path,
            initialize=False,
            read_only=True,
            cache_size=-8_000,
        ) as catalog:
            store = TermStore(term_root, catalog)
            if stage == "transform":
                result_payload = _worker_transform(payload, catalog, store, term_root)
            elif stage == "fingerprint":
                result_payload = _worker_fingerprint(payload, catalog, store)
            elif payload.get("match_kind") == "ramanujan_cf_search":
                result_payload = _worker_ramanujan_cf_search(payload)
            else:
                result_payload = _worker_match(payload, catalog, store)
    finally:
        with _WORKER_HEARTBEAT_LOCK:
            _WORKER_HEARTBEAT_STATE = None
    if _WORKER_HEARTBEAT_FAILURES:
        raise WorkerError(
            "worker heartbeat could not be delivered",
            context={
                "task_id": task_id,
                "error": str(_WORKER_HEARTBEAT_FAILURES[0]),
            },
        )

    output_json = canonical_json(result_payload)
    output_digest = hashlib.sha256(output_json.encode("utf-8")).hexdigest()
    return _WorkerResult(
        task_id=task_id,
        task_key=task_key,
        output_digest=output_digest,
        payload_json=output_json,
    )


def _worker_transform(
    payload: Mapping[str, object],
    catalog: Catalog,
    store: TermStore,
    term_root: str,
) -> Mapping[str, object]:
    ref = _ref_from_json(payload["ref"])
    spec = _transform_from_json(payload["transform"])
    revision = catalog.fetch_one(
        "SELECT offset FROM sequence_revision WHERE revision_id = ?",
        (ref.revision_id,),
    )
    if revision is None:
        raise CatalogError("transform input revision is missing")
    default_start = revision["offset"] if revision["offset"] is not None else 0
    input_start = _integer_parameter(spec, "input_start_index", default_start)
    default_input_count = (
        spec.output_bound + 1 if spec.name == "inverse_euler" else spec.output_bound
    )
    input_count = _integer_parameter(
        spec, "input_term_count", default_input_count
    )
    if input_count <= 0:
        raise TransformError("transform input term count must be positive")
    input_series = store.read_range(ref, input_start, input_count)
    output_start = _integer_parameter(
        spec,
        "output_start_index",
        1 if spec.name in {"inverse_euler", "denumerant"} else 0,
    )
    output_count_value = spec.parameters.get("output_term_count")
    if output_count_value is not None and type(output_count_value) is not int:
        raise TransformError(
            "output_term_count transform parameter must be an integer"
        )
    context = TransformerExecutionContext(
        input_series=input_series,
        spec=spec,
        metadata={
            "source_sequence_id": ref.sequence_id,
            "source_revision_id": ref.revision_id,
            "input_start_index": input_start,
            "output_start_index": output_start,
            "output_term_count": output_count_value,
            "output_bound": spec.output_bound,
        },
    )
    transformer = TransformerRegistry.from_worker_reference(
        str(payload["transformer_reference"]),
        expected_name=spec.name,
        expected_version=spec.version,
    )
    transformer.validate(context)
    transform_result = transformer.transform(context)
    if not transform_result.values:
        raise TransformError("pipeline transformer returned no output terms")
    if len(transform_result.values) > spec.output_bound + 1:
        raise TransformError(
            "pipeline transformer exceeded the declared output bound",
            context={
                "output_count": len(transform_result.values),
                "output_bound": spec.output_bound,
            },
        )
    if (
        output_count_value is not None
        and len(transform_result.values) != output_count_value
    ):
        raise TransformError(
            "pipeline transformer did not return the requested output count",
            context={
                "requested": output_count_value,
                "actual": len(transform_result.values),
            },
        )
    output_values = transform_result.values

    transformer_reference = str(payload["transformer_reference"])
    transform_key = _transform_key(spec, transformer_reference)
    derived_revision = hashlib.sha256(
        json.dumps(
            [ref.revision_id, transform_key, spec.output_bound],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    derived_ref = SequenceRef(
        f"XRAY-{derived_revision[:24].upper()}",
        derived_revision,
        ref.snapshot_id,
    )
    chunk = TermStore(term_root, catalog=None).put_chunk(
        derived_ref,
        transform_result.start_index,
        output_values,
        register=False,
    )
    chunk_json = _chunk_to_json(chunk)
    artifact_digest = digest_json(
        {
            "input_revision": ref.revision_id,
            "transform": _transform_to_json(spec),
            "transformer_reference": transformer_reference,
            "chunk_digest": chunk.payload_digest,
            "output_start": transform_result.start_index,
            "output_count": len(output_values),
            "transformer_metadata": dict(transform_result.metadata),
        }
    )
    artifact_id = hashlib.sha256(
        f"{ref.revision_id}:{transform_key}:{spec.output_bound}".encode("utf-8")
    ).hexdigest()
    return {
        "artifact_id": artifact_id,
        "artifact_digest": artifact_digest,
        "chunk": chunk_json,
        "output_start_index": transform_result.start_index,
        "output_count": len(output_values),
        "transformer_metadata": dict(transform_result.metadata),
    }


def _worker_fingerprint(
    payload: Mapping[str, object],
    catalog: Catalog,
    store: TermStore,
) -> Mapping[str, object]:
    profile = _profile_from_json(payload["profile"])
    if "artifact_id" in payload:
        artifact_id = str(payload["artifact_id"])
        source_ref = _ref_from_json(payload["source_ref"])
        series = _read_artifact_series(
            catalog,
            store,
            artifact_id,
            source_ref.snapshot_id,
        )
        start = int(payload["range_start"])
        end = start + profile.term_count - 1
        values = tuple(series.value_at(index) for index in range(start, end + 1))
        if any(value is None for value in values):
            raise InsufficientTermsError(
                "derived artifact does not cover its fingerprint range",
                context={"artifact_id": artifact_id, "start": start, "end": end},
            )
        fingerprint_series = TermSeries(
            ref=series.ref,
            start_index=start,
            values=values,
            complete_through=end,
        )
        digest = HashIndex.digest_series(fingerprint_series, profile)
        fingerprint = Fingerprint(
            ref=source_ref,
            profile=profile,
            range_start=start,
            range_end=end,
            digest=digest,
        )
        return {
            "range_start": start,
            "range_end": end,
            "digest": digest,
            "profile_key": profile.profile_key,
            "canonicalization_version": fingerprint.canonicalization_version,
        }

    ref = _ref_from_json(payload["ref"])
    row = catalog.fetch_one(
        "SELECT offset FROM sequence_revision WHERE revision_id = ?",
        (ref.revision_id,),
    )
    if row is None:
        raise CatalogError("fingerprint input revision is missing")
    start = profile.start_index
    if start is None:
        start = row["offset"]
    if start is None:
        chunks = catalog.fetch_one(
            """
            SELECT first_index FROM term_chunk
            WHERE revision_id = ? ORDER BY first_index LIMIT 1
            """,
            (ref.revision_id,),
        )
        if chunks is None:
            raise InsufficientTermsError("sequence has no indexed term chunks")
        start = chunks["first_index"]
    end = start + profile.term_count - 1
    series = store.read_range(ref, start, profile.term_count)
    digest = HashIndex.digest_series(series, profile)
    return {
        "range_start": start,
        "range_end": end,
        "digest": digest,
        "profile_key": profile.profile_key,
    }


def _worker_match(
    payload: Mapping[str, object],
    catalog: Catalog,
    store: TermStore,
) -> Mapping[str, object]:
    source = _ref_from_json(payload["source"])
    target = _ref_from_json(payload["target"])
    start = int(payload["range_start"])
    end = int(payload["range_end"])
    count = end - start + 1
    if count <= 0:
        raise ValidationError("match range is empty")
    if "artifact_id" in payload:
        left = _read_artifact_series(
            catalog,
            store,
            str(payload["artifact_id"]),
            source.snapshot_id,
        )
        left_revision_id = str(payload["artifact_id"])
        transform_key = str(payload["transform_key"])
    else:
        left = store.read_range(source, start, count)
        left_revision_id = source.revision_id
        transform_key = ""
    right = store.read_range(target, start, count)
    hasher = hashlib.sha256()
    identity = [left_revision_id, target.revision_id, start, end]
    if transform_key:
        identity.append(transform_key)
    hasher.update(canonical_json(identity).encode("ascii"))
    checked = 0
    for index in range(start, end + 1):
        a = left.value_at(index)
        b = right.value_at(index)
        payload_bytes = canonical_json([index, a, b]).encode("ascii")
        hasher.update(len(payload_bytes).to_bytes(8, "big"))
        hasher.update(payload_bytes)
        if a is None or b is None:
            return {
                "status": "insufficient_data",
                "checked_count": checked,
                "missing_index": index,
                "evidence_digest": hasher.hexdigest(),
            }
        checked += 1
        if a != b:
            return {
                "status": "mismatch",
                "checked_count": checked,
                "first_mismatch": [index, a, b],
                "evidence_digest": hasher.hexdigest(),
            }
    return {
        "status": "exact_match",
        "checked_count": checked,
        "evidence_digest": hasher.hexdigest(),
    }


def _read_artifact_series(
    catalog: Catalog,
    store: TermStore,
    artifact_id: str,
    snapshot_id: str,
) -> TermSeries:
    row = catalog.fetch_one(
        """
        SELECT revision_id, term_manifest_json
        FROM derived_artifact WHERE artifact_id = ?
        """,
        (artifact_id,),
    )
    if row is None:
        raise CatalogError(
            "derived artifact is missing",
            context={"artifact_id": artifact_id},
        )
    manifest = json.loads(row["term_manifest_json"])
    chunk = _chunk_from_json(manifest)
    derived_ref = SequenceRef(
        f"XRAY-{chunk.revision_id[:24].upper()}",
        chunk.revision_id,
        snapshot_id,
    )
    series = store.read_chunk(chunk, ref=derived_ref)
    if (
        "output_start_index" in manifest
        and series.start_index != int(manifest["output_start_index"])
    ):
        raise CatalogError(
            "derived artifact start index conflicts with its term manifest",
            context={"artifact_id": artifact_id},
        )
    if (
        "output_count" in manifest
        and len(series.values) != int(manifest["output_count"])
    ):
        raise CatalogError(
            "derived artifact term count conflicts with its term manifest",
            context={"artifact_id": artifact_id},
        )
    return series


def _worker_ramanujan_cf_search(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    from .transformers.ramanujan.search import (
        CFSearchSpec,
        search_gcf_candidates,
    )

    search_spec = CFSearchSpec.from_json_value(payload["search_spec"])
    candidates = search_gcf_candidates(search_spec)
    return {
        "status": "numerical_candidates",
        "candidates": [
            {
                "lhs_index": candidate.lhs_index,
                "rhs_index": candidate.rhs_index,
                "lhs_key": candidate.lhs_key,
                "lhs_numerator": list(candidate.lhs_numerator),
                "lhs_denominator": list(candidate.lhs_denominator),
                "rhs_a_terms": list(candidate.rhs_a_terms),
                "rhs_b_terms": list(candidate.rhs_b_terms),
                "lhs_value": candidate.lhs_value,
                "rhs_value": candidate.rhs_value,
                "compared_digits": candidate.compared_digits,
            }
            for candidate in candidates
        ],
    }


def _transform_to_json(spec: TransformSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "version": spec.version,
        "parameters": dict(spec.parameters),
        "input_domain": spec.input_domain,
        "output_bound": spec.output_bound,
    }


def _transform_from_json(value: object) -> TransformSpec:
    if not isinstance(value, Mapping):
        raise ValidationError("transform specification must be an object")
    return TransformSpec(
        name=str(value["name"]),
        version=str(value["version"]),
        parameters=value["parameters"],
        input_domain=str(value["input_domain"]),
        output_bound=int(value["output_bound"]),
    )


def _transform_key(spec: TransformSpec, transformer_reference: str) -> str:
    return digest_json(
        {
            "spec": _transform_to_json(spec),
            "transformer_reference": transformer_reference,
        }
    )


def _profile_to_json(profile: FingerprintProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "version": profile.version,
        "mode": profile.mode,
        "term_count": profile.term_count,
        "start_index": profile.start_index,
    }


def _profile_from_json(value: object) -> FingerprintProfile:
    if not isinstance(value, Mapping):
        raise ValidationError("fingerprint profile must be an object")
    return FingerprintProfile(
        name=str(value["name"]),
        version=str(value["version"]),
        mode=value["mode"],
        term_count=int(value["term_count"]),
        start_index=value["start_index"],
    )


def _spec_to_json(spec: PipelineSpec) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "snapshot_id": spec.snapshot_id,
        "stages": list(spec.stages),
        "transform_specs": [_transform_to_json(x) for x in spec.transform_specs],
        "fingerprint_profiles": [
            _profile_to_json(x) for x in spec.fingerprint_profiles
        ],
        "ramanujan_search_specs": [
            search_spec.to_json_value()
            for search_spec in spec.ramanujan_search_specs
        ],
        "page_size": spec.page_size,
        "lease_seconds": spec.lease_seconds,
        "max_attempts": spec.max_attempts,
        "code_revision": spec.code_revision,
    }


def _spec_from_json(value: object) -> PipelineSpec:
    if not isinstance(value, Mapping):
        raise ValidationError("stored pipeline specification is malformed")
    serialized_search_specs = value.get("ramanujan_search_specs", [])
    if serialized_search_specs:
        from .transformers.ramanujan.search import CFSearchSpec

        search_specs = tuple(
            CFSearchSpec.from_json_value(item)
            for item in serialized_search_specs
        )
    else:
        search_specs = ()
    return PipelineSpec(
        run_id=str(value["run_id"]),
        snapshot_id=str(value["snapshot_id"]),
        stages=tuple(value["stages"]),
        transform_specs=tuple(
            _transform_from_json(item) for item in value["transform_specs"]
        ),
        fingerprint_profiles=tuple(
            _profile_from_json(item) for item in value["fingerprint_profiles"]
        ),
        ramanujan_search_specs=search_specs,
        page_size=int(value["page_size"]),
        lease_seconds=int(value["lease_seconds"]),
        max_attempts=int(value["max_attempts"]),
        code_revision=str(value["code_revision"]),
    )


def _ref_to_json(ref: SequenceRef) -> dict[str, str]:
    return {
        "sequence_id": ref.sequence_id,
        "revision_id": ref.revision_id,
        "snapshot_id": ref.snapshot_id,
    }


def _ref_from_json(value: object) -> SequenceRef:
    if not isinstance(value, Mapping):
        raise ValidationError("sequence reference must be an object")
    return SequenceRef(
        sequence_id=str(value["sequence_id"]),
        revision_id=str(value["revision_id"]),
        snapshot_id=str(value["snapshot_id"]),
    )


def _chunk_to_json(chunk: ChunkRef) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "revision_id": chunk.revision_id,
        "first_index": chunk.first_index,
        "term_count": chunk.term_count,
        "last_index": chunk.last_index,
        "codec": chunk.codec,
        "format_version": chunk.format_version,
        "uncompressed_bytes": chunk.uncompressed_bytes,
        "payload_digest": chunk.payload_digest,
        "content_path": chunk.content_path,
        "bundle_id": chunk.bundle_id,
        "byte_offset": chunk.byte_offset,
        "byte_length": chunk.byte_length,
    }


def _chunk_from_json(value: object) -> ChunkRef:
    if not isinstance(value, Mapping):
        raise ValidationError("term chunk manifest must be an object")
    integer_fields = (
        "first_index",
        "term_count",
        "last_index",
        "format_version",
        "uncompressed_bytes",
    )
    text_fields = (
        "chunk_id",
        "revision_id",
        "codec",
        "payload_digest",
        "content_path",
    )
    if any(type(value.get(field)) is not int for field in integer_fields):
        raise ValidationError("term chunk manifest has invalid integer fields")
    if any(not isinstance(value.get(field), str) for field in text_fields):
        raise ValidationError("term chunk manifest has invalid text fields")
    return ChunkRef(
        chunk_id=value["chunk_id"],
        revision_id=value["revision_id"],
        first_index=value["first_index"],
        term_count=value["term_count"],
        last_index=value["last_index"],
        codec=value["codec"],
        format_version=value["format_version"],
        uncompressed_bytes=value["uncompressed_bytes"],
        payload_digest=value["payload_digest"],
        content_path=value["content_path"],
        bundle_id=value.get("bundle_id"),
        byte_offset=value.get("byte_offset"),
        byte_length=value.get("byte_length"),
    )


def _integer_parameter(spec: TransformSpec, name: str, default: int) -> int:
    value = spec.parameters.get(name, default)
    if type(value) is not int:
        raise ValidationError(
            f"transform parameter {name!r} must be an integer",
            context={"value": value},
        )
    return value
