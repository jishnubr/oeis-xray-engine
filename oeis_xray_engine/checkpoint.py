"""Lease-based, idempotent task and run state for resumable execution."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping

from .errors import CatalogError, ValidationError, WorkerError
from .storage.catalog import Catalog

_TASK_STATES = frozenset(
    ("pending", "leased", "complete", "retryable_error", "fatal_error")
)
_STALE_HEARTBEAT_SECONDS = 300
_RUN_STATES = frozenset(
    ("planned", "running", "paused", "complete", "failed", "cancelled")
)


def canonical_json(value: object) -> str:
    """Serialize JSON-compatible values deterministically."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "value cannot be represented in canonical JSON",
            context={"error": str(exc)},
        ) from exc


def digest_json(value: object) -> str:
    """Return SHA-256 for a canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskManifest:
    """Immutable task input and deterministic idempotency identity."""

    task_id: str
    run_id: str
    task_key: str
    partition_key: str
    input_digest: str
    payload_json: str

    @classmethod
    def create(
        cls,
        run_id: str,
        task_key: str,
        partition_key: str,
        payload: Mapping[str, object],
    ) -> "TaskManifest":
        if not run_id or not task_key or not partition_key:
            raise ValueError("run_id, task_key, and partition_key are required")
        payload_json = canonical_json(payload)
        input_digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        task_id = hashlib.sha256(
            f"{run_id}\0{task_key}".encode("utf-8")
        ).hexdigest()
        return cls(
            task_id=task_id,
            run_id=run_id,
            task_key=task_key,
            partition_key=partition_key,
            input_digest=input_digest,
            payload_json=payload_json,
        )


@dataclass(frozen=True, slots=True)
class TaskLease:
    """A task lease returned to a coordinator for execution."""

    task_id: str
    run_id: str
    task_key: str
    partition_key: str
    input_digest: str
    payload_json: str
    lease_owner: str
    lease_until: str
    attempt_count: int


@dataclass(frozen=True, slots=True)
class TaskCounts:
    """Bounded task-state summary for a run."""

    pending: int
    leased: int
    complete: int
    retryable_error: int
    fatal_error: int

    @property
    def unfinished(self) -> int:
        return self.pending + self.leased + self.retryable_error


class CheckpointStore:
    """Coordinator-only persistence for tasks, leases, and run manifests."""

    def __init__(self, catalog: Catalog) -> None:
        if not isinstance(catalog, Catalog):
            raise TypeError("catalog must be a Catalog")
        if catalog.read_only:
            raise CatalogError("checkpoint store requires a writable catalog")
        self.catalog = catalog

    def create_run(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        run_kind: str,
        config: Mapping[str, object],
        code_revision: str,
        manifest: Mapping[str, object],
    ) -> None:
        """Create a planned run idempotently or validate its existing identity."""

        if not all((run_id, snapshot_id, run_kind, code_revision)):
            raise ValueError("run identity fields must be non-empty")
        config_json = canonical_json(config)
        manifest_json = canonical_json(manifest)
        config_digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        with self.catalog.transaction():
            existing = self.catalog.fetch_one(
                """
                SELECT snapshot_id, run_kind, config_digest, code_revision,
                       manifest_json
                FROM run WHERE run_id = ?
                """,
                (run_id,),
            )
            expected = (
                snapshot_id,
                run_kind,
                config_digest,
                code_revision,
                manifest_json,
            )
            if existing is not None:
                if tuple(existing) != expected:
                    raise CatalogError(
                        "run_id already exists with a different immutable manifest",
                        context={"run_id": run_id},
                    )
                return
            snapshot = self.catalog.fetch_one(
                "SELECT state FROM corpus_snapshot WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            if snapshot is None or snapshot["state"] != "ready":
                raise CatalogError(
                    "run requires a ready corpus snapshot",
                    context={"snapshot_id": snapshot_id},
                )
            self.catalog.execute(
                """
                INSERT INTO run (
                  run_id, snapshot_id, run_kind, config_digest, code_revision,
                  state, started_at, finished_at, manifest_json
                ) VALUES (?, ?, ?, ?, ?, 'planned', NULL, NULL, ?)
                """,
                (
                    run_id,
                    snapshot_id,
                    run_kind,
                    config_digest,
                    code_revision,
                    manifest_json,
                ),
            )

    def add_tasks(
        self,
        tasks: Iterable[TaskManifest],
        *,
        batch_size: int = 500,
    ) -> int:
        """Insert tasks in bounded transactions; identical retries are no-ops."""

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        inserted_or_seen = 0
        batch: list[TaskManifest] = []

        def commit_batch(items: list[TaskManifest]) -> None:
            with self.catalog.transaction():
                for task in items:
                    row = self.catalog.fetch_one(
                        """
                        SELECT task_id, partition_key, input_digest,
                               task_manifest_json
                        FROM task WHERE run_id = ? AND task_key = ?
                        """,
                        (task.run_id, task.task_key),
                    )
                    if row is not None:
                        if tuple(row) != (
                            task.task_id,
                            task.partition_key,
                            task.input_digest,
                            task.payload_json,
                        ):
                            raise CatalogError(
                                "task idempotency key conflicts with stored task",
                                context={
                                    "run_id": task.run_id,
                                    "task_key": task.task_key,
                                },
                            )
                        continue
                    self.catalog.execute(
                        """
                        INSERT INTO task (
                          task_id, run_id, task_key, partition_key, state,
                          lease_owner, lease_until, attempt_count, input_digest,
                          task_manifest_json, output_digest, error_json, updated_at
                        ) VALUES (?, ?, ?, ?, 'pending', NULL, NULL, 0, ?, ?,
                                  NULL, NULL, ?)
                        """,
                        (
                            task.task_id,
                            task.run_id,
                            task.task_key,
                            task.partition_key,
                            task.input_digest,
                            task.payload_json,
                            _utc_now(),
                        ),
                    )

        for task in tasks:
            if not isinstance(task, TaskManifest):
                raise TypeError("tasks must contain TaskManifest values")
            batch.append(task)
            inserted_or_seen += 1
            if len(batch) >= batch_size:
                commit_batch(batch)
                batch = []
        if batch:
            commit_batch(batch)
        return inserted_or_seen

    def lease_tasks(
        self,
        run_id: str,
        owner: str | None = None,
        *,
        limit: int = 1,
        lease_seconds: int = 300,
        max_attempts: int = 3,
    ) -> list[TaskLease]:
        """Atomically lease a bounded set of ready or expired tasks."""

        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        owner = owner or f"{socket.gethostname()}:{os.getpid()}"
        now = _utc_now()
        lease_until = (
            datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        ).isoformat()
        leases: list[TaskLease] = []

        with self.catalog.transaction():
            self.recover_expired_leases(
                run_id,
                now=now,
                max_attempts=max_attempts,
                within_transaction=True,
            )
            rows = self.catalog.fetch_all(
                """
                SELECT task_id, run_id, task_key, partition_key, input_digest,
                       task_manifest_json, attempt_count
                FROM task
                WHERE run_id = ?
                  AND state IN ('pending','retryable_error')
                  AND attempt_count < ?
                ORDER BY partition_key, task_key
                LIMIT ?
                """,
                (run_id, max_attempts, limit),
            )
            for row in rows:
                self.catalog.execute(
                    """
                    UPDATE task
                    SET state = 'leased', lease_owner = ?, lease_until = ?,
                        last_heartbeat = ?, attempt_count = attempt_count + 1,
                        error_json = NULL,
                        updated_at = ?
                    WHERE task_id = ? AND state IN ('pending','retryable_error')
                    """,
                    (owner, lease_until, now, now, row["task_id"]),
                )
                leases.append(
                    TaskLease(
                        task_id=row["task_id"],
                        run_id=row["run_id"],
                        task_key=row["task_key"],
                        partition_key=row["partition_key"],
                        input_digest=row["input_digest"],
                        payload_json=row["task_manifest_json"],
                        lease_owner=owner,
                        lease_until=lease_until,
                        attempt_count=row["attempt_count"] + 1,
                    )
                )
        return leases

    def heartbeat_task(
        self,
        task_id: str,
        owner: str,
        *,
        lease_seconds: int,
        now: str | None = None,
    ) -> bool:
        """Refresh a live worker's heartbeat and extend its lease."""

        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        timestamp = now or _utc_now()
        lease_until = (
            datetime.fromisoformat(timestamp) + timedelta(seconds=lease_seconds)
        ).isoformat()
        result = self.catalog.execute(
            """
            UPDATE task
            SET last_heartbeat = ?, lease_until = ?, updated_at = ?
            WHERE task_id = ? AND state = 'leased' AND lease_owner = ?
            """,
            (timestamp, lease_until, timestamp, task_id, owner),
        )
        return result.rowcount == 1

    def recover_expired_leases(
        self,
        run_id: str,
        *,
        now: str | None = None,
        max_attempts: int = 3,
        within_transaction: bool = False,
    ) -> int:
        """Requeue expired tasks or mark exhausted tasks fatal."""

        timestamp = now or _utc_now()
        stale_before = (
            datetime.fromisoformat(timestamp)
            - timedelta(seconds=_STALE_HEARTBEAT_SECONDS)
        ).isoformat()

        def recover() -> int:
            expired = self.catalog.fetch_all(
                """
                SELECT task_id, attempt_count FROM task
                WHERE run_id = ? AND state = 'leased'
                  AND (
                    lease_until <= ? OR last_heartbeat IS NULL
                    OR last_heartbeat <= ?
                  )
                """,
                (run_id, timestamp, stale_before),
            )
            for row in expired:
                state = (
                    "fatal_error"
                    if row["attempt_count"] >= max_attempts
                    else "retryable_error"
                )
                error = canonical_json(
                    {
                        "kind": "lease_expired",
                        "expired_at": timestamp,
                        "attempt_count": row["attempt_count"],
                    }
                )
                self.catalog.execute(
                    """
                    UPDATE task
                    SET state = ?, lease_owner = NULL, lease_until = NULL,
                        last_heartbeat = NULL,
                        error_json = ?, updated_at = ?
                    WHERE task_id = ? AND state = 'leased'
                    """,
                    (state, error, timestamp, row["task_id"]),
                )
            return len(expired)

        if within_transaction:
            return recover()
        with self.catalog.transaction():
            return recover()

    def complete_task(
        self,
        task_id: str,
        owner: str,
        output_digest: str,
        *,
        commit_result: Callable[[Catalog], None] | None = None,
    ) -> None:
        """Commit task outputs and completion atomically in one catalog tx."""

        if not output_digest:
            raise ValueError("output_digest must be non-empty")
        with self.catalog.transaction():
            row = self.catalog.fetch_one(
                "SELECT state, lease_owner, output_digest FROM task WHERE task_id = ?",
                (task_id,),
            )
            if row is None:
                raise CatalogError("task does not exist", context={"task_id": task_id})
            if row["state"] == "complete":
                if row["output_digest"] == output_digest:
                    return
                raise CatalogError(
                    "completed task received a different output digest",
                    context={"task_id": task_id},
                )
            if row["state"] != "leased" or row["lease_owner"] != owner:
                raise CatalogError(
                    "task completion rejected because lease is not owned",
                    context={"task_id": task_id, "owner": owner},
                )
            if commit_result is not None:
                commit_result(self.catalog)
            result = self.catalog.execute(
                """
                UPDATE task
                SET state = 'complete', output_digest = ?, lease_owner = NULL,
                    lease_until = NULL, last_heartbeat = NULL,
                    error_json = NULL, updated_at = ?
                WHERE task_id = ? AND state = 'leased' AND lease_owner = ?
                """,
                (output_digest, _utc_now(), task_id, owner),
            )
            if result.rowcount != 1:
                raise CatalogError(
                    "task lease changed while committing output",
                    context={"task_id": task_id},
                )

    def fail_task(
        self,
        task_id: str,
        owner: str,
        error: BaseException | Mapping[str, object],
        *,
        max_attempts: int = 3,
    ) -> None:
        """Record a bounded error and make the task retryable or fatal."""

        if isinstance(error, BaseException):
            error_data: Mapping[str, object] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        else:
            error_data = error
        with self.catalog.transaction():
            row = self.catalog.fetch_one(
                "SELECT state, lease_owner, attempt_count FROM task WHERE task_id = ?",
                (task_id,),
            )
            if row is None:
                raise CatalogError("task does not exist", context={"task_id": task_id})
            if row["state"] != "leased" or row["lease_owner"] != owner:
                raise CatalogError(
                    "task failure rejected because lease is not owned",
                    context={"task_id": task_id, "owner": owner},
                )
            state = (
                "fatal_error"
                if row["attempt_count"] >= max_attempts
                else "retryable_error"
            )
            result = self.catalog.execute(
                """
                UPDATE task SET state = ?, error_json = ?, lease_owner = NULL,
                    lease_until = NULL, last_heartbeat = NULL, updated_at = ?
                WHERE task_id = ?
                """,
                (state, canonical_json(error_data), _utc_now(), task_id),
            )
            if result.rowcount != 1:
                raise WorkerError("failed to persist worker error", context={"task_id": task_id})

    def task_counts(self, run_id: str) -> TaskCounts:
        """Return task counts using a bounded aggregate query."""

        row = self.catalog.fetch_one(
            """
            SELECT
              SUM(state = 'pending') AS pending,
              SUM(state = 'leased') AS leased,
              SUM(state = 'complete') AS complete,
              SUM(state = 'retryable_error') AS retryable_error,
              SUM(state = 'fatal_error') AS fatal_error
            FROM task WHERE run_id = ?
            """,
            (run_id,),
        )
        return TaskCounts(
            pending=row["pending"] or 0,
            leased=row["leased"] or 0,
            complete=row["complete"] or 0,
            retryable_error=row["retryable_error"] or 0,
            fatal_error=row["fatal_error"] or 0,
        )

    def iter_completed_tasks(
        self,
        run_id: str,
        *,
        page_size: int = 500,
    ):
        """Keyset-page completed task results in deterministic task-key order."""

        if type(page_size) is not int or page_size <= 0:
            raise ValueError("page_size must be positive")
        last_key: str | None = None
        while True:
            if last_key is None:
                rows = self.catalog.fetch_all(
                    """
                    SELECT task_key, output_digest FROM task
                    WHERE run_id = ? AND state = 'complete'
                    ORDER BY task_key LIMIT ?
                    """,
                    (run_id, page_size),
                )
            else:
                rows = self.catalog.fetch_all(
                    """
                    SELECT task_key, output_digest FROM task
                    WHERE run_id = ? AND state = 'complete' AND task_key > ?
                    ORDER BY task_key LIMIT ?
                    """,
                    (run_id, last_key, page_size),
                )
            if not rows:
                return
            for row in rows:
                last_key = row["task_key"]
                yield row["task_key"], row["output_digest"]

    def get_run(self, run_id: str) -> Mapping[str, object]:
        """Return a detached run record, including decoded manifest."""

        row = self.catalog.fetch_one("SELECT * FROM run WHERE run_id = ?", (run_id,))
        if row is None:
            raise CatalogError("run does not exist", context={"run_id": run_id})
        result = dict(row)
        try:
            result["manifest"] = json.loads(result["manifest_json"])
        except json.JSONDecodeError as exc:
            raise CatalogError(
                "run manifest is corrupt", context={"run_id": run_id}
            ) from exc
        return result

    def update_manifest(self, run_id: str, manifest: Mapping[str, object]) -> None:
        with self.catalog.transaction():
            result = self.catalog.execute(
                "UPDATE run SET manifest_json = ? WHERE run_id = ?",
                (canonical_json(manifest), run_id),
            )
            if result.rowcount != 1:
                raise CatalogError("run does not exist", context={"run_id": run_id})

    def set_run_state(self, run_id: str, state: str) -> None:
        if state not in _RUN_STATES:
            raise ValueError(f"invalid run state: {state!r}")
        finished_at = _utc_now() if state in {"complete", "failed", "cancelled"} else None
        result = self.catalog.execute(
            """
            UPDATE run SET state = ?, started_at = COALESCE(started_at, ?),
                finished_at = ? WHERE run_id = ?
            """,
            (state, _utc_now() if state == "running" else None, finished_at, run_id),
        )
        if result.rowcount != 1:
            raise CatalogError("run does not exist", context={"run_id": run_id})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
