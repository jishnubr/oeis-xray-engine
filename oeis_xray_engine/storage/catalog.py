"""Thread-safe, single-connection SQLite catalog for coordinator-owned writes.

Workers should not open this catalog to write results. A coordinator serializes
mutations through this class and commits artifact references together with
task-completion state. The API returns detached row values rather than cursors,
so callers cannot continue using a cursor after the connection lock is released.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence
from urllib.parse import quote

from ..errors import CatalogError
from ..models import ChunkRef, SequenceRef

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_JOURNAL_MODES = frozenset(("DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"))
_SYNCHRONOUS_MODES = frozenset(("OFF", "NORMAL", "FULL", "EXTRA"))


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Detached summary of a completed INSERT/UPDATE/DELETE statement."""

    rowcount: int
    lastrowid: int | None


class Catalog:
    """A serialized SQLite catalog connection owned by one coordinator.

    The re-entrant lock serializes all use of this connection across threads.
    SQLite still has a single writer; this wrapper does not provide multi-process
    coordination or turn SQLite into a multi-writer database.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: float = 30.0,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
        cache_size: int = -64_000,
        initialize: bool = True,
        read_only: bool = False,
    ) -> None:
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        if type(cache_size) is not int or cache_size == 0:
            raise ValueError("cache_size must be a non-zero integer")
        journal_mode = journal_mode.upper()
        synchronous = synchronous.upper()
        if journal_mode not in _JOURNAL_MODES:
            raise ValueError(f"unsupported SQLite journal_mode: {journal_mode!r}")
        if synchronous not in _SYNCHRONOUS_MODES:
            raise ValueError(f"unsupported SQLite synchronous mode: {synchronous!r}")

        self.path = str(path)
        self.read_only = read_only
        if read_only and (self.path == ":memory:" or self.path.startswith("file:")):
            raise ValueError("read-only catalog requires a filesystem path")
        if not read_only and self.path != ":memory:" and not self.path.startswith("file:"):
            Path(self.path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )

        self._lock = threading.RLock()
        self._local = threading.local()
        self._closed = False
        try:
            if read_only:
                absolute_path = Path(self.path).expanduser().resolve().as_posix()
                connection_target = f"file:{quote(absolute_path, safe='/')}?mode=ro"
                self._connection = sqlite3.connect(
                    connection_target,
                    timeout=timeout,
                    isolation_level=None,
                    check_same_thread=False,
                    uri=True,
                )
            else:
                self._connection = sqlite3.connect(
                    self.path,
                    timeout=timeout,
                    isolation_level=None,
                    check_same_thread=False,
                    uri=self.path.startswith("file:"),
                )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
            self._connection.execute(f"PRAGMA cache_size = {cache_size}")
            if not read_only:
                self._connection.execute(f"PRAGMA synchronous = {synchronous}")
                self._connection.execute(f"PRAGMA journal_mode = {journal_mode}")
            if initialize and not read_only:
                self.initialize_schema()
        except CatalogError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            self._closed = True
            raise
        except (sqlite3.Error, OSError) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            self._closed = True
            raise CatalogError(
                "failed to initialize SQLite catalog",
                context={"path": self.path, "error": str(exc)},
            ) from exc

    def initialize_schema(self) -> None:
        """Create the catalog schema if it does not already exist."""

        if self.read_only:
            raise CatalogError("cannot initialize a read-only catalog")
        try:
            schema = _SCHEMA_PATH.read_text(encoding="utf-8")
            with self._lock:
                self._ensure_open()
                # executescript manages its own transaction boundaries. Keep
                # schema initialization outside a caller-managed transaction.
                if getattr(self._local, "transaction_depth", 0):
                    raise CatalogError(
                        "schema initialization cannot run inside a transaction"
                    )
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + schema + "\nCOMMIT;"
                )
                task_columns = {
                    row["name"]
                    for row in self._connection.execute("PRAGMA table_info(task)")
                }
                if "task_manifest_json" not in task_columns:
                    self._connection.execute(
                        "ALTER TABLE task ADD COLUMN task_manifest_json "
                        "TEXT NOT NULL DEFAULT '{}'"
                    )
                if "last_heartbeat" not in task_columns:
                    self._connection.execute(
                        "ALTER TABLE task ADD COLUMN last_heartbeat TEXT"
                    )
                chunk_columns = {
                    row["name"]
                    for row in self._connection.execute(
                        "PRAGMA table_info(term_chunk)"
                    )
                }
                for column, declaration in (
                    ("bundle_id", "TEXT"),
                    ("byte_offset", "INTEGER"),
                    ("byte_length", "INTEGER"),
                ):
                    if column not in chunk_columns:
                        self._connection.execute(
                            f"ALTER TABLE term_chunk ADD COLUMN {column} "
                            f"{declaration}"
                        )
                self._connection.execute(
                    """
                    UPDATE task SET last_heartbeat = updated_at
                    WHERE state = 'leased' AND last_heartbeat IS NULL
                    """
                )
                enabled = self._connection.execute(
                    "PRAGMA foreign_keys"
                ).fetchone()[0]
                if enabled != 1:
                    raise CatalogError("SQLite foreign-key enforcement is disabled")
        except OSError as exc:
            raise CatalogError(
                "failed to read catalog schema",
                context={"schema_path": str(_SCHEMA_PATH), "error": str(exc)},
            ) from exc
        except sqlite3.Error as exc:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise CatalogError(
                "failed to create catalog schema",
                context={"path": self.path, "error": str(exc)},
            ) from exc

    def execute(
        self,
        sql: str,
        parameters: Sequence[object] | Mapping[str, object] = (),
    ) -> ExecutionResult:
        """Execute one statement and return detached mutation metadata.

        Use :meth:`fetch_one` or :meth:`fetch_all` for queries. This method does
        not expose a live cursor outside the connection lock. Mutations outside
        an explicit transaction are wrapped in ``BEGIN IMMEDIATE``.
        """

        with self._lock:
            self._ensure_open()
            if self.read_only:
                raise CatalogError("cannot execute mutations on a read-only catalog")
            if getattr(self._local, "transaction_depth", 0) == 0:
                with self.transaction(immediate=True):
                    return self.execute(sql, parameters)
            try:
                cursor = self._connection.execute(sql, parameters)
                return ExecutionResult(cursor.rowcount, cursor.lastrowid)
            except sqlite3.Error as exc:
                raise CatalogError(
                    "SQLite statement failed",
                    context={"error": str(exc), "sql": sql},
                ) from exc

    def executemany(
        self,
        sql: str,
        parameters: Sequence[Sequence[object] | Mapping[str, object]],
    ) -> ExecutionResult:
        """Execute a bounded batch, using ``BEGIN IMMEDIATE`` if needed."""

        with self._lock:
            self._ensure_open()
            if self.read_only:
                raise CatalogError("cannot execute mutations on a read-only catalog")
            if getattr(self._local, "transaction_depth", 0) == 0:
                with self.transaction(immediate=True):
                    return self.executemany(sql, parameters)
            try:
                cursor = self._connection.executemany(sql, parameters)
                return ExecutionResult(cursor.rowcount, cursor.lastrowid)
            except sqlite3.Error as exc:
                raise CatalogError(
                    "SQLite batch statement failed",
                    context={"error": str(exc), "sql": sql},
                ) from exc

    def fetch_one(
        self,
        sql: str,
        parameters: Sequence[object] | Mapping[str, object] = (),
    ) -> sqlite3.Row | None:
        """Execute a query and return one detached SQLite row, if present."""

        with self._lock:
            self._ensure_open()
            try:
                return self._connection.execute(sql, parameters).fetchone()
            except sqlite3.Error as exc:
                raise CatalogError(
                    "SQLite query failed",
                    context={"error": str(exc), "sql": sql},
                ) from exc

    def fetch_all(
        self,
        sql: str,
        parameters: Sequence[object] | Mapping[str, object] = (),
    ) -> list[sqlite3.Row]:
        """Execute a bounded query and materialize its rows under the lock.

        Callers performing large scans should use keyset-paginated queries;
        this convenience method is intended for bounded result sets.
        """

        with self._lock:
            self._ensure_open()
            try:
                return self._connection.execute(sql, parameters).fetchall()
            except sqlite3.Error as exc:
                raise CatalogError(
                    "SQLite query failed",
                    context={"error": str(exc), "sql": sql},
                ) from exc

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator["Catalog"]:
        """Run an atomic transaction; nested uses are implemented by savepoints.

        The connection lock remains held for the entire context. Catching
        ``BaseException`` ensures cancellation and interrupts also roll back.
        """

        self._lock.acquire()
        depth = getattr(self._local, "transaction_depth", 0)
        savepoint = f"xray_sp_{depth}"
        try:
            self._ensure_open()
            if self.read_only:
                raise CatalogError("cannot begin a transaction on a read-only catalog")
            try:
                if depth == 0:
                    self._connection.execute(
                        "BEGIN IMMEDIATE" if immediate else "BEGIN"
                    )
                else:
                    self._connection.execute(f"SAVEPOINT {savepoint}")
                self._local.transaction_depth = depth + 1
            except sqlite3.Error as exc:
                raise CatalogError(
                    "failed to begin SQLite transaction",
                    context={"error": str(exc)},
                ) from exc

            try:
                yield self
            except BaseException:
                self._rollback_transaction(depth, savepoint)
                raise
            else:
                self._commit_transaction(depth, savepoint)
        finally:
            self._local.transaction_depth = depth
            self._lock.release()

    def record_term_chunk(self, chunk: ChunkRef) -> None:
        """Idempotently register a chunk manifest for an existing revision.

        The chunk file is written before this catalog call. A failed insert may
        therefore leave an unreferenced content object, which is safe to collect
        later; the catalog will never point to a partially written object.
        """

        if not isinstance(chunk, ChunkRef):
            raise TypeError("chunk must be a ChunkRef")
        with self.transaction():
            existing = self.fetch_one(
                """
                SELECT revision_id, first_index, term_count, last_index, codec,
                       format_version, uncompressed_bytes, payload_digest,
                       content_path, bundle_id, byte_offset, byte_length
                FROM term_chunk WHERE chunk_id = ?
                """,
                (chunk.chunk_id,),
            )
            expected = (
                chunk.revision_id,
                chunk.first_index,
                chunk.term_count,
                chunk.last_index,
                chunk.codec,
                chunk.format_version,
                chunk.uncompressed_bytes,
                chunk.payload_digest,
                chunk.content_path,
                chunk.bundle_id,
                chunk.byte_offset,
                chunk.byte_length,
            )
            if existing is not None:
                actual = tuple(existing)
                if actual != expected:
                    raise CatalogError(
                        "chunk_id already exists with different metadata",
                        context={"chunk_id": chunk.chunk_id},
                    )
                return

            try:
                self.execute(
                    """
                    INSERT INTO term_chunk (
                      chunk_id, revision_id, first_index, term_count, last_index,
                      codec, format_version, uncompressed_bytes, payload_digest,
                      content_path, bundle_id, byte_offset, byte_length
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.revision_id,
                        chunk.first_index,
                        chunk.term_count,
                        chunk.last_index,
                        chunk.codec,
                        chunk.format_version,
                        chunk.uncompressed_bytes,
                        chunk.payload_digest,
                        chunk.content_path,
                        chunk.bundle_id,
                        chunk.byte_offset,
                        chunk.byte_length,
                    ),
                )
            except CatalogError as exc:
                # The schema also prevents overlapping chunks beginning at the
                # same index. Make a retry idempotent but reject conflicts.
                existing = self.fetch_one(
                    """
                    SELECT chunk_id, term_count, payload_digest
                    FROM term_chunk WHERE revision_id = ? AND first_index = ?
                    """,
                    (chunk.revision_id, chunk.first_index),
                )
                if (
                    existing is not None
                    and existing["chunk_id"] == chunk.chunk_id
                    and existing["term_count"] == chunk.term_count
                    and existing["payload_digest"] == chunk.payload_digest
                ):
                    return
                raise exc

    def get_sequence_ref(self, revision_id: str) -> SequenceRef:
        """Return the full sequence reference for a stored revision."""

        row = self.fetch_one(
            """
            SELECT r.revision_id, r.snapshot_id, s.sequence_id
            FROM sequence_revision AS r
            JOIN sequence AS s ON s.sequence_id = r.sequence_id
            WHERE r.revision_id = ?
            """,
            (revision_id,),
        )
        if row is None:
            raise CatalogError(
                "sequence revision does not exist",
                context={"revision_id": revision_id},
            )
        return SequenceRef(
            sequence_id=row["sequence_id"],
            revision_id=row["revision_id"],
            snapshot_id=row["snapshot_id"],
        )

    def iter_sequence_refs(
        self,
        snapshot_id: str,
        *,
        page_size: int = 500,
    ) -> Iterator[SequenceRef]:
        """Yield sequence revisions in bounded keyset-paginated pages.

        Rows are ordered by the composite ``(sequence_id, revision_id)`` key,
        since a snapshot may contain more than one immutable revision for a
        sequence. Only one page is materialized at a time.
        """

        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string")
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("page_size must be a positive integer")

        last_sequence_id: str | None = None
        last_revision_id: str | None = None
        while True:
            if last_sequence_id is None:
                rows = self.fetch_all(
                    """
                    SELECT r.sequence_id, r.revision_id, r.snapshot_id
                    FROM sequence_revision AS r
                    WHERE r.snapshot_id = ?
                    ORDER BY r.sequence_id, r.revision_id
                    LIMIT ?
                    """,
                    (snapshot_id, page_size),
                )
            else:
                rows = self.fetch_all(
                    """
                    SELECT r.sequence_id, r.revision_id, r.snapshot_id
                    FROM sequence_revision AS r
                    WHERE r.snapshot_id = ?
                      AND (r.sequence_id > ?
                           OR (r.sequence_id = ? AND r.revision_id > ?))
                    ORDER BY r.sequence_id, r.revision_id
                    LIMIT ?
                    """,
                    (
                        snapshot_id,
                        last_sequence_id,
                        last_sequence_id,
                        last_revision_id,
                        page_size,
                    ),
                )
            if not rows:
                return
            for row in rows:
                last_sequence_id = row["sequence_id"]
                last_revision_id = row["revision_id"]
                yield SequenceRef(
                    sequence_id=row["sequence_id"],
                    revision_id=row["revision_id"],
                    snapshot_id=row["snapshot_id"],
                )

    def integrity_check(self) -> list[str]:
        """Run SQLite's integrity check and return every diagnostic line."""

        with self._lock:
            self._ensure_open()
            try:
                return [
                    str(row[0])
                    for row in self._connection.execute("PRAGMA integrity_check")
                ]
            except sqlite3.Error as exc:
                raise CatalogError(
                    "SQLite integrity check failed",
                    context={"error": str(exc)},
                ) from exc

    def close(self) -> None:
        """Close the connection; repeated calls are harmless."""

        with self._lock:
            if self._closed:
                return
            if getattr(self._local, "transaction_depth", 0):
                raise CatalogError("cannot close catalog inside a transaction")
            try:
                self._connection.close()
            except sqlite3.Error as exc:
                raise CatalogError(
                    "failed to close SQLite catalog",
                    context={"error": str(exc)},
                ) from exc
            finally:
                self._closed = True

    def __enter__(self) -> "Catalog":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _rollback_transaction(self, depth: int, savepoint: str) -> None:
        try:
            if depth == 0:
                self._connection.execute("ROLLBACK")
            else:
                self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        except sqlite3.Error as exc:
            raise CatalogError(
                "failed to roll back SQLite transaction",
                context={"error": str(exc)},
            ) from exc

    def _commit_transaction(self, depth: int, savepoint: str) -> None:
        try:
            if depth == 0:
                self._connection.execute("COMMIT")
            else:
                self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        except sqlite3.Error as exc:
            try:
                if depth == 0 and self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise CatalogError(
                "failed to commit SQLite transaction",
                context={"error": str(exc)},
            ) from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise CatalogError("catalog is closed", context={"path": self.path})
