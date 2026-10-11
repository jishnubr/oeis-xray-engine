"""Streaming ingestion and immutable revision management for OEIS records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator, TextIO

from ..errors import CatalogError, ValidationError
from ..models import (
    SequenceDescriptor,
    SequenceRef,
    SequenceRevision,
)
from ..storage.catalog import Catalog
from ..storage.term_store import TermStore

PARSER_VERSION = "oeis-seq-v2"
_METADATA_TAGS = frozenset({"%N", "%F", "%C", "%K", "%Y", "%H"})
_ANUMBER_RE = re.compile(r"^A\d{6}$", re.IGNORECASE)
_TAGGED_SEQUENCE_ID_RE = re.compile(r"^[A-Z][A-Z0-9_-]{0,63}$")
_MAX_DIAGNOSTICS = 100


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Bounded summary of an ingestion operation; contains no corpus rows."""

    snapshot_id: str
    source_digest: str
    records_seen: int
    added: int
    changed: int
    unchanged: int
    parsed: int
    partial: int
    invalid: int
    terms_ingested: int
    chunks_written: int
    diagnostics_digest: str


@dataclass(frozen=True, slots=True)
class Partition:
    """Deterministic keyset-pagination request for sequence revisions."""

    page_size: int = 500
    after_sequence_id: str | None = None
    start_sequence_id: str | None = None
    end_sequence_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        if self.start_sequence_id is not None and not isinstance(
            self.start_sequence_id, str
        ):
            raise ValueError("start_sequence_id must be a string or None")
        if self.end_sequence_id is not None and not isinstance(
            self.end_sequence_id, str
        ):
            raise ValueError("end_sequence_id must be a string or None")
        if (
            self.start_sequence_id is not None
            and self.end_sequence_id is not None
            and self.start_sequence_id > self.end_sequence_id
        ):
            raise ValueError(
                "start_sequence_id cannot be greater than end_sequence_id"
            )


@dataclass(slots=True)
class _ParsedRecord:
    sequence_id: str | None
    offset: int | None
    metadata: dict[str, list[str]]
    diagnostics: list[str]
    source_digest: str
    terms_file: TextIO
    term_digest: str
    term_count: int
    raw_size: int

    def close(self) -> None:
        self.terms_file.close()


class SequenceIndex:
    """Ingest OEIS ``.seq`` files or a raw stream of OEIS-format lines.

    Source files are visited in deterministic order. Directory ingestion hashes
    the source in a streaming first pass, then parses it in a second pass so
    immutable snapshot metadata is known before any revision rows are written.
    Each record's term values are staged in a bounded-memory temporary file;
    catalog writes and chunk registration happen atomically per record.
    """

    def __init__(
        self,
        catalog: Catalog,
        term_store: TermStore,
        *,
        chunk_terms: int = 4096,
        page_size: int = 500,
    ) -> None:
        if not isinstance(catalog, Catalog):
            raise TypeError("catalog must be a Catalog")
        if not isinstance(term_store, TermStore):
            raise TypeError("term_store must be a TermStore")
        if term_store.catalog is not catalog:
            raise ValueError("term store and sequence index must share a catalog")
        if type(chunk_terms) is not int or chunk_terms <= 0:
            raise ValueError("chunk_terms must be a positive integer")
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        self.catalog = catalog
        self.term_store = term_store
        self.chunk_terms = chunk_terms
        self.page_size = page_size

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        snapshot_id: str,
        source_revision: str,
        source_name: str = "oeis-seq-directory",
    ) -> IngestReport:
        """Ingest all ``*.seq`` files under ``directory`` without corpus loads."""

        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise ValidationError(
                "OEIS source directory does not exist",
                context={"directory": str(root)},
            )
        self._validate_snapshot_args(snapshot_id, source_revision, source_name)
        digest = hashlib.sha256()
        file_count = 0

        def records() -> Iterator[_ParsedRecord]:
            nonlocal file_count
            for path in self._iter_seq_files(root):
                relative = path.relative_to(root).as_posix().encode("utf-8")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                with path.open("rb") as source:
                    yield from self._iter_stream_records(
                        self._iter_hashed_lines(source, digest),
                        source_label=path.relative_to(root).as_posix(),
                    )
                file_count += 1
            digest.update(file_count.to_bytes(8, "big"))

        return self._ingest_records(
            records(),
            snapshot_id=snapshot_id,
            source_revision=source_revision,
            source_name=source_name,
            source_digest=None,
            source_digest_provider=digest.hexdigest,
        )

    def ingest_lines(
        self,
        lines: Iterable[str | bytes],
        *,
        snapshot_id: str,
        source_revision: str,
        source_name: str = "oeis-line-stream",
    ) -> IngestReport:
        """Ingest a raw stream of OEIS tagged lines using a disk-backed spool.

        The spool permits the complete source digest to be known before
        immutable snapshot and revision rows are created, while keeping memory
        bounded independently of corpus size.
        """

        self._validate_snapshot_args(snapshot_id, source_revision, source_name)
        digest = hashlib.sha256()
        with tempfile.TemporaryFile(mode="w+b") as staged:
            for line_number, line in enumerate(lines, start=1):
                if isinstance(line, str):
                    try:
                        encoded = line.encode("utf-8")
                    except UnicodeEncodeError as exc:
                        raise ValidationError(
                            "input line cannot be encoded as UTF-8",
                            context={"line_number": line_number},
                        ) from exc
                elif isinstance(line, bytes):
                    encoded = line
                else:
                    raise TypeError("line stream must yield str or bytes")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                staged.write(encoded)
            staged.flush()
            staged.seek(0)
            records = self._iter_stream_records(staged, source_label="<line-stream>")
            return self._ingest_records(
                records,
                snapshot_id=snapshot_id,
                source_revision=source_revision,
                source_name=source_name,
                source_digest=digest.hexdigest(),
            )

    def iter_sequences(
        self,
        snapshot_id: str,
        partition: Partition | None = None,
    ) -> Iterator[SequenceDescriptor]:
        """Yield descriptors in bounded keyset pages for one snapshot."""

        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        page = partition or Partition(page_size=self.page_size)
        last_id = page.after_sequence_id
        first_page = True
        while True:
            conditions = ["r.snapshot_id = ?"]
            params: list[object] = [snapshot_id]

            if first_page:
                if page.after_sequence_id is not None:
                    conditions.append("s.sequence_id > ?")
                    params.append(page.after_sequence_id)
                elif page.start_sequence_id is not None:
                    conditions.append("s.sequence_id >= ?")
                    params.append(page.start_sequence_id)
            else:
                conditions.append("s.sequence_id > ?")
                params.append(last_id)

            if page.end_sequence_id is not None:
                conditions.append("s.sequence_id <= ?")
                params.append(page.end_sequence_id)

            where_clause = " AND ".join(conditions)
            query = f"""
                SELECT s.sequence_id, r.revision_id, r.snapshot_id, r.offset,
                       r.term_count, r.known_end_index, r.completeness,
                       r.term_digest, r.source_digest, r.parse_status
                FROM sequence_revision AS r
                JOIN sequence AS s ON s.sequence_id = r.sequence_id
                WHERE {where_clause}
                ORDER BY s.sequence_id
                LIMIT ?
            """
            params.append(page.page_size)
            rows = self.catalog.fetch_all(query, tuple(params))
            first_page = False
            if not rows:
                return
            for row in rows:
                last_id = row["sequence_id"]
                yield SequenceDescriptor(
                    ref=SequenceRef(
                        row["sequence_id"], row["revision_id"], row["snapshot_id"]
                    ),
                    offset=row["offset"],
                    term_count=row["term_count"],
                    known_end_index=row["known_end_index"],
                    completeness=row["completeness"],
                    term_digest=row["term_digest"] or "",
                    source_digest=row["source_digest"],
                    parse_status=row["parse_status"],
                )

    def get_revision(self, ref: SequenceRef) -> SequenceRevision:
        """Return one immutable revision from the catalog."""

        row = self.catalog.fetch_one(
            """
            SELECT r.* FROM sequence_revision AS r
            JOIN sequence AS s ON s.sequence_id = r.sequence_id
            WHERE r.revision_id = ? AND r.sequence_id = ? AND r.snapshot_id = ?
            """,
            (ref.revision_id, ref.sequence_id, ref.snapshot_id),
        )
        if row is None:
            raise CatalogError(
                "sequence revision does not exist",
                context={"revision_id": ref.revision_id},
            )
        return SequenceRevision(
            ref=ref,
            source_digest=row["source_digest"],
            term_digest=row["term_digest"],
            offset=row["offset"],
            term_count=row["term_count"],
            known_end_index=row["known_end_index"],
            completeness=row["completeness"],
            parse_status=row["parse_status"],
            raw_metadata_json=row["raw_metadata_json"],
            diagnostics_json=row["diagnostics_json"],
        )

    def _ingest_records(
        self,
        records: Iterable[_ParsedRecord],
        *,
        snapshot_id: str,
        source_revision: str,
        source_name: str,
        source_digest: str | None,
        source_digest_provider: Callable[[], str] | None = None,
    ) -> IngestReport:
        deferred_digest = source_digest is None
        if deferred_digest:
            source_digest = "pending:" + hashlib.sha256(
                json.dumps(
                    [source_name, source_revision, PARSER_VERSION],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        snapshot_state, stored_digest = self._ensure_snapshot(
            snapshot_id=snapshot_id,
            source_revision=source_revision,
            source_name=source_name,
            source_digest=source_digest,
            allow_deferred_digest=deferred_digest,
        )
        counts = {
            "records_seen": 0,
            "added": 0,
            "changed": 0,
            "unchanged": 0,
            "parsed": 0,
            "partial": 0,
            "invalid": 0,
            "terms_ingested": 0,
            "chunks_written": 0,
        }
        diagnostics_hash = hashlib.sha256()
        try:
            for record in records:
                try:
                    counts["records_seen"] += 1
                    outcome, parse_status, term_count, chunk_count = self._persist_record(
                        record,
                        snapshot_id,
                        read_only_snapshot=snapshot_state == "ready",
                    )
                    counts[outcome] += 1
                    counts[
                        "parsed" if parse_status == "valid" else parse_status
                    ] += 1
                    counts["terms_ingested"] += term_count
                    counts["chunks_written"] += chunk_count
                    diagnostics_hash.update(
                        json.dumps(
                            [
                                record.sequence_id,
                                record.source_digest,
                                record.diagnostics,
                            ],
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    if record.diagnostics:
                        diagnostics_hash.update(b"\n")
                finally:
                    record.close()
            with self.catalog.transaction():
                self.catalog.execute(
                    "UPDATE corpus_snapshot SET state = 'ready' WHERE snapshot_id = ?",
                    (snapshot_id,),
                )
        except BaseException:
            # Keep an ingesting snapshot resumable. Individual revision writes
            # are atomic and content-addressed file leftovers are unreferenced.
            close = getattr(records, "close", None)
            if close is not None:
                close()
            raise
        completed_digest = (
            source_digest_provider()
            if source_digest_provider is not None
            else source_digest
        )
        if completed_digest is None:
            raise CatalogError("ingestion completed without a source digest")
        if snapshot_state == "ready":
            if stored_digest != completed_digest:
                raise CatalogError(
            "snapshot source content differs from the immutable snapshot",
            context={"snapshot_id": snapshot_id},
                )
        else:
            if (
                not stored_digest.startswith("pending:")
                and stored_digest != completed_digest
            ):
                raise CatalogError(
            "resumed source content differs from the incomplete snapshot",
            context={"snapshot_id": snapshot_id},
                )
            with self.catalog.transaction():
                self.catalog.execute(
            """
            UPDATE corpus_snapshot
            SET source_digest = ?, state = 'ready'
            WHERE snapshot_id = ?
            """,
            (completed_digest, snapshot_id),
                )
        return IngestReport(
            snapshot_id=snapshot_id,
            source_digest=completed_digest,
            diagnostics_digest=diagnostics_hash.hexdigest(),
            **counts,
        )

    def _persist_record(
        self,
        record: _ParsedRecord,
        snapshot_id: str,
        *,
        read_only_snapshot: bool = False,
    ) -> tuple[str, str, int, int]:
        diagnostics = list(record.diagnostics)
        sequence_id = record.sequence_id
        parse_status = "valid"
        if sequence_id is None:
            sequence_id = f"INVALID-{record.source_digest[:24].upper()}"
            parse_status = "invalid"
            diagnostics.append("Missing or invalid %I sequence identifier.")
        if record.offset is None and record.term_count:
            parse_status = "partial" if parse_status != "invalid" else "invalid"
            diagnostics.append("Terms were not indexed: no valid %O offset was supplied.")
        elif diagnostics and parse_status == "valid":
            parse_status = "partial"

        indexed_term_count = (
            record.term_count
            if record.offset is not None and parse_status == "valid"
            else 0
        )
        indexed_term_digest = record.term_digest if indexed_term_count else None
        known_end = (
            record.offset + indexed_term_count - 1 if indexed_term_count else None
        )
        revision_id = hashlib.sha256(
            json.dumps(
                [
                    sequence_id,
                    snapshot_id,
                    record.source_digest,
                    PARSER_VERSION,
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        ref = SequenceRef(sequence_id, revision_id, snapshot_id)
        stored_revision = self.catalog.fetch_one(
            """
            SELECT source_digest, term_digest
            FROM sequence_revision WHERE revision_id = ?
            """,
            (revision_id,),
        )
        if stored_revision is not None:
            if (
                stored_revision["source_digest"] != record.source_digest
                or stored_revision["term_digest"] != indexed_term_digest
            ):
                raise CatalogError(
                    "revision identity conflicts with persisted content",
                    context={"revision_id": revision_id},
                )
            return "unchanged", parse_status, indexed_term_count, 0
        if read_only_snapshot:
            raise CatalogError(
                "ready snapshot is missing an ingested sequence revision",
                context={
                    "snapshot_id": snapshot_id,
                    "sequence_id": sequence_id,
                },
            )
        conflicting_revision = self.catalog.fetch_one(
            """
            SELECT revision_id FROM sequence_revision
            WHERE snapshot_id = ? AND sequence_id = ?
            LIMIT 1
            """,
            (snapshot_id, sequence_id),
        )
        if conflicting_revision is not None:
            raise CatalogError(
                "immutable snapshot contains conflicting revisions for a sequence",
                context={
                    "snapshot_id": snapshot_id,
                    "sequence_id": sequence_id,
                },
            )
        revision = SequenceRevision(
            ref=ref,
            source_digest=record.source_digest,
            term_digest=indexed_term_digest,
            offset=record.offset,
            term_count=indexed_term_count,
            known_end_index=known_end,
            completeness="prefix",
            parse_status=parse_status,
            raw_metadata_json=json.dumps(
                record.metadata, sort_keys=True, separators=(",", ":")
            ),
            diagnostics_json=json.dumps(
                diagnostics, ensure_ascii=True, separators=(",", ":")
            ),
        )

        previous = self.catalog.fetch_one(
            "SELECT current_revision_id FROM sequence WHERE sequence_id = ?",
            (sequence_id,),
        )
        previous_revision = previous["current_revision_id"] if previous else None
        if previous_revision == revision_id:
            return "unchanged", parse_status, indexed_term_count, 0

        chunks = []
        if indexed_term_count:
            record.terms_file.flush()
            record.terms_file.seek(0)
            position = 0
            while position < indexed_term_count:
                values = []
                for _ in range(min(self.chunk_terms, indexed_term_count - position)):
                    line = record.terms_file.readline()
                    if not line:
                        raise ValidationError(
                            "staged term stream ended before its declared count",
                            context={"sequence_id": sequence_id},
                        )
                    try:
                        values.append(_parse_integer(line.rstrip("\n")))
                    except ValueError as exc:
                        raise ValidationError(
                            "staged term stream contains an invalid integer",
                            context={"sequence_id": sequence_id, "line": line},
                        ) from exc
                chunk = self.term_store.put_chunk(
                    ref, record.offset + position, values, register=False
                )
                chunks.append(chunk)
                position += len(values)

        with self.catalog.transaction():
            self.catalog.execute(
                """
                INSERT INTO sequence (sequence_id, current_revision_id, first_seen_snapshot)
                VALUES (?, ?, ?)
                ON CONFLICT(sequence_id) DO NOTHING
                """,
                (sequence_id, revision_id, snapshot_id),
            )
            existing = self.catalog.fetch_one(
                "SELECT * FROM sequence_revision WHERE revision_id = ?",
                (revision_id,),
            )
            if existing is None:
                self.catalog.execute(
                    """
                    INSERT INTO sequence_revision (
                      revision_id, sequence_id, snapshot_id, source_digest,
                      term_digest, offset, term_count, known_end_index,
                      completeness, parse_status, raw_metadata_json,
                      diagnostics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision.ref.revision_id,
                        sequence_id,
                        snapshot_id,
                        revision.source_digest,
                        revision.term_digest,
                        revision.offset,
                        revision.term_count,
                        revision.known_end_index,
                        revision.completeness,
                        revision.parse_status,
                        revision.raw_metadata_json,
                        revision.diagnostics_json,
                    ),
                )
            elif (
                existing["source_digest"] != revision.source_digest
                or existing["term_digest"] != revision.term_digest
            ):
                raise CatalogError(
                    "revision identity conflicts with persisted content",
                    context={"revision_id": revision_id},
                )
            for chunk in chunks:
                self.catalog.record_term_chunk(chunk)
            self.catalog.execute(
                "UPDATE sequence SET current_revision_id = ? WHERE sequence_id = ?",
                (revision_id, sequence_id),
            )

        outcome = "added" if previous_revision is None else "changed"
        return outcome, parse_status, indexed_term_count, len(chunks)

    def _ensure_snapshot(
        self,
        *,
        snapshot_id: str,
        source_revision: str,
        source_name: str,
        source_digest: str,
        allow_deferred_digest: bool = False,
    ) -> tuple[str, str]:
        now = datetime.now(timezone.utc).isoformat()
        with self.catalog.transaction():
            existing = self.catalog.fetch_one(
                "SELECT * FROM corpus_snapshot WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            if existing is not None:
                expected_metadata = (source_name, source_revision, PARSER_VERSION)
                actual_metadata = (
                    existing["source_name"],
                    existing["source_revision"],
                    existing["parser_version"],
                )
                if actual_metadata != expected_metadata:
                    raise CatalogError(
                        "snapshot_id already exists with different source metadata",
                        context={"snapshot_id": snapshot_id},
                    )
                if not allow_deferred_digest and existing["source_digest"] != source_digest:
                    raise CatalogError(
                        "snapshot_id already exists with different source metadata",
                        context={"snapshot_id": snapshot_id},
                    )
                return existing["state"], existing["source_digest"]
            self.catalog.execute(
                """
                INSERT INTO corpus_snapshot (
                  snapshot_id, source_name, source_revision, source_digest,
                  created_at, parser_version, state
                ) VALUES (?, ?, ?, ?, ?, ?, 'ingesting')
                """,
                (
                    snapshot_id,
                    source_name,
                    source_revision,
                    source_digest,
                    now,
                    PARSER_VERSION,
                ),
            )
        return "ingesting", source_digest

    @staticmethod
    def _validate_snapshot_args(
        snapshot_id: str,
        source_revision: str,
        source_name: str,
    ) -> None:
        for name, value in (
            ("snapshot_id", snapshot_id),
            ("source_revision", source_revision),
            ("source_name", source_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @staticmethod
    def _digest_directory(root: Path) -> str:
        digest = hashlib.sha256()
        file_count = 0
        for path in SequenceIndex._iter_seq_files(root):
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with path.open("rb") as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(len(block).to_bytes(8, "big"))
                    digest.update(block)
            file_count += 1
        digest.update(file_count.to_bytes(8, "big"))
        return digest.hexdigest()

    def _iter_directory_records(self, root: Path) -> Iterator[_ParsedRecord]:
        for path in self._iter_seq_files(root):
            with path.open("rb") as source:
                yield from self._iter_stream_records(
                    source,
                    source_label=path.relative_to(root).as_posix(),
                )

    @staticmethod
    def _iter_hashed_lines(
        source: BinaryIO,
        digest: "hashlib._Hash",
        *,
        block_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        """Yield exact source lines while hashing the original bounded blocks."""

        buffered = bytearray()
        while block := source.read(block_size):
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
            buffered.extend(block)
            start = 0
            while True:
                end = buffered.find(b"\n", start)
                if end < 0:
                    break
                yield bytes(buffered[start : end + 1])
                start = end + 1
            if start:
                del buffered[:start]
        if buffered:
            yield bytes(buffered)

    @staticmethod
    def _iter_seq_files(root: Path) -> Iterator[Path]:
        for current, directories, filenames in os.walk(root):
            directories.sort()
            for filename in sorted(filenames):
                if filename.lower().endswith(".seq"):
                    yield Path(current) / filename

    def _iter_stream_records(
        self,
        source: BinaryIO,
        *,
        source_label: str,
    ) -> Iterator[_ParsedRecord]:
        builder: _RecordBuilder | None = None
        for line_number, raw_line in enumerate(source, start=1):
            try:
                line = raw_line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                if builder is None:
                    builder = _RecordBuilder(source_label)
                builder.add_raw(raw_line)
                builder.add_diagnostic(
                    f"Line {line_number}: invalid UTF-8 in source record."
                )
                continue
            tag = line[:2] if len(line) >= 2 and line[0] == "%" else None
            if (
                builder is not None
                and builder.sequence_id is None
                and not builder.has_recognized_content
                and tag != "%I"
            ):
                builder.close()
                builder = None
            if tag == "%I" and builder is not None and builder.has_content:
                if builder.has_recognized_content:
                    yield builder.finish()
                else:
                    builder.close()
                builder = None
            if builder is None:
                builder = _RecordBuilder(source_label)
            builder.add_line(raw_line, line, line_number)
        if (
            builder is not None
            and builder.has_content
            and builder.has_recognized_content
        ):
            yield builder.finish()
        elif builder is not None:
            builder.close()


class _RecordBuilder:
    """Bounded-memory builder for one tagged source record."""

    def __init__(self, source_label: str) -> None:
        self.source_label = source_label
        self.digest = hashlib.sha256()
        self.term_digest = hashlib.sha256()
        self.terms_file = tempfile.TemporaryFile(
            mode="w+t", encoding="ascii", newline="\n"
        )
        self.sequence_id: str | None = None
        self.offset: int | None = None
        self.metadata: dict[str, list[str]] = {}
        self.diagnostics: list[str] = []
        self.suppressed_diagnostics = 0
        self.term_count = 0
        self.has_content = False
        self.has_recognized_content = False

    def add_raw(self, raw_line: bytes) -> None:
        self.digest.update(len(raw_line).to_bytes(8, "big"))
        self.digest.update(raw_line)
        self.has_content = self.has_content or bool(raw_line.strip())

    def add_line(self, raw_line: bytes, line: str, line_number: int) -> None:
        self.add_raw(raw_line)
        if len(line) < 2 or not line.startswith("%"):
            return
        tag = line[:2]
        if tag not in {"%I", "%S", "%T", "%U", "%O"} | _METADATA_TAGS:
            return
        self.has_recognized_content = True
        payload = line[2:].strip()
        sequence_id, body = _split_identifier(payload)
        if tag == "%I":
            if sequence_id is None:
                self.add_diagnostic(
                    f"Line {line_number}: malformed %I identifier."
                )
            elif not _valid_sequence_id(sequence_id):
                self.add_diagnostic(
                    f"Line {line_number}: invalid sequence identifier {sequence_id!r}."
                )
            elif self.sequence_id is None:
                self.sequence_id = sequence_id.upper()
            elif self.sequence_id != sequence_id.upper():
                self.add_diagnostic(
                    f"Line {line_number}: conflicting sequence ID {sequence_id!r}."
                )
            self.metadata.setdefault("%I", []).append(payload)
            return

        if sequence_id is None:
            self.add_diagnostic(
                f"Line {line_number}: {tag} line is missing its sequence identifier."
            )
            return
        if self.sequence_id is None:
            self.sequence_id = sequence_id.upper()
        elif self.sequence_id != sequence_id.upper():
            self.add_diagnostic(
                f"Line {line_number}: {tag} identifier differs from %I identifier."
            )
            return

        if tag == "%O":
            self.metadata.setdefault("%O", []).append(payload)
            offset_text = body.split(",", 1)[0].strip()
            try:
                parsed_offset = int(offset_text, 10)
            except ValueError:
                self.add_diagnostic(f"Line {line_number}: malformed %O offset.")
            else:
                if self.offset is None:
                    self.offset = parsed_offset
                elif self.offset != parsed_offset:
                    self.add_diagnostic(
                        f"Line {line_number}: conflicting %O offsets."
                    )
            return

        if tag in _METADATA_TAGS:
            self.metadata.setdefault(tag, []).append(payload)
            return

        if not body:
            return
        for token in _csv_tokens(body):
            value_text = token.strip()
            if not value_text:
                continue
            try:
                value = _parse_integer(value_text)
            except ValueError:
                self.add_diagnostic(
                    f"Line {line_number}: malformed integer in {tag} data."
                )
                continue
            canonical = _canonical_integer(value)
            self.terms_file.write(f"{canonical}\n")
            encoded = canonical.encode("ascii")
            self.term_digest.update(len(encoded).to_bytes(8, "big"))
            self.term_digest.update(encoded)
            self.term_count += 1

    def finish(self) -> _ParsedRecord:
        self.terms_file.flush()
        self.terms_file.seek(0)
        if self.suppressed_diagnostics:
            self.diagnostics.append(
                f"{self.suppressed_diagnostics} additional diagnostics suppressed."
            )
        return _ParsedRecord(
            sequence_id=self.sequence_id,
            offset=self.offset,
            metadata=self.metadata,
            diagnostics=self.diagnostics,
            source_digest=self.digest.hexdigest(),
            terms_file=self.terms_file,
            term_digest=self.term_digest.hexdigest(),
            term_count=self.term_count,
            raw_size=0,
        )

    def close(self) -> None:
        self.terms_file.close()

    def add_diagnostic(self, message: str) -> None:
        if len(self.diagnostics) < _MAX_DIAGNOSTICS:
            self.diagnostics.append(message)
        else:
            self.suppressed_diagnostics += 1


def _split_identifier(payload: str) -> tuple[str | None, str]:
    fields = payload.split(None, 1)
    if not fields:
        return None, ""
    identifier = fields[0]
    if not _valid_sequence_id(identifier):
        return None, payload
    return identifier.upper(), fields[1] if len(fields) > 1 else ""


def _valid_sequence_id(value: str) -> bool:
    return bool(_ANUMBER_RE.fullmatch(value) or _TAGGED_SEQUENCE_ID_RE.fullmatch(value))


def _canonical_integer(value: int) -> str:
    """Render a Python integer without relying on the decimal digit limit."""

    if value == 0:
        return "0"
    negative = value < 0
    magnitude = -value if negative else value
    base = 1_000_000_000
    groups: list[int] = []
    while magnitude:
        magnitude, remainder = divmod(magnitude, base)
        groups.append(remainder)
    parts = [str(groups[-1])]
    parts.extend(f"{group:09d}" for group in reversed(groups[:-1]))
    text = "".join(parts)
    return f"-{text}" if negative else text


def _parse_integer(value: str) -> int:
    """Parse signed decimal integers without Python's string digit limit."""

    if not value:
        raise ValueError("empty integer")
    negative = value[0] == "-"
    digits = value[1:] if value[0] in "+-" else value
    if not digits or any(character < "0" or character > "9" for character in digits):
        raise ValueError("invalid decimal integer")
    first_width = len(digits) % 9 or 9
    result = int(digits[:first_width])
    for start in range(first_width, len(digits), 9):
        result = result * 1_000_000_000 + int(digits[start : start + 9])
    return -result if negative else result


def _csv_tokens(value: str) -> Iterator[str]:
    """Yield comma-separated tokens without constructing a token list."""

    start = 0
    while True:
        end = value.find(",", start)
        if end < 0:
            yield value[start:]
            return
        yield value[start:end]
        start = end + 1
