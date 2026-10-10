"""Immutable bundled storage for arbitrary-precision term chunks.

Each chunk is a canonical JSON document compressed with deterministic gzip and
appended to a size-bounded bundle. Integers are encoded as decimal strings,
avoiding JSON consumer precision loss. Catalog references retain the bundle
byte range and the SHA-256 digest of the uncompressed canonical document.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from ..errors import CatalogError, CorruptChunkError, TermStoreError
from ..models import ChunkRef, IntegrityReport, SequenceRef, TermSeries
from .catalog import Catalog

_FORMAT_VERSION = 1
_CODEC = "gzip-json"
_JSON_PREFIX = b'{"format_version":1,"start_index":'
_JSON_MIDDLE = b',"values":['
_JSON_SUFFIX = b"]}"
_DECIMAL_CHUNK_BASE = 1_000_000_000
_DECIMAL_CHUNK_WIDTH = 9
_BUNDLE_MAX_BYTES = 100 * 1024 * 1024


class TermStore:
    """Store and retrieve immutable sequence-term chunks in append-only bundles.

    Args:
        root: Directory for append-only bundle files and legacy chunk payloads.
        catalog: Optional coordinator-owned catalog. When supplied, writes
            register chunk metadata and ``read_range`` can resolve chunks by
            revision. Catalog mutations remain serialized by :class:`Catalog`.

    Chunk inputs should be bounded by the caller (for example, a few thousand
    terms). The implementation streams JSON and gzip output, then appends each
    compressed payload to a bundle under an inter-process write lock.
    """

    def __init__(
        self,
        root: str | Path,
        catalog: Catalog | None = None,
        *,
        max_uncompressed_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if type(max_uncompressed_bytes) is not int or max_uncompressed_bytes <= 0:
            raise ValueError("max_uncompressed_bytes must be a positive integer")
        self.root = Path(root).expanduser().resolve()
        self.catalog = catalog
        self.max_uncompressed_bytes = max_uncompressed_bytes
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TermStoreError(
                "failed to create term-store directory",
                context={"path": str(self.root), "error": str(exc)},
            ) from exc

    def put_chunk(
        self,
        ref: SequenceRef,
        first_index: int,
        values: Iterable[int],
        *,
        register: bool = True,
    ) -> ChunkRef:
        """Persist consecutive terms and return their immutable catalog ref.

        ``values`` must be non-empty and contain exact Python integers (not
        booleans). The chunk's indices are consecutive from ``first_index``;
        sparse source data should be split into contiguous runs or represented
        at a higher layer with explicit sparse-index metadata.
        """

        if not isinstance(ref, SequenceRef):
            raise TypeError("ref must be a SequenceRef")
        if type(first_index) is not int:
            raise TypeError("first_index must be an integer")

        self._validate_revision_id(ref.revision_id)
        digest = hashlib.sha256()
        count = 0
        uncompressed_bytes = 0

        try:
            with tempfile.SpooledTemporaryFile(
                max_size=1024 * 1024,
                mode="w+b",
                dir=self.root,
            ) as temp_file:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=temp_file,
                    compresslevel=6,
                    mtime=0,
                ) as compressed:
                    uncompressed_bytes += self._write_hashed(
                        compressed, digest, _JSON_PREFIX
                    )
                    uncompressed_bytes += self._write_hashed(
                        compressed,
                        digest,
                        str(first_index).encode("ascii"),
                    )
                    uncompressed_bytes += self._write_hashed(
                        compressed, digest, _JSON_MIDDLE
                    )
                    first = True
                    for position, value in enumerate(values):
                        if type(value) is not int:
                            raise TypeError(
                                f"values[{position}] must be an integer, not "
                                f"{type(value).__name__}"
                            )
                        if not first:
                            uncompressed_bytes += self._write_hashed(
                                compressed, digest, b","
                            )
                        uncompressed_bytes += self._write_hashed(
                            compressed,
                            digest,
                            json.dumps(
                                self._int_to_decimal(value), ensure_ascii=True
                            ).encode("ascii"),
                        )
                        count += 1
                        first = False
                    if count == 0:
                        raise ValueError("term chunks cannot be empty")
                    uncompressed_bytes += self._write_hashed(
                        compressed, digest, _JSON_SUFFIX
                    )
                    if uncompressed_bytes > self.max_uncompressed_bytes:
                        raise TermStoreError(
                            "term chunk exceeds configured uncompressed size limit",
                            context={
                                "limit": self.max_uncompressed_bytes,
                                "actual": uncompressed_bytes,
                            },
                        )

                temp_file.flush()
                temp_file.seek(0)
                bundle_id, byte_offset, byte_length = self._append_bundle(
                    temp_file
                )
        except (OSError, gzip.BadGzipFile) as exc:
            raise TermStoreError(
                "failed to write term chunk",
                context={"revision_id": ref.revision_id, "error": str(exc)},
            ) from exc
        payload_digest = digest.hexdigest()
        chunk_id = self._chunk_id(
            ref.revision_id, first_index, count, payload_digest
        )

        chunk = ChunkRef(
            chunk_id=chunk_id,
            revision_id=ref.revision_id,
            first_index=first_index,
            term_count=count,
            last_index=first_index + count - 1,
            codec=_CODEC,
            format_version=_FORMAT_VERSION,
            uncompressed_bytes=uncompressed_bytes,
            payload_digest=payload_digest,
            content_path="",
            bundle_id=bundle_id,
            byte_offset=byte_offset,
            byte_length=byte_length,
        )
        if register and self.catalog is not None:
            try:
                self.catalog.record_term_chunk(chunk)
            except CatalogError:
                # The payload is immutable and unreferenced; a later garbage
                # collector may remove it if no catalog row points to it.
                raise
        return chunk

    def read_chunk(
        self, chunk: ChunkRef, ref: SequenceRef | None = None
    ) -> TermSeries:
        """Read and fully validate one chunk as a dense ``TermSeries``."""

        self._validate_chunk_ref(chunk)
        if ref is None:
            if self.catalog is None:
                raise TermStoreError(
                    "read_chunk requires a SequenceRef or a catalog"
                )
            ref = self.catalog.get_sequence_ref(chunk.revision_id)
        if ref.revision_id != chunk.revision_id:
            raise TermStoreError(
                "chunk revision does not match the requested sequence reference",
                context={
                    "chunk_revision_id": chunk.revision_id,
                    "requested_revision_id": ref.revision_id,
                },
            )
        payload = self._read_chunk_payload(chunk)
        if len(payload) != chunk.uncompressed_bytes:
            raise CorruptChunkError(
                "chunk uncompressed size does not match its manifest",
                context={
                    "chunk_id": chunk.chunk_id,
                    "expected": chunk.uncompressed_bytes,
                    "actual": len(payload),
                },
            )

        document = self._decode_document(payload, chunk)
        values = tuple(self._parse_decimal(value) for value in document["values"])
        if len(values) != chunk.term_count:
            raise CorruptChunkError(
                "chunk term count does not match its manifest",
                context={
                    "chunk_id": chunk.chunk_id,
                    "expected": chunk.term_count,
                    "actual": len(values),
                },
            )
        if document["start_index"] != chunk.first_index:
            raise CorruptChunkError(
                "chunk start index does not match its manifest",
                context={"chunk_id": chunk.chunk_id},
            )

        return TermSeries(
            ref=ref,
            start_index=chunk.first_index,
            values=values,
            complete_through=chunk.last_index,
        )

    def read_range(
        self,
        ref: SequenceRef,
        start_index: int,
        count: int,
    ) -> TermSeries:
        """Read a known index range, preserving missing indices explicitly.

        Requires a catalog supplied to the constructor. ``count`` is the index
        range width, not the number of present terms. Any absent index in the
        requested range is represented by omission from ``TermSeries.indices``.
        """

        if self.catalog is None:
            raise TermStoreError(
                "read_range requires a catalog to resolve revision chunks"
            )
        if not isinstance(ref, SequenceRef):
            raise TypeError("ref must be a SequenceRef")
        if type(start_index) is not int:
            raise TypeError("start_index must be an integer")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        end_index = start_index + count - 1

        rows = self.catalog.fetch_all(
            """
            SELECT chunk_id, revision_id, first_index, term_count, last_index,
                   codec, format_version, uncompressed_bytes, payload_digest,
                   content_path, bundle_id, byte_offset, byte_length
            FROM term_chunk
            WHERE revision_id = ? AND first_index <= ? AND last_index >= ?
            ORDER BY first_index
            """,
            (ref.revision_id, end_index, start_index),
        )
        chunks = [self._chunk_from_row(row) for row in rows]
        indexed_values: list[tuple[int, int]] = []
        for chunk in chunks:
            series = self.read_chunk(chunk, ref=ref)
            for index, value in series.iter_indexed():
                if start_index <= index <= end_index:
                    indexed_values.append((index, value))

        indexed_values.sort(key=lambda item: item[0])
        for previous, current in zip(indexed_values, indexed_values[1:]):
            if previous[0] == current[0]:
                raise CorruptChunkError(
                    "overlapping chunks contain the same term index",
                    context={
                        "revision_id": ref.revision_id,
                        "index": current[0],
                    },
                )

        return TermSeries(
            ref=ref,
            start_index=start_index,
            values=tuple(value for _, value in indexed_values),
            complete_through=end_index,
            indices=tuple(index for index, _ in indexed_values),
        )

    def verify(self, chunk: ChunkRef) -> IntegrityReport:
        """Validate the payload and return a structured integrity report."""

        try:
            self._validate_chunk_ref(chunk)
            payload = self._read_chunk_payload(chunk)
            if len(payload) != chunk.uncompressed_bytes:
                raise CorruptChunkError(
                    "chunk uncompressed size does not match its manifest",
                    context={"chunk_id": chunk.chunk_id},
                )
            document = self._decode_document(payload, chunk)
            actual_count = len(document["values"])
            if actual_count != chunk.term_count:
                raise CorruptChunkError(
                    "chunk term count does not match its manifest",
                    context={"chunk_id": chunk.chunk_id},
                )
            if document["start_index"] != chunk.first_index:
                raise CorruptChunkError(
                    "chunk start index does not match its manifest",
                    context={"chunk_id": chunk.chunk_id},
                )
        except (TermStoreError, OSError, ValueError) as exc:
            actual_digest: str | None = None
            try:
                payload = self._read_chunk_payload(
                    chunk,
                    expected_digest=None,
                    verify_manifest_digest=False,
                )
                actual_digest = hashlib.sha256(payload).hexdigest()
            except (TermStoreError, OSError):
                pass
            return IntegrityReport(
                valid=False,
                chunk_id=chunk.chunk_id,
                payload_digest=chunk.payload_digest,
                actual_digest=actual_digest,
                term_count=None,
                error=str(exc),
            )

        return IntegrityReport(
            valid=True,
            chunk_id=chunk.chunk_id,
            payload_digest=chunk.payload_digest,
            actual_digest=chunk.payload_digest,
            term_count=actual_count,
        )

    def iter_chunks(self, ref: SequenceRef) -> Iterator[ChunkRef]:
        """Yield a revision's cataloged chunks in increasing index order."""

        if self.catalog is None:
            raise TermStoreError("iter_chunks requires a catalog")
        rows = self.catalog.fetch_all(
            """
            SELECT chunk_id, revision_id, first_index, term_count, last_index,
                   codec, format_version, uncompressed_bytes, payload_digest,
                   content_path, bundle_id, byte_offset, byte_length
            FROM term_chunk WHERE revision_id = ? ORDER BY first_index
            """,
            (ref.revision_id,),
        )
        for row in rows:
            yield self._chunk_from_row(row)

    def _read_chunk_payload(
        self,
        chunk: ChunkRef,
        *,
        expected_digest: str | None = None,
        verify_manifest_digest: bool = True,
    ) -> bytes:
        digest = chunk.payload_digest if verify_manifest_digest else expected_digest
        if chunk.bundle_id is None:
            return self._read_payload(
                self._resolve_content_path(chunk),
                digest,
            )

        bundle_path = self._resolve_bundle_path(chunk)
        byte_offset = chunk.byte_offset
        byte_length = chunk.byte_length
        if byte_offset is None or byte_length is None:
            raise CorruptChunkError(
                "bundled chunk is missing its byte range",
                context={"chunk_id": chunk.chunk_id},
            )
        marker_temp_path: Path | None = None
        try:
            with bundle_path.open("rb") as bundle:
                bundle.seek(byte_offset)
                compressed_payload = bundle.read(byte_length)
        except OSError as exc:
            raise CorruptChunkError(
                "unable to read bundled term chunk",
                context={"chunk_id": chunk.chunk_id, "error": str(exc)},
            ) from exc
        if len(compressed_payload) != byte_length:
            raise CorruptChunkError(
                "bundled term chunk is shorter than its manifest",
                context={
                    "chunk_id": chunk.chunk_id,
                    "expected_bytes": byte_length,
                    "actual_bytes": len(compressed_payload),
                },
            )
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(compressed_payload), mode="rb") as compressed:
                payload = compressed.read(self.max_uncompressed_bytes + 1)
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise CorruptChunkError(
                "unable to decompress bundled term chunk",
                context={"chunk_id": chunk.chunk_id, "error": str(exc)},
            ) from exc
        return self._validate_payload_digest(
            payload,
            digest,
            f"{chunk.bundle_id}@{byte_offset}",
        )

    def _read_payload(self, path: Path, expected_digest: str | None) -> bytes:
        try:
            with gzip.open(path, "rb") as compressed:
                payload = compressed.read(self.max_uncompressed_bytes + 1)
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise CorruptChunkError(
                "unable to decompress term chunk",
                context={"path": str(path), "error": str(exc)},
            ) from exc

        if len(payload) > self.max_uncompressed_bytes:
            raise CorruptChunkError(
                "term chunk exceeds configured uncompressed size limit",
                context={
                    "path": str(path),
                    "limit": self.max_uncompressed_bytes,
                },
            )
        return self._validate_payload_digest(
            payload,
            expected_digest,
            str(path),
        )

    def _validate_payload_digest(
        self,
        payload: bytes,
        expected_digest: str | None,
        source: str,
    ) -> bytes:
        if len(payload) > self.max_uncompressed_bytes:
            raise CorruptChunkError(
                "term chunk exceeds configured uncompressed size limit",
                context={
                    "limit": self.max_uncompressed_bytes,
                    "source": source,
                },
            )
        if expected_digest is not None:
            actual_digest = hashlib.sha256(payload).hexdigest()
            if actual_digest != expected_digest:
                raise CorruptChunkError(
                    "term chunk digest mismatch",
                    context={
                        "path": source,
                        "expected": expected_digest,
                        "actual": actual_digest,
                    },
                )
        return payload

    @classmethod
    def _decode_document(cls, payload: bytes, chunk: ChunkRef) -> dict[str, object]:
        try:
            document = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptChunkError(
                "term chunk is not valid ASCII JSON",
                context={"chunk_id": chunk.chunk_id, "error": str(exc)},
            ) from exc
        if not isinstance(document, dict):
            raise CorruptChunkError("term chunk root must be a JSON object")
        if set(document) != {"format_version", "start_index", "values"}:
            raise CorruptChunkError(
                "term chunk has unexpected or missing fields",
                context={"chunk_id": chunk.chunk_id},
            )
        if document["format_version"] != chunk.format_version:
            raise CorruptChunkError(
                "term chunk format version does not match its manifest",
                context={"chunk_id": chunk.chunk_id},
            )
        if not isinstance(document["start_index"], int) or isinstance(
            document["start_index"], bool
        ):
            raise CorruptChunkError("term chunk start_index must be an integer")
        values = document["values"]
        if not isinstance(values, list) or any(
            not cls._is_canonical_decimal(value) for value in values
        ):
            raise CorruptChunkError(
                "term chunk values must be decimal integer strings"
            )
        return document

    @staticmethod
    def _int_to_decimal(value: int) -> str:
        """Convert an arbitrary-size integer without Python's digit limit."""

        if value == 0:
            return "0"
        negative = value < 0
        magnitude = -value if negative else value
        chunks: list[int] = []
        while magnitude:
            magnitude, remainder = divmod(magnitude, _DECIMAL_CHUNK_BASE)
            chunks.append(remainder)
        decimal_parts = [str(chunks.pop())]
        decimal_parts.extend(
            f"{chunk:0{_DECIMAL_CHUNK_WIDTH}d}" for chunk in reversed(chunks)
        )
        decimal = "".join(decimal_parts)
        return f"-{decimal}" if negative else decimal

    @classmethod
    def _parse_decimal(cls, value: str) -> int:
        """Parse canonical decimal text in bounded-size chunks."""

        if not cls._is_canonical_decimal(value):
            raise CorruptChunkError("term value is not a canonical decimal integer")
        negative = value.startswith("-")
        digits = value[1:] if negative else value
        first_width = len(digits) % _DECIMAL_CHUNK_WIDTH or _DECIMAL_CHUNK_WIDTH
        result = int(digits[:first_width])
        for start in range(first_width, len(digits), _DECIMAL_CHUNK_WIDTH):
            result = (
                result * _DECIMAL_CHUNK_BASE
                + int(digits[start : start + _DECIMAL_CHUNK_WIDTH])
            )
        return -result if negative else result

    @staticmethod
    def _is_canonical_decimal(value: object) -> bool:
        if not isinstance(value, str) or not value:
            return False
        negative = value.startswith("-")
        digits = value[1:] if negative else value
        if not digits or any(character < "0" or character > "9" for character in digits):
            return False
        if len(digits) > 1 and digits[0] == "0":
            return False
        return not (negative and digits == "0")

    @staticmethod
    def _write_hashed(
        compressed: gzip.GzipFile,
        digest: "hashlib._Hash",
        data: bytes,
    ) -> int:
        compressed.write(data)
        digest.update(data)
        return len(data)

    def _append_bundle(self, payload: BinaryIO) -> tuple[str, int, int]:
        bundle_dir = self.root / "bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        payload.seek(0, os.SEEK_END)
        payload_length = payload.tell()
        payload.seek(0)
        current_marker = bundle_dir / "current"
        with self._bundle_write_lock():
            try:
                bundle_number = int(current_marker.read_text(encoding="ascii"))
            except FileNotFoundError:
                bundle_number = 0
            except (OSError, ValueError) as exc:
                raise TermStoreError(
                    "bundle index is invalid",
                    context={"path": str(current_marker), "error": str(exc)},
                ) from exc

            while True:
                bundle_id = f"bundles/{bundle_number:08d}.bin"
                bundle_path = self._resolve_bundle_id(bundle_id)
                bundle_size = bundle_path.stat().st_size if bundle_path.exists() else 0
                if (
                    bundle_size == 0
                    or bundle_size + payload_length <= _BUNDLE_MAX_BYTES
                ):
                    break
                bundle_number += 1

            byte_offset = bundle_size
            marker_temp_path: Path | None = None
            try:
                with bundle_path.open("ab") as bundle:
                    shutil.copyfileobj(payload, bundle, length=1024 * 1024)
                    bundle.flush()
                    os.fsync(bundle.fileno())
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="ascii",
                    prefix=".bundle-current-",
                    suffix=".tmp",
                    dir=bundle_dir,
                    delete=False,
                ) as marker_temp:
                    marker_temp.write(str(bundle_number))
                    marker_temp.flush()
                    os.fsync(marker_temp.fileno())
                    marker_temp_path = Path(marker_temp.name)
                os.replace(marker_temp_path, current_marker)
            except OSError:
                self._remove_temp(marker_temp_path)
                raise
        return bundle_id, byte_offset, payload_length

    @contextmanager
    def _bundle_write_lock(self) -> Iterator[None]:
        lock_path = self.root / ".bundle-write.lock"
        with lock_path.open("a+b") as lock_file:
            if os.name == "nt":
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                while True:
                    try:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _resolve_bundle_id(self, bundle_id: str) -> Path:
        bundle_path = Path(bundle_id)
        if bundle_path.is_absolute():
            raise CorruptChunkError(
                "bundle identifier must be relative to the term store",
                context={"bundle_id": bundle_id},
            )
        path = (self.root / bundle_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise CorruptChunkError(
                "bundle identifier escapes the configured term store",
                context={"bundle_id": bundle_id},
            ) from exc
        return path

    def _resolve_bundle_path(self, chunk: ChunkRef) -> Path:
        if chunk.bundle_id is None:
            raise CorruptChunkError(
                "file-backed chunk has no bundle identifier",
                context={"chunk_id": chunk.chunk_id},
            )
        return self._resolve_bundle_id(chunk.bundle_id)

    def _resolve_content_path(self, chunk: ChunkRef) -> Path:
        manifest_path = Path(chunk.content_path)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        path = manifest_path.resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise CorruptChunkError(
                "chunk content path escapes the configured term store",
                context={"chunk_id": chunk.chunk_id, "path": str(path)},
            ) from exc
        return path

    @staticmethod
    def _validate_revision_id(revision_id: str) -> None:
        if not isinstance(revision_id, str) or not revision_id.strip():
            raise ValueError("revision_id must be a non-empty string")
        if any(char in revision_id for char in ("/", "\\", "\x00")):
            raise ValueError("revision_id must not contain path separators")

    @staticmethod
    def _validate_chunk_ref(chunk: ChunkRef) -> None:
        if not isinstance(chunk, ChunkRef):
            raise TypeError("chunk must be a ChunkRef")
        if chunk.codec != _CODEC:
            raise CorruptChunkError(
                "unsupported chunk codec",
                context={"chunk_id": chunk.chunk_id, "codec": chunk.codec},
            )
        if chunk.format_version != _FORMAT_VERSION:
            raise CorruptChunkError(
                "unsupported chunk format version",
                context={
                    "chunk_id": chunk.chunk_id,
                    "version": chunk.format_version,
                },
            )

    @staticmethod
    def _chunk_id(
        revision_id: str,
        first_index: int,
        count: int,
        payload_digest: str,
    ) -> str:
        identity = json.dumps(
            [revision_id, first_index, count, payload_digest],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(identity).hexdigest()

    @staticmethod
    def _remove_temp(path: Path | None) -> None:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # The original write/publish exception is more useful. A stale
                # temporary file is not referenced by the catalog.
                pass

    @staticmethod
    def _chunk_from_row(row: object) -> ChunkRef:
        try:
            return ChunkRef(
                chunk_id=row["chunk_id"],  # type: ignore[index]
                revision_id=row["revision_id"],  # type: ignore[index]
                first_index=row["first_index"],  # type: ignore[index]
                term_count=row["term_count"],  # type: ignore[index]
                last_index=row["last_index"],  # type: ignore[index]
                codec=row["codec"],  # type: ignore[index]
                format_version=row["format_version"],  # type: ignore[index]
                uncompressed_bytes=row["uncompressed_bytes"],  # type: ignore[index]
                payload_digest=row["payload_digest"],  # type: ignore[index]
                content_path=row["content_path"],  # type: ignore[index]
                bundle_id=row["bundle_id"],  # type: ignore[index]
                byte_offset=row["byte_offset"],  # type: ignore[index]
                byte_length=row["byte_length"],  # type: ignore[index]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptChunkError(
                "catalog contains an invalid term chunk manifest",
                context={"error": str(exc)},
            ) from exc
