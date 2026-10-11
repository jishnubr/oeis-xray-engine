"""Create a verified, compact copy of a catalog without selected abandoned runs."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Sequence

TARGET_RUN_IDS = (
    "full-corpus-denumerant-v1",
    "full-corpus-euler-v1",
    "full-corpus-euler-v2",
    "full-corpus-euler-v3",
    "full-corpus-euler-v4",
)
PRESERVED_SNAPSHOT_ID = "full-corpus-v1"
PRESERVED_TABLES = (
    "corpus_snapshot",
    "sequence",
    "sequence_revision",
    "term_chunk",
)
RUN_SCOPED_COLUMNS = {
    "task": "run_id",
    "derived_artifact": "created_by_run",
    "fingerprint": "created_by_run",
    "artifact_fingerprint": "created_by_run",
    "relation_claim": "created_by_run",
    "cf_search_candidate": "run_id",
    "formula_candidate": "created_by_run",
    "anomaly": "created_by_run",
}
ALLOWED_FOREIGN_KEYS = {
    ("task", "run"),
    ("derived_artifact", "run"),
    ("fingerprint", "run"),
    ("artifact_fingerprint", "run"),
    ("artifact_fingerprint", "derived_artifact"),
    ("relation_claim", "run"),
    ("relation_claim", "relation_claim"),
    ("relation_evidence", "relation_claim"),
    ("cf_search_candidate", "run"),
    ("cf_search_candidate", "task"),
    ("formula_candidate", "run"),
    ("anomaly", "run"),
    ("graph_edge", "relation_claim"),
}
REQUIRED_TABLES = frozenset(
    {
        "run",
        "task",
        "derived_artifact",
        "fingerprint",
        "artifact_fingerprint",
        "relation_claim",
        "relation_evidence",
        "cf_search_candidate",
        "formula_candidate",
        "anomaly",
        "graph_edge",
        *PRESERVED_TABLES,
    }
)
PLACEHOLDER = ",".join("?" for _ in TARGET_RUN_IDS)


class PruneError(Exception):
    """Raised when the catalog cannot be pruned safely."""


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }


def _validate_schema(connection: sqlite3.Connection) -> None:
    tables = _table_names(connection)
    missing = sorted(REQUIRED_TABLES - tables)
    if missing:
        raise PruneError(
            "catalog schema is missing required tables: " + ", ".join(missing)
        )

    for table, column in RUN_SCOPED_COLUMNS.items():
        columns = {
            str(row["name"])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        if column not in columns:
            raise PruneError(f"catalog table {table!r} has no {column!r} column")

    for table in tables:
        columns = {
            str(row["name"])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        for column in {"run_id", "created_by_run"} & columns:
            if table == "run" and column == "run_id":
                continue
            if RUN_SCOPED_COLUMNS.get(table) != column:
                raise PruneError(
                    f"catalog has unhandled run-scoped column "
                    f"{table}.{column}; review the pruning logic"
                )

    for child in tables:
        for row in connection.execute(f'PRAGMA foreign_key_list("{child}")'):
            parent = str(row["table"])
            if parent in {"run", "task", "derived_artifact", "relation_claim"}:
                if (child, parent) not in ALLOWED_FOREIGN_KEYS:
                    raise PruneError(
                        "catalog has an unhandled foreign key "
                        f"{child!r} -> {parent!r}; review the pruning logic"
                    )


def _target_runs(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            f"SELECT run_id, snapshot_id, state FROM run "
            f"WHERE run_id IN ({PLACEHOLDER}) ORDER BY run_id",
            TARGET_RUN_IDS,
        )
    )


def _ensure_supported_runs(runs: Sequence[sqlite3.Row]) -> None:
    if not runs:
        raise PruneError("none of the five targeted runs exist in this catalog")
    unexpected = [
        str(run["run_id"])
        for run in runs
        if str(run["snapshot_id"]) != PRESERVED_SNAPSHOT_ID
    ]
    if unexpected:
        raise PruneError(
            "target run(s) do not belong to the expected "
            f"{PRESERVED_SNAPSHOT_ID!r} snapshot: {', '.join(unexpected)}"
        )


def _ensure_preserved_snapshot(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT state FROM corpus_snapshot WHERE snapshot_id = ?",
        (PRESERVED_SNAPSHOT_ID,),
    ).fetchone()
    if row is None or str(row["state"]) != "ready":
        raise PruneError(
            f"expected snapshot {PRESERVED_SNAPSHOT_ID!r} to exist and be ready"
        )


def _run_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table, column in RUN_SCOPED_COLUMNS.items():
        counts[table] = int(
            connection.execute(
                f'SELECT COUNT(*) FROM "{table}" '
                f'WHERE "{column}" IN ({PLACEHOLDER})',
                TARGET_RUN_IDS,
            ).fetchone()[0]
        )
    counts["artifact_fingerprint_for_target_artifacts"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM artifact_fingerprint "
            "WHERE artifact_id IN ("
            f"SELECT artifact_id FROM derived_artifact "
            f"WHERE created_by_run IN ({PLACEHOLDER}))",
            TARGET_RUN_IDS,
        ).fetchone()[0]
    )
    counts["relation_evidence_for_target_claims"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM relation_evidence WHERE claim_id IN ("
            f"SELECT claim_id FROM relation_claim "
            f"WHERE created_by_run IN ({PLACEHOLDER}))",
            TARGET_RUN_IDS,
        ).fetchone()[0]
    )
    counts["graph_edge_for_target_claims"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM graph_edge WHERE claim_id IN ("
            f"SELECT claim_id FROM relation_claim "
            f"WHERE created_by_run IN ({PLACEHOLDER}))",
            TARGET_RUN_IDS,
        ).fetchone()[0]
    )
    return counts


def _preserved_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(
            connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        )
        for table in PRESERVED_TABLES
    }


def _other_run_ids(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            f"SELECT run_id FROM run WHERE run_id NOT IN ({PLACEHOLDER})",
            TARGET_RUN_IDS,
        )
    }


def _check_no_external_claim_dependencies(
    connection: sqlite3.Connection,
    target_run_ids: Sequence[str],
) -> None:
    placeholders = ",".join("?" for _ in target_run_ids)
    dependent_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM relation_claim AS dependent "
            "WHERE dependent.created_by_run NOT IN ("
            f"{placeholders}) AND dependent.supersedes_claim_id IN ("
            "SELECT claim_id FROM relation_claim "
            f"WHERE created_by_run IN ({placeholders}))",
            (*target_run_ids, *target_run_ids),
        ).fetchone()[0]
    )
    if dependent_count:
        raise PruneError(
            f"{dependent_count} relation claim(s) from other runs supersede "
            "claims in the targeted runs; refusing to break that provenance"
        )


def _delete_target_run_rows(
    connection: sqlite3.Connection, target_run_ids: Sequence[str]
) -> None:
    placeholders = ",".join("?" for _ in target_run_ids)
    parameters = tuple(target_run_ids)

    connection.execute(
        "DELETE FROM relation_evidence WHERE claim_id IN ("
        "SELECT claim_id FROM relation_claim "
        f"WHERE created_by_run IN ({placeholders}))",
        parameters,
    )
    connection.execute(
        "DELETE FROM graph_edge WHERE claim_id IN ("
        "SELECT claim_id FROM relation_claim "
        f"WHERE created_by_run IN ({placeholders}))",
        parameters,
    )
    connection.execute(
        "DELETE FROM relation_claim "
        f"WHERE created_by_run IN ({placeholders})",
        parameters,
    )
    connection.execute(
        "DELETE FROM cf_search_candidate WHERE run_id IN "
        f"({placeholders})",
        parameters,
    )
    connection.execute(
        "DELETE FROM artifact_fingerprint WHERE created_by_run IN "
        f"({placeholders}) OR artifact_id IN ("
        "SELECT artifact_id FROM derived_artifact "
        f"WHERE created_by_run IN ({placeholders}))",
        (*parameters, *parameters),
    )
    connection.execute(
        "DELETE FROM derived_artifact WHERE created_by_run IN "
        f"({placeholders})",
        parameters,
    )
    connection.execute(
        "DELETE FROM fingerprint WHERE created_by_run IN "
        f"({placeholders})",
        parameters,
    )
    connection.execute(
        "DELETE FROM formula_candidate WHERE created_by_run IN "
        f"({placeholders})",
        parameters,
    )
    connection.execute(
        "DELETE FROM anomaly WHERE created_by_run IN "
        f"({placeholders})",
        parameters,
    )
    connection.execute(
        "DELETE FROM task WHERE run_id IN " f"({placeholders})",
        parameters,
    )
    connection.execute(
        "DELETE FROM run WHERE run_id IN " f"({placeholders})",
        parameters,
    )


def _integrity_check(connection: sqlite3.Connection) -> None:
    result = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
    if result != ["ok"]:
        raise PruneError("SQLite integrity_check failed: " + "; ".join(result[:10]))
    foreign_key_errors = list(connection.execute("PRAGMA foreign_key_check"))
    if foreign_key_errors:
        raise PruneError(
            f"SQLite foreign_key_check found {len(foreign_key_errors)} violation(s)"
        )


def _sidecars(path: Path) -> list[Path]:
    return [
        Path(f"{path}{suffix}")
        for suffix in ("-wal", "-shm")
        if Path(f"{path}{suffix}").exists()
    ]


def _dirty_sidecars(path: Path) -> list[Path]:
    """Return sidecar files that indicate uncheckpointed or active WAL state.

    On Windows, SQLite WAL mode recreates the -shm sidecar (and a zero-byte
    -wal file) even for read-only connections, and these persist after the
    connection closes.  A zero-byte WAL paired with an orphaned -shm does NOT
    contain uncheckpointed data and is safe to ignore.  Only flag sidecars
    where the WAL file has actual content (non-zero bytes), which indicates a
    genuine active writer or an un-checkpointed WAL.
    """
    wal_path = Path(f"{path}-wal")
    shm_path = Path(f"{path}-shm")
    wal_exists = wal_path.exists()
    wal_has_data = wal_exists and wal_path.stat().st_size > 0
    if wal_has_data:
        # Non-zero WAL: genuine uncheckpointed state — report both files.
        dirty = [wal_path]
        if shm_path.exists():
            dirty.append(shm_path)
        return dirty
    # Zero-byte or absent WAL: harmless Windows WAL-mode orphan, not dirty.
    return []


def _print_report(
    path: Path,
    runs: Sequence[sqlite3.Row],
    counts: dict[str, int],
    preserved_counts: dict[str, int],
) -> None:
    print(f"Catalog: {path}")
    print("Target runs:")
    for run in runs:
        print(
            f"  {run['run_id']}: state={run['state']}, "
            f"snapshot={run['snapshot_id']}"
        )
    print("Rows selected for deletion:")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    print("Corpus rows to preserve:")
    for table, count in preserved_counts.items():
        print(f"  {table}: {count}")


def _create_pruned_copy(source_path: Path, output_path: Path) -> None:
    if source_path.resolve() == output_path.resolve():
        raise PruneError("output path must differ from the source catalog")
    if output_path.exists():
        raise PruneError(f"output already exists; refusing to overwrite: {output_path}")
    if not output_path.parent.is_dir():
        raise PruneError(f"output directory does not exist: {output_path.parent}")

    source_sidecars = _dirty_sidecars(source_path)
    if source_sidecars:
        raise PruneError(
            "source SQLite WAL has uncheckpointed data; close all catalog users "
            "and run PRAGMA wal_checkpoint(TRUNCATE) before pruning: "
            + ", ".join(str(path) for path in source_sidecars)
        )
    required_free_space = source_path.stat().st_size * 2
    free_space = shutil.disk_usage(output_path.parent).free
    if free_space < required_free_space:
        raise PruneError(
            "insufficient free space beside the output: need at least "
            f"{required_free_space:,} bytes for the copy and VACUUM workspace"
        )

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp.sqlite", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with contextlib.closing(_connect_readonly(source_path)) as source:
            _validate_schema(source)
            _ensure_preserved_snapshot(source)
            runs = _target_runs(source)
            _ensure_supported_runs(runs)
            target_ids = [str(run["run_id"]) for run in runs]
            _check_no_external_claim_dependencies(source, target_ids)
            expected_preserved_counts = _preserved_counts(source)
            expected_other_runs = _other_run_ids(source)
            destination = sqlite3.connect(temporary_path)
            try:
                source.backup(destination)
            finally:
                destination.close()

        with contextlib.closing(sqlite3.connect(temporary_path)) as pruned:
            pruned.row_factory = sqlite3.Row
            pruned.execute("PRAGMA foreign_keys = ON")
            _validate_schema(pruned)
            _ensure_preserved_snapshot(pruned)
            copied_runs = _target_runs(pruned)
            _ensure_supported_runs(copied_runs)
            _check_no_external_claim_dependencies(pruned, target_ids)
            pruned.execute("BEGIN IMMEDIATE")
            _delete_target_run_rows(pruned, target_ids)
            pruned.commit()
            pruned.execute("VACUUM")
            _integrity_check(pruned)

            if _target_runs(pruned):
                raise PruneError("one or more target runs remain after cleanup")
            actual_preserved_counts = _preserved_counts(pruned)
            if actual_preserved_counts != expected_preserved_counts:
                raise PruneError(
                    "preserved corpus row counts changed: "
                    f"expected {expected_preserved_counts}, "
                    f"found {actual_preserved_counts}"
                )
            actual_other_runs = _other_run_ids(pruned)
            if actual_other_runs != expected_other_runs:
                raise PruneError("one or more non-target runs changed")

        if output_path.exists():
            raise PruneError(
                f"output appeared during pruning; refusing to overwrite: {output_path}"
            )
        temporary_path.rename(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        for sidecar in _sidecars(temporary_path):
            sidecar.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview cleanup or create a separate, verified compact catalog copy "
            "without the five named abandoned full-corpus runs."
        )
    )
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create a pruned copy; the source catalog is never modified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new output catalog path (required with --apply; must not exist)",
    )
    parser.add_argument(
        "--confirm-no-active-workers",
        action="store_true",
        help=(
            "confirm that no process is using or writing the source catalog; "
            "required with --apply"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_path = args.catalog.expanduser().resolve()
    if not source_path.is_file():
        print(f"error: catalog file does not exist: {source_path}", file=sys.stderr)
        return 2
    if args.apply and not args.confirm_no_active_workers:
        print(
            "error: --apply requires --confirm-no-active-workers after verifying "
            "that all catalog users are stopped",
            file=sys.stderr,
        )
        return 2
    if args.apply and args.output is None:
        print("error: --apply requires --output", file=sys.stderr)
        return 2
    if not args.apply and args.output is not None:
        print("error: --output is only valid with --apply", file=sys.stderr)
        return 2

    # Early sidecar guard: check *before* opening any connection.  Opening the
    # DB in WAL mode (even read-only) recreates the -shm sidecar, so we must
    # verify cleanliness before the first sqlite3.connect() call, not after.
    if args.apply:
        early_sidecars = _dirty_sidecars(source_path)
        if early_sidecars:
            print(
                "error: source SQLite WAL has uncheckpointed data; close all catalog users "
                "and ensure the WAL is checkpointed before pruning: "
                + ", ".join(str(path) for path in early_sidecars),
                file=sys.stderr,
            )
            return 2

    try:
        with contextlib.closing(_connect_readonly(source_path)) as catalog:
            _validate_schema(catalog)
            _ensure_preserved_snapshot(catalog)
            runs = _target_runs(catalog)
            _ensure_supported_runs(runs)
            _check_no_external_claim_dependencies(
                catalog, [str(run["run_id"]) for run in runs]
            )
            counts = _run_counts(catalog)
            preserved_counts = _preserved_counts(catalog)
            _print_report(source_path, runs, counts, preserved_counts)
            sidecars = _dirty_sidecars(source_path)
            if sidecars:
                print(
                    "Warning: SQLite WAL has uncheckpointed data; the preview may observe "
                    "uncheckpointed state and --apply will refuse to proceed: "
                    + ", ".join(str(path) for path in sidecars)
                )
        # The read-only preview connection above may have recreated the -shm
        # sidecar.  Wait for SQLite to release it (it should be gone as soon as
        # the connection is closed), then pass the clean path into
        # _create_pruned_copy which re-checks sidecars itself.

        if not args.apply:
            print("Preview only. No database files were changed.")
            return 0

        output_path = args.output.expanduser().resolve()
        # Re-verify sidecars are gone after the preview connection closed.
        # The -shm sidecar created by the read-only open should have been
        # released; if something else snuck in, fail clearly.
        # After the read-only preview connection closes, a zero-byte WAL and
        # orphaned SHM may linger on Windows — these are harmless WAL-mode
        # artefacts.  Only abort if the WAL actually contains uncheckpointed
        # data, which would indicate a genuine concurrent writer.
        post_preview_sidecars = _dirty_sidecars(source_path)
        if post_preview_sidecars:
            raise PruneError(
                "source SQLite WAL has uncheckpointed data after the preview "
                "connection closed; another process may be writing the catalog: "
                + ", ".join(str(path) for path in post_preview_sidecars)
            )
        _create_pruned_copy(source_path, output_path)
        print(f"Verified pruned copy created: {output_path}")
        print(
            "The source catalog and term store were not modified. Review and "
            "test the new catalog before changing any application paths."
        )
        return 0
    except (OSError, sqlite3.Error, PruneError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
