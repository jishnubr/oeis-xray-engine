"""Dedicated SQLite database store for verified mathematical discoveries.

Decouples pure mathematical results (certified denumerants, parametric towers,
missing teeth, and synthesized master arrays) from operational pipeline state
(worker leases, heartbeats, and task manifests).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Sequence

_LOGGER = logging.getLogger("oeis_xray_engine.discovery.results_store")

DISCOVERIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS certified_denumerant (
    sequence_id TEXT PRIMARY KEY,
    weights_json TEXT NOT NULL,
    distinct_weight_count INTEGER NOT NULL,
    max_weight INTEGER NOT NULL,
    confidence_terms INTEGER NOT NULL,
    extracted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parametric_tower (
    tower_id TEXT PRIMARY KEY,
    family_template TEXT NOT NULL,
    base_weights_json TEXT NOT NULL,
    variable_multiplicity INTEGER NOT NULL,
    discovered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tower_entry (
    tower_id TEXT NOT NULL REFERENCES parametric_tower(tower_id),
    k_value INTEGER NOT NULL,
    status TEXT NOT NULL,             -- 'cataloged' or 'missing_tooth'
    sequence_id TEXT,                 -- e.g. 'A002620' or NULL if uncataloged
    terms_prefix_json TEXT NOT NULL,  -- first 25-50 terms
    generating_function TEXT NOT NULL,
    PRIMARY KEY (tower_id, k_value)
);

CREATE TABLE IF NOT EXISTS master_array (
    array_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    gf_formula TEXT NOT NULL,
    closed_form TEXT,
    antidiagonals_json TEXT NOT NULL,
    crossrefs_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tower_entry_status ON tower_entry(status);
CREATE INDEX IF NOT EXISTS idx_tower_entry_sequence ON tower_entry(sequence_id);
"""


class ResultsDatabase:
    """Standalone, lightweight results store for OEIS discoveries."""

    def __init__(self, db_path: str | Path = "discoveries.sqlite") -> None:
        self.path = Path(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        self.conn.executescript(DISCOVERIES_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self) -> ResultsDatabase:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Certified Denumerants
    # -------------------------------------------------------------------------

    def record_certified_denumerant(
        self,
        sequence_id: str,
        weights: Sequence[int],
        confidence_terms: int = 12,
        extracted_at: str | None = None,
    ) -> bool:
        """Record or update a certified denumerant weight multiset.

        Returns True if a new row was inserted, False if updated.
        """
        sorted_weights = sorted(int(w) for w in weights)
        weights_json = json.dumps(sorted_weights)
        distinct_count = len(set(sorted_weights))
        max_weight = max(sorted_weights) if sorted_weights else 0
        timestamp = extracted_at or datetime.now(timezone.utc).isoformat()

        cur = self.conn.cursor()
        cur.execute(
            "SELECT sequence_id FROM certified_denumerant WHERE sequence_id = ?",
            (sequence_id,),
        )
        is_new = cur.fetchone() is None

        cur.execute(
            """
            INSERT INTO certified_denumerant (
                sequence_id, weights_json, distinct_weight_count,
                max_weight, confidence_terms, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sequence_id) DO UPDATE SET
                weights_json = excluded.weights_json,
                distinct_weight_count = excluded.distinct_weight_count,
                max_weight = excluded.max_weight,
                confidence_terms = excluded.confidence_terms,
                extracted_at = excluded.extracted_at
            """,
            (
                sequence_id,
                weights_json,
                distinct_count,
                max_weight,
                confidence_terms,
                timestamp,
            ),
        )
        self.conn.commit()
        return is_new

    def get_certified_denumerants(self) -> list[dict[str, Any]]:
        """Retrieve all certified denumerants ordered by sequence ID."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT sequence_id, weights_json, distinct_weight_count,
                   max_weight, confidence_terms, extracted_at
            FROM certified_denumerant
            ORDER BY sequence_id
            """
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "sequence_id": r["sequence_id"],
                    "weights": json.loads(r["weights_json"]),
                    "distinct_weight_count": r["distinct_weight_count"],
                    "max_weight": r["max_weight"],
                    "confidence_terms": r["confidence_terms"],
                    "extracted_at": r["extracted_at"],
                }
            )
        return rows

    def is_sequence_certified(self, sequence_id: str) -> bool:
        """Check if a sequence already exists in certified_denumerant."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM certified_denumerant WHERE sequence_id = ?",
            (sequence_id,),
        )
        return cur.fetchone() is not None

    # -------------------------------------------------------------------------
    # Parametric Towers
    # -------------------------------------------------------------------------

    def record_tower(
        self,
        tower_id: str,
        family_template: str,
        base_weights: Sequence[int],
        variable_multiplicity: int = 1,
        discovered_at: str | None = None,
    ) -> bool:
        """Record or update a parametric tower archetype.

        Returns True if a new row was inserted.
        """
        base_json = json.dumps(sorted(int(w) for w in base_weights))
        timestamp = discovered_at or datetime.now(timezone.utc).isoformat()

        cur = self.conn.cursor()
        cur.execute(
            "SELECT tower_id FROM parametric_tower WHERE tower_id = ?",
            (tower_id,),
        )
        is_new = cur.fetchone() is None

        cur.execute(
            """
            INSERT INTO parametric_tower (
                tower_id, family_template, base_weights_json,
                variable_multiplicity, discovered_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tower_id) DO UPDATE SET
                family_template = excluded.family_template,
                base_weights_json = excluded.base_weights_json,
                variable_multiplicity = excluded.variable_multiplicity,
                discovered_at = excluded.discovered_at
            """,
            (
                tower_id,
                family_template,
                base_json,
                variable_multiplicity,
                timestamp,
            ),
        )
        self.conn.commit()
        return is_new

    def record_tower_entry(
        self,
        tower_id: str,
        k_value: int,
        status: str,
        sequence_id: str | None,
        terms_prefix: Sequence[int],
        generating_function: str,
    ) -> bool:
        """Record a tower member (either 'cataloged' or 'missing_tooth').

        Returns True if newly inserted.
        """
        terms_json = json.dumps(list(terms_prefix))
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM tower_entry WHERE tower_id = ? AND k_value = ?",
            (tower_id, k_value),
        )
        is_new = cur.fetchone() is None

        cur.execute(
            """
            INSERT INTO tower_entry (
                tower_id, k_value, status, sequence_id,
                terms_prefix_json, generating_function
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tower_id, k_value) DO UPDATE SET
                status = excluded.status,
                sequence_id = excluded.sequence_id,
                terms_prefix_json = excluded.terms_prefix_json,
                generating_function = excluded.generating_function
            """,
            (
                tower_id,
                k_value,
                status,
                sequence_id,
                terms_json,
                generating_function,
            ),
        )
        self.conn.commit()
        return is_new

    def get_towers(self) -> list[dict[str, Any]]:
        """Retrieve all parametric towers."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT tower_id, family_template, base_weights_json,
                   variable_multiplicity, discovered_at
            FROM parametric_tower
            ORDER BY tower_id
            """
        )
        towers = []
        for r in cur.fetchall():
            towers.append(
                {
                    "tower_id": r["tower_id"],
                    "family_template": r["family_template"],
                    "base_weights": json.loads(r["base_weights_json"]),
                    "variable_multiplicity": r["variable_multiplicity"],
                    "discovered_at": r["discovered_at"],
                }
            )
        return towers

    def get_tower_entries(
        self, tower_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve tower entries, optionally filtered by tower_id."""
        cur = self.conn.cursor()
        if tower_id is not None:
            cur.execute(
                """
                SELECT tower_id, k_value, status, sequence_id,
                       terms_prefix_json, generating_function
                FROM tower_entry
                WHERE tower_id = ?
                ORDER BY k_value
                """,
                (tower_id,),
            )
        else:
            cur.execute(
                """
                SELECT tower_id, k_value, status, sequence_id,
                       terms_prefix_json, generating_function
                FROM tower_entry
                ORDER BY tower_id, k_value
                """
            )
        entries = []
        for r in cur.fetchall():
            entries.append(
                {
                    "tower_id": r["tower_id"],
                    "k_value": r["k_value"],
                    "status": r["status"],
                    "sequence_id": r["sequence_id"],
                    "terms_prefix": json.loads(r["terms_prefix_json"]),
                    "generating_function": r["generating_function"],
                }
            )
        return entries

    def get_missing_teeth(self) -> list[dict[str, Any]]:
        """Retrieve all uncataloged Missing Teeth entries across all towers."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT e.tower_id, e.k_value, e.terms_prefix_json, e.generating_function,
                   t.family_template, t.base_weights_json
            FROM tower_entry e
            JOIN parametric_tower t ON t.tower_id = e.tower_id
            WHERE e.status = 'missing_tooth'
            ORDER BY e.tower_id, e.k_value
            """
        )
        teeth = []
        for r in cur.fetchall():
            teeth.append(
                {
                    "tower_id": r["tower_id"],
                    "k_value": r["k_value"],
                    "terms_prefix": json.loads(r["terms_prefix_json"]),
                    "generating_function": r["generating_function"],
                    "family_template": r["family_template"],
                    "base_weights": json.loads(r["base_weights_json"]),
                }
            )
        return teeth

    # -------------------------------------------------------------------------
    # Master Arrays
    # -------------------------------------------------------------------------

    def record_master_array(
        self,
        array_id: str,
        title: str,
        gf_formula: str,
        closed_form: str | None,
        antidiagonals: Sequence[Any],
        crossrefs: Sequence[str],
        created_at: str | None = None,
    ) -> bool:
        """Record or update a synthesized Master Array."""
        antidiag_json = json.dumps(list(antidiagonals))
        crossrefs_json = json.dumps(sorted(set(crossrefs)))
        timestamp = created_at or datetime.now(timezone.utc).isoformat()

        cur = self.conn.cursor()
        cur.execute(
            "SELECT array_id FROM master_array WHERE array_id = ?",
            (array_id,),
        )
        is_new = cur.fetchone() is None

        cur.execute(
            """
            INSERT INTO master_array (
                array_id, title, gf_formula, closed_form,
                antidiagonals_json, crossrefs_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(array_id) DO UPDATE SET
                title = excluded.title,
                gf_formula = excluded.gf_formula,
                closed_form = excluded.closed_form,
                antidiagonals_json = excluded.antidiagonals_json,
                crossrefs_json = excluded.crossrefs_json,
                created_at = excluded.created_at
            """,
            (
                array_id,
                title,
                gf_formula,
                closed_form,
                antidiag_json,
                crossrefs_json,
                timestamp,
            ),
        )
        self.conn.commit()
        return is_new

    def get_master_arrays(self) -> list[dict[str, Any]]:
        """Retrieve all synthesized Master Arrays."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT array_id, title, gf_formula, closed_form,
                   antidiagonals_json, crossrefs_json, created_at
            FROM master_array
            ORDER BY array_id
            """
        )
        arrays = []
        for r in cur.fetchall():
            arrays.append(
                {
                    "array_id": r["array_id"],
                    "title": r["title"],
                    "gf_formula": r["gf_formula"],
                    "closed_form": r["closed_form"],
                    "antidiagonals": json.loads(r["antidiagonals_json"]),
                    "crossrefs": json.loads(r["crossrefs_json"]),
                    "created_at": r["created_at"],
                }
            )
        return arrays

    # -------------------------------------------------------------------------
    # Metrics & Summaries
    # -------------------------------------------------------------------------

    def get_counts(self) -> dict[str, int]:
        """Return counts of each discovery category."""
        cur = self.conn.cursor()
        n_certified = cur.execute(
            "SELECT COUNT(*) FROM certified_denumerant"
        ).fetchone()[0]
        n_towers = cur.execute(
            "SELECT COUNT(*) FROM parametric_tower"
        ).fetchone()[0]
        n_cataloged = cur.execute(
            "SELECT COUNT(*) FROM tower_entry WHERE status = 'cataloged'"
        ).fetchone()[0]
        n_missing = cur.execute(
            "SELECT COUNT(*) FROM tower_entry WHERE status = 'missing_tooth'"
        ).fetchone()[0]
        n_arrays = cur.execute(
            "SELECT COUNT(*) FROM master_array"
        ).fetchone()[0]
        return {
            "certified_denumerants": n_certified,
            "parametric_towers": n_towers,
            "cataloged_members": n_cataloged,
            "missing_teeth": n_missing,
            "master_arrays": n_arrays,
        }

    def summary_report(self) -> str:
        """Format a human-readable discovery summary."""
        counts = self.get_counts()
        lines = [
            "+--------------------------------------------------+",
            "|           DISCOVERIES DATABASE SUMMARY           |",
            "+--------------------------------------------------+",
            f"|  Certified Denumerant Sequences : {counts['certified_denumerants']:6d}         |",
            f"|  Parametric Towers Discovered    : {counts['parametric_towers']:6d}         |",
            f"|  Cataloged Tower Members        : {counts['cataloged_members']:6d}         |",
            f"|  Uncataloged 'Missing Teeth'    : {counts['missing_teeth']:6d}         |",
            f"|  Synthesized Master Arrays      : {counts['master_arrays']:6d}         |",
            "+--------------------------------------------------+",
        ]
        return "\n".join(lines)
