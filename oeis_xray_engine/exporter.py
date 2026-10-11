"""Human-readable exports for verified OEIS relation claims."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Literal, TextIO

from .errors import ValidationError
from .storage.catalog import Catalog

ExportFormat = Literal["json", "markdown"]
_EVIDENCE_LEVELS = ("E1", "E2", "E3", "E4")
_PAGE_SIZE = 500


def iter_gems(
    catalog: Catalog,
    *,
    run_id: str | None = None,
    page_size: int = _PAGE_SIZE,
) -> Iterator[dict[str, object]]:
    """Yield E1+ claims in claim-id order using bounded keyset pagination."""

    if not isinstance(catalog, Catalog):
        raise TypeError("catalog must be a Catalog")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("run_id must be a non-empty string or None")
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("page_size must be a positive integer")

    last_claim_id = ""
    while True:
        run_clause = "AND rc.created_by_run = ?" if run_id is not None else ""
        parameters: list[object] = [last_claim_id, *_EVIDENCE_LEVELS]
        if run_id is not None:
            parameters.append(run_id)
        parameters.append(page_size)
        rows = catalog.fetch_all(
            f"""
            SELECT rc.claim_id, rc.relation_kind, rc.source_revision_id,
                   rc.target_revision_id, source.sequence_id AS source_sequence,
                   target.sequence_id AS target_sequence,
                   rc.transform_key, rc.index_map_json, rc.parameters_json,
                   rc.compared_start, rc.compared_end, rc.evidence_level,
                   rc.status, rc.evidence_digest, rc.created_by_run,
                   transform.name AS transform_name,
                   transform.implementation_version AS transform_version,
                   transform.canonical_parameters AS transform_parameters_json,
                   (
                     SELECT MAX(artifact.output_bound)
                     FROM derived_artifact AS artifact
                     WHERE artifact.revision_id = rc.source_revision_id
                       AND artifact.transform_key = rc.transform_key
                   ) AS output_bound
            FROM relation_claim AS rc
            JOIN sequence_revision AS source_revision
              ON source_revision.revision_id = rc.source_revision_id
            JOIN sequence AS source
              ON source.sequence_id = source_revision.sequence_id
            JOIN sequence_revision AS target_revision
              ON target_revision.revision_id = rc.target_revision_id
            JOIN sequence AS target
              ON target.sequence_id = target_revision.sequence_id
            LEFT JOIN transform_spec AS transform
              ON transform.transform_key = rc.transform_key
            WHERE rc.claim_id > ?
              AND rc.evidence_level IN (?, ?, ?, ?)
              {run_clause}
            ORDER BY rc.claim_id
            LIMIT ?
            """,
            parameters,
        )
        if not rows:
            return
        for row in rows:
            last_claim_id = str(row["claim_id"])
            try:
                claim_parameters = json.loads(row["parameters_json"])
                index_map = json.loads(row["index_map_json"])
                transform_parameters = _transform_parameters(
                    row["transform_parameters_json"]
                )
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValidationError(
                    "relation claim contains malformed export metadata",
                    context={"claim_id": last_claim_id, "error": str(exc)},
                ) from exc

            compared_start = row["compared_start"]
            compared_end = row["compared_end"]
            compared_terms = (
                compared_end - compared_start + 1
                if type(compared_start) is int
                and type(compared_end) is int
                and compared_end >= compared_start
                else None
            )
            name = row["transform_name"]
            version = row["transform_version"]
            output_bound = _configured_bound(transform_parameters)
            if output_bound is None:
                output_bound = row["output_bound"]
            yield {
                "claim_id": last_claim_id,
                "relation_kind": row["relation_kind"],
                "source": row["source_sequence"],
                "source_revision_id": row["source_revision_id"],
                "target": row["target_sequence"],
                "target_revision_id": row["target_revision_id"],
                "transform": (
                    {
                        "name": name,
                        "version": version,
                        "parameters": transform_parameters,
                        "output_bound": output_bound,
                    }
                    if name is not None
                    else None
                ),
                "transform_key": row["transform_key"],
                "index_map": index_map,
                "parameters": claim_parameters,
                "compared_start": compared_start,
                "compared_end": compared_end,
                "compared_terms": compared_terms,
                "evidence_level": row["evidence_level"],
                "status": row["status"],
                "evidence_digest": row["evidence_digest"],
                "run_id": row["created_by_run"],
            }


def write_gems(
    catalog: Catalog,
    output: TextIO,
    *,
    run_id: str | None = None,
    format: ExportFormat = "markdown",
    page_size: int = _PAGE_SIZE,
) -> int:
    """Stream E1+ claims to a text output and return the number exported."""

    if format not in ("json", "markdown"):
        raise ValueError("format must be 'json' or 'markdown'")
    gems = iter_gems(catalog, run_id=run_id, page_size=page_size)
    if format == "json":
        return _write_json(gems, output)
    return _write_markdown(gems, output, run_id)


def _transform_parameters(raw: object) -> dict[str, object]:
    if raw is None:
        return {}
    parsed = json.loads(str(raw))
    if not isinstance(parsed, dict):
        raise TypeError("transform parameters must be a JSON object")
    nested = parsed.get("parameters")
    if isinstance(nested, dict):
        return nested
    return parsed


def _configured_bound(parameters: Mapping[str, object]) -> object:
    for key in ("term_cap", "cap", "output_bound"):
        if key in parameters:
            return parameters[key]
    return None


def _write_json(gems: Iterator[dict[str, object]], output: TextIO) -> int:
    output.write("[\n")
    count = 0
    for gem in gems:
        if count:
            output.write(",\n")
        encoded = json.dumps(gem, ensure_ascii=False, indent=2)
        output.write("  ")
        output.write(encoded.replace("\n", "\n  "))
        count += 1
    output.write("\n]\n")
    return count


def _write_markdown(
    gems: Iterator[dict[str, object]],
    output: TextIO,
    run_id: str | None,
) -> int:
    output.write("# OEIS X-Ray discoveries\n\n")
    output.write(f"Run: `{run_id}`" if run_id is not None else "Run: all runs")
    output.write("\n\n")
    count = 0
    for gem in gems:
        count += 1
        transform = gem["transform"]
        source = str(gem["source"])
        target = str(gem["target"])
        if isinstance(transform, dict):
            identity = f"{transform['name']}({source}) = {target}"
            transform_label = (
                f"{transform['name']} v{transform['version']}"
            )
            bound = transform.get("output_bound")
            if bound is not None:
                transform_label += f" (cap: {bound})"
        else:
            identity = f"{source} = {target}"
            transform_label = "direct sequence match"

        level = str(gem["evidence_level"])
        compared_terms = gem["compared_terms"]
        if compared_terms is None:
            evidence_detail = str(gem["status"])
        elif level == "E1":
            evidence_detail = f"Exact match for {compared_terms} terms"
        elif level == "E2":
            evidence_detail = (
                f"Independent reconstruction; exact match for {compared_terms} terms"
            )
        else:
            evidence_detail = f"{gem['status']} ({compared_terms} terms)"
        output.write(f"## Discovery {count}\n\n")
        output.write(f"- **Identity:** {identity}\n")
        output.write(f"- **Transform:** {transform_label}\n")
        output.write(f"- **Evidence:** {level} ({evidence_detail})\n")
        output.write(
            f"- **Compared range:** {gem['compared_start']}..{gem['compared_end']}\n"
        )
        output.write(f"- **Run:** `{gem['run_id']}`\n")
        output.write(f"- **Claim:** `{gem['claim_id']}`\n\n")
    if count == 0:
        output.write("No E1-or-higher claims found.\n")
    return count
