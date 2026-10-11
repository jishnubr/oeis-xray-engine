import sqlite3
from collections import Counter
from oeis_xray_engine.storage.catalog import Catalog
from oeis_xray_engine.storage.term_store import TermStore
from oeis_xray_engine.pipeline import _read_artifact_series

catalog = Catalog('oeis_xray_catalog.sqlite', initialize=False, read_only=True)
term_store = TermStore('oeis_xray_terms', catalog)

rows = catalog.fetch_all("""
    SELECT a.artifact_id, s.sequence_id, r.snapshot_id, a.created_by_run
    FROM derived_artifact a
    JOIN sequence_revision r ON r.revision_id = a.revision_id
    JOIN sequence s ON s.sequence_id = r.sequence_id
    JOIN transform_spec ts ON ts.transform_key = a.transform_key
    WHERE ts.name = 'denumerant'
    ORDER BY s.sequence_id
""")

print(f"Total certified denumerant artifacts: {len(rows)}")

shapes = Counter()
multisets = Counter()
seq_by_multiset = {}

for r in rows:
    series = _read_artifact_series(catalog, term_store, r["artifact_id"], r["snapshot_id"])
    w = tuple(series.values)
    multisets[w] += 1
    if w not in seq_by_multiset:
        seq_by_multiset[w] = []
    seq_by_multiset[w].append(r["sequence_id"])
    
    # Generic shape: replace distinct elements with sorted symbols
    # e.g. (1, 1, 3, 3) -> (a, a, b, b)
    mapping = {}
    sym = 0
    shape = []
    for val in w:
        if val not in mapping:
            mapping[val] = chr(ord('a') + sym)
            sym += 1
        shape.append(mapping[val])
    shapes[tuple(shape)] += 1

print("\n--- Top 15 Algebraic Shapes across all 766 sequences ---")
for sh, count in shapes.most_common(15):
    print(f"  Shape {list(sh)}: {count} sequences")

print("\n--- Most Frequent Specific Weight Multisets ---")
for w, count in multisets.most_common(15):
    seqs = ", ".join(seq_by_multiset[w][:4])
    if len(seq_by_multiset[w]) > 4:
        seqs += f", ... (+{len(seq_by_multiset[w])-4} more)"
    print(f"  {list(w)} ({count} sequences): {seqs}")

catalog.close()
