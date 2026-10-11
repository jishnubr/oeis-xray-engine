# OEIS X-Ray Engine

A Python package for ingesting OEIS sequence files, running bounded
sequence-transform and fingerprint searches, and recording provenance-backed
finite relation claims.

This package is a distinct component within the parent repository. It does not
depend on the web application, its UI, its draft database, or its working-tree
files. Corpus files and generated catalogs are supplied by the operator and
are not included in this package.

## Requirements

- Python 3.10 or newer
- Dependencies declared in the root `pyproject.toml`

## Install

From the repository root:

```powershell
python -m pip install -e .
```

## Use

```powershell
oeis-xray --help
```

The CLI accepts explicit paths for its SQLite catalog and content-addressed term
store. Consult each command's `--help` for its inputs and options.

## Prune abandoned catalog runs

`oeis-xray-prune-catalog` targets only the five named full-corpus runs
(`full-corpus-denumerant-v1` and `full-corpus-euler-v1` through `v4`). It first
prints a read-only preview. To produce a **new** compacted catalog, close all
processes using the source catalog, ensure its SQLite WAL is checkpointed, and
provide a new output path with sufficient free disk space:

```powershell
oeis-xray-prune-catalog --catalog .\oeis_xray_catalog.sqlite
oeis-xray-prune-catalog `
  --catalog .\oeis_xray_catalog.sqlite `
  --apply `
  --confirm-no-active-workers `
  --output .\oeis_xray_catalog.pruned.sqlite
```

The source catalog and term store are never modified. The command refuses to
overwrite an existing output, refuses source `-wal`/`-shm` sidecars, preserves
the `full-corpus-v1` snapshot and its sequence/chunk rows, and checks SQLite
integrity and foreign keys before publishing the output copy. The `--apply`
operation requires the explicit `--confirm-no-active-workers` confirmation and
approximately twice the source catalog size free beside the output for the
copy and `VACUUM` workspace. Review the new database and verify application
behavior before switching any configured catalog path; do not delete the
original catalog or term files as part of this command.

## Test

From the repository root:

```powershell
python -m unittest discover -s tests -p "test_xray_*.py" -v
python -m unittest discover -s tests/truth_matrix -v
```

The tests use temporary fixtures; they do not require the web application or a
local OEIS corpus.

Sequence prefix matches are finite evidence only; they do not establish an
infinite mathematical identity.
