# OEIS X-Ray Engine

A standalone Python engine for ingesting OEIS sequence files, running bounded
sequence-transform and fingerprint searches, and recording provenance-backed
finite relation claims.

The engine is separate from the OEIS Research Workbench application. It does not
depend on that application's UI, draft database, or working-tree files. Corpus
files and generated catalogs are supplied by the operator and are not included
in this repository.

## Requirements

- Python 3.10 or newer
- Dependencies are installed from `pyproject.toml`

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

## Test

```powershell
python -m unittest discover -v
```

The test suite uses temporary fixtures; it does not require the parent
application or a local OEIS corpus.

## Package layout

- `oeis_xray_engine/` contains the engine, CLI, storage, indexing, matching,
  transformer, and checkpoint modules.
- `tests/` contains focused CLI, pipeline, indexing/matching, and transformer
  regression tests.

Sequence prefix matches are finite evidence only; they do not establish an
infinite mathematical identity.
