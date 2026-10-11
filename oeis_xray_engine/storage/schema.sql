PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS corpus_snapshot (
  snapshot_id TEXT PRIMARY KEY,
  source_name TEXT NOT NULL,
  source_revision TEXT NOT NULL,
  source_digest TEXT NOT NULL,
  created_at TEXT NOT NULL,
  parser_version TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('ingesting','ready','failed'))
);

CREATE TABLE IF NOT EXISTS sequence (
  sequence_id TEXT PRIMARY KEY,
  current_revision_id TEXT,
  first_seen_snapshot TEXT NOT NULL REFERENCES corpus_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS sequence_revision (
  revision_id TEXT PRIMARY KEY,
  sequence_id TEXT NOT NULL REFERENCES sequence(sequence_id),
  snapshot_id TEXT NOT NULL REFERENCES corpus_snapshot(snapshot_id),
  source_digest TEXT NOT NULL,
  term_digest TEXT,
  offset INTEGER,
  term_count INTEGER NOT NULL,
  known_end_index INTEGER,
  completeness TEXT NOT NULL,
  parse_status TEXT NOT NULL,
  raw_metadata_json TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL,
  UNIQUE(sequence_id, snapshot_id, source_digest)
);
CREATE INDEX IF NOT EXISTS sequence_revision_by_sequence
  ON sequence_revision(sequence_id, snapshot_id);
CREATE INDEX IF NOT EXISTS sequence_revision_by_snapshot_keyset
  ON sequence_revision(snapshot_id, sequence_id, revision_id);
CREATE INDEX IF NOT EXISTS sequence_revision_by_digest
  ON sequence_revision(term_digest);

CREATE TABLE IF NOT EXISTS term_chunk (
  chunk_id TEXT PRIMARY KEY,
  revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  first_index INTEGER NOT NULL,
  term_count INTEGER NOT NULL,
  last_index INTEGER NOT NULL,
  codec TEXT NOT NULL,
  format_version INTEGER NOT NULL,
  uncompressed_bytes INTEGER NOT NULL,
  payload_digest TEXT NOT NULL,
  content_path TEXT NOT NULL,
  bundle_id TEXT,
  byte_offset INTEGER,
  byte_length INTEGER,
  UNIQUE(revision_id, first_index)
);
CREATE INDEX IF NOT EXISTS term_chunk_by_revision
  ON term_chunk(revision_id, first_index);

CREATE TABLE IF NOT EXISTS transform_spec (
  transform_key TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  implementation_version TEXT NOT NULL,
  canonical_parameters TEXT NOT NULL,
  domain_spec TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run (
  run_id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES corpus_snapshot(snapshot_id),
  run_kind TEXT NOT NULL,
  config_digest TEXT NOT NULL,
  code_revision TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('planned','running','paused','complete','failed','cancelled')),
  started_at TEXT,
  finished_at TEXT,
  manifest_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task (
  task_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(run_id),
  task_key TEXT NOT NULL,
  partition_key TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('pending','leased','complete','retryable_error','fatal_error')),
  lease_owner TEXT,
  lease_until TEXT,
  last_heartbeat TEXT,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  input_digest TEXT NOT NULL,
  task_manifest_json TEXT NOT NULL DEFAULT '{}',
  output_digest TEXT,
  error_json TEXT,
  updated_at TEXT NOT NULL,
  UNIQUE(run_id, task_key)
);
CREATE INDEX IF NOT EXISTS task_ready
  ON task(run_id, state, partition_key);

CREATE TABLE IF NOT EXISTS derived_artifact (
  artifact_id TEXT PRIMARY KEY,
  revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  transform_key TEXT NOT NULL REFERENCES transform_spec(transform_key),
  output_bound INTEGER NOT NULL,
  artifact_digest TEXT NOT NULL,
  term_manifest_json TEXT NOT NULL,
  created_by_run TEXT NOT NULL REFERENCES run(run_id),
  UNIQUE(revision_id, transform_key, output_bound, artifact_digest)
);
CREATE INDEX IF NOT EXISTS artifact_lookup
  ON derived_artifact(revision_id, transform_key, output_bound);
CREATE INDEX IF NOT EXISTS artifact_by_transform_cursor
  ON derived_artifact(transform_key, artifact_id);

CREATE TABLE IF NOT EXISTS fingerprint (
  fingerprint_id TEXT PRIMARY KEY,
  revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  profile_key TEXT NOT NULL,
  range_start INTEGER NOT NULL,
  range_end INTEGER NOT NULL,
  digest TEXT NOT NULL,
  canonicalization_version TEXT NOT NULL,
  created_by_run TEXT NOT NULL REFERENCES run(run_id),
  UNIQUE(revision_id, profile_key, range_start, range_end)
);
CREATE INDEX IF NOT EXISTS fingerprint_bucket
  ON fingerprint(profile_key, digest);

CREATE TABLE IF NOT EXISTS artifact_fingerprint (
  fingerprint_id TEXT PRIMARY KEY,
  artifact_id TEXT NOT NULL REFERENCES derived_artifact(artifact_id),
  profile_key TEXT NOT NULL,
  range_start INTEGER NOT NULL,
  range_end INTEGER NOT NULL,
  digest TEXT NOT NULL,
  canonicalization_version TEXT NOT NULL,
  created_by_run TEXT NOT NULL REFERENCES run(run_id),
  UNIQUE(artifact_id, profile_key, range_start, range_end),
  CHECK (range_end >= range_start)
);
CREATE INDEX IF NOT EXISTS artifact_fingerprint_bucket
  ON artifact_fingerprint(
    profile_key, range_start, range_end, canonicalization_version, digest
  );

CREATE TABLE IF NOT EXISTS relation_claim (
  claim_id TEXT PRIMARY KEY,
  relation_kind TEXT NOT NULL,
  source_revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  target_revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  transform_key TEXT,
  index_map_json TEXT NOT NULL,
  parameters_json TEXT NOT NULL,
  compared_start INTEGER,
  compared_end INTEGER,
  evidence_level TEXT NOT NULL CHECK (evidence_level IN ('E0','E1','E2','E3','E4')),
  status TEXT NOT NULL,
  evidence_digest TEXT NOT NULL,
  created_by_run TEXT NOT NULL REFERENCES run(run_id),
  supersedes_claim_id TEXT REFERENCES relation_claim(claim_id),
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS relation_source
  ON relation_claim(source_revision_id, evidence_level);
CREATE INDEX IF NOT EXISTS relation_target
  ON relation_claim(target_revision_id, evidence_level);
CREATE INDEX IF NOT EXISTS relation_claim_by_run_evidence
  ON relation_claim(created_by_run, evidence_level);

CREATE TABLE IF NOT EXISTS relation_evidence (
  evidence_id TEXT PRIMARY KEY,
  claim_id TEXT NOT NULL REFERENCES relation_claim(claim_id),
  evidence_level TEXT NOT NULL CHECK (evidence_level IN ('E2','E3','E4')),
  evidence_kind TEXT NOT NULL,
  verifier_id TEXT NOT NULL,
  range_start INTEGER NOT NULL,
  range_end INTEGER NOT NULL,
  evidence_digest TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  CHECK (range_end >= range_start),
  UNIQUE(claim_id, evidence_level, evidence_digest)
);
CREATE INDEX IF NOT EXISTS relation_evidence_by_claim
  ON relation_evidence(claim_id, evidence_level);

CREATE TABLE IF NOT EXISTS cf_search_candidate (
  candidate_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(run_id),
  task_id TEXT NOT NULL REFERENCES task(task_id),
  candidate_digest TEXT NOT NULL,
  lhs_key TEXT NOT NULL,
  lhs_numerator_json TEXT NOT NULL,
  lhs_denominator_json TEXT NOT NULL,
  rhs_a_terms_json TEXT NOT NULL,
  rhs_b_terms_json TEXT NOT NULL,
  lhs_value TEXT NOT NULL,
  rhs_value TEXT NOT NULL,
  compared_digits INTEGER NOT NULL CHECK (compared_digits > 0),
  UNIQUE(run_id, candidate_digest)
);
CREATE INDEX IF NOT EXISTS cf_search_candidate_by_run
  ON cf_search_candidate(run_id, candidate_digest);

CREATE TABLE IF NOT EXISTS formula_candidate (
  formula_id TEXT PRIMARY KEY,
  revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  family TEXT NOT NULL,
  canonical_expression TEXT NOT NULL,
  parameters_json TEXT NOT NULL,
  training_range_json TEXT NOT NULL,
  holdout_range_json TEXT NOT NULL,
  exact_match_count INTEGER NOT NULL,
  mismatch_json TEXT,
  evidence_level TEXT NOT NULL,
  created_by_run TEXT NOT NULL REFERENCES run(run_id)
);

CREATE TABLE IF NOT EXISTS anomaly (
  anomaly_id TEXT PRIMARY KEY,
  revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  rule_key TEXT NOT NULL,
  severity TEXT NOT NULL,
  classification TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  evidence_digest TEXT NOT NULL,
  created_by_run TEXT NOT NULL REFERENCES run(run_id),
  review_state TEXT NOT NULL DEFAULT 'unreviewed'
);

CREATE TABLE IF NOT EXISTS bfile_validation (
  validation_id TEXT PRIMARY KEY,
  revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  validation_kind TEXT NOT NULL CHECK (
    validation_kind IN ('sequence','denumerant')
  ),
  validation_status TEXT NOT NULL CHECK (
    validation_status IN ('match','mismatch','unavailable','malformed','insufficient_data')
  ),
  source_url TEXT,
  source_digest TEXT,
  comparison_digest TEXT NOT NULL,
  compared_start INTEGER,
  compared_end INTEGER,
  terms_checked INTEGER NOT NULL DEFAULT 0 CHECK (terms_checked >= 0),
  first_mismatch_index INTEGER,
  error_message TEXT,
  validator_version TEXT NOT NULL,
  validated_at TEXT NOT NULL,
  UNIQUE(
    revision_id, validation_kind, source_digest, comparison_digest,
    validator_version
  )
);
CREATE INDEX IF NOT EXISTS bfile_validation_by_revision
  ON bfile_validation(revision_id, validation_kind, validation_status);

CREATE TABLE IF NOT EXISTS graph_node (
  node_id TEXT PRIMARY KEY,
  sequence_id TEXT NOT NULL REFERENCES sequence(sequence_id),
  revision_id TEXT NOT NULL REFERENCES sequence_revision(revision_id),
  node_type TEXT NOT NULL,
  UNIQUE(sequence_id, revision_id, node_type)
);

CREATE TABLE IF NOT EXISTS graph_edge (
  edge_id TEXT PRIMARY KEY,
  claim_id TEXT NOT NULL UNIQUE REFERENCES relation_claim(claim_id),
  source_node_id TEXT NOT NULL REFERENCES graph_node(node_id),
  target_node_id TEXT NOT NULL REFERENCES graph_node(node_id),
  edge_type TEXT NOT NULL,
  evidence_level TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS graph_edges_from
  ON graph_edge(source_node_id, active, evidence_level);
CREATE INDEX IF NOT EXISTS graph_edges_to
  ON graph_edge(target_node_id, active, evidence_level);
