import hashlib
import json
import re
import sqlite3
from pathlib import Path

from psil.store.models import Paper


class Database:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create_tables(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    doi TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    journal TEXT,
                    authors TEXT,
                    affiliations TEXT,
                    pub_date TEXT,
                    toc_image_url TEXT,
                    signal_score INTEGER,
                    signal_tier TEXT,
                    signal_trajectory REAL,
                    signal_action TEXT,
                    llm_reasoning TEXT,
                    concept_name TEXT,
                    concept_drift TEXT,
                    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    -- Layer 1: Causal extraction
                    causal_question TEXT,
                    causal_constraint TEXT,
                    causal_input TEXT,
                    causal_transformation TEXT,
                    causal_output TEXT,
                    causal_outcome TEXT
                );

                CREATE TABLE IF NOT EXISTS ingest_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    papers_fetched INTEGER,
                    papers_new INTEGER,
                    papers_high_signal INTEGER,
                    papers_maybe_signal INTEGER,
                    papers_ignored INTEGER
                );

                CREATE TABLE IF NOT EXISTS concept_tracking (
                    name TEXT PRIMARY KEY,
                    first_seen TEXT,
                    last_seen TEXT,
                    appearances INTEGER DEFAULT 1,
                    source_doi TEXT,
                    why_matters TEXT,
                    connection TEXT,
                    missing_link TEXT,
                    opportunity TEXT,
                    trajectory_weight TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'emerging'
                );

                -- Layer 2: Logic patterns discovered across papers
                CREATE TABLE IF NOT EXISTS logic_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    description TEXT,
                    causal_template TEXT,
                    paper_count INTEGER DEFAULT 0,
                    sample_dois TEXT,
                    discovered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    score REAL DEFAULT 0
                );

                -- Layer 3: Higher-order frameworks
                CREATE TABLE IF NOT EXISTS frameworks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    framework_name TEXT NOT NULL,
                    description TEXT,
                    covered_patterns TEXT,
                    excluded_patterns TEXT,
                    compression_score REAL DEFAULT 0,
                    novelty_score REAL DEFAULT 0,
                    generated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Delta detection: old → new worldview shifts
                CREATE TABLE IF NOT EXISTS deltas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    previous_assumption TEXT,
                    new_assumption TEXT,
                    delta TEXT,
                    source_dois TEXT,
                    detected_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Constraint Discovery: constraints extracted from frameworks
                CREATE TABLE IF NOT EXISTS constraints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    framework_name TEXT NOT NULL,
                    statement TEXT,
                    constraint_type TEXT DEFAULT 'requires',
                    supporting_evidence TEXT,
                    violating_examples TEXT,
                    confidence REAL DEFAULT 0,
                    prediction_power REAL DEFAULT 0,
                    actionability REAL DEFAULT 0,
                    status TEXT DEFAULT 'candidate',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Predictions generated from constraints
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    constraint_id INTEGER,
                    prediction_type TEXT,
                    statement TEXT,
                    expected_relationship TEXT,
                    failure_condition TEXT,
                    status TEXT DEFAULT 'candidate',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Doyle TMS: justification tracking for every concept-paper link
                CREATE TABLE IF NOT EXISTS justifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_name TEXT NOT NULL,
                    paper_doi TEXT NOT NULL,
                    support_type TEXT,
                    evidence_strength TEXT,
                    justification_text TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Constraint verification: cross-check papers against constraints
                CREATE TABLE IF NOT EXISTS constraint_verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    constraint_name TEXT NOT NULL,
                    paper_doi TEXT NOT NULL,
                    result TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    evidence TEXT,
                    verified_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Experiments designed from predictions
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    framework_name TEXT,
                    manipulate_variable TEXT,
                    measure_variable TEXT,
                    expected_result TEXT,
                    failure_condition TEXT,
                    priority TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'candidate',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Trajectories: long-term research directions consuming concepts
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    confidence TEXT DEFAULT 'Stable',
                    supporting_concepts TEXT,
                    evidence_count INTEGER DEFAULT 0,
                    discovery_count INTEGER DEFAULT 0,
                    validation_count INTEGER DEFAULT 0,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Kernel State: accumulated kernel judgments across scans
                CREATE TABLE IF NOT EXISTS kernel_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT,
                    category TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Kernel Objects: first-class scientific judgment objects.
                -- LLMs can propose these; the non-LLM kernel revises them.
                CREATE TABLE IF NOT EXISTS kernel_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_key TEXT NOT NULL UNIQUE,
                    object_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    statement TEXT,
                    status TEXT DEFAULT 'candidate',
                    confidence REAL DEFAULT 0,
                    entrenchment REAL DEFAULT 0,
                    source_type TEXT,
                    source_ref TEXT,
                    evidence TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS kernel_object_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_key TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT,
                    previous_confidence REAL,
                    new_confidence REAL,
                    previous_entrenchment REAL,
                    new_entrenchment REAL,
                    reason TEXT,
                    evidence_delta TEXT,
                    actor TEXT DEFAULT 'kernel',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS kernel_object_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL,
                    target_key TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL DEFAULT 1,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_key, target_key, relation_type)
                );

                CREATE TABLE IF NOT EXISTS kernel_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_key TEXT NOT NULL UNIQUE,
                    task_type TEXT NOT NULL,
                    object_key TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    priority REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    action_hint TEXT,
                    source_type TEXT,
                    source_ref TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS kernel_task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_key TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT,
                    reason TEXT,
                    actor TEXT DEFAULT 'kernel',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Research Memory: approved/rejected concepts, decisions, contradictions
                CREATE TABLE IF NOT EXISTS research_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_type TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    status TEXT DEFAULT 'candidate',
                    reason TEXT,
                    evidence_strength TEXT,
                    affected_projects TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Curated local literature imported from NRB manifests.
                CREATE TABLE IF NOT EXISTS local_sources (
                    doi TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    journal TEXT,
                    pub_year TEXT,
                    bucket TEXT,
                    status TEXT,
                    local_path TEXT,
                    note TEXT,
                    source_manifest TEXT,
                    imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

        # ── Migrations for existing tables ──
        self._migrate()

    def _migrate(self):
        """Add missing columns to existing tables (safe to re-run)."""
        migrations = {
            "papers": [
                "evidence TEXT", "limitation TEXT", "caveat TEXT",
                "project_relevance TEXT", "review_status TEXT DEFAULT 'summarized'",
                "problem_class TEXT", "novelty_type TEXT",
                "strategic_value TEXT", "concept_support_name TEXT",
                "support_type TEXT", "evidence_strength TEXT",
                "concept_support_score REAL DEFAULT 0",
                "evidence_type TEXT",
                "causal_modifier TEXT", "causal_context TEXT",
            ],
            "concept_tracking": [
                "definition TEXT", "covered_papers TEXT", "logic_pattern TEXT",
                "epistemic_entrenchment REAL DEFAULT 3.0",
                "state_transition TEXT", "evidence_strength TEXT",
                "limitation TEXT", "novelty_score REAL DEFAULT 0",
                "actionability TEXT", "project_fit TEXT",
            ],
            "frameworks": [
                "compressed_concepts TEXT", "core_logic TEXT",
                "worldview_shift TEXT", "suggested_experiment TEXT",
                "predictive_power REAL DEFAULT 0",
                "falsifiability REAL DEFAULT 0", "actionability REAL DEFAULT 0",
                "transferability REAL DEFAULT 0", "taste_fit REAL DEFAULT 0",
                "status TEXT DEFAULT 'candidate'",
            ],
            "deltas": [
                "evidence TEXT", "implication TEXT",
                "affected_projects TEXT", "status TEXT DEFAULT 'candidate'",
            ],
            "logic_patterns": [
                "status TEXT DEFAULT 'candidate'",
            ],
        }
        for table, columns in migrations.items():
            with self._connect() as conn:
                for col_def in columns:
                    col_name = col_def.split()[0]
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                    except Exception:
                        pass  # column already exists

    def insert_paper(self, paper: Paper, signal_score: int = 0,
                     signal_tier: str = "", signal_trajectory: float = 0.0,
                     signal_action: str = "", llm_reasoning: str = "",
                     concept_name: str = "", concept_drift: str = "",
                     causal: dict | None = None,
                     problem_class: str = "", novelty_type: str = "",
                     evidence_type: str = "",
                     strategic_value: str = "", concept_support_name: str = "",
                     support_type: str = "", evidence_strength: str = "",
                     concept_support_score: float = 0.0):
        row = paper.to_db_row()
        row["signal_score"] = signal_score
        row["signal_tier"] = signal_tier
        row["signal_trajectory"] = signal_trajectory
        row["signal_action"] = signal_action
        row["llm_reasoning"] = llm_reasoning
        row["concept_name"] = concept_name
        row["concept_drift"] = concept_drift
        row["causal_question"] = ""
        row["causal_constraint"] = ""
        row["causal_input"] = ""
        row["causal_transformation"] = ""
        row["causal_output"] = ""
        row["causal_outcome"] = ""
        row["causal_modifier"] = ""
        row["causal_context"] = ""
        if causal:
            row["causal_question"] = causal.get("question", "")
            row["causal_constraint"] = causal.get("constraint", "")
            row["causal_input"] = causal.get("input_state", "")
            row["causal_transformation"] = causal.get("transformation", "")
            row["causal_output"] = causal.get("output_state", "")
            row["causal_outcome"] = causal.get("outcome", "")
            row["causal_modifier"] = causal.get("modifier", "")
            row["causal_context"] = causal.get("context", "")

        row["problem_class"] = problem_class
        row["novelty_type"] = novelty_type
        row["evidence_type"] = evidence_type
        row["strategic_value"] = strategic_value
        row["concept_support_name"] = concept_support_name
        row["support_type"] = support_type
        row["evidence_strength"] = evidence_strength
        row["concept_support_score"] = concept_support_score

        with self._connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO papers
                (doi, title, abstract, journal, authors, affiliations,
                 pub_date, toc_image_url, signal_score, signal_tier,
                 signal_trajectory, signal_action, llm_reasoning,
                 concept_name, concept_drift,
                 causal_question, causal_constraint, causal_input,
                 causal_transformation, causal_output, causal_outcome,
                 causal_modifier, causal_context,
                 problem_class, novelty_type, evidence_type, strategic_value,
                 concept_support_name, support_type, evidence_strength,
                 concept_support_score)
                VALUES (:doi, :title, :abstract, :journal, :authors, :affiliations,
                        :pub_date, :toc_image_url, :signal_score, :signal_tier,
                        :signal_trajectory, :signal_action, :llm_reasoning,
                        :concept_name, :concept_drift,
                        :causal_question, :causal_constraint, :causal_input,
                        :causal_transformation, :causal_output, :causal_outcome,
                        :causal_modifier, :causal_context,
                        :problem_class, :novelty_type, :evidence_type, :strategic_value,
                        :concept_support_name, :support_type, :evidence_strength,
                        :concept_support_score)
            """, row)

    def doi_exists(self, doi: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM papers WHERE doi = ?", (doi,))
            return cur.fetchone() is not None

    def get_all_papers(self) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM papers ORDER BY ingested_at DESC")
            return [dict(row) for row in cur.fetchall()]

    def update_paper_toc_image_url(self, doi: str, toc_image_url: str) -> int:
        """Fill a missing TOC image URL for an existing paper row."""
        if not doi or not toc_image_url:
            return 0
        with self._connect() as conn:
            cur = conn.execute("""
                UPDATE papers
                SET toc_image_url = ?
                WHERE doi = ?
                  AND (toc_image_url IS NULL OR toc_image_url = '')
            """, (toc_image_url, doi))
            return cur.rowcount

    # ---- Curated Local Sources ----

    def upsert_local_source(self, doi: str, title: str, journal: str = "",
                            pub_year: str = "", bucket: str = "",
                            status: str = "", local_path: str = "",
                            note: str = "", source_manifest: str = ""):
        """Insert or update a curated local source imported from a manifest."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO local_sources
                (doi, title, journal, pub_year, bucket, status, local_path,
                 note, source_manifest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doi) DO UPDATE SET
                    title = excluded.title,
                    journal = excluded.journal,
                    pub_year = excluded.pub_year,
                    bucket = excluded.bucket,
                    status = excluded.status,
                    local_path = excluded.local_path,
                    note = excluded.note,
                    source_manifest = excluded.source_manifest,
                    updated_at = CURRENT_TIMESTAMP
            """, (doi, title, journal, pub_year, bucket, status, local_path,
                  note, source_manifest))

    def get_local_sources(self, bucket: str = None) -> list[dict]:
        with self._connect() as conn:
            if bucket:
                cur = conn.execute(
                    "SELECT * FROM local_sources WHERE bucket = ? ORDER BY pub_year DESC, title",
                    (bucket,))
            else:
                cur = conn.execute(
                    "SELECT * FROM local_sources ORDER BY bucket, pub_year DESC, title"
                )
            return [dict(row) for row in cur.fetchall()]

    def insert_log(self, fetched: int, new: int, high: int, maybe: int, ignored: int):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO ingest_log
                (papers_fetched, papers_new, papers_high_signal,
                 papers_maybe_signal, papers_ignored)
                VALUES (?, ?, ?, ?, ?)
            """, (fetched, new, high, maybe, ignored))

    def get_recent_logs(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM ingest_log ORDER BY run_at DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cur.fetchall()]

    # ---- Concept Accumulation Engine ----

    def upsert_concept(self, name: str, source_doi: str = "",
                       why_matters: str = "", connection: str = "",
                       missing_link: str = "", opportunity: str = "",
                       trajectory_weight: str = "medium",
                       status: str = "emerging",
                       seen_date: str = ""):
        """Insert or update a tracked concept. Increments appearance count on update."""
        seen_date = (seen_date or "").strip()[:10]
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT appearances, first_seen, last_seen FROM concept_tracking WHERE name = ?",
                (name.lower().strip(),)
            ).fetchone()

            if existing:
                new_count = existing["appearances"] + 1
                current_first = existing["first_seen"] or seen_date
                current_last = existing["last_seen"] or seen_date
                first_seen = min([d for d in [current_first, seen_date] if d], default=current_first)
                last_seen = max([d for d in [current_last, seen_date] if d], default=current_last)
                new_status = status
                if new_count >= 5:
                    new_status = "established"
                elif new_count >= 3:
                    new_status = "gaining momentum"

                conn.execute("""
                    UPDATE concept_tracking
                    SET appearances = ?, first_seen = COALESCE(NULLIF(?, ''), first_seen),
                        last_seen = COALESCE(NULLIF(?, ''), last_seen),
                        status = ?, why_matters = ?, connection = ?,
                        missing_link = ?, opportunity = ?,
                        trajectory_weight = ?
                    WHERE name = ?
                """, (new_count, first_seen, last_seen, new_status, why_matters, connection,
                      missing_link, opportunity, trajectory_weight,
                      name.lower().strip()))
            else:
                if not seen_date:
                    seen_date_expr = "date('now')"
                    params = (name.lower().strip(), source_doi, why_matters, connection,
                              missing_link, opportunity, trajectory_weight, status)
                    conn.execute(f"""
                        INSERT INTO concept_tracking
                        (name, first_seen, last_seen, appearances, source_doi,
                         why_matters, connection, missing_link, opportunity,
                         trajectory_weight, status)
                        VALUES (?, {seen_date_expr}, {seen_date_expr}, 1, ?, ?, ?, ?, ?, ?, ?)
                    """, params)
                    return
                conn.execute("""
                    INSERT INTO concept_tracking
                    (name, first_seen, last_seen, appearances, source_doi,
                     why_matters, connection, missing_link, opportunity,
                     trajectory_weight, status)
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                """, (name.lower().strip(), seen_date, seen_date, source_doi,
                      why_matters, connection, missing_link, opportunity,
                      trajectory_weight, status))

    def get_concept_momentum(self, min_appearances: int = 1) -> list[dict]:
        """Get concepts with momentum data, sorted by appearances descending."""
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT * FROM concept_tracking
                WHERE appearances >= ?
                ORDER BY appearances DESC
            """, (min_appearances,))
            return [dict(row) for row in cur.fetchall()]

    def get_emerging_concepts(self, threshold: int = 2) -> list[dict]:
        """Get concepts gaining momentum (appearances >= threshold)."""
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT * FROM concept_tracking
                WHERE appearances >= ?
                ORDER BY appearances DESC
            """, (threshold,))
            return [dict(row) for row in cur.fetchall()]

    def get_concept(self, name: str) -> dict | None:
        """Get a single concept by name."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM concept_tracking WHERE name = ?",
                (name.lower().strip(),)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_papers_with_causal(self, days_back: int = 7) -> list[dict]:
        """Get papers that have causal extraction data."""
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT doi, title, journal, signal_tier,
                       causal_question, causal_constraint, causal_input,
                       causal_transformation, causal_output, causal_outcome
                FROM papers
                WHERE causal_transformation IS NOT NULL AND causal_transformation != ''
                  AND ingested_at >= date('now', ?)
                ORDER BY ingested_at DESC
            """, (f'-{days_back} days',))
            return [dict(row) for row in cur.fetchall()]

    # ---- Logic Patterns ----

    def upsert_logic_pattern(self, pattern_name: str, pattern_type: str,
                             description: str = "", causal_template: str = "",
                             sample_dois: str = "", score: float = 0):
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, paper_count FROM logic_patterns WHERE pattern_name = ?",
                (pattern_name.lower().strip(),)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE logic_patterns
                    SET description = ?, causal_template = ?, sample_dois = ?,
                        paper_count = paper_count + 1, score = ?
                    WHERE id = ?
                """, (description, causal_template, sample_dois, score, existing["id"]))
            else:
                conn.execute("""
                    INSERT INTO logic_patterns
                    (pattern_name, pattern_type, description, causal_template,
                     paper_count, sample_dois, score)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                """, (pattern_name.lower().strip(), pattern_type, description,
                      causal_template, sample_dois, score))

    def get_logic_patterns(self) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM logic_patterns ORDER BY score DESC, paper_count DESC"
            )
            return [dict(row) for row in cur.fetchall()]

    # ---- Frameworks ----

    def insert_framework(self, framework_name: str, description: str = "",
                         covered_patterns: str = "", excluded_patterns: str = "",
                         compression_score: float = 0, novelty_score: float = 0,
                         core_logic: str = "", worldview_shift: str = "",
                         predictive_power: float = 0, falsifiability: float = 0,
                         actionability: float = 0, transferability: float = 0,
                         taste_fit: float = 0, suggested_experiment: str = "",
                         status: str = "candidate"):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO frameworks
                (framework_name, description, covered_patterns, excluded_patterns,
                 compression_score, novelty_score, core_logic, worldview_shift,
                 predictive_power, falsifiability, actionability,
                 transferability, taste_fit, suggested_experiment, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (framework_name.lower().strip(), description, covered_patterns,
                  excluded_patterns, compression_score, novelty_score, core_logic,
                  worldview_shift, predictive_power, falsifiability, actionability,
                  transferability, taste_fit, suggested_experiment, status))

    def set_framework_status(self, name: str, status: str):
        with self._connect() as conn:
            conn.execute("UPDATE frameworks SET status = ? WHERE framework_name = ?",
                         (status, name.lower().strip()))

    def set_concept_status(self, name: str, status: str):
        with self._connect() as conn:
            conn.execute("UPDATE concept_tracking SET status = ? WHERE name = ?",
                         (status, name.lower().strip()))

    def get_frameworks(self) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM frameworks ORDER BY compression_score DESC"
            )
            return [dict(row) for row in cur.fetchall()]

    # ---- Deltas ----

    def insert_delta(self, previous: str, new: str, delta: str, source_dois: str = ""):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO deltas
                (previous_assumption, new_assumption, delta, source_dois)
                VALUES (?, ?, ?, ?)
            """, (previous, new, delta, source_dois))

    def get_deltas(self) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM deltas ORDER BY detected_at DESC")
            return [dict(row) for row in cur.fetchall()]

    # ---- Research Memory ----

    def upsert_memory(self, item_type: str, item_name: str, status: str = "candidate",
                      reason: str = "", evidence_strength: str = "",
                      affected_projects: str = ""):
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM research_memory WHERE item_type = ? AND item_name = ?",
                (item_type, item_name.lower().strip())
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE research_memory
                    SET status = ?, reason = ?, evidence_strength = ?,
                        affected_projects = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, reason, evidence_strength, affected_projects, existing["id"]))
            else:
                conn.execute("""
                    INSERT INTO research_memory
                    (item_type, item_name, status, reason, evidence_strength, affected_projects)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (item_type, item_name.lower().strip(), status, reason,
                      evidence_strength, affected_projects))

    def get_memory(self, item_type: str = None, status: str = None) -> list[dict]:
        with self._connect() as conn:
            query = "SELECT * FROM research_memory WHERE 1=1"
            params = []
            if item_type:
                query += " AND item_type = ?"
                params.append(item_type)
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY updated_at DESC"
            cur = conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    # ---- Constraints / Predictions / Experiments ----

    def insert_constraint(self, name: str, framework_name: str, statement: str = "",
                          constraint_type: str = "requires", supporting_evidence: str = "",
                          violating_examples: str = "", confidence: float = 0,
                          prediction_power: float = 0, actionability: float = 0):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO constraints
                (name, framework_name, statement, constraint_type, supporting_evidence,
                 violating_examples, confidence, prediction_power, actionability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name.lower().strip(), framework_name.lower().strip(), statement,
                  constraint_type, supporting_evidence, violating_examples,
                  confidence, prediction_power, actionability))

    def get_constraints(self, framework_name: str = None) -> list[dict]:
        with self._connect() as conn:
            if framework_name:
                cur = conn.execute(
                    "SELECT * FROM constraints WHERE LOWER(framework_name) = ? ORDER BY prediction_power DESC",
                    (framework_name.lower().strip(),))
            else:
                cur = conn.execute("SELECT * FROM constraints ORDER BY prediction_power DESC")
            return [dict(row) for row in cur.fetchall()]

    def insert_prediction(self, constraint_id: int, prediction_type: str,
                          statement: str, expected_relationship: str = "",
                          failure_condition: str = ""):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO predictions
                (constraint_id, prediction_type, statement, expected_relationship, failure_condition)
                VALUES (?, ?, ?, ?, ?)
            """, (constraint_id, prediction_type, statement, expected_relationship, failure_condition))

    def get_predictions(self, constraint_id: int = None) -> list[dict]:
        with self._connect() as conn:
            if constraint_id:
                cur = conn.execute(
                    "SELECT * FROM predictions WHERE constraint_id = ?", (constraint_id,))
            else:
                cur = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC")
            return [dict(row) for row in cur.fetchall()]

    def insert_experiment(self, prediction_id: int, framework_name: str,
                          manipulate: str, measure: str, expected: str,
                          failure: str, priority: str = "medium"):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO experiments
                (prediction_id, framework_name, manipulate_variable, measure_variable,
                 expected_result, failure_condition, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (prediction_id, framework_name, manipulate, measure, expected, failure, priority))

    def get_experiments(self, framework_name: str = None) -> list[dict]:
        with self._connect() as conn:
            if framework_name:
                cur = conn.execute(
                    "SELECT * FROM experiments WHERE LOWER(framework_name) = ? ORDER BY priority, created_at DESC",
                    (framework_name.lower().strip(),))
            else:
                cur = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC")
            return [dict(row) for row in cur.fetchall()]

    # ---- Constraint Verification ----

    def upsert_verification(self, constraint_name: str, paper_doi: str,
                            result: str, confidence: float = 0, evidence: str = ""):
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO constraint_verifications
                (constraint_name, paper_doi, result, confidence, evidence)
                VALUES (?, ?, ?, ?, ?)
            """, (constraint_name.lower().strip(), paper_doi, result, confidence, evidence))

    def get_verifications(self, constraint_name: str = None) -> list[dict]:
        with self._connect() as conn:
            if constraint_name:
                cur = conn.execute(
                    "SELECT * FROM constraint_verifications WHERE constraint_name = ? ORDER BY verified_at DESC",
                    (constraint_name.lower().strip(),))
            else:
                cur = conn.execute("SELECT * FROM constraint_verifications ORDER BY verified_at DESC")
            return [dict(row) for row in cur.fetchall()]

    # ---- Trajectories ----

    def init_trajectories(self, trajectory_names: list[str]):
        """Seed trajectories from identity (idempotent)."""
        with self._connect() as conn:
            for name in trajectory_names:
                conn.execute(
                    "INSERT OR IGNORE INTO trajectories (name, confidence) VALUES (?, 'Stable')",
                    (name.lower().strip(),))

    def update_trajectory(self, name: str, confidence: str = None,
                          evidence_delta: int = 0, discovery_delta: int = 0,
                          validation_delta: int = 0, supporting_concepts: str = None):
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM trajectories WHERE name = ?", (name.lower().strip(),)
            ).fetchone()
            if not existing:
                conn.execute("INSERT INTO trajectories (name, confidence) VALUES (?, 'Stable')",
                             (name.lower().strip(),))
            updates = []
            params = []
            if confidence:
                updates.append("confidence = ?"); params.append(confidence)
            if evidence_delta:
                updates.append("evidence_count = evidence_count + ?"); params.append(evidence_delta)
            if discovery_delta:
                updates.append("discovery_count = discovery_count + ?"); params.append(discovery_delta)
            if validation_delta:
                updates.append("validation_count = validation_count + ?"); params.append(validation_delta)
            if supporting_concepts:
                updates.append("supporting_concepts = ?"); params.append(supporting_concepts)
            if updates:
                updates.append("last_updated = CURRENT_TIMESTAMP")
                params.append(name.lower().strip())
                conn.execute(f"UPDATE trajectories SET {', '.join(updates)} WHERE name = ?", params)

    def get_trajectories(self) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM trajectories ORDER BY evidence_count DESC")
            return [dict(row) for row in cur.fetchall()]

    # ---- Kernel State ----

    def set_kernel_state(self, key: str, value: str, category: str = "general"):
        with self._connect() as conn:
            conn.execute("""INSERT OR REPLACE INTO kernel_state (key, value, category, updated_at)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                         (key, value, category))

    def get_kernel_state(self, key: str) -> str | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT value FROM kernel_state WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def get_kernel_state_by_category(self, category: str) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM kernel_state WHERE category = ? ORDER BY updated_at DESC", (category,))
            return [dict(row) for row in cur.fetchall()]

    # ---- Kernel Objects / Revision Ledger ----

    def _kernel_json(self, value) -> str:
        if value in (None, ""):
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _kernel_object_key(self, object_type: str, title: str, source_ref: str = "") -> str:
        object_type = re.sub(r"[^a-z0-9]+", "-", (object_type or "object").lower()).strip("-") or "object"
        base = re.sub(r"[^a-z0-9]+", "-", (title or "untitled").lower()).strip("-")[:72] or "untitled"
        digest = hashlib.sha1(f"{object_type}|{title}|{source_ref}".encode("utf-8")).hexdigest()[:8]
        return f"{object_type}-{base}-{digest}"

    def upsert_kernel_object(self, object_type: str, title: str, statement: str = "",
                             status: str = "candidate", confidence: float = 0,
                             entrenchment: float = 0, source_type: str = "",
                             source_ref: str = "", evidence=None, metadata=None,
                             object_key: str = "") -> dict:
        object_type = (object_type or "claim").strip().lower()
        title = (title or "Untitled kernel object").strip()
        object_key = (object_key or self._kernel_object_key(object_type, title, source_ref)).strip()
        row = {
            "object_key": object_key,
            "object_type": object_type,
            "title": title,
            "statement": statement or "",
            "status": status or "candidate",
            "confidence": float(confidence or 0),
            "entrenchment": float(entrenchment or 0),
            "source_type": source_type or "",
            "source_ref": source_ref or "",
            "evidence": self._kernel_json(evidence),
            "metadata": self._kernel_json(metadata),
        }
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO kernel_objects
                (object_key, object_type, title, statement, status, confidence,
                 entrenchment, source_type, source_ref, evidence, metadata)
                VALUES (:object_key, :object_type, :title, :statement, :status,
                        :confidence, :entrenchment, :source_type, :source_ref,
                        :evidence, :metadata)
                ON CONFLICT(object_key) DO UPDATE SET
                    object_type = excluded.object_type,
                    title = excluded.title,
                    statement = excluded.statement,
                    status = excluded.status,
                    confidence = excluded.confidence,
                    entrenchment = excluded.entrenchment,
                    source_type = excluded.source_type,
                    source_ref = excluded.source_ref,
                    evidence = excluded.evidence,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, row)
        return self.get_kernel_object(object_key) or {}

    def get_kernel_object(self, object_key: str) -> dict | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM kernel_objects WHERE object_key = ?", (object_key,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_kernel_objects(self, object_type: str = None, status: str = None) -> list[dict]:
        with self._connect() as conn:
            query = "SELECT * FROM kernel_objects WHERE 1=1"
            params = []
            if object_type:
                query += " AND object_type = ?"
                params.append(object_type.strip().lower())
            if status:
                query += " AND status = ?"
                params.append(status.strip().lower())
            query += " ORDER BY entrenchment DESC, confidence DESC, updated_at DESC"
            cur = conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def revise_kernel_object(self, object_key: str, status: str = None,
                             confidence: float = None, entrenchment: float = None,
                             evidence=None, metadata=None) -> dict | None:
        updates = []
        params = []
        if status is not None:
            updates.append("status = ?"); params.append(status)
        if confidence is not None:
            updates.append("confidence = ?"); params.append(float(confidence))
        if entrenchment is not None:
            updates.append("entrenchment = ?"); params.append(float(entrenchment))
        if evidence is not None:
            updates.append("evidence = ?"); params.append(self._kernel_json(evidence))
        if metadata is not None:
            updates.append("metadata = ?"); params.append(self._kernel_json(metadata))
        if not updates:
            return self.get_kernel_object(object_key)
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(object_key)
        with self._connect() as conn:
            conn.execute(f"UPDATE kernel_objects SET {', '.join(updates)} WHERE object_key = ?", params)
        return self.get_kernel_object(object_key)

    def add_kernel_object_event(self, object_key: str, event_type: str,
                                previous_status: str = "", new_status: str = "",
                                previous_confidence: float = None, new_confidence: float = None,
                                previous_entrenchment: float = None, new_entrenchment: float = None,
                                reason: str = "", evidence_delta=None,
                                actor: str = "kernel") -> dict:
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO kernel_object_events
                (object_key, event_type, previous_status, new_status,
                 previous_confidence, new_confidence, previous_entrenchment,
                 new_entrenchment, reason, evidence_delta, actor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                object_key, event_type, previous_status, new_status,
                previous_confidence, new_confidence, previous_entrenchment,
                new_entrenchment, reason, self._kernel_json(evidence_delta), actor,
            ))
            event_id = cur.lastrowid
            row = conn.execute("SELECT * FROM kernel_object_events WHERE id = ?", (event_id,)).fetchone()
            return dict(row) if row else {}

    def get_kernel_object_events(self, object_key: str = None, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            if object_key:
                cur = conn.execute("""
                    SELECT * FROM kernel_object_events
                    WHERE object_key = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                """, (object_key, limit))
            else:
                cur = conn.execute("""
                    SELECT * FROM kernel_object_events
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                """, (limit,))
            return [dict(row) for row in cur.fetchall()]

    def link_kernel_objects(self, source_key: str, target_key: str,
                            relation_type: str, weight: float = 1,
                            reason: str = "") -> dict:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO kernel_object_links
                (source_key, target_key, relation_type, weight, reason)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_key, target_key, relation_type) DO UPDATE SET
                    weight = excluded.weight,
                    reason = excluded.reason
            """, (source_key, target_key, relation_type, float(weight or 1), reason))
            row = conn.execute("""
                SELECT * FROM kernel_object_links
                WHERE source_key = ? AND target_key = ? AND relation_type = ?
            """, (source_key, target_key, relation_type)).fetchone()
            return dict(row) if row else {}

    def get_kernel_object_links(self, object_key: str = None) -> list[dict]:
        with self._connect() as conn:
            if object_key:
                cur = conn.execute("""
                    SELECT * FROM kernel_object_links
                    WHERE source_key = ? OR target_key = ?
                    ORDER BY created_at DESC
                """, (object_key, object_key))
            else:
                cur = conn.execute("SELECT * FROM kernel_object_links ORDER BY created_at DESC")
            return [dict(row) for row in cur.fetchall()]

    # ---- Kernel Task Queue ----

    def _kernel_task_key(self, task_type: str, object_key: str = "", title: str = "") -> str:
        task_type = re.sub(r"[^a-z0-9]+", "-", (task_type or "task").lower()).strip("-") or "task"
        base_source = object_key or title or "untitled"
        base = re.sub(r"[^a-z0-9]+", "-", base_source.lower()).strip("-")[:72] or "untitled"
        digest = hashlib.sha1(f"{task_type}|{object_key}|{title}".encode("utf-8")).hexdigest()[:8]
        return f"{task_type}-{base}-{digest}"

    def upsert_kernel_task(self, task_type: str, title: str, description: str = "",
                           priority: float = 0, status: str = "open",
                           action_hint: str = "", object_key: str = "",
                           source_type: str = "", source_ref: str = "",
                           metadata=None, task_key: str = "") -> dict:
        task_type = (task_type or "review").strip().lower()
        title = (title or "Untitled kernel task").strip()
        task_key = (task_key or self._kernel_task_key(task_type, object_key, title)).strip()
        row = {
            "task_key": task_key,
            "task_type": task_type,
            "object_key": object_key or "",
            "title": title,
            "description": description or "",
            "priority": float(priority or 0),
            "status": status or "open",
            "action_hint": action_hint or "",
            "source_type": source_type or "",
            "source_ref": source_ref or "",
            "metadata": self._kernel_json(metadata),
        }
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO kernel_tasks
                (task_key, task_type, object_key, title, description, priority,
                 status, action_hint, source_type, source_ref, metadata)
                VALUES (:task_key, :task_type, :object_key, :title, :description,
                        :priority, :status, :action_hint, :source_type,
                        :source_ref, :metadata)
                ON CONFLICT(task_key) DO UPDATE SET
                    task_type = excluded.task_type,
                    object_key = excluded.object_key,
                    title = excluded.title,
                    description = excluded.description,
                    priority = excluded.priority,
                    status = CASE
                        WHEN kernel_tasks.status IN ('done', 'dismissed') THEN kernel_tasks.status
                        ELSE excluded.status
                    END,
                    action_hint = excluded.action_hint,
                    source_type = excluded.source_type,
                    source_ref = excluded.source_ref,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, row)
        return self.get_kernel_task(task_key) or {}

    def get_kernel_task(self, task_key: str) -> dict | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM kernel_tasks WHERE task_key = ?", (task_key,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_kernel_tasks(self, status: str = None, task_type: str = None,
                         limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            query = "SELECT * FROM kernel_tasks WHERE 1=1"
            params = []
            if status:
                query += " AND status = ?"
                params.append(status.strip().lower())
            if task_type:
                query += " AND task_type = ?"
                params.append(task_type.strip().lower())
            query += " ORDER BY priority DESC, updated_at DESC LIMIT ?"
            params.append(limit)
            cur = conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def revise_kernel_task(self, task_key: str, status: str,
                           reason: str = "", actor: str = "human") -> dict | None:
        task = self.get_kernel_task(task_key)
        if not task:
            return None
        previous_status = task.get("status", "open")
        new_status = (status or "open").strip().lower()
        with self._connect() as conn:
            conn.execute("""
                UPDATE kernel_tasks
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE task_key = ?
            """, (new_status, task_key))
            conn.execute("""
                INSERT INTO kernel_task_events
                (task_key, event_type, previous_status, new_status, reason, actor)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (task_key, "status_change", previous_status, new_status, reason, actor))
        return self.get_kernel_task(task_key)

    def get_kernel_task_events(self, task_key: str = None, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            if task_key:
                cur = conn.execute("""
                    SELECT * FROM kernel_task_events
                    WHERE task_key = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                """, (task_key, limit))
            else:
                cur = conn.execute("""
                    SELECT * FROM kernel_task_events
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                """, (limit,))
            return [dict(row) for row in cur.fetchall()]

    # ---- Justification Tracking (Doyle TMS) ----

    def insert_justification(self, concept_name: str, paper_doi: str,
                              support_type: str = "", evidence_strength: str = "",
                              justification_text: str = ""):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO justifications
                (concept_name, paper_doi, support_type, evidence_strength, justification_text)
                VALUES (?, ?, ?, ?, ?)
            """, (concept_name.lower().strip(), paper_doi, support_type,
                  evidence_strength, justification_text))

    def get_justifications(self, concept_name: str = None,
                            paper_doi: str = None,
                            active_only: bool = True) -> list[dict]:
        with self._connect() as conn:
            query = "SELECT * FROM justifications WHERE 1=1"
            params = []
            if concept_name:
                query += " AND concept_name = ?"
                params.append(concept_name.lower().strip())
            if paper_doi:
                query += " AND paper_doi = ?"
                params.append(paper_doi)
            if active_only:
                query += " AND is_active = 1"
            query += " ORDER BY created_at DESC"
            cur = conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def withdraw_justification(self, concept_name: str, paper_doi: str):
        """Deactivate a justification (paper found unreliable)."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE justifications SET is_active = 0
                WHERE concept_name = ? AND paper_doi = ?
            """, (concept_name.lower().strip(), paper_doi))

    def recalculate_concept_confidence(self, concept_name: str):
        """Recalculate concept confidence from active justifications only."""
        justs = self.get_justifications(concept_name=concept_name, active_only=True)
        active_count = len(justs)
        if active_count == 0:
            return 0
        # Weight by evidence strength
        weight = 0
        for j in justs:
            es = (j.get("evidence_strength") or "").strip()
            if "High" in es: weight += 2
            elif "Medium" in es: weight += 1
            else: weight += 0.5
        return min(10, weight)

    # ---- Belief Revision (AGM) ----

    def set_entrenchment(self, concept_name: str, entrenchment: float):
        with self._connect() as conn:
            conn.execute("""
                UPDATE concept_tracking SET epistemic_entrenchment = ?
                WHERE name = ?
            """, (entrenchment, concept_name.lower().strip()))

    def get_entrenchment(self, concept_name: str) -> float:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT epistemic_entrenchment FROM concept_tracking WHERE name = ?",
                (concept_name.lower().strip(),))
            row = cur.fetchone()
            return float(row[0]) if row and row[0] else 3.0

    def contract_concept(self, concept_name: str, reason: str = "") -> dict:
        """AGM contraction: weaken a concept's confidence."""
        current = self.get_concept(concept_name)
        if not current:
            return {"status": "not_found"}
        old_apps = current.get("appearances", 0)
        old_ent = current.get("epistemic_entrenchment", 3.0)
        new_ent = max(0, old_ent - 1.0)
        self.set_entrenchment(concept_name, new_ent)
        # Also deactivate the weakest justification
        justs = self.get_justifications(concept_name=concept_name, active_only=True)
        if justs:
            weakest = min(justs, key=lambda j: {"High": 3, "Medium": 2, "Low": 1}.get(
                (j.get("evidence_strength") or "").strip(), 1))
            self.withdraw_justification(concept_name, weakest["paper_doi"])
        new_confidence = self.recalculate_concept_confidence(concept_name)
        return {
            "concept": concept_name,
            "old_entrenchment": old_ent,
            "new_entrenchment": new_ent,
            "old_appearances": old_apps,
            "new_confidence": new_confidence,
            "reason": reason,
        }

    def get_verification_summary(self) -> dict:
        """Summary: how many constraints are supported vs violated vs untested."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(DISTINCT name) FROM constraints").fetchone()[0]
            supported = conn.execute(
                "SELECT COUNT(DISTINCT constraint_name) FROM constraint_verifications WHERE result = 'supported'"
            ).fetchone()[0]
            violated = conn.execute(
                "SELECT COUNT(DISTINCT constraint_name) FROM constraint_verifications WHERE result = 'violated'"
            ).fetchone()[0]
            return {"total_constraints": total, "supported": supported,
                    "violated": violated, "untested": total - supported - violated}

    def get_memory_summary(self) -> dict:
        """Answer: What do we believe? Why? What changed? What's uncertain? What next?"""
        with self._connect() as conn:
            approved = conn.execute(
                "SELECT item_type, item_name, reason FROM research_memory WHERE status = 'approved'"
            ).fetchall()
            rejected = conn.execute(
                "SELECT item_type, item_name, reason FROM research_memory WHERE status = 'rejected'"
            ).fetchall()
            contradictions = conn.execute(
                "SELECT * FROM research_memory WHERE item_type = 'contradiction' AND status != 'resolved'"
            ).fetchall()
            decisions = conn.execute(
                "SELECT * FROM research_memory WHERE item_type = 'decision' ORDER BY updated_at DESC LIMIT 10"
            ).fetchall()
            next_actions = conn.execute(
                "SELECT * FROM research_memory WHERE item_type = 'next_action' AND status != 'done'"
            ).fetchall()
            return {
                "beliefs": [dict(r) for r in approved],
                "rejected": [dict(r) for r in rejected],
                "contradictions": [dict(r) for r in contradictions],
                "decisions": [dict(r) for r in decisions],
                "next_actions": [dict(r) for r in next_actions],
            }
