import json
import sqlite3

from click.testing import CliRunner

from psil.benchmark.v3_intake import (
    render_ablation_markdown,
    render_legacy_backfill_markdown,
    render_review_smoke_markdown,
    run_v3_intake_ablation,
    run_v3_legacy_digest_backfill,
    run_v3_review_smoke,
    select_legacy_digest_papers,
)
from psil.cli import main


def _row(
    title,
    tier,
    final,
    trajectory,
    bridge,
    concept,
    *,
    abstract=None,
    why=None,
    connection=None,
):
    return {
        "title": title,
        "doi": f"doi:10.0/{title.lower().replace(' ', '-')}",
        "abstract": abstract or "A paper with project relevance.",
        "signal_tier": tier,
        "llm_reasoning": json.dumps(
            {
                "final_score": final,
                "trajectory_influence": trajectory,
                "bridge": bridge,
                "concept_support": concept,
                "why_matters": why or "Provides a testable mechanism.",
                "potential_connection": connection or "",
            }
        ),
    }


def _legacy_row(title, tier, final, trajectory, bridge, concept, abstract):
    row = _row(title, tier, final, trajectory, bridge, concept)
    row["abstract"] = abstract
    row["journal"] = "Legacy Journal"
    row["ingested_at"] = "2026-06-01 00:00:00"
    return row


def _create_papers_table(conn):
    conn.execute(
        """
        CREATE TABLE papers (
            doi TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            journal TEXT,
            signal_tier TEXT,
            signal_score INTEGER,
            signal_trajectory REAL,
            llm_reasoning TEXT,
            concept_support_score REAL,
            problem_class TEXT,
            novelty_type TEXT,
            ingested_at TEXT
        )
        """
    )


def _insert_row(conn, row):
    conn.execute(
        """
        INSERT INTO papers (
            doi, title, abstract, journal, signal_tier, signal_score, signal_trajectory,
            llm_reasoning, concept_support_score, problem_class, novelty_type,
            ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["doi"],
            row["title"],
            row["abstract"],
            row.get("journal", "Test Journal"),
            row["signal_tier"],
            5,
            row.get("signal_trajectory", 0.0),
            row["llm_reasoning"],
            row.get("concept_support_score", 0.0),
            row.get("problem_class", ""),
            row.get("novelty_type", ""),
            row.get("ingested_at", "2026-06-11 00:00:00"),
        ),
    )


def test_v3_intake_ablation_measures_consensus_and_dampening(tmp_path):
    rows = [
        _row(
            "OECT EV functional readout",
            "IMPORTANT",
            8.0,
            8,
            8,
            8,
            abstract="An OECT biosensor preserves extracellular vesicle biological state through a functional readout.",
            why="Demonstrates traceable coupling between molecular recognition, EV disease state, and OECT electronic readout.",
            connection="Directly relevant to OECT-based EV sensing.",
        ),
        _row(
            "Real-time living bioelectronic transducer",
            "IMPORTANT",
            7.5,
            7,
            7,
            7,
            abstract="A real-time bioelectronic sensor couples living cell metabolism to an organic electrochemical transistor signal.",
            why="Demonstrates a direct living transducer platform with biological state and electronic signal coupled.",
            connection="Could inform design of hybrid living-material interfaces for future OECT sensing.",
        ),
        _row(
            "Partial OECT platform",
            "IMPORTANT",
            6.8,
            7,
            7,
            3,
            abstract="An OECT platform improves sensor integration but does not specify the biological state.",
            why="Provides a platform for electronic readout.",
            connection="Could inform OECT sensing design.",
        ),
        _row(
            "Generic assay platform",
            "POTENTIAL",
            5.8,
            7,
            7,
            7,
            abstract="A platform workflow improves sample handling without a specific biological readout.",
            why="Could improve lab workflow.",
        ),
        _row(
            "Off-topic quantum resonator",
            "WATCHLIST",
            4.0,
            2,
            4,
            7,
            abstract="A mechanical resonator for quantum computing.",
            why="Useful for quantum information processing.",
        ),
    ]

    result = run_v3_intake_ablation(rows, kernel_dir=tmp_path / "kernel" / "v3")

    assert result["paper_count"] == 5
    assert result["full_kernel"]["validation_status"] == "ok"
    assert result["full_kernel"]["relation_counts"]["support"] == 2
    assert result["full_kernel"]["relation_counts"]["underdetermined"] == 1
    assert result["full_kernel"]["relation_counts"]["neutral"] == 2
    assert result["full_kernel"]["contested_queue_count"] == 0
    assert result["full_kernel"]["pending_queue_count"] == 1
    assert result["ablations"]["without_relation_consensus"]["unstable_commits_prevented"] > 0
    assert (
        result["ablations"]["without_confidence_dampening"]["final_confidence"]
        > result["full_kernel"]["final_confidence"]
    )
    assert (
        result["ablations"]["without_entrenchment_policy"]["final_entrenchment"]
        > result["full_kernel"]["final_entrenchment"]
    )
    support_deltas = [
        item["confidence_delta"]
        for item in result["papers"]
        if item["relation"] == "support"
    ]
    assert support_deltas
    assert all(delta > 0 for delta in support_deltas)
    assert support_deltas[-1] < support_deltas[0]


def test_render_ablation_markdown_contains_empirical_metrics(tmp_path):
    rows = [
        _row("Unanimous high", "IMPORTANT", 8.0, 8, 8, 8),
        _row("Mixed high", "IMPORTANT", 6.8, 7, 7, 3),
    ]
    result = run_v3_intake_ablation(rows, kernel_dir=tmp_path / "kernel" / "v3")

    markdown = render_ablation_markdown(result)

    assert "Unstable commits prevented by consensus" in markdown
    assert "Pending evidence queue" in markdown
    assert "Overconfidence reduction" in markdown
    assert "Over-entrenchment reduction" in markdown
    assert "confidence_delta" in markdown
    assert "Unanimous high" in markdown


def test_v3_review_smoke_validates_one_real_review_item(tmp_path):
    row = _row("Mixed high", "IMPORTANT", 6.8, 7, 7, 3)

    result = run_v3_review_smoke(row, kernel_dir=tmp_path / "kernel" / "v3")

    assert result["validation_status"] == "ok"
    assert result["paper"]["title"] == "Mixed high"
    assert result["evidence"]["source_ref"] == row["doi"]
    assert result["revision"]["triggering_evidence_ids"] == [result["evidence"]["id"]]
    assert "confidence_delta_policy" in result["revision"]
    assert "entrenchment_delta_policy" in result["revision"]
    assert "pending_queue_count" in result


def test_render_review_smoke_markdown_contains_kernel_path(tmp_path):
    row = _row("Mixed high", "IMPORTANT", 6.8, 7, 7, 3)
    result = run_v3_review_smoke(row, kernel_dir=tmp_path / "kernel" / "v3")

    markdown = render_review_smoke_markdown(result)

    assert "V3 Real Review Smoke Test" in markdown
    assert "Confidence delta policy" in markdown
    assert "Entrenchment delta policy" in markdown
    assert "Pending evidence queue count" in markdown
    assert "Validation: ok" in markdown


def test_select_legacy_digest_papers_excludes_curated_library(tmp_path):
    db_path = tmp_path / "psil.db"
    conn = sqlite3.connect(db_path)
    _create_papers_table(conn)
    rows = [
        _legacy_row("Legacy ranked", "IMPORTANT", 7.5, 7, 7, 7, "sensor paper"),
        _legacy_row("Curated source", "CURATED_LIBRARY", 7.5, 7, 7, 7, "local source"),
        _legacy_row("Blank reasoning", "IMPORTANT", 7.5, 7, 7, 7, "blank"),
    ]
    rows[2]["llm_reasoning"] = ""
    for row in rows:
        _insert_row(conn, row)
    conn.commit()
    conn.close()

    selected = select_legacy_digest_papers(db_path, limit=10)

    assert [row["title"] for row in selected] == ["Legacy ranked"]


def test_v3_legacy_digest_backfill_routes_by_mode(tmp_path):
    rows = [
        _legacy_row(
            "Photochemical polariton mechanism",
            "IMPORTANT",
            8.0,
            8,
            8,
            8,
            "fundamental polariton photochemistry mechanism",
        ),
        _legacy_row(
            "EV sensor platform",
            "POTENTIAL",
            5.8,
            7,
            7,
            7,
            "sensor platform for diagnostic detection",
        ),
        _legacy_row(
            "Bioelectronics review",
            "WATCHLIST",
            4.0,
            2,
            2,
            1,
            "review perspective on biological mechanisms",
        ),
    ]

    result = run_v3_legacy_digest_backfill(rows, kernel_dir=tmp_path / "kernel" / "v3")

    assert result["durable_kernel_impact"] == "none"
    assert result["paper_count"] == 3
    assert result["full_kernel"]["validation_status"] == "ok"
    assert result["full_kernel"]["relation_counts"]["support"] == 2
    assert result["full_kernel"]["relation_counts"]["underdetermined"] == 1
    assert result["full_kernel"]["pending_queue_count"] == 1
    assert result["full_kernel"]["contested_queue_count"] == 0
    assert set(result["full_kernel"]["mode_counts"]) == {
        "mechanism_to_coupling",
        "synthesis_prior",
        "transduction_validity",
    }
    assert len(result["beliefs"]) == 3


def test_v3_legacy_digest_backfill_reads_content_not_tier(tmp_path):
    rows = [
        _legacy_row(
            "Real-time bioelectronic sensors based on electroactive bacteria with organic electrochemical transistors",
            "POTENTIAL",
            5.2,
            5,
            6,
            2,
            "",
        ),
        _legacy_row(
            "Multimodal activity-affinity assay of ADAM-10 extracellular vesicles in untreated plasma reveals metastatic stage of colorectal cancer",
            "POTENTIAL",
            5.2,
            5,
            6,
            3,
            "",
        ),
        _legacy_row(
            "A plant immune receptor mediates tritrophic interactions by linking caterpillar defense and parasitoid attraction",
            "LOW_PRIORITY",
            1.6,
            1,
            1,
            1,
            "A plant immunity paper about ecological interactions.",
        ),
    ]
    first_reasoning = json.loads(rows[0]["llm_reasoning"])
    first_reasoning["why_matters"] = (
        "Demonstrates real-time bioelectronic sensing using electroactive bacteria "
        "coupled to an OECT, a direct living transducer platform."
    )
    first_reasoning["potential_connection"] = (
        "The OECT as a readout platform is directly relevant and could inform design "
        "of hybrid living-material interfaces for organoid or EV sensing."
    )
    rows[0]["llm_reasoning"] = json.dumps(first_reasoning)

    second_reasoning = json.loads(rows[1]["llm_reasoning"])
    second_reasoning["why_matters"] = (
        "Demonstrates a functional readout of ADAM-10 on EVs directly from untreated "
        "plasma, linking EV enzyme activity to colorectal cancer metastatic stage."
    )
    second_reasoning["potential_connection"] = (
        "This activity-based EV phenotyping concept could be coupled with OECT-based "
        "electronic readout and aligns with organoid + EV + sensing platforms."
    )
    rows[1]["llm_reasoning"] = json.dumps(second_reasoning)

    result = run_v3_legacy_digest_backfill(rows, kernel_dir=tmp_path / "kernel" / "v3")
    by_title = {paper["title"]: paper["relation"] for paper in result["papers"]}

    assert by_title[rows[0]["title"]] == "support"
    assert by_title[rows[1]["title"]] == "support"
    assert by_title[rows[2]["title"]] == "neutral"
    assert result["full_kernel"]["relation_counts"].get("challenge", 0) == 0


def test_render_legacy_backfill_markdown_contains_mode_and_queue_metrics(tmp_path):
    rows = [
        _legacy_row(
            "Photochemical polariton mechanism",
            "IMPORTANT",
            8.0,
            8,
            8,
            8,
            "fundamental polariton photochemistry mechanism",
        ),
        _legacy_row(
            "EV sensor platform",
            "POTENTIAL",
            5.8,
            7,
            7,
            7,
            "sensor platform for diagnostic detection",
        ),
    ]
    result = run_v3_legacy_digest_backfill(rows, kernel_dir=tmp_path / "kernel" / "v3")

    markdown = render_legacy_backfill_markdown(result)

    assert "V3 Legacy Digest Backfill Dry Run" in markdown
    assert "Mode Breakdown" in markdown
    assert "Pending evidence queue" in markdown
    assert "Durable kernel impact: none" in markdown


def test_cli_v3_intake_ablation_writes_report(tmp_path):
    db_path = tmp_path / "psil.db"
    output_path = tmp_path / "ablation.md"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE papers (
            doi TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            signal_tier TEXT,
            signal_score INTEGER,
            signal_trajectory REAL,
            llm_reasoning TEXT,
            concept_support_score REAL,
            problem_class TEXT,
            novelty_type TEXT,
            ingested_at TEXT
        )
        """
    )
    for row in [
        _row("Unanimous high", "HIGH_PRIORITY", 8.0, 8, 8, 8),
        _row("Important mixed", "IMPORTANT", 6.8, 7, 7, 3),
        _row("Potential mixed", "POTENTIAL", 5.8, 7, 7, 7),
        _row("Watchlist low", "WATCHLIST", 3.8, 2, 2, 1),
    ]:
        conn.execute(
            """
            INSERT INTO papers (
                doi, title, abstract, signal_tier, signal_score, signal_trajectory,
                llm_reasoning, concept_support_score, problem_class, novelty_type,
                ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["doi"],
                row["title"],
                row["abstract"],
                row["signal_tier"],
                5,
                0.0,
                row["llm_reasoning"],
                0.0,
                "",
                "",
                "2026-06-11 00:00:00",
            ),
        )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        main,
        [
            "v3-intake-ablation",
            "--db-path",
            str(db_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "unstable commits prevented" in result.output
    assert "Overconfidence reduction" in output_path.read_text(encoding="utf-8")
    assert "Over-entrenchment reduction" in output_path.read_text(encoding="utf-8")
    assert "confidence_delta" in output_path.read_text(encoding="utf-8")


def test_cli_v3_review_smoke_writes_report(tmp_path):
    db_path = tmp_path / "psil.db"
    output_path = tmp_path / "review-smoke.md"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE papers (
            doi TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            journal TEXT,
            signal_tier TEXT,
            signal_score INTEGER,
            signal_trajectory REAL,
            llm_reasoning TEXT,
            concept_support_score REAL,
            problem_class TEXT,
            novelty_type TEXT,
            ingested_at TEXT
        )
        """
    )
    row = _row("Mixed high", "IMPORTANT", 6.8, 7, 7, 3)
    conn.execute(
        """
        INSERT INTO papers (
            doi, title, abstract, journal, signal_tier, signal_score, signal_trajectory,
            llm_reasoning, concept_support_score, problem_class, novelty_type,
            ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["doi"],
            row["title"],
            row["abstract"],
            "Test Journal",
            row["signal_tier"],
            5,
            0.0,
            row["llm_reasoning"],
            0.0,
            "",
            "",
            "2026-06-11 00:00:00",
        ),
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        main,
        [
            "v3-review-smoke",
            "--db-path",
            str(db_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "relation=" in result.output
    assert "V3 Real Review Smoke Test" in output_path.read_text(encoding="utf-8")
    assert "Confidence delta policy" in output_path.read_text(encoding="utf-8")


def test_cli_v3_backfill_digest_writes_report(tmp_path):
    db_path = tmp_path / "psil.db"
    output_path = tmp_path / "legacy-backfill.md"
    conn = sqlite3.connect(db_path)
    _create_papers_table(conn)
    for row in [
        _legacy_row(
            "Photochemical polariton mechanism",
            "IMPORTANT",
            8.0,
            8,
            8,
            8,
            "fundamental polariton photochemistry mechanism",
        ),
        _legacy_row(
            "EV sensor platform",
            "POTENTIAL",
            5.8,
            7,
            7,
            7,
            "sensor platform for diagnostic detection",
        ),
    ]:
        _insert_row(conn, row)
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        main,
        [
            "v3-backfill-digest",
            "--db-path",
            str(db_path),
            "--limit",
            "2",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "Backfill dry-run" in result.output
    assert "V3 Legacy Digest Backfill Dry Run" in output_path.read_text(encoding="utf-8")
