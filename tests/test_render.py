from datetime import date
from psil.digest.render import render_digest
from psil.store.models import Paper


def test_render_digest_has_all_sections():
    high_priority = [
        (Paper(
            doi="10.0/high", title="High Priority Paper", journal="Nature Materials",
            abstract="Important work."
        ), 15, ["OECT", "dynamic biointerfaces"], {
            "paper_type": "transduction_or_device_paper",
            "judgment_mode": "transduction_route",
            "relevance": 9, "novelty": 8, "bridge": 7, "trajectory_influence": 9,
            "final_score": 8.4,
            "why_matters": "- Changes everything.\n- Opens new field.",
            "potential_connection": "- Directly relevant to OECT sensing.",
            "weakness": "None.",
            "action": "Read immediately",
            "signal_tier": "HIGH_PRIORITY",
        })
    ]
    important = [
        (Paper(
            doi="10.0/imp", title="Important Paper", journal="ACS Nano",
            abstract="Interesting work."
        ), 8, ["hydrogel", "biosensing"], {
            "relevance": 7, "novelty": 6, "bridge": 5, "trajectory_influence": 6,
            "final_score": 6.0,
            "why_matters": "- Interesting approach.",
            "potential_connection": "- Could inform hydrogel electronics work.",
            "weakness": "Limited validation.",
            "action": "Review this week",
            "signal_tier": "IMPORTANT",
        })
    ]
    potential = [
        (Paper(
            doi="10.0/pot", title="Potential Paper", journal="Adv Materials",
            abstract="Mildly interesting."
        ), 3, ["nanoparticle"], {
            "relevance": 5, "novelty": 4, "bridge": 3, "trajectory_influence": 4,
            "final_score": 4.2,
            "why_matters": "- Incremental advance.",
            "signal_tier": "POTENTIAL",
        })
    ]
    watchlist_entries = [
        (Paper(
            doi="10.0/watch", title="Watchlist Paper", journal="ACS Nano",
            abstract="Gold nanocluster with two-photon properties."
        ), 2, ["gold nanocluster", "two-photon"], {
            "relevance": 3, "novelty": 4, "bridge": 3, "trajectory_influence": 5,
            "final_score": 3.6,
            "why_matters": "- Gold nanocluster with two-photon absorption.",
            "signal_tier": "WATCHLIST",
        })
    ]
    low_priority = [
        (Paper(
            doi="10.0/low", title="Low Priority Paper", journal="JACS",
            abstract="Routine."
        ), 1, ["drug delivery"], {
            "relevance": 2, "novelty": 1, "bridge": 1, "trajectory_influence": 1,
            "final_score": 1.3,
            "signal_tier": "LOW_PRIORITY",
        })
    ]
    ignored = [
        Paper(doi="10.0/ignored", title="Correction to Something", journal="Nature",
              abstract="Author correction.")
    ]
    concept_feed = [
        {
            "name": "radiative Q-factor modulation",
            "source": "High Priority Paper",
            "why_matters": "New paradigm for field localization sensing.",
            "connection": "Connects to nanophotonics and Q-factor sensing.",
            "dictionary_update": "Add to nanophotonics signal dictionary.",
            "action": "Add to dictionary",
        }
    ]
    concept_gap_map = [
        {
            "concept": "radiative Q-factor modulation",
            "current_connection": "EV sensing",
            "potential_connection": "Organoid EV sensing",
            "missing_link": "Bioelectronic readout",
            "opportunity": "Q-factor modulation integrated with OECT platforms",
        }
    ]
    daily_summary = {
        "best_paper": "High Priority Paper",
        "best_concept": "radiative Q-factor modulation",
        "most_relevant_current": "High Priority Paper",
        "most_relevant_future": "Watchlist Paper",
        "read_first": "High Priority Paper",
        "concept_to_add": "radiative Q-factor modulation",
        "emerging_direction": "nanophotonic biosensing",
        "confidence": "Medium",
    }

    output = render_digest(date(2026, 5, 27), high_priority, important, potential,
                           watchlist_entries, [], low_priority, ignored,
                           concept_feed, concept_gap_map, [], [], [],
                           daily_summary)

    assert "Daily Scientific Signals" in output
    assert "HIGH PRIORITY" in output
    assert "IMPORTANT" in output
    assert "POTENTIAL" in output
    assert "WATCHLIST" in output
    assert "LOW PRIORITY" in output
    assert "IGNORE" in output
    assert "CONCEPT FEED" in output
    assert "CONCEPT GAP MAP" in output
    assert "DAILY SUMMARY" in output
    assert "High Priority Paper" in output
    assert "Nature Materials" in output
    assert "Changes everything" in output
    assert "Correction to Something" in output
    assert "radiative Q-factor modulation" in output
    assert "Paper Type" in output
    assert "transduction_route" in output
    assert "Bioelectronic readout" in output
    assert "8.4" in output
    assert "Read immediately" in output
    assert "nanophotonic biosensing" in output
    assert "Medium" in output


def test_render_digest_handles_empty_sections():
    output = render_digest(date(2026, 5, 27), [], [], [], [], [], [], [],
                           [], [], [], [], [], {})
    assert "No high-priority papers today" in output
    assert "No watchlist papers today" in output
    assert "No new concepts detected today" in output
    assert "No concept gaps mapped today" in output


def test_render_digest_includes_doi_links():
    high = [
        (Paper(doi="10.0/test", title="Test", journal="Nature Materials"), 10,
         ["OECT"], {
            "relevance": 9, "novelty": 8, "bridge": 7, "trajectory_influence": 8,
            "final_score": 8.1,
            "why_matters": "- Important.",
            "potential_connection": "- Relevant.",
            "weakness": "None.",
            "action": "Read immediately",
            "signal_tier": "HIGH_PRIORITY",
        })
    ]
    output = render_digest(date(2026, 5, 27), high, [], [], [], [], [], [],
                           [], [], [], [], [], {})
    assert "10.0/test" in output
