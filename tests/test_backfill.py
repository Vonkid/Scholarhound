from datetime import date
from unittest.mock import MagicMock, patch

from psil.backfill import (
    DEFAULT_BACKFILL_START,
    default_backfill_window,
    fetch_crossref_range,
    iter_date_chunks,
    select_backfill_journals,
    summarize_candidates,
)
from psil.store.db import Database
from psil.store.models import Paper


def test_default_project_backfill_start_is_2020():
    assert DEFAULT_BACKFILL_START == date(2020, 1, 1)


def test_default_backfill_window_uses_year_count():
    start, end = default_backfill_window(5, today=date(2026, 6, 8))

    assert end == date(2026, 6, 8)
    assert start == date(2021, 6, 8)


def test_iter_date_chunks_year_boundaries():
    chunks = list(iter_date_chunks(date(2024, 6, 1), date(2026, 2, 3), "year"))

    assert chunks == [
        (date(2024, 6, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 2, 3)),
    ]


def test_select_backfill_journals_prefers_focused_subset():
    journals = [
        {"name": "Nature Biomedical Engineering", "issn": "2157-846X"},
        {"name": "Nature Communications", "issn": "2041-1723"},
        {"name": "Science Advances", "issn": "2375-2548"},
        {"name": "Biosensors and Bioelectronics", "issn": "0956-5663"},
        {"name": "Chemical Society Reviews", "issn": "0306-0012"},
        {"name": "Nature Materials"},
    ]

    selected = select_backfill_journals(journals, focused=True)

    assert [j["name"] for j in selected] == ["Nature Biomedical Engineering"]


def test_select_backfill_journals_allows_explicit_broad_journal():
    journals = [
        {"name": "Biosensors and Bioelectronics", "issn": "0956-5663"},
        {"name": "Nature Biomedical Engineering", "issn": "2157-846X"},
    ]

    selected = select_backfill_journals(
        journals,
        focused=True,
        names=("Biosensors and Bioelectronics",),
    )

    assert [j["name"] for j in selected] == ["Biosensors and Bioelectronics"]


@patch("psil.backfill.requests.get")
def test_fetch_crossref_range_uses_inclusive_pub_date_filter(mock_get):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "message": {
            "total-results": 1,
            "items": [
                {
                    "DOI": "10.1038/example",
                    "title": ["Historical OECT biosensor"],
                    "abstract": "OECT biosensing.",
                    "published-online": {"date-parts": [[2024, 3, 2]]},
                }
            ],
        }
    }
    mock_get.return_value = response

    papers = fetch_crossref_range(
        "2157-846X",
        "Nature Biomedical Engineering",
        date(2024, 1, 1),
        date(2024, 12, 31),
    )

    assert len(papers) == 1
    assert papers[0].doi == "10.1038/example"
    params = mock_get.call_args.kwargs["params"]
    assert params["filter"] == "from-pub-date:2024-01-01,until-pub-date:2024-12-31"


def test_summarize_candidates_skips_existing_and_prefilters(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    db.insert_paper(Paper(doi="10.0/existing", title="Existing OECT paper"))
    papers = [
        Paper(doi="10.0/existing", title="Existing OECT paper"),
        Paper(doi="10.0/pass", title="OECT biosensing with EV diagnostics"),
        Paper(doi="10.0/ignore", title="Regional geology pattern"),
    ]

    passed, ignored, new_count = summarize_candidates(papers, db, threshold=0)

    assert new_count == 2
    assert [p.doi for p, _ in passed] == ["10.0/pass"]
    assert [p.doi for p in ignored] == ["10.0/ignore"]
