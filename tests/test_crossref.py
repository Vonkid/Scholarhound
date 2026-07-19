from unittest.mock import patch, MagicMock
from datetime import date
from psil.ingest.crossref import fetch_crossref, parse_crossref_work
from psil.store.models import Paper


SAMPLE_WORKS_RESPONSE = {
    "message": {
        "items": [
            {
                "DOI": "10.1126/science.adg8758",
                "title": ["Mechanically gated bioelectronic transduction"],
                "abstract": "We report a novel force-coupled electrochemical system.",
                "container-title": ["Science"],
                "author": [
                    {"given": "Alice", "family": "Wang"},
                    {"given": "Bob", "family": "Chen"},
                ],
                "published-print": {"date-parts": [[2026, 5, 25]]},
            }
        ]
    }
}


@patch("psil.ingest.crossref.requests.get")
def test_fetch_crossref_returns_papers(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_WORKS_RESPONSE
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    papers = fetch_crossref("0036-8075", date.today(), journal_name="Science")
    assert len(papers) == 1
    assert papers[0].doi == "10.1126/science.adg8758"
    assert papers[0].journal == "Science"
    assert papers[0].authors == ["Alice Wang", "Bob Chen"]


def test_parse_crossref_work_extracts_all_fields():
    work = {
        "DOI": "10.1038/test",
        "title": ["Test Title"],
        "abstract": "Test abstract.",
        "container-title": ["Nature Materials"],
        "author": [{"given": "First", "family": "Last"}],
        "published-print": {"date-parts": [[2026, 5, 1]]},
        "link": [{"URL": "https://example.com/toc.jpg", "content-type": "image/jpeg"}],
    }
    paper = parse_crossref_work(work, journal_name="Nature Materials")
    assert paper.title == "Test Title"
    assert paper.journal == "Nature Materials"
    assert paper.authors == ["First Last"]
    assert paper.pub_date == "2026-05-01"
    assert "example.com/toc.jpg" in paper.toc_image_url


def test_parse_crossref_work_handles_missing_fields():
    work = {"DOI": "10.0/minimal"}
    paper = parse_crossref_work(work, journal_name="Test")
    assert paper.doi == "10.0/minimal"
    assert paper.title == ""
    assert paper.authors == []
    assert paper.pub_date == ""
