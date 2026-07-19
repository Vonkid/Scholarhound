from unittest.mock import patch, MagicMock
from datetime import date
from psil.ingest.orchestrator import ingest_all_journals
from psil.store.models import Paper


@patch("psil.ingest.orchestrator.fetch_nature_rss")
@patch("psil.ingest.orchestrator.fetch_crossref")
def test_ingest_all_journals_combines_sources(mock_crossref, mock_rss):
    mock_rss.return_value = [
        Paper(doi="10.1038/rss1", title="RSS Paper 1", journal="Nature Materials")
    ]
    mock_crossref.return_value = [
        Paper(doi="10.1126/cr1", title="CrossRef Paper 1", journal="Science")
    ]

    journals = [
        {"name": "Nature Materials", "rss": "https://www.nature.com/nmat.rss", "issn": "1476-1122"},
        {"name": "Science", "issn": "0036-8075"},
    ]

    papers = ingest_all_journals(journals, date.today())

    assert len(papers) == 2
    dois = {p.doi for p in papers}
    assert "10.1038/rss1" in dois
    assert "10.1126/cr1" in dois


@patch("psil.ingest.orchestrator.fetch_nature_rss")
@patch("psil.ingest.orchestrator.fetch_crossref")
def test_ingest_deduplicates_by_doi(mock_crossref, mock_rss):
    mock_rss.return_value = [
        Paper(doi="10.1038/same", title="Same Paper", journal="Nature Materials")
    ]
    mock_crossref.return_value = [
        Paper(doi="10.1038/same", title="Same Paper", journal="Nature Materials")
    ]

    journals = [
        {"name": "Nature Materials", "rss": "https://www.nature.com/nmat.rss", "issn": "1476-1122"},
    ]

    papers = ingest_all_journals(journals, date.today())
    assert len(papers) == 1


@patch("psil.ingest.orchestrator.fetch_nature_rss")
@patch("psil.ingest.orchestrator.fetch_crossref")
def test_ingest_skips_journals_without_rss(mock_crossref, mock_rss):
    mock_crossref.return_value = []

    journals = [
        {"name": "Science Advances", "issn": "2375-2548"},
    ]

    papers = ingest_all_journals(journals, date.today())
    mock_rss.assert_not_called()
    mock_crossref.assert_called_once()
