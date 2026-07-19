from unittest.mock import patch, MagicMock

from psil.ingest.rss import fetch_nature_rss, parse_nature_entry


SAMPLE_ENTRIES = [
    {
        "title": "Excited-state routing in 2D perovskites",
        "link": "https://www.nature.com/articles/s41563-026-02616-4",
        "summary": "We demonstrate a new mechanism for excited-state routing.",
        "published": "2026-05-27",
        "authors": [{"name": "Alice Wang"}, {"name": "Bob Chen"}],
    }
]


@patch("psil.ingest.rss.feedparser.parse")
def test_fetch_nature_rss_returns_papers(mock_parse):
    mock_feed = MagicMock()
    mock_feed.entries = SAMPLE_ENTRIES
    mock_parse.return_value = mock_feed
    papers = fetch_nature_rss("https://www.nature.com/nmat.rss")
    assert len(papers) == 1
    assert papers[0].title == "Excited-state routing in 2D perovskites"
    assert papers[0].doi == "10.1038/s41563-026-02616-4"
    assert papers[0].journal == ""


def test_parse_nature_entry_extracts_doi_from_link():
    entry = {
        "title": "Test Paper",
        "link": "https://www.nature.com/articles/s41563-026-02616-4",
        "summary": "Abstract here.",
        "published": "2026-05-27",
    }
    paper = parse_nature_entry(entry)
    assert paper.doi == "10.1038/s41563-026-02616-4"
    assert paper.title == "Test Paper"
    assert paper.abstract == "Abstract here."
    assert paper.pub_date == "2026-05-27"


def test_parse_nature_entry_handles_missing_link():
    entry = {
        "title": "No Link Paper",
        "summary": "Abstract.",
    }
    paper = parse_nature_entry(entry)
    assert paper.doi == ""
