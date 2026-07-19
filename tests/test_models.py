import json
from psil.store.models import Paper


def test_paper_creation_from_dict():
    data = {
        "doi": "10.1038/s41563-026-02616-4",
        "title": "Tailoring crystal phases of high-entropy alloy catalysts",
        "abstract": "This paper explores...",
        "journal": "Nature Materials",
        "authors": ["Alice Wang", "Bob Chen"],
        "affiliations": ["MIT", "Stanford"],
        "pub_date": "2026-05-27",
        "toc_image_url": "https://example.com/toc.jpg",
    }
    paper = Paper.from_dict(data)

    assert paper.doi == "10.1038/s41563-026-02616-4"
    assert paper.title == "Tailoring crystal phases of high-entropy alloy catalysts"
    assert paper.journal == "Nature Materials"
    assert len(paper.authors) == 2
    assert paper.authors[0] == "Alice Wang"


def test_paper_to_dict_roundtrip():
    paper = Paper(
        doi="10.1038/test",
        title="Test Paper",
        abstract="Abstract",
        journal="Nature Materials",
        authors=["Test Author"],
        affiliations=["Test Univ"],
        pub_date="2026-05-27",
    )
    d = paper.to_dict()
    paper2 = Paper.from_dict(d)
    assert paper2.doi == paper.doi
    assert paper2.title == paper.title


def test_paper_to_db_row_serializes_json_fields():
    paper = Paper(
        doi="10.1038/test",
        title="Test Paper",
        abstract="Abstract",
        journal="Nature Materials",
        authors=["Author One", "Author Two"],
        affiliations=["Univ A"],
        pub_date="2026-05-27",
    )
    row = paper.to_db_row()
    assert isinstance(row["authors"], str)
    assert json.loads(row["authors"]) == ["Author One", "Author Two"]
    assert isinstance(row["affiliations"], str)
    assert json.loads(row["affiliations"]) == ["Univ A"]
