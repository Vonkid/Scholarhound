"""Benchmark ingest: 15-year Nature Reviews + Nature sub-journal reviews via Crossref."""

import sqlite3, time, requests
from datetime import date
from pathlib import Path

USER_AGENT = "ScholarHound-Benchmark/0.1 (https://scholarhound.academy/)"
CROSSREF_JOURNALS = "https://api.crossref.org/journals/{issn}/works"
DB_PATH = Path.home() / ".psil" / "benchmark.db"
RATE_DELAY = 0.15  # seconds between requests (~6 req/s, well under 50 limit)

REVIEW_JOURNALS = [
    ("Nature Reviews Materials", "2058-8437"),
    ("Nature Reviews Chemistry", "2397-3358"),
    ("Nature Reviews Physics", "2522-5820"),
    ("Nature Reviews Molecular Cell Biology", "1471-0072"),
    ("Nature Reviews Genetics", "1471-0056"),
    ("Nature Reviews Cancer", "1474-175X"),
    ("Nature Reviews Immunology", "1474-1733"),
    ("Nature Reviews Neuroscience", "1471-003X"),
    ("Nature Reviews Microbiology", "1740-1526"),
    ("Nature Reviews Drug Discovery", "1474-1776"),
    ("Nature Reviews Disease Primers", "2056-676X"),
    ("Nature Reviews Endocrinology", "1759-5029"),
    ("Nature Reviews Gastroenterology & Hepatology", "1759-5045"),
    ("Nature Reviews Cardiology", "1759-5002"),
    ("Nature Reviews Neurology", "1759-4758"),
    ("Nature Reviews Rheumatology", "1759-4790"),
    ("Nature Reviews Nephrology", "1759-5061"),
    ("Nature Reviews Urology", "1759-4812"),
    ("Nature Reviews Clinical Oncology", "1759-4774"),
    ("Nature Reviews Electrical Engineering", "2948-1201"),
    ("Nature Reviews Bioengineering", "2731-6092"),
]

NATURE_SUB_JOURNALS = [
    ("Nature Materials", "1476-1122"),
    ("Nature Chemistry", "1755-4330"),
    ("Nature Photonics", "1749-4885"),
    ("Nature Nanotechnology", "1748-3387"),
    ("Nature Biotechnology", "1087-0156"),
    ("Nature Biomedical Engineering", "2157-846X"),
    ("Nature Electronics", "2520-1131"),
]


def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    return db


def create_tables():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (
            doi TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            journal TEXT,
            pub_year INTEGER,
            pub_date TEXT,
            authors TEXT,
            source TEXT  -- 'nature_reviews' or 'nature_sub'
        );
        CREATE TABLE IF NOT EXISTS ingest_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            journal TEXT,
            year_start INTEGER,
            papers INTEGER,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    db.commit()
    db.close()


def fetch_journal(issn, journal_name, year_start=2011, source="nature_reviews"):
    """Fetch all papers for a journal from a given year. Returns count of new papers."""
    db = get_db()
    added = 0

    for year in range(year_start, date.today().year + 1):
        # Check if already ingested this year
        existing = db.execute(
            "SELECT COUNT(*) as n FROM ingest_log WHERE journal=? AND year_start=?",
            (journal_name, year)
        ).fetchone()
        if existing["n"] > 0:
            continue

        papers = _fetch_year(issn, year, journal_name)
        for p in papers:
            try:
                db.execute("""
                    INSERT OR IGNORE INTO reviews (doi, title, abstract, journal, pub_year, pub_date, authors, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (p["doi"], p["title"], p["abstract"], journal_name,
                      p["pub_year"], p["pub_date"], p["authors"], source))
                if db.total_changes > 0:
                    added += 1
            except Exception:
                pass

        db.execute(
            "INSERT INTO ingest_log (journal, year_start, papers) VALUES (?, ?, ?)",
            (journal_name, year, len(papers))
        )
        db.commit()
        print(f"  {journal_name}: {year} → {len(papers)} papers ({added} new)")

    db.close()
    return added


def _fetch_year(issn, year, journal_name):
    """Fetch one year of papers with offset pagination."""
    papers = []
    rows_per_page = 200
    offset = 0
    max_pages = 50  # safety limit: 50 × 200 = 10K papers per year

    for page in range(max_pages):
        url = CROSSREF_JOURNALS.format(issn=issn)
        params = {
            "filter": f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31",
            "rows": rows_per_page,
            "offset": offset,
        }
        try:
            resp = requests.get(url, params=params,
                              headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                break
            for item in items:
                paper = _parse_item(item, journal_name, year)
                if paper:
                    papers.append(paper)
            total = data.get("message", {}).get("total-results", 0)
            offset += rows_per_page
            if offset >= total:
                break
            time.sleep(RATE_DELAY)
        except Exception as e:
            print(f"    Error {year} p{page}: {e}")
            break

    return papers


def _parse_item(item, journal_name, year):
    """Parse a Crossref work item into a review dict."""
    doi = item.get("DOI", "")
    if not doi:
        return None

    title = ""
    titles = item.get("title", [])
    if titles:
        title = titles[0]

    abstract = item.get("abstract", "") or ""

    authors = []
    for a in item.get("author", []):
        name = f"{a.get('given','')} {a.get('family','')}".strip()
        if name:
            authors.append(name)

    pub_date = ""
    dp = item.get("published-print", {}).get("date-parts", [[None, None]])[0]
    if not dp or not dp[0]:
        dp = item.get("published-online", {}).get("date-parts", [[None, None]])[0]
    if not dp or not dp[0]:
        dp = item.get("created", {}).get("date-parts", [[None, None]])[0]
    if dp and dp[0]:
        pub_date = f"{dp[0]:04d}-{dp[1] or 1:02d}"

    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "journal": journal_name,
        "pub_year": year,
        "pub_date": pub_date,
        "authors": "; ".join(authors[:20]),
        "source": "nature_reviews",
    }


def get_progress():
    """Return current ingestion progress."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) as n FROM reviews").fetchone()["n"]
    years = db.execute(
        "SELECT COUNT(DISTINCT journal||year_start) as n FROM ingest_log"
    ).fetchone()["n"]
    db.close()
    return {"total_papers": total, "journal_years_done": years}
