from datetime import date

from psil.store.models import Paper
from psil.ingest.rss import fetch_nature_rss
from psil.ingest.crossref import fetch_crossref


def ingest_all_journals(journals: list[dict],
                        query_date: date,
                        days_back: int = 1) -> list[Paper]:
    seen_dois: set[str] = set()
    all_papers: list[Paper] = []

    for journal in journals:
        name = journal["name"]
        issn = journal.get("issn", "")
        rss_url = journal.get("rss", "")

        if rss_url:
            try:
                papers = fetch_nature_rss(rss_url)
                for p in papers:
                    p.journal = name
                    if p.doi and p.doi not in seen_dois:
                        seen_dois.add(p.doi)
                        all_papers.append(p)
            except Exception:
                pass

        if issn:
            try:
                papers = fetch_crossref(issn, query_date,
                                        journal_name=name,
                                        days_back=days_back)
                for p in papers:
                    if p.doi and p.doi not in seen_dois:
                        seen_dois.add(p.doi)
                        all_papers.append(p)
            except Exception:
                pass

    return all_papers
