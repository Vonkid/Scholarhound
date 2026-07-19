from datetime import date, timedelta

import requests

from psil.store.models import Paper

CROSSREF_BASE = "https://api.crossref.org/journals"
USER_AGENT = "PSIL/0.1 (https://scholarhound.academy/)"


def fetch_crossref(issn: str, query_date: date,
                   journal_name: str = "",
                   days_back: int = 1) -> list[Paper]:
    papers = []
    for i in range(days_back):
        d = query_date - timedelta(days=i)
        date_str = f"{d}T00:00:00"
        url = f"{CROSSREF_BASE}/{issn}/works"
        params = {
            "filter": f"from-pub-date:{date_str}",
            "rows": 50,
        }
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        items = data.get("message", {}).get("items", [])

        # Some publishers (e.g., Cell Press) only have month-precision pub dates,
        # so from-pub-date misses them. Fall back to from-created-date.
        if not items:
            params_fb = {
                "filter": f"from-created-date:{date_str}",
                "rows": 50,
            }
            resp_fb = requests.get(url, params=params_fb,
                                   headers={"User-Agent": USER_AGENT})
            resp_fb.raise_for_status()
            items = resp_fb.json().get("message", {}).get("items", [])

        for item in items:
            paper = parse_crossref_work(item, journal_name=journal_name)
            papers.append(paper)
    return papers


def parse_crossref_work(work: dict, journal_name: str = "") -> Paper:
    doi = work.get("DOI", "")
    title = ""
    titles = work.get("title", [])
    if titles:
        title = titles[0]

    abstract = work.get("abstract", "") or ""

    authors = []
    for author in work.get("author", []):
        given = author.get("given", "")
        family = author.get("family", "")
        full = f"{given} {family}".strip()
        if full:
            authors.append(full)

    affiliations = []
    for author in work.get("author", []):
        for affil in author.get("affiliation", []):
            name = affil.get("name", "")
            if name:
                affiliations.append(name)

    pub_date = ""
    date_parts = work.get("published-print", {}).get("date-parts", [[]])[0]
    if not date_parts:
        date_parts = work.get("published-online", {}).get("date-parts", [[]])[0]
    if not date_parts:
        date_parts = work.get("created", {}).get("date-parts", [[]])[0]
    if len(date_parts) >= 3:
        pub_date = f"{date_parts[0]:04d}-{date_parts[1]:02d}-{date_parts[2]:02d}"
    elif len(date_parts) == 2:
        pub_date = f"{date_parts[0]:04d}-{date_parts[1]:02d}"

    toc_image_url = ""
    for link in work.get("link", []):
        if "image" in link.get("content-type", ""):
            toc_image_url = link.get("URL", "")
            break

    return Paper(
        doi=doi,
        title=title,
        abstract=abstract,
        journal=journal_name,
        authors=authors,
        affiliations=affiliations,
        pub_date=pub_date,
        toc_image_url=toc_image_url,
    )
