import re
from datetime import datetime

import feedparser

from psil.store.models import Paper

DOI_PATTERN = re.compile(r"10\.\d{4,}/[^\s]+")
NATURE_ARTICLE_PATTERN = re.compile(r"nature\.com/articles/([^/\s]+)")


def _extract_doi(link: str) -> str:
    if not link:
        return ""

    doi_match = DOI_PATTERN.search(link)
    if doi_match:
        return doi_match.group(0).rstrip(".")

    nature_match = NATURE_ARTICLE_PATTERN.search(link)
    if nature_match:
        return "10.1038/" + nature_match.group(1)

    return ""


def parse_nature_entry(entry: dict) -> Paper:
    title = entry.get("title", "").strip()
    abstract = entry.get("summary", "").strip()
    link = entry.get("link", "")

    doi = _extract_doi(link)

    authors = []
    for author in entry.get("authors", []):
        name = author.get("name", "")
        if name:
            authors.append(name)

    pub_date = ""
    published = entry.get("published", "")
    if published:
        try:
            dt = datetime(*entry["published_parsed"][:6])
            pub_date = dt.strftime("%Y-%m-%d")
        except (AttributeError, KeyError, ValueError):
            pub_date = published[:10]

    # Extract TOC image from RSS media:content or enclosures
    toc_img = ""
    media = entry.get("media_content", []) or []
    for m in media:
        url = m.get("url", "")
        if url:
            toc_img = url
            break
    if not toc_img:
        for enc in entry.get("enclosures", []) or []:
            url = enc.get("href", "") or enc.get("url", "")
            if url:
                toc_img = url
                break

    return Paper(
        doi=doi,
        title=title,
        abstract=abstract,
        authors=authors,
        pub_date=pub_date,
        toc_image_url=toc_img,
    )


def fetch_nature_rss(feed_url: str) -> list[Paper]:
    feed = feedparser.parse(feed_url)
    papers = []
    for entry in feed.entries:
        paper = parse_nature_entry(entry)
        papers.append(paper)
    return papers
