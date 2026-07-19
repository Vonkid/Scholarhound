import html
import re
from urllib.parse import quote, urljoin, urlparse

import requests

from psil.store.models import Paper


USER_AGENT = "ScholarHound/0.1 (https://scholarhound.academy/)"

META_IMAGE_KEYS = {
    "og:image": 0,
    "og:image:url": 0,
    "og:image:secure_url": 0,
    "twitter:image": 1,
    "twitter:image:src": 1,
    "citation_image": 2,
    "thumbnail": 3,
    "image": 4,
}

BAD_IMAGE_HINTS = (
    "favicon",
    "logo",
    "header-",
    "newsletter",
    "captcha",
    "challenge",
    "verify",
    "doubleclick",
    "pubads.",
    "gampad",
    "static/images/favicons",
    "nature-briefing",
    "uploads/product",
)


TAG_RE = re.compile(r"<(meta|link|img)\b[^>]*>", re.IGNORECASE)
ATTR_RE = re.compile(
    r"""([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
    re.IGNORECASE,
)
IMAGE_URL_RE = re.compile(
    r"""https?://[^"'<> ]+\.(?:png|jpg|jpeg|webp)(?:\?[^"'<> ]*)?""",
    re.IGNORECASE,
)


def article_url_for_doi(doi: str) -> str:
    """Return a likely article URL for publisher-page metadata extraction."""
    doi = (doi or "").strip()
    if not doi:
        return ""
    doi_lower = doi.lower()
    if doi_lower.startswith("10.1038/"):
        return "https://www.nature.com/articles/" + doi.split("/", 1)[1]
    return "https://doi.org/" + quote(doi, safe="/")


def extract_toc_image_from_html(page_html: str, page_url: str = "", doi: str = "") -> str:
    """Extract the best article image URL from publisher HTML metadata."""
    if not page_html:
        return ""

    candidates: list[tuple[int, int, str]] = []
    doi_tokens = _doi_tokens(doi)

    for index, match in enumerate(TAG_RE.finditer(page_html)):
        tag_name = match.group(1).lower()
        attrs = _attrs(match.group(0))
        if tag_name == "meta":
            key = (
                attrs.get("property")
                or attrs.get("name")
                or attrs.get("itemprop")
                or ""
            ).lower()
            raw_url = attrs.get("content", "")
            priority = META_IMAGE_KEYS.get(key)
            if priority is not None:
                _add_candidate(candidates, priority, index, raw_url, page_url, doi_tokens)
        elif tag_name == "link":
            rel = attrs.get("rel", "").lower()
            if "image_src" in rel or rel == "preload" and attrs.get("as", "").lower() == "image":
                _add_candidate(
                    candidates,
                    5,
                    index,
                    attrs.get("href", ""),
                    page_url,
                    doi_tokens,
                    require_doi_match=bool(doi_tokens),
                )
        elif tag_name == "img":
            raw_url = attrs.get("data-src") or attrs.get("src") or ""
            priority = 20 if _matches_doi(raw_url, doi_tokens) else 60
            _add_candidate(
                candidates,
                priority,
                index,
                raw_url,
                page_url,
                doi_tokens,
                require_doi_match=bool(doi_tokens),
            )

    for index, match in enumerate(IMAGE_URL_RE.finditer(page_html), start=10_000):
        raw_url = match.group(0)
        priority = 15 if _matches_doi(raw_url, doi_tokens) else 70
        _add_candidate(
            candidates,
            priority,
            index,
            raw_url,
            page_url,
            doi_tokens,
            require_doi_match=bool(doi_tokens),
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2] if candidates else ""


def fetch_toc_image_url(article_url: str, doi: str = "", timeout: int = 12) -> str:
    """Fetch a publisher page and extract its best TOC/article image URL."""
    article_url = (article_url or "").strip()
    if not article_url:
        return ""
    try:
        resp = requests.get(
            article_url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception:
        return ""
    return extract_toc_image_from_html(resp.text, resp.url or article_url, doi=doi)


def enrich_paper_toc_image(
    paper: Paper,
    article_url: str = "",
    timeout: int = 12,
) -> str:
    """Populate ``paper.toc_image_url`` when a publisher page exposes one."""
    if paper.toc_image_url:
        return paper.toc_image_url
    target_url = article_url or article_url_for_doi(paper.doi)
    image_url = fetch_toc_image_url(target_url, doi=paper.doi, timeout=timeout)
    if image_url:
        paper.toc_image_url = image_url
    return image_url


def _attrs(tag: str) -> dict[str, str]:
    attrs = {}
    for match in ATTR_RE.finditer(tag):
        key = match.group(1).lower()
        value = match.group(2) or match.group(3) or match.group(4) or ""
        attrs[key] = html.unescape(value.strip())
    return attrs


def _normalize_url(raw_url: str, page_url: str) -> str:
    raw_url = html.unescape((raw_url or "").strip())
    if not raw_url or raw_url.startswith("data:"):
        return ""
    return urljoin(page_url or "", raw_url)


def _add_candidate(
    candidates: list[tuple[int, int, str]],
    priority: int,
    index: int,
    raw_url: str,
    page_url: str,
    doi_tokens: tuple[str, ...],
    require_doi_match: bool = False,
):
    if require_doi_match and not _matches_doi(raw_url, doi_tokens):
        return
    url = _normalize_url(raw_url, page_url)
    if require_doi_match and not _matches_doi(url, doi_tokens):
        return
    if _is_displayable_image_url(url, doi_tokens):
        candidates.append((priority, index, url))


def _is_displayable_image_url(url: str, doi_tokens: tuple[str, ...]) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    low = url.lower()
    if any(hint in low for hint in BAD_IMAGE_HINTS):
        return False
    path = parsed.path.lower()
    image_like = bool(re.search(r"\.(png|jpe?g|webp)$", path))
    image_like = image_like or "/mediaobjects/" in path
    image_like = image_like or parsed.netloc.lower() == "media.springernature.com"
    return image_like


def _doi_tokens(doi: str) -> tuple[str, ...]:
    doi = (doi or "").strip()
    if not doi:
        return ()
    return (
        doi.lower(),
        quote(doi, safe="").lower(),
        doi.replace("/", "%2F").lower(),
        doi.replace("/", "%2f").lower(),
    )


def _matches_doi(url: str, doi_tokens: tuple[str, ...]) -> bool:
    if not doi_tokens:
        return False
    low = (url or "").lower()
    return any(token and token in low for token in doi_tokens)
