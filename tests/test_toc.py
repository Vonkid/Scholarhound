from unittest.mock import MagicMock, patch

from psil.ingest.toc import (
    article_url_for_doi,
    enrich_paper_toc_image,
    extract_toc_image_from_html,
    fetch_toc_image_url,
)
from psil.store.models import Paper


def test_article_url_for_nature_doi_uses_article_slug():
    assert (
        article_url_for_doi("10.1038/s41565-025-01995-0")
        == "https://www.nature.com/articles/s41565-025-01995-0"
    )


def test_extract_toc_image_prefers_publisher_meta_image():
    html = """
    <html><head>
      <meta property="og:image" content="https://media.springernature.com/m685/springer-static/image/art%3A10.1038%2Fs41565-025-01995-0/MediaObjects/41565_2025_1995_Fig1_HTML.png">
      <meta name="twitter:image" content="https://example.com/fallback.png">
    </head></html>
    """

    image = extract_toc_image_from_html(
        html,
        "https://www.nature.com/articles/s41565-025-01995-0",
        doi="10.1038/s41565-025-01995-0",
    )

    assert "media.springernature.com" in image
    assert "Fig1_HTML.png" in image


def test_extract_toc_image_ignores_header_and_uses_article_figure():
    html = """
    <img src="https://media.springernature.com/full/nature-cms/uploads/product/nnano/header-logo.svg">
    <img src="https://media.springernature.com/lw1200/springer-static/image/art%3A10.1038%2Fs41565-025-01995-0/MediaObjects/41565_2025_1995_Fig2_HTML.png">
    """

    image = extract_toc_image_from_html(
        html,
        "https://www.nature.com/articles/s41565-025-01995-0",
        doi="10.1038/s41565-025-01995-0",
    )

    assert image.endswith("Fig2_HTML.png")


def test_extract_toc_image_normalizes_relative_urls():
    html = '<meta name="citation_image" content="/images/toc.jpg">'

    image = extract_toc_image_from_html(html, "https://example.com/article")

    assert image == "https://example.com/images/toc.jpg"


def test_extract_toc_image_does_not_treat_cloudflare_challenge_as_toc():
    html = """
    <html><head><title>Just a moment...</title></head>
    <body><span id="challenge-error-text">Enable JavaScript and cookies to continue</span></body>
    </html>
    """

    assert extract_toc_image_from_html(html, "https://example.com") == ""


def test_extract_toc_image_rejects_ad_url_even_when_it_contains_doi():
    html = """
    <img src="https://pubads.g.doubleclick.net/gampad/ad?doi=10.1038/s41467-026-73762-1">
    <meta property="og:image" content="https://media.springernature.com/m685/springer-static/image/art%3A10.1038%2Fs41467-026-73762-1/MediaObjects/41467_2026_73762_Fig1_HTML.png">
    """

    image = extract_toc_image_from_html(html, doi="10.1038/s41467-026-73762-1")

    assert image.startswith("https://media.springernature.com/")


def test_extract_toc_image_rejects_recommended_article_image_with_other_doi():
    html = """
    <img src="https://media.springernature.com/w215h120/springer-static/image/art%3A10.1038%2Fs41467-024-54528-z/MediaObjects/41467_2024_54528_Fig1_HTML.png">
    """

    image = extract_toc_image_from_html(html, doi="10.1038/s41467-026-73762-1")

    assert image == ""


@patch("psil.ingest.toc.requests.get")
def test_fetch_toc_image_url_reads_response_url(mock_get):
    response = MagicMock()
    response.url = "https://publisher.example/article"
    response.text = '<meta property="og:image" content="/toc.png">'
    response.raise_for_status.return_value = None
    mock_get.return_value = response

    image = fetch_toc_image_url("https://doi.org/10.0/test", doi="10.0/test")

    assert image == "https://publisher.example/toc.png"


@patch("psil.ingest.toc.fetch_toc_image_url")
def test_enrich_paper_toc_image_mutates_missing_field(mock_fetch):
    mock_fetch.return_value = "https://example.com/toc.png"
    paper = Paper(doi="10.1038/test", title="Test")

    image = enrich_paper_toc_image(paper)

    assert image == "https://example.com/toc.png"
    assert paper.toc_image_url == "https://example.com/toc.png"
