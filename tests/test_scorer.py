from psil.rank.scorer import prefilter_papers
from psil.rank.concepts import TIER3_WEIGHT
from psil.store.models import Paper


def make_paper(doi, title, abstract=""):
    return Paper(doi=doi, title=title, abstract=abstract, journal="Test Journal")


def test_prefilter_discards_no_match():
    papers = [
        make_paper("10.0/1", "A study of regional geology patterns"),
        make_paper("10.0/2", "Quantum computing advances"),
    ]
    passed, ignored = prefilter_papers(papers, threshold=0)
    assert len(ignored) == 2
    assert len(passed) == 0


def test_prefilter_passes_tier1():
    papers = [
        make_paper("10.0/1", "OECT sensing with dynamic biointerfaces"),
        make_paper("10.0/2", "Generic title with no signals"),
    ]
    passed, ignored = prefilter_papers(papers, threshold=0)
    assert len(passed) == 1
    assert passed[0][1] >= 5


def test_prefilter_passes_tier3():
    paper = make_paper("10.0/1", "A hydrogel for drug delivery applications")
    passed, ignored = prefilter_papers([paper], threshold=0)
    assert len(passed) == 1
    assert passed[0][1] >= TIER3_WEIGHT


def test_score_from_abstract():
    paper = make_paper(
        "10.0/1",
        "Generic title",
        "We demonstrate NIR-triggered release with photocleavage mechanisms",
    )
    passed, ignored = prefilter_papers([paper], threshold=0)
    assert len(passed) == 1
    assert passed[0][1] >= 10  # 2 * TIER1_WEIGHT
