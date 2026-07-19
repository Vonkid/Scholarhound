from psil.store.models import Paper
from psil.rank.concepts import get_concept_score, get_matched_concepts


def prefilter_papers(
    papers: list[Paper], threshold: int = 0
) -> tuple[list[tuple[Paper, int]], list[Paper]]:
    passed = []
    ignored = []
    for paper in papers:
        combined_text = f"{paper.title} {paper.abstract}"
        score = get_concept_score(combined_text)
        if score > threshold:
            passed.append((paper, score))
        else:
            ignored.append(paper)
    return passed, ignored


def classify_signal(tier: str) -> str:
    tier = tier.strip().upper()
    if tier not in ("HIGH", "MAYBE", "IGNORE"):
        tier = "IGNORE"
    return tier
