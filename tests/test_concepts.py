from psil.rank.concepts import CONCEPTS, get_concept_score, TIER1_WEIGHT, TIER2_WEIGHT, TIER3_WEIGHT


def test_tier1_concept_matches():
    score = get_concept_score("We demonstrate OECT sensing for organoid monitoring")
    assert score >= TIER1_WEIGHT


def test_tier2_concept_matches():
    score = get_concept_score("A flexible bioelectronics platform for wearable biosensing")
    assert score >= TIER2_WEIGHT


def test_tier3_concept_matches():
    score = get_concept_score("A hydrogel-based drug delivery system")
    assert score >= TIER3_WEIGHT


def test_tier1_plus_tier2():
    score = get_concept_score("NIR-triggered release with OECT and extracellular vesicle analysis")
    assert score >= TIER1_WEIGHT + TIER1_WEIGHT + TIER2_WEIGHT


def test_no_match_returns_zero():
    score = get_concept_score("A study of unrelated topics in geology")
    assert score == 0


def test_new_concept_dictionary_size():
    assert len(CONCEPTS) > 50
