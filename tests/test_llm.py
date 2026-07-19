from unittest.mock import patch, MagicMock
from psil.rank.llm import LLMClient, build_ranking_prompt, parse_llm_response
from psil.rank.identity import ResearchIdentity
from psil.store.models import Paper


def test_build_ranking_prompt_includes_paper_details():
    identity = ResearchIdentity()
    paper = Paper(
        doi="10.1038/test",
        title="Excited-state routing for oxygen-independent photochemistry",
        abstract="We demonstrate a novel mechanism for photon utilization.",
        journal="Nature Chemistry",
    )
    prompt = build_ranking_prompt(paper, identity, matched_signals="OECT, photocleavage")
    assert paper.title in prompt
    assert paper.abstract in prompt
    assert paper.journal in prompt
    assert "OECT" in prompt
    assert "RELEVANCE" in prompt
    assert "NOVELTY" in prompt
    assert "BRIDGE" in prompt
    assert "TRAJECTORY_INFLUENCE" in prompt
    assert "PAPER_TYPE" in prompt
    assert "JUDGMENT_MODE" in prompt
    assert "Current Core" in prompt
    assert "Emerging Directions" in prompt


def test_parse_llm_response_with_scores():
    response = """RELEVANCE: 7/10
NOVELTY: 8/10
BRIDGE: 6/10
TRAJECTORY_INFLUENCE: 7/10
CONCEPT_SUPPORT: 8/10
SIGNAL_TIER: IMPORTANT
WHY_MATTERS: - Decouples photochemistry from oxygen.
POTENTIAL_CONNECTION: - Directly relevant.
WEAKNESS: In vitro only.
ACTION: Review this week
CONCEPT_NAME: oxygen-independent photochemistry
CONCEPT_WHY_MATTERS: Enables new class of photochemical reactions.
CONCEPT_CURRENT_CONNECTION: NIR photocleavage research.
CONCEPT_POTENTIAL_CONNECTION: Could expand to ROS-independent systems.
CONCEPT_MISSING_LINK: In vivo validation needed.
CONCEPT_OPPORTUNITY: Hypoxia-tolerant photoactivation platform.
CONCEPT_ACTION: Add to dictionary"""
    result = parse_llm_response(response)
    assert result["relevance"] == 7
    assert result["novelty"] == 8
    assert result["bridge"] == 6
    assert result["trajectory_influence"] == 7
    # 0.25*7 + 0.20*8 + 0.20*6 + 0.20*7 + 0.15*8 = 7.2
    assert result["final_score"] == 7.2
    assert result["signal_tier"] == "IMPORTANT"
    assert result["concept_name"] == "oxygen-independent photochemistry"
    assert "n vivo" in result["concept_missing_link"].lower()


def test_parse_llm_response_low_priority():
    response = """RELEVANCE: 3/10
NOVELTY: 2/10
BRIDGE: 1/10
TRAJECTORY_INFLUENCE: 1/10
CONCEPT_SUPPORT: 1/10
SIGNAL_TIER: LOW_PRIORITY
WHY_MATTERS: - Incremental optimization.
POTENTIAL_CONNECTION: - Limited connection.
WEAKNESS: None
ACTION: Skip
CONCEPT_NAME: None
CONCEPT_WHY_MATTERS: None
CONCEPT_CURRENT_CONNECTION: None
CONCEPT_POTENTIAL_CONNECTION: None
CONCEPT_MISSING_LINK: None
CONCEPT_OPPORTUNITY: None
CONCEPT_ACTION: Ignore for now"""
    result = parse_llm_response(response)
    # 0.25*3 + 0.20*2 + 0.20*1 + 0.20*1 + 0.15*1 = 1.7
    assert round(result["final_score"], 1) == 1.7
    assert result["signal_tier"] == "LOW_PRIORITY"
    assert result["concept_name"] == "None"


def test_parse_llm_response_routes_score_by_paper_type():
    response = """PAPER_TYPE: Validation or Benchmark Paper
JUDGMENT_MODE: validation_readiness
PROBLEM_CLASS: Sensing
NOVELTY_TYPE: New Validation
EVIDENCE_TYPE: Benchmark Evidence
STRATEGIC_VALUE: Disease-Relevant Functional Readout
RELEVANCE: 8/10
NOVELTY: 2/10
BRIDGE: 3/10
TRAJECTORY_INFLUENCE: 7/10
CONCEPT_SUPPORT: 9/10
SIGNAL_TIER: IMPORTANT
WHY_MATTERS: - Establishes a disease-relevant benchmark.
POTENTIAL_CONNECTION: - Relevant validation template.
WEAKNESS: Limited cohort.
ACTION: Review this week
CONCEPT_NAME: validation benchmark
CONCEPT_ACTION: Watch only"""
    result = parse_llm_response(response)
    assert result["paper_type"] == "validation_or_benchmark_paper"
    assert result["judgment_mode"] == "validation_readiness"
    assert result["final_score"] == 6.8


@patch("psil.rank.llm.OpenAI")
def test_llm_client_calls_api(mock_openai_cls):
    identity = ResearchIdentity()
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = (
        "RELEVANCE: 9/10\nNOVELTY: 8/10\nBRIDGE: 7/10\n"
        "TRAJECTORY_INFLUENCE: 9/10\nCONCEPT_SUPPORT: 9/10\n"
        "SIGNAL_TIER: HIGH_PRIORITY\nWHY_MATTERS: - Test.\n"
        "CONCEPT_NAME: test concept\n"
        "CONCEPT_WHY_MATTERS: Important.\n"
        "CONCEPT_CURRENT_CONNECTION: Bioelectronics.\n"
        "CONCEPT_POTENTIAL_CONNECTION: Nanophotonics.\n"
        "CONCEPT_MISSING_LINK: Integration.\n"
        "CONCEPT_OPPORTUNITY: New platform.\n"
        "CONCEPT_ACTION: Add to dictionary\n"
    )
    mock_client.chat.completions.create.return_value = mock_resp

    client = LLMClient(api_key="sk-test", base_url="https://api.deepseek.com/v1")
    result = client.rank(Paper(doi="10.0/1", title="Test"), identity)
    # 0.25*9 + 0.20*8 + 0.20*7 + 0.20*9 + 0.15*9 = 8.4
    assert result["final_score"] == 8.4
    assert result["signal_tier"] == "HIGH_PRIORITY"
