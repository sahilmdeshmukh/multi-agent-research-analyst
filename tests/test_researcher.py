"""Unit tests for the researcher agent with mocked Tavily + Groq."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_analyst.schemas import AgentState, ResearchNote, ResearchNoteList, ResearchQuery


MOCK_SEARCH_RESULTS = [
    {
        "url": "https://example.com/asml-monopoly",
        "title": "ASML's EUV Monopoly Explained",
        "content": (
            "ASML holds a near-total monopoly on extreme ultraviolet (EUV) lithography machines, "
            "which are essential for manufacturing chips below 7nm. This gives ASML significant "
            "pricing power, with each machine costing roughly $150–200 million."
        ),
    },
    {
        "url": "https://example.com/chip-prices",
        "title": "How ASML Affects Semiconductor Prices",
        "content": (
            "Because chipmakers like TSMC, Samsung, and Intel must purchase ASML machines to "
            "produce advanced chips, the cost of these machines flows through to chip prices "
            "and ultimately consumer electronics."
        ),
    },
]

MOCK_SUB_QUESTIONS = ["What is ASML's market position?", "How does EUV technology affect chip costs?"]


@pytest.fixture()
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")


def _make_mock_llm(sub_questions: list[str], notes: list[ResearchNote]) -> MagicMock:
    mock_llm = MagicMock()

    def structured_side_effect(schema):
        inner = MagicMock()
        if schema is ResearchQuery:
            inner.invoke.return_value = ResearchQuery(
                original_query="test query", sub_questions=sub_questions
            )
        elif schema is ResearchNoteList:
            inner.invoke.return_value = ResearchNoteList(notes=notes)
        else:
            inner.invoke.return_value = notes
        return inner

    mock_llm.with_structured_output.side_effect = structured_side_effect
    return mock_llm


def test_researcher_node_returns_notes(mock_env: None) -> None:
    expected_notes = [
        ResearchNote(
            claim="ASML holds near-total monopoly on EUV lithography machines",
            source_url="https://example.com/asml-monopoly",
            source_title="ASML's EUV Monopoly Explained",
            confidence=0.9,
        )
    ]

    mock_llm = _make_mock_llm(MOCK_SUB_QUESTIONS, expected_notes)

    with (
        patch("research_analyst.agents.researcher.ChatGroq", return_value=mock_llm),
        patch(
            "research_analyst.agents.researcher.tavily_search",
            return_value=MOCK_SEARCH_RESULTS,
        ),
    ):
        from research_analyst.agents.researcher import researcher_node

        initial_state = AgentState(query="How does ASML's monopoly affect chip prices?")
        result = researcher_node(initial_state)

    assert len(result.notes) > 0
    assert result.sub_questions == MOCK_SUB_QUESTIONS
    assert all(isinstance(n, ResearchNote) for n in result.notes)
    assert all(n.source_url for n in result.notes)


def test_researcher_node_preserves_existing_notes(mock_env: None) -> None:
    existing_note = ResearchNote(
        claim="Existing claim from previous round",
        source_url="https://example.com/existing",
        source_title="Existing Source",
        confidence=0.7,
    )
    new_note = ResearchNote(
        claim="New claim from this round",
        source_url="https://example.com/new",
        source_title="New Source",
        confidence=0.8,
    )

    mock_llm = _make_mock_llm(MOCK_SUB_QUESTIONS, [new_note])

    with (
        patch("research_analyst.agents.researcher.ChatGroq", return_value=mock_llm),
        patch(
            "research_analyst.agents.researcher.tavily_search",
            return_value=MOCK_SEARCH_RESULTS,
        ),
    ):
        from research_analyst.agents.researcher import researcher_node

        state_with_existing = AgentState(
            query="test query",
            notes=[existing_note],
            revision_round=1,
        )
        result = researcher_node(state_with_existing)

    assert existing_note in result.notes


def test_researcher_uses_followup_questions_on_revision(mock_env: None) -> None:
    from research_analyst.schemas import Critique

    followup_questions = ["What is ASML's revenue from EUV?", "Who are ASML's competitors?"]
    critique = Critique(
        verdict="request_revision",
        gaps=["revenue data missing"],
        weak_sources=[],
        contradictions=[],
        follow_up_questions=followup_questions,
        reasoning="Need more financial data",
    )

    mock_llm = _make_mock_llm(followup_questions, [])

    with (
        patch("research_analyst.agents.researcher.ChatGroq", return_value=mock_llm),
        patch(
            "research_analyst.agents.researcher.tavily_search",
            return_value=[],
        ),
    ):
        from research_analyst.agents.researcher import researcher_node

        state = AgentState(
            query="ASML monopoly",
            revision_round=1,
            critiques=[critique],
        )
        result = researcher_node(state)

    assert result.sub_questions == followup_questions
