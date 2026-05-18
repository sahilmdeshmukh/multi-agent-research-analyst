"""Unit tests for the synthesizer agent with mocked ChatGroq — no real API calls."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_analyst.schemas import AgentState, Memo, MemoSection, ResearchNote


def _make_notes(count: int = 3) -> list[ResearchNote]:
    """Create *count* ResearchNote objects for use in tests."""
    return [
        ResearchNote(
            claim=f"Claim {i}",
            source_url=f"https://source{i}.com/article-{i}",
            source_title=f"Article {i}",
            confidence=0.9,
        )
        for i in range(count)
    ]


@pytest.fixture()
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")


def test_synthesizer_produces_memo(mock_env: None) -> None:
    """synthesizer_node sets final_memo on state when the LLM returns a valid Memo."""
    expected_memo = Memo(
        title="Test Research Memo",
        summary="This is a two sentence executive summary. It covers the key findings.",
        sections=[
            MemoSection(heading="Background", body="Background text with citation [1]."),
            MemoSection(heading="Findings", body="Key finding supported by [2]."),
        ],
        sources=[
            "https://source0.com/article-0",
            "https://source1.com/article-1",
        ],
    )

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = expected_memo
    mock_llm.with_structured_output.return_value = mock_structured

    state = AgentState(query="What is the state of AI?", notes=_make_notes(3))

    with patch("research_analyst.llm.ChatGroq", return_value=mock_llm):
        from research_analyst.agents.synthesizer import synthesizer_node

        result = synthesizer_node(state)

    assert result.final_memo is not None
    assert result.final_memo.title == "Test Research Memo"
    assert len(result.final_memo.sections) == 2
    assert result.final_memo.sections[0].heading == "Background"
    # All other state fields must be preserved unchanged
    assert result.query == state.query
    assert result.notes == state.notes
    assert result.critiques == state.critiques
    assert result.revision_round == state.revision_round


def test_synthesizer_fallback_on_error(mock_env: None) -> None:
    """When the LLM raises ValueError, synthesizer_node returns a fallback Memo."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = ValueError("model response unparseable")
    mock_llm.with_structured_output.return_value = mock_structured

    state = AgentState(query="fallback query", notes=_make_notes(2))

    with patch("research_analyst.llm.ChatGroq", return_value=mock_llm):
        from research_analyst.agents.synthesizer import synthesizer_node

        result = synthesizer_node(state)

    assert result.final_memo is not None
    assert result.final_memo.title == "Research incomplete"
    assert "model response unparseable" in result.final_memo.summary
    assert result.final_memo.sections == []
    assert result.final_memo.sources == []
    # Other state fields must still be preserved
    assert result.query == state.query
    assert result.notes == state.notes
    assert result.revision_round == state.revision_round
