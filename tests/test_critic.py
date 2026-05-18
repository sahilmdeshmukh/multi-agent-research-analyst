"""Unit tests for the critic agent with mocked ChatGroq — no real API calls."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_analyst.schemas import AgentState, Critique, ResearchNote


def _make_notes(count: int, domains: list[str] | None = None) -> list[ResearchNote]:
    """Create *count* ResearchNote objects spread across the given domain list."""
    if domains is None:
        domains = [f"source{i}.com" for i in range(count)]
    notes = []
    for i in range(count):
        domain = domains[i % len(domains)]
        notes.append(
            ResearchNote(
                claim=f"Claim {i}",
                source_url=f"https://{domain}/article-{i}",
                source_title=f"Article {i}",
                confidence=0.8,
            )
        )
    return notes


@pytest.fixture()
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")


def test_critic_approves_when_criteria_met(mock_env: None) -> None:
    """Critic appends an 'approve' critique when the LLM returns one."""
    approve_critique = Critique(
        verdict="approve",
        gaps=[],
        weak_sources=[],
        contradictions=[],
        follow_up_questions=[],
        reasoning="All criteria met.",
    )

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = approve_critique
    mock_llm.with_structured_output.return_value = mock_structured

    # 5 notes across 3 distinct domains satisfies the approval criteria
    notes = _make_notes(5, domains=["alpha.com", "beta.com", "gamma.com"])
    state = AgentState(query="test query", notes=notes)

    with patch("research_analyst.llm.ChatGroq", return_value=mock_llm):
        from research_analyst.agents.critic import critic_node

        result = critic_node(state)

    assert len(result.critiques) == 1
    assert result.critiques[0].verdict == "approve"
    assert result.critiques[0].follow_up_questions == []


def test_critic_requests_revision_on_fallback(mock_env: None) -> None:
    """When the LLM raises ValueError the fallback critique requests a revision."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = ValueError("model response unparseable")
    mock_llm.with_structured_output.return_value = mock_structured

    state = AgentState(query="fallback test query", notes=_make_notes(2))

    with patch("research_analyst.llm.ChatGroq", return_value=mock_llm):
        from research_analyst.agents.critic import critic_node

        result = critic_node(state)

    assert len(result.critiques) == 1
    critique = result.critiques[0]
    assert critique.verdict == "request_revision"
    assert len(critique.follow_up_questions) > 0
