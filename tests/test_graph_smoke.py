"""Smoke tests for the full LangGraph pipeline (researcher → critic → synthesizer).

All LLM and search calls are mocked — no real API calls are made.
"""
from __future__ import annotations

from itertools import count
from unittest.mock import MagicMock, patch

import pytest

from research_analyst.schemas import (
    AgentState,
    Critique,
    Memo,
    MemoSection,
    ResearchNote,
    ResearchNoteList,
    ResearchQuery,
)


# ---------------------------------------------------------------------------
# Shared canned data
# ---------------------------------------------------------------------------

_SUB_QUESTIONS = ["What is quantum computing?", "How does it work?"]

_SIX_NOTES = [
    ResearchNote(
        claim=f"Claim {i}",
        source_url=f"https://domain{i % 3}.com/article-{i}",
        source_title=f"Article {i}",
        confidence=0.85,
    )
    for i in range(6)
]

_APPROVE_CRITIQUE = Critique(
    verdict="approve",
    gaps=[],
    weak_sources=[],
    contradictions=[],
    follow_up_questions=[],
    reasoning="Research is thorough and well-sourced.",
)

_REVISION_CRITIQUE = Critique(
    verdict="request_revision",
    gaps=["Missing economic impact data"],
    weak_sources=[],
    contradictions=[],
    follow_up_questions=["What is the economic impact of quantum computing?"],
    reasoning="More data needed on economic aspects.",
)

_MEMO = Memo(
    title="Quantum Computing Research Memo",
    summary="Quantum computing leverages quantum mechanics for computation. It promises exponential speed-ups for certain problems.",
    sections=[
        MemoSection(heading="Background", body="Quantum computers use qubits [1]."),
        MemoSection(heading="Findings", body="Key findings show promising results [2]."),
    ],
    sources=["https://domain0.com/article-0", "https://domain1.com/article-1"],
)

_MOCK_SEARCH_RESULTS = [
    {
        "url": "https://domain0.com/article-0",
        "title": "Quantum Computing Overview",
        "content": "Quantum computing uses qubits to perform computations.",
    }
]

# Follow-up notes returned by researcher on revision round
_REVISION_NOTES = [
    ResearchNote(
        claim=f"Revision claim {i}",
        source_url=f"https://revision{i}.com/article-{i}",
        source_title=f"Revision Article {i}",
        confidence=0.8,
    )
    for i in range(3)
]


# ---------------------------------------------------------------------------
# Mock factory helpers
# ---------------------------------------------------------------------------

def _make_llm_mock_approve_first_round() -> MagicMock:
    """LLM mock for test_graph_full_flow_approve_on_first_round.

    - ResearchQuery → _SUB_QUESTIONS
    - ResearchNoteList → _SIX_NOTES
    - Critique → _APPROVE_CRITIQUE
    - Memo → _MEMO
    """
    mock_llm = MagicMock()

    def structured_side_effect(schema):
        inner = MagicMock()
        if schema is ResearchQuery:
            inner.invoke.return_value = ResearchQuery(
                original_query="test query", sub_questions=_SUB_QUESTIONS
            )
        elif schema is ResearchNoteList:
            # Each sub-question search yields 3 notes; 2 sub-questions → 6 notes total
            inner.invoke.return_value = ResearchNoteList(notes=_SIX_NOTES[:3])
        elif schema is Critique:
            inner.invoke.return_value = _APPROVE_CRITIQUE
        elif schema is Memo:
            inner.invoke.return_value = _MEMO
        else:
            inner.invoke.return_value = MagicMock()
        return inner

    mock_llm.with_structured_output.side_effect = structured_side_effect
    return mock_llm


def _make_llm_mock_revision_then_approve() -> MagicMock:
    """LLM mock for test_graph_revision_loop_then_approve.

    Critic returns request_revision on the first call, approve on the second.
    """
    mock_llm = MagicMock()
    critique_call_counter = count(0)

    def structured_side_effect(schema):
        inner = MagicMock()
        if schema is ResearchQuery:
            inner.invoke.return_value = ResearchQuery(
                original_query="test query", sub_questions=_SUB_QUESTIONS
            )
        elif schema is ResearchNoteList:
            inner.invoke.return_value = ResearchNoteList(notes=_SIX_NOTES[:3])
        elif schema is Critique:
            call_n = next(critique_call_counter)
            if call_n == 0:
                inner.invoke.return_value = _REVISION_CRITIQUE
            else:
                inner.invoke.return_value = _APPROVE_CRITIQUE
        elif schema is Memo:
            inner.invoke.return_value = _MEMO
        else:
            inner.invoke.return_value = MagicMock()
        return inner

    mock_llm.with_structured_output.side_effect = structured_side_effect
    return mock_llm


def _make_llm_mock_always_revision() -> MagicMock:
    """LLM mock for test_graph_max_rounds_forces_synthesizer.

    Critic always returns request_revision so we can test the max-rounds guard.
    """
    mock_llm = MagicMock()

    def structured_side_effect(schema):
        inner = MagicMock()
        if schema is ResearchQuery:
            inner.invoke.return_value = ResearchQuery(
                original_query="test query", sub_questions=_SUB_QUESTIONS
            )
        elif schema is ResearchNoteList:
            inner.invoke.return_value = ResearchNoteList(notes=_SIX_NOTES[:3])
        elif schema is Critique:
            inner.invoke.return_value = _REVISION_CRITIQUE
        elif schema is Memo:
            inner.invoke.return_value = _MEMO
        else:
            inner.invoke.return_value = MagicMock()
        return inner

    mock_llm.with_structured_output.side_effect = structured_side_effect
    return mock_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_graph_compiles(mock_env: None) -> None:
    """build_graph() returns a non-None compiled graph without making external calls."""
    with patch("research_analyst.llm.ChatGroq", return_value=MagicMock()):
        from research_analyst.graph import build_graph

        graph = build_graph()

    assert graph is not None


def test_graph_full_flow_approve_on_first_round(mock_env: None) -> None:
    """Full pipeline: researcher → critic (approve) → synthesizer in a single pass.

    Assertions:
    - final_memo is set
    - revision_round == 0 (no revision loops occurred)
    - len(notes) == 6 (3 notes per sub-question × 2 sub-questions)
    - len(critiques) == 1
    """
    mock_llm = _make_llm_mock_approve_first_round()

    with (
        patch("research_analyst.llm.ChatGroq", return_value=mock_llm),
        patch(
            "research_analyst.agents.researcher.tavily_search",
            return_value=_MOCK_SEARCH_RESULTS,
        ),
    ):
        from research_analyst.graph import run

        final_state = run("What is quantum computing?")

    assert final_state.final_memo is not None
    assert final_state.revision_round == 0
    assert len(final_state.notes) == 6
    assert len(final_state.critiques) == 1
    assert final_state.critiques[0].verdict == "approve"


def test_graph_revision_loop_then_approve(mock_env: None) -> None:
    """Pipeline performs one revision loop before approving.

    Assertions:
    - revision_round == 1 (incremented once)
    - len(critiques) == 2 (one request_revision + one approve)
    - final_memo is set
    """
    mock_llm = _make_llm_mock_revision_then_approve()

    with (
        patch("research_analyst.llm.ChatGroq", return_value=mock_llm),
        patch(
            "research_analyst.agents.researcher.tavily_search",
            return_value=_MOCK_SEARCH_RESULTS,
        ),
    ):
        from research_analyst.graph import run

        final_state = run("What is quantum computing?")

    assert final_state.final_memo is not None
    assert final_state.revision_round == 1
    assert len(final_state.critiques) == 2
    assert final_state.critiques[0].verdict == "request_revision"
    assert final_state.critiques[1].verdict == "approve"


def test_graph_max_rounds_forces_synthesizer(
    mock_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When MAX_CRITIQUE_ROUNDS=1 and critic always requests revision, synthesizer still runs.

    With MAX_CRITIQUE_ROUNDS=1, the routing condition `revision_round < MAX_CRITIQUE_ROUNDS`
    allows one revision (round 0 → increment → round 1). After round 1, revision_round is
    no longer < 1, so the graph forces the synthesizer regardless of the critic verdict.
    """
    import research_analyst.graph as graph_module

    monkeypatch.setattr(graph_module, "MAX_CRITIQUE_ROUNDS", 1)

    mock_llm = _make_llm_mock_always_revision()

    with (
        patch("research_analyst.llm.ChatGroq", return_value=mock_llm),
        patch(
            "research_analyst.agents.researcher.tavily_search",
            return_value=_MOCK_SEARCH_RESULTS,
        ),
    ):
        # Rebuild graph AFTER patching MAX_CRITIQUE_ROUNDS so the new value is captured
        # by the closure in _route_after_critic.
        graph = graph_module.build_graph()
        initial_state = AgentState(query="What is quantum computing?")
        result = graph.invoke(initial_state)
        final_state = AgentState(**result)

    assert final_state.final_memo is not None
    # Both critiques should be request_revision
    assert all(c.verdict == "request_revision" for c in final_state.critiques)
