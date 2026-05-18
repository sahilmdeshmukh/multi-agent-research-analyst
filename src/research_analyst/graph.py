"""LangGraph graph for the multi-agent research analyst.

Day 2: single-node graph wiring just the researcher.
Day 3: added critic, synthesizer, and the conditional critique loop.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from research_analyst.agents.critic import critic_node
from research_analyst.agents.researcher import researcher_node
from research_analyst.agents.synthesizer import synthesizer_node
from research_analyst.schemas import AgentState

load_dotenv()

MAX_CRITIQUE_ROUNDS: int = int(os.environ.get("MAX_CRITIQUE_ROUNDS", "3"))


def increment_revision_node(state: AgentState) -> AgentState:
    """Tiny router node: increments revision_round before looping back to researcher."""
    return AgentState(
        query=state.query,
        sub_questions=state.sub_questions,
        notes=state.notes,
        critiques=state.critiques,
        revision_round=state.revision_round + 1,
        final_memo=state.final_memo,
    )


def _route_after_critic(state: AgentState) -> str:
    """Conditional edge: approve → synthesizer; revision within limit → increment; else → synthesizer."""
    if not state.critiques:
        return "synthesizer"

    last_critique = state.critiques[-1]
    if last_critique.verdict == "approve":
        return "synthesizer"
    if state.revision_round < MAX_CRITIQUE_ROUNDS:
        return "increment_revision"
    # Max rounds reached — force synthesizer
    return "synthesizer"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("researcher", researcher_node)
    builder.add_node("critic", critic_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("increment_revision", increment_revision_node)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "critic")

    builder.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "synthesizer": "synthesizer",
            "increment_revision": "increment_revision",
        },
    )

    builder.add_edge("increment_revision", "researcher")
    builder.add_edge("synthesizer", END)

    return builder.compile()


def run(query: str) -> AgentState:
    graph = build_graph()
    initial_state = AgentState(query=query)
    result = graph.invoke(initial_state)
    return AgentState(**result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m research_analyst.graph \"<query>\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\nResearching: {query}\n{'=' * 60}")

    state = run(query)

    print(f"\nSub-questions explored ({len(state.sub_questions)}):")
    for i, q in enumerate(state.sub_questions, 1):
        print(f"  {i}. {q}")

    print(f"\nNotes collected ({len(state.notes)}):")
    for i, note in enumerate(state.notes, 1):
        print(f"\n  [{i}] Claim: {note.claim}")
        print(f"      Source: {note.source_title}")
        print(f"      URL: {note.source_url}")
        print(f"      Confidence: {note.confidence:.2f}")

    if state.critiques:
        print(f"\nCritique rounds ({len(state.critiques)}):")
        for i, critique in enumerate(state.critiques, 1):
            print(f"\n  Round {i} — Verdict: {critique.verdict.upper()}")
            print(f"  Reasoning: {critique.reasoning}")
            if critique.gaps:
                print(f"  Gaps: {', '.join(critique.gaps)}")
            if critique.follow_up_questions:
                print("  Follow-up questions:")
                for fq in critique.follow_up_questions:
                    print(f"    - {fq}")

    if state.final_memo:
        print(f"\n{'=' * 60}")
        print(f"FINAL MEMO: {state.final_memo.title}")
        print(f"{'=' * 60}")
        print(f"\nSummary: {state.final_memo.summary}")
        for section in state.final_memo.sections:
            print(f"\n## {section.heading}")
            print(section.body)
        if state.final_memo.sources:
            print("\nSources:")
            for j, src in enumerate(state.final_memo.sources, 1):
                print(f"  [{j}] {src}")
    else:
        print("\n[No final memo produced]")
