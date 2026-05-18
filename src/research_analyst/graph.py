"""LangGraph graph for the multi-agent research analyst.

Day 2: single-node graph wiring just the researcher.
Day 3: will add critic, synthesizer, and the conditional critique loop.
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from research_analyst.agents.researcher import researcher_node
from research_analyst.schemas import AgentState

load_dotenv()


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("researcher", researcher_node)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", END)
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
    print(f"\nResearching: {query}\n{'='*60}")

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
