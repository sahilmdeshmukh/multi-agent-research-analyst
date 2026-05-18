from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import ValidationError

from research_analyst.schemas import AgentState, ResearchNote, ResearchNoteList, ResearchQuery
from research_analyst.tools.search import tavily_search

load_dotenv()

_DECOMPOSE_PROMPT = """You are a research planning expert.
Given a research question, decompose it into 2–4 focused sub-questions that,
when answered together, fully address the original question.
Be specific — each sub-question should be independently searchable."""

_EXTRACT_PROMPT = """You are a research analyst extracting structured notes from web search results.
For each source provided, extract factual claims that are directly relevant to the sub-question.
Assign a confidence score (0.0–1.0) based on how authoritative and specific the source is.
Return ONLY claims that appear in the provided text — do not hallucinate."""


def _get_llm(model_env_var: str = "RESEARCHER_MODEL") -> ChatGroq:
    model = os.environ.get(model_env_var, "llama-3.3-70b-versatile")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return ChatGroq(model=model, api_key=api_key, temperature=0)


def _decompose_query(query: str) -> list[str]:
    llm = _get_llm()
    structured = llm.with_structured_output(ResearchQuery)
    result: ResearchQuery = structured.invoke(
        [
            {"role": "system", "content": _DECOMPOSE_PROMPT},
            {"role": "user", "content": f"Research question: {query}"},
        ]
    )
    return result.sub_questions


def _extract_notes_from_results(
    sub_question: str, search_results: list[dict]
) -> list[ResearchNote]:
    if not search_results:
        return []

    llm = _get_llm()
    # Groq structured output doesn't reliably handle bare list[T]; wrap in a model
    structured = llm.with_structured_output(ResearchNoteList)

    sources_text = "\n\n".join(
        f"Source [{i+1}]\nTitle: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:800]}"
        for i, r in enumerate(search_results)
    )

    user_msg = (
        f"Sub-question: {sub_question}\n\n"
        f"Search results:\n{sources_text}\n\n"
        "Extract structured research notes from these sources."
    )

    try:
        result: ResearchNoteList = structured.invoke(
            [
                {"role": "system", "content": _EXTRACT_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        return result.notes if result.notes else []
    except (ValidationError, ValueError):
        return []


def researcher_node(state: AgentState) -> AgentState:
    """LangGraph node: decompose query, search, extract notes, update state."""
    # On revision rounds, use critic's follow-up questions as new sub-questions
    if state.revision_round > 0 and state.critiques:
        last_critique = state.critiques[-1]
        sub_questions = last_critique.follow_up_questions or [state.query]
    else:
        sub_questions = _decompose_query(state.query)

    new_notes: list[ResearchNote] = []
    for sub_q in sub_questions:
        results = tavily_search(sub_q, max_results=4)
        notes = _extract_notes_from_results(sub_q, results)
        new_notes.extend(notes)

    return AgentState(
        query=state.query,
        sub_questions=sub_questions,
        notes=state.notes + new_notes,
        critiques=state.critiques,
        revision_round=state.revision_round,
        final_memo=state.final_memo,
    )
