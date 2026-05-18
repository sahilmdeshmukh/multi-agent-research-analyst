from __future__ import annotations

from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import ValidationError

from research_analyst.llm import get_groq_llm
from research_analyst.schemas import AgentState, Critique

load_dotenv()

_CRITIC_SYSTEM_PROMPT = """You are a rigorous research quality critic.
Your job is to review a set of research notes and decide whether they are sufficient to
write a comprehensive answer to the original query, or whether the researcher needs to do
more work.

**Approve** when ALL of the following hold:
- There are at least 5 research notes with source URLs
- There are at least 3 distinct source domains (e.g. bbc.com, reuters.com, gov.uk are 3 domains)
- There are no obvious contradictions between claims
- The core aspects of the original query are covered

**Request revision** when any of the above criteria are NOT met, or when there are
significant gaps, contradictions, or weak/missing sources.

If you approve: set `follow_up_questions` to an empty list.
If you request revision: include 2–4 specific, independently-searchable follow-up questions
that directly target the identified gaps. Make them precise enough for a web search.

Be concise and factual in your reasoning. Do not fabricate information."""


def _format_notes_for_critique(state: AgentState) -> str:
    """Render the current research notes as a readable block for the LLM."""
    if not state.notes:
        return "No research notes have been collected yet."

    lines: list[str] = [f"Original query: {state.query}", "", "Research notes:"]
    for i, note in enumerate(state.notes, start=1):
        lines.append(
            f"\n[{i}] Claim: {note.claim}\n"
            f"    Source: {note.source_url}\n"
            f"    Title: {note.source_title}\n"
            f"    Confidence: {note.confidence:.2f}"
        )
    return "\n".join(lines)


def _count_distinct_domains(state: AgentState) -> int:
    """Count unique second-level domains across all notes."""
    domains: set[str] = set()
    for note in state.notes:
        try:
            parsed = urlparse(note.source_url)
            # skip if url lacks a scheme — urlparse can't extract netloc without one
            if not parsed.scheme:
                continue
            # netloc may be 'www.example.com' — strip leading 'www.'
            netloc = parsed.netloc.lower().removeprefix("www.").strip()
            if netloc:
                domains.add(netloc)
        except ValueError:
            pass
    return len(domains)


def critic_node(state: AgentState) -> AgentState:
    """LangGraph node: critique the current research notes and update state."""
    llm = get_groq_llm("CRITIC_MODEL")
    structured = llm.with_structured_output(Critique)

    notes_text = _format_notes_for_critique(state)
    domain_count = _count_distinct_domains(state)

    user_msg = (
        f"{notes_text}\n\n"
        f"Statistics:\n"
        f"- Total notes collected: {len(state.notes)}\n"
        f"- Distinct source domains: {domain_count}\n"
        f"- Revision round: {state.revision_round}\n\n"
        "Please critique these notes and return a structured verdict."
    )

    try:
        critique: Critique = structured.invoke(
            [
                {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
    except (ValidationError, ValueError) as exc:
        # Fallback: request revision with a generic question so the graph can continue
        critique = Critique(
            verdict="request_revision",
            gaps=["Unable to parse structured critique from model response"],
            weak_sources=[],
            contradictions=[],
            follow_up_questions=[f"Please search for more information about: {state.query}"],
            reasoning=f"Critique parsing failed ({exc}); defaulting to revision request.",
        )

    return AgentState(
        query=state.query,
        sub_questions=state.sub_questions,
        notes=state.notes,
        critiques=state.critiques + [critique],
        revision_round=state.revision_round,
        final_memo=state.final_memo,
    )
