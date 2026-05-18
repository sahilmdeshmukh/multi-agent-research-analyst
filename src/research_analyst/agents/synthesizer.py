from __future__ import annotations

from dotenv import load_dotenv
from pydantic import ValidationError

from research_analyst.llm import get_groq_llm
from research_analyst.schemas import AgentState, Memo

load_dotenv()

_SYNTHESIZER_SYSTEM_PROMPT = """You are an expert research analyst who writes clear, well-structured memos.
Your task is to synthesise a set of research notes into a polished memo with inline citations.

**Requirements:**
- Write 3–5 sections, each with a short heading and a markdown-formatted body
- Use inline citations like [1], [2], etc. wherever you reference a claim from the notes
- The `sources` list MUST contain only URLs that actually appear in the notes provided — do NOT invent URLs
- Citation numbers [N] in the body must correspond to the position of the URL in the `sources` list
  (i.e. [1] → sources[0], [2] → sources[1], etc.)
- Write a concise executive summary of 2–3 sentences covering the key finding
- Be factual, precise, and well-organised — do not fabricate or embellish information
- If the notes are sparse, write what you can from the evidence available"""


def _format_notes_for_synthesis(state: AgentState) -> str:
    """Render all accumulated research notes as a numbered block for the LLM."""
    if not state.notes:
        return "No research notes have been collected."

    lines: list[str] = [f"Original query: {state.query}", "", "Research notes:"]
    for i, note in enumerate(state.notes, start=1):
        lines.append(
            f"\n[{i}] Claim: {note.claim}\n"
            f"    Source URL: {note.source_url}\n"
            f"    Source Title: {note.source_title}\n"
            f"    Confidence: {note.confidence:.2f}"
        )
    return "\n".join(lines)


def synthesizer_node(state: AgentState) -> AgentState:
    """LangGraph node: synthesise research notes into a final cited memo."""
    llm = get_groq_llm("SYNTHESIZER_MODEL")
    structured = llm.with_structured_output(Memo)

    notes_text = _format_notes_for_synthesis(state)
    user_msg = (
        f"{notes_text}\n\n"
        f"Total notes: {len(state.notes)}\n"
        f"Revision rounds completed: {state.revision_round}\n\n"
        "Please write a comprehensive research memo with inline citations."
    )

    try:
        memo: Memo = structured.invoke(
            [
                {"role": "system", "content": _SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
    except (ValidationError, ValueError) as exc:
        memo = Memo(
            title="Research incomplete",
            summary=str(exc),
            sections=[],
            sources=[],
        )

    return AgentState(
        query=state.query,
        sub_questions=state.sub_questions,
        notes=state.notes,
        critiques=state.critiques,
        revision_round=state.revision_round,
        final_memo=memo,
    )
