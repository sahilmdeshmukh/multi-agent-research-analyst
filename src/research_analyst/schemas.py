from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchNote(BaseModel):
    claim: str = Field(description="A factual claim derived from the source")
    source_url: str = Field(description="URL of the source")
    source_title: str = Field(description="Title of the source page or article")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the claim (0–1)")


class ResearchQuery(BaseModel):
    original_query: str
    sub_questions: list[str] = Field(
        description="2–4 focused sub-questions that together answer the original query"
    )


class ResearchNoteList(BaseModel):
    """Wrapper so Groq structured output can return a list of notes reliably."""

    notes: list[ResearchNote] = Field(
        description="List of research notes extracted from the sources"
    )


class Critique(BaseModel):
    verdict: Literal["approve", "request_revision"]
    gaps: list[str] = Field(description="Topics or claims that lack sufficient evidence")
    weak_sources: list[str] = Field(description="Source URLs deemed low-quality or absent")
    contradictions: list[str] = Field(description="Claims that conflict with each other")
    follow_up_questions: list[str] = Field(
        description="Specific questions the researcher should answer next (empty if approved)"
    )
    reasoning: str = Field(description="Brief explanation of the verdict")


class MemoSection(BaseModel):
    heading: str
    body: str = Field(description="Markdown-formatted section body with inline citations like [1]")


class Memo(BaseModel):
    title: str
    summary: str = Field(description="2–3 sentence executive summary")
    sections: list[MemoSection]
    sources: list[str] = Field(
        description="Ordered list of source URLs matching the [N] inline citation numbers"
    )


class AgentState(BaseModel):
    query: str
    sub_questions: list[str] = []
    notes: list[ResearchNote] = []
    critiques: list[Critique] = []
    revision_round: int = 0
    final_memo: Memo | None = None
