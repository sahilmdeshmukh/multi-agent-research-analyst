# Claude Code Notes — Multi-Agent Research Analyst

## Read first
- PLAN.md in this repo has the full project plan, day-by-day tasks, and
  acceptance criteria. Read it before proposing changes.
- README.md is the public-facing description.

## Stack
Python 3.12 · uv · LangGraph · langchain-groq (llama-3.3-70b-versatile) ·
Tavily · Streamlit · Pydantic · pytest

## Conventions
- Conventional commits: feat:, fix:, refactor:, docs:, test:, chore:
- Atomic commits — one logical change per commit
- Type hints everywhere; strict Pydantic models for all agent I/O
- No bare `except` — name the exception class
- Ruff for lint, line length 100
- All keys come from .env via python-dotenv

## Status
- [x] Day 1: scaffold + Streamlit placeholder
- [x] Day 2: researcher agent end-to-end
- [ ] Day 3: critic + synthesizer + graph loop + streaming UI
- [ ] Day 4: evals + deploy + README + GIF

## Notes
- Python 3.14 used (uv resolved 3.14 from system); all deps install fine
- Groq structured output requires list[T] to be wrapped in a Pydantic model (ResearchNoteList) — bare list[T] schema causes malformed responses
- Smoke test: 20 real notes on "How does ASML's monopoly affect chip prices?" ✓
- 3 unit tests passing ✓

## Open questions
(empty — add here as they come up)
