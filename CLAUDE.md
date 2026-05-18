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
- [x] Day 3: critic + synthesizer + graph loop + streaming UI
- [x] Day 4: eval harness + architecture diagram + README + HF Spaces deploy (eval results pending Groq token reset)

## Notes
- Python 3.14 used (uv resolved 3.14 from system); all deps install fine
- Groq structured output requires list[T] to be wrapped in a Pydantic model (ResearchNoteList) — bare list[T] schema causes malformed responses
- Shared LLM factory in `src/research_analyst/llm.py` — all agents use `get_groq_llm(model_env_var)`
- `increment_revision_node` in graph.py handles revision_round increment (keeps conditional edge function pure)
- asyncio/Streamlit: use plain `asyncio.run()` — Streamlit scripts run in a worker thread with a fresh loop
- 11 tests passing (3 researcher, 2 critic, 2 synthesizer, 4 graph smoke) ✓
- Local demo: `uv run streamlit run app.py`

## Open questions
(empty — add here as they come up)
