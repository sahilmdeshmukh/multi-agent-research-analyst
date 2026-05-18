# Multi-Agent Research Analyst — Design

**Date:** 2026-05-18
**Status:** Approved for implementation planning
**Owner:** Sahil Manoj Deshmukh

## 1. Purpose & success criteria

A LangGraph system in which three specialized agents — researcher, critic, synthesizer — collaborate to produce a cited research memo for an open-ended user query. The project is a portfolio piece intended to back up the claims in a public LinkedIn post and to teach the technologies it uses (LangGraph, structured-output LLM calls, multi-agent orchestration, eval harness design, Streamlit streaming, Hugging Face Spaces deployment, LangSmith tracing).

**Success criteria for v1:**

- All three agents implemented and orchestrated via a single LangGraph `StateGraph`.
- Explicit critique loop with a 3-round cap; the synthesizer is honest when the cap is hit.
- Eval over 20 fixed questions: zero hallucinated citations in the manual run that backs the public claim.
- Streamlit UI streams agent traces and the final memo to the user.
- Deployed to Hugging Face Spaces with a public URL; rate-limited at the app layer.
- Tracing on by default via LangSmith free tier.
- Honest README that describes what was actually built — not the LinkedIn post.

**Non-goals for v1:**

- No persistence (memos are session-scoped).
- No user accounts or auth.
- No CI/CD; tests run locally only.
- No production observability (Sentry, metrics, alerts).
- No parallel sub-query planning (planner node is a v2 idea).

## 2. Architecture

```
                    ┌──────────────┐
                    │  user query  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  researcher  │◄────┐
                    └──────┬───────┘     │
                           ▼             │ revise
                    ┌──────────────┐     │ (≤ 3 rounds)
                    │    critic    │─────┘
                    └──────┬───────┘
                       approve
                           ▼
                    ┌──────────────┐
                    │ synthesizer  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  cited memo  │
                    └──────────────┘
```

Single `StateGraph` with three nodes (`researcher`, `critic`, `synthesizer`) and one conditional edge after the critic.

**Routing rule** on the critic's conditional edge:

```
if state["verdict"] == "revise" and state["critique_rounds"] < MAX_CRITIQUE_ROUNDS:
    return "researcher"
else:
    return "synthesizer"
```

When the cap is hit, `state["forced_pass"]` is set to `True` so the synthesizer can write a memo that is honest about unresolved source-quality concerns rather than pretending the critic was satisfied.

## 3. State schema

```python
class GraphState(TypedDict):
    query: str
    notes: list[ResearchNote]           # accumulated across revisions
    critiques: list[Critique]           # one per round, for the trace
    critique_rounds: int                # incremented by critic node
    verdict: Literal["approve", "revise"]
    forced_pass: bool                   # True if cap hit
    memo: str | None                    # filled by synthesizer

class ResearchNote(BaseModel):
    claim: str
    source_url: HttpUrl | None          # None = "no source" sentinel (see §7 error 2)
    source_title: str
    confidence: float                   # 0.0 - 1.0, self-reported by researcher

class Critique(BaseModel):
    verdict: Literal["approve", "revise"]
    weak_notes: list[int]               # indices into notes[] flagged as weak
    reasoning: str
    revision_guidance: str | None       # what researcher should do next round
```

**Accumulation invariant:** `notes` accumulates across rounds rather than being replaced. This lets the synthesizer see the full research trail, lets the trace show progress, and lets the critic distinguish "the researcher didn't fix the weak source" from "the researcher dropped good notes."

## 4. Components & file layout

```
multi-agent-research-analyst/
├── pyproject.toml
├── .env.example
├── src/mara/
│   ├── __init__.py
│   ├── config.py               # pydantic-settings, reads .env
│   ├── schemas.py              # ResearchNote, Critique, GraphState
│   ├── tools/
│   │   └── tavily_search.py    # LangChain Tool wrapper, returns list[dict]
│   ├── agents/
│   │   ├── researcher.py       # node fn
│   │   ├── critic.py           # node fn
│   │   └── synthesizer.py      # node fn
│   ├── graph.py                # build_graph() returns compiled StateGraph
│   └── llm.py                  # ChatGroq factory + retry helpers
├── app/
│   └── streamlit_app.py        # streaming UI, calls graph.stream()
├── eval/
│   ├── questions.jsonl         # 20 fixed eval questions
│   ├── judge.py                # LLM-as-judge harness
│   ├── manual_runner.py        # CLI runner, dumps memos to results/
│   └── results/                # gitignored except for the headline run
├── tests/
│   ├── test_schemas.py
│   ├── test_researcher.py
│   ├── test_critic.py
│   ├── test_graph.py
│   └── conftest.py
├── README.md
└── .gitignore
```

**Design choices worth flagging:**

- **`src/mara/` package** keeps imports stable when we later wrap in a CLI or HTTP API.
- **`llm.py` as a single factory** so model swaps (Groq → OpenAI → Anthropic) are one file change. Backs the "provider-agnostic LangGraph" framing in the README without fakery.
- **`config.py` uses `pydantic-settings`** rather than raw `os.getenv`. Typed config with validation, fits the Pydantic learning thread.

## 5. Configuration

All config lives in `.env` (gitignored) and is loaded via `pydantic-settings` in `src/mara/config.py`. `.env.example` documents every key.

| Key | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — (required) | Groq LLM auth |
| `TAVILY_API_KEY` | — (required) | Tavily search auth |
| `LANGSMITH_API_KEY` | — (optional) | Tracing; if absent, tracing is disabled gracefully |
| `LANGSMITH_PROJECT` | `mara-dev` | LangSmith project name |
| `RESEARCHER_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `CRITIC_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `SYNTHESIZER_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `MAX_CRITIQUE_ROUNDS` | `3` | Critique loop cap |
| `TAVILY_MAX_RESULTS` | `5` | Per-search result cap |
| `RATE_LIMIT_PER_HOUR` | `5` | Streamlit per-IP cap |

## 6. Data flow (end-to-end trace)

Example query: *"How effective is GLP-1 therapy for non-diabetic weight loss?"*

| Step | Node | Reads from state | Does | Writes to state |
|---|---|---|---|---|
| 1 | researcher | `query`, `critiques=[]` | One Tavily call (the researcher does multiple rounds via the graph loop, not multiple calls within one invocation), parses results into `ResearchNote`s via `with_structured_output()` | `notes` (3–6 entries) |
| 2 | critic | `notes` | Inspects URL domain + claim plausibility per note; flags weak sources | `critiques[0]`, `verdict`, `critique_rounds=1` |
| 3 | conditional edge | `verdict`, `critique_rounds` | Routes back to researcher (verdict=revise, rounds<3) | — |
| 4 | researcher | `query`, `critiques[0]`, existing `notes` | Reruns Tavily with refined query informed by `revision_guidance` | appends new notes |
| 5 | critic | updated `notes` | Approves (or revises again, up to round 3) | `critiques[1]`, `verdict=approve` |
| 6 | synthesizer | `notes`, `critiques`, `forced_pass=False` | Writes markdown memo with inline `[1]`-style citations and a sources section | `memo` |

Streamlit reads `graph.stream(...)` and renders each node's update as a collapsible block; the final memo renders below the trace.

## 7. Error handling

Five real failure modes, each with a specific defense:

1. **Llama returns malformed JSON.** `with_structured_output()` is the first line of defense. Wrap each agent's LLM call with `@retry(stop=stop_after_attempt(3), wait=wait_exponential)` (Tenacity). If all 3 retries fail, raise a typed `AgentParseError`. The graph catches it at the top level and propagates a user-facing "agent failed" message — not a stack trace.
2. **Tavily returns zero results.** Researcher returns `notes` containing one synthetic placeholder: `ResearchNote(claim="no sources found for: <query>", confidence=0.0, source_url=None, source_title="no-results")`. Critic detects this (any note with `source_url is None`) and forces `verdict=revise` with guidance "broaden search terms." If still empty after 3 rounds, synthesizer produces a memo that explicitly states no sources were found. Never hallucinates.
3. **Critique cap exhausted.** `forced_pass=True` is passed to the synthesizer, which appends a "Source quality caveat" section listing the unresolved critic objections. Honest by construction.
4. **Groq 429 rate limit.** Tenacity retry with exponential backoff up to 60s total wait. After that, surface to the UI with a "rate limited, retry in 30s" banner.
5. **Streamlit `graph.stream` exception mid-flight.** Wrap the stream loop in try/except; render a partial-result panel with whatever `notes`/`critiques` were collected before the failure. Don't lose the user's work.

## 8. Testing strategy

Three layers, all local:

- **Unit tests** — `tests/test_schemas.py`, `test_researcher.py`, `test_critic.py`. Pydantic validation; agent node functions called with a fake LLM (`langchain_core.language_models.fake.FakeListChatModel`) and fake Tavily (fixture-backed). Assert state transitions, not LLM content.
- **Graph tests** — `tests/test_graph.py`. Full `build_graph().invoke(...)` with the LLM scripted to produce: (a) one-shot approval, (b) two revisions then approve, (c) cap-exhausted forced pass. Verifies routing, counter increments, and the `forced_pass` plumbing.
- **Eval harness** — `eval/`. The real correctness measurement. Two scripts:
  - `eval/judge.py`: runs all 20 questions through the real graph + a Groq judge that scores each citation on **(URL resolves with HTTP 200) AND (snippet from page supports claim)**. Used for fast iteration during development. CSV output.
  - `eval/manual_runner.py`: same 20 questions, no judge — runs the graph and dumps memos to `eval/results/<timestamp>/` for hand verification. The headline "0/20 hallucinated" claim for the LinkedIn post comes from this run, **not the LLM judge**.

**No CI in v1.** Tests run via `pytest` locally. CI is a post-launch addition.

## 9. Eval question set

20 questions, distributed across four bands so the eval surfaces different failure modes:

- 8 factual/numeric (e.g., "What was Nvidia's data center revenue Q3 2024?") — tests citation precision.
- 6 open-ended/analytical (e.g., "Tradeoffs of MoE vs dense LLMs?") — tests synthesizer quality.
- 4 freshness (e.g., "Latest FDA approvals for Alzheimer's, 2025") — tests Tavily date handling.
- 2 deliberately under-sourced (e.g., "Is there scientific evidence for telepathy?") — tests honest "no sources" behavior.

Question set is authored as part of the build and committed at `eval/questions.jsonl`. The headline manual run's output is committed to `eval/results/headline/` (a fixed directory checked into git) so the LinkedIn post is auditable.

## 10. Streamlit UI

Single page. Layout, top to bottom:

1. Title + one-sentence description.
2. Text input + submit button.
3. While running: collapsible blocks per node, populated live from `graph.stream(...)`. Each block shows the node name, the state delta (e.g., new notes, critique verdict), and elapsed time.
4. After completion: the final memo rendered as Markdown, with the sources section at the bottom.
5. A "raw state" expander at the bottom for debugging.

**Rate limit:** in-memory `{ip_hash: [timestamps]}` dict, max `RATE_LIMIT_PER_HOUR` memos per hour per IP hash. Resets on app restart (acceptable for v1; HF Spaces restart frequency is low enough).

## 11. Deployment (Hugging Face Spaces)

Hugging Face Spaces with the **Streamlit SDK**. Required artifacts beyond the code:

- `requirements.txt` (generated from `pyproject.toml` via `uv pip compile` or hand-mirrored — pinned versions).
- Spaces **Secrets** for `GROQ_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`. `.env` never leaves local.
- A `README.md` at the repo root with the Spaces YAML frontmatter (sdk: streamlit, app_file: app/streamlit_app.py).

## 12. Dependencies

```toml
[project]
requires-python = ">=3.11"
dependencies = [
  "langchain-groq",
  "langchain-core",
  "langchain-community",
  "langgraph",
  "tavily-python",
  "pydantic",
  "pydantic-settings",
  "streamlit",
  "langsmith",
  "tenacity",
]

[dependency-groups]
dev = [
  "pytest",
  "pytest-asyncio",
  "ruff",
]
```

Managed with `uv`.

## 13. Out of scope (explicit non-goals)

- Persistence / database.
- Auth / accounts.
- CI/CD pipelines.
- Production-grade metrics or alerting.
- Multi-language UI.
- Cost dashboarding (we rely on Groq's free-tier ceiling).
- Planner / parallel sub-query fan-out (Topology B from brainstorming) — saved for v2.
- Critic-orchestrated topology (Topology C) — requires stronger tool-calling than Llama 3.3 70b reliably provides.
