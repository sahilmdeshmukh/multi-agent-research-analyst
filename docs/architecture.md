# Architecture

The system is a three-agent LangGraph pipeline with an explicit critique loop.

```mermaid
flowchart TD
    A([START]) --> B

    B["🔬 Researcher\n─────────────\nDecomposes query into\n2-4 sub-questions\nSearches Tavily\nExtracts structured notes\n(claim + URL + confidence)"]

    B --> C

    C["🔍 Critic\n─────────────\nReviews all notes\nChecks: >= 5 notes, >= 3 domains\nno contradictions\nVerdicts: approve / request_revision"]

    C --> D{Verdict?}

    D -->|"approve"| E
    D -->|"request_revision\nround < MAX_ROUNDS"| F
    D -->|"request_revision\nround >= MAX_ROUNDS\n(force)"| E

    F["Increment\nrevision_round + 1"]
    F --> B

    E["✍️ Synthesizer\n─────────────\nWrites cited memo\nInline [N] citations\nSources from notes only"]

    E --> G([END])

    style A fill:#27ae60,color:#fff,stroke:none
    style G fill:#27ae60,color:#fff,stroke:none
    style D fill:#e67e22,color:#fff,stroke:none
    style F fill:#2980b9,color:#fff,stroke:none
```

## Agent State

Shared `AgentState` (Pydantic) flows through all nodes:

| Field | Type | Description |
|---|---|---|
| `query` | `str` | Original research question |
| `sub_questions` | `list[str]` | Decomposed search queries |
| `notes` | `list[ResearchNote]` | Accumulated across all rounds |
| `critiques` | `list[Critique]` | History of critic verdicts |
| `revision_round` | `int` | Current loop iteration (0-indexed) |
| `final_memo` | `Memo \| None` | Output — set by synthesizer |

## Key Design Decisions

- **Critique loop capped at `MAX_CRITIQUE_ROUNDS` (default 3)** — prevents infinite loops on hard queries
- **Notes accumulate across rounds** — each revision adds to the evidence base, not replaces it
- **Groq `with_structured_output(Model)`** — all agent I/O is typed Pydantic; bare `list[T]` is wrapped in `ResearchNoteList` to work around Groq structured output limitations
- **Shared `get_groq_llm()` factory** — all three agents use the same factory from `llm.py`
