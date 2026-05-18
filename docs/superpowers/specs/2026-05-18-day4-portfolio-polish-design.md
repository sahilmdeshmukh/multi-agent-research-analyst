---
title: Day 4 — Eval Harness, Architecture Diagram, README Polish
date: 2026-05-18
status: approved
---

# Day 4: Portfolio Polish Design

## Goal

Complete the project so it is portfolio-ready on GitHub. Three deliverables:
1. Eval harness with real comparison numbers
2. Architecture diagram (Mermaid, GitHub-native)
3. Polished README filled with real eval data

GIF recording and Hugging Face Spaces deploy are deferred to a separate session.

---

## Section 1: Eval Harness

### Files

- `eval/__init__.py` — empty, makes eval a package
- `eval/queries.yaml` — 10 diverse research queries across domains (tech, economics, science, geopolitics, health)
- `eval/run_eval.py` — CLI script that runs both systems and produces results

### Baseline agent

A minimal single-agent baseline that is fair and sincere (per PLAN.md anti-patterns):
- Uses the same `researcher_node` (Tavily search + Groq note extraction)
- Uses the same `synthesizer_node`
- No critic, no revision loop — one pass only
- Built inline inside `run_eval.py` as a two-node LangGraph (no separate file needed)

This mirrors what a "single LLM with a search tool" agent would produce.

### Metrics captured per run

| Metric | How captured |
|---|---|
| `source_count` | `len(state.notes)` |
| `unique_domains` | distinct netloc values from note source URLs |
| `latency_seconds` | wall-clock time for `graph.invoke()` |
| `memo_length_words` | word count of all section bodies in final memo |
| `hallucinated_citations` | initially 0 — set manually via CLI flag after human review |

### Output

- Console: formatted comparison table using f-strings (no extra dependencies)
- File: `eval/results.json` with per-query and aggregated results

### CLI flags

```
uv run python eval/run_eval.py              # runs all 10 queries
uv run python eval/run_eval.py --query-limit 3   # run first N queries (for testing)
uv run python eval/run_eval.py --add-hallucination-scores  # prompts for manual scores
```

### Reproducibility

- Model versions logged in results.json
- Queries fixed in queries.yaml (deterministic input)
- temperature=0 on all LLM calls (already the default)

---

## Section 2: Architecture Diagram

### File: `docs/architecture.md`

Contains a single Mermaid flowchart showing the full graph topology:

```
START → researcher → critic → [conditional edge]
                                  ↓ approve
                              synthesizer → END
                                  ↓ request_revision + round < MAX
                          increment_revision → researcher (loop)
                                  ↓ request_revision + round ≥ MAX
                              synthesizer → END
```

GitHub renders Mermaid natively in `.md` files — no PNG generation needed.

---

## Section 3: README Overhaul

### File: `README.md` (full replacement)

Sections in order:

1. **Header** — title, one-line description
2. **Demo GIF** — `![demo](docs/demo.gif)` placeholder (user fills in)
3. **Live demo link** — placeholder for HF Spaces URL (user fills in after deploy)
4. **Why this exists** — multi-agent vs single-agent motivation paragraph
5. **What it does** — three-bullet agent description
6. **Architecture** — embed Mermaid diagram inline (copy from docs/architecture.md)
7. **Results table** — filled with REAL numbers from eval/results.json. No placeholder values.
8. **Tech choices** — Groq, LangGraph, Tavily rationale
9. **Quickstart** — clone → uv sync → .env → streamlit run
10. **Configuration table** — all env vars with defaults
11. **Tech stack** — one-liner list
12. **Roadmap** — checklist matching PLAN.md template

### Constraint

The Results table is written AFTER eval/run_eval.py completes. The README writing step depends on results.json existing with real data.

---

## Implementation Order

1. `eval/queries.yaml` + `eval/__init__.py`
2. `eval/run_eval.py` (baseline graph + metrics + output)
3. Run eval — produces `eval/results.json`
4. `docs/architecture.md` (Mermaid diagram)
5. `README.md` (uses real numbers from results.json)
6. `CLAUDE.md` status update

---

## Success Criteria

- `uv run python eval/run_eval.py` completes without error, produces `eval/results.json`
- Results table in README contains real numbers (not placeholders)
- `docs/architecture.md` renders as a diagram on GitHub
- README has working Quickstart section
- All 11 existing tests still pass
