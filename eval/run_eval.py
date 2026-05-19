"""Eval harness: compare single-agent baseline vs multi-agent system.

Usage:
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --query-limit 3
    uv run python eval/run_eval.py --add-hallucination-scores
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

EVAL_DIR = Path(__file__).parent
QUERIES_FILE = EVAL_DIR / "queries.yaml"
RESULTS_FILE = EVAL_DIR / "results.json"


def _unique_domains(notes: list) -> int:
    """Count distinct source domains across a list of ResearchNote objects."""
    domains: set[str] = set()
    for note in notes:
        try:
            parsed = urlparse(note.source_url)
            if parsed.scheme and parsed.netloc:
                domains.add(parsed.netloc.lower().removeprefix("www.").strip())
        except ValueError:
            pass
    return len(domains)


def _memo_word_count(memo) -> int:
    """Total word count across all memo section bodies."""
    if memo is None:
        return 0
    return len(" ".join(s.body for s in memo.sections).split())


def _build_baseline_graph():
    """Two-node baseline: researcher → synthesizer, no critic loop."""
    from langgraph.graph import END, START, StateGraph
    from research_analyst.agents.researcher import researcher_node
    from research_analyst.agents.synthesizer import synthesizer_node
    from research_analyst.schemas import AgentState

    builder = StateGraph(AgentState)
    builder.add_node("researcher", researcher_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", END)
    return builder.compile()


def _build_multi_graph():
    from research_analyst.graph import build_graph
    return build_graph()


def _run_once(graph, query: str) -> dict:
    from research_analyst.schemas import AgentState

    start = time.perf_counter()
    result = graph.invoke(AgentState(query=query).model_dump())
    elapsed = time.perf_counter() - start

    state = AgentState(**result)
    return {
        "source_count": len(state.notes),
        "unique_domains": _unique_domains(state.notes),
        "latency_seconds": round(elapsed, 2),
        "memo_length_words": _memo_word_count(state.final_memo),
        "hallucinated_citations": 0,
    }


def _load_queries(limit: int | None = None) -> list[str]:
    import yaml

    with open(QUERIES_FILE) as f:
        data = yaml.safe_load(f)
    queries: list[str] = data["queries"]
    return queries[:limit] if limit is not None else queries


def _avg(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs if r.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _print_table(all_results: list[dict]) -> None:
    baseline_runs = [r["baseline"] for r in all_results]
    multi_runs = [r["multi_agent"] for r in all_results]

    metrics = [
        ("Avg unique source domains", "unique_domains"),
        ("Avg memo length (words)",   "memo_length_words"),
        ("Avg latency (s)",           "latency_seconds"),
        ("Hallucinated citations",    "hallucinated_citations"),
    ]

    col_w = 32
    print("\n" + "=" * 76)
    print(f"{'Metric':<{col_w}} {'Baseline':>14} {'Multi-agent':>14} {'Delta':>10}")
    print("-" * 76)
    for label, key in metrics:
        b = _avg(baseline_runs, key)
        m = _avg(multi_runs, key)
        delta = round(m - b, 2)
        sign = "+" if delta > 0 else ""
        print(f"{label:<{col_w}} {b:>14} {m:>14} {sign+str(delta):>10}")
    print("=" * 76)


def _prompt_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("  Please enter a whole number.")


def _add_hallucination_scores(all_results: list[dict]) -> None:
    print("\nEnter hallucination scores (number of fabricated citations per run).")
    print("Review each memo in eval/results.json before scoring.\n")
    for i, entry in enumerate(all_results):
        q = entry["query"][:60]
        entry["baseline"]["hallucinated_citations"] = _prompt_int(
            f"[{i+1}] Baseline    | {q}... : "
        )
        entry["multi_agent"]["hallucinated_citations"] = _prompt_int(
            f"[{i+1}] Multi-agent | {q}... : "
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval harness")
    parser.add_argument("--query-limit", type=int, default=None,
                        help="Only run first N queries")
    parser.add_argument("--add-hallucination-scores", action="store_true",
                        help="Prompt to enter manual hallucination scores")
    args = parser.parse_args()

    queries = _load_queries(args.query_limit)

    # Resume: load any queries already completed in results.json
    existing: list[dict] = []
    if RESULTS_FILE.exists():
        try:
            existing = json.loads(RESULTS_FILE.read_text()).get("results", [])
        except (json.JSONDecodeError, KeyError):
            existing = []
    done_queries = {e["query"] for e in existing}
    remaining = [q for q in queries if q not in done_queries]

    if done_queries:
        print(f"Resuming: {len(done_queries)} already done, {len(remaining)} remaining.")
    print(f"Running eval on {len(remaining)} queries (baseline + multi-agent each)...")

    if not remaining:
        print("All queries already complete. Use --add-hallucination-scores to score them.")
        _print_table(existing)
        return

    baseline_graph = _build_baseline_graph()
    multi_graph = _build_multi_graph()

    all_results: list[dict] = list(existing)

    for i, query in enumerate(remaining, len(done_queries) + 1):
        print(f"\n[{i}/{len(queries)}] {query[:70]}")
        print("  -> baseline ...")
        baseline_metrics = _run_once(baseline_graph, query)
        print(f"     {baseline_metrics['source_count']} notes, "
              f"{baseline_metrics['unique_domains']} domains, "
              f"{baseline_metrics['latency_seconds']}s")

        print("  -> multi-agent ...")
        multi_metrics = _run_once(multi_graph, query)
        print(f"     {multi_metrics['source_count']} notes, "
              f"{multi_metrics['unique_domains']} domains, "
              f"{multi_metrics['latency_seconds']}s")

        all_results.append({
            "query": query,
            "baseline": baseline_metrics,
            "multi_agent": multi_metrics,
        })

        # Save after every query so progress is never lost
        output = {
            "model": "llama-3.3-70b-versatile",
            "query_count": len(all_results),
            "results": all_results,
        }
        RESULTS_FILE.write_text(json.dumps(output, indent=2))
        print(f"  [saved {len(all_results)}/{len(queries)} to results.json]")

    if args.add_hallucination_scores:
        _add_hallucination_scores(all_results)
        RESULTS_FILE.write_text(json.dumps({
            "model": "llama-3.3-70b-versatile",
            "query_count": len(all_results),
            "results": all_results,
        }, indent=2))

    print(f"\nResults saved to {RESULTS_FILE}")

    _print_table(all_results)


if __name__ == "__main__":
    main()
