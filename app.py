"""Multi-Agent Research Analyst — Streamlit UI with live streaming."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research Analyst",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helper: verify required API keys are present
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ["GROQ_API_KEY", "TAVILY_API_KEY"]


def _check_env() -> list[str]:
    """Return a list of missing environment variable names."""
    return [k for k in _REQUIRED_KEYS if not os.environ.get(k)]


# ---------------------------------------------------------------------------
# Async streaming function
# ---------------------------------------------------------------------------


async def _stream_research(
    query: str,
    activity_lines: list[str],
    activity_placeholder: Any,
) -> Any:
    """Run the graph with astream_events and update activity_lines in-place.

    Returns the final AgentState (or None on error).
    """
    from research_analyst.graph import build_graph
    from research_analyst.schemas import AgentState

    graph = build_graph()
    initial_state = AgentState(query=query)

    final_state: AgentState | None = None

    async for event in graph.astream_events(
        initial_state.model_dump(),
        version="v2",
    ):
        event_name: str = event.get("event", "")
        metadata: dict = event.get("metadata", {})
        node: str = metadata.get("langgraph_node", "")

        if not node:
            continue

        if event_name == "on_chain_start":
            if node == "researcher":
                # Determine the round number from data if available
                data = event.get("data", {})
                input_data = data.get("input", {})
                if isinstance(input_data, dict):
                    round_n = input_data.get("revision_round", 0) + 1
                else:
                    # Fallback: count how many researcher lines we already have
                    round_n = sum(1 for ln in activity_lines if "Researcher" in ln) + 1
                activity_lines.append(f"🔬 Researcher searching... (Round {round_n})")

            elif node == "critic":
                activity_lines.append("🔍 Critic reviewing notes...")

            elif node == "increment_revision":
                data = event.get("data", {})
                input_data = data.get("input", {})
                if isinstance(input_data, dict):
                    next_round = input_data.get("revision_round", 0) + 1
                else:
                    next_round = sum(1 for ln in activity_lines if "revision round" in ln.lower()) + 1
                activity_lines.append(f"🔄 Starting revision round {next_round}...")

            elif node == "synthesizer":
                activity_lines.append("✍️ Synthesizer writing memo...")

        elif event_name == "on_chain_end":
            data = event.get("data", {})
            output = data.get("output", {})

            if node == "researcher":
                if isinstance(output, dict):
                    notes_count = len(output.get("notes", []))
                    activity_lines.append(f"   → {notes_count} notes collected so far")

            elif node == "critic":
                if isinstance(output, dict):
                    critiques = output.get("critiques", [])
                    if critiques:
                        last = critiques[-1]
                        # last may be a Critique object or a dict
                        if isinstance(last, dict):
                            verdict = last.get("verdict", "")
                            reasoning = last.get("reasoning", "")
                        else:
                            verdict = getattr(last, "verdict", "")
                            reasoning = getattr(last, "reasoning", "")

                        if verdict == "approve":
                            activity_lines.append("   → ✅ Approved")
                        else:
                            snippet = reasoning[:120] + "..." if len(reasoning) > 120 else reasoning
                            activity_lines.append(f"   → ⚠️ Requesting revision: {snippet}")

            elif node == "synthesizer":
                activity_lines.append("   → ✅ Memo complete")
                # Capture final state
                if isinstance(output, dict):
                    from research_analyst.schemas import AgentState as _AS

                    try:
                        final_state = _AS(**output)
                    except Exception:
                        final_state = None

        # Re-render the activity panel after every relevant event
        activity_placeholder.markdown("\n\n".join(activity_lines))

    # If we didn't capture via synthesizer end event, try one last invoke result
    if final_state is None:
        # Attempt to re-run synchronously to get the state (fallback)
        pass

    return final_state


# ---------------------------------------------------------------------------
# Memo renderer
# ---------------------------------------------------------------------------


def _render_memo(memo: Any) -> str:
    """Render a Memo object as a markdown string."""
    lines: list[str] = []

    lines.append(f"# {memo.title}\n")
    lines.append(f"> {memo.summary}\n")

    for section in memo.sections:
        lines.append(f"## {section.heading}\n")
        lines.append(f"{section.body}\n")

    if memo.sources:
        lines.append("## Sources\n")
        for i, src in enumerate(memo.sources, 1):
            lines.append(f"{i}. {src}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------

st.title("Multi-Agent Research Analyst")
st.caption("A team of LangGraph agents that researches, critiques, and synthesises a cited memo.")

# Show missing-key banner at the top if needed
missing_keys = _check_env()
if missing_keys:
    st.error(
        f"**Missing API keys:** {', '.join(missing_keys)}\n\n"
        "Create a `.env` file in the project root with:\n"
        "```\nGROQ_API_KEY=...\nTAVILY_API_KEY=...\n```\n"
        "then restart the app.",
        icon="🚨",
    )

st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Research Topic")
    query = st.text_area(
        "Enter your research question",
        placeholder="e.g. How does ASML's monopoly affect chip prices?",
        height=120,
    )
    run_button = st.button(
        "Run Research",
        type="primary",
        use_container_width=True,
        disabled=bool(missing_keys),
    )

    st.divider()
    st.subheader("Agent Activity")
    activity_placeholder = st.empty()
    activity_placeholder.info("Agent activity will stream here once you click **Run Research**.")

with col_right:
    st.subheader("Research Memo")
    memo_placeholder = st.empty()
    memo_placeholder.info(
        "The final cited memo will appear here after the agents complete their work."
    )

# ---------------------------------------------------------------------------
# Run on button click
# ---------------------------------------------------------------------------

if run_button:
    if not query.strip():
        st.error("Please enter a research question.")
    elif missing_keys:
        st.error(f"Cannot run — missing API keys: {', '.join(missing_keys)}")
    else:
        activity_lines: list[str] = []
        activity_placeholder.markdown("*Starting agents...*")

        final_state = None
        try:
            # Run the async streaming function from sync Streamlit context.
            # asyncio.run() creates a fresh event loop, which avoids conflicts
            # with any loop Streamlit may have started internally.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Streamlit is running inside an existing event loop (e.g. Tornado).
                    # Use a new thread-based event loop to avoid "cannot run nested" error.
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(
                            asyncio.run,
                            _stream_research(query, activity_lines, activity_placeholder),
                        )
                        final_state = future.result()
                else:
                    final_state = loop.run_until_complete(
                        _stream_research(query, activity_lines, activity_placeholder)
                    )
            except RuntimeError:
                # No event loop exists yet — use asyncio.run()
                final_state = asyncio.run(
                    _stream_research(query, activity_lines, activity_placeholder)
                )

        except Exception as exc:
            st.error(f"**Error during research:** {exc}")
            activity_placeholder.error(f"Pipeline failed: {exc}")

        # Render final memo
        if final_state is not None and final_state.final_memo is not None:
            memo_md = _render_memo(final_state.final_memo)
            memo_placeholder.markdown(memo_md)
        else:
            if final_state is not None:
                memo_placeholder.warning("Research completed but no memo was produced.")
            # If an exception was raised, the error banner above is sufficient
