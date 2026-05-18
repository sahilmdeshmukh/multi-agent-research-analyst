import streamlit as st

st.set_page_config(
    page_title="Multi-Agent Research Analyst",
    page_icon="🔬",
    layout="wide",
)

st.title("Multi-Agent Research Analyst")
st.caption("A team of LangGraph agents that researches, critiques, and synthesizes a cited memo.")

st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Research Topic")
    query = st.text_area(
        "Enter your research question",
        placeholder="e.g. How does ASML's monopoly affect chip prices?",
        height=120,
    )
    run_button = st.button("Run Research", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Agent Activity")
    activity_placeholder = st.empty()
    activity_placeholder.info("Agent activity will stream here once you click Run Research.")

with col_right:
    st.subheader("Research Memo")
    memo_placeholder = st.empty()
    memo_placeholder.info("The final cited memo will appear here after the agents complete their work.")

if run_button:
    if not query.strip():
        st.error("Please enter a research question.")
    else:
        activity_placeholder.warning("🚧 Agent pipeline not yet implemented. Check back after Day 3!")
        memo_placeholder.warning("🚧 Coming soon.")
