# UI Dark Professional Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the Dark Professional (Gradient & Glow) theme to the Streamlit app using a config file + injected CSS.

**Architecture:** `.streamlit/config.toml` handles base Streamlit widget theming; a CSS block injected via `st.markdown()` in `app.py` handles gradients, glow effects, panel styling, and typography. No layout changes — columns, logic, and streaming are untouched.

**Tech Stack:** Streamlit 1.57, CSS (injected via `st.markdown(unsafe_allow_html=True)`)

---

## File Map

| File | Action |
|---|---|
| `.streamlit/config.toml` | Create — base dark widget theming |
| `app.py` | Modify — add CSS block after `st.set_page_config(...)` |

---

## Task 1: Create `.streamlit/config.toml`

**Files:**
- Create: `.streamlit/config.toml`

- [ ] **Step 1: Create `.streamlit/` directory and config file**

```toml
[theme]
base = "dark"
primaryColor = "#00d4ff"
backgroundColor = "#080c14"
secondaryBackgroundColor = "#0d1b2e"
textColor = "#c8d8e8"
```

Save to: `.streamlit/config.toml`

- [ ] **Step 2: Verify Streamlit picks it up**

```
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
uv run streamlit run app.py
```

Open http://localhost:8501. The background should be very dark navy (#080c14) and inputs should have a dark blue background. If the page is still white, the config file is not being read — check the path is exactly `.streamlit/config.toml` at the project root.

- [ ] **Step 3: Commit**

```bash
git add .streamlit/config.toml
git commit -m "feat: add dark professional Streamlit theme config"
```

---

## Task 2: Inject CSS in app.py

**Files:**
- Modify: `app.py` — add CSS block after `st.set_page_config(...)` (line 24)

- [ ] **Step 1: Add the CSS block immediately after `st.set_page_config(...)`**

Insert this block at line 26 (after the closing `)` of `st.set_page_config`):

```python
st.markdown(
    """
    <style>
    /* ── App shell ─────────────────────────────────────── */
    .stApp { background: #080c14; }
    .block-container { padding-top: 1.5rem; }

    /* ── Title gradient ─────────────────────────────────── */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
    }

    /* ── Subheaders ─────────────────────────────────────── */
    h2, h3 {
        color: #4a90a4 !important;
        letter-spacing: 0.04em;
    }

    /* ── Caption ────────────────────────────────────────── */
    [data-testid="stCaptionContainer"] p { color: #2d4a6e !important; }

    /* ── Divider ────────────────────────────────────────── */
    hr { border-color: rgba(0, 212, 255, 0.15) !important; }

    /* ── Text area ──────────────────────────────────────── */
    .stTextArea textarea {
        background: #0d1b2e !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
        color: #c8d8e8 !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(0, 212, 255, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.12) !important;
    }
    .stTextArea label p {
        color: #4a90a4 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    /* ── Primary button (Run Research) ──────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4ff, #0066cc) !important;
        border: none !important;
        box-shadow: 0 0 16px rgba(0, 212, 255, 0.35) !important;
        color: #fff !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        transition: box-shadow 0.2s ease, transform 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 28px rgba(0, 212, 255, 0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Agent activity panel (left column markdown) ─────── */
    [data-testid="column"]:first-child [data-testid="stMarkdownContainer"] > p,
    [data-testid="column"]:first-child [data-testid="stMarkdownContainer"] {
        background: #0d1b2e;
        border: 1px solid rgba(0, 212, 255, 0.12);
        border-radius: 10px;
        padding: 14px !important;
        line-height: 1.7 !important;
    }

    /* ── Memo panel (right column markdown) ──────────────── */
    [data-testid="column"]:last-child [data-testid="stMarkdownContainer"] {
        background: linear-gradient(180deg, #0d1b2e 0%, #090e18 100%);
        border: 1px solid rgba(0, 212, 255, 0.14);
        border-radius: 10px;
        padding: 20px !important;
    }
    [data-testid="column"]:last-child [data-testid="stMarkdownContainer"] h1 {
        color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important;
        font-size: 1.4rem !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        margin-bottom: 0.6rem !important;
    }
    [data-testid="column"]:last-child [data-testid="stMarkdownContainer"] blockquote {
        border-left: 3px solid #00d4ff !important;
        background: rgba(0, 212, 255, 0.05) !important;
        padding: 10px 16px !important;
        border-radius: 0 6px 6px 0 !important;
        color: #8bafd4 !important;
        font-style: italic !important;
        margin: 0.5rem 0 1.2rem 0 !important;
    }
    [data-testid="column"]:last-child [data-testid="stMarkdownContainer"] h2 {
        color: #c8d8e8 !important;
        -webkit-text-fill-color: #c8d8e8 !important;
        font-size: 1rem !important;
        border-bottom: 1px solid rgba(0, 212, 255, 0.1) !important;
        padding-bottom: 4px !important;
        margin-top: 1.2rem !important;
        letter-spacing: 0.02em !important;
    }
    [data-testid="column"]:last-child [data-testid="stMarkdownContainer"] p {
        color: #8bafd4 !important;
        line-height: 1.75 !important;
    }
    [data-testid="column"]:last-child [data-testid="stMarkdownContainer"] ol li {
        color: #5a7a9a !important;
        font-size: 0.85rem !important;
    }

    /* ── Alert / info / warning / error boxes ────────────── */
    [data-testid="stAlert"] {
        background: #0d1b2e !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
```

- [ ] **Step 2: Restart Streamlit and verify visually**

Stop any running Streamlit instance, then:

```
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
uv run streamlit run app.py
```

Open http://localhost:8501. Check:
- [ ] Page background is very dark navy (not white/grey)
- [ ] Title "Multi-Agent Research Analyst" has cyan-to-purple gradient text
- [ ] "Run Research" button is cyan with a glow
- [ ] Text area has a dark blue background with subtle cyan border
- [ ] The two placeholder boxes (Agent Activity, Research Memo) have dark blue backgrounds with subtle borders

If any check fails, inspect via browser DevTools (F12 → Inspector) to find the actual selector and adjust the CSS.

- [ ] **Step 3: Run the app end-to-end to confirm nothing broke**

Enter a query, click Run Research, let it complete. Verify:
- [ ] Streaming activity lines appear correctly styled (no broken layout)
- [ ] Final memo renders with cyan heading, blockquote, and styled sections

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: apply Dark Professional gradient & glow theme to Streamlit UI"
```

---

## Task 3: Push to HF Spaces

**Files:** None (git push only)

- [ ] **Step 1: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 2: Push to HF Spaces** (replace token)

```powershell
git remote set-url hf https://Sahil3717:YOUR_TOKEN@huggingface.co/spaces/Sahil3717/multi-agent-research-analyst
git push hf main
git remote set-url hf https://huggingface.co/spaces/Sahil3717/multi-agent-research-analyst
```

- [ ] **Step 3: Verify HF build succeeds**

Watch the build log at https://huggingface.co/spaces/Sahil3717/multi-agent-research-analyst. Build should complete in ~3 minutes. Open the live URL and confirm the dark theme is visible.

---

## Self-Review

- [x] Task 1 creates config.toml with exact hex values from the spec
- [x] Task 2 CSS targets columns by `:first-child` / `:last-child` — avoids styling ALL markdown globally
- [x] `h1` gradient uses `-webkit-text-fill-color: transparent` (correct pattern for gradient text)
- [x] `h1` inside the memo panel explicitly resets `-webkit-text-fill-color: #00d4ff` so it doesn't inherit the rainbow gradient from the title
- [x] No layout changes — column structure, streaming logic, state management untouched
- [x] Task 3 deploys the change to HF Spaces
