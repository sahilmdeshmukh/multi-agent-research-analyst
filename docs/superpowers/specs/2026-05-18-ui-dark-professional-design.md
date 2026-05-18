# UI Dark Professional Design Spec

**Date:** 2026-05-18  
**Status:** Approved

## Goal

Apply a Dark Professional (Gradient & Glow) visual theme to the Streamlit app. No layout changes — only colours, gradients, borders, and shadows.

## Reference mockup

Variant B from brainstorm: deep navy/charcoal background, cyan (#00d4ff) primary accent, subtle gradients on panels and header, cyan glow on active agent activity lines.

---

## Section 1: `.streamlit/config.toml`

Create this file to set base Streamlit widget theming:

```toml
[theme]
base = "dark"
primaryColor = "#00d4ff"
backgroundColor = "#080c14"
secondaryBackgroundColor = "#0d1b2e"
textColor = "#c8d8e8"
```

- `base = "dark"` — makes all native widgets (dropdowns, checkboxes, etc.) inherit dark styling
- `primaryColor` — cyan; used by Streamlit for button focus rings, active states
- `backgroundColor` — main page background (very dark navy)
- `secondaryBackgroundColor` — sidebar, cards, input backgrounds
- `textColor` — default body text (light blue-grey)

---

## Section 2: Custom CSS block in `app.py`

Inject immediately after `st.set_page_config(...)` via:

```python
st.markdown("""<style>...<style>""", unsafe_allow_html=True)
```

### What the CSS covers

**App header / title**
- `.stApp header` background: matches `#080c14`
- `h1` (app title): gradient text `linear-gradient(90deg, #00d4ff, #7c3aed)` via `-webkit-background-clip: text`
- Caption text: muted `#2d4a6e`

**Run button**
- Background: `linear-gradient(135deg, #00d4ff, #0066cc)`
- Box-shadow: `0 0 16px rgba(0, 212, 255, 0.4)` glow
- Text: white, bold
- Hover: glow intensifies

**Text area (topic input)**
- Border: `1px solid rgba(0, 212, 255, 0.2)`
- Background: `#0d1b2e`
- Focus border: `rgba(0, 212, 255, 0.6)` with subtle glow

**Agent activity panel** (the `st.empty()` markdown area)
- Container background: `#0d1b2e`
- Border: `1px solid rgba(0, 212, 255, 0.1)`
- Border-radius: `8px`, padding: `12px`

**Memo panel** (right column `st.empty()` markdown)
- Background: `linear-gradient(180deg, #0d1b2e 0%, #080c14 100%)`
- Border: `1px solid rgba(0, 212, 255, 0.15)`
- Border-radius: `8px`, padding: `16px`
- `h1` inside memo: cyan `#00d4ff`
- `blockquote` (summary): left border `2px solid #00d4ff`, bg `rgba(0,212,255,0.05)`, italic text

**Divider**
- `hr`: `border-color: rgba(0, 212, 255, 0.15)`

**Error/warning banners**
- `.stAlert`: dark background `#0d1b2e`, border `1px solid rgba(0,212,255,0.2)`

---

## Section 3: No layout changes

- Column ratio stays `[1, 2]`
- All `st.empty()` placeholders, streaming logic, and state management are untouched
- Only the CSS block and config file are added

---

## Files changed

| File | Action |
|---|---|
| `.streamlit/config.toml` | Create |
| `app.py` | Add CSS block after `st.set_page_config(...)` |

## Success criteria

- App loads with dark navy background and cyan accents
- Run button has gradient + glow
- Memo panel has gradient background and styled blockquote
- No existing functionality broken
- Looks like the Variant B mockup
