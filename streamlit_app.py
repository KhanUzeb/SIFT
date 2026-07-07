"""
Minimal single-file Streamlit frontend for the AI Research Agent.

Run:
    uvicorn app.main:app --reload
    streamlit run streamlit.py

Expected backend endpoints:
    GET  /notes
    POST /research
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
APP_DIR = Path(__file__).resolve().parent
FAVICON_PATH = APP_DIR / "fvc.jpg"


def _html_text(value: str) -> str:
    return escape(value).replace("\n", "<br>")


def _shorten(text: str, limit: int = 80) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_count(n: int, singular: str, plural: str | None = None) -> str:
    word = singular if n == 1 else (plural or singular + "s")
    return f"{n} {word}"


def _favicon_icon() -> str:
    if FAVICON_PATH.exists():
        try:
            return str(FAVICON_PATH)
        except Exception:
            pass
    return "SIFT"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_notes() -> list[dict[str, Any]]:
    resp = requests.get(f"{API_BASE}/notes", timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else []


def post_research(query: str) -> dict[str, Any]:
    resp = requests.post(f"{API_BASE}/research", json={"query": query}, timeout=180)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


def check_backend() -> bool:
    try:
        # Pings the base URL to verify connection status
        requests.get(API_BASE, timeout=3)
        return True
    except requests.RequestException:
        return False


@st.cache_data
def _bg_image_data() -> str:
    path = APP_DIR / "Bg.jpg"
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{data}"


def inject_styles() -> None:
    bg_data = _bg_image_data()
    bg_css = (
        f"""
          .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background-image: url({bg_data});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            opacity: 0.12;
            z-index: -1;
            pointer-events: none;
          }}
        """
        if bg_data
        else ""
    )
    st.markdown(
        f"""
        <style>
          :root {{
            --bg: #fafafa;
            --card: #ffffff;
            --text: #111111;
            --muted: #666666;
            --line: #eaeaea;
            --accent: #ff365c;
            --accent-soft: rgba(255, 54, 92, 0.08);
            --radius: 14px;
          }}

          .stApp {{
            background: {"transparent" if bg_data else "var(--bg)"};
            color: var(--text);
          }}

          {bg_css}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
          [data-testid="stAppViewContainer"],
          [data-testid="stHeader"],
          [data-testid="stToolbar"] {
            background: transparent;
          }

          .block-container {
            max-width: 960px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
          }

          .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
          }

          .brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
          }

          .brand img {
            width: 34px;
            height: 34px;
            object-fit: cover;
            border-radius: 10px;
            border: 1px solid var(--line);
            background: #fff;
          }

          .brand-title {
            font-size: 1.1rem;
            font-weight: 700;
            line-height: 1.05;
            margin: 0;
          }

          .brand-sub {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.12rem;
          }

          .page-title {
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.02;
            letter-spacing: -0.03em;
            font-weight: 750;
            margin: 0.15rem 0 0.35rem 0;
          }

          .page-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
            margin: 0 0 1rem 0;
            max-width: 62ch;
          }

          .card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: var(--radius);
          }

          .input-card {
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.03);
          }

          .result-card {
            padding: 1rem;
          }

          .label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent);
            font-weight: 800;
            margin-bottom: 0.5rem;
          }

          .answer {
            color: var(--text);
            line-height: 1.75;
            font-size: 1rem;
          }

          .meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.85rem;
          }

          .meta {
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            background: #fff;
            color: #444;
            font-size: 0.84rem;
          }

          .source-stack {
            display: grid;
            gap: 0.6rem;
          }

          .source-item {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            background: #fff;
          }

          .source-item a {
            color: var(--text);
            font-weight: 650;
            text-decoration: none;
          }

          .source-item a:hover {
            color: var(--accent);
          }

          .source-url {
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 0.2rem;
            word-break: break-word;
          }

          .empty {
            border: 1px dashed var(--line);
            background: #fff;
            border-radius: var(--radius);
            padding: 1rem;
            color: var(--muted);
          }

          .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border: 1px solid var(--line);
            background: #fff;
            color: #444;
            border-radius: 999px;
            padding: 0.36rem 0.72rem;
            font-size: 0.84rem;
            margin-bottom: 0.9rem;
          }

          .stButton > button,
          [data-testid="stFormSubmitButton"] button {
            border: 1px solid var(--line);
            background: var(--card);
            color: var(--text);
            border-radius: 999px;
            padding: 0.52rem 0.92rem;
            font-weight: 650;
            box-shadow: none;
          }

          .stButton > button:hover,
          [data-testid="stFormSubmitButton"] button:hover {
            border-color: rgba(255, 54, 92, 0.25);
            background: var(--accent-soft);
            color: var(--text);
          }

          [data-testid="stTextInputRootElement"] > div {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: #fff;
          }

          [data-testid="stTextInputRootElement"] input {
            background: transparent;
            color: var(--text);
          }

          [data-testid="stTextInputRootElement"] input::placeholder {
            color: #9a9a9a;
          }

          [data-testid="stTextInputRootElement"] > div:focus-within {
            border-color: rgba(255, 54, 92, 0.35);
            box-shadow: 0 0 0 1px rgba(255, 54, 92, 0.08);
          }

          [data-testid="stSidebar"] {
            background: #fff;
            border-right: 1px solid var(--line);
          }

          [data-testid="stSidebar"] * {
            color: var(--text);
          }

          [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #fff;
          }

          code {
            background: #f5f5f5;
            color: #111;
          }

          a {
            color: var(--accent);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    cols = st.columns([1, 7, 2], vertical_alignment="center")
    with cols[0]:
        if FAVICON_PATH.exists():
            st.image(str(FAVICON_PATH), width=34)
        else:
            st.markdown(
                '<div style="width:34px;height:34px;border:1px solid #eaeaea;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#fff;font-weight:700;">S</div>',
                unsafe_allow_html=True,
            )
    with cols[1]:
        st.markdown('<div class="brand-title">SIFT</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">AI Agent For All Your Research Needs</div>', unsafe_allow_html=True)


def render_sidebar() -> list[dict[str, Any]]:
    with st.sidebar:
        st.markdown("### History")

        if st.button("Refresh", use_container_width=True):
            fetch_notes.clear()
            st.rerun()
        st.markdown("### Recent runs")
        try:
            notes = fetch_notes()
        except requests.RequestException:
            notes = []
            st.caption("History unavailable until the backend responds.")

        if not notes:
            st.caption("No saved research yet.")
        else:
            for note in reversed(notes[-5:]):
                q = str(note.get("query", "")).strip()
                s = str(note.get("summary", "")).strip()
                sources = note.get("sources", []) or []
                with st.expander(_shorten(q, 58)):
                    st.markdown(f"**Question**\n\n{escape(q)}", unsafe_allow_html=True)
                    st.markdown(f"**Summary**\n\n{escape(s)}", unsafe_allow_html=True)
                    st.caption(_fmt_count(len(sources), "source"))
                    for src in sources:
                        title = escape(str(src.get("title", "Source")))
                        url = escape(str(src.get("url", "")), quote=True)
                        st.markdown(f"- [{title}]({url})")

        st.markdown("### Notes")
        st.caption("Keep prompts focused: topic, comparison, timeframe, or outcome.")

    return notes if "notes" in locals() else []


def render_search_box(disabled: bool = False) -> str:
    st.markdown('<div class="card input-card">', unsafe_allow_html=True)
    with st.form("research_form", clear_on_submit=False):
        query = st.text_input(
            "Ask a question",
            placeholder="e.g. compare free LLM APIs for a small research app",
            key="research_query",
            label_visibility="collapsed",
            disabled=disabled,
        )
        submitted = st.form_submit_button("Research", disabled=disabled)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        return (query or "").strip()
    return ""


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty">
          Type a focused question and run research. Good prompts include a topic, a comparison, a timeframe, or a specific outcome.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: dict[str, Any]) -> None:
    query = str(result.get("query", "")).strip()
    answer = str(result.get("answer", "")).strip()
    sources = result.get("sources", []) or []
    steps_taken = int(result.get("steps_taken", 0) or 0)
    note_id = str(result.get("note_id", "")).strip()

    st.markdown("### Latest answer")
    st.markdown('<div class="card result-card">', unsafe_allow_html=True)
    st.markdown("**Answer**")
    st.markdown(answer if answer else "_No answer returned._")
    st.markdown(
        f"""
        <div class="meta-row">
          <span class="meta">{_fmt_count(steps_taken, "step")} taken</span>
          <span class="meta">{_fmt_count(len(sources), "source")} kept</span>
          <span class="meta">Note: {escape(note_id or "—")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if query:
        st.download_button(
            "Download answer as markdown",
            data=f"# Question\n\n{query}\n\n# Answer\n\n{answer}\n",
            file_name="answer.md",
            mime="text/markdown",
            use_container_width=False,
        )

    st.markdown("### Sources")
    if sources:
        for idx, src in enumerate(sources, 1):
            title = escape(str(src.get("title", f"Source {idx}")))
            url = escape(str(src.get("url", "")).strip(), quote=True)
            st.markdown(
                f"""
                <div class="source-item">
                  <a href="{url}" target="_blank" rel="noreferrer">{idx}. {title}</a>
                  <div class="source-url">{escape(str(src.get("url", "")))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("The backend returned an answer, but no readable sources were preserved for this run.")


def render_history_tab(notes: list[dict[str, Any]]) -> None:
    st.markdown("### History")
    if not notes:
        st.caption("No saved research yet.")
        return

    for note in reversed(notes[-8:]):
        q = str(note.get("query", "")).strip()
        summary = str(note.get("summary", "")).strip()
        sources = note.get("sources", []) or []

        with st.expander(_shorten(q, 84)):
            st.markdown("**Question**")
            st.markdown(q or "_—_")
            st.markdown("**Summary**")
            st.markdown(summary or "_—_")
            st.caption(_fmt_count(len(sources), "source"))
            for src in sources:
                title = escape(str(src.get("title", "Source")))
                url = escape(str(src.get("url", "")), quote=True)
                st.markdown(f"- [{title}]({url})")


def main() -> None:
    st.set_page_config(
        page_title="SIFT",
        page_icon=_favicon_icon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    render_header()

    # Verify if the backend service is up and active
    backend_active = check_backend()
    if not backend_active:
        st.error(
            "⚠️ **Backend Offline:** Cannot establish connection to the AI Research API at "
            f"`{API_BASE}`. Please ensure your server is running via `uvicorn app.main:app --reload`."
        )

    st.markdown(
        '<div class="page-title">Research anything.</div>'
        '<div class="page-copy">A compact LLM based agent for asking questions, getting grounded answers, and keeping sources visible.</div>',
        unsafe_allow_html=True,
    )

    notes = render_sidebar()

    # Pass the backend status to disable controls if it's down
    query = render_search_box(disabled=not backend_active)

    if query and backend_active:
        cleaned_query = query.strip()
        if len(cleaned_query) < 3:
            st.error("Please enter a fuller question.")
        else:
            with st.spinner("Researching..."):
                try:
                    result = post_research(cleaned_query)
                    fetch_notes.clear()
                    st.session_state["last_result"] = result
                    st.session_state["last_query"] = cleaned_query
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the backend. Start it with `uvicorn app.main:app --reload`.")
                except requests.exceptions.Timeout:
                    st.error("The backend took too long to respond.")
                except requests.exceptions.HTTPError as e:
                    detail = ""
                    try:
                        payload = e.response.json()
                        if isinstance(payload, dict) and payload.get("detail"):
                            detail = f"\n\n{payload['detail']}"
                    except ValueError:
                        pass
                    st.error(f"Research service error: {e}.{detail}")
                else:
                    st.success("Done.")

    last_result = st.session_state.get("last_result")
    if isinstance(last_result, dict) and last_result:
        render_result(last_result)
    else:
        render_empty_state()

    st.markdown("---")
    tab1, tab2 = st.tabs(["Latest answer", "History"])
    with tab1:
        if isinstance(last_result, dict) and last_result:
            st.markdown("This is the latest stored result from your current session.")
        else:
            st.caption("Run a query to see the latest answer here.")
    with tab2:
        render_history_tab(notes)


if __name__ == "__main__":
    main()