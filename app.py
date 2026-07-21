"""Stock Decision Pro — mobile-first bilingual Streamlit application."""

from __future__ import annotations

import re

import streamlit as st

from src.data import load_market_bundle, search_securities
from src.engines import analyze_investment, analyze_swing
from src.i18n import t
from src.indicators import compute_indicators
from src.ui import (
    build_price_chart,
    inject_styles,
    render_brand_header,
    render_breakdown,
    render_empty_state,
    render_footer,
    render_fundamental_snapshot,
    render_hero,
    render_language_gate,
    render_mode_description,
    render_reasons,
    render_result_summary,
    render_technical_snapshot,
)


st.set_page_config(
    page_title="Stock Decision Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "Stock Decision Pro V1.0 — rules-based stock research.",
    },
)


def set_language(lang: str) -> None:
    st.session_state.lang = lang
    st.session_state.language_ready = True
    st.query_params["lang"] = lang


def current_language() -> str:
    query_lang = st.query_params.get("lang")
    if query_lang in {"ar", "en"}:
        st.session_state.lang = query_lang
        st.session_state.language_ready = True
    return st.session_state.get("lang", "en")


def language_gate() -> None:
    if st.session_state.get("language_ready"):
        return
    render_language_gate()
    spacer, english_col, arabic_col, spacer_2 = st.columns([1.1, 1, 1, 1.1])
    with english_col:
        if st.button("English", use_container_width=True, type="primary", key="choose_en"):
            set_language("en")
            st.rerun()
    with arabic_col:
        if st.button("العربية", use_container_width=True, key="choose_ar"):
            set_language("ar")
            st.rerun()
    st.stop()


def ticker_like(value: str) -> bool:
    return bool(re.fullmatch(r"(?=.*[A-Za-z])[A-Za-z0-9.^=\-]{1,20}", value.strip()))


def security_label(candidate: dict[str, str]) -> str:
    exchange = f" · {candidate['exchange']}" if candidate.get("exchange") else ""
    return f"{candidate['symbol']} — {candidate['name']}{exchange}"


def run_analysis(symbol: str, mode: str, target_pct: float, lang: str) -> None:
    if not symbol:
        st.session_state.analysis_error = t(lang, "invalid_symbol")
        return
    st.session_state.analysis_error = None
    try:
        with st.spinner(t(lang, "loading")):
            bundle = load_market_bundle(symbol)
            result = analyze_swing(bundle, target_pct) if mode == "swing" else analyze_investment(bundle)
            st.session_state.analysis_result = result
            st.session_state.analysis_history = compute_indicators(bundle["history"])
            recent = st.session_state.get("recent_symbols", [])
            st.session_state.recent_symbols = [symbol] + [item for item in recent if item != symbol][:4]
    except Exception:
        st.session_state.analysis_result = None
        st.session_state.analysis_history = None
        st.session_state.analysis_error = t(lang, "data_error_body")


lang = current_language()
language_gate()
inject_styles(lang)

header_left, header_right = st.columns([8, 1.3], vertical_alignment="center")
with header_left:
    render_brand_header(lang)
with header_right:
    switch_label = "العربية" if lang == "en" else "English"
    if st.button(f"🌐 {switch_label}", use_container_width=True, key="language_switch"):
        set_language("ar" if lang == "en" else "en")
        st.rerun()

render_hero(lang)

with st.container(border=True):
    search_query = st.text_input(
        t(lang, "search_label"),
        placeholder=t(lang, "search_placeholder"),
        help=t(lang, "search_help"),
        key="search_query",
    ).strip()

    candidates = search_securities(search_query) if search_query else []
    if search_query and ticker_like(search_query):
        typed_symbol = search_query.upper()
        if not any(item["symbol"] == typed_symbol for item in candidates):
            candidates.insert(
                0,
                {
                    "symbol": typed_symbol,
                    "name": t(lang, "typed_symbol"),
                    "exchange": "",
                    "type": "EQUITY",
                    "source": "typed",
                },
            )

    selected_symbol = ""
    if candidates:
        selected = st.selectbox(
            t(lang, "select_security"),
            options=candidates,
            format_func=security_label,
            key="security_match",
        )
        selected_symbol = selected["symbol"]
    elif search_query:
        st.caption(t(lang, "no_matches"))

    mode_labels = {t(lang, "swing"): "swing", t(lang, "investment"): "investment"}
    selected_mode_label = st.radio(
        t(lang, "mode_label"),
        options=list(mode_labels),
        horizontal=True,
        key=f"mode_{lang}",
    )
    mode = mode_labels[selected_mode_label]
    render_mode_description(lang, mode)

    target_pct = 5.0
    if mode == "swing":
        target_pct = float(
            st.select_slider(
                t(lang, "target_label"),
                options=[3, 4, 5, 7, 10],
                value=5,
                format_func=lambda value: f"{value}%",
                help=t(lang, "target_help"),
                key="swing_target",
            )
        )

    analyze_clicked = st.button(
        f"↗  {t(lang, 'analyze')}",
        type="primary",
        use_container_width=True,
        key="analyze_button",
    )

if analyze_clicked:
    fallback_symbol = search_query.upper() if ticker_like(search_query) else ""
    run_analysis(selected_symbol or fallback_symbol, mode, target_pct, lang)

if st.session_state.get("analysis_error"):
    st.error(
        f"**{t(lang, 'data_error_title')}**\n\n{st.session_state.analysis_error}",
        icon="⚠️",
    )

result = st.session_state.get("analysis_result")
history = st.session_state.get("analysis_history")

if result is None:
    render_empty_state(lang)
else:
    render_result_summary(result, lang)
    st.markdown(f"### {t(lang, 'details')}")
    with st.expander(t(lang, "why_decision"), expanded=False):
        render_reasons(result, lang)
    with st.expander(t(lang, "score_breakdown"), expanded=False):
        render_breakdown(result, lang)
    with st.expander(t(lang, "technical_snapshot"), expanded=False):
        render_technical_snapshot(result, lang)
    if result.mode == "investment":
        with st.expander(t(lang, "fundamental_snapshot"), expanded=False):
            render_fundamental_snapshot(result, lang)
    with st.expander(t(lang, "chart"), expanded=False):
        if history is not None and not history.empty:
            figure = build_price_chart(history, result, lang)
            st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False, "responsive": True})
            st.caption(t(lang, "chart_caption"))
    with st.expander(t(lang, "methodology"), expanded=False):
        st.write(t(lang, "methodology_body"))

render_footer(lang)
