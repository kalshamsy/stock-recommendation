"""Stock Decision Pro — bilingual analysis and evaluation workspace."""

from __future__ import annotations

import re

import streamlit as st

from src.backtest import run_swing_backtest
from src.data import (
    load_backtest_bundle,
    load_local_universe,
    load_market_bundle,
    search_securities,
)
from src.engines import analyze_investment, analyze_swing
from src.evaluation_ui import (
    et,
    render_backtest_result,
    render_evaluation_intro,
    render_session_log,
)
from src.i18n import t
from src.indicators import compute_indicators
from src.models import AnalysisResult
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
        "About": "Stock Decision Pro V1.1 — analysis and point-in-time evaluation.",
    },
)


def set_language(lang: str) -> None:
    if st.session_state.get("lang") != lang:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_history", None)
        st.session_state.pop("analysis_error", None)
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


def trading_days_label(lang: str, value: int) -> str:
    if lang == "ar" and value == 1:
        return "1 يوم تداول"
    return f"{value} {et(lang, 'trading_days')}"


def security_picker(lang: str, key_prefix: str) -> str:
    universe_records = load_local_universe().to_dict("records")
    security_options = [security_label(item) for item in universe_records]
    symbol_by_label = {
        security_label(item): str(item["symbol"]).strip().upper()
        for item in universe_records
    }
    selected_search = st.selectbox(
        t(lang, "search_label"),
        options=security_options,
        index=None,
        placeholder=t(lang, "search_placeholder"),
        help=t(lang, "search_help"),
        accept_new_options=True,
        key=f"{key_prefix}_security_search",
    )
    if not selected_search:
        return ""

    known_symbol = symbol_by_label.get(selected_search, "")
    if known_symbol:
        return known_symbol

    custom_query = str(selected_search).strip()
    candidates = search_securities(custom_query)
    if ticker_like(custom_query):
        typed_symbol = custom_query.upper()
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
    if not candidates:
        st.caption(t(lang, "no_matches"))
        return custom_query.upper() if ticker_like(custom_query) else ""
    selected = st.selectbox(
        t(lang, "select_security"),
        options=candidates,
        format_func=security_label,
        key=f"{key_prefix}_security_match",
    )
    return str(selected["symbol"]).upper()


def record_recommendation(result: AnalysisResult) -> None:
    record = {
        "symbol": result.symbol,
        "mode": result.mode,
        "as_of": result.as_of,
        "decision": result.decision,
        "score": result.score,
        "confidence": result.confidence,
        "price": round(result.current_price, 4),
        "entry_low": round(result.entry_low, 4) if result.entry_low is not None else None,
        "entry_high": round(result.entry_high, 4) if result.entry_high is not None else None,
        "stop_loss": round(result.stop_loss, 4) if result.stop_loss is not None else None,
        "target_1": round(result.target_1, 4) if result.target_1 is not None else None,
        "target_pct": result.target_pct,
        "engine_version": "1.1.0",
    }
    records = st.session_state.get("recommendation_log", [])
    identity = (record["symbol"], record["mode"], record["as_of"], record["target_pct"])
    filtered = [
        item
        for item in records
        if (item["symbol"], item["mode"], item["as_of"], item.get("target_pct")) != identity
    ]
    st.session_state.recommendation_log = [record] + filtered[:99]


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
            record_recommendation(result)
    except Exception:
        st.session_state.analysis_result = None
        st.session_state.analysis_history = None
        st.session_state.analysis_error = t(lang, "data_error_body")


def render_analysis_workspace(lang: str) -> None:
    with st.container(border=True):
        selected_symbol = security_picker(lang, "analysis")
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
        run_analysis(selected_symbol, mode, target_pct, lang)

    if st.session_state.get("analysis_error"):
        st.error(
            f"**{t(lang, 'data_error_title')}**\n\n{st.session_state.analysis_error}",
            icon="⚠️",
        )

    result = st.session_state.get("analysis_result")
    history = st.session_state.get("analysis_history")
    if result is None:
        render_empty_state(lang)
        return

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


def render_evaluation_workspace(lang: str) -> None:
    render_evaluation_intro(lang)
    st.info(et(lang, "investment_note"), icon="ℹ️")
    with st.container(border=True):
        symbol = security_picker(lang, "evaluation")
        first, second = st.columns(2)
        period_labels = {
            et(lang, "period_2y"): "2y",
            et(lang, "period_5y"): "5y",
            et(lang, "period_10y"): "10y",
        }
        with first:
            period_label = st.selectbox(
                et(lang, "period_label"),
                options=list(period_labels),
                index=1,
                key=f"backtest_period_{lang}",
            )
            target_pct = float(
                st.select_slider(
                    t(lang, "target_label"),
                    options=[3, 4, 5, 7, 10],
                    value=5,
                    format_func=lambda value: f"{value}%",
                    key="backtest_target",
                )
            )
            entry_window = int(
                st.select_slider(
                    et(lang, "entry_window_label"),
                    options=[1, 2, 3, 5],
                    value=3,
                    format_func=lambda value: trading_days_label(lang, value),
                    key="backtest_entry_window",
                )
            )
        with second:
            holding_days = int(
                st.select_slider(
                    et(lang, "holding_label"),
                    options=[5, 10, 15, 20],
                    value=10,
                    format_func=lambda value: trading_days_label(lang, value),
                    key="backtest_holding_days",
                )
            )
            slippage_pct = float(
                st.select_slider(
                    et(lang, "slippage_label"),
                    options=[0.0, 0.05, 0.10, 0.20],
                    value=0.10,
                    format_func=lambda value: f"{value:.2f}%",
                    key="backtest_slippage",
                )
            )
        evaluate_clicked = st.button(
            f"⌁  {et(lang, 'run_backtest')}",
            type="primary",
            use_container_width=True,
            key="run_backtest_button",
        )

    if evaluate_clicked:
        if not symbol:
            st.session_state.backtest_error = t(lang, "invalid_symbol")
            st.session_state.backtest_result = None
        else:
            st.session_state.backtest_error = None
            try:
                with st.spinner(et(lang, "backtest_loading")):
                    bundle = load_backtest_bundle(symbol, period_labels[period_label])
                    st.session_state.backtest_result = run_swing_backtest(
                        bundle,
                        target_pct=target_pct,
                        holding_days=holding_days,
                        entry_window=entry_window,
                        slippage_pct=slippage_pct,
                    )
            except Exception:
                st.session_state.backtest_result = None
                st.session_state.backtest_error = et(lang, "backtest_error")

    if st.session_state.get("backtest_error"):
        st.error(st.session_state.backtest_error, icon="⚠️")
    backtest_result = st.session_state.get("backtest_result")
    if backtest_result is not None:
        render_backtest_result(backtest_result, lang)

    st.divider()
    render_session_log(st.session_state.get("recommendation_log", []), lang)


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
workspace_labels = {
    et(lang, "workspace_analysis"): "analysis",
    et(lang, "workspace_evaluation"): "evaluation",
}
workspace_label = st.segmented_control(
    et(lang, "workspace_label"),
    options=list(workspace_labels),
    default=list(workspace_labels)[0],
    selection_mode="single",
    key=f"workspace_{lang}",
)
workspace = workspace_labels.get(workspace_label, "analysis")

if workspace == "evaluation":
    render_evaluation_workspace(lang)
else:
    render_analysis_workspace(lang)

render_footer(lang)
