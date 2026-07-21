"""Reusable Streamlit presentation components."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .i18n import explain, t
from .models import AnalysisResult, Factor


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def inject_styles(lang: str) -> None:
    css = (PROJECT_ROOT / "assets" / "style.css").read_text(encoding="utf-8")
    direction = "rtl" if lang == "ar" else "ltr"
    align = "right" if lang == "ar" else "left"
    st.markdown(
        f"<style>:root{{--app-direction:{direction};--app-align:{align};}}{css}</style>",
        unsafe_allow_html=True,
    )


def render_language_gate() -> None:
    inject_styles("en")
    st.markdown(
        """
        <div class="language-shell">
          <div class="brand-mark">SD</div>
          <div class="eyebrow">STOCK DECISION PRO</div>
          <h1>Choose your language<br><span>اختر لغتك</span></h1>
          <p>Select your preferred language to continue.<br>اختر اللغة المفضلة للمتابعة.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header(lang: str) -> None:
    st.markdown(
        f"""
        <div class="brand-lockup">
          <div class="brand-mark small">SD</div>
          <div>
            <div class="brand-name">Stock Decision <span>Pro</span></div>
            <div class="brand-tagline">{escape(t(lang, 'tagline'))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(lang: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-copy">
            <div class="eyebrow">RULES-BASED STOCK RESEARCH</div>
            <h1>{escape(t(lang, 'hero_title'))}</h1>
            <p>{escape(t(lang, 'hero_subtitle'))}</p>
            <div class="hero-badges">
              <span><i class="live-dot"></i>{escape(t(lang, 'live_data'))}</span>
              <span>✓ {escape(t(lang, 'research_tool'))}</span>
            </div>
          </div>
          <div class="hero-visual" aria-hidden="true">
            <div class="visual-grid"></div>
            <svg viewBox="0 0 420 190" preserveAspectRatio="none">
              <defs>
                <linearGradient id="lineGradient" x1="0" x2="1">
                  <stop offset="0" stop-color="#38bdf8"/>
                  <stop offset="1" stop-color="#34d399"/>
                </linearGradient>
                <linearGradient id="fillGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stop-color="#34d399" stop-opacity=".28"/>
                  <stop offset="1" stop-color="#34d399" stop-opacity="0"/>
                </linearGradient>
              </defs>
              <path d="M0 152 L35 138 L70 147 L106 112 L142 120 L177 84 L212 96 L248 59 L282 68 L318 36 L351 53 L386 22 L420 29 L420 190 L0 190 Z" fill="url(#fillGradient)"/>
              <path d="M0 152 L35 138 L70 147 L106 112 L142 120 L177 84 L212 96 L248 59 L282 68 L318 36 L351 53 L386 22 L420 29" fill="none" stroke="url(#lineGradient)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div class="floating-score"><small>SCORE</small><strong>84</strong><span>/ 100</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_search_panel_start() -> None:
    st.markdown('<div class="panel-heading"><span>01</span></div>', unsafe_allow_html=True)


def render_mode_description(lang: str, mode: str) -> None:
    key = "swing_desc" if mode == "swing" else "investment_desc"
    st.markdown(f'<p class="field-note">{escape(t(lang, key))}</p>', unsafe_allow_html=True)


def render_empty_state(lang: str) -> None:
    st.markdown(
        f"""
        <section class="welcome-card">
          <div class="welcome-icon">⌁</div>
          <div>
            <h3>{escape(t(lang, 'welcome_title'))}</h3>
            <p>{escape(t(lang, 'welcome_body'))}</p>
          </div>
        </section>
        <div class="feature-grid">
          <article><span>01</span><h4>{escape(t(lang, 'feature_1_title'))}</h4><p>{escape(t(lang, 'feature_1_body'))}</p></article>
          <article><span>02</span><h4>{escape(t(lang, 'feature_2_title'))}</h4><p>{escape(t(lang, 'feature_2_body'))}</p></article>
          <article><span>03</span><h4>{escape(t(lang, 'feature_3_title'))}</h4><p>{escape(t(lang, 'feature_3_body'))}</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tone(decision: str) -> tuple[str, str]:
    if decision == "ENTER":
        return "positive", "●"
    if decision == "GRADUAL":
        return "teal", "◐"
    if decision == "WAIT":
        return "warning", "◷"
    return "negative", "×"


def _price(value: float | None, currency: str) -> str:
    if value is None:
        return "—"
    decimals = 3 if value < 1 else 2
    return f"{currency} {value:,.{decimals}f}"


def _compact(value: float | None, currency: str = "") -> str:
    if value is None:
        return "—"
    absolute = abs(value)
    if absolute >= 1_000_000_000_000:
        text = f"{value / 1_000_000_000_000:.1f}T"
    elif absolute >= 1_000_000_000:
        text = f"{value / 1_000_000_000:.1f}B"
    elif absolute >= 1_000_000:
        text = f"{value / 1_000_000:.1f}M"
    else:
        text = f"{value:,.0f}"
    return f"{currency} {text}".strip()


def _quick_text(lang: str, decision: str) -> str:
    key = {
        "ENTER": "quick_enter",
        "GRADUAL": "quick_gradual",
        "WAIT": "quick_wait",
        "AVOID": "quick_avoid",
    }[decision]
    return t(lang, key)


def render_result_summary(result: AnalysisResult, lang: str) -> None:
    tone, icon = _tone(result.decision)
    change = result.price_change_pct
    change_text = "—" if change is None else f"{change:+.2f}%"
    change_class = "up" if change is not None and change >= 0 else "down"
    entry = (
        f"{result.entry_low:,.2f} – {result.entry_high:,.2f}"
        if result.entry_low is not None and result.entry_high is not None
        else "—"
    )
    target_1_label = "target_1" if result.mode == "swing" else "fair_value"
    subtitle_parts = [part for part in [result.exchange, result.sector] if part]
    subtitle = " · ".join(subtitle_parts)

    st.markdown(
        f"""
        <section class="result-shell {tone}">
          <div class="security-row">
            <div>
              <div class="symbol-line"><span>{escape(result.symbol)}</span><h2>{escape(result.company_name)}</h2></div>
              <p>{escape(subtitle)} &nbsp;·&nbsp; {escape(t(lang, 'as_of'))} {escape(result.as_of)}</p>
            </div>
            <div class="price-block"><strong>{escape(_price(result.current_price, result.currency))}</strong><span class="{change_class}">{escape(change_text)} {escape(t(lang, 'today'))}</span></div>
          </div>
          <div class="decision-grid">
            <div class="decision-copy">
              <div class="decision-label">{escape(t(lang, 'decision'))}</div>
              <div class="decision-value"><i>{icon}</i>{escape(t(lang, result.decision))}</div>
              <p><strong>{escape(t(lang, 'quick_take'))}:</strong> {escape(_quick_text(lang, result.decision))}</p>
              <div class="confidence-pill">{escape(t(lang, 'confidence'))}: <b>{escape(t(lang, result.confidence))}</b> · {escape(t(lang, 'data_coverage'))}: {result.data_completeness}%</div>
            </div>
            <div class="score-ring" style="--score:{result.score}; --score-color:var(--{tone})">
              <div><strong>{result.score}</strong><span>/100</span><small>{escape(t(lang, 'score'))}</small></div>
            </div>
          </div>
          <div class="metric-grid">
            <article><small>{escape(t(lang, 'entry_zone'))}</small><strong>{escape(entry)}</strong><span>{escape(result.currency)}</span></article>
            <article><small>{escape(t(lang, 'stop_loss'))}</small><strong>{escape(_price(result.stop_loss, result.currency))}</strong><span>{escape(t(lang, 'risk_control'))}</span></article>
            <article><small>{escape(t(lang, target_1_label))}</small><strong>{escape(_price(result.target_1, result.currency))}</strong><span>{f'{result.target_pct:.0f}%' if result.target_pct else escape(t(lang, 'model_reference'))}</span></article>
            <article><small>{escape(t(lang, 'target_2'))}</small><strong>{escape(_price(result.target_2, result.currency))}</strong><span>{f'R:R {result.risk_reward:.1f}:1' if result.risk_reward else escape(t(lang, 'secondary_reference'))}</span></article>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_factor_group(title: str, factors: list[Factor], kind: str, lang: str) -> None:
    st.markdown(f'<h4 class="detail-subtitle">{escape(title)}</h4>', unsafe_allow_html=True)
    if not factors:
        st.markdown('<div class="empty-factor">—</div>', unsafe_allow_html=True)
        return
    for item in factors[:7]:
        icon = "✓" if kind == "positive" else "!" if kind == "risk" else "×"
        st.markdown(
            f'<div class="factor-row {kind}"><i>{icon}</i><span>{escape(explain(lang, item))}</span></div>',
            unsafe_allow_html=True,
        )


def render_reasons(result: AnalysisResult, lang: str) -> None:
    left, right = st.columns(2)
    with left:
        _render_factor_group(t(lang, "strengths"), result.positives, "positive", lang)
    with right:
        _render_factor_group(t(lang, "risks"), result.risks, "risk", lang)
    if result.blockers:
        st.markdown('<div class="blocker-box">', unsafe_allow_html=True)
        _render_factor_group(t(lang, "hard_blocks"), result.blockers, "blocker", lang)
        st.markdown("</div>", unsafe_allow_html=True)


def render_breakdown(result: AnalysisResult, lang: str) -> None:
    max_weights = {
        "trend": 25,
        "momentum": 15,
        "volume_liquidity": 15,
        "entry_quality": 15,
        "room_to_target": 15,
        "risk_reward": 15,
        "growth": 20,
        "profitability": 20,
        "financial_health": 15,
        "valuation": 20,
        "quality": 15,
        "technical_timing": 10,
    }
    rows = []
    for key, value in result.breakdown.items():
        maximum = max_weights.get(key, 100)
        width = _clamp_css(value / maximum * 100 if maximum else 0)
        rows.append(
            f'<div class="score-row"><div><span>{escape(t(lang, key))}</span><b>{value:g}/{maximum}</b></div><div class="score-track"><i style="width:{width:.1f}%"></i></div></div>'
        )
    st.markdown('<div class="breakdown-card">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def _clamp_css(value: float) -> float:
    return max(0, min(100, value))


def render_technical_snapshot(result: AnalysisResult, lang: str) -> None:
    indicators = result.indicators
    metrics = [
        (t(lang, "rsi"), f"{indicators.get('rsi'):.1f}" if indicators.get("rsi") is not None else "—"),
        (
            t(lang, "volume_ratio"),
            f"{indicators.get('volume_ratio'):.2f}×" if indicators.get("volume_ratio") is not None else "—",
        ),
        (t(lang, "sma_50"), _price(indicators.get("sma50"), result.currency)),
        (t(lang, "sma_200"), _price(indicators.get("sma200"), result.currency)),
        (t(lang, "support"), _price(result.support, result.currency)),
        (t(lang, "resistance"), _price(result.resistance, result.currency)),
        (
            t(lang, "market_trend"),
            t(lang, indicators.get("market_trend")) if indicators.get("market_trend") else "—",
        ),
    ]
    html = "".join(
        f'<article><small>{escape(label)}</small><strong>{escape(value)}</strong></article>' for label, value in metrics
    )
    st.markdown(f'<div class="snapshot-grid">{html}</div>', unsafe_allow_html=True)


def render_fundamental_snapshot(result: AnalysisResult, lang: str) -> None:
    f = result.fundamentals
    metrics = [
        (t(lang, "market_cap"), _compact(f.get("market_cap"), result.currency)),
        (t(lang, "pe"), f"{f.get('trailing_pe'):.1f}" if f.get("trailing_pe") is not None else "—"),
        (t(lang, "forward_pe"), f"{f.get('forward_pe'):.1f}" if f.get("forward_pe") is not None else "—"),
        (t(lang, "revenue_growth"), _percentage(f.get("revenue_growth"))),
        (t(lang, "earnings_growth"), _percentage(f.get("earnings_growth"))),
        (t(lang, "profit_margin"), _percentage(f.get("profit_margin"))),
        (t(lang, "debt_equity"), f"{f.get('debt_equity'):.0f}%" if f.get("debt_equity") is not None else "—"),
        (t(lang, "roe"), _percentage(f.get("roe"))),
        (t(lang, "free_cash_flow"), _compact(f.get("free_cash_flow"), result.currency)),
    ]
    html = "".join(
        f'<article><small>{escape(label)}</small><strong>{escape(value)}</strong></article>' for label, value in metrics
    )
    st.markdown(f'<div class="snapshot-grid">{html}</div>', unsafe_allow_html=True)


def _percentage(value: float | None) -> str:
    return f"{value:+.1f}%" if value is not None else "—"


def build_price_chart(history: pd.DataFrame, result: AnalysisResult, lang: str) -> go.Figure:
    data = history.tail(180).copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
    )
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing_line_color="#34d399",
            decreasing_line_color="#fb7185",
            increasing_fillcolor="#34d399",
            decreasing_fillcolor="#fb7185",
            name=result.symbol,
        ),
        row=1,
        col=1,
    )
    for column, color, label in (
        ("EMA20", "#38bdf8", "EMA 20"),
        ("SMA50", "#a78bfa", "SMA 50"),
        ("SMA200", "#fbbf24", "SMA 200"),
    ):
        if column in data:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[column], name=label, line={"color": color, "width": 1.5}),
                row=1,
                col=1,
            )
    volume_colors = ["#34d399" if close >= open_ else "#fb7185" for close, open_ in zip(data["Close"], data["Open"])]
    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], marker_color=volume_colors, opacity=0.45, name="Volume"),
        row=2,
        col=1,
    )
    for level, color, label in (
        (result.support, "#34d399", t(lang, "support")),
        (result.resistance, "#fb7185", t(lang, "resistance")),
    ):
        if level is not None:
            fig.add_hline(y=level, line_dash="dot", line_color=color, opacity=0.7, annotation_text=label, row=1, col=1)

    fig.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1221",
        margin={"l": 8, "r": 8, "t": 25, "b": 8},
        font={"color": "#94a3b8", "family": "Inter, Arial, sans-serif"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.09)", zeroline=False, side="right")
    return fig


def render_footer(lang: str) -> None:
    st.markdown(
        f"""
        <footer class="app-footer">
          <div><strong>Stock Decision Pro</strong><span>V1.2</span></div>
          <p>{escape(t(lang, 'footer_disclaimer'))}</p>
          <small>{escape(t(lang, 'source_note'))}</small>
        </footer>
        """,
        unsafe_allow_html=True,
    )
