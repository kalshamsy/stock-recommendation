"""Bilingual UI for the saved fixed-universe portfolio evaluation report."""

from __future__ import annotations

from html import escape
from math import isfinite
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .portfolio_backtest import load_universe_report


COPY = {
    "en": {
        "eyebrow": "FIXED-UNIVERSE PORTFOLIO TEST",
        "title": "Hundreds of securities, one constrained portfolio",
        "body": "This saved report applies the same swing rules to the bundled security list, then accepts trades subject to cash, risk sizing, and a maximum number of concurrent positions.",
        "missing": "The universe report has not been generated yet. Run scripts/run_universe_backtest.py locally, then commit data/universe_report.json.",
        "warning": "Research result only. The current list is used across the whole history, creating survivorship/current-universe bias. It is not evidence of future performance.",
        "sample_warning": "Fewer than 100 portfolio trades were accepted, so this is still an exploratory sample.",
        "generated": "Generated",
        "period": "Period",
        "symbols": "Securities tested",
        "failed": "Data failures",
        "trades": "Portfolio trades",
        "trade_flow": "Executable candidates: {candidates} · Accepted: {accepted} · Skipped at capacity: {skipped}",
        "target_rate": "Target-hit rate",
        "average_return": "Average return / trade",
        "profit_factor": "Profit factor",
        "total_return": "Portfolio return",
        "drawdown": "Realized drawdown",
        "spy": "SPY reference",
        "curve": "Portfolio equity curve",
        "curve_note": "Book equity is updated when positions close; open trades are held at cost. Therefore intraday and unrealized drawdowns are not shown.",
        "calibration": "Score calibration",
        "symbols_table": "Results by security",
        "ledger": "Accepted portfolio trades",
        "failures": "Symbols without a completed test",
        "download": "Download accepted trades CSV",
        "assumptions": "Portfolio assumptions",
        "assumptions_body": "Initial capital: {capital}. At most {positions} positions can be open. Each position risks no more than {risk}% of book equity at the published stop and is capped at {weight}% of equity. Exits are processed before entries on the same day. Entry and exit slippage is included. Historical earnings dates and point-in-time sector membership are not yet connected.",
    },
    "ar": {
        "eyebrow": "اختبار محفظة على قائمة أسهم ثابتة",
        "title": "مئات الرموز ضمن محفظة واحدة بقيود واقعية",
        "body": "يطبق هذا التقرير المحفوظ قواعد المضاربة نفسها على قائمة الرموز المرفقة، ثم يقبل الصفقات وفق السيولة وحجم المخاطرة والحد الأقصى للصفقات المفتوحة.",
        "missing": "لم يُنشأ تقرير القائمة بعد. شغّل scripts/run_universe_backtest.py محليًا، ثم ارفع data/universe_report.json.",
        "warning": "هذه نتيجة بحثية فقط. استخدام قائمة اليوم على كامل التاريخ يسبب انحياز البقاء/القائمة الحالية، ولا يثبت الأداء المستقبلي.",
        "sample_warning": "تم قبول أقل من 100 صفقة في المحفظة، لذلك ما زالت العينة استكشافية.",
        "generated": "تاريخ الإنشاء",
        "period": "الفترة",
        "symbols": "الرموز المختبرة",
        "failed": "أخطاء البيانات",
        "trades": "صفقات المحفظة",
        "trade_flow": "الفرص القابلة للتنفيذ: {candidates} · المقبولة: {accepted} · المرفوضة لاكتمال السعة: {skipped}",
        "target_rate": "نسبة وصول الهدف",
        "average_return": "متوسط عائد الصفقة",
        "profit_factor": "معامل الربح",
        "total_return": "عائد المحفظة",
        "drawdown": "التراجع المحقق",
        "spy": "مرجع SPY",
        "curve": "منحنى قيمة المحفظة",
        "curve_note": "تُحدّث قيمة المحفظة عند إغلاق الصفقات، وتبقى الصفقات المفتوحة بسعر التكلفة؛ لذلك لا يظهر التراجع غير المحقق أو داخل اليوم.",
        "calibration": "معايرة درجات التوصية",
        "symbols_table": "النتائج حسب السهم",
        "ledger": "صفقات المحفظة المقبولة",
        "failures": "رموز لم يكتمل اختبارها",
        "download": "تحميل الصفقات المقبولة CSV",
        "assumptions": "افتراضات المحفظة",
        "assumptions_body": "رأس المال الأولي: {capital}. الحد الأقصى {positions} صفقات مفتوحة. لا تخاطر الصفقة بأكثر من {risk}% من قيمة المحفظة عند الوقف المنشور، ولا يتجاوز حجمها {weight}% من المحفظة. تُنفذ عمليات الخروج قبل الدخول في اليوم نفسه، ويُحتسب الانزلاق عند الدخول والخروج. لم نربط بعد تقويم الأرباح التاريخي أو عضوية القطاعات المتاحة في وقتها.",
    },
}


def ut(lang: str, key: str) -> str:
    return COPY.get(lang, COPY["en"]).get(key, key)


def _pct(value) -> str:
    return "—" if value is None else f"{float(value):+.2f}%"


def _number(value) -> str:
    if value is None:
        return "—"
    number = float(value)
    return "∞" if not isfinite(number) else f"{number:.2f}"


def _curve_figure(frame: pd.DataFrame, lang: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=pd.to_datetime(frame["date"]),
            y=frame["return_pct"],
            mode="lines",
            line={"color": "#34d399", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(52,211,153,.10)",
        )
    )
    fig.add_hline(y=0, line_color="rgba(148,163,184,.35)", line_width=1)
    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1221",
        margin={"l": 8, "r": 8, "t": 20, "b": 8},
        font={"color": "#94a3b8", "family": "Inter, Arial, sans-serif"},
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.09)", ticksuffix="%", side="right")
    return fig


def render_universe_report(path: str | Path, lang: str) -> None:
    st.markdown(
        f"""
        <section class="evaluation-intro">
          <small>{escape(ut(lang, 'eyebrow'))}</small>
          <h2>{escape(ut(lang, 'title'))}</h2>
          <p>{escape(ut(lang, 'body'))}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    report_path = Path(path)
    if not report_path.exists():
        st.info(ut(lang, "missing"), icon="ℹ️")
        return

    report = load_universe_report(report_path)
    metrics = report["metrics"]
    meta = report["metadata"]
    st.warning(ut(lang, "warning"), icon="⚠️")
    if int(metrics.get("accepted_trades", 0)) < 100:
        st.warning(ut(lang, "sample_warning"), icon="⚠️")
    generated = str(meta.get("generated_at") or "—")[:10]
    st.caption(
        f"{ut(lang, 'generated')}: {generated} · {ut(lang, 'period')}: "
        f"{meta.get('start_date', '—')} → {meta.get('end_date', '—')} · {meta.get('target_pct', 5):g}%"
    )

    first = st.columns(4)
    first[0].metric(ut(lang, "symbols"), int(metrics.get("symbols_succeeded", 0)))
    first[1].metric(ut(lang, "failed"), int(metrics.get("symbols_failed", 0)))
    first[2].metric(ut(lang, "trades"), int(metrics.get("accepted_trades", 0)))
    first[3].metric(ut(lang, "target_rate"), f"{float(metrics.get('target_rate', 0)):.1f}%")
    second = st.columns(4)
    second[0].metric(ut(lang, "average_return"), _pct(metrics.get("average_return")))
    second[1].metric(ut(lang, "profit_factor"), _number(metrics.get("profit_factor")))
    second[2].metric(ut(lang, "total_return"), _pct(metrics.get("total_return")))
    second[3].metric(ut(lang, "drawdown"), _pct(metrics.get("max_drawdown")))
    st.metric(ut(lang, "spy"), _pct(metrics.get("benchmark_return")))
    st.caption(
        ut(lang, "trade_flow").format(
            candidates=int(metrics.get("candidate_trades", 0)),
            accepted=int(metrics.get("accepted_trades", 0)),
            skipped=int(metrics.get("skipped_capacity", 0)),
        )
    )

    equity = pd.DataFrame(report.get("equity_curve", []))
    if not equity.empty:
        st.markdown(f"### {ut(lang, 'curve')}")
        st.plotly_chart(_curve_figure(equity, lang), use_container_width=True, config={"displaylogo": False})
        st.caption(ut(lang, "curve_note"))

    buckets = pd.DataFrame(report.get("score_buckets", []))
    if not buckets.empty:
        st.markdown(f"### {ut(lang, 'calibration')}")
        st.dataframe(buckets, hide_index=True, use_container_width=True)

    per_symbol = pd.DataFrame(report.get("per_symbol", []))
    with st.expander(ut(lang, "symbols_table"), expanded=False):
        st.dataframe(per_symbol, hide_index=True, use_container_width=True)

    trades = pd.DataFrame(report.get("accepted_trades", []))
    with st.expander(ut(lang, "ledger"), expanded=False):
        st.dataframe(trades, hide_index=True, use_container_width=True)
        st.download_button(
            ut(lang, "download"),
            data=trades.to_csv(index=False).encode("utf-8-sig"),
            file_name="stock-decision-universe-trades.csv",
            mime="text/csv",
            use_container_width=True,
        )

    failures = pd.DataFrame(report.get("failures", []))
    if not failures.empty:
        with st.expander(ut(lang, "failures"), expanded=False):
            st.dataframe(failures, hide_index=True, use_container_width=True)

    with st.expander(ut(lang, "assumptions"), expanded=False):
        st.write(
            ut(lang, "assumptions_body").format(
                capital=f"${float(metrics.get('initial_capital', 0)):,.0f}",
                positions=int(meta.get("max_positions", 10)),
                risk=float(meta.get("risk_per_trade_pct", 1)),
                weight=float(meta.get("max_position_pct", 10)),
            )
        )
