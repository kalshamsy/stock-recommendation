"""Bilingual presentation helpers for the historical evaluation center."""

from __future__ import annotations

from html import escape
from math import isfinite

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest import BacktestResult


EVAL_TEXT = {
    "en": {
        "eyebrow": "POINT-IN-TIME EVALUATION",
        "title": "Test the recommendation rules before trusting them",
        "body": "The engine replays each historical day using only information available up to that close, then simulates the published entry zone, stop, and target.",
        "sample_insufficient": "This sample is too small for a reliability claim. Treat the result as exploratory until at least 100 executed trades are available across multiple securities and market regimes.",
        "sample_early": "This is an early sample. It is useful for diagnosing the rules, but not yet sufficient for a public success-rate claim.",
        "sample_established": "The sample size is useful, but robustness still requires multi-stock and out-of-sample testing.",
        "executed": "Executed trades",
        "target_rate": "Target-hit rate",
        "expectancy": "Average return / trade",
        "profit_factor": "Profit factor",
        "total_return": "Sequential model return",
        "drawdown": "Maximum drawdown",
        "buy_hold": "Stock buy & hold",
        "spy": "SPY reference",
        "signals": "Signals",
        "expired": "Expired",
        "targets": "Targets",
        "stops": "Stops",
        "time_exits": "Time exits",
        "equity_title": "Sequential model equity curve",
        "equity_caption": "Each executed trade is compounded sequentially. This is hypothetical model performance, not an actual account.",
        "score_title": "Does a higher score produce better outcomes?",
        "score_range": "Score range",
        "trades": "Trades",
        "profitable_rate": "Profitable rate",
        "average_return": "Average return",
        "ledger": "Complete signal ledger",
        "download": "Download evaluation CSV",
        "signal_date": "Signal date",
        "entry_date": "Entry date",
        "exit_date": "Exit date",
        "score": "Score",
        "entry": "Entry",
        "stop": "Stop",
        "target": "Target",
        "exit": "Exit",
        "outcome": "Outcome",
        "holding": "Days held",
        "return": "Net return",
        "TARGET": "Target",
        "STOP": "Stop",
        "TIME_EXIT": "Time exit",
        "EXPIRED": "No fill",
        "assumptions": "Assumptions and limitations",
        "assumptions_body": "Signals are calculated after the daily close. An order must touch the displayed entry zone within the selected entry window. The stop is assumed to occur first if the target and stop both trade inside one daily candle. One trade per symbol can be open at a time. Slippage is deducted on entry and exit. Earnings dates are excluded because a reliable historical earnings calendar is not yet connected. Investment recommendations are not backtested here because current fundamental data must not be applied to past dates.",
        "no_trades": "The current rules produced no executable entry in this period. Try a longer period or a different security; do not lower the rules only to manufacture trades.",
        "period": "Evaluation period",
        "workspace_analysis": "Stock analysis",
        "workspace_evaluation": "Evaluation center",
        "workspace_label": "Workspace",
        "period_label": "Historical period",
        "period_2y": "2 years",
        "period_5y": "5 years",
        "period_10y": "10 years",
        "holding_label": "Maximum holding period",
        "entry_window_label": "Entry window",
        "trading_days": "trading days",
        "slippage_label": "Slippage each side",
        "run_backtest": "Run historical evaluation",
        "backtest_loading": "Replaying the rules across historical daily data…",
        "backtest_error": "The historical evaluation could not be completed. Try another period or ticker.",
        "investment_note": "Investment backtesting is intentionally excluded until point-in-time fundamental statements are connected.",
        "session_log": "Recommendations generated in this browser session",
        "session_log_note": "This temporary log resets when the Streamlit session ends. Download it before closing; permanent tracking requires the database phase.",
        "session_log_empty": "Analyze a stock first and its recommendation snapshot will appear here.",
        "download_session_log": "Download session log CSV",
    },
    "ar": {
        "eyebrow": "تقييم زمني دون بيانات مستقبلية",
        "title": "اختبر قواعد التوصية قبل الاعتماد عليها",
        "body": "يعيد المحرك تحليل كل يوم تاريخي باستخدام المعلومات المتاحة حتى إغلاق ذلك اليوم فقط، ثم يحاكي منطقة الدخول والوقف والهدف المنشورة.",
        "sample_insufficient": "العينة صغيرة جدًا لإعلان موثوقية. اعتبر النتيجة استكشافية حتى نجمع 100 صفقة منفذة على الأقل عبر أسهم وظروف سوق مختلفة.",
        "sample_early": "هذه عينة أولية مفيدة لتشخيص القواعد، لكنها لا تكفي بعد لإعلان نسبة نجاح للجمهور.",
        "sample_established": "حجم العينة مفيد، لكن المتانة ما زالت تحتاج اختبار عدة أسهم وفترات خارج العينة.",
        "executed": "الصفقات المنفذة",
        "target_rate": "نسبة وصول الهدف",
        "expectancy": "متوسط العائد للصفقة",
        "profit_factor": "معامل الربح",
        "total_return": "عائد النموذج المتتابع",
        "drawdown": "أكبر تراجع",
        "buy_hold": "شراء السهم والاحتفاظ",
        "spy": "مرجع SPY",
        "signals": "الإشارات",
        "expired": "لم تُنفذ",
        "targets": "وصلت الهدف",
        "stops": "وصلت الوقف",
        "time_exits": "خروج زمني",
        "equity_title": "منحنى أداء النموذج المتتابع",
        "equity_caption": "تُركب نتائج الصفقات المنفذة بالتتابع. هذا أداء افتراضي للنموذج وليس حساب تداول فعليًا.",
        "score_title": "هل تتحسن النتائج فعلًا مع ارتفاع الدرجة؟",
        "score_range": "نطاق الدرجة",
        "trades": "الصفقات",
        "profitable_rate": "نسبة الصفقات المربحة",
        "average_return": "متوسط العائد",
        "ledger": "السجل الكامل للإشارات",
        "download": "تحميل نتائج التقييم CSV",
        "signal_date": "تاريخ الإشارة",
        "entry_date": "تاريخ الدخول",
        "exit_date": "تاريخ الخروج",
        "score": "الدرجة",
        "entry": "الدخول",
        "stop": "الوقف",
        "target": "الهدف",
        "exit": "الخروج",
        "outcome": "النتيجة",
        "holding": "أيام الاحتفاظ",
        "return": "العائد الصافي",
        "TARGET": "تحقق الهدف",
        "STOP": "تحقق الوقف",
        "TIME_EXIT": "خروج زمني",
        "EXPIRED": "لم يتم الدخول",
        "assumptions": "الافتراضات والقيود",
        "assumptions_body": "تُحسب الإشارة بعد الإغلاق اليومي. يجب أن يلامس السعر منطقة الدخول المنشورة خلال نافذة الدخول المحددة. إذا لمس السعر الوقف والهدف داخل الشمعة اليومية نفسها نفترض تحقق الوقف أولًا. لا تُفتح أكثر من صفقة واحدة للسهم في الوقت نفسه. يُخصم الانزلاق السعري عند الدخول والخروج. لا يُطبق فلتر الأرباح لأن تقويم الأرباح التاريخي الموثوق غير مربوط بعد. ولا نختبر توصيات الاستثمار هنا لأن تطبيق البيانات المالية الحالية على تواريخ قديمة سيكون مضللًا.",
        "no_trades": "لم تنتج القواعد الحالية دخولًا قابلًا للتنفيذ في هذه الفترة. جرّب فترة أطول أو سهمًا مختلفًا، ولا نخفّض الشروط لمجرد صناعة صفقات.",
        "period": "فترة التقييم",
        "workspace_analysis": "تحليل الأسهم",
        "workspace_evaluation": "مركز التقييم",
        "workspace_label": "مساحة العمل",
        "period_label": "الفترة التاريخية",
        "period_2y": "سنتان",
        "period_5y": "5 سنوات",
        "period_10y": "10 سنوات",
        "holding_label": "أقصى مدة للاحتفاظ",
        "entry_window_label": "نافذة تنفيذ الدخول",
        "trading_days": "أيام تداول",
        "slippage_label": "الانزلاق لكل جهة",
        "run_backtest": "تشغيل التقييم التاريخي",
        "backtest_loading": "جارٍ إعادة تشغيل القواعد على البيانات اليومية التاريخية…",
        "backtest_error": "تعذر إكمال التقييم التاريخي. جرّب فترة أو رمزًا آخر.",
        "investment_note": "تعمدنا عدم اختبار الاستثمار حتى نربط بيانات مالية تاريخية متاحة في وقتها الفعلي.",
        "session_log": "التوصيات الصادرة في جلسة المتصفح الحالية",
        "session_log_note": "هذا سجل مؤقت يُمسح عند انتهاء جلسة Streamlit. حمّله قبل الإغلاق؛ الحفظ الدائم يحتاج مرحلة قاعدة البيانات.",
        "session_log_empty": "حلّل سهمًا أولًا وستظهر لقطة التوصية هنا.",
        "download_session_log": "تحميل سجل الجلسة CSV",
    },
}


def et(lang: str, key: str) -> str:
    return EVAL_TEXT.get(lang, EVAL_TEXT["en"]).get(key, key)


def render_evaluation_intro(lang: str) -> None:
    st.markdown(
        f"""
        <section class="evaluation-intro">
          <small>{escape(et(lang, 'eyebrow'))}</small>
          <h2>{escape(et(lang, 'title'))}</h2>
          <p>{escape(et(lang, 'body'))}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _pct(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value):+.2f}%"


def _number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if not isfinite(numeric):
        return "∞"
    return f"{numeric:.{decimals}f}"


def _equity_figure(result: BacktestResult, lang: str) -> go.Figure:
    curve = result.equity_curve.copy()
    curve["return_pct"] = (curve["equity"] - 1) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["date"],
            y=curve["return_pct"],
            mode="lines",
            line={"color": "#22d3ee", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.10)",
            name=et(lang, "total_return"),
        )
    )
    fig.add_hline(y=0, line_color="rgba(148,163,184,.35)", line_width=1)
    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1221",
        margin={"l": 8, "r": 8, "t": 20, "b": 8},
        font={"color": "#94a3b8", "family": "Inter, Arial, sans-serif"},
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.09)", zeroline=False, ticksuffix="%", side="right")
    return fig


def _trade_table(result: BacktestResult, lang: str) -> pd.DataFrame:
    if result.trades.empty:
        return pd.DataFrame()
    frame = result.trades.copy()
    for column in ("signal_date", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
    frame["outcome"] = frame["outcome"].map(lambda value: et(lang, str(value)))
    selected = frame[
        [
            "signal_date",
            "score",
            "entry_date",
            "entry_price",
            "stop_loss",
            "target_price",
            "exit_date",
            "exit_price",
            "outcome",
            "return_pct",
            "holding_days",
        ]
    ].copy()
    selected.columns = [
        et(lang, "signal_date"),
        et(lang, "score"),
        et(lang, "entry_date"),
        et(lang, "entry"),
        et(lang, "stop"),
        et(lang, "target"),
        et(lang, "exit_date"),
        et(lang, "exit"),
        et(lang, "outcome"),
        et(lang, "return"),
        et(lang, "holding"),
    ]
    return selected


def render_backtest_result(result: BacktestResult, lang: str) -> None:
    metrics = result.metrics
    status = str(metrics.get("sample_status") or "INSUFFICIENT")
    warning_key = {
        "INSUFFICIENT": "sample_insufficient",
        "EARLY": "sample_early",
        "ESTABLISHED": "sample_established",
    }.get(status, "sample_insufficient")
    st.warning(et(lang, warning_key), icon="⚠️")
    st.caption(f"{et(lang, 'period')}: {result.start_date} → {result.end_date} · {result.symbol}")

    row_one = st.columns(4)
    row_one[0].metric(et(lang, "executed"), int(metrics["executed"]))
    row_one[1].metric(et(lang, "target_rate"), f"{float(metrics['target_rate']):.1f}%")
    row_one[2].metric(et(lang, "expectancy"), _pct(metrics["average_return"]))
    row_one[3].metric(et(lang, "profit_factor"), _number(metrics["profit_factor"]))
    row_two = st.columns(4)
    row_two[0].metric(et(lang, "total_return"), _pct(metrics["total_return"]))
    row_two[1].metric(et(lang, "drawdown"), _pct(metrics["max_drawdown"]))
    row_two[2].metric(et(lang, "buy_hold"), _pct(metrics["stock_buy_hold_return"]))
    row_two[3].metric(et(lang, "spy"), _pct(metrics["benchmark_return"]))

    st.caption(
        " · ".join(
            [
                f"{et(lang, 'signals')}: {int(metrics['signals'])}",
                f"{et(lang, 'expired')}: {int(metrics['expired'])}",
                f"{et(lang, 'targets')}: {int(metrics['targets'])}",
                f"{et(lang, 'stops')}: {int(metrics['stops'])}",
                f"{et(lang, 'time_exits')}: {int(metrics['time_exits'])}",
            ]
        )
    )

    if int(metrics["executed"]) == 0:
        st.info(et(lang, "no_trades"), icon="ℹ️")
    else:
        st.markdown(f"### {et(lang, 'equity_title')}")
        st.plotly_chart(
            _equity_figure(result, lang),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
        st.caption(et(lang, "equity_caption"))

    if not result.score_buckets.empty:
        st.markdown(f"### {et(lang, 'score_title')}")
        buckets = result.score_buckets.rename(
            columns={
                "score_range": et(lang, "score_range"),
                "trades": et(lang, "trades"),
                "target_rate": et(lang, "target_rate"),
                "profitable_rate": et(lang, "profitable_rate"),
                "average_return": et(lang, "average_return"),
            }
        )
        st.dataframe(
            buckets,
            hide_index=True,
            use_container_width=True,
            column_config={
                et(lang, "target_rate"): st.column_config.NumberColumn(format="%.1f%%"),
                et(lang, "profitable_rate"): st.column_config.NumberColumn(format="%.1f%%"),
                et(lang, "average_return"): st.column_config.NumberColumn(format="%+.2f%%"),
            },
        )

    with st.expander(et(lang, "ledger"), expanded=False):
        table = _trade_table(result, lang)
        st.dataframe(table, hide_index=True, use_container_width=True)
        csv_bytes = result.trades.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            et(lang, "download"),
            data=csv_bytes,
            file_name=f"{result.symbol.lower()}-swing-backtest.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander(et(lang, "assumptions"), expanded=False):
        st.write(et(lang, "assumptions_body"))


def render_session_log(records: list[dict], lang: str) -> None:
    st.markdown(f"### {et(lang, 'session_log')}")
    st.caption(et(lang, "session_log_note"))
    if not records:
        st.info(et(lang, "session_log_empty"), icon="ℹ️")
        return
    frame = pd.DataFrame(records)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button(
        et(lang, "download_session_log"),
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="stock-decision-session-log.csv",
        mime="text/csv",
        use_container_width=True,
    )
