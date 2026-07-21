import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


st.set_page_config(
    page_title="Stock Decision MVP",
    page_icon="📈",
    layout="wide",
)


@dataclass
class AnalysisResult:
    symbol: str
    mode: str
    decision: str
    score: int
    confidence: str
    current_price: float
    entry_zone: Tuple[float, float]
    stop_loss: float
    targets: List[float]
    positives: List[str]
    risks: List[str]
    metrics: Dict[str, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def load_market_data(symbol: str) -> Tuple[pd.DataFrame, Dict]:
    if yf is None:
        raise RuntimeError("yfinance غير مثبت")

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="2y", interval="1d", auto_adjust=True)
    if hist.empty or len(hist) < 220:
        raise ValueError("لا توجد بيانات تاريخية كافية لهذا الرمز")

    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    return hist, info


def demo_data(symbol: str) -> Tuple[pd.DataFrame, Dict]:
    seed = sum(ord(c) for c in symbol.upper())
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
    trend = np.linspace(70, 115, len(dates))
    noise = rng.normal(0, 1.5, len(dates)).cumsum() * 0.2
    close = np.maximum(10, trend + noise)
    high = close * (1 + rng.uniform(0.002, 0.018, len(dates)))
    low = close * (1 - rng.uniform(0.002, 0.018, len(dates)))
    open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
    volume = rng.integers(1_000_000, 8_000_000, len(dates))
    hist = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    info = {
        "trailingPE": 24 + (seed % 9),
        "forwardPE": 20 + (seed % 7),
        "priceToSalesTrailing12Months": 4.2,
        "debtToEquity": 45 + (seed % 30),
        "returnOnEquity": 0.18,
        "revenueGrowth": 0.11,
        "earningsGrowth": 0.14,
        "freeCashflow": 2_500_000_000,
        "marketCap": 120_000_000_000,
        "sector": "Technology",
    }
    return hist, info


def prepare_metrics(hist: pd.DataFrame) -> Dict[str, float]:
    df = hist.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["HIGH20"] = df["High"].rolling(20).max()
    df["LOW20"] = df["Low"].rolling(20).min()

    last = df.iloc[-1]
    prev_20 = df.iloc[-21:-1]
    resistance = safe_float(prev_20["High"].max(), safe_float(last["Close"]) * 1.05)
    support = safe_float(prev_20["Low"].min(), safe_float(last["Close"]) * 0.95)
    sma50_20d_ago = safe_float(df["SMA50"].iloc[-21], safe_float(last["SMA50"]))

    return {
        "price": safe_float(last["Close"]),
        "sma20": safe_float(last["SMA20"]),
        "sma50": safe_float(last["SMA50"]),
        "sma200": safe_float(last["SMA200"]),
        "rsi": safe_float(last["RSI14"], 50),
        "volume_ratio": safe_float(last["Volume"] / last["VOL20"], 1),
        "resistance": resistance,
        "support": support,
        "sma50_slope_pct": safe_float((last["SMA50"] / sma50_20d_ago - 1) * 100, 0),
        "distance_to_resistance_pct": safe_float((resistance / last["Close"] - 1) * 100, 0),
        "distance_to_support_pct": safe_float((last["Close"] / support - 1) * 100, 0),
        "one_month_return_pct": safe_float((last["Close"] / df["Close"].iloc[-21] - 1) * 100, 0),
        "three_month_return_pct": safe_float((last["Close"] / df["Close"].iloc[-64] - 1) * 100, 0),
    }


def score_trading(symbol: str, hist: pd.DataFrame, info: Dict, target_pct: float) -> AnalysisResult:
    m = prepare_metrics(hist)
    score = 0
    positives, risks = [], []

    if m["price"] > m["sma50"]:
        score += 18
        positives.append("السعر أعلى من متوسط 50 يوم")
    else:
        risks.append("السعر أسفل متوسط 50 يوم")

    if m["sma50_slope_pct"] > 0:
        score += 12
        positives.append("متوسط 50 يوم يتجه للأعلى")
    else:
        risks.append("ميل متوسط 50 يوم غير صاعد")

    if m["price"] > m["sma20"]:
        score += 10
        positives.append("السعر يحافظ على الاتجاه القصير")
    else:
        risks.append("السعر أسفل متوسط 20 يوم")

    if 48 <= m["rsi"] <= 68:
        score += 12
        positives.append("RSI في نطاق صحي للمضاربة")
    elif m["rsi"] > 75:
        risks.append("السهم قريب من التشبع الشرائي")
    else:
        score += 5

    if m["volume_ratio"] >= 1.2:
        score += 13
        positives.append("حجم التداول أعلى من متوسط 20 يوم")
    elif m["volume_ratio"] >= 0.8:
        score += 7
    else:
        risks.append("حجم التداول ضعيف")

    if m["distance_to_resistance_pct"] >= target_pct + 0.5:
        score += 15
        positives.append("توجد مساحة كافية قبل المقاومة لتحقيق الهدف")
    elif m["distance_to_resistance_pct"] >= target_pct:
        score += 10
    else:
        risks.append("المقاومة أقرب من الهدف المطلوب")

    risk_pct = max(1.5, min(3.0, m["distance_to_support_pct"] * 0.55))
    rr = target_pct / risk_pct
    if rr >= 2:
        score += 20
        positives.append("نسبة العائد إلى المخاطرة مناسبة")
    elif rr >= 1.5:
        score += 12
    else:
        risks.append("نسبة العائد إلى المخاطرة غير مناسبة")

    mandatory_fail = (
        m["price"] < m["sma50"]
        or m["distance_to_resistance_pct"] < target_pct * 0.7
        or rr < 1.2
    )

    if mandatory_fail or score < 60:
        decision = "لا دخول"
    elif score < 75:
        decision = "انتظار تأكيد"
    else:
        decision = "دخول مشروط"

    entry_low = m["price"] * 0.995
    entry_high = m["price"] * 1.005
    stop = m["price"] * (1 - risk_pct / 100)
    targets = [m["price"] * 1.03, m["price"] * (1 + target_pct / 100)]

    return AnalysisResult(
        symbol=symbol,
        mode="مضاربة",
        decision=decision,
        score=int(round(clamp(score, 0, 100))),
        confidence="مرتفعة" if score >= 80 else "متوسطة" if score >= 65 else "منخفضة",
        current_price=m["price"],
        entry_zone=(entry_low, entry_high),
        stop_loss=stop,
        targets=targets,
        positives=positives[:4],
        risks=risks[:4],
        metrics={**m, "risk_reward": rr, "target_pct": target_pct},
    )


def score_investment(symbol: str, hist: pd.DataFrame, info: Dict) -> AnalysisResult:
    m = prepare_metrics(hist)
    score = 0
    positives, risks = [], []

    revenue_growth = safe_float(info.get("revenueGrowth")) * 100
    earnings_growth = safe_float(info.get("earningsGrowth")) * 100
    roe = safe_float(info.get("returnOnEquity")) * 100
    debt_to_equity = safe_float(info.get("debtToEquity"), 999)
    pe = safe_float(info.get("trailingPE"), 0)
    fcf = safe_float(info.get("freeCashflow"), 0)

    if revenue_growth >= 10:
        score += 18
        positives.append("نمو الإيرادات جيد")
    elif revenue_growth > 0:
        score += 10
    else:
        risks.append("نمو الإيرادات ضعيف أو سلبي")

    if earnings_growth >= 10:
        score += 17
        positives.append("نمو الأرباح جيد")
    elif earnings_growth > 0:
        score += 9
    else:
        risks.append("نمو الأرباح ضعيف أو سلبي")

    if roe >= 15:
        score += 15
        positives.append("العائد على حقوق المساهمين قوي")
    elif roe >= 8:
        score += 8
    else:
        risks.append("العائد على حقوق المساهمين منخفض")

    if debt_to_equity <= 60:
        score += 15
        positives.append("المديونية ضمن نطاق مقبول")
    elif debt_to_equity <= 120:
        score += 8
    else:
        risks.append("المديونية مرتفعة")

    if fcf > 0:
        score += 15
        positives.append("التدفق النقدي الحر إيجابي")
    else:
        risks.append("التدفق النقدي الحر غير إيجابي أو غير متوفر")

    if 0 < pe <= 25:
        score += 10
    elif 25 < pe <= 40:
        score += 6
    else:
        risks.append("التقييم مرتفع أو بيانات التقييم غير متوفرة")

    if m["price"] > m["sma200"] and m["sma50"] > m["sma200"]:
        score += 10
        positives.append("الاتجاه طويل الأجل صاعد")
    elif m["price"] > m["sma200"]:
        score += 6
    else:
        risks.append("الاتجاه طويل الأجل ضعيف")

    if score >= 80:
        decision = "مناسب للاستثمار التدريجي"
    elif score >= 65:
        decision = "مراقبة وانتظار سعر أفضل"
    else:
        decision = "غير مناسب حاليًا"

    entry_low = min(m["price"], m["sma50"])
    entry_high = m["price"]
    stop = min(m["support"], m["sma200"]) * 0.98
    targets = [m["price"] * 1.10, m["price"] * 1.20]

    metrics = {
        **m,
        "revenue_growth_pct": revenue_growth,
        "earnings_growth_pct": earnings_growth,
        "roe_pct": roe,
        "debt_to_equity": debt_to_equity,
        "pe": pe,
        "free_cashflow": fcf,
    }

    return AnalysisResult(
        symbol=symbol,
        mode="استثمار",
        decision=decision,
        score=int(round(clamp(score, 0, 100))),
        confidence="مرتفعة" if score >= 80 else "متوسطة" if score >= 65 else "منخفضة",
        current_price=m["price"],
        entry_zone=(entry_low, entry_high),
        stop_loss=stop,
        targets=targets,
        positives=positives[:4],
        risks=risks[:4],
        metrics=metrics,
    )


def money(value: float) -> str:
    return f"${value:,.2f}"


st.title("📈 Stock Decision MVP")
st.caption("محرك أولي لتقييم الأسهم للاستثمار أو المضاربة القصيرة")

with st.sidebar:
    st.header("إعداد التحليل")
    symbol = st.text_input("رمز السهم", value="AAPL", max_chars=12).strip().upper()
    mode = st.radio("نوع العملية", ["مضاربة 3–5%", "استثمار"])
    target_pct = st.slider("هدف المضاربة %", 3.0, 5.0, 5.0, 0.5) if mode.startswith("مضاربة") else 5.0
    use_demo = st.toggle("استخدام بيانات تجريبية", value=False)
    analyze = st.button("تحليل السهم", type="primary", use_container_width=True)

st.info("هذه النسخة تعليمية وتجريبية، وليست توصية مالية شخصية أو ضمانًا للربح.")

if analyze:
    if not symbol:
        st.error("أدخل رمز السهم")
        st.stop()

    try:
        with st.spinner("جاري تحميل البيانات وتحليل السهم..."):
            if use_demo:
                hist, info = demo_data(symbol)
                source_note = "بيانات تجريبية"
            else:
                hist, info = load_market_data(symbol)
                source_note = "بيانات سوق عبر yfinance"

            result = score_trading(symbol, hist, info, target_pct) if mode.startswith("مضاربة") else score_investment(symbol, hist, info)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("القرار", result.decision)
        col2.metric("الدرجة", f"{result.score}/100")
        col3.metric("السعر الحالي", money(result.current_price))
        col4.metric("الثقة", result.confidence)

        st.progress(result.score / 100)

        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("خطة الصفقة")
            st.write(f"**منطقة الدخول:** {money(result.entry_zone[0])} – {money(result.entry_zone[1])}")
            st.write(f"**وقف الخسارة الإرشادي:** {money(result.stop_loss)}")
            st.write("**الأهداف:** " + "، ".join(money(x) for x in result.targets))
            st.write(f"**مصدر التحليل:** {source_note}")

        with right:
            st.subheader("لماذا؟")
            if result.positives:
                st.markdown("**نقاط إيجابية**")
                for item in result.positives:
                    st.write(f"✅ {item}")
            if result.risks:
                st.markdown("**مخاطر وملاحظات**")
                for item in result.risks:
                    st.write(f"⚠️ {item}")

        st.subheader("الرسم السعري")
        chart_df = hist[["Close"]].copy()
        chart_df["SMA20"] = hist["Close"].rolling(20).mean()
        chart_df["SMA50"] = hist["Close"].rolling(50).mean()
        st.line_chart(chart_df.tail(180), use_container_width=True)

        with st.expander("المؤشرات والبيانات التفصيلية"):
            metrics_df = pd.DataFrame(
                [{"المؤشر": k, "القيمة": round(v, 3) if isinstance(v, (int, float)) else v} for k, v in result.metrics.items()]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with st.expander("JSON API Output"):
            st.json(asdict(result))

    except Exception as exc:
        st.error(f"تعذر تحليل الرمز: {exc}")
        st.caption("جرّب تشغيل خيار البيانات التجريبية للتأكد من أن التطبيق يعمل.")
else:
    st.markdown(
        """
        ### ما الذي يفعله هذا الـMVP؟
        - إدخال رمز السهم.
        - اختيار استثمار أو مضاربة قصيرة.
        - حساب درجة من 100.
        - إصدار قرار: دخول مشروط، انتظار، أو لا دخول.
        - اقتراح منطقة دخول ووقف خسارة وأهداف.
        - عرض أسباب القرار والمخاطر.
        """
    )
