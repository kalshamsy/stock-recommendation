import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="StockDecision Pro", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
:root { --bg:#0b1020; --panel:#121a2e; --text:#eef2ff; --muted:#9aa6c3; --green:#16c784; --red:#ea3943; --amber:#f5a623; --line:#25304b; }
.stApp { background:linear-gradient(180deg,#09101f 0%,#0d1425 100%); color:var(--text); }
.block-container { max-width:1180px; padding-top:1.1rem; padding-bottom:5rem; }
[data-testid="stSidebar"] { background:#0d1425; border-right:1px solid var(--line); }
h1,h2,h3,p,label,span,div { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
.hero { background:linear-gradient(135deg,#17213b,#10172a); border:1px solid var(--line); border-radius:22px; padding:22px; margin-bottom:16px; }
.eyebrow { color:#7d8baa; font-size:.82rem; letter-spacing:.08em; text-transform:uppercase; }
.ticker { font-size:2rem; font-weight:800; margin:2px 0; }
.price { font-size:1.45rem; font-weight:700; }
.card { background:rgba(18,26,46,.96); border:1px solid var(--line); border-radius:18px; padding:18px; height:100%; }
.metric-label { color:var(--muted); font-size:.82rem; }
.metric-value { font-size:1.55rem; font-weight:800; margin-top:4px; }
.decision { border-radius:20px; padding:22px; text-align:center; margin:10px 0 18px; }
.decision-buy { background:rgba(22,199,132,.12); border:1px solid rgba(22,199,132,.45); }
.decision-wait { background:rgba(245,166,35,.12); border:1px solid rgba(245,166,35,.45); }
.decision-no { background:rgba(234,57,67,.12); border:1px solid rgba(234,57,67,.45); }
.decision-title { font-size:2.15rem; font-weight:900; }
.decision-sub { color:var(--muted); margin-top:5px; }
.pill { display:inline-block; border:1px solid var(--line); background:#0d1425; border-radius:99px; padding:6px 11px; margin:3px; font-size:.82rem; }
.good { color:var(--green); } .bad { color:#ff7077; } .warn { color:var(--amber); }
.section-title { font-size:1.25rem; font-weight:800; margin:14px 0 8px; }
.small { color:var(--muted); font-size:.82rem; }
[data-testid="stMetricValue"] { color:var(--text); }
.stButton button { width:100%; border-radius:12px; min-height:46px; font-weight:750; }
.stTextInput input, [data-baseweb="select"] { border-radius:12px !important; }
@media(max-width:640px){ .ticker{font-size:1.55rem}.decision-title{font-size:1.7rem}.block-container{padding-left:.8rem;padding-right:.8rem} }
</style>
""",
    unsafe_allow_html=True,
)

@dataclass
class Result:
    symbol: str
    company: str
    mode: str
    decision: str
    score: int
    confidence: str
    price: float
    change_pct: float
    entry: Tuple[float, float]
    stop: float
    targets: List[float]
    positives: List[str]
    risks: List[str]
    metrics: Dict[str, float]


def sf(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def rsi(series: pd.Series, n=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)


def atr(df: pd.DataFrame, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-prev).abs(), (df["Low"]-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@st.cache_data(ttl=900, show_spinner=False)
def fetch(symbol: str):
    if yf is None:
        raise RuntimeError("yfinance غير متاح")
    t = yf.Ticker(symbol)
    hist = t.history(period="2y", interval="1d", auto_adjust=False)
    if hist.empty or len(hist) < 220:
        raise ValueError("لا توجد بيانات كافية لهذا الرمز")
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return hist, info


def enrich(df):
    x = df.copy()
    x["EMA20"] = x.Close.ewm(span=20, adjust=False).mean()
    x["SMA50"] = x.Close.rolling(50).mean()
    x["SMA200"] = x.Close.rolling(200).mean()
    x["RSI"] = rsi(x.Close)
    x["ATR"] = atr(x)
    x["VOL20"] = x.Volume.rolling(20).mean()
    return x


def technical_metrics(df):
    x = enrich(df)
    last = x.iloc[-1]
    prev = x.iloc[-2]
    window = x.iloc[-61:-1]
    resistance = sf(window.High.quantile(.92), sf(last.Close)*1.06)
    support = sf(window.Low.quantile(.08), sf(last.Close)*.94)
    price = sf(last.Close)
    return {
        "price": price,
        "change_pct": (price/sf(prev.Close, price)-1)*100,
        "ema20": sf(last.EMA20), "sma50": sf(last.SMA50), "sma200": sf(last.SMA200),
        "rsi": sf(last.RSI,50), "atr": sf(last.ATR,price*.02),
        "volume_ratio": sf(last.Volume/sf(last.VOL20,last.Volume),1),
        "support": support, "resistance": resistance,
        "room_pct": (resistance/price-1)*100,
        "one_month": (price/sf(x.Close.iloc[-22],price)-1)*100,
        "three_month": (price/sf(x.Close.iloc[-65],price)-1)*100,
        "sma50_slope": (sf(last.SMA50)/sf(x.SMA50.iloc[-21],sf(last.SMA50))-1)*100,
    }


def trading_result(symbol, hist, info, target_pct):
    m = technical_metrics(hist); score=0; pos=[]; risk=[]
    if m["price"] > m["sma50"]: score+=16; pos.append("السعر أعلى من متوسط 50 يوم")
    else: risk.append("السعر أسفل متوسط 50 يوم")
    if m["sma50_slope"] > 0: score+=12; pos.append("اتجاه متوسط 50 يوم صاعد")
    else: risk.append("متوسط 50 يوم لا يتجه للأعلى")
    if m["price"] > m["ema20"]: score+=12; pos.append("الزخم القصير إيجابي")
    else: risk.append("السعر أسفل EMA20")
    if 48 <= m["rsi"] <= 68: score+=14; pos.append("RSI في نطاق مناسب")
    elif 40 <= m["rsi"] < 48: score+=7
    elif m["rsi"] > 72: risk.append("السهم ممتد أو قريب من التشبع الشرائي")
    else: risk.append("الزخم ضعيف")
    if m["volume_ratio"] >= 1.2: score+=12; pos.append("الحجم أعلى من المتوسط")
    elif m["volume_ratio"] >= .8: score+=6
    else: risk.append("حجم التداول ضعيف")
    if m["room_pct"] >= target_pct+1: score+=16; pos.append("مساحة كافية قبل المقاومة")
    elif m["room_pct"] >= target_pct: score+=10
    else: risk.append("المقاومة أقرب من الهدف")
    stop_distance = max(1.5, min(3.5, (m["atr"]/m["price"])*100*1.25))
    rr = target_pct/stop_distance
    if rr>=2: score+=18; pos.append("العائد إلى المخاطرة مناسب")
    elif rr>=1.5: score+=11
    else: risk.append("العائد إلى المخاطرة غير كافٍ")
    hard_fail = m["price"] < m["sma50"] or m["room_pct"] < target_pct*.7 or rr<1.15
    decision = "لا دخول حاليًا" if hard_fail or score<60 else "انتظار تأكيد" if score<76 else "دخول مناسب حاليًا"
    confidence = "مرتفعة" if score>=82 else "متوسطة" if score>=68 else "منخفضة"
    price=m["price"]; entry=(price*.997, price*1.003); stop=price*(1-stop_distance/100)
    return Result(symbol, info.get("longName") or symbol, "مضاربة", decision, round(score), confidence, price, m["change_pct"], entry, stop, [price*1.03,price*(1+target_pct/100)],pos,risk,{**m,"risk_reward":rr,"target_pct":target_pct,"stop_pct":stop_distance})


def investment_result(symbol, hist, info):
    m=technical_metrics(hist); score=0; pos=[]; risk=[]
    rev=sf(info.get("revenueGrowth"))*100; earn=sf(info.get("earningsGrowth"))*100
    roe=sf(info.get("returnOnEquity"))*100; debt=sf(info.get("debtToEquity"),999)
    pe=sf(info.get("trailingPE")); fpe=sf(info.get("forwardPE")); fcf=sf(info.get("freeCashflow"))
    if rev>=10: score+=17; pos.append("نمو الإيرادات قوي")
    elif rev>0: score+=9
    else: risk.append("نمو الإيرادات ضعيف أو غير متاح")
    if earn>=10: score+=17; pos.append("نمو الأرباح قوي")
    elif earn>0: score+=9
    else: risk.append("نمو الأرباح ضعيف أو غير متاح")
    if roe>=15: score+=14; pos.append("العائد على حقوق المساهمين جيد")
    elif roe>=8: score+=8
    else: risk.append("ROE منخفض أو غير متاح")
    if debt<=80: score+=13; pos.append("المديونية ضمن مستوى مقبول")
    elif debt<=150: score+=7
    else: risk.append("المديونية مرتفعة أو البيانات غير مكتملة")
    if fcf>0: score+=12; pos.append("التدفق النقدي الحر إيجابي")
    else: risk.append("التدفق النقدي الحر غير إيجابي أو غير متاح")
    if 0<pe<=30: score+=12; pos.append("التقييم السعري غير مبالغ فيه نسبيًا")
    elif 0<pe<=45: score+=7
    else: risk.append("التقييم مرتفع أو غير متاح")
    if m["price"]>m["sma200"]: score+=9; pos.append("الاتجاه طويل الأجل إيجابي")
    else: risk.append("السعر دون متوسط 200 يوم")
    if m["price"]>m["sma50"]: score+=6
    decision="غير مناسب حاليًا" if score<58 else "مراقبة أو شراء تدريجي" if score<76 else "مناسب للاستثمار التدريجي"
    confidence="مرتفعة" if score>=82 else "متوسطة" if score>=67 else "منخفضة"
    price=m["price"]; stop=min(m["support"],price*.90)
    return Result(symbol,info.get("longName") or symbol,"استثمار",decision,round(score),confidence,price,m["change_pct"],(price*.97,price*1.01),stop,[price*1.10,price*1.20],pos,risk,{**m,"revenue_growth":rev,"earnings_growth":earn,"roe":roe,"debt_to_equity":debt,"pe":pe,"forward_pe":fpe})


def money(v):
    return f"${v:,.2f}" if v else "—"


def render_chart(hist, result):
    x=enrich(hist).tail(180)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=x.index,open=x.Open,high=x.High,low=x.Low,close=x.Close,name="Price"))
    fig.add_trace(go.Scatter(x=x.index,y=x.EMA20,name="EMA20",line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=x.index,y=x.SMA50,name="SMA50",line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=x.index,y=x.SMA200,name="SMA200",line=dict(width=1.2)))
    fig.add_hline(y=result.stop,line_dash="dot",annotation_text="Stop")
    for i,t in enumerate(result.targets,1): fig.add_hline(y=t,line_dash="dot",annotation_text=f"T{i}")
    fig.update_layout(height=500,margin=dict(l=5,r=5,t=30,b=5),paper_bgcolor="#121a2e",plot_bgcolor="#121a2e",font_color="#dfe6ff",xaxis_rangeslider_visible=False,legend_orientation="h")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})


with st.sidebar:
    st.markdown("## إعداد التحليل")
    symbol=st.text_input("رمز السهم",value="AAPL",max_chars=12).strip().upper()
    mode=st.radio("نوع العملية",["مضاربة 3–5%","استثمار"],horizontal=False)
    target=st.slider("هدف المضاربة %",3.0,8.0,5.0,.5,disabled=mode=="استثمار")
    analyze=st.button("تحليل السهم",type="primary")
    st.caption("النتائج تحليل آلي تعليمي وليست ضمانًا أو توصية مالية شخصية.")

st.markdown("<div class='eyebrow'>STOCK INTELLIGENCE PLATFORM</div><h1 style='margin:.15rem 0 1rem'>StockDecision <span style='color:#7c8cff'>Pro</span></h1>",unsafe_allow_html=True)

if "last_symbol" not in st.session_state: st.session_state.last_symbol="AAPL"
if analyze: st.session_state.last_symbol=symbol or "AAPL"
symbol=st.session_state.last_symbol

try:
    with st.spinner(f"جاري تحليل {symbol}..."):
        hist,info=fetch(symbol)
        result=trading_result(symbol,hist,info,target) if mode.startswith("مضاربة") else investment_result(symbol,hist,info)
except Exception as e:
    st.error(f"تعذر تحميل بيانات {symbol}: {e}")
    st.stop()

change_class="good" if result.change_pct>=0 else "bad"
st.markdown(f"""
<div class='hero'>
  <div class='eyebrow'>{result.mode}</div>
  <div class='ticker'>{result.company} <span style='color:#7d8baa;font-size:1rem'>({result.symbol})</span></div>
  <div class='price'>{money(result.price)} <span class='{change_class}' style='font-size:1rem'>{result.change_pct:+.2f}%</span></div>
</div>
""",unsafe_allow_html=True)

kind="decision-buy" if "مناسب" in result.decision and "غير" not in result.decision else "decision-wait" if "انتظار" in result.decision or "مراقبة" in result.decision else "decision-no"
st.markdown(f"""
<div class='decision {kind}'>
 <div class='metric-label'>القرار السريع</div>
 <div class='decision-title'>{result.decision}</div>
 <div class='decision-sub'>الدرجة {result.score}/100 · الثقة {result.confidence}</div>
</div>
""",unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)
items=[("منطقة الدخول",f"{money(result.entry[0])} – {money(result.entry[1])}"),("وقف الخسارة",money(result.stop)),("الهدف الأول",money(result.targets[0])),("الهدف الثاني",money(result.targets[1]))]
for c,(lab,val) in zip([c1,c2,c3,c4],items):
    with c: st.markdown(f"<div class='card'><div class='metric-label'>{lab}</div><div class='metric-value'>{val}</div></div>",unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
with st.expander("عرض تفاصيل التحليل",expanded=False):
    tab1,tab2,tab3,tab4=st.tabs(["ملخص","الرسم البياني","أسباب القرار","المؤشرات"])
    with tab1:
        a,b,c=st.columns(3)
        a.metric("RSI",f"{result.metrics['rsi']:.1f}")
        b.metric("الحجم مقابل المتوسط",f"{result.metrics['volume_ratio']:.2f}×")
        if result.mode=="مضاربة": c.metric("العائد/المخاطرة",f"{result.metrics['risk_reward']:.2f}")
        else: c.metric("P/E",f"{result.metrics.get('pe',0):.1f}" if result.metrics.get('pe',0)>0 else "—")
        st.markdown("#### الخلاصة")
        st.write("هذا القرار ناتج عن قواعد كمية تجمع الاتجاه والزخم والسيولة والمقاومة وإدارة المخاطر، أو جودة الشركة وتقييمها في وضع الاستثمار.")
    with tab2: render_chart(hist,result)
    with tab3:
        pcol,rcol=st.columns(2)
        with pcol:
            st.markdown("### نقاط القوة")
            if result.positives:
                for x in result.positives: st.markdown(f"✅ {x}")
            else: st.write("لا توجد نقاط قوة كافية.")
        with rcol:
            st.markdown("### المخاطر")
            if result.risks:
                for x in result.risks: st.markdown(f"⚠️ {x}")
            else: st.write("لم تُرصد مخاطر فنية كبيرة ضمن القواعد الحالية.")
    with tab4:
        metrics = {
            "السعر": result.metrics["price"], "EMA20": result.metrics["ema20"], "SMA50": result.metrics["sma50"], "SMA200": result.metrics["sma200"],
            "RSI": result.metrics["rsi"], "الدعم": result.metrics["support"], "المقاومة": result.metrics["resistance"], "مساحة حتى المقاومة %": result.metrics["room_pct"],
            "أداء شهر %": result.metrics["one_month"], "أداء 3 أشهر %": result.metrics["three_month"],
        }
        st.dataframe(pd.DataFrame({"المؤشر":metrics.keys(),"القيمة":[round(v,2) for v in metrics.values()]}),hide_index=True,use_container_width=True)

st.caption("تنبيه: بيانات yfinance قد تكون متأخرة أو ناقصة. قبل الاستخدام التجاري يلزم مزود بيانات مرخص واختبار تاريخي شامل ومراجعة قانونية.")
