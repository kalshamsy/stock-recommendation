# Stock Decision Pro V1.0

A professional, mobile-first, bilingual Streamlit web app for explainable stock research. Users can search by ticker or company name, choose **Swing Trading** or **Investment**, and see a quick decision first with optional evidence and charts.

> **Important:** This is a research and education tool, not personalized financial advice. It does not guarantee returns. Market data can be delayed, incomplete, or temporarily unavailable.

## What is included

- Arabic and English language selection on first launch.
- One-click language switching after launch.
- Arabic right-to-left layout and mobile-first responsive styling.
- Ticker/company autocomplete using a bundled list plus `yfinance.Search` when available.
- Separate deterministic engines:
  - **Swing Trading:** trend, momentum, liquidity, entry quality, room to resistance, and risk/reward.
  - **Investment:** growth, profitability, financial health, valuation, quality, and technical timing.
- Quick result card with:
  - decision;
  - score and confidence;
  - current price and daily change;
  - entry zone, risk level, and price targets.
- Expandable details for reasons, risks, blockers, score breakdown, indicators, fundamentals, methodology, and charts.
- Interactive daily candlestick chart with volume, EMA 20, SMA 50, SMA 200, support, and resistance.
- Cached requests and graceful data-error handling.
- Transparent source code and unit tests for the calculation layer.

## Project structure

```text
stock-decision-pro/
├── app.py                         # Streamlit entry point
├── requirements.txt              # Community Cloud dependencies
├── README.md
├── assets/
│   └── style.css                  # Responsive financial-dashboard theme
├── data/
│   └── stock_universe.csv         # Offline autocomplete fallback
├── src/
│   ├── config.py                  # Weights and product configuration
│   ├── data.py                    # Search and market-data access
│   ├── engines.py                 # Swing and Investment engines
│   ├── i18n.py                    # Arabic/English interface copy
│   ├── indicators.py              # Technical calculations
│   ├── models.py                  # Typed result models
│   └── ui.py                      # UI components and Plotly chart
├── tests/
│   └── test_engines.py
└── .streamlit/
    └── config.toml                # Streamlit theme and server settings
```

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Replace the current GitHub repository files

Upload the **contents of this project**, not the ZIP itself. The repository root must contain `app.py` and `requirements.txt`.

1. Open the current repository on GitHub.
2. Use **Add file → Upload files**.
3. Upload `app.py`, `requirements.txt`, `README.md`, `.streamlit`, `assets`, `data`, and `src`.
4. Commit the changes to the branch used by Streamlit, normally `main`.
5. Streamlit Community Cloud should redeploy automatically.

If GitHub's mobile upload does not preserve folders, upload from a computer or create each folder in GitHub before adding its files. The folder names must remain exactly as shown above.

## Streamlit Community Cloud settings

- **Repository:** your existing Stock Decision repository
- **Branch:** `main` (or the branch already configured)
- **Main file path:** `app.py`
- **Secrets:** none required for this V1

If the old page remains visible, open **Manage app → Reboot app** and confirm that the main file path is `app.py`.

## Add to the iPhone Home Screen

1. Open the deployed `.streamlit.app` link in Safari.
2. Tap **Share**.
3. Tap **Add to Home Screen**.
4. Tap **Add**.

## How the decisions work

The app does not use an LLM to invent a recommendation. Each engine applies visible rules and weighted scoring. Missing data lowers confidence, and material blockers can prevent an entry result.

### Swing Trading — 100 points

| Component | Weight |
|---|---:|
| Trend and market context | 25 |
| Momentum | 15 |
| Volume and liquidity | 15 |
| Entry quality | 15 |
| Room to the selected target | 15 |
| Risk/reward | 15 |

Potential blockers include insufficient history, low average dollar volume, inadequate room before estimated resistance, weak risk/reward, and a possible earnings event within five days.

### Investment — 100 points

| Component | Weight |
|---|---:|
| Revenue and earnings growth | 20 |
| Profitability and free cash flow | 20 |
| Financial health | 15 |
| Valuation | 20 |
| Business quality | 15 |
| Technical timing | 10 |

Investment results depend heavily on available company fundamentals. Limited coverage forces a lower-confidence **Wait** result rather than pretending the data is complete.

## Data-source limitations

This V1 uses `yfinance`/Yahoo Finance for prototyping. It is convenient but is not a licensed commercial market-data feed. Before charging users or relying on real-time execution:

- replace it with a licensed provider;
- add exchange-specific delay labels;
- store recommendation snapshots and immutable input data;
- backtest the rules without look-ahead bias;
- add monitoring, rate-limit protection, and data-quality checks;
- obtain appropriate legal and regulatory review for the intended markets.

## Development checks

```bash
python -m compileall app.py src
python -m pytest -q
```

## العربية — ملخص سريع

هذه النسخة جاهزة للنشر على Streamlit Community Cloud وتعمل على الآيفون والكمبيوتر. تعرض قرارًا سريعًا أولًا، ثم تتيح فتح أسباب القرار والمخاطر والدرجات والرسم البياني عند الحاجة. التطبيق أداة بحث وتعليم، وليس توصية مالية شخصية أو ضمانًا للربح. قبل الاستخدام التجاري يجب ربط مزود بيانات سوق مرخص وإجراء اختبارات تاريخية وتدقيق قانوني مناسب.
