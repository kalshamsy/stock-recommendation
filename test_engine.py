from app import demo_data, score_investment, score_trading


def test_demo_trading_result():
    hist, info = demo_data("AAPL")
    result = score_trading("AAPL", hist, info, 5.0)
    assert 0 <= result.score <= 100
    assert result.decision in {"دخول مشروط", "انتظار تأكيد", "لا دخول"}


def test_demo_investment_result():
    hist, info = demo_data("MSFT")
    result = score_investment("MSFT", hist, info)
    assert 0 <= result.score <= 100
    assert result.mode == "استثمار"
