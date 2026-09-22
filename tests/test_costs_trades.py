"""Cost model against hand-computed values; trade evaluation arithmetic."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from intraday.config import IST, Config
from intraday.costs import round_trip_costs
from intraday.trades import Trade, evaluate, quantity_for, results_frame


def test_costs_50k_flat_at_10bps_by_hand() -> None:
    c = round_trip_costs(50_000, 50_000, 10)
    assert c.brokerage_buy == pytest.approx(15.0)  # min(20, 0.03% x 50,000 = 15)
    assert c.brokerage_sell == pytest.approx(15.0)
    assert c.stt == pytest.approx(12.5)  # 0.025% x 50,000, sell only
    assert c.exchange_buy == pytest.approx(1.485)  # 0.00297% x 50,000
    assert c.exchange_sell == pytest.approx(1.485)
    assert c.stamp == pytest.approx(1.5)  # 0.003% x 50,000, buy only
    assert c.sebi_buy == pytest.approx(0.05)  # 0.0001% x 50,000
    assert c.sebi_sell == pytest.approx(0.05)
    assert c.gst == pytest.approx(0.18 * (15 + 15 + 1.485 + 1.485))  # 5.9346
    assert c.slippage_buy == pytest.approx(50.0)  # 10 bps x 50,000
    assert c.slippage_sell == pytest.approx(50.0)
    assert c.statutory == pytest.approx(30 + 12.5 + 2.97 + 1.5 + 0.10 + 5.9346)  # 53.0046
    assert c.total == pytest.approx(153.0046)


@pytest.mark.parametrize("bps,expected_total", [(5, 103.0046), (10, 153.0046), (20, 253.0046)])
def test_costs_at_three_slippage_levels(bps: int, expected_total: float) -> None:
    assert round_trip_costs(50_000, 50_000, bps).total == pytest.approx(expected_total)


def test_brokerage_cap_and_side_specific_items() -> None:
    c = round_trip_costs(10_000_000, 9_000_000, 0)  # 1 cr buy, 90 L sell
    assert c.brokerage_buy == 20.0 and c.brokerage_sell == 20.0
    assert c.stt == pytest.approx(0.00025 * 9_000_000)
    assert c.stamp == pytest.approx(0.00003 * 10_000_000)
    assert c.slippage == 0.0
    assert set(c.items()) == {
        "brokerage_buy", "brokerage_sell", "stt", "exchange_buy", "exchange_sell", "stamp",
        "sebi_buy", "sebi_sell", "gst", "slippage_buy", "slippage_sell", "total",
    }


def test_costs_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        round_trip_costs(0, 100, 5)
    with pytest.raises(ValueError):
        round_trip_costs(100, 100, -1)


def make_trade(direction: str = "long", entry: float = 1000.0, exit_: float = 1010.0, stop: float = 990.0, mfe: float = 1012.0) -> Trade:
    day = date(2026, 9, 7)
    return Trade(
        setup="test", variant="base", symbol="X", session_date=day, direction=direction,
        entry_index=10, entry_time=datetime(2026, 9, 7, 10, 5, tzinfo=IST), entry_price=entry,
        exit_index=20, exit_time=datetime(2026, 9, 7, 10, 55, tzinfo=IST), exit_price=exit_,
        exit_reason="eod", stop_price=stop, atr=10.0, stop_atr_multiple=1.0, mfe_price=mfe,
        features={"rvol_breakout_bar": 1.2, "or_width_atr": 0.4},
    )


def test_long_trade_r_arithmetic(config: Config) -> None:
    r = evaluate(make_trade(), 10, config)
    assert r.quantity == 50  # floor(50,000 / 1000)
    assert r.risk_inr == pytest.approx(500.0)  # 50 x (1000 - 990)
    assert r.gross_pnl == pytest.approx(500.0)  # 50 x 10
    assert r.r_gross == pytest.approx(1.0)
    assert r.buy_turnover == 50_000 and r.sell_turnover == 50_500
    assert r.net_pnl == pytest.approx(500.0 - r.costs.total)
    assert r.r_net == pytest.approx((500.0 - r.costs.total) / 500.0)
    assert r.mfe_r == pytest.approx(1.2)
    assert r.duration_bars == 10


def test_short_trade_r_arithmetic(config: Config) -> None:
    r = evaluate(make_trade("short", entry=1000.0, exit_=995.0, stop=1010.0, mfe=990.0), 5, config)
    assert r.risk_inr == pytest.approx(500.0)
    assert r.gross_pnl == pytest.approx(250.0)
    assert r.r_gross == pytest.approx(0.5)
    assert r.sell_turnover == 50_000 and r.buy_turnover == 49_750
    assert r.costs.stt == pytest.approx(0.00025 * 50_000)  # STT on the sell leg (the entry)
    assert r.costs.stamp == pytest.approx(0.00003 * 49_750)
    assert r.mfe_r == pytest.approx(1.0)


def test_costs_scale_with_slippage_but_r_definition_does_not(config: Config) -> None:
    a, b = evaluate(make_trade(), 5, config), evaluate(make_trade(), 20, config)
    assert a.r_gross == b.r_gross and a.risk_inr == b.risk_inr
    assert b.r_net < a.r_net
    assert (a.costs.total - b.costs.total) == pytest.approx(-(15 / 10_000) * (50_000 + 50_500))


def test_stop_on_wrong_side_raises(config: Config) -> None:
    with pytest.raises(ValueError, match="losing side"):
        evaluate(make_trade(stop=1005.0), 5, config)
    with pytest.raises(ValueError, match="losing side"):
        evaluate(make_trade("short", stop=995.0), 5, config)


def test_quantity_guard(config: Config) -> None:
    assert quantity_for(49_999.0, config) == 1
    with pytest.raises(ValueError, match="buys no shares"):
        quantity_for(60_000.0, config)


def test_results_frame_carries_features(config: Config) -> None:
    df = results_frame([evaluate(make_trade(), 10, config)])
    assert df.loc[0, "rvol_breakout_bar"] == 1.2 and df.loc[0, "r_net"] < 1.0
    assert "cost_slippage" in df.columns
