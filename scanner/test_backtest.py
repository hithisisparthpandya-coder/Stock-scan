"""Backtester tests: exit mechanics, accounting, no look-ahead, split adjustment, baseline parity.
Run: python scanner/test_backtest.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtest as bt  # noqa: E402
from demo import make_history, make_long_history  # noqa: E402
from funnel import run_funnel  # noqa: E402
from strategy import build_panel, features, signals  # noqa: E402

CFG = json.loads((Path(__file__).parent / "config.json").read_text())
ZERO = dict(CFG, cost_per_side=0.0, slippage_per_side=0.0, dp_charge=0, bt_warmup=5, bt_capital=100000,
            max_position_pct=1.0, heat_cap=1.0, risk_pct=0.01)


def _panel(bars):
    """bars: list of (open, high, low, close) for one stock 'AAA'."""
    idx = pd.bdate_range("2025-01-01", periods=len(bars))
    a = np.array(bars, float)
    mk = lambda col: pd.DataFrame({"AAA": a[:, col]}, index=idx)
    P = {"O": mk(0), "H": mk(1), "L": mk(2), "C": mk(3)}
    f = {"ll20_prev": P["L"].shift(1).rolling(20, min_periods=1).min(),
         "atr14": pd.DataFrame({"AAA": 2.0}, index=idx),
         "regime": pd.Series("green", index=idx)}
    return P, f


def _run(bars, signal_day=9, stop=95.0, cfg=ZERO, setup="Breakout from base"):
    P, f = _panel(bars)
    orig = bt.signals

    def fake(f_, P_, i, cfg_, strategy, excl):
        if i == signal_day:
            return [{"symbol": "AAA", "close": float(P_["C"].iloc[i, 0]), "stop": stop, "setup": setup}], [], cfg_["risk_pct"]
        return [], [], cfg_["risk_pct"]
    bt.signals = fake
    try:
        return bt.run_backtest(P, f, cfg, "baseline")
    finally:
        bt.signals = orig


FLAT = [(100, 100.5, 99.5, 100)] * 10           # days 0–9, signal on day 9 close = 100


def test_stop_hit_is_minus_1R():
    res = _run(FLAT + [(100, 101, 99, 100), (99, 99.5, 94, 95)])
    t = res.trades.iloc[0]
    assert t["reason"] == "stop" and abs(t["exit"] - 95) < 1e-9 and abs(t["r"] + 1) < 1e-9, t.to_dict()
    assert t["qty"] == 200          # 1% of 1,00,000 = 1,000 / 5 per share


def test_gap_below_stop_fills_at_open():
    res = _run(FLAT + [(100, 101, 99, 100), (90, 91, 88, 89)])
    t = res.trades.iloc[0]
    assert t["reason"] == "stop (gap)" and abs(t["r"] + 2) < 1e-9, t.to_dict()


def test_breakeven_after_1R():
    res = _run(FLAT + [(100, 101, 99, 100), (101, 106, 100.5, 104), (103, 103.5, 99, 100)])
    t = res.trades.iloc[0]
    assert abs(t["r"]) < 1e-9 and "breakeven" in t["reason"], t.to_dict()


def test_partial_at_2R_then_trail():
    bars = FLAT + [(100, 101, 99, 100), (102, 111, 101, 109)] + [(109, 110, 108, 109)] * 3 + \
           [(109, 109, 98, 98.5), (97, 97, 96, 96.5)]
    res = _run(bars)
    t = res.trades.iloc[0]
    # half (100 sh) sold at 110, the rest exits at the next open after a close below the prior 20-day low
    assert t["qty"] == 200 and t["reason"] in ("trail", "breakeven/trail stop"), t.to_dict()
    assert t["pnl"] > 0


def test_drawdown_halt_is_temporary():
    """Regression: the 10% circuit breaker must pause trading, not end it forever."""
    idx = pd.bdate_range("2025-01-01", periods=400)
    # every 3 days: entry at 100, next day gap to 80 → a ~-4R loss each time (forces a deep drawdown)
    o = np.full(400, 100.0); h = np.full(400, 100.5); l = np.full(400, 99.5); c = np.full(400, 100.0)
    for k in range(10, 400, 3):
        o[k] = h[k] = l[k] = c[k] = 80.0
    mk = lambda a: pd.DataFrame({"AAA": a}, index=idx)
    P = {"O": mk(o), "H": mk(h), "L": mk(l), "C": mk(c)}
    f = {"ll20_prev": P["L"].shift(1).rolling(20, min_periods=1).min(),
         "atr14": pd.DataFrame({"AAA": 2.0}, index=idx), "regime": pd.Series("green", index=idx)}
    orig = bt.signals
    bt.signals = lambda f_, P_, i, cfg_, s, e: ([{"symbol": "AAA", "close": 100.0, "stop": 95.0,
                                                  "setup": "x"}], [], cfg_["risk_pct"]) if P_["C"].iloc[i, 0] == 100 else ([], [], cfg_["risk_pct"])
    try:
        res = bt.run_backtest(P, f, ZERO, "adaptive")
    finally:
        bt.signals = orig
    d = pd.to_datetime(res.trades["entry_date"])
    assert len(res.trades) > 5
    assert d.max() > idx[300], "trading never resumed after the drawdown halt"
    gaps = d.diff().dt.days.dropna()
    assert gaps.max() >= 20, "expected a cooling-off pause"


def test_accounting_identity_with_costs():
    hist, nifty = make_long_history(years=3, n_stocks=40)
    P = build_panel(hist); f = features(P, nifty, CFG)
    res = bt.run_backtest(P, f, CFG, "baseline")
    assert abs(res.equity.iloc[-1] - (CFG["bt_capital"] + res.trades["pnl"].sum())) < 1.0


def test_no_lookahead():
    hist, nifty = make_long_history(years=3, n_stocks=40)
    P = build_panel(hist); f = features(P, nifty, CFG)
    a = bt.run_backtest(P, f, CFG, "adaptive")
    cut = P["C"].index[-120]
    h2 = hist.copy()
    later = h2["date"] > cut
    rng = np.random.default_rng(1)
    for col in ("open", "high", "low", "close"):
        h2.loc[later, col] = h2.loc[later, col] * rng.uniform(0.6, 1.6, later.sum())
    h2.loc[later, "prev_close"] = np.nan
    n2 = nifty.copy(); n2[n2.index > cut] *= 0.5
    P2 = build_panel(h2); f2 = features(P2, n2, CFG)
    b = bt.run_backtest(P2, f2, CFG, "adaptive")
    done = lambda r: r.trades[pd.to_datetime(r.trades["exit_date"]) <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(done(a), done(b))
    assert len(done(a)) > 0


def test_split_adjustment():
    hist, _ = make_long_history()
    P = build_panel(hist)
    c = P["C"]["SPLITCO"]
    jumps = (c / c.shift(1) - 1).abs()
    assert jumps.max() < 0.2, f"split not adjusted: max daily move {jumps.max():.2%}"
    raw = hist[hist.symbol == "SPLITCO"].set_index("date")["close"]
    assert (raw / raw.shift(1) - 1).abs().max() > 0.4      # raw data really does contain the split


def test_baseline_matches_live_funnel():
    hist, nifty, excluded, names = make_history()
    live = run_funnel(hist, CFG, excluded, names)["candidates"]
    P = build_panel(hist); f = features(P, nifty, CFG)
    assert bool(f["market_ok"].iloc[-1])
    cands, _, _ = signals(f, P, len(P["C"]) - 1, CFG, "baseline", excluded)
    assert [c["symbol"] for c in cands] == [c["symbol"] for c in live]
    for x, y in zip(cands, live):
        assert abs(x["stop"] - y["stop"]) < 0.01


def test_report_runs_on_demo():
    import bt_report
    hist, nifty = make_long_history(years=3, n_stocks=40)
    rep = bt_report.run(hist, nifty, CFG, demo=True)
    assert set(rep["strategies"]) == {"baseline", "adaptive"}
    assert rep["verdict"]["recommended"] in ("baseline", "adaptive", "none")
    json.dumps(rep, default=str)




def test_paper_ledger_matches_backtest():
    """The live ledger, stepped one day at a time, must make exactly the trades the backtest makes."""
    import tempfile
    import ledger
    hist, nifty = make_long_history(years=2.2, n_stocks=40)
    hist = hist[hist.symbol != "SPLITCO"]
    P = build_panel(hist); f = features(P, nifty, CFG)
    dates = P["C"].index
    s0 = len(dates) - 140
    cfg = dict(CFG, capital=CFG["bt_capital"])
    for strategy in ("baseline", "adaptive"):
        path = Path(tempfile.mkdtemp()) / "ledger.json"
        for i in range(s0, len(dates)):
            d = dates[i]
            cands, _, _ = signals(f, P, i, cfg, strategy, set())
            picks = [{"symbol": c["symbol"], "stop": c["stop"], "setup": c["setup"], "entry": c["close"]} for c in cands]
            fd = {k: (v.loc[:d] if hasattr(v, "loc") else v) for k, v in f.items()}
            led = ledger.update(path, hist[hist.date <= d], fd, picks, cfg, strategy)
        res = bt.run_backtest(P, f, dict(cfg, bt_warmup=0), strategy, start=dates[s0])
        bt_tr = res.trades[res.trades.reason != "end of test"].reset_index(drop=True)
        led_tr = pd.DataFrame(led["sim"]["trades"])
        cols = ["symbol", "entry_date", "exit_date", "qty", "pnl", "reason"]
        assert len(bt_tr) > 3, strategy
        pd.testing.assert_frame_equal(bt_tr[cols], led_tr[cols], check_dtype=False)


if __name__ == "__main__":  # noqa
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
