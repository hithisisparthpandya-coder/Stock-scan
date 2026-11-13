"""Mutual fund engine tests. Run: python scanner/test_mf.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mf  # noqa: E402
from mf_classify import bucket_for, parse_amfi_navall  # noqa: E402


def _nav(monthly_growth=0.01, years=12, start=100.0):
    # calendar-day NAVs so the growth per calendar year is exact
    idx = pd.date_range(end="2026-11-12", periods=int(365.25 * years), freq="D")
    t = (idx - idx[0]).days.values / 365.25
    return pd.Series(start * ((1 + monthly_growth) ** 12) ** t, index=idx)


def test_cagr_exact():
    s = _nav(0.01)
    want = 1.01 ** 12 - 1
    assert abs(mf.cagr(s, 5) - want) < 0.002, mf.cagr(s, 5)
    assert mf.cagr(_nav(0.01, years=2), 5) is None          # not enough history


def test_xirr_and_sip():
    assert abs(mf.xirr([(pd.Timestamp("2020-01-01"), -100), (pd.Timestamp("2021-01-01"), 110)]) - 0.0997) < 0.002
    s = _nav(0.01)
    m = s.resample("ME").last()
    assert abs(mf.sip_xirr(m, 60) - (1.01 ** 12 - 1)) < 0.01   # steady 1%/month → ~12.7% XIRR
    flat = pd.Series(10.0, index=m.index)
    assert abs(mf.sip_xirr(flat, 60)) < 1e-4


def test_max_drawdown():
    s = pd.Series([100, 120, 60, 90, 130], index=pd.date_range("2020", periods=5, freq="ME"), dtype=float)
    assert abs(mf.max_drawdown(s) + 0.5) < 1e-9


def test_projection_maths():
    """Constant 1%/month: lump multiple = 1.01^n; SIP of ₹1 at each month start = sum of 1.01^k, k=1..n."""
    s = _nav(0.01, years=12)
    ranked = {"idx_n50": [{"code": 1, "score": 1}]}
    pr = mf.projections({"idx_n50": 100}, ranked, {1: s}, horizons=(5,))
    n = 60
    assert abs(pr["lump"]["5"]["p50"] - 1.01 ** n) < 0.01, pr["lump"]
    want_sip = sum(1.01 ** k for k in range(1, n + 1))
    assert abs(pr["sip"]["5"]["p50"] - want_sip) < 0.2, (pr["sip"], want_sip)


def test_index_fund_ranking_prefers_cheaper_tracking():
    """Same index, different costs: the higher 3-year return (lower cost) must rank first."""
    funds = {}
    for code, g in ((1, 0.0095), (2, 0.0100), (3, 0.0098)):
        funds[code] = {"name": f"F{code}", "m": mf.fund_metrics(_nav(g, years=6))}
    funds[4] = {"name": "Young", "m": mf.fund_metrics(_nav(0.02, years=1.5))}   # too young to judge
    ranked = mf.score_bucket(funds, "index")
    assert [f["code"] for f in ranked[:3]] == [2, 3, 1]
    assert ranked[-1]["code"] == 4


def test_active_scoring_rewards_consistency_over_luck():
    idx = pd.bdate_range(end="2026-11-12", periods=252 * 8)
    rng = np.random.default_rng(3)
    steady = pd.Series(100 * np.exp(np.cumsum(np.full(len(idx), 0.13 / 252))), index=idx)
    noisy_r = rng.normal(0.11 / 252, 0.03, len(idx)); noisy_r[-60:] += 0.004       # very volatile, lucky spurt
    noisy = pd.Series(100 * np.exp(np.cumsum(noisy_r)), index=idx)
    meh = pd.Series(100 * np.exp(np.cumsum(np.full(len(idx), 0.08 / 252))), index=idx)
    funds = {1: {"name": "Steady", "m": mf.fund_metrics(steady)}, 2: {"name": "Lucky", "m": mf.fund_metrics(noisy)},
             3: {"name": "Meh", "m": mf.fund_metrics(meh)}}
    ranked = mf.score_bucket(funds, "active")
    # Even though this seed makes "Lucky" return more, a −60% drawdown at 50%+ volatility must not rank first.
    assert ranked[0]["name"] == "Steady", [(f["name"], f["score"]) for f in ranked]


def test_classification_on_real_names():
    cases = {
        "Navi Nifty 50 Index Fund-Direct Plan-Growth": "idx_n50",
        "Nippon India Index Fund - Nifty 50 Plan - Direct Plan Growth Plan": "idx_n50",
        "UTI Nifty Next 50 Index Fund - Direct Plan - Growth Option": "idx_next50",
        "Motilal Oswal Nifty Midcap 150 Index Fund - Direct Plan": None,               # no "growth" in name
        "Motilal Oswal Nifty Midcap 150 Index Fund - Direct Plan - Growth": "idx_mid150",
        "UTI Nifty 200 Momentum 30 Index Fund - Direct Plan - Growth Option": "idx_mom30",
        "UTI Nifty50 Equal Weight Index Fund - Direct Plan - Growth": None,
        "NAVI ELSS TAX SAVER NIFTY 50 INDEX FUND - DIRECT PLAN GROWTH": None,
        "HDFC Flexi Cap Fund - Growth Option - Direct Plan": "flexi",
        "Motilal Oswal Diversified Equity Flexicap Passive Fund of Funds- Direct - Growth": None,
        "ICICI Prudential US Bluechip Equity Fund - Direct Plan - Growth": None,
        "Nippon India Large Cap Fund - Direct Plan Growth Plan - Growth Option": "large",
        "Mirae Asset Large & Midcap Fund - Direct Plan - Growth": None,
        "DSP World Gold Mining Overseas Equity Omni FoF - Direct Plan - Growth": None,
        "Nippon India Gold Savings Fund - Direct Plan Growth Plan - Growth Option": "gold",
        "SBI ELSS Tax Saver FUND - DIRECT PLAN -GROWTH": "elss",
        "HDFC Flexi Cap Fund - IDCW Option - Direct Plan": None,
        "HDFC Flexi Cap Fund - Growth Option - Regular Plan": None,
    }
    for name, want in cases.items():
        assert bucket_for(name, None) == want, (name, bucket_for(name, None), want)
    # AMFI category wins over the name for active funds; names refine index funds
    assert bucket_for("XYZ Bluechip Fund - Direct Plan - Growth", "Equity Scheme - Large Cap Fund") == "large"
    assert bucket_for("XYZ Nifty Next 50 Index Fund - Direct - Growth", "Other Scheme - Index Funds") == "idx_next50"


def test_parse_amfi_navall():
    txt = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Flexi Cap Fund)

HDFC Mutual Fund

118955;INF179K01UT0;-;HDFC Flexi Cap Fund - Growth Option - Direct Plan;2100.1234;12-Nov-2026

Close Ended Schemes(ELSS)

ABC Mutual Fund

999999;X;-;ABC Closed ELSS - Direct Growth;10.0;12-Nov-2026
"""
    d = parse_amfi_navall(txt)
    assert d == {118955: {"name": "HDFC Flexi Cap Fund - Growth Option - Direct Plan",
                          "category": "Equity Scheme - Flexi Cap Fund", "amc": "HDFC Mutual Fund",
                          "nav": "2100.1234", "date": "12-Nov-2026"}}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
