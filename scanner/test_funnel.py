"""Run: python -m pytest scanner -q   (or python scanner/test_funnel.py)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import datetime as dt  # noqa: E402

from demo import make_history, make_quality  # noqa: E402
from funnel import apply_quality, check_fundamentals, market_filter, run_funnel  # noqa: E402
from nse import parse_event_calendar, parse_rss, parse_sec_bhavdata, parse_sec_list, _collect_symbols  # noqa: E402
from notify import build_message  # noqa: E402

CFG = json.loads((Path(__file__).parent / "config.json").read_text())


TODAY = dt.date(2026, 11, 13)


def _result():
    hist, nifty, excluded, names = make_history()
    bands, results, funds = make_quality(TODAY)
    tech = run_funnel(hist, CFG, excluded, names, bands)
    picks, q = apply_quality(tech["candidates"], funds, results, TODAY, CFG)
    tech["picks"] = picks
    tech["funnel"] += [{"stage": k, "count": v} for k, v in q]
    return tech, nifty


def test_known_winners_and_losers():
    res, _ = _result()
    syms = {p["symbol"] for p in res["picks"]}
    assert "BRKOUT" in syms, syms
    assert "PULLBK" in syms, syms
    assert "LOWPX" in syms, "a liquid, fundamentally sound low-priced stock should be allowed"
    for bad in ("DOWNTR", "ILLIQ", "SURVL", "NOVOL", "WIDESTP", "LOWTHIN", "BAND5", "RESULTS", "WEAKFUND"):
        assert bad not in syms, f"{bad} should be filtered out"


def test_quality_flags_and_order():
    res, _ = _result()
    by = {p["symbol"]: p for p in res["picks"]}
    assert by["NODATA"]["fund_status"] == "unknown"
    assert by["BRKOUT"]["fund_status"] == "pass" and by["BRKOUT"]["next_results"] == "2026-12-13"
    assert by["LOWPX"]["low_priced"] is True
    statuses = [p["fund_status"] for p in res["picks"]]
    assert statuses == sorted(statuses, key=lambda s: s != "pass"), "verified picks must rank first"


def test_check_fundamentals_rules():
    good = {"roe": 0.2, "de": 0.5, "profitable": True, "mcap_cr": 1000, "sector": "Industrials"}
    assert check_fundamentals(good, CFG)[0] == "pass"
    assert check_fundamentals(dict(good, de=1.8), CFG)[0] == "fail"
    assert check_fundamentals(dict(good, profitable=False), CFG)[0] == "fail"
    assert check_fundamentals(dict(good, mcap_cr=200), CFG)[0] == "fail"
    assert check_fundamentals(dict(good, roe=None), CFG)[0] == "unknown"
    bank = dict(good, de=None, sector="Financial Services")
    assert check_fundamentals(bank, CFG)[0] == "pass", "D/E is skipped for banks/NBFCs"
    assert check_fundamentals({}, CFG)[0] == "unknown"
    # a results date today is inside the window; one beyond it is not
    c = [{"symbol": "A", "score": 1}, {"symbol": "B", "score": 2}]
    picks, _ = apply_quality(c, {"A": good, "B": good},
                             {"A": TODAY, "B": TODAY + dt.timedelta(days=CFG["results_window_days"] + 1)}, TODAY, CFG)
    assert [p["symbol"] for p in picks] == ["B"]


def test_setups_labelled():
    res, _ = _result()
    by = {p["symbol"]: p for p in res["picks"]}
    assert by["BRKOUT"]["setup"] == "Breakout from base"
    assert by["PULLBK"]["setup"] == "Pullback to support"


def test_risk_maths_match_planner_rules():
    res, _ = _result()
    risk_budget = CFG["capital"] * CFG["risk_pct"]
    max_pos = CFG["capital"] * CFG["max_position_pct"]
    for p in res["picks"]:
        r = p["entry"] - p["stop"]
        assert r > 0
        assert p["risk_pct"] <= CFG["max_stop_pct"] * 100 + 1e-9
        assert p["qty"] >= 1
        assert p["max_loss"] <= risk_budget + 1          # never more than 1% of capital
        assert p["capital"] <= max_pos + 1               # never more than 25% in one stock
        assert abs(p["target"] - (p["entry"] + 2 * r)) < 0.02


def test_funnel_is_monotonic():
    res, _ = _result()
    counts = [f["count"] for f in res["funnel"]]
    assert counts == sorted(counts, reverse=True), counts
    assert len(res["picks"]) <= CFG["top_n"]
    assert len(res["picks"]) <= counts[-1]


def test_market_filter():
    _, nifty = _result()
    assert market_filter(nifty)["ok"] is True
    falling = pd.Series(range(100, 0, -1), index=pd.bdate_range("2026-01-01", periods=100), dtype=float)
    assert market_filter(falling)["ok"] is False


def test_parsers():
    csv = (b"SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE,"
           b" AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
           b"RELIANCE, EQ, 13-Nov-2026, 1400, 1401, 1420, 1395, 1410, 1412, 1408, 1000000, 14080.5, 50000, 500000, 50\n"
           b"ABCBOND, N1, 13-Nov-2026, 100, 100, 100, 100, 100, 100, 100, 10, 0.01, 1, 10, 100\n")
    import datetime as dt
    df = parse_sec_bhavdata(csv, dt.date(2026, 11, 13))
    assert list(df["symbol"]) == ["RELIANCE"]
    assert abs(df["turnover_cr"].iloc[0] - 140.805) < 1e-6
    rss = b"""<rss><channel><item><title>Circular A</title><link>https://x/a</link>
    <pubDate>Thu, 12 Nov 2026 10:00:00 +0530</pubDate></item></channel></rss>"""
    items = parse_rss(rss)
    assert items[0]["title"] == "Circular A" and items[0]["published"] is not None
    assert _collect_symbols({"a": {"data": [{"symbol": "X "}, {"symbol": "Y"}]}}) == {"X", "Y"}
    sl = b"Symbol,Series,Security Name,Band,Remarks\nAAA,EQ,Aaa Ltd,20,\nBBB,EQ,Bbb Ltd,5,\nCCC,BE,Ccc Ltd,2,\n"
    assert parse_sec_list(sl) == {"AAA": 20.0, "BBB": 5.0}
    ev = [{"symbol": "AAA", "purpose": "Financial Results", "date": "20-Nov-2026"},
          {"symbol": "AAA", "purpose": "Financial Results/Dividend", "date": "18-Nov-2026"},
          {"symbol": "BBB", "purpose": "Fund Raising", "date": "19-Nov-2026"}]
    assert parse_event_calendar(ev) == {"AAA": dt.date(2026, 11, 18)}


def test_funds_are_recognised():
    from nse import is_fund
    for f in ("LIQUIDBEES", "LIQUID1", "CASHIETF", "MON100", "SBILIQETF", "GOLDBEES", "NIFTYBEES", "SETFNIF50", "BHARATBOND"):
        assert is_fund(f), f
    for st in ("RELIANCE", "GOLDIAM", "CASTROLIND", "APLAPOLLO", "MARKSANS", "GILLETTE", "MIDHANI", "ABSLAMC", "CAMS"):
        assert not is_fund(st), st


def test_message_builds():
    res, nifty = _result()
    res["market"] = market_filter(nifty)
    msg = build_message(res, CFG)
    assert "BRKOUT" in msg and len(msg) < 4096


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
