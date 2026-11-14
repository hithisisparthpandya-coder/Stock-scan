"""Tests for the sentiment engine, movers analysis and the live job (demo mode, no network).
Run: python scanner/test_sentiment.py"""
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import sentiment as S  # noqa: E402
from market_intel import (IST, explain_move, parse_all_indices, parse_fii_dii, parse_index_quotes,  # noqa: E402
                          parse_market_status)

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 11, 16, 6, 0, tzinfo=UTC)


def test_headline_signs():
    pos = ["Tata Motors profit jumps 40%, beats estimates", "L&T bags order worth Rs 5,000 crore",
           "Brokerage upgrades Infosys to buy, raises target", "Sun Pharma gets USFDA approval for drug"]
    neg = ["HDFC Bank shares fall after downgrade", "Paytm loss widens; shares slump",
           "SEBI order bars promoters of XYZ", "Company defaults on bond payment"]
    for h in pos:
        assert S.score_headline(h) > 0.3, h
    for h in neg:
        assert S.score_headline(h) < -0.3, h
    assert S.score_headline("Nifty ends flat in a quiet session") == 0
    for h in pos + neg:
        assert -1 <= S.score_headline(h) <= 1


def test_negation_and_denial():
    assert S.score_headline("Company does not see margin pressure") > 0
    assert S.score_headline("Firm fails to win order") < 0
    assert -1 < S.score_headline("Company denies fraud allegations") < 0     # denial is still not good news
    assert S.red_flag("Auditor resigns at XYZ") and S.red_flag("ED raids promoter premises")
    assert not S.red_flag("Company denies fraud allegations")


def test_specific_phrase_wins_over_generic():
    # "profit jumps" (+2) must be matched before the generic "jumps" (+1.2)
    assert S.score_headline("Profit jumps") > S.score_headline("Shares jump")


def test_stock_sentiment_decay_and_matching():
    keys = S.name_keys("TATAMOTORS", "Tata Motors Limited")
    fresh = [{"title": "Tata Motors bags order", "published": NOW - dt.timedelta(hours=1)}]
    old = [{"title": "Tata Motors bags order", "published": NOW - dt.timedelta(hours=48)}]
    other = [{"title": "Maruti bags order", "published": NOW}]
    a, b = S.stock_sentiment(fresh, keys, NOW), S.stock_sentiment(old, keys, NOW)
    assert a["N"] > b["N"] > 0
    assert S.stock_sentiment(other, keys, NOW)["count"] == 0
    assert S.stock_sentiment([{"title": "Tata Motors bags order", "published": NOW - dt.timedelta(days=5)}], keys, NOW)["count"] == 0
    assert "itc" in S.name_keys("ITC", "ITC Limited")
    rp = S.name_keys("RPOWER", "Reliance Power Limited")
    assert not S.mentions("Reliance Industries shares rebound 15%", rp) and S.mentions("Reliance Power hits upper circuit", rp)
    assert S.mentions("Larsen & Toubro bags order", S.name_keys("LT", "Larsen & Toubro Limited"))


def test_market_mood_bounds_and_bands():
    lo = S.market_mood(breadth=0.1, advances=50, declines=450, vix=30, nifty=90, nifty_sma50=100, news_avg=-0.8, flows_cr=-8000)
    hi = S.market_mood(breadth=0.9, advances=450, declines=50, vix=10, nifty=110, nifty_sma50=100, news_avg=0.8, flows_cr=8000)
    assert 0 <= lo["score"] < 35 and lo["band"] == "fear" and lo["risk_mult"] == 0 and lo["max_new"] == 0
    assert 65 <= hi["score"] <= 100 and hi["risk_mult"] == 1.0
    part = S.market_mood(breadth=0.5)
    assert part["score"] == 50 and len(part["components"]) == 1
    assert S.market_mood()["score"] is None
    for m in (lo, hi, part):
        assert m["risk_mult"] <= 1.0          # mood can never raise risk


def test_adjusted_list_veto_and_limits():
    picks = [{"symbol": "AAA", "score": 80, "entry": 100, "stop": 95}, {"symbol": "BBB", "score": 70, "entry": 50, "stop": 48},
             {"symbol": "CCC", "score": 60, "entry": 20, "stop": 19}, {"symbol": "DDD", "score": 90, "entry": 10, "stop": 9.5}]
    news = {"CCC": {"N": 0.8, "headlines": [], "red_flag": False, "count": 2},
            "DDD": {"N": -0.2, "headlines": [], "red_flag": True, "count": 1}}
    cautious = S.market_mood(breadth=0.42)
    out = S.adjusted_list(picks, news, {}, cautious)
    st = {x["symbol"]: x["status"] for x in out}
    assert st["DDD"] == "vetoed"
    assert sum(1 for x in out if x["status"] == "suggested") == 2         # cautious → max 2
    assert out[0]["symbol"] == "AAA" and out[1]["symbol"] == "CCC"         # 60 + 15·0.8 = 72 > 70
    fear = S.adjusted_list(picks, news, {}, S.market_mood(breadth=0.1))
    assert not any(x["status"] == "suggested" for x in fear)


def test_explain_move_reasons():
    q = {"pchange": 6.0, "open": 104, "prev_close": 100, "last": 106, "high": 106.2, "low": 103.5, "volume": 900_000,
         "year_high": 106.1, "year_low": 60}
    ctx = {"avgvol20": 200_000, "sma50": 95, "hh20": 104, "ll20": 90, "band": None}
    peer = {"industry": "Metals", "median": 2.4, "n": 8}
    news = S.stock_sentiment([{"title": "XYZ bags order worth Rs 900 crore", "published": NOW}], [], NOW, already_matched=True)
    r = explain_move("XYZ", q, ctx, peer, news, [], None, 0.5, dt.date(2026, 11, 16))
    assert r["verdict"].startswith("Likely news")
    txt = " ".join(r["reasons"])
    assert "Heavy volume" in txt and "Sector-wide" in txt
    quiet = explain_move("XYZ", {"pchange": -4, "last": 96, "prev_close": 100}, {}, None, None, [], None, 1.0, dt.date(2026, 11, 16))
    assert quiet["verdict"] == "No clear driver found"


def test_nse_parsers():
    data = {"advance": {"advances": "300", "declines": "190", "unchanged": "10"},
            "data": [{"symbol": "NIFTY 500", "priority": 1, "lastPrice": 23000},
                     {"symbol": "ABC", "priority": 0, "open": 10, "dayHigh": 11, "dayLow": 9.5, "lastPrice": "1,050.5",
                      "previousClose": 10, "pChange": 5, "totalTradedVolume": 1000, "totalTradedValue": 5e7,
                      "meta": {"companyName": "Abc Ltd", "industry": "Metals"}},
                     {"symbol": "DEF", "priority": 0, "lastPrice": 20, "previousClose": 21, "pChange": -4.76,
                      "meta": {"companyName": "Def Ltd", "industry": "IT"}}]}
    q, ad = parse_index_quotes(data)
    assert q.loc["ABC", "last"] == 1050.5 and q.loc["ABC", "industry"] == "Metals" and abs(q.loc["ABC", "value_cr"] - 5) < 1e-9
    assert ad["advances"] == 300 and "NIFTY 500" not in q.index
    assert parse_all_indices({"data": [{"index": "INDIA VIX", "last": 13.2, "percentChange": -1.1}]})["INDIA VIX"]["last"] == 13.2
    assert parse_market_status({"marketState": [{"market": "Capital Market", "marketStatus": "Open"}]})["status"] == "Open"
    f = parse_fii_dii([{"category": "FII/FPI *", "netValue": "-1234.5", "date": "13-Nov-2026"},
                       {"category": "DII **", "netValue": "2000", "date": "13-Nov-2026"}])
    assert f["fii"] == -1234.5 and f["dii"] == 2000


def test_scorecard_ranks_the_best_stock_first():
    import json as _j
    from demo import make_long_history
    from strategy import build_panel, features
    import scorecard
    cfg = _j.loads((Path(__file__).parent / "config.json").read_text())
    hist, nifty = make_long_history(years=2, n_stocks=80)
    P = build_panel(hist); f = features(P, nifty, cfg)
    t = len(P["C"]) - 30
    d = P["C"].index[t].date().isoformat()
    r5 = (P["C"].iloc[t + 5] / P["O"].iloc[t + 1] - 1).where(f["avgturn20"].iloc[t] >= cfg["min_turnover_cr"]).dropna()
    best, worst = r5.idxmax(), r5.idxmin()
    ev = scorecard.evaluate(P, f, cfg, d, [best, worst])
    rows = {x["symbol"]: x for x in ev["5d"]["picks"]}
    assert rows[best]["rank"] == 1 and rows[best]["pctile"] == 100 and rows[worst]["pctile"] < 5
    s = scorecard.summarize([{"date": d, "symbols": [best, worst], "eval": ev}])
    assert s["5d"]["hit_pct"] == 50.0 and s["5d"]["in_top50"] >= 1


def test_live_and_evening_demo_end_to_end():
    """Runs the evening demo scan and the live job in a scratch copy of the repo; checks live.json."""
    root = Path(__file__).resolve().parents[1]
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(root / "scanner", tmp / "scanner", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp / "docs" / "data").mkdir(parents=True)
    for cmd in (["scanner/run.py", "--demo", "--no-telegram"], ["scanner/live.py", "--demo"]):
        p = subprocess.run([sys.executable, *cmd], cwd=tmp, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    picks = json.loads((tmp / "docs/data/picks.json").read_text())
    assert picks["sentiment"]["mood"]["score"] is not None
    assert (tmp / "docs/data/universe.json").exists() and (tmp / "docs/data/ledger_sentiment.json").exists()
    assert (tmp / "docs/data/scorecard.json").exists() and "scorecard" in picks
    live = json.loads((tmp / "docs/data/live.json").read_text())
    assert len(live["movers"]["gainers"]) == 10 and len(live["movers"]["losers"]) == 10
    assert all(g["reasons"] for g in live["movers"]["gainers"][:3])
    assert live["mood"]["score"] is not None and live["watch"]
    assert "NaN" not in (tmp / "docs/data/live.json").read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
