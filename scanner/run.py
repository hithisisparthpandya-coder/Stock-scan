"""Daily scan entry point.

  python scanner/run.py            # real run (used by GitHub Actions)
  python scanner/run.py --demo     # synthetic data, writes demo picks.json, no network
  FORCE=1 python scanner/run.py    # scan latest available day even if today has no file yet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from funnel import apply_quality, market_filter, run_funnel  # noqa: E402
from notify import build_message, send_telegram  # noqa: E402
from strategy import build_panel, features, regime_info, signals  # noqa: E402

CACHE = ROOT / "data_cache"
HIST = CACHE / "history.csv.gz"
NIFTY = CACHE / "nifty.csv"
OUT = ROOT / "docs" / "data" / "picks.json"
IST = ZoneInfo("Asia/Kolkata")


def load_cfg() -> dict:
    cfg = json.loads((ROOT / "scanner" / "config.json").read_text())
    if os.environ.get("APP_URL"):
        cfg["app_url"] = os.environ["APP_URL"]
    return cfg


def write_result(result: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"Wrote {OUT}")


def update_history(nse, cfg, today: dt.date) -> tuple[pd.DataFrame, pd.Series]:
    CACHE.mkdir(exist_ok=True)
    hist = pd.read_csv(HIST, parse_dates=["date"]) if HIST.exists() else pd.DataFrame()
    if len(hist) and "open" not in hist.columns:      # cache from v1.0/1.1 lacks open/prev close: rebuild
        print("Old cache format: rebuilding history with open and previous-close prices")
        hist = pd.DataFrame()
    if len(hist):
        start = hist["date"].max().date() + dt.timedelta(days=1)
    else:
        hist = pd.DataFrame()
        start = today - dt.timedelta(days=cfg["backfill_calendar_days"])
        print(f"No cache found: backfilling from {start} (one-time, ~5-10 min)")
    nifty = pd.read_csv(NIFTY, parse_dates=["date"]).set_index("date")["close"] if NIFTY.exists() \
        else pd.Series(dtype=float)

    new_frames, new_nifty = [], {}
    d = start
    while d <= today:
        if d.weekday() < 5:
            df = nse.bhavcopy(d)
            if df is not None and len(df):
                new_frames.append(df)
                nc = nse.nifty_close(d)
                if nc:
                    new_nifty[pd.Timestamp(d)] = nc
                print(f"  {d}: {len(df)} stocks")
        d += dt.timedelta(days=1)

    if new_frames:
        hist = pd.concat([hist, *new_frames], ignore_index=True)
    if new_nifty:
        nifty = pd.concat([nifty, pd.Series(new_nifty)])
    if hist.empty:
        raise SystemExit("No price data could be downloaded from NSE.")

    hist = hist.drop_duplicates(["date", "symbol"], keep="last")
    keep = sorted(hist["date"].unique())[-cfg["history_trading_days"]:]
    hist = hist[hist["date"].isin(keep)]
    hist.to_csv(HIST, index=False, compression="gzip")
    nifty = nifty[~nifty.index.duplicated(keep="last")].sort_index()
    nifty.rename("close").rename_axis("date").to_csv(NIFTY)
    return hist, nifty


def to_pick(c: dict, cfg: dict, risk: float, names: dict) -> dict:
    """Strategy candidate → pick card, sized on the configured capital and the regime's risk."""
    import math
    entry, stop = c["close"], c["stop"]
    r = entry - stop
    qty = int(min(math.floor(cfg["capital"] * risk / r), math.floor(cfg["capital"] * cfg["max_position_pct"] / entry))) \
        if r > 0 and risk > 0 else 0
    return {"symbol": c["symbol"], "name": names.get(c["symbol"], ""), "setup": c["setup"],
            "low_priced": c["low_priced"], "entry": round(entry, 2), "stop": round(stop, 2),
            "target": round(entry + cfg["reward_risk"] * r, 2), "qty": qty, "capital": round(qty * entry, 0),
            "max_loss": round(qty * r, 0), "risk_pct": round(c["risk_pct"] * 100, 2),
            "vol_ratio": round(c["vol_ratio"], 2),
            "ret63_pct": round(c["ret63"] * 100, 1) if c["ret63"] is not None else None,
            "mom_pct": round(c["mom_pct"] * 100) if c["mom_pct"] is not None else None,
            "sma50": round(c["sma50"], 2), "sma200": round(c["sma200"], 2),
            "turnover_cr": round(c["turnover_cr"], 1), "score": round(c["score"] * 100, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    now = dt.datetime.now(IST)

    if args.demo:
        from demo import make_history, make_quality
        hist, nifty, excluded, names = make_history()
        bands, results, funds_demo = make_quality(now.date())
        surv_ok, circulars = True, [
            {"source": "SEBI", "title": "Demo circular – real ones appear here after the first live scan",
             "link": "https://www.sebi.gov.in/", "date": now.date().isoformat()}]
    else:
        from nse import NSE
        nse = NSE()
        hist, nifty = update_history(nse, cfg, now.date())
        excluded, surv_ok = nse.surveillance_symbols()
        names = nse.company_names()
        circulars = nse.circulars()
        bands = nse.price_bands()
        results = nse.upcoming_results()
        from nse import is_fund
        funds_listed = nse.etf_symbols()
        excluded = set(excluded) | funds_listed | {s for s in hist["symbol"].unique() if is_fund(s)}

    latest = hist["date"].max().date()
    if not args.demo and latest < now.date() and not os.environ.get("FORCE"):
        print(f"No NSE file for {now.date()} yet (latest {latest}). Holiday or not published; no alert.")
        return
    if not args.demo and OUT.exists() and not os.environ.get("FORCE"):
        prev = json.loads(OUT.read_text())
        if prev.get("as_of") == latest.isoformat() and not prev.get("demo"):
            print(f"Already scanned {latest}; nothing to do (backup run).")
            return

    P = build_panel(hist)
    feat = features(P, nifty, cfg)
    strategy = cfg.get("live_strategy", "baseline")
    if strategy == "adaptive":
        tight = {s for s, b in bands.items() if b in set(cfg.get("exclude_bands", [2, 5]))}
        raw, counts, risk = signals(feat, P, len(P["C"]) - 1, cfg, "adaptive", set(excluded) | tight)
        tech = {"as_of": P["C"].index[-1].date().isoformat(),
                "funnel": [{"stage": k, "count": v} for k, v in counts],
                "candidates": [to_pick(c, cfg, risk, names) for c in raw]}
        tech["candidates"] = [c for c in tech["candidates"] if c["qty"] >= 1]
    else:
        tech = run_funnel(hist, cfg, excluded, names, bands)
    cands = tech["candidates"][: cfg["fundamentals_max"]]
    if args.demo:
        funds = funds_demo
    else:
        from fundamentals import fetch
        print(f"Fetching fundamentals for {len(cands)} candidates")
        funds = fetch([c["symbol"] for c in cands], CACHE / "fundamentals.json")
    picks, qcounts = apply_quality(cands, funds, results, now.date(), cfg)
    covered = sum(1 for c in cands if funds.get(c["symbol"]))
    result = {"as_of": tech["as_of"], "funnel": tech["funnel"] + [{"stage": k, "count": v} for k, v in qcounts],
              "picks": picks}
    result["market"] = market_filter(nifty)
    result["regime"] = regime_info(feat)
    result["strategy"] = strategy
    result["surveillance_checked"] = bool(surv_ok)
    result["circulars"] = circulars
    from ledger import summary_line, update as update_ledger
    led = update_ledger(OUT.parent / "ledger.json", hist, feat, picks, cfg, strategy, demo=args.demo)
    result["paper"] = {"summary": summary_line(led), "events": led["events_latest"][-10:],
                       "queued": led["queued"]}
    try:   # did past picks become top performers?
        from scorecard import summary_line as sc_line, update as update_scorecard
        sc = update_scorecard(OUT.parent / "scorecard.json", P, feat, cfg, tech["as_of"], picks, demo=args.demo)
        result["scorecard"] = {"summary": sc["summary"], "line": sc_line(sc)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        result.setdefault("warnings_internal", []).append(f"Scorecard failed: {type(e).__name__}")
    try:   # sentiment layer: universe for the live job, Market Mood, sentiment-adjusted list + its paper ledger
        from market_intel import evening_intel
        universe, sent = evening_intel(None if args.demo else nse, P, feat, picks, names, bands, results, cfg, now,
                                       demo=args.demo)
        (OUT.parent / "universe.json").write_text(json.dumps(universe, ensure_ascii=False, default=str))
        chosen = [x for x in sent["list"] if x["status"] == "suggested"]
        sled = update_ledger(OUT.parent / "ledger_sentiment.json", hist, feat, chosen, cfg, strategy,
                             demo=args.demo, risk_mult=sent["mood"]["risk_mult"])
        sent["paper"] = {"summary": summary_line(sled).replace("Paper account", "Sentiment paper account"),
                         "queued": sled["queued"]}
        result["sentiment"] = sent
    except Exception as e:  # never lose the main scan over the optional layer
        import traceback
        traceback.print_exc()
        result.setdefault("warnings_internal", []).append(f"Sentiment layer failed: {type(e).__name__}")
    result["generated_at"] = now.isoformat(timespec="minutes")
    result["config"] = {k: cfg[k] for k in ("capital", "risk_pct", "max_position_pct", "reward_risk",
                                            "min_turnover_cr", "vol_mult", "max_stop_pct", "min_roe",
                                            "max_de", "min_mcap_cr", "results_window_days")}
    result["demo"] = bool(args.demo)
    warnings = []
    if not surv_ok:
        warnings.append("NSE ASM/GSM lists could not be fetched today – check each pick on nseindia.com.")
    if not bands:
        warnings.append("NSE price-band list unavailable today – check that a pick isn't in a 2% or 5% band.")
    if not results:
        warnings.append("NSE results calendar unavailable today – results dates come from Yahoo where available; "
                        "confirm on nseindia.com before buying.")
    if cands and covered < len(cands):
        warnings.append(f"Fundamentals found for {covered} of {len(cands)} candidates; "
                        "picks marked ⚠ need a manual check on screener.in.")
    if warnings:
        result["warnings"] = warnings
    result["checks"] = {"surveillance": bool(surv_ok), "price_bands": bool(bands),
                        "results_calendar": bool(results), "fundamentals_covered": covered,
                        "fundamentals_checked": len(cands)}
    write_result(result)

    if not args.no_telegram:
        send_telegram(build_message(result, cfg))


if __name__ == "__main__":
    main()
