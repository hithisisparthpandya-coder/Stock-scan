"""Backtest report: baseline vs regime-adaptive, in-sample vs out-of-sample, robustness, verdict.

  python scanner/bt_report.py           # real NSE data (GitHub Actions; first run downloads ~10 years)
  python scanner/bt_report.py --demo    # synthetic data, no network
Writes docs/data/backtest.json, which the app's Strategy Lab tab shows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import group_stats, metrics, run_backtest, slice_metrics, yearly_returns  # noqa: E402
from strategy import build_panel, features  # noqa: E402

OUT = ROOT / "docs" / "data" / "backtest.json"
LONG = ROOT / "data_cache" / "long"
IST = ZoneInfo("Asia/Kolkata")

LABELS = {"baseline": "Current funnel (baseline)", "adaptive": "Regime-adaptive momentum"}

ROBUSTNESS = [
    ("Volume surge 1.3×", {"vol_mult": 1.3}),
    ("Volume surge 1.8×", {"vol_mult": 1.8}),
    ("Max stop 2 ATR", {"atr_stop_max": 2.0}),
    ("Momentum top 30%", {"mom_pct_green": 0.7, "mom_pct_amber": 0.85}),
    ("Risk 0.75% per trade", {"risk_pct": 0.0075}),
]


def has_edge(oos: dict, ins: dict, cfg: dict) -> bool:
    return bool(oos.get("trades", 0) >= cfg["edge_min_trades"]
                and (oos.get("avg_r") or 0) >= cfg["edge_min_avg_r"]
                and (oos.get("profit_factor") or 0) >= cfg["edge_min_pf_oos"]
                and (ins.get("profit_factor") or 0) >= cfg["edge_min_pf_is"])


def verdict(rep: dict, cfg: dict) -> dict:
    b, a = rep["strategies"]["baseline"], rep["strategies"]["adaptive"]
    eb = has_edge(b["out_of_sample"], b["in_sample"], cfg)
    ea = has_edge(a["out_of_sample"], a["in_sample"], cfg)
    robust_ok = sum(1 for r in rep["robustness"] if (r.get("profit_factor") or 0) >= cfg["edge_min_pf_is"])
    robust = robust_ok >= len(rep["robustness"]) - 1
    lines = []
    if ea and robust and (not eb or (a["out_of_sample"]["avg_r"] >= b["out_of_sample"]["avg_r"]
                                     and a["out_of_sample"]["max_drawdown_pct"] >= b["out_of_sample"]["max_drawdown_pct"])):
        choice = "adaptive"
        lines.append("Regime-adaptive momentum held its edge on years it was not designed on, with a smaller "
                     "or equal drawdown, and survived the robustness checks. Switch the live scanner to it.")
    elif eb:
        choice = "baseline"
        lines.append("The current funnel shows the more reliable edge out of sample. Keep it live.")
        if ea and not robust:
            lines.append("The adaptive version looked good but broke when its settings were nudged – a sign of luck.")
    else:
        choice = "none"
        lines.append("Neither strategy cleared the bar on unseen years. Do not trade real money with them; "
                     "keep paper trading, or put the money in a Nifty index fund until a version passes.")
    lines.append(f"Robustness: {robust_ok} of {len(rep['robustness'])} setting changes kept a profit factor "
                 f"≥ {cfg['edge_min_pf_is']}.")
    return {"recommended": choice, "baseline_has_edge": eb, "adaptive_has_edge": ea,
            "robust": robust, "summary": lines}


def run(hist: pd.DataFrame, nifty: pd.Series, cfg: dict, demo: bool) -> dict:
    t0 = time.time()
    P = build_panel(hist)
    f = features(P, nifty, cfg)
    dates = P["C"].index
    split = pd.Timestamp(cfg["bt_oos_start"]) if not demo else dates[int(len(dates) * 0.55)]
    print(f"Panel {P['C'].shape} in {time.time() - t0:.0f}s; out-of-sample from {split.date()}")

    rep = {"generated_at": dt.datetime.now(IST).isoformat(timespec="minutes"), "demo": demo,
           "data": {"start": dates[0].date().isoformat(), "end": dates[-1].date().isoformat(),
                    "symbols": int(P["C"].shape[1]), "trading_days": int(len(dates))},
           "oos_start": split.date().isoformat(), "capital": cfg["bt_capital"],
           "assumptions": [
               f"Starting capital ₹{cfg['bt_capital']:,}; whole shares; no leverage.",
               "Signals on the close, bought at the next open; skipped if the open gaps more than "
               f"{cfg['gap_tolerance']*100:.0f}% above.",
               f"Costs {cfg['cost_per_side']*100:.2f}% + slippage {cfg['slippage_per_side']*100:.2f}% per side, "
               f"plus ₹{cfg['dp_charge']} per sale.",
               "Prices adjusted for splits and bonuses; delisted stocks included (no survivorship bias).",
               "Fundamentals, results dates and ASM/GSM lists are NOT applied – no point-in-time history "
               "is freely available. Live picks get these extra filters, so live results may differ.",
               "Settings are fixed, textbook values – nothing was tuned to this data.",
           ],
           "strategies": {}}
    for name in ("baseline", "adaptive"):
        t1 = time.time()
        res = run_backtest(P, f, cfg, name)
        e = res.equity
        rep["strategies"][name] = {
            "label": LABELS[name],
            "all": metrics(e, res.trades),
            "in_sample": slice_metrics(res, end=split),
            "out_of_sample": slice_metrics(res, start=split),
            "yearly": yearly_returns(e),
            "by_regime": group_stats(res.trades, "regime"),
            "by_setup": group_stats(res.trades, "setup"),
            "by_exit": group_stats(res.trades, "reason"),
            "equity": {d.date().isoformat(): round(v) for d, v in e.resample("W").last().dropna().items()},
            "recent_trades": res.trades.tail(25).to_dict("records") if len(res.trades) else [],
        }
        print(f"{name}: {rep['strategies'][name]['all']} ({time.time() - t1:.0f}s)")
        if name == "adaptive":
            rep["regime_share"] = {k: round(v * 100, 1) for k, v in res.regimes.value_counts(normalize=True).items()}
            first = e.index[0]

    n = nifty.reindex(dates).ffill()
    n = n[n.index >= first]
    rep["nifty"] = {
        "equity": {d.date().isoformat(): round(v / n.iloc[0] * cfg["bt_capital"])
                   for d, v in n.resample("W").last().dropna().items()},
        "all": {**metrics(n / n.iloc[0] * cfg["bt_capital"], None), "trades": None},
        "yearly": yearly_returns(n),
    }

    rob = []
    for label, over in ROBUSTNESS:
        c2 = dict(cfg, **over)          # these settings only affect signals, so features are reused
        res = run_backtest(P, f, c2, "adaptive")
        m = metrics(res.equity, res.trades)
        rob.append({"label": label, **{k: m.get(k) for k in ("cagr_pct", "max_drawdown_pct", "win_rate_pct",
                                                               "avg_r", "profit_factor", "trades")}})
        print("robustness", label, rob[-1])
    rep["robustness"] = rob
    rep["verdict"] = verdict(rep, cfg)
    print("Verdict:", rep["verdict"])
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "scanner" / "config.json").read_text())
    if args.demo:
        from demo import make_long_history
        hist, nifty = make_long_history()
    else:
        from bt_data import liquid_subset, load_long
        from nse import NSE
        hist, nifty = load_long(NSE(), cfg["bt_start_year"], LONG, dt.datetime.now(IST).date())
        hist = liquid_subset(hist, cfg["min_turnover_cr"])
    rep = run(hist, nifty, cfg, args.demo)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
