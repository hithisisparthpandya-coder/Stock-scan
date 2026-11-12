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

from backtest import (group_stats, metrics, portfolio_metrics, run_backtest,  # noqa: E402
                      run_momentum_portfolio, slice_metrics, yearly_returns)
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


# Research queue item (a). Fixed, textbook settings – the midpoint of the 10–20 range asked for and
# the index's 200-stock universe. Not tuned. Never used by the live scanner.
MOMPORT = {"n_hold": 15, "universe": 200}


def research_momentum_portfolio(P: dict, f: dict, cfg: dict, n: pd.Series, split) -> dict:
    t1 = time.time()
    res = run_momentum_portfolio(P, f, cfg, **MOMPORT)
    e = res.equity
    ne = n.reindex(e.index).ffill() / n.reindex(e.index).ffill().iloc[0] * cfg["bt_capital"]
    def side(s, t, a=None, b=None):
        m = pd.Series(True, index=ne.index)
        if a is not None:
            m &= ne.index >= pd.Timestamp(a)
        if b is not None:
            m &= ne.index < pd.Timestamp(b)
        nm = metrics(ne[m], None)
        nm.pop("trades", None)
        return {"strategy": portfolio_metrics(s, t, a, b), "nifty": nm}

    ins, oos = side(e, res.trades, None, split), side(e, res.trades, split, None)
    beat = lambda x: (x["strategy"].get("cagr_pct") or -99) > (x["nifty"].get("cagr_pct") or 99)
    out = {
        "id": "a_momentum_portfolio",
        "label": f"Monthly top-{MOMPORT['n_hold']} momentum portfolio (Nifty200 Momentum style)",
        "status": "test only – not used live",
        "settings": {**MOMPORT, "rebalance": "monthly", "weighting": "equal at entry",
                     "universe_note": "200 most-traded stocks (no point-in-time market caps available)"},
        "all": side(e, res.trades),
        "in_sample": ins, "out_of_sample": oos,
        "beats_nifty_in_sample": beat(ins), "beats_nifty_out_of_sample": beat(oos),
        "yearly": yearly_returns(e), "nifty_yearly": yearly_returns(ne),
        "equity": {d.date().isoformat(): round(v) for d, v in e.resample("W").last().dropna().items()},
        "notes": ["Nifty here is the price index (no dividends), which understates buy-and-hold by about 1–1.5% a year.",
                  "Always invested: no market filter and no stop-losses, so drawdowns follow the market."],
    }
    print(f"research momentum portfolio: IS {ins} | OOS {oos} ({time.time() - t1:.0f}s)")
    return out


def run(hist: pd.DataFrame, nifty: pd.Series, cfg: dict, demo: bool, exclude: set | None = None) -> dict:
    t0 = time.time()
    if exclude:
        hist = hist[~hist["symbol"].isin(exclude)]
        print(f"Excluded {len(exclude)} funds/ETFs from the universe")
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
               "ETFs and liquid/gold/index funds excluded – only company shares are traded.",
               f"Stops must be {cfg.get('min_stop_pct', 0)*100:.1f}–{cfg['max_stop_pct']*100:.0f}% from entry "
               "(tighter stops are eaten by charges).",
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

    rep["research"] = research_momentum_portfolio(P, f, cfg, n, split)

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
    exclude = None
    if not args.demo:
        from nse import NSE as _N, is_fund
        exclude = {s for s in hist["symbol"].unique() if is_fund(s)} | _N().etf_symbols()
    rep = run(hist, nifty, cfg, args.demo, exclude)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
