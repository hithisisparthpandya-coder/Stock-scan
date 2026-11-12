"""Live paper-trading ledger (forward test).

Every scan "buys" its picks on paper at the next trading day's open and manages them with exactly the
backtest rules (same PortfolioSim class). Results build a track record on data no strategy has seen –
the evidence needed before any real money is used. Stored in docs/data/ledger.json (committed daily).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import PortfolioSim, metrics


def _raw(hist: pd.DataFrame):
    hist = hist.drop_duplicates(["date", "symbol"], keep="last")
    piv = lambda c: hist.pivot(index="date", columns="symbol", values=c).sort_index()
    C = piv("close")
    g = lambda c: piv(c).reindex(index=C.index, columns=C.columns) if c in hist.columns and hist[c].notna().any() else None
    O = g("open")
    return {"O": C.copy() if O is None else O.fillna(C), "H": g("high"), "L": g("low"), "C": C, "PC": g("prev_close")}


def _risk_for(strategy: str, regime: str, market_ok: bool, cfg: dict) -> float:
    if strategy == "adaptive":
        return {"green": cfg["risk_pct"], "amber": cfg["risk_pct"] * cfg["amber_risk_mult"], "red": 0.0}[regime]
    return cfg["risk_pct"] if market_ok else 0.0


def update(path: Path, hist: pd.DataFrame, feat: dict, picks: list[dict], cfg: dict, strategy: str,
           demo: bool = False) -> dict:
    lcfg = dict(cfg, bt_capital=cfg["capital"])
    led = json.loads(path.read_text()) if path.exists() else {}
    if led.get("demo") and not demo:
        led = {}                                         # never mix demo data into the real record
    sim = PortfolioSim(lcfg, strategy, led.get("sim"))
    R = _raw(hist)
    dates = R["C"].index
    last = pd.Timestamp(led["last_date"]) if led.get("last_date") else None
    curve = dict(led.get("equity_curve", {}))
    events = []
    LL20 = R["L"].shift(1).rolling(20, min_periods=5).min()
    if last is not None:
        for i in np.where(dates > last)[0]:
            d = dates[i]
            factors = {}
            if R["PC"] is not None and i > 0:
                for sym in list(sim.pos) + [s["symbol"] for s in sim.pending]:
                    pc, prev = R["PC"].iloc[i].get(sym, np.nan), R["C"].iloc[i - 1].get(sym, np.nan)
                    if pc > 0 and prev > 0:
                        factors[sym] = pc / prev
            reg = feat["regime"].get(d, "amber") if hasattr(feat["regime"], "get") else "amber"
            atr = feat["atr14"].loc[d] if d in feat["atr14"].index else pd.Series(dtype=float)
            sim.step(d.date().isoformat(), R["O"].iloc[i], R["H"].iloc[i], R["L"].iloc[i], R["C"].iloc[i],
                     LL20.iloc[i], atr, reg, factors)
            curve[d.date().isoformat()] = round(sim.equity, 2)
            events += [f"{d.date().isoformat()}: {e}" for e in sim.events]
    else:
        led["started"] = dates[-1].date().isoformat()
        curve[dates[-1].date().isoformat()] = round(sim.equity, 2)

    # queue today's picks for tomorrow's open (replaces any queue from an earlier run today)
    sim.pending = []
    reg_today = feat["regime"].iloc[-1]
    risk = _risk_for(strategy, reg_today, bool(feat["market_ok"].iloc[-1]), lcfg) * sim.risk_multiplier()
    cands = [{"symbol": p["symbol"], "stop": p["stop"], "setup": p["setup"], "close": p["entry"]} for p in picks]
    if len(sim.pos) < lcfg["max_positions"]:
        sim.queue(cands, risk, reg_today)

    eq = pd.Series({pd.Timestamp(k): v for k, v in curve.items()}).sort_index()
    trades = pd.DataFrame(sim.trades)
    last_close = R["C"].iloc[-1]
    open_rows = []
    for p in sim.pos.values():
        lc = float(last_close.get(p.sym, p.last_close) or p.last_close)
        unreal = (lc - p.entry) * p.qty + (p.sold_value - p.sold_qty * p.entry) - p.cost
        open_rows.append({"symbol": p.sym, "qty": p.qty, "entry": round(p.entry, 2), "stop": round(p.stop, 2),
                          "last": round(lc, 2), "entry_date": p.entry_date, "setup": p.setup,
                          "partial_done": p.partial_done, "unrealized": round(unreal, 0),
                          "r": round(unreal / p.risk_amt, 2) if p.risk_amt else 0.0})
    out = {
        "demo": demo, "strategy": strategy, "started": led.get("started"),
        "last_date": dates[-1].date().isoformat(), "capital": lcfg["bt_capital"],
        "equity": round(sim.equity, 2), "return_pct": round((sim.equity / lcfg["bt_capital"] - 1) * 100, 2),
        "metrics": metrics(eq, trades) if len(eq) >= 2 else {"trades": len(trades)},
        "open_positions": open_rows,
        "queued": [{"symbol": s["symbol"], "limit": round(s["limit"], 2), "stop": round(s["stop"], 2)} for s in sim.pending],
        "recent_trades": sim.trades[-30:][::-1],
        "events_latest": events[-20:],
        "events_log": (led.get("events_log", []) + events)[-60:],
        "equity_curve": curve,
        "sim": sim.state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return out


def summary_line(led: dict) -> str:
    if not led:
        return ""
    m = led.get("metrics", {})
    s = (f"📒 Paper account: ₹{led['equity']:,.0f} ({led['return_pct']:+.2f}% since {led['started']}) · "
         f"{len(led['open_positions'])} open · {m.get('trades', 0)} closed")
    if m.get("trades"):
        s += f" · win {m.get('win_rate_pct')}% · avg {m.get('avg_r')}R"
    return s
