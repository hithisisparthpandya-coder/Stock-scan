"""Live pick scorecard: did each day's picks end up among the TOP PERFORMERS afterwards?

Every evening the day's picks are recorded. Once the data exists, each pick is judged from the next
day's open to the close 1, 5 and 20 trading days later, against every liquid NSE stock (avg turnover
≥ min_turnover_cr) over the same window – the same yardstick as the backtest's factor study.
Stored in docs/data/scorecard.json and shown in the app.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = {"1d": 1, "5d": 5, "20d": 20}


def evaluate(P: dict, f: dict, cfg: dict, as_of: str, symbols: list[str]) -> dict:
    C, O = P["C"], P["O"]
    dates = C.index
    t = int(np.searchsorted(dates, pd.Timestamp(as_of)))
    if t >= len(dates) or dates[t] != pd.Timestamp(as_of):
        return {}
    uni = (f["avgturn20"].iloc[t] >= cfg["min_turnover_cr"]) & C.iloc[t].notna()
    out = {}
    for h, n in HORIZONS.items():
        if t + n >= len(dates):
            continue
        r = (C.iloc[t + n] / O.iloc[t + 1] - 1).where(uni).dropna()
        if len(r) < 50:
            continue
        pct = r.rank(pct=True)
        top = r.sort_values(ascending=False)
        rows = []
        for s in symbols:
            if s not in r.index:
                continue
            rows.append({"symbol": s, "ret_pct": round(float(r[s]) * 100, 2), "pctile": round(float(pct[s]) * 100, 1),
                         "rank": int((top.index == s).argmax()) + 1})
        if rows:
            out[h] = {"universe": int(len(r)), "median_pct": round(float(r.median()) * 100, 2),
                      "top10_cut_pct": round(float(r.quantile(0.9)) * 100, 2), "picks": rows}
    return out


def summarize(days: list[dict]) -> dict:
    s = {}
    for h in HORIZONS:
        rows = [(p, d["eval"][h]["median_pct"]) for d in days if h in d.get("eval", {}) for p in d["eval"][h]["picks"]]
        if not rows:
            continue
        n = len(rows)
        s[h] = {"picks": n, "hit_pct": round(sum(p["pctile"] >= 90 for p, _ in rows) / n * 100, 1),
                "beat_median_pct": round(sum(p["pctile"] > 50 for p, _ in rows) / n * 100, 1),
                "avg_pctile": round(float(np.mean([p["pctile"] for p, _ in rows])), 1),
                "excess_pct": round(float(np.mean([p["ret_pct"] - m for p, m in rows])), 2),
                "in_top50": sum(p["rank"] <= 50 for p, _ in rows)}
    return s


def update(path: Path, P: dict, f: dict, cfg: dict, as_of: str, picks: list[dict], demo: bool = False) -> dict:
    doc = json.loads(path.read_text()) if path.exists() else {}
    if doc.get("demo") and not demo:
        doc = {}
    days = {d["date"]: d for d in doc.get("days", [])}
    days[as_of] = {**days.get(as_of, {}), "date": as_of, "symbols": [p["symbol"] for p in picks]}
    for d in days.values():
        if d["symbols"] and len(d.get("eval", {})) < len(HORIZONS):
            d["eval"] = evaluate(P, f, cfg, d["date"], d["symbols"])
    lst = sorted(days.values(), key=lambda d: d["date"])[-250:]
    out = {"demo": demo, "updated": as_of, "random_hit_pct": 10.0, "summary": summarize(lst), "days": lst}
    path.write_text(json.dumps(out, indent=1))
    return out


def summary_line(sc: dict) -> str:
    s = sc.get("summary", {}).get("5d") or sc.get("summary", {}).get("1d")
    if not s:
        return ""
    h = "1 week" if "5d" in sc.get("summary", {}) else "next day"
    return (f"🎯 Pick scorecard ({h}): {s['hit_pct']}% of {s['picks']} picks landed in the top 10% (random 10%) · "
            f"avg {s['excess_pct']:+.2f}% vs the median stock")
