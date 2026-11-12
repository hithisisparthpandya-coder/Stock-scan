"""The stock funnel. Pure pandas, no network: same rules as the Planner sheet.

Stages
  0 Universe        all NSE EQ-series stocks traded on the latest day
  1 Surveillance    not on NSE ASM / GSM lists, and not in a 2% / 5% circuit band
  2 Liquidity       20-day avg turnover >= min_turnover_cr, 200+ days of data, price >= min_price;
                    low-priced stocks (< low_price_threshold) need 2x the turnover

  3 Trend           close > 50 DMA > 200 DMA, and the 200 DMA is rising
  4 Volume          today's volume >= vol_mult x the previous 20-day average
  5 Setup           breakout above the 20-day high, or a bounce off the 20-day EMA
  6 Risk fits       stop distance <= max_stop_pct and at least 1 share fits the risk budget
  7 Results         no results board meeting within results_window_days      (apply_quality)
  8 Fundamentals    ROE, debt/equity, profit, market cap; unknown data is flagged (apply_quality)
Ranked by relative strength (63-day return), volume surge and tightness of stop.
"""
from __future__ import annotations

import datetime as dt

import math

import numpy as np
import pandas as pd


def market_filter(nifty: pd.Series) -> dict:
    """nifty: Series of closes indexed by date."""
    nifty = nifty.dropna().sort_index()
    if len(nifty) < 50:
        return {"ok": None, "close": float(nifty.iloc[-1]) if len(nifty) else None, "sma50": None,
                "note": "Not enough Nifty history yet; check manually."}
    sma50 = float(nifty.rolling(50).mean().iloc[-1])
    close = float(nifty.iloc[-1])
    ok = close > sma50
    return {"ok": ok, "close": round(close, 2), "sma50": round(sma50, 2),
            "note": "Nifty above its 50 DMA: new trades allowed." if ok
            else "Nifty below its 50 DMA: NO new trades (rule). Watch only."}


def run_funnel(hist: pd.DataFrame, cfg: dict, excluded: set[str] | None = None,
               names: dict[str, str] | None = None, bands: dict[str, float] | None = None) -> dict:
    """Technical stages 0-6. Returns every survivor as a candidate, best first."""
    excluded = set(excluded or set())
    names = names or {}
    bands = bands or {}
    tight = set(cfg.get("exclude_bands", [2, 5]))
    excluded |= {sym for sym, b in bands.items() if b in tight}
    hist = hist.drop_duplicates(["date", "symbol"], keep="last")

    C = hist.pivot(index="date", columns="symbol", values="close").sort_index()
    H = hist.pivot(index="date", columns="symbol", values="high").reindex(C.index)
    L = hist.pivot(index="date", columns="symbol", values="low").reindex(C.index)
    V = hist.pivot(index="date", columns="symbol", values="volume").reindex(C.index)
    T = hist.pivot(index="date", columns="symbol", values="turnover_cr").reindex(C.index)

    latest = C.index[-1]
    s = pd.DataFrame(index=C.columns)
    s["close"] = C.iloc[-1]
    s["prev_close"] = C.iloc[-2] if len(C) > 1 else np.nan
    s["volume"] = V.iloc[-1]
    s["days"] = C.notna().sum()
    s["sma50"] = C.rolling(50).mean().iloc[-1]
    sma200 = C.rolling(200).mean()
    s["sma200"] = sma200.iloc[-1]
    s["sma200_prev"] = sma200.shift(20).iloc[-1]
    s["ema20"] = C.ewm(span=20, adjust=False, ignore_na=True).mean().iloc[-1]
    s["avgvol20"] = V.shift(1).rolling(20).mean().iloc[-1]
    s["avgturn20"] = T.rolling(20, min_periods=10).mean().iloc[-1]
    s["hh20"] = H.shift(1).rolling(20).max().iloc[-1]
    s["low3"] = L.rolling(3).min().iloc[-1]
    s["low10"] = L.rolling(10).min().iloc[-1]
    s["ret63"] = (C.iloc[-1] / C.shift(63).iloc[-1]) - 1

    counts = []
    m = s["close"].notna()
    counts.append(("Universe (NSE EQ stocks traded today)", int(m.sum())))

    m &= ~s.index.isin(list(excluded))
    counts.append(("Not on ASM/GSM or a 2%/5% circuit band", int(m.sum())))

    low = s["close"] < cfg["low_price_threshold"]
    need_turn = np.where(low, cfg["min_turnover_cr"] * cfg["low_price_turnover_mult"], cfg["min_turnover_cr"])
    m &= (s["close"] >= cfg["min_price"]) & (s["avgturn20"] >= need_turn) & (s["days"] >= 200)
    counts.append((f"Liquid (≥ ₹{cfg['min_turnover_cr']} cr/day; 2× that below ₹{cfg['low_price_threshold']})",
                   int(m.sum())))

    m &= (s["close"] > s["sma50"]) & (s["sma50"] > s["sma200"]) & (s["sma200"] > s["sma200_prev"])
    counts.append(("Uptrend (price > 50 DMA > rising 200 DMA)", int(m.sum())))

    s["vol_ratio"] = s["volume"] / s["avgvol20"]
    m &= s["vol_ratio"] >= cfg["vol_mult"]
    counts.append((f"Volume ≥ {cfg['vol_mult']}× 20-day average", int(m.sum())))

    breakout = s["close"] > s["hh20"]
    pullback = (s["low3"] <= s["ema20"] * (1 + cfg["pullback_tol"])) & (s["close"] > s["ema20"]) & \
               (s["close"] > s["prev_close"])
    s["setup"] = np.where(breakout, "Breakout from base", np.where(pullback, "Pullback to support", ""))
    m &= breakout | pullback
    counts.append(("Valid setup (breakout or pullback bounce)", int(m.sum())))

    buf = 1 - cfg["stop_buffer"]
    s["stop"] = np.where(breakout, s["low10"] * buf, s["low3"] * buf)
    s["risk_ps"] = s["close"] - s["stop"]
    s["risk_pct"] = s["risk_ps"] / s["close"]
    risk_amt = cfg["capital"] * cfg["risk_pct"]
    max_pos = cfg["capital"] * cfg["max_position_pct"]

    def qty(row):
        if not (row["risk_ps"] > 0):
            return 0
        return int(min(math.floor(risk_amt / row["risk_ps"]), math.floor(max_pos / row["close"])))

    s["qty"] = s.apply(qty, axis=1)
    m &= (s["risk_pct"] >= cfg.get("min_stop_pct", 0)) & (s["risk_pct"] <= cfg["max_stop_pct"]) & (s["qty"] >= 1)
    counts.append((f"Stop {cfg.get('min_stop_pct', 0)*100:.1f}–{int(cfg['max_stop_pct']*100)}% away, ≥ 1 share fits",
                   int(m.sum())))

    c = s[m].copy()
    if len(c):
        c["score"] = (0.5 * c["ret63"].rank(pct=True)
                      + 0.3 * c["vol_ratio"].rank(pct=True)
                      + 0.2 * (1 - c["risk_pct"].rank(pct=True)))
        c = c.sort_values("score", ascending=False)

    picks = []
    for sym, r in c.iterrows():
        entry = float(r["close"])
        stop = float(r["stop"])
        q = int(r["qty"])
        picks.append({
            "symbol": sym,
            "name": names.get(sym, ""),
            "setup": r["setup"],
            "low_priced": bool(r["close"] < cfg["low_price_threshold"]),
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(entry + cfg["reward_risk"] * (entry - stop), 2),
            "qty": q,
            "capital": round(q * entry, 0),
            "max_loss": round(q * (entry - stop), 0),
            "risk_pct": round(float(r["risk_pct"]) * 100, 2),
            "vol_ratio": round(float(r["vol_ratio"]), 2),
            "ret63_pct": round(float(r["ret63"]) * 100, 1) if pd.notna(r["ret63"]) else None,
            "sma50": round(float(r["sma50"]), 2),
            "sma200": round(float(r["sma200"]), 2),
            "turnover_cr": round(float(r["avgturn20"]), 1),
            "score": round(float(r["score"]) * 100, 1),
        })

    return {
        "as_of": latest.date().isoformat(),
        "funnel": [{"stage": k, "count": v} for k, v in counts],
        "candidates": picks,
    }


# ---------- stages 7-8: results date and fundamentals (pure; data fetched elsewhere) ----------
def check_fundamentals(f: dict, cfg: dict) -> tuple[str, list[str]]:
    """Returns ("pass" | "fail" | "unknown", reasons)."""
    if not f:
        return "unknown", ["no fundamentals data"]
    fin = (f.get("sector") or "") == "Financial Services"   # D/E is not meaningful for banks/NBFCs
    fails, unknown = [], []
    roe, de, prof, mcap = f.get("roe"), f.get("de"), f.get("profitable"), f.get("mcap_cr")
    if roe is None: unknown.append("ROE")
    elif roe < cfg["min_roe"]: fails.append(f"ROE {roe*100:.1f}% < {cfg['min_roe']*100:.0f}%")
    if not fin:
        if de is None: unknown.append("D/E")
        elif de > cfg["max_de"]: fails.append(f"D/E {de:.2f} > {cfg['max_de']}")
    if prof is None: unknown.append("profit")
    elif prof is False: fails.append("loss-making")
    if mcap is None: unknown.append("market cap")
    elif mcap < cfg["min_mcap_cr"]: fails.append(f"market cap ₹{mcap:,.0f} cr < ₹{cfg['min_mcap_cr']} cr")
    if fails:
        return "fail", fails
    if unknown:
        return "unknown", ["missing: " + ", ".join(unknown)]
    return "pass", []


def apply_quality(candidates: list[dict], funds: dict, results: dict, today: dt.date, cfg: dict):
    """Drops stocks with results due soon or failing fundamentals; flags unknowns. Returns (picks, counts)."""
    window = dt.timedelta(days=cfg["results_window_days"])
    after_results, counts = [], []
    for p in candidates:
        f = funds.get(p["symbol"], {}) or {}
        nxt = results.get(p["symbol"])
        if nxt is None and f.get("next_results"):
            nxt = dt.date.fromisoformat(f["next_results"])
        p["next_results"] = nxt.isoformat() if nxt else None
        if nxt and today <= nxt <= today + window:
            continue
        after_results.append(p)
    counts.append((f"No results due in next {cfg['results_window_days']} days", len(after_results)))

    kept = []
    for p in after_results:
        f = funds.get(p["symbol"], {}) or {}
        status, reasons = check_fundamentals(f, cfg)
        if status == "fail":
            continue
        p.update({
            "fund_status": status, "fund_notes": reasons,
            "roe_pct": round(f["roe"] * 100, 1) if f.get("roe") is not None else None,
            "de": round(f["de"], 2) if f.get("de") is not None else None,
            "mcap_cr": round(f["mcap_cr"]) if f.get("mcap_cr") is not None else None,
            "sector": f.get("sector"),
        })
        kept.append(p)
    counts.append(("Fundamentals OK or unverified (failures dropped)", len(kept)))
    # Verified first, then unverified; best score first within each group.
    kept.sort(key=lambda p: (p["fund_status"] != "pass", -p["score"]))
    return kept[: cfg["top_n"]], counts
