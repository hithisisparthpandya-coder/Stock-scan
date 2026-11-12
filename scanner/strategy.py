"""Strategy engine shared by the live scanner and the backtester.

Two strategies:
  baseline  – the current funnel (trend + volume + breakout/pullback, Nifty > 50 DMA, 1% risk)
  adaptive  – regime-adaptive momentum: the market regime (green / amber / red) decides which
              setups are allowed and how much is risked; stocks are ranked by volatility-adjusted
              6- and 12-month momentum (similar in spirit to NSE's momentum index method).

All signals for day i use only data up to and including day i. Trades fill at day i+1's open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SQRT252 = np.sqrt(252)


# ---------------------------------------------------------------- data panel
def adjustment_multiplier(C: pd.DataFrame, PC: pd.DataFrame | None, tol: float = 0.02) -> pd.DataFrame | float:
    """Back-adjustment for splits/bonuses. On an ex-date NSE reports an adjusted previous close,
    so PC[t] / C[t-1] != 1. Prices before that date are multiplied by that ratio."""
    if PC is None:
        return 1.0
    F = PC / C.shift(1)
    F = F.where((F - 1).abs() > tol, 1.0).fillna(1.0).clip(0.02, 50)
    return F.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)


def build_panel(hist: pd.DataFrame) -> dict:
    hist = hist.drop_duplicates(["date", "symbol"], keep="last")
    C = hist.pivot(index="date", columns="symbol", values="close").sort_index()

    def get(col):
        if col not in hist.columns or hist[col].isna().all():
            return None
        return hist.pivot(index="date", columns="symbol", values=col).reindex(index=C.index, columns=C.columns)

    H, L, V, T = get("high"), get("low"), get("volume"), get("turnover_cr")
    O = get("open")
    O = C.copy() if O is None else O.fillna(C)
    M = adjustment_multiplier(C, get("prev_close"))
    return {"O": O * M, "H": H * M, "L": L * M, "C": C * M, "V": V / M, "T": T, "M": M}


# ---------------------------------------------------------------- features
def features(P: dict, nifty: pd.Series, cfg: dict) -> dict:
    C, H, L, V, T = P["C"], P["H"], P["L"], P["V"], P["T"]
    f = {}
    f["sma50"] = C.rolling(50).mean()
    f["sma200"] = C.rolling(200).mean()
    f["sma200_prev"] = f["sma200"].shift(20)
    f["ema20"] = C.ewm(span=20, adjust=False, ignore_na=True).mean()
    pc = C.shift(1)
    tr = np.maximum.reduce([(H - L).values, (H - pc).abs().values, (L - pc).abs().values])
    f["atr14"] = pd.DataFrame(tr, index=C.index, columns=C.columns).rolling(14).mean()
    f["avgvol20"] = V.shift(1).rolling(20).mean()
    f["avgturn20"] = T.rolling(20, min_periods=10).mean()
    f["hh20"] = H.shift(1).rolling(20).max()
    f["low3"] = L.rolling(3).min()
    f["low10"] = L.rolling(10).min()
    f["ll20_prev"] = L.shift(1).rolling(20).min()
    f["days"] = C.notna().cumsum()
    f["ret63"] = C / C.shift(63) - 1

    liquid = f["avgturn20"] >= cfg["min_turnover_cr"]
    r = C.pct_change(fill_method=None)
    vol = r.rolling(252, min_periods=200).std() * SQRT252
    m6 = ((C / C.shift(126) - 1) / vol).where(liquid)
    m12 = ((C / C.shift(252) - 1) / vol).where(liquid)

    def z(x):
        return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)

    f["mom"] = (z(m6) + z(m12)) / 2
    f["mom_pct"] = f["mom"].rank(axis=1, pct=True)

    above = (C > f["sma50"]) & liquid
    f["breadth"] = above.sum(axis=1) / liquid.sum(axis=1).replace(0, np.nan)

    n = nifty.reindex(C.index).ffill()
    n50, n200 = n.rolling(50).mean(), n.rolling(200).mean()
    nvol = n.pct_change(fill_method=None).rolling(20).std() * SQRT252
    volpct = nvol.rolling(252, min_periods=120).rank(pct=True)
    known = n200.notna() & f["breadth"].notna()
    green = (n > n50) & (n > n200) & (f["breadth"] >= cfg["breadth_green"]) & (volpct.fillna(0.5) <= cfg["vol_pct_green"])
    red = ((n < n50) & (n < n200)) | (volpct.fillna(0) >= cfg["vol_pct_red"]) | (f["breadth"] < cfg["breadth_red"])
    reg = pd.Series(np.where(red, "red", np.where(green, "green", "amber")), index=C.index)
    f["regime"] = reg.where(known, "amber")
    f["market_ok"] = (n > n50).fillna(False)
    f["nifty"], f["nifty50"], f["nifty200"], f["nifty_volpct"] = n, n50, n200, volpct
    return f


def regime_info(f: dict, i: int = -1) -> dict:
    g = lambda k: f[k].iloc[i]
    reg = g("regime")
    notes = {
        "green": "Trend up, broad participation, calm volatility – breakouts and pullbacks allowed at full risk.",
        "amber": "Mixed market – pullbacks only, in the strongest stocks, at half risk.",
        "red": "Downtrend or volatility spike – no new trades; protect open positions.",
    }
    num = lambda x: None if pd.isna(x) else round(float(x), 3)
    return {"regime": reg, "note": notes[reg], "breadth": num(g("breadth")),
            "nifty": num(g("nifty")), "nifty_sma50": num(g("nifty50")), "nifty_sma200": num(g("nifty200")),
            "vol_percentile": num(g("nifty_volpct"))}


# ---------------------------------------------------------------- signals
def _row(f, P, i):
    keys = ["sma50", "sma200", "sma200_prev", "ema20", "atr14", "avgvol20", "avgturn20", "hh20",
            "low3", "low10", "days", "ret63", "mom_pct"]
    s = pd.DataFrame({k: f[k].iloc[i] for k in keys})
    s["close"] = P["C"].iloc[i]
    s["prev_close"] = P["C"].iloc[i - 1] if i != 0 and len(P["C"]) > 1 else np.nan
    s["volume"] = P["V"].iloc[i]
    s["vol_ratio"] = s["volume"] / s["avgvol20"]
    return s


def _liquid_mask(s, cfg):
    low = s["close"] < cfg["low_price_threshold"]
    need = np.where(low, cfg["min_turnover_cr"] * cfg["low_price_turnover_mult"], cfg["min_turnover_cr"])
    return (s["close"] >= cfg["min_price"]) & (s["avgturn20"] >= need) & (s["days"] >= 200)


def signals(f: dict, P: dict, i: int, cfg: dict, strategy: str, excluded: set | None = None):
    """Candidates for day i (list of dicts, best first) and funnel stage counts."""
    excluded = excluded or set()
    s = _row(f, P, i)
    counts = []
    m = s["close"].notna()
    counts.append(("Universe", int(m.sum())))
    m &= ~s.index.isin(list(excluded))
    counts.append(("Not excluded (surveillance / bands)", int(m.sum())))
    m &= _liquid_mask(s, cfg)
    counts.append(("Liquid", int(m.sum())))
    m &= (s["close"] > s["sma50"]) & (s["sma50"] > s["sma200"]) & (s["sma200"] > s["sma200_prev"])
    counts.append(("Uptrend", int(m.sum())))

    breakout = (s["close"] > s["hh20"]) & (s["vol_ratio"] >= cfg["vol_mult"])
    bounce = (s["low3"] <= s["ema20"] * (1 + cfg["pullback_tol"])) & (s["close"] > s["ema20"]) & \
             (s["close"] > s["prev_close"])

    if strategy == "baseline":
        if not bool(f["market_ok"].iloc[i]):
            counts.append(("Market filter (Nifty > 50 DMA)", 0))
            return [], counts, 0.0
        risk = cfg["risk_pct"]
        m &= s["vol_ratio"] >= cfg["vol_mult"]
        counts.append(("Volume surge", int(m.sum())))
        pullback = bounce
        allowed = breakout | pullback
    else:
        reg = f["regime"].iloc[i]
        risk = {"green": cfg["risk_pct"], "amber": cfg["risk_pct"] * cfg["amber_risk_mult"], "red": 0.0}[reg]
        if risk == 0:
            counts.append(("Regime RED – no new trades", 0))
            return [], counts, 0.0
        need = cfg["mom_pct_green"] if reg == "green" else cfg["mom_pct_amber"]
        m &= s["mom_pct"] >= need
        counts.append((f"Momentum top {round((1 - need) * 100)}%", int(m.sum())))
        pullback = bounce & (s["vol_ratio"] >= cfg["pullback_vol_mult"])
        allowed = (breakout | pullback) if reg == "green" else pullback

    m &= allowed
    counts.append(("Setup", int(m.sum())))
    s["setup"] = np.where(breakout, "Breakout from base", np.where(pullback, "Pullback to support", ""))
    struct = np.where(breakout, s["low10"], s["low3"]) * (1 - cfg["stop_buffer"])
    if strategy == "baseline":
        stop = pd.Series(struct, index=s.index)
    else:  # ATR guard: at least 1 ATR of room, never more than 3 ATR
        atr = s["atr14"]
        stop = np.minimum(s["close"] - atr, np.maximum(struct, s["close"] - cfg["atr_stop_max"] * atr))
    s["stop"] = stop
    s["risk_pct"] = (s["close"] - s["stop"]) / s["close"]
    m &= (s["risk_pct"] > 0) & (s["risk_pct"] <= cfg["max_stop_pct"])
    counts.append(("Stop fits", int(m.sum())))

    c = s[m].copy()
    if len(c):
        if strategy == "baseline":
            c["score"] = (0.5 * c["ret63"].rank(pct=True) + 0.3 * c["vol_ratio"].rank(pct=True)
                          + 0.2 * (1 - c["risk_pct"].rank(pct=True)))
        else:
            c["score"] = (0.6 * c["mom_pct"].rank(pct=True) + 0.2 * c["vol_ratio"].rank(pct=True)
                          + 0.2 * (1 - c["risk_pct"].rank(pct=True)))
        c = c.sort_values("score", ascending=False)
    out = [{"symbol": sym, "setup": r["setup"], "close": float(r["close"]), "stop": float(r["stop"]),
            "risk_pct": float(r["risk_pct"]), "vol_ratio": float(r["vol_ratio"]), "score": float(r["score"]),
            "mom_pct": None if pd.isna(r["mom_pct"]) else float(r["mom_pct"]),
            "ret63": None if pd.isna(r["ret63"]) else float(r["ret63"]),
            "sma50": float(r["sma50"]), "sma200": float(r["sma200"]), "turnover_cr": float(r["avgturn20"]),
            "low_priced": bool(r["close"] < cfg["low_price_threshold"])}
           for sym, r in c.iterrows()]
    return out, counts, risk
