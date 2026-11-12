"""Synthetic market data with known answers, for tests and for the demo app data."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_DAYS = 260


def _dates(n=N_DAYS):
    return pd.bdate_range(end="2026-11-13", periods=n)


def _frame(sym, close, vol, highpad=0.01, lowpad=0.01, high=None, low=None):
    d = _dates(len(close))
    close = np.asarray(close, float)
    vol = np.asarray(vol, float)
    h = close * (1 + highpad) if high is None else high
    lo = close * (1 - lowpad) if low is None else low
    return pd.DataFrame({"date": d, "symbol": sym, "high": h, "low": lo, "close": close,
                         "volume": vol, "turnover_cr": close * vol / 1e7})


def _uptrend(start, end, n, rng, noise=0.004):
    base = np.geomspace(start, end, n)
    return base * (1 + rng.normal(0, noise, n))


def make_history(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = N_DAYS
    frames = []

    # 1. Clean breakout: uptrend, 20-day flat base, breakout today on 3x volume
    c = _uptrend(400, 600, n, rng)
    c[-21:-1] = 600 * (1 + rng.normal(0, 0.003, 20))
    c[-1] = 630
    v = np.full(n, 300_000.0); v[-1] = 900_000
    frames.append(_frame("BRKOUT", c, v))

    # 2. Pullback bounce: uptrend, dips to the 20 EMA over 3 days, bounces today on 2x volume
    c = _uptrend(800, 1200, n, rng)
    ema = pd.Series(c).ewm(span=20, adjust=False).mean().values
    c[-4] = ema[-5] * 1.01; c[-3] = ema[-5] * 1.0; c[-2] = ema[-5] * 0.998
    c[-1] = c[-2] * 1.02
    lo = c * 0.99
    v = np.full(n, 150_000.0); v[-1] = 330_000
    frames.append(_frame("PULLBK", c, v, low=lo))

    # 3. Downtrend: fails trend stage
    c = np.geomspace(900, 500, n); v = np.full(n, 400_000.0); v[-1] = 1_200_000
    frames.append(_frame("DOWNTR", c, v))

    # 4. Penny stock: fails price stage
    c = _uptrend(10, 30, n, rng); v = np.full(n, 20_000_000.0); v[-1] = 60_000_000
    frames.append(_frame("PENNY", c, v))

    # 5. Illiquid: fails turnover stage
    c = _uptrend(300, 500, n, rng); v = np.full(n, 5_000.0); v[-1] = 20_000
    frames.append(_frame("ILLIQ", c, v))

    # 6. Perfect breakout but on surveillance list
    c = _uptrend(200, 300, n, rng); c[-21:-1] = 300; c[-1] = 315
    v = np.full(n, 800_000.0); v[-1] = 2_400_000
    frames.append(_frame("SURVL", c, v))

    # 7. Breakout without volume: fails volume stage
    c = _uptrend(500, 700, n, rng); c[-21:-1] = 700; c[-1] = 730
    v = np.full(n, 300_000.0)
    frames.append(_frame("NOVOL", c, v))

    # 8. Breakout with a stop that is too wide (>8%): fails risk stage
    c = _uptrend(500, 700, n, rng); c[-21:-1] = 700 * (1 + rng.normal(0, 0.003, 20)); c[-1] = 735
    lo = c * 0.99; lo[-8] = 600
    v = np.full(n, 300_000.0); v[-1] = 900_000
    frames.append(_frame("WIDESTP", c, v, low=lo))

    def clean_breakout(sym, start, end, vol, mult=3.0):
        c = _uptrend(start, end, n, rng)
        c[-21:-1] = end * (1 + rng.normal(0, 0.003, 20)); c[-1] = end * 1.04
        v = np.full(n, float(vol)); v[-1] = vol * mult
        frames.append(_frame(sym, c, v))

    clean_breakout("LOWPX", 20, 35, 8_000_000)     # low-priced but liquid (~₹28 cr/day): allowed
    clean_breakout("LOWTHIN", 20, 35, 4_300_000)   # low-priced, ~₹15 cr/day: below the 2x bar
    clean_breakout("BAND5", 300, 450, 600_000)     # on a 5% circuit band: excluded
    clean_breakout("RESULTS", 900, 1300, 200_000)  # results in 3 days: excluded
    clean_breakout("WEAKFUND", 400, 560, 400_000)  # ROE 8%: excluded
    clean_breakout("NODATA", 250, 380, 700_000)    # no fundamentals data: kept, flagged

    # Background noise stocks
    for i in range(40):
        drift = rng.normal(0.0004, 0.0008)
        c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n))) + 50
        v = rng.integers(200_000, 2_000_000, n).astype(float)
        frames.append(_frame(f"STK{i:02d}", c, v, highpad=0.012, lowpad=0.012))

    hist = pd.concat(frames, ignore_index=True)
    nifty = pd.Series(np.geomspace(22000, 26000, n), index=_dates(n))
    excluded = {"SURVL"}
    names = {"BRKOUT": "Demo Breakout Ltd", "PULLBK": "Demo Pullback Bank Ltd", "LOWPX": "Demo Low-Price Ltd"}
    return hist, nifty, excluded, names


def make_quality(today):
    """Demo price bands, results calendar and fundamentals with known outcomes."""
    import datetime as dt
    good = {"roe": 0.21, "de": 0.35, "profitable": True, "mcap_cr": 8500, "sector": "Industrials"}
    funds = {f"STK{i:02d}": dict(good) for i in range(40)}
    funds.update({
        "BRKOUT": dict(good, roe=0.24, mcap_cr=12000),
        "PULLBK": {"roe": 0.17, "de": None, "profitable": True, "mcap_cr": 45000, "sector": "Financial Services"},
        "LOWPX": dict(good, roe=0.19, de=0.6, mcap_cr=1200),
        "RESULTS": dict(good),
        "WEAKFUND": dict(good, roe=0.08),
        "BAND5": dict(good), "LOWTHIN": dict(good), "WIDESTP": dict(good), "NOVOL": dict(good),
    })
    bands = {s: 20.0 for s in list(funds) + ["NODATA", "SURVL"]}
    bands["BAND5"] = 5.0
    results = {"RESULTS": today + dt.timedelta(days=3), "BRKOUT": today + dt.timedelta(days=30)}
    return bands, results, funds


def make_long_history(seed: int = 11, years: float = 7.0, n_stocks: int = 90):
    """Synthetic multi-year market with bull / bear / choppy phases, persistent stock momentum,
    volume spikes on big moves, overnight gaps, and one 1:2 split (raw prices + NSE-style adjusted
    previous close) to exercise the split adjustment. For tests and the demo Strategy Lab only."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-11-13", periods=int(252 * years))
    n = len(dates)
    # Market phases (annual drift, annual vol)
    phases = [(0.22, 0.13), (-0.25, 0.28), (0.02, 0.16), (0.28, 0.14), (-0.15, 0.24), (0.18, 0.15), (0.05, 0.18)]
    seg = np.linspace(0, n, len(phases) + 1).astype(int)
    mret = np.zeros(n)
    for k, (mu, sig) in enumerate(phases):
        a, b = seg[k], seg[k + 1]
        mret[a:b] = rng.normal(mu / 252, sig / np.sqrt(252), b - a)
    nifty = pd.Series(15000 * np.exp(np.cumsum(mret)), index=dates)

    frames = []
    split_sym, split_i = "SPLITCO", n - 400
    for s in range(n_stocks):
        sym = split_sym if s == 0 else f"LNG{s:03d}"
        beta = rng.uniform(0.7, 1.4)
        alpha = np.zeros(n)
        k = 0
        while k < n:  # persistent alpha regimes → momentum exists in this synthetic market
            L = int(rng.integers(80, 200))
            alpha[k:k + L] = rng.normal(0, 0.35) / 252
            k += L
        idio = rng.normal(0, 0.018, n)
        r = beta * mret + alpha + idio
        adj = rng.uniform(80, 2500) * np.exp(np.cumsum(r))
        gap = rng.normal(0, 0.006, n)
        open_ = np.r_[adj[0], adj[:-1] * (1 + gap[1:])]
        hi = np.maximum(open_, adj) * (1 + np.abs(rng.normal(0, 0.008, n)))
        lo = np.minimum(open_, adj) * (1 - np.abs(rng.normal(0, 0.008, n)))
        base_vol = rng.uniform(2e5, 3e6) * (500 / adj.mean()) ** 0.5
        vol = base_vol * (1 + 4 * np.abs(r) / 0.02) * rng.uniform(0.7, 1.3, n)
        mult = np.ones(n)
        if sym == split_sym:
            mult[:split_i] = 2.0                 # raw prices were double before the 1:2 split
        raw_c, raw_o, raw_h, raw_l = adj * mult, open_ * mult, hi * mult, lo * mult
        raw_v = vol / mult
        prev = np.r_[np.nan, raw_c[:-1]]
        if sym == split_sym:
            prev[split_i] = raw_c[split_i - 1] / 2   # NSE reports the adjusted base price on the ex-date
        frames.append(pd.DataFrame({"date": dates, "symbol": sym, "open": raw_o, "high": raw_h, "low": raw_l,
                                    "close": raw_c, "prev_close": prev, "volume": raw_v,
                                    "turnover_cr": raw_c * raw_v / 1e7}))
    return pd.concat(frames, ignore_index=True), nifty
