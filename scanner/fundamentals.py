"""Company fundamentals and next results date, fetched only for stocks that survive the
technical funnel (typically 10–60 a day), cached for 7 days.

Source: Yahoo Finance via the yfinance package (free, personal use). Coverage of Indian
stocks is good but not complete; a missing figure is reported as unknown, never as a pass.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path


def _num(x):
    try:
        x = float(x)
        return x if x == x else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _extract(info: dict) -> dict:
    de = _num(info.get("debtToEquity"))
    ni = _num(info.get("netIncomeToCommon"))
    eps = _num(info.get("trailingEps"))
    mcap = _num(info.get("marketCap"))
    return {
        "roe": _num(info.get("returnOnEquity")),             # fraction, 0.18 = 18%
        "de": de / 100 if de is not None else None,          # Yahoo reports D/E in percent
        "profitable": (ni > 0) if ni is not None else ((eps > 0) if eps is not None else None),
        "mcap_cr": mcap / 1e7 if mcap is not None else None,
        "sector": info.get("sector"),
    }


def _next_earnings(t) -> str | None:
    try:
        cal = t.calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            d = dates[0]
            return (d.date() if hasattr(d, "date") and callable(d.date) else d).isoformat()
    except Exception:
        pass
    return None


def fetch(symbols: list[str], cache_path: Path, max_age_days: int = 7, pause: float = 0.3) -> dict:
    """Returns {symbol: {roe, de, profitable, mcap_cr, sector, next_results, fetched}}."""
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}
    today = dt.date.today()
    todo = [s for s in symbols
            if s not in cache or (today - dt.date.fromisoformat(cache[s]["fetched"])).days >= max_age_days]
    if todo:
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance not installed; fundamentals unavailable.")
            return {s: cache.get(s, {}) for s in symbols}
        for sym in todo:
            try:
                t = yf.Ticker(f"{sym}.NS")
                rec = _extract(t.info or {})
                rec["next_results"] = _next_earnings(t)
                rec["fetched"] = today.isoformat()
                cache[sym] = rec
            except Exception as e:  # network / rate limit / delisted
                print(f"  fundamentals {sym}: {type(e).__name__}")
            time.sleep(pause)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return {s: cache.get(s, {}) for s in symbols}
