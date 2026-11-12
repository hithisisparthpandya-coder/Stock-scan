"""Long price history for the backtest: NSE bhavcopies from 2015, cached one parquet file per year
(only the current year is refreshed on later runs). Includes stocks that were later delisted, so
the test has no survivorship bias."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def load_long(nse, start_year: int, cache_dir: Path, today: dt.date) -> tuple[pd.DataFrame, pd.Series]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, nifty_parts = [], []
    for year in range(start_year, today.year + 1):
        eq_path, nf_path = cache_dir / f"eq_{year}.parquet", cache_dir / f"nifty_{year}.csv"
        have = pd.read_parquet(eq_path) if eq_path.exists() else pd.DataFrame()
        nf = pd.read_csv(nf_path, parse_dates=["date"]).set_index("date")["close"] if nf_path.exists() \
            else pd.Series(dtype=float)
        complete = year < today.year and eq_path.exists() and len(have) and have["date"].max() >= pd.Timestamp(f"{year}-12-24")
        if not complete:
            d = (have["date"].max().date() + dt.timedelta(days=1)) if len(have) else dt.date(year, 1, 1)
            last = min(dt.date(year, 12, 31), today)
            new, new_n = [], {}
            print(f"Downloading {year}: {d} → {last}")
            while d <= last:
                if d.weekday() < 5:
                    df = nse.bhavcopy(d)
                    if df is not None and len(df):
                        new.append(df)
                        nc = nse.nifty_close(d)
                        if nc:
                            new_n[pd.Timestamp(d)] = nc
                d += dt.timedelta(days=1)
            if new:
                have = pd.concat([have, *new], ignore_index=True)
                have.to_parquet(eq_path, index=False)
            if new_n:
                nf = pd.concat([nf, pd.Series(new_n)])
                nf = nf[~nf.index.duplicated(keep="last")].sort_index()
                nf.rename("close").rename_axis("date").to_csv(nf_path)
        if len(have):
            frames.append(have)
        if len(nf):
            nifty_parts.append(nf)
    hist = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    nifty = pd.concat(nifty_parts).sort_index() if nifty_parts else pd.Series(dtype=float)
    if len(hist):
        days = hist["date"].nunique()
        if len(nifty) < 0.9 * days:
            nifty = _nifty_fallback(hist["date"].min(), nifty)
    return hist, nifty


def _nifty_fallback(start, nifty: pd.Series) -> pd.Series:
    """NSE's daily index files are missing for some old dates; fill from Yahoo's ^NSEI history."""
    try:
        import yfinance as yf
        y = yf.download("^NSEI", start=pd.Timestamp(start).date().isoformat(), progress=False, auto_adjust=False)
        if y is None or y.empty:
            return nifty
        col = y["Close"]
        col = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
        col.index = pd.to_datetime(col.index).tz_localize(None)
        print(f"Nifty: filled from Yahoo ({len(col)} days)")
        return pd.concat([col, nifty]).groupby(level=0).last().sort_index()
    except Exception as e:
        print("Nifty fallback failed:", type(e).__name__)
        return nifty


def liquid_subset(hist: pd.DataFrame, min_turnover_cr: float) -> pd.DataFrame:
    """Keep only symbols that were liquid at least 10% of the time (memory: ~3,000 → ~1,200 symbols)."""
    q = hist.groupby("symbol")["turnover_cr"].quantile(0.9)
    keep = q[q >= min_turnover_cr].index
    return hist[hist["symbol"].isin(keep)]
