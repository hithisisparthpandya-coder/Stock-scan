"""Fetchers for official NSE / SEBI public files.

All network calls are defensive: a missing file returns None, and optional
sources (surveillance lists, circulars) never stop the scan.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
import datetime as dt
import xml.etree.ElementTree as ET

import pandas as pd
import requests

ARCHIVES = "https://nsearchives.nseindia.com"
WWW = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

COLS = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "turnover_cr"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


class NSE:
    def __init__(self, pause: float = 0.4):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.pause = pause
        self._warm = False

    # ---------- low level ----------
    def get(self, url: str, tries: int = 3, timeout: int = 30):
        for i in range(tries):
            try:
                r = self.s.get(url, timeout=timeout)
                if r.status_code == 200 and r.content:
                    time.sleep(self.pause)
                    return r
                if r.status_code == 404:
                    return None
            except requests.RequestException:
                pass
            time.sleep(2 * (i + 1))
        return None

    def _warm_www(self):
        if not self._warm:
            try:
                self.s.get(WWW, timeout=20)
            except requests.RequestException:
                pass
            self._warm = True

    # ---------- daily equity prices ----------
    def bhavcopy(self, d: dt.date) -> pd.DataFrame | None:
        """EQ-series OHLC + volume for one trading day, or None (holiday / not yet published).
        Tries the formats NSE has used over the years, newest-likely first for the date."""
        sources = [self._sec_full, self._udiff, self._old_bhav]
        if d < dt.date(2024, 7, 1):
            sources = [self._old_bhav, self._sec_full]
        for fn in sources:
            df = fn(d)
            if df is not None and len(df):
                return df
        return None

    def _sec_full(self, d):
        r = self.get(f"{ARCHIVES}/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv")
        return parse_sec_bhavdata(r.content, d) if r is not None else None

    def _udiff(self, d):
        r = self.get(f"{ARCHIVES}/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
        return parse_udiff(r.content, d) if r is not None else None

    def _old_bhav(self, d):
        mon = MONTHS[d.month - 1]
        r = self.get(f"{ARCHIVES}/content/historical/EQUITIES/{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip")
        return parse_old_bhav(r.content, d) if r is not None else None

    def nifty_close(self, d: dt.date) -> float | None:
        r = self.get(f"{ARCHIVES}/content/indices/ind_close_all_{d:%d%m%Y}.csv")
        if r is None:
            return None
        try:
            df = pd.read_csv(io.BytesIO(r.content))
            df.columns = [c.strip() for c in df.columns]
            row = df[df["Index Name"].astype(str).str.strip().str.lower() == "nifty 50"]
            if row.empty:
                return None
            return float(row["Closing Index Value"].iloc[0])
        except Exception:
            return None

    # ---------- reference data ----------
    def company_names(self) -> dict[str, str]:
        r = self.get(f"{ARCHIVES}/content/equities/EQUITY_L.csv")
        if r is None:
            return {}
        try:
            df = pd.read_csv(io.BytesIO(r.content))
            df.columns = [c.strip() for c in df.columns]
            return dict(zip(df["SYMBOL"].str.strip(), df["NAME OF COMPANY"].str.strip()))
        except Exception:
            return {}

    def surveillance_symbols(self) -> tuple[set[str], bool]:
        """Symbols on NSE ASM / GSM lists. Returns (symbols, fetched_ok)."""
        self._warm_www()
        found: set[str] = set()
        ok = False
        for path in ("/api/reportASM", "/api/reportGSM"):
            r = self.get(WWW + path, tries=2)
            if r is None:
                continue
            try:
                found |= _collect_symbols(r.json())
                ok = True
            except (ValueError, json.JSONDecodeError):
                continue
        return found, ok

    def price_bands(self) -> dict[str, float]:
        """Daily circuit band (%) per EQ symbol from NSE's security list. Tight bands (2%/5%) mean a
        stop-loss can fail to fill on a lower-circuit day."""
        r = self.get(f"{ARCHIVES}/content/equities/sec_list.csv")
        if r is None:
            return {}
        try:
            return parse_sec_list(r.content)
        except Exception:
            return {}

    def upcoming_results(self) -> dict[str, dt.date]:
        """Next board meeting for financial results per symbol, from NSE's event calendar."""
        self._warm_www()
        r = self.get(WWW + "/api/event-calendar", tries=2)
        if r is None:
            return {}
        try:
            return parse_event_calendar(r.json())
        except (ValueError, json.JSONDecodeError):
            return {}

    # ---------- regulatory news ----------
    def circulars(self, days: int = 7, limit: int = 8) -> list[dict]:
        feeds = [
            ("SEBI", "https://www.sebi.gov.in/sebirss.xml"),
            ("NSE", f"{ARCHIVES}/content/RSS/Circulars.xml"),
        ]
        out: list[dict] = []
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        for source, url in feeds:
            r = self.get(url, tries=2)
            if r is None:
                continue
            try:
                items = parse_rss(r.content)
            except ET.ParseError:
                continue
            kept = 0
            for it in items:
                when = it.get("published")
                if when is not None and when < cutoff:
                    continue
                out.append({
                    "source": source,
                    "title": it["title"],
                    "link": it["link"],
                    "date": when.date().isoformat() if when else None,
                })
                kept += 1
                if kept >= limit:
                    break
        return out


# ---------- parsers (pure functions, unit-testable) ----------
def parse_sec_bhavdata(content: bytes, d: dt.date) -> pd.DataFrame | None:
    df = pd.read_csv(io.BytesIO(content), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    need = {"SYMBOL", "SERIES", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS"}
    if not need.issubset(df.columns):
        return None
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df[df["SERIES"] == "EQ"]
    out = pd.DataFrame({
        "date": pd.Timestamp(d),
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "open": pd.to_numeric(df.get("OPEN_PRICE", float("nan")), errors="coerce"),
        "high": pd.to_numeric(df["HIGH_PRICE"], errors="coerce"),
        "low": pd.to_numeric(df["LOW_PRICE"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
        "prev_close": pd.to_numeric(df.get("PREV_CLOSE", float("nan")), errors="coerce"),
        "volume": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
        "turnover_cr": pd.to_numeric(df["TURNOVER_LACS"], errors="coerce") / 100.0,
    })
    return out.dropna(subset=["close"])[COLS]


def parse_udiff(content: bytes, d: dt.date) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = z.namelist()[0]
            df = pd.read_csv(z.open(name))
    except (zipfile.BadZipFile, IndexError):
        return None
    df.columns = [c.strip() for c in df.columns]
    if "SctySrs" not in df.columns:
        return None
    df = df[df["SctySrs"].astype(str).str.strip() == "EQ"]
    out = pd.DataFrame({
        "date": pd.Timestamp(d),
        "symbol": df["TckrSymb"].astype(str).str.strip(),
        "open": pd.to_numeric(df.get("OpnPric", float("nan")), errors="coerce"),
        "high": pd.to_numeric(df["HghPric"], errors="coerce"),
        "low": pd.to_numeric(df["LwPric"], errors="coerce"),
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "prev_close": pd.to_numeric(df.get("PrvsClsgPric", float("nan")), errors="coerce"),
        "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        "turnover_cr": pd.to_numeric(df["TtlTrfVal"], errors="coerce") / 1e7,
    })
    return out.dropna(subset=["close"])[COLS]


def parse_sec_list(content: bytes) -> dict[str, float]:
    df = pd.read_csv(io.BytesIO(content), skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    sym = next(c for c in df.columns if c.startswith("symbol"))
    band = next(c for c in df.columns if "band" in c)
    ser = next((c for c in df.columns if c.startswith("series")), None)
    if ser:
        df = df[df[ser].astype(str).str.strip() == "EQ"]
    vals = pd.to_numeric(df[band].astype(str).str.replace("%", "").str.strip(), errors="coerce")
    return {s.strip(): float(v) for s, v in zip(df[sym].astype(str), vals) if pd.notna(v)}


def parse_event_calendar(data) -> dict[str, dt.date]:
    rows = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
    out: dict[str, dt.date] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        purpose = f"{row.get('purpose', '')} {row.get('bm_desc', '')}".lower()
        if "result" not in purpose:
            continue
        sym = str(row.get("symbol", "")).strip()
        raw = str(row.get("date") or row.get("bm_date") or "").strip()
        when = None
        for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                when = dt.datetime.strptime(raw, fmt).date()
                break
            except ValueError:
                continue
        if sym and when and (sym not in out or when < out[sym]):
            out[sym] = when
    return out


def parse_old_bhav(content: bytes, d: dt.date) -> pd.DataFrame | None:
    """Pre-July-2024 'cmDDMONYYYYbhav.csv.zip' format."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            df = pd.read_csv(z.open(z.namelist()[0]))
    except (zipfile.BadZipFile, IndexError):
        return None
    df.columns = [c.strip() for c in df.columns]
    if "SERIES" not in df.columns:
        return None
    df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
    out = pd.DataFrame({
        "date": pd.Timestamp(d),
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH"], errors="coerce"),
        "low": pd.to_numeric(df["LOW"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "prev_close": pd.to_numeric(df["PREVCLOSE"], errors="coerce"),
        "volume": pd.to_numeric(df["TOTTRDQTY"], errors="coerce"),
        "turnover_cr": pd.to_numeric(df["TOTTRDVAL"], errors="coerce") / 1e7,
    })
    return out.dropna(subset=["close"])[COLS]


def _collect_symbols(obj) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == "symbol" and isinstance(v, str):
                found.add(v.strip())
            else:
                found |= _collect_symbols(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _collect_symbols(v)
    return found


def _parse_pub(raw: str):
    """RSS dates: RFC 822 (NSE) or SEBI's looser formats such as '13 Nov, 2026 +0530'."""
    from email.utils import parsedate_to_datetime
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        when = None
    if when is None:
        for fmt in ("%d %b, %Y %z", "%d %b, %Y", "%d %b %Y %H:%M:%S %z", "%d %b %Y", "%d-%m-%Y", "%Y-%m-%d",
                    "%b %d, %Y"):
            try:
                when = dt.datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
    if when is not None and when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when


def parse_rss(content: bytes) -> list[dict]:
    from email.utils import parsedate_to_datetime

    root = ET.fromstring(content)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate")
        when = None
        if pub:
            when = _parse_pub(pub.strip())
        if title:
            items.append({"title": title, "link": link, "published": when})
    return items
