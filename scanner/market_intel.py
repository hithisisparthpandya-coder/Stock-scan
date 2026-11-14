"""Market intelligence shared by the evening scan and the intraday live job:
live quotes, sector moves, India VIX, FII/DII flows, news headlines, NSE announcements,
the "why did it move" analysis of the day's best and worst performers, and the watchlist.

Every fetcher is optional: a failure returns an empty result and is reported in `sources`,
never stops the job.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import urllib.parse
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sentiment import adjusted_list, market_mood, name_keys, score_headline, stock_sentiment

IST = ZoneInfo("Asia/Kolkata")
WWW = "https://www.nseindia.com"

MARKET_FEEDS = [
    ("Economic Times", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/business.xml"),
    ("Mint", "https://www.livemint.com/rss/markets"),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss"),
]
SECTOR_INDICES = ["NIFTY BANK", "NIFTY IT", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY FMCG", "NIFTY METAL",
                  "NIFTY REALTY", "NIFTY ENERGY", "NIFTY MEDIA", "NIFTY PSU BANK", "NIFTY FINANCIAL SERVICES",
                  "NIFTY OIL & GAS", "NIFTY HEALTHCARE INDEX", "NIFTY CONSUMER DURABLES", "NIFTY INFRASTRUCTURE",
                  "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100"]


def _num(x):
    try:
        v = float(str(x).replace(",", ""))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- NSE live endpoints
def nse_json(nse, path: str):
    nse._warm_www()
    r = nse.get(WWW + path, tries=2, timeout=20)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def parse_index_quotes(data) -> tuple[pd.DataFrame, dict]:
    """NSE /api/equity-stockIndices response → per-stock quotes + advance/decline counts."""
    rows = []
    for d in (data or {}).get("data", []):
        meta = d.get("meta") or {}
        if not meta or d.get("priority") == 1:
            continue
        rows.append({"symbol": d.get("symbol"), "name": meta.get("companyName") or "",
                     "industry": meta.get("industry") or "", "open": _num(d.get("open")),
                     "high": _num(d.get("dayHigh")), "low": _num(d.get("dayLow")), "last": _num(d.get("lastPrice")),
                     "prev_close": _num(d.get("previousClose")), "pchange": _num(d.get("pChange")),
                     "volume": _num(d.get("totalTradedVolume")),
                     "value_cr": (_num(d.get("totalTradedValue")) or 0) / 1e7,
                     "year_high": _num(d.get("yearHigh")), "year_low": _num(d.get("yearLow")),
                     "chg30d": _num(d.get("perChange30d"))})
    adv = (data or {}).get("advance") or {}
    ad = {"advances": int(_num(adv.get("advances")) or 0), "declines": int(_num(adv.get("declines")) or 0),
          "unchanged": int(_num(adv.get("unchanged")) or 0)}
    df = pd.DataFrame(rows)
    if len(df):
        df = df.dropna(subset=["symbol", "last"]).drop_duplicates("symbol").set_index("symbol")
    return df, ad


def parse_quote_equity(sym: str, data) -> dict | None:
    if not data:
        return None
    pi, info = data.get("priceInfo") or {}, data.get("info") or {}
    hl = pi.get("intraDayHighLow") or {}
    whl = pi.get("weekHighLow") or {}
    last = _num(pi.get("lastPrice"))
    if not last:
        return None
    return {"symbol": sym, "name": info.get("companyName") or "", "industry": (data.get("industryInfo") or {}).get("basicIndustry") or info.get("industry") or "",
            "open": _num(pi.get("open")), "high": _num(hl.get("max")), "low": _num(hl.get("min")), "last": last,
            "prev_close": _num(pi.get("previousClose")), "pchange": _num(pi.get("pChange")), "volume": None,
            "value_cr": None, "year_high": _num(whl.get("max")), "year_low": _num(whl.get("min")), "chg30d": None}


def parse_all_indices(data) -> dict:
    out = {}
    for d in (data or {}).get("data", []):
        name = d.get("index") or d.get("indexSymbol")
        if name:
            out[name] = {"last": _num(d.get("last")), "pchange": _num(d.get("percentChange")),
                         "prev_close": _num(d.get("previousClose")),
                         "advances": _num(d.get("advances")), "declines": _num(d.get("declines"))}
    return out


def parse_market_status(data) -> dict:
    for m in (data or {}).get("marketState", []):
        if m.get("market") == "Capital Market":
            return {"status": m.get("marketStatus"), "trade_date": m.get("tradeDate"),
                    "message": m.get("marketStatusMessage")}
    return {}


def parse_fii_dii(data) -> dict:
    out = {}
    for d in data or []:
        cat = str(d.get("category", "")).upper()
        key = "fii" if "FII" in cat or "FPI" in cat else "dii" if "DII" in cat else None
        if key:
            out[key] = _num(d.get("netValue"))
            out["date"] = d.get("date")
    return out


def parse_announcements(data, since: dt.datetime) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d in data or []:
        sym, desc = d.get("symbol"), (d.get("desc") or "").strip()
        when = None
        for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M"):
            try:
                when = dt.datetime.strptime(str(d.get("an_dt") or d.get("sort_date") or ""), fmt).replace(tzinfo=IST)
                break
            except ValueError:
                continue
        if not sym or not desc or (when and when < since):
            continue
        text = (d.get("attchmntText") or "").strip()
        out.setdefault(sym, []).append({"title": f"{desc}: {text[:160]}" if text else desc,
                                        "time": when.isoformat() if when else None,
                                        "link": d.get("attchmntFile") or ""})
    return out


# ---------------------------------------------------------------- Yahoo fallbacks
def yahoo_quotes(symbols: list[str]) -> pd.DataFrame:
    """Fallback when NSE's live API is unreachable: today's daily bar so far from Yahoo (~15 min delay)."""
    try:
        import yfinance as yf
        tick = [s + ".NS" for s in symbols]
        y = yf.download(tick, period="5d", interval="1d", group_by="ticker", progress=False,
                        auto_adjust=False, threads=True)
        rows = []
        for s, t in zip(symbols, tick):
            try:
                df = y[t].dropna(subset=["Close"]) if len(tick) > 1 else y.dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df) < 2:
                continue
            a, b = df.iloc[-1], df.iloc[-2]
            rows.append({"symbol": s, "name": "", "industry": "", "open": float(a["Open"]), "high": float(a["High"]),
                         "low": float(a["Low"]), "last": float(a["Close"]), "prev_close": float(b["Close"]),
                         "pchange": (float(a["Close"]) / float(b["Close"]) - 1) * 100, "volume": float(a["Volume"]),
                         "value_cr": float(a["Volume"]) * float(a["Close"]) / 1e7, "year_high": None,
                         "year_low": None, "chg30d": None})
        return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
    except Exception as e:
        print("Yahoo quotes failed:", type(e).__name__, str(e)[:120])
        return pd.DataFrame()


def yahoo_vix_history() -> list[float]:
    try:
        import yfinance as yf
        y = yf.download("^INDIAVIX", period="1y", interval="1d", progress=False, auto_adjust=False)
        col = y["Close"]
        col = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
        return [round(float(v), 2) for v in col.dropna().values]
    except Exception as e:
        print("VIX history failed:", type(e).__name__)
        return []


# ---------------------------------------------------------------- news
def fetch_feed(nse, url: str, source: str) -> list[dict]:
    from nse import parse_rss
    r = nse.get(url, tries=1, timeout=15)
    if r is None:
        return []
    try:
        items = parse_rss(r.content)
    except Exception:
        return []
    return [{**it, "source": source} for it in items]


def market_headlines(nse) -> tuple[list[dict], list[str]]:
    out, ok = [], []
    for source, url in MARKET_FEEDS:
        items = fetch_feed(nse, url, source)
        if items:
            ok.append(source)
            out += items
    return out, sorted(set(ok))


def google_news(nse, name: str, symbol: str) -> list[dict]:
    q = f'"{name}" share OR stock' if name else f"{symbol} NSE share"
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q + " when:3d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    items = fetch_feed(nse, url, "Google News")
    for it in items:                       # Google appends " - Publisher" to titles
        if " - " in it["title"]:
            t, pub = it["title"].rsplit(" - ", 1)
            it["title"], it["source"] = t, pub
    return items[:15]


def news_tone(headlines: list[dict], now: dt.datetime, hours: float = 24) -> dict:
    """Average score of market headlines from the last `hours` that carry any sentiment."""
    scores, top = [], []
    seen = set()
    for it in headlines:
        p = it.get("published")
        if p and (now - p).total_seconds() > hours * 3600:
            continue
        k = it["title"].lower()[:70]
        if k in seen:
            continue
        seen.add(k)
        s = score_headline(it["title"])
        if s != 0:
            scores.append(s)
            top.append({"title": it["title"], "link": it.get("link", ""), "source": it.get("source", ""), "score": s})
    top.sort(key=lambda x: -abs(x["score"]))
    return {"avg": round(float(np.mean(scores)), 3) if scores else None, "count": len(scores),
            "positive": sum(1 for s in scores if s > 0), "negative": sum(1 for s in scores if s < 0),
            "top_positive": [x for x in top if x["score"] > 0][:4], "top_negative": [x for x in top if x["score"] < 0][:4]}


def collect_stock_news(nse, targets: dict[str, str], market_items: list[dict], cache: dict, now: dt.datetime,
                       max_queries: int = 40, ttl_min: int = 45, demo_items: dict | None = None) -> dict:
    """News sentiment for each target symbol (symbol → company name): Google News search (cached for
    `ttl_min` minutes) plus any market-feed headline that names the company."""
    out, queries = {}, 0
    for sym, name in targets.items():
        keys = name_keys(sym, name)
        c = cache.get(sym) or {}
        fresh = c.get("fetched") and (now - dt.datetime.fromisoformat(c["fetched"])).total_seconds() < ttl_min * 60
        if demo_items is not None:
            items = demo_items.get(sym, [])
        elif fresh:
            items = [{**it, "published": dt.datetime.fromisoformat(it["published"]) if it.get("published") else None}
                     for it in c.get("items", [])]
        elif queries < max_queries:
            items = google_news(nse, name, sym)
            queries += 1
            cache[sym] = {"fetched": now.isoformat(timespec="minutes"),
                          "items": [{**it, "published": it["published"].isoformat() if it.get("published") else None}
                                    for it in items]}
        else:
            items = []
        own = [it for it in items if not keys or any(k in it["title"].lower() for k in keys + [sym.lower()])]
        feed = [it for it in market_items if keys and any(re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", it["title"].lower()) for k in keys)]
        out[sym] = stock_sentiment(own + feed, keys, now, already_matched=True)
    return out


# ---------------------------------------------------------------- movers analysis
def vix_percentile(vix: float | None, hist: list[float]) -> float | None:
    if vix is None or len(hist) < 60:
        return None
    return float(np.mean(np.asarray(hist) <= vix))


def day_fraction(now: dt.datetime) -> float:
    """Share of the 9:15–15:30 session elapsed (volume pacing)."""
    t = now.astimezone(IST)
    mins = (t.hour * 60 + t.minute) - (9 * 60 + 15)
    return min(1.0, max(0.05, mins / 375))


def explain_move(sym: str, q: dict, ctx: dict, peer: dict | None, news: dict | None, ann: list[dict],
                 results_date: str | None, frac: float, today: dt.date) -> dict:
    """Rule-based 'why did it move' for one stock. Returns reasons (strongest first) and a summary."""
    reasons = []          # (strength, text, kind)
    ch = q.get("pchange") or 0.0
    up = ch >= 0
    o, pc, last, hi, lo = q.get("open"), q.get("prev_close"), q.get("last"), q.get("high"), q.get("low")
    # news & corporate events
    if ann:
        reasons.append((3.0, f"Company announcement: {ann[0]['title'][:140]}", "event"))
    if results_date:
        rd = dt.date.fromisoformat(results_date)
        if abs((rd - today).days) <= 1:
            reasons.append((2.8, f"Quarterly results {'today' if rd == today else 'due' if rd > today else 'just out'} ({rd.strftime('%d %b')})", "event"))
    if news and news.get("count"):
        h = news["headlines"][0]
        agree = (news["N"] > 0.1 and up) or (news["N"] < -0.1 and not up)
        reasons.append((2.5 if agree else 1.2, f"News {'+' if news['N'] >= 0 else ''}{news['N']:.2f}: “{h['title'][:120]}”", "news"))
        if news.get("red_flag"):
            reasons.append((3.0, "Red-flag headline (regulator / fraud / default) – treat with caution", "news"))
    # sector
    if peer and peer.get("n", 0) >= 3:
        pm = peer["median"]
        if abs(pm) >= 1.0 and (pm > 0) == up:
            reasons.append((2.2, f"Sector-wide move: {peer['industry']} peers median {pm:+.1f}% ({peer['n']} stocks)", "sector"))
        elif abs(pm) < 0.5:
            reasons.append((1.5, f"Stock-specific: {peer['industry']} peers flat ({pm:+.1f}%)", "sector"))
        elif (pm > 0) != up:
            reasons.append((1.6, f"Against its sector: {peer['industry']} peers {pm:+.1f}%", "sector"))
    # volume
    av = ctx.get("avgvol20")
    if q.get("volume") and av:
        vr = q["volume"] / (av * frac)
        if vr >= 2:
            reasons.append((2.0, f"Heavy volume: {vr:.1f}× a normal day's pace – strong conviction", "volume"))
        elif vr < 0.8:
            reasons.append((1.0, f"Light volume ({vr:.1f}× normal pace) – move may not hold", "volume"))
    # gap & intraday shape
    if o and pc:
        gap = (o / pc - 1) * 100
        if abs(gap) >= 2:
            reasons.append((1.8, f"Gapped {'up' if gap > 0 else 'down'} {gap:+.1f}% at the open – overnight news or orders", "price"))
    if hi and lo and last and hi > lo:
        pos = (last - lo) / (hi - lo)
        if up and pos < 0.35 and ch > 0:
            reasons.append((1.4, "Faded from the day's high – sellers stepped in", "price"))
        elif not up and pos > 0.65:
            reasons.append((1.4, "Recovered from the day's low – buyers stepped in", "price"))
        elif up and pos > 0.9:
            reasons.append((1.2, "Closing near the day's high", "price"))
        elif not up and pos < 0.1:
            reasons.append((1.2, "Closing near the day's low", "price"))
    # trend context
    s50, hh20, ll20 = ctx.get("sma50"), ctx.get("hh20"), ctx.get("ll20")
    if last and s50:
        if up and hh20 and last > hh20:
            reasons.append((1.7, f"Broke above its 20-day high (₹{hh20:,.0f})" + (" in an uptrend" if last > s50 else ""), "trend"))
        elif up and last < s50:
            reasons.append((1.3, "Bounce inside a downtrend (still below its 50-day average)", "trend"))
        elif not up and ll20 and last < ll20:
            reasons.append((1.7, f"Broke below its 20-day low (₹{ll20:,.0f})", "trend"))
        elif not up and pc and pc > s50 >= last:
            reasons.append((1.6, "Fell through its 50-day average", "trend"))
    yh, yl = q.get("year_high"), q.get("year_low")
    if up and yh and last and last >= yh * 0.995:
        reasons.append((1.6, "At a 52-week high", "trend"))
    if not up and yl and last and last <= yl * 1.005:
        reasons.append((1.6, "At a 52-week low", "trend"))
    band = ctx.get("band")
    if band and abs(ch) >= band * 0.98:
        reasons.append((2.6, f"Hit its {band:g}% {'upper' if up else 'lower'} circuit", "price"))
    reasons.sort(key=lambda r: -r[0])
    kinds = {k for _, _, k in reasons[:3]}
    if kinds & {"event", "news"}:
        verdict = "Likely news / event driven"
    elif "sector" in kinds and any("Sector-wide" in t for _, t, _ in reasons):
        verdict = "Moving with its sector"
    elif reasons:
        verdict = "Technical / flow driven – no clear news found"
    else:
        verdict = "No clear driver found"
    return {"verdict": verdict, "reasons": [t for _, t, _ in reasons[:5]]}


def peer_moves(quotes: pd.DataFrame) -> dict[str, dict]:
    out = {}
    if not len(quotes) or "industry" not in quotes:
        return out
    for ind, g in quotes[quotes["industry"].astype(bool)].groupby("industry"):
        out[ind] = {"industry": ind, "median": round(float(g["pchange"].median()), 2), "n": int(len(g))}
    return out


def movers(quotes: pd.DataFrame, universe: dict, peers: dict, news: dict, anns: dict, results: dict,
           now: dt.datetime, n: int = 10, min_value_cr: float = 5.0) -> dict:
    q = quotes[(quotes["value_cr"].fillna(0) >= min_value_cr) | quotes["value_cr"].isna()].copy()
    q = q.dropna(subset=["pchange"])
    frac, today = day_fraction(now), now.astimezone(IST).date()
    ctxs = universe.get("stocks", {})

    def pack(df):
        rows = []
        for sym, r in df.iterrows():
            rd = r.to_dict()
            ex = explain_move(sym, rd, ctxs.get(sym, {}), peers.get(rd.get("industry") or ""),
                              news.get(sym), anns.get(sym, []), results.get(sym), frac, today)
            rows.append({"symbol": sym, "name": rd.get("name") or ctxs.get(sym, {}).get("name", ""),
                         "industry": rd.get("industry") or "", "last": rd.get("last"),
                         "pchange": round(rd["pchange"], 2), "value_cr": round(rd["value_cr"], 1) if rd.get("value_cr") else None,
                         **ex, "headlines": (news.get(sym) or {}).get("headlines", [])[:3]})
        return rows
    return {"gainers": pack(q.sort_values("pchange", ascending=False).head(n)),
            "losers": pack(q.sort_values("pchange").head(n))}


def mover_takeaways(mv: dict, peers: dict) -> list[str]:
    """A few plain-English lines about the day's movers."""
    out = []
    for side, word in (("gainers", "gainers"), ("losers", "losers")):
        rows = mv.get(side, [])
        if not rows:
            continue
        inds = pd.Series([r["industry"] for r in rows if r["industry"]]).value_counts()
        if len(inds) and inds.iloc[0] >= 3:
            out.append(f"{inds.iloc[0]} of the top 10 {word} are {inds.index[0]} – a sector theme, not single-stock news.")
        newsy = sum(1 for r in rows if r["verdict"].startswith("Likely news"))
        out.append(f"{newsy} of 10 {word} have a news or event explanation; the rest look technical or flow-driven.")
    if peers:
        s = sorted(peers.values(), key=lambda p: p["median"])
        s = [p for p in s if p["n"] >= 5]
        if s:
            out.append(f"Strongest industry: {s[-1]['industry']} ({s[-1]['median']:+.1f}%). Weakest: {s[0]['industry']} ({s[0]['median']:+.1f}%).")
    return out


# ---------------------------------------------------------------- universe file (written each evening)
def build_universe(P: dict, f: dict, names: dict, bands: dict, results: dict, cfg: dict, vix_hist: list[float],
                   industries: dict | None = None) -> dict:
    """Context the live job needs about each liquid stock, from the evening's price history."""
    C = P["C"]
    i = len(C) - 1
    turn = f["avgturn20"].iloc[i]
    liquid = turn[turn >= cfg["min_turnover_cr"]].index
    hh20 = P["H"].iloc[-20:].max() if P["H"] is not None else C.iloc[-20:].max()
    ll20 = P["L"].iloc[-20:].min() if P["L"] is not None else C.iloc[-20:].min()
    avgvol = P["V"].iloc[-20:].mean()
    stocks = {}
    for s in liquid:
        c = C[s].iloc[i]
        if not (c > 0):
            continue
        g = lambda x: None if x is None or pd.isna(x) else round(float(x), 2)
        stocks[s] = {"name": names.get(s, ""), "close": g(c), "sma50": g(f["sma50"][s].iloc[i]),
                     "sma200": g(f["sma200"][s].iloc[i]), "hh20": g(hh20.get(s)), "ll20": g(ll20.get(s)),
                     "avgvol20": g(avgvol.get(s)), "band": bands.get(s),
                     "industry": (industries or {}).get(s, "")}
    today = C.index[-1].date()
    res = {s: d.isoformat() for s, d in results.items() if 0 <= (d - today).days <= 7} if results else {}
    return {"as_of": today.isoformat(), "breadth": None if pd.isna(f["breadth"].iloc[i]) else round(float(f["breadth"].iloc[i]), 3),
            "nifty_sma50": None if pd.isna(f["nifty50"].iloc[i]) else round(float(f["nifty50"].iloc[i]), 2),
            "vix_hist": vix_hist[-260:], "results": res, "stocks": stocks}


def breadth_live(quotes: pd.DataFrame, universe: dict) -> float | None:
    st = universe.get("stocks", {})
    n = above = 0
    for sym, r in quotes.iterrows():
        s50 = (st.get(sym) or {}).get("sma50")
        if s50 and r.get("last"):
            n += 1
            above += r["last"] > s50
    return above / n if n >= 50 else universe.get("breadth")


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


# ---------------------------------------------------------------- evening: universe + sentiment list
def evening_intel(nse, P: dict, f: dict, picks: list[dict], names: dict, bands: dict, results: dict,
                  cfg: dict, now: dt.datetime, demo: bool = False) -> tuple[dict, dict]:
    """Returns (universe.json content, sentiment section for picks.json). `nse` is None in demo mode."""
    utc_now = now.astimezone(dt.timezone.utc)
    sources = {}
    if demo:
        vix_hist = list(np.round(np.linspace(11, 20, 250), 2))
        snap, idx, flows, mkt_items = pd.DataFrame(), {"INDIA VIX": {"last": 13.5}}, {"fii": -900.0, "dii": 1400.0}, [
            {"title": "Nifty ends higher as banks rally; FII buying continues", "link": "", "source": "Demo",
             "published": utc_now - dt.timedelta(hours=3)}]
        stock_items = {p["symbol"]: [] for p in picks}
        if picks:
            stock_items[picks[-1]["symbol"]] = [{"title": f"{picks[-1]['symbol']} shares slump after SEBI probe report",
                                                 "link": "", "source": "Demo", "published": utc_now - dt.timedelta(hours=5)}]
    else:
        vix_hist = yahoo_vix_history()
        snap, _ = parse_index_quotes(nse_json(nse, "/api/equity-stockIndices?index=NIFTY%20500"))
        idx = parse_all_indices(nse_json(nse, "/api/allIndices"))
        flows = parse_fii_dii(nse_json(nse, "/api/fiidiiTradeReact"))
        mkt_items, ok = market_headlines(nse)
        sources = {"vix_history": len(vix_hist), "nifty500_snapshot": len(snap), "indices": bool(idx),
                   "fii_dii": bool(flows), "news_feeds": ok}
        stock_items = None
    industries = snap["industry"].to_dict() if len(snap) else {}
    universe = build_universe(P, f, names, bands, results, cfg, vix_hist, industries)

    C = P["C"]
    liq = list(universe["stocks"])
    ch = (C[liq].iloc[-1] / C[liq].iloc[-2] - 1).dropna() if len(C) > 1 and liq else pd.Series(dtype=float)
    adv, dec = int((ch > 0).sum()), int((ch < 0).sum())
    vix = (idx.get("INDIA VIX") or {}).get("last") or (vix_hist[-1] if vix_hist else None)
    tone = news_tone(mkt_items, utc_now)
    i = len(C) - 1
    mood = market_mood(breadth=universe["breadth"], advances=adv, declines=dec, vix=vix,
                       vix_pct=vix_percentile(vix, vix_hist), nifty=float(f["nifty"].iloc[i]) if not pd.isna(f["nifty"].iloc[i]) else None,
                       nifty_sma50=universe["nifty_sma50"], news_avg=tone["avg"],
                       flows_cr=(flows.get("fii") or 0) + (flows.get("dii") or 0) if flows.get("fii") is not None else None)
    targets = {p["symbol"]: p.get("name") or names.get(p["symbol"], "") for p in picks}
    news = collect_stock_news(nse, targets, mkt_items, {}, utc_now, demo_items=stock_items)
    peers = peer_moves(snap.assign(pchange=snap["pchange"])) if len(snap) else {}
    base = [{**p, "industry": industries.get(p["symbol"], "")} for p in picks]
    lst = adjusted_list(base, news, {k: v["median"] for k, v in peers.items()}, mood)
    sent = {"mood": mood, "news_tone": tone, "list": lst, "flows": flows, "sources": sources,
            "vix": vix, "advances": adv, "declines": dec}
    return universe, sent
