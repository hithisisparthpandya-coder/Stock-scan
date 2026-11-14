"""Intraday live job (every ~15 minutes in market hours, GitHub Actions → docs/data/live.json).

  python scanner/live.py          # real run
  python scanner/live.py --demo   # synthetic quotes and headlines, no network
  FORCE=1 python scanner/live.py  # run even outside market hours

What it does each run:
  • Live prices for the Nifty 500 (NSE's own index snapshot; Yahoo as fallback) and all watched stocks.
  • Best and worst performers of the day, each with a rule-based "why it moved" analysis
    (news, results, NSE announcements, sector, volume, gap, trend, circuit).
  • Market Mood (sentiment equation in sentiment.py) from breadth, advancers, India VIX, Nifty trend,
    news tone and FII/DII flows.
  • Your watchlist: today's picks, paper orders and paper positions with live P&L and stop distance.
  • The sentiment-adjusted ("dynamic") list, refreshed with live prices and news.
  • Telegram pings only for big moves: ±3% on a watched stock, a stop or target touched.
Never places orders and never changes the strategy or the paper ledgers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from market_intel import (IST, REF_INDEX, SECTOR_INDICES, breadth_live, collect_stock_news, load_json, nifty500_list,  # noqa: E402
                          market_headlines, mover_takeaways, movers, news_tone, nse_json, parse_all_indices,
                          parse_announcements, parse_fii_dii, parse_index_quotes, parse_market_status,
                          parse_quote_equity, peer_moves, vix_percentile, yahoo_quotes)
from sentiment import adjusted_list, market_mood  # noqa: E402

DATA = ROOT / "docs" / "data"
OUT = DATA / "live.json"
ALERT_MOVE = 3.0


def in_session(now: dt.datetime) -> bool:
    t = now.astimezone(IST)
    m = t.hour * 60 + t.minute
    return t.weekday() < 5 and 9 * 60 + 5 <= m <= 15 * 60 + 50


def watchlist(picks: list[dict], ledger: dict, sent_list: list[dict], quotes: pd.DataFrame) -> list[dict]:
    rows = {}

    def q(sym):
        return quotes.loc[sym].to_dict() if sym in quotes.index else {}

    for p in picks:
        rows[p["symbol"]] = {"symbol": p["symbol"], "role": "Today's pick", "entry": p["entry"], "stop": p["stop"],
                             "target": p.get("target"), "qty": p.get("qty")}
    for o in ledger.get("queued", []):
        r = rows.setdefault(o["symbol"], {"symbol": o["symbol"], "entry": None})
        r.update({"role": "Paper order (next open)", "limit": o["limit"], "stop": o["stop"]})
    for p in ledger.get("open_positions", []):
        rows[p["symbol"]] = {"symbol": p["symbol"], "role": "Paper position", "entry": p["entry"], "stop": p["stop"],
                             "qty": p["qty"], "target": None}
    for s in sent_list:
        if s.get("status") == "suggested" and s["symbol"] not in rows:
            rows[s["symbol"]] = {"symbol": s["symbol"], "role": "Sentiment pick", "entry": s["entry"], "stop": s["stop"],
                                 "target": s.get("target"), "qty": s.get("qty")}
    out = []
    for sym, r in rows.items():
        x = q(sym)
        last = x.get("last")
        r.update({"name": x.get("name", ""), "last": last, "pchange": None if x.get("pchange") is None else round(x["pchange"], 2),
                  "open": x.get("open"), "low": x.get("low"), "high": x.get("high")})
        if last and r.get("entry"):
            r["from_entry_pct"] = round((last / r["entry"] - 1) * 100, 2)
            if r.get("qty"):
                r["pnl"] = round((last - r["entry"]) * r["qty"], 0)
        if last and r.get("stop"):
            r["to_stop_pct"] = round((last / r["stop"] - 1) * 100, 2)
        flags = []
        if r.get("stop") and x.get("low") and x["low"] <= r["stop"]:
            flags.append("stop touched")
        if r.get("target") and x.get("high") and x["high"] >= r["target"]:
            flags.append("target touched")
        lim = r.get("limit") or (r["entry"] * 1.01 if r.get("entry") and r["role"] != "Paper position" else None)
        if lim and x.get("open") and r["role"] != "Paper position" and x["open"] > lim:
            flags.append("opened above the limit – skip (rule: no chasing)")
        r["flags"] = flags
        out.append(r)
    return out


def alerts_for(watch: list[dict], sent: set[str], today: str) -> list[tuple[str, str]]:
    new = []
    for w in watch:
        sym = w["symbol"]
        for f in w["flags"]:
            if f.startswith("opened above"):
                continue
            k = f"{today}|{sym}|{f}"
            if k not in sent:
                new.append((k, f"{'🛑' if 'stop' in f else '🎯'} {sym} ({w['role']}): {f} · last ₹{w['last']:,.2f}"))
        ch = w.get("pchange")
        if ch is not None and abs(ch) >= ALERT_MOVE:
            k = f"{today}|{sym}|move{'+' if ch > 0 else '-'}"
            if k not in sent:
                new.append((k, f"{'📈' if ch > 0 else '📉'} {sym} ({w['role']}) {ch:+.1f}% today · last ₹{w['last']:,.2f}"))
    return new


def send_alerts(lines: list[str], app_url: str):
    from notify import send_telegram
    if lines:
        send_telegram("⚡ <b>Intraday alert</b>\n" + "\n".join(lines) + (f"\n{app_url}" if app_url else ""))


def demo_inputs(universe: dict, now: dt.datetime):
    rng = np.random.default_rng(int(now.strftime("%H%M")) if os.environ.get("DEMO_RANDOM") else 3)
    inds = ["Capital Goods", "Information Technology", "Financial Services", "Healthcare", "Automobile"]
    rows = []
    for k, (s, c) in enumerate(universe["stocks"].items()):
        ch = float(rng.normal(0.3, 2.2))
        if k == 0:
            ch = 7.5
        pc = c["close"]
        last = pc * (1 + ch / 100)
        rows.append({"symbol": s, "name": c.get("name") or s.title(), "industry": c.get("industry") or inds[k % len(inds)],
                     "open": pc * (1 + ch / 300), "high": max(last, pc) * 1.004, "low": min(last, pc) * 0.996,
                     "last": last, "prev_close": pc, "pchange": ch, "volume": (c.get("avgvol20") or 1e5) * rng.uniform(0.5, 3),
                     "value_cr": 20.0, "year_high": pc * 1.3, "year_low": pc * 0.7, "chg30d": None})
    quotes = pd.DataFrame(rows).set_index("symbol")
    adv = {"advances": int((quotes.pchange > 0).sum()), "declines": int((quotes.pchange < 0).sum()), "unchanged": 0}
    first = quotes.index[0]
    utc = dt.timezone.utc
    headlines = [{"title": "Sensex, Nifty rise as FII inflows return; metals rally", "link": "", "source": "Demo",
                  "published": now.astimezone(utc) - dt.timedelta(hours=2)},
                 {"title": "Rupee slips; crude prices weigh on oil marketing companies", "link": "", "source": "Demo",
                  "published": now.astimezone(utc) - dt.timedelta(hours=3)}]
    stock_items = {first: [{"title": f"{quotes.loc[first, 'name']} bags order worth Rs 1,200 crore", "link": "",
                            "source": "Demo", "published": now.astimezone(utc) - dt.timedelta(hours=1)}]}
    indices = {"NIFTY 50": {"last": 25350.0, "pchange": 0.42}, "INDIA VIX": {"last": 13.2, "pchange": -2.1},
               "NIFTY BANK": {"last": 52000.0, "pchange": 0.8}, "NIFTY IT": {"last": 41000.0, "pchange": -0.6}}
    return quotes, adv, headlines, stock_items, indices, {"fii": 1250.0, "dii": 800.0, "date": "demo"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()
    now = dt.datetime.now(IST)
    utc_now = now.astimezone(dt.timezone.utc)
    prev = load_json(OUT, {}) or {}
    if not args.demo and not os.environ.get("FORCE") and not in_session(now):
        print("Outside market hours; nothing to do.")
        return

    cfg = json.loads((ROOT / "scanner" / "config.json").read_text())
    app_url = os.environ.get("APP_URL", cfg.get("app_url", ""))
    universe = load_json(DATA / "universe.json", {"stocks": {}}) or {"stocks": {}}
    picks_doc = load_json(DATA / "picks.json", {}) or {}
    ledger = load_json(DATA / "ledger.json", {}) or {}
    picks = picks_doc.get("picks", [])
    sent_doc = picks_doc.get("sentiment") or {}
    sources = {}

    if args.demo:
        quotes, adv, mkt_items, demo_stock_items, indices, flows = demo_inputs(universe, now)
        status, anns = {"status": "Demo"}, {}
        nse = None
        sources = {"prices": "demo"}
    else:
        from nse import NSE
        nse = NSE(pause=0.2)
        status = parse_market_status(nse_json(nse, "/api/marketStatus"))
        if not os.environ.get("FORCE") and status.get("status") and status["status"].lower() != "open":
            last_run = prev.get("generated_at", "")
            if last_run[:10] == now.date().isoformat() and last_run[11:16] >= "15:30":
                print(f"Market {status['status']} and today's closing snapshot is already saved.")
                return
        quotes, adv = parse_index_quotes(nse_json(nse, "/api/equity-stockIndices?index=NIFTY%20500", REF_INDEX))
        sources["prices"] = "NSE Nifty 500 snapshot" if len(quotes) else None
        want = {p["symbol"] for p in picks} | {p["symbol"] for p in ledger.get("open_positions", [])} | \
               {o["symbol"] for o in ledger.get("queued", [])} | {s["symbol"] for s in sent_doc.get("list", [])}
        missing = [s for s in want if s not in quotes.index]
        if len(quotes):
            extra = [parse_quote_equity(s, nse_json(nse, f"/api/quote-equity?symbol={s}")) for s in missing[:25]]
            extra = [e for e in extra if e]
            if extra:
                quotes = pd.concat([quotes, pd.DataFrame(extra).set_index("symbol")])
        n500 = nifty500_list(nse)
        if not len(quotes):
            syms = list(n500) or list(universe.get("stocks", {}))[:600]
            quotes = yahoo_quotes(sorted(set(syms) | want))
            sources["prices"] = "Yahoo Finance (~15 min delayed)" if len(quotes) else None
            adv = {"advances": int((quotes.get("pchange", pd.Series(dtype=float)) > 0).sum()),
                   "declines": int((quotes.get("pchange", pd.Series(dtype=float)) < 0).sum()), "unchanged": 0}
        if not len(quotes):
            print("No live prices from NSE or Yahoo; keeping the previous snapshot.")
            sys.exit(1)
        for s in quotes.index:            # fill names/industries from the Nifty 500 list and the evening universe
            u = {**universe.get("stocks", {}).get(s, {}), **{k: v for k, v in n500.get(s, {}).items() if v}}
            if not quotes.at[s, "name"] and u.get("name"):
                quotes.at[s, "name"] = u["name"]
            if not quotes.at[s, "industry"] and u.get("industry"):
                quotes.at[s, "industry"] = u["industry"]
        sources["industries"] = int((quotes["industry"].astype(str) != "").sum())
        indices = parse_all_indices(nse_json(nse, "/api/allIndices"))
        sources["indices"] = bool(indices)
        flows = parse_fii_dii(nse_json(nse, "/api/fiidiiTradeReact"))
        sources["fii_dii"] = bool(flows)
        anns = parse_announcements(nse_json(nse, "/api/corporate-announcements?index=equities"),
                                   now - dt.timedelta(hours=24))
        sources["announcements"] = bool(anns)
        mkt_items, ok = market_headlines(nse)
        sources["news_feeds"] = ok
        demo_stock_items = None

    # sector + peer moves
    peers = peer_moves(quotes)
    sectors = sorted([{"name": k.replace("NIFTY ", "").title().replace("Psu", "PSU").replace("It", "IT").replace("Fmcg", "FMCG"),
                       "pchange": v["pchange"]} for k, v in indices.items() if k in SECTOR_INDICES and v.get("pchange") is not None],
                     key=lambda x: -x["pchange"])
    nifty = indices.get("NIFTY 50", {})
    vix = indices.get("INDIA VIX", {})

    # news
    tone = news_tone(mkt_items, utc_now)
    gainers_losers = pd.concat([quotes.sort_values("pchange", ascending=False).head(10), quotes.sort_values("pchange").head(10)])
    targets = {s: (quotes.at[s, "name"] if s in quotes.index else "") for s in
               list(dict.fromkeys(list(gainers_losers.index) + [p["symbol"] for p in picks] +
                                  [p["symbol"] for p in ledger.get("open_positions", [])] +
                                  [s["symbol"] for s in sent_doc.get("list", [])]))}
    cache = prev.get("news_cache", {}) if prev.get("generated_at", "")[:10] == now.date().isoformat() else {}
    news = collect_stock_news(nse, targets, mkt_items, cache, utc_now, demo_items=demo_stock_items)

    # mood
    mood = market_mood(breadth=breadth_live(quotes, universe), advances=adv.get("advances"), declines=adv.get("declines"),
                       vix=vix.get("last"), vix_pct=vix_percentile(vix.get("last"), universe.get("vix_hist", [])),
                       nifty=nifty.get("last"), nifty_sma50=universe.get("nifty_sma50"), news_avg=tone["avg"],
                       flows_cr=(flows.get("fii") or 0) + (flows.get("dii") or 0) if flows.get("fii") is not None else None)

    # movers
    results = universe.get("results", {})
    mv = movers(quotes, universe, peers, news, anns, results, now)

    # dynamic sentiment list on live data
    base = [{**p, "industry": quotes.at[p["symbol"], "industry"] if p["symbol"] in quotes.index else ""} for p in picks]
    dyn = adjusted_list(base, news, {k: v["median"] for k, v in peers.items()}, mood)
    for d in dyn:
        x = quotes.loc[d["symbol"]].to_dict() if d["symbol"] in quotes.index else {}
        d["last"], d["pchange"] = x.get("last"), x.get("pchange")
        if x.get("open") and x["open"] > d["entry"] * 1.01:
            d["live_note"] = "opened >1% above entry – rules say skip today"
        elif x.get("low") and x["low"] <= d["stop"]:
            d["live_note"] = "traded below the stop – setup failed"

    watch = watchlist(picks, ledger, sent_doc.get("list", []), quotes)
    today = now.date().isoformat()
    sent = set(prev.get("alerts_sent", [])) if prev.get("generated_at", "")[:10] == today else set()
    new = alerts_for(watch, sent, today)
    if new and not args.demo and not args.no_telegram:
        send_alerts([t for _, t in new], app_url)
    sent |= {k for k, _ in new}

    out = {
        "generated_at": now.isoformat(timespec="minutes"), "demo": bool(args.demo),
        "market_status": status.get("status") or ("Open" if in_session(now) else "Closed"),
        "delay_note": "NSE website snapshot, refreshed about every 15 minutes in market hours (GitHub's scheduler can run late)."
                      if sources.get("prices", "").startswith("NSE") else "Prices ~15 min delayed (Yahoo), refreshed about every 15 minutes.",
        "sources": sources,
        "nifty": {"last": nifty.get("last"), "pchange": nifty.get("pchange")},
        "vix": {"last": vix.get("last"), "pchange": vix.get("pchange")},
        "advance_decline": adv, "flows": flows, "sectors": sectors,
        "industries": sorted([p for p in peers.values() if p["n"] >= 3], key=lambda p: -p["median"]),
        "mood": mood, "news_tone": tone,
        "movers": mv, "takeaways": mover_takeaways(mv, peers),
        "watch": watch, "dynamic": dyn,
        "alerts_today": [k.split("|", 1)[1] for k in sorted(sent)],
        "alerts_sent": sorted(sent), "news_cache": cache,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(f"Wrote {OUT}: {len(quotes)} quotes · mood {mood.get('score')} ({mood.get('band')}) · "
          f"{len(new)} new alerts · sources {sources}")


if __name__ == "__main__":
    main()
