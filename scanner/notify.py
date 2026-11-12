"""Telegram alerts (Bot API). Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."""
from __future__ import annotations

import html
import os

import requests


def _inr(x: float, decimals: int = 2) -> str:
    return f"₹{x:,.{decimals}f}"


def build_message(result: dict, cfg: dict) -> str:
    mk = result.get("market", {})
    picks = result.get("picks", [])[: cfg.get("telegram_top", 10)]
    lines = [f"<b>📊 Stock scan · {result['as_of']}</b>"]
    if result.get("demo"):
        lines.append("<i>DEMO DATA – not real prices</i>")
    if mk.get("ok") is False:
        lines.append("🔴 <b>Market filter OFF</b> – Nifty below 50 DMA. No new trades; watch only.")
    elif mk.get("ok"):
        lines.append("🟢 Market filter ON – Nifty above 50 DMA.")
    else:
        lines.append("⚪ Market filter unknown – check Nifty manually.")
    rg = result.get("regime") or {}
    if rg.get("regime"):
        icon = {"green": "🟢", "amber": "🟡", "red": "🔴"}[rg["regime"]]
        lines.append(f"{icon} Regime {rg['regime'].upper()} – {rg['note']}")
    f = result.get("funnel", [])
    if f:
        lines.append("Funnel: " + " → ".join(str(x["count"]) for x in f))
    lines.append("")
    if not picks:
        lines.append("No stock passed every rule today. No trade is a valid outcome.")
    for i, p in enumerate(picks, 1):
        fund = {"pass": "✅ fundamentals OK", "unknown": "⚠️ check fundamentals"}.get(p.get("fund_status"), "")
        extra = " · ".join(x for x in [
            fund,
            f"ROE {p['roe_pct']}%" if p.get("roe_pct") is not None else "",
            f"results {p['next_results']}" if p.get("next_results") else "",
            "low-priced" if p.get("low_priced") else "",
        ] if x)
        lines.append(
            f"<b>{i}. {html.escape(p['symbol'])}</b> · {html.escape(p['setup'])}\n"
            + (f"   {html.escape(extra)}\n" if extra else "") +
            f"   Entry {_inr(p['entry'])} · Stop {_inr(p['stop'])} · Target {_inr(p['target'])}\n"
            f"   Qty {p['qty']} · Max loss {_inr(p['max_loss'], 0)} · Vol {p['vol_ratio']}×"
        )
    if result.get("circulars"):
        lines.append(f"\n📰 {len(result['circulars'])} new SEBI/NSE circulars – see app.")
    lines.append(
        "\nBefore buying: confirm ⚠️ picks on screener.in and run every pick through the Planner. "
        "Use a limit order; skip if it opens >1% above entry. Place the stop the same day."
    )
    if cfg.get("app_url"):
        lines.append(cfg["app_url"])
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram not configured; skipping alert.")
        return False
    ok = True
    # Telegram limit is 4096 chars per message; split on blank lines.
    chunks, cur = [], ""
    for block in text.split("\n"):
        if len(cur) + len(block) + 1 > 3800:
            chunks.append(cur)
            cur = ""
        cur += block + "\n"
    if cur.strip():
        chunks.append(cur)
    for c in chunks:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": c, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=30,
        )
        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text[:300])
            ok = False
    return ok
