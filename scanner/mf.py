"""Mutual fund selection and long-term planning model.

  python scanner/mf.py                  # GitHub Actions: downloads NAV history, writes docs/data/mf.json
  python scanner/mf.py --db funds.db    # use a local copy of the NAV database
  python scanner/mf.py --no-telegram

Method (all from daily NAVs of Direct-Growth plans):
  • Index funds: among funds tracking the same index, the higher return over 3 years (then 5) wins –
    the index is identical, so the gap is the fund's costs and tracking error.
  • Active funds (5+ years old), all measured over the SAME recent window so no fund skips a crash:
    consistency of beating the category's median 3-year rolling return (30%), return per unit of
    volatility (30%), downside – worst 5-year drawdown and worst 3-year return (25%), 5-year CAGR (15%).
    Each active category is also checked against its index fund.
  • Model portfolios: index funds as the core, a small active slot, gold and debt for stability.
  • Outcome ranges: every historical start month for lump sums and SIPs into the blended portfolio.
Past returns do not guarantee future returns. This is a screen, not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mf_classify import BENCHMARK_INDEX, BUCKETS, bucket_for, category_family, parse_amfi_navall  # noqa: E402

OUT = ROOT / "docs" / "data" / "mf.json"
CACHE = ROOT / "data_cache" / "mf"
IST = ZoneInfo("Asia/Kolkata")
DB_URL = "https://github.com/captn3m0/historical-mf-data/releases/latest/download/funds.db.zst"
AMFI_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"

PROFILES = {
    "conservative": {
        "label": "Conservative", "horizon": "3–5 years", "equity_pct": 40,
        "about": "Capital protection first; equity for some growth, debt and gold to cushion falls.",
        "weights": {"idx_n50": 30, "flexi": 10, "corp_bond": 30, "arbitrage": 20, "gold": 10},
    },
    "balanced": {
        "label": "Balanced", "horizon": "5–7 years", "equity_pct": 65,
        "about": "Growth with a debt and gold cushion; mostly low-cost index funds.",
        "weights": {"idx_n50": 35, "idx_next50": 10, "idx_mid150": 10, "flexi": 10, "corp_bond": 25, "gold": 10},
    },
    "aggressive": {
        "label": "Aggressive", "horizon": "7+ years", "equity_pct": 85,
        "about": "Maximum long-term growth; expect deep temporary falls along the way.",
        "weights": {"idx_n50": 30, "idx_next50": 15, "idx_mid150": 15, "idx_mom30": 10, "flexi": 10,
                    "idx_small250": 5, "gold": 10, "liquid": 5},
    },
}
SIMPLE = {  # 3-fund versions for small SIPs
    "conservative": {"idx_n50": 40, "corp_bond": 45, "gold": 15},
    "balanced": {"idx_n50": 55, "idx_mid150": 15, "corp_bond": 30},
    "aggressive": {"idx_n50": 60, "idx_mid150": 25, "gold": 15},
}
# Young buckets borrow a longer-lived proxy for the projection maths only
PROXY = {"idx_mom30": "idx_mid150", "idx_n500": "flexi", "idx_small250": "small", "idx_mid150": "mid",
         "idx_next50": "large", "idx_n50": "large"}


# ------------------------------------------------------------------ data
def download_db(dest: Path) -> Path:
    import requests
    import zstandard
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading NAV history (~380 MB)…")
    with requests.get(DB_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        dctx = zstandard.ZstdDecompressor()
        with open(dest, "wb") as out:
            dctx.copy_stream(r.raw, out)
    return dest


def amfi_categories() -> dict[int, dict]:
    import requests
    try:
        r = requests.get(AMFI_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return parse_amfi_navall(r.text)
    except Exception as e:
        print("AMFI category file unavailable, using scheme names:", type(e).__name__)
        return {}


def load_universe_by_name(con, schemes, fresh):
    meta = {}
    for code, name in schemes.itertuples(index=False):
        if code in fresh:
            b = bucket_for(name.strip(), None)
            if b:
                meta[int(code)] = {"name": name.strip(), "bucket": b, "amc": None}
    codes = ",".join(str(c) for c in meta)
    nav = pd.read_sql(f"select scheme_code, date, nav from nav where scheme_code in ({codes})", con, parse_dates=["date"])
    con.close()
    return nav, meta, False


def load_universe(db: Path, amfi: dict) -> tuple[pd.DataFrame, dict, bool]:
    con = sqlite3.connect(db)
    latest = pd.read_sql("select scheme_code, max(date) d from nav group by scheme_code", con)
    newest = latest["d"].max()
    fresh = set(latest.loc[latest["d"] >= (pd.Timestamp(newest) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                           "scheme_code"])
    schemes = pd.read_sql("select scheme_code, scheme_name from schemes", con)
    if amfi:
        cats = pd.Series([v["category"] for v in amfi.values()]).value_counts()
        known = {c for c in cats.index if category_family(c)}
        print(f"AMFI list: {len(amfi)} schemes; {len(known)} categories map to ours: {sorted(known)}")
        print("  top categories:", dict(cats.head(12)))
        overlap = len(set(amfi) & set(int(c) for c in schemes["scheme_code"]))
        if overlap < 0.5 * len(amfi) or not known:
            print("AMFI list doesn't line up with the NAV database – classifying by scheme name instead")
            amfi = {}
    known = {v["category"] for v in amfi.values() if category_family(v["category"])} if amfi else set()
    meta = {}
    diag = {"fresh": 0, "in_amfi": 0, "our_category": 0, "direct_growth": 0}
    samples = []
    for code, name in schemes.itertuples(index=False):
        if code not in fresh:
            continue                                  # closed / merged schemes
        diag["fresh"] += 1
        a = amfi.get(int(code))
        if amfi and a is None:
            continue                                  # not in AMFI's live list
        diag["in_amfi"] += 1
        # AMFI's current file drops "Direct Plan - Growth" from names, so the plan/option test uses the
        # NAV database's full scheme name; AMFI supplies only the category.
        nm = name.strip()
        if a and a["category"] in known:
            diag["our_category"] += 1
            from mf_classify import is_direct_growth
            if is_direct_growth(nm):
                diag["direct_growth"] += 1
            elif len(samples) < 3:
                samples.append(nm)
        cat = a["category"] if a else None
        # AMFI category known to us → use it; another AMFI category (sectoral, overseas…) → skip; none → name
        b = bucket_for(nm, cat) if cat in known else (bucket_for(nm, None) if cat is None else None)
        if b:
            meta[int(code)] = {"name": nm, "bucket": b, "amc": a["amc"] if a else None}
    print("Universe funnel:", diag, "→ classified", len(meta))
    if samples:
        print("  e.g. not direct-growth:", samples)
    if not meta and amfi:
        print("Nothing classified via AMFI categories – retrying by scheme name")
        return load_universe_by_name(con, schemes, fresh)
    if not meta:
        raise SystemExit("No funds could be classified – check the AMFI file format and NAV database.")
    codes = ",".join(str(c) for c in meta)
    nav = pd.read_sql(f"select scheme_code, date, nav from nav where scheme_code in ({codes})", con,
                      parse_dates=["date"])
    con.close()
    return nav, meta, bool(amfi)


# ------------------------------------------------------------------ metrics
def cagr(s: pd.Series, years: float) -> float | None:
    if s.empty:
        return None
    end = s.index[-1]
    start = end - pd.DateOffset(days=round(365.25 * years))
    if s.index[0] > start + pd.Timedelta(days=7):
        return None
    v0 = s.asof(start)
    return float((s.iloc[-1] / v0) ** (1 / years) - 1) if v0 and v0 > 0 else None


def max_drawdown(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min()) if len(s) else 0.0


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]
    ts = np.array([(d - t0).days / 365.25 for d, _ in cashflows])
    cf = np.array([v for _, v in cashflows])
    f = lambda r: np.sum(cf / (1 + r) ** ts)
    lo, hi = -0.99, 10.0
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)


def sip_xirr(monthly: pd.Series, months: int) -> float | None:
    """XIRR of a ₹1 SIP on each month-end for the last `months` months, valued at the latest NAV."""
    m = monthly.dropna()
    if len(m) < months + 1:
        return None
    window = m.iloc[-(months + 1):-1]
    units = (1.0 / window).sum()
    flows = [(d, -1.0) for d in window.index] + [(m.index[-1], float(units * m.iloc[-1]))]
    return xirr(flows)


def fund_metrics(s: pd.Series) -> dict:
    s = s.dropna().sort_index()
    m = s.resample("ME").last().dropna()
    r3 = (m / m.shift(36)) ** (1 / 3) - 1
    r3 = r3.iloc[-24:].dropna()          # 3-year windows ending in the last 2 years: same span for every fund
    mret = m.pct_change().dropna().iloc[-60:]
    last5 = s[s.index >= s.index[-1] - pd.DateOffset(years=5)]   # common window, so no fund "skips" a crash
    return {
        "age_years": round((s.index[-1] - s.index[0]).days / 365.25, 1),
        "nav": round(float(s.iloc[-1]), 4), "nav_date": s.index[-1].date().isoformat(),
        "cagr_1y": cagr(s, 1), "cagr_3y": cagr(s, 3), "cagr_5y": cagr(s, 5), "cagr_10y": cagr(s, 10),
        "max_dd": max_drawdown(last5), "vol": float(mret.std() * math.sqrt(12)) if len(mret) > 12 else None,
        "worst_r3": float(r3.min()) if len(r3) else None, "median_r3": float(r3.median()) if len(r3) else None,
        "sip_xirr_5y": sip_xirr(m, 60), "r3_series": r3,
    }


def _pct_rank(vals: pd.Series, higher_better=True) -> pd.Series:
    return vals.rank(pct=True, ascending=higher_better)


def score_bucket(funds: dict[int, dict], kind: str) -> list[dict]:
    """funds: code → {name, metrics}. Returns ranked list of dicts (best first)."""
    if not funds:
        return []
    df = pd.DataFrame({c: {k: v for k, v in f["m"].items() if k != "r3_series"} for c, f in funds.items()}).T
    if kind == "index":
        ok = df[df["age_years"] >= 3]
        key = ["cagr_3y", "cagr_5y"]
        if ok.empty:
            ok, key = df[df["age_years"] >= 1], ["cagr_1y"]
        # Returns within 0.05%/yr are a tie (noise); then the longer track record wins.
        ok = ok.assign(_b=(ok[key[0]].astype(float) * 2000).apply(np.floor)).sort_values(
            ["_b", "age_years"], ascending=False, na_position="last").drop(columns="_b")
        ranked = list(ok.index) + [c for c in df.sort_values("age_years", ascending=False).index if c not in ok.index]
        df["score"] = np.nan
        for i, c in enumerate(ranked):
            df.loc[c, "score"] = round(100 * (1 - i / max(1, len(ranked))), 1)
        df["why"] = np.where(df.index.isin(ok.index),
                             f"Top {key[0].replace('cagr_', '')} return for this index (ties go to the longer record)",
                             "Too young to judge tracking yet")
    elif kind == "active":
        ok = df[df["age_years"] >= 5].copy()
        if ok.empty:
            ok = df[df["age_years"] >= 3].copy()
        # consistency: share of months the fund's 3y rolling return beat the category median
        r3 = pd.DataFrame({c: funds[c]["m"]["r3_series"] for c in ok.index})
        med = r3.median(axis=1)
        # Every 5-year-old fund has the same 24 rolling windows, so the raw share is comparable.
        cons = {}
        for c in ok.index:
            v = r3[c].notna()
            cons[c] = float((r3[c] >= med)[v].mean()) if v.any() else np.nan
        ok["consistency"] = pd.Series(cons)
        ok["risk_adj"] = ok["cagr_5y"].astype(float) / ok["vol"].astype(float).clip(lower=0.02)
        down = (_pct_rank(ok["max_dd"].astype(float)) + _pct_rank(ok["worst_r3"].astype(float))) / 2
        ok["score"] = (0.30 * _pct_rank(ok["consistency"].astype(float)) + 0.30 * _pct_rank(ok["risk_adj"])
                       + 0.25 * down + 0.15 * _pct_rank(ok["cagr_5y"].astype(float))) * 100
        df = df.join(ok[["consistency"]], how="left")
        df["score"] = ok["score"].round(1)
        df["why"] = np.where(df["score"].notna(), "Most consistent, best return per unit of risk, smaller falls",
                             "Under 5 years of history")
    else:  # debt / gold: steadiest return per unit of fall
        df["score"] = ((_pct_rank(df["cagr_3y"].astype(float)) * 0.6 + _pct_rank(df["max_dd"].astype(float)) * 0.4)
                       * 100).where(df["age_years"] >= 3).round(1)
        df["why"] = np.where(df["score"].notna(), "Best 3-year return with the smallest drawdown", "Under 3 years old")
    df = df.sort_values("score", ascending=False, na_position="last")
    out = []
    for code, r in df.iterrows():
        rec = {"code": int(code), "name": funds[code]["name"]}
        for k in ("age_years", "nav", "nav_date", "cagr_1y", "cagr_3y", "cagr_5y", "cagr_10y", "max_dd", "vol",
                  "worst_r3", "median_r3", "sip_xirr_5y", "score", "consistency", "why"):
            v = r.get(k)
            if isinstance(v, (float, np.floating)):
                v = None if np.isnan(v) else (round(float(v) * 100, 2) if k in (
                    "cagr_1y", "cagr_3y", "cagr_5y", "cagr_10y", "max_dd", "vol", "worst_r3", "median_r3",
                    "sip_xirr_5y", "consistency") else round(float(v), 2))
            rec[k] = v
        out.append(rec)
    return out


# ------------------------------------------------------------------ projections
def proxy_series(bucket: str, ranked: dict, navs: dict) -> tuple[pd.Series | None, str]:
    """Longest-lived fund of a bucket (or its proxy bucket) as a monthly return series."""
    chain, b = [bucket], bucket
    while b in PROXY and PROXY[b] not in chain:
        b = PROXY[b]
        chain.append(b)
    best, used = None, bucket
    for b in chain:
        for f in ranked.get(b, []):
            s = navs.get(f["code"])
            if s is not None and (best is None or s.index[0] < best.index[0]):
                best, used = s, b
        if best is not None and (best.index[-1] - best.index[0]).days > 365.25 * 8:
            break
    if best is None:
        return None, used
    return best.resample("ME").last().pct_change().dropna(), used


def projections(weights: dict, ranked: dict, navs: dict, horizons=(3, 5, 7, 10)) -> dict:
    series, notes = {}, []
    for b in weights:
        s, used = proxy_series(b, ranked, navs)
        if s is None:
            continue
        series[b] = s
        if used != b:
            notes.append(f"{BUCKETS[b][0]} uses {BUCKETS[used][0]} history (too new for its own)")
    if not series:
        return {}
    R = pd.DataFrame(series).dropna()
    w = pd.Series({b: weights[b] for b in R.columns}, dtype=float)
    port = (R * (w / w.sum())).sum(axis=1)               # monthly rebalanced blend
    out = {"start": R.index[0].date().isoformat(), "end": R.index[-1].date().isoformat(), "notes": notes,
           "lump": {}, "sip": {}, "max_drawdown_pct": round(max_drawdown((1 + port).cumprod()) * 100, 1)}
    g = (1 + port).values
    for h in horizons:
        n = 12 * h
        if len(g) < n + 12:
            continue
        lump, sip = [], []
        for i in range(len(g) - n + 1):
            seg = g[i:i + n]
            lump.append(float(np.prod(seg)))
            sip.append(float(np.sum(np.cumprod(seg[::-1])[::-1])))  # ₹1 at the start of each month
        q = lambda a, p: round(float(np.percentile(a, p)), 3)
        out["lump"][str(h)] = {"p10": q(lump, 10), "p50": q(lump, 50), "p90": q(lump, 90),
                               "worst": round(min(lump), 3), "windows": len(lump)}
        out["sip"][str(h)] = {"p10": q(sip, 10), "p50": q(sip, 50), "p90": q(sip, 90),
                              "worst": round(min(sip), 3), "windows": len(sip), "invested": n}
    return out


# ------------------------------------------------------------------ report
def build(db: Path, amfi: dict) -> dict:
    nav, meta, used_amfi = load_universe(db, amfi)
    navs = {int(c): g.set_index("date")["nav"].sort_index() for c, g in nav.groupby("scheme_code")}
    by_bucket: dict[str, dict] = {k: {} for k in BUCKETS}
    for code, info in meta.items():
        s = navs.get(code)
        if s is None or len(s) < 250:
            continue
        by_bucket[info["bucket"]][code] = {"name": info["name"], "m": fund_metrics(s)}
    ranked = {k: score_bucket(v, BUCKETS[k][1]) for k, v in by_bucket.items()}

    buckets = {}
    for k, lst in ranked.items():
        buckets[k] = {"label": BUCKETS[k][0], "kind": BUCKETS[k][1], "funds_considered": len(lst), "top": lst[:5]}
    # Active vs index evidence (5-year CAGR, after fees, Direct plans)
    evidence = []
    for act, idx in BENCHMARK_INDEX.items():
        for yrs in (5, 10):
            key = f"cagr_{yrs}y"
            best_idx = max((f for f in ranked.get(idx, []) if f.get(key) is not None),
                           key=lambda f: f[key], default=None)
            acts = [f for f in ranked.get(act, []) if f.get(key) is not None]
            if best_idx and acts:
                beat = sum(1 for f in acts if f[key] > best_idx[key])
                evidence.append({"category": BUCKETS[act][0], "years": yrs, "index_fund": best_idx["name"],
                                 "index_cagr": best_idx[key], "active_funds": len(acts), "beat_index": beat,
                                 "median_active_cagr": round(float(np.median([f[key] for f in acts])), 2)})
    profiles = {}
    for key, p in PROFILES.items():
        alloc = []
        for b, wgt in p["weights"].items():
            top = ranked.get(b, [])
            pick = next((f for f in top if f.get("score") is not None), top[0] if top else None)
            alt = next((f for f in top if f.get("score") is not None and pick and f["code"] != pick["code"]), None)
            alloc.append({"bucket": b, "label": BUCKETS[b][0], "weight": wgt,
                          "fund": pick and {k2: pick[k2] for k2 in ("code", "name", "cagr_3y", "cagr_5y", "max_dd")},
                          "alternative": alt and {"code": alt["code"], "name": alt["name"]}})
        simple = [{"bucket": b, "label": BUCKETS[b][0], "weight": wgt,
                   "fund": (lambda f: f and {"code": f["code"], "name": f["name"]})(
                       next((f for f in ranked.get(b, []) if f.get("score") is not None), None))}
                  for b, wgt in SIMPLE[key].items()]
        profiles[key] = {**{k2: p[k2] for k2 in ("label", "horizon", "equity_pct", "about")},
                         "allocation": alloc, "simple": simple,
                         "projections": projections(p["weights"], ranked, navs),
                         "simple_projections": projections(SIMPLE[key], ranked, navs)}
    nav_date = max(s.index[-1] for s in navs.values()).date().isoformat()
    return {
        "generated_at": dt.datetime.now(IST).isoformat(timespec="minutes"), "nav_date": nav_date,
        "universe": {"schemes": len(meta), "source": "AMFI NAV history (captn3m0/historical-mf-data)",
                     "categories": "AMFI scheme categories" if used_amfi else "scheme names (AMFI list unavailable)"},
        "buckets": buckets, "evidence": evidence, "profiles": profiles,
        "rules": [
            "Direct plan, Growth option only – no distributor commission, no IDCW payouts.",
            "Rebalance once a year, or when any part drifts more than 5 percentage points – redirect new SIPs first, sell only if needed.",
            "Review a fund only if it falls into the bottom half of its category in two reviews in a row, or its fund manager or mandate changes.",
            "Keep 3–6 months of expenses in a liquid fund before investing for the long term.",
            "Equity funds: gains after 12 months taxed at 12.5% above ₹1.25 lakh a year; within 12 months at 20%. Debt funds: taxed at your slab rate.",
            "ELSS saves tax only under the old tax regime (Section 80C) and locks each SIP for 3 years.",
            "Check exit loads before switching (many equity funds charge 1% if sold within 1 year).",
        ],
    }


def telegram_text(rep: dict) -> str:
    lines = [f"<b>🏦 Mutual fund review · NAV {rep['nav_date']}</b>"]
    for key, p in rep["profiles"].items():
        pr = p.get("projections", {}).get("lump", {}).get("5")
        lines.append(f"\n<b>{p['label']}</b> ({p['horizon']}, {p['equity_pct']}% equity)")
        for a in p["allocation"]:
            if a["fund"]:
                lines.append(f"  {a['weight']}% · {a['fund']['name']}")
        if pr:
            lines.append(f"  ₹50,000 over 5 yrs, past range: ₹{50000*pr['p10']:,.0f} – ₹{50000*pr['p90']:,.0f} "
                         f"(median ₹{50000*pr['p50']:,.0f})")
    for e in rep.get("evidence", [])[:2]:
        lines.append(f"\n{e['beat_index']} of {e['active_funds']} {e['category'].lower()} funds beat the best "
                     f"index fund over {e['years']} years (index {e['index_cagr']}%/yr).")
    lines.append("\nPast returns don't guarantee future returns. A screen, not advice.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()
    db = Path(args.db) if args.db else download_db(CACHE / "funds.db")
    rep = build(db, amfi_categories())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(f"Wrote {OUT}: {rep['universe']}")
    if not args.no_telegram:
        from notify import send_telegram
        send_telegram(telegram_text(rep))


if __name__ == "__main__":
    main()
