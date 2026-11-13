"""Mutual fund universe and category buckets.

Only Direct plans with the Growth option are considered (no distributor commission, no IDCW payouts).
Categories come from AMFI's scheme headers when available; scheme names are the fallback.
"""
from __future__ import annotations

import re

BUCKETS = {
    # key: (label, kind, AMFI category it belongs to)
    "idx_n50":      ("Nifty 50 index fund", "index", "Other Scheme - Index Funds"),
    "idx_next50":   ("Nifty Next 50 index fund", "index", "Other Scheme - Index Funds"),
    "idx_n500":     ("Nifty 500 / total market index fund", "index", "Other Scheme - Index Funds"),
    "idx_mid150":   ("Nifty Midcap 150 index fund", "index", "Other Scheme - Index Funds"),
    "idx_small250": ("Nifty Smallcap 250 index fund", "index", "Other Scheme - Index Funds"),
    "idx_mom30":    ("Nifty200 Momentum 30 index fund", "index", "Other Scheme - Index Funds"),
    "large":        ("Large cap (active)", "active", "Equity Scheme - Large Cap Fund"),
    "flexi":        ("Flexi cap (active)", "active", "Equity Scheme - Flexi Cap Fund"),
    "mid":          ("Mid cap (active)", "active", "Equity Scheme - Mid Cap Fund"),
    "small":        ("Small cap (active)", "active", "Equity Scheme - Small Cap Fund"),
    "elss":         ("ELSS tax saver (active)", "active", "Equity Scheme - ELSS"),
    "liquid":       ("Liquid fund", "debt", "Debt Scheme - Liquid Fund"),
    "arbitrage":    ("Arbitrage fund", "debt", "Hybrid Scheme - Arbitrage Fund"),
    "corp_bond":    ("Corporate bond fund", "debt", "Debt Scheme - Corporate Bond Fund"),
    "gold":         ("Gold fund (FoF)", "gold", "Other Scheme - FoF Domestic"),
}

# Which index an active category is judged against (active-vs-index evidence)
BENCHMARK_INDEX = {"large": "idx_n50", "flexi": "idx_n500", "mid": "idx_mid150", "small": "idx_small250"}

_EXCLUDE = re.compile(r"\b(idcw|dividend|payout|reinvest|bonus|segregat|unclaimed|regular|retail|institutional)\b", re.I)
_SMART_BETA = r"(equal|value|quality|momentum|low\s*vol|alpha|shariah|dividend|esg|sensex|smart|enhanced|multicap|50:50|arbitrage)"


def is_direct_growth(name: str) -> bool:
    n = name.lower()
    return "direct" in n and "growth" in n and not _EXCLUDE.search(n)


def _has(pattern: str, n: str) -> bool:
    return re.search(pattern, n) is not None


def index_bucket(n: str) -> str | None:
    """Which index an index fund tracks, from its name (n is lower-case)."""
    if not _has(r"index|\betf fof|fund of fund", n) or _has(r"\belss\b|tax", n):
        return None
    if _has(r"nifty\s*200\s*momentum\s*30", n):
        return "idx_mom30"
    if _has(_SMART_BETA, n):
        return None
    if _has(r"next\s*50|junior", n):
        return "idx_next50"
    if _has(r"midcap\s*150", n):
        return "idx_mid150"
    if _has(r"small\s*cap\s*250|smallcap\s*250", n):
        return "idx_small250"
    if _has(r"nifty\s*500\b|total\s*market", n):
        return "idx_n500"
    if _has(r"nifty\s*50\b", n) and not _has(r"nifty\s*50\s*(value|equal)|bank|it\b|financial", n):
        return "idx_n50"
    return None


def bucket_from_name(name: str) -> str | None:
    n = name.lower()
    ib = index_bucket(n)
    if ib:
        return ib
    if _has(r"\bindex\b|\betf\b(?!.*fund of fund)|\bfof\b(?!.*gold)", n) and not _has(r"\bgold\b", n):
        return None
    if _has(r"\bgold\b", n) and not _has(r"silver|multi|asset|&|and\s+gold|mining|world|overseas|international", n):
        return "gold"
    if _has(r"passive|fund\s*of\s*funds?|\bfof\b|overseas|international|global|\bu\.?s\.?\b|world|emerging|"
            r"asia|china|japan|europe|nasdaq|s&p|hang\s*seng|greater\s*china", n):
        return None
    if _has(r"\belss\b|tax\s*saver|taxsaver|tax\s*relief|long\s*term\s*equity|tax\s*plan", n):
        return "elss"
    if _has(r"arbitrage", n):
        return "arbitrage"
    if _has(r"liquid", n) and not _has(r"etf|overnight", n):
        return "liquid"
    if _has(r"corporate\s*bond", n) and not _has(r"index|target|maturity|\d{4}", n):
        return "corp_bond"
    if _has(r"flexi\s*cap|flexicap", n):
        return "flexi"
    if _has(r"large\s*(&|and)\s*mid|large\s*&\s*midcap|large\s*and\s*midcap", n):
        return None
    if _has(r"\bsmall\s*cap|smallcap", n):
        return "small"
    if _has(r"\bmid\s*cap|midcap", n):
        return "mid"
    if _has(r"\blarge\s*cap|largecap|bluechip|top\s*100", n):
        return "large"
    return None


def category_family(cat: str | None) -> str | None:
    """Map an AMFI category – old style ('Equity Scheme - Mid Cap Fund') or new style
    ('Income/Debt Oriented Schemes - Liquid Fund', 'Index Funds - Equity Funds') – to our family."""
    if not cat:
        return None
    c = cat.lower()
    if "overseas" in c or "sectoral" in c or "thematic" in c:
        return None
    if "index fund" in c:
        return "index"
    if "fund of fund" in c or "fof" in c:
        return "fof"
    if "large & mid" in c or "large and mid" in c:
        return None
    for key, word in (("elss", "elss"), ("liquid", "liquid fund"), ("arbitrage", "arbitrage"),
                      ("corp_bond", "corporate bond"), ("flexi", "flexi cap"), ("large", "large cap"),
                      ("small", "small cap"), ("mid", "mid cap")):
        if word in c:
            return key
    return None


def bucket_for(name: str, amfi_category: str | None) -> str | None:
    """AMFI category decides the family; the name refines index funds and guards against mislabels."""
    if not is_direct_growth(name):
        return None
    by_name = bucket_from_name(name)
    if amfi_category is None:
        return by_name
    fam = category_family(amfi_category)
    if fam == "index":
        return index_bucket(name.lower())
    if fam == "fof":
        return "gold" if by_name == "gold" else None
    return fam


def parse_amfi_navall(text: str) -> dict[int, dict]:
    """AMFI NAVAll.txt → {scheme_code: {name, category, amc, nav, date}} for open-ended schemes."""
    out, category, amc = {}, None, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Scheme Code"):
            continue
        if ";" not in line:
            m = re.match(r"(Open|Close|Interval) Ended Schemes\s*\((.*)\)", line)
            if m:
                category = m.group(2).strip() if m.group(1) == "Open" else None
            else:
                amc = line
            continue
        parts = line.split(";")
        if len(parts) < 6 or category is None:
            continue
        try:
            code = int(parts[0])
        except ValueError:
            continue
        out[code] = {"name": parts[3].strip(), "category": category, "amc": amc,
                     "nav": parts[4].strip(), "date": parts[5].strip()}
    return out
