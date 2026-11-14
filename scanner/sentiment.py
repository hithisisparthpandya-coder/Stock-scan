"""Sentiment engine: headline scoring, per-stock news sentiment, Market Mood index and the
sentiment-adjusted ("dynamic") scan.

Everything here is a fixed, written-down formula – no weights were fitted to past prices, because
there is no free history of old headlines to fit them on. The sentiment list is therefore tracked in
its own paper ledger (docs/data/ledger_sentiment.json) to learn whether it helps before anyone relies
on it.

THE EQUATIONS
-------------
1. Headline score  h ∈ [−1, 1]
     h = tanh( Σ lexicon weights of the phrases found / 2 )
   Finance-specific phrases ("bags order" +2, "profit jumps" +2, "downgrade" −2, "SEBI order" −2 …).
   A negation ("not", "no", "fails to") just before a phrase flips it and halves it; "denies"
   keeps the sign at 30% strength (a denial of fraud is still not good news).

2. Stock news sentiment  N ∈ [−1, 1]  (headlines that name the stock, last 72 hours)
     N = tanh( Σ h_i · 0.5^(age_i / 24h) / 1.5 )          (a day-old headline counts half)

3. Market Mood  M ∈ [0, 100]  (weighted average of whatever components are available today)
     breadth   0.25 × share of liquid stocks above their 50-day average
     advance   0.20 × advances / (advances + declines) today
     calm      0.20 × (1 − India VIX percentile over the past year)   [or (25 − VIX)/15 if no history]
     trend     0.15 × clip(0.5 + (Nifty / 50-DMA − 1) / 0.06, 0, 1)
     news      0.20 × (average market-headline score + 1) / 2
     flows     0.10 × (0.5 + tanh(FII+DII net ₹ cr / 3000) / 2)   [only when NSE publishes it]
   Bands: < 35 Fear → no new suggestions · 35–50 Cautious → half risk, max 2 · ≥ 50 → normal.
   Mood can only LOWER risk, never raise it.

4. Sentiment-adjusted score for a technical pick (T = its 0–100 rank score)
     A = T + 15 · N + 5 · clip(sector move today / 1%, −1, 1)
   Vetoed when N ≤ −0.4 or a red-flag headline appears (fraud, raid, default, SEBI order …).
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

# ---------------------------------------------------------------- 1. lexicon
# (regex, weight). Earlier (more specific) phrases are matched first and consume their words.
LEXICON: list[tuple[str, float]] = [
    # results & guidance
    (r"profit (jumps?|surges?|soars?|rises?|climbs?|up|doubles?|grows?)", 2.0),
    (r"(revenue|sales|income) (jumps?|surges?|rises?|grows?|climbs?)", 1.5),
    (r"(beats?|tops?|exceeds?) (estimates|expectations|street)", 2.0),
    (r"(misses?|below) (estimates|expectations)", -2.0),
    (r"profit (falls?|drops?|declines?|slumps?|plunges?|dips?|down|halves)", -2.0),
    (r"(revenue|sales) (falls?|drops?|declines?|slumps?)", -1.5),
    (r"loss (widens?|deepens?)", -2.0), (r"loss narrows?", 1.0), (r"net loss", -1.2),
    (r"swings? to (profit|black)", 2.0), (r"swings? to (loss|red)", -2.0),
    (r"margins? (expand|expansion|improves?)", 1.5), (r"margins? (contract|pressure|squeeze|shrink)", -1.5),
    (r"guidance (raised|upgrade)|raises (guidance|outlook)", 2.0),
    (r"guidance (cut|lowered)|cuts (guidance|outlook)", -2.0),
    (r"record (high|profit|revenue|sales)", 1.5), (r"all[- ]time high", 1.5),
    (r"52[- ]week high", 1.2), (r"52[- ]week low", -1.2),
    (r"upper circuit", 1.5), (r"lower circuit", -2.0),
    # brokers
    (r"(rating )?upgrades?d?|upgraded to buy", 2.0), (r"(rating )?downgrades?d?|downgraded to sell", -2.0),
    (r"target (price )?(raised|hiked|increased)|(raises?|lifts?|lifted|hikes?) [^.;]{0,60}?(price )?target", 1.5),
    (r"target (price )?(cut|lowered|reduced)|cuts? target", -1.5),
    (r"\bbuy (call|rating)\b|\boutperform\b|\boverweight\b", 1.2),
    (r"\bsell (call|rating)\b|\bunderperform\b|\bunderweight\b", -1.2),
    # business events
    (r"(bags?|wins?|secures?|bagged|won|receives?|gets?) (an? )?(order|orders|contract|contracts|deal)", 2.0),
    (r"order (book|inflow) (rises|grows|jumps)", 1.5),
    (r"(usfda|fda|dcgi) approval|approval from (usfda|fda)", 2.0), (r"\bapproval\b", 0.8),
    (r"\bbuyback\b", 1.5), (r"\bbonus (issue|shares)\b", 1.0), (r"\bdividend\b", 0.5),
    (r"(raises|hikes|increases) stake|stake (hike|increase)", 1.2),
    (r"promoters? (sells?|offloads?|cuts?) stake|stake sale|offloads? stake", -1.2),
    (r"\bpledge (invoked|invocation)\b", -2.5), (r"\bpledged?\b", -0.8),
    (r"\bexpansion\b|\bcapacity addition\b|new plant", 0.8),
    (r"\bpartnership\b|\btie[- ]up\b|\bjoint venture\b", 0.8),
    (r"\bacquires?\b|\bacquisition\b", 0.5),
    (r"\bdebt[- ]free\b|\bdeleverag", 1.2), (r"\bturnaround\b", 1.2),
    (r"\bqip\b|\bfund ?rais", -0.3), (r"\bdilution\b", -1.0),
    (r"\bwarning letter\b|\bimport alert\b|\bform 483\b|\bobservations\b", -2.0),
    (r"\brecall\b", -1.2), (r"\blayoffs?\b|\bjob cuts\b", -1.0), (r"\bstrike\b", -1.0),
    (r"\bfire\b|\bblast\b|\bexplosion\b", -1.5), (r"\bshutdown\b|\bhalts? (production|operations)\b", -1.5),
    (r"\blawsuit\b|\bsued\b|\blitigation\b", -1.2),
    (r"(ceo|cfo|md|auditor|director)s? (resigns?|quits?|steps? down)", -1.8), (r"\bresigns?\b", -0.8),
    # regulators & red flags
    (r"\bfraud\b", -3.0), (r"\braids?\b|\bsearch(es)? (operation|by)\b", -2.0),
    (r"\b(enforcement directorate|money laundering)\b", -2.5), (r"\bforensic audit\b", -2.0),
    (r"\bdefaults?\b|\binsolvency\b|\bbankruptcy\b|\bnclt\b", -2.5),
    (r"\bsebi (order|ban|bars|penalty|probe|notice|show[- ]cause)", -2.0),
    (r"\bprobe\b|\binvestigation\b", -1.5), (r"\bpenalty\b|\bfined\b", -1.2), (r"\bban(s|ned)?\b", -1.2),
    # price action / market tone
    (r"\bsurg(e|es|ed|ing)\b|\bsoar(s|ed|ing)?\b|\bzoom(s|ed)?\b|\bskyrocket", 1.5),
    (r"\bjump(s|ed)?\b|\brall(y|ies|ied)\b|\bspurt(s|ed)?\b", 1.2),
    (r"\b(gain|gains|gained|rise|rises|rose|climbs?|climbed|advances?|higher)\b", 0.7),
    (r"\bplung(e|es|ed|ing)\b|\bcrash(es|ed)?\b|\btank(s|ed)?\b|\bnosedive", -1.8),
    (r"\bslump(s|ed)?\b|\btumbl(e|es|ed)\b|\bsink(s)?\b|\bsank\b|\bselloff\b|\bsell-off\b", -1.5),
    (r"\b(fall|falls|fell|drop|drops|dropped|decline|declines|declined|slip|slips|slipped|lower|slide|slides)\b", -0.7),
    (r"\bprofit[- ]booking\b", -0.5),
    (r"\bbullish\b|\boptimism\b|\brecovery\b|\brebound(s|ed)?\b", 1.0),
    (r"\bbearish\b|\bpessimism\b|\bfears?\b|\bjitters\b|\bpanic\b", -1.0),
    (r"\b(fii|fpi|foreign) (inflows?|buying)\b", 1.0), (r"\b(fii|fpi|foreign) (outflows?|selling)\b", -1.0),
    (r"\brate cut\b", 1.0), (r"\brate hike\b", -1.0), (r"\brecession\b", -1.5),
    (r"\btariffs?\b|\bsanctions?\b", -0.6), (r"\bwar\b|\bconflict\b", -1.0),
]
_COMPILED = [(re.compile(p, re.I), w) for p, w in LEXICON]        # list order = priority (specific first)
_NEG = re.compile(r"\b(not|no|never|without|fails? to|failed to|denies|denied|despite)\b\W+(\w+\W+){0,2}$", re.I)

RED_FLAGS = re.compile(
    r"\bfraud\b|\braids?\b|\bdefaults?\b|\binsolvency\b|\bbankruptcy\b|\bforensic audit\b|"
    r"\benforcement directorate\b|\bmoney laundering\b|\bsebi (order|ban|bars)\b|\bpledge invo|"
    r"\bwarning letter\b|\bimport alert\b|auditors? (resigns?|quits?)", re.I)


def score_headline(text: str) -> float:
    """Headline → score in [−1, 1] (see module docstring, equation 1)."""
    t = " " + re.sub(r"\s+", " ", text.replace("’", "'")) + " "
    taken = [False] * len(t)
    total = 0.0
    for rx, w in _COMPILED:
        for m in rx.finditer(t):
            a, b = m.span()
            if any(taken[a:b]):
                continue
            for k in range(a, b):
                taken[k] = True
            neg = _NEG.search(t[max(0, a - 40):a])
            if neg:     # "not see margin pressure" flips; "denies fraud" stays mildly negative
                w = 0.3 * w if neg.group(1).lower().startswith("den") else -0.5 * w
            total += w
    return round(math.tanh(total / 2), 3)


def red_flag(text: str) -> bool:
    return bool(RED_FLAGS.search(text)) and not re.search(r"\b(denies|denied|no|clean chit|acquitted)\b", text, re.I)


# ---------------------------------------------------------------- 2. per-stock sentiment
STOP = {"limited", "ltd", "ltd.", "the", "company", "co", "corporation", "corp", "india", "(india)", "of", "and", "&"}


def name_keys(symbol: str, name: str) -> list[str]:
    """Phrases that identify a company in a headline: its symbol (if distinctive) and the first one
    or two significant words of its name."""
    words = [w for w in re.split(r"[\s,.]+", (name or "").lower()) if w and w not in STOP]
    keys = []
    if len(words) >= 2:
        keys.append(" ".join(words[:2]))
    if words and (len(words[0]) >= 6 or len(words) == 1) and len(words[0]) >= 4:
        keys.append(words[0])
    if words and words[0] == symbol.lower() and len(symbol) >= 3:
        keys.append(words[0])
    if len(symbol) >= 4 and symbol.isalpha():
        keys.append(symbol.lower())
    return list(dict.fromkeys(keys))


def mentions(headline: str, keys: list[str]) -> bool:
    h = headline.lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", h) for k in keys)


def _age_hours(item: dict, now: datetime) -> float:
    p = item.get("published")
    if not p:
        return 12.0
    if isinstance(p, str):
        try:
            p = datetime.fromisoformat(p)
        except ValueError:
            return 12.0
    if p.tzinfo is None:
        p = p.replace(tzinfo=timezone.utc)
    return max(0.0, (now - p).total_seconds() / 3600)


def stock_sentiment(headlines: list[dict], keys: list[str], now: datetime | None = None,
                    already_matched: bool = False, max_age_h: float = 72) -> dict:
    """Equation 2. headlines: [{title, link, published, source}]. Returns N, the headlines used and
    whether a red flag was seen."""
    now = now or datetime.now(timezone.utc)
    used, total, flag = [], 0.0, False
    seen = set()
    for it in headlines:
        title = it.get("title", "")
        k = re.sub(r"\W+", " ", title.lower())[:80]
        if not title or k in seen or (not already_matched and not mentions(title, keys)):
            continue
        age = _age_hours(it, now)
        if age > max_age_h:
            continue
        seen.add(k)
        h = score_headline(title)
        total += h * 0.5 ** (age / 24)
        rf = red_flag(title)
        flag |= rf
        used.append({"title": title, "link": it.get("link", ""), "source": it.get("source", ""),
                     "age_h": round(age, 1), "score": h, "red_flag": rf})
    used.sort(key=lambda x: (x["age_h"], -abs(x["score"])))
    return {"N": round(math.tanh(total / 1.5), 3), "headlines": used[:6], "red_flag": flag, "count": len(used)}


# ---------------------------------------------------------------- 3. market mood
MOOD_WEIGHTS = {"breadth": 0.25, "advance": 0.20, "calm": 0.20, "trend": 0.15, "news": 0.20, "flows": 0.10}
MOOD_LABELS = {"breadth": "Breadth (stocks above 50-DMA)", "advance": "Advancers today",
               "calm": "Calm volatility (India VIX)", "trend": "Nifty trend vs 50-DMA",
               "news": "News tone", "flows": "FII + DII flows"}


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def market_mood(breadth=None, advances=None, declines=None, vix=None, vix_pct=None, nifty=None,
                nifty_sma50=None, news_avg=None, flows_cr=None) -> dict:
    """Equation 3. Any argument can be None; the index uses whatever is available."""
    c = {}
    if breadth is not None:
        c["breadth"] = clip01(breadth)
    if advances is not None and declines is not None and advances + declines > 0:
        c["advance"] = clip01(advances / (advances + declines))
    if vix_pct is not None:
        c["calm"] = clip01(1 - vix_pct)
    elif vix is not None:
        c["calm"] = clip01((25 - vix) / 15)
    if nifty and nifty_sma50:
        c["trend"] = clip01(0.5 + (nifty / nifty_sma50 - 1) / 0.06)
    if news_avg is not None:
        c["news"] = clip01((news_avg + 1) / 2)
    if flows_cr is not None:
        c["flows"] = clip01(0.5 + math.tanh(flows_cr / 3000) / 2)
    if not c:
        return {"score": None, "band": "unknown", "risk_mult": 1.0, "max_new": 3, "components": []}
    wsum = sum(MOOD_WEIGHTS[k] for k in c)
    score = 100 * sum(MOOD_WEIGHTS[k] * v for k, v in c.items()) / wsum
    band, mult, max_new, note = mood_band(score)
    return {"score": round(score, 1), "band": band, "risk_mult": mult, "max_new": max_new, "note": note,
            "components": [{"key": k, "label": MOOD_LABELS[k], "value": round(v * 100), "weight": MOOD_WEIGHTS[k]}
                           for k, v in c.items()]}


def mood_band(score: float):
    if score < 35:
        return "fear", 0.0, 0, "Fear – no new suggestions today; protect open positions."
    if score < 50:
        return "cautious", 0.5, 2, "Cautious – at most 2 new ideas, at half the usual risk."
    if score < 65:
        return "neutral", 1.0, 3, "Neutral – normal rules."
    return "greed", 1.0, 3, "Greed – normal rules, but don't chase gap-ups; risk is never raised."


# ---------------------------------------------------------------- 4. sentiment-adjusted scan
def adjusted_list(picks: list[dict], news_by_sym: dict, sector_move: dict, mood: dict) -> list[dict]:
    """Equation 4. picks: technical picks (with 'score' 0–100 and optional 'industry').
    Returns the re-ranked list with reasons, vetoes marked, limited by the mood band."""
    out = []
    for p in picks:
        ns = news_by_sym.get(p["symbol"]) or {"N": 0.0, "headlines": [], "red_flag": False, "count": 0}
        sec = sector_move.get(p.get("industry") or "", None)
        sec_term = 5 * max(-1.0, min(1.0, (sec or 0) / 1.0))
        a = float(p.get("score") or 0) + 15 * ns["N"] + sec_term
        reasons = []
        if ns["count"]:
            reasons.append(f"news {'positive' if ns['N'] > 0.1 else 'negative' if ns['N'] < -0.1 else 'neutral'} "
                           f"({ns['N']:+.2f}, {ns['count']} headline{'s' * (ns['count'] > 1)})")
        else:
            reasons.append("no recent news (neutral)")
        if sec is not None:
            reasons.append(f"sector {sec:+.1f}% today")
        veto = None
        if ns["red_flag"]:
            veto = "red-flag headline"
        elif ns["N"] <= -0.4:
            veto = "strongly negative news"
        out.append({**p, "adj_score": round(a, 1), "news_N": ns["N"], "sector_chg": sec,
                    "reasons": reasons, "veto": veto, "headlines": ns["headlines"][:3]})
    out.sort(key=lambda x: (x["veto"] is not None, -x["adj_score"]))
    kept = 0
    for x in out:
        if x["veto"]:
            x["status"] = "vetoed"
        elif kept < mood.get("max_new", 3):
            x["status"] = "suggested"
            kept += 1
        else:
            x["status"] = "waitlist" if mood.get("max_new", 3) else "blocked by mood"
    return out
