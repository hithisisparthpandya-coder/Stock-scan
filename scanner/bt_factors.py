"""Research: which ranking puts a day's picks among the TOP PERFORMERS of the following days?

For every trading day t, each ranking picks its top K stocks; each pick is "bought" at day t+1's open
and judged by its return to the close of t+1 (1 day), t+5 (1 week) and t+20 (1 month), compared with
every other liquid stock over the same window.

  hit rate  = share of picks that land in the top 10% of all liquid stocks (random picking ≈ 10%)
  bottom    = share that land in the bottom 10% (a ranking that only picks jumpy stocks raises both)
  excess    = average pick return minus the median stock's return over the same window
  beat      = share of picks that beat the median stock

Rankings are textbook factors chosen before looking at results (no tuning). Results are reported
separately for the design years and the unseen years; only a ranking that wins on BOTH is worth
proposing for the live scanner, and switching stays Paarth's decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = {"1d": 1, "5d": 5, "20d": 20}
K = 15


def _rank_pct(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return df.where(mask).rank(axis=1, pct=True)


def baseline_mask_and_score(P: dict, f: dict, cfg: dict):
    """Vectorised version of strategy.signals(..., 'baseline') for every day at once (same formulas;
    surveillance lists not applied, as in the backtest)."""
    C, V, L = P["C"], P["V"], P["L"]
    low = C < cfg["low_price_threshold"]
    need = pd.DataFrame(np.where(low, cfg["min_turnover_cr"] * cfg["low_price_turnover_mult"], cfg["min_turnover_cr"]),
                        index=C.index, columns=C.columns)
    liquid = (C >= cfg["min_price"]) & (f["avgturn20"] >= need) & (f["days"] >= 200)
    up = (C > f["sma50"]) & (f["sma50"] > f["sma200"]) & (f["sma200"] > f["sma200_prev"])
    vr = V / f["avgvol20"]
    breakout = (C > f["hh20"]) & (vr >= cfg["vol_mult"])
    bounce = (f["low3"] <= f["ema20"] * (1 + cfg["pullback_tol"])) & (C > f["ema20"]) & (C > C.shift(1))
    stop = pd.DataFrame(np.where(breakout, f["low10"], f["low3"]), index=C.index, columns=C.columns) * (1 - cfg["stop_buffer"])
    risk = (C - stop) / C
    m = liquid & up & (vr >= cfg["vol_mult"]) & (breakout | bounce) & \
        (risk >= cfg.get("min_stop_pct", 0)) & (risk <= cfg["max_stop_pct"])
    m = m & pd.DataFrame(np.repeat(f["market_ok"].values[:, None], C.shape[1], axis=1), index=C.index, columns=C.columns)
    score = 0.5 * _rank_pct(f["ret63"], m) + 0.3 * _rank_pct(vr, m) + 0.2 * (1 - _rank_pct(risk, m))
    return m.fillna(False), score, vr, liquid


def factor_study(P: dict, f: dict, cfg: dict, split, start_i: int) -> dict:
    C, O = P["C"], P["O"]
    m_base, s_base, vr, liquid = baseline_mask_and_score(P, f, cfg)
    universe = liquid & C.notna()
    hi252 = C.rolling(252, min_periods=200).max()
    near_hi = C / hi252
    mom = f["mom"]
    ret5 = C / C.shift(5) - 1
    comp = (_rank_pct(mom, universe) + _rank_pct(near_hi, universe)) / 2
    comp_in_funnel = (_rank_pct(mom, m_base) + _rank_pct(near_hi, m_base)) / 2

    rankings = [
        ("picks", "Current picks (funnel + current ranking)", s_base, m_base),
        ("picks_composite", "Funnel picks re-ranked by momentum + 52-week high", comp_in_funnel, m_base),
        ("funnel_all", "Every stock passing the funnel (no ranking)", m_base.astype(float), m_base),
        ("momentum", "Top volatility-adjusted 6+12-month momentum", mom, universe),
        ("near_high", "Closest to 52-week high", near_hi, universe),
        ("composite", "Momentum + 52-week high (composite)", comp, universe),
        ("volume", "Biggest volume surge today", vr, universe),
        ("reversal", "Biggest 1-week losers (reversal)", -ret5, universe),
    ]
    dates = C.index
    valid_rows = pd.Series(False, index=dates)
    valid_rows.iloc[start_i:len(dates) - 1] = True
    oos = pd.Series(dates >= pd.Timestamp(split), index=dates)

    out = {"k": K, "horizons": list(HORIZONS), "random_hit_pct": 10.0, "rankings": []}
    fwd, fpct = {}, {}
    entry = O.shift(-1)
    for hname, h in HORIZONS.items():
        r = C.shift(-h) / entry - 1
        r = r.where(universe & r.notna())
        fwd[hname] = r
        fpct[hname] = r.rank(axis=1, pct=True)
    for key, label, score, elig in rankings:
        sc = score.where(elig & universe)
        if key == "funnel_all":
            pick = elig & universe
        else:
            pick = sc.rank(axis=1, ascending=False, method="first") <= K
        row = {"key": key, "label": label}
        for part, rows in (("in_sample", valid_rows & ~oos), ("out_of_sample", valid_rows & oos)):
            res = {}
            for hname in HORIZONS:
                pk = pick & fwd[hname].notna()
                pk = pk[rows.values]
                n = int(pk.values.sum())
                if n < 30:
                    res[hname] = {"picks": n}
                    continue
                pp = fpct[hname][rows.values].where(pk)
                med = fwd[hname][rows.values].median(axis=1)
                ex = fwd[hname][rows.values].where(pk).sub(med, axis=0)
                res[hname] = {"picks": n, "days_with_picks": int(pk.any(axis=1).sum()),
                              "hit_pct": round(float((pp >= 0.9).sum().sum() / n * 100), 1),
                              "bottom_pct": round(float((pp <= 0.1).sum().sum() / n * 100), 1),
                              "beat_median_pct": round(float((pp > 0.5).sum().sum() / n * 100), 1),
                              "excess_pct": round(float(np.nanmean(ex.values)) * 100, 2)}
            row[part] = res
        out["rankings"].append(row)

    def wins(r, h="5d"):
        i, o = r["in_sample"].get(h, {}), r["out_of_sample"].get(h, {})
        return (i.get("excess_pct") or -9), (o.get("excess_pct") or -9)
    base = next(r for r in out["rankings"] if r["key"] == "picks")
    bi, bo = wins(base)
    better = [r for r in out["rankings"] if r["key"] not in ("picks", "funnel_all") and wins(r)[0] > bi and wins(r)[1] > bo]
    best = max(better, key=lambda r: wins(r)[1]) if better else None
    lines = [f"Current picks, 1 week later: {base['out_of_sample'].get('5d', {}).get('hit_pct', '–')}% land in the top 10% "
             f"(random ≈ 10%), {base['out_of_sample'].get('5d', {}).get('excess_pct', '–')}% vs the median stock (unseen years)."]
    if best:
        lines.append(f"'{best['label']}' beat the current ranking 1 week out on both design and unseen years "
                     f"({wins(best)[1]:+.2f}% vs {bo:+.2f}% excess, unseen). Candidate for a paper test – your decision.")
    else:
        lines.append("No alternative ranking beat the current one on both design and unseen years.")
    d1 = [r["out_of_sample"].get("1d", {}).get("hit_pct") for r in out["rankings"]]
    d1 = [x for x in d1 if x is not None]
    if d1:
        lines.append(f"Next day: the best ranking put {max(d1)}% of its picks in the top 10% (random ≈ 10%). "
                     "One-day moves are mostly news and noise; the 1-week and 1-month columns matter for swing trades.")
    out["best"] = best["key"] if best else None
    out["summary"] = lines
    return out
