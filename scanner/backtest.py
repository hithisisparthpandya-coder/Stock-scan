"""Portfolio backtester for the baseline and adaptive strategies.

Rules simulated (same as the live rules):
  • Signal at day-t close → buy at day t+1 open, only if the open is ≤ close × (1 + gap_tolerance)
    and above the stop. Position size = risk budget / (entry − stop), capped at max_position_pct of
    equity and by available cash. Whole shares only.
  • Stop-loss: if the open gaps below the stop, exit at the open (a loss bigger than 1R, as in real life);
    otherwise exit at the stop when the day's low touches it. When a stop and a target are both touched
    on the same day, the stop is assumed to come first (conservative).
  • Once +1R is reached, the stop moves to the entry price from the next day.
  • Half the position is sold at +2R (green / baseline) or +1R (amber trades). The rest exits at the
    next open after a close below the prior 20-day low.
  • Time stop: exit at the next open after 15 trading days if the trade never reached +0.5R.
  • Adaptive only: in a RED regime, stops are tightened to (close − 1 ATR); after a 5% fall from the
    equity peak risk is halved; after a 10% fall new trades pause for dd_cooloff_days (20), then resume
    with the peak reset (so a further 5% fall halves risk again).
  • Costs: cost_per_side (STT, exchange, stamp, GST) + slippage per side + a flat DP charge per sale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy import signals


@dataclass
class Pos:
    sym: str
    qty: int
    entry: float
    stop: float
    R: float
    risk_amt: float
    entry_i: int
    setup: str
    regime: str
    partial_R: float
    partial_done: bool = False
    trail_on: bool = False
    be_next: bool = False
    exit_next: str | None = None
    mfe: float = 0.0
    last_close: float = 0.0
    missing: int = 0
    realized: float = 0.0
    sold_value: float = 0.0
    sold_qty: int = 0
    cost: float = 0.0


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    regimes: pd.Series
    exposure: pd.Series = field(default=None)


def run_backtest(P: dict, f: dict, cfg: dict, strategy: str, start=None, end=None,
                 excluded_by_date=None) -> Result:
    C, O, H, L = P["C"], P["O"], P["H"], P["L"]
    dates = C.index
    i0 = max(int(np.searchsorted(dates, pd.Timestamp(start))) if start is not None else 0, cfg["bt_warmup"])
    i1 = int(np.searchsorted(dates, pd.Timestamp(end), side="right")) if end is not None else len(dates)
    cost_side = cfg["cost_per_side"] + cfg["slippage_per_side"]
    cash = float(cfg["bt_capital"])
    peak = cash
    halt_until = -1
    pos: dict[str, Pos] = {}
    pending: list[dict] = []
    trades, eq, expo, regs = [], [], [], []
    ll20, atr = f["ll20_prev"], f["atr14"]

    def close_pos(p: Pos, px: float, i: int, reason: str):
        nonlocal cash
        gross = p.qty * px
        c = gross * cost_side + cfg["dp_charge"]
        cash += gross - c
        p.cost += c
        total_qty = p.qty + p.sold_qty
        exit_value = p.sold_value + gross
        pnl = exit_value - total_qty * p.entry - p.cost
        trades.append({
            "symbol": p.sym, "entry_date": dates[p.entry_i].date().isoformat(),
            "exit_date": dates[i].date().isoformat(), "entry": round(p.entry, 2),
            "exit": round(exit_value / total_qty, 2), "qty": total_qty, "pnl": round(pnl, 2),
            "r": round(pnl / p.risk_amt, 3) if p.risk_amt else 0.0, "reason": reason,
            "setup": p.setup, "regime": p.regime, "days": i - p.entry_i,
        })

    for i in range(i0, i1):
        o_row, h_row, l_row, c_row = O.iloc[i], H.iloc[i], L.iloc[i], C.iloc[i]

        # 1) fills at today's open for yesterday's signals
        equity_prev = eq[-1] if eq else cash
        open_risk = sum(max(0.0, (p.entry - p.stop)) * p.qty for p in pos.values())
        for sig in pending:
            if len(pos) >= cfg["max_positions"]:
                break
            sym = sig["symbol"]
            if sym in pos:
                continue
            o = o_row.get(sym, np.nan)
            if not (o > 0) or o > sig["limit"] or o <= sig["stop"]:
                continue
            R = o - sig["stop"]
            risk_amt = equity_prev * sig["risk"]
            if open_risk + risk_amt > cfg["heat_cap"] * equity_prev + 1e-9:
                continue
            qty = math.floor(min(risk_amt / R, cfg["max_position_pct"] * equity_prev / o,
                                 cash / (o * (1 + cost_side))))
            if qty < 1:
                continue
            c = qty * o * cost_side
            cash -= qty * o + c
            open_risk += qty * R
            pos[sym] = Pos(sym, qty, o, sig["stop"], R, qty * R, i, sig["setup"], sig["regime"],
                           sig["partial_R"], cost=c, last_close=o)
        pending = []

        # 2) manage open positions with today's bar
        reg_today = f["regime"].iloc[i]
        for sym, p in list(pos.items()):
            o, h, l, c = o_row.get(sym, np.nan), h_row.get(sym, np.nan), l_row.get(sym, np.nan), c_row.get(sym, np.nan)
            if not (c > 0):
                p.missing += 1
                if p.missing > 5:            # suspended / delisted: exit at last known price
                    close_pos(p, p.last_close, i, "no data")
                    del pos[sym]
                continue
            p.missing = 0
            if p.exit_next:
                close_pos(p, o, i, p.exit_next); del pos[sym]; continue
            if p.be_next:
                p.stop = max(p.stop, p.entry); p.be_next = False
            if o <= p.stop:
                close_pos(p, o, i, "stop (gap)"); del pos[sym]; continue
            if l <= p.stop:
                close_pos(p, p.stop, i, "stop" if p.stop < p.entry - 1e-9 else "breakeven/trail stop")
                del pos[sym]; continue
            level = p.entry + p.partial_R * p.R
            if not p.partial_done and p.qty >= 2 and h >= level:
                px = max(o, level)
                q = p.qty // 2
                gross = q * px
                cc = gross * cost_side + cfg["dp_charge"]
                cash += gross - cc
                p.cost += cc; p.sold_value += gross; p.sold_qty += q; p.qty -= q
                p.partial_done = True
            if h >= p.entry + p.R:
                p.be_next = True
                p.trail_on = True
            p.mfe = max(p.mfe, (h - p.entry) / p.R)
            p.last_close = c
            prior_low = ll20.iloc[i].get(sym, np.nan)
            if p.trail_on and c < prior_low:
                p.exit_next = "trail"
            elif i - p.entry_i >= cfg["time_stop_days"] and p.mfe < 0.5:
                p.exit_next = "time stop"
            if strategy == "adaptive" and reg_today == "red":
                a = atr.iloc[i].get(sym, np.nan)
                if a > 0:
                    p.stop = max(p.stop, c - a)

        # 3) mark to market
        equity = cash + sum(p.qty * p.last_close for p in pos.values())
        eq.append(equity)
        expo.append(len(pos))
        regs.append(reg_today)
        peak = max(peak, equity)
        dd = equity / peak - 1

        # 4) circuit breaker (adaptive): a 10% fall from peak pauses new trades for dd_cooloff_days;
        #    trading then resumes with the peak reset, so the pause can't become permanent.
        risk_mult = 1.0
        if strategy == "adaptive":
            if i < halt_until:
                risk_mult = 0.0
            elif dd <= -cfg["dd_halt"]:
                halt_until = i + cfg["dd_cooloff_days"]
                peak = equity
                risk_mult = 0.0
            elif dd <= -cfg["dd_half"]:
                risk_mult = 0.5

        # 5) signals at today's close for tomorrow's open
        if i + 1 >= i1 or len(pos) >= cfg["max_positions"] or risk_mult == 0:
            continue
        excl = excluded_by_date(dates[i]) if excluded_by_date else set()
        cands, _, risk = signals(f, P, i, cfg, strategy, excl)
        risk *= risk_mult
        if risk <= 0:
            continue
        partial_R = cfg["partial_R_amber"] if (strategy == "adaptive" and reg_today == "amber") else cfg["partial_R"]
        for cnd in cands[: cfg["max_positions"] * 3]:
            if cnd["symbol"] in pos:
                continue
            pending.append({"symbol": cnd["symbol"], "stop": cnd["stop"], "setup": cnd["setup"],
                            "limit": cnd["close"] * (1 + cfg["gap_tolerance"]), "risk": risk,
                            "regime": reg_today, "partial_R": partial_R})

    # close anything still open at the last close
    last = i1 - 1
    for sym, p in list(pos.items()):
        close_pos(p, p.last_close, last, "end of test")
        del pos[sym]
    if eq:
        eq[-1] = cash
    idx = dates[i0:i1]
    return Result(pd.Series(eq, index=idx, dtype=float), pd.DataFrame(trades),
                  pd.Series(regs, index=idx), pd.Series(expo, index=idx))


# ---------------------------------------------------------------- metrics
def metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    if len(equity) < 2:
        return {}
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1
    dd = equity / equity.cummax() - 1
    daily = equity.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    out = {"start": equity.index[0].date().isoformat(), "end": equity.index[-1].date().isoformat(),
           "total_return_pct": round(float(ret) * 100, 1), "cagr_pct": round(float(cagr) * 100, 1),
           "max_drawdown_pct": round(float(dd.min()) * 100, 1), "sharpe": round(sharpe, 2), "trades": 0}
    if trades is not None and len(trades):
        t = trades
        wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]
        gross_loss = -losses["pnl"].sum()
        streak = cur = 0
        for v in t["pnl"]:
            cur = cur + 1 if v <= 0 else 0
            streak = max(streak, cur)
        out.update({
            "trades": int(len(t)), "trades_per_year": round(len(t) / years, 1),
            "win_rate_pct": round(len(wins) / len(t) * 100, 1),
            "avg_r": round(float(t["r"].mean()), 3),
            "avg_win_r": round(float(wins["r"].mean()), 2) if len(wins) else 0.0,
            "avg_loss_r": round(float(losses["r"].mean()), 2) if len(losses) else 0.0,
            "profit_factor": round(float(wins["pnl"].sum() / gross_loss), 2) if gross_loss > 0 else None,
            "max_losing_streak": int(streak), "avg_days_held": round(float(t["days"].mean()), 1),
        })
    return out


def slice_metrics(res: Result, start=None, end=None) -> dict:
    e = res.equity
    if start is not None:
        e = e[e.index >= pd.Timestamp(start)]
    if end is not None:
        e = e[e.index < pd.Timestamp(end)]
    t = res.trades
    if len(t):
        d = pd.to_datetime(t["entry_date"])
        m = pd.Series(True, index=t.index)
        if start is not None:
            m &= d >= pd.Timestamp(start)
        if end is not None:
            m &= d < pd.Timestamp(end)
        t = t[m]
    return metrics(e, t)


def yearly_returns(equity: pd.Series) -> dict:
    y = equity.groupby(equity.index.year).last()
    first = equity.iloc[0]
    prev = pd.concat([pd.Series([first], index=[y.index[0] - 1]), y]).shift(1).iloc[1:]
    return {int(k): round(float(v / prev[k] - 1) * 100, 1) for k, v in y.items()}


def group_stats(trades: pd.DataFrame, col: str) -> dict:
    if not len(trades):
        return {}
    out = {}
    for k, g in trades.groupby(col):
        out[str(k)] = {"trades": int(len(g)), "win_rate_pct": round((g["pnl"] > 0).mean() * 100, 1),
                       "avg_r": round(float(g["r"].mean()), 3), "pnl": round(float(g["pnl"].sum()), 0)}
    return out
