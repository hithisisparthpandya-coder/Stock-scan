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
from dataclasses import asdict, dataclass, field

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
    entry_n: int
    entry_date: str
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
    sold_value: float = 0.0
    sold_qty: int = 0
    cost: float = 0.0


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    regimes: pd.Series
    exposure: pd.Series = field(default=None)


class PortfolioSim:
    """One portfolio, stepped one trading day at a time. The backtest and the live paper ledger both
    drive this class, so they follow exactly the same rules. State is JSON-serialisable."""

    def __init__(self, cfg: dict, strategy: str, state: dict | None = None):
        self.cfg, self.strategy = cfg, strategy
        st = state or {}
        self.cash = float(st.get("cash", cfg["bt_capital"]))
        self.peak = float(st.get("peak", self.cash))
        self.halt_until = int(st.get("halt_until", -1))
        self.n = int(st.get("n", -1))                     # trading-day counter
        self.pos = {k: Pos(**v) for k, v in st.get("positions", {}).items()}
        self.pending = list(st.get("pending", []))
        self.trades = list(st.get("trades", []))
        self.events: list[str] = []                       # human-readable fills/exits of the latest day
        self.cost_side = cfg["cost_per_side"] + cfg["slippage_per_side"]
        self.equity = float(st.get("equity", self.cash))

    def state(self) -> dict:
        return {"cash": round(self.cash, 2), "peak": round(self.peak, 2), "halt_until": self.halt_until,
                "n": self.n, "equity": round(self.equity, 2),
                "positions": {k: asdict(v) for k, v in self.pos.items()},
                "pending": self.pending, "trades": self.trades}

    # -- helpers
    def _close(self, p: Pos, px: float, date: str, reason: str):
        gross = p.qty * px
        c = gross * self.cost_side + self.cfg["dp_charge"]
        self.cash += gross - c
        p.cost += c
        total_qty = p.qty + p.sold_qty
        exit_value = p.sold_value + gross
        pnl = exit_value - total_qty * p.entry - p.cost
        r = round(pnl / p.risk_amt, 3) if p.risk_amt else 0.0
        self.trades.append({
            "symbol": p.sym, "entry_date": p.entry_date, "exit_date": date, "entry": round(p.entry, 2),
            "exit": round(exit_value / total_qty, 2), "qty": total_qty, "pnl": round(pnl, 2), "r": r,
            "reason": reason, "setup": p.setup, "regime": p.regime, "days": self.n - p.entry_n,
        })
        self.events.append(f"SOLD {p.sym} {total_qty} @ ₹{exit_value / total_qty:,.2f} ({reason}) → "
                           f"{'+' if pnl >= 0 else '−'}₹{abs(pnl):,.0f} ({r:+.2f}R)")
        del self.pos[p.sym]

    # -- one trading day
    def step(self, date: str, O, H, L, C, LL20, ATR, regime: str, factors=None):
        """Process one day's bars (pandas Series indexed by symbol). `factors` (optional) maps symbol →
        corporate-action factor for raw prices on an ex-date (split 1:2 → 0.5)."""
        self.n += 1
        self.events = []
        cfg = self.cfg
        if factors:
            for sym, f in factors.items():
                p = self.pos.get(sym)
                if p and f and abs(f - 1) > 0.02:
                    p.entry *= f; p.stop *= f; p.R *= f; p.last_close *= f
                    p.qty = max(1, round(p.qty / f)); p.sold_qty = round(p.sold_qty / f)
            for sig in self.pending:
                f = factors.get(sig["symbol"])
                if f and abs(f - 1) > 0.02:
                    sig["stop"] *= f; sig["limit"] *= f

        # 1) fills at the open
        equity_prev = self.equity
        open_risk = sum(max(0.0, p.entry - p.stop) * p.qty for p in self.pos.values())
        for sig in self.pending:
            if len(self.pos) >= cfg["max_positions"]:
                break
            sym = sig["symbol"]
            if sym in self.pos:
                continue
            o = O.get(sym, np.nan)
            if not (o > 0) or o > sig["limit"] or o <= sig["stop"]:
                continue
            R = o - sig["stop"]
            risk_amt = equity_prev * sig["risk"]
            if open_risk + risk_amt > cfg["heat_cap"] * equity_prev + 1e-9:
                continue
            qty = math.floor(min(risk_amt / R, cfg["max_position_pct"] * equity_prev / o,
                                 self.cash / (o * (1 + self.cost_side))))
            if qty < 1:
                continue
            c = qty * o * self.cost_side
            self.cash -= qty * o + c
            open_risk += qty * R
            self.pos[sym] = Pos(sym, qty, o, sig["stop"], R, qty * R, self.n, date, sig["setup"],
                                sig["regime"], sig["partial_R"], cost=c, last_close=o)
            self.events.append(f"BOUGHT {sym} {qty} @ ₹{o:,.2f}, stop ₹{sig['stop']:,.2f}")
        self.pending = []

        # 2) manage open positions with today's bar
        for sym, p in list(self.pos.items()):
            o, h, l, c = O.get(sym, np.nan), H.get(sym, np.nan), L.get(sym, np.nan), C.get(sym, np.nan)
            if not (c > 0):
                p.missing += 1
                if p.missing > 5:
                    self._close(p, p.last_close, date, "no data")
                continue
            p.missing = 0
            if not (o > 0):
                o = c
            if p.exit_next:
                self._close(p, o, date, p.exit_next); continue
            if p.be_next:
                p.stop = max(p.stop, p.entry); p.be_next = False
            if o <= p.stop:
                self._close(p, o, date, "stop (gap)"); continue
            if l <= p.stop:
                self._close(p, p.stop, date, "stop" if p.stop < p.entry - 1e-9 else "breakeven/trail stop"); continue
            level = p.entry + p.partial_R * p.R
            if not p.partial_done and p.qty >= 2 and h >= level:
                px = max(o, level)
                q = p.qty // 2
                gross = q * px
                cc = gross * self.cost_side + cfg["dp_charge"]
                self.cash += gross - cc
                p.cost += cc; p.sold_value += gross; p.sold_qty += q; p.qty -= q
                p.partial_done = True
                self.events.append(f"PARTIAL {sym}: sold {q} @ ₹{px:,.2f} (+{p.partial_R:g}R)")
            if h >= p.entry + p.R:
                p.be_next = True
                p.trail_on = True
            p.mfe = max(p.mfe, (h - p.entry) / p.R)
            p.last_close = c
            prior_low = LL20.get(sym, np.nan)
            if p.trail_on and c < prior_low:
                p.exit_next = "trail"
            elif self.n - p.entry_n >= cfg["time_stop_days"] and p.mfe < 0.5:
                p.exit_next = "time stop"
            if self.strategy == "adaptive" and regime == "red":
                a = ATR.get(sym, np.nan)
                if a > 0:
                    p.stop = max(p.stop, c - a)

        # 3) mark to market
        self.equity = self.cash + sum(p.qty * p.last_close for p in self.pos.values())
        self.peak = max(self.peak, self.equity)
        return self.equity

    def risk_multiplier(self) -> float:
        """Adaptive circuit breaker: a 10% fall from peak pauses new trades for dd_cooloff_days, then
        trading resumes with the peak reset (so the pause can never become permanent)."""
        if self.strategy != "adaptive":
            return 1.0
        dd = self.equity / self.peak - 1
        if self.n < self.halt_until:
            return 0.0
        if dd <= -self.cfg["dd_halt"]:
            self.halt_until = self.n + self.cfg["dd_cooloff_days"]
            self.peak = self.equity
            return 0.0
        return 0.5 if dd <= -self.cfg["dd_half"] else 1.0

    def queue(self, cands: list[dict], risk: float, regime: str):
        """Signals from today's close become limit orders for the next open."""
        if risk <= 0:
            return
        cfg = self.cfg
        partial_R = cfg["partial_R_amber"] if (self.strategy == "adaptive" and regime == "amber") else cfg["partial_R"]
        for c in cands[: cfg["max_positions"] * 3]:
            if c["symbol"] in self.pos:
                continue
            self.pending.append({"symbol": c["symbol"], "stop": c["stop"], "setup": c["setup"],
                                 "limit": c["close"] * (1 + cfg["gap_tolerance"]), "risk": risk,
                                 "regime": regime, "partial_R": partial_R})

    def close_all(self, date: str):
        for p in list(self.pos.values()):
            self._close(p, p.last_close, date, "end of test")
        self.equity = self.cash


def run_backtest(P: dict, f: dict, cfg: dict, strategy: str, start=None, end=None,
                 excluded_by_date=None) -> Result:
    C, O, H, L = P["C"], P["O"], P["H"], P["L"]
    dates = C.index
    i0 = max(int(np.searchsorted(dates, pd.Timestamp(start))) if start is not None else 0, cfg["bt_warmup"])
    i1 = int(np.searchsorted(dates, pd.Timestamp(end), side="right")) if end is not None else len(dates)
    sim = PortfolioSim(cfg, strategy)
    eq, expo, regs = [], [], []
    for i in range(i0, i1):
        d = dates[i].date().isoformat()
        reg = f["regime"].iloc[i]
        sim.step(d, O.iloc[i], H.iloc[i], L.iloc[i], C.iloc[i], f["ll20_prev"].iloc[i], f["atr14"].iloc[i], reg)
        eq.append(sim.equity); expo.append(len(sim.pos)); regs.append(reg)
        mult = sim.risk_multiplier()
        if i + 1 >= i1 or len(sim.pos) >= cfg["max_positions"] or mult == 0:
            continue
        excl = excluded_by_date(dates[i]) if excluded_by_date else set()
        cands, _, risk = signals(f, P, i, cfg, strategy, excl)
        sim.queue(cands, risk * mult, reg)
    if eq:
        sim.close_all(dates[i1 - 1].date().isoformat())
        eq[-1] = sim.equity
    idx = dates[i0:i1]
    return Result(pd.Series(eq, index=idx, dtype=float), pd.DataFrame(sim.trades),
                  pd.Series(regs, index=idx), pd.Series(expo, index=idx))


def run_momentum_portfolio(P: dict, f: dict, cfg: dict, start=None, end=None,
                           n_hold: int = 15, universe: int = 200) -> Result:
    """Research variant (not live): monthly-rebalanced, equal-weight portfolio of the top `n_hold`
    volatility-adjusted momentum stocks, always invested (no market filter, no stops).
      • Ranked on the last trading day of each month's close; trades at the next day's open.
      • Stocks that drop out of the top `n_hold` are sold; new entrants are bought with a slot of
        equity / n_hold each (whole shares; a stock whose one share costs more than a slot is skipped).
        Holdings that stay in the list are not resized (saves charges).
      • Same costs as the swing backtest. A stock with no price for 6 days exits at its last close."""
    from strategy import momentum_ranking
    C, O = P["C"], P["O"]
    dates = C.index
    i0 = max(int(np.searchsorted(dates, pd.Timestamp(start))) if start is not None else 0, cfg["bt_warmup"])
    i1 = int(np.searchsorted(dates, pd.Timestamp(end), side="right")) if end is not None else len(dates)
    cost_side = cfg["cost_per_side"] + cfg["slippage_per_side"]
    cash = float(cfg["bt_capital"])
    pos: dict[str, dict] = {}
    trades, eq, expo = [], [], []
    target = None

    def sell(sym, px, date, n, reason):
        nonlocal cash
        p = pos.pop(sym)
        gross = p["qty"] * px
        c = gross * cost_side + cfg["dp_charge"]
        cash += gross - c
        pnl = gross - p["qty"] * p["entry"] - p["cost"] - c
        trades.append({"symbol": sym, "entry_date": p["entry_date"], "exit_date": date,
                       "entry": round(p["entry"], 2), "exit": round(px, 2), "qty": p["qty"],
                       "pnl": round(pnl, 2), "ret_pct": round(pnl / (p["qty"] * p["entry"]) * 100, 2),
                       "reason": reason, "days": n - p["entry_n"]})

    for i in range(i0, i1):
        d = dates[i].date().isoformat()
        o_row, c_row = O.iloc[i], C.iloc[i]
        if target is not None:                                     # rebalance at today's open
            for sym in [s for s in pos if s not in target]:
                o = o_row.get(sym, np.nan)
                sell(sym, o if o > 0 else pos[sym]["last"], d, i, "rebalance")
            eq_now = cash + sum(p["qty"] * (o_row.get(s, np.nan) if o_row.get(s, np.nan) > 0 else p["last"])
                                for s, p in pos.items())
            slot = eq_now / n_hold
            for sym in target:
                if len(pos) >= n_hold:
                    break
                o = o_row.get(sym, np.nan)
                if sym in pos or not (o > 0):
                    continue
                qty = math.floor(min(slot, cash / (1 + cost_side)) / o)
                if qty < 1:
                    continue
                c = qty * o * cost_side
                cash -= qty * o + c
                pos[sym] = {"qty": qty, "entry": o, "entry_date": d, "entry_n": i, "cost": c, "last": o, "missing": 0}
            target = None
        for sym, p in list(pos.items()):
            c = c_row.get(sym, np.nan)
            if c > 0:
                p["last"], p["missing"] = c, 0
            else:
                p["missing"] += 1
                if p["missing"] > 5:
                    sell(sym, p["last"], d, i, "no data")
        equity = cash + sum(p["qty"] * p["last"] for p in pos.values())
        eq.append(equity); expo.append(len(pos))
        if i + 1 < i1 and dates[i + 1].month != dates[i].month:   # month end → pick next month's list
            slot = equity / n_hold
            r = momentum_ranking(f, P, i, cfg, universe)
            r = r[r["close"] <= slot * 0.95]
            target = list(r.index[:n_hold])
    if eq:
        last_d = dates[i1 - 1].date().isoformat()
        for sym in list(pos):
            sell(sym, pos[sym]["last"], last_d, i1 - 1, "end of test")
        eq[-1] = cash
    idx = dates[i0:i1]
    return Result(pd.Series(eq, index=idx, dtype=float), pd.DataFrame(trades),
                  pd.Series("n/a", index=idx), pd.Series(expo, index=idx))


def portfolio_metrics(equity: pd.Series, trades: pd.DataFrame, start=None, end=None) -> dict:
    """Metrics for the momentum portfolio. R-multiples don't apply (no stops), so trade stats are
    in % return per holding instead."""
    e = equity
    if start is not None:
        e = e[e.index >= pd.Timestamp(start)]
    if end is not None:
        e = e[e.index < pd.Timestamp(end)]
    out = metrics(e, None)
    out.pop("trades", None)
    t = trades
    if len(t):
        d = pd.to_datetime(t["exit_date"])
        m = pd.Series(True, index=t.index)
        if start is not None:
            m &= d >= pd.Timestamp(start)
        if end is not None:
            m &= d < pd.Timestamp(end)
        t = t[m]
    if len(t):
        gl = -t.loc[t["pnl"] <= 0, "pnl"].sum()
        years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
        out.update({"holdings_closed": int(len(t)), "closed_per_year": round(len(t) / years, 1),
                    "win_rate_pct": round(float((t["pnl"] > 0).mean() * 100), 1),
                    "avg_ret_pct": round(float(t["ret_pct"].mean()), 2),
                    "profit_factor": round(float(t.loc[t["pnl"] > 0, "pnl"].sum() / gl), 2) if gl > 0 else None,
                    "avg_days_held": round(float(t["days"].mean()), 1)})
    return out


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
