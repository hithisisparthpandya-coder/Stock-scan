# ATM Stock Scan — daily NSE funnel + Telegram + iPhone & Windows apps

Branded for Absolute Tech Management LLP. Internal tool for personal use — a screen, not investment advice.

Every weekday at 7:30 pm IST, GitHub runs the scanner in the cloud:
NSE official bhavcopy → drops ASM/GSM surveillance stocks → liquidity → uptrend →
volume surge → breakout / pullback setup → stop fits your 1% risk budget → top 15 ranked.
Picks go to Telegram and to the web app on your iPhone. Free to run.

## One-time setup (about 15 minutes)

1. **Telegram bot**
   - In Telegram, open **@BotFather** → send `/newbot` → pick a name → copy the **token**.
   - Open your new bot and send it "hi".
   - In a browser open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy the number after `"chat":{"id":` — that is your **chat ID**.
2. **GitHub repo** — create a free account, then a new **public** repository (e.g. `stock-scan`) and upload everything in this folder (keep the folder structure, including `.github/`).
   *Free GitHub Pages needs a public repo. The repo shows only the daily picks — no personal data.*
3. **Secrets** — repo → Settings → Secrets and variables → Actions → New repository secret:
   - `TELEGRAM_BOT_TOKEN` = the token
   - `TELEGRAM_CHAT_ID` = the chat ID
4. **App hosting** — repo → Settings → Pages → Source: *Deploy from a branch* → `main` / `/docs` → Save.
5. **First run** — repo → Actions → *Daily stock scan* → Run workflow → tick **force** → Run.
   The first run downloads ~14 months of NSE history (≈10 min). After that each run takes ~1–2 min.
6. **iPhone / iPad** — open `https://<your-github-username>.github.io/stock-scan/` in **Safari** → Share → **Add to Home Screen**.

## Windows app
1. Run `ATM-Stock-Scan-Setup-1.2.0.exe`. It installs for your user only (no admin rights) with Start-menu and desktop shortcuts.
   The installer is not code-signed, so Windows SmartScreen may say "Windows protected your PC": click **More info → Run anyway**.
2. On first launch, paste your GitHub Pages address (step 6 above) and tick **Start with Windows** if you want it in the tray.
3. The app checks for a new scan every 30 minutes while it runs and shows a Windows notification when one lands.
   Closing the window keeps it in the system tray; right-click the tray icon to quit.
- Rebuild after changes: Actions → *Build Windows app* → Run workflow (downloads the .exe as an artifact),
  or locally on Linux with `cd desktop && npm ci && npm run dist` (needs `makensis`).
- `docs/index.html` is the single UI for both the iPhone and Windows apps; `desktop/sync-ui.js` copies it in.

## Strategy lab (backtest)
1. Actions → **Strategy backtest** → Run workflow. The first run downloads NSE bhavcopies from 2015 (1.5–3 h);
   it then refreshes itself on the first Sunday of each month in a few minutes.
2. Open the app → **Strategy lab**. It compares the current funnel with regime-adaptive momentum on
   design years (2015–2020) and unseen years (2021→), shows yearly returns vs Nifty, a robustness check,
   and a verdict: use adaptive / keep baseline / no proven edge.
3. To switch the live scanner, set `"live_strategy": "adaptive"` in `scanner/config.json`
   (only if the verdict recommends it). The app shows today's market regime either way.

### Research queue (test-only — never switched live without your approval)
One idea a week is added to the backtest report under `research` in `docs/data/backtest.json`.
- [x] (a) Monthly-rebalanced top-15 volatility-adjusted momentum portfolio (Nifty200 Momentum 30 style) vs Nifty buy-and-hold — added 2026-11-12
- [ ] (b) Breakouts only, no pullbacks
- [ ] (c) Wider trailing exit (close below the 50-day average) to let winners run

## Your settings
Edit `scanner/config.json`: `capital` (update as your account grows), `risk_pct` (0.01 = 1%),
`max_position_pct`, `min_turnover_cr`, `vol_mult`, `max_stop_pct`, `top_n`.

## Files
- `scanner/nse.py` — official NSE/SEBI data (bhavcopy, Nifty close, ASM/GSM lists, circular RSS)
- `scanner/funnel.py` — the live baseline funnel, results-date and fundamentals stages
- `scanner/strategy.py` — features, market regime, baseline + adaptive signals (shared by live and backtest)
- `scanner/backtest.py`, `bt_report.py`, `bt_data.py` — portfolio backtester, report, 10-year data cache
- `scanner/notify.py` — Telegram message
- `scanner/run.py` — daily entry point (`--demo` runs on synthetic data with no network)
- `scanner/test_funnel.py` — tests; the workflow runs them before every scan
- `docs/` — the iPhone web app (reads `docs/data/picks.json`), ATM logo and fonts
- `desktop/` — the Windows app (Electron): tray, notifications, installer script

## Limits (read these)
- End-of-day only, by design. Entry = that day's close; buy next morning with a limit order.
- Fundamentals (ROE > 15%, D/E < 1 except banks/NBFCs, profitable, market cap > ₹500 cr) come from Yahoo Finance
  via yfinance, only for stocks that pass the price/volume funnel, cached 7 days. Missing data is flagged ⚠, never passed.
- Results dates come from NSE's event calendar (Yahoo as fallback); stocks with results in the next 7 days are dropped.
- Low-priced stocks (₹10–50) are allowed if they have 2× the normal liquidity, aren't in a 2%/5% circuit band,
  and pass fundamentals (market cap > ₹500 cr). Below ₹10 is excluded.
- If NSE's surveillance lists can't be fetched, the app shows a warning; check picks on nseindia.com.
- This is a screen, not investment advice. The decision and the risk are yours.
