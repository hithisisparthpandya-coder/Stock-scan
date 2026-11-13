// Copies the shared UI from ../docs into ./ui so the iPhone app and the Windows app stay identical.
const fs = require("fs"), path = require("path");
const src = path.join(__dirname, "..", "docs"), dst = path.join(__dirname, "ui");
fs.rmSync(dst, { recursive: true, force: true });
fs.mkdirSync(path.join(dst, "data"), { recursive: true });
for (const f of ["index.html", "atm-mark.svg", "fonts"]) fs.cpSync(path.join(src, f), path.join(dst, f), { recursive: true });
// Bundled demo data (used only when "Try with demo data" is chosen)
fs.copyFileSync(path.join(__dirname, "demo-picks.json"), path.join(dst, "data", "picks.json"));
fs.copyFileSync(path.join(__dirname, "demo-backtest.json"), path.join(dst, "data", "backtest.json"));
fs.copyFileSync(path.join(__dirname, "demo-ledger.json"), path.join(dst, "data", "ledger.json"));
fs.copyFileSync(path.join(__dirname, "demo-mf.json"), path.join(dst, "data", "mf.json"));
console.log("UI synced from docs/");
