// ATM Stock Scan — Windows desktop app (Electron).
// Reads the daily picks published by the cloud scanner, shows them in the ATM-branded UI,
// and raises a Windows notification when a new scan lands. Lives in the system tray.
const { app, BrowserWindow, Tray, Menu, Notification, ipcMain, shell, nativeImage } = require("electron");
const path = require("path");
const fs = require("fs");

const APP_ID = "in.absolutetechno.stockscan";
const UI = path.join(__dirname, "ui");
const ICON = path.join(__dirname, "build", process.platform === "win32" ? "icon.ico" : "icon.png");
const TRAY_ICON = path.join(__dirname, "build", "tray.png");
const POLL_MINUTES = 30;

let win = null, tray = null, quitting = false;
const userFile = (n) => path.join(app.getPath("userData"), n);
const readJSON = (f, d) => { try { return JSON.parse(fs.readFileSync(f, "utf8")); } catch { return d; } };

function getSettings() {
  return Object.assign({ dataUrl: "", autoStart: false, lastNotified: "" }, readJSON(userFile("settings.json"), {}));
}
function saveSettings(patch) {
  const s = Object.assign(getSettings(), patch);
  fs.mkdirSync(app.getPath("userData"), { recursive: true });
  fs.writeFileSync(userFile("settings.json"), JSON.stringify(s, null, 1));
  if ("autoStart" in patch && app.isPackaged) {
    app.setLoginItemSettings({ openAtLogin: !!s.autoStart, args: ["--hidden"] });
  }
  buildTrayMenu();
  return s;
}
const DATA_FILES = new Set(["picks.json", "backtest.json"]);
function dataUrl(base, name) {
  const root = base.replace(/data\/[^/]*\.json$/i, "").replace(/\/?$/, "/");
  return root + "data/" + name;
}

async function fetchJSON(name = "picks.json") {
  if (!DATA_FILES.has(name)) return { ok: false, error: "Unknown file" };
  const s = getSettings();
  if (!s.dataUrl) return { ok: false, error: "No scanner address set" };
  if (s.dataUrl === "demo") return { ok: true, data: readJSON(path.join(UI, "data", name), null) };
  const cacheFile = userFile("last-" + name);
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 20000);
    const r = await fetch(dataUrl(s.dataUrl, name) + "?t=" + Date.now(), { cache: "no-store", signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    if (!data || (name === "picks.json" && !data.as_of)) throw new Error("Not a scan file");
    fs.writeFileSync(cacheFile, JSON.stringify(data));
    return { ok: true, data };
  } catch (e) {
    const data = readJSON(cacheFile, null);
    return { ok: false, cached: !!data, data, error: String(e.message || e) };
  }
}
const fetchScan = () => fetchJSON("picks.json");

async function checkForNewScan() {
  const s = getSettings();
  if (!s.dataUrl || s.dataUrl === "demo") return;
  const r = await fetchScan();
  if (!r.ok || !r.data) return;
  const d = r.data;
  const key = `${d.as_of}|${d.generated_at || ""}`;
  if (key === s.lastNotified) return;
  saveSettings({ lastNotified: key });
  if (Notification.isSupported()) {
    const picks = d.picks || [];
    const m = d.market || {};
    const rg = d.regime && d.regime.regime ? `Regime ${d.regime.regime.toUpperCase()}. ` : "";
  const body = rg + (m.ok === false ? "Market filter OFF – watch only. " : "") +
      `${picks.length} candidate${picks.length === 1 ? "" : "s"}` +
      (picks.length ? ": " + picks.slice(0, 4).map((p) => p.symbol).join(", ") : " – no trade today.");
    const n = new Notification({ title: `ATM Stock Scan · ${d.as_of}`, body, icon: ICON });
    n.on("click", showWindow);
    n.show();
  }
  if (win) win.webContents.send("scan-updated");
}

function createWindow(hidden) {
  win = new BrowserWindow({
    width: 1240, height: 860, minWidth: 420, minHeight: 560,
    title: "ATM Stock Scan", icon: ICON, backgroundColor: "#FFFDF5", show: !hidden,
    autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  win.removeMenu();
  win.loadFile(path.join(UI, "index.html"));
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//i.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (e, url) => { if (!url.startsWith("file://")) e.preventDefault(); });
  win.on("close", (e) => {
    if (!quitting) { e.preventDefault(); win.hide(); }
  });
}

function showWindow() {
  if (!win) createWindow(false);
  win.show(); win.focus();
}

function buildTrayMenu() {
  if (!tray) return;
  const s = getSettings();
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: "Open ATM Stock Scan", click: showWindow },
    { label: "Check for new scan now", click: () => checkForNewScan() },
    { type: "separator" },
    { label: "Start with Windows", type: "checkbox", checked: !!s.autoStart, click: (i) => saveSettings({ autoStart: i.checked }) },
    { type: "separator" },
    { label: "Quit", click: () => { quitting = true; app.quit(); } },
  ]));
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", showWindow);
  app.whenReady().then(() => {
    app.setAppUserModelId(APP_ID);
    ipcMain.handle("get-settings", () => { const { lastNotified, ...rest } = getSettings(); return rest; });
    ipcMain.handle("save-settings", (_e, patch) => {
      const allowed = {};
      if (typeof patch?.dataUrl === "string") allowed.dataUrl = patch.dataUrl.trim();
      if (typeof patch?.autoStart === "boolean") allowed.autoStart = patch.autoStart;
      saveSettings(allowed);
      setTimeout(checkForNewScan, 1000);
      return true;
    });
    ipcMain.handle("fetch-scan", () => fetchScan());
    ipcMain.handle("fetch-json", (_e, name) => fetchJSON(String(name)));
    ipcMain.handle("open-external", (_e, url) => { if (/^https:\/\//i.test(String(url))) shell.openExternal(url); });

    tray = new Tray(nativeImage.createFromPath(TRAY_ICON));
    tray.setToolTip("ATM Stock Scan");
    tray.on("click", showWindow);
    buildTrayMenu();

    createWindow(process.argv.includes("--hidden"));
    setTimeout(checkForNewScan, 5000);
    setInterval(checkForNewScan, POLL_MINUTES * 60 * 1000);
  });
  app.on("before-quit", () => { quitting = true; });
  app.on("window-all-closed", (e) => e.preventDefault()); // keep running in tray
}
