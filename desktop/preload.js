const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("atmDesktop", {
  getSettings: () => ipcRenderer.invoke("get-settings"),
  saveSettings: (s) => ipcRenderer.invoke("save-settings", s),
  fetchScan: () => ipcRenderer.invoke("fetch-scan"),
  fetchJSON: (name) => ipcRenderer.invoke("fetch-json", name),
  openExternal: (url) => ipcRenderer.invoke("open-external", url),
  onScanUpdated: (cb) => ipcRenderer.on("scan-updated", () => cb()),
});
