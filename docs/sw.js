// App shell cached for offline; scan data always fetched fresh, falling back to the last copy.
const CACHE = "atm-scan-v6";
const SHELL = ["./", "index.html", "manifest.webmanifest", "icon-192.png", "icon-512.png", "atm-mark.svg", "fonts/oswald-latin-700-normal.woff2", "fonts/archivo-latin-400-normal.woff2", "fonts/archivo-latin-600-normal.woff2", "fonts/archivo-latin-700-normal.woff2", "fonts/archivo-latin-800-normal.woff2"];
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;
  if (url.pathname.endsWith("/data/picks.json")) {
    e.respondWith(fetch(e.request).then(r => {
      const copy = r.clone(); caches.open(CACHE).then(c => c.put("data/picks.json", copy)); return r;
    }).catch(() => caches.match("data/picks.json")));
    return;
  }
  e.respondWith(fetch(e.request).then(r => {
    const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return r;
  }).catch(() => caches.match(e.request)));
});
