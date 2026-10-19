/// <reference lib="webworker" />
declare const self: ServiceWorkerGlobalScope

// PWA scaffold - Phase 0 (offline support in Phase 2)
// Workbox manifest - do not modify this line:
// self.__WB_MANIFEST

self.addEventListener('install', () => {
  self.skipWaiting()
})

self.addEventListener('activate', (event: ExtendableEvent) => {
  event.waitUntil(self.clients.claim())
})

export {}
