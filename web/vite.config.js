import { defineConfig } from "vite";

// Dev: `npm run dev` proxies API + scene files to the Python server on :8000.
// Build: output goes into the Python package so `python -m depthwizard` serves it offline.
export default defineConfig({
  base: "./",
  build: { outDir: "../depthwizard/static", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
  server: { proxy: { "/api": "http://127.0.0.1:8000", "/scenes": "http://127.0.0.1:8000" } },
});
