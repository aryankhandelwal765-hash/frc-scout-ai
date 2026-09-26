import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Frontend runs on 5173; /api is proxied to the FastAPI backend on :8000 so the
// browser only ever talks to one origin (and streaming works cleanly).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
