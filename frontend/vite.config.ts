import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend runs on :8000; /api/* is proxied there so the browser never needs CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', rewrite: (path) => path.replace(/^\/api/, '') },
    },
  },
})
