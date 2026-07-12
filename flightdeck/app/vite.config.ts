import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The "export" mode produces a single-chunk bundle (no code splitting) so the
// static export can inline the app into index.html and open from file://,
// where module scripts and dynamic import() are blocked by CORS.
export default defineConfig(({ mode }) => ({
  base: mode === 'export' ? './' : '/',
  plugins: [react()],
  server: { port: 5199, proxy: { '/api': 'http://127.0.0.1:8799', '/healthz': 'http://127.0.0.1:8799' } },
  build: mode === 'export'
    ? { outDir: 'dist-export', cssCodeSplit: false, rollupOptions: { output: { inlineDynamicImports: true } } }
    : undefined,
}));
