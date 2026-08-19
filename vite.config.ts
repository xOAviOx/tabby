import { defineConfig } from 'vite';

export default defineConfig({
  root: 'src/ui',
  publicDir: '../../public',
  build: {
    outDir: '../../dist',
    emptyOutDir: true,
    target: 'esnext',
  },
  server: {
    // OPFS + cross-origin isolation headroom for later milestones (M1 onward).
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
