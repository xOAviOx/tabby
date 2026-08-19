import { defineConfig } from 'vite';

export default defineConfig({
  root: 'src/ui',
  publicDir: '../../public',
  build: {
    outDir: '../../dist',
    emptyOutDir: true,
    target: 'esnext',
    // publicDir holds ~1 GB of converted weights so the dev server and the test runner
    // can serve them over HTTP. They are deployment assets, not build inputs -- copying
    // them on every build would make `npm run build` move a gigabyte for nothing.
    // How they reach the CDN is an M6 decision.
    copyPublicDir: false,
  },
  server: {
    // OPFS + cross-origin isolation headroom for later milestones (M1 onward).
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
