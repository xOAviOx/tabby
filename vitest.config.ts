import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// GPU kernels can only be validated against a real WebGPU implementation, so the whole
// suite runs inside headless Chromium.
//
// `channel: 'chromium'` matters: Playwright's default headless build is
// chromium-headless-shell, which ships no GPU stack -- it exposes `navigator.gpu` but
// `requestAdapter()` always resolves to null. The full Chromium build in new-headless
// mode reaches the real Metal adapter.
export default defineConfig({
  // Serves public/models/** so the loader tests can fetch real chunk files over HTTP
  // rather than through a mock -- Range requests and streaming bodies are the parts of
  // the download path most likely to break, so they are exercised for real.
  publicDir: 'public',
  test: {
    include: ['tests/**/*.test.ts'],
    // One test file at a time. Files share a browser origin, so they share one OPFS
    // quota (4 GiB in an ephemeral Playwright context); running the ~1 GB Qwen gate
    // concurrently with the other files intermittently blew it. They also each create
    // a GPUDevice, which is likewise better serialised.
    fileParallelism: false,
    // The M0 gate requires the negotiated limits and per-shape errors to be visible,
    // and the default reporter swallows passing tests' stdout.
    reporters: ['verbose'],
    testTimeout: 120_000,
    hookTimeout: 120_000,
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright({
        launchOptions: {
          channel: 'chromium',
          args: ['--enable-unsafe-webgpu'],
        },
        // An ephemeral browser context keeps OPFS in memory and sizes its storage quota
        // from free RAM, which made the ~1 GB M1 gate fail intermittently (observed
        // quota swinging between 3072 and 4096 MiB run to run). A persistent context
        // puts OPFS on disk, which is both stable and what a real browser profile does.
        // Requires fileParallelism: false, which is set above.
        persistentContext: true,
      }),
      instances: [{ browser: 'chromium' }],
    },
  },
});
