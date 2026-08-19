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
  test: {
    include: ['tests/**/*.test.ts'],
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
      }),
      instances: [{ browser: 'chromium' }],
    },
  },
});
