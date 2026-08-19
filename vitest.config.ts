import { defineConfig } from 'vitest/config';

// GPU kernels can only be validated on a real WebGPU implementation, so the whole
// suite runs inside headless Chromium (full build, not headless-shell -- the shell
// has no GPU stack and therefore no navigator.gpu).
export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    testTimeout: 120_000,
    hookTimeout: 120_000,
    browser: {
      enabled: true,
      provider: 'playwright',
      headless: true,
      screenshotFailures: false,
      instances: [
        {
          browser: 'chromium',
          launch: {
            channel: 'chromium',
            args: [
              '--enable-unsafe-webgpu',
              '--enable-features=WebGPU',
              '--use-angle=metal',
            ],
          },
        },
      ],
    },
  },
});
