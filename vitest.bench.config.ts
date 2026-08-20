/**
 * Benchmarks run under their own config: they sweep configurations, take minutes, and are
 * evidence for the M5 optimization table rather than gates. The main config excludes them
 * so `npm test` stays fast.
 *
 * Standalone rather than merged with vitest.config.ts, because mergeConfig unions the
 * `include` arrays and would run the whole suite alongside the sweep. The browser settings
 * below are deliberately identical -- see vitest.config.ts for why each one matters.
 */
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
  publicDir: 'public',
  test: {
    include: ['tests/**/*.bench.test.ts'],
    reporters: ['verbose'],
    fileParallelism: false,
    testTimeout: 3_600_000,
    hookTimeout: 900_000,
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright({
        launchOptions: { channel: 'chromium', args: ['--enable-unsafe-webgpu'] },
        persistentContext: true,
      }),
      instances: [{ browser: 'chromium' }],
    },
  },
});
