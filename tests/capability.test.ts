/**
 * M6 capability detection: the three ways WebGPU can be missing must stay
 * distinguishable, because they have completely different fixes and look identical from
 * the page. A blank screen is the failure this guards against; so is confident advice
 * that points the wrong way.
 *
 * The suite runs in a browser that *does* have WebGPU, so the absent cases are produced
 * by stubbing `navigator.gpu` rather than by finding a machine without it.
 */

import { afterEach, describe, expect, it } from 'vitest';

import {
  requestGpuContext,
  WebGpuUnavailableError,
  type WebGpuUnavailableReason,
} from '../src/engine/device.js';

const realGpu = navigator.gpu;

function stubGpu(value: unknown): void {
  Object.defineProperty(navigator, 'gpu', { value, configurable: true, writable: true });
}

afterEach(() => {
  stubGpu(realGpu);
});

async function reasonOf(): Promise<WebGpuUnavailableReason> {
  try {
    await requestGpuContext();
  } catch (error) {
    if (error instanceof WebGpuUnavailableError) return error.reason;
    throw error;
  }
  throw new Error('expected requestGpuContext to throw');
}

describe('M6: WebGPU capability detection', () => {
  it('reports no-webgpu when the browser does not expose the API', async () => {
    stubGpu(undefined);
    expect(await reasonOf()).toBe('no-webgpu');
  });

  it('reports no-adapter when the API exists but yields no adapter', async () => {
    // The headless-shell case, and the blocklisted-driver case: navigator.gpu is present
    // and requestAdapter resolves null, which is not a browser-support problem at all.
    stubGpu({ requestAdapter: async () => null });
    expect(await reasonOf()).toBe('no-adapter');
  });

  it('still succeeds on this machine once the stub is removed', async () => {
    const ctx = await requestGpuContext();
    try {
      expect(ctx.device).toBeDefined();
    } finally {
      ctx.device.destroy();
    }
  });

  it('carries a reason on every unavailable error', () => {
    // The UI switches on this to choose what to tell someone, so an error without one
    // would silently fall through to undefined guidance.
    const reasons: WebGpuUnavailableReason[] = ['insecure-context', 'no-webgpu', 'no-adapter'];
    for (const reason of reasons) {
      const error = new WebGpuUnavailableError('test', reason);
      expect(error.reason).toBe(reason);
      expect(error).toBeInstanceOf(WebGpuUnavailableError);
    }
  });
});
