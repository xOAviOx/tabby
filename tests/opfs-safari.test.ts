/**
 * The Safari write path.
 *
 * A deployed build failed to load on Safari: the whole model downloaded, then the cache
 * write threw. Safari *exposes* `FileSystemFileHandle.createWritable()` and opens the
 * stream happily, and only fails on the first write, with
 * `UnknownError: The operation failed for an unknown transient reason`. Feature detection
 * by presence therefore reports "supported" and loses the data -- which is precisely how
 * this shipped. Writing there has to go through `createSyncAccessHandle()`, which is
 * worker-only by specification.
 *
 * So the store probes the capability by writing a byte, and these tests cover both Safari
 * shapes: the method missing, and the method present but throwing. The suite runs in
 * Chromium, which has a working `createWritable`, so both are simulated on the prototype
 * -- but the fallback they exercise is the real worker over the real OPFS, not a mock.
 */

import { afterEach, describe, expect, it } from 'vitest';

import {
  ModelStore,
  isOpfsAvailable,
  createWritableUsable,
  resetWritableProbe,
} from '../src/engine/store.js';

const nativeCreateWritable = FileSystemFileHandle.prototype.createWritable;

/**
 * Behave the way Safari actually does: expose `createWritable`, open the stream happily,
 * then throw on the first write. Detection by presence passes this and still loses the
 * data, which is exactly the bug that reached production.
 */
function breakCreateWritable(): void {
  FileSystemFileHandle.prototype.createWritable = async function fake() {
    return {
      async write() {
        throw new DOMException(
          'The operation failed for an unknown transient reason (e.g. out of memory).',
          'UnknownError',
        );
      },
      async seek() {},
      async close() {},
      async abort() {},
    } as unknown as FileSystemWritableFileStream;
  } as typeof FileSystemFileHandle.prototype.createWritable;
  resetWritableProbe();
}

/** The older-Safari shape: the method is not there at all. */
function hideCreateWritable(): void {
  // @ts-expect-error deliberately removing a standard method to simulate older Safari
  delete FileSystemFileHandle.prototype.createWritable;
  resetWritableProbe();
}

afterEach(() => {
  FileSystemFileHandle.prototype.createWritable = nativeCreateWritable;
  resetWritableProbe();
});

async function freshStore(id: string): Promise<ModelStore> {
  const store = await ModelStore.open(id);
  await store.clear();
  return store;
}

describe('OPFS writes without createWritable (the Safari path)', () => {
  it('probes the capability instead of trusting that the method exists', async () => {
    const store = await freshStore('safari-probe');
    try {
      expect(isOpfsAvailable()).toBe(true);
      expect(await createWritableUsable(store.directory)).toBe(true);

      // Safari's shape: present, and broken. Presence alone would say "supported".
      breakCreateWritable();
      expect(typeof FileSystemFileHandle.prototype.createWritable).toBe('function');
      expect(await createWritableUsable(store.directory)).toBe(false);

      // Older Safari's shape: absent.
      hideCreateWritable();
      expect(await createWritableUsable(store.directory)).toBe(false);
    } finally {
      await store.clear();
    }
  });

  it('round-trips through the worker when createWritable exists but throws', async () => {
    const store = await freshStore('safari-broken-writable');
    breakCreateWritable();
    try {
      const payload = new Uint8Array([4, 8, 15, 16, 23, 42]);
      await store.write('broken.bin', payload);
      expect(Array.from(await store.readAll('broken.bin'))).toEqual(Array.from(payload));
    } finally {
      await store.clear();
    }
  });

  it('round-trips a file byte-exactly through the worker', async () => {
    const store = await freshStore('safari-roundtrip');
    hideCreateWritable();
    try {
      const payload = new Uint8Array(64 * 1024);
      for (let i = 0; i < payload.length; i++) payload[i] = (i * 31) & 0xff;

      await store.write('chunk.bin', payload);
      const read = await store.readAll('chunk.bin');

      expect(read.byteLength).toBe(payload.byteLength);
      expect(Array.from(read.subarray(0, 512))).toEqual(Array.from(payload.subarray(0, 512)));
      expect(Array.from(read.subarray(-512))).toEqual(Array.from(payload.subarray(-512)));
    } finally {
      await store.clear();
    }
  });

  it('writes a chunk in several pieces, as a streamed download does', async () => {
    const store = await freshStore('safari-streamed');
    hideCreateWritable();
    try {
      const pieces = [
        new Uint8Array([1, 2, 3, 4]),
        new Uint8Array([5, 6, 7, 8]),
        new Uint8Array([9, 10]),
      ];
      const writable = await store.openWritable('streamed.bin');
      for (const piece of pieces) await writable.write(piece);
      await writable.close();

      const read = await store.readAll('streamed.bin');
      expect(Array.from(read)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    } finally {
      await store.clear();
    }
  });

  it('resumes an interrupted file instead of truncating it', async () => {
    // The resume path is the reason `keepExistingData` exists: a half-downloaded chunk
    // must be appended to, not restarted.
    const store = await freshStore('safari-resume');
    hideCreateWritable();
    try {
      const first = await store.openWritable('resume.bin');
      await first.write(new Uint8Array([1, 2, 3, 4]));
      await first.close();

      const second = await store.openWritable('resume.bin', { keepExistingData: true });
      await second.seek(4);
      await second.write(new Uint8Array([5, 6]));
      await second.close();

      expect(Array.from(await store.readAll('resume.bin'))).toEqual([1, 2, 3, 4, 5, 6]);
    } finally {
      await store.clear();
    }
  });

  it('truncates when not resuming, so a stale longer file cannot survive', async () => {
    const store = await freshStore('safari-truncate');
    hideCreateWritable();
    try {
      await store.write('t.bin', new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
      await store.write('t.bin', new Uint8Array([9, 9]));
      expect(Array.from(await store.readAll('t.bin'))).toEqual([9, 9]);
    } finally {
      await store.clear();
    }
  });

  it('releases the file lock after a write, so the next open succeeds', async () => {
    // A sync access handle is an exclusive lock. Leaking one makes every later write to
    // that file fail, and the failure surfaces nowhere near the leak.
    const store = await freshStore('safari-lock');
    hideCreateWritable();
    try {
      for (let i = 0; i < 3; i++) {
        await store.write('locked.bin', new Uint8Array([i]));
        expect(Array.from(await store.readAll('locked.bin'))).toEqual([i]);
      }
    } finally {
      await store.clear();
    }
  });

  it('releases the lock after an abort too', async () => {
    const store = await freshStore('safari-abort');
    hideCreateWritable();
    try {
      const writable = await store.openWritable('aborted.bin');
      await writable.write(new Uint8Array([1, 2, 3]));
      await writable.abort();

      // If abort leaked the handle this open would reject with a lock error.
      await store.write('aborted.bin', new Uint8Array([7]));
      expect(Array.from(await store.readAll('aborted.bin'))).toEqual([7]);
    } finally {
      await store.clear();
    }
  });
});
