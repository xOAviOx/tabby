/**
 * Chunk transport and persistence.
 *
 * Weight chunks are fetched once, verified, and written to OPFS; every later load
 * reads from OPFS and never touches the network. Interrupted downloads resume from
 * the byte they reached rather than starting over -- on a ~1 GB model over a bad
 * connection that is the difference between "eventually works" and "never works".
 *
 * The chunk files are only a transport unit. `ChunkedByteReader` presents them as the
 * single contiguous byte stream that `model.json` offsets are expressed in, so a
 * tensor spanning several chunks is read with one call.
 *
 * Two OPFS characteristics shape this file, both learned the hard way:
 *
 *   - `createWritable({ keepExistingData: true })` copies the entire existing file into
 *     a swap file when it opens. Opening one per network packet turns a 32 MB chunk into
 *     tens of gigabytes of copying. A download therefore opens *one* writable per chunk
 *     and streams into it.
 *   - Every OPFS write goes through a swap file, and deleted bytes are not reclaimed
 *     against the quota promptly. An earlier design wrote `<name>.part`, then moved it
 *     to `<name>`, which meant ~3x the payload counted against quota at once -- 942 MiB
 *     of weights needed over 3 GiB, and an ephemeral Playwright context grants as
 *     little as 3 GiB. Chunks are now written straight to their final name, and
 *     completion is tracked in a small `verified.json` manifest instead.
 */

export interface ChunkMeta {
  name: string;
  bytes: number;
  sha256: string;
}

export type LoadPhase = 'download' | 'cache' | 'upload';

export interface LoadProgress {
  phase: LoadPhase;
  /** Bytes completed in this phase. */
  loadedBytes: number;
  /** Total bytes this phase will move. */
  totalBytes: number;
  /** Whatever is currently being worked on, for a status line. */
  detail: string;
  /** True when the bytes came from OPFS rather than the network. */
  fromCache: boolean;
}

export type ProgressCallback = (progress: LoadProgress) => void;

export class ModelStoreError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelStoreError';
  }
}

/**
 * Records which chunks have been fully downloaded *and* hash-verified. Without it a
 * complete-looking file would have to be re-hashed on every load, which would cost a
 * full pass over ~1 GB on what should be an instant warm start.
 */
const MANIFEST_FILE = 'verified.json';

interface VerifiedEntry {
  bytes: number;
  sha256: string;
}

async function sha256Hex(data: BufferSource): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * OPFS failures arrive as bare DOMExceptions with no stack, which makes a
 * QuotaExceededError from deep in a download almost impossible to place. Every OPFS
 * call that can fail goes through here so the message says what was being attempted.
 */
async function opfs<T>(what: string, fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    if (err instanceof DOMException) {
      throw new ModelStoreError(`${what}: ${err.name} — ${err.message}`);
    }
    throw err;
  }
}

export function isOpfsAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory;
}

/**
 * OPFS-backed store for one model's chunk files.
 *
 * Uses `createWritable()` rather than `createSyncAccessHandle()`: the sync handle is
 * faster but has historically been worker-only, and the loader must work on the main
 * thread too (the dev page and the test suite both run there).
 */
export class ModelStore {
  /**
   * `File` snapshots, kept so the upload phase's hundreds of ranged reads do not
   * re-open a handle each time. Invalidated on every mutation through this class.
   */
  private readonly files = new Map<string, File>();
  private manifest: Record<string, VerifiedEntry> | null = null;

  private constructor(
    private readonly dir: FileSystemDirectoryHandle,
    readonly modelId: string,
  ) {}

  static async open(modelId: string): Promise<ModelStore> {
    if (!isOpfsAvailable()) {
      throw new ModelStoreError('OPFS is unavailable: navigator.storage.getDirectory is missing.');
    }
    const root = await navigator.storage.getDirectory();
    const models = await root.getDirectoryHandle('models', { create: true });
    const dir = await models.getDirectoryHandle(modelId, { create: true });
    return new ModelStore(dir, modelId);
  }

  private invalidate(name: string): void {
    this.files.delete(name);
  }

  private async file(name: string): Promise<File> {
    const cached = this.files.get(name);
    if (cached) return cached;
    const handle = await this.dir.getFileHandle(name);
    const file = await handle.getFile();
    this.files.set(name, file);
    return file;
  }

  /** Size on disk of a file, or -1 when it is absent. */
  async sizeOf(name: string): Promise<number> {
    try {
      return (await this.file(name)).size;
    } catch {
      return -1;
    }
  }

  async readRange(name: string, offset: number, length: number): Promise<Uint8Array<ArrayBuffer>> {
    const file = await this.file(name);
    if (offset + length > file.size) {
      throw new ModelStoreError(
        `${name}: range ${offset}..${offset + length} exceeds file size ${file.size}`,
      );
    }
    return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
  }

  async readAll(name: string): Promise<Uint8Array<ArrayBuffer>> {
    return new Uint8Array(await (await this.file(name)).arrayBuffer());
  }

  /**
   * Open a stream for writing. The caller must close it. `keepExistingData` triggers a
   * full copy of the current contents, so pass it only when actually resuming.
   */
  async openWritable(
    name: string,
    options: { keepExistingData?: boolean } = {},
  ): Promise<FileSystemWritableFileStream> {
    return opfs(`open ${name} for writing`, async () => {
      const handle = await this.dir.getFileHandle(name, { create: true });
      this.invalidate(name);
      return handle.createWritable({ keepExistingData: options.keepExistingData ?? false });
    });
  }

  async write(name: string, data: BufferSource): Promise<void> {
    const writable = await this.openWritable(name);
    await opfs(`write ${name}`, async () => {
      await writable.write(data);
      await writable.close();
    });
    this.invalidate(name);
  }

  async remove(name: string): Promise<void> {
    this.invalidate(name);
    try {
      await this.dir.removeEntry(name);
    } catch {
      // Already absent.
    }
  }

  private async loadManifest(): Promise<Record<string, VerifiedEntry>> {
    if (this.manifest) return this.manifest;
    try {
      this.manifest = JSON.parse(new TextDecoder().decode(await this.readAll(MANIFEST_FILE)));
    } catch {
      this.manifest = {};
    }
    return this.manifest!;
  }

  /** True when this exact chunk (size and hash) has already been verified here. */
  async isVerified(name: string, entry: VerifiedEntry): Promise<boolean> {
    const known = (await this.loadManifest())[name];
    return !!known && known.bytes === entry.bytes && known.sha256 === entry.sha256;
  }

  async markVerified(name: string, entry: VerifiedEntry): Promise<void> {
    const manifest = await this.loadManifest();
    manifest[name] = { bytes: entry.bytes, sha256: entry.sha256 };
    await this.write(MANIFEST_FILE, new TextEncoder().encode(JSON.stringify(manifest)));
  }

  private async entryNames(): Promise<string[]> {
    const names: string[] = [];
    for await (const name of (this.dir as unknown as { keys(): AsyncIterable<string> }).keys()) {
      names.push(name);
    }
    return names;
  }

  /** Drop every file for this model. Backs the dev page's "clear cache" action. */
  async clear(): Promise<void> {
    for (const name of await this.entryNames()) await this.remove(name);
    this.manifest = null;
  }

  async list(): Promise<string[]> {
    return (await this.entryNames()).sort();
  }

  /** Bytes this model currently occupies in OPFS. */
  async usageBytes(): Promise<number> {
    let total = 0;
    for (const name of await this.entryNames()) {
      const size = await this.sizeOf(name);
      if (size > 0) total += size;
    }
    return total;
  }
}

export interface EnsureOptions {
  baseUrl: string;
  signal?: AbortSignal;
  /** Called with the number of bytes accounted for as they arrive. */
  onBytes?: (delta: number, fromCache: boolean) => void;
  /** Verify sha256 after download. Defaults to true. */
  verify?: boolean;
}

export interface EnsureResult {
  fromCache: boolean;
  /** Bytes actually pulled over the network (0 on a cache hit, partial on a resume). */
  networkBytes: number;
  resumedFrom: number;
}

/**
 * Make one chunk present and verified in the store, downloading only what is missing.
 *
 * Resume keeps partial data in `<name>.part` and issues a ranged request for the
 * remainder. A server that ignores `Range` answers 200 instead of 206, which is
 * detected and handled by restarting the chunk rather than corrupting it.
 */
export async function ensureChunk(
  store: ModelStore,
  chunk: ChunkMeta,
  options: EnsureOptions,
): Promise<EnsureResult> {
  const verify = options.verify ?? true;
  const existing = await store.sizeOf(chunk.name);

  if (existing === chunk.bytes) {
    // Full size already. Trust the manifest if it vouches for this exact chunk,
    // otherwise hash it once and record the result.
    if (!verify || (await store.isVerified(chunk.name, chunk))) {
      options.onBytes?.(chunk.bytes, true);
      return { fromCache: true, networkBytes: 0, resumedFrom: 0 };
    }
    const digest = await sha256Hex(await store.readAll(chunk.name));
    if (digest === chunk.sha256) {
      await store.markVerified(chunk.name, chunk);
      options.onBytes?.(chunk.bytes, true);
      return { fromCache: true, networkBytes: 0, resumedFrom: 0 };
    }
    await store.remove(chunk.name);
  } else if (existing > chunk.bytes) {
    // Longer than it should be: not a resumable prefix, so start over.
    await store.remove(chunk.name);
  }

  // Whatever is left is a prefix of the chunk, and resumable.
  let have = await store.sizeOf(chunk.name);
  if (have < 0) have = 0;
  const resumedFrom = have;
  if (have > 0) options.onBytes?.(have, true);

  const url = new URL(chunk.name, options.baseUrl).href;
  const headers: HeadersInit = have > 0 ? { Range: `bytes=${have}-` } : {};
  const response = await fetch(url, { headers, signal: options.signal });

  if (!response.ok) {
    throw new ModelStoreError(`${chunk.name}: HTTP ${response.status} ${response.statusText}`);
  }
  if (have > 0 && response.status !== 206) {
    // Range was ignored; the body is the whole file. Rewind rather than append.
    options.onBytes?.(-have, true);
    have = 0;
  }
  if (!response.body) {
    throw new ModelStoreError(`${chunk.name}: response has no body`);
  }

  // One writable for the whole chunk. Opening one per network packet would re-copy the
  // file on every packet, which is quadratic and, at 32 MB a chunk, ruinous.
  const writable = await store.openWritable(chunk.name, { keepExistingData: have > 0 });
  let networkBytes = 0;
  try {
    if (have > 0) await opfs(`seek ${chunk.name}`, () => writable.seek(have));
    const reader = response.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value?.byteLength) continue;
        await opfs(
          `write ${value.byteLength} B at ${have + networkBytes} of ${chunk.name}`,
          () => writable.write(value),
        );
        networkBytes += value.byteLength;
        options.onBytes?.(value.byteLength, false);
      }
    } finally {
      reader.releaseLock();
    }
    await opfs(`close ${chunk.name}`, () => writable.close());
  } catch (err) {
    await writable.abort().catch(() => undefined);
    throw err;
  }

  const finalSize = await store.sizeOf(chunk.name);
  if (finalSize !== chunk.bytes) {
    throw new ModelStoreError(
      `${chunk.name}: expected ${chunk.bytes} bytes, downloaded ${finalSize}`,
    );
  }

  if (verify) {
    const digest = await sha256Hex(await store.readAll(chunk.name));
    if (digest !== chunk.sha256) {
      // A resumed download can end up full-size but wrong if the existing prefix was
      // corrupt. Drop it so the next attempt starts clean rather than resuming onto bad
      // bytes forever.
      await store.remove(chunk.name);
      throw new ModelStoreError(
        `${chunk.name}: sha256 mismatch (expected ${chunk.sha256}, got ${digest})`,
      );
    }
    await store.markVerified(chunk.name, chunk);
  }

  return { fromCache: false, networkBytes, resumedFrom };
}

/**
 * Presents the chunk files as one contiguous byte stream.
 *
 * `model.json` records every tensor as an offset into this stream, which is what makes
 * chunk size a pure transport decision -- a 260 MiB embedding matrix spans nine 32 MB
 * chunks and is still read with a single call.
 */
export class ChunkedByteReader {
  private readonly total: number;

  constructor(
    private readonly store: ModelStore,
    private readonly chunks: readonly ChunkMeta[],
    private readonly chunkBytes: number,
  ) {
    this.total = chunks.reduce((sum, c) => sum + c.bytes, 0);
  }

  get totalBytes(): number {
    return this.total;
  }

  async read(offset: number, length: number): Promise<Uint8Array<ArrayBuffer>> {
    if (length === 0) return new Uint8Array(0);
    if (offset < 0 || offset + length > this.total) {
      throw new ModelStoreError(
        `read ${offset}..${offset + length} is outside the ${this.total}-byte stream`,
      );
    }

    const out = new Uint8Array(length);
    let written = 0;
    let cursor = offset;
    while (written < length) {
      const index = Math.floor(cursor / this.chunkBytes);
      const chunk = this.chunks[index];
      if (!chunk) throw new ModelStoreError(`no chunk at index ${index} for offset ${cursor}`);
      const withinChunk = cursor - index * this.chunkBytes;
      const take = Math.min(chunk.bytes - withinChunk, length - written);
      out.set(await this.store.readRange(chunk.name, withinChunk, take), written);
      written += take;
      cursor += take;
    }
    return out;
  }
}
