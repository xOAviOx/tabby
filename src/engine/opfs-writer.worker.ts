/**
 * OPFS writes for browsers without `createWritable()`.
 *
 * Safari implements the origin private file system but not
 * `FileSystemFileHandle.createWritable()`; the only way to write there is
 * `createSyncAccessHandle()`, which is available *only* inside a dedicated worker. That is
 * the entire reason this file exists. Everything else about the loader is happy on the
 * main thread, so the fallback is kept as narrow as possible: one worker, one operation
 * per message, no knowledge of chunks or models.
 *
 * A sync access handle is an exclusive lock on the file, so a handle is opened per write
 * session and closed at the end of it. `store.ts` already opens one writable per chunk
 * rather than per packet, so the lock is held for exactly as long as a chunk download.
 */

interface OpenMessage {
  id: number;
  op: 'open';
  handle: FileSystemFileHandle;
  keepExistingData: boolean;
}
interface SeekMessage {
  id: number;
  op: 'seek';
  position: number;
}
interface WriteMessage {
  id: number;
  op: 'write';
  buffer: ArrayBuffer;
}
interface CloseMessage {
  id: number;
  op: 'close' | 'abort';
}

type WriterMessage = OpenMessage | SeekMessage | WriteMessage | CloseMessage;

/** The sync-handle surface this file uses. Not in every lib.dom yet. */
interface SyncAccessHandle {
  write(buffer: BufferSource, options?: { at?: number }): number;
  truncate(size: number): void;
  getSize(): number;
  flush(): void;
  close(): void;
}

let access: SyncAccessHandle | null = null;
let position = 0;

function reply(id: number, error?: unknown): void {
  if (error === undefined) {
    self.postMessage({ id, ok: true });
  } else {
    self.postMessage({ id, ok: false, error: String(error) });
  }
}

self.onmessage = async (event: MessageEvent<WriterMessage>) => {
  const message = event.data;
  try {
    switch (message.op) {
      case 'open': {
        if (access) {
          // A leaked handle would hold the file's exclusive lock forever, and the next
          // open would fail with a lock error far from the cause.
          access.close();
          access = null;
        }
        const handle = message.handle as unknown as {
          createSyncAccessHandle(): Promise<SyncAccessHandle>;
        };
        access = await handle.createSyncAccessHandle();
        if (message.keepExistingData) {
          position = access.getSize();
        } else {
          access.truncate(0);
          position = 0;
        }
        reply(message.id);
        return;
      }
      case 'seek': {
        if (!access) throw new Error('seek before open');
        position = message.position;
        reply(message.id);
        return;
      }
      case 'write': {
        if (!access) throw new Error('write before open');
        const written = access.write(new Uint8Array(message.buffer), { at: position });
        position += written;
        reply(message.id);
        return;
      }
      case 'close':
      case 'abort': {
        if (access) {
          // Only a clean close is worth flushing; an abort is discarding the file anyway.
          if (message.op === 'close') access.flush();
          access.close();
          access = null;
        }
        position = 0;
        reply(message.id);
        return;
      }
    }
  } catch (error) {
    // Any failure leaves the lock held unless it is dropped here.
    try {
      access?.close();
    } catch {
      // Already closed.
    }
    access = null;
    position = 0;
    reply(message.id, error);
  }
};
