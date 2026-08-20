/// <reference types="vite/client" />
/// <reference types="@webgpu/types" />

declare module '*.wgsl?raw' {
  const src: string;
  export default src;
}

interface ImportMetaEnv {
  /**
   * Base URL the model weights are fetched from. Unset in development, where Vite serves
   * them from public/. Must send CORS headers and support Range requests.
   */
  readonly VITE_MODEL_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
