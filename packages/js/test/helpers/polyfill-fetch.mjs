import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const nativeFetch = globalThis.fetch;

export async function ensureFileFetchPolyfill() {
  if (globalThis.__numjsFetchPolyfilled) {
    return;
  }
  globalThis.fetch = async (resource, init) => {
    let urlString = null;
    if (resource instanceof URL) {
      urlString = resource.href;
    } else if (typeof resource === "string") {
      urlString = resource;
    } else if (resource && typeof resource === "object" && "url" in resource) {
      const candidate = resource.url;
      if (typeof candidate === "string") {
        urlString = candidate;
      }
    }
    if (urlString && urlString.startsWith("file://")) {
      const path = fileURLToPath(urlString);
      const data = await readFile(path);
      return new Response(data);
    }
    if (resource instanceof URL && resource.protocol === "file:") {
      const path = fileURLToPath(resource);
      const data = await readFile(path);
      return new Response(data);
    }
    if (nativeFetch) {
      return nativeFetch(resource, init);
    }
    throw new Error("No fetch implementation available; please polyfill.");
  };
  Object.defineProperty(globalThis, "__numjsFetchPolyfilled", {
    value: true,
    configurable: false,
    enumerable: false,
    writable: false,
  });
}

await ensureFileFetchPolyfill();
