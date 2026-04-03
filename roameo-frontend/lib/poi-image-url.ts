import { BACKEND_URL } from "./types";

function trimTrailingSlash(value: string) {
  return value.replace(/\/+$/, "");
}

export function resolvePoiImageUrl(url?: string) {
  if (!url) {
    return undefined;
  }

  if (url === "/placeholder.svg") {
    return url;
  }

  if (url.startsWith("/api/proxy/photo")) {
    return `${trimTrailingSlash(BACKEND_URL)}${url}`;
  }

  try {
    const parsed = new URL(url);
    if (parsed.pathname === "/api/proxy/photo") {
      return `${trimTrailingSlash(BACKEND_URL)}${parsed.pathname}${parsed.search}`;
    }
  } catch {
    return url;
  }

  return url;
}
