"use client";

import type { POI } from "@/lib/types";

interface PoiTypeIconProps {
  poi?: Pick<POI, "type" | "name" | "tags">;
  className?: string;
}

function classifyPoiIcon(poi?: Pick<POI, "type" | "name" | "tags">) {
  if (!poi) {
    return "location";
  }

  const haystack = `${poi.name || ""} ${(poi.tags || []).join(" ")}`.toLowerCase();
  if (poi.type === "stay") {
    return "stay";
  }
  if (poi.type === "destination") {
    return "location";
  }
  if (poi.type === "restaurant") {
    return "restaurant";
  }
  if (/\btemple\b/.test(haystack)) {
    return "temple";
  }
  if (/\bbeach\b/.test(haystack)) {
    return "beach";
  }
  return "attraction";
}

export function PoiTypeIcon({ poi, className = "h-4 w-4" }: PoiTypeIconProps) {
  const kind = classifyPoiIcon(poi);

  if (kind === "stay") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
      >
        <path d="M5.5 11.188h13.875a1.5 1.5 0 0 1 1.5 1.5v3.562H4v-3.563a1.5 1.5 0 0 1 1.5-1.5ZM4 16.25v2.25M20.875 16.25v2.25" />
        <path d="M19.188 11.188V6.125A1.125 1.125 0 0 0 18.063 5H6.813a1.125 1.125 0 0 0-1.125 1.125v5.063" />
        <path d="M9.813 8.375h5.25a.75.75 0 0 1 .75.75v2.063h-6.75V9.124a.75.75 0 0 1 .75-.75Z" />
      </svg>
    );
  }

  if (kind === "restaurant") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
      >
        <path d="M16.629 11.357c1.704 0 3.085-1.727 3.085-3.857s-1.381-3.857-3.085-3.857c-1.705 0-3.086 1.727-3.086 3.857s1.381 3.857 3.086 3.857Zm0 0v9M7.5 3.643v16.714m3.214-16.714v3.214a3.215 3.215 0 1 1-6.428 0V3.643" />
      </svg>
    );
  }

  if (kind === "temple") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="currentColor"
        stroke="currentColor"
        strokeWidth="0"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
      >
        <path d="M19.255 16.605a.527.527 0 0 0-.528-.527h-.188v-.779c0-3.34-1.793-6.4-4.694-8.033v-.444a3.48 3.48 0 0 0-1.318-2.73v-.565a.527.527 0 1 0-1.055 0v.565a3.48 3.48 0 0 0-1.317 2.73v.444a9.204 9.204 0 0 0-4.694 8.033v.779h-.188a.527.527 0 0 0 0 1.055h.188V21h13.078v-3.867h.188a.527.527 0 0 0 .527-.528Zm-1.77-1.306v.779H14.53v-2.702a10.76 10.76 0 0 0-1.38-5.27h.035a8.152 8.152 0 0 1 4.299 7.193Zm-6.962.779v-2.702c0-1.716.458-3.406 1.323-4.888L12 8.224l.154.264a9.704 9.704 0 0 1 1.323 4.888v2.702h-2.954Zm2.954 1.055v2.812h-2.954v-2.813h2.954ZM12 5.027c.5.457.79 1.105.79 1.795v.23h-1.58v-.23c0-.69.29-1.338.79-1.795ZM6.515 15.299a8.152 8.152 0 0 1 4.3-7.192h.034a10.76 10.76 0 0 0-1.38 5.27v2.7H6.515V15.3Zm0 1.834H9.47v2.812H6.515v-2.813Zm10.97 2.812H14.53v-2.813h2.954v2.813Z" />
        <path d="M11.473 18.539a.527.527 0 0 1 1.055 0 .527.527 0 0 1-1.055 0ZM15.48 18.539a.527.527 0 0 1 1.055 0 .527.527 0 0 1-1.055 0ZM7.465 18.539a.527.527 0 0 1 1.055 0 .527.527 0 0 1-1.055 0Z" />
      </svg>
    );
  }

  if (kind === "beach") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
      >
        <path d="m3.563 19.337.637.838a.733.733 0 0 0 1.125 0l.707-.848a.734.734 0 0 1 1.125 0l.709.848a.733.733 0 0 0 1.125 0l.707-.848a.734.734 0 0 1 1.125 0l.712.848a.733.733 0 0 0 1.125 0l.707-.848a.734.734 0 0 1 1.124 0l.708.848a.734.734 0 0 0 1.125 0l.707-.848a.734.734 0 0 1 1.125 0l.707.848a.733.733 0 0 0 1.125 0l.438-.525m-9.16-2.422a5.933 5.933 0 0 1 3.546-1.29h5.626m-8.72-10.99a7.34 7.34 0 0 1 8.142 2.346.734.734 0 0 1-.331 1.151L7.765 12.539a.735.735 0 0 1-.974-.697 7.34 7.34 0 0 1 4.927-6.894Zm0 0-.482-1.386m2.411 6.93 1.895 5.445" />
      </svg>
    );
  }

  if (kind === "attraction") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.3125"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
      >
        <path d="M19.599 10.071H4.402c-.72 0-1.03-.784-.463-1.157l7.598-4.975a.913.913 0 0 1 .926 0l7.598 4.975c.566.373.258 1.157-.462 1.157ZM19.714 17.143H4.286a.643.643 0 0 0-.643.643v1.928c0 .355.288.643.643.643h15.428a.643.643 0 0 0 .643-.643v-1.928a.643.643 0 0 0-.643-.643ZM5.571 10.071v7.072M8.786 10.071v7.072M12 10.071v7.072M15.214 10.071v7.072M18.428 10.071v7.072" />
      </svg>
    );
  }

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M18.75 9c0 3.735-6.75 12.75-6.75 12.75S5.25 12.735 5.25 9a6.75 6.75 0 0 1 13.5 0Z" />
      <path d="M12 11.25a2.25 2.25 0 1 0 0-4.5 2.25 2.25 0 0 0 0 4.5Z" />
    </svg>
  );
}

