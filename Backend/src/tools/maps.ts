import { env } from "../config/env.js";
import type { POI } from "../types/schemas.js";

// Helper function to get the backend base URL
function getBackendBaseUrl(): string {
  // In production, use the deployed backend URL
  if (process.env.NODE_ENV === 'production') {
    return process.env.BACKEND_URL || 'https://roameo.onrender.com';
  }
  // In development, use localhost
  return process.env.BACKEND_URL || 'http://localhost:4000';
}

// Helper function to generate absolute photo proxy URLs
function createPhotoProxyUrl(photoReference: string): string {
  const baseUrl = getBackendBaseUrl();
  return `${baseUrl}/api/proxy/photo?photo_reference=${encodeURIComponent(photoReference)}&maxwidth=800&key=${encodeURIComponent(env.GOOGLE_MAPS_API_KEY!)}`;
}

export interface PlaceQuery {
  q: string; // e.g. "hotels in Goa" or "restaurants near Goa"
  lat?: number;
  lng?: number;
}

export class GoogleMapsClient {
  async searchPlaces(query: PlaceQuery, type: POI["type"] = "attraction"): Promise<POI[]> {
    if (!env.GOOGLE_MAPS_API_KEY) {
      console.warn("[maps] No Google Maps API key configured");
      return [];
    }
    const base = "https://maps.googleapis.com/maps/api/place/textsearch/json";
    const params = new URLSearchParams({
      query: query.q,
      key: env.GOOGLE_MAPS_API_KEY,
    });
    if (query.lat && query.lng) {
      params.set("location", `${query.lat},${query.lng}`);
      params.set("radius", "20000");
    }
    const url = `${base}?${params.toString()}`;
    console.log("[maps] Searching places:", query.q, "Type:", type);
    try {
      const res = await fetch(url);
      if (!res.ok) {
        console.warn("[maps] textsearch HTTP error", res.status, res.statusText);
        return [];
      }
      const data: any = await res.json();
      console.log("[maps] API response status:", data?.status, "Results count:", data?.results?.length || 0);
      if (data?.status && data.status !== "OK") {
        console.warn("[maps] textsearch API status", data.status, data?.error_message || "");
        return [];
      }
      const results: any[] = data?.results || [];
      const mapped: POI[] = results.map((r) => ({
        id: r.place_id as string,
        name: r.name as string,
        type,
        lat: r.geometry?.location?.lat as number,
        lng: r.geometry?.location?.lng as number,
        address: r.formatted_address as string | undefined,
        rating: typeof r.rating === "number" ? (r.rating as number) : undefined,
        photoUrl:
          (r.photos && r.photos[0]?.photo_reference &&
            createPhotoProxyUrl(r.photos[0].photo_reference)) || undefined,
        source: "google",
      }));
      try {
        const withPhotos = mapped.filter((m) => !!m.photoUrl).length;
        if (results.length) {
        }
      } catch {}
      return mapped.slice(0, 10)
    } catch (e: any) {
      console.warn("[maps] textsearch failed:", e?.message || e)
      return []
    }
  }

  async directions(from: [number, number], to: [number, number]): Promise<{ polyline?: string }> {
    if (!env.GOOGLE_MAPS_API_KEY) return {};
    const base = "https://maps.googleapis.com/maps/api/directions/json";
    const params = new URLSearchParams({
      origin: `${from[0]},${from[1]}`,
      destination: `${to[0]},${to[1]}`,
      key: env.GOOGLE_MAPS_API_KEY,
    });
    const url = `${base}?${params.toString()}`;
    try {
      const res = await fetch(url);
      if (!res.ok) return {};
      const data: any = await res.json();
      const polyline: string | undefined = data?.routes?.[0]?.overview_polyline?.points;
      return polyline ? { polyline } : {};
    } catch {
      return {};
    }
  }
}
