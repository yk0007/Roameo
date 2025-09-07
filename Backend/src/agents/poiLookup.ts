import type { POI } from "../types/schemas.js";

function haversineKm(a: { lat: number; lng: number }, b: { lat: number; lng: number }): number {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLng = ((b.lng - a.lng) * Math.PI) / 180;
  const s1 = Math.sin(dLat / 2) ** 2;
  const s2 = Math.sin(dLng / 2) ** 2;
  const c = Math.cos((a.lat * Math.PI) / 180) * Math.cos((b.lat * Math.PI) / 180);
  const d = 2 * R * Math.asin(Math.sqrt(s1 + c * s2));
  return d;
}

export function estimateTravelMinutes(a: { lat: number; lng: number }, b: { lat: number; lng: number }): number {
  const dist = haversineKm(a, b);
  // Assume avg speed ~ 30 km/h in city; clamp 5..180
  const mins = Math.max(5, Math.min(180, (dist / 30) * 60));
  return Math.round(mins);
}

export async function searchPOIByName(name: string, near?: { lat: number; lng: number }): Promise<POI | null> {
  const key = process.env.GOOGLE_MAPS_API_KEY || process.env.PLACES_API_KEY || process.env.GOOGLE_PLACES_API_KEY;
  if (!key) {
    return null; // fallback gracefully when server-side key not configured
  }
  try {
    const params = new URLSearchParams({ query: name, key });
    if (near) {
      params.set("location", `${near.lat},${near.lng}`);
      params.set("radius", "50000");
    }
    const url = `https://maps.googleapis.com/maps/api/place/textsearch/json?${params.toString()}`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    const r = data?.results?.[0];
    if (!r) return null;
    const poi: POI = {
      id: r.place_id,
      name: r.name,
      type: "attraction",
      lat: r.geometry?.location?.lat,
      lng: r.geometry?.location?.lng,
      address: r.formatted_address,
      rating: r.rating,
      source: "google",
      photoUrl: undefined,
    } as POI;
    return poi;
  } catch {
    return null;
  }
}
