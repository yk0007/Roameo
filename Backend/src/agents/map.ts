import type { WsEvent, POI } from "../types/schemas.js";
import { GoogleMapsClient } from "../tools/maps.js";

export async function mapAgent(pois: POI[]): Promise<WsEvent> {
  const maps = new GoogleMapsClient();
  const routes: Array<{ from: [number, number]; to: [number, number]; polyline?: string }> = [];
  // Build sequential segments between adjacent POIs (limit to first 6 to avoid quota spikes)
  const max = Math.min(pois.length, 6);
  for (let i = 0; i < max - 1; i++) {
    const from: [number, number] = [pois[i].lat, pois[i].lng];
    const to: [number, number] = [pois[i + 1].lat, pois[i + 1].lng];
    try {
      const { polyline } = await maps.directions(from, to);
      routes.push({ from, to, polyline });
    } catch {
      routes.push({ from, to });
    }
  }
  return { type: "map.update", data: { pois, routes } };
}
