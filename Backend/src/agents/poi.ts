import type { WsEvent, POI } from "../types/schemas.js";
import { GoogleMapsClient } from "../tools/maps.js";

export async function poiAgent(q: { destination?: string }): Promise<WsEvent | null> {
  const maps = new GoogleMapsClient();
  if (!q.destination) return null;

  const [staysRaw, restaurantsRaw, attractionsRaw] = await Promise.all([
    maps.searchPlaces({ q: `hotels in ${q.destination}` }),
    maps.searchPlaces({ q: `restaurants in ${q.destination}` }),
    maps.searchPlaces({ q: `tourist attractions in ${q.destination}` }),
  ]);

  const mapToPOI = (r: { id: string; name: string; lat: number; lng: number; rating?: number; address?: string; photoUrl?: string }, type: POI["type"]): POI => ({
    id: r.id,
    name: r.name,
    type,
    lat: r.lat,
    lng: r.lng,
    rating: r.rating,
    address: r.address,
    photoUrl: r.photoUrl,
    source: "google",
  });

  const stays: POI[] = staysRaw.slice(0, 10).map((r) => mapToPOI(r, "stay"));
  const restaurants: POI[] = restaurantsRaw.slice(0, 10).map((r) => mapToPOI(r, "restaurant"));
  const attractions: POI[] = attractionsRaw.slice(0, 10).map((r) => mapToPOI(r, "attraction"));

  return { type: "search.results", data: { stays, restaurants, attractions } };
}
