import type { WsEvent, POI } from "../types/schemas.js";
import { GoogleMapsClient } from "../tools/maps.js";

// Fallback POI data when Google Maps API fails
const getFallbackPOIs = (destination: string) => {
  const fallbackData: any = {
    "Ooty": {
      attractions: [
        { id: "ooty-lake", name: "Ooty Lake", lat: 11.4064, lng: 76.6932, rating: 4.2, address: "Ooty Lake Road, Ooty", type: "attraction" },
        { id: "botanical-garden", name: "Government Botanical Garden", lat: 11.4125, lng: 76.7085, rating: 4.3, address: "Botanical Garden Rd, Ooty", type: "attraction" },
        { id: "doddabetta-peak", name: "Doddabetta Peak", lat: 11.3995, lng: 76.7337, rating: 4.1, address: "Doddabetta Peak, Ooty", type: "attraction" },
        { id: "tea-museum", name: "Tea Museum", lat: 11.4102, lng: 76.6950, rating: 4.0, address: "Tea Museum, Ooty", type: "attraction" },
        { id: "rose-garden", name: "Centenary Rose Garden", lat: 11.4086, lng: 76.6969, rating: 4.2, address: "Elk Hill, Ooty", type: "attraction" }
      ],
      restaurants: [
        { id: "earl-secret", name: "Earl's Secret", lat: 11.4102, lng: 76.6950, rating: 4.3, address: "Elk Hill, Ooty", type: "restaurant" },
        { id: "place-be", name: "Place To Bee", lat: 11.4064, lng: 76.6932, rating: 4.1, address: "Commercial Road, Ooty", type: "restaurant" },
        { id: "nahar-sidewalk", name: "Nahar's Sidewalk Cafe", lat: 11.4125, lng: 76.7085, rating: 4.0, address: "Commissioner Road, Ooty", type: "restaurant" }
      ],
      stays: [
        { id: "taj-savoy", name: "Taj Savoy Hotel", lat: 11.4086, lng: 76.6969, rating: 4.4, address: "77 Sylks Rd, Club Road, Ooty", type: "stay" },
        { id: "sterling-ooty", name: "Sterling Ooty", lat: 11.4102, lng: 76.6950, rating: 4.2, address: "Fernhill Post, Ooty", type: "stay" },
        { id: "club-mahindra", name: "Club Mahindra Derby Green", lat: 11.4064, lng: 76.6932, rating: 4.1, address: "Ketti Valley Road, Ooty", type: "stay" }
      ]
    }
  };
  return fallbackData[destination] || fallbackData["Ooty"];
};

export async function poiAgent(q: { destination?: string }): Promise<WsEvent | null> {
  const maps = new GoogleMapsClient();
  if (!q.destination) return null;

  const [staysRaw, restaurantsRaw, attractionsRaw] = await Promise.all([
    maps.searchPlaces({ q: `hotels in ${q.destination}` }).catch(() => []),
    maps.searchPlaces({ q: `restaurants in ${q.destination}` }).catch(() => []),
    maps.searchPlaces({ q: `tourist attractions in ${q.destination}` }).catch(() => []),
  ]);

  // Use fallback data if API calls failed
  const fallbackData = getFallbackPOIs(q.destination);
  const finalStays = staysRaw.length > 0 ? staysRaw : fallbackData.stays;
  const finalRestaurants = restaurantsRaw.length > 0 ? restaurantsRaw : fallbackData.restaurants;
  const finalAttractions = attractionsRaw.length > 0 ? attractionsRaw : fallbackData.attractions;

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

  const stays: POI[] = finalStays.slice(0, 10).map((r: any) => mapToPOI(r, "stay"));
  const restaurants: POI[] = finalRestaurants.slice(0, 10).map((r: any) => mapToPOI(r, "restaurant"));
  const attractions: POI[] = finalAttractions.slice(0, 10).map((r: any) => mapToPOI(r, "attraction"));

  console.log(`[poi] Returning ${stays.length} stays, ${restaurants.length} restaurants, ${attractions.length} attractions for ${q.destination}`);
  return { type: "search.results", data: { stays, restaurants, attractions } };
}
