import type { Poi, PoiCatalog, PoiType } from "@roameo/contracts";
import { env } from "../config/env.js";

const GOOGLE_PLACES_BASE = "https://maps.googleapis.com/maps/api/place";
const GOOGLE_DIRECTIONS_BASE = "https://maps.googleapis.com/maps/api/directions/json";

type PlacesBucket = {
  stays: Poi[];
  restaurants: Poi[];
  attractions: Poi[];
  catalog: PoiCatalog;
};

type RouteEstimate = {
  distanceKm: number;
  durationMinutes: number;
};

type TravelFact = {
  title: string;
  url: string;
  snippet: string;
};

export class PlanningToolError extends Error {
  constructor(
    message: string,
    readonly source: "places" | "directions",
    readonly retryable = true
  ) {
    super(message);
    this.name = "PlanningToolError";
  }
}

function getBackendUrl(): string {
  return env.APP_BASE_URL || "http://localhost:4000";
}

function createPhotoUrl(photoReference?: string): string | undefined {
  if (!photoReference || !env.GOOGLE_MAPS_API_KEY) {
    return undefined;
  }

  return `${getBackendUrl()}/api/proxy/photo?photo_reference=${encodeURIComponent(
    photoReference
  )}&maxwidth=900`;
}

function toTypeQuery(type: PoiType, destination: string): string {
  switch (type) {
    case "stay":
      return `boutique hotels in ${destination}`;
    case "restaurant":
      return `best local restaurants in ${destination}`;
    case "attraction":
      return `top tourist attractions in ${destination}`;
    default:
      return `${type} in ${destination}`;
  }
}

function sanitizePlaceType(type: PoiType): PoiType {
  if (type === "stay" || type === "restaurant" || type === "attraction") {
    return type;
  }
  return "attraction";
}

export class TravelToolsService {
  async searchPlacesForDestination(destination: string): Promise<PlacesBucket> {
    const [stays, restaurants, attractions] = await Promise.all([
      this.searchPlaces(destination, "stay"),
      this.searchPlaces(destination, "restaurant"),
      this.searchPlaces(destination, "attraction")
    ]);

    const items = [...stays, ...restaurants, ...attractions].reduce<
      PoiCatalog["items"]
    >((catalog, poi) => {
      catalog[poi.id] = poi;
      return catalog;
    }, {});

    return {
      stays,
      restaurants,
      attractions,
      catalog: {
        version: 1,
        items
      }
    };
  }

  async searchPlaces(destination: string, type: PoiType): Promise<Poi[]> {
    if (!env.GOOGLE_MAPS_API_KEY) {
      throw new PlanningToolError(
        "Google Maps Places is unavailable because the API key is missing.",
        "places",
        false
      );
    }

    const response = await fetch(
      `${GOOGLE_PLACES_BASE}/textsearch/json?query=${encodeURIComponent(
        toTypeQuery(type, destination)
      )}&key=${encodeURIComponent(env.GOOGLE_MAPS_API_KEY)}`
    );

    if (!response.ok) {
      throw new PlanningToolError(
        `Places search failed with ${response.status}`,
        "places"
      );
    }

    const data = (await response.json()) as any;
    if (data.status && data.status !== "OK" && data.status !== "ZERO_RESULTS") {
      throw new PlanningToolError(
        data.error_message || `Places search returned ${data.status}`,
        "places"
      );
    }

    return (data.results || []).slice(0, 8).map((place: any) => ({
      id: place.place_id,
      name: place.name,
      type: sanitizePlaceType(type),
      lat: place.geometry?.location?.lat,
      lng: place.geometry?.location?.lng,
      address: place.formatted_address || undefined,
      description: place.editorial_summary?.overview || undefined,
      photoUrl: createPhotoUrl(place.photos?.[0]?.photo_reference),
      website: undefined,
      phone: undefined,
      openingHours: place.opening_hours?.weekday_text || [],
      rating: typeof place.rating === "number" ? place.rating : undefined,
      priceLevel:
        typeof place.price_level === "number" ? place.price_level : undefined,
      source: "google_places",
      sourceId: place.place_id,
      tags: [destination]
    }));
  }

  async getPlaceDetails(placeId: string): Promise<Partial<Poi>> {
    if (!env.GOOGLE_MAPS_API_KEY) {
      return {};
    }

    const response = await fetch(
      `${GOOGLE_PLACES_BASE}/details/json?place_id=${encodeURIComponent(
        placeId
      )}&fields=website,formatted_phone_number,opening_hours,url&key=${encodeURIComponent(
        env.GOOGLE_MAPS_API_KEY
      )}`
    );

    if (!response.ok) {
      return {};
    }

    const data = (await response.json()) as any;
    const result = data.result || {};
    return {
      website: result.website || undefined,
      phone: result.formatted_phone_number || undefined,
      openingHours: result.opening_hours?.weekday_text || []
    };
  }

  async estimateRoute(
    from: { lat: number; lng: number },
    to: { lat: number; lng: number }
  ): Promise<RouteEstimate | null> {
    if (!env.GOOGLE_MAPS_API_KEY) {
      return null;
    }

    const response = await fetch(
      `${GOOGLE_DIRECTIONS_BASE}?origin=${from.lat},${from.lng}&destination=${to.lat},${to.lng}&key=${encodeURIComponent(
        env.GOOGLE_MAPS_API_KEY
      )}`
    );

    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as any;
    const leg = data.routes?.[0]?.legs?.[0];
    if (!leg) {
      return null;
    }

    return {
      distanceKm: Number(((leg.distance?.value || 0) / 1000).toFixed(1)),
      durationMinutes: Math.max(10, Math.round((leg.duration?.value || 0) / 60))
    };
  }

  async getDestinationFacts(destination: string): Promise<TravelFact[]> {
    if (!env.TAVILY_API_KEY) {
      return [];
    }

    const response = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        api_key: env.TAVILY_API_KEY,
        query: `current travel advice, weather, safety and logistics for ${destination}`,
        max_results: 3,
        search_depth: "basic"
      })
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as any;
    return (data.results || []).map((result: any) => ({
      title: result.title,
      url: result.url,
      snippet: result.content
    }));
  }
}
