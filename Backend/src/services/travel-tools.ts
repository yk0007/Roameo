import type {
  DateAdvisoryItem,
  DateContext,
  Poi,
  PoiCatalog,
  PoiType
} from "@roameo/contracts";
import { env } from "../config/env.js";

const GOOGLE_PLACES_BASE = "https://maps.googleapis.com/maps/api/place";
const GOOGLE_DIRECTIONS_BASE = "https://maps.googleapis.com/maps/api/directions/json";
const GOOGLE_GEOCODE_BASE = "https://maps.googleapis.com/maps/api/geocode/json";
const OPEN_METEO_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast";
const TAVILY_SEARCH_BASE = "https://api.tavily.com/search";

/**
 * Maps canonical POI types to Google Places `type` filter values.
 * Appending this to every text-search request prevents category bleed
 * (e.g. viewpoints appearing in restaurant results).
 */
const GOOGLE_PLACE_TYPE_FILTER: Record<"stay" | "restaurant" | "attraction", string> = {
  stay: "lodging",
  restaurant: "restaurant",
  attraction: "tourist_attraction"
};

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

export type DateWeatherSummary = {
  summary?: string;
  daily: Array<{ date: string; summary: string }>;
  advisories: DateAdvisoryItem[];
};

export type EventSummary = {
  summary?: string;
  items: Array<{ title: string; detail: string; sourceLabel?: string }>;
  advisories: DateAdvisoryItem[];
};

type DestinationGeo = {
  lat: number;
  lng: number;
  formatted: string;
  country?: string;
  countryCode?: string;
};

export type DiscoveryFocus =
  | "general"
  | "greeting"
  | "capabilities"
  | "hidden_gems"
  | "beaches"
  | "seafood"
  | "restaurants"
  | "attractions"
  | "culture"
  | "hotels"
  | "day_trips"
  | "family";

/**
 * Discovery query bank — maps a DiscoveryFocus + POI type to concrete
 * text-search strings sent to the Google Places API.  Queries are intentionally
 * typed so that the API type-filter can narrow results correctly.
 *
 * IMPORTANT: Only keys that match the focus's primary intent should be present.
 * For example, `restaurants` focus must NOT have an `attraction` key — that
 * would defeat the category isolation added to `searchPlacesByQueries`.
 */
const discoveryQueries: Record<
  DiscoveryFocus,
  Partial<Record<"stay" | "restaurant" | "attraction", string[]>>
> = {
  general: {
    stay: ["boutique hotels", "best stays"],
    restaurant: ["best local restaurants", "popular dining spots"],
    attraction: ["top tourist attractions", "local highlights"]
  },
  greeting: {
    attraction: ["best local highlights", "popular places to visit"]
  },
  capabilities: {
    attraction: ["popular local attractions", "scenic highlights"]
  },
  hidden_gems: {
    attraction: ["hidden gems", "quiet local spots", "off the beaten path"],
    restaurant: ["local favorites", "neighbourhood food spots"]
  },
  beaches: {
    attraction: ["best beaches near", "coastal spots"]
  },
  seafood: {
    // Type-filtered to `restaurant` only — no attractions here
    restaurant: ["best seafood restaurant", "local seafood dining", "fish restaurant"]
  },
  restaurants: {
    // Type-filtered to `restaurant` only — prevents viewpoints bleeding in
    restaurant: ["best restaurant", "popular local dining", "top rated restaurant"]
  },
  attractions: {
    // Type-filtered to `tourist_attraction` only
    attraction: ["top tourist attraction", "best places to visit", "popular landmark"]
  },
  culture: {
    attraction: ["cultural attraction", "heritage site", "museum"]
  },
  hotels: {
    stay: ["best hotel", "boutique hotel", "popular accommodation"]
  },
  day_trips: {
    attraction: ["best day trip from", "nearby scenic escape", "popular excursion"]
  },
  family: {
    stay: ["family-friendly hotel", "resort for families"],
    attraction: ["family-friendly attraction", "park for kids"]
  }
};

export class PlanningToolError extends Error {
  constructor(
    message: string,
    readonly source: "places" | "directions" | "weather" | "events" | "holidays" | "stays",
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

function toDiscoveryQueries(
  destination: string,
  focus: DiscoveryFocus,
  type: PoiType
): string[] {
  const scopedType = sanitizePlaceType(type);
  const base = discoveryQueries[focus]?.[scopedType] || [];
  if (base.length > 0) {
    return base.map((query: string) => `${query} in ${destination}`);
  }

  return [];
}

function sanitizePlaceType(
  type: PoiType
): "stay" | "restaurant" | "attraction" {
  if (type === "stay" || type === "restaurant" || type === "attraction") {
    return type;
  }
  return "attraction";
}

export class TravelToolsService {
  async geocodeDestination(destination: string): Promise<DestinationGeo | null> {
    const apiKey = env.GOOGLE_MAPS_API_KEY;
    if (!apiKey) {
      return null;
    }

    const response = await fetch(
      `${GOOGLE_GEOCODE_BASE}?address=${encodeURIComponent(
        destination
      )}&key=${encodeURIComponent(apiKey)}`
    );
    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as any;
    const result = data.results?.[0];
    if (!result?.geometry?.location) {
      return null;
    }

    const countryComponent = (result.address_components || []).find((component: any) =>
      Array.isArray(component.types) && component.types.includes("country")
    );

    return {
      lat: result.geometry.location.lat,
      lng: result.geometry.location.lng,
      formatted: result.formatted_address || destination,
      country: countryComponent?.long_name || undefined,
      countryCode: countryComponent?.short_name || undefined
    };
  }

  async searchPlacesForDestination(
    destination: string,
    focus: DiscoveryFocus = "general"
  ): Promise<PlacesBucket> {
    const [stays, restaurants, attractions] = await Promise.all([
      this.searchPlacesByQueries(destination, "stay", toDiscoveryQueries(destination, focus, "stay")),
      this.searchPlacesByQueries(
        destination,
        "restaurant",
        toDiscoveryQueries(destination, focus, "restaurant")
      ),
      this.searchPlacesByQueries(
        destination,
        "attraction",
        toDiscoveryQueries(destination, focus, "attraction")
      )
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
    return this.searchPlacesByQueries(destination, type, [toTypeQuery(type, destination)]);
  }

  async searchPlacesByQueries(
    destination: string,
    type: PoiType,
    queries: string[]
  ): Promise<Poi[]> {
    if (!queries || queries.length === 0) {
      return [];
    }
    
    const apiKey = env.GOOGLE_MAPS_API_KEY;
    if (!apiKey) {
      throw new PlanningToolError(
        "Google Maps Places is unavailable because the API key is missing.",
        "places",
        false
      );
    }

    // Apply a strict type filter to every query so Google Places cannot return
    // cross-category results (e.g. viewpoints for a restaurant query).
    const typeFilter = GOOGLE_PLACE_TYPE_FILTER[sanitizePlaceType(type)];

    const results = await Promise.all(
      queries.slice(0, 3).map(async (query) => {
        const url = new URL(`${GOOGLE_PLACES_BASE}/textsearch/json`);
        url.searchParams.set("query", query);
        url.searchParams.set("type", typeFilter);
        url.searchParams.set("key", apiKey);

        const response = await fetch(url.toString());

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

        return (data.results || []).map((place: any) => ({
          id: place.place_id,
          name: place.name,
          // Always use the caller's canonical type — never derive from Google's types array
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
          source: "google_places" as const,
          sourceId: place.place_id,
          tags: [destination, query]
        }));
      })
    );

    return Array.from(
      new Map(
        results.flat().map((poi) => [poi.id, poi] as const)
      ).values()
    ).slice(0, 8);
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

  async getWeatherSummary(
    destination: string,
    dateContext?: Pick<DateContext, "inferredStartDate" | "inferredEndDate" | "flexibility">
  ): Promise<DateWeatherSummary> {
    const geo = await this.geocodeDestination(destination);
    if (!geo) {
      return { daily: [], advisories: [] };
    }

    if (!dateContext?.inferredStartDate || !dateContext?.inferredEndDate) {
      return {
        daily: [],
        advisories:
          dateContext?.flexibility === "open_ended"
            ? [
                {
                  kind: "seasonal",
                  title: "Dates are still flexible",
                  detail: `I can give sharper weather guidance for ${destination} once you narrow the dates a bit.`
                }
              ]
            : []
      };
    }

    const start = new Date(dateContext.inferredStartDate);
    const end = new Date(dateContext.inferredEndDate);
    const today = new Date();
    const diffDays = Math.round(
      (start.getTime() - today.getTime()) / (1000 * 60 * 60 * 24)
    );

    if (diffDays > 16) {
      return {
        daily: [],
        advisories: [
          {
            kind: "seasonal",
            title: "Forecast window is still too far out",
            detail: `Open-Meteo gives reliable daily forecasts closer to the trip. For now I’ll treat weather as a seasonal consideration for ${destination}.`,
            startDate: dateContext.inferredStartDate,
            endDate: dateContext.inferredEndDate
          }
        ]
      };
    }

    const response = await fetch(
      `${OPEN_METEO_FORECAST_BASE}?latitude=${encodeURIComponent(
        String(geo.lat)
      )}&longitude=${encodeURIComponent(
        String(geo.lng)
      )}&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto`
    );
    if (!response.ok) {
      throw new PlanningToolError(
        `Weather lookup failed with ${response.status}`,
        "weather"
      );
    }

    const data = (await response.json()) as any;
    const times = data.daily?.time || [];
    const highs = data.daily?.temperature_2m_max || [];
    const lows = data.daily?.temperature_2m_min || [];
    const rain = data.daily?.precipitation_probability_max || [];
    const codes = data.daily?.weather_code || [];
    const daily = times
      .map((date: string, index: number) => ({
        date,
        summary: `${weatherCodeLabel(codes[index])}, ${Math.round(
          lows[index] || 0
        )}–${Math.round(highs[index] || 0)}°C, rain chance ${Math.round(
          rain[index] || 0
        )}%`,
        rainChance: Math.round(rain[index] || 0)
      }))
      .filter(
        (entry: { date: string }) =>
          entry.date >= dateContext.inferredStartDate! &&
          entry.date <= dateContext.inferredEndDate!
      );

    const advisories = daily
      .filter((entry: { rainChance: number }) => entry.rainChance >= 70)
      .map(
        (entry: { date: string; summary: string }): DateAdvisoryItem => ({
          kind: "weather",
          title: `High rain risk on ${entry.date}`,
          detail: entry.summary,
          startDate: entry.date,
          endDate: entry.date
        })
      );

    return {
      summary:
        daily.length > 0
          ? `Weather is looking ${daily.some((entry: any) => entry.rainChance >= 70) ? "mixed" : "fairly manageable"} across your current dates.`
          : undefined,
      daily: daily.map((entry: any) => ({
        date: entry.date,
        summary: entry.summary
      })),
      advisories
    };
  }

  async getHolidaySummary(
    destination: string,
    dateContext?: Pick<DateContext, "inferredStartDate" | "inferredEndDate">
  ): Promise<EventSummary> {
    if (!dateContext?.inferredStartDate || !dateContext?.inferredEndDate) {
      return { items: [], advisories: [] };
    }

    const geo = await this.geocodeDestination(destination);
    if (!geo?.countryCode) {
      return { items: [], advisories: [] };
    }

    const years = Array.from(
      new Set([
        dateContext.inferredStartDate.slice(0, 4),
        dateContext.inferredEndDate.slice(0, 4)
      ])
    );
    const holidays = (
      await Promise.all(
        years.map(async (year) => {
          const response = await fetch(
            `https://date.nager.at/api/v3/PublicHolidays/${encodeURIComponent(
              year
            )}/${encodeURIComponent(geo.countryCode!)}`
          );
          if (!response.ok) {
            return [];
          }
          return ((await response.json()) as any[]).map((item) => ({
            date: item.date,
            title: item.localName || item.name,
            detail: `${item.name} public holiday`,
            sourceLabel: "Nager.Date"
          }));
        })
      )
    ).flat();

    const items = holidays.filter(
      (item) =>
        item.date >= dateContext.inferredStartDate! &&
        item.date <= dateContext.inferredEndDate!
    );

    return {
      summary: items.length
        ? `There ${items.length === 1 ? "is" : "are"} ${items.length} public holiday note${items.length === 1 ? "" : "s"} during your current window.`
        : undefined,
      items,
      advisories: items.map(
        (item): DateAdvisoryItem => ({
          kind: "holiday",
          title: item.title,
          detail: `${item.detail}. Expect busier transport and local movement around ${destination}.`,
          startDate: item.date,
          endDate: item.date
        })
      )
    };
  }

  async getEventSummary(
    destination: string,
    dateContext?: Pick<DateContext, "inferredStartDate" | "inferredEndDate">
  ): Promise<EventSummary> {
    if (!env.TAVILY_API_KEY || !dateContext?.inferredStartDate || !dateContext?.inferredEndDate) {
      return { items: [], advisories: [] };
    }

    const response = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        api_key: env.TAVILY_API_KEY,
        query: `major festivals, local events, closures, seasonal cautions in ${destination} between ${dateContext.inferredStartDate} and ${dateContext.inferredEndDate}`,
        max_results: 4,
        search_depth: "advanced"
      })
    });

    if (!response.ok) {
      throw new PlanningToolError(
        `Event search failed with ${response.status}`,
        "events"
      );
    }

    const data = (await response.json()) as any;
    const items = (data.results || []).slice(0, 3).map((result: any) => ({
      title: result.title,
      detail: result.content,
      sourceLabel: hostnameLabel(result.url)
    }));

    return {
      summary: items.length
        ? `I found a few event and seasonal notes worth factoring into the trip.`
        : undefined,
      items,
      advisories: items.slice(0, 2).map(
        (item: { title: string; detail: string }): DateAdvisoryItem => ({
          kind: "event",
          title: item.title,
          detail: item.detail,
          startDate: dateContext.inferredStartDate,
          endDate: dateContext.inferredEndDate
        })
      )
    };
  }

  /**
   * Deep web research via Tavily.
   *
   * Use this to pull rich, up-to-date editorial content about a destination —
   * best time to visit, current advisories, top hidden tips, local culture.
   * Results are injected into the planning context so the LLM can cite them.
   *
   * @param destination - City or region name (e.g. "Araku Valley")
   * @param topic - Optional narrow focus (defaults to general travel guide)
   * @param depth - "basic" for fast lookups, "advanced" for richer research
   */
  async deepWebResearch(
    destination: string,
    topic = "travel guide tips local culture food attractions",
    depth: "basic" | "advanced" = "basic"
  ): Promise<Array<{ title: string; url: string; snippet: string; sourceLabel?: string }>> {
    if (!env.TAVILY_API_KEY) {
      return [];
    }

    const query = `${destination} ${topic}`;
    try {
      const response = await fetch(TAVILY_SEARCH_BASE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: env.TAVILY_API_KEY,
          query,
          max_results: depth === "advanced" ? 6 : 3,
          search_depth: depth,
          include_answer: false
        })
      });

      if (!response.ok) return [];

      const data = (await response.json()) as any;
      return (data.results || []).map((r: any) => ({
        title: r.title,
        url: r.url,
        snippet: r.content,
        sourceLabel: hostnameLabel(r.url)
      }));
    } catch {
      return [];
    }
  }
}

function weatherCodeLabel(code?: number): string {
  if (code === 0) {
    return "clear";
  }
  if ([1, 2].includes(code || -1)) {
    return "partly cloudy";
  }
  if (code === 3) {
    return "overcast";
  }
  if ([45, 48].includes(code || -1)) {
    return "foggy";
  }
  if ([51, 53, 55, 61, 63, 65, 80, 81, 82].includes(code || -1)) {
    return "rainy";
  }
  if ([71, 73, 75, 77, 85, 86].includes(code || -1)) {
    return "snowy";
  }
  if ([95, 96, 99].includes(code || -1)) {
    return "stormy";
  }
  return "mixed conditions";
}

function hostnameLabel(url?: string): string | undefined {
  try {
    return url ? new URL(url).hostname.replace(/^www\./, "") : undefined;
  } catch {
    return undefined;
  }
}
