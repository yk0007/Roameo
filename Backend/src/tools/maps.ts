import { env } from "../config/env.js";
import type { POI } from "../types/schemas.js";

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
        // Return mock data when API fails
        return this.getMockPois(query.q, type);
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
            `/api/proxy/photo?photo_reference=${encodeURIComponent(r.photos[0].photo_reference)}&maxwidth=800&key=${encodeURIComponent(
              env.GOOGLE_MAPS_API_KEY!
            )}`) || undefined,
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
      return this.getMockPois(query.q, type)
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

  private getMockPois(query: string, type: POI["type"]): POI[] {
    const destination = this.extractDestination(query);
    const mockData = this.getMockDataForDestination(destination, type);
    return mockData.slice(0, 8); // Return 8 mock POIs
  }

  private extractDestination(query: string): string {
    // Extract destination from queries like "hotels in Mumbai", "restaurants in Ooty"
    const match = query.match(/(?:in|near)\s+([^,]+)/i);
    return match ? match[1].trim() : query;
  }

  private getMockDataForDestination(destination: string, type: POI["type"]): POI[] {
    const dest = destination.toLowerCase();
    
    if (dest.includes('ooty') || dest.includes('udhagamandalam')) {
      return this.getOotyMockPois(type);
    } else if (dest.includes('mumbai') || dest.includes('bombay')) {
      return this.getMumbaiMockPois(type);
    } else if (dest.includes('goa')) {
      return this.getGoaMockPois(type);
    } else if (dest.includes('delhi')) {
      return this.getDelhiMockPois(type);
    }
    
    // Generic fallback POIs
    return this.getGenericMockPois(destination, type);
  }

  private getOotyMockPois(type: POI["type"]): POI[] {
    const base = { source: "mock" as const };
    
    if (type === "attraction") {
      return [
        { ...base, id: "ooty_lake", name: "Ooty Lake", type, lat: 11.4064, lng: 76.6932, rating: 4.2, address: "Lake Road, Ooty, Tamil Nadu" },
        { ...base, id: "botanical_gardens", name: "Government Botanical Gardens", type, lat: 11.4086, lng: 76.7047, rating: 4.3, address: "Botanical Garden Rd, Ooty" },
        { ...base, id: "doddabetta_peak", name: "Doddabetta Peak", type, lat: 11.3969, lng: 76.7342, rating: 4.1, address: "Doddabetta, Ooty" },
        { ...base, id: "rose_garden", name: "Centenary Rose Garden", type, lat: 11.4119, lng: 76.6969, rating: 4.0, address: "Elk Hill, Ooty" },
        { ...base, id: "tea_museum", name: "Tea Museum & Factory", type, lat: 11.4102, lng: 76.6950, rating: 4.2, address: "Doddabetta Rd, Ooty" }
      ];
    } else if (type === "restaurant") {
      return [
        { ...base, id: "place_to_bee", name: "Place To Bee", type, lat: 11.4102, lng: 76.6950, rating: 4.3, address: "Charing Cross, Ooty" },
        { ...base, id: "shinkows", name: "Shinkows Chinese Restaurant", type, lat: 11.4086, lng: 76.6969, rating: 4.1, address: "42, Commissioner's Rd, Ooty" },
        { ...base, id: "sidewalk_cafe", name: "Sidewalk Cafe", type, lat: 11.4119, lng: 76.6932, rating: 4.0, address: "Commercial Rd, Ooty" }
      ];
    } else if (type === "stay") {
      return [
        { ...base, id: "gateway_hotel", name: "The Gateway Hotel Church Road", type, lat: 11.4064, lng: 76.6969, rating: 4.4, address: "Church Rd, Ooty" },
        { ...base, id: "hotel_lakeview", name: "Hotel Lakeview", type, lat: 11.4086, lng: 76.6932, rating: 4.0, address: "Lake Road, Ooty" },
        { ...base, id: "fabhotel_hillview", name: "FabHotel Prime Hill View", type, lat: 11.4102, lng: 76.6950, rating: 3.8, address: "Commercial Rd, Ooty" }
      ];
    }
    return [];
  }

  private getMumbaiMockPois(type: POI["type"]): POI[] {
    const base = { source: "mock" as const };
    
    if (type === "attraction") {
      return [
        { ...base, id: "gateway_of_india", name: "Gateway of India", type, lat: 18.9220, lng: 72.8347, rating: 4.3, address: "Apollo Bandar, Colaba, Mumbai" },
        { ...base, id: "marine_drive", name: "Marine Drive", type, lat: 18.9441, lng: 72.8230, rating: 4.4, address: "Netaji Subhashchandra Bose Rd, Mumbai" },
        { ...base, id: "chhatrapati_shivaji", name: "Chhatrapati Shivaji Maharaj Terminus", type, lat: 18.9401, lng: 72.8352, rating: 4.2, address: "Fort, Mumbai" },
        { ...base, id: "elephanta_caves", name: "Elephanta Caves", type, lat: 18.9633, lng: 72.9315, rating: 4.1, address: "Elephanta Island, Mumbai" }
      ];
    } else if (type === "restaurant") {
      return [
        { ...base, id: "trishna", name: "Trishna", type, lat: 18.9220, lng: 72.8310, rating: 4.5, address: "Sai Baba Marg, Fort, Mumbai" },
        { ...base, id: "bademiya", name: "Bademiya", type, lat: 18.9220, lng: 72.8330, rating: 4.2, address: "Tulloch Rd, Colaba, Mumbai" },
        { ...base, id: "cafe_mondegar", name: "Cafe Mondegar", type, lat: 18.9200, lng: 72.8320, rating: 4.0, address: "Colaba Causeway, Mumbai" }
      ];
    } else if (type === "stay") {
      return [
        { ...base, id: "taj_mahal_palace", name: "The Taj Mahal Palace", type, lat: 18.9216, lng: 72.8330, rating: 4.6, address: "Apollo Bandar, Colaba, Mumbai" },
        { ...base, id: "abode_bombay", name: "Abode Bombay", type, lat: 18.9200, lng: 72.8310, rating: 4.2, address: "Lansdowne Rd, Colaba, Mumbai" }
      ];
    }
    return [];
  }

  private getGoaMockPois(type: POI["type"]): POI[] {
    const base = { source: "mock" as const };
    
    if (type === "attraction") {
      return [
        { ...base, id: "baga_beach", name: "Baga Beach", type, lat: 15.5557, lng: 73.7516, rating: 4.2, address: "Baga, Goa" },
        { ...base, id: "calangute_beach", name: "Calangute Beach", type, lat: 15.5438, lng: 73.7553, rating: 4.1, address: "Calangute, Goa" },
        { ...base, id: "fort_aguada", name: "Fort Aguada", type, lat: 15.4942, lng: 73.7737, rating: 4.0, address: "Candolim, Goa" }
      ];
    } else if (type === "restaurant") {
      return [
        { ...base, id: "fishermans_wharf", name: "Fisherman's Wharf", type, lat: 15.5000, lng: 73.7500, rating: 4.3, address: "Cavelossim, Goa" },
        { ...base, id: "vinayak_family", name: "Vinayak Family Restaurant", type, lat: 15.5500, lng: 73.7600, rating: 4.1, address: "Mapusa, Goa" }
      ];
    } else if (type === "stay") {
      return [
        { ...base, id: "taj_exotica", name: "Taj Exotica Resort & Spa", type, lat: 15.4000, lng: 73.7000, rating: 4.5, address: "Benaulim, Goa" },
        { ...base, id: "alila_diwa", name: "Alila Diwa Goa", type, lat: 15.2500, lng: 73.9500, rating: 4.4, address: "Majorda, Goa" }
      ];
    }
    return [];
  }

  private getDelhiMockPois(type: POI["type"]): POI[] {
    const base = { source: "mock" as const };
    
    if (type === "attraction") {
      return [
        { ...base, id: "red_fort", name: "Red Fort", type, lat: 28.6562, lng: 77.2410, rating: 4.3, address: "Netaji Subhash Marg, Delhi" },
        { ...base, id: "india_gate", name: "India Gate", type, lat: 28.6129, lng: 77.2295, rating: 4.4, address: "Rajpath, New Delhi" },
        { ...base, id: "qutub_minar", name: "Qutub Minar", type, lat: 28.5245, lng: 77.1855, rating: 4.2, address: "Mehrauli, Delhi" }
      ];
    } else if (type === "restaurant") {
      return [
        { ...base, id: "karim", name: "Karim's", type, lat: 28.6500, lng: 77.2400, rating: 4.2, address: "Jama Masjid, Delhi" },
        { ...base, id: "bukhara", name: "Bukhara", type, lat: 28.6139, lng: 77.2090, rating: 4.5, address: "ITC Maurya, Delhi" }
      ];
    } else if (type === "stay") {
      return [
        { ...base, id: "itc_maurya", name: "ITC Maurya", type, lat: 28.6139, lng: 77.2090, rating: 4.5, address: "Sardar Patel Marg, Delhi" },
        { ...base, id: "oberoi_delhi", name: "The Oberoi, New Delhi", type, lat: 28.6304, lng: 77.2177, rating: 4.6, address: "Dr APJ Abdul Kalam Rd, Delhi" }
      ];
    }
    return [];
  }

  private getGenericMockPois(destination: string, type: POI["type"]): POI[] {
    const base = { source: "mock" as const, lat: 20.5937, lng: 78.9629 }; // Center of India
    
    if (type === "attraction") {
      return [
        { ...base, id: `${destination}_attraction_1`, name: `${destination} Heritage Site`, type, rating: 4.2, address: `Heritage Area, ${destination}` },
        { ...base, id: `${destination}_attraction_2`, name: `${destination} Museum`, type, rating: 4.0, address: `Museum Road, ${destination}` },
        { ...base, id: `${destination}_attraction_3`, name: `${destination} Park`, type, rating: 4.1, address: `Central Park, ${destination}` }
      ];
    } else if (type === "restaurant") {
      return [
        { ...base, id: `${destination}_restaurant_1`, name: `${destination} Spice Kitchen`, type, rating: 4.3, address: `Main Street, ${destination}` },
        { ...base, id: `${destination}_restaurant_2`, name: `Local Flavors ${destination}`, type, rating: 4.1, address: `Food Street, ${destination}` }
      ];
    } else if (type === "stay") {
      return [
        { ...base, id: `${destination}_hotel_1`, name: `${destination} Grand Hotel`, type, rating: 4.2, address: `Hotel District, ${destination}` },
        { ...base, id: `${destination}_hotel_2`, name: `${destination} Palace Resort`, type, rating: 4.0, address: `Resort Area, ${destination}` }
      ];
    }
    return [];
  }
}
