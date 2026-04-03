import { GoogleMapsClient } from "./maps.js";
import { env } from "../config/env.js";

interface DestinationImageResult {
  imageUrl?: string;
  photoReference?: string;
}

export class DestinationImageService {
  private maps: GoogleMapsClient;

  constructor() {
    this.maps = new GoogleMapsClient();
  }

  private getPublicBaseUrl(): string {
    return (env.APP_BASE_URL || "http://localhost:4000").replace(/\/+$/, "");
  }

  private buildPhotoProxyUrl(photoReference: string): string {
    return `${this.getPublicBaseUrl()}/api/proxy/photo?maxwidth=1200&photo_reference=${encodeURIComponent(photoReference)}`;
  }

  /**
   * Fetch a high-quality destination image for a location
   * Returns the best available image URL or undefined if not found
   */
  async getDestinationImage(destination: string): Promise<DestinationImageResult> {
    try {
      console.log(`[destination-images] Searching for image: ${destination}`);
      
      // Check if Google Maps API key is available
      if (!env.GOOGLE_MAPS_API_KEY) {
        console.warn(`[destination-images] Google Maps API key not configured`);
        return {};
      }
      
      const searchQueries = [
        `${destination} famous landmark`,
        `${destination} iconic attraction`,
        `${destination} most famous place`,
        `${destination} tourism attraction`
      ];

      let bestPlace: any | undefined;
      for (const searchQuery of searchQueries) {
        const response = await fetch(
          `https://maps.googleapis.com/maps/api/place/textsearch/json?` +
          `query=${encodeURIComponent(searchQuery)}&` +
          `type=tourist_attraction&` +
          `key=${env.GOOGLE_MAPS_API_KEY}`
        );

        if (!response.ok) {
          console.error(`[destination-images] Places API error: ${response.status}`);
          continue;
        }

        const data = await response.json();
        if (data.status !== "OK" || !Array.isArray(data.results) || data.results.length === 0) {
          continue;
        }

        const normalizedDestination = destination.trim().toLowerCase();
        const placesWithPhotos = data.results
          .filter((place: any) => Array.isArray(place.photos) && place.photos.length > 0)
          .sort((a: any, b: any) => {
            const aHaystack = `${a.name || ""} ${a.formatted_address || ""}`.toLowerCase();
            const bHaystack = `${b.name || ""} ${b.formatted_address || ""}`.toLowerCase();
            const aDestinationBoost = aHaystack.includes(normalizedDestination) ? 2 : 0;
            const bDestinationBoost = bHaystack.includes(normalizedDestination) ? 2 : 0;
            const aRatings = typeof a.user_ratings_total === "number" ? Math.min(a.user_ratings_total / 500, 3) : 0;
            const bRatings = typeof b.user_ratings_total === "number" ? Math.min(b.user_ratings_total / 500, 3) : 0;
            const aScore = aDestinationBoost + (a.rating || 0) + aRatings;
            const bScore = bDestinationBoost + (b.rating || 0) + bRatings;
            return bScore - aScore;
          });

        if (placesWithPhotos.length > 0) {
          bestPlace = placesWithPhotos[0];
          break;
        }
      }

      if (!bestPlace) {
        console.log(`[destination-images] No landmark photos found for: ${destination}`);
        return {};
      }

      const photoReference = bestPlace.photos?.[0]?.photo_reference;
      if (!photoReference) {
        console.log(`[destination-images] No photo reference found`);
        return {};
      }

      console.log(`[destination-images] Found landmark image for ${destination}: ${bestPlace.name}`);

      return {
        imageUrl: this.buildPhotoProxyUrl(photoReference),
        photoReference
      };

    } catch (error) {
      console.error(`[destination-images] Error fetching image for ${destination}:`, error);
      return {};
    }
  }

  /**
   * Get multiple destination images for multi-destination trips
   * Returns the first successful image found
   */
  async getDestinationImageForTrip(destinations: string[]): Promise<DestinationImageResult> {
    if (!destinations || destinations.length === 0) {
      return {};
    }

    // Try each destination until we find an image
    for (const destination of destinations) {
      const result = await this.getDestinationImage(destination);
      if (result.imageUrl) {
        return result;
      }
    }

    return {};
  }
}
