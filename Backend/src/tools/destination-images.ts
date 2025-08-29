import { GoogleMapsClient } from "./maps.js";

interface DestinationImageResult {
  imageUrl?: string;
  photoReference?: string;
}

export class DestinationImageService {
  private maps: GoogleMapsClient;

  constructor() {
    this.maps = new GoogleMapsClient();
  }

  /**
   * Fetch a high-quality destination image for a location
   * Returns the best available image URL or undefined if not found
   */
  async getDestinationImage(destination: string): Promise<DestinationImageResult> {
    try {
      console.log(`[destination-images] Searching for image: ${destination}`);
      
      // Use Places API to find the destination
      const searchQuery = `${destination} tourism attractions landmark`;
      const response = await fetch(
        `https://maps.googleapis.com/maps/api/place/textsearch/json?` +
        `query=${encodeURIComponent(searchQuery)}&` +
        `type=tourist_attraction&` +
        `key=${process.env.GOOGLE_MAPS_API_KEY}`
      );

      if (!response.ok) {
        console.error(`[destination-images] Places API error: ${response.status}`);
        return {};
      }

      const data = await response.json();
      
      if (data.status !== 'OK' || !data.results || data.results.length === 0) {
        console.log(`[destination-images] No places found for: ${destination}`);
        return {};
      }

      // Look for places with photos, prioritize highly-rated tourist attractions
      const placesWithPhotos = data.results
        .filter((place: any) => place.photos && place.photos.length > 0)
        .sort((a: any, b: any) => {
          // Prioritize by rating, then by number of photos
          const ratingDiff = (b.rating || 0) - (a.rating || 0);
          if (ratingDiff !== 0) return ratingDiff;
          return (b.photos?.length || 0) - (a.photos?.length || 0);
        });

      if (placesWithPhotos.length === 0) {
        console.log(`[destination-images] No photos found for places in: ${destination}`);
        return {};
      }

      const bestPlace = placesWithPhotos[0];
      const photo = bestPlace.photos[0]; // Get the first (usually best) photo
      
      if (!photo.photo_reference) {
        console.log(`[destination-images] No photo reference found`);
        return {};
      }

      // Generate high-quality image URL using Places API Photo service
      const imageUrl = `https://maps.googleapis.com/maps/api/place/photo?` +
        `maxwidth=800&` +
        `maxheight=600&` +
        `photo_reference=${photo.photo_reference}&` +
        `key=${process.env.GOOGLE_MAPS_API_KEY}`;

      console.log(`[destination-images] Found image for ${destination}: ${bestPlace.name}`);
      
      return {
        imageUrl,
        photoReference: photo.photo_reference
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