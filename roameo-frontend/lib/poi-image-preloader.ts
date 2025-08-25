"use client"

import { takumiOptimizer } from "@/lib/takumi-image-optimizer"
import type { POI } from "@/lib/types"

class PoiImagePreloader {
  private preloadQueue = new Set<string>()
  private isPreloading = false

  // Preload images for POIs that are likely to be viewed
  async preloadPoiImages(pois: POI[], priority: 'high' | 'medium' | 'low' = 'medium') {
    if (this.isPreloading) return

    this.isPreloading = true
    
    try {
      // Extract unique image URLs
      const imageUrls = pois
        .map(poi => poi.photoUrl)
        .filter((url): url is string => !!url && !this.preloadQueue.has(url))

      // Add to preload queue
      imageUrls.forEach(url => this.preloadQueue.add(url))

      // Determine dimensions based on priority
      const dimensions = this.getDimensionsByPriority(priority)

      // Preload in batches to avoid overwhelming the browser
      const batchSize = priority === 'high' ? 3 : priority === 'medium' ? 2 : 1
      
      for (let i = 0; i < imageUrls.length; i += batchSize) {
        const batch = imageUrls.slice(i, i + batchSize)
        
        await Promise.allSettled(
          batch.map(url => 
            takumiOptimizer.optimizeImage(url, dimensions.width, dimensions.height, priority === 'high' ? 'high' : 'medium')
          )
        )

        // Small delay between batches to prevent blocking
        if (i + batchSize < imageUrls.length) {
          await new Promise(resolve => setTimeout(resolve, 100))
        }
      }
    } catch (error) {
      console.warn('[poi-preloader] Preloading failed:', error)
    } finally {
      this.isPreloading = false
    }
  }

  // Preload images for POIs in viewport or about to enter viewport
  async preloadVisiblePois(pois: POI[]) {
    return this.preloadPoiImages(pois, 'high')
  }

  // Preload images for search results
  async preloadSearchResults(searchResults: { stays?: POI[], restaurants?: POI[], attractions?: POI[] }) {
    const allPois = [
      ...(searchResults.stays || []),
      ...(searchResults.restaurants || []),
      ...(searchResults.attractions || [])
    ]
    
    return this.preloadPoiImages(allPois, 'medium')
  }

  // Preload images for itinerary POIs (highest priority)
  async preloadItineraryPois(pois: POI[]) {
    return this.preloadPoiImages(pois, 'high')
  }

  private getDimensionsByPriority(priority: 'high' | 'medium' | 'low') {
    switch (priority) {
      case 'high':
        return { width: 400, height: 300 }
      case 'medium':
        return { width: 320, height: 240 }
      case 'low':
        return { width: 240, height: 180 }
    }
  }

  // Clear preload queue
  clearQueue() {
    this.preloadQueue.clear()
  }

  // Get preload stats
  getStats() {
    return {
      queueSize: this.preloadQueue.size,
      isPreloading: this.isPreloading,
      optimizerStats: takumiOptimizer.getCacheStats()
    }
  }
}

export const poiImagePreloader = new PoiImagePreloader()
