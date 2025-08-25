"use client"

import { browserImageOptimizer } from "./browser-image-optimizer"

// Legacy compatibility wrapper for the browser-based image optimizer
// This maintains the same API as the original Takumi optimizer but uses browser-compatible Canvas API
class TakumiImageOptimizer {
  async optimizeImage(
    src: string, 
    width: number = 400, 
    height: number = 300,
    quality: 'high' | 'medium' | 'low' = 'medium'
  ): Promise<string> {
    return browserImageOptimizer.optimizeImage(src, width, height, quality)
  }

  async createPoiThumbnail(
    src: string,
    poiName: string,
    rating?: number,
    width: number = 400,
    height: number = 300
  ): Promise<string> {
    return browserImageOptimizer.createPoiThumbnail(src, poiName, rating, width, height)
  }

  async preloadImages(urls: string[], dimensions: { width: number; height: number } = { width: 400, height: 300 }) {
    return browserImageOptimizer.preloadImages(urls, dimensions)
  }

  clearCache() {
    return browserImageOptimizer.clearCache()
  }

  getCacheStats() {
    return browserImageOptimizer.getCacheStats()
  }
}

export const takumiOptimizer = new TakumiImageOptimizer()
