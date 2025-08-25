"use client"

import { Renderer } from "@takumi-rs/core"
import { container, image, percentage } from "@takumi-rs/helpers"

class TakumiImageOptimizer {
  private renderer: Renderer | null = null
  private cache = new Map<string, string>()
  private loading = new Set<string>()

  async initRenderer() {
    if (!this.renderer) {
      this.renderer = new Renderer({
        fonts: [], // Add fonts if needed for text overlays
        persistentImages: []
      })
    }
    return this.renderer
  }

  async optimizeImage(
    src: string, 
    width: number = 400, 
    height: number = 300,
    quality: 'high' | 'medium' | 'low' = 'medium'
  ): Promise<string> {
    const cacheKey = `${src}-${width}x${height}-${quality}`
    
    // Return cached optimized image if available
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!
    }

    // Prevent duplicate processing
    if (this.loading.has(cacheKey)) {
      return src // Return original while processing
    }

    this.loading.add(cacheKey)

    try {
      const renderer = await this.initRenderer()
      
      // Create optimized image layout
      const layout = container({
        style: {
          width,
          height,
          backgroundColor: 0xf5f5f5, // Light gray background
          borderRadius: 8
        },
        children: [
          image({
            src,
            style: {
              width: percentage(100),
              height: percentage(100),
              objectFit: 'cover'
            }
          })
        ]
      })

      // Render to optimized format
      const format = quality === 'high' ? 'PNG' : 'WebP'
      const imageBuffer = await renderer.renderAsync(layout, {
        width,
        height,
        format: format as any
      })

      // Convert to data URL for immediate use
      const uint8Array = new Uint8Array(imageBuffer)
      const blob = new Blob([uint8Array], { 
        type: format === 'PNG' ? 'image/png' : 'image/webp' 
      })
      const dataUrl = URL.createObjectURL(blob)

      // Cache the optimized image
      this.cache.set(cacheKey, dataUrl)
      this.loading.delete(cacheKey)
      
      return dataUrl
    } catch (error) {
      console.warn('[takumi-optimizer] Failed to optimize image:', error)
      this.loading.delete(cacheKey)
      return src // Fallback to original
    }
  }

  // Create thumbnail with overlay text (for POI cards)
  async createPoiThumbnail(
    src: string,
    poiName: string,
    rating?: number,
    width: number = 400,
    height: number = 300
  ): Promise<string> {
    const cacheKey = `poi-${src}-${poiName}-${width}x${height}`
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!
    }

    if (this.loading.has(cacheKey)) {
      return src
    }

    this.loading.add(cacheKey)

    try {
      const renderer = await this.initRenderer()
      
      const layout = container({
        style: {
          width,
          height,
          position: 'relative',
          borderRadius: 12
        },
        children: [
          // Background image
          image({
            src,
            style: {
              width: percentage(100),
              height: percentage(100),
              objectFit: 'cover'
            }
          }),
          // Gradient overlay for text readability
          container({
            style: {
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: 60,
              backgroundImage: 'linear-gradient(transparent, rgba(0,0,0,0.7))'
            }
          })
        ]
      })

      const imageBuffer = await renderer.renderAsync(layout, {
        width,
        height,
        format: 'WebP' as any
      })

      const uint8Array = new Uint8Array(imageBuffer)
      const blob = new Blob([uint8Array], { type: 'image/webp' })
      const dataUrl = URL.createObjectURL(blob)

      this.cache.set(cacheKey, dataUrl)
      this.loading.delete(cacheKey)
      
      return dataUrl
    } catch (error) {
      console.warn('[takumi-optimizer] Failed to create POI thumbnail:', error)
      this.loading.delete(cacheKey)
      return src
    }
  }

  // Preload and optimize multiple images
  async preloadImages(urls: string[], dimensions: { width: number; height: number } = { width: 400, height: 300 }) {
    const promises = urls.map(url => 
      this.optimizeImage(url, dimensions.width, dimensions.height, 'medium')
    )
    
    try {
      await Promise.allSettled(promises)
    } catch (error) {
      console.warn('[takumi-optimizer] Some images failed to preload:', error)
    }
  }

  // Clear cache to free memory
  clearCache() {
    // Revoke object URLs to free memory
    this.cache.forEach(url => {
      if (url.startsWith('blob:')) {
        URL.revokeObjectURL(url)
      }
    })
    this.cache.clear()
    this.loading.clear()
  }

  // Get cache stats
  getCacheStats() {
    return {
      cached: this.cache.size,
      loading: this.loading.size
    }
  }
}

export const takumiOptimizer = new TakumiImageOptimizer()
