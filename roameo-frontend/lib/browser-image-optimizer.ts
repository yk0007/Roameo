"use client"

class BrowserImageOptimizer {
  private cache = new Map<string, string>()
  private loading = new Set<string>()

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
      // Load the image
      const img = await this.loadImage(src)
      
      // Create canvas for optimization
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      
      if (!ctx) {
        throw new Error('Canvas context not available')
      }

      // Set canvas dimensions
      canvas.width = width
      canvas.height = height

      // Calculate aspect ratio and positioning
      const aspectRatio = img.width / img.height
      const targetAspectRatio = width / height

      let drawWidth = width
      let drawHeight = height
      let offsetX = 0
      let offsetY = 0

      if (aspectRatio > targetAspectRatio) {
        // Image is wider than target
        drawWidth = height * aspectRatio
        offsetX = (width - drawWidth) / 2
      } else {
        // Image is taller than target
        drawHeight = width / aspectRatio
        offsetY = (height - drawHeight) / 2
      }

      // Fill background
      ctx.fillStyle = '#f5f5f5'
      ctx.fillRect(0, 0, width, height)

      // Draw image with cover behavior
      ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight)

      // Convert to optimized format
      const qualityValue = quality === 'high' ? 0.9 : quality === 'medium' ? 0.8 : 0.7
      const format = quality === 'high' ? 'image/png' : 'image/webp'
      
      const dataUrl = canvas.toDataURL(format, qualityValue)

      // Cache the optimized image
      this.cache.set(cacheKey, dataUrl)
      this.loading.delete(cacheKey)
      
      return dataUrl
    } catch (error) {
      console.warn('[browser-optimizer] Failed to optimize image:', error)
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
      // First optimize the base image
      const optimizedBase = await this.optimizeImage(src, width, height, 'medium')
      
      // Load the optimized image
      const img = await this.loadImage(optimizedBase)
      
      // Create canvas for thumbnail with overlay
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      
      if (!ctx) {
        throw new Error('Canvas context not available')
      }

      canvas.width = width
      canvas.height = height

      // Draw the base image
      ctx.drawImage(img, 0, 0, width, height)

      // Add gradient overlay for text readability
      const gradient = ctx.createLinearGradient(0, height - 60, 0, height)
      gradient.addColorStop(0, 'rgba(0, 0, 0, 0)')
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)')
      
      ctx.fillStyle = gradient
      ctx.fillRect(0, height - 60, width, 60)

      // Add POI name text
      ctx.fillStyle = 'white'
      ctx.font = 'bold 16px system-ui, -apple-system, sans-serif'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'bottom'
      
      // Truncate text if too long
      const maxWidth = width - 20
      let displayName = poiName
      const metrics = ctx.measureText(displayName)
      
      if (metrics.width > maxWidth) {
        while (ctx.measureText(displayName + '...').width > maxWidth && displayName.length > 0) {
          displayName = displayName.slice(0, -1)
        }
        displayName += '...'
      }
      
      ctx.fillText(displayName, 10, height - 30)

      // Add rating if available
      if (rating && rating > 0) {
        ctx.font = '14px system-ui, -apple-system, sans-serif'
        ctx.fillText(`⭐ ${rating.toFixed(1)}`, 10, height - 10)
      }

      const dataUrl = canvas.toDataURL('image/webp', 0.8)

      this.cache.set(cacheKey, dataUrl)
      this.loading.delete(cacheKey)
      
      return dataUrl
    } catch (error) {
      console.warn('[browser-optimizer] Failed to create POI thumbnail:', error)
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
      console.warn('[browser-optimizer] Some images failed to preload:', error)
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

  // Helper method to load image as Promise
  private loadImage(src: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = 'anonymous' // Enable CORS for external images
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error(`Failed to load image: ${src}`))
      img.src = src
    })
  }
}

export const browserImageOptimizer = new BrowserImageOptimizer()
