// Simple image cache to avoid reloading images
class ImageCache {
  private cache = new Map<string, string>()
  private loading = new Set<string>()

  async getImage(url: string): Promise<string> {
    // Return cached image if available
    if (this.cache.has(url)) {
      return this.cache.get(url)!
    }

    // Return placeholder if already loading
    if (this.loading.has(url)) {
      return "/placeholder.svg"
    }

    // Start loading the image
    this.loading.add(url)
    
    try {
      // Convert relative proxy URLs to full URLs
      const fullUrl = this.getFullUrl(url)
      
      // Preload the image to cache it in browser
      const img = new Image()
      img.crossOrigin = "anonymous"
      
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = () => reject(new Error("Failed to load image"))
        img.src = fullUrl
      })
      
      // Cache the successful URL
      this.cache.set(url, fullUrl)
      this.loading.delete(url)
      return fullUrl
    } catch (error) {
      // Cache the placeholder for failed images
      this.cache.set(url, "/placeholder.svg")
      this.loading.delete(url)
      return "/placeholder.svg"
    }
  }

  private getFullUrl(url: string): string {
    // If it's already a full URL, return as is
    if (url.startsWith('http://') || url.startsWith('https://')) {
      return url
    }
    
    // If it's a proxy URL, prepend the backend base URL
    if (url.startsWith('/api/proxy/')) {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:4000'
      return `${backendUrl}${url}`
    }
    
    // For other relative URLs, return as is (like /placeholder.svg)
    return url
  }

  // Check if image is cached
  isCached(url: string): boolean {
    return this.cache.has(url)
  }

  // Get cached image or placeholder
  getCachedImage(url: string): string {
    return this.cache.get(url) || "/placeholder.svg"
  }

  // Clear cache
  clear(): void {
    this.cache.clear()
    this.loading.clear()
  }
}

export const imageCache = new ImageCache()
