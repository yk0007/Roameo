"use client"

import { useState, useEffect } from "react"
import { imageCache } from "@/lib/image-cache"

interface CachedImageProps {
  src?: string
  alt: string
  className?: string
  onError?: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void
}

export function CachedImage({ src, alt, className, onError }: CachedImageProps) {
  const [imageSrc, setImageSrc] = useState<string>("/placeholder.svg")
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (!src) {
      setImageSrc("/placeholder.svg")
      return
    }

    // Check if image is already cached
    if (imageCache.isCached(src)) {
      setImageSrc(imageCache.getCachedImage(src))
      return
    }

    // Load and cache the image
    setIsLoading(true)
    imageCache.getImage(src).then((cachedSrc) => {
      setImageSrc(cachedSrc)
      setIsLoading(false)
    }).catch(() => {
      setImageSrc("/placeholder.svg")
      setIsLoading(false)
    })
  }, [src])

  const handleError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    const target = e.currentTarget as HTMLImageElement
    if (target.src.endsWith("/placeholder.svg")) return
    
    console.log(`[cached-image] Image failed to load: ${target.src}`)
    setImageSrc("/placeholder.svg")
    
    if (onError) {
      onError(e)
    }
  }

  return (
    <img
      src={imageSrc}
      alt={alt}
      className={className}
      onError={handleError}
      style={{ 
        opacity: isLoading ? 0.7 : 1,
        transition: 'opacity 0.2s ease-in-out'
      }}
    />
  )
}
