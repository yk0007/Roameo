"use client"

import { useState, useEffect } from "react"
import Image from "next/image"

interface CachedImageProps {
  src?: string
  alt: string
  className?: string
  onError?: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void
  priority?: boolean
  quality?: number
}

export function CachedImage({ 
  src, 
  alt, 
  className, 
  onError, 
  priority = true, 
  quality = 90 
}: CachedImageProps) {
  const [imageSrc, setImageSrc] = useState<string>(src || "/placeholder.svg")
  const [hasError, setHasError] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // Update imageSrc when src prop changes
  useEffect(() => {
    if (src && src !== imageSrc && !hasError) {
      setImageSrc(src)
      setHasError(false)
      setIsLoading(true)
    }
  }, [src, imageSrc, hasError])

  const handleError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    const target = e.currentTarget as HTMLImageElement
    if (target.src.endsWith("/placeholder.svg") || hasError) return
    
    console.log(`[cached-image] Image failed to load: ${target.src}`)
    setImageSrc("/placeholder.svg")
    setHasError(true)
    setIsLoading(false)
    
    if (onError) {
      onError(e)
    }
  }

  const handleLoad = () => {
    setIsLoading(false)
  }

  if (!imageSrc) {
    return (
      <div className={`bg-gray-200 flex items-center justify-center ${className}`}>
        <span className="text-gray-400 text-sm">No image</span>
      </div>
    )
  }

  return (
    <>
      {isLoading && !hasError && (
        <div className={`absolute inset-0 bg-gray-100 flex items-center justify-center ${className}`}>
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      )}
      {hasError ? (
        <div className={`bg-gradient-to-br from-gray-200 to-gray-300 flex items-center justify-center ${className}`}>
          <span className="text-gray-500 text-sm font-medium">{alt.charAt(0).toUpperCase()}</span>
        </div>
      ) : (
        <Image
          src={imageSrc}
          alt={alt}
          className={`transition-all duration-700 ease-out ${className}`}
          onError={handleError}
          onLoad={handleLoad}
          priority={priority}
          placeholder="empty"
          quality={quality}
          fill
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          style={{ objectFit: 'cover' }}
        />
      )}
    </>
  )
}
