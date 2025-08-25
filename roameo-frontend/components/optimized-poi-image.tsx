"use client"

import { useState } from "react"
import Image from "next/image"

interface OptimizedPoiImageProps {
  src?: string
  alt: string
  className?: string
  width?: number
  height?: number
  quality?: 'high' | 'medium' | 'low'
  poiName?: string
  rating?: number
  enableThumbnail?: boolean
  priority?: boolean
  fill?: boolean
  sizes?: string
  onError?: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void
}

export function OptimizedPoiImage({ 
  src, 
  alt, 
  className, 
  width = 400, 
  height = 300,
  quality = 'medium',
  poiName,
  rating,
  enableThumbnail = false,
  priority = false,
  fill = false,
  sizes,
  onError 
}: OptimizedPoiImageProps) {
  const [imageSrc, setImageSrc] = useState<string>(src || "/placeholder.svg")
  const [isError, setIsError] = useState(false)

  const handleError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    if (imageSrc === "/placeholder.svg") return
    
    console.log(`[optimized-poi-image] Image failed to load: ${imageSrc}`)
    setImageSrc("/placeholder.svg")
    setIsError(true)
    
    if (onError) {
      onError(e)
    }
  }

  // Convert quality to numeric value for Vercel
  const qualityValue = quality === 'high' ? 95 : quality === 'medium' ? 80 : 65

  // If no src provided or error occurred, show placeholder
  if (!src || isError) {
    return (
      <div className={`bg-gray-100 flex items-center justify-center ${className}`}>
        <div className="text-gray-400 text-sm">No image</div>
      </div>
    )
  }

  // Handle POI thumbnail with overlay (if needed)
  if (enableThumbnail && poiName) {
    return (
      <div className="relative">
        {fill ? (
          <Image
            src={imageSrc}
            alt={alt}
            fill
            className={className}
            quality={qualityValue}
            priority={priority}
            sizes={sizes || "(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"}
            onError={handleError}
            style={{ objectFit: 'cover' }}
          />
        ) : (
          <Image
            src={imageSrc}
            alt={alt}
            width={width}
            height={height}
            className={className}
            quality={qualityValue}
            priority={priority}
            sizes={sizes}
            onError={handleError}
            style={{ objectFit: 'cover' }}
          />
        )}
        
        {/* POI overlay */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-3">
          <div className="text-white font-semibold text-sm truncate">{poiName}</div>
          {rating && rating > 0 && (
            <div className="text-white/90 text-xs">⭐ {rating.toFixed(1)}</div>
          )}
        </div>
      </div>
    )
  }

  // Standard optimized image
  if (fill) {
    return (
      <Image
        src={imageSrc}
        alt={alt}
        fill
        className={className}
        quality={qualityValue}
        priority={priority}
        sizes={sizes || "(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"}
        onError={handleError}
        style={{ objectFit: 'cover' }}
      />
    )
  }

  return (
    <Image
      src={imageSrc}
      alt={alt}
      width={width}
      height={height}
      className={className}
      quality={qualityValue}
      priority={priority}
      sizes={sizes}
      onError={handleError}
      style={{ objectFit: 'cover' }}
    />
  )
}
