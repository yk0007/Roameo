"use client"

import { useState } from "react"
import Image from "next/image"

interface VercelOptimizedImageProps {
  src?: string
  alt: string
  className?: string
  width?: number
  height?: number
  quality?: number
  priority?: boolean
  sizes?: string
  fill?: boolean
  onError?: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void
}

export function VercelOptimizedImage({ 
  src, 
  alt, 
  className, 
  width = 400, 
  height = 300,
  quality = 80,
  priority = false,
  sizes,
  fill = false,
  onError 
}: VercelOptimizedImageProps) {
  const [imageSrc, setImageSrc] = useState<string>(src || "/placeholder.svg")
  const [isError, setIsError] = useState(false)

  const handleError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    if (imageSrc === "/placeholder.svg") return
    
    console.log(`[vercel-optimized-image] Image failed to load: ${imageSrc}`)
    setImageSrc("/placeholder.svg")
    setIsError(true)
    
    if (onError) {
      onError(e)
    }
  }

  // If no src provided, show placeholder
  if (!src || isError) {
    return (
      <div className={`bg-gray-100 flex items-center justify-center ${className}`}>
        <div className="text-gray-400 text-sm">No image</div>
      </div>
    )
  }

  // Use fill prop for responsive containers
  if (fill) {
    return (
      <Image
        src={imageSrc}
        alt={alt}
        fill
        className={className}
        quality={quality}
        priority={priority}
        sizes={sizes || "(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"}
        onError={handleError}
        style={{ objectFit: 'cover' }}
      />
    )
  }

  // Use explicit dimensions
  return (
    <Image
      src={imageSrc}
      alt={alt}
      width={width}
      height={height}
      className={className}
      quality={quality}
      priority={priority}
      sizes={sizes}
      onError={handleError}
      style={{ objectFit: 'cover' }}
    />
  )
}
