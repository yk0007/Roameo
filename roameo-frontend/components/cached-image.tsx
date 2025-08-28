"use client"

import { useState } from "react"

interface CachedImageProps {
  src?: string
  alt: string
  className?: string
  onError?: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void
}

export function CachedImage({ src, alt, className, onError }: CachedImageProps) {
  const [imageSrc, setImageSrc] = useState<string>(src || "/placeholder.svg")

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
      loading="eager"
      decoding="sync"
    />
  )
}
