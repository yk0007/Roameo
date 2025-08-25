"use client"

import { useState, useEffect } from "react"
import { takumiOptimizer } from "@/lib/takumi-image-optimizer"

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
  onError 
}: OptimizedPoiImageProps) {
  const [imageSrc, setImageSrc] = useState<string>("/placeholder.svg")
  const [isLoading, setIsLoading] = useState(false)
  const [isOptimized, setIsOptimized] = useState(false)

  useEffect(() => {
    if (!src) {
      setImageSrc("/placeholder.svg")
      setIsOptimized(false)
      return
    }

    const optimizeImage = async () => {
      setIsLoading(true)
      
      try {
        let optimizedSrc: string
        
        if (enableThumbnail && poiName) {
          // Create POI thumbnail with overlay
          optimizedSrc = await takumiOptimizer.createPoiThumbnail(
            src, 
            poiName, 
            rating, 
            width, 
            height
          )
        } else {
          // Standard image optimization
          optimizedSrc = await takumiOptimizer.optimizeImage(
            src, 
            width, 
            height, 
            quality
          )
        }
        
        setImageSrc(optimizedSrc)
        setIsOptimized(optimizedSrc !== src)
      } catch (error) {
        console.warn('[optimized-poi-image] Optimization failed, using original:', error)
        setImageSrc(src)
        setIsOptimized(false)
      } finally {
        setIsLoading(false)
      }
    }

    optimizeImage()
  }, [src, width, height, quality, poiName, rating, enableThumbnail])

  const handleError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    const target = e.currentTarget as HTMLImageElement
    if (target.src.endsWith("/placeholder.svg")) return
    
    console.log(`[optimized-poi-image] Image failed to load: ${target.src}`)
    setImageSrc("/placeholder.svg")
    setIsOptimized(false)
    
    if (onError) {
      onError(e)
    }
  }

  return (
    <div className="relative">
      <img
        src={imageSrc}
        alt={alt}
        className={className}
        onError={handleError}
        style={{ 
          opacity: isLoading ? 0.7 : 1,
          transition: 'opacity 0.3s ease-in-out'
        }}
      />
      
      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-75">
          <div className="w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
        </div>
      )}
      
      {/* Optimization indicator (for development) */}
      {process.env.NODE_ENV === 'development' && isOptimized && (
        <div className="absolute top-1 left-1 bg-green-500 text-white text-xs px-1 py-0.5 rounded">
          ⚡
        </div>
      )}
    </div>
  )
}
