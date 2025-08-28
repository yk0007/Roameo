"use client"

import { Heart, Check, Star, ChevronLeft, ChevronRight, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import Image from "next/image"
import type { POI } from "@/lib/types"
import { useState } from "react"

interface SearchCardProps {
  poi: POI
  isSaved: boolean
  isItineraryItem?: boolean
  onToggleSave: (poi: POI, next: boolean) => void
  onAddPoi: (poi: POI) => void
  onReplan: (poi: POI) => void
  compact?: boolean
}

export function SearchCard({ poi, isSaved, isItineraryItem, onToggleSave, onAddPoi, onReplan, compact = false }: SearchCardProps) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [imageError, setImageError] = useState(false)
  
  // Mock multiple images - in real app, poi would have multiple photos
  const images = poi.photoUrl ? [poi.photoUrl] : []
  
  const nextImage = () => {
    if (images.length > 1) {
      setCurrentImageIndex((prev) => (prev + 1) % images.length)
    }
  }
  
  const prevImage = () => {
    if (images.length > 1) {
      setCurrentImageIndex((prev) => (prev - 1 + images.length) % images.length)
    }
  }

  const handleImageError = () => {
    setImageError(true)
  }

  const imgClass = "w-full h-48 object-cover"
  const padClass = compact ? "p-3" : "p-4"
  const titleClass = compact ? "font-semibold text-base text-gray-900 leading-tight" : "font-semibold text-lg text-gray-900 leading-tight"

  return (
    <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
      <div className="relative">
        <Image 
          src={imageError ? '/placeholder.svg' : (images[currentImageIndex] || poi.photoUrl || '/placeholder.svg')} 
          alt={poi.name} 
          className={imgClass}
          width={400}
          height={300}
          quality={75}
          priority={false}
          loading="lazy"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          placeholder="blur"
          blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkqGx0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q=="
          onError={handleImageError}
        />
        
        {/* Navigation arrows */}
        {images.length > 1 && (
          <>
            <Button
              size="icon"
              variant="secondary"
              className="absolute left-3 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/80 hover:bg-white shadow-md"
              onClick={prevImage}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <Button
              size="icon"
              variant="secondary"
              className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/80 hover:bg-white shadow-md"
              onClick={nextImage}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </>
        )}
        
        {/* Top right buttons (only save) */}
        <div className="absolute top-3 right-3 flex gap-2">
          <Button
            size="icon"
            variant="secondary"
            className={`rounded-full bg-white/80 hover:bg-white shadow-md ${compact ? "w-7 h-7" : "w-8 h-8"}`}
            onClick={() => onToggleSave(poi, !isSaved)}
          >
            <Heart className={`w-4 h-4 ${isSaved ? 'fill-red-500 text-red-500' : 'text-gray-600'}`} />
          </Button>
        </div>
        
        {/* Add to trip / Added state */}
        <div className="absolute top-3 left-3">
          {isItineraryItem ? (
            <Button
              size="sm"
              variant="secondary"
              disabled
              className={`rounded-full bg-white text-gray-700 shadow-sm cursor-default ${compact ? "px-3 py-0.5 text-xs" : "px-4 py-1 text-sm"}`}
            >
              <Check className="w-4 h-4 mr-1" /> Added
            </Button>
          ) : (
            <Button
              size="sm"
              className={`bg-black/80 hover:bg-black text-white rounded-full ${compact ? "px-3 py-0.5 text-xs" : "px-4 py-1 text-sm"}`}
              onClick={() => onAddPoi(poi)}
            >
              Add to trip
            </Button>
          )}
        </div>
        
        {/* Image dots indicator */}
        {images.length > 1 && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-1">
            {images.map((_, index) => (
              <div
                key={index}
                className={`w-2 h-2 rounded-full ${
                  index === currentImageIndex ? 'bg-white' : 'bg-white/50'
                }`}
              />
            ))}
          </div>
        )}
        
        {/* Info button */}
        {!compact && (
          <Button
            size="icon"
            variant="secondary"
            className="absolute bottom-3 right-3 w-8 h-8 rounded-full bg-black/60 hover:bg-black/80 text-white"
          >
            <Info className="w-4 h-4" />
          </Button>
        )}
      </div>

      <div className={padClass}>
        <div className="flex justify-between items-start mb-2">
          <h3 className={titleClass}>{poi.name}</h3>
          {poi.rating && (
            <div className="flex items-center gap-1 ml-2">
              <Star className="w-4 h-4 fill-black text-black" />
              <span className="text-sm font-medium text-gray-900">{poi.rating}</span>
              <span className="text-sm text-gray-500">({Math.floor(Math.random() * 1000)})</span>
            </div>
          )}
        </div>
        
        <div className={`flex items-center gap-2 ${compact ? "text-xs" : "text-sm"} text-gray-600 mb-2`}>
          <span>🍽️</span>
          <span>{poi.type}</span>
        </div>
        
        <p className={`${compact ? "text-xs" : "text-sm"} text-gray-600 ${compact ? "mb-2" : "mb-3"}`}>Address : {poi.address}</p>
    
      </div>
    </div>
  )
}
