"use client"

import { Heart, Check, Star } from "lucide-react"
import { Button } from "@/components/ui/button"
import Image from "next/image"
import { motion } from "framer-motion"
import { memo, useCallback } from "react"
import { resolvePoiImageUrl } from "@/lib/poi-image-url"
import type { POI } from "@/lib/types"
import { PoiTypeIcon } from "./poi-type-icon"

interface PoiCardProps {
  poi: POI
  isSaved: boolean
  isItineraryItem?: boolean
  isAddPending?: boolean
  onToggleSave: (poi: POI, next: boolean) => void
  onAddPoi: (poi: POI) => void
  onReplan: (poi: POI) => void
}


// Compact version for map hover cards
export const CompactPoiCard = memo(function CompactPoiCard({ poi, isSaved, isItineraryItem, isAddPending, onToggleSave, onAddPoi, onReplan }: PoiCardProps) {
  const handleToggleSave = useCallback(() => {
    onToggleSave(poi, !isSaved)
  }, [poi, isSaved, onToggleSave])

  const handleAddPoi = useCallback(() => {
    onAddPoi(poi)
  }, [poi, onAddPoi])

  const handleReplan = useCallback(() => {
    onReplan(poi)
  }, [poi, onReplan])
  const imageUrl = resolvePoiImageUrl(poi.photoUrl) || "/placeholder.svg"
  const addLabel = isItineraryItem ? "In Itinerary" : isAddPending ? "Adding..." : "Add to Itinerary"
  return (
    <motion.div 
      className="h-[218px] w-[248px] overflow-hidden rounded-[24px] border border-white/70 bg-white shadow-[0_18px_48px_rgba(15,23,42,0.18)]"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
    >
      <div className="relative overflow-hidden rounded-b-[20px]">
        <Image 
          src={imageUrl} 
          alt={poi.name} 
          width={400}
          height={160}
          className="h-32 w-full rounded-b-[20px] object-cover"
          quality={90}
          priority={true}
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          placeholder="blur"
          blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkqGx0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q=="
        />
        
        {/* Top right love button */}
        <div className="absolute top-3 right-3 flex gap-2">
          <Button
            size="icon"
            variant="secondary"
            className="w-7 h-7 rounded-full bg-white/80 hover:bg-white shadow-md"
            onClick={handleToggleSave}
          >
            <Heart className={`w-4 h-4 ${isSaved ? 'fill-red-500 text-red-500' : 'text-gray-600'}`} />
          </Button>
        </div>
        
        {/* Info badge / Add to trip button */}
        <div className="absolute top-3 left-3">
          <Button
            type="button"
            size="sm"
            onClick={handleAddPoi}
            disabled={isItineraryItem}
            aria-busy={isAddPending}
            className={`h-7 px-3 text-xs rounded-full font-medium shadow-sm ${
              isItineraryItem
                ? "bg-white/90 text-gray-700 hover:bg-white"
                : isAddPending
                  ? "bg-white/90 text-gray-700 hover:bg-white"
                  : "bg-white/90 text-gray-700 hover:bg-white"
            }`}
          >
            {isItineraryItem ? (
              <>
                <Check className="w-3.5 h-3.5 mr-1 inline" /> {addLabel}
              </>
            ) : (
              <>
                <PoiTypeIcon poi={poi} className="mr-1 inline h-3 w-3" /> {addLabel}
              </>
            )}
          </Button>
        </div>
        
      </div>
      
      <div className="space-y-2 p-3">
        <div className="flex justify-between items-start mb-2">
          <h3 className="line-clamp-2 font-semibold text-sm leading-5 text-gray-900">{poi.name}</h3>
          {poi.rating && (
            <div className="ml-2 flex items-center gap-1 rounded-full bg-slate-50 px-2 py-1">
              <Star className="h-3.5 w-3.5 fill-yellow-400 text-yellow-400" />
              <span className="text-xs font-medium text-gray-900">{poi.rating}</span>
            </div>
          )}
        </div>

        <p className="line-clamp-3 text-xs leading-5 text-gray-600">
          {poi.address || "Location available on the map"}
        </p>
      </div>
    </motion.div>
  )
})
