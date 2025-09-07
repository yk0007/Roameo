"use client"

import { Heart, Plus, Hotel, MapPin, Star, RefreshCw, X, Check, Bookmark } from "lucide-react"
import { Button } from "@/components/ui/button"
import Image from "next/image"
import { motion } from "framer-motion"
import { memo, useCallback } from "react"
import type { POI } from "@/lib/types"

interface PoiCardProps {
  poi: POI
  isSaved: boolean
  isItineraryItem?: boolean
  onToggleSave: (poi: POI, next: boolean) => void
  onAddPoi: (poi: POI) => void
  onReplan: (poi: POI) => void
}


// Compact version for map hover cards
export const CompactPoiCard = memo(function CompactPoiCard({ poi, isSaved, isItineraryItem, onToggleSave, onAddPoi, onReplan }: PoiCardProps) {
  const handleToggleSave = useCallback(() => {
    onToggleSave(poi, !isSaved)
  }, [poi, isSaved, onToggleSave])

  const handleAddPoi = useCallback(() => {
    onAddPoi(poi)
  }, [poi, onAddPoi])

  const handleReplan = useCallback(() => {
    onReplan(poi)
  }, [poi, onReplan])
  return (
    <motion.div 
      className="bg-white rounded-2xl overflow-hidden shadow-sm w-full max-w-sm"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{
        scale: 1.05,
        y: -8,
        boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
        transition: { duration: 0.3, ease: "easeOut" }
      }}
      transition={{ duration: 0.2 }}
    >
      <div className="relative">
        <Image 
          src={poi.photoUrl || '/placeholder.svg'} 
          alt={poi.name} 
          width={400}
          height={160}
          className="w-full h-40 object-cover"
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
        
        {/* Info badge - no longer clickable for auto-add */}
        <div className="absolute top-3 left-3">
          {isItineraryItem ? (
            <div className="bg-white/90 text-gray-700 rounded-full px-3 py-0.5 text-xs font-medium shadow-sm">
              <Check className="w-4 h-4 mr-1 inline" /> In Itinerary
            </div>
          ) : (
            <div className="bg-white/90 text-gray-700 rounded-full px-3 py-0.5 text-xs font-medium shadow-sm">
              <MapPin className="w-3 h-3 mr-1 inline" /> Available
            </div>
          )}
        </div>
        
      </div>
      
      <div className="p-3">
        <div className="flex justify-between items-start mb-2">
          <h3 className="font-semibold text-base text-gray-900 leading-tight">{poi.name}</h3>
          {poi.rating && (
            <div className="flex items-center gap-1 ml-2">
              <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
              <span className="text-sm font-medium text-gray-900">{poi.rating}</span>
              <span className="text-sm text-gray-500">(995)</span>
            </div>
          )}
        </div>
        
        <p className="text-xs text-gray-600 mb-2">Address : {poi.address}</p>
      </div>
    </motion.div>
  )
})
