"use client"

import { Heart, Plus, Hotel, MapPin, Star, RefreshCw, X, Check, Bookmark } from "lucide-react"
import { Button } from "@/components/ui/button"
import Image from "next/image"
import { motion } from "framer-motion"
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
export function CompactPoiCard({ poi, isSaved, isItineraryItem, onToggleSave, onAddPoi, onReplan }: PoiCardProps) {
  return (
    <motion.div 
      className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100"
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
          quality={80}
        />
        
        {/* Top right love button */}
        <div className="absolute top-3 right-3 flex gap-2">
          <Button
            size="icon"
            variant="secondary"
            className="w-7 h-7 rounded-full bg-white/80 hover:bg-white shadow-md"
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
              className="rounded-full bg-white text-gray-700 shadow-sm cursor-default px-3 py-0.5 text-xs"
            >
              <Check className="w-4 h-4 mr-1" /> Added
            </Button>
          ) : (
            <Button
              size="sm"
              className="bg-black/80 hover:bg-black text-white rounded-full px-3 py-0.5 text-xs"
              onClick={() => onAddPoi(poi)}
            >
              Add to trip
            </Button>
          )}
        </div>
        
      </div>
      
      <div className="p-3">
        <div className="flex justify-between items-start mb-2">
          <h3 className="font-semibold text-base text-gray-900 leading-tight">{poi.name}</h3>
          {poi.rating && (
            <div className="flex items-center gap-1 ml-2">
              <Star className="w-4 h-4 fill-black text-black" />
              <span className="text-sm font-medium text-gray-900">{poi.rating}</span>
              <span className="text-sm text-gray-500">(995)</span>
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-2 text-xs text-gray-600 mb-2">
          <span>🍽️</span>
          <span>{poi.type}</span>
        </div>
        
        <p className="text-xs text-gray-600 mb-2">Address : {poi.address}</p>
      </div>
    </motion.div>
  )
}
