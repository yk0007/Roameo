"use client"

import { Heart, Plus, Hotel, MapPin, Star, RefreshCw, X, Check, Bookmark } from "lucide-react"
import { Button } from "@/components/ui/button"
import { OptimizedPoiImage } from "@/components/optimized-poi-image"
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
    <div className="bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-all duration-300 hover:scale-[1.02] border border-gray-100">
      <div className="relative">
        <OptimizedPoiImage 
          src={poi.photoUrl} 
          alt={poi.name} 
          className="w-full h-40 object-cover"
          width={400}
          height={160}
          quality="medium"
          poiName={poi.name}
          rating={poi.rating}
          enableThumbnail={true}
        />
        
        {/* Top right buttons (only save) */}
        <div className="absolute top-3 right-3 flex gap-2">
          <Button
            size="icon"
            variant="ghost"
            className="w-7 h-7 rounded-full bg-white/90 hover:bg-white shadow-sm border-0 transition-all duration-200 hover:scale-110"
            onClick={() => onToggleSave(poi, !isSaved)}
          >
            <Heart className={`w-4 h-4 transition-colors duration-200 ${isSaved ? 'fill-red-500 text-red-500' : 'text-gray-600'}`} />
          </Button>
        </div>
        
        {/* Add to trip / Added state */}
        <div className="absolute top-3 left-3">
          {isItineraryItem ? (
            <Button
              size="sm"
              variant="secondary"
              disabled
              className="rounded-full px-3 py-0.5 text-xs bg-emerald-50 text-emerald-700 shadow-sm cursor-default border-0"
            >
              <Check className="w-4 h-4 mr-1" /> Added
            </Button>
          ) : (
            <Button
              size="sm"
              className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white rounded-full px-3 py-0.5 text-xs transition-all duration-200 hover:scale-105 border-0"
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
    </div>
  )
}
