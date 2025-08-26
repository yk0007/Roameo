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
        
      </div>
      
      <div className="p-3">
        {/* Add to trip / Added state */}
        <div className="flex gap-2 mb-3">
          {!isSaved && (
            <button
              onClick={() => onToggleSave(poi, true)}
              className="shadow-[inset_0_0_0_2px_#ef4444] text-red-500 px-4 py-2 rounded-full tracking-wider uppercase font-bold bg-transparent hover:bg-red-500 hover:text-white dark:text-red-400 transition duration-200 text-xs flex-1 flex items-center justify-center gap-1"
            >
              <Heart className="w-3 h-3" />
              Save
            </button>
          )}
          
          {!isItineraryItem && (
            <button
              onClick={() => onAddPoi(poi)}
              className="shadow-[inset_0_0_0_2px_#3b82f6] text-blue-500 px-4 py-2 rounded-full tracking-wider uppercase font-bold bg-transparent hover:bg-blue-500 hover:text-white dark:text-blue-400 transition duration-200 text-xs flex-1 flex items-center justify-center gap-1"
            >
              <Plus className="w-3 h-3" />
              Add
            </button>
          )}
          
          {isItineraryItem && (
            <button
              onClick={() => onReplan(poi)}
              className="shadow-[inset_0_0_0_2px_#f59e0b] text-amber-500 px-4 py-2 rounded-full tracking-wider uppercase font-bold bg-transparent hover:bg-amber-500 hover:text-white dark:text-amber-400 transition duration-200 text-xs flex-1 flex items-center justify-center gap-1"
            >
              <RefreshCw className="w-3 h-3" />
              Replan
            </button>
          )}
        </div>
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
