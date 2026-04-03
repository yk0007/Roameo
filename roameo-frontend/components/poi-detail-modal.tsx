"use client"

import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"
import { Heart, Plus, MapPin, Star, Clock, Phone, Globe, Navigation, Check } from "lucide-react"
import { resolvePoiImageUrl } from "@/lib/poi-image-url"
import type { POI } from "@/lib/types"

interface PoiDetailModalProps {
  poi: POI | null
  isOpen: boolean
  onClose: () => void
  isSaved: boolean
  isItineraryItem?: boolean
  onToggleSave: (poi: POI, nextSaved: boolean) => void
  onAddPoi: (poi: POI) => void
  onReplan: (poi: POI) => void
}

export function PoiDetailModal({
  poi,
  isOpen,
  onClose,
  isSaved,
  isItineraryItem,
  onToggleSave,
  onAddPoi,
  onReplan
}: PoiDetailModalProps) {
  if (!poi) return null
  const imageUrl = resolvePoiImageUrl(poi.photoUrl) || "/placeholder.svg"

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl font-semibold">{poi.name}</DialogTitle>
        </DialogHeader>
        
        <div className="space-y-6 animate-in fade-in-50 duration-300">
          {/* Hero Image */}
          <div className="relative">
            <Image
              src={imageUrl}
              alt={poi.name}
              className="w-full h-64 object-cover rounded-lg"
              width={600}
              height={256}
              quality={90}
              priority={true}
              placeholder="empty"
            />
            
            {/* Action buttons overlay */}
            <div className="absolute top-4 right-4 flex gap-2">
              <Button
                size="icon"
                variant="ghost"
                className="w-10 h-10 rounded-full bg-white/90 hover:bg-white shadow-sm border-0 transition-all duration-200 hover:scale-105"
                onClick={() => onToggleSave(poi, !isSaved)}
              >
                <Heart className={`w-5 h-5 transition-colors duration-200 ${isSaved ? 'fill-red-500 text-red-500' : 'text-gray-600'}`} />
              </Button>
            </div>
            
            {/* Add to trip button */}
            <div className="absolute bottom-4 left-4">
              {isItineraryItem ? (
                <Button
                  size="sm"
                  variant="secondary"
                  disabled
                  className="rounded-full px-4 py-2 bg-emerald-50 text-emerald-700 shadow-sm cursor-default border-0"
                >
                  <Check className="w-4 h-4 mr-2" /> Added to Trip
                </Button>
              ) : (
                <Button
                  size="sm"
                  className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white rounded-full px-4 py-2 shadow-sm transition-all duration-200 hover:scale-105 border-0"
                  onClick={() => onAddPoi(poi)}
                >
                  <Plus className="w-4 h-4 mr-2" /> Add to Trip
                </Button>
              )}
            </div>
          </div>

          {/* Basic Info */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Badge variant="secondary" className="text-sm bg-blue-50 text-blue-700 border-0 rounded-full px-3 py-1">
                {poi.type}
              </Badge>
              {poi.rating && (
                <div className="flex items-center gap-1">
                  <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
                  <span className="font-semibold">{poi.rating}</span>
                  <span className="text-gray-500 text-sm">(995+ reviews)</span>
                </div>
              )}
            </div>

            {/* Address */}
            <div className="flex items-start gap-3">
              <MapPin className="w-5 h-5 text-gray-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-gray-700">{poi.address}</p>
                <Button variant="link" className="p-0 h-auto text-blue-600 text-sm hover:text-blue-700 transition-colors duration-200">
                  <Navigation className="w-4 h-4 mr-1" />
                  Get Directions
                </Button>
              </div>
            </div>

            {/* Description */}
            {poi.description && (
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900">About</h3>
                <p className="text-gray-700 leading-relaxed">{poi.description}</p>
              </div>
            )}

            {/* Opening Hours */}
            {poi.openingHours && (
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  Opening Hours
                </h3>
                <div className="text-gray-700 space-y-1">
                  {poi.openingHours.map((hours, index) => (
                    <div key={index} className="text-sm">
                      {hours}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Contact Info */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {poi.phone && (
                <div className="flex items-center gap-2">
                  <Phone className="w-4 h-4 text-gray-400" />
                  <a href={`tel:${poi.phone}`} className="text-blue-600 hover:text-blue-700 transition-colors duration-200">
                    {poi.phone}
                  </a>
                </div>
              )}
              {poi.website && (
                <div className="flex items-center gap-2">
                  <Globe className="w-4 h-4 text-gray-400" />
                  <a href={poi.website} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 transition-colors duration-200">
                    Visit Website
                  </a>
                </div>
              )}
            </div>

            {/* Price Range */}
            {poi.priceLevel && (
              <div className="space-y-2">
                <h3 className="font-semibold text-gray-900">Price Range</h3>
                <div className="flex items-center gap-1">
                  {Array.from({ length: 4 }, (_, i) => (
                    <span
                      key={i}
                      className={`text-lg ${i < (poi.priceLevel || 0) ? 'text-green-600' : 'text-gray-300'}`}
                    >
                      ₹
                    </span>
                  ))}
                  <span className="text-sm text-gray-500 ml-2">
                    {poi.priceLevel === 1 && "Budget-friendly"}
                    {poi.priceLevel === 2 && "Moderate"}
                    {poi.priceLevel === 3 && "Expensive"}
                    {poi.priceLevel === 4 && "Very Expensive"}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4 border-t border-gray-100">
            <Button
              variant="outline"
              className="flex-1 border-gray-200 hover:bg-gray-50 transition-all duration-200 hover:scale-[1.02]"
              onClick={() => onReplan(poi)}
            >
              Replan Around This
            </Button>
            <Button
              className="flex-1 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white transition-all duration-200 hover:scale-[1.02] border-0"
              onClick={() => {
                onAddPoi(poi)
                onClose()
              }}
              disabled={isItineraryItem}
            >
              {isItineraryItem ? "Already Added" : "Add to Trip"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
