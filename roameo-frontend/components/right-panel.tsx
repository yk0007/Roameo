"use client"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { MapPin, Calendar, ChevronRight } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { MapView } from "./map-view"
import { ItineraryPanel } from "./itinerary-panel"
import { ShareButton } from "./share-button"
import { CachedImage } from "./cached-image"
import type { Itinerary, POI, Activity } from "@/lib/types"

interface RightPanelProps {
  activeView: "map" | "itinerary"
  onViewChange: (view: "map" | "itinerary") => void
  trip: any
  itinerary?: Itinerary
  mapData?: { pois: any[]; routes: Array<{ from: [number, number]; to: [number, number]; polyline?: string }> }
  onClose: () => void
  savedIds?: Set<string>
  itineraryPoiIds?: Set<string>
  onToggleSave?: (poi: POI, nextSaved: boolean) => void
  onAddPoi?: (poi: POI) => void
  onReplan?: (poi: POI) => void
}

export function RightPanel({
  activeView,
  onViewChange,
  trip,
  itinerary,
  mapData,
  onClose,
  savedIds,
  itineraryPoiIds,
  onToggleSave,
  onAddPoi,
  onReplan,
}: RightPanelProps) {

  // Nudge Google Maps to render when the Map tab becomes visible
  useEffect(() => {
    if (activeView === "map") {
      // Two frames to allow layout to settle
      const t = setTimeout(() => {
        window.dispatchEvent(new Event("resize"))
      }, 50)
      return () => clearTimeout(t)
    }
  }, [activeView])

  return (
    <div className="bg-white flex flex-col relative h-full">
      {/* Navigation Tabs */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-white/95 backdrop-blur-md rounded-full p-1 border border-white/30 shadow-xl transition-all duration-300">
        <Button
          variant={activeView === "map" ? "default" : "ghost"}
          size="sm"
          onClick={() => onViewChange("map")}
          className={`rounded-full px-4 ${
            activeView === "map"
              ? "bg-black text-white hover:bg-gray-800 hover:text-white"
              : "text-gray-600 hover:bg-gray-100 hover:text-black"
          }`}
        >
          <MapPin className="w-4 h-4 mr-1" />
          Map
        </Button>
        <Button
          variant={activeView === "itinerary" ? "default" : "ghost"}
          size="sm"
          onClick={() => onViewChange("itinerary")}
          className={`rounded-full px-4 ${
            activeView === "itinerary"
              ? "bg-black text-white hover:bg-gray-800 hover:text-white"
              : "text-gray-600 hover:bg-gray-100 hover:text-black"
          }`}
        >
          <Calendar className="w-4 h-4 mr-1" />
          Itinerary
        </Button>
      </div>

      <Button
        variant="ghost"
        size="sm"
        onClick={onClose}
        className="absolute top-4 right-4 z-10 bg-white/90 backdrop-blur-sm rounded-full shadow-lg hover:bg-gray-100 w-8 h-8 p-0 flex items-center justify-center"
      >
        <ChevronRight className="w-4 h-4" />
      </Button>

      {/* Content */}
      <div className="flex-1 overflow-hidden h-full">
        <div className={`h-full ${activeView === "map" ? "block" : "hidden"}`}>
          {mapData && (
            <MapView
              mapData={mapData}
              savedIds={savedIds}
              itinerary={itinerary}
              itineraryPoiIds={itineraryPoiIds}
              onToggleSave={onToggleSave}
              onAddPoi={onAddPoi}
              onReplan={onReplan}
              isVisible={activeView === "map"}
            />
          )}
        </div>

        {activeView === "itinerary" && (
          <div className="h-full overflow-y-auto">
            <ItineraryPanel itinerary={itinerary} trip={trip} />
          </div>
        )}
        
      </div>
    </div>
  )
}
