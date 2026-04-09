"use client"
import { useEffect } from "react"
import { Button } from "@/components/ui/button"
import { MapPin, Calendar, ChevronRight } from "lucide-react"
import MapView from "./map-view"
import { ItineraryPanel } from "./itinerary-panel"
import type { Itinerary, POI, SessionPlanningState } from "@/lib/types"

interface RightPanelProps {
  activeView: "map" | "itinerary"
  onViewChange: (view: "map" | "itinerary") => void
  trip: any
  itinerary?: Itinerary
  planVersionKey?: string
  mapData?: { pois: any[]; routes: Array<{ from: [number, number]; to: [number, number]; polyline?: string }> }
  onClose: () => void
  planningState?: SessionPlanningState
  savedIds?: Set<string>
  itineraryPoiIds?: Set<string>
  onToggleSave?: (poi: POI, nextSaved: boolean) => void
  onAddPoi?: (poi: POI) => void
  onReplan?: (poi: POI) => void
  hasBillingError?: boolean
}

export function RightPanel({
  activeView,
  onViewChange,
  trip,
  itinerary,
  planVersionKey = "no-plan",
  mapData,
  onClose,
  planningState,
  savedIds,
  itineraryPoiIds,
  onToggleSave,
  onAddPoi,
  onReplan,
  hasBillingError = false,
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
    <div className="relative flex h-full flex-col overflow-hidden bg-transparent lg:rounded-l-[24px]">
      <div className="pointer-events-none absolute left-1/2 top-4 z-50 flex -translate-x-1/2 items-center gap-2">
        <div className="pointer-events-auto flex items-center gap-0.5 rounded-full border border-white/40 bg-white/50 p-[5px] shadow-[0_8px_32px_rgba(15,23,42,0.12),0_2px_8px_rgba(15,23,42,0.06)] backdrop-blur-2xl">
          <Button
            variant={activeView === "map" ? "default" : "ghost"}
            size="sm"
            onClick={() => onViewChange("map")}
            className={`h-[31px] rounded-full px-4 text-[13px] ${
              activeView === "map"
                ? "bg-black text-white hover:bg-gray-800 hover:text-white"
                : "text-gray-500 hover:bg-transparent hover:text-black"
            }`}
          >
            <MapPin className="mr-1 w-4 h-4" />
            Map
          </Button>
          <Button
            variant={activeView === "itinerary" ? "default" : "ghost"}
            size="sm"
            onClick={() => onViewChange("itinerary")}
            className={`h-[31px] rounded-full px-4 text-[13px] ${
              activeView === "itinerary"
                ? "bg-black text-white hover:bg-gray-800 hover:text-white"
                : "text-gray-500 hover:bg-transparent hover:text-black"
            }`}
          >
            <Calendar className="mr-1 w-4 h-4" />
            Itinerary
          </Button>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          className="pointer-events-auto h-9 w-9 rounded-full border border-white/40 bg-white/50 p-0 shadow-[0_8px_32px_rgba(15,23,42,0.12),0_2px_8px_rgba(15,23,42,0.06)] backdrop-blur-2xl hover:bg-white/70"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {planningState?.status === "unavailable" ? (
        <div className="absolute left-4 top-16 z-50 rounded-full border border-amber-200 bg-amber-50/95 px-3 py-1 text-xs font-medium text-amber-800 shadow-sm">
          AI unavailable
        </div>
      ) : null}

      <div className="relative h-full flex-1 overflow-hidden rounded-l-[24px] bg-transparent">
        <div className="h-full">
          <MapView
            key={`map-${planVersionKey}`}
            mapData={mapData || { pois: [], routes: [] }}
            savedIds={savedIds}
            itinerary={itinerary}
            itineraryPoiIds={itineraryPoiIds}
            onToggleSave={onToggleSave}
            onAddPoi={onAddPoi}
            onReplan={onReplan}
            isVisible={activeView === "map"}
            hasBillingError={hasBillingError}
          />
        </div>

        {activeView === "itinerary" && (
          <div className="absolute inset-0 z-40 overflow-y-auto bg-white/40 backdrop-blur-xl border-l border-white/20">
            <ItineraryPanel 
              key={`itinerary-${planVersionKey}`}
              itinerary={itinerary} 
              trip={trip}
              savedIds={savedIds}
              onToggleSave={onToggleSave}
              onAddPoi={onAddPoi}
              onReplan={onReplan}
              planningStatus={planningState?.status === "unavailable" ? "AI unavailable. Existing itinerary remains visible until planning recovers." : undefined}
            />
          </div>
        )}
        
      </div>
    </div>
  )
}
