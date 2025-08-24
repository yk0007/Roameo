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
          <MapView
            mapData={mapData}
            savedIds={savedIds}
            itinerary={itinerary}
            onToggleSave={onToggleSave}
            onAddPoi={onAddPoi}
            onReplan={onReplan}
            isVisible={activeView === "map"}
          />
        </div>

        {activeView === "itinerary" && (
          <div className="h-full flex flex-col">
            {/* Fixed header */}
            <div className="flex items-center justify-between p-4 pt-16 pb-4 bg-white border-b border-gray-100">
              <h3 className="font-semibold">Itinerary</h3>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">{itinerary?.days ?? 0} days</span>
                <ShareButton tripId={trip.id} tripTitle={trip.title} itinerary={itinerary} />
              </div>
            </div>

            {/* Scrollable content */}
            <div className="flex-1 overflow-y-auto">
              {!itinerary && (
                <div className="text-sm text-gray-500 p-4">No itinerary yet. Tell Roameo your origin, destination and days.</div>
              )}

              {itinerary?.daysPlan && itinerary.daysPlan.length > 0 && itinerary.daysPlan.map((day, dayIndex) => {
                if (!day || typeof day.day !== 'number') return null;
                
                return (
                  <div key={day.day} className="relative">
                    <div className="flex items-center gap-3 p-3 bg-white/95 backdrop-blur-md border-b border-gray-100 shadow-sm sticky top-0 z-10">
                      <span className="text-sm font-bold bg-zinc-800 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">{day.day}</span>
                      <h4 className="font-semibold text-md italic">{day.title || `Day ${day.day}`}</h4>
                    </div>

                    <div className="space-y-3 pl-4 border-l-2 border-zinc-200 ml-4 p-4">
                    {day.activities?.length > 0 && day.activities.map((activity, index) => {
                      if (!activity || !activity.name) return null;
                      
                      return (
                        <div
                          key={index}
                          className="flex gap-4 p-2 rounded-lg hover:bg-zinc-50 relative"
                        >
                          <div className="absolute left-[-26px] top-5 w-3 h-3 bg-zinc-300 rounded-full border-4 border-white"></div>
                          <div className="w-12 h-12 rounded-lg overflow-hidden bg-gray-200 flex items-center justify-center">
                            {activity.photoUrl ? (
                              <CachedImage
                                src={activity.photoUrl}
                                alt={activity.name}
                                className="w-full h-full object-cover"
                              />
                            ) : (
                              <span className="text-xs text-gray-600">⛳</span>
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between">
                              <div>
                                <h5 className="font-medium text-sm">{activity.name}</h5>
                                {activity.start && activity.end && (
                                  <p className="text-xs text-gray-500">{activity.start} - {activity.end}</p>
                                )}
                                {activity.location && <p className="text-xs text-gray-500">{activity.location}</p>}
                              </div>
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <Button variant="outline" size="sm" className="text-xs border-0 bg-white shadow-md hover:shadow-lg transition-shadow rounded-xl">
                                    Details
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end" className="w-48 z-[10001] border-0 shadow-xl rounded-xl bg-white/95 backdrop-blur-md">
                                  <DropdownMenuItem>View on Map</DropdownMenuItem>
                                  <DropdownMenuItem>Get Directions</DropdownMenuItem>
                                  <DropdownMenuItem>More Info</DropdownMenuItem>
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </div>
                          </div>
                        </div>
                      );
                    })}

                    {day.accommodation && day.accommodation.name && (
                      <div className="flex items-center gap-4 p-2 relative"> 
                        <div className="absolute left-[-26px] top-5 w-3 h-3 bg-zinc-300 rounded-full border-4 border-white"></div>
                        <div className="w-8 h-8 bg-gray-300 rounded flex items-center justify-center">
                          <span className="text-xs">🏨</span>
                        </div>
                        <div className="flex-1">
                          <h5 className="font-medium text-sm">{day.accommodation.name}</h5>
                          {day.accommodation.checkIn && (
                            <p className="text-xs text-gray-500">{day.accommodation.checkIn}</p>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
            </div>
          </div>
        )}
        
      </div>
    </div>
  )
}
