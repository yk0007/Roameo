"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import {
  Clock,
  ChevronDown,
  ChevronUp,
  Plus,
  MoreHorizontal,
  Trash2,
  Calendar,
  Bookmark,
  RotateCcw,
  RotateCw,
  MapPin,
  Bed,
  Heart,
  Hotel,
  Star,
  RefreshCw,
  X,
  Check,
} from "lucide-react"
import { OptimizedPoiImage } from "@/components/optimized-poi-image"
import { PoiDetailModal } from "@/components/poi-detail-modal"
import { ExpandablePoiCard } from "@/components/expandable-poi-card"
import { Itinerary, ItineraryDay, Activity, POI } from "@/lib/types"

interface ItineraryPanelProps {
  itinerary?: Itinerary
  onPOISelect?: (pois: any[]) => void
}

export function ItineraryPanel({ itinerary, onPOISelect }: ItineraryPanelProps) {
  const [activeTab, setActiveTab] = useState("Itinerary")
  const [expandedDays, setExpandedDays] = useState<number[]>([1])
  const [distancesEnabled, setDistancesEnabled] = useState(true)
  const [selectedPoi, setSelectedPoi] = useState<POI | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const tabs = ["Itinerary", "Calendar", "Bookings"]

  const currentItinerary = itinerary

  const toggleDay = (day: number) => {
    setExpandedDays((prev) => (prev.includes(day) ? prev.filter((d) => d !== day) : [...prev, day]))
  }

  if (activeTab === "Calendar") {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-center gap-2 mb-6">
          {tabs.map((tab) => (
            <Button
              key={tab}
              variant={activeTab === tab ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab(tab)}
              className={`backdrop-blur-md border rounded-full px-4 ${
                activeTab === tab
                  ? "bg-black/80 text-white border-black/20"
                  : "bg-white/80 text-gray-700 border-white/30 hover:bg-white/90"
              }`}
            >
              {tab === "Calendar" && <Calendar className="w-4 h-4 mr-2 border border-current rounded p-0.5" />}
              {tab === "Bookings" && <Bookmark className="w-4 h-4 mr-2 border border-current rounded p-0.5" />}
              {tab}
            </Button>
          ))}
        </div>

        <div className="text-center text-gray-500">
          <Calendar className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Calendar view coming soon</p>
        </div>
      </div>
    )
  }

  if (activeTab === "Bookings") {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-center gap-2 mb-6">
          {tabs.map((tab) => (
            <Button
              key={tab}
              variant={activeTab === tab ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab(tab)}
              className={`backdrop-blur-md border rounded-full px-4 ${
                activeTab === tab
                  ? "bg-black/80 text-white border-black/20"
                  : "bg-white/80 text-gray-700 border-white/30 hover:bg-white/90"
              }`}
            >
              {tab === "Calendar" && <Calendar className="w-4 h-4 mr-2 border border-current rounded p-0.5" />}
              {tab === "Bookings" && <Bookmark className="w-4 h-4 mr-2 border border-current rounded p-0.5" />}
              {tab}
            </Button>
          ))}
        </div>

        <div className="text-center text-gray-500">
          <Bookmark className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No bookings yet</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-center gap-2 mb-6">
        {tabs.map((tab) => (
          <Button
            key={tab}
            variant={activeTab === tab ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab(tab)}
            className={`backdrop-blur-md border-0 rounded-full px-4 transition-all duration-200 ${
              activeTab === tab
                ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-sm"
                : "bg-white/80 text-gray-700 hover:bg-white/90 shadow-sm"
            }`}
          >
            {tab === "Calendar" && <Calendar className="w-4 h-4 mr-2 border border-current rounded p-0.5" />}
            {tab === "Bookings" && <Bookmark className="w-4 h-4 mr-2 border border-current rounded p-0.5" />}
            {tab}
          </Button>
        ))}
      </div>

      {/* Header with controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Itinerary</h2>
          {currentItinerary && <p className="text-sm text-gray-600">{currentItinerary.days} days</p>}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Distances</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDistancesEnabled(!distancesEnabled)}
              className={`w-12 h-6 rounded-full p-0 transition-all duration-200 ${
                distancesEnabled 
                  ? "bg-gradient-to-r from-blue-500 to-purple-600" 
                  : "bg-gray-200 hover:bg-gray-300"
              }`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full transition-all duration-200 ${
                  distancesEnabled ? "translate-x-3" : "translate-x-1"
                }`}
              />
            </Button>
          </div>
          <div className="flex gap-1">
            <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
              <RotateCcw className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
              <RotateCw className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Daily Plans */}
      <div className="space-y-4">
        {currentItinerary?.daysPlan?.map((day: ItineraryDay) => (
          <div key={day.day} className="space-y-3">
            {/* Day Header */}
            <div className="flex items-center justify-between">
              <Button
                variant="ghost"
                onClick={() => toggleDay(day.day)}
                className="flex items-center gap-2 p-0 h-auto hover:bg-transparent"
              >
                {expandedDays.includes(day.day) ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronUp className="w-4 h-4" />
                )}
                <span className="font-semibold">Day {day.day}</span>
                <span className="text-green-600">🌲</span>
                <span className="font-medium">{day.title}</span>
                <span className="text-sm text-gray-500">{day.date}</span>
              </Button>
              <div className="flex gap-1">
                <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
                  <MoreHorizontal className="w-4 h-4" />
                </Button>
                <Button variant="ghost" size="sm" className="w-8 h-8 p-0">
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Day Activities */}
            {expandedDays.includes(day.day) && (
              <div className="space-y-3 ml-6">
                {day.activities.map((activity: Activity, index: number) => {
                  // Create a POI object from activity data
                  const activityPoi: POI = {
                    id: activity.poiId || `activity-${index}`,
                    name: activity.name,
                    address: activity.location || '',
                    photoUrl: activity.photoUrl,
                    type: 'Activity',
                    rating: 4.5,
                    description: `${activity.name} scheduled from ${activity.start} to ${activity.end}. Duration: ${activity.start} - ${activity.end}`
                  }
                  
                  return (
                    <ExpandablePoiCard
                      key={index}
                      poi={activityPoi}
                      isInItinerary={true}
                      className="bg-white/90 backdrop-blur-sm border-0 rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-all duration-200"
                    />
                  )
                })}

                {/* Accommodation */}
                {day.accommodation && (() => {
                  // Create a POI object from accommodation data
                  const accommodationPoi: POI = {
                    id: day.accommodation.poiId || `accommodation-${day.day}`,
                    name: day.accommodation.name || '',
                    address: '',
                    photoUrl: day.accommodation.photoUrl,
                    type: 'Hotel',
                    rating: 4.5,
                    description: `Accommodation for ${day.accommodation.nights} night(s). Check-in: ${day.accommodation.checkIn || 'TBD'}`
                  }
                  
                  return (
                    <ExpandablePoiCard
                      poi={accommodationPoi}
                      isInItinerary={true}
                      className="bg-white/90 backdrop-blur-sm border-0 rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-all duration-200"
                    />
                  )
                })()}

                {/* Add Button */}
                <button className="shadow-[inset_0_0_0_2px_#10b981] text-emerald-500 px-8 py-3 rounded-full tracking-wider uppercase font-bold bg-transparent hover:bg-emerald-500 hover:text-white dark:text-emerald-400 transition duration-200 w-full flex items-center justify-center gap-2">
                  <Plus className="w-4 h-4" />
                  Add Activity
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
      
      {/* POI Detail Modal */}
      <PoiDetailModal
        poi={selectedPoi}
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false)
          setSelectedPoi(null)
        }}
        isSaved={false}
        isItineraryItem={true}
        onToggleSave={() => {}}
        onAddPoi={() => {}}
        onReplan={() => {}}
      />
    </div>
  )
}
