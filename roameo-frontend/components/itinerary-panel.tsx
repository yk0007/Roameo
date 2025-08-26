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
  RotateCcw,
  RotateCw,
  MapPin,
  Bed,
} from "lucide-react"
import { Itinerary, ItineraryDay, Activity } from "@/lib/types"
import { ShareButton } from "./share-button"

interface ItineraryPanelProps {
  itinerary?: Itinerary
  trip: any
  onPOISelect?: (pois: any[]) => void
}

export function ItineraryPanel({ itinerary, trip, onPOISelect }: ItineraryPanelProps) {
  const [expandedDays, setExpandedDays] = useState<number[]>([1])
  const [distancesEnabled, setDistancesEnabled] = useState(true)

  const currentItinerary = itinerary

  const toggleDay = (day: number) => {
    setExpandedDays((prev) => (prev.includes(day) ? prev.filter((d) => d !== day) : [...prev, day]))
  }

  return (
    <div className="p-6 space-y-6">
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
              className={`w-12 h-6 rounded-full p-0 ${distancesEnabled ? "bg-black" : "bg-gray-300"}`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full transition-transform ${
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
                {day.activities.map((activity: Activity, index: number) => (
                  <Card
                    key={index}
                    className="bg-white/80 backdrop-blur-sm border-white/30 rounded-2xl overflow-hidden"
                  >
                    <CardContent className="p-4">
                      <div className="flex gap-3">
                        {activity.photoUrl ? (
                          <img
                            src={activity.photoUrl}
                            alt={activity.name}
                            className="w-16 h-16 rounded-xl object-cover flex-shrink-0"
                          />
                        ) : (
                          <div className="w-16 h-16 rounded-xl bg-gray-100 flex items-center justify-center flex-shrink-0">
                            <MapPin className="w-6 h-6 text-gray-400" />
                          </div>
                        )}
                        <div className="flex-1 min-w-0 overflow-hidden">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <h4 className="font-semibold text-sm truncate">{activity.name}</h4>
                              <div className="flex items-center gap-1 text-xs text-gray-500">
                                <Clock className="w-3 h-3 flex-shrink-0" />
                                <span className="truncate">
                                  {activity.start} - {activity.end}
                                </span>
                              </div>
                            </div>
                            <Button variant="outline" size="sm" className="rounded-full text-xs bg-transparent flex-shrink-0">
                              Details
                            </Button>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}

                {/* Accommodation */}
                {day.accommodation && (
                  <Card className="bg-white/80 backdrop-blur-sm border-white/30 rounded-2xl overflow-hidden">
                    <CardContent className="p-4">
                      <div className="flex gap-3">
                        {day.accommodation.photoUrl ? (
                          <img
                            src={day.accommodation.photoUrl}
                            alt={day.accommodation.name}
                            className="w-16 h-16 rounded-xl object-cover flex-shrink-0"
                          />
                        ) : (
                          <div className="w-16 h-16 rounded-xl bg-gray-100 flex items-center justify-center flex-shrink-0">
                            <Bed className="w-6 h-6 text-gray-400" />
                          </div>
                        )}
                        <div className="flex-1 min-w-0 overflow-hidden">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <h4 className="font-semibold text-sm truncate">{day.accommodation.name}</h4>
                              <p className="text-xs text-gray-500 truncate">
                                {day.accommodation.checkIn && `Check-in from ${day.accommodation.checkIn}`}
                                {day.accommodation.nights && ` (${day.accommodation.nights} night)`}
                              </p>
                            </div>
                            <Button variant="outline" size="sm" className="rounded-full text-xs bg-transparent flex-shrink-0">
                              Book
                            </Button>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Add Button */}
                <Button
                  variant="outline"
                  className="w-full rounded-2xl bg-white/50 backdrop-blur-sm border-white/30 hover:bg-white/80"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Add
                </Button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
