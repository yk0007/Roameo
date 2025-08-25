"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { ShareButton } from "./share-button"
import { Itinerary, POI, Activity } from "@/lib/types"
import { SearchCard } from "./search-card"
import { ExpandableCard } from "./ui/expandable-card"
import Image from "next/image"

interface ItineraryPanelProps {
  itinerary?: Itinerary
  trip: any
  onPOISelect?: (pois: any[]) => void
}


export function ItineraryPanel({ itinerary, trip, onPOISelect }: ItineraryPanelProps) {
  const [selectedActivity, setSelectedActivity] = useState<Activity | null>(null)

  const activityToPoi = (activity: Activity): POI => {
    return {
      id: activity.id || `${activity.name}-${activity.location}`,
      name: activity.name,
      photoUrl: activity.photoUrl,
      type: 'attraction',
      rating: activity.rating,
      address: activity.location,
      lat: activity.lat ?? 0,
      lng: activity.lng ?? 0,
    }
  }
  return (
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
                    className="flex gap-4 p-3 rounded-lg hover:bg-zinc-50 relative bg-white shadow-lg border border-gray-100"
                  >
                    <div className="absolute left-[-26px] top-5 w-3 h-3 bg-zinc-300 rounded-full border-4 border-white"></div>
                    <div className="w-16 h-16 rounded-lg overflow-hidden bg-gray-200 flex items-center justify-center">
                      {activity.photoUrl ? (
                        <Image
                          src={activity.photoUrl}
                          alt={activity.name}
                          width={64}
                          height={64}
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
                        <ExpandableCard 
                          cards={[{
                            title: activity.name,
                            description: activity.location || 'Activity location',
                            src: activity.photoUrl || '/placeholder-activity.jpg',
                            ctaText: 'View Details',
                            content: () => (
                              <div className="space-y-4">
                                {activity.description && (
                                  <p className="text-sm text-gray-600">{activity.description}</p>
                                )}
                                {activity.start && activity.end && (
                                  <div className="flex items-center gap-2">
                                    <span className="text-sm font-medium">Time:</span>
                                    <span className="text-sm text-gray-600">{activity.start} - {activity.end}</span>
                                  </div>
                                )}
                                {activity.rating && (
                                  <div className="flex items-center gap-2">
                                    <span className="text-sm font-medium">Rating:</span>
                                    <span className="text-sm text-gray-600">{activity.rating}/5</span>
                                  </div>
                                )}
                                {activity.location && (
                                  <div className="flex items-center gap-2">
                                    <span className="text-sm font-medium">Location:</span>
                                    <span className="text-sm text-gray-600">{activity.location}</span>
                                  </div>
                                )}
                              </div>
                            )
                          }]}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
      </div>
    </div>
  )
}
