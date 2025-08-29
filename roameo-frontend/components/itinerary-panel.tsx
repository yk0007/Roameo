"use client"

import { useState, useMemo, useCallback, useEffect, useId, useRef } from "react"
import { Button } from "@/components/ui/button"
import { ShareButton } from "./share-button"
import { CompactPoiCard } from "./poi-card"
import { Itinerary, POI, Activity } from "@/lib/types"
import Image from "next/image"
import { AnimatePresence, motion } from "framer-motion"
import { useOutsideClick } from "@/hooks/use-outside-click"
import {
  Clock,
  MapPin,
  Bed,
  Star,
  Heart,
  ExternalLink,
} from "lucide-react"

interface ItineraryPanelProps {
  itinerary?: Itinerary
  trip: any
  onPOISelect?: (pois: any[]) => void
  savedIds?: Set<string>
  onToggleSave?: (poi: POI, nextSaved: boolean) => void
  onAddPoi?: (poi: POI) => void
  onReplan?: (poi: POI) => void
  isLoading?: boolean
  planningStatus?: string
}

export function ItineraryPanel({ 
  itinerary, 
  trip, 
  onPOISelect, 
  savedIds = new Set(), 
  onToggleSave = () => {}, 
  onAddPoi = () => {}, 
  onReplan = () => {},
  isLoading = false,
  planningStatus
}: ItineraryPanelProps) {
  const [selectedActivity, setSelectedActivity] = useState<Activity | null>(null)
  const [expandedPoiCards, setExpandedPoiCards] = useState<Set<string>>(new Set())
  const [activeCard, setActiveCard] = useState<Activity | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const id = useId()

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setActiveCard(null)
      }
    }

    if (activeCard) {
      document.body.style.overflow = "hidden"
    } else {
      document.body.style.overflow = "auto"
    }

    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [activeCard])

  useOutsideClick(ref, () => setActiveCard(null))

  // Memoized function to convert activity to POI
  const activityToPoi = useCallback((activity: Activity): POI => {
    return {
      id: activity.poiId || `${activity.name}-${activity.location}`,
      name: activity.name,
      photoUrl: activity.photoUrl,
      type: 'attraction',
      rating: activity.rating,
      address: activity.location,
      lat: activity.lat ?? 0,
      lng: activity.lng ?? 0,
    }
  }, [])

  // Memoized accommodation to POI converter
  const accommodationToPoi = useCallback((accommodation: any): POI => {
    return {
      id: accommodation.poiId || `${accommodation.name}-${accommodation.location}`,
      name: accommodation.name,
      photoUrl: accommodation.photoUrl,
      type: 'stay',
      rating: 4.0, // Default rating for accommodations
      address: accommodation.location,
      lat: 0,
      lng: 0,
    }
  }, [])

  // Toggle POI card expansion
  const togglePoiCard = useCallback((poiId: string) => {
    setExpandedPoiCards(prev => {
      const newSet = new Set(prev)
      if (newSet.has(poiId)) {
        newSet.delete(poiId)
      } else {
        newSet.add(poiId)
      }
      return newSet
    })
  }, [])
  
  return (
    <div className="h-full flex flex-col relative">
      {/* Loading Overlay */}
      {(isLoading || planningStatus) && (
        <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="flex flex-col items-center space-y-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <div className="text-sm font-medium text-gray-700">
              {planningStatus || "Planning your itinerary..."}
            </div>
          </div>
        </div>
      )}
      
      {/* Expandable Card Overlay and Content */}
      <AnimatePresence>
        {activeCard && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/20 h-full w-full z-10"
          />
        )}
      </AnimatePresence>
      
      <AnimatePresence>
        {activeCard ? (
          <div className="fixed inset-0 grid place-items-center z-[100]">
            <motion.button
              key={`button-${activeCard.name}-${id}`}
              layout
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{
                opacity: 0,
                transition: {
                  duration: 0.05,
                },
              }}
              className="flex absolute top-2 right-2 lg:hidden items-center justify-center bg-white rounded-full h-6 w-6"
              onClick={() => setActiveCard(null)}
            >
              <CloseIcon />
            </motion.button>
            <motion.div
              layoutId={`card-${activeCard.name}-${id}`}
              ref={ref}
              className="w-full max-w-[500px] h-full md:h-fit md:max-h-[90%] flex flex-col bg-white dark:bg-neutral-900 sm:rounded-3xl overflow-hidden"
            >
              <motion.div layoutId={`image-${activeCard.name}-${id}`}>
                <Image
                  width={500}
                  height={320}
                  src={activeCard.photoUrl || '/placeholder.svg'}
                  alt={activeCard.name}
                  className="w-full h-80 lg:h-80 sm:rounded-tr-lg sm:rounded-tl-lg object-cover object-top"
                  priority={true}
                  quality={90}
                  placeholder="empty"
                  sizes="(max-width: 768px) 100vw, 500px"
                />
              </motion.div>

              <div>
                <div className="flex justify-between items-start p-4">
                  <div className="flex-1">
                    <motion.h3
                      layoutId={`title-${activeCard.name}-${id}`}
                      className="font-bold text-neutral-700 dark:text-neutral-200 mb-2"
                    >
                      {activeCard.name}
                    </motion.h3>
                    <motion.p
                      layoutId={`description-${activeCard.location}-${id}`}
                      className="text-neutral-600 dark:text-neutral-400 mb-3"
                    >
                      {activeCard.location}
                    </motion.p>
                    
                    {/* Time Information */}
                    {activeCard.start && activeCard.end && (
                      <div className="flex items-center gap-2 text-sm text-gray-600 mb-3">
                        <Clock className="w-4 h-4" />
                        <span>{activeCard.start} - {activeCard.end}</span>
                      </div>
                    )}
                    
                    {/* Rating */}
                    {activeCard.rating && (
                      <div className="flex items-center gap-1 mb-3">
                        <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                        <span className="text-sm font-medium">{activeCard.rating}</span>
                      </div>
                    )}
                  </div>

                  <div className="flex flex-col gap-2 ml-4">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const poi = activityToPoi(activeCard)
                        onToggleSave(poi, !savedIds.has(poi.id))
                      }}
                      className="rounded-full"
                    >
                      <Heart className={`w-4 h-4 ${savedIds.has(activityToPoi(activeCard).id) ? 'fill-red-500 text-red-500' : ''}`} />
                    </Button>
                    {activeCard.lat && activeCard.lng && (
                      <Button size="sm" variant="outline" className="rounded-full">
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    )}
                  </div>
                </div>
                <div className="pt-4 relative px-4">
                  <motion.div
                    layout
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-neutral-600 text-xs md:text-sm lg:text-base h-40 md:h-fit pb-10 flex flex-col items-start gap-4 overflow-auto dark:text-neutral-400 [mask:linear-gradient(to_bottom,white,white,transparent)] [scrollbar-width:none] [-ms-overflow-style:none] [-webkit-overflow-scrolling:touch]"
                  >
                    {activeCard.description && (
                      <div>
                        <h4 className="font-semibold mb-2">Description</h4>
                        <p>{activeCard.description}</p>
                      </div>
                    )}
                    {activeCard.distanceKm && (
                      <div>
                        <h4 className="font-semibold mb-2">Distance</h4>
                        <p>{activeCard.distanceKm}km from previous location</p>
                      </div>
                    )}
                    <div>
                      <h4 className="font-semibold mb-2">Location</h4>
                      <p>{activeCard.location}</p>
                      {activeCard.lat && activeCard.lng && (
                        <p className="text-xs text-gray-500 mt-1">
                          Coordinates: {activeCard.lat}, {activeCard.lng}
                        </p>
                      )}
                    </div>
                  </motion.div>
                </div>
              </div>
            </motion.div>
          </div>
        ) : null}
      </AnimatePresence>
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
                  
                  const poi = activityToPoi(activity)
                  const isExpanded = expandedPoiCards.has(poi.id)
                  
                  return (
                    <div key={index} className="relative">
                      <div className="absolute left-[-26px] top-5 w-3 h-3 bg-zinc-300 rounded-full border-4 border-white z-10"></div>
                      
                      {/* Expandable Activity Card */}
                      <motion.div
                        layoutId={`card-${activity.name}-${id}`}
                        onClick={() => setActiveCard(activity)}
                        className="p-4 flex items-center gap-4 hover:bg-neutral-50 dark:hover:bg-neutral-800 rounded-xl cursor-pointer border border-gray-200 mb-3 transition-all"
                      >
                        <motion.div layoutId={`image-${activity.name}-${id}`}>
                          <Image
                            width={60}
                            height={60}
                            src={activity.photoUrl || '/placeholder.svg'}
                            alt={activity.name}
                            className="h-14 w-14 rounded-lg object-cover object-top"
                            quality={90}
                            priority={true}
                            placeholder="empty"
                            sizes="60px"
                          />
                        </motion.div>
                        <div className="flex-1">
                          <motion.h3
                            layoutId={`title-${activity.name}-${id}`}
                            className="font-medium text-neutral-800 dark:text-neutral-200"
                          >
                            {activity.name}
                          </motion.h3>
                          <motion.p
                            layoutId={`description-${activity.location}-${id}`}
                            className="text-neutral-600 dark:text-neutral-400 text-sm"
                          >
                            {activity.location}
                          </motion.p>
                          {activity.start && activity.end && (
                            <div className="flex items-center gap-2 text-xs text-gray-500 mt-1">
                              <Clock className="w-3 h-3" />
                              <span>{activity.start} - {activity.end}</span>
                            </div>
                          )}
                        </div>
                        <motion.button
                          layoutId={`button-${activity.name}-${id}`}
                          className="px-3 py-1 text-xs rounded-full font-bold bg-gray-100 hover:bg-blue-500 hover:text-white text-black transition-colors"
                        >
                          View Details
                        </motion.button>
                      </motion.div>
                    </div>
                  );
                })}
                
                {/* Accommodation info text only - no separate card */}
                {day.accommodation && (
                  <div className="text-sm text-gray-600 mt-2 px-3 py-2 bg-blue-50 rounded-lg border border-blue-200">
                    🏨 <span className="font-medium">{day.accommodation.name}</span>
                    {day.accommodation.checkIn && (
                      <span className="ml-2 text-xs">• Check-in: {day.accommodation.checkIn}</span>
                    )}
                    {day.accommodation.nights && (
                      <span className="ml-2 text-xs">• {day.accommodation.nights} night{day.accommodation.nights > 1 ? 's' : ''}</span>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  )
}

export const CloseIcon = () => {
  return (
    <motion.svg
      initial={{
        opacity: 0,
      }}
      animate={{
        opacity: 1,
      }}
      exit={{
        opacity: 0,
        transition: {
          duration: 0.05,
        },
      }}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-4 w-4 text-black"
    >
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path d="M18 6l-12 12" />
      <path d="M6 6l12 12" />
    </motion.svg>
  );
};
