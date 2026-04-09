"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import { AnimatePresence, motion } from "framer-motion";
import { Clock, ExternalLink, Heart, MapPin, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ShareButton } from "./share-button";
import { PoiTypeIcon } from "./poi-type-icon";
import { CachedImage } from "./cached-image";
import { useOutsideClick } from "@/hooks/use-outside-click";
import { resolvePoiImageUrl } from "@/lib/poi-image-url";
import type { Activity, Itinerary, POI } from "@/lib/types";

interface ItineraryPanelProps {
  itinerary?: Itinerary;
  trip: any;
  pois?: POI[];
  onPOISelect?: (pois: any[]) => void;
  savedIds?: Set<string>;
  onToggleSave?: (poi: POI, nextSaved: boolean) => void;
  onAddPoi?: (poi: POI) => void;
  onReplan?: (poi: POI) => void;
  isLoading?: boolean;
  planningStatus?: string;
}

function normalizeMatchText(value?: string) {
  return (value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function resolveActivityImage(activity: Activity, pois?: POI[]) {
  const direct = resolvePoiImageUrl(activity.photoUrl);
  if (direct) {
    return direct;
  }

  const candidates = pois || [];
  const byId = activity.poiId
    ? candidates.find((poi) => poi.id === activity.poiId)
    : undefined;
  const byIdImage = resolvePoiImageUrl(byId?.photoUrl);
  if (byIdImage) {
    return byIdImage;
  }

  const activityName = normalizeMatchText(activity.name);
  const activityLocation = normalizeMatchText(activity.location);

  const byName = candidates.find((poi) => {
    const poiName = normalizeMatchText(poi.name);
    return (
      poiName &&
      activityName &&
      (activityName.includes(poiName) || poiName.includes(activityName))
    );
  });
  const byNameImage = resolvePoiImageUrl(byName?.photoUrl);
  if (byNameImage) {
    return byNameImage;
  }

  const byLocation = candidates.find((poi) => {
    const poiAddress = normalizeMatchText(poi.address);
    return (
      poiAddress &&
      activityLocation &&
      (poiAddress.includes(activityLocation) || activityLocation.includes(poiAddress))
    );
  });

  return resolvePoiImageUrl(byLocation?.photoUrl);
}

function activityToPoi(activity: Activity): POI {
  const lowerName = activity.name.toLowerCase();
  const type =
    lowerName.includes("hotel") ||
    lowerName.includes("resort") ||
    lowerName.includes("stay")
      ? "stay"
      : lowerName.includes("restaurant") ||
          lowerName.includes("cafe") ||
          lowerName.includes("dinner") ||
          lowerName.includes("lunch")
        ? "restaurant"
        : "attraction";
  return {
    id: activity.poiId || `${activity.name}-${activity.location}`,
    name: activity.name,
    photoUrl: activity.photoUrl,
    type,
    rating: activity.rating,
    address: activity.location,
    lat: activity.lat ?? 0,
    lng: activity.lng ?? 0,
    openingHours: [],
    source: "manual",
    tags: [],
  };
}

function CloseIcon() {
  return (
    <motion.svg
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.05 } }}
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
}

function ActivityModal({
  activity,
  savedIds,
  onClose,
  onToggleSave,
}: {
  activity: Activity | null;
  savedIds: Set<string>;
  onClose: () => void;
  onToggleSave: (poi: POI, nextSaved: boolean) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useOutsideClick(ref, onClose);

  useEffect(() => {
    if (!activity) {
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [activity, onClose]);

  if (!activity) {
    return null;
  }

  const poi = activityToPoi(activity);
  const isSaved = savedIds.has(poi.id);
  const imageUrl = resolvePoiImageUrl(activity.photoUrl) || "/placeholder.svg";

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[100] grid place-items-center p-6">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/30 backdrop-blur-[2px]"
        />
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 24, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 16, scale: 0.985 }}
          transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
          className="relative z-10 flex h-full w-full max-w-[520px] flex-col overflow-hidden rounded-[28px] bg-white shadow-[0_24px_80px_rgba(15,23,42,0.18)] md:h-auto md:max-h-[90%]"
        >
          <button
            className="absolute right-4 top-4 z-20 flex h-10 w-10 items-center justify-center rounded-full bg-white/92 text-black shadow-sm ring-1 ring-black/10 backdrop-blur-sm"
            onClick={onClose}
          >
            <CloseIcon />
          </button>

          <Image
            width={520}
            height={320}
            src={imageUrl}
            alt={activity.name}
            className="h-72 w-full object-cover object-top md:h-80"
            priority
            quality={90}
            sizes="(max-width: 768px) 100vw, 520px"
          />

          <div className="px-4 pb-6 pt-5">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <h3 className="mb-2 font-bold text-neutral-800">{activity.name}</h3>
                <p className="mb-3 text-neutral-600">{activity.location}</p>

                {activity.start && activity.end && (
                  <div className="mb-3 flex items-center gap-2 text-sm text-gray-600">
                    <Clock className="h-4 w-4" />
                    <span>
                      {activity.start} - {activity.end}
                    </span>
                  </div>
                )}

                {activity.rating && (
                  <div className="mb-3 flex items-center gap-1">
                    <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                    <span className="text-sm font-medium">{activity.rating}</span>
                  </div>
                )}
              </div>

              <div className="ml-4 flex flex-col gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => onToggleSave(poi, !isSaved)}
                  className="rounded-full"
                >
                  <Heart className={`h-4 w-4 ${isSaved ? "fill-red-500 text-red-500" : ""}`} />
                </Button>
                {activity.lat && activity.lng && (
                  <Button size="sm" variant="outline" className="rounded-full">
                    <ExternalLink className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>

            <div className="mt-4 h-40 overflow-auto pb-6 text-sm text-neutral-600 [mask:linear-gradient(to_bottom,white,white,transparent)] [-ms-overflow-style:none] [-webkit-overflow-scrolling:touch] [scrollbar-width:none] md:h-fit">
              {activity.description && (
                <div className="mb-4">
                  <h4 className="mb-2 font-semibold">Description</h4>
                  <p>{activity.description}</p>
                </div>
              )}
              {activity.distanceKm && (
                <div className="mb-4">
                  <h4 className="mb-2 font-semibold">Distance</h4>
                  <p>{activity.distanceKm}km from previous location</p>
                </div>
              )}
              <div>
                <h4 className="mb-2 font-semibold">Location</h4>
                <p>{activity.location}</p>
                {activity.lat && activity.lng && (
                  <p className="mt-1 text-xs text-gray-500">
                    Coordinates: {activity.lat}, {activity.lng}
                  </p>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}

export function ItineraryPanel({
  itinerary,
  trip,
  pois,
  savedIds = new Set(),
  onToggleSave = () => {},
  isLoading = false,
  planningStatus,
}: ItineraryPanelProps) {
  const [activeCard, setActiveCard] = useState<Activity | null>(null);
  const [activeDestinationTab, setActiveDestinationTab] = useState("");

  useEffect(() => {
    if (itinerary?.destinationSegments?.length) {
      setActiveDestinationTab(itinerary.destinationSegments[0].destination);
    } else if (itinerary?.destination) {
      setActiveDestinationTab(itinerary.destination);
    }
  }, [itinerary?.destination, itinerary?.destinationSegments]);

  const getActiveDays = useCallback(() => {
    if (!itinerary?.daysPlan) {
      return [];
    }

    if (!itinerary.destinationSegments || itinerary.destinationSegments.length <= 1) {
      return itinerary.daysPlan;
    }

    const activeSegment = itinerary.destinationSegments.find(
      (segment) => segment.destination === activeDestinationTab,
    );

    if (!activeSegment) {
      return [];
    }

    return itinerary.daysPlan.filter(
      (day) => day.day >= activeSegment.startDay && day.day <= activeSegment.endDay,
    );
  }, [activeDestinationTab, itinerary]);

  return (
    <div className="relative flex h-full flex-col">
      {isLoading && !itinerary?.daysPlan?.length && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-white/80 backdrop-blur-sm">
          <div className="flex flex-col items-center space-y-4">
            <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-blue-600" />
            <div className="text-sm font-medium text-gray-700">
              Planning your itinerary...
            </div>
          </div>
        </div>
      )}

      <ActivityModal
        activity={activeCard}
        savedIds={savedIds}
        onClose={() => setActiveCard(null)}
        onToggleSave={onToggleSave}
      />

      <div className="border-b border-white/20 bg-transparent shrink-0">
        <div className="flex flex-col gap-2 px-5 pb-5 pt-[88px] text-black">
          <div className="flex items-center justify-between">
            <h3 className="text-2xl font-bold tracking-tight">Itinerary</h3>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-slate-600">{itinerary?.days ?? 0} days</span>
              <ShareButton tripId={trip.id} tripTitle={trip.title} itinerary={itinerary} />
            </div>
          </div>
        </div>

        {itinerary?.destinationSegments && itinerary.destinationSegments.length > 1 && (
          <div className="flex border-b border-white/20 bg-transparent">
            {itinerary.destinationSegments.map((segment) => (
              <button
                key={segment.destination}
                onClick={() => setActiveDestinationTab(segment.destination)}
                className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                  activeDestinationTab === segment.destination
                    ? "border-b-2 border-blue-600 bg-white/40 text-blue-800 font-bold"
                    : "text-slate-800 hover:bg-white/20 hover:text-black"
                }`}
              >
                <div className="flex flex-col items-center">
                  <span>{segment.destination}</span>
                  <span className="mt-1 text-xs text-gray-500">{segment.days} days</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        {!itinerary && (
          <div className="p-4 text-sm text-gray-500">
            No itinerary yet. Tell Roameo your origin, destination and days.
          </div>
        )}

        {itinerary?.daysPlan &&
          itinerary.daysPlan.length > 0 &&
          getActiveDays().map((day) => (
            <div key={day.day} className="relative">
              <div className="sticky top-0 z-20 flex items-center gap-3 border-b border-white/20 bg-white/70 p-3.5 shadow-sm backdrop-blur-xl">
                <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-zinc-900 text-sm font-bold text-white shadow-sm">
                  {day.day}
                </span>
                <h4 className="text-md font-bold text-slate-800">
                  {day.title || `Day ${day.day}`}
                </h4>
              </div>

              <div className="ml-4 space-y-3 border-l-[2px] border-dashed border-blue-400/40 p-4 pl-6">
                {day.activities?.map((activity, index) => (
                  <div key={`${activity.name}-${index}`} className="relative">
                    <div className="absolute left-[-35px] top-[32px] z-10 flex h-[22px] w-[22px] items-center justify-center rounded-full border-[3px] border-white/80 bg-blue-500 shadow-sm backdrop-blur-md">
                      <div className="h-1.5 w-1.5 rounded-full bg-white shadow-sm" />
                    </div>

                    <motion.div
                      onClick={() => setActiveCard(activity)}
                      whileHover={{ y: -1 }}
                      transition={{ duration: 0.16, ease: "easeOut" }}
                      className="mb-4 flex cursor-pointer items-center gap-4 rounded-2xl border border-white/60 bg-white/50 px-4 py-4 shadow-[0_8px_20px_rgba(15,23,42,0.04)] backdrop-blur-md transition-[border-color,box-shadow,background-color] hover:border-white/80 hover:bg-white/70 hover:shadow-[0_12px_26px_rgba(15,23,42,0.06)]"
                    >
                      {resolveActivityImage(activity, pois) ? (
                        <div className="relative h-14 w-14 shrink-0 overflow-hidden rounded-xl border border-white/40 shadow-sm">
                          <CachedImage
                            src={resolveActivityImage(activity, pois)}
                            alt={activity.name}
                            className="h-14 w-14 object-cover object-top"
                            quality={90}
                            priority
                          />
                        </div>
                      ) : (
                        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl border border-white/60 bg-blue-50/60 text-blue-500 shadow-sm backdrop-blur-md">
                          <PoiTypeIcon
                            poi={activityToPoi(activity)}
                            className="h-6 w-6 stroke-[1.5]"
                          />
                        </div>
                      )}
                      <div className="flex-1">
                        <h3 className="font-medium text-neutral-800">{activity.name}</h3>
                        <p className="text-sm text-neutral-600">{activity.location}</p>
                        {activity.start && activity.end && (
                          <div className="mt-1 flex items-center gap-2 text-xs text-gray-500">
                            <Clock className="h-3 w-3" />
                            <span>
                              {activity.start} - {activity.end}
                            </span>
                          </div>
                        )}
                      </div>
                      <button className="rounded-full border border-slate-200 bg-white px-3.5 py-1.5 text-xs font-semibold text-slate-700 shadow-sm transition-colors hover:border-slate-300 hover:bg-slate-100 hover:text-black">
                        View Details
                      </button>
                    </motion.div>
                  </div>
                ))}

                {day.accommodation && (
                  <div className="mt-2 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-gray-600">
                    🏨 <span className="font-medium">{day.accommodation.name}</span>
                    {day.accommodation.checkIn && (
                      <span className="ml-2 text-xs">• Check-in: {day.accommodation.checkIn}</span>
                    )}
                    {day.accommodation.nights && (
                      <span className="ml-2 text-xs">
                        • {day.accommodation.nights} night
                        {day.accommodation.nights > 1 ? "s" : ""}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}
