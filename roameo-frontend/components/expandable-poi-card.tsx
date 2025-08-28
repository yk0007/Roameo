"use client";

import React, { useEffect, useId, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useOutsideClick } from "../hooks/use-outside-click";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Heart, Plus, MapPin, Star, Clock, Phone, Globe, Navigation, Check } from "lucide-react";
import type { POI } from "@/lib/types";

interface ExpandablePoiCardProps {
  poi: POI;
  onSave?: (poi: POI) => void;
  onAdd?: (poi: POI) => void;
  onReplan?: (poi: POI) => void;
  isSaved?: boolean;
  isInItinerary?: boolean;
  className?: string;
}

export function ExpandablePoiCard({
  poi,
  onSave,
  onAdd,
  onReplan,
  isSaved = false,
  isInItinerary = false,
  className = ""
}: ExpandablePoiCardProps) {
  const [active, setActive] = useState<boolean>(false);
  const ref = useRef<HTMLDivElement>(null);
  const id = useId();

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setActive(false);
      }
    }

    if (active) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [active]);

  useOutsideClick(ref, () => setActive(false));

  const getPriceRange = (priceLevel?: number) => {
    if (!priceLevel) return "Price not available";
    const ranges = ["Free", "$", "$$", "$$$", "$$$$"];
    return ranges[priceLevel] || "$$";
  };

  const formatOpeningHours = (hours?: string | string[]) => {
    if (!hours) return "Hours not available";
    if (Array.isArray(hours)) return hours.join(", ");
    return hours;
  };

  return (
    <>
      <AnimatePresence>
        {active && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/20 h-full w-full z-10"
          />
        )}
      </AnimatePresence>
      
      <AnimatePresence>
        {active ? (
          <div className="fixed inset-0 grid place-items-center z-[100]">
            <motion.button
              key={`button-${poi.name}-${id}`}
              layout
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: 0.05 } }}
              className="flex absolute top-2 right-2 lg:hidden items-center justify-center bg-white rounded-full h-6 w-6"
              onClick={() => setActive(false)}
            >
              <CloseIcon />
            </motion.button>
            
            <motion.div
              layoutId={`card-${poi.name}-${id}`}
              ref={ref}
              className="w-full max-w-[500px] h-full md:h-fit md:max-h-[90%] flex flex-col bg-white dark:bg-neutral-900 sm:rounded-3xl overflow-hidden"
            >
              <motion.div layoutId={`image-${poi.name}-${id}`}>
                <Image
                  src={poi.photoUrl || '/placeholder.svg'}
                  alt={poi.name}
                  className="w-full h-80 lg:h-80 sm:rounded-tr-lg sm:rounded-tl-lg object-cover"
                  width={500}
                  height={320}
                  quality={90}
                  priority={true}
                  placeholder="empty"
                />
              </motion.div>

              <div>
                <div className="flex justify-between items-start p-4">
                  <div className="flex-1">
                    <motion.h3
                      layoutId={`title-${poi.name}-${id}`}
                      className="font-bold text-neutral-700 dark:text-neutral-200 text-lg"
                    >
                      {poi.name}
                    </motion.h3>
                    <motion.p
                      layoutId={`description-${poi.name}-${id}`}
                      className="text-neutral-600 dark:text-neutral-400 text-sm"
                    >
                      {poi.type} • {poi.address}
                    </motion.p>
                  </div>

                  <div className="flex gap-2 ml-4">
                    {!isSaved && onSave && (
                      <button
                        onClick={() => onSave(poi)}
                        className="shadow-[inset_0_0_0_2px_#ef4444] text-red-500 px-4 py-2 rounded-full tracking-wider uppercase font-bold bg-transparent hover:bg-red-500 hover:text-white dark:text-red-400 transition duration-200 text-xs flex items-center gap-1"
                      >
                        <Heart className="w-3 h-3" />
                        Save
                      </button>
                    )}
                    
                    {!isInItinerary && onAdd && (
                      <button
                        onClick={() => onAdd(poi)}
                        className="shadow-[inset_0_0_0_2px_#3b82f6] text-blue-500 px-4 py-2 rounded-full tracking-wider uppercase font-bold bg-transparent hover:bg-blue-500 hover:text-white dark:text-blue-400 transition duration-200 text-xs flex items-center gap-1"
                      >
                        <Plus className="w-3 h-3" />
                        Add
                      </button>
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
                    {/* Rating */}
                    {poi.rating && (
                      <div className="flex items-center gap-2">
                        <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                        <span className="font-medium">{poi.rating.toFixed(1)}</span>
                        <span className="text-gray-500">(reviews)</span>
                      </div>
                    )}

                    {/* Price Range */}
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">
                        {getPriceRange(poi.priceLevel)}
                      </Badge>
                    </div>

                    {/* Description */}
                    {poi.description && (
                      <div>
                        <h4 className="font-semibold mb-2">About</h4>
                        <p className="text-sm leading-relaxed">{poi.description}</p>
                      </div>
                    )}

                    {/* Opening Hours */}
                    {poi.openingHours && (
                      <div className="flex items-start gap-2">
                        <Clock className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <div>
                          <span className="font-medium">Hours: </span>
                          <span className="text-sm">{formatOpeningHours(poi.openingHours)}</span>
                        </div>
                      </div>
                    )}

                    {/* Contact Info */}
                    {poi.phone && (
                      <div className="flex items-center gap-2">
                        <Phone className="w-4 h-4" />
                        <span className="text-sm">{poi.phone}</span>
                      </div>
                    )}

                    {poi.website && (
                      <div className="flex items-center gap-2">
                        <Globe className="w-4 h-4" />
                        <a 
                          href={poi.website} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-sm text-blue-600 hover:underline"
                        >
                          Visit Website
                        </a>
                      </div>
                    )}

                    {/* Location */}
                    <div className="flex items-start gap-2">
                      <MapPin className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span className="text-sm">{poi.address || 'Address not available'}</span>
                    </div>
                  </motion.div>
                </div>
              </div>
            </motion.div>
          </div>
        ) : null}
      </AnimatePresence>

      {/* Card Preview */}
      <motion.div
        layoutId={`card-${poi.name}-${id}`}
        onClick={() => setActive(true)}
        className={`p-4 flex flex-col md:flex-row justify-between items-center hover:bg-neutral-50 dark:hover:bg-neutral-800 rounded-xl cursor-pointer transition-all duration-300 hover:scale-[1.02] ${className}`}
      >
        <div className="flex gap-4 flex-col md:flex-row">
          <motion.div layoutId={`image-${poi.name}-${id}`}>
            <Image
              src={poi.photoUrl || '/placeholder.svg'}
              alt={poi.name}
              className="h-40 w-40 md:h-14 md:w-14 rounded-lg object-cover"
              width={160}
              height={160}
              quality={80}
            />
          </motion.div>
          <div className="">
            <motion.h3
              layoutId={`title-${poi.name}-${id}`}
              className="font-medium text-neutral-800 dark:text-neutral-200 text-center md:text-left"
            >
              {poi.name}
            </motion.h3>
            <motion.p
              layoutId={`description-${poi.name}-${id}`}
              className="text-neutral-600 dark:text-neutral-400 text-center md:text-left text-sm"
            >
              {poi.type} • {poi.rating ? `⭐ ${poi.rating.toFixed(1)}` : 'No rating'}
            </motion.p>
          </div>
        </div>
        <motion.button
          layoutId={`button-${poi.name}-${id}`}
          className="shadow-[inset_0_0_0_2px_#616467] text-black px-6 py-2 rounded-full tracking-wider uppercase font-bold bg-transparent hover:bg-[#616467] hover:text-white dark:text-neutral-200 transition duration-200 text-xs mt-4 md:mt-0"
        >
          Details
        </motion.button>
      </motion.div>
    </>
  );
}

export const CloseIcon = () => {
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
};
