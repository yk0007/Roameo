"use client";

import { Check, ChevronLeft, ChevronRight, Heart, Info, MapPin, Star } from "lucide-react";
import Image from "next/image";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { resolvePoiImageUrl } from "@/lib/poi-image-url";
import type { POI } from "@/lib/types";
import { PoiTypeIcon } from "./poi-type-icon";

interface SearchCardProps {
  poi: POI;
  isSaved: boolean;
  isItineraryItem?: boolean;
  onToggleSave: (poi: POI, next: boolean) => void;
  onAddPoi: (poi: POI) => void;
  onReplan: (poi: POI) => void;
  compact?: boolean;
}

function poiTypeLabel(poi: POI) {
  switch (poi.type) {
    case "restaurant":
      return "Restaurant";
    case "stay":
      return "Stay";
    case "destination":
      return "Destination";
    case "transit":
      return "Transit";
    default:
      return "Attraction";
  }
}

function formatPriceLevel(level?: number) {
  if (typeof level !== "number") {
    return null;
  }
  return "$".repeat(Math.max(1, Math.min(level, 4)));
}

function compactAddress(address?: string) {
  if (!address) {
    return "Location available on the map";
  }
  const parts = address.split(",").map((part) => part.trim()).filter(Boolean);
  return parts.slice(0, 2).join(", ");
}

export function SearchCard({
  poi,
  isSaved,
  isItineraryItem,
  onToggleSave,
  onAddPoi,
  onReplan,
  compact = false,
}: SearchCardProps) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [imageError, setImageError] = useState(false);

  const images = useMemo(() => {
    const resolved = resolvePoiImageUrl(poi.photoUrl);
    return resolved ? [resolved] : [];
  }, [poi.photoUrl]);
  const hasImage = images.length > 0 && !imageError;
  const priceLabel = formatPriceLevel(poi.priceLevel);

  const showPrevious = () => {
    if (images.length > 1) {
      setCurrentImageIndex((current) => (current - 1 + images.length) % images.length);
    }
  };

  const showNext = () => {
    if (images.length > 1) {
      setCurrentImageIndex((current) => (current + 1) % images.length);
    }
  };

  return (
    <article className="overflow-hidden rounded-[28px] bg-white shadow-none transition-transform duration-200 hover:-translate-y-0.5">
      <div className="relative aspect-[4/3.2] overflow-hidden rounded-[26px] bg-[#eef2f6]">
        {hasImage ? (
          <Image
            src={images[currentImageIndex]}
            alt={poi.name}
            fill
            className="object-cover"
            quality={90}
            priority
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center bg-[#edf1f5] text-slate-400">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-300/90 text-slate-500">
              <PoiTypeIcon poi={poi} className="h-8 w-8" />
            </div>
            <div className="mt-4 text-[15px] font-medium">No Image</div>
          </div>
        )}

        <div className="absolute inset-x-0 top-0 flex items-start justify-between p-4">
          <Button
            type="button"
            size="sm"
            onClick={() => onAddPoi(poi)}
            disabled={isItineraryItem}
            className={`${compact ? "h-8 px-3 text-[12px]" : "h-11 px-5 text-[14px]"} rounded-full font-medium shadow-[0_6px_18px_rgba(15,23,42,0.16)] ${
              isItineraryItem
                ? "bg-white text-slate-600 hover:bg-white"
                : "bg-[#2d2d2d] text-white hover:bg-black"
            }`}
          >
            {isItineraryItem ? (
              <>
                <Check className={`${compact ? "mr-1 h-3 w-3" : "mr-2 h-4 w-4"}`} />
                Added
              </>
            ) : (
              "Add to trip"
            )}
          </Button>

          <Button
            type="button"
            size="icon"
            variant="secondary"
            className={`${compact ? "h-8 w-8" : "h-11 w-11"} rounded-full border-0 bg-white/92 text-slate-700 shadow-[0_6px_18px_rgba(15,23,42,0.14)] hover:bg-white`}
            onClick={() => onToggleSave(poi, !isSaved)}
          >
            <Heart className={`${compact ? "h-4 w-4" : "h-5 w-5"} ${isSaved ? "fill-slate-900 text-slate-900" : ""}`} />
          </Button>
        </div>

        {images.length > 1 ? (
          <>
            <Button
              type="button"
              size="icon"
              variant="secondary"
              className="absolute left-3 top-1/2 h-9 w-9 -translate-y-1/2 rounded-full border-0 bg-white/90 text-slate-700 shadow-[0_6px_18px_rgba(15,23,42,0.14)] hover:bg-white"
              onClick={showPrevious}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              type="button"
              size="icon"
              variant="secondary"
              className="absolute right-3 top-1/2 h-9 w-9 -translate-y-1/2 rounded-full border-0 bg-white/90 text-slate-700 shadow-[0_6px_18px_rgba(15,23,42,0.14)] hover:bg-white"
              onClick={showNext}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 gap-1.5">
              {images.map((_, index) => (
                <span
                  key={`${poi.id}-dot-${index}`}
                  className={`h-2.5 w-2.5 rounded-full ${
                    index === currentImageIndex ? "bg-white" : "bg-white/55"
                  }`}
                />
              ))}
            </div>
          </>
        ) : null}

        <Button
          type="button"
          size="icon"
          variant="secondary"
          className={`absolute bottom-4 right-4 ${compact ? "h-8 w-8" : "h-10 w-10"} rounded-full border-0 bg-black/58 text-white shadow-[0_8px_20px_rgba(15,23,42,0.18)] hover:bg-black/70`}
          onClick={() => onReplan(poi)}
        >
          <Info className={compact ? "h-3.5 w-3.5" : "h-4 w-4"} />
        </Button>
      </div>

      <div className="px-1 pb-1 pt-4">
        <div className="flex items-start justify-between gap-2">
          <h3 className={`line-clamp-2 ${compact ? "text-[14px]" : "text-[18px]"} font-semibold leading-[1.18] tracking-[-0.03em] text-slate-900`}>
            {poi.name}
          </h3>
          {typeof poi.rating === "number" ? (
            <div className={`mt-0.5 flex shrink-0 items-center gap-1 ${compact ? "text-[12px]" : "text-[14px]"} font-semibold text-slate-900`}>
              <Star className={`${compact ? "h-3 w-3" : "h-4 w-4"} fill-black text-black`} />
              <span>{poi.rating.toFixed(1)}</span>
            </div>
          ) : null}
        </div>

        <div className={`mt-1.5 flex items-center gap-2 ${compact ? "text-[12px]" : "text-[15px]"} text-slate-500`}>
          <PoiTypeIcon poi={poi} className={`${compact ? "h-3 w-3" : "h-4 w-4"} shrink-0`} />
          <span>{poiTypeLabel(poi)}</span>
        </div>

        <div className={`mt-1 flex items-start gap-2 ${compact ? "text-[12px]" : "text-[15px]"} leading-5 text-slate-500`}>
          <MapPin className={`mt-0.5 ${compact ? "h-3 w-3" : "h-4 w-4"} shrink-0 text-slate-400`} />
          <span className="line-clamp-2">{compactAddress(poi.address)}</span>
        </div>

        {priceLabel ? (
          <div className="mt-2 text-[16px] font-medium tracking-[0.08em] text-slate-500">
            {priceLabel}
          </div>
        ) : null}
      </div>
    </article>
  );
}
