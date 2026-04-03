"use client";

import { motion } from "framer-motion";
import { ArrowRight, CalendarDays, MapPin, Plane } from "lucide-react";
import { CachedImage } from "@/components/cached-image";
import DestinationCardArt from "@/components/DestinationCardArt";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";
import { BACKEND_URL } from "@/lib/types";

export type DashboardTripSummary = {
  id: string;
  title: string;
  destination?: string;
  days?: number;
  createdAt: string;
  updatedAt: string;
  destinationImageUrl?: string;
};

type DashboardTripCardProps = {
  trip: DashboardTripSummary;
  index: number;
  animateIn: boolean;
  skipEntryAnimation?: boolean;
  onClick: () => void;
};

const BADGE_STYLES: Record<TripCardBadge["tone"], string> = {
  mint: "border-white/80 bg-white/88 text-slate-800 shadow-[0_10px_30px_rgba(15,23,42,0.12)]",
  indigo: "border-white/80 bg-white/88 text-slate-800 shadow-[0_10px_30px_rgba(15,23,42,0.12)]",
  amber: "border-white/80 bg-white/88 text-slate-800 shadow-[0_10px_30px_rgba(15,23,42,0.12)]"
};

const ART_RECIPES = [
  {
    variant: "stamp" as const,
    shellClass:
      "bg-[linear-gradient(135deg,#f3e4e8_0%,#f6eadc_52%,#ede7df_100%)]"
  },
  {
    variant: "topo" as const,
    shellClass:
      "bg-[linear-gradient(135deg,#e6edf6_0%,#f2ece3_52%,#ddd5cb_100%)]"
  },
  {
    variant: "stamp" as const,
    shellClass:
      "bg-[linear-gradient(135deg,#dce5ee_0%,#e7ddd1_52%,#cab7a8_100%)]"
  },
  {
    variant: "topo" as const,
    shellClass:
      "bg-[linear-gradient(135deg,#e2ebdd_0%,#dde4d1_50%,#cbb99b_100%)]"
  }
] as const;

const STOCK_IMAGE_LIBRARY = [
  "https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=1600&q=80",
  "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1600&q=80",
  "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1600&q=80",
  "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=1600&q=80",
  "https://images.unsplash.com/photo-1506929562872-bb421503ef21?auto=format&fit=crop&w=1600&q=80",
  "https://images.unsplash.com/photo-1519046904884-53103b34b206?auto=format&fit=crop&w=1600&q=80"
] as const;

const STOCK_IMAGE_MATCHERS: Array<{ pattern: RegExp; url: string }> = [
  {
    pattern: /\b(kerala|munnar|alleppey|kochi|india)\b/i,
    url: "https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=1600&q=80"
  },
  {
    pattern: /\b(mumbai|bombay)\b/i,
    url: "https://images.unsplash.com/photo-1595658658481-d53d3f999875?auto=format&fit=crop&w=1600&q=80"
  },
  {
    pattern: /\b(kyoto|japan|tokyo|osaka)\b/i,
    url: "https://images.unsplash.com/photo-1545569341-9eb8b30979d9?auto=format&fit=crop&w=1600&q=80"
  },
  {
    pattern: /\b(swiss|switzerland|zermatt|alps)\b/i,
    url: "https://images.unsplash.com/photo-1508261305436-4e4f0f7dc663?auto=format&fit=crop&w=1600&q=80"
  },
  {
    pattern: /\b(amalfi|italy|campania|rome|florence)\b/i,
    url: "https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=1600&q=80"
  }
] as const;

type TripCardBadge = {
  label: string;
  tone: "mint" | "indigo" | "amber";
};

function buildTripBadge(trip: DashboardTripSummary): TripCardBadge {
  const normalizedTitle = trip.title.trim().toLowerCase();
  const hasCorePlan = Boolean(trip.destination && trip.days);

  if (!hasCorePlan || normalizedTitle === "untitled trip") {
    return { label: "AI Draft", tone: "indigo" };
  }

  const createdAt = Date.parse(trip.createdAt);
  const updatedAt = Date.parse(trip.updatedAt);
  const modifiedRecently =
    Number.isFinite(createdAt) &&
    Number.isFinite(updatedAt) &&
    updatedAt - createdAt > 1000 * 60 * 60;

  if (modifiedRecently) {
    return { label: "Upcoming", tone: "mint" };
  }

  return { label: "Planned", tone: "amber" };
}

function formatTripDate(trip: DashboardTripSummary) {
  const source = trip.updatedAt || trip.createdAt;
  const parsed = Date.parse(source);
  if (!Number.isFinite(parsed)) {
    return "Dates flexible";
  }

  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit"
  }).format(parsed);
}

function formatDestination(destination?: string) {
  if (!destination?.trim()) {
    return "Destination to be decided";
  }

  const value = destination.trim();
  return value
    .split(",")
    .map((part) =>
      part
        .trim()
        .split(/\s+/)
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ")
    )
    .join(", ");
}

function formatDays(days?: number) {
  if (!days || days < 1) {
    return "Flexible";
  }

  return `${days} Day${days === 1 ? "" : "s"}`;
}

function getArtRecipe(index: number) {
  return ART_RECIPES[index % ART_RECIPES.length];
}

function getBadgeDotClass(tone: TripCardBadge["tone"]) {
  if (tone === "mint") {
    return "bg-emerald-500";
  }
  if (tone === "indigo") {
    return "bg-slate-500";
  }
  return "bg-amber-500";
}

export function DashboardTripCard({
  trip,
  index,
  animateIn,
  skipEntryAnimation = false,
  onClick
}: DashboardTripCardProps) {
  const badge = buildTripBadge(trip);
  const artRecipe = getArtRecipe(index);
  const useAnimatedStamp = !trip.destination && !trip.days && !trip.destinationImageUrl;
  const [dynamicImage, setDynamicImage] = useState<string | null>(null);

  useEffect(() => {
    if (useAnimatedStamp || trip.destinationImageUrl) return;

    const haystack = `${trip.title} ${trip.destination || ""}`;
    const matched = STOCK_IMAGE_MATCHERS.find(({ pattern }) => pattern.test(haystack));
    if (matched) {
      setDynamicImage(matched.url);
      return;
    }

    if (!trip.destination) {
      setDynamicImage(STOCK_IMAGE_LIBRARY[index % STOCK_IMAGE_LIBRARY.length]);
      return;
    }

    let isMounted = true;
    const fetchImage = async () => {
      try {
        const primaryLocation = trip.destination!.split(',')[0].trim();
        let fetchedUrl = null;

        // Step 1: Try the robust backend DestinationImageService (Google Places)
        try {
          const res = await fetch(`${BACKEND_URL}/api/destination-image?q=${encodeURIComponent(primaryLocation)}`);
          if (res.ok) {
            const data = await res.json();
            if (data.imageUrl) {
              fetchedUrl = data.imageUrl;
            }
          }
        } catch (backendErr) {
          console.error("Backend image fetch failed:", backendErr);
        }

        if (!isMounted) return;

        // Step 2: Fallback to Wikipedia if Google Places failed or has no image
        if (!fetchedUrl) {
          const wikiRes = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(primaryLocation)}`);
          if (wikiRes.ok) {
            const data = await wikiRes.json();
            if (data.originalimage?.source) {
              fetchedUrl = data.originalimage.source;
            } else if (data.thumbnail?.source) {
              fetchedUrl = data.thumbnail.source;
            }
          }
        }

        if (!isMounted) return;

        if (fetchedUrl) {
          setDynamicImage(fetchedUrl);
        } else {
          setDynamicImage(STOCK_IMAGE_LIBRARY[index % STOCK_IMAGE_LIBRARY.length]);
        }
      } catch (err) {
        if (isMounted) {
          setDynamicImage(STOCK_IMAGE_LIBRARY[index % STOCK_IMAGE_LIBRARY.length]);
        }
      }
    };

    fetchImage();

    return () => {
      isMounted = false;
    };
  }, [trip.destination, trip.title, trip.destinationImageUrl, useAnimatedStamp, index]);

  const coverImageUrl = trip.destinationImageUrl || dynamicImage || STOCK_IMAGE_LIBRARY[index % STOCK_IMAGE_LIBRARY.length];

  const entryAnimation = skipEntryAnimation
    ? {}
    : {
        initial: { opacity: 0, y: 20 },
        animate: animateIn ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 },
        transition: { duration: 0.45, delay: index * 0.08, ease: [0.22, 1, 0.36, 1] as const }
      };

  return (
    <motion.article
      {...entryAnimation}
      whileHover={{ y: -6 }}
      onClick={onClick}
      className="group h-full cursor-pointer"
    >
      <div className="flex h-full flex-col overflow-hidden rounded-[2rem] border border-slate-200 bg-white p-5 shadow-[0_8px_24px_rgba(15,23,42,0.08)] transition-all duration-300 group-hover:shadow-[0_18px_36px_rgba(15,23,42,0.12)]">
        <div
          className={cn(
            "relative mb-5 aspect-[4/3] overflow-hidden rounded-[1.65rem]",
            artRecipe.shellClass
          )}
        >
          {useAnimatedStamp ? (
            <div className="h-full w-full overflow-hidden rounded-[1.65rem] transition-transform duration-700 group-hover:scale-[1.03]">
              <DestinationCardArt
                destination={trip.destination || trip.title}
                variant={artRecipe.variant}
                className="h-full w-full"
              />
            </div>
          ) : (
            <div className="relative h-full w-full overflow-hidden rounded-[1.65rem] transition-transform duration-700 group-hover:scale-[1.03]">
              <CachedImage
                src={coverImageUrl}
                alt={trip.destination || trip.title}
                className="h-full w-full object-cover"
                quality={88}
              />
            </div>
          )}

          <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(15,23,42,0.05),rgba(15,23,42,0),rgba(15,23,42,0.12))]" />

          <div className="absolute left-4 top-4">
            <span
              className={cn(
                "inline-flex items-center gap-2 rounded-full border px-4 py-2 text-[0.72rem] font-semibold uppercase tracking-[0.18em] backdrop-blur-md",
                BADGE_STYLES[badge.tone]
              )}
            >
              <span className={cn("h-1.5 w-1.5 rounded-full", getBadgeDotClass(badge.tone))} />
              {badge.label}
            </span>
          </div>

          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/8 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            <div className="flex h-12 w-12 translate-y-4 items-center justify-center rounded-full bg-white/90 text-slate-900 shadow-lg backdrop-blur-sm transition-all duration-300 group-hover:translate-y-0">
              <ArrowRight className="h-5 w-5" />
            </div>
          </div>
        </div>

        <div className="flex flex-1 flex-col px-1 pb-1">
          <h3 className="mb-3 min-h-[4.75rem] line-clamp-2 text-[1.8rem] font-semibold tracking-[-0.055em] text-slate-950 transition-colors duration-300 group-hover:text-slate-700">
            {trip.title || "Untitled trip"}
          </h3>

          <div className="mt-auto space-y-4 text-[0.98rem] text-slate-500">
            <div className="flex items-center gap-2.5">
              <MapPin className="h-[19px] w-[19px] shrink-0 text-slate-400" />
              <span className="truncate">{formatDestination(trip.destination)}</span>
            </div>

            <div className="h-px w-full bg-slate-200" />

            <div className="flex items-center justify-between gap-3 text-[1rem]">
              <div className="flex min-w-0 items-center gap-2.5">
                <CalendarDays className="h-[19px] w-[19px] shrink-0 text-slate-400" />
                <span className="truncate">{formatTripDate(trip)}</span>
              </div>

              <div className="inline-flex shrink-0 items-center rounded-[1rem] border border-slate-200 bg-slate-50 px-3 py-2 font-medium text-slate-700">
                <Plane className="mr-2 h-4 w-4" />
                {trip.days ? `${trip.days}d` : "Flexible"}
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.article>
  );
}
