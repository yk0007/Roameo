import type {
  CanonicalSession,
  Itinerary,
  MapData,
  POI,
  SearchResults,
  SessionPlanMutation,
  TripContext
} from "./types";
import { getBudgetOptionByLabel, getBudgetOptionByTotal } from "./budget-options";

const CANONICAL_POI_SOURCES = new Set(["google_places", "google_maps", "web_research"]);

function latestAcceptedDecision(
  session: CanonicalSession | undefined,
  prefix: string
) {
  return [...(session?.memory.acceptedDecisions || [])]
    .reverse()
    .find((decision) => decision.startsWith(prefix))
    ?.slice(prefix.length)
    .trim();
}

function acceptedDurationDays(session?: CanonicalSession) {
  const decision = latestAcceptedDecision(session, "Duration: ");
  const match = decision?.match(/(\d+)/);
  if (!match) {
    return undefined;
  }

  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

function acceptedTravelers(session?: CanonicalSession) {
  const decision = latestAcceptedDecision(session, "Travelers: ");
  const match = decision?.match(/(\d+)/);
  if (!match) {
    return undefined;
  }

  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? String(value) : undefined;
}

function acceptedBudget(session?: CanonicalSession) {
  return latestAcceptedDecision(session, "Budget target: ");
}

function acceptedOrigin(session?: CanonicalSession) {
  return latestAcceptedDecision(session, "Origin: ");
}

function acceptedDestination(session?: CanonicalSession) {
  return latestAcceptedDecision(session, "Destination: ");
}

function deriveTripTitle(
  session: CanonicalSession | undefined,
  destination?: string,
  totalDays?: number
) {
  const currentTitle = session?.title?.trim();
  if (currentTitle && !/^untitled trip$/i.test(currentTitle)) {
    return currentTitle;
  }

  if (destination && totalDays) {
    return `${destination}, ${totalDays} days`;
  }
  if (destination) {
    return destination;
  }

  return currentTitle || "Untitled trip";
}

function getCanonicalPois(session?: CanonicalSession): POI[] {
  return Object.values(session?.poiCatalog.items || {}).filter((poi) =>
    CANONICAL_POI_SOURCES.has(poi.source)
  );
}

function roundDistanceKm(
  from?: { lat: number; lng: number },
  to?: { lat: number; lng: number }
) {
  if (!from || !to) {
    return undefined;
  }

  const toRadians = (value: number) => (value * Math.PI) / 180;
  const R = 6371;
  const dLat = toRadians(to.lat - from.lat);
  const dLng = toRadians(to.lng - from.lng);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRadians(from.lat)) *
      Math.cos(toRadians(to.lat)) *
      Math.sin(dLng / 2) ** 2;

  return Number((2 * R * Math.asin(Math.sqrt(a))).toFixed(1));
}

export function buildSearchResults(session?: CanonicalSession): SearchResults {
  const values = getCanonicalPois(session);
  return {
    stays: values.filter((poi) => poi.type === "stay"),
    restaurants: values.filter((poi) => poi.type === "restaurant"),
    attractions: values.filter((poi) => poi.type === "attraction")
  };
}

export function buildItinerary(session?: CanonicalSession): Itinerary | undefined {
  const plan = session?.plan;
  if (!plan) {
    return undefined;
  }

  return {
    origin: plan.origin,
    destination: plan.destination,
    destinations: plan.destinations,
    days: plan.totalDays,
    destinationSegments: plan.destinationSegments.map((segment) => ({
      destination: segment.destination,
      startDay: segment.startDay,
      endDay: segment.endDay,
      days: segment.endDay - segment.startDay + 1
    })),
    daysPlan: plan.days.map((day) => {
      const accommodation = day.accommodationPoiId
        ? session?.poiCatalog.items[day.accommodationPoiId]
        : undefined;
      const canonicalAccommodation =
        accommodation && CANONICAL_POI_SOURCES.has(accommodation.source)
          ? accommodation
          : undefined;
      const pois = day.activities.map((activity) => {
        const poi = activity.poiId
          ? session?.poiCatalog.items[activity.poiId]
          : undefined;
        return poi && CANONICAL_POI_SOURCES.has(poi.source) ? poi : undefined;
      });

      return {
        day: day.day,
        date: day.date,
        title: day.title,
        theme: day.theme,
        summary: day.summary,
        accommodation: canonicalAccommodation
          ? {
              name: canonicalAccommodation.name,
              poiId: canonicalAccommodation.id,
              location: canonicalAccommodation.address,
              photoUrl: canonicalAccommodation.photoUrl
            }
          : undefined,
        activities: day.activities.map((activity) => {
          const poi = activity.poiId
            ? session?.poiCatalog.items[activity.poiId]
            : undefined;
          const canonicalPoi =
            poi && CANONICAL_POI_SOURCES.has(poi.source) ? poi : undefined;
          const activityIndex = day.activities.findIndex(
            (candidate) => candidate.id === activity.id
          );
          const previousPoi =
            activityIndex > 0 ? pois[activityIndex - 1] : undefined;

          return {
            id: activity.id,
            name: activity.title,
            start: activity.startTime,
            end: activity.endTime,
            location: canonicalPoi?.address,
            poiId: activity.poiId,
            lat: canonicalPoi?.lat,
            lng: canonicalPoi?.lng,
            distanceKm: getActivityDistanceKm(previousPoi, canonicalPoi),
            photoUrl: canonicalPoi?.photoUrl,
            rating: canonicalPoi?.rating,
            description: activity.summary || canonicalPoi?.description,
            notes: activity.notes
          };
        })
      };
    })
  };
}

export function buildMapData(session?: CanonicalSession): MapData {
  const plan = session?.plan;
  const poiMap = new Map<string, POI>();
  const routes: MapData["routes"] = [];

  if (!plan) {
    for (const poi of getCanonicalPois(session)) {
      poiMap.set(poi.id, poi);
    }

    return {
      pois: Array.from(poiMap.values()),
      routes
    };
  }

  for (const day of plan.days) {
    const dayPois = day.activities
      .map((activity) =>
        activity.poiId ? session?.poiCatalog.items[activity.poiId] : undefined
      )
      .filter(Boolean) as POI[];

      if (day.accommodationPoiId) {
        const accommodation = session?.poiCatalog.items[day.accommodationPoiId];
      if (accommodation && CANONICAL_POI_SOURCES.has(accommodation.source)) {
        poiMap.set(accommodation.id, accommodation);
      }
    }

    dayPois.forEach((poi) => {
      if (CANONICAL_POI_SOURCES.has(poi.source)) {
        poiMap.set(poi.id, poi);
      }
    });

    for (let index = 1; index < dayPois.length; index += 1) {
      routes.push({
        from: [dayPois[index - 1].lat, dayPois[index - 1].lng],
        to: [dayPois[index].lat, dayPois[index].lng],
        durationMinutes:
          day.activities[index].travelTimeMinutesFromPrevious || undefined
      });
    }
  }

  return {
    pois: Array.from(poiMap.values()),
    routes
  };
}

export function buildTripContext(session?: CanonicalSession): TripContext {
  const plan = session?.plan;
  const destination =
    plan?.destination ||
    acceptedDestination(session) ||
    session?.memory.destinationsDiscussed.at(-1) ||
    "";
  const totalDays = plan?.totalDays || acceptedDurationDays(session) || 0;
  const budgetLabel =
    getBudgetOptionByTotal(plan?.budgetTarget?.total || plan?.budget?.total)?.label ||
    acceptedBudget(session) ||
    "";
  return {
    id: session?.id || "",
    title: deriveTripTitle(session, destination, totalDays),
    origin: plan?.origin || acceptedOrigin(session) || "",
    destination,
    destinations:
      plan?.destinations?.length
        ? plan.destinations
        : session?.memory.destinationsDiscussed.length
          ? session.memory.destinationsDiscussed
          : destination
            ? [destination]
            : [],
    startDate: plan?.startDate || session?.memory.dateContext.inferredStartDate,
    endDate: plan?.endDate || session?.memory.dateContext.inferredEndDate,
    dateFlexibility: session?.memory.dateContext.flexibility,
    days: totalDays,
    travelers:
      plan?.travelerCount ? String(plan.travelerCount) : acceptedTravelers(session) || "",
    budget: budgetLabel
  };
}

export function buildItineraryPoiIds(session?: CanonicalSession) {
  const ids = new Set<string>();
  for (const day of session?.plan?.days || []) {
    if (day.accommodationPoiId) {
      ids.add(day.accommodationPoiId);
    }
    for (const activity of day.activities) {
      if (activity.poiId) {
        ids.add(activity.poiId);
      }
    }
  }
  return ids;
}

function parseBudgetInput(value: string): {
  total: number;
  currency: string;
} | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const option = getBudgetOptionByLabel(trimmed);
  if (option) {
    return {
      total: option.total,
      currency: option.currency
    };
  }

  const amountMatch = trimmed.match(/(\d[\d,]*)/);
  if (!amountMatch) {
    return null;
  }

  const total = Number.parseInt(amountMatch[1].replace(/,/g, ""), 10);
  if (Number.isNaN(total)) {
    return null;
  }

  const currencyMatch = trimmed.match(/^[A-Za-z]{3}/);
  return {
    total,
    currency: currencyMatch?.[0]?.toUpperCase() || "INR"
  };
}

export function buildOverviewMutation(
  previous: TripContext,
  next: TripContext
): SessionPlanMutation | null {
  const budgetTarget = parseBudgetInput(next.budget);
  const mutation: Extract<SessionPlanMutation, { type: "update_overview" }> = {
    type: "update_overview"
  };
  let hasChanges = false;

  if (previous.title !== next.title && next.title) {
    mutation.title = next.title;
    hasChanges = true;
  }
  if (previous.origin !== next.origin && next.origin) {
    mutation.origin = next.origin;
    hasChanges = true;
  }
  if (previous.destination !== next.destination && next.destination) {
    mutation.destination = next.destination;
    hasChanges = true;
  }
  if (
    JSON.stringify(previous.destinations || []) !==
      JSON.stringify(next.destinations || []) &&
    (next.destinations || []).length > 0
  ) {
    mutation.destinations = next.destinations;
    mutation.destination = next.destinations[0];
    hasChanges = true;
  }
  if (previous.startDate !== next.startDate && next.startDate) {
    mutation.startDate = next.startDate;
    hasChanges = true;
  }
  if (previous.endDate !== next.endDate && next.endDate) {
    mutation.endDate = next.endDate;
    hasChanges = true;
  }
  if (previous.dateFlexibility !== next.dateFlexibility && next.dateFlexibility) {
    mutation.dateFlexibility = next.dateFlexibility;
    hasChanges = true;
  }
  if (previous.days !== next.days && next.days) {
    mutation.totalDays = next.days;
    hasChanges = true;
  }
  if (previous.travelers !== next.travelers && next.travelers) {
    const travelerCount = Number.parseInt(next.travelers, 10);
    if (!Number.isNaN(travelerCount) && travelerCount > 0) {
      mutation.travelerCount = travelerCount;
      hasChanges = true;
    }
  }
  if (previous.budget !== next.budget && budgetTarget) {
    mutation.budgetTotal = budgetTarget.total;
    mutation.currency = budgetTarget.currency;
    hasChanges = true;
  }

  return hasChanges ? mutation : null;
}

export function getActivityDistanceKm(
  previousPoi?: POI,
  currentPoi?: POI
) {
  return roundDistanceKm(previousPoi, currentPoi);
}
