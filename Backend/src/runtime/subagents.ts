import { randomUUID } from "node:crypto";
import { z } from "zod";
import type {
  AssistantResponseBlock,
  DateContext,
  DateAdvisoryItem,
  FollowUpDomain,
  FollowUpOption,
  PendingFollowUpContext,
  PlanSnapshot,
  Poi,
  PoiCatalog,
  Provider,
  SessionSnapshot,
  TravelIntent,
  TravelStyle
} from "@roameo/contracts";
import { planSnapshotSchema, travelIntentSchema } from "@roameo/contracts";
import {
  ProviderService,
  type ResolvedProvider
} from "../services/provider-service.js";
import {
  TravelToolsService,
  type DiscoveryFocus
} from "../services/travel-tools.js";

export type TurnResolution = {
  intent: TravelIntent;
  destination?: string;
  destinations: string[];
  origin?: string;
  totalDays?: number;
  travelerCount?: number;
  budgetNote?: string;
  styles: TravelStyle[];
  questionFocus?: string;
  dateContext: DateContext;
  stayMode?: boolean;
  explicitNewTrip?: boolean;
  followUpDomain?: FollowUpDomain;
  restaurantCategory?: RestaurantCategoryKey;
  followUpAmbiguous?: boolean;
  followUpOptions?: FollowUpOption[];
};

export type DestinationResearch = {
  destinations: string[];
  focus: DiscoveryFocus;
  catalog: PoiCatalog;
  grouped: {
    stays: Poi[];
    restaurants: Poi[];
    attractions: Poi[];
  };
  facts: Array<{ title: string; url: string; snippet: string }>;
};

export type PlanningContext = {
  weather?: {
    summary?: string;
    daily: Array<{ date: string; summary: string }>;
    advisories: DateAdvisoryItem[];
  };
  events?: {
    summary?: string;
    items: Array<{ title: string; detail: string; sourceLabel?: string }>;
    advisories: DateAdvisoryItem[];
  };
  holidays?: {
    summary?: string;
    items: Array<{ title: string; detail: string; sourceLabel?: string }>;
    advisories: DateAdvisoryItem[];
  };
  workerProgress?: Array<{
    label: string;
    detail?: string;
    state: "running" | "completed";
  }>;
};

type RestaurantCategoryKey =
  | "famous"
  | "budget_friendly"
  | "vegetarian"
  | "non_vegetarian"
  | "premium";

const intentAliases: Record<string, TravelIntent> = {
  plan_trip: "plan_trip",
  plantrip: "plan_trip",
  plan: "plan_trip",
  planning: "plan_trip",
  refine_trip: "refine_trip",
  refinetrip: "refine_trip",
  refine: "refine_trip",
  update_trip: "refine_trip",
  update: "refine_trip",
  modify: "refine_trip",
  search_places: "search_places",
  searchplaces: "search_places",
  destination_search: "search_places",
  destinationsearch: "search_places",
  place_search: "search_places",
  places_search: "search_places",
  search: "search_places",
  places: "search_places",
  question: "question",
  chat: "question",
  ask: "question",
  settings: "settings",
  preferences: "settings",
  config: "settings",
  meta: "meta",
  misc: "meta"
};

export function normalizeTravelIntent(value: unknown): TravelIntent {
  const normalizedKey = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  const intent = intentAliases[normalizedKey];

  if (!intent) {
    throw new Error(`Unsupported travel intent "${String(value ?? "")}"`);
  }

  return intent;
}

const styleCueMatchers: Array<{
  style: TurnResolution["styles"][number];
  pattern: RegExp;
}> = [
  { style: "relaxed", pattern: /\b(relaxed|slow|calm|easygoing|laid[- ]back)\b/i },
  { style: "balanced", pattern: /\bbalanced\b/i },
  { style: "packed", pattern: /\b(packed|fast[- ]paced|busy)\b/i },
  { style: "luxury", pattern: /\b(luxury|premium|high[- ]end)\b/i },
  { style: "budget", pattern: /\b(budget|budget[- ]friendly|affordable|cheap)\b/i },
  { style: "family", pattern: /\b(family|kids?|children)\b/i },
  { style: "romantic", pattern: /\b(romantic|honeymoon|couple)\b/i },
  { style: "adventure", pattern: /\b(adventure|wildlife|hiking|trekking|scenic|nature)\b/i },
  { style: "culture", pattern: /\b(culture|cultural|heritage|museum|history|historic)\b/i }
];

const styleAliases: Array<{
  style: TravelStyle;
  pattern: RegExp;
}> = [
  { style: "relaxed", pattern: /\b(relaxed|slow|calm|easygoing|laid[- ]back|leisure|chill)\b/i },
  { style: "balanced", pattern: /\b(balanced|mixed|moderate)\b/i },
  { style: "packed", pattern: /\b(packed|fast[- ]paced|busy|intense|full)\b/i },
  { style: "luxury", pattern: /\b(luxury|premium|high[- ]end|upscale)\b/i },
  { style: "budget", pattern: /\b(budget|budget[- ]friendly|affordable|cheap|value)\b/i },
  { style: "family", pattern: /\b(family|kid|kids|child|children)\b/i },
  { style: "romantic", pattern: /\b(romantic|honeymoon|couple)\b/i },
  { style: "adventure", pattern: /\b(adventure|wildlife|hiking|trekking|scenic|nature|outdoor|explore)\b/i },
  { style: "culture", pattern: /\b(culture|cultural|heritage|museum|history|historic|local)\b/i }
];

function normalizeTravelStyle(value: unknown): TravelStyle | null {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  for (const alias of styleAliases) {
    if (alias.pattern.test(normalized)) {
      return alias.style;
    }
  }

  return null;
}

function normalizeDestinationValue(value: string): string {
  return value
    .replace(/\bfor\s+\d+\s+days?\b/gi, "")
    .replace(/\bfor\s*$/i, "")
    .replace(/\bwith\b.*$/i, "")
    .replace(/\bfocus(?:ing)?\s+on\b.*$/i, "")
    .replace(/\bplease\b.*$/i, "")
    .replace(/\b(day[- ]by[- ]day|itinerary|budget|accommodation|transport|food|safety)\b.*$/i, "")
    .replace(/[^\p{L}\p{N}\s,'&.-]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[,\s]+|[,\s]+$/g, "");
}

function inferDestinationFromMessage(message: string): string[] {
  const patterns = [
    /\btrip to\s+([A-Za-z][A-Za-z\s.'&-]{1,60})/i,
    /\btravel to\s+([A-Za-z][A-Za-z\s.'&-]{1,60})/i,
    /\bvisit\s+([A-Za-z][A-Za-z\s.'&-]{1,60})/i,
    /\bin\s+([A-Za-z][A-Za-z\s.'&-]{1,60})\s+for\s+\d+\s+days?\b/i,
    /\bfor\s+\d+\s+days?\s+in\s+([A-Za-z][A-Za-z\s.'&-]{1,60})/i,
    /\bto\s+([A-Za-z][A-Za-z\s.'&-]{1,60})\s+for\s+\d+\s+days?\b/i
  ];

  for (const pattern of patterns) {
    const match = message.match(pattern);
    const destination = normalizeDestinationValue(match?.[1] || "");
    if (destination) {
      return [destination];
    }
  }

  return [];
}

function isGreetingMessage(message: string): boolean {
  return /^(hi|hello|hey|hii|yo|good\s+(morning|afternoon|evening))\b/i.test(
    message.trim()
  );
}

function isCapabilitiesMessage(message: string): boolean {
  return /\b(what can you do|how can you help|what do you do|help me explore)\b/i.test(
    message
  );
}

function inferQuestionFocus(message: string, intent: TravelIntent): string | undefined {
  const suppressSecondaryServiceFocus =
    (intent === "plan_trip" || intent === "refine_trip") &&
    (
      /please provide a detailed itinerary/i.test(message) ||
      /\b(include|accommodation suggestions|local cuisine recommendations|transportation options|travel tips|budget estimates|important travel information|safety tips)\b/i.test(
        message
      )
    );

  if (isCapabilitiesMessage(message)) {
    return "capabilities";
  }
  if (isGreetingMessage(message)) {
    return "greeting";
  }
  if (/\b(hidden gems?|unique local activities?|offbeat|lesser[- ]known)\b/i.test(message)) {
    return "hidden_gems";
  }
  if (/\b(beaches?|sunset spots?|sea view)\b/i.test(message)) {
    return "beaches";
  }
  if (/\bseafood\b/i.test(message)) {
    return "seafood";
  }
  if (
    !suppressSecondaryServiceFocus &&
    /\b(restaurants?|food spots?|where to eat|dining|cafes?|cuisine|eat(?:ing|en)?)\b/i.test(message)
  ) {
    return "restaurants";
  }
  if (/\b(cultural attractions?|heritage|museums?|history)\b/i.test(message)) {
    return "culture";
  }
  if (!suppressSecondaryServiceFocus && /\b(hotels?|stays?|accommodation)\b/i.test(message)) {
    return "hotels";
  }
  if (/\b(attractions?|things to do|places to visit|sightseeing|landmarks?|viewpoints?)\b/i.test(message)) {
    return "attractions";
  }
  if (/\b(festivals?|events?|holidays?|best time to go|best dates?)\b/i.test(message)) {
    return "events";
  }
  if (/\b(day trips?|nearby trips?|excursions)\b/i.test(message)) {
    return "day_trips";
  }
  if (/\b(family[- ]friendly|kids?|children)\b/i.test(message)) {
    return "family";
  }
  if (intent === "search_places") {
    return "general";
  }
  return undefined;
}

function getPendingFollowUpContext(
  session: SessionSnapshot
): PendingFollowUpContext | undefined {
  const fromMemory = session.memory.pendingFollowUp;
  if (fromMemory) {
    return fromMemory;
  }

  for (let index = session.messages.length - 1; index >= 0; index -= 1) {
    const message = session.messages[index];
    if (message.role !== "assistant") {
      continue;
    }
    const context = message.meta?.followUpContext as PendingFollowUpContext | undefined;
    if (context) {
      return context;
    }
  }

  return undefined;
}

function isAffirmationMessage(message: string): boolean {
  return /^(yes|yeah|yep|sure|ok|okay|do that|show me|go ahead|sounds good)\b/i.test(
    message.trim()
  );
}

function inferOfferedDomainsFromText(text: string): FollowUpDomain[] {
  const domains = new Set<FollowUpDomain>();
  const normalized = text.toLowerCase();

  if (/\b(stays?|hotels?|accommodation)\b/.test(normalized)) {
    domains.add("stays");
  }
  if (/\b(restaurants?|food|eat|dining|cafes?)\b/.test(normalized)) {
    domains.add("restaurants");
  }
  if (/\b(attractions?|things to do|activities?|places to visit|hidden gems?)\b/.test(normalized)) {
    domains.add("attractions");
  }
  if (/\b(transport|train|bus|drive|flight|get to)\b/.test(normalized)) {
    domains.add("transport");
  }
  if (/\b(events?|festivals?|best dates?|best time)\b/.test(normalized)) {
    domains.add("events");
  }
  if (/\b(dates?|timing)\b/.test(normalized)) {
    domains.add("dates");
  }

  return Array.from(domains);
}

function inferPromptDomain(promptOrLabel: string): FollowUpDomain | undefined {
  const [domain] = inferOfferedDomainsFromText(promptOrLabel);
  return domain;
}

function inferRestaurantCategoryFromMessage(
  message: string
): RestaurantCategoryKey | undefined {
  const normalized = message.toLowerCase();
  if (/\b(cheap|cheaper|budget|budget-friendly|affordable|value)\b/.test(normalized)) {
    return "budget_friendly";
  }
  if (/\b(veg|vegetarian|pure veg)\b/.test(normalized)) {
    return "vegetarian";
  }
  if (/\b(non veg|non-veg|meat|chicken|mutton|seafood|grill|bbq|barbecue)\b/.test(normalized)) {
    return "non_vegetarian";
  }
  if (/\b(premium|rich|fancy|upscale|luxury|fine dining)\b/.test(normalized)) {
    return "premium";
  }
  if (/\b(famous|iconic|must[- ]try|signature|popular)\b/.test(normalized)) {
    return "famous";
  }
  return undefined;
}

function inferDateContext(
  session: SessionSnapshot,
  message: string,
  totalDays?: number
): DateContext {
  const existing = session.memory.dateContext;
  const isoRanges = [
    ...message.matchAll(/\b(\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(\d{4}-\d{2}-\d{2})\b/g)
  ];
  if (isoRanges[0]) {
    return {
      requestedStartDate: isoRanges[0][1],
      requestedEndDate: isoRanges[0][2],
      inferredStartDate: isoRanges[0][1],
      inferredEndDate: isoRanges[0][2],
      flexibility: "exact",
      derivedFrom: "explicit",
      advisoryItems: existing.advisoryItems
    };
  }

  const explicitDates = [...message.matchAll(/\b(\d{4}-\d{2}-\d{2})\b/g)].map(
    (match) => match[1]
  );
  if (explicitDates[0]) {
    const startDate = explicitDates[0];
    const endDate =
      explicitDates[1] || addDaysIso(startDate, Math.max((totalDays || 1) - 1, 0));
    return {
      requestedStartDate: startDate,
      requestedEndDate: endDate,
      inferredStartDate: startDate,
      inferredEndDate: endDate,
      flexibility: explicitDates[1] ? "exact" : totalDays ? "approximate" : "exact",
      derivedFrom: "explicit",
      advisoryItems: existing.advisoryItems
    };
  }

  if (/\bthis weekend\b/i.test(message)) {
    const startDate = nextWeekendStart();
    return {
      requestedStartDate: undefined,
      requestedEndDate: undefined,
      inferredStartDate: startDate,
      inferredEndDate: addDaysIso(startDate, Math.max((totalDays || 2) - 1, 1)),
      flexibility: "approximate",
      derivedFrom: "relative",
      advisoryItems: existing.advisoryItems
    };
  }

  const monthMatch = message.match(
    /\b(?:around|in|during|some time in)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b/i
  );
  if (monthMatch) {
    const startDate = monthStartIso(monthMatch[1], new Date().getUTCFullYear());
    return {
      requestedStartDate: undefined,
      requestedEndDate: undefined,
      inferredStartDate: startDate,
      inferredEndDate: addDaysIso(startDate, Math.max((totalDays || 3) - 1, 2)),
      flexibility: "approximate",
      derivedFrom: "relative",
      advisoryItems: existing.advisoryItems
    };
  }

  if (/\b(best time to go|whenever|sometime|some time)\b/i.test(message)) {
    return {
      ...existing,
      flexibility: "open_ended",
      derivedFrom: existing.derivedFrom === "none" ? "suggested" : existing.derivedFrom
    };
  }

  return existing;
}

function addDaysIso(date: string, days: number): string {
  const next = new Date(`${date}T00:00:00.000Z`);
  next.setUTCDate(next.getUTCDate() + days);
  return next.toISOString().slice(0, 10);
}

function nextWeekendStart(): string {
  const now = new Date();
  const day = now.getUTCDay();
  const daysUntilSaturday = (6 - day + 7) % 7 || 7;
  now.setUTCDate(now.getUTCDate() + daysUntilSaturday);
  return now.toISOString().slice(0, 10);
}

function monthStartIso(monthName: string, year: number): string {
  const index = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december"
  ].indexOf(monthName.toLowerCase());
  const date = new Date(Date.UTC(year, Math.max(index, 0), 1));
  return date.toISOString().slice(0, 10);
}

function localDestinationFromSession(session: SessionSnapshot): string | undefined {
  const acceptedDestination = session.memory.acceptedDecisions
    .find((decision) => decision.startsWith("Destination: "))
    ?.replace(/^Destination:\s*/, "")
    .trim();

  return (
    session.plan?.destination ||
    session.memory.destinationsDiscussed.at(-1) ||
    acceptedDestination ||
    undefined
  );
}

function localOriginFromSession(session: SessionSnapshot): string | undefined {
  const acceptedOrigin = [...session.memory.acceptedDecisions]
    .reverse()
    .find((decision) => decision.startsWith("Origin: "))
    ?.replace(/^Origin:\s*/, "")
    .trim();

  return session.plan?.origin || acceptedOrigin || undefined;
}

function primaryLocationLabel(location?: string): string | undefined {
  if (!location) {
    return undefined;
  }
  const [firstPart] = location
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  return firstPart || location.trim() || undefined;
}

function localTotalDaysFromSession(session: SessionSnapshot): number | undefined {
  const durationDecision = [...session.memory.acceptedDecisions]
    .reverse()
    .find((decision) => decision.startsWith("Duration: "));
  const match = durationDecision?.match(/Duration:\s*(\d+)\s*days?/i);
  if (match) {
    const value = Number.parseInt(match[1], 10);
    if (Number.isFinite(value) && value > 0) {
      return value;
    }
  }

  const startDate = session.memory.dateContext.inferredStartDate;
  const endDate = session.memory.dateContext.inferredEndDate;
  if (!startDate || !endDate) {
    return undefined;
  }

  const start = new Date(`${startDate}T00:00:00.000Z`);
  const end = new Date(`${endDate}T00:00:00.000Z`);
  const diffDays = Math.round((end.getTime() - start.getTime()) / 86_400_000) + 1;
  return diffDays > 0 ? diffDays : undefined;
}

function localTravelerCountFromSession(session: SessionSnapshot): number | undefined {
  const travelerDecision = [...session.memory.acceptedDecisions]
    .reverse()
    .find((decision) => decision.startsWith("Travelers: "));
  const match = travelerDecision?.match(/Travelers:\s*(\d+)/i);
  if (!match) {
    return undefined;
  }

  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

export function resolveDiscoveryFocus(resolution: TurnResolution): DiscoveryFocus {
  switch (resolution.questionFocus) {
    case "greeting":
      return "greeting";
    case "capabilities":
      return "capabilities";
    case "hidden_gems":
      return "hidden_gems";
    case "beaches":
      return "beaches";
    case "seafood":
      return "seafood";
    case "restaurants":
      return "restaurants";
    case "attractions":
      return "attractions";
    case "culture":
      return "culture";
    case "hotels":
      return "hotels";
    case "day_trips":
      return "day_trips";
    case "family":
      return "family";
    case "events":
      return "general";
    default:
      return "general";
  }
}

/**
 * Determines whether the current turn resolution requires a live Places +
 * Tavily research pass before generating a response.
 *
 * Rules:
 *  - plan_trip / refine_trip  → always research (need POIs for itinerary)
 *  - search_places            → always research
 *  - question with a place / POI focus → research so we can show real cards
 *    (restaurants, attractions, culture, beaches, hidden_gems, seafood, hotels,
 *    day_trips, family, greeting, capabilities)
 *  - question about events → no POI research; date-context agent handles it
 *  - pure meta / settings     → no research
 */
export function shouldResearchResolution(resolution: TurnResolution): boolean {
  if (resolution.destinations.length === 0) {
    return false;
  }

  if (
    resolution.intent === "plan_trip" ||
    resolution.intent === "refine_trip"
  ) {
    return !!resolution.totalDays;
  }

  if (resolution.intent === "search_places") {
    return true;
  }

  if (resolution.intent === "question") {
    if (resolution.questionFocus === "followup_clarify") {
      return false;
    }
    // POI-focused questions need real cards — research them
    const poiFocuses = [
      "restaurants",
      "attractions",
      "culture",
      "beaches",
      "hidden_gems",
      "seafood",
      "hotels",
      "day_trips",
      "family"
    ];
    return poiFocuses.includes(resolution.questionFocus || "");
  }

  return false;
}

function inferTotalDays(message: string): number | undefined {
  const match = message.match(/\b(\d+)\s*[- ]?\s*days?\b/i);
  if (!match) {
    return undefined;
  }

  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

function inferTravelerCount(message: string): number | undefined {
  const digitMatch = message.match(/\bfor\s+(\d+)\s+(travellers?|travelers?|people|persons|adults?)\b/i);
  if (digitMatch) {
    const value = Number.parseInt(digitMatch[1], 10);
    return Number.isFinite(value) && value > 0 ? value : undefined;
  }

  if (/\b(solo|just me|myself)\b/i.test(message)) {
    return 1;
  }

  if (/\b(couple|two of us)\b/i.test(message)) {
    return 2;
  }

  return undefined;
}

function isComprehensivePlanningRequest(
  message: string,
  rememberedDestination?: string,
  rememberedTotalDays?: number
): boolean {
  const normalized = message.toLowerCase();
  const asksForPlan = /\b(plan|itinerary|trip)\b/.test(normalized);
  const hasDurationCue =
    /\b(\d+\s*days?|weekend|day[- ]by[- ]day)\b/.test(normalized) ||
    Boolean(rememberedDestination && rememberedTotalDays);
  const hasPlanningDeliverables =
    /\b(detailed itinerary|day[- ]by[- ]day schedule|day[- ]by[- ]day|accommodation suggestions|transportation options|travel tips|budget estimates|important travel information|safety tips)\b/i.test(
      message
    );
  const hasStructuredRequirements =
    /please provide a detailed itinerary/i.test(message) ||
    /\n[-•]/.test(message);

  return asksForPlan && (hasDurationCue || hasPlanningDeliverables || hasStructuredRequirements);
}

function inferStyles(message: string): TurnResolution["styles"] {
  const styles = new Set<TurnResolution["styles"][number]>();
  for (const cue of styleCueMatchers) {
    if (cue.pattern.test(message)) {
      styles.add(cue.style);
    }
  }
  return Array.from(styles);
}

function inferIntentFromMessage(
  session: SessionSnapshot,
  message: string,
  current: TravelIntent
): TravelIntent {
  const normalized = message.toLowerCase();
  const rememberedDestination = localDestinationFromSession(session);
  const rememberedTotalDays = localTotalDaysFromSession(session);
  const pendingFollowUp = getPendingFollowUpContext(session);
  const asksForStays = /\b(show|find|recommend|best)\b.*\b(hotels?|stays?|accommodation)\b|\bwhere should i stay\b/.test(
    normalized
  );
  const affirmation = isAffirmationMessage(message);
  const refineCue =
    /\b(add|remove|replace|swap|move|adjust|refine|change|update|tweak|slower|relax|rebalance|upgrade)\b/.test(
      normalized
    ) && /\b(day\s*\d+|itinerary|plan|trip|stay|dinner|lunch|breakfast|pace)\b/.test(normalized);

  if (session.plan && refineCue) {
    return "refine_trip";
  }

  if (
    pendingFollowUp &&
    (affirmation || Boolean(inferRestaurantCategoryFromMessage(message)))
  ) {
    if (pendingFollowUp.primaryDomain === "restaurants") {
      return "search_places";
    }
    if (pendingFollowUp.primaryDomain === "stays") {
      return "search_places";
    }
    if (
      pendingFollowUp.primaryDomain === "attractions" ||
      pendingFollowUp.primaryDomain === "activities"
    ) {
      return "search_places";
    }
    return "question";
  }

  if (asksForStays) {
    return "search_places";
  }

  if (isCapabilitiesMessage(message) || isGreetingMessage(message)) {
    return "question";
  }

  const comprehensivePlanningRequest = isComprehensivePlanningRequest(
    message,
    rememberedDestination,
    rememberedTotalDays
  );
  if (current === "plan_trip" && comprehensivePlanningRequest) {
    return "plan_trip";
  }

  const itineraryFollowUp =
    /\b(generate|build|create|make|show|give|need|want)\b.*\b(itinerary|trip plan|plan)\b/.test(
      normalized
    ) ||
    /\bwhere(?:'s| is)\s+(?:the\s+)?itinerary\b/.test(normalized) ||
    /\b(itinerary|trip plan)\s+(?:now|please)\b/.test(normalized);
  if (
    comprehensivePlanningRequest ||
    (!session.plan && itineraryFollowUp && rememberedDestination && rememberedTotalDays)
  ) {
    return "plan_trip";
  }

  const explicitDiscovery =
    /\b(hidden gems?|unique local activities?|best beaches?|seafood|restaurants?|cultural attractions?|heritage|museums?|day trips?|hotels?|stays?|family[- ]friendly)\b/.test(
      normalized
    );
  if (explicitDiscovery) {
    return "search_places";
  }

  const asksForPlan =
    /\b(plan|itinerary|trip)\b/.test(normalized) &&
    (
      /\b(\d+\s*days?|weekend|day-by-day)\b/.test(normalized) ||
      (!!rememberedDestination && !!rememberedTotalDays)
    );
  if (asksForPlan) {
    return "plan_trip";
  }

  return current;
}

function finalizeResolution(
  session: SessionSnapshot,
  message: string,
  parsed: z.infer<typeof resolverSchema>
): TurnResolution {
  const pendingFollowUp = getPendingFollowUpContext(session);
  const inferredIntent = inferIntentFromMessage(session, message, parsed.intent);
  const heuristicFocus = inferQuestionFocus(message, inferredIntent);
  const requestedRestaurantCategory = inferRestaurantCategoryFromMessage(message);
  const affirmation = isAffirmationMessage(message);
  const inferredFollowUpOptions = pendingFollowUp?.options || [];
  const followUpAmbiguous =
    Boolean(pendingFollowUp) &&
    affirmation &&
    !pendingFollowUp?.primaryDomain &&
    inferredFollowUpOptions.length > 1 &&
    !requestedRestaurantCategory;
  const followUpDomain =
    pendingFollowUp?.primaryDomain ||
    (requestedRestaurantCategory ? "restaurants" : undefined);
  const pendingFocus =
    pendingFollowUp?.primaryDomain === "stays"
      ? "hotels"
      : pendingFollowUp?.primaryDomain === "restaurants"
        ? "restaurants"
        : pendingFollowUp?.primaryDomain === "attractions" ||
            pendingFollowUp?.primaryDomain === "activities"
          ? "attractions"
          : pendingFollowUp?.primaryDomain === "events"
            ? "events"
            : pendingFollowUp?.primaryDomain === "transport"
              ? "transport"
              : undefined;
  const inferredFocus =
    followUpAmbiguous
      ? "followup_clarify"
      : (pendingFollowUp &&
          (affirmation || Boolean(requestedRestaurantCategory)) &&
          pendingFocus)
        ? pendingFocus
        : heuristicFocus || parsed.questionFocus;
  const inferredDestinations = inferDestinationFromMessage(message);
  const explicitNewTrip =
    inferredIntent === "plan_trip" &&
    /\b(plan|itinerary|trip)\b/i.test(message) &&
    inferredDestinations.length > 0;
  const rawDestinations = Array.from(
    new Set(
      [
        ...normalizeDestinations(parsed),
        ...inferredDestinations,
        ...(pendingFollowUp?.destination ? [pendingFollowUp.destination] : []),
        ...(explicitNewTrip ? [] : session.memory.destinationsDiscussed.slice(-2))
      ].filter(Boolean)
    )
  );
  const localDestination = localDestinationFromSession(session);
  const destinations =
    rawDestinations.length > 0
      ? rawDestinations
      : inferredFocus && ["greeting", "capabilities"].includes(inferredFocus) && localDestination
        ? [localDestination]
        : rawDestinations;
  const inferredDays = inferTotalDays(message);
  const inferredTravelers = inferTravelerCount(message);
  const inferredStyles = inferStyles(message);
  const dateContext = inferDateContext(
    session,
    message,
    parsed.totalDays ||
      inferredDays ||
      session.plan?.totalDays ||
      localTotalDaysFromSession(session)
  );
  const parsedStyles = parsed.styles
    .map((style) => normalizeTravelStyle(style))
    .filter((style): style is TravelStyle => Boolean(style));

  return {
    ...parsed,
    intent: inferredIntent,
    destination: parsed.destination || destinations[0],
    destinations,
    totalDays:
      parsed.totalDays ||
      inferredDays ||
      session.plan?.totalDays ||
      localTotalDaysFromSession(session),
    travelerCount:
      parsed.travelerCount ||
      inferredTravelers ||
      session.plan?.travelerCount ||
      localTravelerCountFromSession(session),
    styles: Array.from(new Set([...parsedStyles, ...inferredStyles])),
    questionFocus: inferredFocus,
    dateContext,
    stayMode:
      inferredFocus === "hotels" ||
      (pendingFollowUp?.primaryDomain === "stays" &&
        (affirmation || Boolean(requestedRestaurantCategory))),
    explicitNewTrip,
    followUpDomain,
    restaurantCategory: requestedRestaurantCategory,
    followUpAmbiguous,
    followUpOptions: inferredFollowUpOptions
  };
}

function shouldResolveDeterministically(
  session: SessionSnapshot,
  message: string
): boolean {
  const trimmed = message.trim();
  if (!trimmed) {
    return false;
  }

  const inferredIntent = inferIntentFromMessage(session, trimmed, "question");
  const inferredFocus = inferQuestionFocus(trimmed, inferredIntent);
  const inferredDestinations = inferDestinationFromMessage(trimmed);
  const pendingFollowUp = getPendingFollowUpContext(session);
  const hasLocalDestination = Boolean(localDestinationFromSession(session));
  const hasLocalDuration = Boolean(localTotalDaysFromSession(session));
  const requestedRestaurantCategory = inferRestaurantCategoryFromMessage(trimmed);
  const referentialFollowUp =
    /\b(that one|those|this one|more|show me more|another one|another|similar|like this)\b/i.test(
      trimmed
    );

  if (
    pendingFollowUp &&
    (isAffirmationMessage(trimmed) ||
      Boolean(requestedRestaurantCategory) ||
      referentialFollowUp)
  ) {
    return true;
  }

  if (session.plan && inferredIntent === "refine_trip") {
    return true;
  }

  if (
    inferredIntent === "search_places" &&
    Boolean(inferredFocus) &&
    (inferredDestinations.length > 0 || hasLocalDestination || Boolean(pendingFollowUp))
  ) {
    return true;
  }

  if (
    inferredIntent === "plan_trip" &&
    ((inferredDestinations.length > 0 &&
      (Boolean(inferTotalDays(trimmed)) || /\b(weekend|day[- ]by[- ]day)\b/i.test(trimmed))) ||
      (hasLocalDestination &&
        hasLocalDuration &&
        /\b(plan|itinerary|trip)\b/i.test(trimmed)))
  ) {
    return true;
  }

  if (
    inferredIntent === "question" &&
    inferredFocus === "followup_clarify"
  ) {
    return true;
  }

  if (
    inferredIntent === "question" &&
    Boolean(inferredFocus) &&
    (inferredDestinations.length > 0 || hasLocalDestination)
  ) {
    return true;
  }

  return false;
}

export function resolveDeterministicTurnIntent(
  session: SessionSnapshot,
  message: string
): TurnResolution | null {
  if (!shouldResolveDeterministically(session, message)) {
    return null;
  }

  const inferredDestinations = inferDestinationFromMessage(message);
  const heuristicFocus = inferQuestionFocus(message, "question");

  return finalizeResolution(session, message, {
    intent: "question",
    destination: inferredDestinations[0],
    destinations: inferredDestinations,
    totalDays: inferTotalDays(message),
    travelerCount: inferTravelerCount(message),
    styles: [],
    questionFocus: heuristicFocus
  });
}

const resolverSchema = z.object({
  intent: z.preprocess(normalizeTravelIntent, travelIntentSchema),
  destination: z.string().optional(),
  destinations: z.array(z.string()).default([]),
  origin: z.string().optional(),
  totalDays: z.number().int().positive().optional(),
  travelerCount: z.number().int().positive().optional(),
  budgetNote: z.string().optional(),
  styles: z.array(z.string()).default([]),
  questionFocus: z.string().optional()
});

const planDraftSchema = z.object({
  title: z.string(),
  destination: z.string().optional(),
  destinations: z.array(z.string()).default([]),
  totalDays: z.number().int().positive(),
  travelerCount: z.number().int().positive().default(1),
  notes: z.array(z.string()).default([]),
  destinationSegments: z.array(
    z.object({
      destination: z.string(),
      startDay: z.number().int().positive(),
      endDay: z.number().int().positive(),
      nights: z.number().int().nonnegative()
    })
  ),
  days: z.array(
    z.object({
      day: z.number().int().positive(),
      title: z.string(),
      theme: z.string().optional(),
      summary: z.string().optional(),
      destination: z.string(),
      accommodationPoiId: z.string().optional(),
      activities: z.array(
        z.object({
          id: z.string().optional(),
          poiId: z.string().optional(),
          title: z.string(),
          summary: z.string().optional(),
          startTime: z.string(),
          endTime: z.string(),
          notes: z.array(z.string()).default([])
        })
      )
    })
  )
});

const assistantNarrativeSchema = z.object({
  introEyebrow: z.string().optional(),
  introTitle: z.string(),
  introBody: z.string(),
  moodEmoji: z.string().optional(),
  leadText: z.string(),
  promptChips: z.array(
    z.object({
      label: z.string(),
      prompt: z.string()
    })
  ).max(4).default([]),
  clarifyingQuestions: z.array(z.string()).max(3).default([])
});

type AssistantNarrative = z.infer<typeof assistantNarrativeSchema>;

function buildConversationDigest(session: SessionSnapshot): string {
  const recentMessages = session.messages
    .slice(-8)
    .map((message) => `${message.role}: ${message.content}`)
    .join("\n");

  return [
    `Session title: ${session.title}`,
    `Current destinations: ${session.plan?.destinations.join(", ") || "none"}`,
    `Current dates: ${session.plan?.startDate || session.memory.dateContext.inferredStartDate || "none"} to ${session.plan?.endDate || session.memory.dateContext.inferredEndDate || "none"}`,
    `Pending follow-up: ${session.memory.pendingFollowUp ? JSON.stringify(session.memory.pendingFollowUp) : "none"}`,
    `Accepted decisions: ${session.memory.acceptedDecisions.join(" | ") || "none"}`,
    `Preferences: ${session.memory.preferences.styles.join(", ") || "none"}`,
    `Recent messages:\n${recentMessages || "none"}`
  ].join("\n");
}

function nextDate(index: number, startDate?: string): string {
  const date = startDate ? new Date(`${startDate}T00:00:00.000Z`) : new Date();
  date.setUTCDate(date.getUTCDate() + index);
  return date.toISOString().slice(0, 10);
}

function isCanonicalPoiId(catalog: PoiCatalog, poiId?: string): poiId is string {
  return Boolean(poiId && catalog.items[poiId]);
}

const BUDGET_EXPECTATION_BANDS = [
  { label: "Budget-friendly", maxPerTravelerPerDay: 3500 },
  { label: "Mid-range", maxPerTravelerPerDay: 7000 },
  { label: "Comfortable", maxPerTravelerPerDay: 12000 },
  { label: "Premium", maxPerTravelerPerDay: Number.POSITIVE_INFINITY }
] as const;

function inferBudgetExpectation(plan?: PlanSnapshot): string | undefined {
  const total =
    plan?.budgetTarget?.total ||
    plan?.budget?.total;
  const days = plan?.totalDays || plan?.days.length || 0;
  const travelers = Math.max(plan?.travelerCount || 1, 1);

  if (!total || days <= 0) {
    return undefined;
  }

  const perTravelerPerDay = total / days / travelers;
  return (
    BUDGET_EXPECTATION_BANDS.find(
      (band) => perTravelerPerDay <= band.maxPerTravelerPerDay
    )?.label || "Mid-range"
  );
}

function formatBudgetLabel(plan?: PlanSnapshot): string | undefined {
  const expectation = inferBudgetExpectation(plan);
  if (expectation) {
    return `Expected spend: ${expectation}`;
  }
  return undefined;
}

function periodForTimeLabel(time?: string): "morning" | "afternoon" | "evening" | "flex" {
  if (!time) {
    return "flex";
  }

  const match = time.match(/^(\d{1,2}):(\d{2})$/);
  if (!match) {
    return "flex";
  }

  const hour = Number.parseInt(match[1], 10);
  if (hour < 12) {
    return "morning";
  }
  if (hour < 17) {
    return "afternoon";
  }
  return "evening";
}

function labelForPeriod(period: "morning" | "afternoon" | "evening" | "flex") {
  switch (period) {
    case "morning":
      return { label: "Morning", emoji: "☀️" };
    case "afternoon":
      return { label: "Afternoon", emoji: "🧡" };
    case "evening":
      return { label: "Evening", emoji: "🌙" };
    default:
      return { label: "Flexible", emoji: "✨" };
  }
}

function normalizeDestinations(resolution: z.infer<typeof resolverSchema>): string[] {
  const set = new Set(
    [resolution.destination, ...resolution.destinations].filter(Boolean)
  );
  return Array.from(set) as string[];
}

function aggregateCatalog(research: DestinationResearch[]): DestinationResearch {
  const combined: DestinationResearch = {
    destinations: [],
    focus: research[0]?.focus || "general",
    catalog: { version: 1, items: {} },
    grouped: { stays: [], restaurants: [], attractions: [] },
    facts: []
  };

  for (const chunk of research) {
    combined.destinations.push(...chunk.destinations);
    combined.grouped.stays.push(...chunk.grouped.stays);
    combined.grouped.restaurants.push(...chunk.grouped.restaurants);
    combined.grouped.attractions.push(...chunk.grouped.attractions);
    combined.facts.push(...chunk.facts);
    Object.assign(combined.catalog.items, chunk.catalog.items);
  }

  return combined;
}

export async function researchPlanningDestinations(
  tools: TravelToolsService,
  destinations: string[],
  focus: DiscoveryFocus = "general",
  includeDeepResearch = false
): Promise<DestinationResearch> {
  const baseline = await researchDestinations(
    tools,
    destinations,
    "general",
    includeDeepResearch
  );

  if (
    focus === "general" ||
    focus === "greeting" ||
    focus === "capabilities"
  ) {
    return baseline;
  }

  const focused = await researchDestinations(tools, destinations, focus, false);
  return aggregateCatalog([baseline, focused]);
}

function scorePoi(
  poi: Poi,
  resolution: TurnResolution,
  category: "stay" | "restaurant" | "attraction"
): number {
  const ratingScore = (poi.rating || 0) * 20;
  const priceFit =
    resolution.styles.includes("luxury")
      ? (poi.priceLevel || 0) * 6
      : resolution.styles.includes("budget")
        ? 20 - (poi.priceLevel || 0) * 5
        : 10;
  const stayBoost =
    category === "stay" && resolution.stayMode ? 20 : category === "attraction" && resolution.questionFocus === "hidden_gems" ? 12 : 0;
  const metadataBoost = poi.photoUrl ? 6 : 0;
  return ratingScore + priceFit + stayBoost + metadataBoost;
}

function sortPois(
  pois: Poi[],
  resolution: TurnResolution,
  category: "stay" | "restaurant" | "attraction"
): Poi[] {
  return [...pois].sort((left, right) => scorePoi(right, resolution, category) - scorePoi(left, resolution, category));
}

function matchesDestinationName(poi: Poi, destination: string): boolean {
  const haystack = `${poi.name} ${poi.address || ""} ${(poi.tags || []).join(" ")}`
    .toLowerCase();
  return haystack.includes(destination.toLowerCase());
}

function destinationSortedPois(
  pois: Poi[],
  resolution: TurnResolution,
  category: "stay" | "restaurant" | "attraction",
  destination: string
): Poi[] {
  const sorted = sortPois(pois, resolution, category);
  const matched = sorted.filter((poi) => matchesDestinationName(poi, destination));
  return matched.length > 0 ? matched : sorted;
}

const MEAL_ACTIVITY_PATTERN =
  /\b(breakfast|brunch|lunch|dinner|meal|cafe|coffee|snack|eat)\b/i;
const STAY_ACTIVITY_PATTERN =
  /\b(check[- ]?in|check in|hotel|resort|stay|accommodation|settle in)\b/i;

function labelForMealActivity(activity: { title: string; startTime: string }): string {
  const title = activity.title.toLowerCase();
  if (/\bbreakfast|brunch|coffee|cafe\b/.test(title)) {
    return "Breakfast";
  }
  if (/\bdinner\b/.test(title)) {
    return "Dinner";
  }
  if (/\blunch\b/.test(title)) {
    return "Lunch";
  }

  const period = periodForTimeLabel(activity.startTime);
  if (period === "evening") {
    return "Dinner";
  }
  if (period === "morning") {
    return "Breakfast";
  }
  return "Lunch";
}

function findMealInsertionIndex(
  activities: Array<{ startTime: string }>
): number {
  const afternoonIndex = activities.findIndex(
    (activity) => periodForTimeLabel(activity.startTime) === "afternoon"
  );
  if (afternoonIndex >= 0) {
    return afternoonIndex;
  }

  const morningActivities = activities.filter(
    (activity) => periodForTimeLabel(activity.startTime) === "morning"
  ).length;
  return Math.min(Math.max(morningActivities, 1), activities.length);
}

export function applyPlanHospitalitySelections(
  plan: PlanSnapshot,
  resolution: TurnResolution,
  research: DestinationResearch
): PlanSnapshot {
  const rankedRestaurants = sortPois(
    research.grouped.restaurants,
    resolution,
    "restaurant"
  );
  const rankedStays = sortPois(research.grouped.stays, resolution, "stay");
  if (!rankedRestaurants.length && !rankedStays.length) {
    return plan;
  }

  const catalog = research.catalog;
  const usedRestaurantIds = new Set<string>();
  const days = plan.days.map((day, index) => {
    let activities = [...day.activities];
    let accommodationPoiId = day.accommodationPoiId;
    const restaurantsForDay = destinationSortedPois(
      rankedRestaurants,
      resolution,
      "restaurant",
      day.destination
    );
    const staysForDay = destinationSortedPois(
      rankedStays,
      resolution,
      "stay",
      day.destination
    );

    const selectedRestaurant =
      restaurantsForDay.find((poi) => !usedRestaurantIds.has(poi.id)) ||
      restaurantsForDay[0];
    if (selectedRestaurant) {
      const mealIndex = activities.findIndex((activity) => {
        const activityPoi = activity.poiId ? catalog.items[activity.poiId] : undefined;
        return activityPoi?.type === "restaurant" || MEAL_ACTIVITY_PATTERN.test(activity.title);
      });

      if (mealIndex >= 0) {
        const existing = activities[mealIndex];
        const existingPoi = existing.poiId ? catalog.items[existing.poiId] : undefined;
        if (existingPoi?.type !== "restaurant") {
          const mealLabel = labelForMealActivity(existing);
          activities[mealIndex] = {
            ...existing,
            poiId: selectedRestaurant.id,
            title: `${mealLabel} at ${selectedRestaurant.name}`,
            summary:
              existing.summary ||
              `Use ${selectedRestaurant.name} for a dependable local ${mealLabel.toLowerCase()} stop in ${day.destination}.`
          };
        }
        usedRestaurantIds.add(selectedRestaurant.id);
      } else if (activities.length < 5) {
        const insertionIndex = findMealInsertionIndex(activities);
        activities.splice(insertionIndex, 0, {
          id: randomUUID(),
          poiId: selectedRestaurant.id,
          title: `Lunch at ${selectedRestaurant.name}`,
          summary: `Pause at ${selectedRestaurant.name} for a well-placed local meal without breaking the route.`,
          startTime: "13:00",
          endTime: "14:15",
          notes: []
        });
        usedRestaurantIds.add(selectedRestaurant.id);
      }
    }

    const isOvernightDay = plan.days.length > 1 && index < plan.days.length - 1;
    const selectedStay = staysForDay[0];
    if (selectedStay && isOvernightDay) {
      accommodationPoiId = accommodationPoiId || selectedStay.id;
      const stayActivityIndex = activities.findIndex((activity) => {
        const activityPoi = activity.poiId ? catalog.items[activity.poiId] : undefined;
        return activityPoi?.type === "stay" || STAY_ACTIVITY_PATTERN.test(activity.title);
      });

      if (stayActivityIndex >= 0) {
        const existing = activities[stayActivityIndex];
        const existingPoi = existing.poiId ? catalog.items[existing.poiId] : undefined;
        if (existingPoi?.type !== "stay") {
          activities[stayActivityIndex] = {
            ...existing,
            poiId: selectedStay.id,
            title: `Check in at ${selectedStay.name}`,
            summary:
              existing.summary ||
              `Settle in at ${selectedStay.name} so the rest of the trip stays easy and family-friendly.`
          };
        }
      } else if (activities.length < 5) {
        activities.push({
          id: randomUUID(),
          poiId: selectedStay.id,
          title: `Check in at ${selectedStay.name}`,
          summary: `Use ${selectedStay.name} as your base for the night in ${day.destination}.`,
          startTime: "18:30",
          endTime: "19:15",
          notes: []
        });
      }
    }

    return {
      ...day,
      accommodationPoiId,
      activities
    };
  });

  return { ...plan, days };
}

function mergeAdvisories(...groups: Array<DateAdvisoryItem[] | undefined>): DateAdvisoryItem[] {
  return Array.from(
    new Map(
      groups
        .flat()
        .filter((item): item is DateAdvisoryItem => Boolean(item))
        .map((item) => [`${item.kind}:${item.title}:${item.startDate || ""}`, item] as const)
    ).values()
  );
}

export async function resolveTurnIntent(
  providerService: ProviderService,
  resolvedProvider: ResolvedProvider,
  session: SessionSnapshot,
  message: string
): Promise<TurnResolution> {
  const digest = buildConversationDigest(session);
  const parsed = await providerService.generateObject({
    resolved: resolvedProvider,
    schema: resolverSchema,
    schemaName: "turn_resolution",
    instructions:
      "You are the intent and slot resolver for a travel planning assistant. Be conservative and return structured travel intent only from the current session context. Return JSON only. The intent value must be exactly one of: plan_trip, refine_trip, search_places, question, settings, meta. Never emit legacy labels, uppercase variants, or extra commentary.",
    input: `Conversation digest:\n${digest}\n\nUser message:\n${message}`
  });

  return finalizeResolution(session, message, parsed);
}

/**
 * Research one or more destinations by fetching live Google Places data and
 * optionally augmenting with deep Tavily editorial facts.
 *
 * The `focus` parameter controls:
 *  - Which Google Places query templates are fired (from `discoveryQueries`)
 *  - Which POI type filter is applied (preventing category bleed)
 *  - Which response blocks are rendered on the frontend
 *
 * @param tools - Instantiated TravelToolsService
 * @param destinations - Array of destination strings
 * @param focus - Discovery focus (restaurants | attractions | general | …)
 * @param includeDeepResearch - Set true to also fetch Tavily editorial content
 */
export async function researchDestinations(
  tools: TravelToolsService,
  destinations: string[],
  focus: DiscoveryFocus = "general",
  includeDeepResearch = false
): Promise<DestinationResearch> {
  const chunks = await Promise.all(
    destinations.map(async (destination) => {
      const result = await tools.searchPlacesForDestination(destination, focus);

      // Deep web research: pull Tavily editorial facts only when opted in.
      // This adds ~1-2 s for a much richer set of facts the LLM can cite.
      const factsPromise = includeDeepResearch
        ? tools.deepWebResearch(destination, topicForFocus(focus))
        : tools.getDestinationFacts(destination);

      const facts = await factsPromise;

      return {
        destinations: [destination],
        focus,
        catalog: result.catalog,
        grouped: {
          stays: result.stays,
          restaurants: result.restaurants,
          attractions: result.attractions
        },
        facts
      };
    })
  );

  return aggregateCatalog(chunks);
}

/**
 * Maps a DiscoveryFocus to a Tavily query topic string.
 * Used when `includeDeepResearch` is true.
 */
function topicForFocus(focus: DiscoveryFocus): string {
  switch (focus) {
    case "restaurants": return "best restaurants local food guide";
    case "seafood": return "best seafood restaurants local guide";
    case "attractions": return "top tourist attractions things to do";
    case "culture": return "cultural heritage museums history";
    case "beaches": return "best beaches coastal travel guide";
    case "hidden_gems": return "hidden gems off the beaten path local tips";
    case "hotels": return "best hotels accommodation guide";
    case "day_trips": return "best day trips nearby excursions";
    case "family": return "family friendly activities kids travel";
    default: return "travel guide tips local culture food attractions";
  }
}

export async function synthesizePlan(
  providerService: ProviderService,
  resolvedProvider: ResolvedProvider,
  session: SessionSnapshot,
  resolution: TurnResolution,
  research: DestinationResearch
): Promise<PlanSnapshot> {
  const rankedStays = sortPois(research.grouped.stays, resolution, "stay");
  const rankedRestaurants = sortPois(
    research.grouped.restaurants,
    resolution,
    "restaurant"
  );
  const poiSummary = JSON.stringify(
    {
      stays: rankedStays.map((poi) => ({
        id: poi.id,
        name: poi.name,
        address: poi.address,
        priceLevel: poi.priceLevel,
        rating: poi.rating
      })),
      restaurants: rankedRestaurants.map((poi) => ({
        id: poi.id,
        name: poi.name,
        address: poi.address,
        priceLevel: poi.priceLevel,
        rating: poi.rating
      })),
      attractions: sortPois(research.grouped.attractions, resolution, "attraction").map((poi) => ({
        id: poi.id,
        name: poi.name,
        rating: poi.rating
      }))
    },
    null,
    2
  );

  const digest = buildConversationDigest(session);
  const draft = await providerService.generateObject({
    resolved: resolvedProvider,
    schema: planDraftSchema,
    schemaName: "plan_draft",
    instructions:
      "You are the itinerary synthesis agent. Build a realistic plan that references the provided POI ids. Keep the output purely JSON.",
    input: `Conversation digest:\n${digest}

Resolved travel request:
${JSON.stringify(resolution, null, 2)}

Available POIs:
${poiSummary}

Rules:
- Use the exact poiId values when you reference a POI.
- Create a plan with ${resolution.totalDays || session.plan?.totalDays || 3} days.
- Keep each day feasible, with 3 to 5 activities.
- Keep the same destination structure the user asked for.
- When stay POIs are available for an overnight stop, set accommodationPoiId and anchor the day to a specific stay choice.
- When restaurant POIs are available, use them for meal stops instead of leaving meals generic.
- If the traveller asked for accommodation, food, local cuisine, or a family-friendly plan, treat stay and restaurant choices as required parts of the itinerary, not optional side notes.
- Do not fabricate POIs that are not in the list.`
  });

  const totalDays = draft.totalDays || resolution.totalDays || 3;
  const version = (session.plan?.version || 0) + 1;
  const normalized = planSnapshotSchema.parse({
    schemaVersion: 1,
    sessionId: session.id,
    version,
    title: draft.title,
    origin: resolution.origin || session.plan?.origin || session.memory.preferences.homeAirport,
    destination: draft.destination || resolution.destination || research.destinations[0],
    destinations:
      draft.destinations.length > 0 ? draft.destinations : research.destinations,
    startDate: resolution.dateContext.inferredStartDate,
    endDate:
      resolution.dateContext.inferredEndDate ||
      (resolution.dateContext.inferredStartDate
        ? addDaysIso(
            resolution.dateContext.inferredStartDate,
            Math.max(totalDays - 1, 0)
          )
        : undefined),
    totalDays,
    travelerCount:
      resolution.travelerCount || draft.travelerCount || session.plan?.travelerCount || 1,
    notes: draft.notes,
    destinationSegments: draft.destinationSegments,
    days: draft.days.map((day, index) => ({
      ...day,
      date: nextDate(index, resolution.dateContext.inferredStartDate),
      activities: day.activities.map((activity) => ({
        ...activity,
        id: activity.id || randomUUID()
      }))
    })),
    generatedAt: new Date().toISOString(),
    lastUserIntent: resolution.intent
  });

  return applyPlanHospitalitySelections(normalized, resolution, research);
}

export async function enrichPlanLogistics(
  tools: TravelToolsService,
  plan: PlanSnapshot,
  catalog: PoiCatalog
): Promise<PlanSnapshot> {
  const days = await Promise.all(
    plan.days.map(async (day) => {
      const nextActivities = [...day.activities];
      for (let index = 1; index < nextActivities.length; index += 1) {
        const previousPoiId = nextActivities[index - 1].poiId;
        const currentPoiId = nextActivities[index].poiId;
        const previousPoi = previousPoiId
          ? catalog.items[previousPoiId]
          : undefined;
        const currentPoi = currentPoiId
          ? catalog.items[currentPoiId]
          : undefined;
        if (!previousPoi || !currentPoi) {
          continue;
        }

        const route = await tools.estimateRoute(previousPoi, currentPoi);
        if (route) {
          nextActivities[index].travelTimeMinutesFromPrevious =
            route.durationMinutes;
        } else {
          delete nextActivities[index].travelTimeMinutesFromPrevious;
        }
      }

      const accommodation = day.accommodationPoiId
        ? catalog.items[day.accommodationPoiId]
        : undefined;
      const budget = {
        accommodation: accommodation?.priceLevel ? accommodation.priceLevel * 2500 : 2500,
        food: 1400,
        transport: nextActivities.reduce(
          (total, activity) => total + (activity.travelTimeMinutesFromPrevious || 0) * 4,
          300
        ),
        activities: nextActivities.length * 350,
        misc: 500,
        total: 0,
        currency: "INR"
      };
      budget.total =
        budget.accommodation +
        budget.food +
        budget.transport +
        budget.activities +
        budget.misc;

      return {
        ...day,
        activities: nextActivities,
        budget
      };
    })
  );

  const total = days.reduce(
    (sum, day) => sum + (day.budget?.total || 0),
    0
  );

  return {
    ...plan,
    days,
    budget: {
      accommodation: days.reduce((sum, day) => sum + (day.budget?.accommodation || 0), 0),
      food: days.reduce((sum, day) => sum + (day.budget?.food || 0), 0),
      transport: days.reduce((sum, day) => sum + (day.budget?.transport || 0), 0),
      activities: days.reduce((sum, day) => sum + (day.budget?.activities || 0), 0),
      misc: days.reduce((sum, day) => sum + (day.budget?.misc || 0), 0),
      total,
      currency: "INR"
    }
  };
}

/**
 * Feasibility Critic sub-agent — analyses a synthesised plan and emits
 * lightweight text fixes to obvious logistical problems WITHOUT making a full
 * re-plan call (keeps latency low).
 *
 * Problems it detects and patches:
 *  - Too many stops per day (> 5 activities → trim weakest)
 *  - Back-to-back long-distance legs (> 120 min travel between activities)
 *  - Departure day has too many activities (should be light/travel day)
 *  - Missing accommodation on multi-day trips (fills from catalog)
 *
 * Returns the (possibly patched) plan plus a list of human-readable critique
 * strings that turn-runner.ts injects as agent traces.
 *
 * @param plan    - Plan produced by synthesizePlan + enrichPlanLogistics
 * @param catalog - Canonical POI catalog for the session
 */
export function criticizeAndRefinePlan(
  plan: PlanSnapshot,
  catalog: PoiCatalog
): { plan: PlanSnapshot; critiques: string[] } {
  const critiques: string[] = [];
  const days = plan.days.map((day, idx) => {
    let activities = [...day.activities];

    // Rule 1: cap activities at 5 per day
    if (activities.length > 5) {
      const removed = activities.splice(5).map((a) => a.title);
      critiques.push(
        `Day ${day.day}: Trimmed ${removed.length} over-scheduled stop(s) (${removed.join(", ")}) to keep the day comfortable.`
      );
    }

    // Rule 2: flag long transit legs
    for (let i = 1; i < activities.length; i++) {
      const travel = activities[i].travelTimeMinutesFromPrevious;
      if (travel && travel > 120) {
        critiques.push(
          `Day ${day.day}: ${Math.round(travel)} min transfer between "${activities[i - 1].title}" and "${activities[i].title}" — consider removing one stop.`
        );
      }
    }

    // Rule 3: last day should be light
    const isLastDay = idx === plan.days.length - 1;
    if (isLastDay && activities.length > 3) {
      const removed = activities.splice(3).map((a) => a.title);
      critiques.push(
        `Day ${day.day} (departure): Reduced to 3 activities for a comfortable travel day. Dropped: ${removed.join(", ")}.`
      );
    }

    // Rule 4: accommodation gap on multi-day plans
    if (plan.days.length > 1 && !isLastDay && !day.accommodationPoiId) {
      const fallback = Object.values(catalog.items).find((p) => p.type === "stay");
      if (fallback) {
        critiques.push(
          `Day ${day.day}: No accommodation set — defaulting to "${fallback.name}" from the catalog.`
        );
        return { ...day, activities, accommodationPoiId: fallback.id };
      }
    }

    return { ...day, activities };
  });

  return { plan: { ...plan, days }, critiques };
}

/**
 * Transit Advisor sub-agent — adds heuristic inter-city transit guidance for
 * each destination-segment boundary in a multi-stop plan.
 *
 * For every (from → to) city pair it estimates:
 *  - Transport mode: flight / train / bus / drive
 *  - Rough travel time label
 *  - Booking lead-time note
 *
 * Runs after enrichPlanLogistics (route distances already populated).
 *
 * @param plan     - Enriched plan (destinationSegments populated)
 * @param _research - Research object (destination list used for context)
 */
export function transitAdvisor(
  plan: PlanSnapshot,
  _research: DestinationResearch
): { segments: TransitSegment[] } {
  const segments: TransitSegment[] = [];

  for (let idx = 0; idx < plan.destinationSegments.length - 1; idx++) {
    const from = plan.destinationSegments[idx]?.destination;
    const to   = plan.destinationSegments[idx + 1]?.destination;
    if (!from || !to) continue;

    const fromNorm = from.toLowerCase();
    const toNorm   = to.toLowerCase();

    const flightHubs = ["hyderabad", "bangalore", "bengaluru", "mumbai", "delhi", "chennai", "kolkata", "pune"];
    const isLongHaul = flightHubs.some((h) => fromNorm.includes(h) || toNorm.includes(h));

    const hillKeywords = ["araku", "ooty", "munnar", "coorg", "kodagu", "darjeeling", "manali", "shimla", "mussoorie"];
    const isHillRoute  = hillKeywords.some((k) => fromNorm.includes(k) || toNorm.includes(k));

    let mode: TransitSegment["mode"];
    let durationLabel: string;
    let bookingNote: string;

    if (isLongHaul) {
      mode          = "flight";
      durationLabel = "1–2 hr flight";
      bookingNote   = "Book at least 3–4 weeks ahead for best fares.";
    } else if (isHillRoute) {
      mode          = "train";
      durationLabel = "3–5 hr scenic train or 4–6 hr drive";
      bookingNote   = "IRCTC Tatkal available 1 day before; road accessible by hired cab.";
    } else {
      mode          = "drive";
      durationLabel = "2–4 hr drive or intercity bus";
      bookingNote   = "Cabs or KSRTC / APSRTC buses are the most flexible option.";
    }

    segments.push({ from, to, mode, durationLabel, bookingNote });
  }

  return { segments };
}

/** Structured transit segment returned by transitAdvisor. */
export type TransitSegment = {
  from: string;
  to: string;
  /** Best heuristic mode for this segment. */
  mode: "flight" | "train" | "bus" | "drive";
  /** Human-readable time estimate (e.g. "3–5 hr scenic train or 4–6 hr drive"). */
  durationLabel: string;
  /** Booking advice (e.g. "Book 3–4 weeks ahead"). */
  bookingNote: string;
};

export type FastTurnResponse = {
  reply: string;
  responseBlocks: AssistantResponseBlock[];
  memory: SessionSnapshot["memory"];
};

export type ImmediateAssistantReply = {
  reply: string;
  responseBlocks: AssistantResponseBlock[];
};

// ─── Conversation Mode ───────────────────────────────────────────────────────

export type ConversationMode =
  | "greeting"
  | "missing_info"
  | "discovery"
  | "planning"
  | "refinement";

export type MissingSlot =
  | "missing_destination"
  | "missing_duration"
  | "missing_budget"
  | null;

const MAX_DAY_BY_DAY_TRIP_DAYS = 21;

/**
 * Determines the high-level conversation mode for the current turn.
 *
 * This drives which branch of AUTONOMOUS_NARRATIVE_INSTRUCTIONS the LLM
 * follows, and which response block templates buildResponseBlocks renders.
 *
 * Ordering matters — more specific checks come first.
 */
export function resolveConversationMode(
  session: SessionSnapshot,
  resolution: TurnResolution
): ConversationMode {
  if (resolution.intent === "refine_trip") return "refinement";

  if (resolution.intent === "plan_trip") {
    if (resolution.destinations.length > 0 && resolution.totalDays) {
      return "planning";
    }
    return "missing_info";
  }

  if (resolution.intent === "search_places") return "discovery";

  // ── question intent ──────────────────────────────────────────────────────
  // Only return "greeting" when this really is the very first exchange.
  // We check session memory (not committed messages) because the previous
  // assistant message may not be hydrated into session.messages yet at the
  // time we evaluate this.
  const hasHistory =
    session.memory.destinationsDiscussed.length > 0 ||
    session.memory.acceptedDecisions.length > 0 ||
    session.messages.filter((m) => m.role === "user").length >= 1;

  if (!hasHistory && (
    resolution.questionFocus === "greeting" ||
    resolution.questionFocus === "capabilities" ||
    session.messages.filter((m) => m.role === "assistant").length === 0
  )) {
    return "greeting";
  }

  // Question with a destination focus → discovery (show POI cards etc.)
  if (
    resolution.destinations.length > 0 &&
    resolution.questionFocus &&
    !["greeting", "capabilities"].includes(resolution.questionFocus)
  ) {
    return "discovery";
  }

  // Returning user with no clear destination context → still discovery not greeting
  if (hasHistory) return "discovery";

  // True zero-state greeting
  return "greeting";
}

/** Return the primary missing slot needed to proceed for plan_trip intent. */
export function resolveMissingSlot(resolution: TurnResolution): MissingSlot {
  if (resolution.intent !== "plan_trip") return null;
  if (!resolution.destinations.length) return "missing_destination";
  if (!resolution.totalDays) return "missing_duration";
  return null;
}

function formatLongTripPromptPrefix(destination: string | undefined, totalDays: number): string {
  if (destination) {
    return `${destination} for ${totalDays} days`;
  }
  return `${totalDays} days`;
}

export function resolveOversizedPlanResponse(
  resolution: TurnResolution
): ImmediateAssistantReply | null {
  const totalDays = resolution.totalDays;
  if (resolution.intent !== "plan_trip" || !totalDays || totalDays <= MAX_DAY_BY_DAY_TRIP_DAYS) {
    return null;
  }

  const destination = resolution.destination || resolution.destinations[0];
  const planLabel = formatLongTripPromptPrefix(destination, totalDays);
  const title = destination
    ? `${destination} needs a phased plan`
    : "This trip needs a phased plan";

  let body =
    `I can help with ${planLabel}, but once a trip stretches past about three weeks, a strict day-by-day itinerary stops being useful.`;
  if (totalDays >= 365) {
    body =
      `I can help with ${planLabel}, but a day-by-day itinerary that long is less “trip planning” and more “new life chapter.”`;
  } else if (totalDays >= 45) {
    body =
      `I can help with ${planLabel}, but a day-by-day itinerary that long would turn into a small travel novel.`;
  }

  const lead =
    destination
      ? `Smarter move: I break ${destination} into phases, base cities, and priorities, then detail only the first stretch you actually want to book now.`
      : "Smarter move: I break it into phases, base cities, and priorities, then detail only the first stretch you actually want to book now.";

  const prompts = totalDays >= 365
    ? [
        {
          label: "First 14 days",
          prompt: destination
            ? `Plan the first 14 days of my ${totalDays}-day trip to ${destination}.`
            : `Plan the first 14 days of my ${totalDays}-day trip.`
        },
        {
          label: "First month",
          prompt: destination
            ? `Plan the first 30 days of my ${totalDays}-day trip to ${destination}.`
            : `Plan the first 30 days of my ${totalDays}-day trip.`
        },
        {
          label: "Quarterly phases",
          prompt: destination
            ? `Break my ${totalDays}-day trip to ${destination} into quarterly phases with base cities, seasons, and priorities.`
            : `Break my ${totalDays}-day trip into quarterly phases with base cities, seasons, and priorities.`
        },
        {
          label: "High-level route",
          prompt: destination
            ? `Give me a high-level route and seasonal strategy for a ${totalDays}-day trip to ${destination}.`
            : `Give me a high-level route and seasonal strategy for a ${totalDays}-day trip.`
        }
      ]
    : [
        {
          label: "First 7 days",
          prompt: destination
            ? `Plan the first 7 days of my ${totalDays}-day trip to ${destination}.`
            : `Plan the first 7 days of my ${totalDays}-day trip.`
        },
        {
          label: "First 10 days",
          prompt: destination
            ? `Plan the first 10 days of my ${totalDays}-day trip to ${destination}.`
            : `Plan the first 10 days of my ${totalDays}-day trip.`
        },
        {
          label: "Weekly phases",
          prompt: destination
            ? `Break my ${totalDays}-day trip to ${destination} into weekly phases with base cities and priorities.`
            : `Break my ${totalDays}-day trip into weekly phases with base cities and priorities.`
        },
        {
          label: "Route only",
          prompt: destination
            ? `Give me a high-level route for a ${totalDays}-day trip to ${destination}.`
            : `Give me a high-level route for a ${totalDays}-day trip.`
        }
      ];

  return {
    reply: `${body} ${lead}`,
    responseBlocks: [
      {
        type: "trip_intro",
        title,
        body
      },
      {
        type: "lead",
        text: lead
      },
      {
        type: "assistant_prompt_chips",
        title: "Better ways to structure it",
        prompts
      }
    ]
  };
}

// ─── Narrative instructions ───────────────────────────────────────────────────

const AUTONOMOUS_NARRATIVE_INSTRUCTIONS = `
You are the lead conversation agent for Roameo — an autonomous AI travel planner.
Return valid JSON only.

DETECT your conversation mode from the input and follow the matching rules:

**GREETING** (no plan exists, questionFocus = greeting/capabilities, and the user has NO prior history):
- Warm, short, enthusiastic. Sound like you just met the traveller.
- introTitle: personalised welcome (never "Hello!" or "Hi there!" verbatim).
- introBody: 1-2 sentences, inviting, zero pressure.
- leadText: 1 sentence that bridges to what the user can do next.
- promptChips: 3-4 inviting low-pressure starters.
- clarifyingQuestions: EMPTY array — never ask questions on a greeting.
- CRITICAL: If the conversation digest shows ANY previous assistant messages, you are NOT in a first-time greeting. Re-acknowledge the conversation naturally instead.

**MISSING_INFO** (intent = plan_trip but no destination or no duration):
- You are collecting ONE missing slot — address only that single gap.
- If destination is missing: ask WHERE in a warm, curious single sentence.
- If destination is known but duration is missing: ask HOW LONG — suggest realistic durations.
- introTitle: something like "Where to?" or "How long are you thinking?"
- promptChips: fill-in suggestions (e.g. ["3 days", "Long weekend", "1 week", "10 days"]).
- clarifyingQuestions: EMPTY — use promptChips instead.

**DISCOVERY** (intent = search_places OR question with a place focus, OR returning user):
- Sound like a knowledgeable local guide.
- Reference the actual destination and focus (restaurants / attractions / etc.) by name.
- introTitle: punchy, place-aware headline that reflects the SPECIFIC request.
- introBody: 1-2 sentences of local context or intrigue.
- leadText: bridge to the suggestions naturally.
- promptChips: 3-4 focused follow-up ideas.
- clarifyingQuestions: EMPTY.
- CRITICAL: If the user asked specifically about restaurants, say so. If they asked about attractions, say so. Never swap them.

**PLANNING** (intent = plan_trip, destination + duration known):
- Confident, forward-moving.
- introTitle: trip-branded heading (use destination + theme).
- introBody: Brief acknowledgement of what you are building.
- leadText: Bridge to the itinerary with one punchy sentence.
- promptChips: 2-3 refinement ideas for AFTER the plan is presented.
- clarifyingQuestions: EMPTY unless a genuine blocker exists.

**REFINEMENT** (intent = refine_trip):
- Acknowledge the specific change the user requested.
- introTitle: short confirmation (e.g. "Updated day 2" or "Slower pace locked in").
- introBody: 1 sentence confirming what changed.
- leadText: 1 sentence pivoting to what is next.
- promptChips: 2-3 further refinement prompts.
- clarifyingQuestions: EMPTY.

GLOBAL RULES:
- NEVER start a response as if you are meeting the user for the first time when the conversation digest shows previous messages.
- NEVER repeat the same introTitle as any previous assistant message in the digest.
- Use at most one emoji across all text fields combined.
- Keep each field SHORT — introBody max 2 sentences, leadText max 1 sentence.
- NEVER give fake-precise currency totals or exact INR-style estimates. Use expectation-level language like Budget-friendly, Mid-range, Comfortable, Premium, lower-cost, moderate, or upscale instead.
- Never bullet-list the missing fields; always address only the single most important gap.
- Never say "As an AI" or "I don't have access to real-time data".
- Never open with "Of course!", "Certainly!", "Absolutely!", "Great choice!", or "Sure!".
- When given a research summary with restaurants, ONLY reference those specific restaurants. Never make up restaurant names.
- When given a research summary with attractions, ONLY reference those specific attractions.
`;

export async function answerConversationally(
  providerService: ProviderService,
  resolvedProvider: ResolvedProvider,
  session: SessionSnapshot,
  resolution: TurnResolution,
  research?: DestinationResearch,
  planningContext?: PlanningContext
): Promise<AssistantNarrative> {
  const mode = resolveConversationMode(session, resolution);
  const missingSlot = resolveMissingSlot(resolution);
  if (resolution.questionFocus === "followup_clarify") {
    return {
      introTitle: "Which one should I pull up?",
      introBody:
        resolution.destination || resolution.destinations[0]
          ? `I can keep going for ${resolution.destination || resolution.destinations[0]}, but there are a couple of different directions from the last message.`
          : "I can keep going, but there are a couple of different directions from the last message.",
      leadText: "Pick the one you want and I’ll continue from there.",
      promptChips: (resolution.followUpOptions || []).slice(0, 4).map((option) => ({
        label: option.label,
        prompt: option.prompt
      })),
      clarifyingQuestions: []
    };
  }
  const digest = buildConversationDigest(session);

  const researchSummary = research
    ? JSON.stringify(
        {
          destinations: research.destinations,
          focus: research.focus,
          topStays: research.grouped.stays.slice(0, 3).map((poi) => poi.name),
          topRestaurants: research.grouped.restaurants
            .slice(0, 4)
            .map((poi) => poi.name),
          topAttractions: research.grouped.attractions
            .slice(0, 5)
            .map((poi) => poi.name),
          facts: research.facts.slice(0, 3)
        },
        null,
        2
      )
    : "none";

  return providerService.generateObject({
    resolved: resolvedProvider,
    schema: assistantNarrativeSchema,
    schemaName: "assistant_narrative",
    instructions: AUTONOMOUS_NARRATIVE_INSTRUCTIONS,
    input: `Conversation digest:\n${digest}

Detected conversation mode: ${mode}
Missing slot (if any): ${missingSlot || "none"}

Resolved request:\n${JSON.stringify(resolution, null, 2)}

Research summary:\n${researchSummary}

Date context:\n${JSON.stringify(resolution.dateContext, null, 2)}

Planning context:\n${JSON.stringify(planningContext || {}, null, 2)}

Return fields:
- introTitle: heading (short, creative, never repeated from prior messages)
- introBody: 1-2 sentence warm body
- moodEmoji: optional single emoji
- leadText: 1-sentence bridge
- promptChips: context-appropriate follow-up chips
- clarifyingQuestions: ONLY if a genuine blocker; otherwise empty array`
  });
}

function buildCapabilitiesSections(destination?: string, origin?: string) {
  const locality = destination ? ` around ${destination}` : "";
  const nearbyOrigin = !destination && origin ? ` near ${origin}` : "";
  return [
    {
      title: "Personalized recommendations",
      body: `Find restaurants, attractions, stays, and hidden gems${locality || nearbyOrigin} based on the vibe you want.`
    },
    {
      title: "Itinerary planning",
      body: "Build day-by-day trips for quick outings, weekend escapes, or longer multi-stop journeys."
    },
    {
      title: "Hotel search",
      body: "Surface stay options that fit your budget, pace, and trip style."
    },
    {
      title: "Restaurant suggestions",
      body: "Recommend local favorites, seafood, street food, cafes, and special-occasion spots."
    },
    {
      title: "Transport help",
      body: "Help you think through how to get there, move between stops, and keep the route practical."
    },
    {
      title: "Local tips",
      body: origin
        ? `Use your current base in ${origin} to suggest nearby states, city breaks, cultural spots, and practical travel timing.`
        : "Share timing, neighborhood context, and practical travel notes that make the plan easier to use."
    }
  ];
}

function buildCapabilityExamples(destination?: string, origin?: string) {
  if (destination) {
    return [
      `Best beaches near ${destination}?`,
      `Local seafood spots in ${destination}?`,
      `Plan a 2-day trip from ${destination}.`,
      `Family-friendly hotel options in ${destination}?`
    ];
  }

  if (origin) {
    return [
      `Weekend trips near ${origin}?`,
      `Best local food spots around ${origin}?`,
      `Cultural places near ${origin}?`,
      `Suggest a good 2-day getaway from ${origin}.`
    ];
  }

  return [
    "Best beaches nearby?",
    "Plan a 2-day trip for me.",
    "Suggest a great boutique stay.",
    "Show me local food spots."
  ];
}

function getPrimaryPoiForFocus(research?: DestinationResearch): Poi | undefined {
  if (!research) {
    return undefined;
  }

  const picks =
    research.focus === "seafood"
      ? research.grouped.restaurants
      : research.focus === "hotels" || research.focus === "family"
        ? research.grouped.stays
        : research.grouped.attractions;

  return picks[0] || research.grouped.attractions[0] || research.grouped.restaurants[0] || research.grouped.stays[0];
}

function buildDiscoveryBadge(focus: DiscoveryFocus, poi: Poi): string | undefined {
  // Use POI type as primary signal, then focus for flavor
  if (poi.type === "restaurant") {
    switch (focus) {
      case "seafood":  return "Seafood spot";
      case "hidden_gems": return "Local food find";
      default: return "Restaurant pick";
    }
  }
  if (poi.type === "stay") {
    switch (focus) {
      case "family":   return "Family stay";
      case "hotels":   return "Hotel pick";
      default: return "Stay option";
    }
  }
  // attraction type
  switch (focus) {
    case "hidden_gems": return "Quiet local pick";
    case "beaches":     return "Coastal favorite";
    case "culture":     return "Cultural stop";
    case "family":      return "Easy family pick";
    case "day_trips":   return "Good escape";
    default:            return "Place to explore";
  }
}

function buildPoiStoryBody(
  poi: Poi,
  focus: DiscoveryFocus,
  destination?: string
): string {
  if (poi.description) {
    return poi.description;
  }

  const locality = destination ? ` in ${destination}` : "";

  // Use POI type as primary signal
  if (poi.type === "restaurant") {
    switch (focus) {
      case "seafood":
        return `A dependable seafood pick${locality} when you want something local and easy to build into a trip day.`;
      case "hidden_gems":
        return `A lower-key local food spot${locality} that gives you a more authentic meal than the usual tourist restaurants.`;
      default:
        return `A well-regarded dining spot${locality} that balances local flavors, convenience, and a solid reputation.`;
    }
  }

  if (poi.type === "stay") {
    return `A practical stay option${locality} that works well as a comfortable base for the rest of the trip.`;
  }

  // Attraction type — use focus for flavor
  switch (focus) {
    case "hidden_gems":
      return `A lower-key local favorite${locality} that gives you a calmer, more authentic stop than the usual headline picks.`;
    case "beaches":
      return `A strong coastal stop${locality} if you want sea breeze, wide views, and an easy sunrise or sunset plan.`;
    case "culture":
      return `A worthwhile culture stop${locality} if you want heritage, local context, and a stronger sense of place.`;
    case "day_trips":
      return `A good nearby escape${locality} when you want a scenic break without overcomplicating the route.`;
    case "family":
      return `An easy family-friendly stop${locality} with enough to keep everyone engaged without too much fuss.`;
    default:
      return `A strong place to consider${locality} if you want a balanced mix of atmosphere, convenience, and local appeal.`;
  }
}

function classifyRestaurantCategory(
  poi: Poi,
  destination?: string
): RestaurantCategoryKey {
  const haystack = [poi.name, poi.description, poi.address, ...(poi.tags || [])]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  const localDestination = destination?.toLowerCase() || "";

  if (/\b(veg|vegetarian|pure veg|thali|satvik|jain)\b/.test(haystack)) {
    return "vegetarian";
  }

  if (
    /\b(non veg|non-veg|seafood|fish|chicken|mutton|biryani|grill|bbq|barbecue)\b/.test(
      haystack
    )
  ) {
    return "non_vegetarian";
  }

  if (
    (typeof poi.priceLevel === "number" && poi.priceLevel >= 3) ||
    /\b(fine dining|upscale|premium|luxury|chef|signature tasting)\b/.test(haystack)
  ) {
    return "premium";
  }

  if (
    (typeof poi.priceLevel === "number" && poi.priceLevel <= 1) ||
    /\b(budget|cheap|cheap eats|street food|mess|canteen|value)\b/.test(haystack)
  ) {
    return "budget_friendly";
  }

  if (
    (typeof poi.rating === "number" && poi.rating >= 4.4) ||
    /\b(famous|popular|must try|must-try|iconic|legendary|local favorite)\b/.test(haystack) ||
    (localDestination && haystack.includes(localDestination))
  ) {
    return "famous";
  }

  return "famous";
}

function bucketRestaurants(
  pois: Poi[],
  destination?: string
): Array<{ key: RestaurantCategoryKey; title: string; poiIds: string[] }> {
  const buckets = new Map<RestaurantCategoryKey, Poi[]>();
  for (const poi of pois) {
    const bucket = classifyRestaurantCategory(poi, destination);
    const items = buckets.get(bucket) || [];
    items.push(poi);
    buckets.set(bucket, items);
  }

  const definitions: Array<{ key: RestaurantCategoryKey; title: string }> = [
    { key: "famous", title: "Famous" },
    { key: "budget_friendly", title: "Budget-friendly" },
    { key: "vegetarian", title: "Vegetarian" },
    { key: "non_vegetarian", title: "Non-vegetarian" },
    { key: "premium", title: "Premium" }
  ];

  return definitions
    .map((definition) => ({
      ...definition,
      poiIds: Array.from(
        new Set((buckets.get(definition.key) || []).map((poi) => poi.id))
      ).slice(0, 8)
    }))
    .filter((section) => section.poiIds.length > 0);
}

export function derivePendingFollowUpContext(params: {
  resolution: TurnResolution;
  narrative: AssistantNarrative;
  responseBlocks: AssistantResponseBlock[];
  assistantReply: string;
  plan?: PlanSnapshot;
}): PendingFollowUpContext | null {
  const { resolution, narrative, responseBlocks, assistantReply, plan } = params;
  const destination = resolution.destination || resolution.destinations[0] || plan?.destination;
  const startDate = resolution.dateContext.inferredStartDate || plan?.startDate;
  const endDate = resolution.dateContext.inferredEndDate || plan?.endDate;
  const promptOptions = narrative.promptChips
    .map((chip) => {
      const domain = inferPromptDomain(`${chip.label} ${chip.prompt}`);
      return domain
        ? {
            domain,
            label: chip.label,
            prompt: chip.prompt
          }
        : null;
    })
    .filter((option): option is FollowUpOption => Boolean(option));
  const textDomains = inferOfferedDomainsFromText(assistantReply);

  const stayBlock = responseBlocks.find(
    (block) => block.type === "stay_recommendation_list"
  );
  if (stayBlock && stayBlock.type === "stay_recommendation_list") {
    return {
      primaryDomain: "stays",
      destination,
      startDate,
      endDate,
      focus: "hotels",
      categoryKeys: [],
      poiIds: [
        stayBlock.bestOption.poiId,
        ...stayBlock.alternatives.map((item) => item.poiId)
      ],
      options: [{ domain: "stays", label: "Stay options", prompt: "Show me stay options" }]
    };
  }

  const categorizedRows = responseBlocks.find(
    (block) => block.type === "categorized_place_rows"
  );
  if (categorizedRows && categorizedRows.type === "categorized_place_rows") {
    return {
      primaryDomain: "restaurants",
      destination,
      startDate,
      endDate,
      focus: "restaurants",
      categoryKeys: categorizedRows.sections.map((section) => section.key),
      poiIds: categorizedRows.sections.flatMap((section) => section.poiIds),
      options: categorizedRows.sections.map((section) => ({
        domain: "restaurants",
        label: section.title,
        prompt: `Show me ${section.title.toLowerCase()} restaurants`,
        categoryKey: section.key
      }))
    };
  }

  const placeRows = responseBlocks.filter((block) => block.type === "place_card_row");
  if (resolution.stayMode) {
    return {
      primaryDomain: "stays",
      destination,
      startDate,
      endDate,
      focus: "hotels",
      categoryKeys: [],
      poiIds: placeRows.flatMap((block) => block.poiIds),
      options: [{ domain: "stays", label: "Stay options", prompt: "Show me stay options" }]
    };
  }

  if (resolution.questionFocus === "restaurants" || resolution.questionFocus === "seafood") {
    return {
      primaryDomain: "restaurants",
      destination,
      startDate,
      endDate,
      focus: "restaurants",
      categoryKeys: [],
      poiIds: placeRows.flatMap((block) => block.poiIds),
      options: promptOptions
    };
  }

  const uniqueDomains = Array.from(new Set([...textDomains, ...promptOptions.map((option) => option.domain)]));
  if (uniqueDomains.length === 0) {
    return null;
  }

  return {
    primaryDomain: uniqueDomains.length === 1 ? uniqueDomains[0] : undefined,
    destination,
    startDate,
    endDate,
    focus: resolution.questionFocus,
    categoryKeys: [],
    poiIds: placeRows.flatMap((block) => block.poiIds),
    options: promptOptions
  };
}

// ─── Smart default chips ──────────────────────────────────────────────────────

function buildMissingDestinationChips(): Array<{ label: string; prompt: string; slotAction?: { field: string; value: string | number } }> {
  return [
    { label: "Goa", prompt: "Goa", slotAction: { field: "destination", value: "Goa" } },
    { label: "Rajasthan", prompt: "Rajasthan", slotAction: { field: "destination", value: "Rajasthan" } },
    { label: "Kerala backwaters", prompt: "Kerala backwaters", slotAction: { field: "destination", value: "Kerala backwaters" } },
    { label: "Inspire me", prompt: "Suggest an interesting destination for me" }
  ];
}

function buildMissingDurationChips(
  destination: string
): Array<{ label: string; prompt: string; slotAction?: { field: string; value: string | number } }> {
  return [
    { label: "3 days", prompt: "3 days", slotAction: { field: "days", value: 3 } },
    { label: "Long weekend", prompt: "Long weekend", slotAction: { field: "days", value: 4 } },
    { label: "1 week", prompt: "1 week", slotAction: { field: "days", value: 7 } },
    { label: "10 days", prompt: "10 days", slotAction: { field: "days", value: 10 } }
  ];
}

function buildGreetingChips(
  sessionDestination?: string,
  sessionOrigin?: string
): Array<{ label: string; prompt: string }> {
  if (sessionDestination) {
    return [
      { label: `Plan ${sessionDestination} trip`, prompt: `Plan me a trip to ${sessionDestination}` },
      { label: "Best beaches nearby", prompt: `Best beaches near ${sessionDestination}?` },
      { label: "Local food spots", prompt: `Best local food spots in ${sessionDestination}?` },
      { label: "What can you do?", prompt: "What can you help me with?" }
    ];
  }
  if (sessionOrigin) {
    return [
      { label: `Trips near ${sessionOrigin}`, prompt: `Suggest weekend trips near ${sessionOrigin}` },
      { label: `Food near ${sessionOrigin}`, prompt: `Best local food spots around ${sessionOrigin}?` },
      { label: `Culture near ${sessionOrigin}`, prompt: `Show me cultural places near ${sessionOrigin}` },
      { label: "What can you do?", prompt: "What can you help me with?" }
    ];
  }
  return [
    { label: "Plan a trip", prompt: "Help me plan a trip" },
    { label: "Inspire me", prompt: "Suggest an interesting destination for me" },
    { label: "What can you do?", prompt: "What can you help me with?" },
    { label: "Show local gems", prompt: "Show me some hidden gems to explore" }
  ];
}

function inferredUserName(session: SessionSnapshot): string | undefined {
  return session.memory.acceptedDecisions
    .find((decision: string) => decision.startsWith("Name: "))
    ?.replace(/^Name:\s*/, "")
    .trim();
}

function normalizePersonName(raw: string): string {
  return raw
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

function isTravelIntentLikeMessage(message: string): boolean {
  return /\b(plan|trip|itinerary|destination|destinations|travel|visit|days?|restaurants?|food|stays?|hotels?|accommodation|beaches?|attractions?|budget|transport|events?|festivals?)\b/i.test(
    message
  );
}

function isFastCapabilitiesMessage(message: string): boolean {
  return /^(?:so\s+)?(?:what|how)\s+(?:can\s+(?:you|u)\s+do|can\s+(?:you|u)\s+help|do\s+(?:you|u)\s+do)\b.*[?.!]*$/i.test(
    message
  ) || /^(?:can\s+(?:you|u)\s+help\s+me|help)\b[\s!,.?]*$/i.test(message);
}

function isFastIdentityMessage(message: string): boolean {
  return /^(?:who|what)\s+are\s+(?:you|u)\b.*[?.!]*$/i.test(message);
}

function isFastSmallTalkMessage(message: string): boolean {
  return /^(?:how are you|how's it going|whats up|what's up|sup|bye|goodbye|see you|good night|goodnight|lol|haha|hmm+|huh+|nice one)[\s!,.?]*$/i.test(
    message
  );
}

export function resolveFastTurnResponse(
  session: SessionSnapshot,
  message: string
): FastTurnResponse | null {
  const trimmed = message.trim();
  if (!trimmed || isTravelIntentLikeMessage(trimmed)) {
    return null;
  }

  const existingName = inferredUserName(session);
  const origin = primaryLocationLabel(localOriginFromSession(session));
  const introMatch = trimmed.match(
    /^(?:hi|hello|hey)?[\s,!-]*(?:my name is|i am|i'm|im|call me)\s+([\p{L}\p{M}][\p{L}\p{M}\s'-]{0,40})$/iu
  );
  const greetingOnly = /^(hi|hello|hey|hii|yo|good\s+(morning|afternoon|evening))[\s!,.?]*$/i.test(
    trimmed
  );
  const acknowledgementOnly = /^(thanks|thank you|ok|okay|cool|nice|got it|alright)[\s!,.?]*$/i.test(
    trimmed
  );
  const capabilitiesOnly = isFastCapabilitiesMessage(trimmed);
  const identityOnly = isFastIdentityMessage(trimmed);
  const smallTalkOnly = isFastSmallTalkMessage(trimmed);

  if (
    !introMatch &&
    !greetingOnly &&
    !acknowledgementOnly &&
    !capabilitiesOnly &&
    !identityOnly &&
    !smallTalkOnly
  ) {
    return null;
  }

  const destination = localDestinationFromSession(session);
  const prompts = buildGreetingChips(destination, origin);
  const acceptedDecisions = session.memory.acceptedDecisions.filter(
    (decision: string) => !decision.startsWith("Name: ")
  );

  let reply = "";
  let title = "";
  let body = "";
  let lead = "";

  if (introMatch) {
    const name = normalizePersonName(introMatch[1]);
    acceptedDecisions.push(`Name: ${name}`);
    reply = `Nice to meet you, ${name}. Tell me where you want to go, when you’re thinking, or just the kind of trip you want.`;
    title = `Nice to meet you, ${name}`;
    body = origin
      ? `I’ll keep that in mind, and I can also use your current base in ${origin} when nearby suggestions would help.`
      : "I’ll keep that in mind for the rest of this conversation.";
    lead = origin
      ? `Tell me a destination, dates, or just a vibe and I can also suggest smart options around ${origin}.`
      : "Tell me a destination, dates, or just a vibe and I’ll take it from there.";
  } else if (greetingOnly) {
    const greetingName = existingName ? `, ${existingName}` : "";
    reply = `Hey${greetingName}. Tell me where you want to go, when you want to travel, or what kind of trip you want.`;
    title = `Hey${greetingName}`;
    body = origin ? `I’m ready when you are, and I can use ${origin} as your starting context for nearby ideas.` : "I’m ready when you are.";
    lead = origin
      ? `Give me a destination, dates, or a travel vibe and I can also suggest good options from around ${origin}.`
      : "Give me a destination, dates, or a travel vibe and I’ll start there.";
  } else if (capabilitiesOnly || identityOnly) {
    const guideName = existingName ? `, ${existingName}` : "";
    reply = `I’m Roameo${guideName}. I can plan trips, find restaurants and stays, suggest attractions, surface local events, and help refine dates, budget, and route decisions.`;
    title = "Your Roameo travel guide";
    body =
      origin
        ? `I’m here to help you plan trips, discover places worth visiting, and use your current base in ${origin} for nearby states, cities, food, and culture when that helps.`
        : "I’m here to help you plan trips, discover places worth visiting, and refine the details without making you spell out everything up front.";
    lead =
      destination
        ? `I can start with ${destination}, or you can ask for restaurants, stays, local gems, better dates, or a full itinerary.`
        : origin
          ? `I can start from ${origin}, suggest nearby escapes, or help with restaurants, stays, and local culture anywhere you want.`
          : "Give me a destination, a rough vibe, or just tell me whether you want ideas for a trip, places to eat, or places to stay.";
  } else if (smallTalkOnly) {
    const smallTalkName = existingName ? `, ${existingName}` : "";
    reply = `I’m good${smallTalkName}. Whenever you’re ready, give me a destination, a date range, or even just a rough travel vibe.`;
    title = `Ready when you are${smallTalkName}`;
    body = origin
      ? `I can keep things simple and even start from your current location in ${origin}.`
      : "I can keep things simple and start from very little information.";
    lead =
      destination
        ? `We can keep building around ${destination}, or switch to a different place anytime.`
        : origin
          ? `A destination name, a budget, or even “show me places near ${origin}” is enough to get started.`
          : "A destination name, a budget, or even a one-line idea is enough to get started.";
  } else {
    const acknowledgementName = existingName ? `, ${existingName}` : "";
    reply = `Got it${acknowledgementName}. Tell me where you want to go or what you want to figure out next.`;
    title = `Got it${acknowledgementName}`;
    body = origin
      ? `I’m ready for the next step, and I can use ${origin} for nearby suggestions if that helps.`
      : "I’m ready for the next step whenever you are.";
    lead = origin
      ? `A destination, dates, or even “what’s worth doing near ${origin}?” is enough to get moving.`
      : "A destination, dates, or even a rough travel idea is enough to get moving.";
  }

  return {
    reply,
    responseBlocks: [
      {
        type: "trip_intro",
        title,
        body
      },
      {
        type: "lead",
        text: lead
      },
      {
        type: "assistant_prompt_chips",
        title: "Easy next steps",
        prompts
      }
    ],
    memory: {
      ...session.memory,
      summary: reply.slice(0, 400),
      acceptedDecisions,
      pendingFollowUp: null,
      planningState: {
        ...session.memory.planningState,
        status: "ready",
        stage: "ready",
        source: undefined,
        reason: undefined,
        retryable: true,
        updatedAt: new Date().toISOString()
      }
    }
  };
}

function buildDiscoveryChips(
  destination: string,
  focus: string
): Array<{ label: string; prompt: string }> {
  const d = destination;
  const base: Array<{ label: string; prompt: string }> = [];

  if (focus !== "restaurants" && focus !== "seafood") {
    base.push({ label: "Food spots", prompt: `Best restaurants and food spots in ${d}?` });
  }
  if (focus !== "hotels") {
    base.push({ label: "Where to stay", prompt: `Best hotels and stays in ${d}?` });
  }
  if (focus !== "attractions" && focus !== "hidden_gems") {
    base.push({ label: "Top attractions", prompt: `Top attractions and things to do in ${d}?` });
  }
  base.push({ label: `Plan ${d} trip`, prompt: `Plan me a trip to ${d}` });

  return base.slice(0, 4);
}

export function buildResponseBlocks(params: {
  session: SessionSnapshot;
  resolution: TurnResolution;
  narrative: AssistantNarrative;
  research?: DestinationResearch;
  plan?: PlanSnapshot;
  planningContext?: PlanningContext;
}): AssistantResponseBlock[] {
  const { resolution, narrative, research, plan, session, planningContext } = params;
  const discoveryFocus = research?.focus || resolveDiscoveryFocus(resolution);
  const mode = resolveConversationMode(session, resolution);
  const missingSlot = resolveMissingSlot(resolution);

  // ── SHARED: intro + lead are always first ────────────────────────────────────
  const blocks: AssistantResponseBlock[] = [
    {
      type: "trip_intro",
      eyebrow: narrative.introEyebrow,
      title: narrative.introTitle,
      body: narrative.introBody,
      moodEmoji: narrative.moodEmoji
    },
    {
      type: "lead",
      text: narrative.leadText
    }
  ];

  // ── MISSING INFO: short circuit, return chips only ──────────────────────────
  if (mode === "missing_info") {
    const dest = resolution.destination || resolution.destinations[0] || "";
    const missingChips =
      missingSlot === "missing_destination"
        ? buildMissingDestinationChips()
        : missingSlot === "missing_duration" && dest
          ? buildMissingDurationChips(dest)
          : [];

    const chips = narrative.promptChips.length > 0 ? narrative.promptChips : missingChips;

    if (chips.length > 0) {
      blocks.push({
        type: "assistant_prompt_chips",
        title:
          missingSlot === "missing_destination"
            ? "Popular destinations"
            : missingSlot === "missing_duration"
              ? "Pick a duration"
              : "Quick options",
        prompts: chips
      });
    }
    return blocks;
  }

  // ── GREETING: capabilities + featured POI + greeting chips ──────────────────
  if (mode === "greeting") {
    const sessionDestination = localDestinationFromSession(session);
    const sessionOrigin = primaryLocationLabel(localOriginFromSession(session));

    if (resolution.questionFocus === "capabilities") {
      blocks.push({
        type: "capabilities_overview",
        title: "What I can do for you",
        intro: resolution.destination
          ? `I can help you explore ${resolution.destination} with real places, practical planning, and low-friction next steps.`
          : sessionOrigin
            ? `I can help with real places, practical planning, and nearby suggestions from ${sessionOrigin}.`
            : "I can help with real places, practical planning, and easy next steps.",
        sections: buildCapabilitiesSections(resolution.destination, sessionOrigin),
        examplesTitle: "Example help",
        examples: buildCapabilityExamples(resolution.destination, sessionOrigin)
      });
    }

    const featuredPoi = getPrimaryPoiForFocus(research);
    if (featuredPoi) {
      blocks.push({
        type: "featured_poi",
        title: sessionDestination
          ? `A local place to start in ${sessionDestination}`
          : "A good place to start",
        body: "If you want, I can pull more nearby food, beaches, culture, or build a full trip plan from here.",
        poiId: featuredPoi.id
      });
    }

    const greetingPoiIds = [
      ...(research?.grouped.attractions.slice(0, 2).map((p) => p.id) || []),
      ...(research?.grouped.restaurants.slice(0, 2).map((p) => p.id) || [])
    ].filter((id) => isCanonicalPoiId(session.poiCatalog, id));

    if (greetingPoiIds.length > 0) {
      blocks.push({
        type: "place_card_row",
        title: "Popular nearby places",
        poiIds: greetingPoiIds.slice(0, 4),
        display: "carousel"
      });
    }

    blocks.push({
      type: "assistant_prompt_chips",
      title: "Easy next steps",
      prompts:
        narrative.promptChips.length > 0
          ? narrative.promptChips
          : buildGreetingChips(sessionDestination, sessionOrigin)
    });

    return blocks;
  }

  // ── DISCOVERY: typed restaurant + attraction carousels ───────────────────────
  if (mode === "discovery") {
    const dest = resolution.destination || resolution.destinations[0] || "";

    const mergedAdvisories = mergeAdvisories(
      resolution.dateContext.advisoryItems,
      planningContext?.weather?.advisories,
      planningContext?.events?.advisories,
      planningContext?.holidays?.advisories
    );
    if (mergedAdvisories.length) {
      blocks.push({
        type: "date_advisory",
        title: "A timing note for your dates",
        summary:
          resolution.dateContext.flexibility === "exact"
            ? "Your requested dates work, but there are a few timing notes worth keeping in mind."
            : "Because your dates are flexible, I found a few timing notes that could improve the trip.",
        advisories: mergedAdvisories.slice(0, 4)
      });
    }

    const eventItems = [...(planningContext?.events?.items || []), ...(planningContext?.holidays?.items || [])];
    if (eventItems.length) {
      blocks.push({
        type: "event_window_summary",
        title: "Around your dates",
        summary: planningContext?.events?.summary || planningContext?.holidays?.summary,
        items: eventItems.slice(0, 4)
      });
    }

    if (research && !resolution.stayMode) {
      // Determine which POI types to show based on discovery focus
      const isFoodFocus = ["seafood", "restaurants"].includes(discoveryFocus);
      const isAttractionFocus = ["attractions", "culture", "beaches", "hidden_gems", "day_trips", "family"].includes(discoveryFocus);
      const showRestaurants = isFoodFocus || discoveryFocus === "general";
      const showAttractions = isAttractionFocus || discoveryFocus === "general";

      const rankedRestaurants = showRestaurants
        ? sortPois(research.grouped.restaurants, resolution, "restaurant").filter((poi) =>
            isCanonicalPoiId(session.poiCatalog, poi.id)
          )
        : [];

      // poi_story_list: use focus-appropriate POIs only
      const storySource = isFoodFocus
        ? rankedRestaurants
        : isAttractionFocus
          ? sortPois(research.grouped.attractions, resolution, "attraction")
          : [
              ...sortPois(research.grouped.attractions, resolution, "attraction"),
              ...rankedRestaurants
            ];

      const storyCandidates = Array.from(
        new Map(
          storySource
            .filter((poi) => isCanonicalPoiId(session.poiCatalog, poi.id))
            .map((poi) => [poi.id, poi] as const)
        ).values()
      ).slice(0, 3);

      if (storyCandidates.length > 0) {
        const storyTitle = isFoodFocus
          ? (dest ? `A few strong picks in ${dest}` : "Worth trying")
          : isAttractionFocus
            ? (dest ? `Worth visiting in ${dest}` : "Worth visiting")
            : (dest ? `A few strong picks in ${dest}` : "Worth exploring");

        blocks.push({
          type: "poi_story_list",
          title: storyTitle,
          intro: discoveryFocus === "hidden_gems" ? "These lean more local and quieter than the usual tourist picks." : undefined,
          items: storyCandidates.map((poi) => ({
            poiId: poi.id,
            title: poi.name,
            badge: buildDiscoveryBadge(discoveryFocus, poi),
            body: buildPoiStoryBody(poi, discoveryFocus, dest)
          }))
        });
      }

      if (isFoodFocus && rankedRestaurants.length > 0) {
        const filteredRestaurants = resolution.restaurantCategory
          ? rankedRestaurants.filter(
              (poi) =>
                classifyRestaurantCategory(poi, dest) ===
                resolution.restaurantCategory
            )
          : rankedRestaurants;
        const sections = bucketRestaurants(filteredRestaurants, dest)
          .filter((section) =>
            resolution.restaurantCategory
              ? section.key === resolution.restaurantCategory
              : true
          )
          .map((section) => ({
            key: section.key,
            title: section.title,
            poiIds: section.poiIds,
            display: "carousel" as const
          }));

        if (sections.length > 0) {
          blocks.push({
            type: "categorized_place_rows",
            title:
              discoveryFocus === "seafood"
                ? dest
                  ? `Seafood picks in ${dest}`
                  : "Seafood picks"
                : dest
                  ? `Restaurants in ${dest}`
                  : "Restaurants",
            sections
          });
        }
      }
    }

    if (research && resolution.stayMode) {
      const dest = resolution.destination || resolution.destinations[0] || "";
      const rankedStays = sortPois(research.grouped.stays, resolution, "stay")
        .filter((poi) => isCanonicalPoiId(session.poiCatalog, poi.id))
        .slice(0, 4);
      const best = rankedStays[0];
      if (best) {
        blocks.push({
          type: "stay_recommendation_list",
          title: dest ? `Stay ideas for ${dest}` : "Stay ideas for this trip",
          intro: "Recommendation-level picks based on the route, trip style, and current place data.",
          bookingDisclaimer: "Rates are approximate. Confirm availability directly with the property.",
          bestOption: {
            poiId: best.id,
            title: "Best option",
            rateLabel: typeof best.priceLevel === "number" ? estimatedRateLabel(best.priceLevel) : undefined,
            body: best.description || "Balances location, practical routing, and overall fit.",
            caveat: best.openingHours.length === 0 ? "I don't have live availability data yet." : undefined
          },
          alternativesTitle: "Other options that could work",
          alternatives: rankedStays.slice(1).map((poi) => ({
            poiId: poi.id,
            title: poi.name,
            rateLabel: typeof poi.priceLevel === "number" ? estimatedRateLabel(poi.priceLevel) : undefined,
            body: poi.description || "A workable alternative if you want a different vibe, location, or budget."
          })),
          notFitTitle: "Not the best fit",
          notFit: research.grouped.stays.length > rankedStays.length
            ? [{ label: "Lower-confidence local listings", reason: "Some results had weaker metadata or felt less aligned with the trip style." }]
            : []
        });
        blocks.push({
          type: "place_card_row",
          title: "Stay options",
          poiIds: rankedStays.map((p) => p.id).slice(0, 4),
          display: "carousel"
        });
      }
    }

    const discoveryChips =
      narrative.promptChips.length > 0
        ? narrative.promptChips
        : buildDiscoveryChips(resolution.destination || resolution.destinations[0] || "", discoveryFocus);

    const hasCategorizedRows = blocks.some(
      (block) => block.type === "categorized_place_rows"
    );
    if (discoveryChips.length > 0 && !hasCategorizedRows) {
      blocks.push({ type: "assistant_prompt_chips", title: "Explore more", prompts: discoveryChips });
    }

    return blocks;
  }

  // ── PLANNING: itinerary + POI cards + refinement chips ───────────────────────
  if (mode === "planning") {

    const mergedAdvisories = mergeAdvisories(
      resolution.dateContext.advisoryItems,
      planningContext?.weather?.advisories,
      planningContext?.events?.advisories,
      planningContext?.holidays?.advisories
    );
    if (mergedAdvisories.length) {
      blocks.push({
        type: "date_advisory",
        title: "A timing note for your dates",
        summary:
          resolution.dateContext.flexibility === "exact"
            ? "Your requested dates work, but there are a few timing notes worth keeping in mind."
            : "Because your dates are flexible, I found a few timing notes that could improve the trip.",
        advisories: mergedAdvisories.slice(0, 4)
      });
    }

    const planEventItems = [...(planningContext?.events?.items || []), ...(planningContext?.holidays?.items || [])];
    if (planEventItems.length) {
      blocks.push({
        type: "event_window_summary",
        title: "Around your dates",
        summary: planningContext?.events?.summary || planningContext?.holidays?.summary,
        items: planEventItems.slice(0, 4)
      });
    }

    if (plan?.days.length) {
      blocks.push({
        type: "itinerary_template",
        title: plan.title,
        subtitle:
          plan.destinationSegments.length > 1
            ? `${plan.destinationSegments[0]?.destination} to ${plan.destinationSegments.at(-1)?.destination}`
            : plan.destination || plan.destinations[0] || "Trip plan",
        budgetLabel: formatBudgetLabel(plan),
        days: plan.days.map((day) => {
          const grouped = new Map<
            "morning" | "afternoon" | "evening" | "flex",
            Array<{ title: string; poiId?: string; timeLabel?: string; description?: string }>
          >();
          for (const activity of day.activities) {
            const period = periodForTimeLabel(activity.startTime);
            const items = grouped.get(period) || [];
            items.push({
              title: activity.title,
              poiId: isCanonicalPoiId(session.poiCatalog, activity.poiId) ? activity.poiId : undefined,
              timeLabel: activity.startTime,
              description: activity.summary || activity.notes[0]
            });
            grouped.set(period, items);
          }
          return {
            day: day.day,
            date: day.date,
            title: day.title,
            summary: day.summary,
            destination: day.destination,
            accent: day.theme,
            periods: Array.from(grouped.entries()).map(([key, entries]) => ({ key, ...labelForPeriod(key), entries })),
            stayPoiId: isCanonicalPoiId(session.poiCatalog, day.accommodationPoiId) ? day.accommodationPoiId : undefined,
            footer: undefined
          };
        })
      });

      const planPoiIds = plan.days
        .flatMap((day) => [day.accommodationPoiId, ...day.activities.map((a) => a.poiId)])
        .filter((id): id is string => isCanonicalPoiId(session.poiCatalog, id))
        .slice(0, 6);
      if (planPoiIds.length > 0) {
        blocks.push({ type: "place_card_row", title: "Places in this plan", poiIds: planPoiIds, display: "inline" });
      }
    }

    blocks.push({
      type: "assistant_prompt_chips",
      title: "Refine the plan",
      prompts: narrative.promptChips.length > 0
        ? narrative.promptChips
        : [
            { label: "Relax the pace", prompt: "Make day 2 a bit slower and easier." },
            { label: "Better stays", prompt: "Keep the route, but upgrade the stay recommendations." },
            { label: "More food stops", prompt: "Add a couple of stronger local food stops." }
          ]
    });
    return blocks;
  }

  // ── REFINEMENT ───────────────────────────────────────────────────────────────
  if (mode === "refinement") {
    if (plan?.days.length) {
      blocks.push({
        type: "itinerary_template",
        title: plan.title,
        subtitle:
          plan.destinationSegments.length > 1
            ? `${plan.destinationSegments[0]?.destination} to ${plan.destinationSegments.at(-1)?.destination}`
            : plan.destination || plan.destinations[0] || "Updated plan",
        budgetLabel: formatBudgetLabel(plan),
        days: plan.days.map((day) => {
          const grouped = new Map<
            "morning" | "afternoon" | "evening" | "flex",
            Array<{ title: string; poiId?: string; timeLabel?: string; description?: string }>
          >();
          for (const activity of day.activities) {
            const period = periodForTimeLabel(activity.startTime);
            const items = grouped.get(period) || [];
            items.push({
              title: activity.title,
              poiId: isCanonicalPoiId(session.poiCatalog, activity.poiId) ? activity.poiId : undefined,
              timeLabel: activity.startTime,
              description: activity.summary || activity.notes[0]
            });
            grouped.set(period, items);
          }
          return {
            day: day.day,
            date: day.date,
            title: day.title,
            summary: day.summary,
            destination: day.destination,
            accent: day.theme,
            periods: Array.from(grouped.entries()).map(([key, entries]) => ({ key, ...labelForPeriod(key), entries })),
            stayPoiId: isCanonicalPoiId(session.poiCatalog, day.accommodationPoiId) ? day.accommodationPoiId : undefined,
            footer: undefined
          };
        })
      });
    }

    blocks.push({
      type: "assistant_prompt_chips",
      title: "Further refinements",
      prompts: narrative.promptChips.length > 0
        ? narrative.promptChips
        : [
            { label: "More changes", prompt: "What else can I adjust?" },
            { label: "Add food stops", prompt: "Add more local food recommendations." },
            { label: "Adjust budget", prompt: "Can you adjust the budget breakdown?" }
          ]
    });
    return blocks;
  }

  // ── FALLBACK ─────────────────────────────────────────────────────────────────
  if (narrative.clarifyingQuestions.length > 0) {
    blocks.push({ type: "clarifying_questions", title: "A couple of quick checks", questions: narrative.clarifyingQuestions });
  }
  if (narrative.promptChips.length > 0) {
    blocks.push({ type: "assistant_prompt_chips", title: "Easy next steps", prompts: narrative.promptChips });
  }

  return blocks;
}



function estimatedRateLabel(priceLevel: number): string {
  switch (priceLevel) {
    case 0:
    case 1:
      return "Estimated budget-friendly";
    case 2:
      return "Estimated mid-range";
    case 3:
      return "Estimated upscale";
    default:
      return "Estimated premium";
  }
}

export function updateSessionMemory(
  session: SessionSnapshot,
  resolution: TurnResolution,
  assistantReply: string,
  plan?: PlanSnapshot,
  pendingFollowUp?: PendingFollowUpContext | null
): SessionSnapshot["memory"] {
  const destinations = new Set(session.memory.destinationsDiscussed);
  for (const destination of resolution.destinations) {
    destinations.add(destination);
  }

  const acceptedDecisions = new Set(session.memory.acceptedDecisions);
  if (plan?.destination) {
    acceptedDecisions.add(`Destination: ${plan.destination}`);
  } else if (resolution.destination) {
    acceptedDecisions.add(`Destination: ${resolution.destination}`);
  }
  if (plan?.totalDays) {
    acceptedDecisions.add(`Duration: ${plan.totalDays} days`);
  } else if (resolution.totalDays) {
    acceptedDecisions.add(`Duration: ${resolution.totalDays} days`);
  }
  if (plan?.startDate && plan?.endDate) {
    acceptedDecisions.add(`Dates: ${plan.startDate} to ${plan.endDate}`);
  } else if (
    resolution.dateContext.inferredStartDate &&
    resolution.dateContext.inferredEndDate &&
    resolution.dateContext.flexibility === "exact"
  ) {
    acceptedDecisions.add(
      `Dates: ${resolution.dateContext.inferredStartDate} to ${resolution.dateContext.inferredEndDate}`
    );
  }
  if (resolution.origin) {
    acceptedDecisions.add(`Origin: ${resolution.origin}`);
  }
  if (resolution.travelerCount) {
    acceptedDecisions.add(`Travelers: ${resolution.travelerCount}`);
  }
  if (resolution.budgetNote) {
    acceptedDecisions.add(`Budget target: ${resolution.budgetNote}`);
  }

  return {
    ...session.memory,
    summary: assistantReply.slice(0, 400),
    destinationsDiscussed: Array.from(destinations),
    acceptedDecisions: Array.from(acceptedDecisions),
    dateContext: {
      ...session.memory.dateContext,
      ...resolution.dateContext
    },
    pendingFollowUp: pendingFollowUp ?? null,
    lastPlanVersion: plan?.version || session.memory.lastPlanVersion,
    preferences: {
      ...session.memory.preferences,
      styles: Array.from(
        new Set([...session.memory.preferences.styles, ...resolution.styles])
      ) as SessionSnapshot["memory"]["preferences"]["styles"]
    }
  };
}
