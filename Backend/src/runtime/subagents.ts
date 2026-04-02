import { randomUUID } from "node:crypto";
import { z } from "zod";
import type {
  AssistantResponseBlock,
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
import { TravelToolsService } from "../services/travel-tools.js";

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
};

export type DestinationResearch = {
  destinations: string[];
  catalog: PoiCatalog;
  grouped: {
    stays: Poi[];
    restaurants: Poi[];
    attractions: Poi[];
  };
  facts: Array<{ title: string; url: string; snippet: string }>;
};

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
  const asksForPlan =
    /\b(plan|itinerary|trip)\b/.test(normalized) &&
    /\b(\d+\s*days?|weekend|day-by-day)\b/.test(normalized);
  if (asksForPlan) {
    return "plan_trip";
  }

  const refineCue =
    /\b(add|remove|replace|swap|move|adjust|refine|change|update|tweak|slower|relax|rebalance|upgrade)\b/.test(
      normalized
    ) && /\b(day\s*\d+|itinerary|plan|trip|stay|dinner|lunch|breakfast|pace)\b/.test(normalized);

  if (session.plan && refineCue && (current === "question" || current === "search_places")) {
    return "refine_trip";
  }

  return current;
}

function finalizeResolution(
  session: SessionSnapshot,
  message: string,
  parsed: z.infer<typeof resolverSchema>
): TurnResolution {
  const inferredDestinations = inferDestinationFromMessage(message);
  const destinations = Array.from(
    new Set(
      [
        ...normalizeDestinations(parsed),
        ...inferredDestinations,
        ...session.memory.destinationsDiscussed.slice(-2)
      ].filter(Boolean)
    )
  );
  const inferredDays = inferTotalDays(message);
  const inferredTravelers = inferTravelerCount(message);
  const inferredStyles = inferStyles(message);
  const parsedStyles = parsed.styles
    .map((style) => normalizeTravelStyle(style))
    .filter((style): style is TravelStyle => Boolean(style));

  return {
    ...parsed,
    intent: inferIntentFromMessage(session, message, parsed.intent),
    destination: parsed.destination || destinations[0],
    destinations,
    totalDays: parsed.totalDays || inferredDays || session.plan?.totalDays,
    travelerCount:
      parsed.travelerCount || inferredTravelers || session.plan?.travelerCount,
    styles: Array.from(new Set([...parsedStyles, ...inferredStyles]))
  };
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
    `Accepted decisions: ${session.memory.acceptedDecisions.join(" | ") || "none"}`,
    `Preferences: ${session.memory.preferences.styles.join(", ") || "none"}`,
    `Recent messages:\n${recentMessages || "none"}`
  ].join("\n");
}

function nextDate(index: number): string {
  const date = new Date();
  date.setUTCDate(date.getUTCDate() + index);
  return date.toISOString().slice(0, 10);
}

function isCanonicalPoiId(catalog: PoiCatalog, poiId?: string): poiId is string {
  return Boolean(poiId && catalog.items[poiId]);
}

function formatBudgetLabel(plan?: PlanSnapshot): string | undefined {
  if (plan?.budget?.total) {
    return `${plan.budget.currency} ${plan.budget.total.toLocaleString()}`;
  }
  if (plan?.budgetTarget?.total) {
    return `${plan.budgetTarget.currency} ${plan.budgetTarget.total.toLocaleString()}`;
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

export async function researchDestinations(
  tools: TravelToolsService,
  destinations: string[]
): Promise<DestinationResearch> {
  const chunks = await Promise.all(
    destinations.map(async (destination) => {
      const result = await tools.searchPlacesForDestination(destination);
      const facts = await tools.getDestinationFacts(destination);
      return {
        destinations: [destination],
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

export async function synthesizePlan(
  providerService: ProviderService,
  resolvedProvider: ResolvedProvider,
  session: SessionSnapshot,
  resolution: TurnResolution,
  research: DestinationResearch
): Promise<PlanSnapshot> {
  const poiSummary = JSON.stringify(
    {
      stays: research.grouped.stays.map((poi) => ({
        id: poi.id,
        name: poi.name,
        priceLevel: poi.priceLevel,
        rating: poi.rating
      })),
      restaurants: research.grouped.restaurants.map((poi) => ({
        id: poi.id,
        name: poi.name,
        priceLevel: poi.priceLevel,
        rating: poi.rating
      })),
      attractions: research.grouped.attractions.map((poi) => ({
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
- Include accommodationPoiId when a stay is available.
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
    totalDays,
    travelerCount:
      resolution.travelerCount || draft.travelerCount || session.plan?.travelerCount || 1,
    notes: draft.notes,
    destinationSegments: draft.destinationSegments,
    days: draft.days.map((day, index) => ({
      ...day,
      date: nextDate(index),
      activities: day.activities.map((activity) => ({
        ...activity,
        id: activity.id || randomUUID()
      }))
    })),
    generatedAt: new Date().toISOString(),
    lastUserIntent: resolution.intent
  });

  return normalized;
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

export async function answerConversationally(
  providerService: ProviderService,
  resolvedProvider: ResolvedProvider,
  session: SessionSnapshot,
  resolution: TurnResolution,
  research?: DestinationResearch
): Promise<AssistantNarrative> {
  const digest = buildConversationDigest(session);
  const researchSummary = research
    ? JSON.stringify(
        {
          destinations: research.destinations,
          topStays: research.grouped.stays.slice(0, 3).map((poi) => poi.name),
          topRestaurants: research.grouped.restaurants
            .slice(0, 3)
            .map((poi) => poi.name),
          topAttractions: research.grouped.attractions
            .slice(0, 4)
            .map((poi) => poi.name),
          facts: research.facts
        },
        null,
        2
      )
    : "none";

  return providerService.generateObject({
    resolved: resolvedProvider,
    schema: assistantNarrativeSchema,
    schemaName: "assistant_narrative",
    instructions:
      "You are the lead conversation agent for Roameo. Return JSON only. Keep the tone warm, friendly, concise, and easy to continue. Use at most one light emoji in intro fields. Avoid sounding strict, robotic, or question-heavy. Ask clarifying questions only when the user has not given enough information to produce valid travel output. Prompt chips should feel easy and low-pressure.",
    input: `Conversation digest:\n${digest}

Resolved request:\n${JSON.stringify(resolution, null, 2)}

Research summary:\n${researchSummary}

Return:
- introTitle: friendly heading for the response
- introBody: short warm opener
- leadText: a concise natural-language bridge into the plan or suggestions
- promptChips: 0-4 low-friction follow-up prompts
- clarifyingQuestions: only include when truly required to proceed safely`
  });
}

export function buildResponseBlocks(params: {
  session: SessionSnapshot;
  resolution: TurnResolution;
  narrative: AssistantNarrative;
  research?: DestinationResearch;
  plan?: PlanSnapshot;
}): AssistantResponseBlock[] {
  const { resolution, narrative, research, plan, session } = params;
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
          Array<{
            title: string;
            poiId?: string;
            timeLabel?: string;
            description?: string;
          }>
        >();

        for (const activity of day.activities) {
          const period = periodForTimeLabel(activity.startTime);
          const items = grouped.get(period) || [];
          items.push({
            title: activity.title,
            poiId: isCanonicalPoiId(session.poiCatalog, activity.poiId)
              ? activity.poiId
              : undefined,
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
          periods: Array.from(grouped.entries()).map(([key, entries]) => ({
            key,
            ...labelForPeriod(key),
            entries
          })),
          stayPoiId: isCanonicalPoiId(session.poiCatalog, day.accommodationPoiId)
            ? day.accommodationPoiId
            : undefined,
          footer: day.budget?.total
            ? `${day.budget.currency} ${day.budget.total.toLocaleString()} for the day`
            : undefined
        };
      })
    });
  }

  const candidatePoiIds = Array.from(
    new Set(
      plan?.days.length
        ? plan.days
            .flatMap((day) => [
              day.accommodationPoiId,
              ...day.activities.map((activity) => activity.poiId)
            ])
            .filter((poiId): poiId is string => isCanonicalPoiId(session.poiCatalog, poiId))
            .slice(0, 6)
        : [
            ...(research?.grouped.attractions.slice(0, 3).map((poi) => poi.id) || []),
            ...(research?.grouped.stays.slice(0, 2).map((poi) => poi.id) || []),
            ...(research?.grouped.restaurants.slice(0, 2).map((poi) => poi.id) || [])
          ].filter((poiId) => isCanonicalPoiId(session.poiCatalog, poiId))
    )
  );

  if (candidatePoiIds.length > 0) {
    blocks.push({
      type: "place_card_row",
      title: plan?.days.length
        ? "Places in this plan"
        : resolution.intent === "search_places"
          ? "Places to start with"
          : "A few strong matches",
      poiIds: candidatePoiIds.slice(0, 4),
      display: "inline"
    });
  }

  if (narrative.clarifyingQuestions.length > 0) {
    blocks.push({
      type: "clarifying_questions",
      title: "A couple of quick checks",
      questions: narrative.clarifyingQuestions
    });
  }

  const prompts =
    narrative.promptChips.length > 0
      ? narrative.promptChips
      : plan?.days.length
        ? [
            {
              label: "Relax day 2",
              prompt: "Make day 2 a bit slower and easier."
            },
            {
              label: "Better stays",
              prompt: "Keep the route, but upgrade the stay recommendations."
            },
            {
              label: "More food spots",
              prompt: "Add a couple of stronger local food stops."
            }
          ]
        : resolution.destinations[0]
          ? [
              {
                label: "Build the itinerary",
                prompt: `Build me a friendly itinerary for ${resolution.destinations[0]}.`
              },
              {
                label: "Show top places",
                prompt: `Show me the best places to consider in ${resolution.destinations[0]}.`
              }
            ]
          : [];

  if (prompts.length > 0) {
    blocks.push({
      type: "assistant_prompt_chips",
      title: "Easy next steps",
      prompts
    });
  }

  return blocks;
}

export function updateSessionMemory(
  session: SessionSnapshot,
  resolution: TurnResolution,
  assistantReply: string,
  plan?: PlanSnapshot
): SessionSnapshot["memory"] {
  const destinations = new Set(session.memory.destinationsDiscussed);
  for (const destination of resolution.destinations) {
    destinations.add(destination);
  }

  const acceptedDecisions = new Set(session.memory.acceptedDecisions);
  if (plan?.destination) {
    acceptedDecisions.add(`Destination: ${plan.destination}`);
  }
  if (plan?.totalDays) {
    acceptedDecisions.add(`Duration: ${plan.totalDays} days`);
  }

  return {
    ...session.memory,
    summary: assistantReply.slice(0, 400),
    destinationsDiscussed: Array.from(destinations),
    acceptedDecisions: Array.from(acceptedDecisions),
    lastPlanVersion: plan?.version || session.memory.lastPlanVersion,
    preferences: {
      ...session.memory.preferences,
      styles: Array.from(
        new Set([...session.memory.preferences.styles, ...resolution.styles])
      ) as SessionSnapshot["memory"]["preferences"]["styles"]
    }
  };
}
