import { randomUUID } from "node:crypto";
import {
  planSnapshotSchema,
  type ConversationMessage,
  type ItineraryActivity,
  type ItineraryDay,
  type PlanMutationInput,
  type PlanSnapshot,
  type Poi,
  type PoiCatalog,
  type SessionSnapshot
} from "@roameo/contracts";
import { z } from "zod";
import {
  enrichPlanLogistics,
  researchDestinations,
  synthesizePlan,
  type TurnResolution
} from "../runtime/subagents.js";
import { ProviderService } from "./provider-service.js";
import { SessionRepository } from "./session-repository.js";
import { TravelToolsService } from "./travel-tools.js";

const DAY_START_MINUTES = 8 * 60 + 30;
const BETWEEN_ACTIVITY_MINUTES = 30;

const regeneratedDaySchema = z.object({
  title: z.string(),
  theme: z.string().optional(),
  summary: z.string().optional(),
  accommodationPoiId: z.string().optional(),
  activities: z.array(
    z.object({
      poiId: z.string().optional(),
      title: z.string(),
      summary: z.string().optional(),
      notes: z.array(z.string()).default([])
    })
  )
});

function formatTime(totalMinutes: number): string {
  const minutes = Math.max(totalMinutes, 0);
  const hours = Math.floor(minutes / 60)
    .toString()
    .padStart(2, "0");
  const remainder = (minutes % 60).toString().padStart(2, "0");
  return `${hours}:${remainder}`;
}

function appendUnique(values: string[], next: string): string[] {
  return values.includes(next) ? values : [...values, next];
}

function upsertDecision(values: string[], prefix: string, next?: string): string[] {
  if (!next) {
    return values;
  }

  return [...values.filter((value) => !value.startsWith(prefix)), `${prefix}${next}`];
}

function matchesDestination(poi: Poi, destination: string): boolean {
  const haystack = `${poi.name} ${poi.address || ""} ${poi.tags.join(" ")}`.toLowerCase();
  return haystack.includes(destination.toLowerCase());
}

function estimateActivityDurationMinutes(activity: ItineraryActivity, poi?: Poi): number {
  const title = activity.title.toLowerCase();
  if (poi?.type === "restaurant" || /\bbreakfast\b|\blunch\b|\bdinner\b/.test(title)) {
    return 75;
  }
  if (poi?.type === "stay") {
    return 45;
  }
  if (/\bmarket\b|\bwalk\b|\bshopping\b/.test(title)) {
    return 90;
  }
  return 120;
}

function distanceKm(from?: Poi, to?: Poi): number {
  if (!from || !to) {
    return 0;
  }

  const toRadians = (value: number) => (value * Math.PI) / 180;
  const earthRadiusKm = 6371;
  const dLat = toRadians(to.lat - from.lat);
  const dLng = toRadians(to.lng - from.lng);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRadians(from.lat)) *
      Math.cos(toRadians(to.lat)) *
      Math.sin(dLng / 2) ** 2;

  return 2 * earthRadiusKm * Math.asin(Math.sqrt(a));
}

function clonePlan(plan: PlanSnapshot): PlanSnapshot {
  return structuredClone(plan);
}

function finalizePlan(
  plan: PlanSnapshot,
  overrides: Partial<PlanSnapshot> = {}
): PlanSnapshot {
  return planSnapshotSchema.parse({
    ...plan,
    ...overrides,
    version: (plan.version || 0) + 1,
    generatedAt: new Date().toISOString(),
    lastUserIntent: "refine_trip"
  });
}

function shiftPlanDates(
  plan: PlanSnapshot,
  startDate?: string,
  endDate?: string
): PlanSnapshot {
  if (!startDate || !endDate) {
    return plan;
  }

  return {
    ...plan,
    startDate,
    endDate,
    days: plan.days.map((day, index) => ({
      ...day,
      date: addDaysIso(startDate, index)
    }))
  };
}

function addDaysIso(date: string, days: number): string {
  const next = new Date(`${date}T00:00:00.000Z`);
  next.setUTCDate(next.getUTCDate() + days);
  return next.toISOString().slice(0, 10);
}

function buildActivityFromPoi(poi: Poi): ItineraryActivity {
  const notes = [];
  if (typeof poi.rating === "number") {
    notes.push(`Rated ${poi.rating.toFixed(1)} on Google Maps.`);
  }
  if (poi.openingHours.length > 0) {
    notes.push(`Hours: ${poi.openingHours[0]}`);
  }

  return {
    id: randomUUID(),
    poiId: poi.id,
    title: poi.name,
    summary: poi.description,
    startTime: "09:00",
    endTime: "10:30",
    notes
  };
}

function normalizeDaySchedule(day: ItineraryDay, catalog: PoiCatalog): ItineraryDay {
  let cursor = DAY_START_MINUTES;

  return {
    ...day,
    activities: day.activities.map((activity) => {
      const poi = activity.poiId ? catalog.items[activity.poiId] : undefined;
      const durationMinutes = estimateActivityDurationMinutes(activity, poi);
      const startTime = formatTime(cursor);
      const endTime = formatTime(cursor + durationMinutes);
      cursor += durationMinutes + BETWEEN_ACTIVITY_MINUTES;

      return {
        ...activity,
        startTime,
        endTime
      };
    })
  };
}

function insertAt<T>(values: T[], index: number, item: T): T[] {
  return [...values.slice(0, index), item, ...values.slice(index)];
}

function mergeCatalogs(base: PoiCatalog, next?: PoiCatalog): PoiCatalog {
  if (!next) {
    return base;
  }

  return {
    version: Math.max(base.version, next.version) + 1,
    items: {
      ...base.items,
      ...next.items
    }
  };
}

function findBestInsertionPosition(
  day: ItineraryDay,
  catalog: PoiCatalog,
  poi: Poi
): number {
  if (day.activities.length === 0) {
    return 0;
  }

  let bestIndex = day.activities.length;
  let bestCost = Number.POSITIVE_INFINITY;

  for (let index = 0; index <= day.activities.length; index += 1) {
    const before = index > 0 ? day.activities[index - 1] : undefined;
    const after = index < day.activities.length ? day.activities[index] : undefined;
    const beforePoi = before?.poiId ? catalog.items[before.poiId] : undefined;
    const afterPoi = after?.poiId ? catalog.items[after.poiId] : undefined;
    const baseline = beforePoi && afterPoi ? distanceKm(beforePoi, afterPoi) : 0;
    const cost =
      distanceKm(beforePoi, poi) +
      distanceKm(poi, afterPoi) -
      baseline +
      index * 0.01;

    if (cost < bestCost) {
      bestCost = cost;
      bestIndex = index;
    }
  }

  return bestIndex;
}

function pickBestDayForPoi(
  plan: PlanSnapshot,
  catalog: PoiCatalog,
  poi: Poi,
  preferredDay?: number
): { dayIndex: number; position: number } {
  if (preferredDay) {
    const dayIndex = plan.days.findIndex((day) => day.day === preferredDay);
    if (dayIndex === -1) {
      throw new Error(`Day ${preferredDay} does not exist in this itinerary`);
    }

    return {
      dayIndex,
      position: findBestInsertionPosition(plan.days[dayIndex], catalog, poi)
    };
  }

  const scored = plan.days.map((day, index) => {
    const destinationBoost = matchesDestination(poi, day.destination) ? -1 : 0;
    const position = findBestInsertionPosition(day, catalog, poi);
    return {
      dayIndex: index,
      position,
      score: day.activities.length + destinationBoost
    };
  });

  scored.sort((left, right) => left.score - right.score);
  return {
    dayIndex: scored[0].dayIndex,
    position: scored[0].position
  };
}

function reorderActivities(
  day: ItineraryDay,
  catalog: PoiCatalog,
  focusPoiId?: string
): ItineraryActivity[] {
  const remaining = [...day.activities];
  const ordered: ItineraryActivity[] = [];

  if (focusPoiId) {
    const focusIndex = remaining.findIndex((activity) => activity.poiId === focusPoiId);
    if (focusIndex >= 0) {
      ordered.push(remaining.splice(focusIndex, 1)[0]);
    }
  }

  while (remaining.length > 0) {
    const anchor = ordered.at(-1);
    const anchorPoi = anchor?.poiId ? catalog.items[anchor.poiId] : undefined;

    if (!anchorPoi) {
      ordered.push(remaining.shift()!);
      continue;
    }

    let bestIndex = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let index = 0; index < remaining.length; index += 1) {
      const candidatePoi = remaining[index].poiId
        ? catalog.items[remaining[index].poiId!]
        : undefined;
      const nextDistance = distanceKm(anchorPoi, candidatePoi);
      if (nextDistance < bestDistance) {
        bestDistance = nextDistance;
        bestIndex = index;
      }
    }

    ordered.push(remaining.splice(bestIndex, 1)[0]);
  }

  return ordered;
}

function buildMutationMemory(
  session: SessionSnapshot,
  plan: PlanSnapshot,
  decision: string,
  dateContextOverrides?: Partial<SessionSnapshot["memory"]["dateContext"]>
): SessionSnapshot["memory"] {
  let acceptedDecisions = appendUnique(session.memory.acceptedDecisions, decision);
  const destinationsDiscussed = Array.from(
    new Set([...session.memory.destinationsDiscussed, ...plan.destinations])
  );
  const effectiveFlexibility =
    dateContextOverrides?.flexibility ||
    (plan.startDate && plan.endDate ? "exact" : session.memory.dateContext.flexibility);

  if (plan.destination) {
    acceptedDecisions = appendUnique(
      acceptedDecisions,
      `Destination: ${plan.destination}`
    );
  }
  acceptedDecisions = appendUnique(
    acceptedDecisions,
    `Duration: ${plan.totalDays} days`
  );
  if (plan.budgetTarget) {
    acceptedDecisions = appendUnique(
      acceptedDecisions,
      `Budget target: ${plan.budgetTarget.currency} ${plan.budgetTarget.total.toLocaleString()}`
    );
  }
  if (plan.startDate && plan.endDate && effectiveFlexibility === "exact") {
    acceptedDecisions = appendUnique(
      acceptedDecisions,
      `Dates: ${plan.startDate} to ${plan.endDate}`
    );
  } else if (plan.startDate && effectiveFlexibility !== "exact") {
    acceptedDecisions = appendUnique(
      acceptedDecisions,
      `Flexible timing: ${plan.startDate} for ${plan.totalDays} days`
    );
  }

  return {
    ...session.memory,
    summary: decision.slice(0, 400),
    lastPlanVersion: plan.version,
    destinationsDiscussed,
    acceptedDecisions,
    dateContext: {
      ...session.memory.dateContext,
      ...dateContextOverrides,
      inferredStartDate: plan.startDate || session.memory.dateContext.inferredStartDate,
      inferredEndDate: plan.endDate || session.memory.dateContext.inferredEndDate,
      flexibility: effectiveFlexibility,
      derivedFrom:
        dateContextOverrides?.derivedFrom ||
        (plan.startDate && plan.endDate
          ? "explicit"
          : session.memory.dateContext.derivedFrom)
    }
  };
}

function buildOverviewOnlyMemory(
  session: SessionSnapshot,
  mutation: Extract<PlanMutationInput, { type: "update_overview" }>
): SessionSnapshot["memory"] {
  const nextDestinations = mutation.destinations?.length
    ? mutation.destinations
    : mutation.destination
      ? [mutation.destination]
      : [];
  const destinationsDiscussed = Array.from(
    new Set([...session.memory.destinationsDiscussed, ...nextDestinations])
  );

  let acceptedDecisions = [...session.memory.acceptedDecisions];
  acceptedDecisions = upsertDecision(
    acceptedDecisions,
    "Destination: ",
    mutation.destination || mutation.destinations?.[0]
  );
  acceptedDecisions = upsertDecision(acceptedDecisions, "Origin: ", mutation.origin);
  acceptedDecisions = upsertDecision(
    acceptedDecisions,
    "Duration: ",
    mutation.totalDays ? `${mutation.totalDays} days` : undefined
  );
  acceptedDecisions = upsertDecision(
    acceptedDecisions,
    "Travelers: ",
    mutation.travelerCount ? String(mutation.travelerCount) : undefined
  );
  acceptedDecisions = upsertDecision(
    acceptedDecisions,
    "Budget target: ",
    typeof mutation.budgetTotal === "number"
      ? `${mutation.currency || session.memory.preferences.currency || "INR"} ${mutation.budgetTotal.toLocaleString()}`
      : undefined
  );
  acceptedDecisions = upsertDecision(
    acceptedDecisions,
    "Dates: ",
    mutation.startDate && mutation.endDate ? `${mutation.startDate} to ${mutation.endDate}` : undefined
  );

  return {
    ...session.memory,
    summary: "Updated trip overview",
    destinationsDiscussed,
    acceptedDecisions,
    dateContext: {
      ...session.memory.dateContext,
      inferredStartDate: mutation.startDate || session.memory.dateContext.inferredStartDate,
      inferredEndDate: mutation.endDate || session.memory.dateContext.inferredEndDate,
      flexibility:
        mutation.dateFlexibility ||
        (mutation.startDate && mutation.endDate ? "exact" : session.memory.dateContext.flexibility),
      derivedFrom:
        mutation.dateFlexibility || (mutation.startDate && mutation.endDate)
          ? "explicit"
          : session.memory.dateContext.derivedFrom
    }
  };
}

function buildDateContextOverrides(
  session: SessionSnapshot,
  mutation: Extract<PlanMutationInput, { type: "update_overview" }>,
  flexibility?: SessionSnapshot["memory"]["dateContext"]["flexibility"]
): Partial<SessionSnapshot["memory"]["dateContext"]> | undefined {
  if (!flexibility) {
    return undefined;
  }

  const derivedFrom: SessionSnapshot["memory"]["dateContext"]["derivedFrom"] =
    flexibility === "exact" ? "explicit" : "suggested";

  return {
    requestedStartDate: mutation.startDate || session.memory.dateContext.requestedStartDate,
    requestedEndDate: mutation.endDate || session.memory.dateContext.requestedEndDate,
    inferredStartDate: mutation.startDate || session.memory.dateContext.inferredStartDate,
    inferredEndDate: mutation.endDate || session.memory.dateContext.inferredEndDate,
    flexibility,
    derivedFrom
  };
}

function buildConfirmationMessage(
  sessionId: string,
  content: string
): ConversationMessage {
  return {
    id: randomUUID(),
    sessionId,
    role: "assistant",
    content,
    createdAt: new Date().toISOString(),
    phase: "final",
    meta: {
      source: "plan-mutation"
    }
  };
}

function ensurePlan(session: SessionSnapshot): PlanSnapshot {
  if (!session.plan) {
    throw new Error("This session does not have an itinerary yet");
  }

  return clonePlan(session.plan);
}

export class PlanMutationService {
  constructor(
    private repository: SessionRepository,
    private providerService: ProviderService,
    private tools: TravelToolsService
  ) {}

  async apply(
    session: SessionSnapshot,
    userId: string | undefined,
    mutation: PlanMutationInput
  ): Promise<SessionSnapshot> {
    switch (mutation.type) {
      case "add_poi":
        return this.commitPlanChange(
          session,
          userId,
          await this.addPoi(session, mutation)
        );
      case "remove_poi":
        return this.commitPlanChange(
          session,
          userId,
          await this.removePoi(session, mutation)
        );
      case "move_activity":
        return this.commitPlanChange(
          session,
          userId,
          await this.moveActivity(session, mutation)
        );
      case "regenerate_day":
        return this.commitPlanChange(
          session,
          userId,
          await this.regenerateDay(session, userId, mutation)
        );
      case "rebalance_trip":
        return this.commitPlanChange(
          session,
          userId,
          await this.rebalanceTrip(session, userId, mutation)
        );
      case "update_overview":
        return this.commitPlanChange(
          session,
          userId,
          await this.updateOverview(session, userId, mutation)
        );
      default:
        throw new Error("Unsupported plan mutation");
    }
  }

  private async commitPlanChange(
    session: SessionSnapshot,
    userId: string | undefined,
    change: {
      plan?: PlanSnapshot;
      catalog?: PoiCatalog;
      decision: string;
      confirmation?: string;
      sessionTitle?: string;
      sessionMemory?: SessionSnapshot["memory"];
      dateContextOverrides?: Partial<SessionSnapshot["memory"]["dateContext"]>;
    }
  ): Promise<SessionSnapshot> {
    if (!change.plan) {
      const updated = await this.repository.updateSession(session.id, {
        title: change.sessionTitle,
        memory: change.sessionMemory
      });
      if (!updated) {
        throw new Error("Session not found");
      }
      return updated;
    }

    await this.repository.savePlan(session.id, change.plan, change.catalog || session.poiCatalog);
    const memory = buildMutationMemory(
      session,
      change.plan,
      change.decision,
      change.dateContextOverrides
    );
    await this.repository.updateSession(session.id, {
      title: change.plan.title,
      memory
    });

    if (change.confirmation) {
      await this.repository.saveMessage(
        buildConfirmationMessage(session.id, change.confirmation)
      );
    }

    const updated = await this.repository.getSession(session.id, userId);
    if (!updated) {
      throw new Error("Session not found");
    }

    return updated;
  }

  private async addPoi(
    session: SessionSnapshot,
    mutation: Extract<PlanMutationInput, { type: "add_poi" }>
  ) {
    const plan = ensurePlan(session);
    const poi = session.poiCatalog.items[mutation.poiId];
    if (!poi) {
      throw new Error("POI not found in the current session catalog");
    }

    const alreadyPresent = plan.days.some((day) =>
      day.activities.some((activity) => activity.poiId === poi.id)
    );
    if (alreadyPresent) {
      throw new Error(`${poi.name} is already in the itinerary`);
    }

    const { dayIndex, position } = pickBestDayForPoi(
      plan,
      session.poiCatalog,
      poi,
      mutation.day
    );

    plan.days = plan.days.map((day, index) => {
      if (index !== dayIndex) {
        return day;
      }

      const nextDay = {
        ...day,
        activities: insertAt(day.activities, position, buildActivityFromPoi(poi))
      };

      return normalizeDaySchedule(nextDay, session.poiCatalog);
    });

    const nextPlan = await enrichPlanLogistics(
      this.tools,
      finalizePlan(plan),
      session.poiCatalog
    );

    return {
      plan: nextPlan,
      decision: `Added ${poi.name} to day ${plan.days[dayIndex].day}`,
      confirmation: `Added ${poi.name} to day ${plan.days[dayIndex].day} and refreshed travel times.`
    };
  }

  private async removePoi(
    session: SessionSnapshot,
    mutation: Extract<PlanMutationInput, { type: "remove_poi" }>
  ) {
    const plan = ensurePlan(session);
    const poi = session.poiCatalog.items[mutation.poiId];
    const matches = plan.days.flatMap((day) =>
      day.activities.filter((activity) => activity.poiId === mutation.poiId)
    );

    if (matches.length === 0 && !plan.days.some((day) => day.accommodationPoiId === mutation.poiId)) {
      throw new Error("POI is not part of the current itinerary");
    }

    plan.days = plan.days.map((day) =>
      normalizeDaySchedule(
        {
          ...day,
          accommodationPoiId:
            day.accommodationPoiId === mutation.poiId ? undefined : day.accommodationPoiId,
          activities: day.activities.filter((activity) => activity.poiId !== mutation.poiId)
        },
        session.poiCatalog
      )
    );

    const nextPlan = await enrichPlanLogistics(
      this.tools,
      finalizePlan(plan),
      session.poiCatalog
    );

    return {
      plan: nextPlan,
      decision: `Removed ${poi?.name || "the selected place"} from the itinerary`,
      confirmation: `${poi?.name || "The selected place"} was removed and the itinerary was rebalanced.`
    };
  }

  private async moveActivity(
    session: SessionSnapshot,
    mutation: Extract<PlanMutationInput, { type: "move_activity" }>
  ) {
    const plan = ensurePlan(session);
    let activityToMove: ItineraryActivity | undefined;

    plan.days = plan.days.map((day) => {
      const activityIndex = day.activities.findIndex(
        (activity) => activity.id === mutation.activityId
      );
      if (activityIndex === -1) {
        return day;
      }

      activityToMove = day.activities[activityIndex];
      return {
        ...day,
        activities: day.activities.filter(
          (activity) => activity.id !== mutation.activityId
        )
      };
    });

    if (!activityToMove) {
      throw new Error("Activity not found in the current itinerary");
    }

    const targetDayIndex = plan.days.findIndex((day) => day.day === mutation.toDay);
    if (targetDayIndex === -1) {
      throw new Error(`Day ${mutation.toDay} does not exist in this itinerary`);
    }

    plan.days = plan.days.map((day, index) => {
      if (index !== targetDayIndex) {
        return normalizeDaySchedule(day, session.poiCatalog);
      }

      const nextPosition = Math.min(
        mutation.position ?? day.activities.length,
        day.activities.length
      );
      return normalizeDaySchedule(
        {
          ...day,
          activities: insertAt(day.activities, nextPosition, activityToMove!)
        },
        session.poiCatalog
      );
    });

    const nextPlan = await enrichPlanLogistics(
      this.tools,
      finalizePlan(plan),
      session.poiCatalog
    );

    return {
      plan: nextPlan,
      decision: `Moved ${activityToMove.title} to day ${mutation.toDay}`,
      confirmation: `Moved ${activityToMove.title} to day ${mutation.toDay} and updated the route timings.`
    };
  }

  private async regenerateDay(
    session: SessionSnapshot,
    userId: string | undefined,
    mutation: Extract<PlanMutationInput, { type: "regenerate_day" }>
  ) {
    const plan = ensurePlan(session);
    const dayIndex = plan.days.findIndex((day) => day.day === mutation.day);
    if (dayIndex === -1) {
      throw new Error(`Day ${mutation.day} does not exist in this itinerary`);
    }

    const targetDay = plan.days[dayIndex];
    const nextDay = await this.buildRegeneratedDay(
      session,
      userId,
      targetDay,
      mutation.focusPoiId
    );
    plan.days = plan.days.map((day, index) => (index === dayIndex ? nextDay : day));

    const nextPlan = await enrichPlanLogistics(
      this.tools,
      finalizePlan(plan),
      session.poiCatalog
    );

    return {
      plan: nextPlan,
      decision: `Regenerated day ${mutation.day} for ${targetDay.destination}`,
      confirmation: `Regenerated day ${mutation.day} with a fresher flow for ${targetDay.destination}.`
    };
  }

  private async rebalanceTrip(
    session: SessionSnapshot,
    userId: string | undefined,
    mutation: Extract<PlanMutationInput, { type: "rebalance_trip" }>
  ) {
    let plan = ensurePlan(session);
    const focusPoi = mutation.focusPoiId
      ? session.poiCatalog.items[mutation.focusPoiId]
      : undefined;

    const missingFocusedPoi =
      focusPoi &&
      !plan.days.some((day) => day.activities.some((activity) => activity.poiId === focusPoi.id));

    if (focusPoi && missingFocusedPoi) {
      const addResult = await this.addPoi(session, {
        type: "add_poi",
        poiId: focusPoi.id
      });
      plan = clonePlan(addResult.plan!);
    }

    const reorderedDays = plan.days.map((day) =>
      normalizeDaySchedule(
        {
          ...day,
          activities: reorderActivities(day, session.poiCatalog, mutation.focusPoiId)
        },
        session.poiCatalog
      )
    );

    plan.days = await Promise.all(
      reorderedDays.map((day) =>
        this.buildRegeneratedDay(
          session,
          userId,
          day,
          day.activities.some((activity) => activity.poiId === mutation.focusPoiId)
            ? mutation.focusPoiId
            : undefined
        )
      )
    );

    const nextPlan = await enrichPlanLogistics(
      this.tools,
      finalizePlan(plan),
      session.poiCatalog
    );

    return {
      plan: nextPlan,
      decision: focusPoi
        ? `Rebalanced the itinerary around ${focusPoi.name}`
        : "Rebalanced the current itinerary",
      confirmation: focusPoi
        ? `Rebalanced the itinerary around ${focusPoi.name} and refreshed travel times.`
        : "Rebalanced the itinerary for smoother day pacing and travel flow."
    };
  }

  private async updateOverview(
    session: SessionSnapshot,
    userId: string | undefined,
    mutation: Extract<PlanMutationInput, { type: "update_overview" }>
  ) {
    if (!session.plan) {
      return {
        sessionTitle: mutation.title || session.title,
        sessionMemory: buildOverviewOnlyMemory(session, mutation),
        decision: "Updated trip overview"
      };
    }

    const plan = ensurePlan(session);
    const nextBudgetTarget =
      typeof mutation.budgetTotal === "number"
        ? {
            total: mutation.budgetTotal,
            currency:
              mutation.currency ||
              plan.budgetTarget?.currency ||
              plan.budget?.currency ||
              "INR"
          }
        : plan.budgetTarget;
    const nextDateFlexibility =
      mutation.dateFlexibility || (mutation.startDate && mutation.endDate ? "exact" : undefined);

    const requiresRegeneration =
      Boolean(mutation.destination && mutation.destination !== plan.destination) ||
      Boolean(
        mutation.destinations &&
          JSON.stringify(mutation.destinations) !== JSON.stringify(plan.destinations)
      ) ||
      Boolean(mutation.totalDays && mutation.totalDays !== plan.totalDays);

    if (!requiresRegeneration) {
      const nextPlan = await enrichPlanLogistics(
        this.tools,
        finalizePlan(
          shiftPlanDates(
            {
              ...plan,
              title: mutation.title || plan.title,
              origin: mutation.origin || plan.origin,
              destination: mutation.destination || mutation.destinations?.[0] || plan.destination,
              destinations:
                mutation.destinations && mutation.destinations.length > 0
                  ? mutation.destinations
                  : plan.destinations,
              travelerCount: mutation.travelerCount || plan.travelerCount,
              budgetTarget: nextBudgetTarget
            },
            mutation.startDate || plan.startDate,
            mutation.endDate || plan.endDate
          )
        ),
        session.poiCatalog
      );

      return {
        plan: nextPlan,
        decision: "Updated trip overview",
        confirmation: "Updated the trip overview and kept the itinerary synced.",
        dateContextOverrides: buildDateContextOverrides(
          session,
          mutation,
          nextDateFlexibility
        )
      };
    }

    const destination = mutation.destination || mutation.destinations?.[0] || plan.destination;
    if (!destination) {
      throw new Error("A destination is required to regenerate this itinerary");
    }

    const destinations =
      mutation.destinations && mutation.destinations.length > 0
        ? mutation.destinations
        : mutation.destination && mutation.destination !== plan.destination
        ? [destination]
        : plan.destinations.length > 0
          ? plan.destinations
          : [destination];

    const resolvedProvider = await this.providerService.resolveProvider(
      userId,
      session.providerSettings
    );
    const research = await researchDestinations(this.tools, destinations);
    const resolution: TurnResolution = {
      intent: "refine_trip",
      destination,
      destinations,
      origin: mutation.origin || plan.origin,
      totalDays: mutation.totalDays || plan.totalDays,
      travelerCount: mutation.travelerCount || plan.travelerCount,
      budgetNote: nextBudgetTarget
        ? `${nextBudgetTarget.currency} ${nextBudgetTarget.total}`
        : undefined,
      styles: session.memory.preferences.styles,
      dateContext: {
        ...session.memory.dateContext,
        inferredStartDate:
          mutation.startDate ||
          plan.startDate ||
          session.memory.dateContext.inferredStartDate,
        inferredEndDate:
          mutation.endDate ||
          plan.endDate ||
          session.memory.dateContext.inferredEndDate,
        flexibility:
          nextDateFlexibility || session.memory.dateContext.flexibility,
        derivedFrom:
          nextDateFlexibility === "exact"
            ? "explicit"
            : nextDateFlexibility
              ? "suggested"
              : session.memory.dateContext.derivedFrom
      }
    };

    let nextPlan = await synthesizePlan(
      this.providerService,
      resolvedProvider,
      session,
      resolution,
      {
        ...research,
        catalog: mergeCatalogs(session.poiCatalog, research.catalog)
      }
    );
    nextPlan = planSnapshotSchema.parse({
      ...nextPlan,
      title: mutation.title || nextPlan.title,
      origin: mutation.origin || nextPlan.origin,
      destination,
      destinations,
      startDate: mutation.startDate || nextPlan.startDate,
      endDate: mutation.endDate || nextPlan.endDate,
      travelerCount: mutation.travelerCount || nextPlan.travelerCount,
      budgetTarget: nextBudgetTarget
    });
    nextPlan = await enrichPlanLogistics(
      this.tools,
      nextPlan,
      mergeCatalogs(session.poiCatalog, research.catalog)
    );

    return {
      plan: nextPlan,
      catalog: mergeCatalogs(session.poiCatalog, research.catalog),
      decision: "Regenerated the trip overview",
      confirmation: "Updated the trip overview and regenerated the itinerary to keep every panel in sync.",
      dateContextOverrides: buildDateContextOverrides(
        session,
        mutation,
        nextDateFlexibility
      )
    };
  }

  private async buildRegeneratedDay(
    session: SessionSnapshot,
    userId: string | undefined,
    day: ItineraryDay,
    focusPoiId?: string
  ): Promise<ItineraryDay> {
    const matchingPois = Object.values(session.poiCatalog.items).filter(
      (poi) =>
        poi.type !== "stay" &&
        matchesDestination(poi, day.destination)
    );

    const accommodations = Object.values(session.poiCatalog.items).filter(
      (poi) => poi.type === "stay" && matchesDestination(poi, day.destination)
    );

    if (matchingPois.length === 0) {
      throw new Error(`No POIs were found for ${day.destination} in this session`);
    }

    if (!userId) {
      throw new Error("A signed-in user is required to regenerate a day");
    }

    const resolvedProvider = await this.providerService.resolveProvider(
      userId,
      session.providerSettings
    );
    const draft = await this.providerService.generateObject({
      resolved: resolvedProvider,
      schema: regeneratedDaySchema,
      schemaName: "regenerated_day",
      instructions:
        "You are regenerating one travel day. Use only the provided poiId values and keep the flow realistic.",
      input: `Current session title: ${session.title}
Current destination: ${day.destination}
Current day number: ${day.day}
Focus POI: ${focusPoiId || "none"}
Available accommodation IDs: ${accommodations.map((poi) => poi.id).join(", ") || "none"}
Available activity POIs:
${JSON.stringify(
  matchingPois.map((poi) => ({
    id: poi.id,
    name: poi.name,
    type: poi.type,
    rating: poi.rating,
    description: poi.description
  })),
  null,
  2
)}

Rules:
- Return 3 to 5 activities.
- Use the exact poiId values.
- Include the focus POI if one was provided.
- Keep the day grounded in ${day.destination}.`
    });

    return normalizeDaySchedule(
      {
        ...day,
        title: draft.title,
        theme: draft.theme,
        summary: draft.summary,
        accommodationPoiId: draft.accommodationPoiId || day.accommodationPoiId,
        activities: draft.activities.map((activity) => ({
          id: randomUUID(),
          poiId: activity.poiId,
          title: activity.title,
          summary: activity.summary,
          startTime: "09:00",
          endTime: "10:00",
          notes: activity.notes
        }))
      },
      session.poiCatalog
    );
  }
}
