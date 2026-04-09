import type {
  PendingFollowUpContext,
  PlanMutationInput,
  SessionMemory,
  SessionSnapshot
} from "@roameo/contracts";
import { PlanMutationService } from "./plan-mutation-service.js";
import { SessionRepository } from "./session-repository.js";

type HeaderToolInput = Extract<PlanMutationInput, { type: "update_overview" }>;
type ItineraryToolInput = Exclude<PlanMutationInput, { type: "update_overview" }>;

export type SessionMemoryToolInput = {
  summary?: string;
  replaceDestinationsDiscussed?: string[];
  appendDestinationsDiscussed?: string[];
  replaceAcceptedDecisions?: string[];
  appendAcceptedDecisions?: string[];
  dateContext?: Partial<SessionMemory["dateContext"]>;
  planningState?: Partial<SessionMemory["planningState"]>;
  pendingFollowUp?: PendingFollowUpContext | null;
  clearPendingFollowUp?: boolean;
};

export type ActiveTripContextInput = {
  destination?: string;
  destinations?: string[];
  origin?: string;
  totalDays?: number;
  travelerCount?: number;
  budgetNote?: string;
  startDate?: string;
  endDate?: string;
  explicitNewTrip?: boolean;
};

function replaceDecision(values: string[], prefix: string, next?: string): string[] {
  const filtered = values.filter((value) => !value.startsWith(prefix));
  return next ? [...filtered, `${prefix}${next}`] : filtered;
}

function appendUnique(values: string[], next: string): string[] {
  return values.includes(next) ? values : [...values, next];
}

/**
 * First-class internal tool surface for autonomous agents.
 * Keeps all session and itinerary mutations on canonical repository/service paths.
 */
export class AgentToolService {
  constructor(
    private repository: SessionRepository,
    private planMutationService: PlanMutationService
  ) {}

  async getSessionSnapshot(
    sessionId: string,
    userId?: string
  ): Promise<SessionSnapshot> {
    const session = await this.repository.getSession(sessionId, userId);
    if (!session) {
      throw new Error("Session not found");
    }
    return session;
  }

  async updateTripHeader(
    sessionId: string,
    userId: string | undefined,
    input: Omit<HeaderToolInput, "type">
  ): Promise<SessionSnapshot> {
    const session = await this.getSessionSnapshot(sessionId, userId);
    return this.planMutationService.apply(session, userId, {
      type: "update_overview",
      ...input
    });
  }

  async editItinerary(
    sessionId: string,
    userId: string | undefined,
    input: ItineraryToolInput
  ): Promise<SessionSnapshot> {
    const session = await this.getSessionSnapshot(sessionId, userId);
    return this.planMutationService.apply(session, userId, input);
  }

  async updateSessionMemory(
    sessionId: string,
    userId: string | undefined,
    input: SessionMemoryToolInput
  ): Promise<SessionSnapshot> {
    const session = await this.getSessionSnapshot(sessionId, userId);
    const nextMemory: SessionMemory = {
      ...session.memory,
      summary: input.summary ?? session.memory.summary,
      destinationsDiscussed:
        input.replaceDestinationsDiscussed ??
        (input.appendDestinationsDiscussed?.reduce(appendUnique, [
          ...session.memory.destinationsDiscussed
        ]) || session.memory.destinationsDiscussed),
      acceptedDecisions:
        input.replaceAcceptedDecisions ??
        (input.appendAcceptedDecisions?.reduce(appendUnique, [
          ...session.memory.acceptedDecisions
        ]) || session.memory.acceptedDecisions),
      dateContext: {
        ...session.memory.dateContext,
        ...(input.dateContext || {})
      },
      planningState: {
        ...session.memory.planningState,
        ...(input.planningState || {})
      },
      pendingFollowUp: input.clearPendingFollowUp
        ? null
        : input.pendingFollowUp !== undefined
          ? input.pendingFollowUp
          : session.memory.pendingFollowUp
    };

    const updated = await this.repository.updateSession(sessionId, {
      memory: nextMemory
    });
    if (!updated) {
      throw new Error("Session not found");
    }
    return updated;
  }

  async resetActiveTripContext(
    sessionId: string,
    userId: string | undefined,
    input: ActiveTripContextInput
  ): Promise<SessionSnapshot> {
    const session = await this.getSessionSnapshot(sessionId, userId);
    const destinations =
      input.destinations?.filter(Boolean) ||
      (input.destination ? [input.destination] : []);
    let acceptedDecisions = [...session.memory.acceptedDecisions];

    acceptedDecisions = replaceDecision(
      acceptedDecisions,
      "Destination: ",
      input.destination || destinations[0]
    );
    acceptedDecisions = replaceDecision(
      acceptedDecisions,
      "Origin: ",
      input.origin
    );
    acceptedDecisions = replaceDecision(
      acceptedDecisions,
      "Duration: ",
      input.totalDays ? `${input.totalDays} days` : undefined
    );
    acceptedDecisions = replaceDecision(
      acceptedDecisions,
      "Travelers: ",
      input.travelerCount ? String(input.travelerCount) : undefined
    );
    acceptedDecisions = replaceDecision(
      acceptedDecisions,
      "Budget target: ",
      input.budgetNote
    );
    acceptedDecisions = replaceDecision(
      acceptedDecisions,
      "Dates: ",
      input.startDate && input.endDate ? `${input.startDate} to ${input.endDate}` : undefined
    );

    const nextMemory: SessionMemory = {
      ...session.memory,
      destinationsDiscussed: input.explicitNewTrip
        ? destinations
        : destinations.reduce(appendUnique, [...session.memory.destinationsDiscussed]),
      acceptedDecisions,
      dateContext: {
        ...session.memory.dateContext,
        inferredStartDate: input.startDate || session.memory.dateContext.inferredStartDate,
        inferredEndDate: input.endDate || session.memory.dateContext.inferredEndDate,
        flexibility:
          input.startDate && input.endDate
            ? "exact"
            : session.memory.dateContext.flexibility,
        derivedFrom:
          input.startDate && input.endDate
            ? "explicit"
            : session.memory.dateContext.derivedFrom
      },
      pendingFollowUp: null
    };

    const updated = await this.repository.updateSession(sessionId, {
      title: input.destination || session.title,
      memory: nextMemory
    });
    if (!updated) {
      throw new Error("Session not found");
    }
    return updated;
  }

  async saveFollowUpContext(
    sessionId: string,
    userId: string | undefined,
    pendingFollowUp: PendingFollowUpContext | null
  ): Promise<SessionSnapshot> {
    return this.updateSessionMemory(sessionId, userId, {
      pendingFollowUp
    });
  }
}
