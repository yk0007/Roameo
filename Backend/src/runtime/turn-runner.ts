import { randomUUID } from "node:crypto";
import type {
  PlanningState,
  ConversationMessage,
  SendMessageInput,
  SessionSnapshot,
  StreamEvent
} from "@roameo/contracts";
import { SessionRepository } from "../services/session-repository.js";
import { ProviderService, type ResolvedProvider } from "../services/provider-service.js";
import {
  PlanningToolError,
  TravelToolsService
} from "../services/travel-tools.js";
import { StreamHub } from "../core/stream-hub.js";
import {
  answerConversationally,
  buildResponseBlocks,
  criticizeAndRefinePlan,
  derivePendingFollowUpContext,
  enrichPlanLogistics,
  type ImmediateAssistantReply,
  researchPlanningDestinations,
  resolveFastTurnResponse,
  resolveDeterministicTurnIntent,
  resolveOversizedPlanResponse,
  type PlanningContext,
  researchDestinations,
  resolveDiscoveryFocus,
  resolveTurnIntent,
  shouldResearchResolution,
  synthesizePlan,
  transitAdvisor,
  updateSessionMemory
} from "./subagents.js";

function chunkText(value: string): string[] {
  const sentences = value
    .split(/(?<=[.!?])\s+/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);
  return sentences.length > 0 ? sentences : [value];
}

function createPlanningState(
  current: SessionSnapshot["memory"]["planningState"],
  next: Partial<PlanningState>
): PlanningState {
  return {
    ...current,
    ...next,
    updatedAt: new Date().toISOString()
  };
}

function deriveSessionTitle(
  session: SessionSnapshot,
  resolution: {
    destination?: string;
    destinations: string[];
    totalDays?: number;
    explicitNewTrip?: boolean;
  },
  plan?: SessionSnapshot["plan"]
): string | undefined {
  if (plan?.title?.trim()) {
    return plan.title.trim();
  }

  const currentTitle = session.title?.trim();
  const destination =
    resolution.destination ||
    resolution.destinations[0] ||
    session.plan?.destination ||
    session.memory.destinationsDiscussed.at(-1);
  const totalDays = resolution.totalDays || session.plan?.totalDays;

  if (!destination) {
    return currentTitle;
  }

  if (!currentTitle || /^untitled trip$/i.test(currentTitle) || resolution.explicitNewTrip) {
    return totalDays ? `${destination}, ${totalDays} days` : destination;
  }

  return currentTitle;
}

export class TurnRunner {
  private inflight = new Map<string, Promise<void>>();

  constructor(
    private repository: SessionRepository,
    private providerService: ProviderService,
    private tools: TravelToolsService,
    private streamHub: StreamHub
  ) {}

  async runTurn(
    session: SessionSnapshot,
    userId: string | undefined,
    input: SendMessageInput
  ): Promise<void> {
    const existing = this.inflight.get(session.id);
    if (existing) {
      await existing;
    }

    const job = this.executeTurn(session, userId, input);
    this.inflight.set(session.id, job);

    try {
      await job;
    } finally {
      this.inflight.delete(session.id);
    }
  }

  private async executeTurn(
    session: SessionSnapshot,
    userId: string | undefined,
    input: SendMessageInput
  ): Promise<void> {
    const turnId = randomUUID();
    const settings = input.providerSettings || session.providerSettings;

    const userMessage: ConversationMessage = {
      id: randomUUID(),
      sessionId: session.id,
      role: "user",
      content: input.content,
      createdAt: new Date().toISOString(),
      meta: {}
    };
    await this.repository.saveMessage(userMessage);
    this.streamHub.emit(session.id, {
      type: "message.committed",
      data: userMessage
    });

    this.streamHub.emit(session.id, {
      type: "turn.started",
      data: {
        sessionId: session.id,
        turnId,
        providerSettings: settings
      }
    });

    const fastTurn = resolveFastTurnResponse(session, input.content);
    if (fastTurn) {
      await commitImmediateAssistantReply({
        repository: this.repository,
        streamHub: this.streamHub,
        session,
        settings,
        turnId,
        reply: fastTurn,
        memory: fastTurn.memory
      });
      return;
    }

    let resolvedProvider: ResolvedProvider | null = null;
    const getResolvedProvider = async (): Promise<ResolvedProvider> => {
      if (resolvedProvider) {
        return resolvedProvider;
      }
      resolvedProvider = await this.providerService.resolveProvider(
        userId,
        settings
      );
      return resolvedProvider;
    };
    const currentProvider = () =>
      resolvedProvider ? resolvedProvider.provider : undefined;

    const persistPlanningState = async (
      baseSession: SessionSnapshot,
      next: Partial<PlanningState>
    ) => {
      const updated = await this.repository.updateSession(session.id, {
        providerSettings: settings,
        memory: {
          ...baseSession.memory,
          planningState: createPlanningState(baseSession.memory.planningState, next)
        }
      });

      if (updated) {
        this.streamHub.emit(session.id, {
          type: "session.snapshot",
          data: updated
        });
        return updated;
      }

      return baseSession;
    };

    try {
      let nextSession = await persistPlanningState(session, {
        status: "running",
        stage: "understanding",
        source: "provider",
        reason: undefined,
        retryable: true
      });

      let resolution = resolveDeterministicTurnIntent(nextSession, input.content);
      if (!resolution) {
        resolution = await resolveTurnIntent(
          this.providerService,
          await getResolvedProvider(),
          nextSession,
          input.content
        );
      }

      const oversizedPlanResponse = resolveOversizedPlanResponse(resolution);
      if (oversizedPlanResponse) {
        const nextMemory = updateSessionMemory(
          nextSession,
          resolution,
          oversizedPlanResponse.reply,
          undefined,
          null
        );
        nextMemory.planningState = createPlanningState(
          nextSession.memory.planningState,
          {
            status: "ready",
            stage: "ready",
            source: undefined,
            reason: undefined,
            retryable: true
          }
        );
        await commitImmediateAssistantReply({
          repository: this.repository,
          streamHub: this.streamHub,
          session,
          settings,
          turnId,
          reply: oversizedPlanResponse,
          memory: nextMemory,
          title: deriveSessionTitle(nextSession, resolution, undefined)
        });
        return;
      }

      let reply = "";
      let latestPlan = nextSession.plan;
      const planningContext: PlanningContext = {
        workerProgress: []
      };
      const pushProgress = (
        label: string,
        detail?: string,
        state: "running" | "completed" = "completed"
      ) => {
        planningContext.workerProgress?.push({ label, detail, state });
      };

      const shouldResearch = shouldResearchResolution(resolution);

      let research;
      if (shouldResearch) {
        nextSession = await persistPlanningState(nextSession, {
          status: "running",
          stage: resolution.stayMode ? "researching_stays" : "researching",
          source: resolution.stayMode ? "stays" : "places",
          reason: undefined,
          retryable: true
        });
        const focusLabel = (() => {
          const focus = resolveDiscoveryFocus(resolution);
          switch (focus) {
            case "restaurants": return "Searching restaurants";
            case "seafood": return "Finding seafood spots";
            case "hotels": return "Scouting stays";
            case "hidden_gems": return "Discovering hidden gems";
            case "beaches": return "Finding beaches";
            case "culture": return "Researching culture spots";
            case "attractions": return "Discovering attractions";
            case "family": return "Finding family-friendly places";
            case "day_trips": return "Looking up day trips";
            default: return resolution.stayMode ? "Scouting stays" : "Discovering places";
          }
        })();
        research =
          resolution.intent === "plan_trip" || resolution.intent === "refine_trip"
            ? await researchPlanningDestinations(
                this.tools,
                resolution.destinations,
                resolveDiscoveryFocus(resolution),
                resolution.intent === "plan_trip"
              )
            : await researchDestinations(
                this.tools,
                resolution.destinations,
                resolveDiscoveryFocus(resolution),
                false
              );
        const poiCount = Object.keys(research.catalog.items).length;
        pushProgress(
          resolution.stayMode ? "Scouting local stays" : `Discovering places in ${resolution.destinations.join(", ")}`,
          `Pulled ${poiCount} real places into the catalog.`
        );

        if (!(resolution.intent === "plan_trip" || resolution.intent === "refine_trip")) {
          await this.repository.savePoiCatalog(session.id, research.catalog);
          nextSession = (await this.repository.getSession(session.id, userId)) || nextSession;
        }
      }

      const isMissingInfo = resolution.intent === "plan_trip" && (!resolution.totalDays && !nextSession.plan?.totalDays);

      if (
        !isMissingInfo &&
        (resolution.dateContext.inferredStartDate ||
          resolution.questionFocus === "events" ||
          resolution.intent === "plan_trip" ||
          resolution.intent === "refine_trip")
      ) {
        nextSession = await persistPlanningState(nextSession, {
          status: "running",
          stage: "checking_dates",
          source: "weather",
          reason: undefined,
          retryable: true
        });
        planningContext.weather = await this.tools.getWeatherSummary(
          resolution.destination || resolution.destinations[0] || nextSession.plan?.destination || "",
          resolution.dateContext
        );
        pushProgress(
          "Checking your dates",
          planningContext.weather.summary || "Dates normalized and weather notes prepared."
        );

        nextSession = await persistPlanningState(nextSession, {
          status: "running",
          stage: "researching_events",
          source: "events",
          reason: undefined,
          retryable: true
        });
        const destinationForEvents =
          resolution.destination || resolution.destinations[0] || nextSession.plan?.destination || "";
        planningContext.events = await this.tools.getEventSummary(
          destinationForEvents,
          resolution.dateContext
        );
        planningContext.holidays = await this.tools.getHolidaySummary(
          destinationForEvents,
          resolution.dateContext
        );
        pushProgress(
          "Checking festivals and local timing",
          planningContext.events.summary ||
            planningContext.holidays.summary ||
            "Added event and holiday context where available."
        );
      }

      if (research && (resolution.intent === "plan_trip" || resolution.intent === "refine_trip") && !isMissingInfo) {
        nextSession = await persistPlanningState(nextSession, {
          status: "running",
          stage:
            resolution.intent === "refine_trip" ? "refining" : "building_plan",
          source: "provider",
          reason: undefined,
          retryable: true
        });
        let plan = await synthesizePlan(
          this.providerService,
          await getResolvedProvider(),
          nextSession,
          resolution,
          research
        );
        plan = await enrichPlanLogistics(this.tools, plan, research.catalog);

        // ── Feasibility Critic pass ────────────────────────────────────────────
        // Non-blocking: the critic mutates the plan in-memory (trim over-scheduled
        // days, flag long transfers, fill missing accommodation).  Each critique
        // is emitted as an agent trace visible in the agentic status panel.
        const { plan: critiquedPlan, critiques } = criticizeAndRefinePlan(plan, research.catalog);
        plan = critiquedPlan;

        // ── Transit Advisor pass ───────────────────────────────────────────────
        // Only meaningful for multi-destination trips.
        if (plan.destinationSegments.length > 1) {
          const { segments } = transitAdvisor(plan, research);
          void segments;
        }

        void critiques;

        latestPlan = plan;
        pushProgress(
          "Building the itinerary",
          "Scored the strongest places, checked routing, and assembled the day-by-day plan."
        );

        await this.repository.savePlan(session.id, plan, research.catalog);
        this.streamHub.emit(session.id, {
          type: "plan.updated",
          data: {
            sessionId: session.id,
            plan,
            poiCatalog: research.catalog
          }
        });

        nextSession = (await this.repository.getSession(session.id, userId)) || nextSession;
      }

      const narrative = await answerConversationally(
        this.providerService,
        await getResolvedProvider(),
        nextSession,
        resolution,
        research,
        planningContext
      );
      reply = [narrative.introBody, narrative.leadText].filter(Boolean).join(" ");
      const responseBlocks = buildResponseBlocks({
        session: nextSession,
        resolution,
        narrative,
        research,
        plan: latestPlan,
        planningContext
      });
      const followUpContext = derivePendingFollowUpContext({
        resolution,
        narrative,
        responseBlocks,
        assistantReply: reply,
        plan: latestPlan
      });

      const assistantMessageId = randomUUID();
      const chunks = chunkText(reply);
      for (let index = 0; index < chunks.length; index += 1) {
        const event: StreamEvent = {
          type: "message.delta",
          data: {
            sessionId: session.id,
            turnId,
            messageId: assistantMessageId,
            role: "assistant",
            delta: chunks[index],
            done: index === chunks.length - 1
          }
        };
        this.streamHub.emit(session.id, event);
      }

      const assistantMessage: ConversationMessage = {
        id: assistantMessageId,
        sessionId: session.id,
        role: "assistant",
        content: reply,
        createdAt: new Date().toISOString(),
        phase: "final",
        meta: {
          provider: currentProvider(),
          turnId,
          responseBlocks,
          followUpContext: followUpContext || undefined
        }
      };
      await this.repository.saveMessage(assistantMessage);
      this.streamHub.emit(session.id, {
        type: "message.committed",
        data: assistantMessage
      });

      const nextMemory = updateSessionMemory(
        nextSession,
        resolution,
        reply,
        latestPlan,
        followUpContext
      );
      nextMemory.planningState = createPlanningState(
        nextSession.memory.planningState,
        {
          status: "ready",
          stage: "ready",
          source: undefined,
          reason: undefined,
          retryable: true
        }
      );
      const updated = await this.repository.updateSession(session.id, {
        providerSettings: settings,
        preferences: nextMemory.preferences,
        memory: nextMemory,
        title: deriveSessionTitle(nextSession, resolution, latestPlan)
      });

      if (updated) {
        updated.memory = nextMemory;
        this.streamHub.emit(session.id, {
          type: "session.snapshot",
          data: updated
        });
      }

      this.streamHub.emit(session.id, {
        type: "turn.completed",
        data: {
          sessionId: session.id,
          turnId
        }
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unexpected turn failure";
      const failedPlanningState = createPlanningState(
        session.memory.planningState,
        error instanceof PlanningToolError
          ? {
              status: "unavailable",
              stage: "unavailable",
              source: error.source,
              reason: message,
              retryable: error.retryable
            }
          : {
              status: "unavailable",
              stage: "unavailable",
              source: "provider",
              reason: message,
              retryable: true
          }
      );
      const updated = await this.repository.updateSession(session.id, {
        providerSettings: settings,
        memory: {
          ...(session.memory || {}),
          planningState: failedPlanningState
        }
      });
      if (updated) {
        this.streamHub.emit(session.id, {
          type: "session.snapshot",
          data: updated
        });
      }

      const assistantMessage: ConversationMessage = {
        id: randomUUID(),
        sessionId: session.id,
        role: "assistant",
        content:
          "I hit a temporary planning issue, so I kept your current trip unchanged. You can retry in a moment and I’ll pick it back up.",
        createdAt: new Date().toISOString(),
        phase: "final",
        meta: {
          provider: currentProvider(),
          turnId,
          responseBlocks: [
            {
              type: "trip_intro",
              title: "I kept your current trip safe",
              body: "Planning hit a temporary issue, so I left your accepted itinerary exactly as it was.",
              moodEmoji: "⚠️"
            },
            {
              type: "lead",
              text: "You can retry in a moment, or tell me what you want to adjust and I’ll continue from the last accepted plan."
            },
            {
              type: "planning_status",
              state: "unavailable",
              stage: "unavailable",
              label: "Planning is temporarily unavailable",
              detail: message
            },
            {
              type: "assistant_prompt_chips",
              title: "Try next",
              prompts: [
                {
                  label: "Retry now",
                  prompt: "Please try building the plan again."
                }
              ]
            }
          ]
        }
      };
      await this.repository.saveMessage(assistantMessage);
      this.streamHub.emit(session.id, {
        type: "message.committed",
        data: assistantMessage
      });
      this.streamHub.emit(session.id, {
        type: "turn.failed",
        data: {
          sessionId: session.id,
          turnId,
          error: message
        }
      });
      throw error;
    }
  }
}

async function commitImmediateAssistantReply(params: {
  repository: SessionRepository;
  streamHub: StreamHub;
  session: SessionSnapshot;
  settings: SessionSnapshot["providerSettings"];
  turnId: string;
  reply: ImmediateAssistantReply;
  memory: SessionSnapshot["memory"];
  title?: string;
}): Promise<void> {
  const {
    repository,
    streamHub,
    session,
    settings,
    turnId,
    reply,
    memory,
    title
  } = params;
  const assistantMessageId = randomUUID();
  const chunks = chunkText(reply.reply);

  for (let index = 0; index < chunks.length; index += 1) {
    streamHub.emit(session.id, {
      type: "message.delta",
      data: {
        sessionId: session.id,
        turnId,
        messageId: assistantMessageId,
        role: "assistant",
        delta: chunks[index],
        done: index === chunks.length - 1
      }
    });
  }

  const assistantMessage: ConversationMessage = {
    id: assistantMessageId,
    sessionId: session.id,
    role: "assistant",
    content: reply.reply,
    createdAt: new Date().toISOString(),
    phase: "final",
    meta: {
      turnId,
      responseBlocks: reply.responseBlocks
    }
  };
  await repository.saveMessage(assistantMessage);
  streamHub.emit(session.id, {
    type: "message.committed",
    data: assistantMessage
  });

  const updated = await repository.updateSession(session.id, {
    providerSettings: settings,
    memory,
    title
  });
  if (updated) {
    updated.memory = memory;
    streamHub.emit(session.id, {
      type: "session.snapshot",
      data: updated
    });
  }

  streamHub.emit(session.id, {
    type: "turn.completed",
    data: {
      sessionId: session.id,
      turnId
    }
  });
}
