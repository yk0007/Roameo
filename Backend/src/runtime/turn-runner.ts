import { randomUUID } from "node:crypto";
import type {
  AgentTraceEvent,
  PlanningState,
  ConversationMessage,
  SendMessageInput,
  SessionSnapshot,
  StreamEvent
} from "@roameo/contracts";
import { SessionRepository } from "../services/session-repository.js";
import { ProviderService } from "../services/provider-service.js";
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
  type PlanningContext,
  researchDestinations,
  resolveDiscoveryFocus,
  resolveTurnIntent,
  shouldResearchResolution,
  synthesizePlan,
  transitAdvisor,
  updateSessionMemory
} from "./subagents.js";

function createTrace(
  sessionId: string,
  turnId: string,
  agent: string,
  status: AgentTraceEvent["status"],
  label: string,
  detail?: string
): AgentTraceEvent {
  return {
    id: randomUUID(),
    sessionId,
    turnId,
    agent,
    status,
    label,
    detail,
    createdAt: new Date().toISOString()
  };
}

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
    const resolvedProvider = await this.providerService.resolveProvider(
      userId,
      settings
    );

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

    const emitTrace = async (
      agent: string,
      status: AgentTraceEvent["status"],
      label: string,
      detail?: string
    ) => {
      const trace = createTrace(session.id, turnId, agent, status, label, detail);
      await this.repository.saveTrace(trace);
      this.streamHub.emit(session.id, { type: "trace.updated", data: trace });
    };

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

      await emitTrace("lead", "running", "Resolving intent");
      const resolution = await resolveTurnIntent(
        this.providerService,
        resolvedProvider,
        nextSession,
        input.content
      );
      await emitTrace("intent-slot-resolver", "completed", resolution.intent);

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
        await emitTrace(
          resolution.stayMode ? "stay-search-agent" : "discovery-search-agent",
          "running",
          `${focusLabel} in ${resolution.destinations.join(", ")}`
        );
        research = await researchDestinations(
          this.tools,
          resolution.destinations,
          resolveDiscoveryFocus(resolution),
          // Use Tavily deep-research for plan_trip so the LLM has richer editorial
          // context (best time to visit, local tips, culture notes, etc.)
          resolution.intent === "plan_trip"
        );
        const poiCount = Object.keys(research.catalog.items).length;
        await emitTrace(
          resolution.stayMode ? "stay-search-agent" : "discovery-search-agent",
          "completed",
          `Found ${poiCount} places`,
          `Pulled ${poiCount} real places into the catalog.`
        );
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
        await emitTrace("date-context-agent", "running", "Checking weather and timing");
        planningContext.weather = await this.tools.getWeatherSummary(
          resolution.destination || resolution.destinations[0] || nextSession.plan?.destination || "",
          resolution.dateContext
        );
        await emitTrace("date-context-agent", "completed", "Date and weather notes ready");
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
        await emitTrace("events-culture-agent", "running", "Searching festivals and local events");
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
        await emitTrace("events-culture-agent", "completed", "Festival and holiday notes ready");
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
        await emitTrace("itinerary-planner", "running", "Building your itinerary");
        let plan = await synthesizePlan(
          this.providerService,
          resolvedProvider,
          nextSession,
          resolution,
          research
        );
        await emitTrace("feasibility-validator", "running", "Evaluating logistics and routing");
        plan = await enrichPlanLogistics(this.tools, plan, research.catalog);

        // ── Feasibility Critic pass ────────────────────────────────────────────
        // Non-blocking: the critic mutates the plan in-memory (trim over-scheduled
        // days, flag long transfers, fill missing accommodation).  Each critique
        // is emitted as an agent trace visible in the agentic status panel.
        await emitTrace("feasibility-critic", "running", "Running feasibility check");
        const { plan: critiquedPlan, critiques } = criticizeAndRefinePlan(plan, research.catalog);
        plan = critiquedPlan;
        for (const critique of critiques) {
          await emitTrace("feasibility-critic", "completed", critique);
        }
        await emitTrace("feasibility-validator", "completed", "Logistics validated");

        // ── Transit Advisor pass ───────────────────────────────────────────────
        // Only meaningful for multi-destination trips.
        if (plan.destinationSegments.length > 1) {
          await emitTrace("transit-advisor", "running", "Planning inter-city transit");
          const { segments } = transitAdvisor(plan, research);
          for (const seg of segments) {
            await emitTrace(
              "transit-advisor",
              "completed",
              `${seg.from} → ${seg.to}: ${seg.durationLabel}`,
              seg.bookingNote
            );
          }
        }

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

      const narrativeTraceLabel =
        resolution.intent === "plan_trip" || resolution.intent === "refine_trip"
          ? "Crafting your travel narrative"
          : "Drafting response";
      await emitTrace("narrator", "running", narrativeTraceLabel);
      const narrative = await answerConversationally(
        this.providerService,
        resolvedProvider,
        nextSession,
        resolution,
        research,
        planningContext
      );
      await emitTrace("narrator", "completed", "Response ready");
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
          provider: resolvedProvider.provider,
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
        title: latestPlan?.title
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
          provider: resolvedProvider.provider,
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
      await emitTrace("lead", "failed", "Turn failed", message);
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
