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
  enrichPlanLogistics,
  researchDestinations,
  resolveTurnIntent,
  synthesizePlan,
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

      const shouldResearch =
        resolution.destinations.length > 0 &&
        (resolution.intent === "plan_trip" ||
          resolution.intent === "refine_trip" ||
          resolution.intent === "search_places");

      let research;
      if (shouldResearch) {
        nextSession = await persistPlanningState(nextSession, {
          status: "running",
          stage: "researching",
          source: "places",
          reason: undefined,
          retryable: true
        });
        await emitTrace(
          "destination-research-agent",
          "running",
          `Researching ${resolution.destinations.join(", ")}`
        );
        research = await researchDestinations(this.tools, resolution.destinations);
        await emitTrace(
          "destination-research-agent",
          "completed",
          `Fetched ${Object.keys(research.catalog.items).length} POIs`
        );

        if (!(resolution.intent === "plan_trip" || resolution.intent === "refine_trip")) {
          await this.repository.savePoiCatalog(session.id, research.catalog);
          nextSession = (await this.repository.getSession(session.id, userId)) || nextSession;
        }
      }

      if (research && (resolution.intent === "plan_trip" || resolution.intent === "refine_trip")) {
        nextSession = await persistPlanningState(nextSession, {
          status: "running",
          stage:
            resolution.intent === "refine_trip" ? "refining" : "building_plan",
          source: "provider",
          reason: undefined,
          retryable: true
        });
        await emitTrace("itinerary-synthesis-agent", "running", "Building plan");
        let plan = await synthesizePlan(
          this.providerService,
          resolvedProvider,
          nextSession,
          resolution,
          research
        );
        await emitTrace("feasibility-validator", "running", "Validating logistics");
        plan = await enrichPlanLogistics(this.tools, plan, research.catalog);
        await emitTrace("feasibility-validator", "completed", "Plan validated");
        latestPlan = plan;

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
        resolvedProvider,
        nextSession,
        resolution,
        research
      );
      reply = [narrative.introBody, narrative.leadText].filter(Boolean).join(" ");
      const responseBlocks = buildResponseBlocks({
        session: nextSession,
        resolution,
        narrative,
        research,
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
          responseBlocks
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
        latestPlan
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
