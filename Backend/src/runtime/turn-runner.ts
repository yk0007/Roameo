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

    try {
      let nextSession =
        (await this.repository.updateSession(session.id, {
          providerSettings: settings,
          memory: {
            ...session.memory,
            planningState: createPlanningState(session.memory.planningState, {
              status: "running",
              source: "provider",
              reason: undefined,
              retryable: true
            })
          }
        })) || session;

      this.streamHub.emit(session.id, {
        type: "session.snapshot",
        data: nextSession
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
      }

      if (research && (resolution.intent === "plan_trip" || resolution.intent === "refine_trip")) {
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

      reply = await answerConversationally(
        this.providerService,
        resolvedProvider,
        nextSession,
        resolution,
        research
      );
      const responseBlocks = buildResponseBlocks({
        session: nextSession,
        resolution,
        reply,
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
              source: error.source,
              reason: message,
              retryable: error.retryable
            }
          : {
              status: "unavailable",
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
