import { z } from "zod";

export const providerSchema = z.enum(["gemini", "openai"]);
export type Provider = z.infer<typeof providerSchema>;

export const runModeSchema = z.enum(["fast", "balanced", "deep"]);
export type RunMode = z.infer<typeof runModeSchema>;

export const keySourceSchema = z.enum(["platform", "user"]);
export type KeySource = z.infer<typeof keySourceSchema>;

export const travelStyleSchema = z.enum([
  "relaxed",
  "balanced",
  "packed",
  "luxury",
  "budget",
  "family",
  "romantic",
  "adventure",
  "culture"
]);
export type TravelStyle = z.infer<typeof travelStyleSchema>;

export const sessionProviderSettingsSchema = z.object({
  provider: providerSchema.default("gemini"),
  runMode: runModeSchema.default("balanced"),
  keySource: keySourceSchema.default("platform")
});
export type SessionProviderSettings = z.infer<
  typeof sessionProviderSettingsSchema
>;

export const userPreferenceSchema = z.object({
  homeAirport: z.string().optional(),
  currency: z.string().default("INR"),
  locale: z.string().default("en-IN"),
  styles: z.array(travelStyleSchema).default([]),
  dietaryNotes: z.array(z.string()).default([]),
  accessibilityNotes: z.array(z.string()).default([])
});
export type UserPreference = z.infer<typeof userPreferenceSchema>;

export const planningStateStatusSchema = z.enum([
  "ready",
  "running",
  "unavailable"
]);
export type PlanningStateStatus = z.infer<typeof planningStateStatusSchema>;

export const planningStateStageSchema = z.enum([
  "understanding",
  "researching",
  "building_plan",
  "refining",
  "ready",
  "unavailable"
]);
export type PlanningStateStage = z.infer<typeof planningStateStageSchema>;

export const planningStateSourceSchema = z.enum([
  "provider",
  "places",
  "directions"
]);
export type PlanningStateSource = z.infer<typeof planningStateSourceSchema>;

export const planningStateSchema = z.object({
  status: planningStateStatusSchema.default("ready"),
  stage: planningStateStageSchema.default("ready"),
  source: planningStateSourceSchema.optional(),
  reason: z.string().optional(),
  retryable: z.boolean().default(true),
  updatedAt: z.string().optional()
});
export type PlanningState = z.infer<typeof planningStateSchema>;

export const sessionMemorySchema = z.object({
  summary: z.string().default(""),
  destinationsDiscussed: z.array(z.string()).default([]),
  acceptedDecisions: z.array(z.string()).default([]),
  lastPlanVersion: z.number().default(0),
  planningState: planningStateSchema.default({
    status: "ready",
    stage: "ready",
    retryable: true
  }),
  preferences: userPreferenceSchema.default({
    currency: "INR",
    locale: "en-IN",
    styles: [],
    dietaryNotes: [],
    accessibilityNotes: []
  })
});
export type SessionMemory = z.infer<typeof sessionMemorySchema>;

export const budgetBreakdownSchema = z.object({
  accommodation: z.number().nonnegative().default(0),
  food: z.number().nonnegative().default(0),
  transport: z.number().nonnegative().default(0),
  activities: z.number().nonnegative().default(0),
  misc: z.number().nonnegative().default(0),
  total: z.number().nonnegative().default(0),
  currency: z.string().default("INR")
});
export type BudgetBreakdown = z.infer<typeof budgetBreakdownSchema>;

export const budgetTargetSchema = z.object({
  total: z.number().nonnegative(),
  currency: z.string().default("INR")
});
export type BudgetTarget = z.infer<typeof budgetTargetSchema>;

export const poiTypeSchema = z.enum([
  "destination",
  "stay",
  "restaurant",
  "attraction",
  "transit"
]);
export type PoiType = z.infer<typeof poiTypeSchema>;

export const poiSourceSchema = z.enum([
  "google_places",
  "google_maps",
  "web_research",
  "manual"
]);
export type PoiSource = z.infer<typeof poiSourceSchema>;

export const poiSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: poiTypeSchema,
  lat: z.number(),
  lng: z.number(),
  address: z.string().optional(),
  description: z.string().optional(),
  photoUrl: z.string().optional(),
  website: z.string().optional(),
  phone: z.string().optional(),
  openingHours: z.array(z.string()).default([]),
  rating: z.number().optional(),
  priceLevel: z.number().int().min(0).max(4).optional(),
  source: poiSourceSchema,
  sourceId: z.string().optional(),
  tags: z.array(z.string()).default([])
});
export type Poi = z.infer<typeof poiSchema>;

export const poiCatalogSchema = z.object({
  version: z.number().int().default(1),
  items: z.record(z.string(), poiSchema).default({})
});
export type PoiCatalog = z.infer<typeof poiCatalogSchema>;

export const itineraryActivitySchema = z.object({
  id: z.string(),
  poiId: z.string().optional(),
  title: z.string(),
  summary: z.string().optional(),
  startTime: z.string(),
  endTime: z.string(),
  travelTimeMinutesFromPrevious: z.number().int().nonnegative().optional(),
  notes: z.array(z.string()).default([])
});
export type ItineraryActivity = z.infer<typeof itineraryActivitySchema>;

export const itineraryDaySchema = z.object({
  day: z.number().int().positive(),
  date: z.string(),
  title: z.string(),
  theme: z.string().optional(),
  summary: z.string().optional(),
  destination: z.string(),
  accommodationPoiId: z.string().optional(),
  activities: z.array(itineraryActivitySchema).default([]),
  budget: budgetBreakdownSchema.optional()
});
export type ItineraryDay = z.infer<typeof itineraryDaySchema>;

export const destinationSegmentSchema = z.object({
  destination: z.string(),
  startDay: z.number().int().positive(),
  endDay: z.number().int().positive(),
  nights: z.number().int().nonnegative()
});
export type DestinationSegment = z.infer<typeof destinationSegmentSchema>;

export const travelIntentSchema = z.enum([
  "plan_trip",
  "refine_trip",
  "search_places",
  "question",
  "settings",
  "meta"
]);
export type TravelIntent = z.infer<typeof travelIntentSchema>;

export const planSnapshotSchema = z.object({
  schemaVersion: z.literal(1),
  sessionId: z.string(),
  version: z.number().int().nonnegative(),
  title: z.string(),
  origin: z.string().optional(),
  destination: z.string().optional(),
  destinations: z.array(z.string()).default([]),
  totalDays: z.number().int().positive().default(1),
  travelerCount: z.number().int().positive().default(1),
  budgetTarget: budgetTargetSchema.optional(),
  budget: budgetBreakdownSchema.optional(),
  notes: z.array(z.string()).default([]),
  destinationSegments: z.array(destinationSegmentSchema).default([]),
  days: z.array(itineraryDaySchema).default([]),
  generatedAt: z.string(),
  lastUserIntent: travelIntentSchema.default("question")
});
export type PlanSnapshot = z.infer<typeof planSnapshotSchema>;

export const chatMessageRoleSchema = z.enum([
  "user",
  "assistant",
  "system",
  "tool"
]);
export type ChatMessageRole = z.infer<typeof chatMessageRoleSchema>;

export const messagePhaseSchema = z.enum([
  "thinking",
  "tooling",
  "draft",
  "final"
]);
export type MessagePhase = z.infer<typeof messagePhaseSchema>;

export const assistantResponseBlockSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("trip_intro"),
    eyebrow: z.string().optional(),
    title: z.string(),
    body: z.string(),
    moodEmoji: z.string().optional()
  }),
  z.object({
    type: z.literal("lead"),
    text: z.string()
  }),
  z.object({
    type: z.literal("itinerary_template"),
    title: z.string(),
    subtitle: z.string().optional(),
    budgetLabel: z.string().optional(),
    days: z.array(
      z.object({
        day: z.number().int().positive(),
        date: z.string().optional(),
        title: z.string(),
        summary: z.string().optional(),
        destination: z.string(),
        accent: z.string().optional(),
        periods: z.array(
          z.object({
            key: z.enum(["morning", "afternoon", "evening", "flex"]),
            label: z.string(),
            emoji: z.string().optional(),
            entries: z.array(
              z.object({
                title: z.string(),
                poiId: z.string().optional(),
                timeLabel: z.string().optional(),
                description: z.string().optional()
              })
            ).default([])
          })
        ).default([]),
        stayPoiId: z.string().optional(),
        footer: z.string().optional()
      })
    ).min(1)
  }),
  z.object({
    type: z.literal("clarifying_questions"),
    title: z.string().optional(),
    questions: z.array(z.string()).min(1)
  }),
  z.object({
    type: z.literal("place_card_row"),
    title: z.string().optional(),
    poiIds: z.array(z.string()).default([]),
    display: z.enum(["inline", "carousel"]).default("inline")
  }),
  z.object({
    type: z.literal("recommendation_cards"),
    title: z.string().optional(),
    poiIds: z.array(z.string()).default([])
  }),
  z.object({
    type: z.literal("itinerary_summary"),
    title: z.string().optional(),
    days: z.array(
      z.object({
        day: z.number().int().positive(),
        title: z.string(),
        summary: z.string().optional()
      })
    )
  }),
  z.object({
    type: z.literal("assistant_prompt_chips"),
    title: z.string().optional(),
    prompts: z.array(
      z.object({
        label: z.string(),
        prompt: z.string()
      })
    ).min(1)
  }),
  z.object({
    type: z.literal("quick_actions"),
    title: z.string().optional(),
    actions: z.array(
      z.object({
        label: z.string(),
        prompt: z.string()
      })
    )
  }),
  z.object({
    type: z.literal("planning_status"),
    state: planningStateStatusSchema,
    stage: planningStateStageSchema,
    label: z.string(),
    detail: z.string().optional()
  })
]);
export type AssistantResponseBlock = z.infer<typeof assistantResponseBlockSchema>;

export const conversationMessageMetaSchema = z
  .object({
    provider: providerSchema.optional(),
    turnId: z.string().optional(),
    responseBlocks: z.array(assistantResponseBlockSchema).optional()
  })
  .catchall(z.any());
export type ConversationMessageMeta = z.infer<typeof conversationMessageMetaSchema>;

export const conversationMessageSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  role: chatMessageRoleSchema,
  content: z.string(),
  createdAt: z.string(),
  phase: messagePhaseSchema.optional(),
  meta: conversationMessageMetaSchema.default({})
});
export type ConversationMessage = z.infer<typeof conversationMessageSchema>;

export const agentTraceStatusSchema = z.enum([
  "queued",
  "running",
  "completed",
  "failed"
]);
export type AgentTraceStatus = z.infer<typeof agentTraceStatusSchema>;

export const agentTraceEventSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  turnId: z.string(),
  agent: z.string(),
  status: agentTraceStatusSchema,
  label: z.string(),
  detail: z.string().optional(),
  createdAt: z.string()
});
export type AgentTraceEvent = z.infer<typeof agentTraceEventSchema>;

export const sessionSnapshotSchema = z.object({
  id: z.string(),
  userId: z.string().optional(),
  title: z.string().default("Untitled trip"),
  providerSettings: sessionProviderSettingsSchema,
  memory: sessionMemorySchema,
  plan: planSnapshotSchema.optional(),
  poiCatalog: poiCatalogSchema.default({ version: 1, items: {} }),
  messages: z.array(conversationMessageSchema).default([]),
  savedPoiIds: z.array(z.string()).default([]),
  traces: z.array(agentTraceEventSchema).default([]),
  createdAt: z.string(),
  updatedAt: z.string()
});
export type SessionSnapshot = z.infer<typeof sessionSnapshotSchema>;

export const sessionSummarySchema = z.object({
  id: z.string(),
  title: z.string(),
  destination: z.string().optional(),
  totalDays: z.number().int().positive().optional(),
  providerSettings: sessionProviderSettingsSchema,
  updatedAt: z.string(),
  createdAt: z.string()
});
export type SessionSummary = z.infer<typeof sessionSummarySchema>;

export const providerCredentialSchema = z.object({
  provider: providerSchema,
  keySource: keySourceSchema,
  configured: z.boolean(),
  lastUpdatedAt: z.string().optional()
});
export type ProviderCredential = z.infer<typeof providerCredentialSchema>;

export const toolCallResultSchema = z.object({
  toolName: z.string(),
  success: z.boolean(),
  payload: z.record(z.string(), z.any()).default({}),
  error: z.string().optional()
});
export type ToolCallResult = z.infer<typeof toolCallResultSchema>;

export const streamEventSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("session.snapshot"),
    data: sessionSnapshotSchema
  }),
  z.object({
    type: z.literal("turn.started"),
    data: z.object({
      sessionId: z.string(),
      turnId: z.string(),
      providerSettings: sessionProviderSettingsSchema
    })
  }),
  z.object({
    type: z.literal("message.delta"),
    data: z.object({
      sessionId: z.string(),
      turnId: z.string(),
      messageId: z.string(),
      role: chatMessageRoleSchema,
      delta: z.string(),
      done: z.boolean().default(false)
    })
  }),
  z.object({
    type: z.literal("message.committed"),
    data: conversationMessageSchema
  }),
  z.object({
    type: z.literal("trace.updated"),
    data: agentTraceEventSchema
  }),
  z.object({
    type: z.literal("plan.updated"),
    data: z.object({
      sessionId: z.string(),
      plan: planSnapshotSchema,
      poiCatalog: poiCatalogSchema
    })
  }),
  z.object({
    type: z.literal("turn.completed"),
    data: z.object({
      sessionId: z.string(),
      turnId: z.string()
    })
  }),
  z.object({
    type: z.literal("turn.failed"),
    data: z.object({
      sessionId: z.string(),
      turnId: z.string(),
      error: z.string()
    })
  })
]);
export type StreamEvent = z.infer<typeof streamEventSchema>;

export const sendMessageInputSchema = z.object({
  content: z.string().min(1),
  providerSettings: sessionProviderSettingsSchema.optional()
});
export type SendMessageInput = z.infer<typeof sendMessageInputSchema>;

export const planMutationInputSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("add_poi"),
    poiId: z.string().min(1),
    day: z.number().int().positive().optional()
  }),
  z.object({
    type: z.literal("remove_poi"),
    poiId: z.string().min(1)
  }),
  z.object({
    type: z.literal("move_activity"),
    activityId: z.string().min(1),
    toDay: z.number().int().positive(),
    position: z.number().int().nonnegative().optional()
  }),
  z.object({
    type: z.literal("regenerate_day"),
    day: z.number().int().positive(),
    focusPoiId: z.string().min(1).optional()
  }),
  z.object({
    type: z.literal("rebalance_trip"),
    focusPoiId: z.string().min(1).optional()
  }),
  z.object({
    type: z.literal("update_overview"),
    title: z.string().optional(),
    origin: z.string().optional(),
    destination: z.string().optional(),
    totalDays: z.number().int().positive().optional(),
    travelerCount: z.number().int().positive().optional(),
    budgetTotal: z.number().nonnegative().optional(),
    currency: z.string().optional()
  })
]);
export type PlanMutationInput = z.infer<typeof planMutationInputSchema>;

export const sessionMutationSchema = z.object({
  title: z.string().optional(),
  providerSettings: sessionProviderSettingsSchema.optional(),
  preferences: userPreferenceSchema.optional(),
  memory: sessionMemorySchema.optional()
});
export type SessionMutation = z.infer<typeof sessionMutationSchema>;

export const createSessionInputSchema = z.object({
  initialMessage: z.string().optional(),
  providerSettings: sessionProviderSettingsSchema.optional(),
  title: z.string().optional()
});
export type CreateSessionInput = z.infer<typeof createSessionInputSchema>;

export const userSettingsUpdateSchema = z.object({
  providerSettings: sessionProviderSettingsSchema,
  preferences: userPreferenceSchema
});
export type UserSettingsUpdate = z.infer<typeof userSettingsUpdateSchema>;

export const providerCredentialInputSchema = z.object({
  apiKey: z.string().min(1)
});
export type ProviderCredentialInput = z.infer<
  typeof providerCredentialInputSchema
>;
