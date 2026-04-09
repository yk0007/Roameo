export {
  budgetBreakdownSchema,
  chatMessageRoleSchema,
  conversationMessageSchema,
  createSessionInputSchema,
  destinationSegmentSchema,
  itineraryActivitySchema,
  itineraryDaySchema,
  keySourceSchema,
  messagePhaseSchema,
  planSnapshotSchema,
  poiCatalogSchema,
  poiSchema,
  providerCredentialSchema,
  providerSchema,
  runModeSchema,
  sendMessageInputSchema,
  sessionMemorySchema,
  sessionMutationSchema,
  sessionProviderSettingsSchema,
  sessionSnapshotSchema,
  sessionSummarySchema,
  streamEventSchema,
  toolCallResultSchema,
  travelIntentSchema,
  userPreferenceSchema
} from "@roameo/contracts";

export type {
  BudgetBreakdown,
  ChatMessageRole,
  ConversationMessage as ChatMessage,
  CreateSessionInput,
  DestinationSegment,
  ItineraryActivity as Activity,
  ItineraryDay,
  KeySource,
  MessagePhase,
  PlanSnapshot as Itinerary,
  Poi as POI,
  PoiCatalog,
  PoiType,
  Provider,
  ProviderCredential,
  RunMode,
  SendMessageInput,
  SessionMemory,
  SessionMutation,
  SessionProviderSettings,
  SessionSnapshot,
  SessionSummary,
  StreamEvent as WsEvent,
  ToolCallResult,
  TravelIntent,
  UserPreference
} from "@roameo/contracts";

export type TripContext = {
  sessionId: string;
  inviteId?: string;
  title?: string;
  origin?: string;
  destination?: string;
  destinations?: string[];
  days?: number;
  travelers?: number;
  budget?: string;
  destinationImageUrl?: string;
};
