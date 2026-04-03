import type {
  PlanningState,
  ConversationMessage,
  DateFlexibility,
  PlanSnapshot,
  PlanMutationInput,
  Poi,
  SessionProviderSettings,
  SessionSnapshot,
  SessionSummary,
  StreamEvent,
  AgentTraceEvent
} from "@roameo/contracts";

export type SessionId = string;
export type ChatMessage = ConversationMessage;
export type POI = Poi;
export type CanonicalPlan = PlanSnapshot;
export type CanonicalSession = SessionSnapshot;
export type CanonicalSessionSummary = SessionSummary;
export type SessionPlanMutation = PlanMutationInput;
export type ProviderSettings = SessionProviderSettings;
export type WsEvent = StreamEvent;
export type SessionPlanningState = PlanningState;
export type { AgentTraceEvent };


export interface Activity {
  id: string;
  name: string;
  start: string;
  end: string;
  location?: string;
  poiId?: string;
  lat?: number;
  lng?: number;
  distanceKm?: number;
  photoUrl?: string;
  rating?: number;
  description?: string;
  notes?: string[];
}

export interface ItineraryDay {
  day: number;
  date: string;
  title?: string;
  activities: Activity[];
  accommodation?: {
    name: string;
    checkIn?: string;
    checkOut?: string;
    nights?: number;
    poiId?: string;
    location?: string;
    photoUrl?: string;
  };
  theme?: string;
  summary?: string;
}

export interface Itinerary {
  origin?: string;
  destination?: string;
  destinations?: string[];
  days: number;
  daysPlan: ItineraryDay[];
  destinationSegments?: Array<{
    destination: string;
    startDay: number;
    endDay: number;
    days: number;
  }>;
}

export interface SearchResults {
  stays: POI[];
  restaurants: POI[];
  attractions: POI[];
}

export interface MapData {
  pois: POI[];
  routes: Array<{
    from: [number, number];
    to: [number, number];
    durationMinutes?: number;
  }>;
}

export interface TripContext {
  id: string;
  title: string;
  origin: string;
  destination: string;
  destinations: string[];
  startDate?: string;
  endDate?: string;
  dateFlexibility?: DateFlexibility;
  days: number;
  travelers: string;
  budget: string;
}

export interface SessionSettingsPayload {
  providerSettings: ProviderSettings;
  preferences: {
    homeAirport?: string;
    currency: string;
    locale: string;
    styles: string[];
    dietaryNotes: string[];
    accessibilityNotes: string[];
  };
  credentials: Array<{
    provider: "gemini" | "openai";
    keySource: "user";
    configured: boolean;
    lastUpdatedAt?: string;
  }>;
}

function resolveBackendUrl() {
  const configured = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();

  if (typeof window !== "undefined") {
    const { hostname } = window.location;
    const isLocalHost =
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "0.0.0.0";

    if (isLocalHost) {
      if (configured && /localhost|127\.0\.0\.1|0\.0\.0\.0/.test(configured)) {
        return configured;
      }

      return "http://localhost:4000";
    }
  }

  return configured || "http://localhost:4000";
}

export const BACKEND_URL = resolveBackendUrl();
