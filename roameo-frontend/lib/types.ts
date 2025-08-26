// Shared types mirrored from Backend/src/types/schemas.ts (keep in sync)
export type SessionId = string
export type InviteId = string

export interface TripContext {
  sessionId: SessionId
  inviteId?: InviteId
  title?: string
  origin?: string
  destination?: string
  destinations?: string[] // Support multiple destinations
  days?: number
  travelers?: number
  budget?: string
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant" | "system" | "tool"
  content: string
  createdAt: string // ISO
}

export interface POI {
  id: string
  name: string
  type: "stay" | "restaurant" | "attraction" | "Activity" | "Hotel"
  lat?: number
  lng?: number
  rating?: number
  price?: string
  address?: string
  photoUrl?: string
  source?: "google" | "foursquare" | "custom"
  description?: string
  openingHours?: string[]
  phone?: string
  website?: string
  priceLevel?: number
}

export interface Activity {
  id?: string;
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
}

export interface Itinerary {
  origin: string
  destination: string
  days: number
  daysPlan: ItineraryDay[]
}

export interface SearchResults {
  stays: POI[]
  restaurants: POI[]
  attractions: POI[]
}

export type WsEvent =
  | { type: "chat.append"; data: ChatMessage }
  | { type: "navbar.update"; data: Partial<TripContext> }
  | { type: "itinerary.update"; data: Itinerary }
  | { type: "search.results"; data: SearchResults }
  | { type: "map.update"; data: { pois: POI[]; routes: Array<{ from: [number, number]; to: [number, number]; polyline?: string }> } }
  | { type: "session.ready"; data: { sessionId: SessionId; inviteId: InviteId } }
  | { type: "chat.history"; data: ChatMessage[] }

export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "https://roameo.onrender.com"
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "wss://roameo.onrender.com/ws"
