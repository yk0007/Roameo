// Shared schemas for Roameo backend (mirror in frontend as lib/types.ts)
import { z } from "zod";

export const TripContextSchema = z.object({
  sessionId: z.string(),
  inviteId: z.string().optional(),
  title: z.string().optional(),
  origin: z.string().optional(),
  destination: z.string().optional(),
  destinations: z.array(z.string()).optional(), // Support multiple destinations
  days: z.number().optional(),
  travelers: z.number().optional(),
  budget: z.string().optional(),
  destinationImageUrl: z.string().optional(), // Add destination image URL
});
export type TripContext = z.infer<typeof TripContextSchema>;

export const ChatMessageSchema = z.object({
  id: z.string(),
  role: z.union([z.literal("user"), z.literal("assistant"), z.literal("system"), z.literal("tool")]),
  content: z.string(),
  createdAt: z.string(),
});
export type ChatMessage = z.infer<typeof ChatMessageSchema>;

export const POISchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.union([z.literal("stay"), z.literal("restaurant"), z.literal("attraction")]),
  lat: z.number(),
  lng: z.number(),
  rating: z.number().optional(),
  price: z.string().optional(),
  address: z.string().optional(),
  photoUrl: z.string().optional(),
  source: z.union([z.literal("google"), z.literal("foursquare"), z.literal("custom"), z.literal("mock")]).optional(),
});
export type POI = z.infer<typeof POISchema>;

export const ActivitySchema = z.object({
  name: z.string(),
  start: z.string(),
  end: z.string(),
  location: z.string().optional(),
  poiId: z.string().optional(),
  lat: z.number().optional(),
  lng: z.number().optional(),
  distanceKm: z.number().optional(),
  photoUrl: z.string().optional(),
  rating: z.number().optional(),
  description: z.string().optional(),
});
export type Activity = z.infer<typeof ActivitySchema>;

export const ItineraryDaySchema = z.object({
  day: z.number(),
  date: z.string(),
  title: z.string().optional(),
  activities: z.array(ActivitySchema),
  accommodation: z
    .object({ name: z.string(), checkIn: z.string().optional(), checkOut: z.string().optional(), nights: z.number().optional(), poiId: z.string().optional() })
    .optional(),
});

export const ItinerarySchema = z.object({
  origin: z.string(),
  destination: z.string(),
  destinations: z.array(z.string()).optional(), // Support multiple destinations
  days: z.number(),
  daysPlan: z.array(ItineraryDaySchema),
  destinationSegments: z.array(z.object({
    destination: z.string(),
    startDay: z.number(),
    endDay: z.number(),
    days: z.number()
  })).optional(), // Track which days belong to which destination
});
export type Itinerary = z.infer<typeof ItinerarySchema>;

export const SearchResultsSchema = z.object({
  stays: z.array(POISchema),
  restaurants: z.array(POISchema),
  attractions: z.array(POISchema),
});
export type SearchResults = z.infer<typeof SearchResultsSchema>;

export const MapUpdateSchema = z.object({
  pois: z.array(POISchema),
  routes: z.array(
    z.object({ from: z.tuple([z.number(), z.number()]), to: z.tuple([z.number(), z.number()]), polyline: z.string().optional() })
  ),
});

export const WsEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("chat.append"), data: ChatMessageSchema }),
  z.object({ type: z.literal("chat.history"), data: z.array(ChatMessageSchema) }),
  z.object({ type: z.literal("navbar.update"), data: TripContextSchema.partial() }),
  z.object({ type: z.literal("itinerary.update"), data: ItinerarySchema }),
  z.object({ type: z.literal("search.results"), data: SearchResultsSchema }),
  z.object({ type: z.literal("map.update"), data: MapUpdateSchema }),
  z.object({ type: z.literal("session.ready"), data: z.object({ sessionId: z.string(), inviteId: z.string() }) }),
  z.object({ type: z.literal("intent.detected"), data: z.object({ intent: z.enum(["PLAN_TRIP", "DESTINATION_SEARCH", "CHAT"]), message: z.string() }) }),
  z.object({ type: z.literal("planning.status"), data: z.object({ status: z.string() }) }),
  z.object({ type: z.literal("search.status"), data: z.object({ status: z.string() }) }),
  z.object({ type: z.literal("map.status"), data: z.object({ status: z.string() }) }),
]);
export type WsEvent = z.infer<typeof WsEventSchema>;
