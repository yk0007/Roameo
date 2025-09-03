import "dotenv/config";
import express, { type Request, type Response } from "express";
import cors from "cors";
import { createServer, type IncomingMessage } from "http";
import { WebSocketServer } from "ws";
import type WebSocket from "ws";
import { randomUUID } from "crypto";
import { WsHub } from "./ws/emit.js";
import type { WsEvent, TripContext } from "./types/schemas.js";
import { buildApiRouter } from "./api/router.js";
import { runRouter } from "./graph/graph.js";
import type { Db } from "./db/types.js";
import { CachedDb } from "./cache/cached-db.js";
import { SimpleRateLimiter } from "./utils/rateLimiter.js";

const app = express();
app.use(cors());
app.use(express.json());

const httpServer = createServer(app);
const wss = new WebSocketServer({ server: httpServer, path: "/ws" });
const hub = new WsHub();
// Prefer cached DB with Memcached for better performance; falls back gracefully if cache unavailable
const db: Db = new CachedDb(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const chatLimiter = new SimpleRateLimiter(60, 60_000); // 60 req/min per IP

// Mount REST API router
app.use("/api", buildApiRouter(hub, db, {
  chatLimiter,
  runRouter,
  onNewSession: (sid, inviteId, trip) => {
    sessions.set(sid, { inviteId, trip });
  },
  onDeleteSession: (sid: string) => {
    sessions.delete(sid);
  },
}));

wss.on("connection", (ws: WebSocket, req: IncomingMessage) => {
  // Expect: ws://host/ws?sessionId=...
  const url = new URL(req.url || "", `http://${req.headers.host}`);
  const sessionId = url.searchParams.get("sessionId");
  if (!sessionId) {
    ws.close(1008, "sessionId required");
    return;
  }

  // Do NOT implicitly create sessions on WS connect. If the session does not exist,
  // close the socket. New sessions must be created via REST /api/chat/send.
  const existing = db.getSession(sessionId);
  if (!existing) {
    ws.close(1008, "unknown sessionId");
    return;
  }

  hub.attach(sessionId, ws);

  // Keep inviteId from either DB or parallel in-memory map (if present)
  const inviteId = existing.inviteId || sessions.get(sessionId)?.inviteId || undefined;
  if (inviteId && !existing.inviteId) db.setInvite(sessionId, inviteId);

  // Send session ready to all clients in the session
  hub.emit(sessionId, { type: "session.ready", data: { sessionId, inviteId } as any });
  
  // Send current trip navbar data
  if (existing.trip) {
    console.log(`[ws] Restoring trip data for session ${sessionId}:`, JSON.stringify(existing.trip, null, 2));
    hub.emit(sessionId, { type: "navbar.update", data: existing.trip as any });
  }
  
  // If we have a persisted itinerary from prior runs, replay it so UI restores
  const maybeItin = (existing.trip as any)?.itinerary;
  if (maybeItin) {
    console.log(`[ws] Restoring itinerary for session ${sessionId} with ${maybeItin.daysPlan?.length || 0} days`);
    hub.emit(sessionId, { type: "itinerary.update", data: maybeItin });
  } else {
    console.log(`[ws] No itinerary found for session ${sessionId}`);
  }
  // Replay last search results and map snapshot if present
  const maybeSearch = (existing.trip as any)?.searchResults;
  if (maybeSearch) {
    console.log(`[ws] Restoring search results for session ${sessionId}:`, {
      stays: maybeSearch.stays?.length || 0,
      restaurants: maybeSearch.restaurants?.length || 0,
      attractions: maybeSearch.attractions?.length || 0
    });
    hub.emit(sessionId, { type: "search.results", data: maybeSearch });
  } else {
    console.log(`[ws] No search results found for session ${sessionId}`);
    
    // If we have an itinerary but no search results, and we have a destination, 
    // trigger a search to populate POI data for the map hover functionality
    const destination = (existing.trip as any)?.destination;
    if (maybeItin && destination) {
      console.log(`[ws] Triggering POI search for destination ${destination} to restore map functionality`);
      
      // Import and use POI agent to regenerate search results
      import('./agents/poi.js').then(async ({ poiAgent }) => {
        try {
          const poiResult = await poiAgent({ destination });
          if (poiResult && poiResult.type === 'search.results') {
            console.log(`[ws] Generated search results for session ${sessionId}:`, {
              stays: poiResult.data.stays?.length || 0,
              restaurants: poiResult.data.restaurants?.length || 0,
              attractions: poiResult.data.attractions?.length || 0
            });
            
            // Emit the search results and persist them
            hub.emit(sessionId, poiResult);
            db.patchTrip(sessionId, { searchResults: poiResult.data });
            
            // Also generate map data if we don't have it
            const currentMapData = (existing.trip as any)?.mapData;
            if (!currentMapData) {
              // Extract POI IDs from itinerary
              const itineraryPoiIds = new Set<string>();
              maybeItin.daysPlan.forEach((day: any) => {
                day.activities?.forEach((activity: any) => {
                  if (activity.poiId) itineraryPoiIds.add(activity.poiId);
                });
                if (day.accommodation?.poiId) itineraryPoiIds.add(day.accommodation.poiId);
              });
              
              // Find matching POIs from the newly generated search results
              const allSearchPois = [
                ...poiResult.data.stays,
                ...poiResult.data.restaurants,
                ...poiResult.data.attractions
              ];
              
              const itineraryPois = allSearchPois.filter((poi: any) => itineraryPoiIds.has(poi.id));
              
              if (itineraryPois.length > 0) {
                console.log(`[ws] Generated map data with ${itineraryPois.length} POIs from regenerated search results`);
                const mapData = { pois: itineraryPois, routes: [] };
                hub.emit(sessionId, { type: "map.update", data: mapData });
                db.patchTrip(sessionId, { mapData });
              }
            }
          }
        } catch (error) {
          console.error(`[ws] Failed to regenerate POI search for session ${sessionId}:`, error);
        }
      }).catch(error => {
        console.error(`[ws] Failed to import POI agent for session ${sessionId}:`, error);
      });
    }
  }
  
  const maybeMap = (existing.trip as any)?.mapData;
  if (maybeMap) {
    console.log(`[ws] Restoring map data for session ${sessionId}:`, {
      pois: maybeMap.pois?.length || 0,
      routes: maybeMap.routes?.length || 0
    });
    hub.emit(sessionId, { type: "map.update", data: maybeMap });
  } else {
    console.log(`[ws] No map data found for session ${sessionId}`);
    
    // Fallback: If we have an itinerary but no map data, try to generate map data from itinerary POIs
    if (maybeItin && maybeItin.daysPlan && maybeSearch) {
      console.log(`[ws] Attempting to generate map data from itinerary and search results for session ${sessionId}`);
      
      // Extract POI IDs from itinerary
      const itineraryPoiIds = new Set<string>();
      maybeItin.daysPlan.forEach((day: any) => {
        day.activities?.forEach((activity: any) => {
          if (activity.poiId) itineraryPoiIds.add(activity.poiId);
        });
        if (day.accommodation?.poiId) itineraryPoiIds.add(day.accommodation.poiId);
      });
      
      // Find matching POIs from search results
      const allSearchPois = [
        ...(maybeSearch.stays || []),
        ...(maybeSearch.restaurants || []),
        ...(maybeSearch.attractions || [])
      ];
      
      const itineraryPois = allSearchPois.filter((poi: any) => itineraryPoiIds.has(poi.id));
      
      if (itineraryPois.length > 0) {
        console.log(`[ws] Generated map data with ${itineraryPois.length} POIs from itinerary for session ${sessionId}`);
        const mapData = { pois: itineraryPois, routes: [] };
        hub.emit(sessionId, { type: "map.update", data: mapData });
        
        // Persist the generated map data for future sessions
        db.patchTrip(sessionId, { mapData });
      }
    }
  }
  // Replay prior messages to rebuild chat UI on fresh connects
  if (existing.messages?.length) {
    hub.emit(sessionId, { type: "chat.history", data: existing.messages as any });
  }
});

// In-memory session store (MVP) — replace with Supabase later (db abstracts storage)
const sessions = new Map<string, { inviteId: string; trip: Partial<TripContext> }>();


const port = process.env.PORT || 4000;

async function init() {
  // Best-effort: hydrate from Supabase at startup so existing trips/messages are available immediately
  try {
    // @ts-ignore - enhanced method for cached DB
    if (typeof (db as any).hydrateFromRemote === "function") {
      console.log("[roameo-backend] hydrating from Supabase with cache warming...");
      await (db as any).hydrateFromRemote();
      console.log("[roameo-backend] hydration and cache warming complete");
    }
  } catch (e) {
    console.warn("[roameo-backend] hydrateFromRemote failed (continuing):", e);
  }

  // Perform cache health check
  try {
    // @ts-ignore - cache health check method
    if (typeof (db as any).cacheHealthCheck === "function") {
      const healthCheck = await (db as any).cacheHealthCheck();
      console.log(`[roameo-backend] Cache health: ${healthCheck.healthy ? 'OK' : 'WARN'} - ${healthCheck.message}`);
    }
  } catch (e) {
    console.warn("[roameo-backend] Cache health check failed:", e);
  }

  httpServer.listen(port, () => {
    console.log(`[roameo-backend] listening on http://localhost:${port}`);
  });
}

init();
