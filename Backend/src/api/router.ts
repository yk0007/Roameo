import { Router, type Request, type Response } from "express";
import { randomUUID } from "crypto";
import type { WsEvent, TripContext } from "../types/schemas.js";
import type { WsHub } from "../ws/emit.js";
import type { Db } from "../db/types.js";
import type { SimpleRateLimiter } from "../utils/rateLimiter.js";
import type { runRouter } from "../graph/graph.js";
import { buildMapsRouter } from "./maps.js";
import { buildCacheRouter } from "./cache.js";
import { optionalAuth, authenticateUser, type AuthenticatedRequest } from "../middleware/auth.js";

export function buildApiRouter(
  hub: WsHub,
  db: Db,
  opts: {
    chatLimiter: SimpleRateLimiter;
    runRouter: typeof runRouter;
    onNewSession?: (sessionId: string, inviteId: string, trip: Partial<TripContext>) => void;
    onDeleteSession?: (sessionId: string) => void;
  }
) {
  const r = Router();

  // Update trip details (emits navbar.update)
  r.post("/trip/update", async (req: Request, res: Response) => {
    const { sessionId, patch } = req.body as { sessionId?: string; patch?: Partial<TripContext> };
    if (!sessionId || !patch) return res.status(400).json({ error: "sessionId and patch required" });
    const event: WsEvent = { type: "navbar.update", data: patch };
    db.patchTrip(sessionId, patch as Record<string, any>);
    hub.emit(sessionId, event);
    res.json({ ok: true });
  });


  // Create a new invite id for session
  r.post("/invite/create", async (req: Request, res: Response) => {
    const { sessionId } = req.body as { sessionId?: string };
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    const inviteId = randomUUID().slice(0, 8);
    db.setInvite(sessionId, inviteId);
    res.json({ inviteId });
  });

  // Save/unsave POI (stub)
  r.post("/poi/save", async (req: Request, res: Response) => {
    const { sessionId, poiId, saved } = req.body as { sessionId?: string; poiId?: string; saved?: boolean };
    if (!sessionId || !poiId) return res.status(400).json({ error: "sessionId and poiId required" });
    db.setPoiSaved(sessionId, poiId, Boolean(saved));
    res.json({ ok: true, saved: Boolean(saved) });
  });

  // Clear chat messages for a session
  r.post("/chat/clear", async (req: Request, res: Response) => {
    const { sessionId } = req.body as { sessionId?: string };
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    db.clearMessages(sessionId);
    hub.emit(sessionId, { type: "chat.append", data: { id: "sys", role: "assistant", content: "Chat cleared.", createdAt: new Date().toISOString() } as any });
    res.json({ ok: true });
  });

  // Delete a trip/session entirely
  r.delete("/sessions/:sessionId", authenticateUser, async (req: AuthenticatedRequest, res: Response) => {
    const { sessionId } = req.params;
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    db.deleteSession(sessionId);
    // Also clear any parallel in-memory cache, if provided
    try {
      opts?.onDeleteSession?.(sessionId);
    } catch {}
    res.json({ ok: true });
  });

  // Expose saved POI IDs for a session so the client can restore Saved tab state
  r.get("/sessions/:sessionId/saved-pois", authenticateUser, async (req: AuthenticatedRequest, res: Response) => {
    const { sessionId } = req.params;
    const s = db.getSession(sessionId);
    if (!s) return res.status(404).json({ error: "Session not found" });
    res.json({ ids: Array.from(s.savedPoiIds || []) });
  });

  // Alternative endpoint for frontend compatibility
  r.get("/session/saved", async (req: Request, res: Response) => {
    const { sessionId } = req.query as { sessionId?: string };
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    const s = db.getSession(sessionId);
    if (!s) return res.status(404).json({ error: "Session not found" });
    res.json({ ids: Array.from(s.savedPoiIds || []) });
  });

  // List trips (sessions) — MVP: return everything in memory DB
  // Health check endpoint
  r.get("/health", (_req: Request, res: Response) => {
    res.status(200).send("OK");
  });

  // Apply optional auth to all routes
  r.use(optionalAuth);

  r.post("/chat/send", async (req: AuthenticatedRequest, res: Response) => {
    const ip = (req.headers["x-forwarded-for"] as string)?.split(",")[0]?.trim() || req.ip || "unknown";
    if (!opts.chatLimiter.allow(`chat:${ip}`)) return res.status(429).json({ error: "Rate limit exceeded" });

    const { sessionId: incoming, inviteId: incomingInvite, message } = req.body || {};
    let sessionId = incoming as string | undefined;
    let inviteId = incomingInvite as string | undefined;

    const isNew = !sessionId;
    if (!sessionId) {
      sessionId = randomUUID();
      inviteId = randomUUID().slice(0, 8);
      opts.onNewSession?.(sessionId, inviteId, { sessionId });
      db.upsertSession(sessionId, { inviteId, trip: { sessionId }, userId: req.userId });
      hub.emit(sessionId, { type: "session.ready", data: { sessionId: sessionId, inviteId: inviteId! } });
    }

    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "message required" });
    }

    const sid = sessionId as string;
    db.appendMessage(sid, { id: randomUUID(), role: "user", content: message, createdAt: new Date().toISOString() });

    try {
      const session = db.getSession(sid);
      let wsEvents: any[] = [];
      if (session) {
        // Fetch chat history for context
        const history = session?.messages || [];
        wsEvents = await opts.runRouter({ sessionId: sid, message, trip: (session?.trip as Partial<TripContext>) || {} }, history);
        
        for (const e of wsEvents) {
          hub.emit(sid, e);
          // Persist assistant messages to database
          if (e.type === "chat.append" && e.data.role === "assistant") {
            db.appendMessage(sid, e.data);
          }
          // Update trip data when navbar update event occurs
          if (e.type === "navbar.update") {
            const tripUpdate = {
              destination: e.data.destination,
              destinations: e.data.destinations,
              days: e.data.days,
              title: e.data.title
            };
            db.patchTrip(sid, tripUpdate);
          }
          // Persist itinerary updates with better error handling
          if (e.type === "itinerary.update") {
            try {
              // Only persist if we have valid itinerary data
              if (e.data && typeof e.data === 'object' && e.data.daysPlan) {
                db.patchTrip(sid, { itinerary: e.data });
                console.log(`[router] Itinerary updated for session ${sid} with ${e.data.daysPlan?.length || 0} days`);
              } else {
                console.warn(`[router] Skipping invalid itinerary update for session ${sid}:`, e.data);
              }
            } catch (error) {
              console.error(`[router] Failed to persist itinerary for session ${sid}:`, error);
            }
          }
          // Persist search results
          if (e.type === "search.results") {
            db.patchTrip(sid, { searchResults: e.data });
          }
          // Persist map data
          if (e.type === "map.update") {
            db.patchTrip(sid, { mapData: e.data });
          }
        }
      }
      return res.json({ sessionId, inviteId, created: isNew, events: wsEvents });
    } catch (err) {
      const errorMessage = { id: randomUUID(), role: "assistant" as const, content: "Sorry, something went wrong.", createdAt: new Date().toISOString() };
      hub.emit(sid, {
        type: "chat.append",
        data: errorMessage,
      });
      // Persist error message to database
      db.appendMessage(sid, errorMessage as any);
      return res.json({ sessionId, inviteId, created: isNew, events: [] });
    }
  });

  r.get("/user/stats", async (req: AuthenticatedRequest, res: Response) => {
    if (!req.userId) {
      return res.status(401).json({ error: "Authentication required" });
    }
    
    try {
      // Count user's trips from database instead of memory
      const { createClient } = await import('@supabase/supabase-js');
      const supabaseUrl = process.env.SUPABASE_URL;
      const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

      if (!supabaseUrl || !supabaseKey) {
        console.error('Supabase environment variables not set');
        return res.status(503).json({ error: 'Database not configured' });
      }

      const supabase = createClient(supabaseUrl, supabaseKey);

      const { count, error } = await supabase
        .from('chat_sessions')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', req.userId);

      if (error) {
        console.error('Error counting user sessions:', error);
        return res.status(500).json({ error: 'Failed to fetch user statistics' });
      }

      res.json({ 
        tripCount: count || 0,
        userId: req.userId 
      });
    } catch (error) {
      console.error('Error fetching user stats:', error);
      res.status(500).json({ error: "Failed to fetch user statistics" });
    }
  });

  r.get("/trips/list", authenticateUser, async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.userId) {
        console.log("No userId in trips/list request");
        return res.json({ trips: [] });
      }
      
      console.log("Fetching trips for userId:", req.userId);

      // Query sessions from database instead of memory
      const { createClient } = await import('@supabase/supabase-js');
      const supabaseUrl = process.env.SUPABASE_URL;
      const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

      if (!supabaseUrl || !supabaseKey) {
        console.error('Supabase environment variables not set');
        return res.status(503).json({ error: 'Database not configured' });
      }

      const supabase = createClient(supabaseUrl, supabaseKey);

      // First check if there are any sessions at all
      const { data: allSessions, error: allError } = await supabase
        .from('chat_sessions')
        .select('session_id, user_id, created_at')
        .limit(10);
      
      console.log("All sessions in database:", allSessions);
      console.log("Looking for user_id:", req.userId);

      const { data: sessions, error } = await supabase
        .from('chat_sessions')
        .select('*')
        .eq('user_id', req.userId)
        .order('updated_at', { ascending: false });

      if (error) {
        console.error('Error fetching user sessions:', error);
        return res.status(500).json({ error: 'Failed to fetch trips' });
      }

      console.log("Found sessions:", sessions?.length || 0);

      const rows = (sessions || []).map((s) => {
        const trip = (s.trip as any) || {};
        const days: number | undefined = trip.days;
        const duration: string | null = typeof days === "number" ? `${days} days` : (trip.duration || null);
        const title: string = trip.title || trip.destination || "Untitled trip";
        
        console.log('Processing session:', s.session_id);
        console.log('Trip data:', JSON.stringify(trip, null, 2));
        console.log('Extracted title:', title);
        console.log('Extracted destination:', trip.destination);
        
        return {
          id: s.session_id,
          sessionId: s.session_id,
          inviteId: s.invite_id,
          title,
          destination: trip.destination || null,
          duration,
          createdAt: s.created_at,
          updatedAt: s.updated_at,
          destinationImageUrl: trip.destinationImageUrl || null,
        };
      });
      
      console.log('Final trips array:', JSON.stringify(rows, null, 2));

      res.json({ trips: rows });
    } catch (error) {
      console.error('Error in trips/list:', error);
      res.status(500).json({ error: 'Failed to fetch trips' });
    }
  });

  // Delete trip endpoint
  r.delete("/trip", authenticateUser, async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { sessionId } = req.query as { sessionId?: string };
      if (!sessionId) {
        return res.status(400).json({ error: 'sessionId required' });
      }

      const { createClient } = await import('@supabase/supabase-js');
      const supabaseUrl = process.env.SUPABASE_URL;
      const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

      if (!supabaseUrl || !supabaseKey) {
        return res.status(503).json({ error: 'Database not configured' });
      }

      const supabase = createClient(supabaseUrl, supabaseKey);

      // Delete the session
      const { error } = await supabase
        .from('chat_sessions')
        .delete()
        .eq('session_id', sessionId)
        .eq('user_id', req.userId);

      if (error) {
        console.error('Error deleting trip:', error);
        return res.status(500).json({ error: 'Failed to delete trip' });
      }

      console.log('Deleted trip session:', sessionId);
      res.json({ ok: true });
    } catch (error) {
      console.error('Error in DELETE /trip:', error);
      res.status(500).json({ error: 'Failed to delete trip' });
    }
  });

  // Test endpoint to add trip data to existing session
  r.post("/test/add-trip-data", authenticateUser, async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { sessionId } = req.body;
      if (!sessionId) {
        return res.status(400).json({ error: 'sessionId required' });
      }

      const { createClient } = await import('@supabase/supabase-js');
      const supabaseUrl = process.env.SUPABASE_URL;
      const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

      if (!supabaseUrl || !supabaseKey) {
        return res.status(503).json({ error: 'Database not configured' });
      }

      const supabase = createClient(supabaseUrl, supabaseKey);

      const testTripData = {
        title: "3-day Ooty Adventure",
        destination: "Ooty",
        days: 3,
        duration: "3 days"
      };

      const { error } = await supabase
        .from('chat_sessions')
        .update({ trip: testTripData })
        .eq('session_id', sessionId)
        .eq('user_id', req.userId);

      if (error) {
        console.error('Error updating trip data:', error);
        return res.status(500).json({ error: 'Failed to update trip data' });
      }

      console.log('Added test trip data to session:', sessionId);
      res.json({ success: true, tripData: testTripData });
    } catch (error) {
      console.error('Error in test/add-trip-data:', error);
      res.status(500).json({ error: 'Failed to add trip data' });
    }
  });

  // Proxy endpoint for Google Photos API to fix CORS issues
  r.get("/proxy/photo", async (req: Request, res: Response) => {
    const { photo_reference, maxwidth = "400", key } = req.query as { 
      photo_reference?: string; 
      maxwidth?: string; 
      key?: string; 
    };
    
    if (!photo_reference || !key) {
      return res.status(400).json({ error: "photo_reference and key required" });
    }

    try {
      const photoUrl = `https://maps.googleapis.com/maps/api/place/photo?photo_reference=${photo_reference}&maxwidth=${maxwidth}&key=${key}`;
      
      // Add timeout and abort controller for fetch
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch(photoUrl, {
        signal: controller.signal,
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; RoameoBot/1.0)'
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        return res.status(response.status).json({ error: "Failed to fetch photo" });
      }

      // Set appropriate CORS headers
      res.set({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': response.headers.get('content-type') || 'image/jpeg',
        'Cache-Control': 'public, max-age=86400' // Cache for 24 hours
      });

      // Stream the image data
      if (response.body) {
        const reader = response.body.getReader();
        const pump = async () => {
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              res.write(value);
            }
            res.end();
          } catch (error) {
            console.error('[proxy] Error streaming photo:', error);
            res.status(500).end();
          }
        };
        await pump();
      } else {
        res.status(500).json({ error: "No response body" });
      }
    } catch (error: any) {
      console.error('[proxy] Error fetching photo:', error);
      
      // Handle timeout errors specifically
      if (error?.name === 'AbortError') {
        return res.status(408).json({ error: "Photo fetch timeout" });
      }
      
      // Handle network errors
      if (error?.code === 'ETIMEDOUT' || error?.code === 'ECONNREFUSED') {
        return res.status(503).json({ error: "Photo service unavailable" });
      }
      
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Google Maps API key endpoint
  r.get("/maps/api-key", async (req: Request, res: Response) => {
    const apiKey = process.env.GOOGLE_MAPS_API_KEY
    
    if (!apiKey) {
      return res.status(500).json({ error: 'Google Maps API key not configured' })
    }
    
    // Validate API key format (Google Maps API keys are 39 characters)
    if (apiKey.length !== 39 || !apiKey.startsWith('AIza')) {
      console.error('Invalid Google Maps API key format:', apiKey.substring(0, 10) + '...')
      return res.status(500).json({ error: 'Invalid Google Maps API key format' })
    }
    
    return res.json({ apiKey })
  })

  // Mount maps proxy router
  r.use("/maps", buildMapsRouter());

  // Mount cache management router
  r.use("/cache", buildCacheRouter(db));

  return r;
}
