import { Router, type Request, type Response } from "express";
import { randomUUID } from "crypto";
import type { WsEvent, TripContext } from "../types/schemas.js";
import type { WsHub } from "../ws/emit.js";
import type { Db } from "../db/types.js";
import type { SimpleRateLimiter } from "../utils/rateLimiter.js";
import type { runRouter } from "../graph/graph.js";
import { buildMapsRouter } from "./maps.js";
import { optionalAuth, type AuthenticatedRequest } from "../middleware/auth.js";

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
  r.post("/trip/update", (req: Request, res: Response) => {
    const { sessionId, patch } = req.body as { sessionId?: string; patch?: Partial<TripContext> };
    if (!sessionId || !patch) return res.status(400).json({ error: "sessionId and patch required" });
    const event: WsEvent = { type: "navbar.update", data: patch };
    db.patchTrip(sessionId, patch as Record<string, any>);
    hub.emit(sessionId, event);
    res.json({ ok: true });
  });

  // Create a new invite id for session
  r.post("/invite/create", (req: Request, res: Response) => {
    const { sessionId } = req.body as { sessionId?: string };
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    const inviteId = randomUUID().slice(0, 8);
    db.setInvite(sessionId, inviteId);
    res.json({ inviteId });
  });

  // Save/unsave POI (stub)
  r.post("/poi/save", (req: Request, res: Response) => {
    const { sessionId, poiId, saved } = req.body as { sessionId?: string; poiId?: string; saved?: boolean };
    if (!sessionId || !poiId) return res.status(400).json({ error: "sessionId and poiId required" });
    db.setPoiSaved(sessionId, poiId, Boolean(saved));
    res.json({ ok: true, saved: Boolean(saved) });
  });

  // Clear chat messages for a session
  r.post("/chat/clear", (req: Request, res: Response) => {
    const { sessionId } = req.body as { sessionId?: string };
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    db.clearMessages(sessionId);
    hub.emit(sessionId, { type: "chat.append", data: { id: "sys", role: "assistant", content: "Chat cleared.", createdAt: new Date().toISOString() } as any });
    res.json({ ok: true });
  });

  // Delete a trip/session entirely
  r.delete("/trip", (req: Request, res: Response) => {
    const sessionId = (req.query?.sessionId as string) || (req.body?.sessionId as string);
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    db.deleteSession(sessionId);
    // Also clear any parallel in-memory cache, if provided
    try {
      opts?.onDeleteSession?.(sessionId);
    } catch {}
    res.json({ ok: true });
  });

  // Expose saved POI IDs for a session so the client can restore Saved tab state
  r.get("/session/saved", (req: Request, res: Response) => {
    const sessionId = (req.query?.sessionId as string) || (req.body?.sessionId as string);
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    const s = db.getSession(sessionId);
    if (!s) return res.status(404).json({ error: "not found" });
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
            const history = session?.messages || [];
      const events = await opts.runRouter({ sessionId: sid, message, trip: (session?.trip as Partial<TripContext>) || {} }, history);
      for (const e of events) {
        if (e.type === "navbar.update") {
          const prev = db.getSession(sid)?.trip || {};
          db.upsertSession(sid, { trip: { ...(prev as Partial<TripContext>), ...e.data } });
        } else if (e.type === "chat.append") {
          db.appendMessage(sid, e.data as any);
        } else if (e.type === "itinerary.update") {
          const prev = db.getSession(sid)?.trip || {};
          db.upsertSession(sid, { trip: { ...prev, itinerary: e.data } as any });
        } else if (e.type === "search.results") {
          const prev = db.getSession(sid)?.trip || {};
          db.upsertSession(sid, { trip: { ...prev, searchResults: e.data } as any });
        } else if (e.type === "map.update") {
          const prev = db.getSession(sid)?.trip || {};
          db.upsertSession(sid, { trip: { ...prev, mapData: e.data } as any });
        }
        hub.emit(sid, e);
      }
      return res.json({ sessionId, inviteId, created: isNew, events });
    } catch (err) {
      hub.emit(sid, {
        type: "chat.append",
        data: { id: randomUUID(), role: "assistant", content: "Sorry, something went wrong.", createdAt: new Date().toISOString() },
      });
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
      const supabase = createClient(
        process.env.SUPABASE_URL!,
        process.env.SUPABASE_SERVICE_ROLE_KEY!
      );

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

  r.get("/trips/list", async (req: AuthenticatedRequest, res: Response) => {
    try {
      if (!req.userId) {
        console.log("No userId in trips/list request");
        return res.json({ trips: [] });
      }
      
      console.log("Fetching trips for userId:", req.userId);

      // Query sessions from database instead of memory
      const { createClient } = await import('@supabase/supabase-js');
      const supabase = createClient(
        process.env.SUPABASE_URL!,
        process.env.SUPABASE_SERVICE_ROLE_KEY!
      );

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
        return {
          id: s.session_id,
          sessionId: s.session_id,
          inviteId: s.invite_id,
          title,
          destination: trip.destination || null,
          duration,
          createdAt: s.created_at,
          updatedAt: s.updated_at,
        };
      });

      res.json({ trips: rows });
    } catch (error) {
      console.error('Error in trips/list:', error);
      res.status(500).json({ error: 'Failed to fetch trips' });
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
      
      const response = await fetch(photoUrl);
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
    } catch (error) {
      console.error('[proxy] Error fetching photo:', error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Mount maps proxy router
  r.use("/maps", buildMapsRouter());

  return r;
}
