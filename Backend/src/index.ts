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
import { SupabaseDb } from "./db/supabase.js";
import { SimpleRateLimiter } from "./utils/rateLimiter.js";

interface AuthenticatedMessage extends IncomingMessage {
  userId?: string;
}

const app = express();
app.use(cors({
  origin: [
    'https://roameo-app.vercel.app',
    'http://localhost:3000',
    'http://localhost:3001'
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'x-api-key'],
  optionsSuccessStatus: 200
}));
app.use(express.json());

const httpServer = createServer(app);
const wss = new WebSocketServer({ server: httpServer, path: "/ws" });
const hub = new WsHub();
// Prefer persistent DB if Supabase env is configured; write-through to Supabase while serving from memory for speed
const db: Db = new SupabaseDb(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!);
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

wss.on("connection", async (ws: WebSocket, req: AuthenticatedMessage) => {
  const url = new URL(req.url || "", `http://${req.headers.host}`);
  const sessionId = url.searchParams.get("sessionId");
  if (!sessionId) {
    ws.close(1008, "Missing sessionId");
    return;
  }

  let userId: string | undefined;
  let authProcessed = false;

  // Handle auth message from client
  const handleAuth = async (token: string) => {
    try {
      const { createClient } = await import('@supabase/supabase-js');
      const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!);
      const { data: { user } } = await supabase.auth.getUser(token);
      if (user) {
        userId = user.id;
        req.userId = userId;
        console.log(`[ws] Authenticated user ${userId} for session ${sessionId}`);
        
        // Update or create session with userId
        const existing = await db.getSession(sessionId);
        if (!existing) {
          console.log(`[ws] Creating new session ${sessionId} for user ${userId}`);
          await db.upsertSession(sessionId, { inviteId: sessions.get(sessionId)?.inviteId, trip: { sessionId }, userId });
        } else if (!existing.userId) {
          console.log(`[ws] Updating session ${sessionId} with userId ${userId}`);
          await db.upsertSession(sessionId, { userId });
        }
        
        authProcessed = true;
        initializeSession();
      }
    } catch (error) {
      console.error('WebSocket auth error:', error);
    }
  };

  // Initialize session after auth
  const initializeSession = async () => {
    const existing = await db.getSession(sessionId);
    
    // Keep inviteId from either DB or parallel in-memory map (if present)
    const inviteId = existing?.inviteId || sessions.get(sessionId)?.inviteId || undefined;
    if (inviteId && !existing?.inviteId) await db.setInvite(sessionId, inviteId);

    // Send session ready to all clients in the session
    hub.emit(sessionId, { type: "session.ready", data: { sessionId, inviteId } as any });
    // Send current trip navbar data
    if (existing?.trip) hub.emit(sessionId, { type: "navbar.update", data: existing.trip as any });
    // If we have a persisted itinerary from prior runs, replay it so UI restores
    const maybeItin = (existing?.trip as any)?.itinerary;
    if (maybeItin) hub.emit(sessionId, { type: "itinerary.update", data: maybeItin });
    // Replay last search results and map snapshot if present
    const maybeSearch = (existing?.trip as any)?.searchResults;
    if (maybeSearch) hub.emit(sessionId, { type: "search.results", data: maybeSearch });
    const maybeMap = (existing?.trip as any)?.mapData;
    if (maybeMap) hub.emit(sessionId, { type: "map.update", data: maybeMap });
    // Replay prior messages to rebuild chat UI on fresh connects
    if (existing?.messages?.length) {
      hub.emit(sessionId, { type: "chat.history", data: existing.messages as any });
    }
  };

  console.log(`[ws] Client connected to session ${sessionId}, waiting for auth`);

  // Handle incoming messages for auth
  ws.on('message', async (data) => {
    try {
      const message = JSON.parse(data.toString());
      if (message.type === 'auth' && message.token && !authProcessed) {
        await handleAuth(message.token);
      }
    } catch (error) {
      console.error('WebSocket message error:', error);
    }
  });

  // Fallback: try header auth first, then wait for message auth
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.substring(7);
    await handleAuth(token);
  }

  // If no auth after 2 seconds, create anonymous session
  setTimeout(async () => {
    if (!authProcessed) {
      console.log(`[ws] No auth received for session ${sessionId}, creating anonymous session`);
      const existing = await db.getSession(sessionId);
      if (!existing) {
        await db.upsertSession(sessionId, { inviteId: sessions.get(sessionId)?.inviteId, trip: { sessionId } });
      }
      authProcessed = true;
      initializeSession();
    }
  }, 2000);

  hub.attach(sessionId, ws);
});

// In-memory session store (MVP) — replace with Supabase later (db abstracts storage)
const sessions = new Map<string, { inviteId: string; trip: Partial<TripContext> }>();


const port = process.env.PORT || 4000;

async function init() {
  // Initialize database connection
  try {
    console.log("[roameo-backend] initializing database connection...");
    console.log("[roameo-backend] database initialized");
  } catch (e) {
    console.warn("[roameo-backend] database initialization failed (continuing):", e);
  }

  httpServer.listen(port, () => {
    console.log(`[roameo-backend] listening on http://localhost:${port}`);
  });
}

init();
