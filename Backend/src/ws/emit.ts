import type { WsEvent } from "../types/schemas.js";
import type WebSocket from "ws";
import { SimpleRateLimiter } from "../utils/rateLimiter.js";

export type SessionId = string;

export class WsHub {
  private sessions = new Map<SessionId, Set<WebSocket>>();
  // Limit to 100 events per 10 seconds per session (tune as needed)
  private limiter = new SimpleRateLimiter(100, 10_000);

  attach(sessionId: SessionId, ws: WebSocket) {
    if (!this.sessions.has(sessionId)) this.sessions.set(sessionId, new Set());
    this.sessions.get(sessionId)!.add(ws);

    ws.on("close", () => {
      const set = this.sessions.get(sessionId);
      if (!set) return;
      set.delete(ws);
      if (set.size === 0) this.sessions.delete(sessionId);
    });
  }

  emit(sessionId: SessionId, event: WsEvent) {
    // Drop bursts to avoid overwhelming clients/network
    if (!this.limiter.allow(`ws-out:${sessionId}`)) return;
    const set = this.sessions.get(sessionId);
    if (!set) return;
    const payload = JSON.stringify(event);
    for (const ws of set) {
      if (ws.readyState === ws.OPEN) ws.send(payload);
    }
  }
}
