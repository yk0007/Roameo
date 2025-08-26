import type { Db, SessionRecord } from "./types.js";

export class MemoryDb implements Db {
  private sessions = new Map<string, SessionRecord>();

  upsertSession(sessionId: string, data: Partial<SessionRecord>): SessionRecord {
    const existing = this.sessions.get(sessionId) || { sessionId, trip: {}, messages: [], savedPoiIds: new Set() };
    const updated = { ...existing, ...data };
    if (data.savedPoiIds) {
      updated.savedPoiIds = new Set([...existing.savedPoiIds, ...data.savedPoiIds]);
    }
    this.sessions.set(sessionId, updated);
    return updated;
  }

  getSession(sessionId: string): SessionRecord | undefined {
    return this.sessions.get(sessionId);
  }

  appendMessage(sessionId: string, msg: SessionRecord["messages"][number]): void {
    const s = this.upsertSession(sessionId, {});
    s.messages.push(msg);
    console.log(`[memory] appendMessage: session ${sessionId} now has ${s.messages.length} messages`);
  }

  patchTrip(sessionId: string, patch: Record<string, any>): void {
    const s = this.upsertSession(sessionId, {});
    s.trip = { ...s.trip, ...patch };
  }

  setInvite(sessionId: string, inviteId: string): void {
    const s = this.upsertSession(sessionId, {});
    s.inviteId = inviteId;
  }

  setPoiSaved(sessionId: string, poiId: string, saved: boolean): void {
    const s = this.upsertSession(sessionId, {});
    if (saved) s.savedPoiIds.add(poiId);
    else s.savedPoiIds.delete(poiId);
  }

  clearMessages(sessionId: string): void {
    const s = this.upsertSession(sessionId, {});
    s.messages = [];
  }

  deleteSession(sessionId: string): void {
    this.sessions.delete(sessionId);
  }

  listSessions(): SessionRecord[] {
    return Array.from(this.sessions.values());
  }
}
