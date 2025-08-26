import type { Db, SessionRecord } from "./types.js";

export class MemoryDb implements Db {
  private sessions = new Map<string, SessionRecord>();

  async upsertSession(sessionId: string, data: Partial<SessionRecord>): Promise<SessionRecord> {
    const existing = this.sessions.get(sessionId) || { sessionId, trip: {}, messages: [], savedPoiIds: new Set() };
    const updated = { ...existing, ...data };
    if (data.savedPoiIds) {
      updated.savedPoiIds = new Set([...existing.savedPoiIds, ...data.savedPoiIds]);
    }
    this.sessions.set(sessionId, updated);
    return updated;
  }

  async getSession(sessionId: string): Promise<SessionRecord | undefined> {
    return this.sessions.get(sessionId);
  }

  async appendMessage(sessionId: string, msg: SessionRecord["messages"][number]): Promise<void> {
    const s = await this.upsertSession(sessionId, {});
    s.messages.push(msg);
    console.log(`[memory] appendMessage: session ${sessionId} now has ${s.messages.length} messages`);
  }

  async patchTrip(sessionId: string, patch: Record<string, any>): Promise<void> {
    const s = await this.upsertSession(sessionId, {});
    s.trip = { ...s.trip, ...patch };
  }

  async setInvite(sessionId: string, inviteId: string): Promise<void> {
    const s = await this.upsertSession(sessionId, {});
    s.inviteId = inviteId;
  }

  async setPoiSaved(sessionId: string, poiId: string, saved: boolean): Promise<void> {
    const s = await this.upsertSession(sessionId, {});
    if (saved) s.savedPoiIds.add(poiId);
    else s.savedPoiIds.delete(poiId);
  }

  async clearMessages(sessionId: string): Promise<void> {
    const s = await this.upsertSession(sessionId, {});
    s.messages = [];
  }

  async deleteSession(sessionId: string): Promise<void> {
    this.sessions.delete(sessionId);
  }

  async listSessions(): Promise<SessionRecord[]> {
    return Array.from(this.sessions.values());
  }
}
