export type Message = {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  createdAt: string;
};

export interface SessionRecord {
  sessionId: string;
  inviteId?: string;
  trip: Record<string, any>;
    messages: Message[];
  savedPoiIds: Set<string>;
}

export interface Db {
  upsertSession(sessionId: string, data: Partial<SessionRecord>): SessionRecord;
  getSession(sessionId: string): SessionRecord | undefined;
  appendMessage(sessionId: string, msg: SessionRecord["messages"][number]): void;
  patchTrip(sessionId: string, patch: Record<string, any>): void;
  setInvite(sessionId: string, inviteId: string): void;
  setPoiSaved(sessionId: string, poiId: string, saved: boolean): void;
  clearMessages(sessionId: string): void;
  deleteSession(sessionId: string): void;
  listSessions(): SessionRecord[];
}
