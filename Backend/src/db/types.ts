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
  userId?: string;
}

export interface Db {
  upsertSession(sessionId: string, data: Partial<SessionRecord>): Promise<SessionRecord>;
  getSession(sessionId: string): Promise<SessionRecord | undefined>;
  appendMessage(sessionId: string, msg: SessionRecord["messages"][number]): Promise<void>;
  patchTrip(sessionId: string, patch: Record<string, any>): Promise<void>;
  setInvite(sessionId: string, inviteId: string): Promise<void>;
  setPoiSaved(sessionId: string, poiId: string, saved: boolean): Promise<void>;
  clearMessages(sessionId: string): Promise<void>;
  deleteSession(sessionId: string): Promise<void>;
  listSessions(): Promise<SessionRecord[]>;
}
