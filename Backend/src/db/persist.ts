import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { Db, SessionRecord } from "./types.js";
import { MemoryDb } from "./memory.js";

export class WriteThroughDb implements Db {
  private mem = new MemoryDb();
  private client: SupabaseClient | null = null;

  constructor(url?: string, serviceKey?: string) {
    if (url && serviceKey) {
      this.client = createClient(url, serviceKey, { auth: { persistSession: false } });
    }
  }

  // One-time hydrate from Supabase -> memory
  async hydrateFromRemote(): Promise<void> {
    if (!this.client) return;
    console.log("[persist] Starting hydration from Supabase...");
    const { data: sessions, error } = await this.client
      .from("sessions")
      .select("session_id, invite_id, trip");
    if (error) {
      console.error("[persist] Failed to fetch sessions:", error);
      throw error;
    }
    console.log(`[persist] Found ${sessions?.length || 0} sessions`);
    for (const s of sessions || []) {
      const { data: messages, error: msgError } = await this.client
        .from("messages")
        .select("id, role, content, created_at")
        .eq("session_id", s.session_id)
        .order("created_at", { ascending: true });
      if (msgError) {
        console.error(`[persist] Failed to fetch messages for session ${s.session_id}:`, msgError);
        continue;
      }
      const { data: saved } = await this.client
        .from("saved_pois")
        .select("poi_id")
        .eq("session_id", s.session_id);
      console.log(`[persist] Session ${s.session_id}: ${messages?.length || 0} messages, ${saved?.length || 0} saved POIs`);
      this.mem.upsertSession(s.session_id, {
        inviteId: s.invite_id ?? undefined,
        trip: (s.trip as any) || {},
        messages: (messages || []).map((m: any) => ({ id: m.id, role: m.role, content: m.content, createdAt: new Date(m.created_at).toISOString() })),
        savedPoiIds: new Set((saved || []).map((r: any) => r.poi_id)),
      });
    }
    console.log("[persist] Hydration complete");
  }

  async upsertSession(sessionId: string, data: Partial<SessionRecord>): Promise<SessionRecord> {
    const rec = this.mem.upsertSession(sessionId, data);
    this.flushUpsert(sessionId, data).catch(() => {});
    return rec;
  }

  async getSession(sessionId: string): Promise<SessionRecord | undefined> {
    return this.mem.getSession(sessionId);
  }

  async appendMessage(sessionId: string, msg: SessionRecord["messages"][number]): Promise<void> {
    this.mem.appendMessage(sessionId, msg);
    console.log(`[persist] appendMessage called for session ${sessionId}, message ${msg.id}`);
    this.flushAppendMessage(sessionId, msg).catch((e) => {
      console.error(`[persist] flushAppendMessage failed:`, e);
    });
  }

  async patchTrip(sessionId: string, patch: Record<string, any>): Promise<void> {
    this.mem.patchTrip(sessionId, patch);
    this.flushPatchTrip(sessionId, patch).catch(() => {});
  }

  async setInvite(sessionId: string, inviteId: string): Promise<void> {
    this.mem.setInvite(sessionId, inviteId);
    this.flushSetInvite(sessionId, inviteId).catch(() => {});
  }

  async setPoiSaved(sessionId: string, poiId: string, saved: boolean): Promise<void> {
    this.mem.setPoiSaved(sessionId, poiId, saved);
    this.flushSetPoiSaved(sessionId, poiId, saved).catch(() => {});
  }

  async clearMessages(sessionId: string): Promise<void> {
    this.mem.clearMessages(sessionId);
    this.flushClearMessages(sessionId).catch(() => {});
  }

  async deleteSession(sessionId: string): Promise<void> {
    this.mem.deleteSession(sessionId);
    this.flushDeleteSession(sessionId).catch(() => {});
  }

  async listSessions(): Promise<SessionRecord[]> {
    return this.mem.listSessions();
  }

  // Flush helpers (best-effort, non-blocking)
  private async flushUpsert(sessionId: string, data: Partial<SessionRecord>) {
    if (!this.client) return;
    const base: any = { session_id: sessionId };
    if (data.inviteId !== undefined) base.invite_id = data.inviteId;
    if (data.trip !== undefined) base.trip = data.trip as any;
    if (data.userId !== undefined) base.user_id = data.userId;
    await this.client.from("chat_sessions").upsert(base, { onConflict: "session_id" });

    if (data.messages && data.messages.length) {
      try {
        const rows = data.messages.map((m) => ({ 
          id: m.id, 
          session_id: sessionId, 
          role: m.role, 
          content: m.content, 
          created_at: m.createdAt,
          user_id: data.userId || '00000000-0000-0000-0000-000000000000'
        }));
        const { error } = await this.client.from("messages").upsert(rows, { onConflict: "id" });
        
        if (error) {
          console.error(`[persist] Failed to bulk insert messages:`, error);
        }
      } catch (e) {
        console.error(`[persist] Exception during bulk message insert:`, e);
      }
    }
    if (data.savedPoiIds && data.savedPoiIds.size) {
      const rows = Array.from(data.savedPoiIds).map((poiId) => ({ session_id: sessionId, poi_id: poiId }));
      await this.client.from("saved_pois").upsert(rows, { onConflict: "session_id,poi_id" });
    }
  }

  private async flushAppendMessage(sessionId: string, msg: SessionRecord["messages"][number]) {
    if (!this.client) return;
    console.log(`[persist] Saving message ${msg.id} for session ${sessionId}`);
    
    // Ensure session exists in database first
    await this.ensureSessionExists(sessionId);
    
    try {
      // Save message with session_id that matches the chat_sessions.id
      const session = await this.mem.getSession(sessionId);
      const { error } = await this.client.from("messages").upsert({
        id: msg.id, 
        session_id: sessionId, 
        role: msg.role, 
        content: msg.content, 
        created_at: msg.createdAt,
        user_id: session?.userId || '00000000-0000-0000-0000-000000000000'
      });
      
      if (error) {
        console.error(`[persist] Failed to save message ${msg.id}:`, error);
      } else {
        console.log(`[persist] Successfully saved message ${msg.id}`);
      }
    } catch (e) {
      console.error(`[persist] Exception saving message ${msg.id}:`, e);
    }
  }

  private async flushPatchTrip(sessionId: string, patch: Record<string, any>) {
    if (!this.client) return;
    const session = await this.mem.getSession(sessionId);
    const cur = session?.trip || {};
    await this.client.from("chat_sessions").upsert({ session_id: sessionId, trip: cur }, { onConflict: "session_id" });
  }

  private async flushSetInvite(sessionId: string, inviteId: string) {
    if (!this.client) return;
    await this.client.from("chat_sessions").upsert({ session_id: sessionId, invite_id: inviteId }, { onConflict: "session_id" });
  }

  private async flushSetPoiSaved(sessionId: string, poiId: string, saved: boolean) {
    if (!this.client) return;
    if (saved) await this.client.from("saved_pois").upsert({ session_id: sessionId, poi_id: poiId });
    else await this.client.from("saved_pois").delete().eq("session_id", sessionId).eq("poi_id", poiId);
  }

  private async flushClearMessages(sessionId: string) {
    if (!this.client) return;
    await this.client.from("messages").delete().eq("session_id", sessionId);
  }

  private async flushDeleteSession(sessionId: string) {
    if (!this.client) return;
    // Delete related rows first to avoid orphans
    await this.client.from("messages").delete().eq("session_id", sessionId);
    await this.client.from("saved_pois").delete().eq("session_id", sessionId);
    await this.client.from("chat_sessions").delete().eq("session_id", sessionId);
  }

  private async ensureSessionExists(sessionId: string) {
    if (!this.client) return;
    try {
      // Check if session exists in chat_sessions table using 'session_id' column
      const { data: existingSession, error: selectError } = await this.client
        .from("chat_sessions")
        .select("session_id")
        .eq("session_id", sessionId)
        .maybeSingle();
      
      if (selectError && selectError.code !== 'PGRST116') {
        console.error(`[persist] Error checking session existence:`, selectError);
        return;
      }
      
      if (!existingSession) {
        console.log(`[persist] Creating session ${sessionId} in chat_sessions table`);
        const session = await this.mem.getSession(sessionId);
        const { error: insertError } = await this.client.from("chat_sessions").insert({
          session_id: sessionId,
          user_id: session?.userId,
          trip: session?.trip || {}
        });
        
        if (insertError) {
          console.error(`[persist] Failed to create session ${sessionId}:`, insertError);
        } else {
          console.log(`[persist] Successfully created session ${sessionId}`);
        }
      }
    } catch (e) {
      console.error(`[persist] Exception ensuring session exists:`, e);
    }
  }
}
