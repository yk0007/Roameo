import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { SessionRecord } from "./types.js";

export class SupabaseDb {
  private client: SupabaseClient;
  private static clientPool: Map<string, SupabaseClient> = new Map();

  constructor(url: string, serviceKey: string) {
    // Use connection pooling to reuse clients
    const poolKey = `${url}:${serviceKey}`;
    if (SupabaseDb.clientPool.has(poolKey)) {
      this.client = SupabaseDb.clientPool.get(poolKey)!;
    } else {
      this.client = createClient(url, serviceKey, {
        auth: { persistSession: false },
        db: { schema: "public" },
        global: {
          headers: {
            "x-client-info": "roameo-backend",
          },
        },
      });
      SupabaseDb.clientPool.set(poolKey, this.client);
    }
  }

  async upsertSession(
    sessionId: string,
    data: Partial<SessionRecord>,
  ): Promise<SessionRecord> {
    // Ensure base row exists
    const base: any = { session_id: sessionId };
    if (data.inviteId !== undefined) base.invite_id = data.inviteId;
    if (data.trip !== undefined) base.trip = data.trip as any;
    if (data.userId !== undefined) base.user_id = data.userId;

    const { error } = await this.client
      .from("chat_sessions")
      .upsert(base, { onConflict: "session_id" });
    if (error) throw error;

    // Append messages if provided
    if (data.messages && data.messages.length) {
      const rows = data.messages.map((m) => ({
        id: m.id,
        session_id: sessionId,
        role: m.role,
        content: m.content,
        created_at: m.createdAt,
      }));
      const { error: merr } = await this.client
        .from("messages")
        .upsert(rows, { onConflict: "id" });
      if (merr) throw merr;
    }

    // Merge saved POIs if provided
    if (data.savedPoiIds && data.savedPoiIds.size) {
      const inserts = Array.from(data.savedPoiIds).map((poiId) => ({
        session_id: sessionId,
        poi_id: poiId,
      }));
      const { error: perr } = await this.client
        .from("saved_pois")
        .upsert(inserts, { onConflict: "session_id,poi_id" });
      if (perr) throw perr;
    }

    return (await this.getSession(sessionId))!;
  }

  async getSession(sessionId: string): Promise<SessionRecord | undefined> {
    const { data: srow, error } = await this.client
      .from("chat_sessions")
      .select("session_id, invite_id, trip, user_id")
      .eq("session_id", sessionId)
      .maybeSingle();
    if (error) throw error;
    if (!srow) return undefined;

    const { data: mrows, error: merr } = await this.client
      .from("messages")
      .select("id, role, content, created_at")
      .eq("session_id", sessionId)
      .order("created_at", { ascending: true });
    if (merr) throw merr;

    const { data: prow, error: perr } = await this.client
      .from("saved_pois")
      .select("poi_id")
      .eq("session_id", sessionId);
    if (perr) throw perr;

    return {
      sessionId: srow.session_id,
      inviteId: srow.invite_id ?? undefined,
      trip: (srow.trip as any) || {},
      userId: srow.user_id ?? undefined,
      messages: (mrows || []).map((m) => ({
        id: m.id,
        role: m.role as any,
        content: m.content,
        createdAt: new Date(m.created_at!).toISOString(),
      })),
      savedPoiIds: new Set((prow || []).map((p) => p.poi_id)),
    };
  }

  async appendMessage(
    sessionId: string,
    msg: SessionRecord["messages"][number],
  ): Promise<void> {
    const { error } = await this.client.from("messages").insert({
      id: msg.id,
      session_id: sessionId,
      role: msg.role,
      content: msg.content,
      created_at: msg.createdAt,
    });
    if (error) throw error;
  }

  async patchTrip(
    sessionId: string,
    patch: Record<string, any>,
  ): Promise<void> {
    const current = (await this.getSession(sessionId))?.trip || {};
    const merged = { ...current, ...patch };
    const { error } = await this.client
      .from("chat_sessions")
      .upsert({ session_id: sessionId, trip: merged });
    if (error) throw error;
  }

  async setInvite(sessionId: string, inviteId: string): Promise<void> {
    const { error } = await this.client
      .from("chat_sessions")
      .upsert({ session_id: sessionId, invite_id: inviteId });
    if (error) throw error;
  }

  async setPoiSaved(
    sessionId: string,
    poiId: string,
    saved: boolean,
  ): Promise<void> {
    if (saved) {
      const { error } = await this.client
        .from("saved_pois")
        .upsert({ session_id: sessionId, poi_id: poiId });
      if (error) throw error;
    } else {
      const { error } = await this.client
        .from("saved_pois")
        .delete()
        .eq("session_id", sessionId)
        .eq("poi_id", poiId);
      if (error) throw error;
    }
  }

  async clearMessages(sessionId: string): Promise<void> {
    const { error } = await this.client
      .from("messages")
      .delete()
      .eq("session_id", sessionId);
    if (error) throw error;
  }

  async deleteSession(sessionId: string): Promise<void> {
    const { error } = await this.client
      .from("chat_sessions")
      .delete()
      .eq("session_id", sessionId);
    if (error) throw error;
  }

  async listSessions(): Promise<SessionRecord[]> {
    const { data: sessions, error } = await this.client
      .from("chat_sessions")
      .select("session_id, invite_id, trip, user_id, updated_at")
      .order("updated_at", { ascending: false })
      .limit(100); // Reasonable limit
    if (error) throw error;

    const results: SessionRecord[] = [];
    for (const s of sessions || []) {
      const sess = await this.getSession(s.session_id);
      if (sess) results.push(sess);
    }
    return results;
  }

  // Optimized method for fetching user trips without loading all session data
  async getUserTrips(
    userId: string,
    limit: number = 50,
  ): Promise<
    Array<{
      sessionId: string;
      inviteId?: string;
      trip: any;
      createdAt: string;
      updatedAt: string;
    }>
  > {
    const { data: sessions, error } = await this.client
      .from("chat_sessions")
      .select("session_id, invite_id, trip, created_at, updated_at")
      .eq("user_id", userId)
      .order("updated_at", { ascending: false })
      .limit(limit);

    if (error) throw error;

    return (sessions || []).map((s) => ({
      sessionId: s.session_id,
      inviteId: s.invite_id ?? undefined,
      trip: (s.trip as any) || {},
      createdAt: s.created_at,
      updatedAt: s.updated_at,
    }));
  }
}
