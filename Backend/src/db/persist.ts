import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { Db, SessionRecord } from "./types.js";
import { MemoryDb } from "./memory.js";

// Connection pool configuration for better performance
const SUPABASE_CONFIG = {
  auth: { persistSession: false },
  global: {
    headers: {
      'x-application-name': 'roameo-backend'
    }
  }
} as const;

export class WriteThroughDb implements Db {
  private mem = new MemoryDb();
  private client: SupabaseClient | null = null;
  private connectionHealth = true;
  private lastHealthCheck = 0;
  private readonly HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

  constructor(url?: string, serviceKey?: string) {
    if (url && serviceKey) {
      this.client = createClient(url, serviceKey, SUPABASE_CONFIG);
      this.setupHealthMonitoring();
    }
  }

  // Health monitoring for database connection
  private setupHealthMonitoring(): void {
    if (!this.client) return;
    
    setInterval(async () => {
      try {
        const { error } = await this.client!.from('chat_sessions').select('session_id').limit(1);
        this.connectionHealth = !error;
        this.lastHealthCheck = Date.now();
      } catch (e) {
        console.warn('[persist] Database health check failed:', e);
        this.connectionHealth = false;
      }
    }, this.HEALTH_CHECK_INTERVAL);
  }

  private isConnectionHealthy(): boolean {
    const timeSinceLastCheck = Date.now() - this.lastHealthCheck;
    return this.connectionHealth && timeSinceLastCheck < this.HEALTH_CHECK_INTERVAL * 2;
  }

  // Optimized one-time hydrate from Supabase -> memory with batch processing
  async hydrateFromRemote(): Promise<void> {
    if (!this.client) {
      console.warn('[persist] No Supabase client available, skipping hydration');
      return;
    }
    
    // For startup hydration, don't require health check - just try to connect
    console.log("[persist] Starting optimized hydration from Supabase...");
    
    try {
      // Test connection first with a simple query
      const { error: testError } = await this.client
        .from('chat_sessions')
        .select('session_id')
        .limit(1);
        
      if (testError) {
        console.warn('[persist] Database connection test failed:', testError);
        console.warn('[persist] Skipping hydration due to connection issues');
        return;
      }
      
      // Mark connection as healthy since the test succeeded
      this.connectionHealth = true;
      this.lastHealthCheck = Date.now();
      
      // Use a custom query to get all sessions with recent messages for hydration
      // Since get_session_with_recent_messages requires a specific session_id, 
      // we'll use a different approach for bulk hydration
      const { data: sessions, error: sessionsError } = await this.client
        .from('chat_sessions')
        .select('session_id, invite_id, trip, created_at, updated_at');
        
      if (sessionsError) {
        console.error("[persist] Failed to fetch sessions:", sessionsError);
        return this.hydrateFromRemoteClassic();
      }
      
      if (!sessions || sessions.length === 0) {
        console.log("[persist] No sessions found");
        return;
      }
      
      // Get recent messages for all sessions in batches
      const sessionIds = sessions.map(s => s.session_id);
      const { data: allMessages, error: messagesError } = await this.client
        .from('messages')
        .select('id, role, content, created_at, session_id')
        .in('session_id', sessionIds)
        .order('created_at', { ascending: false })
        .limit(100 * sessions.length); // Reasonable limit per session
        
      if (messagesError) {
        console.error("[persist] Failed to fetch messages:", messagesError);
        return this.hydrateFromRemoteClassic();
      }
      
      // Group messages by session and limit to recent messages
      const messagesBySession = new Map<string, any[]>();
      (allMessages || []).forEach(msg => {
        if (!messagesBySession.has(msg.session_id)) {
          messagesBySession.set(msg.session_id, []);
        }
        const sessionMessages = messagesBySession.get(msg.session_id)!;
        if (sessionMessages.length < 100) { // Limit per session
          sessionMessages.push(msg);
        }
      });
      
      // Reverse messages to get chronological order for each session
      messagesBySession.forEach(messages => {
        messages.reverse();
      });
      
      const sessionsWithMessages = sessions.map(session => ({
        ...session,
        messages: messagesBySession.get(session.session_id) || []
      }));
      
      console.log(`[persist] Found ${sessionsWithMessages?.length || 0} sessions`);
      
      // Batch fetch all saved POIs for all sessions
      const { data: allSavedPois } = await this.client
        .from("saved_pois")
        .select("session_id, poi_id")
        .in('session_id', sessionIds);
      
      // Group saved POIs by session
      const savedPoisBySession = new Map<string, string[]>();
      (allSavedPois || []).forEach((poi: any) => {
        if (!savedPoisBySession.has(poi.session_id)) {
          savedPoisBySession.set(poi.session_id, []);
        }
        savedPoisBySession.get(poi.session_id)!.push(poi.poi_id);
      });
      
      // Process each session
      for (const session of sessionsWithMessages || []) {
        const messages = Array.isArray(session.messages) ? session.messages : [];
        const savedPoiIds = savedPoisBySession.get(session.session_id) || [];
        
        console.log(`[persist] Session ${session.session_id}: ${messages.length} messages, ${savedPoiIds.length} saved POIs`);
        
        this.mem.upsertSession(session.session_id, {
          inviteId: session.invite_id ?? undefined,
          trip: session.trip || {},
          messages: messages.map((m: any) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            createdAt: m.createdAt || new Date().toISOString()
          })),
          savedPoiIds: new Set(savedPoiIds),
        });
      }
      
      console.log("[persist] Optimized hydration complete");
    } catch (error) {
      console.error('[persist] Error during optimized hydration:', error);
      // Fallback to classic method
      return this.hydrateFromRemoteClassic();
    }
  }
  
  // Fallback classic hydration method
  private async hydrateFromRemoteClassic(): Promise<void> {
    if (!this.client) return;
    console.log("[persist] Using classic hydration method...");
    const { data: sessions, error } = await this.client
      .from("chat_sessions")
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

  upsertSession(sessionId: string, data: Partial<SessionRecord>): SessionRecord {
    const rec = this.mem.upsertSession(sessionId, data);
    this.flushUpsert(sessionId, data).catch(() => {});
    return rec;
  }

  getSession(sessionId: string): SessionRecord | undefined {
    return this.mem.getSession(sessionId);
  }

  // Performance monitoring helper
  private async withPerformanceMonitoring<T>(
    operation: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const start = Date.now();
    try {
      const result = await fn();
      const duration = Date.now() - start;
      if (duration > 1000) { // Log slow queries
        console.warn(`[persist] Slow ${operation}: ${duration}ms`);
      }
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      console.error(`[persist] Failed ${operation} after ${duration}ms:`, error);
      throw error;
    }
  }

  // Optimized message append with batching capability
  private messageQueue = new Map<string, SessionRecord["messages"][number][]>();
  private flushTimeout: NodeJS.Timeout | null = null;
  
  appendMessage(sessionId: string, msg: SessionRecord["messages"][number]): void {
    this.mem.appendMessage(sessionId, msg);
    
    // Add to queue for batch processing
    if (!this.messageQueue.has(sessionId)) {
      this.messageQueue.set(sessionId, []);
    }
    this.messageQueue.get(sessionId)!.push(msg);
    
    // Batch flush messages for better performance
    if (this.flushTimeout) {
      clearTimeout(this.flushTimeout);
    }
    
    this.flushTimeout = setTimeout(() => {
      this.flushQueuedMessages().catch((e) => {
        console.error(`[persist] Failed to flush queued messages:`, e);
      });
    }, 100); // 100ms batching window
  }
  
  private async flushQueuedMessages(): Promise<void> {
    if (!this.client || this.messageQueue.size === 0) return;
    
    const allMessages: any[] = [];
    
    for (const [sessionId, messages] of this.messageQueue.entries()) {
      const session = this.mem.getSession(sessionId);
      for (const msg of messages) {
        allMessages.push({
          id: msg.id,
          session_id: sessionId,
          role: msg.role,
          content: msg.content,
          created_at: msg.createdAt,
          user_id: session?.userId || '00000000-0000-0000-0000-000000000000'
        });
      }
    }
    
    if (allMessages.length > 0) {
      await this.withPerformanceMonitoring('batch-insert-messages', async () => {
        const { error } = await this.client!.from("messages").upsert(allMessages, { onConflict: "id" });
        if (error) throw error;
      });
      
      console.log(`[persist] Batch flushed ${allMessages.length} messages`);
    }
    
    this.messageQueue.clear();
  }

  patchTrip(sessionId: string, patch: Record<string, any>): void {
    this.mem.patchTrip(sessionId, patch);
    // For critical data like itinerary, ensure immediate persistence
    if (patch.itinerary) {
      console.log(`[persist] Immediate flush for itinerary update on session ${sessionId}`);
      this.flushPatchTrip(sessionId, patch).catch((error) => {
        console.error(`[persist] CRITICAL: Failed to persist itinerary for session ${sessionId}:`, error);
      });
    } else {
      this.flushPatchTrip(sessionId, patch).catch(() => {});
    }
  }

  setInvite(sessionId: string, inviteId: string): void {
    this.mem.setInvite(sessionId, inviteId);
    this.flushSetInvite(sessionId, inviteId).catch(() => {});
  }

  setPoiSaved(sessionId: string, poiId: string, saved: boolean): void {
    this.mem.setPoiSaved(sessionId, poiId, saved);
    this.flushSetPoiSaved(sessionId, poiId, saved).catch(() => {});
  }

  clearMessages(sessionId: string): void {
    this.mem.clearMessages(sessionId);
    this.flushClearMessages(sessionId).catch(() => {});
  }

  deleteSession(sessionId: string): void {
    this.mem.deleteSession(sessionId);
    this.flushDeleteSession(sessionId).catch(() => {});
  }

  listSessions(): SessionRecord[] {
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
      const session = this.mem.getSession(sessionId);
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
    if (!this.client) {
      console.warn(`[persist] No client available for flushing trip patch on session ${sessionId}`);
      return;
    }
    
    try {
      const cur = this.mem.getSession(sessionId)?.trip || {};
      console.log(`[persist] Flushing trip patch for session ${sessionId}:`, JSON.stringify(patch, null, 2));
      
      const { error } = await this.client
        .from("chat_sessions")
        .upsert({ session_id: sessionId, trip: cur }, { onConflict: "session_id" });
      
      if (error) {
        console.error(`[persist] Database error flushing trip patch for session ${sessionId}:`, error);
        throw error;
      }
      
      console.log(`[persist] Successfully flushed trip patch for session ${sessionId}`);
    } catch (error) {
      console.error(`[persist] Exception flushing trip patch for session ${sessionId}:`, error);
      throw error;
    }
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
