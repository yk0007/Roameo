import { WriteThroughDb } from '../db/persist.js';
// import { getMemcachedCache, type CacheLayer } from './memcached.js';
import type { Db, SessionRecord } from '../db/types.js';

// Simple cache interface
interface CacheLayer {
  setSession(sessionId: string, session: SessionRecord, ttl?: number): Promise<void>;
  getSession(sessionId: string): Promise<SessionRecord | null>;
  deleteSession(sessionId: string): Promise<void>;
  setSearchResults(query: string, results: any, ttl?: number): Promise<void>;
  getSearchResults(query: string): Promise<any>;
  setPoi(poiId: string, poi: any, ttl?: number): Promise<void>;
  getPoi(poiId: string): Promise<any>;
  setItinerary(sessionId: string, itinerary: any, ttl?: number): Promise<void>;
  getItinerary(sessionId: string): Promise<any>;
  setTripData(sessionId: string, tripData: any, ttl?: number): Promise<void>;
  getTripData(sessionId: string): Promise<any>;
  del(key: string): Promise<void>;
  flush(): Promise<boolean>;
  getStats(): Promise<any>;
}

// Fallback cache implementation when Memcached is unavailable
class NoOpCache implements CacheLayer {
  async setSession(): Promise<void> {}
  async getSession(): Promise<SessionRecord | null> { return null; }
  async deleteSession(): Promise<void> {}
  async setSearchResults(): Promise<void> {}
  async getSearchResults(): Promise<any> { return null; }
  async setPoi(): Promise<void> {}
  async getPoi(): Promise<any> { return null; }
  async setItinerary(): Promise<void> {}
  async getItinerary(): Promise<any> { return null; }
  async setTripData(): Promise<void> {}
  async getTripData(): Promise<any> { return null; }
  async del(): Promise<void> {}
  async flush(): Promise<boolean> { return true; }
  async getStats(): Promise<any> { return { healthy: false, message: 'No-op cache' }; }
}

export class CachedDb implements Db {
  private writeThruDb: WriteThroughDb;
  private cache: CacheLayer;
  private readonly SESSION_TTL = 3600; // 1 hour
  private readonly SEARCH_TTL = 1800; // 30 minutes
  private readonly POI_TTL = 7200; // 2 hours

  constructor(supabaseUrl?: string, supabaseServiceKey?: string) {
    this.writeThruDb = new WriteThroughDb(supabaseUrl, supabaseServiceKey);
    
    // Use simple no-op cache to avoid memcached connection issues
    console.log('[cached-db] Using no-op cache (memcached disabled)');
    this.cache = new NoOpCache();
  }

  async hydrateFromRemote(): Promise<void> {
    // First hydrate the write-through database
    await this.writeThruDb.hydrateFromRemote();
    
    // Then warm the cache with frequently accessed sessions
    console.log('[cached-db] Warming cache with active sessions...');
    const allSessions = this.writeThruDb.listSessions();
    
    // Cache the most recent sessions (up to 100)
    const recentSessions = allSessions
      .sort((a, b) => {
        const aTime = a.messages[a.messages.length - 1]?.createdAt || '0';
        const bTime = b.messages[b.messages.length - 1]?.createdAt || '0';
        return new Date(bTime).getTime() - new Date(aTime).getTime();
      })
      .slice(0, 100);

    const cachePromises = recentSessions.map(session => 
      this.cache.setSession(session.sessionId, session, this.SESSION_TTL)
    );
    
    await Promise.allSettled(cachePromises);
    console.log(`[cached-db] Cached ${recentSessions.length} recent sessions`);
  }

  upsertSession(sessionId: string, data: Partial<SessionRecord>): SessionRecord {
    // Update the database first
    const session = this.writeThruDb.upsertSession(sessionId, data);
    
    // Then update the cache (fire and forget)
    this.cache.setSession(sessionId, session, this.SESSION_TTL).catch((err: any) => 
      console.warn(`[cached-db] Failed to cache session ${sessionId}:`, err)
    );
    
    return session;
  }

  getSession(sessionId: string): SessionRecord | undefined {
    // For synchronous compatibility, we'll return from memory first
    // and do async cache operations in the background
    const memoryResult = this.writeThruDb.getSession(sessionId);
    
    if (memoryResult) {
      // Cache the result in the background
      this.cache.setSession(sessionId, memoryResult, this.SESSION_TTL).catch(() => {});
      return memoryResult;
    }
    
    // If not in memory, try cache asynchronously but return undefined for now
    // This maintains interface compatibility while still providing cache benefits
    this.tryAsyncCacheRetrieve(sessionId);
    
    return undefined;
  }
  
  // Helper method for async cache retrieval
  private async tryAsyncCacheRetrieve(sessionId: string): Promise<void> {
    try {
      const cached = await this.cache.getSession(sessionId);
      if (cached) {
        // Update memory database with cached data
        this.writeThruDb.upsertSession(sessionId, cached);
      }
    } catch (error) {
      console.warn(`[cached-db] Async cache retrieve failed for ${sessionId}:`, error);
    }
  }

  // Load session from database when not found in memory (for WebSocket connections)
  async loadSessionFromDatabase(sessionId: string): Promise<SessionRecord | undefined> {
    try {
      // Check cache first
      const cached = await this.cache.getSession(sessionId);
      if (cached) {
        console.log(`[cached-db] Found session ${sessionId} in cache`);
        // Update memory database with cached data
        this.writeThruDb.upsertSession(sessionId, cached);
        return cached;
      }

      // If not in cache, load from database using the writeThruDb's hydrateFromRemote logic
      const writeThruDb = this.writeThruDb as any;
      if (writeThruDb.client && typeof writeThruDb.client.from === 'function') {
        console.log(`[cached-db] Loading session ${sessionId} from database`);
        
        const { data: sessionData, error } = await writeThruDb.client
          .from('chat_sessions')
          .select('session_id, invite_id, trip')
          .eq('session_id', sessionId)
          .maybeSingle();

        if (error || !sessionData) {
          console.log(`[cached-db] Session ${sessionId} not found in database`);
          return undefined;
        }

        // Load messages and saved POIs
        const { data: messages } = await writeThruDb.client
          .from('messages')
          .select('id, role, content, created_at')
          .eq('session_id', sessionId)
          .order('created_at', { ascending: true });

        const { data: savedPois } = await writeThruDb.client
          .from('saved_pois')
          .select('poi_id')
          .eq('session_id', sessionId);

        // Build session record
        const sessionRecord: SessionRecord = {
          sessionId: sessionId,
          inviteId: sessionData.invite_id || undefined,
          trip: sessionData.trip || {},
          messages: (messages || []).map((m: any) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            createdAt: new Date(m.created_at).toISOString()
          })),
          savedPoiIds: new Set((savedPois || []).map((p: any) => p.poi_id))
        };

        // Update memory and cache
        this.writeThruDb.upsertSession(sessionId, sessionRecord);
        this.cache.setSession(sessionId, sessionRecord, this.SESSION_TTL).catch(() => {});

        console.log(`[cached-db] Successfully loaded session ${sessionId} from database`);
        return sessionRecord;
      }

      return undefined;
    } catch (error) {
      console.error(`[cached-db] Failed to load session ${sessionId} from database:`, error);
      return undefined;
    }
  }

  appendMessage(sessionId: string, msg: SessionRecord["messages"][number]): void {
    // Update database
    this.writeThruDb.appendMessage(sessionId, msg);
    
    // Invalidate cache to ensure consistency
    this.cache.deleteSession(sessionId).catch((err: any) => 
      console.warn(`[cached-db] Failed to invalidate cache for session ${sessionId}:`, err)
    );
  }

  patchTrip(sessionId: string, patch: Record<string, any>): void {
    // Update database first
    this.writeThruDb.patchTrip(sessionId, patch);
    
    // Invalidate related caches
    Promise.allSettled([
      this.cache.deleteSession(sessionId),
      this.cache.del(`trip:${sessionId}`),
      this.cache.del(`itinerary:${sessionId}`)
    ]).catch(() => {});
    
    // Special handling for itinerary updates
    if (patch.itinerary) {
      console.log(`[cached-db] Itinerary patch for session ${sessionId}`);
      // Immediately cache the updated itinerary
      this.setItineraryData(sessionId, patch.itinerary).catch((error) => {
        console.warn(`[cached-db] Failed to cache updated itinerary for ${sessionId}:`, error);
      });
    }
  }

  setInvite(sessionId: string, inviteId: string): void {
    // Update database
    this.writeThruDb.setInvite(sessionId, inviteId);
    
    // Invalidate cache
    this.cache.deleteSession(sessionId).catch(() => {});
  }

  setPoiSaved(sessionId: string, poiId: string, saved: boolean): void {
    // Update database
    this.writeThruDb.setPoiSaved(sessionId, poiId, saved);
    
    // Invalidate session cache
    this.cache.deleteSession(sessionId).catch(() => {});
  }

  clearMessages(sessionId: string): void {
    // Update database
    this.writeThruDb.clearMessages(sessionId);
    
    // Invalidate cache
    this.cache.deleteSession(sessionId).catch(() => {});
  }

  deleteSession(sessionId: string): void {
    // Update database
    this.writeThruDb.deleteSession(sessionId);
    
    // Clean up all related cache entries
    Promise.allSettled([
      this.cache.deleteSession(sessionId),
      this.cache.del(`trip:${sessionId}`),
      this.cache.del(`itinerary:${sessionId}`),
      this.cache.del(`search:${sessionId}`)
    ]).catch(() => {});
  }

  listSessions(): SessionRecord[] {
    // This always returns fresh data from the database
    // as it's not a frequently called operation
    return this.writeThruDb.listSessions();
  }

  // Enhanced methods with caching
  async getSearchResults(query: string, sessionId?: string): Promise<any | null> {
    try {
      const cacheKey = sessionId ? `${query}:${sessionId}` : query;
      return await this.cache.getSearchResults(cacheKey);
    } catch (error) {
      console.warn('[cached-db] Search cache read failed:', error);
      return null;
    }
  }

  async setSearchResults(query: string, results: any, sessionId?: string): Promise<void> {
    try {
      const cacheKey = sessionId ? `${query}:${sessionId}` : query;
      await this.cache.setSearchResults(cacheKey, results, this.SEARCH_TTL);
    } catch (error) {
      console.warn('[cached-db] Search cache write failed:', error);
    }
  }

  async getPoiData(poiId: string): Promise<any | null> {
    try {
      return await this.cache.getPoi(poiId);
    } catch (error) {
      console.warn(`[cached-db] POI cache read failed for ${poiId}:`, error);
      return null;
    }
  }

  async setPoiData(poiId: string, poi: any): Promise<void> {
    try {
      await this.cache.setPoi(poiId, poi, this.POI_TTL);
    } catch (error) {
      console.warn(`[cached-db] POI cache write failed for ${poiId}:`, error);
    }
  }

  async getItineraryData(sessionId: string): Promise<any | null> {
    try {
      return await this.cache.getItinerary(sessionId);
    } catch (error) {
      console.warn(`[cached-db] Itinerary cache read failed for ${sessionId}:`, error);
      return null;
    }
  }

  async setItineraryData(sessionId: string, itinerary: any): Promise<void> {
    try {
      await this.cache.setItinerary(sessionId, itinerary, this.SESSION_TTL);
    } catch (error) {
      console.warn(`[cached-db] Itinerary cache write failed for ${sessionId}:`, error);
    }
  }

  async getTripData(sessionId: string): Promise<any | null> {
    try {
      return await this.cache.getTripData(sessionId);
    } catch (error) {
      console.warn(`[cached-db] Trip data cache read failed for ${sessionId}:`, error);
      return null;
    }
  }

  async setTripData(sessionId: string, tripData: any): Promise<void> {
    try {
      await this.cache.setTripData(sessionId, tripData, this.SESSION_TTL);
    } catch (error) {
      console.warn(`[cached-db] Trip data cache write failed for ${sessionId}:`, error);
    }
  }

  // Cache management methods
  async flushCache(): Promise<boolean> {
    try {
      return await this.cache.flush();
    } catch (error) {
      console.error('[cached-db] Cache flush failed:', error);
      return false;
    }
  }

  async getCacheStats(): Promise<any> {
    try {
      return await this.cache.getStats();
    } catch (error) {
      console.error('[cached-db] Cache stats failed:', error);
      return { error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async cacheHealthCheck(): Promise<{ healthy: boolean; message: string }> {
    try {
      const memcachedCache = this.cache as any;
      if (memcachedCache.healthCheck) {
        return await memcachedCache.healthCheck();
      }
      return { healthy: true, message: 'Cache health check not available' };
    } catch (error) {
      return { 
        healthy: false, 
        message: `Cache health check failed: ${error instanceof Error ? error.message : 'Unknown error'}` 
      };
    }
  }

  // Batch operations for better performance
  async batchGetSessions(sessionIds: string[]): Promise<Record<string, SessionRecord | null>> {
    const results: Record<string, SessionRecord | null> = {};
    
    // Try to get all from cache first
    const cachePromises = sessionIds.map(async (sessionId) => {
      try {
        const cached = await this.cache.getSession(sessionId);
        return { sessionId, session: cached };
      } catch {
        return { sessionId, session: null };
      }
    });
    
    const cacheResults = await Promise.allSettled(cachePromises);
    const cacheMisses: string[] = [];
    
    cacheResults.forEach((result, index) => {
      const sessionId = sessionIds[index];
      if (result.status === 'fulfilled' && result.value.session) {
        results[sessionId] = result.value.session;
      } else {
        cacheMisses.push(sessionId);
      }
    });
    
    // Get cache misses from database
    for (const sessionId of cacheMisses) {
      const session = this.writeThruDb.getSession(sessionId);
      results[sessionId] = session || null;
      
      // Cache the result if found
      if (session) {
        this.cache.setSession(sessionId, session, this.SESSION_TTL).catch(() => {});
      }
    }
    
    return results;
  }

  // Warm cache with frequently accessed data
  async warmCache(sessionIds?: string[]): Promise<void> {
    const targetSessions = sessionIds || this.writeThruDb.listSessions()
      .sort((a, b) => {
        const aTime = a.messages[a.messages.length - 1]?.createdAt || '0';
        const bTime = b.messages[b.messages.length - 1]?.createdAt || '0';
        return new Date(bTime).getTime() - new Date(aTime).getTime();
      })
      .slice(0, 50)
      .map(s => s.sessionId);

    console.log(`[cached-db] Warming cache for ${targetSessions.length} sessions`);
    
    const warmupPromises = targetSessions.map(async (sessionId) => {
      const session = this.writeThruDb.getSession(sessionId);
      if (session) {
        await this.cache.setSession(sessionId, session, this.SESSION_TTL);
      }
    });
    
    await Promise.allSettled(warmupPromises);
    console.log('[cached-db] Cache warmup complete');
  }
}