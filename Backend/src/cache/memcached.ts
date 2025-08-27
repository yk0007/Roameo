import Memcached from 'memcached';
import type { SessionRecord } from '../db/types.js';

export interface CacheLayer {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T, ttl?: number): Promise<boolean>;
  del(key: string): Promise<boolean>;
  exists(key: string): Promise<boolean>;
  flush(): Promise<boolean>;
  getStats(): Promise<any>;
  
  // Session-specific methods
  getSession(sessionId: string): Promise<SessionRecord | null>;
  setSession(sessionId: string, session: SessionRecord, ttl?: number): Promise<boolean>;
  deleteSession(sessionId: string): Promise<boolean>;
  
  // Search results methods
  getSearchResults(query: string): Promise<any | null>;
  setSearchResults(query: string, results: any, ttl?: number): Promise<boolean>;
  
  // POI methods
  getPoi(poiId: string): Promise<any | null>;
  setPoi(poiId: string, poi: any, ttl?: number): Promise<boolean>;
  
  // Itinerary methods
  getItinerary(sessionId: string): Promise<any | null>;
  setItinerary(sessionId: string, itinerary: any, ttl?: number): Promise<boolean>;
  
  // Trip data methods
  getTripData(sessionId: string): Promise<any | null>;
  setTripData(sessionId: string, tripData: any, ttl?: number): Promise<boolean>;
}

export class MemcachedCache implements CacheLayer {
  private client: Memcached;
  private isConnected = false;
  private connectionRetries = 0;
  private readonly maxRetries = 5;
  private readonly retryDelay = 1000; // 1 second
  
  constructor(servers?: string | string[], options?: Memcached.options) {
    const defaultServers = servers || process.env.MEMCACHED_SERVERS || 'localhost:11211';
    const defaultOptions: Memcached.options = {
      timeout: 5000,
      retries: 3,
      retry: 30000,
      remove: true,
      failOverServers: Array.isArray(defaultServers) ? defaultServers.slice(1) : undefined,
      ...options
    };
    
    this.client = new Memcached(defaultServers, defaultOptions);
    this.setupEventHandlers();
    this.testConnection();
  }

  private setupEventHandlers(): void {
    this.client.on('issue', (issue) => {
      console.warn('[memcached] Connection issue:', issue);
      this.isConnected = false;
    });

    this.client.on('failure', (failure) => {
      console.error('[memcached] Connection failure:', failure);
      this.isConnected = false;
    });

    this.client.on('reconnecting', (details) => {
      console.log('[memcached] Reconnecting:', details);
    });

    this.client.on('reconnect', (details) => {
      console.log('[memcached] Reconnected:', details);
      this.isConnected = true;
      this.connectionRetries = 0;
    });
  }

  private async testConnection(): Promise<void> {
    try {
      await this.set('test_connection', 'ok', 10);
      const result = await this.get<string>('test_connection');
      if (result === 'ok') {
        this.isConnected = true;
        console.log('[memcached] Connected successfully');
        await this.del('test_connection');
      } else {
        throw new Error('Connection test failed');
      }
    } catch (error) {
      console.error('[memcached] Connection failed:', error);
      this.isConnected = false;
      
      if (this.connectionRetries < this.maxRetries) {
        this.connectionRetries++;
        console.log(`[memcached] Retrying connection (${this.connectionRetries}/${this.maxRetries})`);
        setTimeout(() => this.testConnection(), this.retryDelay * this.connectionRetries);
      }
    }
  }

  async get<T>(key: string): Promise<T | null> {
    if (!this.isConnected) {
      console.warn('[memcached] Not connected, skipping get');
      return null;
    }

    return new Promise((resolve, reject) => {
      this.client.get(key, (err, data) => {
        if (err) {
          console.error(`[memcached] Get error for key ${key}:`, err);
          resolve(null); // Graceful degradation
        } else {
          try {
            const parsed = data ? JSON.parse(data) : null;
            resolve(parsed);
          } catch (parseError) {
            console.error(`[memcached] Parse error for key ${key}:`, parseError);
            resolve(null);
          }
        }
      });
    });
  }

  async set<T>(key: string, value: T, ttl: number = 3600): Promise<boolean> {
    if (!this.isConnected) {
      console.warn('[memcached] Not connected, skipping set');
      return false;
    }

    return new Promise((resolve) => {
      try {
        const serialized = JSON.stringify(value);
        this.client.set(key, serialized, ttl, (err) => {
          if (err) {
            console.error(`[memcached] Set error for key ${key}:`, err);
            resolve(false);
          } else {
            resolve(true);
          }
        });
      } catch (serializeError) {
        console.error(`[memcached] Serialization error for key ${key}:`, serializeError);
        resolve(false);
      }
    });
  }

  async del(key: string): Promise<boolean> {
    if (!this.isConnected) {
      console.warn('[memcached] Not connected, skipping delete');
      return false;
    }

    return new Promise((resolve) => {
      this.client.del(key, (err) => {
        if (err) {
          console.error(`[memcached] Delete error for key ${key}:`, err);
          resolve(false);
        } else {
          resolve(true);
        }
      });
    });
  }

  async exists(key: string): Promise<boolean> {
    const value = await this.get(key);
    return value !== null;
  }

  async flush(): Promise<boolean> {
    if (!this.isConnected) {
      console.warn('[memcached] Not connected, skipping flush');
      return false;
    }

    return new Promise((resolve) => {
      this.client.flush((err) => {
        if (err) {
          console.error('[memcached] Flush error:', err);
          resolve(false);
        } else {
          console.log('[memcached] Cache flushed successfully');
          resolve(true);
        }
      });
    });
  }

  async getStats(): Promise<any> {
    if (!this.isConnected) {
      return { connected: false };
    }

    return new Promise((resolve) => {
      this.client.stats((err, stats) => {
        if (err) {
          console.error('[memcached] Stats error:', err);
          resolve({ connected: true, error: err.message });
        } else {
          resolve({ connected: true, stats });
        }
      });
    });
  }

  // Convenience methods for common session operations
  async getSession(sessionId: string): Promise<SessionRecord | null> {
    return this.get<SessionRecord>(`session:${sessionId}`);
  }

  async setSession(sessionId: string, session: SessionRecord, ttl: number = 3600): Promise<boolean> {
    return this.set(`session:${sessionId}`, session, ttl);
  }

  async deleteSession(sessionId: string): Promise<boolean> {
    return this.del(`session:${sessionId}`);
  }

  // Search results caching
  async getSearchResults(query: string): Promise<any | null> {
    const key = `search:${Buffer.from(query).toString('base64')}`;
    return this.get(key);
  }

  async setSearchResults(query: string, results: any, ttl: number = 1800): Promise<boolean> {
    const key = `search:${Buffer.from(query).toString('base64')}`;
    return this.set(key, results, ttl);
  }

  // POI data caching
  async getPoi(poiId: string): Promise<any | null> {
    return this.get(`poi:${poiId}`);
  }

  async setPoi(poiId: string, poi: any, ttl: number = 7200): Promise<boolean> {
    return this.set(`poi:${poiId}`, poi, ttl);
  }

  // Itinerary caching
  async getItinerary(sessionId: string): Promise<any | null> {
    return this.get(`itinerary:${sessionId}`);
  }

  async setItinerary(sessionId: string, itinerary: any, ttl: number = 3600): Promise<boolean> {
    return this.set(`itinerary:${sessionId}`, itinerary, ttl);
  }

  // Trip data caching
  async getTripData(sessionId: string): Promise<any | null> {
    return this.get(`trip:${sessionId}`);
  }

  async setTripData(sessionId: string, tripData: any, ttl: number = 3600): Promise<boolean> {
    return this.set(`trip:${sessionId}`, tripData, ttl);
  }

  // Graceful shutdown
  async close(): Promise<void> {
    return new Promise((resolve) => {
      if (this.client) {
        this.client.end();
        this.isConnected = false;
        console.log('[memcached] Connection closed');
      }
      resolve();
    });
  }

  // Health check method
  async healthCheck(): Promise<{ healthy: boolean; message: string }> {
    try {
      const testKey = 'health_check_' + Date.now();
      const testValue = 'healthy';
      
      const setResult = await this.set(testKey, testValue, 10);
      if (!setResult) {
        return { healthy: false, message: 'Failed to set test value' };
      }
      
      const getValue = await this.get<string>(testKey);
      if (getValue !== testValue) {
        return { healthy: false, message: 'Failed to retrieve test value' };
      }
      
      await this.del(testKey);
      return { healthy: true, message: 'Memcached is healthy' };
    } catch (error) {
      return { 
        healthy: false, 
        message: `Health check failed: ${error instanceof Error ? error.message : 'Unknown error'}` 
      };
    }
  }
}

// Singleton instance
let memcachedInstance: MemcachedCache | null = null;

export function getMemcachedCache(): MemcachedCache {
  if (!memcachedInstance) {
    memcachedInstance = new MemcachedCache();
  }
  return memcachedInstance;
}

// Graceful shutdown handler
process.on('SIGINT', async () => {
  if (memcachedInstance) {
    await memcachedInstance.close();
  }
});

process.on('SIGTERM', async () => {
  if (memcachedInstance) {
    await memcachedInstance.close();
  }
});