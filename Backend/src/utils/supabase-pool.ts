import { createClient, type SupabaseClient } from "@supabase/supabase-js";

interface PoolOptions {
  maxConnections?: number;
  idleTimeoutMs?: number;
  connectionTimeoutMs?: number;
}

class SupabaseConnectionPool {
  private static instance: SupabaseConnectionPool;
  private pools: Map<string, SupabaseClient[]> = new Map();
  private activeConnections: Map<string, Set<SupabaseClient>> = new Map();
  private readonly maxConnections: number;
  private readonly idleTimeoutMs: number;
  private readonly connectionTimeoutMs: number;

  private constructor(options: PoolOptions = {}) {
    this.maxConnections = options.maxConnections || 10;
    this.idleTimeoutMs = options.idleTimeoutMs || 300000; // 5 minutes
    this.connectionTimeoutMs = options.connectionTimeoutMs || 10000; // 10 seconds

    // Cleanup idle connections periodically
    setInterval(() => this.cleanupIdleConnections(), this.idleTimeoutMs / 2);
  }

  static getInstance(options?: PoolOptions): SupabaseConnectionPool {
    if (!SupabaseConnectionPool.instance) {
      SupabaseConnectionPool.instance = new SupabaseConnectionPool(options);
    }
    return SupabaseConnectionPool.instance;
  }

  private getPoolKey(url: string, serviceKey: string): string {
    return `${url}:${serviceKey.substring(0, 10)}`;
  }

  private createConnection(url: string, serviceKey: string): SupabaseClient {
    return createClient(url, serviceKey, {
      auth: { persistSession: false },
      db: { schema: "public" },
      global: {
        headers: {
          "x-client-info": "roameo-backend-pool",
          "x-connection-pool": "true",
        },
      },
    });
  }

  async getConnection(url: string, serviceKey: string): Promise<SupabaseClient> {
    const poolKey = this.getPoolKey(url, serviceKey);

    // Initialize pool if it doesn't exist
    if (!this.pools.has(poolKey)) {
      this.pools.set(poolKey, []);
      this.activeConnections.set(poolKey, new Set());
    }

    const pool = this.pools.get(poolKey)!;
    const active = this.activeConnections.get(poolKey)!;

    // Try to get an available connection from the pool
    if (pool.length > 0) {
      const connection = pool.pop()!;
      active.add(connection);
      return connection;
    }

    // Create new connection if we haven't reached the limit
    if (active.size < this.maxConnections) {
      const connection = this.createConnection(url, serviceKey);
      active.add(connection);
      return connection;
    }

    // Wait for an available connection
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Connection pool timeout"));
      }, this.connectionTimeoutMs);

      const checkForConnection = () => {
        if (pool.length > 0) {
          clearTimeout(timeout);
          const connection = pool.pop()!;
          active.add(connection);
          resolve(connection);
        } else {
          setTimeout(checkForConnection, 100);
        }
      };

      checkForConnection();
    });
  }

  releaseConnection(url: string, serviceKey: string, connection: SupabaseClient): void {
    const poolKey = this.getPoolKey(url, serviceKey);
    const pool = this.pools.get(poolKey);
    const active = this.activeConnections.get(poolKey);

    if (pool && active && active.has(connection)) {
      active.delete(connection);
      pool.push(connection);
    }
  }

  private cleanupIdleConnections(): void {
    for (const [poolKey, pool] of this.pools.entries()) {
      // Keep at least 1 connection in each pool, remove extras
      while (pool.length > 1) {
        pool.pop();
      }
    }
  }

  async closeAllConnections(): Promise<void> {
    for (const [poolKey, pool] of this.pools.entries()) {
      pool.length = 0; // Clear the pool
    }
    for (const [poolKey, active] of this.activeConnections.entries()) {
      active.clear(); // Clear active connections
    }
    this.pools.clear();
    this.activeConnections.clear();
  }

  getPoolStats(): Record<string, { available: number; active: number }> {
    const stats: Record<string, { available: number; active: number }> = {};

    for (const [poolKey, pool] of this.pools.entries()) {
      const active = this.activeConnections.get(poolKey) || new Set();
      stats[poolKey] = {
        available: pool.length,
        active: active.size,
      };
    }

    return stats;
  }
}

export default SupabaseConnectionPool;

// Convenience function for getting a pooled connection
export async function getPooledSupabaseClient(
  url: string,
  serviceKey: string,
): Promise<{ client: SupabaseClient; release: () => void }> {
  const pool = SupabaseConnectionPool.getInstance();
  const client = await pool.getConnection(url, serviceKey);

  return {
    client,
    release: () => pool.releaseConnection(url, serviceKey, client),
  };
}
