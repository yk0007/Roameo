import { Router, type Request, type Response } from "express";
import type { CachedDb } from "../cache/cached-db.js";

export function buildCacheRouter(db: any): Router {
  const router = Router();

  // Get cache statistics
  router.get("/stats", async (req: Request, res: Response) => {
    try {
      if (typeof db.getCacheStats !== "function") {
        return res.status(501).json({ error: "Cache stats not available" });
      }

      const stats = await db.getCacheStats();
      res.json({
        success: true,
        stats,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error("[cache-api] Stats error:", error);
      res.status(500).json({ 
        error: "Failed to get cache stats",
        message: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Cache health check
  router.get("/health", async (req: Request, res: Response) => {
    try {
      if (typeof db.cacheHealthCheck !== "function") {
        return res.status(501).json({ error: "Cache health check not available" });
      }

      const health = await db.cacheHealthCheck();
      const statusCode = health.healthy ? 200 : 503;
      
      res.status(statusCode).json({
        ...health,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error("[cache-api] Health check error:", error);
      res.status(500).json({ 
        healthy: false,
        message: `Health check failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Flush entire cache (admin operation)
  router.post("/flush", async (req: Request, res: Response) => {
    try {
      // This is a destructive operation, so we could add authentication here
      const authHeader = req.headers.authorization;
      const adminToken = process.env.ADMIN_TOKEN;
      
      if (adminToken && authHeader !== `Bearer ${adminToken}`) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      if (typeof db.flushCache !== "function") {
        return res.status(501).json({ error: "Cache flush not available" });
      }

      const success = await db.flushCache();
      
      if (success) {
        console.log("[cache-api] Cache flushed by admin request");
        res.json({ 
          success: true, 
          message: "Cache flushed successfully",
          timestamp: new Date().toISOString()
        });
      } else {
        res.status(500).json({ 
          success: false, 
          message: "Cache flush failed",
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      console.error("[cache-api] Flush error:", error);
      res.status(500).json({ 
        success: false,
        message: `Cache flush failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Warm cache with recent sessions
  router.post("/warm", async (req: Request, res: Response) => {
    try {
      const { sessionIds } = req.body;
      
      if (typeof db.warmCache !== "function") {
        return res.status(501).json({ error: "Cache warming not available" });
      }

      await db.warmCache(sessionIds);
      
      res.json({ 
        success: true, 
        message: sessionIds 
          ? `Cache warmed for ${sessionIds.length} specific sessions`
          : "Cache warmed with recent sessions",
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error("[cache-api] Warm error:", error);
      res.status(500).json({ 
        success: false,
        message: `Cache warming failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Get cached session data
  router.get("/session/:sessionId", async (req: Request, res: Response) => {
    try {
      const { sessionId } = req.params;
      
      if (typeof db.getSession !== "function") {
        return res.status(501).json({ error: "Session retrieval not available" });
      }

      const session = await db.getSession(sessionId);
      
      if (session) {
        res.json({ 
          success: true,
          session: {
            sessionId: session.sessionId,
            messageCount: session.messages?.length || 0,
            savedPoiCount: session.savedPoiIds?.size || 0,
            hasTrip: !!session.trip,
            hasInvite: !!session.inviteId
          },
          timestamp: new Date().toISOString()
        });
      } else {
        res.status(404).json({ 
          success: false,
          message: "Session not found",
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      console.error("[cache-api] Session retrieval error:", error);
      res.status(500).json({ 
        success: false,
        message: `Session retrieval failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Delete cached session data
  router.delete("/session/:sessionId", async (req: Request, res: Response) => {
    try {
      const { sessionId } = req.params;
      
      // This could be enhanced with proper authentication
      const authHeader = req.headers.authorization;
      const adminToken = process.env.ADMIN_TOKEN;
      
      if (adminToken && authHeader !== `Bearer ${adminToken}`) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      if (typeof db.deleteSession !== "function") {
        return res.status(501).json({ error: "Session deletion not available" });
      }

      db.deleteSession(sessionId);
      
      res.json({ 
        success: true,
        message: `Session ${sessionId} deleted from cache and database`,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error("[cache-api] Session deletion error:", error);
      res.status(500).json({ 
        success: false,
        message: `Session deletion failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Cache key management - get specific cached item
  router.get("/key/:key", async (req: Request, res: Response) => {
    try {
      const { key } = req.params;
      
      if (typeof db.cache?.get !== "function") {
        return res.status(501).json({ error: "Direct cache access not available" });
      }

      const value = await db.cache.get(key);
      
      if (value !== null) {
        res.json({ 
          success: true,
          key,
          hasValue: true,
          valueType: typeof value,
          timestamp: new Date().toISOString()
        });
      } else {
        res.status(404).json({ 
          success: false,
          key,
          hasValue: false,
          message: "Key not found in cache",
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      console.error("[cache-api] Key retrieval error:", error);
      res.status(500).json({ 
        success: false,
        message: `Key retrieval failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Delete specific cache key
  router.delete("/key/:key", async (req: Request, res: Response) => {
    try {
      const { key } = req.params;
      
      const authHeader = req.headers.authorization;
      const adminToken = process.env.ADMIN_TOKEN;
      
      if (adminToken && authHeader !== `Bearer ${adminToken}`) {
        return res.status(401).json({ error: "Unauthorized" });
      }

      if (typeof db.cache?.del !== "function") {
        return res.status(501).json({ error: "Direct cache access not available" });
      }

      const success = await db.cache.del(key);
      
      res.json({ 
        success,
        key,
        message: success ? "Key deleted successfully" : "Key deletion failed",
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error("[cache-api] Key deletion error:", error);
      res.status(500).json({ 
        success: false,
        message: `Key deletion failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: new Date().toISOString()
      });
    }
  });

  return router;
}