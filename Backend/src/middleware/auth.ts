import type { NextFunction, Request, Response } from "express";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { env } from "../config/env.js";

let supabase: SupabaseClient | null = null;

if (env.SUPABASE_URL && env.SUPABASE_SERVICE_ROLE_KEY) {
  supabase = createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_ROLE_KEY, {
    auth: { persistSession: false }
  });
}

export interface AuthenticatedRequest extends Request {
  userId?: string;
  user?: unknown;
}

async function resolveUser(token: string) {
  if (!supabase) {
    throw new Error("Supabase auth is not configured");
  }

  const {
    data: { user },
    error
  } = await supabase.auth.getUser(token);

  if (error || !user) {
    return null;
  }

  return user;
}

export async function authenticateUser(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
) {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader?.startsWith("Bearer ")) {
      return res
        .status(401)
        .json({ error: "No authorization token provided" });
    }

    const user = await resolveUser(authHeader.slice(7));
    if (!user) {
      return res.status(401).json({ error: "Invalid or expired token" });
    }

    req.userId = user.id;
    req.user = user;
    return next();
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Authentication failed";
    const status = message.includes("not configured") ? 503 : 500;
    return res.status(status).json({ error: message });
  }
}

export async function optionalAuth(
  req: AuthenticatedRequest,
  _res: Response,
  next: NextFunction
) {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader?.startsWith("Bearer ")) {
      return next();
    }

    const user = await resolveUser(authHeader.slice(7));
    if (user) {
      req.userId = user.id;
      req.user = user;
    }
  } catch (error) {
    console.warn("Optional auth middleware error:", error);
  }

  return next();
}
