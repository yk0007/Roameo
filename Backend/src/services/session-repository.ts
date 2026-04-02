import { randomUUID } from "node:crypto";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type {
  AgentTraceEvent,
  ConversationMessage,
  CreateSessionInput,
  KeySource,
  PlanSnapshot,
  PoiCatalog,
  Provider,
  SessionMemory,
  SessionMutation,
  SessionProviderSettings,
  SessionSnapshot,
  SessionSummary,
  UserPreference
} from "@roameo/contracts";
import {
  sessionMemorySchema,
  sessionProviderSettingsSchema,
  sessionSnapshotSchema,
  userPreferenceSchema
} from "@roameo/contracts";
import { env } from "../config/env.js";

type CredentialRecord = {
  encryptedKey: string;
  keySource: KeySource;
  updatedAt: string;
};

type UserSettingsRecord = {
  providerSettings: SessionProviderSettings;
  preferences: UserPreference;
  credentials: Partial<Record<Provider, CredentialRecord>>;
};

function now(): string {
  return new Date().toISOString();
}

function buildEmptyMemory(): SessionMemory {
  return sessionMemorySchema.parse({});
}

function buildDefaultSettings(): SessionProviderSettings {
  return sessionProviderSettingsSchema.parse({});
}

function buildSessionSnapshot(
  id: string,
  userId?: string,
  input?: CreateSessionInput
): SessionSnapshot {
  return sessionSnapshotSchema.parse({
    id,
    userId,
    title: input?.title || "Untitled trip",
    providerSettings: input?.providerSettings || buildDefaultSettings(),
    memory: buildEmptyMemory(),
    poiCatalog: { version: 1, items: {} },
    messages: [],
    savedPoiIds: [],
    traces: [],
    createdAt: now(),
    updatedAt: now()
  });
}

export class SessionRepository {
  private client: SupabaseClient | null;
  private memorySessions = new Map<string, SessionSnapshot>();
  private userSettings = new Map<string, UserSettingsRecord>();

  constructor() {
    if (env.SUPABASE_URL && env.SUPABASE_SERVICE_ROLE_KEY) {
      this.client = createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_ROLE_KEY, {
        auth: { persistSession: false }
      });
    } else {
      this.client = null;
    }
  }

  async createSession(
    userId: string | undefined,
    input: CreateSessionInput = {}
  ): Promise<SessionSnapshot> {
    const id = randomUUID();
    const snapshot = buildSessionSnapshot(id, userId, input);
    this.memorySessions.set(id, snapshot);

    if (this.client) {
      await this.client.from("travel_sessions").insert({
        id,
        user_id: userId,
        title: snapshot.title,
        provider_settings: snapshot.providerSettings,
        memory: snapshot.memory,
        created_at: snapshot.createdAt,
        updated_at: snapshot.updatedAt
      });
    }

    return snapshot;
  }

  async listSessions(userId: string): Promise<SessionSummary[]> {
    if (!this.client) {
      return Array.from(this.memorySessions.values())
        .filter((session) => session.userId === userId)
        .map((session) => ({
          id: session.id,
          title: session.title,
          destination: session.plan?.destination,
          totalDays: session.plan?.totalDays,
          providerSettings: session.providerSettings,
          updatedAt: session.updatedAt,
          createdAt: session.createdAt
        }))
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    }

    const { data, error } = await this.client
      .from("travel_sessions")
      .select("id,title,destination_summary,total_days,provider_settings,updated_at,created_at")
      .eq("user_id", userId)
      .order("updated_at", { ascending: false });

    if (error) {
      throw error;
    }

    return (data || []).map((row: any) => ({
      id: row.id,
      title: row.title,
      destination: row.destination_summary || undefined,
      totalDays: row.total_days || undefined,
      providerSettings: sessionProviderSettingsSchema.parse(row.provider_settings || {}),
      updatedAt: row.updated_at,
      createdAt: row.created_at
    }));
  }

  async getSession(
    sessionId: string,
    userId?: string
  ): Promise<SessionSnapshot | null> {
    if (!this.client) {
      const snapshot = this.memorySessions.get(sessionId);
      if (!snapshot) {
        return null;
      }
      if (userId && snapshot.userId && snapshot.userId !== userId) {
        return null;
      }
      return snapshot;
    }

    const { data: sessionRow, error: sessionError } = await this.client
      .from("travel_sessions")
      .select("*")
      .eq("id", sessionId)
      .maybeSingle();

    if (sessionError) {
      throw sessionError;
    }
    if (!sessionRow) {
      return null;
    }
    if (userId && sessionRow.user_id && sessionRow.user_id !== userId) {
      return null;
    }

    const [messageRows, planRows, poiRows, savedRows, traceRows] = await Promise.all([
      this.client
        .from("session_messages")
        .select("*")
        .eq("session_id", sessionId)
        .order("created_at", { ascending: true }),
      this.client
        .from("session_plan_snapshots")
        .select("*")
        .eq("session_id", sessionId)
        .order("version", { ascending: false })
        .limit(1),
      this.client
        .from("session_poi_catalogs")
        .select("*")
        .eq("session_id", sessionId)
        .maybeSingle(),
      this.client
        .from("session_saved_pois")
        .select("poi_id")
        .eq("session_id", sessionId),
      this.client
        .from("session_agent_traces")
        .select("*")
        .eq("session_id", sessionId)
        .order("created_at", { ascending: true })
    ]);

    const snapshot = sessionSnapshotSchema.parse({
      id: sessionRow.id,
      userId: sessionRow.user_id || undefined,
      title: sessionRow.title,
      providerSettings: sessionRow.provider_settings || {},
      memory: sessionRow.memory || {},
      plan: planRows.data?.[0]?.snapshot || undefined,
      poiCatalog: poiRows.data?.catalog || { version: 1, items: {} },
      messages: (messageRows.data || []).map((row: any) => ({
        id: row.id,
        sessionId: row.session_id,
        role: row.role,
        content: row.content,
        createdAt: row.created_at,
        phase: row.phase || undefined,
        meta: row.meta || {}
      })),
      savedPoiIds: (savedRows.data || []).map((row: any) => row.poi_id),
      traces: (traceRows.data || []).map((row: any) => ({
        id: row.id,
        sessionId: row.session_id,
        turnId: row.turn_id,
        agent: row.agent,
        status: row.status,
        label: row.label,
        detail: row.detail || undefined,
        createdAt: row.created_at
      })),
      createdAt: sessionRow.created_at,
      updatedAt: sessionRow.updated_at
    });

    this.memorySessions.set(sessionId, snapshot);
    return snapshot;
  }

  async saveMessage(message: ConversationMessage): Promise<void> {
    const snapshot = this.memorySessions.get(message.sessionId);
    if (snapshot) {
      snapshot.messages = [...snapshot.messages, message];
      snapshot.updatedAt = now();
    }

    if (!this.client) {
      return;
    }

    await this.client.from("session_messages").insert({
      id: message.id,
      session_id: message.sessionId,
      role: message.role,
      content: message.content,
      phase: message.phase || null,
      meta: message.meta,
      created_at: message.createdAt
    });

    await this.touchSession(message.sessionId);
  }

  async saveTrace(trace: AgentTraceEvent): Promise<void> {
    const snapshot = this.memorySessions.get(trace.sessionId);
    if (snapshot) {
      snapshot.traces = [...snapshot.traces, trace];
      snapshot.updatedAt = now();
    }

    if (!this.client) {
      return;
    }

    await this.client.from("session_agent_traces").insert({
      id: trace.id,
      session_id: trace.sessionId,
      turn_id: trace.turnId,
      agent: trace.agent,
      status: trace.status,
      label: trace.label,
      detail: trace.detail || null,
      created_at: trace.createdAt
    });

    await this.touchSession(trace.sessionId);
  }

  async savePlan(
    sessionId: string,
    plan: PlanSnapshot,
    poiCatalog: PoiCatalog
  ): Promise<void> {
    const snapshot = this.memorySessions.get(sessionId);
    if (snapshot) {
      snapshot.plan = plan;
      snapshot.poiCatalog = poiCatalog;
      snapshot.title = plan.title;
      snapshot.updatedAt = now();
    }

    if (!this.client) {
      return;
    }

    await this.client.from("session_plan_snapshots").insert({
      id: randomUUID(),
      session_id: sessionId,
      version: plan.version,
      snapshot: plan,
      created_at: now()
    });

    await this.client.from("session_poi_catalogs").upsert(
      {
        session_id: sessionId,
        catalog: poiCatalog,
        updated_at: now()
      },
      { onConflict: "session_id" }
    );

    await this.client
      .from("travel_sessions")
      .update({
        title: plan.title,
        destination_summary: plan.destination || null,
        total_days: plan.totalDays,
        current_plan_version: plan.version,
        updated_at: now()
      })
      .eq("id", sessionId);
  }

  async updateSession(
    sessionId: string,
    mutation: SessionMutation
  ): Promise<SessionSnapshot | null> {
    const snapshot = this.memorySessions.get(sessionId) || (await this.getSession(sessionId));
    if (!snapshot) {
      return null;
    }

    snapshot.title = mutation.title || snapshot.title;
    snapshot.providerSettings =
      mutation.providerSettings || snapshot.providerSettings;
    if (mutation.memory) {
      snapshot.memory = sessionMemorySchema.parse(mutation.memory);
    }
    if (mutation.preferences) {
      snapshot.memory.preferences = userPreferenceSchema.parse(
        mutation.preferences
      );
    }
    snapshot.updatedAt = now();

    if (this.client) {
      await this.client
        .from("travel_sessions")
        .update({
          title: snapshot.title,
          provider_settings: snapshot.providerSettings,
          memory: snapshot.memory,
          updated_at: snapshot.updatedAt
        })
        .eq("id", sessionId);
    }

    this.memorySessions.set(sessionId, snapshot);
    return snapshot;
  }

  async setSavedPoi(
    sessionId: string,
    poiId: string,
    saved: boolean
  ): Promise<SessionSnapshot | null> {
    const snapshot = this.memorySessions.get(sessionId) || (await this.getSession(sessionId));
    if (!snapshot) {
      return null;
    }

    const next = new Set(snapshot.savedPoiIds);
    if (saved) {
      next.add(poiId);
    } else {
      next.delete(poiId);
    }
    snapshot.savedPoiIds = Array.from(next);
    snapshot.updatedAt = now();
    this.memorySessions.set(sessionId, snapshot);

    if (this.client) {
      if (saved) {
        await this.client.from("session_saved_pois").upsert({
          session_id: sessionId,
          poi_id: poiId,
          created_at: now()
        });
      } else {
        await this.client
          .from("session_saved_pois")
          .delete()
          .eq("session_id", sessionId)
          .eq("poi_id", poiId);
      }
      await this.touchSession(sessionId);
    }

    return snapshot;
  }

  async deleteSession(sessionId: string, userId?: string): Promise<void> {
    const snapshot = await this.getSession(sessionId, userId);
    if (!snapshot) {
      return;
    }

    this.memorySessions.delete(sessionId);

    if (!this.client) {
      return;
    }

    await Promise.all([
      this.client.from("session_messages").delete().eq("session_id", sessionId),
      this.client.from("session_plan_snapshots").delete().eq("session_id", sessionId),
      this.client.from("session_poi_catalogs").delete().eq("session_id", sessionId),
      this.client.from("session_saved_pois").delete().eq("session_id", sessionId),
      this.client.from("session_agent_traces").delete().eq("session_id", sessionId),
      this.client.from("travel_sessions").delete().eq("id", sessionId)
    ]);
  }

  async getUserSettings(userId: string): Promise<UserSettingsRecord> {
    const cached = this.userSettings.get(userId);
    if (cached) {
      return cached;
    }

    if (!this.client) {
      const record = {
        providerSettings: buildDefaultSettings(),
        preferences: userPreferenceSchema.parse({}),
        credentials: {}
      };
      this.userSettings.set(userId, record);
      return record;
    }

    const [settingsRow, credentialRows] = await Promise.all([
      this.client
        .from("user_provider_settings")
        .select("*")
        .eq("user_id", userId)
        .maybeSingle(),
      this.client
        .from("user_provider_credentials")
        .select("*")
        .eq("user_id", userId)
    ]);

    const record: UserSettingsRecord = {
      providerSettings: buildDefaultSettings(),
      preferences: userPreferenceSchema.parse({}),
      credentials: {}
    };

    if (settingsRow.data) {
      record.providerSettings = sessionProviderSettingsSchema.parse(
        settingsRow.data.provider_settings || {}
      );
      record.preferences = userPreferenceSchema.parse(
        settingsRow.data.preferences || {}
      );
    }

    for (const row of credentialRows.data || []) {
      record.credentials[row.provider as Provider] = {
        encryptedKey: row.encrypted_key,
        keySource: row.key_source,
        updatedAt: row.updated_at
      };
    }

    this.userSettings.set(userId, record);
    return record;
  }

  async saveUserSettings(
    userId: string,
    providerSettings: SessionProviderSettings,
    preferences: UserPreference
  ): Promise<UserSettingsRecord> {
    const next: UserSettingsRecord = {
      ...(await this.getUserSettings(userId)),
      providerSettings,
      preferences
    };
    this.userSettings.set(userId, next);

    if (this.client) {
      await this.client.from("user_provider_settings").upsert(
        {
          user_id: userId,
          provider_settings: providerSettings,
          preferences,
          updated_at: now()
        },
        { onConflict: "user_id" }
      );
    }

    return next;
  }

  async saveUserCredential(
    userId: string,
    provider: Provider,
    keySource: KeySource,
    encryptedKey: string
  ): Promise<void> {
    const settings = await this.getUserSettings(userId);
    settings.credentials[provider] = {
      encryptedKey,
      keySource,
      updatedAt: now()
    };
    this.userSettings.set(userId, settings);

    if (this.client) {
      await this.client.from("user_provider_credentials").upsert(
        {
          user_id: userId,
          provider,
          key_source: keySource,
          encrypted_key: encryptedKey,
          updated_at: now()
        },
        { onConflict: "user_id,provider,key_source" }
      );
    }
  }

  private async touchSession(sessionId: string): Promise<void> {
    if (!this.client) {
      return;
    }

    await this.client
      .from("travel_sessions")
      .update({ updated_at: now() })
      .eq("id", sessionId);
  }
}
