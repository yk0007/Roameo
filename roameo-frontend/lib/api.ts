import type {
  CreateSessionInput,
  PlanMutationInput,
  SendMessageInput,
  SessionMutation
} from "@roameo/contracts";
import { supabase } from "./supabase/client";
import { BACKEND_URL, type CanonicalSession, type CanonicalSessionSummary, type SessionSettingsPayload } from "./types";

async function buildHeaders() {
  const {
    data: { session }
  } = await supabase.auth.getSession();

  const headers: HeadersInit = {
    "Content-Type": "application/json"
  };

  if (session?.access_token) {
    headers.Authorization = `Bearer ${session.access_token}`;
  }

  return headers;
}

async function apiFetch<T>(path: string, init: RequestInit = {}) {
  const headers = await buildHeaders();
  const response = await fetch(`${BACKEND_URL}${path}`, {
    ...init,
    headers: {
      ...headers,
      ...(init.headers || {})
    }
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function createSession(input: CreateSessionInput = {}) {
  return apiFetch<CanonicalSession>("/api/sessions", {
    method: "POST",
    body: JSON.stringify(input)
  });
}

export async function getSession(sessionId: string) {
  return apiFetch<CanonicalSession>(`/api/sessions/${encodeURIComponent(sessionId)}`);
}

export async function updateSession(sessionId: string, mutation: SessionMutation) {
  return apiFetch<CanonicalSession>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    body: JSON.stringify(mutation)
  });
}

export async function sendMessage(sessionId: string, input: SendMessageInput) {
  return apiFetch<{ accepted: boolean; sessionId: string }>(
    `/api/sessions/${encodeURIComponent(sessionId)}/messages`,
    {
      method: "POST",
      body: JSON.stringify(input)
    }
  );
}

export async function mutatePlan(
  sessionId: string,
  mutation: PlanMutationInput
) {
  return apiFetch<CanonicalSession>(
    `/api/sessions/${encodeURIComponent(sessionId)}/plan-mutations`,
    {
      method: "POST",
      body: JSON.stringify(mutation)
    }
  );
}

export async function deleteSession(sessionId: string) {
  return apiFetch<void>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE"
  });
}

export async function savePoi(sessionId: string, poiId: string, saved: boolean) {
  return apiFetch<{ ids: string[] }>(
    `/api/sessions/${encodeURIComponent(sessionId)}/saved-pois`,
    {
      method: "POST",
      body: JSON.stringify({ poiId, saved })
    }
  );
}

export async function getSavedPoiIds(sessionId: string) {
  return apiFetch<{ ids: string[] }>(
    `/api/sessions/${encodeURIComponent(sessionId)}/saved-pois`
  );
}

export async function listSessionSummaries() {
  const data = await apiFetch<{ sessions: CanonicalSessionSummary[] }>("/api/sessions");
  return data.sessions;
}

export async function getSessionSettings() {
  return apiFetch<SessionSettingsPayload>("/api/me/settings");
}

export async function updateSessionSettings(payload: Pick<SessionSettingsPayload, "providerSettings" | "preferences">) {
  return apiFetch<Pick<SessionSettingsPayload, "providerSettings" | "preferences">>("/api/me/settings", {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export async function saveProviderCredential(provider: "gemini" | "openai", apiKey: string) {
  return apiFetch<void>(`/api/me/credentials/${provider}`, {
    method: "PUT",
    body: JSON.stringify({ apiKey })
  });
}

export async function sendChat({
  sessionId,
  message,
  providerSettings
}: {
  sessionId?: string;
  message: string;
  providerSettings?: CreateSessionInput["providerSettings"];
}) {
  if (sessionId) {
    await sendMessage(sessionId, { content: message, providerSettings });
    return { sessionId };
  }

  const session = await createSession({
    initialMessage: message,
    providerSettings
  });

  return { sessionId: session.id };
}

export async function deleteTrip(sessionId: string) {
  await deleteSession(sessionId);
  return { ok: true as const };
}

export async function tripUpdate(sessionId: string, mutation: SessionMutation) {
  return updateSession(sessionId, mutation);
}

export async function listTrips() {
  const sessions = await listSessionSummaries();
  return {
    trips: sessions.map((session) => ({
      id: session.id,
      title: session.title,
      destination: session.destination,
      days: session.totalDays,
      providerSettings: session.providerSettings,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt
    }))
  };
}
