import { BACKEND_URL, type SessionId, type InviteId, type TripContext, type WsEvent } from "./types"

export async function sendChat({ sessionId, inviteId, message }: { sessionId?: SessionId; inviteId?: InviteId; message: string }) {
  const res = await fetch(`${BACKEND_URL}/api/chat/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, inviteId, message }),
  })
  if (!res.ok) throw new Error(`sendChat failed: ${res.status}`)
  return (await res.json()) as { sessionId: SessionId; inviteId?: InviteId; created?: boolean; events?: WsEvent[] }
}

export async function tripUpdate(sessionId: SessionId, patch: Partial<TripContext>) {
  const res = await fetch(`${BACKEND_URL}/api/trip/update`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, patch }),
  })
  if (!res.ok) throw new Error(`tripUpdate failed: ${res.status}`)
  return (await res.json()) as { ok: true }
}

export async function createInvite(sessionId: SessionId) {
  const res = await fetch(`${BACKEND_URL}/api/invite/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId }),
  })
  if (!res.ok) throw new Error(`createInvite failed: ${res.status}`)
  return (await res.json()) as { inviteId: InviteId }
}

export async function savePoi(sessionId: SessionId, poiId: string, saved: boolean) {
  const res = await fetch(`${BACKEND_URL}/api/poi/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, poiId, saved }),
  })
  if (!res.ok) throw new Error(`savePoi failed: ${res.status}`)
  return (await res.json()) as { ok: true; saved: boolean }
}

export async function clearChat(sessionId: SessionId) {
  const res = await fetch(`${BACKEND_URL}/api/chat/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId }),
  })
  if (!res.ok) throw new Error(`clearChat failed: ${res.status}`)
  return (await res.json()) as { ok: true }
}

export async function deleteTrip(sessionId: SessionId) {
  // Send sessionId via query string to align with backend router reading req.query
  const res = await fetch(`${BACKEND_URL}/api/trip?sessionId=${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  })
  if (!res.ok) throw new Error(`deleteTrip failed: ${res.status}`)
  return (await res.json()) as { ok: true }
}

export async function getSavedPoiIds(sessionId: SessionId) {
  const res = await fetch(`${BACKEND_URL}/api/session/saved?sessionId=${encodeURIComponent(sessionId)}`, {
    method: "GET",
  })
  if (!res.ok) throw new Error(`getSavedPoiIds failed: ${res.status}`)
  return (await res.json()) as { ids: string[] }
}

export async function listTrips() {
  const res = await fetch(`${BACKEND_URL}/api/trips/list`, { cache: "no-store" })
  if (!res.ok) throw new Error(`listTrips failed: ${res.status}`)
  return (await res.json()) as { trips: Array<{ id: string; title: string; destination?: string | null; duration?: string | null; travelers?: number | null; image?: string | null; updatedAt?: string | null }> }
}
