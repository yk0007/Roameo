import type { TripContext, WsEvent } from "../types/schemas.js";
import { GeminiClient } from "../tools/gemini.js";

// Very light-weight heuristic extractor for MVP
export async function extractorAgent(msg: string): Promise<Partial<TripContext>> {
  const patch: Partial<TripContext> = {};
  const text = (msg || "").trim();
  if (!text) return patch;

  // Heuristic extraction
  const clean = (s: string) => s.replace(/\b(for|days?|people|travellers|travelers)\b/gi, "").replace(/[.,;:]+$/g, "").trim();

  // from X to Y / X to Y
  const fromTo = text.match(/(?:from\s+)?([A-Za-z][\w\s,.-]{2,})\s+to\s+([A-Za-z][\w\s,.-]{2,})/i);
  if (fromTo) {
    const [, from, to] = fromTo;
    if (from) patch.origin = clean(from);
    if (to) patch.destination = clean(to);
  }
  // to Y from X (reverse order)
  if (!patch.origin || !patch.destination) {
    const toFrom = text.match(/to\s+([A-Za-z][\w\s,.-]{2,})\s+from\s+([A-Za-z][\w\s,.-]{2,})/i);
    if (toFrom) {
      const [, to, from] = toFrom;
      patch.destination = patch.destination || clean(to);
      patch.origin = patch.origin || clean(from);
    }
  }
  // "trip to Y" / "to Y"
  if (!patch.destination) {
    const toOnly = text.match(/(?:trip\s+)?to\s+([A-Za-z][\w\s,.-]{2,})/i);
    if (toOnly) patch.destination = clean(toOnly[1]);
  }
  // "from X"
  if (!patch.origin) {
    const fromOnly = text.match(/from\s+([A-Za-z][\w\s,.-]{2,})/i);
    if (fromOnly) patch.origin = clean(fromOnly[1]);
  }

  // days: "for 3 days" or "3 days"
  const days = text.match(/(?:for\s+)?(\d{1,2})\s+day?s?/i);
  if (days) patch.days = parseInt(days[1], 10);

  // travelers: "for 2 people/travelers"
  const trav = text.match(/(?:for\s+)?(\d{1,2})\s+(?:people|travellers|travelers|guests)/i);
  if (trav) patch.travelers = parseInt(trav[1], 10);

  // budget keywords
  const budget = text.match(/\b(budget|mid(?:\s|-)?range|luxury|premium)\b/i);
  if (budget) patch.budget = budget[1].toLowerCase();

  // LLM normalization + short title
  try {
    const gemini = new GeminiClient({ model: "flash" });
    const prompt = `You are extracting trip info from a user message. Return strict JSON with keys: origin, destination, days (number), travelers (number|omit), budget (string|omit), title (<=32 chars, concise, like "Vizag → Coonoor, 4 days").
Message: ${text}
Initial guess: ${JSON.stringify(patch)}
Rules: Prefer interpreting "to <dest> from <origin>" order correctly. Use common city names. Do not include extra words in city names. If you cannot infer a field, omit it.`;
    const raw = await gemini.chat(prompt);
    try {
      const json = JSON.parse(raw.replace(/^[^\{]*\{/s, "{").replace(/\}[^\}]*$/s, "}"));
      if (json.origin && typeof json.origin === "string") patch.origin = json.origin;
      if (json.destination && typeof json.destination === "string") patch.destination = json.destination;
      if (typeof json.days === "number") patch.days = json.days;
      if (typeof json.travelers === "number") patch.travelers = json.travelers;
      if (typeof json.budget === "string") patch.budget = json.budget;
      if (typeof json.title === "string") patch.title = json.title;
    } catch {
      // fallback to heuristic title
    }
  } catch {}

  // title heuristic if LLM didn't set one
  if (!patch.title && (patch.destination || patch.origin)) {
    const o = patch.origin ? `${patch.origin}` : "";
    const d = patch.destination ? `${patch.destination}` : "Trip";
    const base = o && patch.destination ? `${o} → ${d}` : d;
    patch.title = patch.days ? `${base}, ${patch.days} day${patch.days > 1 ? "s" : ""}` : base;
  }
  return patch;
}

export function emitTripPatch(_sessionId: string, patch: Partial<TripContext>): WsEvent {
  return { type: "navbar.update", data: patch };
}
