import { GeminiClient } from "../tools/gemini.js";

export async function generateSessionTitle(input: {
  message?: string;
  origin?: string;
  destination?: string;
  days?: number;
  existingTitle?: string;
}): Promise<string> {
  const { message = "", origin, destination, days, existingTitle } = input || {};

  // Heuristic fallback first, in case LLM is unavailable
  const fallback = (() => {
    const o = origin ? `${origin}` : "";
    const d = destination ? `${destination}` : "Trip";
    const base = o && destination ? `${o} → ${d}` : d;
    return typeof days === "number" && days > 0 ? `${base}, ${days} day${days > 1 ? "s" : ""}` : base;
  })();

  try {
    const gemini = new GeminiClient({ model: "flash" });
    const prompt = `You create concise, highly-informative chat session titles for a travel planning conversation.

Rules:
- Output ONLY the title text. No quotes, no punctuation wrappers, no extra words.
- Keep it under 32 characters.
- Prefer: "+" or "→" as a separator between origin and destination when both are present.
- Include duration if clearly specified (e.g., ", 4 days").
- Use common city names. Avoid adjectives unless essential to the user's intent.
- If origin is unknown, omit it.
- If neither origin nor destination is known, infer from message intent (e.g., "Japan autumn plan").

Context:
- Existing title (may be generic): ${existingTitle || "(none)"}
- Origin: ${origin || "(unknown)"}
- Destination: ${destination || "(unknown)"}
- Days: ${typeof days === "number" ? days : "(unknown)"}
- Latest user message: ${message}

Return ONLY the title.`;

    const raw = await gemini.chat(prompt, "flash");
    const title = (raw || "").trim().replace(/^\s*["'`]|["'`]\s*$/g, "");

    // Basic sanity checks
    if (!title || title.startsWith("[gemini:")) return fallback;
    if (title.length > 64) return title.slice(0, 64).trim();
    return title;
  } catch {
    return fallback;
  }
}
