import { GeminiClient } from "../tools/gemini.js";

export async function generateSessionTitle(input: {
  message?: string;
  origin?: string;
  destination?: string;
  days?: number;
  existingTitle?: string;
}): Promise<string> {
  const { message = "", origin, destination, days, existingTitle } = input || {};

  // Generate unique session ID suffix for uniqueness
  const sessionSuffix = Math.random().toString(36).substring(2, 5).toUpperCase();
  
  // Creative fallback with emojis and unique identifiers
  const fallback = (() => {
    const emojis = ["🌟", "✨", "🎯", "🚀", "💫", "🔥", "⚡", "🎨", "🌈", "🎭"];
    const randomEmoji = emojis[Math.floor(Math.random() * emojis.length)];
    
    if (destination && origin) {
      return `${randomEmoji} ${origin} → ${destination} #${sessionSuffix}`;
    } else if (destination) {
      return `${randomEmoji} ${destination} Adventure #${sessionSuffix}`;
    } else {
      return `${randomEmoji} Dream Trip #${sessionSuffix}`;
    }
  })();

  try {
    const gemini = new GeminiClient({ model: "flash" });
    const prompt = `Create a unique, catchy travel session title that's memorable and fun.

Rules:
- Output ONLY the title text. No quotes or extra words.
- Keep it under 28 characters total.
- Make it unique and interesting - use creative themes, emojis, or travel vibes.
- Include the session ID suffix: #${sessionSuffix}
- Examples: "🏝️ Bali Bliss #A7X", "🗼 Tokyo Tales #B2M", "🌊 Coastal Quest #R9K"

Context:
- Origin: ${origin || "unknown"}
- Destination: ${destination || "unknown"}
- Days: ${typeof days === "number" ? days : "unknown"}
- User message: ${message}

Create a title that captures the travel spirit and destination vibe with the #${sessionSuffix} suffix.`;

    const raw = await gemini.chat(prompt, "flash");
    const title = (raw || "").trim().replace(/^\s*["'`]|["'`]\s*$/g, "");

    // Basic sanity checks - reject any error responses
    if (!title || title.startsWith("[gemini:") || title.includes("error") || title.includes("503")) {
      console.log(`[title] Gemini returned invalid response: ${title}, using fallback`);
      return fallback;
    }
    if (title.length > 64) return title.slice(0, 64).trim();
    return title;
  } catch (error: any) {
    console.log(`[title] Title generation failed: ${error?.message || error}, using fallback`);
    return fallback;
  }
}
