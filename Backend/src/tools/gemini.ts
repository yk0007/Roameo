import { env, config } from "../config/env.js";

export type GeminiModel = "flash" | "pro";

export interface GeminiClientOptions {
  model?: GeminiModel;
}

export class GeminiClient {
  constructor(private opts: GeminiClientOptions = {}) {}

    async chat(prompt: string, model: GeminiModel = this.opts.model || "pro"): Promise<string> {
    if (!env.GEMINI_API_KEY) {
      return "[gemini: no API key configured]";
    }
    const modelId = model === "pro" ? config.models.pro : config.models.flash;
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(modelId)}:generateContent?key=${encodeURIComponent(
      env.GEMINI_API_KEY!
    )}`;

    // Retry logic for 500 errors
    const maxRetries = 3;
    let lastError = "";
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [{ text: prompt }],
            },
          ],
          generationConfig: {
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 8192,
          },
          safetySettings: [
            {
              category: "HARM_CATEGORY_HARASSMENT",
              threshold: "BLOCK_ONLY_HIGH",
            },
            {
              category: "HARM_CATEGORY_HATE_SPEECH",
              threshold: "BLOCK_ONLY_HIGH",
            },
            {
              category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              threshold: "BLOCK_ONLY_HIGH",
            },
            {
              category: "HARM_CATEGORY_DANGEROUS_CONTENT",
              threshold: "BLOCK_ONLY_HIGH",
            },
          ],
        }),
      });
      
      if (!res.ok) {
        const text = await res.text();
        lastError = `[${modelId}] error ${res.status}: ${text.slice(0, 200)}`;
        
        // Retry on 500 errors
        if (res.status === 500 && attempt < maxRetries) {
          console.log(`[gemini] Attempt ${attempt}/${maxRetries} failed with 500 error, retrying in ${attempt * 1000}ms...`);
          await new Promise(resolve => setTimeout(resolve, attempt * 1000));
          continue;
        }
        
        return lastError;
      }
      
      const data: any = await res.json();
      
      // Try multiple extraction paths for different response formats
      let text: string | undefined = data?.candidates?.[0]?.content?.parts?.[0]?.text;
      
      // Fallback: check if content exists but parts is missing
      if (!text && data?.candidates?.[0]?.content) {
        const content = data.candidates[0].content;
        if (typeof content === 'string') {
          text = content;
        } else if (content.parts && Array.isArray(content.parts) && content.parts.length > 0) {
          text = content.parts[0]?.text;
        }
      }

      if (text && text.trim()) {
        return text.trim();
      }

      // If no text, log the full response for debugging and check for safety blocks
      console.error("[gemini] Failed to extract text from response. Full response:", JSON.stringify(data, null, 2));

      if (data?.promptFeedback?.blockReason) {
        return `[gemini] Request was blocked. Reason: ${data.promptFeedback.blockReason}`;
      }

      return "[gemini: empty response]";
    } catch (err: any) {
      lastError = `[${modelId}] request failed: ${String(err?.message || err)}`;
      
      if (attempt < maxRetries) {
        console.log(`[gemini] Attempt ${attempt}/${maxRetries} failed with error: ${lastError}, retrying in ${attempt * 1000}ms...`);
        await new Promise(resolve => setTimeout(resolve, attempt * 1000));
        continue;
      }
    }
    }
    
    return lastError || "[gemini: all retry attempts failed]";
  }
}
