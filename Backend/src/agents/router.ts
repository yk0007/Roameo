import type { WsEvent, TripContext } from "../types/schemas.js";
import { GeminiClient } from "../tools/gemini.js";
import { randomUUID } from "crypto";

export async function routerAgent(message: string): Promise<"planner" | "chat"> {
  const plannerKeywords = ["plan", "trip", "itinerary", "go to", "travel"];
  const lowerMessage = message.toLowerCase();
  if (plannerKeywords.some((k) => lowerMessage.includes(k))) {
    return "planner";
  }
  return "chat";
}
