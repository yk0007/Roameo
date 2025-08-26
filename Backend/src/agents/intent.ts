import { GeminiClient } from "../tools/gemini.js";

export type Intent = "PLAN_TRIP" | "DESTINATION_SEARCH" | "CHAT";

export async function intentAgent(message: string): Promise<Intent> {
  const gemini = new GeminiClient({ model: "flash" });
  const prompt = `
    You are an intent detection agent for a travel planning chatbot.
    Your goal is to classify the user's message into one of three categories:
    
    1. **PLAN_TRIP**: The user wants to plan a complete trip with specific details like duration, or is asking for a full itinerary. This includes messages that mention both destination AND duration/days.
        Examples:
        - "plan a 3 day trip to coonoor"
        - "create 5-day itinerary for Japan"
        - "I want a week-long vacation in Kerala"
        - "plan my 4 day ooty trip"

    2. **DESTINATION_SEARCH**: The user mentions a destination or multiple destinations but doesn't specify duration or ask for a complete itinerary. This should trigger immediate POI search.
        Examples:
        - "show me places in ooty"
        - "what to do in coonoor"
        - "ooty attractions"
        - "find hotels in mumbai"
        - "restaurants in goa"
        - "I want to visit kerala"
        - "tell me about places to visit in rajasthan"
        - "what are the top attractions in delhi"

    3. **CHAT**: General conversation, questions about the bot, or travel-related knowledge questions without specific destinations.
        Examples:
        - "who are you?"
        - "what is your name?"
        - "how do you create a travel plan?"
        - "that sounds great"
        - "What is the best time of year to visit?"
        - "how does this work?"

    User message: "${message}"

    IMPORTANT: If a destination is mentioned (even without duration), prefer DESTINATION_SEARCH over CHAT to provide immediate value.

    Based on the message, what is the user's intent?
    Respond with only "PLAN_TRIP", "DESTINATION_SEARCH", or "CHAT".
  `;

  const response = await gemini.chat(prompt);
  const intent = response.trim() as Intent;

  if (intent === "PLAN_TRIP" || intent === "DESTINATION_SEARCH") {
    return intent;
  }
  return "CHAT";
}
