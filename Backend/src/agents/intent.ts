import { GeminiClient } from "../tools/gemini.js";

export type Intent = "PLAN_TRIP" | "CHAT";

export async function intentAgent(message: string): Promise<Intent> {
  const gemini = new GeminiClient({ model: "flash" });
  const prompt = `
    You are an intent detection agent for a travel planning chatbot.
    Your goal is to classify the user's message into one of two categories:
    1.  **PLAN_TRIP**: The user wants to plan a new trip, modify an existing one, or is asking for travel-related information that requires generating a *specific, personalized* itinerary. This is for when the user provides actionable details like a destination, duration, or specific interests for a trip. It should not be used for general knowledge questions about a location.
        Examples:
        - "plan a 3 day trip to coonoor"
        - "find me a hotel in paris"
        - "show me what to do in rome for a week"
        - "can you suggest a 5-day itinerary for Japan?"

    2.  **CHAT**: The user is having a general conversation, asking a general knowledge question (even if travel-related), or making a statement that is not a direct travel planning request. This includes questions about *how* to plan a trip or general questions about a destination.
        Examples:
        - "who are you?"
        - "what is your name?"
        - "what do you need to plan an itinerary?"
        - "how do you create a travel plan?"
        - "that sounds great"
        - "What is the best time of year to visit Ooty?"

    User message: "${message}"

    Based on the message, what is the user's intent?
    Respond with only "PLAN_TRIP" or "CHAT".
  `;

  const response = await gemini.chat(prompt);

  if (response.trim() === "PLAN_TRIP") {
    return "PLAN_TRIP";
  }
  return "CHAT";
}
