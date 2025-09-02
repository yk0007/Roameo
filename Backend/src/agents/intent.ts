import { GeminiClient } from "../tools/gemini.js";
import type { Message } from "../db/types.js";

export type Intent = "PLAN_TRIP" | "DESTINATION_SEARCH" | "CHAT";

export async function intentAgent(message: string, history: Message[] = []): Promise<Intent> {
  try {
    const gemini = new GeminiClient({ model: "flash" });
    
    // Extract conversation context for better intent classification
    const conversationContext = extractIntentContext(history);
    
    // Build context-aware prompt
    let contextPrompt = "";
    if (conversationContext.previousDestinations.length > 0) {
      contextPrompt = `\n\nCONVERSATION CONTEXT: User has previously discussed these destinations: ${conversationContext.previousDestinations.join(', ')}. This helps understand references to "there", "that place", or continuing conversations about planning.`;
    }
    if (conversationContext.planningInProgress) {
      contextPrompt += `\nUser appears to be in the middle of trip planning. References to "continue", "also", "next" may indicate continued planning intent.`;
    }
    if (conversationContext.recentIntents.length > 0) {
      contextPrompt += `\nRecent conversation intents: ${conversationContext.recentIntents.join(', ')}.`;
    }
  const prompt = `
    You are an intent detection agent for a travel planning chatbot with conversation memory.
    Your goal is to classify the user's message into one of three categories:${contextPrompt}
    
    1. **PLAN_TRIP**: The user wants to plan a complete trip with specific details like duration, or is asking for a full itinerary. This includes messages that mention both destination AND duration/days, OR explicitly ask for trip planning.
        Examples:
        - "plan a 3 day trip to coonoor"
        - "create 5-day itinerary for Japan"
        - "I want a week-long vacation in Kerala"
        - "plan my 4 day ooty trip"
        - "help me plan a trip to goa"
        - "make an itinerary for mumbai"

    2. **DESTINATION_SEARCH**: The user mentions a destination AND explicitly asks for information about places, attractions, hotels, or restaurants in that destination. Must have BOTH destination mention AND clear information-seeking intent.
        Examples:
        - "show me places in ooty"
        - "what to do in coonoor"
        - "ooty attractions"
        - "find hotels in mumbai"
        - "restaurants in goa"
        - "tell me about places to visit in rajasthan"
        - "what are the top attractions in delhi"
        - "what's good about manali?"
        - "show me goa beaches"
        
        NOT DESTINATION_SEARCH (should be CHAT):
        - "I want to visit kerala" (vague, no specific information request)
        - "tell me about goa" (too general)
        - "is manali good?" (opinion question)
        - "kerala looks nice" (statement, not query)

    3. **CHAT**: General conversation, questions about the bot, travel-related knowledge questions without specific destinations, or responses that don't clearly indicate destination search or planning intent.
        Examples:
        - "who are you?"
        - "what is your name?"
        - "how do you create a travel plan?"
        - "that sounds great"
        - "What is the best time of year to visit?" (without specific destination)
        - "how does this work?"
        - "thank you"
        - "ok"
        - "yes"
        - "no"
        - "tell me more"
        - "what do you recommend?"

    IMPORTANT RULES:
    - **BE CONSERVATIVE**: When in doubt between any categories, prefer CHAT
    - DESTINATION_SEARCH requires BOTH: (1) clear destination mention AND (2) explicit information-seeking language ("show", "what", "find", "tell me about")
    - Vague statements like "I want to visit X" or "X looks nice" should be CHAT, not DESTINATION_SEARCH
    - Use conversation context to understand references like "there", "that place", "continue planning"
    - If there's explicit mention of planning, creating itinerary, or duration, prefer PLAN_TRIP
    - If the message is conversational, opinion-based, or doesn't have clear action intent, use CHAT
    - Use context to resolve ambiguous messages: "plan a trip there" with previous destination context should be PLAN_TRIP
    - "Let's continue" in a planning context should maintain the previous intent
    - Questions about travel in general (not destination-specific) should be CHAT

    User message: "${message}"

    Based on the message, what is the user's intent?
    Respond with only "PLAN_TRIP", "DESTINATION_SEARCH", or "CHAT".
  `;

  const response = await gemini.chat(prompt);
  const intent = response.trim() as Intent;

  if (intent === "PLAN_TRIP" || intent === "DESTINATION_SEARCH") {
    return intent;
  }
  return "CHAT";
} catch (error: any) {
  console.warn("[intent] Gemini failed, using heuristic fallback:", error.message);
  
  // Fallback: Use simple heuristic pattern matching when Gemini is unavailable
  const lowerMessage = message.toLowerCase();
  
  // Check for strong planning keywords
  const planningKeywords = ["plan", "planning", "create itinerary", "make itinerary", "trip plan"];
  const hasPlanningKeyword = planningKeywords.some(keyword => lowerMessage.includes(keyword));
  
  // Check for destination + planning pattern  
  const hasDestination = /\b(to|visit|visiting|in)\s+[a-zA-Z]{2,}/i.test(message);
  
  // Strong indicators for PLAN_TRIP
  if (hasPlanningKeyword && hasDestination) {
    console.log(`[intent] Heuristic classified as PLAN_TRIP: planning=${hasPlanningKeyword}, destination=${hasDestination}`);
    return "PLAN_TRIP";
  }
  
  // Check for destination search patterns
  const searchKeywords = ["show me", "what to do", "places in", "attractions", "restaurants", "hotels", "find"];
  const hasSearchKeyword = searchKeywords.some(keyword => lowerMessage.includes(keyword));
  
  if (hasSearchKeyword && hasDestination) {
    console.log(`[intent] Heuristic classified as DESTINATION_SEARCH: search=${hasSearchKeyword}, destination=${hasDestination}`);
    return "DESTINATION_SEARCH";
  }
  
  console.log(`[intent] Heuristic fallback to CHAT`);
  return "CHAT";
}
}

/**
 * Extract conversation context for better intent classification
 */
function extractIntentContext(history: Message[]) {
  const context = {
    previousDestinations: new Set<string>(),
    planningInProgress: false,
    recentIntents: [] as string[]
  };
  
  if (!history || history.length === 0) {
    return {
      previousDestinations: [],
      planningInProgress: false,
      recentIntents: []
    };
  }
  
  // Analyze recent conversation history (last 8 messages)
  const recentHistory = history.slice(-8);
  
  recentHistory.forEach((msg, index) => {
    const content = msg.content.toLowerCase();
    
    // Extract destinations mentioned
    const destPatterns = [
      /(?:to|visit|visiting|in|plan.*trip.*to)\s+([A-Za-z][A-Za-z\s]{2,20}?)(?:\s|$|[,.!?])/g,
      /([A-Za-z][A-Za-z\s]{2,20}?)\s+(?:trip|travel|vacation|itinerary)/g
    ];
    
    destPatterns.forEach(pattern => {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const dest = match[1].trim();
        if (dest.length > 2 && dest.length < 20 && !isCommonWord(dest)) {
          context.previousDestinations.add(dest.charAt(0).toUpperCase() + dest.slice(1));
        }
      }
    });
    
    // Check for planning indicators
    const planningKeywords = ['plan', 'itinerary', 'trip', 'vacation', 'days', 'create', 'schedule'];
    if (planningKeywords.some(keyword => content.includes(keyword))) {
      context.planningInProgress = true;
    }
    
    // Infer recent intents based on content
    if (content.includes('plan') || content.includes('itinerary') || /\d+\s*days?/.test(content)) {
      if (index >= recentHistory.length - 3) { // Only count very recent
        context.recentIntents.push('PLAN_TRIP');
      }
    } else if (content.includes('places') || content.includes('attractions') || content.includes('restaurants')) {
      if (index >= recentHistory.length - 3) {
        context.recentIntents.push('DESTINATION_SEARCH');
      }
    }
  });
  
  return {
    previousDestinations: Array.from(context.previousDestinations),
    planningInProgress: context.planningInProgress,
    recentIntents: [...new Set(context.recentIntents)] // Remove duplicates
  };
}

// Helper function to filter out common words that aren't destinations
function isCommonWord(word: string): boolean {
  const commonWords = [
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
    'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see',
    'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use',
    'trip', 'plan', 'visit', 'travel', 'vacation', 'holiday', 'days', 'time', 'place', 'good',
    'great', 'nice', 'best', 'love', 'like', 'want', 'need', 'help', 'please', 'thank'
  ];
  return commonWords.includes(word.toLowerCase());
}
