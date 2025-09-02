import { GeminiClient } from "../tools/gemini.js";
import type { Message } from "../db/types.js";

const GEMINI_PROMPT = `
You are Roameo, a friendly and helpful AI travel assistant with conversational memory.

Your primary goal is to provide helpful, harmless, and non-toxic responses to user questions while maintaining context from previous conversations. Keep your responses concise and conversational.

CONVERSATIONAL MEMORY GUIDELINES:
1. **Remember previous interactions**: Reference earlier messages when relevant
2. **Maintain context**: Keep track of destinations, preferences, and trip details mentioned
3. **Build on conversations**: Connect current responses to previous topics
4. **Use personal pronouns**: Refer to "your trip", "you mentioned", "as we discussed"
5. **Acknowledge continuity**: Use phrases like "following up on", "as planned", "continuing from earlier"

CONTEXT AWARENESS:
- If user previously mentioned destinations, remember them
- If trip details were discussed, reference them appropriately
- If preferences were shared, incorporate them into responses
- If the user asked questions before, acknowledge their ongoing planning process

If the user asks what you need to plan an itinerary, provide a clear and friendly list of the information you require:
"To plan the perfect trip for you, I'll need a few details:
- **Destination**: Where do you want to go?
- **Duration**: How many days will your trip be?
- **Interests**: What do you enjoy doing? (e.g., hiking, museums, cafes)
- **Budget**: What's your approximate budget for the trip?

Once I have these details, I can create a personalized itinerary for you!"

If the user asks for a travel plan without providing details, you should politely decline and ask for the information listed above.

REMEMBER: Always maintain a conversational tone and reference previous context when appropriate. If this is a new conversation with no history, be welcoming and ready to help plan their travel.

DO NOT output JSON or any other structured data.
`.trim();

export async function chatAgent(message: string, history: Message[]): Promise<string> {
  const gemini = new GeminiClient();
  
  // Enhanced conversation history formatting with context analysis
  let conversationContext = "";
  if (history && history.length > 0) {
    // Get recent conversation history (last 10 messages for context)
    const recentHistory = history.slice(-10);
    
    // Extract key context from conversation
    const destinations = new Set<string>();
    const tripDetails: string[] = [];
    const userPreferences: string[] = [];
    
    // Analyze history for context
    recentHistory.forEach(msg => {
      const content = msg.content.toLowerCase();
      
      // Extract mentioned destinations
      const destPatterns = [
        /(?:to|visit|visiting|in)\s+([A-Za-z][A-Za-z\s]{2,20}?)(?:\s|$|[,.!?])/g,
        /([A-Za-z][A-Za-z\s]{2,20}?)\s+(?:trip|travel|vacation)/g
      ];
      
      destPatterns.forEach(pattern => {
        let match;
        while ((match = pattern.exec(content)) !== null) {
          const dest = match[1].trim();
          if (dest.length > 2 && dest.length < 20) {
            destinations.add(dest.charAt(0).toUpperCase() + dest.slice(1));
          }
        }
      });
      
      // Extract trip details
      if (content.includes('day') && /\d+\s*days?/.test(content)) {
        const dayMatch = content.match(/(\d+)\s*days?/);
        if (dayMatch) tripDetails.push(`${dayMatch[1]} days duration`);
      }
      
      // Extract preferences
      const preferenceKeywords = ['like', 'enjoy', 'prefer', 'interested in', 'love'];
      preferenceKeywords.forEach(keyword => {
        if (content.includes(keyword)) {
          const parts = content.split(keyword);
          if (parts.length > 1) {
            const pref = parts[1].split(/[,.!?]/)[0].trim();
            if (pref.length > 3 && pref.length < 50) {
              userPreferences.push(pref);
            }
          }
        }
      });
    });
    
    // Build context summary
    let contextSummary = "CONVERSATION CONTEXT:\n";
    if (destinations.size > 0) {
      contextSummary += `Destinations discussed: ${Array.from(destinations).join(', ')}\n`;
    }
    if (tripDetails.length > 0) {
      contextSummary += `Trip details: ${tripDetails.join(', ')}\n`;
    }
    if (userPreferences.length > 0) {
      contextSummary += `User preferences: ${userPreferences.slice(0, 3).join(', ')}\n`;
    }
    
    // Add recent message history
    const historyStr = recentHistory
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");
    
    conversationContext = `${contextSummary}\nRECENT CONVERSATION HISTORY:\n${historyStr}\n\n`;
  } else {
    conversationContext = "CONVERSATION CONTEXT: This is the start of a new conversation.\n\n";
  }
  
  const fullPrompt = `${GEMINI_PROMPT}\n\n${conversationContext}Current user message: ${message}\n\nProvide a helpful, contextual response that references previous conversation when relevant:`;
  
  try {
    const chatResponse = await gemini.chat(fullPrompt);
    return chatResponse || "Sorry, I'm having trouble thinking right now.";
  } catch (error: any) {
    console.error('[chat] Error generating response:', error);
    
    if (error.message && error.message.includes("API configuration")) {
      return "I'm experiencing some technical difficulties right now. Please try again in a few minutes!";
    }
    
    return "I'm having a bit of trouble right now. Could you please try again?";
  }
}
