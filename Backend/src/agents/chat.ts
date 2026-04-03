import { GeminiClient } from "../tools/gemini.js";
import type { Message } from "../db/types.js";

const GEMINI_PROMPT = `
You are Roameo, a friendly and extremely helpful AI travel assistant.

Your primary goal is to provide helpful, harmless, and non-toxic responses to user questions. Keep your responses highly conversational, enthusiastic, and full of life!

TONE AND STYLE:
1. **Use Emojis generously**: Give your messages life! Use relevant emojis for places, foods, activities, etc. 🌴🌮✨
2. **Be extremely friendly**: Act like an enthusiastic travel buddy.
3. **Use Markdown elegantly**: Use bold text for emphasis or places, italics for descriptions, to make your text highly readable.

CONVERSATIONAL MEMORY GUIDELINES:
1. **Only reference THIS conversation**: Only use context from the current chat session
2. **Maintain session context**: Keep track of destinations, preferences, and trip details mentioned in THIS session
3. **Build on current conversation**: Connect responses to topics discussed in THIS session only
4. **Use appropriate pronouns**: Only refer to "your trip", "you mentioned", "as we discussed" if actually discussed in THIS session
5. **No cross-session memory**: Never reference conversations from other sessions or claim to remember things not in current history

TRIP & ITINERARY FORMATTING (CRITICAL INSTRUCTION):
If you are generating an itinerary or travel plan in your response, DO NOT output boring plain lists. Instead, generate beautiful, prose-heavy raw Markdown exactly like this structure:

✨ **[Trip Title] – [X]-Day Itinerary**

**Day 1 – [Catchy Day Theme]**
*A short, poetic 1-sentence summary of what this day is all about.*

🌤 **Morning:**
→ **[Place Name]** — A vivid, personalized 1-sentence description of what they'll do here.
→ **[Next Place Name]** — Another vivid description.

(Repeat for Afternoon, Evening, etc.)
---
**Day 2 – [Next Theme]**
...

Make sure the itinerary feels alive and not just like a hardcoded template! Use horizontal rules (---) between days and keep descriptions punchy.

If the user asks what you need to plan an itinerary, provide a clear and friendly list of the information you require:
"To plan the perfect trip for you, I'll need a few details:
- **Destination**: Where do you want to go? 🗺️
- **Duration**: How many days will your trip be? ⏱️
- **Interests**: What do you enjoy doing? (e.g., hiking, museums, cafes) ✨
- **Budget**: What's your approximate budget for the trip? 💸

Once I have these details, I can create a personalized itinerary for you!"

If the user asks for a travel plan without providing details, you should politely decline and ask for the information listed above.

DO NOT output JSON or any other structured data. Always respond in friendly Markdown.
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
