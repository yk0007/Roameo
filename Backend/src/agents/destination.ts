import type { WsEvent } from "../types/schemas.js";
import { GeminiClient } from "../tools/gemini.js";
import { poiAgent } from "./poi.js";
import { mapAgent } from "./map.js";

/**
 * Destination Extraction Agent
 * Immediately extracts destination(s) from user messages and triggers POI search
 * This allows for instant POI results without waiting for full itinerary generation
 */

export interface DestinationExtractionResult {
  destination?: string;
  destinations?: string[];
  days?: number;
  origin?: string;
  hasDestination: boolean;
}

export async function destinationExtractionAgent(message: string): Promise<DestinationExtractionResult> {
  const gemini = new GeminiClient({ model: "flash" });
  
  const prompt = `You are a destination extraction agent. Extract travel destinations and trip duration from user messages.

User message: "${message}"

CRITICAL RULES:
1. Extract EXACT destination names as mentioned by the user
2. If user says "ooty", extract "Ooty" (capitalize properly)
3. If user says "coonoor", extract "Coonoor"
4. Do NOT substitute with other destinations
5. For multiple destinations, use "destinations" array
6. For single destination, use "destination" field
7. Extract days if mentioned (e.g., "3 days", "for a week" = 7 days)
8. Extract origin if mentioned (e.g., "from Mumbai")

Examples:
- "plan a trip to ooty" → {"destination": "Ooty", "hasDestination": true}
- "3 day trip to ooty and coonoor" → {"destinations": ["Ooty", "Coonoor"], "days": 3, "hasDestination": true}
- "kerala, goa and rajasthan for 10 days" → {"destinations": ["Kerala", "Goa", "Rajasthan"], "days": 10, "hasDestination": true}
- "from mumbai to goa" → {"origin": "Mumbai", "destination": "Goa", "hasDestination": true}
- "what's the weather like?" → {"hasDestination": false}

Respond with ONLY a JSON object with "destination", "destinations", "days", "origin", and "hasDestination" keys.
If a field is not present, omit it (except hasDestination which is always required).`;

  try {
    const response = await gemini.chat(prompt);
    const cleanedJson = response.replace(/^```json\s*/i, "").replace(/\s*```\s*$/i, "").trim();
    const jsonMatch = cleanedJson.match(/\{[\s\S]*\}/);

    if (jsonMatch) {
      const result = JSON.parse(jsonMatch[0]) as DestinationExtractionResult;
      console.log(`[destination] Extracted:`, result);
      return result;
    }
  } catch (error) {
    console.warn("[destination] Failed to parse extraction result:", error);
  }

  // Fallback to simple heuristic extraction
  return heuristicDestinationExtraction(message);
}

/**
 * Enhanced Chat Response Generator
 * Uses AI to generate rich conversational responses about destinations with POI formatting
 */
export async function generateDestinationChatResponse(
  extraction: DestinationExtractionResult,
  userMessage: string,
  pois?: { stays: any[]; restaurants: any[]; attractions: any[] }
): Promise<string> {
  const gemini = new GeminiClient({ model: "flash" });
  
  const destinations = extraction.destinations || (extraction.destination ? [extraction.destination] : []);
  
  if (destinations.length === 0) {
    return "I'll help you discover some amazing places! Could you tell me which destination you'd like to explore?";
  }
  
  // Create POI context if available
  let poiContext = "";
  if (pois && (pois.stays.length > 0 || pois.restaurants.length > 0 || pois.attractions.length > 0)) {
    const topStays = pois.stays.slice(0, 3).map(poi => poi.name);
    const topRestaurants = pois.restaurants.slice(0, 3).map(poi => poi.name);
    const topAttractions = pois.attractions.slice(0, 3).map(poi => poi.name);
    
    poiContext = `
Found POIs:
Top Stays: ${topStays.join(", ")}
Top Restaurants: ${topRestaurants.join(", ")}
Top Attractions: ${topAttractions.join(", ")}`;
  }
  
  const prompt = `You are a friendly travel assistant. Generate an engaging conversational response about searching for places in the destination(s).

User asked: "${userMessage}"
Destination(s): ${destinations.join(", ")}
${poiContext}

IMPORTANT FORMATTING RULES:
1. Make ALL POI names (hotels, restaurants, attractions) **bold** using markdown **text** syntax
2. Make ALL destination names **bold** using markdown **text** syntax  
3. Use EXACT POI names from the context when available
4. Keep the response conversational, enthusiastic, and helpful
5. Mention that you're searching and will show results
6. If POIs are provided, briefly mention some highlights
7. Keep response concise (2-3 sentences max)
8. DO NOT use HTML spans or data attributes - only use **bold** markdown

Examples:
- For "show me places in ooty": "Great choice! I'm searching for amazing places in **Ooty** and will show you the best options. You'll find excellent stays like **Sterling Ooty Elk Hill** and attractions like **Government Rose Garden**!"
- For "restaurants in mumbai": "Perfect! I'm finding the best restaurants in **Mumbai** for you. I'll show you top-rated places including **Trishna** and **Bombay Canteen**!"

Generate an engaging response now:`;
  
  try {
    const response = await gemini.chat(prompt);
    return response.trim();
  } catch (error) {
    console.warn("[destination] Failed to generate AI chat response:", error);
    // Fallback response with simple bold formatting
    if (destinations.length === 1) {
      return `Great choice! I'm searching for amazing places in **${destinations[0]}** and will show you the best stays, restaurants, and attractions!`;
    } else {
      return `Excellent! I'm searching for fantastic places across **${destinations.slice(0, -1).join(", ")} and ${destinations[destinations.length - 1]}** and will show you the best options!`;
    }
  }
}

/**
 * Immediate POI Search Agent
 * Triggers immediate POI search when destinations are detected
 * Returns POI search results and map data without waiting for itinerary
 */
export async function immediatePoiSearchAgent(
  extraction: DestinationExtractionResult
): Promise<WsEvent[]> {
  if (!extraction.hasDestination) {
    return [];
  }

  const events: WsEvent[] = [];
  
  try {
    // Determine which destinations to search
    const destinations = extraction.destinations || (extraction.destination ? [extraction.destination] : []);
    
    if (destinations.length === 0) {
      return events;
    }

    // Use the first destination for initial POI search
    // Later we can enhance this to search multiple destinations
    const primaryDestination = destinations[0];
    
    console.log(`[immediatePoiSearch] Searching POIs for: ${primaryDestination}`);
    
    // Trigger POI search immediately
    const poiEvent = await poiAgent({ destination: primaryDestination });
    
    if (poiEvent && poiEvent.type === "search.results") {
      events.push(poiEvent);
      
      // Also generate map data with the POIs
      const allPois = [
        ...poiEvent.data.stays,
        ...poiEvent.data.restaurants,
        ...poiEvent.data.attractions
      ];
      
      if (allPois.length > 0) {
        const mapEvent = await mapAgent(allPois);
        events.push(mapEvent);
      }
      
      console.log(`[immediatePoiSearch] Generated ${events.length} events for ${primaryDestination}`);
    }
    
  } catch (error) {
    console.error("[immediatePoiSearch] Error during POI search:", error);
  }
  
  return events;
}

/**
 * Fallback heuristic extraction when AI parsing fails
 */
function heuristicDestinationExtraction(message: string): DestinationExtractionResult {
  const text = message.toLowerCase().trim();
  
  // Check for common destination patterns
  const toPattern = /(?:trip\s+)?to\s+([a-zA-Z][\w\s,.-]{2,})/i;
  const fromToPattern = /from\s+([a-zA-Z][\w\s,.-]{2,})\s+to\s+([a-zA-Z][\w\s,.-]{2,})/i;
  const visitPattern = /(?:visit|explore|go\s+to)\s+([a-zA-Z][\w\s,.-]{2,})/i;
  
  // Extract days
  const daysPattern = /(?:for\s+)?(\d{1,2})\s+days?/i;
  const weekPattern = /(?:for\s+)?(?:a\s+)?week/i;
  
  let destination: string | undefined;
  let origin: string | undefined;
  let days: number | undefined;
  
  // Check for from-to pattern
  const fromToMatch = text.match(fromToPattern);
  if (fromToMatch) {
    origin = fromToMatch[1].trim();
    destination = fromToMatch[2].trim();
  } else {
    // Check for simple "to" pattern
    const toMatch = text.match(toPattern) || text.match(visitPattern);
    if (toMatch) {
      destination = toMatch[1].trim();
    }
  }
  
  // Extract days
  const daysMatch = text.match(daysPattern);
  if (daysMatch) {
    days = parseInt(daysMatch[1], 10);
  } else if (text.match(weekPattern)) {
    days = 7;
  }
  
  // Clean up destination name
  if (destination) {
    destination = destination
      .replace(/\b(for|days?|people|travelers?)\b/gi, '')
      .replace(/[.,;:]+$/g, '')
      .trim();
    
    // Capitalize first letter of each word
    destination = destination
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }
  
  const hasDestination = Boolean(destination);
  
  const result: DestinationExtractionResult = { hasDestination };
  if (destination) result.destination = destination;
  if (origin) result.origin = origin;
  if (days) result.days = days;
  
  console.log(`[destination] Heuristic extraction:`, result);
  return result;
}