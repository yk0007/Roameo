import { randomUUID } from "crypto";
import type { Itinerary, ChatMessage, WsEvent, POI, Activity } from "../types/schemas.js";
import { GoogleMapsClient } from "../tools/maps.js";
import { GeminiClient } from "../tools/gemini.js";
import { DestinationImageService } from "../tools/destination-images.js";

const TRIP_DETAIL_EXTRACTION_PROMPT = `
You are a travel planning assistant. Extract trip details from the user's message and determine their intent.

IMPORTANT: Analyze the context carefully to determine if the user wants to:
1. "plan" - Create a completely new trip (replace existing)
2. "add" - Add a destination to their existing trip 
3. "remove" - Remove a destination from their existing trip
4. "clarify" - Intent is ambiguous, need clarification

Look for keywords like:
- "add", "also visit", "include", "extend", "then go to" → action: "add"
- "remove", "skip", "cancel", "drop", "don't go to" → action: "remove"  
- "plan", "trip to", "visit", "go to" with clear new trip context → action: "plan"
- Ambiguous cases (e.g., "plan for Mysore" when existing trip exists) → action: "clarify"

CONTEXT CLUES for disambiguation:
- If user says "plan trip to X" and there's an EXISTING TRIP to a different destination → "clarify" 
- If user says "plan for X" and there's an EXISTING TRIP to a different destination → "clarify"
- If user mentions specific days for a new destination → likely "add"
- If user says "instead of" or "change to" → likely "plan" (new trip)
- If no days mentioned for new destination → likely needs clarification
- If user says "add X" but current trip is for Y, ask about X not Y
- IMPORTANT: Always check EXISTING TRIP context before deciding action

Extract the following information:
- action: "plan" | "add" | "remove" | "clarify"
- destination: string (primary destination mentioned)
- destinations: string[] (if multiple destinations mentioned)
- days: number (total trip days or days for this destination)
- origin: string (departure city)
- budget: number (if mentioned)
- travellers: number (if mentioned)
- clarificationNeeded: string (what needs to be clarified, only if action is "clarify")

Return valid JSON only.

Examples:
User: "Plan a 5-day trip to Ooty from Bangalore"
{"action": "plan", "destination": "Ooty", "days": 5, "origin": "Bangalore"}

User: "Add Mysore for 3 days to my trip"  
{"action": "add", "destination": "Mysore", "days": 3}

User: "Remove Mysore from my trip"
{"action": "remove", "destination": "Mysore"}

User: "Plan trip to Mysore" (when existing trip to Ooty exists)
{"action": "clarify", "destination": "Mysore", "clarificationNeeded": "Do you want to add Mysore to your existing Ooty trip, or plan a completely new trip to Mysore instead?"}

User: "Plan trip to Mysore for 2 days" (when existing trip to Ooty exists)
{"action": "clarify", "destination": "Mysore", "clarificationNeeded": "Do you want to add Mysore for 2 days to your existing Ooty trip, or plan a completely new 2-day trip to Mysore instead?"}

User: "add ooty too" (when existing trip is for Mysore)
{"action": "clarify", "destination": "Ooty", "clarificationNeeded": "Do you want to add Ooty to your existing Mysore trip? How many days would you like to spend in Ooty?"}
`;

export async function plannerAgent(
  _ctx: {
    origin?: string;
    destination?: string;
    destinations?: string[];
    days?: number;
    existingItinerary?: Itinerary;
  },
  message: string,
  history: ChatMessage[] = [],
): Promise<{
  chatResponse: string;
  itinerary: Itinerary;
  destination: string;
  destinations?: string[];
  days: number;
  destinationImageUrl?: string;
} | null> {
  // Extract conversation context for personalized planning
  const conversationContext = extractConversationContext(history);

  const extractedDetails = await extractTripDetails(
    message,
    conversationContext,
    _ctx, // Pass existing trip context for better disambiguation
  );
  
  let action = extractedDetails.action || "plan";
  let newDestination = extractedDetails.destination;
  const newDestinations = extractedDetails.destinations;
  const newDays = extractedDetails.days;

  // Spell-correct destination name if provided
  if (newDestination) {
    newDestination = await correctDestinationSpelling(newDestination);
  }

  // Guardrail: If an itinerary already exists and user mentions a different destination
  // but didn't explicitly say to replace (e.g., "new trip", "instead", "replace"),
  // force a clarification instead of overwriting the current plan.
  if (
    _ctx.existingItinerary &&
    newDestination &&
    action === "plan"
  ) {
    const currentDest = _ctx.existingItinerary.destination?.toLowerCase().trim();
    const requestedDest = newDestination.toLowerCase().trim();
    const msg = (message || "").toLowerCase();
    const explicitReplace = /(new trip|instead|replace|start over|fresh plan|separate trip|another trip)/.test(msg);
    if (currentDest && requestedDest && currentDest !== requestedDest && !explicitReplace) {
      action = "clarify";
      // fall through to clarification handler below
    }
  }
  
  // Handle clarification needed
  if (action === "clarify") {
    const clarificationMessage = extractedDetails.clarificationNeeded || 
      `I need clarification about "${newDestination}". Do you want to:
      
1. Add ${newDestination} to your existing trip${_ctx.destination ? ` (currently planning for ${_ctx.destination})` : ''}
2. Plan a completely new trip to ${newDestination} instead

Also, how many days would you like to spend there?`;

    return {
      chatResponse: clarificationMessage,
      // Preserve existing itinerary during clarification so UI doesn't clear
      itinerary: _ctx.existingItinerary || createDummyItinerary(_ctx),
      destination: _ctx.destination || newDestination || "",
      destinations: _ctx.destinations,
      days: _ctx.days || 1,
      destinationImageUrl: undefined
    };
  }

  // Handle different actions
  if (action === "add" && _ctx.existingItinerary && newDestination) {
    return await addDestinationToItinerary(_ctx.existingItinerary, newDestination, newDays || 3, _ctx.origin);
  }
  
  if (action === "remove" && _ctx.existingItinerary && newDestination) {
    return await removeDestinationFromItinerary(_ctx.existingItinerary, newDestination);
  }
  
  // Default to creating new itinerary
  const destination = newDestination || _ctx.destination;
  const destinations = newDestinations || _ctx.destinations;
  const days = newDays || _ctx.days;
  const origin =
    _ctx.origin || conversationContext.preferredOrigin || "Current location";
  const maps = new GoogleMapsClient();

  try {
    // --- Step 1: Generate the conversational markdown response --- //
    const gemini = new GeminiClient({ model: "flash" });

    let chatPrompt;
    const finalDestinations =
      destinations || (destination ? [destination] : []);
    const destinationText =
      finalDestinations.length > 1
        ? finalDestinations.join(", ")
        : destination || finalDestinations[0] || "your destination";

    if (!days) {
      // Build context-aware prompt for missing information
      let contextInfo = "";
      if (conversationContext.previousDestinations.length > 0) {
        contextInfo += `The user has previously discussed trips to: ${conversationContext.previousDestinations.join(", ")}. `;
      }
      if (conversationContext.travelPreferences.length > 0) {
        contextInfo += `Their travel preferences include: ${conversationContext.travelPreferences.join(", ")}. `;
      }
      if (conversationContext.previousDurations.length > 0) {
        contextInfo += `They have previously planned trips of ${conversationContext.previousDurations.join(", ")} days. `;
      }

      chatPrompt = `You are a friendly travel planning assistant with conversation memory. ${contextInfo}The user wants to plan a trip to ${destinationText}. Ask them for the number of days they want to stay. Be enthusiastic and suggest some popular attraction types like coffee plantations, waterfalls, and viewpoints. Reference their previous conversations when relevant.`;
    } else {
      // Build comprehensive context-aware planning prompt
      let contextualInfo = "";
      if (conversationContext.previousDestinations.length > 0) {
        contextualInfo += `**CONVERSATION CONTEXT**: The user has previously discussed trips to: ${conversationContext.previousDestinations.join(", ")}. `;
      }
      if (conversationContext.travelPreferences.length > 0) {
        contextualInfo += `Their stated preferences include: ${conversationContext.travelPreferences.join(", ")}. `;
      }
      if (conversationContext.budgetPreferences.length > 0) {
        contextualInfo += `Budget considerations mentioned: ${conversationContext.budgetPreferences.join(", ")}. `;
      }
      if (conversationContext.groupType) {
        contextualInfo += `This appears to be a ${conversationContext.groupType} trip. `;
      }
      if (contextualInfo) {
        contextualInfo += "Incorporate these insights into your planning.\n\n";
      }

      chatPrompt = `You are an expert travel planning assistant with conversation memory. Your goal is to create a beautifully formatted travel itinerary in **Markdown** for a ${days}-day trip to ${destinationText} from ${origin}.

${contextualInfo}**CRITICAL**: You MUST create the itinerary for "${destinationText}" ONLY. Do NOT substitute with any other destination like Goa, Mumbai, or Delhi. The user specifically requested "${destinationText}".

${finalDestinations.length > 1 ? `**MULTI-DESTINATION TRIP**: This is a multi-destination trip covering ${finalDestinations.join(", ")}. Allocate days appropriately across destinations and include travel time between locations.` : ""}

**IMPORTANT**: Start the response *directly* with the itinerary. Do NOT include any conversational introduction, greeting, or lead-in paragraph.

## ✍️ Formatting Rules:
- Main Title: \`# *[Destination]: [Descriptive Title] - A [X]-Day [Type] Escape*\`
- Day Headers: \`## *Day X: Theme*\` (with italics and creative themes)
- Time Sections: \`### [Emoji] [Period] ([Time Range])\`
- Use emojis for each period:
  ☀️ Morning, ⛵/🚗 Afternoon, 🌅/🍽️ Evening, 🛍️ Shopping, 🏰 Heritage, ✈️ Departure
- The 📍 emoji **must** be placed *immediately before* the full name of any location or Point of Interest (POI). For example: **Correct:** \`📍 Ooty Botanical Gardens\`, **Incorrect:** \`Ooty 📍 Botanical Gardens\`.
- Use **bold text only** for times and prices.
- Add a **short descriptive paragraph before each time-block**.
- Leave **ample blank lines** between sections for readability.
- End with required sections (see below)

## 📌 Required Sections:
1. **Trip Title** → Destination, theme, type (romantic, cultural, adventure, etc.)
2. **Daily Plan** → Morning, Afternoon, Evening with activities & timings
3. **Accommodation & Meals** → Hotels, restaurants with price ranges
4. **Estimated Budget** → Accommodation, Food, Transport, Activities, Misc.
5. **Follow-up Questions** → 3–5 questions in quotes

**EXACT STRUCTURE EXAMPLE**:

# *[Destination]: [Descriptive Title] - A [X]-Day [Type] Escape*

## *Day 1: [Creative Theme]*

### ☀️ Morning (9:00 AM - 1:00 PM)
[Short descriptive paragraph about morning activities and atmosphere]

* **9:00 AM:** [Activity description]
* **9:30 AM:** [Activity with 📍 Location Name]
* **10:30 AM:** [Activity with 📍 Location Name and details]
* **11:30 AM:** [Activity description]

### ⛵ Afternoon (1:00 PM - 6:00 PM)
[Short descriptive paragraph about afternoon activities]

* **1:00 PM:** [Activity with 📍 Location Name and detailed description]
* **3:00 PM:** [Activity with 📍 Location Name and detailed description]
* **4:30 PM:** [Activity description]
* **5:00 PM:** [Activity description]

### 🌅 Evening (6:00 PM - 9:00 PM)
[Short descriptive paragraph about evening activities]

* **6:00 PM:** [Activity with 📍 Location Name]
* **7:30 PM:** [Activity description]
* **8:30 PM:** [Activity description]

### 🏨 Accommodation & 🍴 Meals
* **Luxury:** 📍 [Hotel Name] - [Description and price range]
* **Mid-Range:** 📍 [Hotel Name] - [Description and price range]
* **Budget:** 📍 [Hotel Name] - [Description and price range]
* **Meals:** [Restaurant recommendations with 📍 locations and price ranges]

### 💰 Estimated Budget (Per Person)
* **Accommodation:** ₹[X]-₹[Y] per night
* **Food:** ₹[X]-₹[Y] per day
* **Transport:** ₹[X]-₹[Y] total
* **Activities:** ₹[X]-₹[Y] total
* **Miscellaneous:** ₹[X]-₹[Y] total
* **Total:** ₹[X]-₹[Y] for ${days} days

### 🤔 Got More Questions?
"[Question 1 about transportation]"
"[Question 2 about local cuisine]"
"[Question 3 about cultural sites]"
"[Question 4 about hidden gems]"
"[Question 5 about best times to visit]"

Make the itinerary **engaging, structured, and easy to follow** with ample spacing between sections.`;
    }
    let chatResponse = await gemini.chat(chatPrompt);

    // Programmatically remove horizontal rules from the response.
    chatResponse = chatResponse.replace(/^\s*---+\s*$/gm, "");

    // Check for error responses or invalid content
    if (
      !chatResponse?.trim() ||
      chatResponse.startsWith("[gemini:") ||
      chatResponse.includes("error 503") ||
      chatResponse.includes("error 500") ||
      chatResponse.includes("error 429")
    ) {
      console.warn(
        `[planner] Gemini returned an empty or invalid response: ${chatResponse}. Using fallback.`,
      );

      // Create a basic fallback response based on available information
      const fallbackResponse =
        destination && days
          ? `I'd be happy to help you plan your ${days}-day trip to ${destination}! I'm having some technical difficulties generating the detailed itinerary right now, but I can get you started with the basics. Would you like me to try again?`
          : destination
            ? `Great choice! I'd love to help you plan a trip to ${destination}. Could you let me know how many days you'd like to stay so I can create a detailed itinerary for you?`
            : "I'm having a little trouble generating that itinerary right now. Could you try rephrasing your request?";

      return {
        chatResponse: fallbackResponse,
        itinerary: {
          origin,
          destination: destination || "",
          days: days || 0,
          daysPlan: [],
        },
        destination: destination || "",
        destinations,
        days: days || 0,
      };
    }

    // If we don't have enough info for an itinerary, return just the chat response.
    if ((!destination && !destinations) || !days) {
      return {
        chatResponse,
        itinerary: {
          origin,
          destination: destination || "",
          days: 0,
          daysPlan: [],
        },
        destination: destination || "",
        destinations,
        days: days || 0,
      };
    }

    // --- Step 2: Fetch real POIs for all destinations (optimized) --- //
    const allDestinations = destinations || (destination ? [destination] : []);
    const poiPromises = [];

    // Limit to first 3 destinations to prevent timeout
    const limitedDestinations = allDestinations.slice(0, 3);
    console.log(
      `[planner] Fetching POIs for ${limitedDestinations.length} destinations:`,
      limitedDestinations,
    );

    for (const dest of limitedDestinations) {
      poiPromises.push(
        maps
          .searchPlaces({ q: `tourist attractions in ${dest}` }, "attraction")
          .catch((e: any) => {
            console.warn(
              `[planner] Attractions search failed for ${dest}:`,
              e.message,
            );
            return [];
          }),
        maps
          .searchPlaces({ q: `restaurants in ${dest}` }, "restaurant")
          .catch((e: any) => {
            console.warn(
              `[planner] Restaurants search failed for ${dest}:`,
              e.message,
            );
            return [];
          }),
        maps.searchPlaces({ q: `hotels in ${dest}` }, "stay").catch((e: any) => {
          console.warn(
            `[planner] Hotels search failed for ${dest}:`,
            e.message,
          );
          return [];
        }),
      );
    }

    const poiResults = await Promise.all(poiPromises);

    // Combine all POIs from all destinations (limit to prevent huge payloads)
    const attractions: POI[] = [];
    const restaurants: POI[] = [];
    const stays: POI[] = [];

    for (let i = 0; i < poiResults.length; i += 3) {
      attractions.push(...(poiResults[i] || []).slice(0, 10)); // Limit to 10 per destination
      restaurants.push(...(poiResults[i + 1] || []).slice(0, 10));
      stays.push(...(poiResults[i + 2] || []).slice(0, 10));
    }

    console.log(
      `[planner] Found ${attractions.length} attractions, ${restaurants.length} restaurants, ${stays.length} stays`,
    );

    const itinerary = await createStructuredItinerary(
      chatResponse,
      { ..._ctx, destination, days },
      { attractions, restaurants, stays },
    );

    // --- Step 3: Fetch destination image for trip card --- //
    let destinationImageUrl: string | undefined;
    try {
      const imageService = new DestinationImageService();
      const allDestinations =
        destinations || (destination ? [destination] : []);
      if (allDestinations.length > 0) {
        const imageResult =
          await imageService.getDestinationImageForTrip(allDestinations);
        destinationImageUrl = imageResult.imageUrl;
        console.log(
          `[planner] Destination image ${destinationImageUrl ? "found" : "not found"} for ${allDestinations[0]}`,
        );
      }
    } catch (error) {
      console.warn(`[planner] Failed to fetch destination image:`, error);
    }

    return {
      chatResponse,
      itinerary,
      destination: destination || allDestinations[0],
      destinations,
      days,
      destinationImageUrl,
    };
  } catch (e: any) {
    console.warn("[planner] Gemini or Maps failed:", e);

    // Handle API configuration issues
    if (e.message && e.message.includes("API configuration")) {
      return {
        chatResponse:
          "I'm experiencing some technical difficulties right now. Please try again in a few minutes, or contact support if the issue persists.",
        itinerary: {
          origin,
          destination: destination || "",
          days: days || 0,
          daysPlan: [],
        },
        destination: destination || "",
        destinations,
        days: days || 0,
      };
    }

    // Handle specific error types
    if (
      e.message &&
      (e.message.includes("429") || e.message.includes("rate limit"))
    ) {
      return {
        chatResponse:
          "It looks like I'm very popular right now! I've hit my request limit. Please try again in a little while.",
        itinerary: {
          origin,
          destination: destination || "",
          days: days || 0,
          daysPlan: [],
        },
        destination: destination || "",
        destinations,
        days: days || 0,
      };
    }

    if (e.message && (e.message.includes("503") || e.message.includes("500"))) {
      return {
        chatResponse:
          "I'm experiencing some temporary technical difficulties. Let me try to help you plan your trip with a simpler approach. Could you tell me more about what you'd like to do?",
        itinerary: {
          origin,
          destination: destination || "",
          days: days || 0,
          daysPlan: [],
        },
        destination: destination || "",
        destinations,
        days: days || 0,
      };
    }

    if (e.message && e.message.includes("timeout")) {
      return {
        chatResponse: `I'm taking a bit longer to plan your ${destination || "trip"}. Let me give you a quick overview while I work on the details!`,
        itinerary: createDummyItinerary({ ..._ctx, destination, days }),
        destination: destination || "",
        destinations,
        days: days || 0,
      };
    }

    // Return fallback itinerary instead of null
    console.log("[planner] Using fallback itinerary due to error:", e.message);
    const fallbackResponse = destination
      ? `I ran into a little trouble creating your detailed ${destination} itinerary, but here's a sample to get you started!`
      : "I ran into a little trouble, but I'm here to help! Could you tell me which destination you'd like to explore?";

    return {
      chatResponse: fallbackResponse,
      itinerary: createDummyItinerary({ ..._ctx, destination, days }),
      destination: destination || "",
      destinations,
      days: days || 0,
    };
  }
}

// --- Helper to create the structured itinerary from a text description and POIs --- //
async function createStructuredItinerary(
  description: string,
  ctx: { origin?: string; destination?: string; days?: number },
  pois: { attractions: POI[]; restaurants: POI[]; stays: POI[] },
): Promise<Itinerary> {
  const gemini = new GeminiClient({ model: "flash" });
  
  // Reduce POI lists to minimize token usage
  const compactPois = {
    attractions: pois.attractions.slice(0, 8).map(p => ({ id: p.id, name: p.name })),
    restaurants: pois.restaurants.slice(0, 6).map(p => ({ id: p.id, name: p.name })),
    stays: pois.stays.slice(0, 3).map(p => ({ id: p.id, name: p.name }))
  };

  const jsonPrompt = `Create JSON itinerary for ${ctx.destination}, ${ctx.days || 3} days.

POIs:
A: ${JSON.stringify(compactPois.attractions)}
R: ${JSON.stringify(compactPois.restaurants)}  
S: ${JSON.stringify(compactPois.stays)}

JSON format:
{
  "origin": "${ctx.origin || "Current location"}",
  "destination": "${ctx.destination}",
  "days": ${ctx.days || 3},
  "daysPlan": [
    {
      "day": 1,
      "date": "2024-01-01", 
      "title": "Day 1 Title",
      "activities": [
        {"name": "Activity", "start": "09:00", "end": "11:00", "poiId": "poi_id"}
      ]
    }
  ]
}

Requirements:
- Use POI IDs from above lists
- ${ctx.days || 3} days exactly
- 2-4 activities per day
- Times 09:00-20:00
- Valid JSON only`;

  // Retry logic for JSON generation
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      console.log(`[planner] JSON generation attempt ${attempt}/3`);
      const jsonResponse = await gemini.chat(jsonPrompt);
      console.log("[planner] Raw JSON response length:", jsonResponse.length);
      console.log(
        "[planner] Raw response preview:",
        jsonResponse.substring(0, 200) + "...",
      );

      // Better JSON extraction and cleaning
      let cleanedJson = jsonResponse
        .replace(/^```json\s*/i, "")
        .replace(/\s*```\s*$/i, "")
        .trim();

      // Remove any markdown or extra text before/after JSON
      const jsonStart = cleanedJson.indexOf("{");
      const jsonEnd = cleanedJson.lastIndexOf("}");

      if (jsonStart === -1 || jsonEnd === -1) {
        throw new Error(
          `JSON structure not found. Response: ${cleanedJson.substring(0, 500)}`,
        );
      }

      if (jsonEnd <= jsonStart) {
        throw new Error(
          `Invalid JSON structure: end position ${jsonEnd} <= start position ${jsonStart}`,
        );
      }

      cleanedJson = cleanedJson.substring(jsonStart, jsonEnd + 1);
      console.log("[planner] Cleaned JSON length:", cleanedJson.length);
      console.log(
        "[planner] JSON preview:",
        cleanedJson.substring(0, 300) + "...",
      );

      // Validate basic JSON structure
      if (!cleanedJson.startsWith("{") || !cleanedJson.endsWith("}")) {
        throw new Error(
          `Invalid JSON structure: doesn't start/end with braces. Got: ${cleanedJson.substring(0, 50)}...${cleanedJson.substring(-50)}`,
        );
      }

      // Additional validation for common JSON issues
      if (cleanedJson.length < 50) {
        throw new Error(
          `JSON too short (${cleanedJson.length} chars): ${cleanedJson}`,
        );
      }

      // Check for truncated JSON (common AI issue)
      const openBraces = (cleanedJson.match(/\{/g) || []).length;
      const closeBraces = (cleanedJson.match(/\}/g) || []).length;
      if (openBraces !== closeBraces) {
        throw new Error(
          `Mismatched braces: ${openBraces} open, ${closeBraces} close`,
        );
      }

      let parsed: Itinerary;
      try {
        parsed = JSON.parse(cleanedJson) as Itinerary;
      } catch (parseError) {
        console.error("[planner] JSON parse error:", parseError);
        console.error("[planner] Problematic JSON:", cleanedJson);
        throw new Error(
          `JSON parsing failed: ${parseError instanceof Error ? parseError.message : String(parseError)}. JSON length: ${cleanedJson.length}`,
        );
      }

      // Validate required fields with detailed error messages
      if (!parsed) {
        throw new Error("Parsed result is null or undefined");
      }

      if (!parsed.daysPlan) {
        throw new Error("Invalid itinerary structure: daysPlan is missing");
      }

      if (!Array.isArray(parsed.daysPlan)) {
        throw new Error(
          `Invalid itinerary structure: daysPlan is not an array, got: ${typeof parsed.daysPlan}`,
        );
      }

      if (parsed.daysPlan.length === 0) {
        throw new Error("Invalid itinerary structure: daysPlan is empty");
      }

      // Validate each day has required fields
      parsed.daysPlan.forEach((day, index) => {
        if (!day.day && day.day !== 0) {
          throw new Error(`Day ${index} missing 'day' field`);
        }
        if (!day.activities || !Array.isArray(day.activities)) {
          throw new Error(`Day ${index} missing or invalid 'activities' field`);
        }
      });

      // Enrich activities with full POI data
      parsed.daysPlan.forEach((day) => {
        if (day.activities && Array.isArray(day.activities)) {
          day.activities.forEach((act) => {
            const allPois = [
              ...pois.attractions,
              ...pois.restaurants,
              ...pois.stays,
            ];
            const poi = allPois.find((p) => p.id === act.poiId);
            if (poi) {
              act.name = poi.name;
              act.location = poi.address;
              act.photoUrl = poi.photoUrl;
              act.rating = poi.rating;
              act.lat = poi.lat;
              act.lng = poi.lng;
            }
          });
        }
      });

      console.log(
        "[planner] Successfully created structured itinerary with",
        parsed.daysPlan.length,
        "days",
      );
      return parsed;
    } catch (e) {
      lastError = e as Error;
      console.log(
        `[planner] Attempt ${attempt} failed:`,
        e,
      );
      
      // If Gemini is overloaded (503) or MAX_TOKENS, try a different approach
      const errorMsg = (e as Error).message || '';
      if (errorMsg.includes('503') || errorMsg.includes('MAX_TOKENS') || errorMsg.includes('overloaded')) {
        console.log('[planner] Gemini overloaded, trying simplified generation...');
        try {
          return await createSimplifiedItinerary(ctx, pois);
        } catch (fallbackError) {
          console.log('[planner] Simplified generation also failed:', fallbackError);
        }
      }
      
      // If this isn't the last attempt, wait a bit before retrying
      if (attempt < 3) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }
  
  console.error("[planner] All JSON generation attempts failed, using fallback. Last error:", lastError);
  return createDummyItinerary(ctx);
}

// --- Multi-destination Trip Management --- //
async function addDestinationToItinerary(
  existingItinerary: Itinerary,
  newDestination: string,
  days: number,
  origin?: string
): Promise<{
  chatResponse: string;
  itinerary: Itinerary;
  destination: string;
  destinations?: string[];
  days: number;
  destinationImageUrl?: string;
} | null> {
  try {
    const maps = new GoogleMapsClient();
    const destinationImageService = new DestinationImageService();
    
    // Get POIs for the new destination
    console.log(`[planner] Adding ${newDestination} (${days} days) to existing itinerary`);
    const [attractions, restaurants, stays] = await Promise.all([
      maps.searchPlaces({ q: `tourist attractions in ${newDestination}` }, "attraction"),
      maps.searchPlaces({ q: `restaurants in ${newDestination}` }, "restaurant"),
      maps.searchPlaces({ q: `hotels in ${newDestination}` }, "stay")
    ]);
    const pois = { attractions, restaurants, stays };
    
    // Create itinerary for the new destination
    const newDestinationItinerary = await createStructuredItinerary(
      `Create a ${days}-day itinerary for ${newDestination}`,
      { origin: existingItinerary.destination, destination: newDestination, days },
      pois
    );
    
    // Merge with existing itinerary
    const totalDays = existingItinerary.days + days;
    const mergedDaysPlan = [
      ...existingItinerary.daysPlan,
      ...newDestinationItinerary.daysPlan.map(day => ({
        ...day,
        day: day.day + existingItinerary.days,
        date: new Date(new Date(existingItinerary.daysPlan[existingItinerary.daysPlan.length - 1].date).getTime() + day.day * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
      }))
    ];
    
    // Create destination segments
    const destinationSegments = [
      {
        destination: existingItinerary.destination,
        startDay: 1,
        endDay: existingItinerary.days,
        days: existingItinerary.days
      },
      {
        destination: newDestination,
        startDay: existingItinerary.days + 1,
        endDay: totalDays,
        days: days
      }
    ];
    
    const mergedItinerary: Itinerary = {
      origin: existingItinerary.origin,
      destination: existingItinerary.destination, // Keep primary destination
      destinations: [existingItinerary.destination, newDestination],
      days: totalDays,
      daysPlan: mergedDaysPlan,
      destinationSegments
    };
    
    // Get destination image
    let destinationImageUrl: string | undefined;
    try {
      const imageResult = await destinationImageService.getDestinationImage(newDestination);
      destinationImageUrl = imageResult?.imageUrl;
    } catch (e) {
      console.log(`[planner] Could not fetch destination image for ${newDestination}:`, e);
    }
    
    const chatResponse = `Great! I've added ${newDestination} (${days} days) to your existing trip. Your journey will now continue from ${existingItinerary.destination} to ${newDestination}, making it a ${totalDays}-day multi-destination adventure!`;
    
    return {
      chatResponse,
      itinerary: mergedItinerary,
      destination: existingItinerary.destination,
      destinations: [existingItinerary.destination, newDestination],
      days: totalDays,
      destinationImageUrl
    };
  } catch (e) {
    console.error("[planner] Error adding destination:", e);
    return null;
  }
}

async function removeDestinationFromItinerary(
  existingItinerary: Itinerary,
  destinationToRemove: string
): Promise<{
  chatResponse: string;
  itinerary: Itinerary;
  destination: string;
  destinations?: string[];
  days: number;
  destinationImageUrl?: string;
} | null> {
  try {
    if (!existingItinerary.destinationSegments) {
      return {
        chatResponse: `I can only remove destinations from multi-destination trips. Your current trip appears to be a single destination.`,
        itinerary: existingItinerary,
        destination: existingItinerary.destination,
        days: existingItinerary.days
      };
    }
    
    const segmentToRemove = existingItinerary.destinationSegments.find(
      seg => seg.destination.toLowerCase().includes(destinationToRemove.toLowerCase())
    );
    
    if (!segmentToRemove) {
      return {
        chatResponse: `I couldn't find ${destinationToRemove} in your current trip itinerary.`,
        itinerary: existingItinerary,
        destination: existingItinerary.destination,
        destinations: existingItinerary.destinations,
        days: existingItinerary.days
      };
    }
    
    // Remove the destination segment and associated days
    const remainingSegments = existingItinerary.destinationSegments.filter(
      seg => seg.destination !== segmentToRemove.destination
    );
    
    const remainingDaysPlan = existingItinerary.daysPlan.filter(
      day => day.day < segmentToRemove.startDay || day.day > segmentToRemove.endDay
    );
    
    // Adjust day numbers for remaining days
    const adjustedDaysPlan = remainingDaysPlan.map(day => {
      if (day.day > segmentToRemove.endDay) {
        return {
          ...day,
          day: day.day - segmentToRemove.days
        };
      }
      return day;
    });
    
    // Update remaining segments
    const adjustedSegments = remainingSegments.map(seg => {
      if (seg.startDay > segmentToRemove.endDay) {
        return {
          ...seg,
          startDay: seg.startDay - segmentToRemove.days,
          endDay: seg.endDay - segmentToRemove.days
        };
      }
      return seg;
    });
    
    const updatedItinerary: Itinerary = {
      ...existingItinerary,
      days: existingItinerary.days - segmentToRemove.days,
      daysPlan: adjustedDaysPlan,
      destinationSegments: adjustedSegments.length > 1 ? adjustedSegments : undefined,
      destinations: adjustedSegments.length > 1 ? adjustedSegments.map(seg => seg.destination) : undefined
    };
    
    const chatResponse = `I've removed ${destinationToRemove} from your trip. Your itinerary is now ${updatedItinerary.days} days focusing on ${adjustedSegments.map(seg => seg.destination).join(" and ")}.`;
    
    return {
      chatResponse,
      itinerary: updatedItinerary,
      destination: updatedItinerary.destination,
      destinations: updatedItinerary.destinations,
      days: updatedItinerary.days
    };
  } catch (e) {
    console.error("[planner] Error removing destination:", e);
    return null;
  }
}

// --- Simplified Itinerary Generator (when AI is overloaded) --- //
async function createSimplifiedItinerary(
  ctx: { origin?: string; destination?: string; days?: number },
  pois: { attractions: POI[]; restaurants: POI[]; stays: POI[] }
): Promise<Itinerary> {
  const today = new Date();
  const totalDays = ctx.days || 3;
  
  const daysPlan = Array.from({ length: totalDays }, (_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() + i);
    const dateStr = d.toISOString().slice(0, 10);
    
    const activities: Activity[] = [];
    
    // Morning attraction
    if (pois.attractions[i % pois.attractions.length]) {
      const poi = pois.attractions[i % pois.attractions.length];
      activities.push({
        name: poi.name,
        start: "09:00",
        end: "11:00",
        location: poi.address,
        poiId: poi.id,
        lat: poi.lat,
        lng: poi.lng,
        photoUrl: poi.photoUrl,
        rating: poi.rating
      });
    }
    
    // Lunch
    if (pois.restaurants[i % pois.restaurants.length]) {
      const poi = pois.restaurants[i % pois.restaurants.length];
      activities.push({
        name: poi.name,
        start: "12:30",
        end: "14:00",
        location: poi.address,
        poiId: poi.id,
        lat: poi.lat,
        lng: poi.lng,
        photoUrl: poi.photoUrl,
        rating: poi.rating
      });
    }
    
    // Afternoon attraction
    if (pois.attractions[(i + 1) % pois.attractions.length]) {
      const poi = pois.attractions[(i + 1) % pois.attractions.length];
      activities.push({
        name: poi.name,
        start: "15:00",
        end: "17:00",
        location: poi.address,
        poiId: poi.id,
        lat: poi.lat,
        lng: poi.lng,
        photoUrl: poi.photoUrl,
        rating: poi.rating
      });
    }
    
    // Dinner
    if (pois.restaurants[(i + 1) % pois.restaurants.length]) {
      const poi = pois.restaurants[(i + 1) % pois.restaurants.length];
      activities.push({
        name: poi.name,
        start: "19:00",
        end: "21:00",
        location: poi.address,
        poiId: poi.id,
        lat: poi.lat,
        lng: poi.lng,
        photoUrl: poi.photoUrl,
        rating: poi.rating
      });
    }
    
    return {
      day: i + 1,
      date: dateStr,
      title: i === 0 ? `Arrival in ${ctx.destination}` : 
             i === totalDays - 1 ? `Farewell ${ctx.destination}` : 
             `Explore ${ctx.destination}`,
      activities
    };
  });
  
  return {
    origin: ctx.origin || "Current location",
    destination: ctx.destination!,
    days: ctx.days!,
    daysPlan,
  };
}

// --- Fallback Itinerary Generator --- //
function createDummyItinerary(ctx: {
  origin?: string;
  destination?: string;
  days?: number;
}): Itinerary {
  const today = new Date();
  const totalDays = ctx.days || 1;
  const daysPlan = Array.from({ length: totalDays }, (_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() + i);
    const dateStr = d.toISOString().slice(0, 10);
    const baseTitle =
      i === 0
        ? `Arrival in ${ctx.destination}`
        : i === totalDays - 1
          ? `Farewell ${ctx.destination}`
          : `Explore ${ctx.destination}`;
    const activities: Activity[] = [
      {
        name: `Breakfast near ${ctx.destination}`,
        start: "09:00",
        end: "10:00",
      },
      { name: `Top sight #${i + 1}`, start: "11:00", end: "12:30" },
      { name: "Local lunch", start: "13:00", end: "14:00" },
      { name: "Scenic walk", start: "16:00", end: "17:30" },
      { name: "Dinner", start: "19:30", end: "21:00" },
    ];
    return { day: i + 1, date: dateStr, title: baseTitle, activities };
  });
  return {
    origin: ctx.origin || "",
    destination: ctx.destination!,
    days: ctx.days!,
    daysPlan,
  };
}

async function extractTripDetails(
  message: string,
  context?: any,
  existingTrip?: any,
): Promise<{ destination?: string; destinations?: string[]; days?: number; action?: string; clarificationNeeded?: string }> {
  const gemini = new GeminiClient({ model: "flash" });

  // Include context in extraction prompt if available
  let contextPrompt = "";
  if (context && context.previousDestinations.length > 0) {
    contextPrompt = `\nCONTEXT: Previous destinations discussed: ${context.previousDestinations.join(", ")}`;
  }
  
  // Include existing trip context for better disambiguation
  if (existingTrip && existingTrip.destination) {
    contextPrompt += `\nEXISTING TRIP: Currently planning for ${existingTrip.destination}${existingTrip.days ? ` (${existingTrip.days} days)` : ''}`;
  } else if (context && context.currentDestination) {
    contextPrompt += `\nCURRENT TRIP: Active trip planning for ${context.currentDestination}`;
  }

  const prompt = `${TRIP_DETAIL_EXTRACTION_PROMPT}

User message: "${message}"${contextPrompt}`;

  const jsonResponse = await gemini.chat(prompt);
  console.log(`[planner] Trip details extraction response: ${jsonResponse}`);
  const cleanedJson = jsonResponse
    .replace(/^```json\s*/i, "")
    .replace(/\s*```\s*$/i, "")
    .trim();
  const jsonMatch = cleanedJson.match(/\{[\s\S]*\}/);

  if (jsonMatch) {
    try {
      const result = JSON.parse(jsonMatch[0]);
      console.log(`[planner] Extracted trip details:`, result);
      return result;
    } catch (e) {
      console.warn(
        "[planner] Failed to parse trip details JSON from Gemini.",
        e,
      );
    }
  }
  return {};
}

export function emitItineraryUpdate(it: Itinerary): WsEvent {
  return { type: "itinerary.update", data: it };
}

// Spell correction for destination names using Google Places API
async function correctDestinationSpelling(destination: string): Promise<string> {
  try {
    const maps = new GoogleMapsClient();
    
    // Use Google Places Text Search to find the best match
    const searchResults = await maps.searchPlaces({ q: destination }, "attraction");
    
    if (searchResults && searchResults.length > 0) {
      // Extract city/location name from the first result's address
      const firstResult = searchResults[0];
      if (firstResult.address) {
        // Try to extract city name from address components
        const addressParts = firstResult.address.split(',').map((part: string) => part.trim());
        
        // Look for the main city/destination name (usually first or second part)
        for (const part of addressParts) {
          // Skip country codes, postal codes, and state abbreviations
          if (part.length > 2 && !part.match(/^\d+/) && !part.match(/^[A-Z]{2,3}$/)) {
            // If this part contains the original destination (fuzzy match), use it
            const similarity = calculateSimilarity(destination.toLowerCase(), part.toLowerCase());
            if (similarity > 0.6) {
              console.log(`[planner] Corrected "${destination}" to "${part}" (similarity: ${similarity})`);
              return part;
            }
          }
        }
      }
      
      // Fallback: use the POI name if it's a place name
      if (firstResult.name && firstResult.name.length > 2) {
        const similarity = calculateSimilarity(destination.toLowerCase(), firstResult.name.toLowerCase());
        if (similarity > 0.5) {
          console.log(`[planner] Corrected "${destination}" to "${firstResult.name}" (similarity: ${similarity})`);
          return firstResult.name;
        }
      }
    }
    
    // If no good match found, return original
    console.log(`[planner] No spelling correction found for "${destination}"`);
    return destination;
  } catch (error) {
    console.error(`[planner] Error correcting spelling for "${destination}":`, error);
    return destination; // Return original on error
  }
}

// Calculate string similarity using Levenshtein distance
function calculateSimilarity(str1: string, str2: string): number {
  const len1 = str1.length;
  const len2 = str2.length;
  
  if (len1 === 0) return len2 === 0 ? 1 : 0;
  if (len2 === 0) return 0;
  
  const matrix = Array(len1 + 1).fill(null).map(() => Array(len2 + 1).fill(null));
  
  for (let i = 0; i <= len1; i++) matrix[i][0] = i;
  for (let j = 0; j <= len2; j++) matrix[0][j] = j;
  
  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,     // deletion
        matrix[i][j - 1] + 1,     // insertion
        matrix[i - 1][j - 1] + cost // substitution
      );
    }
  }
  
  const maxLen = Math.max(len1, len2);
  return (maxLen - matrix[len1][len2]) / maxLen;
}

// Helper function to extract conversation context
function extractConversationContext(history: ChatMessage[]) {
  const context = {
    previousDestinations: new Set<string>(),
    travelPreferences: new Set<string>(),
    budgetPreferences: new Set<string>(),
    previousDurations: new Set<number>(),
    groupType: undefined as string | undefined,
    preferredOrigin: undefined as string | undefined,
    currentDestination: undefined as string | undefined,
  };

  if (!history || history.length === 0) {
    return {
      previousDestinations: [],
      travelPreferences: [],
      budgetPreferences: [],
      previousDurations: [],
      groupType: undefined,
      preferredOrigin: undefined,
      currentDestination: undefined,
    };
  }

  // Analyze recent conversation history (last 15 messages)
  const recentHistory = history.slice(-15);

  recentHistory.forEach((msg) => {
    const content = msg.content.toLowerCase();

    // Extract destinations mentioned and track current destination
    const destPatterns = [
      /(?:to|visit|visiting|in|plan.*trip.*to)\s+([A-Za-z][A-Za-z\s]{2,20}?)(?:\s|$|[,.!?])/g,
      /([A-Za-z][A-Za-z\s]{2,20}?)\s+(?:trip|travel|vacation|itinerary)/g,
    ];

    destPatterns.forEach((pattern) => {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const dest = match[1].trim();
        if (dest.length > 2 && dest.length < 20 && !isCommonWord(dest)) {
          const formattedDest = dest.charAt(0).toUpperCase() + dest.slice(1);
          context.previousDestinations.add(formattedDest);
          
          // Track most recent destination as current
          if (msg.role === 'assistant' && content.includes('itinerary')) {
            context.currentDestination = formattedDest;
          }
        }
      }
    });

    // Extract duration preferences
    const dayMatches = content.match(/(\d+)\s*days?/g);
    if (dayMatches) {
      dayMatches.forEach((match) => {
        const days = parseInt(match.match(/\d+/)?.[0] || "0");
        if (days > 0 && days <= 30) {
          context.previousDurations.add(days);
        }
      });
    }

    // Extract travel preferences
    const preferenceKeywords = [
      "adventure",
      "relaxing",
      "cultural",
      "historical",
      "nature",
      "beach",
      "mountain",
      "food",
      "shopping",
      "photography",
      "trekking",
      "family",
      "romantic",
      "budget",
      "luxury",
      "backpacking",
      "local experience",
      "nightlife",
      "wellness",
      "spiritual",
    ];

    preferenceKeywords.forEach((keyword) => {
      if (content.includes(keyword)) {
        context.travelPreferences.add(keyword);
      }
    });

    // Extract budget preferences
    const budgetKeywords = [
      "budget",
      "cheap",
      "affordable",
      "luxury",
      "expensive",
      "mid-range",
      "premium",
    ];
    budgetKeywords.forEach((keyword) => {
      if (content.includes(keyword)) {
        context.budgetPreferences.add(keyword);
      }
    });

    // Detect group type
    if (
      content.includes("family") ||
      content.includes("kids") ||
      content.includes("children")
    ) {
      context.groupType = "family";
    } else if (
      content.includes("couple") ||
      content.includes("romantic") ||
      content.includes("honeymoon")
    ) {
      context.groupType = "couple";
    } else if (content.includes("friends") || content.includes("group")) {
      context.groupType = "friends";
    } else if (content.includes("solo") || content.includes("alone")) {
      context.groupType = "solo";
    }

    // Extract origin preferences
    const originPattern =
      /(?:from|coming from|starting from|live in|based in)\s+([A-Za-z][A-Za-z\s]{2,20}?)(?:\s|$|[,.!?])/g;
    let originMatch;
    while ((originMatch = originPattern.exec(content)) !== null) {
      const origin = originMatch[1].trim();
      if (origin.length > 2 && origin.length < 20) {
        context.preferredOrigin =
          origin.charAt(0).toUpperCase() + origin.slice(1);
      }
    }
  });

  return {
    previousDestinations: Array.from(context.previousDestinations),
    travelPreferences: Array.from(context.travelPreferences),
    budgetPreferences: Array.from(context.budgetPreferences),
    previousDurations: Array.from(context.previousDurations),
    groupType: context.groupType,
    preferredOrigin: context.preferredOrigin,
    currentDestination: context.currentDestination,
  };
}

// Helper function to filter out common words that aren't destinations
function isCommonWord(word: string): boolean {
  const commonWords = [
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "get",
    "has",
    "him",
    "his",
    "how",
    "man",
    "new",
    "now",
    "old",
    "see",
    "two",
    "way",
    "who",
    "boy",
    "did",
    "its",
    "let",
    "put",
    "say",
    "she",
    "too",
    "use",
    "trip",
    "plan",
    "visit",
    "travel",
    "vacation",
    "holiday",
    "days",
    "time",
    "place",
    "good",
    "great",
    "nice",
    "best",
    "love",
    "like",
    "want",
    "need",
    "help",
    "please",
    "thank",
  ];
  return commonWords.includes(word.toLowerCase());
}
