import type { Itinerary, WsEvent, POI, Activity } from "../types/schemas.js";
import type { Message } from "../db/types.js";
import { GroqClient } from "../tools/groq.js";
import { GeminiClient } from "../tools/gemini.js";
import type { GroqModel } from "../tools/groq.js";
import { GoogleMapsClient } from "../tools/maps.js";
import { DestinationImageService } from "../tools/destination-images.js";

export async function plannerAgent(
  _ctx: {
    origin?: string;
    destination?: string;
    destinations?: string[];
    days?: number;
  },
  message: string,
  history: Message[] = [],
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
  );
  const destination = extractedDetails.destination || _ctx.destination;
  const destinations = extractedDetails.destinations || _ctx.destinations;
  const days = extractedDetails.days || _ctx.days;
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

      chatPrompt = `You are a friendly travel planning assistant. ${contextInfo}The user wants to plan a trip to ${destinationText}. Ask them for the number of days they want to stay. Be enthusiastic and suggest some popular attraction types like coffee plantations, waterfalls, and viewpoints. ${contextInfo ? 'Use any relevant context from our conversation.' : ''}`;
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

      // Decide whether we need deeper follow-up questions based on missing context
      const needDeepFollowUps =
        (conversationContext.travelPreferences?.length || 0) === 0 ||
        (conversationContext.budgetPreferences?.length || 0) === 0 ||
        !conversationContext.groupType ||
        !origin;

      const followUpSpec = needDeepFollowUps
        ? `5. **Follow-up Questions** → Provide 4–7 highly context-aware questions (each in quotes, one per line) targeting only missing preferences or constraints (budget, pace, group type, mobility needs, season/weather, must-see POIs, hotel class).`
        : `5. **Next-Step Questions (Optional)** → If refinements would meaningfully improve the trip, ask 1–2 concise questions in quotes (e.g., tweak pace, swap a POI, dietary needs). If nothing crucial is missing, omit this section entirely.`;

      chatPrompt = `You are an expert travel planning assistant. Your goal is to create a beautifully formatted travel itinerary in **Markdown** for a ${days}-day trip to ${destinationText} from ${origin}.

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
${followUpSpec}

If you include a questions section, each question must be on its own line and wrapped in quotes. If nothing important is missing, omit the entire questions section.

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
    let chatResponse = await gemini.chat(chatPrompt, "flash");

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
        `[planner] Groq returned an empty or invalid response: ${chatResponse}. Using fallback.`,
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
          .catch((e) => {
            console.warn(
              `[planner] Attractions search failed for ${dest}:`,
              e.message,
            );
            return [];
          }),
        maps
          .searchPlaces({ q: `restaurants in ${dest}` }, "restaurant")
          .catch((e) => {
            console.warn(
              `[planner] Restaurants search failed for ${dest}:`,
              e.message,
            );
            return [];
          }),
        maps.searchPlaces({ q: `hotels in ${dest}` }, "stay").catch((e) => {
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
    console.warn("[planner] LLM or Maps failed:", e);

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
  const jsonModel = (process.env.GROQ_JSON_MODEL as GroqModel) || "llama-3.1-8b-instant";
  const groq = new GroqClient({ model: jsonModel });
  const jsonPrompt = `Create a structured JSON itinerary based on the description and available POIs. Follow this EXACT format:

Description: ${description}

Available POIs (use these IDs in your response):
Attractions: ${JSON.stringify(
    pois.attractions
      .slice(0, 15)
      .map((p) => ({ id: p.id, name: p.name, address: p.address })),
    null,
    2,
  )}
Restaurants: ${JSON.stringify(
    pois.restaurants
      .slice(0, 10)
      .map((p) => ({ id: p.id, name: p.name, address: p.address })),
    null,
    2,
  )}
Stays: ${JSON.stringify(
    pois.stays
      .slice(0, 5)
      .map((p) => ({ id: p.id, name: p.name, address: p.address })),
    null,
    2,
  )}

CRITICAL REQUIREMENTS:
- Use ONLY POI IDs from the lists above (required for each activity)
- Create exactly ${ctx.days || 3} days
- Each day must have 2-4 activities minimum
- Use realistic time slots (09:00-20:00)
- Include accommodation for multi-day trips
- RESPOND WITH COMPLETE, VALID JSON ONLY
- DO NOT truncate the response
- ENSURE all braces are properly closed
 - DO NOT include comments, trailing commas, or any text outside the JSON
 - DO NOT use // or /* */ anywhere

You MUST respond with ONLY this complete JSON structure:
{
  "origin": "${ctx.origin || "Current location"}",
  "destination": "${ctx.destination}",
  "days": ${ctx.days || 3},
  "daysPlan": [
    {
      "day": 1,
      "date": "2024-01-01",
      "title": "Day title",
      "activities": [
        {"name": "Activity", "start": "09:00", "end": "11:00", "location": "Address", "poiId": "poi_id_from_above"}
      ],
      "accommodation": {"name": "Hotel", "checkIn": "15:00", "poiId": "stay_poi_id"}
    }
  ]
}`;

  try {
    const jsonResponse = await groq.chat(jsonPrompt);
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
    // Strip line and block comments that models sometimes add
    cleanedJson = cleanedJson
      .replace(/\/\/.*$/gm, "") // remove // comments
      .replace(/\/\*[\s\S]*?\*\//g, ""); // remove /* */ comments
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
        `JSON parsing failed: ${parseError.message}. JSON length: ${cleanedJson.length}`,
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
    console.log(
      "[planner] Failed to create structured itinerary, using fallback.",
      e,
    );
  }
  return createDummyItinerary(ctx);
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
): Promise<{ destination?: string; destinations?: string[]; days?: number }> {
  const groq = new GroqClient({ model: "llama-3.1-8b-instant" });

  // Include context in extraction prompt if available
  let contextPrompt = "";
  if (
    context &&
    context.previousDestinations &&
    context.previousDestinations.length > 0
  ) {
    contextPrompt = `\n\nCONVERSATION CONTEXT: The user has previously mentioned these destinations: ${context.previousDestinations.join(", ")}. Use this context to better understand location references.`;
  }

  const prompt = `You are an extraction tool. Output STRICT JSON ONLY.
Do not include any explanation, code fences, or markdown. ${contextPrompt}

USER_MESSAGE: "${message}"

REQUIREMENTS:
- Extract destination names faithfully. If a destination looks misspelled (e.g., "delgi"), correct it to the most likely real place ("Delhi"). Do not substitute with a different place.
- If multiple destinations are mentioned (e.g., "ooty and coonoor"), return a "destinations" array.
- If one destination, return a single "destination" string.
- Include "days" if present (number). Omit keys that are not present.

Respond with ONLY one JSON object: {"destination"?: string, "destinations"?: string[], "days"?: number}`;
  const jsonResponse = await groq.chat(prompt, { temperature: 0 });
  console.log(`[planner] Trip details extraction response: ${jsonResponse}`);
  // Robust cleaning: strip code fences, prose, and take the first top-level JSON object
  let cleanedJson = jsonResponse
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();
  const jsonMatch = cleanedJson.match(/\{[\s\S]*\}/);

  if (jsonMatch) {
    try {
      const result = JSON.parse(jsonMatch[0]);
      // Light post-processing: correct common misspellings
      const correctName = (name: string) => normalizeDestinationName(name);
      if (result.destination && typeof result.destination === 'string') {
        result.destination = correctName(result.destination);
      }
      if (Array.isArray(result.destinations)) {
        result.destinations = result.destinations.map((d: string) => correctName(d));
      }
      console.log(`[planner] Extracted trip details:`, result);
      return result;
    } catch (e) {
      console.warn("[planner] Failed to parse trip details JSON from Groq.", e);
    }
  }
  return {};
}

export function emitItineraryUpdate(it: Itinerary): WsEvent {
  return { type: "itinerary.update", data: it };
}

// Heuristic normalizer for common destination misspellings/variants
function normalizeDestinationName(name: string): string {
  const n = (name || "").trim();
  if (!n) return n;
  const lower = n.toLowerCase();
  const map: Record<string, string> = {
    delgi: "Delhi",
    delhi: "Delhi",
    mumbay: "Mumbai",
    bombay: "Mumbai",
    cochin: "Kochi",
    kochi: "Kochi",
    banglore: "Bangalore",
    bengaluru: "Bangalore",
    ooti: "Ooty",
    udhagamandalam: "Ooty",
    kunoor: "Coonoor",
    coonur: "Coonoor",
    kodaikannal: "Kodaikanal",
    udaypur: "Udaipur",
  };
  if (map[lower]) return map[lower];
  // Title-case fallback
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

// Helper function to extract conversation context
function extractConversationContext(history: Message[]) {
  const context = {
    previousDestinations: new Set<string>(),
    travelPreferences: new Set<string>(),
    budgetPreferences: new Set<string>(),
    previousDurations: new Set<number>(),
    groupType: undefined as string | undefined,
    preferredOrigin: undefined as string | undefined,
  };

  if (!history || history.length === 0) {
    return {
      previousDestinations: [],
      travelPreferences: [],
      budgetPreferences: [],
      previousDurations: [],
      groupType: undefined,
      preferredOrigin: undefined,
    };
  }

  // Analyze recent conversation history (last 15 messages)
  const recentHistory = history.slice(-15);

  recentHistory.forEach((msg) => {
    const content = msg.content.toLowerCase();

    // Extract destinations mentioned
    const destPatterns = [
      /(?:to|visit|visiting|in|plan.*trip.*to)\s+([A-Za-z][A-Za-z\s]{2,20}?)(?:\s|$|[,.!?])/g,
      /([A-Za-z][A-Za-z\s]{2,20}?)\s+(?:trip|travel|vacation|itinerary)/g,
    ];

    destPatterns.forEach((pattern) => {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const dest = match[1].trim();
        if (dest.length > 2 && dest.length < 20 && !isCommonWord(dest)) {
          context.previousDestinations.add(
            dest.charAt(0).toUpperCase() + dest.slice(1),
          );
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
