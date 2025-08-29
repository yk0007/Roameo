import type { Itinerary, WsEvent, POI, Activity } from "../types/schemas.js";
import { GeminiClient } from "../tools/gemini.js";
import { GoogleMapsClient } from "../tools/maps.js";

export async function plannerAgent(
  _ctx: { origin?: string; destination?: string; destinations?: string[]; days?: number },
  message: string
): Promise<{ chatResponse: string; itinerary: Itinerary; destination: string; destinations?: string[]; days: number } | null> {
    const extractedDetails = await extractTripDetails(message);
  const destination = extractedDetails.destination || _ctx.destination;
  const destinations = extractedDetails.destinations || _ctx.destinations;
  const days = extractedDetails.days || _ctx.days;
  const origin = _ctx.origin || "Current location";
  const maps = new GoogleMapsClient();

  try {
    // --- Step 1: Generate the conversational markdown response --- //
    const gemini = new GeminiClient({ model: "flash" });

    let chatPrompt;
    const finalDestinations = destinations || (destination ? [destination] : []);
    const destinationText = finalDestinations.length > 1 
      ? finalDestinations.join(", ") 
      : (destination || finalDestinations[0] || "your destination");

    if (!days) {
      chatPrompt = `You are a friendly travel planning assistant. The user wants to plan a trip to ${destinationText}. Ask them for the number of days they want to stay. Be enthusiastic and suggest some popular attraction types like coffee plantations, waterfalls, and viewpoints.`;
    } else {
    chatPrompt = `You are an expert travel planning assistant. Your goal is to create a beautifully formatted travel itinerary in **Markdown** for a ${days}-day trip to ${destinationText} from ${origin}.

**CRITICAL**: You MUST create the itinerary for "${destinationText}" ONLY. Do NOT substitute with any other destination like Goa, Mumbai, or Delhi. The user specifically requested "${destinationText}".

${finalDestinations.length > 1 ? `**MULTI-DESTINATION TRIP**: This is a multi-destination trip covering ${finalDestinations.join(", ")}. Allocate days appropriately across destinations and include travel time between locations.` : ''}

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

    if (!chatResponse?.trim() || chatResponse.startsWith("[gemini:")) {
      console.warn(`[planner] Gemini returned an empty or invalid response: ${chatResponse}. Using fallback.`);
      return {
        chatResponse: "I'm having a little trouble generating that itinerary right now. Could you try rephrasing your request?",
        itinerary: { origin, destination: destination || "", days: days || 0, daysPlan: [] },
        destination: destination || "",
        days: days || 0,
      };
    }

    // If we don't have enough info for an itinerary, return just the chat response.
    if ((!destination && !destinations) || !days) {
      return {
        chatResponse,
        itinerary: { origin, destination: destination || "", days: 0, daysPlan: [] },
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
    console.log(`[planner] Fetching POIs for ${limitedDestinations.length} destinations:`, limitedDestinations);
    
    for (const dest of limitedDestinations) {
      poiPromises.push(
        maps.searchPlaces({ q: `tourist attractions in ${dest}` }, "attraction")
          .catch((e) => { console.warn(`[planner] Attractions search failed for ${dest}:`, e.message); return []; }),
        maps.searchPlaces({ q: `restaurants in ${dest}` }, "restaurant")
          .catch((e) => { console.warn(`[planner] Restaurants search failed for ${dest}:`, e.message); return []; }),
        maps.searchPlaces({ q: `hotels in ${dest}` }, "stay")
          .catch((e) => { console.warn(`[planner] Hotels search failed for ${dest}:`, e.message); return []; })
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
    
    console.log(`[planner] Found ${attractions.length} attractions, ${restaurants.length} restaurants, ${stays.length} stays`);

    const itinerary = await createStructuredItinerary(chatResponse, { ..._ctx, destination, days }, { attractions, restaurants, stays });

    return { chatResponse, itinerary, destination: destination || allDestinations[0], destinations, days };

  } catch (e: any) {
    console.warn("[planner] Gemini or Maps failed:", e);
    
    // Handle specific error types
    if (e.message && e.message.includes("429")) {
      return {
        chatResponse: "It looks like I'm very popular right now! I've hit my request limit. Please try again in a little while.",
        itinerary: { origin, destination: destination || "", days: days || 0, daysPlan: [] },
        destination: destination || "",
        days: days || 0,
      };
    }
    
    if (e.message && e.message.includes("timeout")) {
      return {
        chatResponse: `I'm taking a bit longer to plan your ${destination || 'trip'}. Let me give you a quick overview while I work on the details!`,
        itinerary: createDummyItinerary({ ..._ctx, destination, days }),
        destination: destination || "",
        days: days || 0,
      };
    }
    
    // Return fallback itinerary instead of null
    console.log("[planner] Using fallback itinerary due to error:", e.message);
    return {
      chatResponse: `I ran into a little trouble creating your detailed ${destination || 'itinerary'}, but here's a sample to get you started!`,
      itinerary: createDummyItinerary({ ..._ctx, destination, days }),
      destination: destination || "",
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
  const jsonPrompt = `Create a structured JSON itinerary based on the description and available POIs. Follow this EXACT format:

Description: ${description}

Available POIs (use these IDs in your response):
Attractions: ${JSON.stringify(pois.attractions.slice(0, 15).map(p => ({id: p.id, name: p.name, address: p.address})), null, 2)}
Restaurants: ${JSON.stringify(pois.restaurants.slice(0, 10).map(p => ({id: p.id, name: p.name, address: p.address})), null, 2)}
Stays: ${JSON.stringify(pois.stays.slice(0, 5).map(p => ({id: p.id, name: p.name, address: p.address})), null, 2)}

IMPORTANT CONSTRAINTS:
- Use ONLY POI IDs from the lists above
- Maximum ${ctx.days || 3} days
- Each day should have 3-5 activities
- Use realistic time slots (09:00-21:00)
- Include accommodation for multi-day trips

Respond with ONLY this JSON structure (no additional text):
{
  "origin": "${ctx.origin || 'Current location'}",
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
    const jsonResponse = await gemini.chat(jsonPrompt);
    console.log("[planner] Raw JSON response length:", jsonResponse.length);
    
    // Better JSON extraction and cleaning
    let cleanedJson = jsonResponse.replace(/^```json\s*/i, "").replace(/\s*```\s*$/i, "").trim();
    
    // Remove any trailing content after the JSON object
    const jsonStart = cleanedJson.indexOf('{');
    const jsonEnd = cleanedJson.lastIndexOf('}');
    
    if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
      cleanedJson = cleanedJson.substring(jsonStart, jsonEnd + 1);
    }
    
    console.log("[planner] Cleaned JSON length:", cleanedJson.length);
    
    // Validate JSON structure before parsing
    if (!cleanedJson.startsWith('{') || !cleanedJson.endsWith('}')) {
      throw new Error('Invalid JSON structure: missing braces');
    }
    
    const parsed = JSON.parse(cleanedJson) as Itinerary;
    
    // Validate required fields
    if (!parsed.daysPlan || !Array.isArray(parsed.daysPlan)) {
      throw new Error('Invalid itinerary structure: missing or invalid daysPlan');
    }
    
    // Enrich activities with full POI data
    parsed.daysPlan.forEach((day) => {
      if (day.activities && Array.isArray(day.activities)) {
        day.activities.forEach((act) => {
          const allPois = [...pois.attractions, ...pois.restaurants, ...pois.stays];
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
    
    console.log("[planner] Successfully created structured itinerary with", parsed.daysPlan.length, "days");
    return parsed;
  } catch (e) {
    console.log("[planner] Failed to create structured itinerary, using fallback.", e);
  }
  return createDummyItinerary(ctx);
}

// --- Fallback Itinerary Generator --- //
function createDummyItinerary(ctx: { origin?: string; destination?: string; days?: number }): Itinerary {
  const today = new Date();
  const totalDays = ctx.days || 1;
  const daysPlan = Array.from({ length: totalDays }, (_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() + i);
    const dateStr = d.toISOString().slice(0, 10);
    const baseTitle = i === 0 ? `Arrival in ${ctx.destination}` : i === totalDays - 1 ? `Farewell ${ctx.destination}` : `Explore ${ctx.destination}`;
    const activities: Activity[] = [
      { name: `Breakfast near ${ctx.destination}`, start: "09:00", end: "10:00" },
      { name: `Top sight #${i + 1}`, start: "11:00", end: "12:30" },
      { name: "Local lunch", start: "13:00", end: "14:00" },
      { name: "Scenic walk", start: "16:00", end: "17:30" },
      { name: "Dinner", start: "19:30", end: "21:00" },
    ];
    return { day: i + 1, date: dateStr, title: baseTitle, activities };
  });
  return { origin: ctx.origin || "", destination: ctx.destination!, days: ctx.days!, daysPlan };
}

async function extractTripDetails(message: string): Promise<{ destination?: string; destinations?: string[]; days?: number }> {
  const gemini = new GeminiClient({ model: "flash" });
  const prompt = `Extract the destination(s) and number of days from the user's message. Be very precise with destination names.

User message: "${message}"

CRITICAL: Extract the EXACT destination(s) mentioned by the user. Do NOT change or substitute destination names.
- If user says "ooty", extract "Ooty"
- If user says "coonoor", extract "Coonoor" 
- If user says "kodaikanal", extract "Kodaikanal"
- Do NOT substitute with other destinations like Goa, Mumbai, etc.

For MULTIPLE destinations:
- If user mentions multiple places like "ooty and coonoor" or "kerala, goa and rajasthan", extract all destinations
- Use "destinations" array for multiple places
- Use "destination" for single place

Respond with ONLY a JSON object with "destination", "destinations", and "days" keys. If a value is not present, omit the key.

Examples:
Single destination: {"destination": "Ooty", "days": 3}
Multiple destinations: {"destinations": ["Kerala", "Goa", "Rajasthan"], "days": 10}
Multiple with days per location: {"destinations": ["Ooty", "Coonoor"], "days": 5}`;
  const jsonResponse = await gemini.chat(prompt);
  console.log(`[planner] Trip details extraction response: ${jsonResponse}`);
  const cleanedJson = jsonResponse.replace(/^```json\s*/i, "").replace(/\s*```\s*$/i, "").trim();
  const jsonMatch = cleanedJson.match(/\{[\s\S]*\}/);

  if (jsonMatch) {
    try {
      const result = JSON.parse(jsonMatch[0]);
      console.log(`[planner] Extracted trip details:`, result);
      return result;
    } catch (e) {
      console.warn("[planner] Failed to parse trip details JSON from Gemini.", e);
    }
  }
  return {};
}

export function emitItineraryUpdate(it: Itinerary): WsEvent {
  return { type: "itinerary.update", data: it };
}
