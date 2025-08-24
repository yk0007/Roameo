import type { Itinerary, WsEvent, POI, Activity } from "../types/schemas.js";
import { GeminiClient } from "../tools/gemini.js";
import { GoogleMapsClient } from "../tools/maps.js";

export async function plannerAgent(
  _ctx: { origin?: string; destination?: string; days?: number },
  message: string
): Promise<{ chatResponse: string; itinerary: Itinerary; destination: string; days: number } | null> {
    const extractedDetails = await extractTripDetails(message);
  const destination = extractedDetails.destination || _ctx.destination;
  const days = extractedDetails.days || _ctx.days;
  const origin = _ctx.origin || "Current location";
  const maps = new GoogleMapsClient();

  try {
    // --- Step 1: Generate the conversational markdown response --- //
    const gemini = new GeminiClient({ model: "flash" });

    let chatPrompt;
    if (!days) {
      chatPrompt = `You are a friendly travel planning assistant. The user wants to plan a trip to ${destination}. Ask them for the number of days they want to stay. Be enthusiastic and suggest some popular attraction types like coffee plantations, waterfalls, and viewpoints.`;
    } else {
    chatPrompt = `You are an expert travel planning assistant. Your goal is to create a beautifully formatted travel itinerary in **Markdown** for a ${days}-day trip to ${destination} from ${origin}.

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
* **Accommodation:** €[X]-€[Y] per night
* **Food:** €[X]-€[Y] per day
* **Transport:** €[X]-€[Y] total
* **Activities:** €[X]-€[Y] total
* **Miscellaneous:** €[X]-€[Y] total
* **Total:** €[X]-€[Y] for ${days} days

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
    // If we don't have enough info for an itinerary, return just the chat response.
    if (!destination || !days) {
      return {
        chatResponse,
        itinerary: { origin, destination: destination || "", days: 0, daysPlan: [] },
        destination: destination || "",
        days: days || 0,
      };
    }

    // --- Step 2: Fetch real POIs to build a structured itinerary --- //
        const [attractions, restaurants, stays] = await Promise.all([
      maps.searchPlaces({ q: `tourist attractions in ${destination}` }, "attraction").catch(() => []),
      maps.searchPlaces({ q: `restaurants in ${destination}` }, "restaurant").catch(() => []),
      maps.searchPlaces({ q: `hotels in ${destination}` }, "stay").catch(() => []),
    ]);

        const itinerary = await createStructuredItinerary(chatResponse, { ..._ctx, destination, days }, { attractions, restaurants, stays });

    return { chatResponse, itinerary, destination, days };

  } catch (e: any) {
    console.warn("[planner] Gemini or Maps failed:", e);
    if (e.message && e.message.includes("429")) {
      return {
        chatResponse: "It looks like I'm very popular right now! I've hit my request limit. Please try again in a little while.",
        itinerary: { origin, destination: destination || "", days: days || 0, daysPlan: [] },
        destination: destination || "",
        days: days || 0,
      };
    }
    
    // Return fallback itinerary instead of null
    console.log("[planner] Using fallback itinerary due to error.");
    return {
      chatResponse: "I ran into a little trouble creating your itinerary, but here's a sample to get you started!",
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
  const jsonPrompt = `Based on the following itinerary description, and a list of available POIs, create a structured JSON object representing the plan. Pick the most relevant POIs for each activity.

Description:
${description}

Available POIs:
- Attractions: ${JSON.stringify(pois.attractions, null, 2)}
- Restaurants: ${JSON.stringify(pois.restaurants, null, 2)}
- Stays: ${JSON.stringify(pois.stays, null, 2)}

JSON Structure:
{
  "origin": "${ctx.origin || 'Current location'}",
  "destination": "${ctx.destination}",
  "days": ${ctx.days},
  "daysPlan": [
    {
      "day": 1,
      "date": "YYYY-MM-DD",
      "title": "Title for Day 1",
      "activities": [
        {"name": "Activity name", "start": "HH:MM", "end": "HH:MM", "location": "Address or area", "poiId": "poi_id_from_list"}
      ],
      "accommodation": {"name": "Hotel name", "checkIn": "HH:MM", "poiId": "poi_id_from_list"}
    }
  ]
}

Respond with ONLY the JSON object.`;

  const jsonResponse = await gemini.chat(jsonPrompt);
  const cleanedJson = jsonResponse.replace(/^```json\s*/i, "").replace(/\s*```\s*$/i, "").trim();
  const jsonMatch = cleanedJson.match(/\{[\s\S]*\}/);

  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]) as Itinerary;
      // Enrich activities with full POI data
      parsed.daysPlan.forEach((day) => {
        day.activities.forEach((act) => {
          const allPois = [...pois.attractions, ...pois.restaurants, ...pois.stays];
          const poi = allPois.find((p) => p.id === act.poiId);
          if (poi) {
            act.name = poi.name;
            act.location = poi.address;
            act.photoUrl = poi.photoUrl; // This is the key part
            act.rating = poi.rating;
            act.lat = poi.lat;
            act.lng = poi.lng;
          }
        });
      });
      return parsed;
    } catch (e) {
      console.warn("[planner] Failed to parse JSON from Gemini, falling back.", e);
    }
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

async function extractTripDetails(message: string): Promise<{ destination?: string; days?: number }> {
  const gemini = new GeminiClient({ model: "flash" });
  const prompt = `Extract the destination and number of days from the user's message.

User message: "${message}"

Respond with ONLY a JSON object with "destination" and "days" keys. If a value is not present, omit the key.
For example:
{
  "destination": "Rameswaram",
  "days": 3
}`;
  const jsonResponse = await gemini.chat(prompt);
  const cleanedJson = jsonResponse.replace(/^```json\s*/i, "").replace(/\s*```\s*$/i, "").trim();
  const jsonMatch = cleanedJson.match(/\{[\s\S]*\}/);

  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]);
    } catch (e) {
      console.warn("[planner] Failed to parse trip details JSON from Gemini.", e);
    }
  }
  return {};
}

export function emitItineraryUpdate(it: Itinerary): WsEvent {
  return { type: "itinerary.update", data: it };
}
