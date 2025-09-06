import { randomUUID } from "crypto";
import type { WsEvent, TripContext } from "../types/schemas.js";
import type { Message } from "../db/types.js";
import { StateGraph, END, StateGraphArgs } from "@langchain/langgraph";
import { plannerAgent, emitItineraryUpdate } from "../agents/planner.js";
import { poiAgent } from "../agents/poi.js";
import { mapAgent } from "../agents/map.js";
import { chatAgent } from "../agents/chat.js";
import { intentAgent } from "../agents/intent.js";
import { generateSessionTitle } from "../agents/title.js";
import { destinationExtractionAgent, immediatePoiSearchAgent, generateDestinationChatResponse } from "../agents/destination.js";
import { DestinationImageService } from "../tools/destination-images.js";

export type GraphInput = {
  sessionId: string;
  message: string;
  trip: Partial<TripContext>;
};

interface State {
  messages: Message[];
  input: GraphInput;
  events: WsEvent[];
  route: "planner" | "destination_search" | "chat";
  extractedDestination?: {
    destination?: string;
    destinations?: string[];
    days?: number;
    origin?: string;
    hasDestination: boolean;
  };
}

const graphState: StateGraphArgs<State>["channels"] = {
  messages: {
    reducer: (x, y) => x.concat(y),
    default: () => [],
  },
  input: {
    reducer: (x, y) => y ?? x,
    default: () => ({ sessionId: "", message: "", trip: {}, messages: [] }),
  },
  events: {
    reducer: (x, y) => x.concat(y),
    default: () => [],
  },
  route: {
    reducer: (x, y) => y ?? x,
    default: () => "chat",
  },
  extractedDestination: {
    reducer: (x, y) => y ?? x,
    default: () => undefined,
  },
};

const graph = new StateGraph<State>({ channels: graphState })
  .addNode("router", async (state: State) => {
    const { message } = state.input;
    const intent = await intentAgent(message, state.messages);
    console.log(`[router] Intent detected: ${intent} for message: "${message}"`);
    
    // Emit intent detection event for planning and destination search intents
    const events: WsEvent[] = [];
    if (intent === "PLAN_TRIP" || intent === "DESTINATION_SEARCH") {
      events.push({
        type: "intent.detected",
        data: { intent, message }
      });
    }
    
    if (intent === "PLAN_TRIP") {
      return { route: "planner", events };
    } else if (intent === "DESTINATION_SEARCH") {
      return { route: "destination_search", events };
    }
    return { route: "chat", events };
  })
  .addNode("planner", async (state: State) => {
    const { trip, message } = state.input;
    
    // Emit planning start event
    const plannerEvents: WsEvent[] = [{
      type: "planning.status",
      data: { status: "Analyzing your request..." }
    }];
    
    // Clarify intent when a different destination is mentioned without additive phrasing
    const existingDests: string[] = (trip.destinations && trip.destinations.length)
      ? (trip.destinations as string[])
      : (trip.destination ? [trip.destination] : []);
    let extracted;
    try {
      extracted = await destinationExtractionAgent(message, state.messages);
    } catch {}

    // Detect additive phrasing
    const additiveIntentInline = /\b(add|also|plus|and\s+then|plan\s+also|as\s+well)\b/i.test(message);

    if (extracted && extracted.hasDestination) {
      const newDests: string[] = extracted.destinations && extracted.destinations.length
        ? extracted.destinations
        : (extracted.destination ? [extracted.destination] : []);

      const normalizedSet = (arr: string[]) => new Set(arr.map((d) => (d || "").toLowerCase().trim()));
      const existingSet = normalizedSet(existingDests);
      const newSet = normalizedSet(newDests);
      const hasDifferent = Array.from(newSet).some((d) => !existingSet.has(d));

      if (existingDests.length > 0 && hasDifferent && !additiveIntentInline) {
        // Ask the user to choose whether to add or start fresh (rendered as clickable options)
        const nd = newDests.join(", ");
        const ed = existingDests.join(", ");
        const clarification = `I see a different destination ("${nd}") from your current trip (${ed}).\n\n`+
`Would you like me to:\n\n`+
`🤔 Got More Questions?\n`+
`"Add to current trip"\n`+
`"Start a new trip"\n`+
`"Add ${nd} for 2 days after current plan"\n`+
`"Start ${nd} as a fresh ${extracted?.days || 2}-day trip"\n`+
`"Replace current trip with ${nd}"`;

        return {
          events: [
            // End any active planning animation since we're waiting for user's choice
            { type: "planning.status", data: { status: "Done" } },
            {
              type: "chat.append",
              data: {
                id: randomUUID(),
                role: "assistant",
                content: clarification,
                createdAt: new Date().toISOString(),
              },
            },
          ],
        };
      }
    }

    const res = await plannerAgent(trip, message, state.messages);
    if (!res) {
      return {
        events: [
          {
            type: "chat.append",
            data: {
              id: randomUUID(),
              role: "assistant",
              content: "Something went wrong, please try again.",
              createdAt: new Date().toISOString(),
            },
          },
        ],
      };
    }

    // Heuristic: detect additive planning intent ("add", "also", "plus", "add it", "add to")
    const additiveIntent = /\b(add|also|plus|and\s+then|plan\s+also|add\s+it|add\s+to)\b/i.test(message);

    // Merge itinerary if additive intent and we have an existing itinerary
    let itineraryToEmit = res.itinerary;
    if (additiveIntent && (trip as any)?.itinerary?.daysPlan?.length) {
      try {
        const prev = (trip as any).itinerary as any;
        const prevDays = prev.daysPlan || [];
        const offset = prevDays.length;
        const shiftedDays = (res.itinerary.daysPlan || []).map((d: any, idx: number) => ({
          ...d,
          day: offset + (idx + 1),
        }));
        itineraryToEmit = {
          origin: prev.origin || res.itinerary.origin,
          destination: res.itinerary.destination, // keep latest destination label for summary
          days: (prev.days || prevDays.length) + (res.itinerary.days || shiftedDays.length),
          daysPlan: [...prevDays, ...shiftedDays],
        } as any;
      } catch (e) {
        console.warn("[planner] Failed to merge itineraries, falling back to latest only:", (e as any)?.message || e);
        itineraryToEmit = res.itinerary;
      }
    }

    // Merge destinations list
    const mergedDestinations = Array.from(
      new Set([
        ...((trip.destinations as string[] | undefined) || (trip.destination ? [trip.destination] : [])),
        ...((res.destinations as string[] | undefined) || (res.destination ? [res.destination] : [])),
      ].filter(Boolean) as string[])
    );

    // Build/merge itinerary segments for multi-destination UX
    let itinerarySegments: Array<{ destination: string; startDay: number; endDay: number }> | undefined;
    try {
      const prevItin = (trip as any)?.itinerary;
      const prevSegs: Array<{ destination: string; startDay: number; endDay: number }> | undefined = (trip as any)?.itinerarySegments;
      if (additiveIntent && prevItin?.daysPlan?.length && res.itinerary?.days) {
        const prevLen = prevItin.daysPlan.length;
        const newLen = res.itinerary.days;
        const baseSegs = prevSegs && prevSegs.length
          ? prevSegs
          : [{ destination: (trip.destination as string) || ((trip.destinations as string[]|undefined)?.[0] || res.destination), startDay: 1, endDay: prevLen }];
        itinerarySegments = [
          ...baseSegs,
          { destination: res.destination, startDay: prevLen + 1, endDay: prevLen + newLen },
        ];
      } else if (res.itinerary?.days && res.destination) {
        itinerarySegments = [{ destination: res.destination, startDay: 1, endDay: res.itinerary.days }];
      }
    } catch {}

    const updatedTrip = { 
      ...trip, 
      destination: res.destination, 
      destinations: mergedDestinations.length ? mergedDestinations : res.destinations,
      days: itineraryToEmit?.days || res.days,
      destinationImageUrl: res.destinationImageUrl,
      itinerary: itineraryToEmit || res.itinerary,
      itinerarySegments,
    };
    
    // Generate title with fallback handling
    let title: string;
    try {
      title = await generateSessionTitle({
        message,
        origin: updatedTrip.origin,
        destination: res.destination,
        days: res.days,
        existingTitle: trip.title,
      });
    } catch (error: any) {
      console.log(`[planner] Title generation failed: ${error?.message || error}, using fallback`);
      // Create fallback title
      const sessionSuffix = Math.random().toString(36).substring(2, 5).toUpperCase();
      title = res.destination ? `✨ ${res.destination} Adventure #${sessionSuffix}` : `✨ Dream Trip #${sessionSuffix}`;
    }

    plannerEvents.push(
      {
        type: "chat.append",
        data: {
          id: randomUUID(),
          role: "assistant",
          content: res.chatResponse,
          createdAt: new Date().toISOString(),
        },
      },
      // Emit navbar update with trip details
      {
        type: "navbar.update",
        data: {
          destination: res.destination,
          destinations: res.destinations,
          days: res.days,
          title,
          destinationImageUrl: res.destinationImageUrl,
          itinerarySegments,
        },
      },
    );

    if ((itineraryToEmit?.daysPlan?.length || res.itinerary.daysPlan.length) > 0) {
      // Always update itinerary panel when a valid plan is generated
      plannerEvents.push(emitItineraryUpdate(itineraryToEmit || res.itinerary));

      // Handle POI search for multiple destinations
      const searchDestination = res.destinations && res.destinations.length > 0 
        ? res.destinations[0] // Use first destination for POI search
        : res.destination;

      // Emit search status
      plannerEvents.push({
        type: "search.status",
        data: { status: `Finding places in ${searchDestination}...` }
      });

      let emittedNewMap = false;
      const poiEvt = await poiAgent({ destination: searchDestination });
      if (poiEvt) {
        plannerEvents.push(poiEvt);
        if (poiEvt.type === "search.results") {
          let pois = [...poiEvt.data.stays, ...poiEvt.data.restaurants, ...poiEvt.data.attractions];

          // If additive planning and we have previous map data, merge POIs
          if (additiveIntent && (trip as any)?.mapData?.pois?.length) {
            const prevPois: any[] = (trip as any).mapData.pois || [];
            const byId = new Map<string, any>();
            for (const p of prevPois) byId.set(p.id, p);
            for (const p of pois) byId.set(p.id, p);
            pois = Array.from(byId.values());
          }
          if (pois.length > 0) {
            // Emit map status
            plannerEvents.push({
              type: "map.status",
              data: { status: "Calculating routes and updating map..." }
            });
            plannerEvents.push(await mapAgent(pois));
            emittedNewMap = true;
          }
        }
      }

      // Fallback: if we couldn't emit a new map update, keep previous map markers
      if (!emittedNewMap) {
        const prevMap = (trip as any)?.mapData;
        if (prevMap && prevMap.pois && prevMap.pois.length > 0) {
          plannerEvents.push({ type: "map.update", data: prevMap });
        }
      }
    }

    // Signal planning done so frontend can stop inline animation
    plannerEvents.push({ type: "planning.status", data: { status: "Done" } });
    return { events: plannerEvents, input: { ...state.input, trip: updatedTrip } };
  })
  .addNode("destination_search", async (state: State) => {
    const { message, trip } = state.input;
    
    const events: WsEvent[] = [];
    
    // OPTIMIZATION: Run destination extraction and intent re-classification in parallel
    // This allows us to be more selective about whether to proceed with destination search
    const [extraction, reClassifiedIntent] = await Promise.all([
      destinationExtractionAgent(message, state.messages),
      intentAgent(message, state.messages) // Re-classify with conversation context
    ]);
    
    console.log(`[destination_search] Parallel results - Extracted:`, extraction);
    console.log(`[destination_search] Re-classified intent:`, reClassifiedIntent);
    
    // Step 1: Check if this should actually be handled as general chat
    // If re-classification suggests CHAT, or if no valid destination is found
    const isActualDestinationSearch = extraction.hasDestination && 
      (extraction.destination || (extraction.destinations && extraction.destinations.length > 0)) &&
      reClassifiedIntent !== "CHAT";
    
    if (!isActualDestinationSearch) {
      // This should be treated as general chat - provide immediate response and exit
      console.log(`[destination_search] Routing to chat - hasDestination: ${extraction.hasDestination}, reClassified: ${reClassifiedIntent}`);
      const chatResponse = await chatAgent(message, state.messages);
      events.push({
        type: "chat.append",
        data: {
          id: randomUUID(),
          role: "assistant",
          content: chatResponse,
          createdAt: new Date().toISOString(),
        },
      });
      
      return { events, extractedDestination: extraction };
    }
    
    // Check if this is a clarification response (user chose an option after our question)
    const clarificationResponses = [
      /add\s+to\s+current\s+trip/i,
      /start\s+a?\s*new\s+trip/i,
      /add\s+.*\s+for\s+\d+\s+days/i,
      /start\s+.*\s+as\s+a?\s*fresh/i,
      /replace\s+current\s+trip/i,
      /^add\s+it$/i  // Handle simple "add it" response
    ];
    
    const isClarificationResponse = clarificationResponses.some(pattern => pattern.test(message));
    
    if (isClarificationResponse) {
      // User has made their choice, proceed with planning based on their selection
      const isAdditive = /add\s+to\s+current|add\s+.*\s+for\s+\d+\s+days|^add\s+it$/i.test(message);
      
      if (isAdditive) {
        // Extract days from the message if specified, otherwise default to 2-3 days
        const daysMatch = message.match(/(\d+)\s+days?/i);
        const specifiedDays = daysMatch ? parseInt(daysMatch[1]) : (extraction?.days || 3);
        
        // Create trip context for additive planning
        // For "add it" responses, we need to get the destination from the previous clarification context
        // Look for the destination mentioned in the previous assistant message
        const lastAssistantMessage = state.messages.filter(m => m.role === 'assistant').pop();
        let newDestination = 'Delhi'; // fallback
        
        if (lastAssistantMessage?.content) {
          // Extract destination from clarification message like "I see a different destination ("Delhi")"
          const destMatch = lastAssistantMessage.content.match(/different destination \("([^"]+)"\)/);
          if (destMatch) {
            newDestination = destMatch[1];
          }
        }
        
        // Fallback to extraction if available
        if (!newDestination || newDestination === 'Delhi') {
          newDestination = extraction.destination || (extraction.destinations ? extraction.destinations[0] : 'Delhi');
        }
        
        const planningTrip = {
          ...trip,
          destination: newDestination,
          destinations: [...(trip.destinations as string[] || [trip.destination].filter(Boolean)), newDestination],
          days: specifiedDays,
          origin: extraction.origin || trip.origin,
        };
        
        // Use planner agent with additive context
        const res = await plannerAgent(planningTrip, `Plan ${specifiedDays} days in ${newDestination} to add to existing trip`, state.messages);
        
        if (!res) {
          return {
            events: [
              {
                type: "chat.append",
                data: {
                  id: randomUUID(),
                  role: "assistant", 
                  content: "I had trouble planning the addition to your trip. Could you try again?",
                  createdAt: new Date().toISOString(),
                },
              },
            ],
          };
        }
        
        // Merge with existing itinerary
        const existingItinerary = (trip as any)?.itinerary;
        if (existingItinerary?.daysPlan?.length) {
          const offset = existingItinerary.daysPlan.length;
          const newDays = (res.itinerary.daysPlan || []).map((d: any, idx: number) => ({
            ...d,
            day: offset + idx + 1,
          }));
          
          const mergedItinerary = {
            ...existingItinerary,
            days: existingItinerary.days + res.days,
            daysPlan: [...existingItinerary.daysPlan, ...newDays],
          };
          
          const updatedTrip = {
            ...trip,
            destinations: planningTrip.destinations,
            days: mergedItinerary.days,
            itinerary: mergedItinerary,
          };
          
          return {
            events: [
              {
                type: "chat.append",
                data: {
                  id: randomUUID(),
                  role: "assistant",
                  content: `Perfect! I've added ${specifiedDays} days in ${newDestination} to your existing trip. ${res.chatResponse}`,
                  createdAt: new Date().toISOString(),
                },
              },
              { type: "itinerary.update", data: mergedItinerary },
              {
                type: "navbar.update", 
                data: {
                  destinations: planningTrip.destinations,
                  days: mergedItinerary.days,
                },
              },
            ],
            input: { ...state.input, trip: updatedTrip },
          };
        }
      } else {
        // Start new trip - replace current trip
        const planningTrip = {
          sessionId: trip.sessionId,
          destination: extraction.destination || (extraction.destinations ? extraction.destinations[0] : undefined),
          destinations: extraction.destinations || (extraction.destination ? [extraction.destination] : undefined),
          days: extraction.days || 3,
          origin: extraction.origin,
        };
        
        const res = await plannerAgent(planningTrip, message, state.messages);
        
        if (!res) {
          return {
            events: [
              {
                type: "chat.append",
                data: {
                  id: randomUUID(),
                  role: "assistant",
                  content: "I had trouble planning your new trip. Could you try again?",
                  createdAt: new Date().toISOString(),
                },
              },
            ],
          };
        }
        
        const updatedTrip = {
          ...trip,
          destination: res.destination,
          destinations: res.destinations,
          days: res.days,
          itinerary: res.itinerary,
        };
        
        return {
          events: [
            {
              type: "chat.append",
              data: {
                id: randomUUID(),
                role: "assistant",
                content: res.chatResponse,
                createdAt: new Date().toISOString(),
              },
            },
            { type: "itinerary.update", data: res.itinerary },
            {
              type: "navbar.update",
              data: {
                destination: res.destination,
                destinations: res.destinations,
                days: res.days,
              },
            },
          ],
          input: { ...state.input, trip: updatedTrip },
        };
      }
    }
    
    // Step 2: If re-classification suggests this should be planning, route to planner
    if (reClassifiedIntent === "PLAN_TRIP") {
      // Emit planning status BEFORE chat response to show inline animation first
      events.push({ type: "planning.status", data: { status: "Creating your itinerary..." } });
      
      const quickResponse = await chatAgent(message, state.messages);
      events.push({
        type: "chat.append",
        data: {
          id: randomUUID(),
          role: "assistant",
          content: quickResponse,
          createdAt: new Date().toISOString(),
        },
      });
      
      // Route to planner for full itinerary generation
      return { route: "planner", events, extractedDestination: extraction };
    }
    
    // Step 3: Proceed with destination search flow
    // Run POI search in parallel for better performance
    const poiSearchPromise = immediatePoiSearchAgent(extraction);
    
    // Generate AI-powered conversational response with POI formatting + ask for missing info
    const destinationChatResponse = await generateDestinationChatResponse(extraction, message, {});
    
    // Wait for POI search to complete
    const poiSearchEvent = await poiSearchPromise;
    const poiEvents: WsEvent[] = poiSearchEvent ? [poiSearchEvent] : [];
    
    // Extract POI data for response generation
    const poiData = poiSearchEvent ? {
      stays: poiSearchEvent.data.stays || [],
      restaurants: poiSearchEvent.data.restaurants || [],
      attractions: poiSearchEvent.data.attractions || []
    } : { stays: [], restaurants: [], attractions: [] };
    
    // Generate follow-up questions for missing information
    let followUpResponse = "";
    const missingInfo = [];
    if (!extraction.days) missingInfo.push("duration");
    if (!extraction.origin && !trip.origin) missingInfo.push("origin");
    
    if (missingInfo.length > 0) {
      followUpResponse = `\n\nTo create a perfect itinerary for you, could you please tell me:\n`;
      if (missingInfo.includes("duration")) {
        followUpResponse += `• How many days will you be visiting?\n`;
      }
      if (missingInfo.includes("origin")) {
        followUpResponse += `• Where will you be traveling from?\n`;
      }
    }
    
    // Add destination-specific chat response only if we have additional info
    if (followUpResponse || destinationChatResponse.length > 50) {
      events.push({
        type: "chat.append",
        data: {
          id: randomUUID(),
          role: "assistant",
          content: destinationChatResponse + followUpResponse,
          createdAt: new Date().toISOString(),
        },
      });
    }
    
    // Update navbar with destination info if available (and update image url dynamically)
    const destinations = extraction.destinations || (extraction.destination ? [extraction.destination] : []);
    if (extraction.hasDestination && destinations.length > 0) {
      let destinationImageUrl: string | undefined;
      try {
        const imageService = new DestinationImageService();
        const img = await imageService.getDestinationImageForTrip(destinations);
        destinationImageUrl = img.imageUrl;
      } catch (e) {
        console.warn("[destination_search] Failed to fetch destination image:", (e as any)?.message || e);
      }

      events.push({
        type: "navbar.update",
        data: {
          destination: extraction.destination || destinations[0],
          destinations: extraction.destinations,
          days: extraction.days,
          title: destinations.length === 1 
            ? `Exploring ${destinations[0]}` 
            : `Multi-destination trip`,
          destinationImageUrl,
        },
      });
    }
    
    // Add POI search results and map data
    events.push(...poiEvents);
    
    // Update trip context with extracted information
    const updatedTrip = {
      ...trip,
      destination: extraction.destination || trip.destination,
      destinations: extraction.destinations || trip.destinations,
      days: extraction.days || trip.days,
      origin: extraction.origin || trip.origin,
    };
    
    return { 
      events, 
      extractedDestination: extraction,
      input: { ...state.input, trip: updatedTrip }
    };
  })
  .addNode("chat", async (state: State) => {
    // For pure chat queries, provide immediate response
    const chatResponse = await chatAgent(state.input.message, state.messages);
    const event: WsEvent = {
      type: "chat.append",
      data: { id: randomUUID(), role: "assistant", content: chatResponse, createdAt: new Date().toISOString() },
    };
    return { events: [event] };
  })
  .setEntryPoint("router")
  .addConditionalEdges("router", (state: State) => state.route, {
    planner: "planner",
    destination_search: "destination_search",
    chat: "chat",
  })
  .addEdge("planner", END)
  .addEdge("destination_search", END)
  .addEdge("chat", END);

const app = graph.compile();

export async function runRouter(input: GraphInput, history: Message[]): Promise<WsEvent[]> {
  try {
    // Set up timeout for AI processing - increased for complex planning
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('AI processing timeout')), 90000); // 90 second timeout
    });
    
    const processingPromise = app.invoke({ input: input, messages: history });
    
    const result = (await Promise.race([processingPromise, timeoutPromise])) as unknown as State;
    
    if (!result || !result.events) {
      return [];
    }
    return result.events;
  } catch (error) {
    console.error('Graph processing error:', error);
    
    // Return error event for timeout or other failures
    const errorEvent: WsEvent = {
      type: "chat.append",
      data: {
        id: randomUUID(),
        role: "assistant",
        content: error instanceof Error && error.message.includes('timeout') 
          ? "I'm taking longer than usual to process your request. Please try again or rephrase your question."
          : "Something went wrong while processing your request. Please try again.",
        createdAt: new Date().toISOString(),
      },
    };
    
    return [errorEvent];
  }
}
