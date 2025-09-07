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
  itinerary?: any; // Current itinerary for multi-destination support
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
    
    // Pass existing itinerary context to planner for multi-destination support
    const plannerContext = {
      ...trip,
      existingItinerary: state.itinerary // Pass current itinerary for add/remove operations
    };
    
    const res = await plannerAgent(plannerContext, message, state.messages);
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

    // If this is a clarification, do NOT alter trip state yet.
    const updatedTrip = res.clarify
      ? { ...trip }
      : {
          ...trip,
          destination: res.destination,
          destinations: res.destinations,
          days: res.days,
          destinationImageUrl: res.destinationImageUrl,
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

    // Always send chat response
    plannerEvents.push({
      type: "chat.append",
      data: {
        id: randomUUID(),
        role: "assistant",
        content: res.chatResponse,
        createdAt: new Date().toISOString(),
      },
    });

    // Only update navbar when this is not a clarification
    if (!res.clarify) {
      plannerEvents.push({
        type: "navbar.update",
        data: {
          destination: res.destination,
          destinations: res.destinations,
          days: res.days,
          title,
          destinationImageUrl: res.destinationImageUrl,
        },
      });
    }

    if (res.itinerary.daysPlan.length > 0) {
      // Always emit itinerary update (preserves existing itinerary during clarification)
      plannerEvents.push(emitItineraryUpdate(res.itinerary));

      // Only trigger POI/map updates when not clarifying
      if (!res.clarify) {
        const searchDestination =
          res.destinations && res.destinations.length > 0
            ? res.destinations[0]
            : res.destination;

        plannerEvents.push({
          type: "search.status",
          data: { status: `Finding places in ${searchDestination}...` },
        });

        const poiEvt = await poiAgent({ destination: searchDestination });
        if (poiEvt) {
          plannerEvents.push(poiEvt);
          if (poiEvt.type === "search.results") {
            const pois = [
              ...poiEvt.data.stays,
              ...poiEvt.data.restaurants,
              ...poiEvt.data.attractions,
            ];

            plannerEvents.push({
              type: "map.status",
              data: { status: "Calculating routes and updating map..." },
            });

            plannerEvents.push(await mapAgent(pois));
          }
        }
      }
    }

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
    
    // Step 2: Check if this should actually be trip planning instead
    // If re-classification suggests PLAN_TRIP, handle it as planning
    if (reClassifiedIntent === "PLAN_TRIP") {
      console.log(`[destination_search] Re-classified as PLAN_TRIP, routing to planning logic`);
      
      // Provide immediate acknowledgment
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
      
      // Emit planning status
      events.push({
        type: "planning.status",
        data: { status: "Creating your itinerary..." }
      });
      
      // Create trip context for planner agent
      const planningTrip = {
        ...trip,
        destination: extraction.destination || (extraction.destinations ? extraction.destinations[0] : trip.destination),
        destinations: extraction.destinations || (extraction.destination ? [extraction.destination] : trip.destinations),
        days: extraction.days || trip.days || 3,
        origin: extraction.origin || trip.origin,
      };
      
      // Use planner agent to generate full itinerary
      const res = await plannerAgent(planningTrip, message, state.messages);
      
      if (res) {
        const updatedTrip = { 
          ...planningTrip, 
          destination: res.destination, 
          destinations: res.destinations,
          days: res.days,
          destinationImageUrl: res.destinationImageUrl
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
          console.log(`[destination_search] Title generation failed: ${error?.message || error}, using fallback`);
          // Create fallback title
          const sessionSuffix = Math.random().toString(36).substring(2, 5).toUpperCase();
          title = res.destination ? `✨ ${res.destination} Adventure #${sessionSuffix}` : `✨ Dream Trip #${sessionSuffix}`;
        }

        events.push({
          type: "chat.append",
          data: {
            id: randomUUID(),
            role: "assistant",
            content: res.chatResponse,
            createdAt: new Date().toISOString(),
          },
        });
        
        events.push({
          type: "navbar.update",
          data: {
            destination: res.destination,
            destinations: res.destinations,
            days: res.days,
            title,
            destinationImageUrl: res.destinationImageUrl,
          },
        });

        if (res.itinerary.daysPlan.length > 0) {
          events.push(emitItineraryUpdate(res.itinerary));
          
          const searchDestination = res.destinations && res.destinations.length > 0 
            ? res.destinations[0]
            : res.destination;
          const poiEvt = await poiAgent({ destination: searchDestination });
          if (poiEvt) {
            events.push(poiEvt);
            if (poiEvt.type === "search.results") {
              const pois = [...poiEvt.data.stays, ...poiEvt.data.restaurants, ...poiEvt.data.attractions];
              events.push(await mapAgent(pois));
            }
          }
        }
        
        return { 
          events, 
          extractedDestination: extraction,
          input: { ...state.input, trip: updatedTrip }
        };
      }
    }
    
    // Step 3: This is pure destination search - provide information about the destination
    console.log(`[destination_search] Proceeding with destination search for: ${extraction.destination || extraction.destinations?.[0]}`);
    
    // PRIORITY: Generate immediate response for better UX
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
    
    // Then proceed with destination search in background
    events.push({
      type: "search.status",
      data: { status: `Searching for places in ${extraction.destination || extraction.destinations?.[0] || 'your destination'}...` }
    });
    
    const poiEvents = await immediatePoiSearchAgent(extraction);
    
    // Extract POI data for chat response generation
    let poiData: { stays: any[]; restaurants: any[]; attractions: any[] } | undefined;
    const poiSearchEvent = poiEvents.find(event => event.type === "search.results");
    if (poiSearchEvent && poiSearchEvent.data) {
      poiData = {
        stays: poiSearchEvent.data.stays || [],
        restaurants: poiSearchEvent.data.restaurants || [],
        attractions: poiSearchEvent.data.attractions || []
      };
    }
    
    // Generate AI-powered conversational response with POI formatting + ask for missing info
    const destinationChatResponse = await generateDestinationChatResponse(extraction, message, poiData);
    
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
    
    // Update navbar with destination info if available
    const destinations = extraction.destinations || (extraction.destination ? [extraction.destination] : []);
    if (extraction.hasDestination && destinations.length > 0) {
      events.push({
        type: "navbar.update",
        data: {
          destination: extraction.destination || destinations[0],
          destinations: extraction.destinations,
          days: extraction.days,
          title: destinations.length === 1 
            ? `Exploring ${destinations[0]}` 
            : `Multi-destination trip`,
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
