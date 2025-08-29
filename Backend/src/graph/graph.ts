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
    const intent = await intentAgent(message);
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
    
    const res = await plannerAgent(trip, message);
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

    const updatedTrip = { 
      ...trip, 
      destination: res.destination, 
      destinations: res.destinations,
      days: res.days,
      destinationImageUrl: res.destinationImageUrl
    };
    const title = await generateSessionTitle({
      message,
      origin: updatedTrip.origin,
      destination: res.destination,
      days: res.days,
      existingTitle: trip.title,
    });

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
        },
      },
    );

    if (res.itinerary.daysPlan.length > 0) {
      plannerEvents.push(emitItineraryUpdate(res.itinerary));
      // Handle POI search for multiple destinations
      const searchDestination = res.destinations && res.destinations.length > 0 
        ? res.destinations[0] // Use first destination for POI search
        : res.destination;
      
      // Emit search status
      plannerEvents.push({
        type: "search.status",
        data: { status: `Finding places in ${searchDestination}...` }
      });
      
      const poiEvt = await poiAgent({ destination: searchDestination });
      if (poiEvt) {
        plannerEvents.push(poiEvt);
        if (poiEvt.type === "search.results") {
          const pois = [...poiEvt.data.stays, ...poiEvt.data.restaurants, ...poiEvt.data.attractions];
          
          // Emit map status
          plannerEvents.push({
            type: "map.status",
            data: { status: "Calculating routes and updating map..." }
          });
          
          plannerEvents.push(await mapAgent(pois));
        }
      }
    }

    return { events: plannerEvents, input: { ...state.input, trip: updatedTrip } };
  })
  .addNode("destination_search", async (state: State) => {
    const { message, trip } = state.input;
    
    const events: WsEvent[] = [];
    
    // PRIORITY: Generate immediate AI response first for quick user feedback
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
    
    // Extract destination information after sending immediate response
    const extraction = await destinationExtractionAgent(message);
    console.log(`[destination_search] Extracted destinations:`, extraction);

    // Check if user is asking for trip planning (even without explicit days)
    const planningKeywords = ["plan", "trip", "itinerary", "travel", "visit"];
    const messageContainsPlanningKeywords = planningKeywords.some(keyword => 
      message.toLowerCase().includes(keyword)
    );
    
    // Check if we have destination and either:
    // 1. Both destination and days, OR
    // 2. Destination and planning keywords (user wants trip planning)
    const shouldTriggerPlanning = extraction.hasDestination && 
      (extraction.destination || (extraction.destinations && extraction.destinations.length > 0)) &&
      (extraction.days || messageContainsPlanningKeywords);
    
    if (shouldTriggerPlanning) {
      console.log(`[destination_search] Triggering itinerary planning - days: ${extraction.days}, planning keywords: ${messageContainsPlanningKeywords}`);
      
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
        days: extraction.days || trip.days || 3, // Default to 3 days if not specified
        origin: extraction.origin || trip.origin,
      };
      
      // Use planner agent to generate full itinerary
      const res = await plannerAgent(planningTrip, message);
      
      if (res) {
        const updatedTrip = { 
          ...planningTrip, 
          destination: res.destination, 
          destinations: res.destinations,
          days: res.days,
          destinationImageUrl: res.destinationImageUrl
        };
        
        const title = await generateSessionTitle({
          message,
          origin: updatedTrip.origin,
          destination: res.destination,
          days: res.days,
          existingTitle: trip.title,
        });

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
          
          // Handle POI search for multiple destinations
          const searchDestination = res.destinations && res.destinations.length > 0 
            ? res.destinations[0] // Use first destination for POI search
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
    
    // Otherwise, proceed with immediate POI search and ask for remaining details
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
