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
    
    if (intent === "PLAN_TRIP") {
      return { route: "planner" };
    } else if (intent === "DESTINATION_SEARCH") {
      return { route: "destination_search" };
    }
    return { route: "chat" };
  })
  .addNode("planner", async (state: State) => {
    const { trip, message } = state.input;
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
      days: res.days 
    };
    const title = await generateSessionTitle({
      message,
      origin: updatedTrip.origin,
      destination: res.destination,
      days: res.days,
      existingTitle: trip.title,
    });

    const events: WsEvent[] = [
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
        },
      },
    ];

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

    return { events, input: { ...state.input, trip: updatedTrip } };
  })
  .addNode("destination_search", async (state: State) => {
    const { message, trip } = state.input;
    
    // Extract destination information immediately
    const extraction = await destinationExtractionAgent(message);
    console.log(`[destination_search] Extracted destinations:`, extraction);
    
    const events: WsEvent[] = [];
    
    // Trigger immediate POI search first to get POI data
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
    
    // Generate AI-powered conversational response with POI formatting
    const chatResponse = await generateDestinationChatResponse(extraction, message, poiData);
    
    // Add chat response
    events.push({
      type: "chat.append",
      data: {
        id: randomUUID(),
        role: "assistant",
        content: chatResponse,
        createdAt: new Date().toISOString(),
      },
    });
    
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
    // Set up timeout for AI processing
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('AI processing timeout')), 60000); // 60 second timeout
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
