import { randomUUID } from "crypto";
import type { WsEvent, TripContext } from "../types/schemas.js";
import type { Message } from "../db/types.js";
import { StateGraph, END, StateGraphArgs } from "@langchain/langgraph";
import { plannerAgent, emitItineraryUpdate } from "../agents/planner.js";
import { editorAgent } from "../agents/editor.js";
import { mapEditorAgent } from "../agents/mapEditor.js";
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
  itineraryHistory?: any[]; // simple undo stack
  itineraryRedo?: any[]; // simple redo stack
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
  itineraryHistory: {
    reducer: (x, y) => y ?? x,
    default: () => [],
  },
  itineraryRedo: {
    reducer: (x, y) => y ?? x,
    default: () => [],
  },
};

// Create a very small rolling summary from the last few messages to improve grounding
function summarizeConversation(messages: Message[], itinerary?: any): string {
  const lastUser = [...messages].reverse().find(m => m.role === "user")?.content || "";
  const lastAssistant = [...messages].reverse().find(m => m.role === "assistant")?.content || "";
  const itin = itinerary?.destination ? `${itinerary.destination} for ${itinerary.days || "?"} days` : "no itinerary yet";
  const u = lastUser.replace(/\s+/g, " ").trim().slice(0, 220);
  const a = lastAssistant.replace(/\s+/g, " ").trim().slice(0, 220);
  return `Context: current itinerary: ${itin}. Last user: "${u}". Last assistant: "${a}".`;
}

const graph = new StateGraph<State>({ channels: graphState })
  .addNode("router", async (state: State) => {
    const { message } = state.input;
    const recent = state.messages.slice(-30);
    const intent = await intentAgent(message, recent);
    console.log(`[router] Intent detected: ${intent} for message: "${message}"`);
    // Quick regex-based detection for edit commands (add day, remove day, add/move/remove activity)
    const m = message.toLowerCase();
    const isEdit = /(add (a )?day( after (day )?\d+)?|remove day \d+|add .+ to day \d+|remove .+ from day \d+|move .+ from day \d+ to day \d+)/.test(m);
    const isMapEdit = /(show routes?|hide routes?|clear map|reset map|add marker [-+]?\d+\.?\d*,\s*[-+]?\d+\.?\d*|fit (map|to itinerary|to route|to points))/i.test(message);
    
    // Emit intent detection event for planning and destination search intents
    const events: WsEvent[] = [];
    if (intent === "PLAN_TRIP" || intent === "DESTINATION_SEARCH") {
      events.push({
        type: "intent.detected",
        data: { intent, message }
      });
    }
    
    if (isEdit) {
      return { route: "editor", events };
    } else if (isMapEdit) {
      return { route: "map_editor", events };
    } else if (intent === "PLAN_TRIP") {
      return { route: "planner", events };
    } else if (intent === "DESTINATION_SEARCH") {
      return { route: "destination_search", events };
    }
    return { route: "chat", events };
  })
  .addNode("planner", async (state: State) => {
    const { trip, message } = state.input;
    const recent = state.messages.slice(-30);
    const conversationSummary = summarizeConversation(recent, state.itinerary);
    
    // Emit planning start event
    const plannerEvents: WsEvent[] = [{
      type: "planning.status",
      data: { status: "Analyzing your request..." }
    }];
    
    // Pass existing itinerary context to planner for multi-destination support
    const plannerContext = {
      ...trip,
      existingItinerary: state.itinerary, // Pass current itinerary for add/remove operations
      conversationSummary,
    };
    
    const res = await plannerAgent(plannerContext, message, recent);
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
    const recent = state.messages.slice(-30);
    const events: WsEvent[] = [];

    const [extraction, reClassifiedIntent] = await Promise.all([
      destinationExtractionAgent(message, recent),
      intentAgent(message, recent),
    ]);

    const isActualDestinationSearch =
      extraction.hasDestination &&
      (extraction.destination || (extraction.destinations && extraction.destinations.length > 0)) &&
      reClassifiedIntent !== "CHAT";

    if (!isActualDestinationSearch) {
      const chatResponse = await chatAgent(message, recent);
      events.push({
        type: "chat.append",
        data: { id: randomUUID(), role: "assistant", content: chatResponse, createdAt: new Date().toISOString() },
      });
      return { events, extractedDestination: extraction };
    }

    // Quick plan if user actually wants a plan now
    const planningTrip = {
      ...trip,
      destination:
        extraction.destination || (extraction.destinations ? extraction.destinations[0] : trip.destination),
      destinations: extraction.destinations || (extraction.destination ? [extraction.destination] : trip.destinations),
      days: extraction.days || trip.days || 3,
      origin: extraction.origin || trip.origin,
    };

    const res = await plannerAgent(planningTrip, message, recent);
    if (res) {
      events.push({
        type: "chat.append",
        data: { id: randomUUID(), role: "assistant", content: res.chatResponse, createdAt: new Date().toISOString() },
      });
      if (res.itinerary?.daysPlan?.length >= 0) {
        events.push(emitItineraryUpdate(res.itinerary));
      }
    }
    return { events };
  })
  .addNode("editor", async (state: State) => {
    const { message } = state.input;
    const events: WsEvent[] = [];
    // undo
    if (/^undo\b/i.test(message)) {
      const hist = (state.itineraryHistory || []).slice();
      const redo = (state.itineraryRedo || []).slice();
      if (hist.length > 0) {
        const prev = hist.pop();
        if (state.itinerary) {
          redo.push(state.itinerary);
          if (redo.length > 10) redo.shift();
        }
        events.push({ type: "chat.append", data: { id: randomUUID(), role: "assistant", content: "Undid the last change.", createdAt: new Date().toISOString() } });
        events.push(emitItineraryUpdate(prev));
        return { events, itinerary: prev, itineraryHistory: hist, itineraryRedo: redo } as any;
      } else {
        events.push({ type: "chat.append", data: { id: randomUUID(), role: "assistant", content: "Nothing to undo.", createdAt: new Date().toISOString() } });
        return { events } as any;
      }
    }
    // redo
    if (/^redo\b/i.test(message)) {
      const redo = (state.itineraryRedo || []).slice();
      const hist = (state.itineraryHistory || []).slice();
      if (redo.length > 0) {
        const nextItin = redo.pop();
        if (state.itinerary) {
          hist.push(state.itinerary);
          if (hist.length > 10) hist.shift();
        }
        events.push({ type: "chat.append", data: { id: randomUUID(), role: "assistant", content: "Redid the last undone change.", createdAt: new Date().toISOString() } });
        events.push(emitItineraryUpdate(nextItin));
        return { events, itinerary: nextItin, itineraryHistory: hist, itineraryRedo: redo } as any;
      } else {
        events.push({ type: "chat.append", data: { id: randomUUID(), role: "assistant", content: "Nothing to redo.", createdAt: new Date().toISOString() } });
        return { events } as any;
      }
    }

    const result = await editorAgent(state.itinerary, message, state.messages.slice(-30));
    if (result) {
      events.push({ type: "chat.append", data: { id: randomUUID(), role: "assistant", content: result.chatResponse, createdAt: new Date().toISOString() } });
      if (result.itinerary && result.itinerary.daysPlan?.length >= 0) {
        const hist = (state.itineraryHistory || []).slice();
        if (state.itinerary) {
          hist.push(state.itinerary);
          if (hist.length > 10) hist.shift();
        }
        events.push(emitItineraryUpdate(result.itinerary));
        return { events, itinerary: result.itinerary, itineraryHistory: hist, itineraryRedo: [] } as any;
      }
    }
    return { events } as any;
  })
  .addNode("map_editor", async (state: State) => {
    const { message } = state.input;
    const events: WsEvent[] = [];
    const res = await mapEditorAgent(state.itinerary, message, state.messages.slice(-30));
    if (res) {
      events.push({ type: "chat.append", data: { id: randomUUID(), role: "assistant", content: res.chatResponse, createdAt: new Date().toISOString() } });
      events.push({ type: "map.update", data: res.map as any });
    }
    return { events } as any;
  })
  .addNode("chat", async (state: State) => {
    const { message } = state.input;
    const recent = state.messages.slice(-30);
    const content = await chatAgent(message, recent);
    const events: WsEvent[] = [
      { type: "chat.append", data: { id: randomUUID(), role: "assistant", content, createdAt: new Date().toISOString() } },
    ];
    return { events };
  })
  .addEdge("planner", END)
  .addEdge("destination_search", END)
  .addEdge("editor", END)
  .addEdge("map_editor", END)
  .addEdge("chat", END);

const app = graph.compile();

export async function runRouter(input: GraphInput, history: Message[]): Promise<WsEvent[]> {
  try {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("AI processing timeout")), 90000);
    });
    const processingPromise = app.invoke({ input, messages: history });
    const result = (await Promise.race([processingPromise, timeoutPromise])) as unknown as State;
    if (!result || !result.events) return [];
    return result.events;
  } catch (error) {
    console.error("Graph processing error:", error);
    const errorEvent: WsEvent = {
      type: "chat.append",
      data: {
        id: randomUUID(),
        role: "assistant",
        content:
          error instanceof Error && error.message.includes("timeout")
            ? "I'm taking longer than usual to process your request. Please try again or rephrase your question."
            : "Something went wrong while processing your request. Please try again.",
        createdAt: new Date().toISOString(),
      },
    };
    return [errorEvent];
  }
}
