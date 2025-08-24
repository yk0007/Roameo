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

export type GraphInput = {
  sessionId: string;
  message: string;
  trip: Partial<TripContext>;
};

interface State {
  messages: Message[];
  input: GraphInput;
  events: WsEvent[];
  route: "planner" | "chat";
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
};

const graph = new StateGraph<State>({ channels: graphState })
  .addNode("router", async (state: State) => {
    const { message } = state.input;
    const intent = await intentAgent(message);
    if (intent === "PLAN_TRIP") {
      return { route: "planner" };
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

    const updatedTrip = { ...trip, destination: res.destination, days: res.days };
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
          days: res.days,
          title,
        },
      },
    ];

    if (res.itinerary.daysPlan.length > 0) {
      events.push(emitItineraryUpdate(res.itinerary));
      const poiEvt = await poiAgent({ destination: res.destination });
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
    chat: "chat",
  })
  .addEdge("planner", END)
  .addEdge("chat", END);

const app = graph.compile();

export async function runRouter(input: GraphInput, history: Message[]): Promise<WsEvent[]> {
  const result = (await app.invoke({ input: input, messages: history })) as unknown as State;
  if (!result || !result.events) {
    return [];
  }
  return result.events;
}
