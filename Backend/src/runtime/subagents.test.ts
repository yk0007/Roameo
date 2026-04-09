import test from "node:test";
import assert from "node:assert/strict";
import {
  researchPlanningDestinations,
  buildResponseBlocks,
  normalizeTravelIntent,
  resolveDeterministicTurnIntent,
  resolveFastTurnResponse,
  resolveOversizedPlanResponse,
  resolveTurnIntent,
  synthesizePlan,
  updateSessionMemory
} from "./subagents.js";
import type { SessionSnapshot } from "@roameo/contracts";

function buildProviderStub(result: unknown) {
  return {
    generateObject: async () => result,
    routerModels: () => undefined,
    narrativeModels: () => undefined
  } as any;
}

function makeSession(): SessionSnapshot {
  return {
    id: "session-1",
    title: "Test Trip",
    providerSettings: {
      provider: "gemini",
      runMode: "balanced",
      keySource: "platform"
    },
    memory: {
      summary: "",
      destinationsDiscussed: [],
      acceptedDecisions: [],
      lastPlanVersion: 0,
      pendingFollowUp: null,
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      },
      planningState: {
        status: "ready",
        stage: "ready",
        retryable: true
      },
      preferences: {
        currency: "INR",
        locale: "en-IN",
        styles: [],
        dietaryNotes: [],
        accessibilityNotes: []
      }
    },
    poiCatalog: {
      version: 1,
      items: {
        "poi-1": {
          id: "poi-1",
          name: "Fort Kochi",
          type: "attraction",
          lat: 9.96,
          lng: 76.24,
          openingHours: [],
          source: "google_places",
          tags: []
        },
        "poi-2": {
          id: "poi-2",
          name: "Seagull",
          type: "restaurant",
          lat: 9.95,
          lng: 76.23,
          openingHours: [],
          source: "google_places",
          tags: []
        }
      }
    },
    messages: [],
    savedPoiIds: [],
    createdAt: "2026-04-01T00:00:00.000Z",
    updatedAt: "2026-04-01T00:00:00.000Z"
  };
}

test("normalizeTravelIntent maps legacy provider labels to canonical intent values", () => {
  assert.equal(normalizeTravelIntent("PLAN_TRIP"), "plan_trip");
  assert.equal(normalizeTravelIntent("destination_search"), "search_places");
  assert.equal(normalizeTravelIntent("chat"), "question");
  assert.equal(normalizeTravelIntent("preferences"), "settings");
});

test("normalizeTravelIntent fails fast for unsupported values", () => {
  assert.throws(
    () => normalizeTravelIntent("unknown_mode"),
    /Unsupported travel intent/
  );
});

test("resolveFastTurnResponse handles simple greetings without travel pipeline work", () => {
  const fast = resolveFastTurnResponse(makeSession(), "hi");

  assert.ok(fast);
  assert.match(fast.reply, /tell me where you want to go/i);
  assert.deepEqual(
    fast.responseBlocks.map((block) => block.type),
    ["trip_intro", "lead", "assistant_prompt_chips"]
  );
  assert.equal(fast.memory.planningState.status, "ready");
});

test("resolveFastTurnResponse stores the user's name from a simple introduction", () => {
  const fast = resolveFastTurnResponse(makeSession(), "my name is yk");

  assert.ok(fast);
  assert.match(fast.reply, /nice to meet you, Yk/i);
  assert.ok(
    fast.memory.acceptedDecisions.includes("Name: Yk")
  );
});

test("resolveFastTurnResponse uses origin context in casual replies and chips", () => {
  const session = makeSession();
  session.memory.acceptedDecisions.push("Origin: Visakhapatnam, Andhra Pradesh");

  const fast = resolveFastTurnResponse(session, "hi");

  assert.ok(fast);
  assert.match(fast.reply, /tell me where you want to go/i);
  assert.match(
    (fast.responseBlocks.find((block) => block.type === "lead") as any).text,
    /visakhapatnam/i
  );
  const chips = fast.responseBlocks.find(
    (block) => block.type === "assistant_prompt_chips"
  );
  assert.ok(chips && chips.type === "assistant_prompt_chips");
  assert.ok(chips.prompts.some((prompt) => /visakhapatnam/i.test(prompt.prompt)));
});

test("resolveFastTurnResponse handles capability questions without the full pipeline", () => {
  const fast = resolveFastTurnResponse(makeSession(), "what can u do?");

  assert.ok(fast);
  assert.match(fast.reply, /i can plan trips, find restaurants and stays/i);
  assert.equal(fast.memory.planningState.status, "ready");
});

test("resolveFastTurnResponse handles identity questions without the full pipeline", () => {
  const fast = resolveFastTurnResponse(makeSession(), "what are you?");

  assert.ok(fast);
  assert.match(fast.reply, /roameo/i);
  assert.equal(fast.memory.planningState.status, "ready");
});

test("resolveDeterministicTurnIntent leaves explicit restaurant discovery to the semantic router", () => {
  const session = makeSession();
  session.memory.destinationsDiscussed = ["Goa"];

  const resolution = resolveDeterministicTurnIntent(
    session,
    "show me some restaurants"
  );

  assert.equal(resolution, null);
});

test("resolveDeterministicTurnIntent leaves explicit planning requests to the semantic router", () => {
  const resolution = resolveDeterministicTurnIntent(
    makeSession(),
    "plan trip to araku for 2 days"
  );

  assert.equal(resolution, null);
});

test("resolveDeterministicTurnIntent leaves full itinerary prompts to the semantic router", () => {
  const resolution = resolveDeterministicTurnIntent(
    makeSession(),
    `Plan a family trip to shimla for 4 days. Include family-friendly activities, suitable accommodations, and child-safe attractions. Please provide a detailed itinerary with:

Day-by-day schedule with specific activities
Recommended places to visit with brief descriptions
Accommodation suggestions
Transportation options and travel tips
Local cuisine recommendations
Important travel information and safety tips
Budget estimates for different expense categories`
  );

  assert.equal(resolution, null);
});

test("researchPlanningDestinations keeps hospitality categories available for planning", async () => {
  const attraction = {
    id: "poi-attraction",
    name: "Ridge",
    type: "attraction" as const,
    lat: 31.1,
    lng: 77.1,
    openingHours: [],
    source: "google_places" as const,
    tags: ["Shimla"]
  };
  const stay = {
    id: "poi-stay",
    name: "Family Pine Resort",
    type: "stay" as const,
    lat: 31.2,
    lng: 77.2,
    openingHours: [],
    source: "google_places" as const,
    tags: ["Shimla"]
  };
  const restaurant = {
    id: "poi-restaurant",
    name: "Himachal Family Kitchen",
    type: "restaurant" as const,
    lat: 31.3,
    lng: 77.3,
    openingHours: [],
    source: "google_places" as const,
    tags: ["Shimla"]
  };

  const calls: string[] = [];
  const tools = {
    async searchPlacesForDestination(_destination: string, focus: string) {
      calls.push(focus);
      return {
        stays: focus === "family" || focus === "general" ? [stay] : [],
        restaurants: focus === "general" ? [restaurant] : [],
        attractions: [attraction],
        catalog: {
          version: 1,
          items: {
            [attraction.id]: attraction,
            ...(focus === "general" ? { [restaurant.id]: restaurant } : {}),
            ...(focus === "family" || focus === "general" ? { [stay.id]: stay } : {})
          }
        }
      };
    },
    async getDestinationFacts() {
      return [];
    }
  } as any;

  const research = await researchPlanningDestinations(
    tools,
    ["Shimla"],
    "family",
    false
  );

  assert.deepEqual(calls, ["general", "family"]);
  assert.equal(research.grouped.restaurants[0]?.id, "poi-restaurant");
  assert.equal(research.grouped.stays[0]?.id, "poi-stay");
});

test("synthesizePlan upgrades generic meals and accommodation to real POIs", async () => {
  const session = makeSession();
  const stay = {
    id: "stay-1",
    name: "Pine View Resort",
    type: "stay" as const,
    lat: 31.1,
    lng: 77.1,
    address: "Shimla",
    openingHours: [],
    source: "google_places" as const,
    tags: ["Shimla"]
  };
  const restaurant = {
    id: "restaurant-1",
    name: "Himachal Rasoi",
    type: "restaurant" as const,
    lat: 31.2,
    lng: 77.2,
    address: "Shimla",
    openingHours: [],
    source: "google_places" as const,
    tags: ["Shimla"]
  };
  const attraction = {
    id: "attraction-1",
    name: "The Ridge",
    type: "attraction" as const,
    lat: 31.3,
    lng: 77.3,
    address: "Shimla",
    openingHours: [],
    source: "google_places" as const,
    tags: ["Shimla"]
  };
  const research = {
    destinations: ["Shimla"],
    focus: "family" as const,
    catalog: {
      version: 1,
      items: {
        [stay.id]: stay,
        [restaurant.id]: restaurant,
        [attraction.id]: attraction
      }
    },
    grouped: {
      stays: [stay],
      restaurants: [restaurant],
      attractions: [attraction]
    },
    facts: []
  };

  const plan = await synthesizePlan(
    buildProviderStub({
      title: "Shimla family reset",
      destination: "Shimla",
      destinations: ["Shimla"],
      totalDays: 2,
      travelerCount: 2,
      notes: [],
      destinationSegments: [],
      days: [
        {
          day: 1,
          title: "Soft arrival",
          destination: "Shimla",
          activities: [
            {
              title: "The Ridge walk",
              poiId: attraction.id,
              startTime: "09:00",
              endTime: "10:30",
              notes: []
            },
            {
              title: "Lunch",
              startTime: "13:00",
              endTime: "14:00",
              notes: []
            }
          ]
        },
        {
          day: 2,
          title: "Garden day",
          destination: "Shimla",
          activities: [
            {
              title: "Morning stroll",
              poiId: attraction.id,
              startTime: "09:00",
              endTime: "10:30",
              notes: []
            }
          ]
        }
      ]
    }),
    {} as any,
    session,
    {
      intent: "plan_trip",
      destination: "Shimla",
      destinations: ["Shimla"],
      totalDays: 2,
      travelerCount: 2,
      styles: ["family"],
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    research
  );

  assert.equal(plan.days[0]?.accommodationPoiId, stay.id);
  assert.ok(
    plan.days[0]?.activities.some(
      (activity) =>
        activity.poiId === restaurant.id &&
        /lunch at himachal rasoi/i.test(activity.title)
    )
  );
});

test("resolveOversizedPlanResponse pushes very long trips into phased planning", () => {
  const response = resolveOversizedPlanResponse({
    intent: "plan_trip",
    destination: "Araku Valley",
    destinations: ["Araku Valley"],
    totalDays: 50,
    styles: [],
    dateContext: {
      flexibility: "open_ended",
      derivedFrom: "none",
      advisoryItems: []
    }
  });

  assert.ok(response);
  assert.match(response.reply, /small travel novel/i);
  assert.deepEqual(
    response.responseBlocks.map((block) => block.type),
    ["trip_intro", "lead", "assistant_prompt_chips"]
  );
});

test("updateSessionMemory keeps pre-plan destination and duration in canonical memory", () => {
  const memory = updateSessionMemory(
    makeSession(),
    {
      intent: "plan_trip",
      destination: "Meghalaya",
      destinations: ["Meghalaya"],
      origin: "Shillong",
      totalDays: 3,
      travelerCount: 2,
      styles: [],
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    "Working on it."
  );

  assert.ok(memory.acceptedDecisions.includes("Origin: Shillong"));
  assert.ok(memory.acceptedDecisions.includes("Destination: Meghalaya"));
  assert.ok(memory.acceptedDecisions.includes("Duration: 3 days"));
  assert.ok(memory.acceptedDecisions.includes("Travelers: 2"));
  assert.deepEqual(memory.destinationsDiscussed, ["Meghalaya"]);
});

test("updateSessionMemory replaces stale active trip context on an explicit new trip", () => {
  const session = makeSession();
  session.memory.destinationsDiscussed = ["Paris"];
  session.memory.acceptedDecisions = [
    "Destination: Paris",
    "Duration: 5 days",
    "Dates: 2026-04-10 to 2026-04-14"
  ];

  const memory = updateSessionMemory(
    session,
    {
      intent: "plan_trip",
      destination: "Araku",
      destinations: ["Araku"],
      totalDays: 2,
      explicitNewTrip: true,
      styles: [],
      dateContext: {
        inferredStartDate: "2026-05-01",
        inferredEndDate: "2026-05-02",
        flexibility: "exact",
        derivedFrom: "explicit",
        advisoryItems: []
      }
    },
    "Planning Araku now."
  );

  assert.deepEqual(memory.destinationsDiscussed, ["Araku"]);
  assert.ok(memory.acceptedDecisions.includes("Destination: Araku"));
  assert.ok(!memory.acceptedDecisions.includes("Destination: Paris"));
  assert.ok(memory.acceptedDecisions.includes("Duration: 2 days"));
  assert.ok(!memory.acceptedDecisions.includes("Duration: 5 days"));
});

test("updateSessionMemory preserves multi-city destination sets for the active plan", () => {
  const memory = updateSessionMemory(
    makeSession(),
    {
      intent: "plan_trip",
      destination: "Paris",
      destinations: ["Paris", "Lyon"],
      totalDays: 5,
      explicitNewTrip: true,
      styles: [],
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    "Planning a France route."
  );

  assert.deepEqual(memory.destinationsDiscussed, ["Paris", "Lyon"]);
  assert.ok(memory.acceptedDecisions.includes("Destination: Paris"));
});

test("buildResponseBlocks emits canonical itinerary presentation blocks", () => {
  const session = makeSession();
  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "plan_trip",
      destinations: ["Kochi"],
      destination: "Kochi",
      totalDays: 1,
      travelerCount: 1,
      styles: [],
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    narrative: {
      introTitle: "Kochi looks like a great fit",
      introBody: "I pulled together a first draft you can tweak easily.",
      leadText: "Here’s a warm first pass with food and waterfront time.",
      moodEmoji: "🌴",
      promptChips: [
        {
          label: "Relax it",
          prompt: "Make this more relaxed."
        }
      ],
      clarifyingQuestions: []
    },
    plan: {
      schemaVersion: 1 as const,
      sessionId: "session-1",
      version: 1,
      title: "1-day Kochi reset",
      destination: "Kochi",
      destinations: ["Kochi"],
      totalDays: 1,
      travelerCount: 1,
      budget: {
        accommodation: 2500,
        food: 1400,
        transport: 600,
        activities: 700,
        misc: 500,
        total: 5700,
        currency: "INR"
      },
      notes: [],
      destinationSegments: [],
      days: [
        {
          day: 1,
          date: "2026-04-02",
          title: "Fort Kochi by the water",
          destination: "Kochi",
          activities: [
            {
              id: "activity-1",
              poiId: "poi-1",
              title: "Fort Kochi walk",
              startTime: "09:00",
              endTime: "10:30",
              notes: []
            },
            {
              id: "activity-2",
              poiId: "poi-2",
              title: "Lunch by the harbor",
              startTime: "13:00",
              endTime: "14:15",
              notes: []
            }
          ]
        }
      ],
      generatedAt: "2026-04-01T00:00:00.000Z",
      lastUserIntent: "plan_trip"
    }
  });

  assert.deepEqual(
    blocks.map((block) => block.type),
    ["trip_intro", "lead", "itinerary_template", "place_card_row", "assistant_prompt_chips"]
  );

  const itinerary = blocks.find((block) => block.type === "itinerary_template");
  assert.ok(itinerary && itinerary.type === "itinerary_template");
  assert.equal(itinerary.budgetLabel, "Expected spend: Mid-range");
  assert.ok(itinerary.days.every((day) => day.footer === undefined));
});

test("buildResponseBlocks never references unknown POI ids", () => {
  const session = makeSession();
  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "search_places",
      destinations: ["Kochi"],
      destination: "Kochi",
      styles: [],
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    narrative: {
      introTitle: "A few nice options are ready",
      introBody: "You can start with these and I can tighten the vibe after.",
      leadText: "These are the strongest current matches.",
      promptChips: [],
      clarifyingQuestions: []
    },
    research: {
      destinations: ["Kochi"],
      focus: "general",
      catalog: session.poiCatalog,
      grouped: {
        stays: [],
        restaurants: [],
        attractions: [
          session.poiCatalog.items["poi-1"],
          {
            id: "missing-poi",
            name: "Missing",
            type: "attraction",
            lat: 0,
            lng: 0,
            openingHours: [],
            source: "google_places",
            tags: []
          }
        ]
      },
      facts: []
    }
  });

  const stories = blocks.find((block) => block.type === "poi_story_list");
  assert.ok(stories && stories.type === "poi_story_list");
  assert.deepEqual(stories.items.map((item) => item.poiId), ["poi-1"]);
});

test("resolveTurnIntent upgrades natural follow-up edit requests into refine_trip", async () => {
  const session: SessionSnapshot = {
    ...makeSession(),
    plan: {
      schemaVersion: 1,
      sessionId: "session-1",
      version: 1,
      title: "Kochi weekend",
      destination: "Kochi",
      destinations: ["Kochi"],
      totalDays: 2,
      travelerCount: 1,
      notes: [],
      destinationSegments: [],
      days: [],
      generatedAt: "2026-04-01T00:00:00.000Z",
      lastUserIntent: "plan_trip"
    }
  };

  const providerService = buildProviderStub({
    intent: "question",
    destinations: [],
    styles: []
  });

  const resolution = await resolveTurnIntent(
    providerService,
    {} as any,
    session,
    "Please make day 2 slower and add a seafood dinner."
  );

  assert.equal(resolution.intent, "refine_trip");
});

test("resolveTurnIntent anchors broad greetings to the current local context", async () => {
  const session: SessionSnapshot = {
    ...makeSession(),
    memory: {
      ...makeSession().memory,
      destinationsDiscussed: ["Visakhapatnam"]
    }
  };

  const providerService = buildProviderStub({
    intent: "question",
    destinations: [],
    styles: []
  });

  const resolution = await resolveTurnIntent(
    providerService,
    {} as any,
    session,
    "hello"
  );

  assert.equal(resolution.intent, "question");
  assert.equal(resolution.questionFocus, "greeting");
  assert.equal(resolution.destination, "Visakhapatnam");
});

test("resolveTurnIntent classifies plan trip requests deterministically", async () => {
  const providerService = buildProviderStub({
    intent: "question",
    destinations: [],
    styles: []
  });

  const resolution = await resolveTurnIntent(
    providerService,
    {} as any,
    makeSession(),
    "plan trip to Araku for 2 days"
  );

  assert.equal(resolution.intent, "plan_trip");
  assert.equal(resolution.destination, "Araku");
  assert.equal(resolution.totalDays, 2);
});

test("resolveTurnIntent parses flexible date context from natural language", async () => {
  const providerService = buildProviderStub({
    intent: "question",
    destinations: [],
    styles: []
  });

  const resolution = await resolveTurnIntent(
    providerService,
    {} as any,
    makeSession(),
    "plan trip to Hanoi for 2 days this weekend"
  );

  assert.equal(resolution.intent, "plan_trip");
  assert.equal(resolution.dateContext.flexibility, "approximate");
  assert.equal(resolution.dateContext.derivedFrom, "relative");
  assert.ok(resolution.dateContext.inferredStartDate);
});

test("buildResponseBlocks emits capabilities and featured local card blocks", () => {
  const session = makeSession();
  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "question",
      destination: "Visakhapatnam",
      destinations: ["Visakhapatnam"],
      styles: [],
      questionFocus: "capabilities",
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    narrative: {
      introTitle: "I can help you explore",
      introBody: "I can make planning feel easy and concrete.",
      leadText: "Here’s the main way I can help.",
      promptChips: [],
      clarifyingQuestions: []
    },
    research: {
      destinations: ["Visakhapatnam"],
      focus: "capabilities",
      catalog: session.poiCatalog,
      grouped: {
        stays: [],
        restaurants: [],
        attractions: [session.poiCatalog.items["poi-1"]]
      },
      facts: []
    }
  });

  assert.deepEqual(
    blocks.map((block) => block.type),
    [
      "trip_intro",
      "lead",
      "capabilities_overview",
      "featured_poi",
      "place_card_row",
      "assistant_prompt_chips"
    ]
  );
});

test("buildResponseBlocks emits discovery stories from canonical POIs", () => {
  const session = makeSession();
  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "search_places",
      destination: "Visakhapatnam",
      destinations: ["Visakhapatnam"],
      styles: [],
      questionFocus: "hidden_gems",
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    narrative: {
      introTitle: "A few quieter local picks are ready",
      introBody: "These give you a more local feel than the obvious first stops.",
      leadText: "Here are a few strong places to start.",
      promptChips: [],
      clarifyingQuestions: []
    },
    research: {
      destinations: ["Visakhapatnam"],
      focus: "hidden_gems",
      catalog: session.poiCatalog,
      grouped: {
        stays: [],
        restaurants: [session.poiCatalog.items["poi-2"]],
        attractions: [session.poiCatalog.items["poi-1"]]
      },
      facts: []
    }
  });

  assert.ok(blocks.some((block) => block.type === "poi_story_list"));
  assert.ok(blocks.some((block) => block.type === "place_card_row"));
});

test("buildResponseBlocks emits categorized restaurant rows for food discovery", () => {
  const session = makeSession();
  session.poiCatalog.items["poi-3"] = {
    id: "poi-3",
    name: "Araku Coffee House",
    type: "restaurant",
    lat: 18.32,
    lng: 82.87,
    openingHours: [],
    source: "google_places",
    tags: ["Araku", "famous local restaurant in Araku"],
    rating: 4.6,
    priceLevel: 2
  };
  session.poiCatalog.items["poi-4"] = {
    id: "poi-4",
    name: "Sai Pure Veg",
    type: "restaurant",
    lat: 18.33,
    lng: 82.88,
    openingHours: [],
    source: "google_places",
    tags: ["Araku", "vegetarian restaurant in Araku"],
    rating: 4.2,
    priceLevel: 1
  };
  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "search_places",
      destination: "Araku",
      destinations: ["Araku"],
      styles: [],
      questionFocus: "restaurants",
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    narrative: {
      introTitle: "Araku food picks are ready",
      introBody: "I pulled together a better spread than one generic restaurant row.",
      leadText: "Here are the strongest food categories to explore.",
      promptChips: [],
      clarifyingQuestions: []
    },
    research: {
      destinations: ["Araku"],
      focus: "restaurants",
      catalog: session.poiCatalog,
      grouped: {
        stays: [],
        restaurants: [
          session.poiCatalog.items["poi-2"],
          session.poiCatalog.items["poi-3"],
          session.poiCatalog.items["poi-4"]
        ],
        attractions: []
      },
      facts: []
    }
  });

  const grouped = blocks.find((block) => block.type === "categorized_place_rows");
  assert.ok(grouped && grouped.type === "categorized_place_rows");
  assert.ok(grouped.sections.length >= 2);
});

test("resolveDeterministicTurnIntent leaves generic places requests to the semantic router", () => {
  const session = makeSession();
  session.memory.destinationsDiscussed = ["Goa"];

  const resolution = resolveDeterministicTurnIntent(session, "show me some places");

  assert.equal(resolution, null);
});

test("buildResponseBlocks attraction discovery emits a horizontal attraction row", () => {
  const session = makeSession();
  session.poiCatalog.items["poi-3"] = {
    id: "poi-3",
    name: "RK Beach",
    type: "attraction",
    lat: 17.71,
    lng: 83.32,
    openingHours: [],
    source: "google_places",
    tags: ["Visakhapatnam"],
    rating: 4.4
  };

  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "search_places",
      destination: "Visakhapatnam",
      destinations: ["Visakhapatnam"],
      styles: [],
      questionFocus: "attractions",
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    narrative: {
      introTitle: "A few strong places are ready",
      introBody: "Here are some good attractions to start with.",
      leadText: "Start with these stronger scenic picks.",
      promptChips: [],
      clarifyingQuestions: []
    },
    research: {
      destinations: ["Visakhapatnam"],
      focus: "attractions",
      catalog: session.poiCatalog,
      grouped: {
        stays: [],
        restaurants: [session.poiCatalog.items["poi-2"]],
        attractions: [session.poiCatalog.items["poi-1"], session.poiCatalog.items["poi-3"]]
      },
      facts: []
    }
  });

  const row = blocks.find((block) => block.type === "place_card_row");
  assert.ok(row && row.type === "place_card_row");
  assert.deepEqual(new Set(row.poiIds), new Set(["poi-1", "poi-3"]));
});

test("buildResponseBlocks does not persist worker progress in final replies", () => {
  const session = makeSession();
  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "search_places",
      destination: "Shimla",
      destinations: ["Shimla"],
      styles: [],
      questionFocus: "restaurants",
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    narrative: {
      introTitle: "Shimla food worth your time",
      introBody: "A few picks are ready.",
      leadText: "Start with these.",
      promptChips: [],
      clarifyingQuestions: []
    },
    research: {
      destinations: ["Shimla"],
      focus: "restaurants",
      catalog: session.poiCatalog,
      grouped: {
        stays: [],
        restaurants: [session.poiCatalog.items["poi-2"]],
        attractions: []
      },
      facts: []
    },
    planningContext: {
      workerProgress: [
        {
          label: "Discovering places in Shimla",
          detail: "Pulled real places into the catalog.",
          state: "completed"
        }
      ]
    }
  });

  assert.ok(!blocks.some((block) => block.type === "worker_progress"));
});

test("buildResponseBlocks emits stay and date-aware blocks for stay mode", () => {
  const session = makeSession();
  session.poiCatalog.items["poi-3"] = {
    id: "poi-3",
    name: "Haritha Valley Resort",
    type: "stay",
    lat: 18.3,
    lng: 82.8,
    openingHours: [],
    source: "google_places",
    tags: [],
    rating: 4.1,
    priceLevel: 2
  };
  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "search_places",
      destination: "Araku",
      destinations: ["Araku"],
      styles: ["relaxed"],
      questionFocus: "hotels",
      stayMode: true,
      dateContext: {
        inferredStartDate: "2026-07-02",
        inferredEndDate: "2026-07-03",
        flexibility: "exact",
        derivedFrom: "explicit",
        advisoryItems: [
          {
            kind: "weather",
            title: "Rain is possible",
            detail: "Carry a light waterproof layer."
          }
        ]
      }
    },
    narrative: {
      introTitle: "A few stay options are ready",
      introBody: "I scoped the strongest stay bases for the route.",
      leadText: "Here are the best places to stay around Araku right now.",
      promptChips: [],
      clarifyingQuestions: []
    },
    research: {
      destinations: ["Araku"],
      focus: "hotels",
      catalog: session.poiCatalog,
      grouped: {
        stays: [session.poiCatalog.items["poi-3"]],
        restaurants: [],
        attractions: [session.poiCatalog.items["poi-1"]]
      },
      facts: []
    },
    planningContext: {
      workerProgress: [{ label: "Scouting Araku stays", state: "completed" }],
      holidays: {
        items: [
          {
            title: "Regional holiday",
            detail: "Expect busier transport after lunch.",
            sourceLabel: "Nager.Date"
          }
        ],
        advisories: [],
        summary: "There is one holiday note during this window."
      }
    }
  });

  assert.ok(!blocks.some((block) => block.type === "worker_progress"));
  assert.ok(blocks.some((block) => block.type === "date_advisory"));
  assert.ok(blocks.some((block) => block.type === "event_window_summary"));
  assert.ok(blocks.some((block) => block.type === "stay_recommendation_list"));
});

test("resolveTurnIntent resolves yes against pending stay follow-up context", async () => {
  const session = makeSession();
  session.memory.pendingFollowUp = {
    primaryDomain: "stays",
    destination: "Araku",
    startDate: "2026-07-02",
    endDate: "2026-07-03",
    categoryKeys: [],
    poiIds: ["poi-1"],
    options: [{ domain: "stays", label: "Stay options", prompt: "Show me stay options" }]
  };

  const providerService = buildProviderStub({
    intent: "question",
    destinations: [],
    styles: []
  });

  const resolution = await resolveTurnIntent(providerService, {} as any, session, "yes");
  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.stayMode, true);
  assert.equal(resolution.destination, "Araku");
});

test("resolveTurnIntent resolves yes against pending restaurant follow-up context", async () => {
  const session = makeSession();
  session.memory.pendingFollowUp = {
    primaryDomain: "restaurants",
    destination: "Goa",
    poiIds: ["poi-2"],
    categoryKeys: ["famous", "budget_friendly"],
    options: [{ domain: "restaurants", label: "Restaurants", prompt: "Show me restaurants" }]
  };

  const providerService = buildProviderStub({
    intent: "question",
    destinations: [],
    styles: []
  });

  const resolution = await resolveTurnIntent(providerService, {} as any, session, "sure");
  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.questionFocus, "restaurants");
  assert.equal(resolution.destination, "Goa");
});

test("resolveDeterministicTurnIntent lets an explicit stay request override restaurant follow-up context", () => {
  const session = makeSession();
  session.memory.pendingFollowUp = {
    primaryDomain: "restaurants",
    destination: "Goa",
    poiIds: ["poi-2"],
    categoryKeys: ["famous"],
    options: [{ domain: "restaurants", label: "Restaurants", prompt: "Show me restaurants" }]
  };
  session.memory.destinationsDiscussed = ["Goa"];

  const resolution = resolveDeterministicTurnIntent(session, "show me some stays");

  assert.ok(resolution);
  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.questionFocus, "hotels");
  assert.equal(resolution.followUpDomain, "stays");
  assert.equal(resolution.stayMode, true);
  assert.equal(resolution.destination, "Goa");
});

test("resolveTurnIntent lets an explicit restaurant request override stale attraction follow-up context", async () => {
  const session = makeSession();
  session.memory.pendingFollowUp = {
    destination: "Goa",
    categoryKeys: [],
    poiIds: ["poi-1"],
    options: [
      { domain: "activities", label: "Add to itinerary", prompt: "Add to itinerary" },
      { domain: "attractions", label: "More nature spots", prompt: "More nature spots" },
      { domain: "dates", label: "Best time to visit", prompt: "Best time to visit" }
    ]
  };

  const resolution = await resolveTurnIntent(
    buildProviderStub({
      intent: "question",
      destinations: [],
      styles: []
    }),
    {} as any,
    session,
    "show me some restaurants"
  );

  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.questionFocus, "restaurants");
  assert.equal(resolution.followUpAmbiguous, false);
  assert.equal(resolution.destination, "Goa");
});

test("resolveTurnIntent lets an explicit hotel request override stale attraction follow-up context", async () => {
  const session = makeSession();
  session.memory.pendingFollowUp = {
    destination: "Goa",
    categoryKeys: [],
    poiIds: ["poi-1"],
    options: [
      { domain: "activities", label: "Add to itinerary", prompt: "Add to itinerary" },
      { domain: "attractions", label: "More nature spots", prompt: "More nature spots" },
      { domain: "dates", label: "Best time to visit", prompt: "Best time to visit" }
    ]
  };

  const resolution = await resolveTurnIntent(
    buildProviderStub({
      intent: "question",
      destinations: [],
      styles: []
    }),
    {} as any,
    session,
    "show me some hotels"
  );

  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.questionFocus, "hotels");
  assert.equal(resolution.stayMode, true);
  assert.equal(resolution.followUpAmbiguous, false);
  assert.equal(resolution.destination, "Goa");
});

test("resolveTurnIntent treats explicit stay requests as discovery, not itinerary edits", async () => {
  const session = makeSession();
  session.plan = {
    schemaVersion: 1,
    sessionId: "session-1",
    version: 1,
    title: "Goa trip",
    destination: "Goa",
    destinations: ["Goa"],
    totalDays: 4,
    travelerCount: 2,
    notes: [],
    destinationSegments: [],
    days: [],
    generatedAt: "2026-04-01T00:00:00.000Z",
    lastUserIntent: "plan_trip"
  };

  const resolution = await resolveTurnIntent(
    buildProviderStub({
      intent: "refine_trip",
      destinations: ["Goa"],
      styles: []
    }),
    {} as any,
    session,
    "show me some stays"
  );

  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.questionFocus, "hotels");
  assert.equal(resolution.stayMode, true);
});

test("resolveTurnIntent treats explicit restaurant requests as discovery, not itinerary edits", async () => {
  const session = makeSession();
  session.plan = {
    schemaVersion: 1,
    sessionId: "session-1",
    version: 1,
    title: "Goa trip",
    destination: "Goa",
    destinations: ["Goa"],
    totalDays: 4,
    travelerCount: 2,
    notes: [],
    destinationSegments: [],
    days: [],
    generatedAt: "2026-04-01T00:00:00.000Z",
    lastUserIntent: "plan_trip"
  };

  const resolution = await resolveTurnIntent(
    buildProviderStub({
      intent: "refine_trip",
      destinations: ["Goa"],
      styles: []
    }),
    {} as any,
    session,
    "show me some restaurants"
  );

  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.questionFocus, "restaurants");
});

test("resolveTurnIntent treats food places as restaurant discovery", async () => {
  const session = makeSession();
  session.memory.destinationsDiscussed = ["Goa"];

  const resolution = await resolveTurnIntent(
    buildProviderStub({
      intent: "question",
      destinations: [],
      styles: []
    }),
    {} as any,
    session,
    "show me some food places"
  );

  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.questionFocus, "restaurants");
  assert.equal(resolution.destination, "Goa");
});

test("resolveTurnIntent refines restaurant category from follow-up text", async () => {
  const session = makeSession();
  session.memory.pendingFollowUp = {
    primaryDomain: "restaurants",
    destination: "Goa",
    poiIds: ["poi-2"],
    categoryKeys: ["famous", "budget_friendly", "vegetarian", "non_vegetarian", "premium"],
    options: [{ domain: "restaurants", label: "Restaurants", prompt: "Show me restaurants" }]
  };

  const providerService = buildProviderStub({
    intent: "question",
    destinations: [],
    styles: []
  });

  const resolution = await resolveTurnIntent(providerService, {} as any, session, "show me cheaper ones");
  assert.equal(resolution.intent, "search_places");
  assert.equal(resolution.restaurantCategory, "budget_friendly");
});

test("resolveTurnIntent asks to clarify ambiguous follow-up branches", async () => {
  const session = makeSession();
  session.memory.pendingFollowUp = {
    destination: "Araku",
    categoryKeys: [],
    poiIds: [],
    options: [
      { domain: "stays", label: "Stay options", prompt: "Show me stay options" },
      { domain: "restaurants", label: "Restaurants", prompt: "Show me restaurants" }
    ]
  };

  const providerService = buildProviderStub({
    intent: "question",
    destinations: [],
    styles: []
  });

  const resolution = await resolveTurnIntent(providerService, {} as any, session, "yes");
  assert.equal(resolution.intent, "question");
  assert.equal(resolution.questionFocus, "followup_clarify");
});

test("buildResponseBlocks stay discovery never emits restaurant carousels", () => {
  const session = makeSession();
  const stay = {
    id: "stay-1",
    name: "Goa Palm Retreat",
    type: "stay" as const,
    lat: 15.5,
    lng: 73.8,
    openingHours: [],
    source: "google_places" as const,
    tags: ["Goa"]
  };
  const restaurant = {
    id: "restaurant-1",
    name: "Maka Zai Goan Restaurant",
    type: "restaurant" as const,
    lat: 15.51,
    lng: 73.81,
    openingHours: [],
    source: "google_places" as const,
    tags: ["Goa"]
  };

  session.poiCatalog.items[stay.id] = stay;
  session.poiCatalog.items[restaurant.id] = restaurant;

  const blocks = buildResponseBlocks({
    session,
    resolution: {
      intent: "search_places",
      destination: "Goa",
      destinations: ["Goa"],
      styles: [],
      questionFocus: "hotels",
      stayMode: true,
      dateContext: {
        flexibility: "open_ended",
        derivedFrom: "none",
        advisoryItems: []
      }
    },
    narrative: {
      introTitle: "Your Goa stay options",
      introBody: "Goa has a good mix of stays for this trip.",
      leadText: "Here are the strongest stay picks first.",
      promptChips: [],
      clarifyingQuestions: []
    },
    research: {
      destinations: ["Goa"],
      focus: "hotels",
      catalog: {
        version: 1,
        items: {
          [stay.id]: stay,
          [restaurant.id]: restaurant
        }
      },
      grouped: {
        stays: [stay],
        restaurants: [restaurant],
        attractions: []
      },
      facts: []
    }
  });

  assert.ok(blocks.some((block) => block.type === "stay_recommendation_list"));
  assert.ok(!blocks.some((block) => block.type === "categorized_place_rows"));
  const row = blocks.find((block) => block.type === "place_card_row");
  assert.ok(row && row.type === "place_card_row");
  assert.deepEqual(row.poiIds, [stay.id]);
});
