import test from "node:test";
import assert from "node:assert/strict";
import {
  buildResponseBlocks,
  normalizeTravelIntent,
  resolveTurnIntent
} from "./subagents.js";
import type { SessionSnapshot } from "@roameo/contracts";

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
    traces: [],
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

  const row = blocks.find((block) => block.type === "place_card_row");
  assert.ok(row && row.type === "place_card_row");
  assert.deepEqual(row.poiIds, ["poi-1"]);
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

  const providerService = {
    generateObject: async () => ({
      intent: "question",
      destinations: [],
      styles: []
    })
  } as any;

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

  const providerService = {
    generateObject: async () => ({
      intent: "question",
      destinations: [],
      styles: []
    })
  } as any;

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
  const providerService = {
    generateObject: async () => ({
      intent: "question",
      destinations: [],
      styles: []
    })
  } as any;

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
  const providerService = {
    generateObject: async () => ({
      intent: "question",
      destinations: [],
      styles: []
    })
  } as any;

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
  const row = blocks.find((block) => block.type === "place_card_row");
  assert.ok(row && row.type === "place_card_row");
  assert.deepEqual(row.poiIds, ["poi-1"]);
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

  assert.ok(blocks.some((block) => block.type === "worker_progress"));
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

  const providerService = {
    generateObject: async () => ({
      intent: "question",
      destinations: [],
      styles: []
    })
  } as any;

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

  const providerService = {
    generateObject: async () => ({
      intent: "question",
      destinations: [],
      styles: []
    })
  } as any;

  const resolution = await resolveTurnIntent(providerService, {} as any, session, "sure");
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

  const providerService = {
    generateObject: async () => ({
      intent: "question",
      destinations: [],
      styles: []
    })
  } as any;

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

  const providerService = {
    generateObject: async () => ({
      intent: "question",
      destinations: [],
      styles: []
    })
  } as any;

  const resolution = await resolveTurnIntent(providerService, {} as any, session, "yes");
  assert.equal(resolution.intent, "question");
  assert.equal(resolution.questionFocus, "followup_clarify");
});
