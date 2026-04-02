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
      styles: []
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
      styles: []
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
