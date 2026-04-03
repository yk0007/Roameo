import test from "node:test";
import assert from "node:assert/strict";
import {
  assistantResponseBlockSchema,
  planMutationInputSchema,
  planSnapshotSchema,
  planningStateSchema,
  sessionMutationSchema,
  sessionSnapshotSchema
} from "./index.js";

test("session snapshot fills canonical defaults", () => {
  const snapshot = sessionSnapshotSchema.parse({
    id: "session-1",
    providerSettings: { provider: "gemini", runMode: "balanced", keySource: "platform" },
    memory: {},
    createdAt: "2026-04-01T00:00:00.000Z",
    updatedAt: "2026-04-01T00:00:00.000Z"
  });

  assert.equal(snapshot.title, "Untitled trip");
  assert.deepEqual(snapshot.savedPoiIds, []);
  assert.equal(snapshot.memory.preferences.currency, "INR");
  assert.equal(snapshot.memory.planningState.stage, "ready");
  assert.equal(snapshot.memory.dateContext.flexibility, "open_ended");
});

test("plan snapshot keeps schema version and day structure", () => {
  const plan = planSnapshotSchema.parse({
    schemaVersion: 1,
    sessionId: "session-1",
    version: 2,
    title: "Japan Spring Circuit",
    destinations: ["Tokyo"],
    totalDays: 3,
    travelerCount: 2,
    destinationSegments: [],
    days: [
      {
        day: 1,
        date: "2026-04-01",
        title: "Arrival",
        destination: "Tokyo",
        activities: []
      }
    ],
    generatedAt: "2026-04-01T00:00:00.000Z",
    lastUserIntent: "plan_trip"
  });

  assert.equal(plan.days.length, 1);
  assert.equal(plan.days[0].destination, "Tokyo");
  assert.equal(plan.startDate, undefined);
});

test("session mutation accepts memory updates", () => {
  const mutation = sessionMutationSchema.parse({
    title: "Updated title",
    memory: {
      summary: "A short summary",
      destinationsDiscussed: ["Lisbon"],
      acceptedDecisions: ["Duration: 4 days"],
      lastPlanVersion: 1,
      dateContext: {
        requestedStartDate: "2026-05-01",
        requestedEndDate: "2026-05-03",
        flexibility: "exact",
        derivedFrom: "explicit",
        advisoryItems: []
      },
      preferences: {
        currency: "EUR",
        locale: "en-US",
        styles: ["culture"],
        dietaryNotes: [],
        accessibilityNotes: []
      }
    }
  });

  assert.equal(mutation.memory?.lastPlanVersion, 1);
  assert.deepEqual(mutation.memory?.preferences.styles, ["culture"]);
  assert.equal(mutation.memory?.dateContext?.flexibility, "exact");
});

test("plan mutation accepts direct structured actions", () => {
  const add = planMutationInputSchema.parse({
    type: "add_poi",
    poiId: "poi-1",
    day: 2
  });
  const overview = planMutationInputSchema.parse({
    type: "update_overview",
    destination: "Kyoto",
    destinations: ["Osaka", "Kyoto"],
    startDate: "2026-05-01",
    endDate: "2026-05-04",
    dateFlexibility: "exact",
    totalDays: 4,
    travelerCount: 2,
    budgetTotal: 125000,
    currency: "JPY"
  });

  assert.equal(add.type, "add_poi");
  assert.equal(overview.type, "update_overview");
  assert.equal(overview.destination, "Kyoto");
  assert.deepEqual(overview.destinations, ["Osaka", "Kyoto"]);
  assert.equal(overview.startDate, "2026-05-01");
  assert.equal(overview.dateFlexibility, "exact");
});

test("planning state accepts stage-aware canonical progress", () => {
  const state = planningStateSchema.parse({
    status: "running",
    stage: "researching",
    source: "places",
    retryable: true
  });

  assert.equal(state.status, "running");
  assert.equal(state.stage, "researching");
});

test("planning state accepts date and stay specific canonical progress", () => {
  const state = planningStateSchema.parse({
    status: "running",
    stage: "researching_stays",
    source: "stays",
    retryable: true
  });

  assert.equal(state.stage, "researching_stays");
  assert.equal(state.source, "stays");
});

test("assistant response blocks accept canonical itinerary presentation blocks", () => {
  const intro = assistantResponseBlockSchema.parse({
    type: "trip_intro",
    title: "Kerala is shaping up beautifully",
    body: "I pulled together a relaxed first draft you can keep refining.",
    moodEmoji: "🌴"
  });
  const itinerary = assistantResponseBlockSchema.parse({
    type: "itinerary_template",
    title: "3-day Kerala escape",
    subtitle: "Relaxed pace with scenic stops",
    days: [
      {
        day: 1,
        title: "Fort Kochi by the water",
        destination: "Kochi",
        periods: [
          {
            key: "morning",
            label: "Morning",
            emoji: "☀️",
            entries: [
              {
                title: "Fort Kochi walk",
                poiId: "poi-1",
                timeLabel: "9:00 AM"
              }
            ]
          }
        ]
      }
    ]
  });

  assert.equal(intro.type, "trip_intro");
  assert.equal(itinerary.type, "itinerary_template");
});

test("assistant response blocks accept capability and discovery presentation blocks", () => {
  const capabilities = assistantResponseBlockSchema.parse({
    type: "capabilities_overview",
    title: "What I can do for you",
    intro: "I can help you explore and plan with real places.",
    sections: [
      {
        title: "Recommendations",
        body: "Find restaurants, attractions, stays, and local gems."
      }
    ],
    examples: ["Best seafood nearby?", "Plan a 2-day trip to Araku"]
  });
  const stories = assistantResponseBlockSchema.parse({
    type: "poi_story_list",
    title: "Hidden gems",
    items: [
      {
        poiId: "poi-1",
        title: "Titanic Sea View Point",
        badge: "Quiet local pick",
        body: "A strong sunrise stop with wider bay views than the usual waterfront."
      }
    ]
  });
  const featuredPoi = assistantResponseBlockSchema.parse({
    type: "featured_poi",
    poiId: "poi-1",
    title: "A good place to start",
    body: "A local favorite to anchor the conversation."
  });
  const workerProgress = assistantResponseBlockSchema.parse({
    type: "worker_progress",
    title: "Thinking longer",
    steps: [
      {
        label: "Scouting Araku stays",
        detail: "Checking the strongest stay options for your route.",
        state: "running"
      }
    ]
  });
  const stayList = assistantResponseBlockSchema.parse({
    type: "stay_recommendation_list",
    title: "Best stays for Araku",
    bestOption: {
      poiId: "poi-1",
      title: "Best option",
      body: "A comfortable base close to the core sights."
    }
  });
  const dateAdvisory = assistantResponseBlockSchema.parse({
    type: "date_advisory",
    title: "A small timing note",
    summary: "A nearby weekend may work better for weather and crowds.",
    advisories: [
      {
        kind: "prefer",
        title: "Prefer the following weekend",
        detail: "The weather looks easier and the festival crowds taper off."
      }
    ]
  });
  const eventSummary = assistantResponseBlockSchema.parse({
    type: "event_window_summary",
    title: "Around your dates",
    items: [
      {
        title: "Regional holiday",
        detail: "Expect heavier local movement on the first evening."
      }
    ]
  });

  assert.equal(capabilities.type, "capabilities_overview");
  assert.equal(stories.type, "poi_story_list");
  assert.equal(featuredPoi.type, "featured_poi");
  assert.equal(workerProgress.type, "worker_progress");
  assert.equal(stayList.type, "stay_recommendation_list");
  assert.equal(dateAdvisory.type, "date_advisory");
  assert.equal(eventSummary.type, "event_window_summary");
});

test("assistant response blocks fail fast on invalid itinerary template payloads", () => {
  assert.throws(
    () =>
      assistantResponseBlockSchema.parse({
        type: "itinerary_template",
        title: "Missing day payload",
        days: []
    }),
    /Too small|at least 1/
  );
});

test("assistant response blocks fail fast on invalid discovery payloads", () => {
  assert.throws(
    () =>
      assistantResponseBlockSchema.parse({
        type: "poi_story_list",
        title: "Missing stories",
        items: []
      }),
    /Too small|at least 1/
  );
});
