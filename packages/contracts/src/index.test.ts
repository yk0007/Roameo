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
});

test("session mutation accepts memory updates", () => {
  const mutation = sessionMutationSchema.parse({
    title: "Updated title",
    memory: {
      summary: "A short summary",
      destinationsDiscussed: ["Lisbon"],
      acceptedDecisions: ["Duration: 4 days"],
      lastPlanVersion: 1,
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
    totalDays: 4,
    travelerCount: 2,
    budgetTotal: 125000,
    currency: "JPY"
  });

  assert.equal(add.type, "add_poi");
  assert.equal(overview.type, "update_overview");
  assert.equal(overview.destination, "Kyoto");
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
