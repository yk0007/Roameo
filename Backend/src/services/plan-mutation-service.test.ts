import test from "node:test";
import assert from "node:assert/strict";
import type { PlanSnapshot, PoiCatalog } from "@roameo/contracts";
import { ProviderService } from "./provider-service.js";
import { PlanMutationService } from "./plan-mutation-service.js";
import { SessionRepository } from "./session-repository.js";
import { TravelToolsService } from "./travel-tools.js";

class ThrowingProviderService extends ProviderService {
  override async resolveProvider() {
    return {
      provider: "gemini" as const,
      model: "gemini-2.5-flash",
      keySource: "platform" as const,
      apiKey: "test-key"
    };
  }

  override async generateObject<T>() : Promise<T> {
    throw new Error("provider unavailable");
  }
}

function buildCatalog(): PoiCatalog {
  return {
    version: 1,
    items: {
      "stay-1": {
        id: "stay-1",
        name: "Cinnamon House",
        type: "stay",
        lat: 12.9716,
        lng: 77.5946,
        address: "Bengaluru",
        source: "manual",
        openingHours: [],
        tags: ["Bengaluru"]
      },
      "poi-1": {
        id: "poi-1",
        name: "Cubbon Park",
        type: "attraction",
        lat: 12.9763,
        lng: 77.5929,
        address: "Bengaluru",
        source: "manual",
        openingHours: [],
        tags: ["Bengaluru"]
      },
      "poi-2": {
        id: "poi-2",
        name: "Indian Coffee House",
        type: "restaurant",
        lat: 12.9758,
        lng: 77.6034,
        address: "Bengaluru",
        source: "manual",
        openingHours: [],
        tags: ["Bengaluru"]
      },
      "poi-3": {
        id: "poi-3",
        name: "Lalbagh Botanical Garden",
        type: "attraction",
        lat: 12.9507,
        lng: 77.5848,
        address: "Bengaluru",
        source: "manual",
        openingHours: [],
        tags: ["Bengaluru"]
      }
    }
  };
}

function buildPlan(sessionId: string): PlanSnapshot {
  return {
    schemaVersion: 1,
    sessionId,
    version: 1,
    title: "Bengaluru Weekend",
    destination: "Bengaluru",
    destinations: ["Bengaluru"],
    totalDays: 2,
    travelerCount: 2,
    notes: [],
    destinationSegments: [
      {
        destination: "Bengaluru",
        startDay: 1,
        endDay: 2,
        nights: 1
      }
    ],
    days: [
      {
        day: 1,
        date: "2026-04-10",
        title: "Garden city arrival",
        destination: "Bengaluru",
        accommodationPoiId: "stay-1",
        activities: [
          {
            id: "activity-1",
            poiId: "poi-1",
            title: "Cubbon Park",
            startTime: "09:00",
            endTime: "11:00",
            notes: []
          }
        ]
      },
      {
        day: 2,
        date: "2026-04-11",
        title: "Slow day",
        destination: "Bengaluru",
        activities: []
      }
    ],
    generatedAt: "2026-04-01T00:00:00.000Z",
    lastUserIntent: "plan_trip"
  };
}

test("plan mutations persist canonical itinerary changes", async () => {
  const repository = new SessionRepository();
  const providerService = new ProviderService(repository);
  const service = new PlanMutationService(
    repository,
    providerService,
    new TravelToolsService()
  );

  const session = await repository.createSession("user-1", {
    title: "Bengaluru Weekend"
  });
  await repository.savePlan(session.id, buildPlan(session.id), buildCatalog());

  const added = await service.apply(
    (await repository.getSession(session.id, "user-1"))!,
    "user-1",
    {
      type: "add_poi",
      poiId: "poi-2"
    }
  );

  assert.equal(added.plan?.version, 2);
  assert.ok(
    added.plan?.days.some((day) =>
      day.activities.some((activity) => activity.poiId === "poi-2")
    )
  );
  assert.ok(
    added.messages.some((message) =>
      message.content.includes("Added Indian Coffee House")
    )
  );

  const removed = await service.apply(added, "user-1", {
    type: "remove_poi",
    poiId: "poi-2"
  });

  assert.equal(removed.plan?.version, 3);
  assert.ok(
    removed.plan?.days.every((day) =>
      day.activities.every((activity) => activity.poiId !== "poi-2")
    )
  );
});

test("overview mutations keep title and budget target in the plan snapshot", async () => {
  const repository = new SessionRepository();
  const providerService = new ProviderService(repository);
  const service = new PlanMutationService(
    repository,
    providerService,
    new TravelToolsService()
  );

  const session = await repository.createSession("user-2", {
    title: "Bengaluru Weekend"
  });
  await repository.savePlan(session.id, buildPlan(session.id), buildCatalog());

  const updated = await service.apply(
    (await repository.getSession(session.id, "user-2"))!,
    "user-2",
    {
      type: "update_overview",
      title: "Bengaluru Food and Gardens",
      travelerCount: 3,
      budgetTotal: 45000,
      currency: "INR"
    }
  );

  assert.equal(updated.title, "Bengaluru Food and Gardens");
  assert.equal(updated.plan?.title, "Bengaluru Food and Gardens");
  assert.equal(updated.plan?.travelerCount, 3);
  assert.equal(updated.plan?.budgetTarget?.total, 45000);
  assert.equal(updated.plan?.budgetTarget?.currency, "INR");
});

test("overview mutations persist canonical pre-plan trip details in memory", async () => {
  const repository = new SessionRepository();
  const providerService = new ProviderService(repository);
  const service = new PlanMutationService(
    repository,
    providerService,
    new TravelToolsService()
  );

  const session = await repository.createSession("user-4", {
    title: "Untitled trip"
  });

  const updated = await service.apply(
    (await repository.getSession(session.id, "user-4"))!,
    "user-4",
    {
      type: "update_overview",
      origin: "Shillong",
      destination: "Meghalaya",
      totalDays: 3,
      travelerCount: 2
    }
  );

  assert.ok(updated.memory.acceptedDecisions.includes("Origin: Shillong"));
  assert.ok(updated.memory.acceptedDecisions.includes("Destination: Meghalaya"));
  assert.ok(updated.memory.acceptedDecisions.includes("Duration: 3 days"));
  assert.ok(updated.memory.acceptedDecisions.includes("Travelers: 2"));
  assert.deepEqual(updated.memory.destinationsDiscussed, ["Meghalaya"]);
});

test("rebalance_trip fails fast instead of generating fallback POIs when provider generation fails", async () => {
  const repository = new SessionRepository();
  const providerService = new ThrowingProviderService(repository);
  const service = new PlanMutationService(
    repository,
    providerService,
    new TravelToolsService()
  );

  const session = await repository.createSession("user-3", {
    title: "Bengaluru Weekend"
  });
  await repository.savePlan(session.id, buildPlan(session.id), buildCatalog());

  await assert.rejects(
    service.apply((await repository.getSession(session.id, "user-3"))!, "user-3", {
      type: "rebalance_trip"
    }),
    /provider unavailable/
  );
});
