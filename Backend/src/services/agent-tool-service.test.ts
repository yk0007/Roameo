import test from "node:test";
import assert from "node:assert/strict";
import { ProviderService } from "./provider-service.js";
import { SessionRepository } from "./session-repository.js";
import { PlanMutationService } from "./plan-mutation-service.js";
import { TravelToolsService } from "./travel-tools.js";
import { AgentToolService } from "./agent-tool-service.js";

test("agent tools can reset active trip context for a new trip", async () => {
  const repository = new SessionRepository();
  const providerService = new ProviderService(repository);
  const planMutationService = new PlanMutationService(
    repository,
    providerService,
    new TravelToolsService()
  );
  const tools = new AgentToolService(repository, planMutationService);

  const session = await repository.createSession("user-ctx", {
    title: "Paris"
  });
  await repository.updateSession(session.id, {
    memory: {
      ...session.memory,
      destinationsDiscussed: ["Paris"],
      acceptedDecisions: ["Destination: Paris", "Duration: 5 days"]
    }
  });

  const updated = await tools.resetActiveTripContext(session.id, "user-ctx", {
    destination: "Araku",
    destinations: ["Araku"],
    totalDays: 2,
    explicitNewTrip: true
  });

  assert.deepEqual(updated.memory.destinationsDiscussed, ["Araku"]);
  assert.ok(updated.memory.acceptedDecisions.includes("Destination: Araku"));
  assert.ok(!updated.memory.acceptedDecisions.includes("Destination: Paris"));
});

test("agent tools can update session memory and clear follow-up context", async () => {
  const repository = new SessionRepository();
  const providerService = new ProviderService(repository);
  const planMutationService = new PlanMutationService(
    repository,
    providerService,
    new TravelToolsService()
  );
  const tools = new AgentToolService(repository, planMutationService);

  const session = await repository.createSession("user-mem", {
    title: "Memory"
  });

  const updated = await tools.updateSessionMemory(session.id, "user-mem", {
    appendDestinationsDiscussed: ["Goa"],
    appendAcceptedDecisions: ["Destination: Goa"],
    pendingFollowUp: {
      destination: "Goa",
      primaryDomain: "restaurants",
      categoryKeys: [],
      poiIds: [],
      options: []
    }
  });

  assert.ok(updated.memory.destinationsDiscussed.includes("Goa"));
  assert.equal(updated.memory.pendingFollowUp?.primaryDomain, "restaurants");

  const cleared = await tools.updateSessionMemory(session.id, "user-mem", {
    clearPendingFollowUp: true
  });
  assert.equal(cleared.memory.pendingFollowUp, null);
});
