import test from "node:test";
import assert from "node:assert/strict";
import { SessionRepository } from "./session-repository.js";

test("session repository persists in-memory canonical session state", async () => {
  const repository = new SessionRepository();
  const created = await repository.createSession("user-1", {
    title: "Summer in Japan"
  });

  assert.equal(created.title, "Summer in Japan");

  await repository.saveMessage({
    id: "message-1",
    sessionId: created.id,
    role: "user",
    content: "Plan 5 days in Tokyo",
    createdAt: "2026-04-01T00:00:00.000Z",
    meta: {}
  });

  const updated = await repository.updateSession(created.id, {
    memory: {
      ...created.memory,
      summary: "Planning Tokyo",
      destinationsDiscussed: ["Tokyo"],
      acceptedDecisions: ["Duration: 5 days"],
      lastPlanVersion: 1,
      preferences: created.memory.preferences
    }
  });

  assert.ok(updated);
  assert.equal(updated?.messages.length, 1);
  assert.equal(updated?.memory.summary, "Planning Tokyo");
});
