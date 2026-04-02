import test from "node:test";
import assert from "node:assert/strict";
import { normalizeTravelIntent } from "./subagents.js";

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
