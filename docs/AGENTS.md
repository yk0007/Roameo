# Roameo Agent Index

Quick reference for the canonical per-turn pipeline. The full narrative and agent contract details live in [AUTONOMOUS_AGENTS.md](/Users/yk0007/MyRepos/Roameo/docs/AUTONOMOUS_AGENTS.md).

## Pipeline order

| # | Agent | Function | File | Type |
|---|---|---|---|---|
| 1 | Intent Resolver | `resolveTurnIntent()` | `Backend/src/runtime/subagents.ts` | LLM, Zod-validated |
| 2 | Research Gate | `shouldResearchResolution()` | `Backend/src/runtime/subagents.ts` | Pure predicate |
| 3 | Discovery Agent | `researchDestinations()` | `Backend/src/runtime/subagents.ts` | Async, Google Places plus optional Tavily |
| 4 | Date Context Agent | `getWeatherSummary()`, `getEventSummary()`, `getHolidaySummary()` | `Backend/src/services/travel-tools.ts` | Async, Open-Meteo + Tavily + Nager.Date |
| 5 | Itinerary Planner | `synthesizePlan()` | `Backend/src/runtime/subagents.ts` | Structured LLM call for planning/refinement turns |
| 6 | Logistics Enricher | `enrichPlanLogistics()` | `Backend/src/runtime/subagents.ts` | Async, Google Directions |
| 7 | Feasibility Critic | `criticizeAndRefinePlan()` | `Backend/src/runtime/subagents.ts` | Pure sync |
| 8 | Transit Advisor | `transitAdvisor()` | `Backend/src/runtime/subagents.ts` | Pure sync |
| 9 | Narrator | `answerConversationally()` + `buildResponseBlocks()` | `Backend/src/runtime/subagents.ts` | LLM plus deterministic block assembly |
| 10 | Session Commit | `updateSessionMemory()` + final session update | `Backend/src/runtime/subagents.ts`, `Backend/src/runtime/turn-runner.ts` | Async persistence |

## LLM call pattern

- All turns use two model calls: intent resolution and narration.
- `plan_trip` and `refine_trip` turns add a third structured planning call through `synthesizePlan()`.

## Core contracts

- `SessionSnapshot`: persisted source of truth for the workspace
- `TurnResolution`: typed intent, destinations, date context, and follow-up metadata
- `DestinationResearch`: grouped POIs, canonical catalog, and optional fact research
- `PlanSnapshot`: trip overview, destination segments, days, budgets, and notes
- `PlanningContext`: weather, events, holidays, and worker progress for narration
- `StreamEvent`: the SSE event schema shared between backend and frontend

## Delivery rules

- Keep new planning logic in `Backend/src/runtime/subagents.ts` or `Backend/src/services/travel-tools.ts`.
- Keep route handlers thin; they should validate, orchestrate, and persist.
- Emit agent traces through the turn runner so the frontend can surface live progress.
- End successful turns with `planningState.status = "ready"`.
- Apply Google Places `type` filtering on every `searchPlacesByQueries()` call.
