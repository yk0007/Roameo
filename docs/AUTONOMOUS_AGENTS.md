# Autonomous Agents Architecture

Roameo’s canonical agent model is a router-first runtime with deterministic tool execution and snapshot-based state commits.

The product goal is:

- semantic understanding by model
- explicit tool planning and execution in code
- one canonical session state
- no parallel truth systems for chat, itinerary, map, or saved POIs

## Core principle

Most of the “agents” are plain TypeScript stages coordinated by [Backend/src/runtime/turn-runner.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/turn-runner.ts).

LLMs are used where meaning or synthesis is genuinely needed:

- turn understanding / routing
- narrative generation
- structured itinerary synthesis for plan/refine turns

Everything else should stay deterministic:

- provider resolution
- tool selection and execution
- itinerary mutation
- memory updates
- POI catalog persistence
- stream-event emission

## Current per-turn pipeline

1. Persist user message.
2. Run fast conversational short-circuit only for trivial non-travel turns.
3. Resolve turn intent and semantic focus with `resolveTurnIntent()`.
4. Decide whether discovery/research is required.
5. Run destination discovery and date/event/weather enrichment when required.
6. For `plan_trip` / `refine_trip`, synthesize a structured plan and enrich logistics.
7. Run deterministic feasibility and transit passes.
8. Generate the final assistant narrative and structured blocks.
9. Derive and persist the next follow-up context.
10. Update canonical session memory and emit final snapshot events.

## Current role split

### 1. Conversational fast path

File: [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)

`resolveFastTurnResponse()` handles:

- greetings
- acknowledgements
- identity / capability questions
- simple small talk

These turns skip the heavy planning pipeline and update memory directly.

### 2. Semantic router

File: [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)

`resolveTurnIntent()` is the primary travel understanding layer.

Current model routing:

- router / understanding: `gemma-4-31b-it`
- fallback: `gemini-2.5-flash`

The router is responsible for:

- intent
- active domain
- destination scope
- date context
- traveler count
- style cues
- whether the turn is a new trip, refinement, or discovery

The runtime still keeps minimal deterministic guards for:

- trivial fast-path conversation
- shorthand follow-ups like `yes` / `that one`
- fail-fast state validation

### 3. Discovery and grounding

Files:

- [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)
- [Backend/src/services/travel-tools.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/travel-tools.ts)

Canonical data grounding remains direct:

- Google Places for POIs
- Google Geocoding
- Google Directions / Maps
- Open-Meteo
- Tavily
- Nager.Date

Current rule:

- direct Places/Maps is the canonical POI truth
- model-based Maps grounding is useful as future semantic reranking/expansion, not as the canonical catalog source

### 4. Itinerary synthesis

File: [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)

`synthesizePlan()` uses structured planning output plus deterministic hospitality repair:

- real stay assignment for overnight days
- restaurant-linked meal stops when the request implies food/cuisine support
- no fabricated POIs outside the catalog

Current model routing:

- narrative / plan synthesis: `gemini-flash-latest`
- fallback: `gemini-2.5-flash`

### 5. Internal agent tools

File: [Backend/src/services/agent-tool-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/agent-tool-service.ts)

Current first-class internal tools:

- `getSessionSnapshot`
- `updateTripHeader`
- `editItinerary`
- `updateSessionMemory`
- `resetActiveTripContext`
- `saveFollowUpContext`

These are the canonical mutation primitives the agent system should build on.

They intentionally route through:

- [Backend/src/services/session-repository.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/session-repository.ts)
- [Backend/src/services/plan-mutation-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/plan-mutation-service.ts)

so state writes stay centralized.

## Context management rules

Roameo’s agent system must maintain and invalidate context at the session level.

Current invariants:

- explicit new-trip requests replace stale active-trip context
- multi-city trips preserve the full active destination set
- explicit new discovery asks override stale follow-up context
- itinerary and map route updates happen only when the canonical plan changes
- discovery POIs accumulate in the session catalog across turns instead of being overwritten

Important current helpers:

- `updateSessionMemory()`
- `derivePendingFollowUpContext()`
- `resetActiveTripContext()` in `AgentToolService`

## What “autonomous” means here

For Roameo, autonomous should mean:

- understands the current user intent from the whole conversation
- selects the right tool plan
- modifies canonical session state only through approved tool surfaces
- re-reads the session after tool execution
- narrates the result from the updated canonical state

It should not mean:

- uncontrolled freeform state mutation
- parallel business logic paths
- hidden non-canonical frontend-only state
- inventing route/POI truth outside the catalog
