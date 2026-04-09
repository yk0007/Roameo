# Planning Runtime

The canonical planning runtime lives in:

- [Backend/src/runtime/turn-runner.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/turn-runner.ts)
- [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)
- [Backend/src/services/travel-tools.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/travel-tools.ts)
- [Backend/src/services/plan-mutation-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/plan-mutation-service.ts)
- [Backend/src/services/agent-tool-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/agent-tool-service.ts)

## Canonical intents

The shared `travelIntentSchema` defines:

- `plan_trip`
- `refine_trip`
- `search_places`
- `question`
- `settings`
- `meta`

The runtime derives conversational modes on top of those intents:

- greeting
- missing-info
- discovery
- planning
- refinement

## Current execution model

### 1. Fast path

Non-travel trivial messages are handled by `resolveFastTurnResponse()`.

This path exists to avoid paying the full planning pipeline for:

- greetings
- acknowledgements
- identity/capability questions
- low-value small talk

### 2. Semantic router

Normal travel turns should flow through the semantic router, not a keyword-only local branch.

Current task-oriented Gemini routing:

- router: `gemma-4-31b-it`
- fallback: `gemini-2.5-flash`

The router resolves:

- intent
- destination scope
- traveler count
- style cues
- date context
- active domain
- whether the turn is a new trip or follow-up refinement

### 3. Discovery and enrichment

Discovery builds a canonical `PoiCatalog` from live integrations.

Current source types:

- `google_places`
- `google_maps`
- `web_research`
- `manual`

Rules:

- every Places request applies strict category filtering
- planning turns use `researchPlanningDestinations()` so hospitality categories remain available even for family/culture/nature planning prompts
- discovery POIs are merged into the session catalog across turns

### 4. Plan synthesis and repair

Planning/refinement turns use:

- `synthesizePlan()`
- `enrichPlanLogistics()`
- `criticizeAndRefinePlan()`
- `transitAdvisor()`

Current synthesis behavior includes deterministic hospitality repair:

- overnight days get real stay POIs when available
- generic meal slots are upgraded into actual restaurant POIs when the request calls for food/cuisine support

### 5. Narration and blocks

`answerConversationally()` plus `buildResponseBlocks()` create the final assistant response.

Current narrative model routing:

- primary: `gemini-flash-latest`
- fallback: `gemini-2.5-flash`

### 6. Session commit

At the end of the turn the runtime:

- persists the assistant message
- persists follow-up context
- updates session memory
- sets `planningState.status = "ready"`
- emits the final `session.snapshot`

## Current planning state model

Canonical statuses:

- `ready`
- `running`
- `unavailable`

Canonical stages:

- `understanding`
- `researching`
- `checking_dates`
- `researching_events`
- `researching_stays`
- `building_plan`
- `refining`
- `ready`
- `unavailable`

Canonical sources:

- `provider`
- `places`
- `directions`
- `weather`
- `events`
- `holidays`
- `stays`

## Context maintenance and invalidation

This is now an explicit runtime concern.

Current rules:

- explicit new-trip requests replace stale active trip context
- multi-city trips preserve their active destination set
- explicit new discovery asks override stale follow-up context
- accepted decisions are replaced by key, not appended forever
- the latest destination/origin/duration/budget/travelers entries are the authoritative remembered values

This behavior is implemented in:

- `updateSessionMemory()` in [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)
- `resetActiveTripContext()` in [Backend/src/services/agent-tool-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/agent-tool-service.ts)

## Plan-only projection rules

Current product rule:

- itinerary tab is driven only by the canonical plan
- itinerary route polylines and numbered route markers are driven only by itinerary-linked POIs
- broader discovered POIs may appear on the map as non-route markers
- non-itinerary discovery turns should not mutate the canonical plan

## Deterministic plan mutations

Direct itinerary edits should go through `PlanMutationService`.

Supported mutations:

- `add_poi`
- `remove_poi`
- `move_activity`
- `regenerate_day`
- `rebalance_trip`
- `update_overview`

The new `AgentToolService` wraps this into a cleaner tool surface for autonomous agents.

## Failure behavior

If provider or tool execution fails:

- the current accepted plan is preserved
- planning state moves to `unavailable`
- a fallback assistant message is committed
- `turn.failed` is emitted

The runtime should not fabricate replacement planning content after a failed turn.
