# Autonomous Agents Architecture

Roameo’s backend turn execution is a deterministic, typed pipeline. Most of the “agents” are plain TypeScript functions coordinated by `TurnRunner`; the LLM is used only where free-form interpretation or synthesis is required.

## Core principle

- deterministic orchestration in code
- shared contracts at the boundaries
- live tool-backed grounding for POIs, weather, events, and holidays
- one session snapshot updated at the end of the turn

## Per-turn pipeline

1. `resolveTurnIntent()`
Intent resolver. Uses a structured model call to classify the message, normalize destinations, and infer date context.

2. `shouldResearchResolution()`
Pure predicate that decides whether POI research should run.

3. `researchDestinations()`
Runs Google Places research and optional Tavily deep research into a canonical `PoiCatalog`.

4. `getWeatherSummary()`, `getEventSummary()`, `getHolidaySummary()`
Date-context enrichment using Open-Meteo, Tavily, and Nager.Date.

5. `synthesizePlan()`
Structured planner call used only for `plan_trip` and `refine_trip` turns.

6. `enrichPlanLogistics()`
Backfills travel times with Google Directions and computes day and trip budgets.

7. `criticizeAndRefinePlan()`
Pure in-memory feasibility pass that trims obviously overloaded plans and fills accommodation gaps.

8. `transitAdvisor()`
Pure heuristic pass that emits inter-city transit suggestions for multi-destination trips.

9. `answerConversationally()` plus `buildResponseBlocks()`
Creates the final assistant text and canonical response blocks.

10. `updateSessionMemory()` plus final repository update
Commits the accepted result back into the session snapshot and marks planning as ready.

## LLM call pattern

Normal turns use exactly two model calls:

- intent resolution
- narration

Planning turns use a third model call:

- structured itinerary synthesis

That means `plan_trip` and `refine_trip` turns are the only ones that perform three provider calls.

## Agent responsibilities

### Intent Resolver

File: `Backend/src/runtime/subagents.ts`

`resolveTurnIntent()` returns a `TurnResolution` with:

- `intent`
- `destination` and `destinations`
- `origin`
- `totalDays`
- `travelerCount`
- `budgetNote`
- `styles`
- `questionFocus`
- `stayMode`
- `dateContext`

The result is finalized against session context before the rest of the pipeline runs.

### Research Gate

File: `Backend/src/runtime/subagents.ts`

`shouldResearchResolution()` prevents unnecessary discovery work. It typically runs for:

- `plan_trip`
- `refine_trip`
- `search_places`
- question turns that are really place-discovery requests

### Discovery Agent

Files:

- `Backend/src/runtime/subagents.ts`
- `Backend/src/services/travel-tools.ts`

`researchDestinations()` calls `searchPlacesForDestination()`, which fans out into typed Google Places searches for:

- stays
- restaurants
- attractions

Every `searchPlacesByQueries()` request applies a strict Google Places `type` filter:

- stay -> `lodging`
- restaurant -> `restaurant`
- attraction -> `tourist_attraction`

This category isolation is part of the canonical behavior and prevents cross-category bleed.

For planning turns, discovery can also run `deepWebResearch()` when `TAVILY_API_KEY` is configured.

### Date Context Agent

File: `Backend/src/services/travel-tools.ts`

The date context agent is a grouped set of direct integrations:

- `getWeatherSummary()` -> Open-Meteo
- `getEventSummary()` -> Tavily
- `getHolidaySummary()` -> Nager.Date

These functions populate the `PlanningContext` consumed by narration and block rendering.

### Itinerary Planner

File: `Backend/src/runtime/subagents.ts`

`synthesizePlan()` is the structured planning call. It receives the current session, resolved intent, and researched catalog, then produces a validated `PlanSnapshot`.

Important rules enforced by prompt plus normalization:

- use provided POI ids only
- create feasible day counts
- preserve destination structure
- include accommodation when available
- do not fabricate unknown POIs

### Logistics Enricher

File: `Backend/src/runtime/subagents.ts`

`enrichPlanLogistics()`:

- computes travel times between adjacent POIs with Google Directions
- stores `travelTimeMinutesFromPrevious`
- computes day budgets
- computes trip-total budget

### Feasibility Critic

File: `Backend/src/runtime/subagents.ts`

`criticizeAndRefinePlan()` is a pure post-processing pass. It currently:

- trims days above 5 activities
- flags transfer legs above 120 minutes
- lightens the last day if overloaded
- fills missing accommodation on multi-day trips when a stay exists

Each critique is emitted as a trace event by the turn runner.

### Transit Advisor

File: `Backend/src/runtime/subagents.ts`

`transitAdvisor()` runs for multi-destination plans and emits heuristic inter-city guidance. The output is trace-only today; it does not mutate the saved plan.

Current mode selection is heuristic:

- major hub routes bias toward flight
- hill-station routes bias toward train
- otherwise default toward drive

### Narrator

File: `Backend/src/runtime/subagents.ts`

`answerConversationally()` creates the assistant prose. `buildResponseBlocks()` deterministically turns the narrative, planning context, and plan/catalog state into shared structured blocks.

Common emitted blocks include:

- `trip_intro`
- `lead`
- `capabilities_overview`
- `featured_poi`
- `poi_story_list`
- `place_card_row`
- `itinerary_template`
- `stay_recommendation_list`
- `assistant_prompt_chips`
- `worker_progress`
- `date_advisory`
- `event_window_summary`

### Session Commit

Files:

- `Backend/src/runtime/subagents.ts`
- `Backend/src/runtime/turn-runner.ts`

At commit time, the runtime:

- persists the assistant message
- updates session memory
- sets `planningState.status = "ready"` and `stage = "ready"`
- persists the final session snapshot
- emits `turn.completed`

On failure it preserves the last accepted trip state, marks planning unavailable, commits a fallback assistant message, and emits `turn.failed`.

## Trace model

The turn runner emits traces through `trace.updated` SSE events. Current agent labels include:

- `lead`
- `intent-slot-resolver`
- `discovery-search-agent`
- `stay-search-agent`
- `date-context-agent`
- `events-culture-agent`
- `itinerary-planner`
- `feasibility-validator`
- `feasibility-critic`
- `transit-advisor`
- `narrator`

The frontend consumes these traces as live planning feedback.

## Deterministic mutation path

Not all trip updates require a full conversational turn. `PlanMutationService` handles direct mutations such as:

- add/remove POI
- move activity
- regenerate day
- rebalance trip
- update overview

`update_overview` only regenerates the itinerary when the destination structure or trip length changes. Pure metadata/date shifts reuse the current plan and rerun logistics.
