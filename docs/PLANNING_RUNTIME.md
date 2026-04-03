# Planning Runtime

The canonical planning runtime lives in:

- [Backend/src/runtime/turn-runner.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/turn-runner.ts)
- [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)
- [Backend/src/services/travel-tools.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/travel-tools.ts)
- [Backend/src/services/plan-mutation-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/plan-mutation-service.ts)

## Canonical intents

The shared `travelIntentSchema` defines:

- `plan_trip`
- `refine_trip`
- `search_places`
- `question`
- `settings`
- `meta`

The runtime derives more specific conversational modes on top of those intents, including greeting, discovery, planning, refinement, and missing-info branches.

## Turn phases

The standard turn pipeline is:

1. intent resolution
2. research gate
3. destination discovery
4. weather, event, and holiday enrichment
5. structured plan synthesis for planning/refinement turns
6. logistics enrichment
7. feasibility critic
8. transit advisor
9. narration and response block generation
10. session commit

## Planning state

Canonical planning statuses:

- `ready`
- `running`
- `unavailable`

Canonical planning stages:

- `understanding`
- `researching`
- `checking_dates`
- `researching_events`
- `researching_stays`
- `building_plan`
- `refining`
- `ready`
- `unavailable`

Canonical planning sources:

- `provider`
- `places`
- `directions`
- `weather`
- `events`
- `holidays`
- `stays`

Successful turns must end with `planningState.status = "ready"` and `stage = "ready"`.

## Discovery runtime

Discovery builds a canonical `PoiCatalog` from live integrations.

Current source types:

- `google_places`
- `google_maps`
- `web_research`
- `manual`

The active discovery behavior enforces strict type filtering in Google Places:

- stays -> `lodging`
- restaurants -> `restaurant`
- attractions -> `tourist_attraction`

This is a required guardrail against category bleed.

## Date-aware runtime

Canonical date context is stored in `session.memory.dateContext`.

Tracked fields include:

- requested dates
- inferred dates
- flexibility: `exact`, `approximate`, `open_ended`
- derivation source
- advisory items

Behavioral rules:

- relative dates are normalized before planning
- open-ended dates can still produce seasonal guidance
- long-range trips outside forecast windows fall back to advisory-only weather notes
- the runtime should not silently rewrite accepted dates

## External enrichments

Current direct integrations:

- Open-Meteo for weather forecasts
- Tavily for destination facts and event research
- Nager.Date for public holiday windows

These integrations produce advisory context, not guaranteed booking or inventory truth.

## Structured response model

The runtime emits shared response blocks instead of relying on markdown-only rendering.

Block types actively used by the current runtime include:

- `trip_intro`
- `lead`
- `capabilities_overview`
- `featured_poi`
- `poi_story_list`
- `place_card_row`
- `itinerary_template`
- `stay_recommendation_list`
- `assistant_prompt_chips`
- `planning_status`
- `worker_progress`
- `date_advisory`
- `event_window_summary`
- `clarifying_questions`

The frontend should treat `responseBlocks` as the canonical assistant presentation surface.

## Planning and refinement

`synthesizePlan()` is used for:

- new trip creation
- itinerary refinement
- some regenerated overview flows triggered by plan mutations

`enrichPlanLogistics()` then computes:

- `travelTimeMinutesFromPrevious`
- day budgets
- trip budget totals

`criticizeAndRefinePlan()` applies deterministic post-processing:

- caps overloaded days
- flags long transfers
- trims overloaded final days
- fills missing accommodation on multi-day plans

`transitAdvisor()` emits trace guidance for multi-destination trips but does not currently persist transit segments into the plan schema.

## Session memory

Canonical memory fields include:

- rolling summary
- destinations discussed
- accepted decisions
- last plan version
- date context
- planning state
- user preferences

The runtime updates memory after each successful turn so follow-up turns can reuse accepted trip context.

## Failure behavior

If provider or tool execution fails:

- the current accepted plan is preserved
- planning state moves to `unavailable`
- a fallback assistant message is committed
- `turn.failed` is emitted

The runtime should not fabricate replacement planning content after a failed turn.

## Deterministic plan mutations

Not every change requires a broad conversational turn.

`PlanMutationService` handles direct edits for:

- header changes
- add/remove POI
- move activity
- regenerate day
- rebalance trip

This preserves the snapshot as the source of truth and avoids parallel business logic paths.
