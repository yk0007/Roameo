# Canonical Architecture

## Overview

Roameo’s primary architecture is session-snapshot based.

One `SessionSnapshot` drives:

- chat history
- structured assistant responses
- itinerary
- map POIs and route inputs
- saved POIs
- trace history
- planning state
- session memory and preferences

The schema source of truth lives in [packages/contracts/src/index.ts](/Users/yk0007/MyRepos/Roameo/packages/contracts/src/index.ts).

## Canonical backend path

### API surface

The public API lives in [Backend/src/api/router.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/api/router.ts).

The active routes cover:

- health
- Google Maps helper endpoints
- session CRUD
- message submission
- plan mutations
- discovery loading
- SSE stream subscription
- saved POI updates
- user settings
- BYOK credential storage

### Turn execution

[Backend/src/runtime/turn-runner.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/turn-runner.ts) is the canonical turn orchestrator.

Per turn it:

1. persists the user message
2. marks planning state as running
3. runs fast-path conversation when appropriate
4. resolves turn meaning with the semantic router
5. executes discovery / enrichment / planning tools as needed
6. persists plan and catalog updates when applicable
7. persists the assistant message
8. updates memory and follow-up context
9. emits final snapshot and turn events

### Runtime behavior layer

[Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts) is the main behavior layer for:

- intent resolution
- discovery focus handling
- destination research orchestration
- planning synthesis
- logistics enrichment
- feasibility criticism
- transit advice
- narrative generation
- structured response block assembly
- memory updates
- follow-up context derivation

### Internal tool layer

[Backend/src/services/agent-tool-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/agent-tool-service.ts) is the first-class internal tool surface for autonomous agents.

Current canonical internal tools:

- read session snapshot
- update trip header / overview
- edit itinerary
- update session memory
- reset active trip context
- save or clear follow-up context

### Direct integrations

[Backend/src/services/travel-tools.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/travel-tools.ts) contains the direct external integrations for:

- Google Places
- Google Geocoding
- Google Directions
- Open-Meteo
- Tavily
- Nager.Date

### Persistence

[Backend/src/services/session-repository.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/session-repository.ts) is the persistence gateway.

The canonical persisted model includes:

- `travel_sessions`
- `session_messages`
- `session_plan_snapshots`
- `session_poi_catalogs`
- `session_saved_pois`
- `session_agent_traces`
- `user_provider_settings`
- `user_provider_credentials`

Important current behavior:

- POI catalogs are merged across discovery turns instead of replaced
- plan saves also preserve the broader session catalog
- if Supabase is not configured, the backend falls back to in-memory storage for local development

### Plan mutations

[Backend/src/services/plan-mutation-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/plan-mutation-service.ts) is the canonical direct-edit path.

Supported mutations:

- `add_poi`
- `remove_poi`
- `move_activity`
- `regenerate_day`
- `rebalance_trip`
- `update_overview`

`update_overview` is the path used by the header editors.

## Canonical frontend path

The main workspace lives in [roameo-frontend/app/chat/chat-page-client.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/app/chat/chat-page-client.tsx).

The page:

- loads the current session
- opens the session event stream
- hydrates Zustand state
- derives trip, map, itinerary, and search models from the snapshot
- renders the top navigation, left panel, and right panel

Supporting files:

- [roameo-frontend/lib/api.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/api.ts)
- [roameo-frontend/lib/ws.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/ws.ts)
- [roameo-frontend/lib/session-store.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-store.ts)
- [roameo-frontend/lib/session-view.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-view.ts)
- [roameo-frontend/lib/types.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/types.ts)

## Live update model

The product uses SSE over `GET /api/sessions/:sessionId/stream`.

Current stream events:

- `session.snapshot`
- `turn.started`
- `message.delta`
- `message.committed`
- `trace.updated`
- `plan.updated`
- `turn.completed`
- `turn.failed`

## Current context invariants

The canonical runtime now enforces:

- explicit new-trip requests replace stale active trip context
- multi-city trips preserve the active destination set
- explicit new discovery asks override stale follow-up branches
- itinerary tab content is driven only by the canonical plan
- itinerary map routes are driven only by itinerary-linked POIs
- map markers may still include other discovered canonical POIs without turning them into itinerary routes

## Map and itinerary projection rules

Current frontend projection rules:

- itinerary tab shows itinerary-derived data only
- map routes and numbered markers come from itinerary-linked POIs only
- the broader canonical POI catalog can still appear on the map as non-itinerary markers
- right-panel map and itinerary views remount on canonical plan-version changes so stale internal state is discarded

## Provider settings and model routing

Provider resolution lives in [Backend/src/services/provider-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/provider-service.ts).

Current guarantees:

- Gemini and OpenAI are first-class providers
- each provider keeps `fast`, `balanced`, and `deep` defaults
- Gemini task-specific model lists are available for router and narrative work
- users can store encrypted provider keys
- run-time provider resolution chooses platform or BYOK credentials per request

## Legacy code policy

The repository still contains older code under paths such as:

- `Backend/src/agents/*`
- `Backend/src/graph/*`

These are not the canonical extension path. New behavior should land in the session runtime, internal tool layer, direct integrations, shared contracts, and snapshot-driven frontend.
