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
- SSE stream subscription
- saved POI updates
- user settings
- BYOK credential storage

### Turn execution

[Backend/src/runtime/turn-runner.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/turn-runner.ts) is the canonical turn orchestrator.

Per turn it:

1. persists the user message
2. resolves provider settings and credentials
3. marks planning state as running
4. emits trace events
5. runs the deterministic planning pipeline
6. persists plan and catalog updates when applicable
7. persists the assistant message
8. updates memory and final planning state
9. emits final snapshot and turn events

### Business logic

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

If Supabase is not configured, the backend falls back to in-memory storage for local development.

### Plan mutations

[Backend/src/services/plan-mutation-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/plan-mutation-service.ts) is the canonical direct-edit path.

Supported mutations:

- `add_poi`
- `remove_poi`
- `move_activity`
- `regenerate_day`
- `rebalance_trip`
- `update_overview`

`update_overview` is the path used by the header editors. It:

- updates title, origin, dates, travelers, and budget
- preserves the current itinerary when only metadata/date shifts change
- regenerates the plan when destinations or total day count materially change

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

The frontend helper is still named `connectWs()`, but the implementation in [roameo-frontend/lib/ws.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/ws.ts) uses `fetch()` against the SSE endpoint, not a WebSocket connection.

Current stream events:

- `session.snapshot`
- `turn.started`
- `message.delta`
- `message.committed`
- `trace.updated`
- `plan.updated`
- `turn.completed`
- `turn.failed`

## Shared contracts

The contracts package is the only authoritative schema source for:

- providers and run modes
- preferences and provider settings
- planning state and date context
- POIs and POI catalogs
- plan snapshots
- conversation messages
- structured response blocks
- traces
- stream events
- API payloads

Avoid mirrored frontend/backend copies for those business concepts.

## Map and route rendering

The canonical map component is [roameo-frontend/components/map-view.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/map-view.tsx).

Important implementation note: the current map still renders routes through the Google Maps JavaScript Directions APIs (`DirectionsService` and `DirectionsRenderer`). Do not document a Routes API migration as completed until the code actually changes.

## Provider settings and BYOK

Provider resolution is handled in [Backend/src/services/provider-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/provider-service.ts).

Current guarantees:

- Gemini and OpenAI are first-class providers
- each provider has `fast`, `balanced`, and `deep` model picks
- session settings and user defaults are canonical backend state
- users can store encrypted provider keys
- run-time provider resolution chooses platform or BYOK credentials per request

## Legacy code policy

The repository still contains older code under paths such as:

- `Backend/src/agents/*`
- `Backend/src/graph/*`

These are not the canonical product path. New behavior should land in the session runtime, direct integrations, shared contracts, and snapshot-driven frontend.
