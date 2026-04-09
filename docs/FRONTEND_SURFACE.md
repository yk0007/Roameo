# Frontend Surface

The frontend is a thin consumer of the canonical session snapshot and shared contracts.

## Main workspace

The canonical workspace page is [roameo-frontend/app/chat/chat-page-client.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/app/chat/chat-page-client.tsx).

It is responsible for:

- loading the current session snapshot
- opening the live session stream
- hydrating the Zustand store
- deriving trip, itinerary, map, and search models from the snapshot
- rendering the split workspace

## Data flow

Key files:

- [roameo-frontend/lib/api.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/api.ts)
- [roameo-frontend/lib/ws.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/ws.ts)
- [roameo-frontend/lib/session-store.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-store.ts)
- [roameo-frontend/lib/session-view.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-view.ts)
- [roameo-frontend/lib/types.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/types.ts)

Current responsibilities:

- `api.ts`: authenticated REST calls
- `ws.ts`: SSE stream transport
- `session-store.ts`: live snapshot/event merge
- `session-view.ts`: derived trip, search, itinerary, and map projections
- `types.ts`: frontend aliases over shared contracts

## Current projection rules

### Chat

- structured assistant rendering comes from shared `responseBlocks`
- chat POI cards now receive the full canonical catalog, not just the current map slice

### Search and saved

- search and saved views are derived from the canonical session catalog
- category-specific discovery can hydrate more POIs into the catalog through `/api/sessions/:sessionId/discovery`

### Itinerary tab

- itinerary tab content is driven only by the canonical plan
- it should not change on pure discovery turns unless the plan itself changed

### Map

- numbered route markers and polylines come only from itinerary-linked POIs
- the rest of the canonical POI catalog can still appear as non-route markers
- map hover cards render from the current canonical marker set

## Right panel update behavior

Files:

- [roameo-frontend/components/right-panel.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/right-panel.tsx)
- [roameo-frontend/components/map-view.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/map-view.tsx)
- [roameo-frontend/components/itinerary-panel.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/itinerary-panel.tsx)

Current rule:

- `MapView` and `ItineraryPanel` are keyed by canonical plan version state
- when the real plan changes, the right panel remounts its derived internal state
- when only discovery changes, the itinerary should stay plan-only

## Header editing

Files:

- [roameo-frontend/components/top-navigation.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/top-navigation.tsx)
- [roameo-frontend/components/trip-date-dialog.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/trip-date-dialog.tsx)
- [roameo-frontend/components/trip-destination-dialog.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/trip-destination-dialog.tsx)
- [roameo-frontend/components/trip-travelers-dialog.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/trip-travelers-dialog.tsx)
- [roameo-frontend/components/trip-budget-dialog.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/trip-budget-dialog.tsx)

Current behavior:

- optimistic trip metadata updates
- canonical `update_overview` mutations
- date flexibility support
- destination editing with multi-stop support
- origin autofill from browser location + reverse geocode

## Visual rule of thumb

Keep frontend business logic thin.

The backend/runtime decides:

- what the active trip is
- what the current domain is
- whether the turn is itinerary-changing or discovery-only

The frontend should:

- render the snapshot
- apply stream events
- keep local UI state only for presentation and interaction
