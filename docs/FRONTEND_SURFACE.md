# Frontend Surface

The frontend is intentionally a thin consumer of the canonical session snapshot and shared contracts.

## Main workspace

The canonical workspace page is [roameo-frontend/app/chat/chat-page-client.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/app/chat/chat-page-client.tsx).

It is responsible for:

- loading the current session snapshot
- opening the live session stream
- hydrating the Zustand store
- deriving chat, trip, itinerary, map, and search models
- rendering the left and right panel workspace

## Data flow

Key files:

- [roameo-frontend/lib/api.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/api.ts)
- [roameo-frontend/lib/ws.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/ws.ts)
- [roameo-frontend/lib/session-store.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-store.ts)
- [roameo-frontend/lib/session-view.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-view.ts)
- [roameo-frontend/lib/types.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/types.ts)

Responsibilities by layer:

- `api.ts`: authenticated REST calls
- `ws.ts`: SSE stream transport and chunk parsing
- `session-store.ts`: live snapshot and stream-event merging
- `session-view.ts`: derived view models and overview mutation construction
- `types.ts`: frontend aliases over shared contracts plus display models

## Chat surface

Key files:

- [roameo-frontend/components/chat-interface.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/chat-interface.tsx)
- [roameo-frontend/components/structured-response-blocks.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/structured-response-blocks.tsx)
- [roameo-frontend/components/agentic-status.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/agentic-status.tsx)
- [roameo-frontend/components/inline-planning-status.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/inline-planning-status.tsx)

Current rendering rules:

- committed assistant messages render from shared `responseBlocks`
- streamed text arrives through `message.delta` events and is merged into a draft message
- planning traces render separately from the final assistant message
- worker progress and planning status blocks can be hidden once a final itinerary block exists

## Header editing

Key files:

- [roameo-frontend/components/top-navigation.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/top-navigation.tsx)
- [roameo-frontend/components/trip-date-dialog.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/trip-date-dialog.tsx)
- [roameo-frontend/components/trip-destination-dialog.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/trip-destination-dialog.tsx)
- [roameo-frontend/components/trip-travelers-dialog.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/trip-travelers-dialog.tsx)
- [roameo-frontend/components/trip-budget-dialog.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/trip-budget-dialog.tsx)

Current behaviors:

- optimistic title and trip-metadata updates
- destination editing with support for multi-stop destination arrays
- date editing with flexibility support
- traveler count editing
- budget editing via normalized budget options
- browser geolocation origin autofill through `/api/maps/reverse-geocode`

The frontend sends these changes through `update_overview` plan mutations built in `session-view.ts`.

## Search, saved, and POI surfaces

Key files:

- [roameo-frontend/components/search-interface.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/search-interface.tsx)
- [roameo-frontend/components/search-card.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/search-card.tsx)
- [roameo-frontend/components/poi-card.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/poi-card.tsx)
- [roameo-frontend/components/poi-detail-modal.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/poi-detail-modal.tsx)
- [roameo-frontend/components/poi-type-icon.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/poi-type-icon.tsx)
- [roameo-frontend/lib/poi-image-url.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/poi-image-url.ts)

Current direction:

- image-first POI cards
- canonical save state from `savedPoiIds`
- no-image fallbacks with POI-type-aware iconography
- derived search categories from the session catalog, not ad hoc panel state

## Map surface

The canonical map component is [roameo-frontend/components/map-view.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/components/map-view.tsx).

Current rules:

- markers come from snapshot-derived canonical POIs
- saved and itinerary selections come from canonical ids
- routes are rendered with Google Maps `DirectionsService` and `DirectionsRenderer`
- map-side filtering is local UI state only and must not become a second source of truth

## Styling and tokens

Global design tokens and base styling live in [roameo-frontend/app/globals.css](/Users/yk0007/MyRepos/Roameo/roameo-frontend/app/globals.css).

Shared visual expectations:

- keep typography, spacing, and panel rhythm aligned with existing tokens
- preserve the warm, image-first travel product direction
- avoid reintroducing sessionStorage-based itinerary recovery or duplicate backend contracts
