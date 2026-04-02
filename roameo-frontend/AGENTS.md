# Frontend Scope

## Canonical Frontend
- Preserve the current Roameo structure and warm visual direction, but drive it from the canonical session snapshot.
- Chat, map, itinerary, saved places, and session title must stay synchronized through one state path.
- Prefer route shells, shared view-model helpers, React Query for server fetches, and Zustand for active session state.

## Guardrails
- Do not reintroduce sessionStorage itinerary recovery, duplicated auth checks, or mirrored backend contracts.
- Keep typography, spacing, color, and loading states consistent with the shared design tokens in `app/globals.css`.
