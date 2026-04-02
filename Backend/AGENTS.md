# Backend Scope

## Canonical Backend
- `src/index.ts`, `src/api/`, `src/runtime/`, `src/services/`, `src/core/`, and `packages/contracts` define the live backend path.
- Keep one session model: `travel_sessions`, `session_messages`, `session_plan_snapshots`, `session_poi_catalogs`, `session_saved_pois`, `session_agent_traces`, `user_provider_settings`, and `user_provider_credentials`.
- Stream live updates over authenticated SSE. Do not add new product behavior to the retired WebSocket path.

## Implementation Notes
- Session state must be validated at the boundary and persisted before UI projections depend on it.
- Keep provider integration direct and current for Gemini and OpenAI.
