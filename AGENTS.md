# Roameo Modernization

## Current Canonical Direction
- The primary product is a text-first travel planning workspace with chat on the left and itinerary or map on the right.
- The canonical runtime is session-snapshot based. Chat, itinerary, map, saved POIs, and session metadata must derive from the same persisted session state.
- Gemini and OpenAI are both first-class providers. Provider defaults and BYOK settings live in canonical user settings.

## Delivery Rules
- Prefer the new session API, SSE streaming path, and shared contracts package.
- Do not introduce new business logic into legacy WebSocket, old LangGraph routing, or mirrored frontend/backend type copies.
- Delete or retire duplicate paths once the new primary path is in place.
