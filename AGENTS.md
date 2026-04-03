# Roameo Modernization

## Current Canonical Direction
- The primary product is a text-first travel planning workspace with chat on the left and itinerary or map on the right.
- The canonical runtime is session-snapshot based. Chat, itinerary, map, saved POIs, and session metadata must derive from the same persisted session state.
- Gemini and OpenAI are both first-class providers. Provider defaults and BYOK settings live in canonical user settings.

## Agentic Architecture
- Each turn runs a 10-step deterministic pipeline of specialised sub-agents.
- The LLM is called exactly twice per turn (intent resolution + narrative).  Plans use a third structured call.
- New agents: **Feasibility Critic** (`criticizeAndRefinePlan`) and **Transit Advisor** (`transitAdvisor`) run after logistics enrichment.
- Deep web research via Tavily (`deepWebResearch`) runs for `plan_trip` turns when `TAVILY_API_KEY` is set.
- Full spec: [docs/AUTONOMOUS_AGENTS.md](docs/AUTONOMOUS_AGENTS.md) | Quick ref: [docs/AGENTS.md](docs/AGENTS.md)

## Delivery Rules
- Prefer the new session API, SSE streaming path, and shared contracts package.
- Do not introduce new business logic into legacy WebSocket, old LangGraph routing, or mirrored frontend/backend type copies.
- Delete or retire duplicate paths once the new primary path is in place.
- All new sub-agents must live in `Backend/src/runtime/subagents.ts` or `Backend/src/services/travel-tools.ts` with JSDoc.
- Set `planningState.status = "ready"` at the end of every successful turn — the frontend animation gate depends on it.
- Apply Google Places `type` filter on every `searchPlacesByQueries` call to prevent category bleed.
