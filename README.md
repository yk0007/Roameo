# Roameo

Roameo is a text-first AI travel planning workspace with one canonical session snapshot driving:

- chat
- itinerary
- map routes and markers
- saved POIs
- agent traces
- planning state
- session memory and preferences

The primary product surface is the split workspace:

- chat on the left
- map or itinerary on the right
- header controls for destination, dates, travelers, budget, and trip metadata

## Workspace layout

- `Backend/`: Express + TypeScript backend
- `roameo-frontend/`: Next.js 16 + React 19 frontend
- `packages/contracts/`: shared Zod schemas and TypeScript types
- `database/`: SQL and persistence assets

## Canonical architecture

The source of truth is `SessionSnapshot` in [packages/contracts/src/index.ts](/Users/yk0007/MyRepos/Roameo/packages/contracts/src/index.ts).

It contains:

- session metadata and provider settings
- session memory and planning state
- the active plan snapshot
- the canonical POI catalog
- conversation messages
- saved POI ids
- agent trace events

Canonical codepaths:

- [Backend/src/api/router.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/api/router.ts)
- [Backend/src/runtime/turn-runner.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/turn-runner.ts)
- [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)
- [Backend/src/services/plan-mutation-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/plan-mutation-service.ts)
- [Backend/src/services/agent-tool-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/agent-tool-service.ts)
- [Backend/src/services/travel-tools.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/travel-tools.ts)
- [Backend/src/services/session-repository.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/session-repository.ts)
- [roameo-frontend/app/chat/chat-page-client.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/app/chat/chat-page-client.tsx)
- [roameo-frontend/lib/session-store.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-store.ts)
- [roameo-frontend/lib/session-view.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-view.ts)

## Current runtime direction

Roameo is moving toward a router-first agentic runtime:

1. lightweight conversational fast path for trivial turns
2. semantic router for normal travel understanding
3. deterministic tool execution and validation
4. narrative/planning generation from canonical state

Important current runtime behavior:

- explicit new asks like `show me some restaurants` or `show me some stays` override stale follow-up context
- explicit new trip requests replace stale active-trip destination context
- multi-city trips preserve the active destination set instead of collapsing to one city
- itinerary and map route state update only when the canonical itinerary changes
- the map can still display discovered non-itinerary POIs without turning them into itinerary routes

## Agent and tool model

Roameo uses deterministic orchestration plus LLM understanding. The goal is not “freeform autonomous code everywhere”; it is:

- semantic understanding by model
- explicit tool selection
- deterministic canonical state writes
- fail-fast validation

Current first-class internal agent tools live in [Backend/src/services/agent-tool-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/agent-tool-service.ts):

- `getSessionSnapshot`
- `updateTripHeader`
- `editItinerary`
- `updateSessionMemory`
- `resetActiveTripContext`
- `saveFollowUpContext`

Those tools let the runtime and future autonomous agents read and modify the active session without bypassing the canonical repository and plan mutation paths.

## Providers and integrations

AI providers:

- Gemini
- OpenAI

Direct product integrations:

- Google Places
- Google Geocoding
- Google Directions / Maps
- Open-Meteo
- Tavily
- Nager.Date
- Supabase

The current product direction is:

- direct Google Places/Maps remain the canonical POI and route truth
- Tavily remains editorial enrichment, not the primary POI source
- Gemini Maps grounding is useful as an optional semantic expansion/reranking layer, not as a replacement for canonical Places retrieval

### Verified Gemini model IDs

The current backend config is aligned to live-tested Gemini model IDs:

- working text models:
  - `gemini-flash-latest`
  - `gemini-2.5-flash`
  - `gemini-2.5-flash-lite`
  - `gemma-3-27b-it`
  - `gemma-4-31b-it`
- working embedding model:
  - `gemini-embedding-001`

Current defaults in the backend:

- router / understanding: `gemma-4-31b-it` with `gemini-2.5-flash` fallback
- narrative / synthesis: `gemini-flash-latest` with `gemini-2.5-flash` fallback

## Development

Requirements:

- Node.js `>=22`
- npm workspaces
- `GOOGLE_MAPS_API_KEY`
- at least one of `GEMINI_API_KEY` or `OPENAI_API_KEY`

Install:

```bash
npm install
```

Run backend:

```bash
npm run dev
```

Run frontend:

```bash
npm run dev:frontend
```

Build all workspaces:

```bash
npm run build
```

Typecheck:

```bash
npm run typecheck
```

Test:

```bash
npm run test
```

## Environment

Core backend environment variables:

- `PORT`
- `APP_BASE_URL`
- `WS_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_MODEL_FAST`
- `GEMINI_MODEL_BALANCED`
- `GEMINI_MODEL_DEEP`
- `GEMINI_MODEL_ROUTER`
- `GEMINI_MODEL_ROUTER_FALLBACK`
- `GEMINI_MODEL_NARRATIVE`
- `GEMINI_MODEL_NARRATIVE_FALLBACK`
- `GEMINI_MODEL_GROUNDING`
- `GEMINI_MODEL_GROUNDING_FALLBACK`
- `GEMINI_MODEL_EMBEDDING`
- `OPENAI_API_KEY`
- `OPENAI_MODEL_FAST`
- `OPENAI_MODEL_BALANCED`
- `OPENAI_MODEL_DEEP`
- `GOOGLE_MAPS_API_KEY`
- `TAVILY_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `ROAMEO_ENCRYPTION_SECRET`

Frontend runtime:

- `NEXT_PUBLIC_BACKEND_URL`

## Documentation map

Start with [docs/README.md](/Users/yk0007/MyRepos/Roameo/docs/README.md).

Core docs:

- [docs/AUTONOMOUS_AGENTS.md](/Users/yk0007/MyRepos/Roameo/docs/AUTONOMOUS_AGENTS.md)
- [docs/CANONICAL_ARCHITECTURE.md](/Users/yk0007/MyRepos/Roameo/docs/CANONICAL_ARCHITECTURE.md)
- [docs/API_REFERENCE.md](/Users/yk0007/MyRepos/Roameo/docs/API_REFERENCE.md)
- [docs/PLANNING_RUNTIME.md](/Users/yk0007/MyRepos/Roameo/docs/PLANNING_RUNTIME.md)
- [docs/FRONTEND_SURFACE.md](/Users/yk0007/MyRepos/Roameo/docs/FRONTEND_SURFACE.md)
- [docs/OPERATIONS_AND_TESTING.md](/Users/yk0007/MyRepos/Roameo/docs/OPERATIONS_AND_TESTING.md)

Supporting note:

- [PERFORMANCE_OPTIMIZATIONS.md](/Users/yk0007/MyRepos/Roameo/PERFORMANCE_OPTIMIZATIONS.md)
