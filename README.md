# Roameo

Roameo is a text-first AI travel planning workspace with a canonical split view:

- chat on the left
- itinerary or map on the right
- one persisted session snapshot driving chat, itinerary, map, saved POIs, traces, and planning state

The authoritative implementation is the session API, SSE stream, shared contracts package, and snapshot-based runtime. Legacy LangGraph and WebSocket-era code remains in the tree, but it is not the primary extension path.

## Workspace layout

- `Backend/`: Express + TypeScript backend
- `roameo-frontend/`: Next.js 16 + React 19 frontend
- `packages/contracts/`: shared Zod schemas and TypeScript types
- `database/`: SQL and persistence assets

## Canonical architecture

The single source of truth is `SessionSnapshot` in [packages/contracts/src/index.ts](/Users/yk0007/MyRepos/Roameo/packages/contracts/src/index.ts). It contains:

- session metadata and provider settings
- session memory and planning state
- the active trip plan
- the canonical POI catalog
- conversation messages
- saved POI ids
- agent trace events

The canonical runtime path is:

1. Frontend creates or loads a session.
2. Frontend opens `GET /api/sessions/:sessionId/stream`.
3. User sends a message to `POST /api/sessions/:sessionId/messages`.
4. `TurnRunner` persists the user message, resolves provider settings, runs the deterministic planning pipeline, and emits trace, message, plan, and snapshot events.
5. Backend persists messages, plan, POI catalog, traces, and updated memory.
6. Frontend rebuilds chat, map, itinerary, and header state from the updated snapshot.

Primary codepaths:

- [Backend/src/api/router.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/api/router.ts)
- [Backend/src/runtime/turn-runner.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/turn-runner.ts)
- [Backend/src/runtime/subagents.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/runtime/subagents.ts)
- [Backend/src/services/plan-mutation-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/plan-mutation-service.ts)
- [Backend/src/services/travel-tools.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/travel-tools.ts)
- [Backend/src/services/session-repository.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/session-repository.ts)
- [roameo-frontend/app/chat/chat-page-client.tsx](/Users/yk0007/MyRepos/Roameo/roameo-frontend/app/chat/chat-page-client.tsx)
- [roameo-frontend/lib/session-store.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-store.ts)
- [roameo-frontend/lib/session-view.ts](/Users/yk0007/MyRepos/Roameo/roameo-frontend/lib/session-view.ts)

## Runtime behavior

Each turn runs a deterministic 10-step pipeline:

1. intent resolution
2. research gate
3. destination discovery
4. date context enrichment
5. itinerary synthesis for planning/refinement turns
6. logistics enrichment
7. feasibility critic
8. transit advisor
9. conversational narration plus structured response block assembly
10. session commit

The LLM call pattern is:

- all turns: intent resolution + narration
- planning/refinement turns: an additional structured planning call for itinerary synthesis

Current product behavior in the canonical path includes:

- greeting and capability turns
- destination discovery and category search
- date-aware planning and refinement
- stay recommendations
- weather, event, and holiday advisories
- structured itinerary replies
- live agent trace feedback during planning

## Shared contracts

The contracts package defines the canonical schemas for:

- providers and run modes
- session settings and preferences
- planning state and date context
- POIs and POI catalogs
- plans, itinerary days, and destination segments
- assistant response blocks
- conversation messages
- agent traces
- stream events
- message, mutation, and settings payloads

Do not introduce mirrored frontend/backend business schemas for these concepts.

## Providers and integrations

AI providers:

- Gemini
- OpenAI

Tooling and data providers:

- Google Places, Geocoding, Directions, and photo proxying
- Open-Meteo for forecast summaries
- Tavily for destination facts, deep research, and event research
- Nager.Date for public holiday context
- Supabase for auth and persistent storage

Provider resolution and BYOK handling live in [Backend/src/services/provider-service.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/services/provider-service.ts).

## Development

Requirements:

- Node.js `>=22`
- npm workspaces
- Supabase credentials for persistent mode
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

Build:

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

Core backend environment variables include:

- `PORT`
- `APP_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_MODEL_FAST`
- `GEMINI_MODEL_BALANCED`
- `GEMINI_MODEL_DEEP`
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

Frontend runtime points at the backend via `NEXT_PUBLIC_BACKEND_URL`.

## Docs map

Start with [docs/README.md](/Users/yk0007/MyRepos/Roameo/docs/README.md).

Canonical docs:

- [docs/AUTONOMOUS_AGENTS.md](/Users/yk0007/MyRepos/Roameo/docs/AUTONOMOUS_AGENTS.md)
- [docs/CANONICAL_ARCHITECTURE.md](/Users/yk0007/MyRepos/Roameo/docs/CANONICAL_ARCHITECTURE.md)
- [docs/API_REFERENCE.md](/Users/yk0007/MyRepos/Roameo/docs/API_REFERENCE.md)
- [docs/PLANNING_RUNTIME.md](/Users/yk0007/MyRepos/Roameo/docs/PLANNING_RUNTIME.md)
- [docs/FRONTEND_SURFACE.md](/Users/yk0007/MyRepos/Roameo/docs/FRONTEND_SURFACE.md)
- [docs/OPERATIONS_AND_TESTING.md](/Users/yk0007/MyRepos/Roameo/docs/OPERATIONS_AND_TESTING.md)
- [docs/AGENTS.md](/Users/yk0007/MyRepos/Roameo/docs/AGENTS.md)

Supporting notes:

- [PERFORMANCE_OPTIMIZATIONS.md](/Users/yk0007/MyRepos/Roameo/PERFORMANCE_OPTIMIZATIONS.md)
