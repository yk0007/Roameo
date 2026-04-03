# Operations and Testing

## Workspace commands

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

Typecheck all workspaces:

```bash
npm run typecheck
```

Test all workspaces:

```bash
npm run test
```

## Workspace-specific commands

Contracts:

```bash
npm run build -w @roameo/contracts
npm run test -w @roameo/contracts
```

Backend:

```bash
npm run build -w roameo-backend
npm run typecheck -w roameo-backend
npm run test -w roameo-backend
```

Frontend:

```bash
npm run build -w roameo-frontend
npm run typecheck -w roameo-frontend
npm run test -w roameo-frontend
```

## What to validate after runtime changes

For backend, contracts, or shared-schema changes, verify:

- contracts build and tests
- backend build and tests
- frontend typecheck
- session stream behavior
- one end-to-end turn when provider keys are available

Recommended manual scenarios:

1. Create a session and send an initial planning message.
2. Watch the stream-driven planning states and traces update live.
3. Refine the itinerary with a follow-up message.
4. Edit destination, dates, travelers, and budget from the header.
5. Save and unsave POIs.
6. Confirm chat, itinerary, saved POIs, and map stay synchronized.
7. Trigger a failure case and confirm the accepted plan remains intact.

## Persistence modes

Persistent mode requires:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Without those values, the backend falls back to in-memory session storage.

## Provider and integration requirements

Meaningful planning requires:

- `GOOGLE_MAPS_API_KEY`
- at least one of `GEMINI_API_KEY` or `OPENAI_API_KEY`

Optional but important enrichments:

- `TAVILY_API_KEY` for destination facts, deep research, and event research

## Frontend production notes

The current frontend stack includes:

- Next.js 16
- React 19
- TanStack Query
- Zustand
- Radix primitives
- shared contract types from `@roameo/contracts`

Operationally important notes:

- live updates arrive over SSE, even though the client helper is named `connectWs()`
- route rendering still uses Google Maps DirectionsService/DirectionsRenderer in the current code
- image surfaces rely on the backend photo proxy

## Documentation policy

When the canonical implementation changes:

- update the root README
- update the canonical docs pack in `docs/`
- keep compatibility notes short and clearly secondary
- do not document retired WebSocket/LangGraph-first flows as current architecture
