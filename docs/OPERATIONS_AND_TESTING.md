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
2. Watch planning state and traces update live.
3. Ask for stays, restaurants, attractions, and generic places after a plan already exists.
4. Change destination completely and confirm the active trip context resets.
5. Try a multi-city trip and confirm the full destination set survives later turns.
6. Refresh after discovery turns and confirm POI cards still resolve from the canonical catalog.
7. Confirm itinerary tab changes only when the actual plan changes.
8. Confirm map routes come only from itinerary-linked POIs while non-itinerary POIs still appear as plain markers.

## Persistence and state checks

Current state invariants worth validating:

- POI catalogs merge across discovery turns
- stale follow-up context is cleared or overridden when the user changes topic
- accepted decision entries are replaced by key, not appended forever
- new-trip requests replace stale active destination context

## Provider and integration requirements

Meaningful planning requires:

- `GOOGLE_MAPS_API_KEY`
- at least one of `GEMINI_API_KEY` or `OPENAI_API_KEY`

Optional enrichments:

- `TAVILY_API_KEY`

## Current Gemini model defaults

Verified working IDs in the current environment:

- `gemini-flash-latest`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemma-3-27b-it`
- `gemma-4-31b-it`
- `gemini-embedding-001`

Current backend defaults:

- router: `gemma-4-31b-it`
- router fallback: `gemini-2.5-flash`
- narrative: `gemini-flash-latest`
- narrative fallback: `gemini-2.5-flash`

## Current frontend notes

- live updates arrive over SSE
- chat cards use the full canonical POI catalog
- right-panel itinerary/map remount on canonical plan changes
- map hover cards depend on Google Maps info-window lifecycle and marker refresh behavior

## Documentation policy

When the canonical implementation changes:

- update the root README
- update the docs pack in `docs/`
- document current verified model IDs if model routing changed
- keep frontend notes lighter than backend/runtime details
