# Performance and Scalability Notes

This file reflects the current canonical implementation, not the older cache-wrapper or WebSocket-first architecture.

## Current performance priorities

### Snapshot-first state flow

The biggest performance and correctness win is architectural consistency:

- one `SessionSnapshot`
- one canonical plan
- one canonical POI catalog
- one live stream per session

That prevents duplicated reconstruction across chat, itinerary, map, and saved POI panels.

### Shared contracts and fail-fast validation

The contracts package centralizes validation for:

- session payloads
- messages
- plan mutations
- stream events
- settings
- response blocks

This keeps bad payloads from leaking deep into the runtime.

### Coarse-grained persistence

The repository persists a small number of coarse shapes:

- session root
- messages
- latest plan snapshot
- latest POI catalog
- saved POI ids
- trace events

That keeps session rehydration straightforward and predictable.

### SSE over duplicate live channels

The canonical live path is SSE.

Benefits:

- easier reconnect behavior
- immediate snapshot replay
- less state duplication than parallel live transports
- schema-validated stream payloads

### Direct integration model

The current runtime uses direct integrations instead of adapters:

- Google Places and Directions
- Open-Meteo
- Tavily
- Nager.Date
- Supabase

That reduces translation layers and keeps failure handling explicit.

## Current bottlenecks to watch

- provider latency from Gemini/OpenAI model calls
- Google Places and Directions latency
- Tavily latency on deep-research turns
- Supabase round-trips for snapshot rebuilds
- large itinerary payloads and long trace histories
- client-side map route rendering cost on dense itineraries

## Current mitigations

- snapshot-derived UI instead of panel-specific business state
- preserved last accepted plan on failures
- plan and catalog stored separately so discovery can survive failed replans
- deterministic plan mutations for targeted edits
- structured response blocks instead of markdown parsing heuristics
- optimistic header editing with reconciliation back to the snapshot

## Scale-oriented implementation rules

For production-grade behavior, the repo direction remains:

- one canonical codepath
- one source of truth for business schemas
- direct integrations
- fail-fast validation
- snapshot-first UI projections

These matter more than micro-optimizations because the main scaling risk is state drift across surfaces.

## Future optimization directions

- route-result caching per waypoint set
- POI metadata and image normalization
- trace compaction for long-lived sessions
- snapshot diff emission for especially chatty streams
- selective caching around destination research and geocoding
