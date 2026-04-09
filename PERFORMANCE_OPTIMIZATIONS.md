# Performance and Scalability Notes

This file reflects the current canonical implementation.

## Current performance priorities

### Router-first execution

The biggest practical win is not a cache trick, it is using the heavy pipeline only when a turn actually needs it.

Current direction:

- trivial conversation -> fast local response
- normal travel turns -> semantic router
- only then -> discovery / enrichment / planning tools

That reduces unnecessary provider calls and noisy “thinking” states.

### Snapshot-first state flow

The biggest correctness win remains architectural consistency:

- one `SessionSnapshot`
- one canonical plan
- one canonical POI catalog
- one live stream per session

That prevents duplicated reconstruction across chat, itinerary, map, and saved POI panels.

### Merged POI catalogs

Discovery no longer behaves like a throwaway turn-local cache.

By merging POI catalogs across turns:

- refreshes keep prior stay/restaurant/attraction cards resolvable
- the model and frontend can work from a richer accumulated context
- the user does not lose useful discovery state every time they change topic

### Direct integration model

The current runtime uses direct integrations instead of wrappers:

- Google Places and Directions
- Open-Meteo
- Tavily
- Nager.Date
- Supabase

That keeps latency and failure handling explicit.

## Current bottlenecks to watch

- semantic router latency
- narrative/synthesis latency
- Google Places and Directions latency
- Tavily latency on deep-research turns
- Supabase round-trips for full snapshot rebuilds
- large itinerary payloads and long trace histories
- client-side Google Maps marker and hover-card churn

## Current mitigations

- fast-path conversation handling
- router-first turn understanding
- deterministic plan mutations for targeted edits
- merged session POI catalogs
- right-panel remounting on plan-version changes
- plan-only itinerary projection
- full-catalog chat card rendering instead of map-slice-only rendering

## Scale-oriented implementation rules

For production-grade behavior, the repo direction remains:

- one canonical codepath
- one source of truth for schemas and state
- direct integrations
- fail-fast validation
- explicit tool surfaces for autonomous agents
- session-level context invalidation instead of sticky branch logic

## Future optimization directions

- route-result caching per waypoint set
- cached semantic reranking for fuzzy prompts
- trace compaction for long-lived sessions
- snapshot diff emission for especially chatty streams
- selective caching around geocoding and destination research
- optional Gemini Maps-grounded semantic expansion layered on top of direct Places retrieval
