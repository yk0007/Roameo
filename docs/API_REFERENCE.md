# API and Streaming Reference

The canonical API surface lives in [Backend/src/api/router.ts](/Users/yk0007/MyRepos/Roameo/Backend/src/api/router.ts).

All authenticated endpoints require the current Supabase-backed user context.

## Utility endpoints

### `GET /api/health`

Returns:

- `status`
- `timestamp`

### `GET /api/maps/api-key`

Returns the configured browser Google Maps key.

Failure:

- `503` when `GOOGLE_MAPS_API_KEY` is missing

### `GET /api/maps/reverse-geocode?lat=<lat>&lng=<lng>`

Reverse geocodes browser coordinates for origin autofill.

Returns:

- `label`
- `locality`
- `region`
- `country`
- `formatted`

Failures:

- `400` for missing or invalid coordinates
- `404` when no location is found
- `502` for upstream failures

### `GET /api/proxy/photo?photo_reference=<ref>&maxwidth=<n>`

Proxies Google Places photos through the backend.

Behavior:

- requires `GOOGLE_MAPS_API_KEY`
- returns the upstream image bytes
- sets `Cache-Control: public, max-age=86400`
- sets `Access-Control-Allow-Origin: *`

### `GET /api/destination-image?q=<destination>`

Fetches a high-quality destination cover image using Google Places API.

Returns:

```json
{ "imageUrl": "http://localhost:4000/api/proxy/photo?..." }
```

Failures:

- `400` for missing `q` parameter
- `404` when no image is found
- `502` for upstream API failures

## Session endpoints

### `GET /api/sessions`

Returns:

```json
{ "sessions": [...] }
```

### `POST /api/sessions`

Accepted body:

- `title`
- `providerSettings`
- `initialMessage`

Behavior:

- creates a session
- if `initialMessage` is present, starts a background turn immediately

Response:

- `201` with the created `SessionSnapshot`

### `GET /api/sessions/:sessionId`

Returns the full `SessionSnapshot`.

### `PATCH /api/sessions/:sessionId`

Accepted body matches `sessionMutationSchema`:

- `title`
- `providerSettings`
- `preferences`
- `memory`

Behavior:

- updates the session directly
- emits a fresh `session.snapshot` stream event

### `DELETE /api/sessions/:sessionId`

Deletes the session and returns `204`.

## Conversation endpoints

### `POST /api/sessions/:sessionId/messages`

Accepted body:

- `content`
- optional `providerSettings`

Behavior:

- validates `sendMessageInputSchema`
- starts the turn asynchronously

Response:

```json
{ "accepted": true, "sessionId": "..." }
```

Status:

- `202 Accepted`

### `GET /api/sessions/:sessionId/stream`

Opens the session SSE stream.

Behavior:

- validates access to the session
- sends an immediate `session.snapshot`
- streams live turn, message, trace, plan, and completion events
- emits keepalive comments every 15 seconds

## Plan mutation endpoints

### `POST /api/sessions/:sessionId/plan-mutations`

Accepted body matches `planMutationInputSchema`.

Supported mutations:

- `add_poi`
- `remove_poi`
- `move_activity`
- `regenerate_day`
- `rebalance_trip`
- `update_overview`

Behavior:

- applies the mutation through `PlanMutationService`
- emits `plan.updated` when a plan exists
- always emits an updated `session.snapshot`

Response:

- updated `SessionSnapshot`

## Saved POI endpoints

### `GET /api/sessions/:sessionId/saved-pois`

Returns:

```json
{ "ids": ["poi-id-1", "poi-id-2"] }
```

### `POST /api/sessions/:sessionId/saved-pois`

Accepted body:

- `poiId`
- `saved`

Behavior:

- updates the saved POI set
- emits an updated `session.snapshot`

Returns:

```json
{ "ids": [...] }
```

## User settings endpoints

### `GET /api/me/settings`

Returns:

- `providerSettings`
- `preferences`
- `credentials`

Each credential item includes:

- `provider`
- `keySource`
- `configured`
- `lastUpdatedAt`

### `PUT /api/me/settings`

Accepted body:

- `providerSettings`
- `preferences`

Returns the updated settings payload.

### `PUT /api/me/credentials/:provider`

Providers:

- `gemini`
- `openai`

Accepted body:

- `apiKey`

Behavior:

- encrypts the key
- stores it as a user credential

Response:

- `204 No Content`

## Stream events

The shared stream schema is `streamEventSchema` in [packages/contracts/src/index.ts](/Users/yk0007/MyRepos/Roameo/packages/contracts/src/index.ts).

Current event types:

- `session.snapshot`
- `turn.started`
- `message.delta`
- `message.committed`
- `trace.updated`
- `plan.updated`
- `turn.completed`
- `turn.failed`

## Canonical schemas

Important request and response schemas in the contracts package:

- `CreateSessionInput`
- `SendMessageInput`
- `PlanMutationInput`
- `SessionMutation`
- `UserSettingsUpdate`
- `ProviderCredentialInput`
- `SessionSnapshot`
- `PlanSnapshot`
- `ConversationMessage`
- `AssistantResponseBlock`
- `AgentTraceEvent`
- `StreamEvent`

## Errors

Validation uses Zod at the router boundary.

Common error behavior:

- `400` for invalid request payloads
- `404` for unknown or unauthorized session resources
- `500` for unexpected server failures
- `503` for missing Google Maps configuration on helper endpoints

Planning failures during turn execution are surfaced through:

- updated planning state in the snapshot
- fallback assistant message
- `turn.failed` SSE event
