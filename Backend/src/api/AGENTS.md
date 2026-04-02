# API Surface

## Rules
- Treat `src/api/router.ts` as the single authoritative HTTP contract for the app shell.
- Keep routes narrow, validated, and fail-fast.
- Session mutations must emit a fresh canonical snapshot or a typed stream event when the client needs live updates.
