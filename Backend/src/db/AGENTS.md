# Legacy DB Layer

## Status
- The canonical persistence layer is `src/services/session-repository.ts`.
- Do not add new app behavior to the old DB abstractions in this directory.

## Migration Rule
- Prefer explicit session tables over opaque trip blobs and compatibility tables.
