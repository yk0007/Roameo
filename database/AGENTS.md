# Database

## Canonical Schema
- New work should target the canonical session runtime schema, not the old `chat_sessions` or `sessions` compatibility tables.
- Prefer additive, explicit tables with indexes and RLS over opaque JSON blobs.

## Migration Rule
- Keep schema, repository assumptions, and contracts aligned. If one changes, update the others in the same delivery.
