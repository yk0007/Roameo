# App Routes

## Rules
- Route files should stay thin. Push state orchestration into shared helpers or stores when a page starts to sprawl.
- Authenticated routes must use one consistent auth bootstrap path.
- Keep query-param behavior deterministic, especially for `sessionId` and prefilled `message` flows.
