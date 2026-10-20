# frontend

React + TypeScript dashboard for the FRC Scout AI strategy API.

Two pages, no router — page state lives in `App.tsx`:

- **Match Analysis** — quantitative dashboard (win probability, per-alliance OPR,
  chemistry, phase split, per-robot cards) plus the Strategy and Captain LLM
  reads, streamed in parallel.
- **Captain** — a single alliance briefing on its own page.

Streaming reads arrive as NDJSON over HTTP and are consumed by `useReader.ts`
(one read) and `useBots.ts` (six per-robot reads). Both render the model's
thinking phase as a progress bar before the answer text appears.

## Run

The API must be running on `:8000` first — see the repo root README. Vite
proxies `/api` to it.

```bash
npm install
npm run dev      # http://localhost:5173
```

## Build and check

```bash
npm run build        # production bundle -> dist/
npm run lint         # oxlint
npx tsc --noEmit     # type check
```

`dist/` is also served directly by the FastAPI app, so a single-container
deploy needs no separate web server.
