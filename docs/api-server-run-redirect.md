# Active run redirect

`POST /v1/runs/{run_id}/redirect` sends a correction to a currently running
Hermes run owned by the calling API scope. It is separate from `steer`: an
active model request is interrupted and restarted with the correction in the
same logical turn; a tool batch may intentionally defer the correction.

The request accepts one non-empty `input`, `message`, or `text` field. A
successful response is:

```json
{"object":"hermes.run.redirect","run_id":"run_…","accepted":true}
```

`accepted` means only that the agent accepted the correction. It is not proof
that a replacement response has been generated or delivered. Read the run's
SSE stream and terminal status before acting on a replacement response.

The endpoint returns 404 for an unknown or differently scoped run, and 409
when the run is terminal, no longer accepts redirects, or the agent rejects
the correction. It does not stop a run or cancel any tool work.
