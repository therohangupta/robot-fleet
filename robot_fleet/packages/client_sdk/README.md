# client_sdk

Shared, **multi-client** SDKs for talking to the **Gateway/BFF**.

Why this exists:
- Web, mobile, CLI, scripts should all call the same Gateway contract.
- The Gateway is the *only* client-facing API surface.
- This folder contains:
  - a contract snapshot (OpenAPI + realtime event schema)
  - a **TypeScript SDK** (`typescript/`) &mdash; used by the dashboard frontend (`dashboard-web`)
  - a **Python SDK** (`python/`) &mdash; used by CLI/scripts

### TypeScript SDK

The dashboard frontend imports this as `@robot-fleet/client-sdk` (linked via `file:` in `package.json`). Key exports:

- `gatewayClient` &mdash; typed HTTP client for all Gateway REST endpoints (robots, plans, tasks, goals, world, methods).
- `GatewayRealtimeClient` &mdash; WebSocket client for real-time event subscriptions.
- `fetchApi` &mdash; low-level fetch wrapper used by the gateway client.

What this is NOT:
- It is **not** gateway server code.
- It is **not** fleet-server gRPC client code (that is a separate "fleet SDK" used by the gateway).
