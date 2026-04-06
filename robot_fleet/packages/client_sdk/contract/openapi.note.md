### OpenAPI contract snapshot

For production-grade multi-client support, the Gateway should publish a stable OpenAPI schema.

In this repo today, the Gateway is FastAPI, so you can export OpenAPI at runtime from:
- `GET /openapi.json`

Recommended workflow:
- On release (or after API changes), snapshot the schema into `contract/openapi.json` (or `openapi.yaml`).
- Generate SDKs (TypeScript + Python) from that snapshot, or validate the handwritten SDKs against it.

This file is a placeholder to keep the contract directory “real” in v2 before we automate exports.

