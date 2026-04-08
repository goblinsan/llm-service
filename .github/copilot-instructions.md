# Copilot Instructions

This repository provides a wrapper around `llama.cpp` and is intended to be deployed by `gateway-control-plane` as a `container-service` using `build.strategy: repo-compose`.

## Deployment contract

- Keep `docker-compose.yml`, `wrapper/Dockerfile`, `.env.example`, and `README.md` aligned.
- The wrapper service must continue listening on container port `8080`.
- The published host port must remain configurable through `HOST_PORT`.
- The health endpoint contract is `GET /health`.
- Model-management endpoints are:
  - `GET /api/models`
  - `POST /api/models/download`
  - `GET /api/models/download/{task_id}`
  - `POST /api/models/load`

## Configuration rules

- Do not hardcode private LAN IPs, node IDs, hostnames, usernames, or local filesystem paths in tracked files.
- Keep operator-specific values as placeholders in docs and examples.
- If you add or rename an environment variable, update:
  - `docker-compose.yml`
  - `.env.example`
  - `README.md`
  - tests that rely on the config

## Runtime behavior

- Preserve the wrapper's readiness semantics:
  - `/health` returns `loading`, `ok`, or `error`
  - inference requests return `503` while the model is loading
- Do not hardcode a single upstream `llama-server` binary path unless it is verified in the current base image. Prefer auto-detection plus an explicit `LLAMA_BIN` override.
- Preserve the concurrency guard and request-timeout behavior unless intentionally changed and documented.
- Treat download and model-switch endpoints as security-sensitive. Maintain path sanitization and auth requirements.

## Scope discipline

- Do not assume gateway-control-plane, gateway-api, or chat-platform changes are done in this repo.
- Cross-repo work should be documented as follow-up, not silently implied as complete.

## Validation

Before considering a task complete, prefer verifying:

- Python syntax for `wrapper/`
- README examples still match the implemented endpoints
- Compose/env wiring still matches the documented deployment flow
