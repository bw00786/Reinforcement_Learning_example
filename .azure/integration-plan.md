# Integration Record

## Application

- Frontend: React 18 and Vite in `frontend/`
- API: FastAPI and Uvicorn in `src/alien_garden/api.py`
- Trainer: PyTorch MAPPO implementation in `src/alien_garden/`
- Persistence: none; run data and model weights are in memory
- Azure services: none

## Local Development

- Start both services on Windows with `.\scripts\start-dev.ps1`.
- Frontend: <http://127.0.0.1:5173/>
- API docs: <http://127.0.0.1:8000/docs>
- API health: `GET /api/health`
- Compose: `docker compose up --build`

The startup script reuses a healthy process already listening on a service port.
It writes logs and process IDs for services it starts to `.dev-run/`, which is
git-ignored.

## API

- `POST /api/runs`: validate configuration and start one background run.
- `GET /api/runs/current`: fetch the latest run.
- `GET /api/runs/{run_id}`: poll run progress and episode results.
- A second run is rejected with HTTP 409 while training is in progress.
- Run history is process-local and is lost when the API restarts.

## Training

The trainer uses local per-agent actors, a centralized critic over the full
garden state, per-step rollouts, GAE, clipped PPO updates, entropy regularization,
and gradient clipping. Defaults and algorithm details are documented in
`README.md`.

## Validation

- `python -m pytest -q`: 10 passed.
- `python -m compileall -q src tests`: passed.
- CLI one-episode smoke test: passed.
- Vite production build (`npm run build` in `frontend/`): passed.
- Live frontend and API smoke test, including a completed training run: passed.
- `docker-compose config`: passed.

The npm dependency audit reports a high-severity advisory in transitive
`picomatch`; no patched release was available during implementation.
