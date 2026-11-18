# Alien Garden MAPPO

A cooperative alien-garden grid simulation with decentralized PyTorch policies,
MAPPO-style training, a terminal interface, and a React dashboard backed by FastAPI.

## Requirements

- Python 3.11 or newer
- Node.js 22 or newer and npm
- Docker Desktop with Compose (optional, for containerized startup)

## Run the web app locally

Install the Python dependencies once:

```powershell
python -m pip install -e .
```

Install the frontend dependencies once:

```powershell
cd frontend
npm ci
cd ..
```

Start both services from the repository root in Windows PowerShell:

```powershell
.\scripts\start-dev.ps1
```

If PowerShell's execution policy prevents running local scripts, use a
process-scoped bypass instead of changing the machine-wide policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-dev.ps1
```

The script starts whichever service is not already healthy. The dashboard is at
<http://127.0.0.1:5173/> and the FastAPI Swagger UI is at
<http://127.0.0.1:8000/docs>. Logs and process IDs for services started by the
script are written to `.dev-run/`.

To run either service manually:

```powershell
python -m uvicorn alien_garden.api:app --reload
```

```powershell
cd frontend
npm run dev
```

Vite proxies `/api` requests to `http://localhost:8000` by default. Set
`VITE_API_PROXY_TARGET` to override the backend URL.

## Run with Docker Compose

Start the API and frontend:

```bash
docker compose up --build
```

Open <http://localhost:5173/>. The API docs are at
<http://localhost:8000/docs>. To run only the terminal trainer:

```bash
docker compose --profile cli up --build alien-garden-trainer
```

## Training dashboard and API

The dashboard lets you configure episode count, seed, agent count, learning rate,
garden size, and maximum steps per episode. It displays the latest run's progress,
cooperative reward curve, harvests, plantings, and per-agent rewards.

The API accepts a `POST /api/runs` JSON body with these fields:

| Field | Default | Allowed range |
|---|---:|---:|
| `episodes` | 10 | 1–500 |
| `seed` | 42 | integer |
| `agents` | 3 | 1–8 |
| `learning_rate` | 0.0003 | greater than 0 through 0.1 |
| `grid_size` | 8 | 3–24 |
| `max_steps` | 100 | 1–1000 |

`GET /api/health` reports API health. `GET /api/runs/current` returns the current
or most recently started run, and `GET /api/runs/{run_id}` returns that run's
status and completed episode summaries. Only one run can train at a time; run
history and model weights are held in memory and are not persisted.

## Training algorithm

Each agent has a decentralized actor that chooses actions from its local
observation. A shared centralized critic estimates per-agent values from the full
garden state, seedling ages, all agent positions, and episode progress. The
trainer collects transitions at every step, computes generalized advantage
estimates (GAE), and applies clipped PPO updates over four epochs with minibatches
of up to 64 transitions. Defaults are `gamma=0.99`, `lambda=0.95`, PPO clip
epsilon `0.2`, entropy coefficient `0.01`, and value coefficient `0.5`; actor and
critic gradients are clipped to a norm of `0.5`.

The environment resets to the same initial garden layout for each episode of a
run; the seed also initializes the policy parameters and random action sampling.
The implementation is an educational MAPPO-style example, not a guarantee of
convergence or improved reward. Training metrics are useful for comparing
experiments, but assess learning trends across multiple seeds.

## Run from the terminal

Install the project and run training without the web services:

```powershell
python -m pip install -e .
python -m alien_garden --episodes 10 --seed 42
```

Available CLI options:

| Option | Default | Description |
|---|---:|---|
| `--episodes` | 10 | Number of episodes |
| `--seed` | 42 | Experiment random seed |
| `--agents` | 3 | Number of gardeners |
| `--learning-rate` | 0.0003 | Actor and critic optimizer learning rate |
| `--grid-size` | 8 | Width and height of the square garden (minimum 3) |
| `--max-steps` | 100 | Maximum steps per episode |
| `--no-render` | off | Disable ANSI terminal colors |

## Tests

Run the Python tests:

```powershell
python -m pytest
```

Build the frontend:

```powershell
cd frontend
npm run build
```
