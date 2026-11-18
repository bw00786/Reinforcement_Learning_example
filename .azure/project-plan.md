# Project Plan

**Status**: Integrated
**Created**: 2026-11-17
**Mode**: NEW

---

## 1. Project Overview

**Goal**: Build a local cooperative alien-garden simulation where multiple learning agents use Python/PyTorch MAPPO to coordinate harvesting, planting, and garden management. Provide both a terminal CLI and a React dashboard backed by a FastAPI training API.

**App Type**: Web application and background training worker

**API Login**: No

**Mode**: NEW

**Deployment Plan**: No deployment plan found

---

## 2. Application Components

| Component | Technology |
|-----------|-----------|
| **Frontend** | React 18, JavaScript, Vite |
| **Frontend package manager** | npm |
| **API and worker language** | Python |
| **Runtime** | CPython 3.11+ |
| **API framework** | FastAPI and Uvicorn |
| **Learning framework** | PyTorch |
| **Python package manager** | pip |
| **Test runner** | pytest |
| **Orchestration** | Docker Compose |

The backend implements a configurable grid garden environment, decentralized
actors, a centralized critic, PPO clipped updates with GAE, cooperative and
agent-level rewards, a FastAPI run/status API, and a terminal renderer. The React
dashboard configures training and displays episode metrics. Run status and model
weights are in-memory only; the application does not persist experiments.

---

## 3. Services Required

| Azure Service | Purpose | Environment Variable | Local Default | Classification |
|---------------|---------|----------------------|---------------|----------------|
| None | The initial application runs entirely locally and requires no Azure-hosted data or compute service. | — | — | Enhancement |

---

## 4. Prerequisites

### Run

| Tool | Service(s) | Installed | Version |
|------|------------|-----------|---------|
| Python | alien-garden-trainer | ✅ | 3.13.14 |
| pip | alien-garden-trainer | ✅ | 25.3 |
| PyTorch | alien-garden-trainer | ✅ | 2.7.1+cpu |
| pytest | alien-garden-trainer | ✅ | 7.4.3 |

### Debug

| Tool | Service(s) | Installed | Version |
|------|------------|-----------|---------|
| Docker | alien-garden-trainer | ✅ | 28.1.1 |
| Docker Compose | alien-garden-trainer | ✅ | Installed |
| Python VS Code Extension | alien-garden-trainer | ❓ | Unknown |

Before implementation, verify the required pip packages and test runner are available in the active Python environment.

---

## 5. Project Structure

```
Reinforcement_Learning_example/
├── README.md
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       └── styles.css
├── scripts/
│   └── start-dev.ps1
├── src/
│   └── alien_garden/
│       ├── __init__.py
│       ├── api.py
│       ├── cli.py
│       ├── environment.py
│       ├── agents.py
│       ├── mappo.py
│       ├── rewards.py
│       └── terminal.py
└── tests/
    ├── test_environment.py
    ├── test_agents.py
    ├── test_mappo.py
    └── test_terminal.py
```

---

## 6. Entry Points and API Routes

| # | Method | Path | Description | Request Body | Response Body | Status Codes |
|---|--------|------|-------------|-------------|--------------|-------------|
| 1 | CLI | `python -m alien_garden` | Train and run a complete set of episodes. | CLI arguments | Terminal episode summaries and final metrics | 0 success, 2 invalid arguments |
| 2 | GET | `/api/health` | Check API health. | None | Health status | 200 |
| 3 | POST | `/api/runs` | Start a training run. | Episode count, seed, agents, learning rate, grid size, max steps | Run ID and initial status | 202 accepted, 409 already running, 422 invalid config |
| 4 | GET | `/api/runs/current` | Get current or latest run. | None | Run status, progress, and completed episodes | 200 |
| 5 | GET | `/api/runs/{run_id}` | Get a specific run. | None | Run status, progress, and completed episodes | 200, 404 unknown run |

---

## 7. Implementation Status

The CLI, FastAPI backend, React/Vite dashboard, local startup script, and
Docker Compose configuration are implemented. The training algorithm uses
per-step rollouts, GAE, PPO clipping, and a centralized critic. No database or
Azure service is currently required. Run the instructions in `README.md` for
local development and testing.
