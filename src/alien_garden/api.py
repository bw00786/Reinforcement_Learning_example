"""HTTP API for running and monitoring alien-garden training sessions."""

import logging
from dataclasses import asdict
from threading import Lock
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .environment import AlienGardenEnv
from .mappo import MAPPOTrainer

logger = logging.getLogger(__name__)
app = FastAPI(title="Alien Garden Trainer API", version="0.1.0")

_lock = Lock()
_runs: dict[str, dict] = {}
_current_run_id: str | None = None


class TrainingConfig(BaseModel):
    episodes: int = Field(default=10, ge=1, le=500)
    seed: int = Field(default=42)
    agents: int = Field(default=3, ge=1, le=8)
    learning_rate: float = Field(default=3e-4, gt=0, le=0.1)
    grid_size: int = Field(default=8, ge=3, le=24)
    max_steps: int = Field(default=100, ge=1, le=1000)


def _snapshot(run_id: str) -> dict:
    with _lock:
        run = _runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Training run not found")
        return {
            **run,
            "config": dict(run["config"]),
            "episodes": [dict(episode) for episode in run["episodes"]],
        }


def _train_run(run_id: str, config: TrainingConfig) -> None:
    try:
        env = AlienGardenEnv(
            grid_size=config.grid_size,
            num_agents=config.agents,
            max_steps=config.max_steps,
            seed=config.seed,
        )
        trainer = MAPPOTrainer(env=env, learning_rate=config.learning_rate, seed=config.seed)
        for episode_number in range(1, config.episodes + 1):
            summary = asdict(trainer.run_episode(episode_number))
            with _lock:
                run = _runs[run_id]
                run["episodes"].append(summary)
                run["progress"] = episode_number / config.episodes

        with _lock:
            _runs[run_id]["status"] = "completed"
            _runs[run_id]["progress"] = 1.0
    except Exception as error:
        logger.exception("Training run %s failed", run_id)
        with _lock:
            _runs[run_id]["status"] = "failed"
            _runs[run_id]["error"] = str(error)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs/current")
def current_run() -> dict | None:
    with _lock:
        run_id = _current_run_id
    return _snapshot(run_id) if run_id is not None else None


@app.post("/api/runs", status_code=202)
def start_run(config: TrainingConfig, background_tasks: BackgroundTasks) -> dict:
    global _current_run_id
    with _lock:
        if _current_run_id is not None and _runs[_current_run_id]["status"] == "running":
            raise HTTPException(status_code=409, detail="A training run is already in progress")

        run_id = str(uuid4())
        _runs[run_id] = {
            "id": run_id,
            "status": "running",
            "progress": 0.0,
            "config": config.model_dump(),
            "episodes": [],
            "error": None,
        }
        _current_run_id = run_id

    background_tasks.add_task(_train_run, run_id, config)
    return _snapshot(run_id)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _snapshot(run_id)
