import time

from fastapi.testclient import TestClient

from alien_garden.api import app


def test_training_run_completes_and_returns_episode_metrics():
    client = TestClient(app)
    response = client.post(
        "/api/runs",
        json={
            "episodes": 1,
            "seed": 11,
            "agents": 1,
            "grid_size": 3,
            "max_steps": 1,
        },
    )

    assert response.status_code == 202
    run = response.json()
    deadline = time.monotonic() + 60
    while run["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
        run = client.get(f"/api/runs/{run['id']}").json()

    assert run["status"] == "completed"
    assert run["progress"] == 1
    assert len(run["episodes"]) == 1
    assert run["episodes"][0]["episode"] == 1
    assert len(run["episodes"][0]["agent_rewards"]) == 1


def test_training_run_rejects_invalid_configuration():
    response = TestClient(app).post("/api/runs", json={"episodes": 0})

    assert response.status_code == 422
