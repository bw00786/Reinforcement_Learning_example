from alien_garden.environment import AlienGardenEnv, HARVEST, PLANT


def test_reset_is_reproducible():
    first = AlienGardenEnv(seed=7)
    second = AlienGardenEnv(seed=7)
    assert (first.reset() == second.reset()).all()
    assert first.harvested == 0


def test_invalid_action_count():
    env = AlienGardenEnv(num_agents=2)
    env.reset()
    try:
        env.step([0])
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_plant_and_harvest_state_change():
    env = AlienGardenEnv(grid_size=5, num_agents=1, seed=3)
    env.reset()
    env.positions[0] = [0, 0]
    env.grid.fill(0)
    result = env.step([PLANT])
    assert env.planted == 1
    env.step([0])
    env.step([0])
    harvest = env.step([HARVEST])
    assert env.harvested == 1
    assert harvest.done is True
