import torch

from alien_garden.environment import AlienGardenEnv
from alien_garden.mappo import MAPPOTrainer, compute_gae


def test_episode_summary_is_deterministic_shape():
    env = AlienGardenEnv(grid_size=5, num_agents=2, max_steps=3, seed=11)
    trainer = MAPPOTrainer(env=env, seed=11)
    summary = trainer.run_episode(1)
    assert summary.episode == 1
    assert summary.steps == 3
    assert len(summary.agent_rewards) == 2
    assert len(summary.actions) == 2


def test_gae_uses_discounted_rewards_and_terminal_mask():
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[0.5], [0.25]])
    dones = torch.tensor([False, True])

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([[1.5], [0.75]]))
    torch.testing.assert_close(returns, torch.tensor([[2.0], [1.0]]))


def test_gae_bootstraps_nonterminal_rollout():
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.5]])
    dones = torch.tensor([False])

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        gamma=0.9,
        gae_lambda=0.8,
        next_value=torch.tensor([2.0]),
    )

    torch.testing.assert_close(advantages, torch.tensor([[2.3]]))
    torch.testing.assert_close(returns, torch.tensor([[2.8]]))
