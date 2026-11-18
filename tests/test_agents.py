import torch

from alien_garden.agents import ActorCriticPolicy


def test_policy_returns_action_logprob_and_value():
    policy = ActorCriticPolicy(observation_size=12)
    observation = torch.zeros(12)
    action, log_prob, value = policy.act(observation)
    assert action.shape == torch.Size([])
    assert log_prob.shape == torch.Size([])
    assert value.shape == torch.Size([])
