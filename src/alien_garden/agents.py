"""PyTorch actor and centralized critic networks."""

import torch
from torch import nn


class ActorPolicy(nn.Module):
    """A decentralized policy that acts only on one agent's local observation."""

    def __init__(self, observation_size: int, action_size: int = 7, hidden_size: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)


class CentralizedCritic(nn.Module):
    """Estimate each agent's value from the full shared environment state."""

    def __init__(self, state_size: int, num_agents: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_agents),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ActorCriticPolicy(nn.Module):
    def __init__(self, observation_size: int, action_size: int = 7, hidden_size: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        value = self.critic(observation).squeeze(-1)
        return action, distribution.log_prob(action), value
