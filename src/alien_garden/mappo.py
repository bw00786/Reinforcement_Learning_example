"""Small centralized-critic multi-agent PPO loop."""

from dataclasses import dataclass

import torch
from torch import nn, optim

from .agents import ActorPolicy, CentralizedCritic
from .environment import AlienGardenEnv
from .rewards import calculate_rewards


@dataclass
class EpisodeSummary:
    episode: int
    cooperative_reward: float
    agent_rewards: list[float]
    harvested: int
    planted: int
    steps: int
    actions: list[str]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-agent generalized advantages and value targets."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(gae) if next_value is None else next_value

    for step in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[step].to(rewards.dtype)
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
        next_value = values[step]

    return advantages, advantages + values


class MAPPOTrainer:
    def __init__(
        self,
        env: AlienGardenEnv,
        learning_rate: float = 3e-4,
        seed: int = 42,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= gae_lambda <= 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if not 0 < clip_epsilon < 1:
            raise ValueError("clip_epsilon must be between 0 and 1")
        if update_epochs < 1 or minibatch_size < 1:
            raise ValueError("update_epochs and minibatch_size must be at least 1")

        self.env = env
        torch.manual_seed(seed)
        self.policies = [
            ActorPolicy(observation_size=12) for _ in range(env.num_agents)
        ]
        self.optimizers = [
            optim.Adam(policy.parameters(), lr=learning_rate) for policy in self.policies
        ]
        self.critic = CentralizedCritic(
            state_size=env.global_state().size,
            num_agents=env.num_agents,
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    def _update(
        self,
        observations: torch.Tensor,
        global_states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        rollout_size = observations.shape[0]
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        for _ in range(self.update_epochs):
            indices = torch.randperm(rollout_size)
            for batch_indices in indices.split(self.minibatch_size):
                for agent_index, (policy, optimizer) in enumerate(
                    zip(self.policies, self.optimizers)
                ):
                    distribution = torch.distributions.Categorical(
                        logits=policy(observations[batch_indices, agent_index])
                    )
                    new_log_probs = distribution.log_prob(actions[batch_indices, agent_index])
                    ratio = (new_log_probs - old_log_probs[batch_indices, agent_index]).exp()
                    batch_advantages = normalized_advantages[
                        batch_indices, agent_index
                    ]
                    unclipped = ratio * batch_advantages
                    clipped = ratio.clamp(
                        1 - self.clip_epsilon, 1 + self.clip_epsilon
                    ) * batch_advantages
                    actor_loss = -torch.minimum(unclipped, clipped).mean()
                    actor_loss -= self.entropy_coef * distribution.entropy().mean()

                    optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                    optimizer.step()

                predicted_values = self.critic(global_states[batch_indices])
                previous_values = old_values[batch_indices]
                clipped_values = previous_values + (
                    predicted_values - previous_values
                ).clamp(-self.clip_epsilon, self.clip_epsilon)
                value_loss = torch.maximum(
                    (predicted_values - returns[batch_indices]).pow(2),
                    (clipped_values - returns[batch_indices]).pow(2),
                ).mean()

                self.critic_optimizer.zero_grad()
                (self.value_coef * 0.5 * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

    def run_episode(self, episode: int) -> EpisodeSummary:
        observations = self.env.reset()
        episode_cooperative = 0.0
        episode_agent_rewards = [0.0] * self.env.num_agents
        rollout_observations = []
        rollout_global_states = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []
        last_actions: list[int] = []

        while True:
            observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
            global_state = torch.as_tensor(self.env.global_state(), dtype=torch.float32)
            with torch.no_grad():
                actions = []
                log_probs = []
                for agent_index, policy in enumerate(self.policies):
                    distribution = torch.distributions.Categorical(
                        logits=policy(observation_tensor[agent_index])
                    )
                    action = distribution.sample()
                    actions.append(action)
                    log_probs.append(distribution.log_prob(action))
                values = self.critic(global_state)

            action_ids = [int(action.item()) for action in actions]
            last_actions = action_ids
            result = self.env.step(action_ids)
            shaped_rewards = calculate_rewards(result.cooperative_reward, result.agent_rewards)

            rollout_observations.append(observation_tensor)
            rollout_global_states.append(global_state)
            rollout_actions.append(torch.stack(actions))
            rollout_log_probs.append(torch.stack(log_probs))
            rollout_values.append(values)
            rollout_rewards.append(torch.tensor(shaped_rewards, dtype=torch.float32))
            rollout_dones.append(result.done)

            episode_cooperative += result.cooperative_reward
            episode_agent_rewards = [
                total + shaped for total, shaped in zip(episode_agent_rewards, shaped_rewards)
            ]
            observations = result.observations
            if result.done:
                break

        with torch.no_grad():
            final_global_state = torch.as_tensor(self.env.global_state(), dtype=torch.float32)
            final_value = self.critic(final_global_state)
        values = torch.stack(rollout_values)
        dones = torch.tensor(rollout_dones, dtype=torch.bool)
        advantages, returns = compute_gae(
            torch.stack(rollout_rewards),
            values,
            dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            next_value=final_value,
        )

        self._update(
            observations=torch.stack(rollout_observations),
            global_states=torch.stack(rollout_global_states),
            actions=torch.stack(rollout_actions),
            old_log_probs=torch.stack(rollout_log_probs),
            old_values=values,
            advantages=advantages,
            returns=returns,
        )

        human_actions = [
            ("wait", "up", "down", "left", "right", "plant", "harvest")[int(value)]
            for value in last_actions
        ]
        return EpisodeSummary(
            episode=episode,
            cooperative_reward=round(episode_cooperative, 3),
            agent_rewards=[round(value, 3) for value in episode_agent_rewards],
            harvested=self.env.harvested,
            planted=self.env.planted,
            steps=self.env.steps,
            actions=human_actions,
        )
