"""Reward shaping helpers."""


def calculate_rewards(cooperative_reward: float, agent_rewards: list[float]) -> list[float]:
    """Combine shared cooperation and individual agent rewards."""
    if not agent_rewards:
        return []
    return [0.8 * cooperative_reward + 0.2 * personal for personal in agent_rewards]
