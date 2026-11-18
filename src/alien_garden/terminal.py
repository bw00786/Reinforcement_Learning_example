"""Colorful terminal rendering for episode summaries."""

from .mappo import EpisodeSummary


class TerminalRenderer:
    RESET = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _paint(self, color: str, text: str) -> str:
        return f"{color}{text}{self.RESET}" if self.enabled else text

    def episode(self, summary: EpisodeSummary) -> str:
        agents = ", ".join(
            f"A{i + 1}: {reward:.2f}"
            for i, reward in enumerate(summary.agent_rewards)
        )
        actions = ", ".join(f"{name}" for name in summary.actions)
        return (
            f"{self._paint(self.CYAN, f'Episode {summary.episode:>3}')}  "
            f"{self._paint(self.GREEN, f'co-op {summary.cooperative_reward:.2f}')}  "
            f"{self._paint(self.YELLOW, f'harvest {summary.harvested}')}  "
            f"{self._paint(self.MAGENTA, f'plant {summary.planted}')}  "
            f"steps {summary.steps} | {agents} | actions: {actions}"
        )

    def final(self, summaries: list[EpisodeSummary]) -> str:
        if not summaries:
            return "No episodes completed."
        best = max(summaries, key=lambda item: item.cooperative_reward)
        mean_reward = sum(item.cooperative_reward for item in summaries) / len(summaries)
        total_harvested = sum(item.harvested for item in summaries)
        return (
            f"{self._paint(self.CYAN, 'Alien Garden Complete')}\n"
            f"episodes={len(summaries)}  mean_coop={mean_reward:.3f}  "
            f"total_harvested={total_harvested}  "
            f"{self._paint(self.GREEN, f'best_episode={best.episode}')} "
            f"(reward={best.cooperative_reward:.3f})"
        )
