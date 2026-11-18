from alien_garden.mappo import EpisodeSummary
from alien_garden.terminal import TerminalRenderer


def test_terminal_summary_and_final_output():
    summary = EpisodeSummary(
        episode=1,
        cooperative_reward=1.5,
        agent_rewards=[0.8, 0.7],
        harvested=2,
        planted=3,
        steps=10,
        actions=["harvest", "plant"],
    )
    renderer = TerminalRenderer(enabled=False)
    episode_line = renderer.episode(summary)
    final_line = renderer.final([summary])
    assert "Episode   1" in episode_line
    assert "Alien Garden Complete" in final_line
