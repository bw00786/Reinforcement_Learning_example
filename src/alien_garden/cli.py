"""Command-line entry point for the alien garden trainer."""

import argparse
import random

import numpy as np
import torch

from .environment import AlienGardenEnv
from .mappo import MAPPOTrainer
from .terminal import TerminalRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train cooperative alien gardeners")
    parser.add_argument("--episodes", type=int, default=10, help="number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="experiment random seed")
    parser.add_argument("--agents", type=int, default=3, help="number of learning agents")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PyTorch optimizer learning rate")
    parser.add_argument("--grid-size", type=int, default=8, help="garden grid size")
    parser.add_argument("--max-steps", type=int, default=100, help="maximum steps per episode")
    parser.add_argument("--no-render", action="store_true", help="disable ANSI terminal colors")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.agents < 1:
        parser.error("--agents must be at least 1")
    if args.grid_size < 3:
        parser.error("--grid-size must be at least 3")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = AlienGardenEnv(
        grid_size=args.grid_size,
        num_agents=args.agents,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    trainer = MAPPOTrainer(env=env, learning_rate=args.learning_rate, seed=args.seed)
    renderer = TerminalRenderer(enabled=not args.no_render)

    summaries = []
    for episode in range(1, args.episodes + 1):
        summary = trainer.run_episode(episode)
        summaries.append(summary)
        print(renderer.episode(summary))
    print(renderer.final(summaries))
    return 0
