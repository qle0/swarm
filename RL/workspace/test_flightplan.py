#!/usr/bin/env python3
"""
Test script for hybrid flight planning approach combining RL and path planning.

This script demonstrates a hybrid approach that uses:
1. RRT for global path planning
2. RL policy for local control and obstacle avoidance

Usage
-----
$ python test_flightplan.py --model model/ppo_policy.zip [--seed 42] [--gui]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO

from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode
from swarm.utils.gui_isolation import run_isolated
from swarm.utils.env_factory import make_env
from swarm.planners.hybrid_policy import HybridPolicy


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test hybrid flight planning approach")
parser.add_argument(
    "--model", type=Path, default=Path("model/ppo_policy.zip"),
    help="Path to the RL policy .zip file",
)
parser.add_argument(
    "--seed", type=int, default=1,
    help="Random seed for MapTask generation",
)
parser.add_argument(
    "--gui", action="store_true", default=False,
    help="Show the PyBullet GUI during evaluation",
)
parser.add_argument(
    "--planning_horizon", type=float, default=5.0,
    help="Planning horizon for RRT in seconds",
)
parser.add_argument(
    "--replan_threshold", type=float, default=1.0,
    help="Distance threshold for replanning",
)
args = parser.parse_args()

if not args.model.exists():
    print(f"Warning: RL policy file not found: {args.model}")
    print("Using direct control fallback")
    rl_policy = None
else:
    rl_policy = PPO.load(args.model, device="cpu")


# ──────────────────────────────────────────────────────────────────────
# Deterministic MapTask
# ──────────────────────────────────────────────────────────────────────
task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)

# ──────────────────────────────────────────────────────────────────────
# Create environment and get obstacle IDs
# ──────────────────────────────────────────────────────────────────────
env = make_env(task, gui=args.gui)
cli = env.getPyBulletClient()

# Get all object IDs in the simulation
import pybullet as p
num_objects = p.getNumBodies(physicsClientId=cli)
obstacle_ids = []
for i in range(num_objects):
    body_info = p.getBodyInfo(i, physicsClientId=cli)
    body_name = body_info[1].decode('utf-8')
    # Exclude the drone and the goal
    if "drone" not in body_name.lower() and "goal" not in body_name.lower():
        obstacle_ids.append(i)

# ──────────────────────────────────────────────────────────────────────
# Create hybrid policy
# ──────────────────────────────────────────────────────────────────────
print("Testing hybrid flight planning approach...")
hybrid_policy = HybridPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rl_policy=rl_policy,
    client_id=cli,
    obstacle_ids=obstacle_ids,
    planning_horizon=args.planning_horizon,
    replan_threshold=args.replan_threshold,
    max_iterations=1000,
    step_size=0.3,
    goal_sample_rate=0.2,
    search_radius=2.0,
)

# ──────────────────────────────────────────────────────────────────────
# Evaluate the hybrid policy
# ──────────────────────────────────────────────────────────────────────
result = _run_episode(task=task, uid=0, model=hybrid_policy, gui=args.gui)

# Get statistics from the hybrid policy
stats = hybrid_policy.get_statistics()

print("====================================================")
print("HYBRID FLIGHT PLANNING RESULTS")
print("====================================================")
print(f"Success: {result.success}")
print(f"Time    : {result.time_sec:.2f} s")
print(f"Energy  : {result.energy:.1f} J")
print(f"Score   : {result.score:.3f}")
print("----------------------------------------------------")
print("CONTROL STATISTICS:")
print(f"Planning count      : {stats['planning_count']}")
print(f"RL control count    : {stats['rl_control_count']}")
print(f"Waypoint control    : {stats['waypoint_control_count']}")
total_actions = stats['rl_control_count'] + stats['waypoint_control_count']
if total_actions > 0:
    rl_percentage = (stats['rl_control_count'] / total_actions) * 100
    waypoint_percentage = (stats['waypoint_control_count'] / total_actions) * 100
    print(f"RL control usage    : {rl_percentage:.1f}%")
    print(f"Waypoint usage      : {waypoint_percentage:.1f}%")
print("====================================================")

# Close the environment
env.close()