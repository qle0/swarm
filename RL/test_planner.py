#!/usr/bin/env python3
"""
Test script for path planning algorithms.

Usage
-----
$ python test_planner.py --planner rrt [--seed 42] [--gui]
$ python test_planner.py --planner dijkstra [--seed 42] [--gui]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode
from swarm.utils.gui_isolation import run_isolated
from swarm.utils.env_factory import make_env
from swarm.planners.planner_policy import PlannerPolicy


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test path planning algorithms")
parser.add_argument(
    "--planner", type=str, choices=["rrt", "dijkstra"], default="rrt",
    help="Path planning algorithm to use",
)
parser.add_argument(
    "--seed", type=int, default=1,
    help="Random seed for MapTask generation",
)
parser.add_argument(
    "--gui", action="store_true", default=False,
    help="Show the PyBullet GUI during evaluation",
)
args = parser.parse_args()


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
# Create planner policy
# ──────────────────────────────────────────────────────────────────────
print(f"Testing {args.planner} planner...")

# Prepare planner-specific parameters
planner_kwargs = {}
if args.planner == "rrt":
    planner_kwargs["max_iterations"] = 5000
elif args.planner == "dijkstra":
    planner_kwargs["resolution"] = 0.2

planner_policy = PlannerPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    planner_type=args.planner,
    client_id=cli,
    obstacle_ids=obstacle_ids,
    **planner_kwargs
)

# ──────────────────────────────────────────────────────────────────────
# Evaluate the planner
# ──────────────────────────────────────────────────────────────────────
result = _run_episode(task=task, uid=0, model=planner_policy, gui=args.gui)

print("----------------------------------------------------")
print(f"Success: {result.success}")
print(f"Time    : {result.time_sec:.2f} s")
print(f"Energy  : {result.energy:.1f} J")
print(f"Score   : {result.score:.3f}")
print("----------------------------------------------------")

# Close the environment
env.close()