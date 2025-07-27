#!/usr/bin/env python3
"""
Quick offline validator for a single Stable‑Baselines 3 policy or path planner.

If you pass the optional ``--gui`` flag the policy is first evaluated
head‑less (exactly like the on‑chain validator) and the metrics are
printed.  Afterwards **the identical episode is replayed once more**
with a PyBullet GUI.

Usage
-----
$ python test_RL.py --model model/ppo_policy.zip [--seed 42] [--gui]
$ python test_RL.py --model model/a2c_policy.zip [--seed 42] [--gui]
$ python test_RL.py --model model/sac_policy.zip [--seed 42] [--gui]
$ python test_RL.py --planner rrt [--seed 42] [--gui]
$ python test_RL.py --planner dijkstra [--seed 42] [--gui]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO, A2C, SAC

from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode
from swarm.utils.gui_isolation import run_isolated
from swarm.utils.env_factory import make_env
from swarm.planners.planner_policy import PlannerPolicy


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Local Swarm policy validator")
parser.add_argument(
    "--model", type=Path, default=None,
    help="Path to the Stable‑Baselines 3 .zip file",
)
parser.add_argument(
    "--planner", type=str, choices=["rrt", "dijkstra"], default=None,
    help="Path planning algorithm to use instead of an RL policy",
)
parser.add_argument(
    "--seed", type=int, default=1,
    help="Random seed for MapTask generation",
)
parser.add_argument(
    "--gui", action="store_true", default=False,
    help="After evaluation, replay the episode in a PyBullet GUI",
)
args = parser.parse_args()

# Check that either model or planner is specified
if args.model is None and args.planner is None:
    parser.error("Either --model or --planner must be specified")
    
if args.model is not None and args.planner is not None:
    parser.error("Cannot specify both --model and --planner")

if args.model is not None and not args.model.exists():
    raise FileNotFoundError(f"Policy file not found: {args.model}")


# ──────────────────────────────────────────────────────────────────────
# Deterministic MapTask
# ──────────────────────────────────────────────────────────────────────
task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)

# ──────────────────────────────────────────────────────────────────────
# Load model or create planner
# ──────────────────────────────────────────────────────────────────────
if args.model is not None:
    print(f"Evaluating policy at {args.model} …")
    
    # Determine the algorithm from the model path
    model_name = args.model.stem
    if "ppo" in model_name.lower():
        model = PPO.load(args.model, device="cpu")
    elif "a2c" in model_name.lower():
        model = A2C.load(args.model, device="cpu")
    elif "sac" in model_name.lower():
        model = SAC.load(args.model, device="cpu")
    else:
        # Default to PPO
        model = PPO.load(args.model, device="cpu")
else:
    print(f"Using {args.planner} planner...")
    
    # Create environment to get obstacle IDs
    env = make_env(task, gui=args.gui)
    cli = env.getPyBulletClient()
    
    # Get all object IDs in the simulation
    num_objects = cli.getNumBodies()
    obstacle_ids = []
    for i in range(num_objects):
        body_info = cli.getBodyInfo(i)
        body_name = body_info[1].decode('utf-8')
        # Exclude the drone and the goal
        if "drone" not in body_name.lower() and "goal" not in body_name.lower():
            obstacle_ids.append(i)
    
    # Create planner policy
    model = PlannerPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        planner_type=args.planner,
        client_id=cli,
        obstacle_ids=obstacle_ids,
        # Additional planner-specific parameters
        max_iterations=5000 if args.planner == "rrt" else None,
        resolution=0.2 if args.planner == "dijkstra" else None,
    )
    
    # Close the environment (will be recreated in _run_episode)
    env.close()

# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
result = _run_episode(task=task, uid=0, model=model, gui=args.gui)

print("----------------------------------------------------")
print(f"Success: {result.success}")
print(f"Time    : {result.time_sec:.2f} s")
print(f"Energy  : {result.energy:.1f} J")
print(f"Score   : {result.score:.3f}")
print("----------------------------------------------------")