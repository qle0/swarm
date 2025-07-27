#!/usr/bin/env python3
"""
Test the hybrid approach for drone flight control.
"""
from __future__ import annotations

import argparse
import time
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO

from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode
from swarm.planners.hybrid_policy import HybridPolicy


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test hybrid flight planning approach")
    parser.add_argument(
        "--model", type=Path, default=Path("model/ppo_policy.zip"),
        help="Path to the RL policy .zip file",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Show the PyBullet GUI during evaluation",
    )
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    
    # Load RL policy
    rl_policy = None
    if args.model.exists():
        rl_policy = PPO.load(args.model, device="cpu")
        print(f"Loaded RL policy from {args.model}")
    else:
        print(f"Warning: RL policy file not found: {args.model}")
    
    # Generate task
    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)
    
    # Create hybrid policy
    hybrid_policy = HybridPolicy(
        observation_space=None,
        action_space=None,
        rl_policy=rl_policy,
        client_id=None,
        obstacle_ids=[],  # Will be set during environment creation
        planning_horizon=5.0,
        replan_threshold=1.0,
        max_iterations=500,
        step_size=0.3,
        goal_sample_rate=0.2,
        search_radius=2.0,
    )
    
    # Print task information
    print(f"Task start: {task.start}")
    print(f"Task goal: {task.goal}")
    print(f"Task horizon: {task.horizon}")
    
    # Run episode with debug
    print("Testing hybrid flight planning approach...")
    start_time = time.time()
    
    # Create a wrapper to fix the goal in the observation
    original_predict = hybrid_policy.predict
    
    def fixed_predict(observation, *args, **kwargs):
        # Исправляем формат наблюдения - преобразуем из 2D в 1D
        if len(observation.shape) > 1:
            # Если наблюдение двумерное, берем первую строку
            obs_flat = observation[0].copy()
        else:
            obs_flat = observation.copy()
            
        # Заменяем цель в наблюдении на реальную цель
        # Цель находится в индексах 6-9
        obs_flat[6:9] = task.goal
        
        # Выводим отладочную информацию каждые 100 шагов
        if hybrid_policy.waypoint_control_count % 100 == 0:
            print(f"FIXED OBSERVATION:")
            print(f"  Position: {obs_flat[0:3]}")
            print(f"  Velocity: {obs_flat[3:6]}")
            print(f"  Goal: {obs_flat[6:9]}")
            print(f"  Distance to goal: {np.linalg.norm(obs_flat[0:3] - obs_flat[6:9]):.2f}")
        
        # Вызываем оригинальный метод predict с исправленным наблюдением
        return original_predict(obs_flat, *args, **kwargs)
    
    hybrid_policy.predict = fixed_predict
    
    result = _run_episode(task=task, uid=0, model=hybrid_policy, gui=args.gui)
    test_time = time.time() - start_time
    
    # Get statistics
    stats = {}
    if hasattr(hybrid_policy, 'get_statistics'):
        stats = hybrid_policy.get_statistics()
    
    # Print results
    print("=" * 52)
    print("HYBRID FLIGHT PLANNING RESULTS")
    print("=" * 52)
    print(f"Success: {result.success}")
    print(f"Time    : {result.time_sec:.2f} s")
    print(f"Energy  : {result.energy:.1f} J")
    print(f"Score   : {result.score:.3f}")
    print("-" * 52)
    print("CONTROL STATISTICS:")
    if stats:
        for key, value in stats.items():
            print(f"{key:<20}: {value}")
    print("=" * 52)


if __name__ == "__main__":
    main()