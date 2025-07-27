#!/usr/bin/env python3
"""
Compare different flight control approaches:
1. Pure RL policy
2. Pure RRT path planning
3. Hybrid approach (RRT + RL)

Usage
-----
$ python compare_approaches.py --model model/ppo_policy.zip [--seeds 1,2,3,4,5] [--gui]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import time

from stable_baselines3 import PPO

from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode
from swarm.utils.gui_isolation import run_isolated
from swarm.utils.env_factory import make_env
from swarm.planners.planner_policy import PlannerPolicy
from swarm.planners.hybrid_policy import HybridPolicy


def test_approach(approach_name: str, policy, task, gui: bool = False):
    """Test a single approach and return results."""
    print(f"\nTesting {approach_name}...")
    start_time = time.time()
    
    try:
        result = _run_episode(task=task, uid=0, model=policy, gui=gui)
        test_time = time.time() - start_time
        
        # Get statistics if available
        stats = {}
        if hasattr(policy, 'get_statistics'):
            stats = policy.get_statistics()
        
        return {
            'approach': approach_name,
            'success': result.success,
            'time_sec': result.time_sec,
            'energy': result.energy,
            'score': result.score,
            'test_time': test_time,
            'stats': stats
        }
    except Exception as e:
        print(f"Error testing {approach_name}: {e}")
        return {
            'approach': approach_name,
            'success': False,
            'time_sec': 30.0,
            'energy': 30000.0,
            'score': 0.0,
            'test_time': time.time() - start_time,
            'stats': {},
            'error': str(e)
        }


def main():
    # ──────────────────────────────────────────────────────────────────────
    # CLI
    # ──────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Compare flight control approaches")
    parser.add_argument(
        "--model", type=Path, default=Path("model/ppo_policy.zip"),
        help="Path to the RL policy .zip file",
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3",
        help="Comma-separated list of seeds to test",
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Show the PyBullet GUI during evaluation",
    )
    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Load RL policy if available
    rl_policy = None
    if args.model.exists():
        rl_policy = PPO.load(args.model, device="cpu")
        print(f"Loaded RL policy from {args.model}")
    else:
        print(f"Warning: RL policy file not found: {args.model}")

    # Results storage
    all_results = []
    
    # Test each seed
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"TESTING SEED {seed}")
        print(f"{'='*60}")
        
        # Generate task
        task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=seed)
        
        # Create environment to get obstacle IDs
        env = make_env(task, gui=False)  # Always headless for setup
        cli = env.getPyBulletClient()
        
        # Get obstacle IDs
        import pybullet as p
        num_objects = p.getNumBodies(physicsClientId=cli)
        obstacle_ids = []
        for i in range(num_objects):
            body_info = p.getBodyInfo(i, physicsClientId=cli)
            body_name = body_info[1].decode('utf-8')
            if "drone" not in body_name.lower() and "goal" not in body_name.lower():
                obstacle_ids.append(i)
        
        env.close()
        
        # Test approaches
        approaches_to_test = []
        
        # 1. Pure RL (if available)
        if rl_policy is not None:
            approaches_to_test.append(("Pure RL", rl_policy))
        
        # 2. Pure RRT
        rrt_policy = PlannerPolicy(
            planner_type="rrt",
            observation_space=None,
            action_space=None,
            client_id=None,
            obstacle_ids=obstacle_ids,
            max_iterations=500,
            step_size=0.3,
            goal_sample_rate=0.2,
            search_radius=2.0,
        )
        approaches_to_test.append(("Pure RRT", rrt_policy))
        
        # 3. Hybrid approach (if RL is available)
        if rl_policy is not None:
            hybrid_policy = HybridPolicy(
                observation_space=None,
                action_space=None,
                rl_policy=rl_policy,
                client_id=None,
                obstacle_ids=obstacle_ids,
                planning_horizon=5.0,
                replan_threshold=1.0,
                max_iterations=500,
                step_size=0.3,
                goal_sample_rate=0.2,
                search_radius=2.0,
            )
            approaches_to_test.append(("Hybrid (RRT+RL)", hybrid_policy))
        
        # Test each approach
        for approach_name, policy in approaches_to_test:
            result = test_approach(approach_name, policy, task, args.gui)
            result['seed'] = seed
            all_results.append(result)
    
    # ──────────────────────────────────────────────────────────────────────
    # Analyze and display results
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Group results by approach
    approaches = {}
    for result in all_results:
        approach = result['approach']
        if approach not in approaches:
            approaches[approach] = []
        approaches[approach].append(result)
    
    # Display summary for each approach
    for approach_name, results in approaches.items():
        print(f"\n{approach_name}:")
        print("-" * (len(approach_name) + 1))
        
        successes = [r['success'] for r in results]
        times = [r['time_sec'] for r in results]
        energies = [r['energy'] for r in results]
        scores = [r['score'] for r in results]
        test_times = [r['test_time'] for r in results]
        
        success_rate = np.mean(successes) * 100
        avg_time = np.mean(times)
        avg_energy = np.mean(energies)
        avg_score = np.mean(scores)
        avg_test_time = np.mean(test_times)
        
        print(f"Success rate    : {success_rate:.1f}% ({sum(successes)}/{len(successes)})")
        print(f"Avg time        : {avg_time:.2f} ± {np.std(times):.2f} s")
        print(f"Avg energy      : {avg_energy:.1f} ± {np.std(energies):.1f} J")
        print(f"Avg score       : {avg_score:.3f} ± {np.std(scores):.3f}")
        print(f"Avg test time   : {avg_test_time:.2f} ± {np.std(test_times):.2f} s")
        
        # Display control statistics if available
        if results and 'stats' in results[0] and results[0]['stats']:
            print("Control statistics (last run):")
            stats = results[-1]['stats']
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    # Display detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    print(f"{'Seed':<6} {'Approach':<15} {'Success':<8} {'Time':<8} {'Energy':<10} {'Score':<8}")
    print("-" * 80)
    
    for result in all_results:
        success_str = "✓" if result['success'] else "✗"
        print(f"{result['seed']:<6} {result['approach']:<15} {success_str:<8} "
              f"{result['time_sec']:<8.2f} {result['energy']:<10.1f} {result['score']:<8.3f}")


if __name__ == "__main__":
    main()