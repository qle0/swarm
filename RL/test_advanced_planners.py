#!/usr/bin/env python3
"""
Test script for advanced path planning algorithms with optimizations.

This script tests:
1. Optimized RRT with adaptive sampling
2. A* algorithm 
3. Path smoothing techniques
4. Hybrid approaches

Usage
-----
$ python test_advanced_planners.py [--seed 42] [--gui] [--planner rrt_optimized]
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


def test_planner(planner_name: str, planner_config: dict, task, gui: bool = False):
    """Test a single planner configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {planner_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Create environment to get obstacle IDs
        env = make_env(task, gui=False)
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
        
        # Create planner policy
        if planner_config['type'] == 'hybrid':
            # Load RL policy for hybrid approach
            rl_policy = None
            model_path = Path("model/ppo_policy.zip")
            if model_path.exists():
                rl_policy = PPO.load(model_path, device="cpu")
            
            policy = HybridPolicy(
                observation_space=None,
                action_space=None,
                rl_policy=rl_policy,
                client_id=None,
                obstacle_ids=obstacle_ids,
                **planner_config.get('params', {})
            )
        else:
            policy = PlannerPolicy(
                observation_space=None,
                action_space=None,
                planner_type=planner_config['type'],
                client_id=None,
                obstacle_ids=obstacle_ids,
                **planner_config.get('params', {})
            )
        
        # Run test
        result = _run_episode(task=task, uid=0, model=policy, gui=gui)
        test_time = time.time() - start_time
        
        # Get statistics
        stats = {}
        if hasattr(policy, 'get_statistics'):
            stats = policy.get_statistics()
        elif hasattr(policy, 'planner') and hasattr(policy.planner, 'get_statistics'):
            stats = policy.planner.get_statistics()
        
        print(f"Results:")
        print(f"  Success: {result.success}")
        print(f"  Time: {result.time_sec:.2f} s")
        print(f"  Energy: {result.energy:.1f} J")
        print(f"  Score: {result.score:.3f}")
        print(f"  Test time: {test_time:.2f} s")
        
        if stats:
            print(f"  Statistics:")
            for key, value in stats.items():
                print(f"    {key}: {value}")
        
        return {
            'planner': planner_name,
            'success': result.success,
            'time_sec': result.time_sec,
            'energy': result.energy,
            'score': result.score,
            'test_time': test_time,
            'stats': stats
        }
        
    except Exception as e:
        print(f"Error testing {planner_name}: {e}")
        return {
            'planner': planner_name,
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
    parser = argparse.ArgumentParser(description="Test advanced path planning algorithms")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for MapTask generation",
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Show the PyBullet GUI during evaluation",
    )
    parser.add_argument(
        "--planner", type=str, default="all",
        help="Specific planner to test (or 'all' for all planners)",
    )
    parser.add_argument(
        "--smoothing", action="store_true", default=False,
        help="Test path smoothing techniques",
    )
    args = parser.parse_args()

    # Generate task
    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)
    
    # Define planner configurations
    planner_configs = {
        "RRT (Basic)": {
            'type': 'rrt',
            'params': {
                'max_iterations': 1000,
                'step_size': 0.5,
                'goal_sample_rate': 0.1,
                'search_radius': 1.0,
                'use_path_smoothing': False
            }
        },
        "RRT (Optimized)": {
            'type': 'rrt_optimized',
            'params': {
                'max_iterations': 1000,
                'step_size': 0.5,
                'goal_sample_rate': 0.1,
                'search_radius': 1.0,
                'adaptive_sampling': True,
                'early_termination': True,
                'use_path_smoothing': False
            }
        },
        "A* (Grid-based)": {
            'type': 'astar',
            'params': {
                'grid_resolution': 0.5,
                'heuristic_weight': 1.0,
                'allow_diagonal': True,
                'use_path_smoothing': False
            }
        },
        "Dijkstra (Grid-based)": {
            'type': 'dijkstra',
            'params': {
                'grid_resolution': 0.5,
                'use_path_smoothing': False
            }
        }
    }
    
    # Add smoothed versions if requested
    if args.smoothing:
        smoothing_methods = ['spline', 'shortcut', 'bezier']
        for method in smoothing_methods:
            planner_configs[f"RRT + {method.title()} Smoothing"] = {
                'type': 'rrt_optimized',
                'params': {
                    'max_iterations': 500,
                    'step_size': 0.5,
                    'goal_sample_rate': 0.2,
                    'search_radius': 1.0,
                    'adaptive_sampling': True,
                    'early_termination': True,
                    'use_path_smoothing': True,
                    'smoothing_method': method
                }
            }
            
            planner_configs[f"A* + {method.title()} Smoothing"] = {
                'type': 'astar',
                'params': {
                    'grid_resolution': 0.5,
                    'heuristic_weight': 1.0,
                    'allow_diagonal': True,
                    'use_path_smoothing': True,
                    'smoothing_method': method
                }
            }
    
    # Add hybrid approach if RL model exists
    model_path = Path("model/ppo_policy.zip")
    if model_path.exists():
        planner_configs["Hybrid (RRT + RL)"] = {
            'type': 'hybrid',
            'params': {
                'planning_horizon': 5.0,
                'replan_threshold': 1.0,
                'max_iterations': 500,
                'step_size': 0.3,
                'goal_sample_rate': 0.2,
                'search_radius': 2.0,
            }
        }
    
    # Filter planners if specific one requested
    if args.planner != "all":
        filtered_configs = {}
        for name, config in planner_configs.items():
            if args.planner.lower() in name.lower():
                filtered_configs[name] = config
        planner_configs = filtered_configs
        
        if not planner_configs:
            print(f"No planners found matching '{args.planner}'")
            print("Available planners:")
            for name in planner_configs.keys():
                print(f"  - {name}")
            return
    
    # Test all planners
    results = []
    for planner_name, config in planner_configs.items():
        result = test_planner(planner_name, config, task, args.gui)
        results.append(result)
    
    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    
    # Sort by success rate, then by score
    results.sort(key=lambda x: (x['success'], x['score']), reverse=True)
    
    print(f"{'Planner':<25} {'Success':<8} {'Time':<8} {'Energy':<10} {'Score':<8} {'Test Time':<10}")
    print("-" * 80)
    
    for result in results:
        success_str = "✓" if result['success'] else "✗"
        print(f"{result['planner']:<25} {success_str:<8} "
              f"{result['time_sec']:<8.2f} {result['energy']:<10.1f} "
              f"{result['score']:<8.3f} {result['test_time']:<10.2f}")
    
    # Performance analysis
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\nBest performing planners:")
        for i, result in enumerate(successful_results[:3]):
            print(f"{i+1}. {result['planner']} (Score: {result['score']:.3f})")
    
    print(f"\nFastest planners (by test time):")
    fastest_results = sorted(results, key=lambda x: x['test_time'])
    for i, result in enumerate(fastest_results[:3]):
        print(f"{i+1}. {result['planner']} ({result['test_time']:.2f}s)")


if __name__ == "__main__":
    main()