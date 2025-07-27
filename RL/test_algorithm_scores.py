#!/usr/bin/env python3
"""
Test script to evaluate and compare scores of different path planning algorithms.
"""
import sys
import os
sys.path.append('/workspace/swarm')

import numpy as np
import time
from typing import Dict, List, Tuple
import argparse

from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode
from swarm.utils.env_factory import make_env
from swarm.planners.planner_policy import PlannerPolicy
from swarm.constants import SIM_DT, HORIZON_SEC
from stable_baselines3 import PPO


def test_algorithm_score(algorithm: str, num_tests: int = 5, seeds: List[int] = None) -> Dict:
    """
    Test an algorithm's performance across multiple episodes.
    
    Parameters
    ----------
    algorithm : str
        Algorithm name to test
    num_tests : int
        Number of test episodes
    seeds : List[int]
        Random seeds for reproducible testing
        
    Returns
    -------
    Dict
        Performance statistics
    """
    if seeds is None:
        seeds = list(range(1, num_tests + 1))
    
    results = {
        'algorithm': algorithm,
        'scores': [],
        'times': [],
        'energies': [],
        'successes': [],
        'episodes': []
    }
    
    print(f"\n{'='*60}")
    print(f"TESTING {algorithm.upper()}")
    print(f"{'='*60}")
    
    for i, seed in enumerate(seeds[:num_tests]):
        print(f"\nTest {i+1}/{num_tests} (seed={seed})")
        
        try:
            # Generate task
            task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=seed)
            
            # Create policy based on algorithm type
            if algorithm == "ppo_rl":
                # Test trained RL model
                if os.path.exists("model/ppo_policy.zip"):
                    model = PPO.load("model/ppo_policy.zip")
                else:
                    print(f"  ❌ PPO model not found, skipping")
                    continue
            else:
                # Test path planning algorithm
                env = make_env(task, gui=False)
                
                # Algorithm-specific parameters
                kwargs = {}
                if algorithm == "rrt":
                    kwargs = {"max_iterations": 3000}
                elif algorithm == "rrt_optimized":
                    kwargs = {"max_iterations": 2000, "adaptive_sampling": True, "early_termination": True}
                elif algorithm == "adaptive_rrt":
                    kwargs = {"base_max_iterations": 1500, "adaptation_enabled": True}
                elif algorithm in ["dijkstra", "astar"]:
                    kwargs = {"grid_resolution": 0.3}
                
                model = PlannerPolicy(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    planner_type=algorithm,
                    client_id=None,
                    obstacle_ids=[],
                    **kwargs
                )
                env.close()
            
            # Run episode
            start_time = time.time()
            episode_result = _run_episode(task, uid=0, model=model, gui=False)
            test_time = time.time() - start_time
            
            # Extract results
            success = episode_result.success
            score = episode_result.score
            energy = episode_result.energy
            
            results['scores'].append(score)
            results['times'].append(test_time)
            results['energies'].append(energy)
            results['successes'].append(success)
            results['episodes'].append(episode_result)
            
            # Print episode result
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {status} | Score: {score:.3f} | Time: {test_time:.2f}s | Energy: {energy:.1f}J")
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            results['scores'].append(0.0)
            results['times'].append(0.0)
            results['energies'].append(0.0)
            results['successes'].append(False)
            results['episodes'].append({'error': str(e)})
    
    # Calculate statistics
    scores = [s for s in results['scores'] if s > 0]
    times = [t for t in results['times'] if t > 0]
    energies = [e for e in results['energies'] if e > 0]
    
    results['stats'] = {
        'success_rate': sum(results['successes']) / len(results['successes']) if results['successes'] else 0,
        'avg_score': np.mean(scores) if scores else 0,
        'max_score': np.max(scores) if scores else 0,
        'min_score': np.min(scores) if scores else 0,
        'std_score': np.std(scores) if scores else 0,
        'avg_time': np.mean(times) if times else 0,
        'avg_energy': np.mean(energies) if energies else 0,
        'total_tests': len(results['successes'])
    }
    
    return results


def print_algorithm_summary(results: Dict):
    """Print summary statistics for an algorithm."""
    stats = results['stats']
    algorithm = results['algorithm']
    
    print(f"\n📊 {algorithm.upper()} SUMMARY:")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Average Score: {stats['avg_score']:.3f}")
    print(f"  Score Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
    print(f"  Score Std Dev: {stats['std_score']:.3f}")
    print(f"  Average Time: {stats['avg_time']:.2f}s")
    print(f"  Average Energy: {stats['avg_energy']:.1f}J")


def compare_algorithms(results_list: List[Dict]):
    """Compare multiple algorithms and print comparison table."""
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*80}")
    
    # Sort by average score (descending)
    results_list.sort(key=lambda x: x['stats']['avg_score'], reverse=True)
    
    # Print header
    print(f"{'Algorithm':<15} {'Success%':<9} {'Avg Score':<10} {'Max Score':<10} {'Avg Time':<10} {'Avg Energy':<12}")
    print("-" * 80)
    
    # Print results
    for results in results_list:
        stats = results['stats']
        algorithm = results['algorithm']
        
        print(f"{algorithm:<15} "
              f"{stats['success_rate']:.1%}    "
              f"{stats['avg_score']:<10.3f} "
              f"{stats['max_score']:<10.3f} "
              f"{stats['avg_time']:<10.2f} "
              f"{stats['avg_energy']:<12.1f}")
    
    # Find best performers
    if results_list:
        best_score = results_list[0]
        best_success = max(results_list, key=lambda x: x['stats']['success_rate'])
        best_time = min(results_list, key=lambda x: x['stats']['avg_time'] if x['stats']['avg_time'] > 0 else float('inf'))
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"  Highest Score: {best_score['algorithm']} ({best_score['stats']['avg_score']:.3f})")
        print(f"  Best Success Rate: {best_success['algorithm']} ({best_success['stats']['success_rate']:.1%})")
        print(f"  Fastest: {best_time['algorithm']} ({best_time['stats']['avg_time']:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Test and compare path planning algorithm scores")
    parser.add_argument("--algorithms", nargs="+", 
                       choices=["ppo_rl", "rrt", "rrt_optimized", "adaptive_rrt", "dijkstra", "astar"],
                       default=["rrt", "rrt_optimized", "astar"],
                       help="Algorithms to test")
    parser.add_argument("--num-tests", type=int, default=5,
                       help="Number of test episodes per algorithm")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                       help="Specific seeds to use for testing")
    
    args = parser.parse_args()
    
    print("ALGORITHM SCORE TESTING")
    print(f"Testing algorithms: {', '.join(args.algorithms)}")
    print(f"Number of tests per algorithm: {args.num_tests}")
    
    # Test each algorithm
    all_results = []
    
    for algorithm in args.algorithms:
        try:
            results = test_algorithm_score(algorithm, args.num_tests, args.seeds)
            print_algorithm_summary(results)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Failed to test {algorithm}: {str(e)}")
    
    # Compare all algorithms
    if len(all_results) > 1:
        compare_algorithms(all_results)
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()