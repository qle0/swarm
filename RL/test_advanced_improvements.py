#!/usr/bin/env python3
"""
Test script for advanced path planning improvements:
1. Adaptive RRT with environment complexity analysis
2. RRT* with asymptotic optimality
3. Parallel planning with multiple algorithms
"""
import sys
import os
sys.path.append('/workspace/swarm')

import numpy as np
import time
from swarm.planners.adaptive_rrt import AdaptiveRRTPlanner, MultiStrategyRRTPlanner
from swarm.planners.rrt_star import RRTStarPlanner


def test_adaptive_rrt():
    """Test adaptive RRT planner."""
    print("=" * 60)
    print("TESTING ADAPTIVE RRT")
    print("=" * 60)
    
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([8.0, 8.0, 2.0])
    
    print("Testing with adaptation enabled...")
    planner_adaptive = AdaptiveRRTPlanner(
        start=start,
        goal=goal,
        base_max_iterations=500,
        adaptation_enabled=True,
        complexity_analysis_samples=30
    )
    
    start_time = time.time()
    path_adaptive = planner_adaptive.plan()
    adaptive_time = time.time() - start_time
    
    print(f"Adaptive RRT results:")
    print(f"  Path found: {len(path_adaptive) > 0}")
    print(f"  Path length: {len(path_adaptive)} waypoints")
    print(f"  Planning time: {adaptive_time:.2f}s")
    
    stats_adaptive = planner_adaptive.get_statistics()
    print(f"  Complexity score: {stats_adaptive.get('complexity_metrics', {}).get('complexity_score', 'N/A')}")
    print(f"  Adapted iterations: {stats_adaptive.get('adapted_params', {}).get('max_iterations', 'N/A')}")
    
    print("\nTesting without adaptation...")
    planner_basic = AdaptiveRRTPlanner(
        start=start,
        goal=goal,
        base_max_iterations=500,
        adaptation_enabled=False
    )
    
    start_time = time.time()
    path_basic = planner_basic.plan()
    basic_time = time.time() - start_time
    
    print(f"Basic RRT results:")
    print(f"  Path found: {len(path_basic) > 0}")
    print(f"  Path length: {len(path_basic)} waypoints")
    print(f"  Planning time: {basic_time:.2f}s")
    
    return path_adaptive, path_basic


def test_rrt_star():
    """Test RRT* planner."""
    print("\n" + "=" * 60)
    print("TESTING RRT*")
    print("=" * 60)
    
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([6.0, 6.0, 2.0])
    
    planner = RRTStarPlanner(
        start=start,
        goal=goal,
        max_iterations=2000,
        step_size=0.5,
        goal_sample_rate=0.1,
        rewire_radius_factor=1.5,
        adaptive_radius=True
    )
    
    start_time = time.time()
    path = planner.plan()
    planning_time = time.time() - start_time
    
    print(f"RRT* results:")
    print(f"  Path found: {len(path) > 0}")
    print(f"  Path length: {len(path)} waypoints")
    print(f"  Planning time: {planning_time:.2f}s")
    
    stats = planner.get_statistics()
    print(f"  Final path cost: {stats.get('final_path_cost', 'N/A'):.2f}")
    print(f"  Rewire count: {stats.get('rewire_count', 'N/A')}")
    print(f"  Cost improvements: {stats.get('cost_improvements', 'N/A')}")
    print(f"  Nodes created: {stats.get('nodes_created', 'N/A')}")
    
    return path


def test_multi_strategy():
    """Test multi-strategy RRT planner."""
    print("\n" + "=" * 60)
    print("TESTING MULTI-STRATEGY PLANNER")
    print("=" * 60)
    
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([7.0, 7.0, 2.0])
    
    planner = MultiStrategyRRTPlanner(
        start=start,
        goal=goal,
        strategies=['adaptive', 'optimized', 'basic'],
        time_budget=8.0,
        parallel_execution=False
    )
    
    start_time = time.time()
    path = planner.plan()
    planning_time = time.time() - start_time
    
    print(f"Multi-strategy results:")
    print(f"  Path found: {len(path) > 0}")
    print(f"  Path length: {len(path)} waypoints")
    print(f"  Total planning time: {planning_time:.2f}s")
    
    stats = planner.get_statistics()
    print(f"  Strategies tried: {stats.get('strategies_tried', 'N/A')}")
    print(f"  Successful strategies: {stats.get('successful_strategies', 'N/A')}")
    
    # Show individual results
    for result in stats.get('results', []):
        success_str = "✓" if result['success'] else "✗"
        print(f"    {result['strategy']}: {success_str} "
              f"(cost: {result['cost']:.2f}, time: {result['time']:.2f}s)")
    
    return path


def calculate_path_cost(path):
    """Calculate total path cost."""
    if len(path) < 2:
        return float('inf')
    
    total_cost = 0.0
    for i in range(1, len(path)):
        total_cost += np.linalg.norm(path[i] - path[i-1])
    
    return total_cost


def main():
    print("TESTING ADVANCED PATH PLANNING IMPROVEMENTS")
    print("=" * 80)
    
    # Test adaptive RRT
    adaptive_path, basic_path = test_adaptive_rrt()
    
    # Test RRT*
    rrt_star_path = test_rrt_star()
    
    # Test multi-strategy planner
    multi_strategy_path = test_multi_strategy()
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    paths = {
        'Adaptive RRT': adaptive_path,
        'Basic RRT': basic_path,
        'RRT*': rrt_star_path,
        'Multi-Strategy': multi_strategy_path
    }
    
    print(f"{'Algorithm':<15} {'Success':<8} {'Waypoints':<10} {'Path Cost':<10}")
    print("-" * 50)
    
    for name, path in paths.items():
        success = "✓" if len(path) > 1 else "✗"
        waypoints = len(path)
        cost = calculate_path_cost(path)
        cost_str = f"{cost:.2f}" if cost != float('inf') else "∞"
        
        print(f"{name:<15} {success:<8} {waypoints:<10} {cost_str:<10}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()