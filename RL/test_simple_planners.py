#!/usr/bin/env python3
"""
Simple test for path planning algorithms.
"""
import sys
import os
sys.path.append('/workspace/swarm')

import numpy as np
from swarm.planners.rrt_optimized import OptimizedRRTPlanner
from swarm.planners.astar import AStarPlanner
from swarm.planners.path_smoother import PathSmoother


def test_rrt_optimized():
    """Test optimized RRT planner."""
    print("Testing Optimized RRT...")
    
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([5.0, 5.0, 2.0])
    
    planner = OptimizedRRTPlanner(
        start=start,
        goal=goal,
        max_iterations=500,
        step_size=0.5,
        goal_sample_rate=0.2,
        search_radius=1.0,
        adaptive_sampling=True,
        early_termination=True
    )
    
    path = planner.plan()
    stats = planner.get_statistics()
    
    print(f"  Path found: {len(path) > 0}")
    print(f"  Path length: {len(path)} waypoints")
    print(f"  Statistics: {stats}")
    
    return path


def test_astar():
    """Test A* planner."""
    print("Testing A*...")
    
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([5.0, 5.0, 2.0])
    
    planner = AStarPlanner(
        start=start,
        goal=goal,
        grid_resolution=0.5,
        heuristic_weight=1.0,
        allow_diagonal=True
    )
    
    path = planner.plan()
    stats = planner.get_statistics()
    
    print(f"  Path found: {len(path) > 0}")
    print(f"  Path length: {len(path)} waypoints")
    print(f"  Statistics: {stats}")
    
    return path


def test_path_smoothing(path):
    """Test path smoothing."""
    if len(path) < 3:
        print("Path too short for smoothing")
        return path
    
    print("Testing Path Smoothing...")
    
    smoother = PathSmoother()
    
    # Test different smoothing methods
    methods = ['spline', 'shortcut', 'bezier']
    
    for method in methods:
        try:
            smoothed_path = smoother.smooth_path(path, method=method)
            print(f"  {method.title()} smoothing: {len(path)} -> {len(smoothed_path)} waypoints")
        except Exception as e:
            print(f"  {method.title()} smoothing failed: {e}")
    
    return smoother.smooth_path(path, method='spline')


def main():
    print("=" * 60)
    print("TESTING ADVANCED PATH PLANNING ALGORITHMS")
    print("=" * 60)
    
    # Test RRT Optimized
    rrt_path = test_rrt_optimized()
    print()
    
    # Test A*
    astar_path = test_astar()
    print()
    
    # Test path smoothing on RRT path
    if len(rrt_path) > 2:
        smoothed_path = test_path_smoothing(rrt_path)
        print()
    
    print("=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()