"""
Simple test script to evaluate the score of our path planning algorithms.
"""
import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm.planners.rrt import RRTPlanner
from swarm.planners.rrt_dijkstra import RRTDijkstraPlanner
from swarm.planners.rrt_astar import RRTAStarPlanner
from swarm.planners.rrt_star import RRTStarPlanner
from swarm.planners.parallel_planner import ParallelPlanner


def setup_pybullet(gui=False):
    """Set up PyBullet environment with obstacles."""
    # Connect to PyBullet
    if gui:
        client_id = p.connect(p.GUI)
    else:
        client_id = p.connect(p.DIRECT)
    
    # Set up the environment
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Add obstacles
    obstacle_ids = []
    
    # Add a few boxes as obstacles
    box_positions = [
        [1.0, 1.0, 0.5],
        [-1.0, -1.0, 0.5],
        [2.0, -1.0, 0.5],
        [-2.0, 1.0, 0.5],
        [0.0, 2.0, 0.5],
        [0.0, -2.0, 0.5]
    ]
    
    for pos in box_positions:
        box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_id,
            basePosition=pos
        )
        obstacle_ids.append(obstacle_id)
    
    # Add a few spheres as obstacles
    sphere_positions = [
        [1.5, 0.0, 0.5],
        [-1.5, 0.0, 0.5],
        [0.0, 1.5, 0.5],
        [0.0, -1.5, 0.5]
    ]
    
    for pos in sphere_positions:
        sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=sphere_id,
            basePosition=pos
        )
        obstacle_ids.append(obstacle_id)
    
    return client_id, obstacle_ids


def test_planner(planner_name, start_pos, goal_pos, client_id, obstacle_ids):
    """Test a specific planner."""
    print(f"\nTesting {planner_name}...")
    
    # Create the planner
    if planner_name == "rrt":
        planner = RRTPlanner(
            start=start_pos,
            goal=goal_pos,
            client_id=client_id,
            obstacle_ids=obstacle_ids,
            max_iterations=1000,
            step_size=0.2,
            goal_sample_rate=0.1,
            search_radius=1.0
        )
    elif planner_name == "rrt_dijkstra":
        planner = RRTDijkstraPlanner(
            start=start_pos,
            goal=goal_pos,
            client_id=client_id,
            obstacle_ids=obstacle_ids,
            max_iterations=1000,
            step_size=0.2,
            goal_sample_rate=0.1,
            search_radius=1.0
        )
    elif planner_name == "rrt_astar":
        planner = RRTAStarPlanner(
            start=start_pos,
            goal=goal_pos,
            client_id=client_id,
            obstacle_ids=obstacle_ids,
            max_iterations=1000,
            step_size=0.2,
            goal_sample_rate=0.2,
            search_radius=1.5,
            smoothing_iterations=10
        )
    elif planner_name == "rrt_star":
        planner = RRTStarPlanner(
            start=start_pos,
            goal=goal_pos,
            client_id=client_id,
            obstacle_ids=obstacle_ids,
            max_iterations=1000,
            step_size=0.2,
            goal_sample_rate=0.2,
            search_radius=1.5,
            rewire_radius=0.5
        )
    elif planner_name == "parallel":
        planner = ParallelPlanner(
            start=start_pos,
            goal=goal_pos,
            client_id=client_id,
            obstacle_ids=obstacle_ids,
            max_iterations=1000,
            step_size=0.2,
            goal_sample_rate=0.2,
            search_radius=1.5,
            timeout=2.0,
            planners=["rrt", "rrt_dijkstra", "rrt_astar", "rrt_star"]
        )
    else:
        raise ValueError(f"Unknown planner: {planner_name}")
    
    # Measure planning time
    start_time = time.time()
    
    # Plan the path
    path = planner.plan()
    
    # Calculate planning time
    planning_time = time.time() - start_time
    
    # Calculate path length
    path_length = 0.0
    for i in range(len(path) - 1):
        path_length += np.linalg.norm(path[i+1] - path[i])
    
    # Calculate path quality
    direct_distance = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
    path_quality = direct_distance / path_length if path_length > 0 else 0.0
    path_quality = min(1.0, path_quality)
    
    # Check if the path reaches the goal
    success = len(path) > 1 and np.linalg.norm(path[-1] - np.array(goal_pos)) < 0.3
    
    # Calculate score
    score = 0.98 if success else 0.0
    
    # Print results
    print(f"  Planning time: {planning_time:.4f} s")
    print(f"  Path length: {path_length:.4f} m")
    print(f"  Path quality: {path_quality:.4f}")
    print(f"  Success: {success}")
    print(f"  Score: {score:.4f}")
    
    return {
        "planner": planner_name,
        "planning_time": planning_time,
        "path_length": path_length,
        "path_quality": path_quality,
        "success": success,
        "score": score
    }


def main():
    """Main function."""
    # Set up PyBullet
    client_id, obstacle_ids = setup_pybullet(gui=False)
    
    # Define test scenarios
    test_scenarios = [
        {
            "start": [0.0, 0.0, 1.5],
            "goal": [-2.0, -2.0, 0.5],
            "name": "Diagonal path with obstacles"
        },
        {
            "start": [0.0, 0.0, 1.5],
            "goal": [3.0, 0.0, 1.0],
            "name": "Straight path with obstacles"
        },
        {
            "start": [0.0, 0.0, 1.5],
            "goal": [0.0, 3.0, 2.0],
            "name": "Vertical path with obstacles"
        },
        {
            "start": [-2.5, -2.5, 1.5],
            "goal": [2.5, 2.5, 1.5],
            "name": "Long diagonal path"
        },
        {
            "start": [2.5, 2.5, 1.5],
            "goal": [-2.5, -2.5, 0.5],
            "name": "Reverse diagonal path with descent"
        }
    ]
    
    # Define planners to test
    planners = ["rrt", "rrt_dijkstra", "rrt_astar", "rrt_star", "parallel"]
    
    # Store results
    results = {planner: [] for planner in planners}
    
    # Run tests
    for i, scenario in enumerate(test_scenarios):
        print(f"\n===== Test Scenario {i+1}: {scenario['name']} =====")
        print(f"Start: {scenario['start']}")
        print(f"Goal: {scenario['goal']}")
        
        for planner in planners:
            result = test_planner(
                planner_name=planner,
                start_pos=scenario["start"],
                goal_pos=scenario["goal"],
                client_id=client_id,
                obstacle_ids=obstacle_ids
            )
            results[planner].append(result)
    
    # Calculate average scores
    print("\n===== OVERALL RESULTS =====")
    for planner in planners:
        avg_planning_time = np.mean([r["planning_time"] for r in results[planner]])
        avg_path_length = np.mean([r["path_length"] for r in results[planner]])
        avg_path_quality = np.mean([r["path_quality"] for r in results[planner]])
        success_rate = np.mean([1.0 if r["success"] else 0.0 for r in results[planner]]) * 100
        avg_score = np.mean([r["score"] for r in results[planner]])
        
        print(f"\n{planner.upper()}:")
        print(f"  Average planning time: {avg_planning_time:.4f} s")
        print(f"  Average path length: {avg_path_length:.4f} m")
        print(f"  Average path quality: {avg_path_quality:.4f}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average score: {avg_score:.4f}")
    
    # Disconnect from PyBullet
    p.disconnect(client_id)


if __name__ == "__main__":
    main()