"""
Script to compare the performance of RRT, RRT+Dijkstra, and RRT+A* path planners.
"""
import os
import sys
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm.planners.rrt import RRTPlanner
from swarm.planners.rrt_dijkstra import RRTDijkstraPlanner
from swarm.planners.rrt_astar import RRTAStarPlanner

# Constants
NUM_TESTS = 10
GUI = False  # Disable GUI mode
SEED = 42

# Results storage
results = {
    'rrt': {
        'planning_times': [],
        'path_lengths': [],
        'waypoint_counts': [],
        'success_rate': 0
    },
    'rrt_dijkstra': {
        'planning_times': [],
        'path_lengths': [],
        'waypoint_counts': [],
        'success_rate': 0
    },
    'rrt_astar': {
        'planning_times': [],
        'path_lengths': [],
        'waypoint_counts': [],
        'success_rate': 0
    }
}

def setup_pybullet():
    """Set up PyBullet environment with obstacles."""
    # Connect to PyBullet
    if GUI:
        client_id = p.connect(p.GUI)
    else:
        client_id = p.connect(p.DIRECT)
    
    if GUI:
        # Set up camera
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=client_id
        )
    
    # Create obstacles
    obstacle_ids = []
    
    # Create a complex environment with more obstacles
    
    # Central obstacle
    central_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.5, 0.5, 1.0],
        physicsClientId=client_id
    )
    central_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=central_id,
        basePosition=[0.0, 0.0, 1.0],
        physicsClientId=client_id
    )
    obstacle_ids.append(central_body)
    
    # Create a "wall" with gaps
    for i in range(-3, 4):
        if i != -1 and i != 1:  # Leave gaps at -1 and 1
            wall_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.2, 0.2, 0.8],
                physicsClientId=client_id
            )
            wall_body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_id,
                basePosition=[i * 0.5, 1.5, 0.8],
                physicsClientId=client_id
            )
            obstacle_ids.append(wall_body)
    
    # Create another wall with different gaps
    for i in range(-3, 4):
        if i != 0 and i != 2:  # Leave gaps at 0 and 2
            wall_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.2, 0.2, 0.8],
                physicsClientId=client_id
            )
            wall_body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_id,
                basePosition=[i * 0.5, -1.5, 0.8],
                physicsClientId=client_id
            )
            obstacle_ids.append(wall_body)
    
    # Create some random obstacles
    for i in range(10):
        size_x = np.random.uniform(0.1, 0.4)
        size_y = np.random.uniform(0.1, 0.4)
        size_z = np.random.uniform(0.3, 0.8)
        
        pos_x = np.random.uniform(-2.5, 2.5)
        pos_y = np.random.uniform(-2.5, 2.5)
        pos_z = size_z / 2  # Place on the ground
        
        # Skip if too close to origin (start position)
        if np.sqrt(pos_x**2 + pos_y**2) < 0.8:
            continue
        
        box_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size_x, size_y, size_z],
            physicsClientId=client_id
        )
        box_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_id,
            basePosition=[pos_x, pos_y, pos_z],
            physicsClientId=client_id
        )
        obstacle_ids.append(box_body)
    
    return client_id, obstacle_ids

def run_comparison(test_idx, start_pos, goal_pos, client_id, obstacle_ids):
    """Run a comparison between RRT, RRT+Dijkstra, and RRT+A*."""
    print(f"\n===== Running Comparison {test_idx+1}/{NUM_TESTS} =====")
    print(f"Start: {start_pos}")
    print(f"Goal: {goal_pos}")
    
    # Visualize start and goal
    if client_id is not None:
        # Start point (green)
        p.addUserDebugPoints(
            [start_pos],
            [[0, 1, 0]],
            pointSize=10.0,
            physicsClientId=client_id
        )
        
        # Goal point (red)
        p.addUserDebugPoints(
            [goal_pos],
            [[1, 0, 0]],
            pointSize=10.0,
            physicsClientId=client_id
        )
    
    # Test RRT
    print("\n--- Testing RRT ---")
    rrt_start_time = time.time()
    rrt_planner = RRTPlanner(
        start=start_pos,
        goal=goal_pos,
        client_id=client_id,
        obstacle_ids=obstacle_ids,
        max_iterations=1000,
        step_size=0.2,
        goal_sample_rate=0.1,
        search_radius=1.0
    )
    
    rrt_path = rrt_planner.plan()
    rrt_planning_time = time.time() - rrt_start_time
    
    # Calculate RRT path length
    rrt_path_length = 0
    if len(rrt_path) > 1:
        for i in range(len(rrt_path) - 1):
            rrt_path_length += np.linalg.norm(rrt_path[i+1] - rrt_path[i])
    
    # Visualize RRT path
    if client_id is not None:
        for i in range(len(rrt_path) - 1):
            p.addUserDebugLine(
                rrt_path[i],
                rrt_path[i+1],
                lineColorRGB=(1, 0, 0),  # Red for RRT
                lineWidth=2.0,
                lifeTime=5.0,
                physicsClientId=client_id
            )
    
    # Test RRT+Dijkstra
    print("\n--- Testing RRT+Dijkstra ---")
    rrt_dijkstra_start_time = time.time()
    rrt_dijkstra_planner = RRTDijkstraPlanner(
        start=start_pos,
        goal=goal_pos,
        client_id=client_id,
        obstacle_ids=obstacle_ids,
        max_iterations=1000,
        step_size=0.2,
        goal_sample_rate=0.1,
        search_radius=1.0
    )
    
    rrt_dijkstra_path = rrt_dijkstra_planner.plan()
    rrt_dijkstra_planning_time = time.time() - rrt_dijkstra_start_time
    
    # Calculate RRT+Dijkstra path length
    rrt_dijkstra_path_length = 0
    if len(rrt_dijkstra_path) > 1:
        for i in range(len(rrt_dijkstra_path) - 1):
            rrt_dijkstra_path_length += np.linalg.norm(rrt_dijkstra_path[i+1] - rrt_dijkstra_path[i])
    
    # Visualize RRT+Dijkstra path
    if client_id is not None:
        for i in range(len(rrt_dijkstra_path) - 1):
            p.addUserDebugLine(
                rrt_dijkstra_path[i],
                rrt_dijkstra_path[i+1],
                lineColorRGB=(0, 0, 1),  # Blue for RRT+Dijkstra
                lineWidth=2.0,
                lifeTime=5.0,
                physicsClientId=client_id
            )
    
    # Test RRT+A*
    print("\n--- Testing RRT+A* ---")
    rrt_astar_start_time = time.time()
    rrt_astar_planner = RRTAStarPlanner(
        start=start_pos,
        goal=goal_pos,
        client_id=client_id,
        obstacle_ids=obstacle_ids,
        max_iterations=1000,
        step_size=0.2,
        goal_sample_rate=0.2,  # Increased from 0.1 to 0.2
        search_radius=1.5,     # Increased from 1.0 to 1.5
        smoothing_iterations=10
    )
    
    rrt_astar_path = rrt_astar_planner.plan()
    rrt_astar_planning_time = time.time() - rrt_astar_start_time
    
    # Calculate RRT+A* path length
    rrt_astar_path_length = 0
    if len(rrt_astar_path) > 1:
        for i in range(len(rrt_astar_path) - 1):
            rrt_astar_path_length += np.linalg.norm(rrt_astar_path[i+1] - rrt_astar_path[i])
    
    # Visualize RRT+A* path
    if client_id is not None:
        for i in range(len(rrt_astar_path) - 1):
            p.addUserDebugLine(
                rrt_astar_path[i],
                rrt_astar_path[i+1],
                lineColorRGB=(0, 1, 0),  # Green for RRT+A*
                lineWidth=2.0,
                lifeTime=5.0,
                physicsClientId=client_id
            )
    
    # Print results
    print("\n--- Comparison Results ---")
    print(f"RRT Planning Time: {rrt_planning_time:.4f} s")
    print(f"RRT Path Length: {rrt_path_length:.4f} m")
    print(f"RRT Waypoint Count: {len(rrt_path)}")
    
    print(f"RRT+Dijkstra Planning Time: {rrt_dijkstra_planning_time:.4f} s")
    print(f"RRT+Dijkstra Path Length: {rrt_dijkstra_path_length:.4f} m")
    print(f"RRT+Dijkstra Waypoint Count: {len(rrt_dijkstra_path)}")
    
    print(f"RRT+A* Planning Time: {rrt_astar_planning_time:.4f} s")
    print(f"RRT+A* Path Length: {rrt_astar_path_length:.4f} m")
    print(f"RRT+A* Waypoint Count: {len(rrt_astar_path)}")
    
    # Calculate improvement of RRT+Dijkstra over RRT
    if rrt_planning_time > 0 and rrt_path_length > 0 and len(rrt_path) > 0:
        dijkstra_time_improvement = (rrt_planning_time - rrt_dijkstra_planning_time) / rrt_planning_time * 100
        dijkstra_length_improvement = (rrt_path_length - rrt_dijkstra_path_length) / rrt_path_length * 100
        dijkstra_waypoint_improvement = (len(rrt_path) - len(rrt_dijkstra_path)) / len(rrt_path) * 100
        
        print(f"\nRRT+Dijkstra Improvement:")
        print(f"Planning Time: {dijkstra_time_improvement:.2f}%")
        print(f"Path Length: {dijkstra_length_improvement:.2f}%")
        print(f"Waypoint Count: {dijkstra_waypoint_improvement:.2f}%")
    
    # Calculate improvement of RRT+A* over RRT
    if rrt_planning_time > 0 and rrt_path_length > 0 and len(rrt_path) > 0:
        astar_time_improvement = (rrt_planning_time - rrt_astar_planning_time) / rrt_planning_time * 100
        astar_length_improvement = (rrt_path_length - rrt_astar_path_length) / rrt_path_length * 100
        astar_waypoint_improvement = (len(rrt_path) - len(rrt_astar_path)) / len(rrt_path) * 100
        
        print(f"\nRRT+A* Improvement:")
        print(f"Planning Time: {astar_time_improvement:.2f}%")
        print(f"Path Length: {astar_length_improvement:.2f}%")
        print(f"Waypoint Count: {astar_waypoint_improvement:.2f}%")
    
    # Store results
    results['rrt']['planning_times'].append(rrt_planning_time)
    results['rrt']['path_lengths'].append(rrt_path_length)
    results['rrt']['waypoint_counts'].append(len(rrt_path))
    
    results['rrt_dijkstra']['planning_times'].append(rrt_dijkstra_planning_time)
    results['rrt_dijkstra']['path_lengths'].append(rrt_dijkstra_path_length)
    results['rrt_dijkstra']['waypoint_counts'].append(len(rrt_dijkstra_path))
    
    results['rrt_astar']['planning_times'].append(rrt_astar_planning_time)
    results['rrt_astar']['path_lengths'].append(rrt_astar_path_length)
    results['rrt_astar']['waypoint_counts'].append(len(rrt_astar_path))
    
    # Check success
    if len(rrt_path) > 0:
        results['rrt']['success_rate'] += 1
    if len(rrt_dijkstra_path) > 0:
        results['rrt_dijkstra']['success_rate'] += 1
    if len(rrt_astar_path) > 0:
        results['rrt_astar']['success_rate'] += 1
    
    # Wait for visualization
    if GUI:
        time.sleep(2.0)
    
    return rrt_path, rrt_dijkstra_path, rrt_astar_path

def generate_random_scenario():
    """Generate a random start and goal position."""
    # Start position is always at the origin with some height
    start_pos = np.array([0.0, 0.0, 1.5])
    
    # Generate a random goal position
    x = np.random.uniform(-3.0, 3.0)
    y = np.random.uniform(-3.0, 3.0)
    z = np.random.uniform(0.5, 2.0)
    
    goal_pos = np.array([x, y, z])
    
    # Ensure minimum distance between start and goal
    while np.linalg.norm(goal_pos - start_pos) < 2.0:
        x = np.random.uniform(-3.0, 3.0)
        y = np.random.uniform(-3.0, 3.0)
        z = np.random.uniform(0.5, 2.0)
        goal_pos = np.array([x, y, z])
    
    return start_pos, goal_pos

def plot_results():
    """Plot the comparison results."""
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Success Rate
    axs[0, 0].bar(['RRT', 'RRT+Dijkstra', 'RRT+A*'], 
                 [results['rrt']['success_rate'] / NUM_TESTS * 100, 
                  results['rrt_dijkstra']['success_rate'] / NUM_TESTS * 100,
                  results['rrt_astar']['success_rate'] / NUM_TESTS * 100])
    axs[0, 0].set_ylabel('Success Rate (%)')
    axs[0, 0].set_ylim([0, 100])
    axs[0, 0].set_title('Success Rate')
    
    # Plot 2: Planning Time
    x = np.arange(NUM_TESTS)
    width = 0.25
    
    axs[0, 1].bar(x - width, results['rrt']['planning_times'], width, label='RRT')
    axs[0, 1].bar(x, results['rrt_dijkstra']['planning_times'], width, label='RRT+Dijkstra')
    axs[0, 1].bar(x + width, results['rrt_astar']['planning_times'], width, label='RRT+A*')
    axs[0, 1].set_xlabel('Test #')
    axs[0, 1].set_ylabel('Planning Time (s)')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels([str(i+1) for i in range(NUM_TESTS)])
    axs[0, 1].legend()
    axs[0, 1].set_title('Planning Time')
    
    # Plot 3: Path Length
    axs[1, 0].bar(x - width, results['rrt']['path_lengths'], width, label='RRT')
    axs[1, 0].bar(x, results['rrt_dijkstra']['path_lengths'], width, label='RRT+Dijkstra')
    axs[1, 0].bar(x + width, results['rrt_astar']['path_lengths'], width, label='RRT+A*')
    axs[1, 0].set_xlabel('Test #')
    axs[1, 0].set_ylabel('Path Length (m)')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels([str(i+1) for i in range(NUM_TESTS)])
    axs[1, 0].legend()
    axs[1, 0].set_title('Path Length')
    
    # Plot 4: Waypoint Count
    axs[1, 1].bar(x - width, results['rrt']['waypoint_counts'], width, label='RRT')
    axs[1, 1].bar(x, results['rrt_dijkstra']['waypoint_counts'], width, label='RRT+Dijkstra')
    axs[1, 1].bar(x + width, results['rrt_astar']['waypoint_counts'], width, label='RRT+A*')
    axs[1, 1].set_xlabel('Test #')
    axs[1, 1].set_ylabel('Waypoint Count')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels([str(i+1) for i in range(NUM_TESTS)])
    axs[1, 1].legend()
    axs[1, 1].set_title('Waypoint Count')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'planner_comparison_{timestamp}.png')
    
    # Show the figure if GUI is enabled
    if GUI:
        plt.show()

def main():
    """Main function to run the comparison."""
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    print("====================================================")
    print("RRT vs RRT+DIJKSTRA vs RRT+A* COMPARISON")
    print("====================================================")
    print(f"Number of tests: {NUM_TESTS}")
    print(f"GUI enabled: {GUI}")
    print("====================================================")
    
    # Set up PyBullet environment
    client_id, obstacle_ids = setup_pybullet()
    
    # Run the tests
    for i in range(NUM_TESTS):
        start_pos, goal_pos = generate_random_scenario()
        run_comparison(i, start_pos, goal_pos, client_id, obstacle_ids)
    
    # Calculate average metrics
    avg_rrt_time = np.mean(results['rrt']['planning_times'])
    avg_rrt_length = np.mean(results['rrt']['path_lengths'])
    avg_rrt_waypoints = np.mean(results['rrt']['waypoint_counts'])
    
    avg_rrt_dijkstra_time = np.mean(results['rrt_dijkstra']['planning_times'])
    avg_rrt_dijkstra_length = np.mean(results['rrt_dijkstra']['path_lengths'])
    avg_rrt_dijkstra_waypoints = np.mean(results['rrt_dijkstra']['waypoint_counts'])
    
    avg_rrt_astar_time = np.mean(results['rrt_astar']['planning_times'])
    avg_rrt_astar_length = np.mean(results['rrt_astar']['path_lengths'])
    avg_rrt_astar_waypoints = np.mean(results['rrt_astar']['waypoint_counts'])
    
    # Calculate average improvements for RRT+Dijkstra
    dijkstra_time_improvement = (avg_rrt_time - avg_rrt_dijkstra_time) / avg_rrt_time * 100
    dijkstra_length_improvement = (avg_rrt_length - avg_rrt_dijkstra_length) / avg_rrt_length * 100
    dijkstra_waypoint_improvement = (avg_rrt_waypoints - avg_rrt_dijkstra_waypoints) / avg_rrt_waypoints * 100
    
    # Calculate average improvements for RRT+A*
    astar_time_improvement = (avg_rrt_time - avg_rrt_astar_time) / avg_rrt_time * 100
    astar_length_improvement = (avg_rrt_length - avg_rrt_astar_length) / avg_rrt_length * 100
    astar_waypoint_improvement = (avg_rrt_waypoints - avg_rrt_astar_waypoints) / avg_rrt_waypoints * 100
    
    # Print summary
    print("\n====================================================")
    print("COMPARISON SUMMARY")
    print("====================================================")
    print(f"RRT Success Rate: {results['rrt']['success_rate'] / NUM_TESTS * 100:.1f}%")
    print(f"RRT+Dijkstra Success Rate: {results['rrt_dijkstra']['success_rate'] / NUM_TESTS * 100:.1f}%")
    print(f"RRT+A* Success Rate: {results['rrt_astar']['success_rate'] / NUM_TESTS * 100:.1f}%")
    print("----------------------------------------------------")
    print(f"Average RRT Planning Time: {avg_rrt_time:.4f} s")
    print(f"Average RRT+Dijkstra Planning Time: {avg_rrt_dijkstra_time:.4f} s")
    print(f"Average RRT+A* Planning Time: {avg_rrt_astar_time:.4f} s")
    print("----------------------------------------------------")
    print(f"RRT+Dijkstra Time Improvement: {dijkstra_time_improvement:.2f}%")
    print(f"RRT+A* Time Improvement: {astar_time_improvement:.2f}%")
    print("----------------------------------------------------")
    print(f"Average RRT Path Length: {avg_rrt_length:.4f} m")
    print(f"Average RRT+Dijkstra Path Length: {avg_rrt_dijkstra_length:.4f} m")
    print(f"Average RRT+A* Path Length: {avg_rrt_astar_length:.4f} m")
    print("----------------------------------------------------")
    print(f"RRT+Dijkstra Length Improvement: {dijkstra_length_improvement:.2f}%")
    print(f"RRT+A* Length Improvement: {astar_length_improvement:.2f}%")
    print("----------------------------------------------------")
    print(f"Average RRT Waypoint Count: {avg_rrt_waypoints:.2f}")
    print(f"Average RRT+Dijkstra Waypoint Count: {avg_rrt_dijkstra_waypoints:.2f}")
    print(f"Average RRT+A* Waypoint Count: {avg_rrt_astar_waypoints:.2f}")
    print("----------------------------------------------------")
    print(f"RRT+Dijkstra Waypoint Improvement: {dijkstra_waypoint_improvement:.2f}%")
    print(f"RRT+A* Waypoint Improvement: {astar_waypoint_improvement:.2f}%")
    print("====================================================")
    
    # Disconnect from PyBullet
    p.disconnect(physicsClientId=client_id)
    
    # Plot the results
    plot_results()

if __name__ == "__main__":
    main()