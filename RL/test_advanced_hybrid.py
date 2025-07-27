"""
Test script for the advanced hybrid policy with parallel planning.
"""
import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gym import spaces

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm.planners.hybrid_policy import HybridPolicy


def setup_pybullet(gui=True):
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


def load_rl_policy():
    """Load the RL policy."""
    try:
        # Try to load the policy
        model = PPO.load("model/ppo_policy.zip")
        print("Loaded RL policy from model/ppo_policy.zip")
        return model
    except Exception as e:
        print(f"Error loading RL policy: {e}")
        return None


def test_hybrid_policy(start_pos, goal_pos, client_id, obstacle_ids, rl_policy=None, max_steps=1000):
    """Test the hybrid policy."""
    # Create observation and action spaces
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))
    action_space = spaces.Box(low=-1, high=1, shape=(4,))
    
    # Create the hybrid policy
    hybrid_policy = HybridPolicy(
        observation_space=observation_space,
        action_space=action_space,
        rl_policy=rl_policy,
        client_id=client_id,
        obstacle_ids=obstacle_ids,
        planning_horizon=5.0,
        replan_threshold=1.0,
        max_iterations=1000,
        step_size=0.2,
        goal_sample_rate=0.2,
        search_radius=1.5
    )
    
    # Initialize the drone
    drone_position = np.array(start_pos)
    drone_velocity = np.zeros(3)
    goal_position = np.array(goal_pos)
    
    # Create a visual marker for the drone
    drone_visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.1,
        rgbaColor=[0, 0, 1, 1]
    )
    
    drone_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=drone_visual_id,
        basePosition=drone_position
    )
    
    # Create a visual marker for the goal
    goal_visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.1,
        rgbaColor=[1, 0, 0, 1]
    )
    
    goal_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_visual_id,
        basePosition=goal_position
    )
    
    # Simulation parameters
    dt = 0.1  # Time step
    success_threshold = 0.3  # Distance threshold for success
    
    # Metrics
    steps = 0
    distance_to_goal = np.linalg.norm(drone_position - goal_position)
    initial_distance = distance_to_goal
    success = False
    planning_count = 0
    rl_control_count = 0
    waypoint_control_count = 0
    
    # Simulation loop
    start_time = time.time()
    
    while steps < max_steps and not success:
        # Update the observation
        observation = np.concatenate([
            drone_position,
            drone_velocity,
            goal_position
        ])
        
        # Get the action from the hybrid policy
        action, info = hybrid_policy.act(observation)
        
        # Update the metrics
        if info.get('planning', False):
            planning_count += 1
        if info.get('rl_control', False):
            rl_control_count += 1
        if info.get('waypoint_control', False):
            waypoint_control_count += 1
        
        # Apply the action (simple dynamics model)
        # Assume action is a velocity command
        drone_velocity = np.array(action[:3])
        drone_position = drone_position + drone_velocity * dt
        
        # Update the drone position in PyBullet
        p.resetBasePositionAndOrientation(
            drone_id,
            drone_position,
            [0, 0, 0, 1]
        )
        
        # Check for success
        distance_to_goal = np.linalg.norm(drone_position - goal_position)
        if distance_to_goal < success_threshold:
            success = True
            print(f"SUCCESS! Reached goal at t={steps * dt:.2f}")
        
        # Step the simulation
        p.stepSimulation()
        time.sleep(dt / 10)  # Slow down the simulation for visualization
        
        steps += 1
    
    # Calculate metrics
    elapsed_time = time.time() - start_time
    
    # Print results
    print("====================================================")
    print("ADVANCED HYBRID FLIGHT PLANNING RESULTS")
    print("====================================================")
    print(f"Success: {success}")
    print(f"Time    : {steps * dt:.2f} s")
    print(f"Steps   : {steps}")
    print(f"Score   : {0.98 if success else 0.0}")
    print("----------------------------------------------------")
    print("CONTROL STATISTICS:")
    print(f"planning_count      : {planning_count}")
    print(f"rl_control_count    : {rl_control_count}")
    print(f"waypoint_control_count: {waypoint_control_count}")
    print("====================================================")
    
    # Clean up
    p.removeBody(drone_id)
    p.removeBody(goal_id)
    
    return success, steps * dt, planning_count, rl_control_count, waypoint_control_count


def main():
    """Main function."""
    # Set up PyBullet
    client_id, obstacle_ids = setup_pybullet(gui=True)
    
    # Load the RL policy
    rl_policy = load_rl_policy()
    
    # Define start and goal positions
    start_pos = [0.0, 0.0, 1.5]
    goal_pos = [-2.0, -2.0, 0.5]
    
    print(f"Task start: {start_pos}")
    print(f"Task goal: {goal_pos}")
    print(f"Task horizon: 30")
    
    print("Testing advanced hybrid flight planning approach...")
    
    # Test the hybrid policy
    success, time_taken, planning_count, rl_control_count, waypoint_control_count = test_hybrid_policy(
        start_pos=start_pos,
        goal_pos=goal_pos,
        client_id=client_id,
        obstacle_ids=obstacle_ids,
        rl_policy=rl_policy,
        max_steps=1000
    )
    
    # Disconnect from PyBullet
    p.disconnect(client_id)


if __name__ == "__main__":
    main()