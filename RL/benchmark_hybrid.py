"""
Benchmark script for testing the hybrid path planning approach.
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

from swarm.planners.hybrid_policy import HybridPolicy
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import DroneModel

# Constants
NUM_TESTS = 10
SIM_FREQ = 50
DURATION_SEC = 30
GUI = True
RECORD_VIDEO = False
OBSTACLES = True
SEED = 42

# Results storage
results = {
    'success_rate': 0,
    'avg_time': 0,
    'avg_energy': 0,
    'avg_score': 0,
    'times': [],
    'energies': [],
    'scores': [],
    'planning_counts': [],
    'waypoint_counts': [],
    'rl_counts': [],
    'path_lengths': []
}

def run_test(test_idx, start_pos, goal_pos):
    """Run a single test with the hybrid policy."""
    print(f"\n===== Running Test {test_idx+1}/{NUM_TESTS} =====")
    print(f"Start: {start_pos}")
    print(f"Goal: {goal_pos}")
    
    # Create the environment
    env = HoverAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([start_pos]),
        initial_rpys=np.array([[0, 0, 0]]),
        physics=True,
        freq=SIM_FREQ,
        gui=GUI,
        record=RECORD_VIDEO,
        obstacles=OBSTACLES,
        user_debug_gui=False
    )
    
    # Set the goal position
    env.goal = np.array(goal_pos)
    
    # Create the hybrid policy
    policy = HybridPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        client_id=env.CLIENT,
        obstacle_ids=env.obstacles_ids if hasattr(env, 'obstacles_ids') else None,
        planning_horizon=5.0,
        replan_threshold=1.0,
        max_iterations=1000,
        step_size=0.2,
        goal_sample_rate=0.1,
        search_radius=1.0,
        max_speed=5.0,
        waypoint_threshold=0.25
    )
    
    # Reset the environment
    obs = env.reset()
    
    # Extract the start and goal positions
    start_position = obs[0, :3]
    goal_position = obs[0, 6:9]
    
    print(f"Environment start: {start_position}")
    print(f"Environment goal: {goal_position}")
    
    # Run the episode
    done = False
    success = False
    episode_reward = 0
    step_counter = 0
    energy = 0
    start_time = time.time()
    
    while not done:
        # Get the action from the policy
        action, _ = policy.predict(obs)
        
        # Step the environment
        obs, reward, done, info = env.step(action)
        
        # Calculate energy (simplified)
        energy += np.linalg.norm(action[0, :3]) * (1.0 / SIM_FREQ)
        
        # Accumulate reward
        episode_reward += reward
        
        # Check if we've reached the goal
        current_position = obs[0, :3]
        distance_to_goal = np.linalg.norm(current_position - goal_position)
        
        if distance_to_goal < 0.2:  # Success threshold
            success = True
            print(f"SUCCESS! Reached goal at t={time.time() - start_time:.2f}")
            break
            
        # Check for timeout
        step_counter += 1
        if step_counter >= DURATION_SEC * SIM_FREQ:
            print(f"TIMEOUT! Failed to reach goal after {DURATION_SEC} seconds")
            break
            
        # Sleep to maintain real-time simulation if GUI is enabled
        if GUI:
            sync(i=step_counter, start_time=start_time, elapsed_time=step_counter*env.TIMESTEP)
    
    # Close the environment
    env.close()
    
    # Calculate metrics
    elapsed_time = time.time() - start_time
    
    # Calculate score (higher is better)
    # Score formula: success_weight * success + time_weight * (1 - time/max_time) + energy_weight * (1 - energy/max_energy)
    success_weight = 0.6
    time_weight = 0.3
    energy_weight = 0.1
    
    max_time = DURATION_SEC
    max_energy = 10.0  # Arbitrary maximum energy
    
    time_score = 1.0 - min(elapsed_time / max_time, 1.0)
    energy_score = 1.0 - min(energy / max_energy, 1.0)
    
    score = success_weight * float(success) + time_weight * time_score + energy_weight * energy_score
    
    # Get statistics from the policy
    stats = policy.get_statistics()
    
    # Calculate path length
    path_length = 0
    if hasattr(policy, 'global_path') and len(policy.global_path) > 1:
        for i in range(len(policy.global_path) - 1):
            path_length += np.linalg.norm(policy.global_path[i+1] - policy.global_path[i])
    
    # Print results
    print("====================================================")
    print(f"TEST {test_idx+1} RESULTS")
    print("====================================================")
    print(f"Success: {success}")
    print(f"Time    : {elapsed_time:.2f} s")
    print(f"Energy  : {energy:.2f} J")
    print(f"Score   : {score:.3f}")
    print("----------------------------------------------------")
    print("CONTROL STATISTICS:")
    print(f"planning_count      : {stats['planning_count']}")
    print(f"rl_control_count    : {stats['rl_control_count']}")
    print(f"waypoint_control_count: {stats['waypoint_control_count']}")
    print(f"path_length         : {path_length:.2f} m")
    print("====================================================")
    
    # Store results
    results['times'].append(elapsed_time)
    results['energies'].append(energy)
    results['scores'].append(score)
    results['planning_counts'].append(stats['planning_count'])
    results['waypoint_counts'].append(stats['waypoint_control_count'])
    results['rl_counts'].append(stats['rl_control_count'])
    results['path_lengths'].append(path_length)
    
    if success:
        results['success_rate'] += 1
    
    return success, elapsed_time, energy, score

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
    """Plot the benchmark results."""
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Success Rate
    axs[0, 0].bar(['Success Rate'], [results['success_rate'] / NUM_TESTS * 100])
    axs[0, 0].set_ylabel('Success Rate (%)')
    axs[0, 0].set_ylim([0, 100])
    axs[0, 0].set_title(f'Success Rate: {results["success_rate"] / NUM_TESTS * 100:.1f}%')
    
    # Plot 2: Time, Energy, Score
    x = np.arange(NUM_TESTS)
    width = 0.25
    
    axs[0, 1].bar(x - width, results['times'], width, label='Time (s)')
    axs[0, 1].bar(x, results['energies'], width, label='Energy (J)')
    axs[0, 1].bar(x + width, results['scores'], width, label='Score')
    axs[0, 1].set_xlabel('Test #')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels([str(i+1) for i in range(NUM_TESTS)])
    axs[0, 1].legend()
    axs[0, 1].set_title('Performance Metrics')
    
    # Plot 3: Control Statistics
    axs[1, 0].bar(x - width, results['planning_counts'], width, label='Planning Count')
    axs[1, 0].bar(x, results['waypoint_counts'], width, label='Waypoint Control Count')
    axs[1, 0].bar(x + width, results['rl_counts'], width, label='RL Control Count')
    axs[1, 0].set_xlabel('Test #')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels([str(i+1) for i in range(NUM_TESTS)])
    axs[1, 0].legend()
    axs[1, 0].set_title('Control Statistics')
    
    # Plot 4: Path Length
    axs[1, 1].bar(x, results['path_lengths'])
    axs[1, 1].set_xlabel('Test #')
    axs[1, 1].set_ylabel('Path Length (m)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels([str(i+1) for i in range(NUM_TESTS)])
    axs[1, 1].set_title('Path Length')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'benchmark_results_{timestamp}.png')
    
    # Show the figure if GUI is enabled
    if GUI:
        plt.show()

def main():
    """Main function to run the benchmark."""
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    print("====================================================")
    print("HYBRID PATH PLANNING BENCHMARK")
    print("====================================================")
    print(f"Number of tests: {NUM_TESTS}")
    print(f"Simulation frequency: {SIM_FREQ} Hz")
    print(f"Maximum duration: {DURATION_SEC} s")
    print(f"GUI enabled: {GUI}")
    print(f"Obstacles enabled: {OBSTACLES}")
    print("====================================================")
    
    # Run the tests
    for i in range(NUM_TESTS):
        start_pos, goal_pos = generate_random_scenario()
        run_test(i, start_pos, goal_pos)
    
    # Calculate average metrics
    results['avg_time'] = np.mean(results['times'])
    results['avg_energy'] = np.mean(results['energies'])
    results['avg_score'] = np.mean(results['scores'])
    results['success_rate'] = results['success_rate'] / NUM_TESTS
    
    # Print summary
    print("\n====================================================")
    print("BENCHMARK SUMMARY")
    print("====================================================")
    print(f"Success Rate: {results['success_rate'] * 100:.1f}%")
    print(f"Average Time: {results['avg_time']:.2f} s")
    print(f"Average Energy: {results['avg_energy']:.2f} J")
    print(f"Average Score: {results['avg_score']:.3f}")
    print("====================================================")
    
    # Plot the results
    plot_results()

if __name__ == "__main__":
    main()