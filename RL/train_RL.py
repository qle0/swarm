#!/usr/bin/env python3

import argparse
import os


from stable_baselines3 import PPO, A2C, SAC

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC
from swarm.planners.planner_policy import PlannerPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--algorithm", type=str, choices=["ppo", "a2c", "sac"], default="ppo",
                        help="RL algorithm to use for training")
    parser.add_argument("--planner", type=str, choices=["none", "rrt", "dijkstra"], default="none",
                        help="Use a path planner instead of RL training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)
    env = make_env(task, gui=False)

    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    if args.planner != "none":
        print(f"Using {args.planner} planner instead of RL training")
        
        # Get obstacle IDs from the environment
        cli = env.getPyBulletClient()
        num_objects = cli.getNumBodies()
        obstacle_ids = []
        for i in range(num_objects):
            body_info = cli.getBodyInfo(i)
            body_name = body_info[1].decode('utf-8')
            # Exclude the drone and the goal
            if "drone" not in body_name.lower() and "goal" not in body_name.lower():
                obstacle_ids.append(i)
        
        # Create planner policy
        planner_policy = PlannerPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            planner_type=args.planner,
            client_id=cli,
            obstacle_ids=obstacle_ids,
            # Additional planner-specific parameters
            max_iterations=5000 if args.planner == "rrt" else None,
            resolution=0.2 if args.planner == "dijkstra" else None,
        )
        
        # Save the planner policy
        model_path = f"model/{args.planner}_policy"
        
        # We can't directly save the planner policy, so we'll create a dummy file
        with open(f"{model_path}.txt", "w") as f:
            f.write(f"Planner type: {args.planner}\n")
            f.write(f"This file indicates that a {args.planner} planner should be used instead of an RL policy.\n")
            f.write(f"To use this planner, run: python RL/test_planner.py --planner {args.planner}\n")
        
        print(f"Created planner policy file at {model_path}.txt")
    else:
        # Choose the RL algorithm
        if args.algorithm == "ppo":
            model = PPO("MlpPolicy", env, verbose=1)
        elif args.algorithm == "a2c":
            model = A2C("MlpPolicy", env, verbose=1)
        elif args.algorithm == "sac":
            model = SAC("MlpPolicy", env, verbose=1)
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")
        
        # Train the model
        model.learn(total_timesteps=args.timesteps)
        
        # Save the model
        model_path = f"model/{args.algorithm}_policy"
        model.save(model_path)
        print(f"Model saved to {model_path}")

    env.close()


if __name__ == "__main__":
    main()
