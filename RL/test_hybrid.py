#!/usr/bin/env python3
"""
Test the hybrid approach for drone flight control.
"""
from __future__ import annotations

import argparse
import time
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO

from swarm.constants import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen import random_task
from swarm.validator.forward import _run_episode, flight_reward
from swarm.utils.env_factory import make_env
from swarm.protocol import ValidationResult
from swarm.planners.hybrid_policy import HybridPolicy


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test hybrid flight planning approach")
    parser.add_argument(
        "--model", type=Path, default=Path("model/ppo_policy.zip"),
        help="Path to the RL policy .zip file",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Show the PyBullet GUI during evaluation",
    )
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    
    # Load RL policy
    rl_policy = None
    if args.model.exists():
        rl_policy = PPO.load(args.model, device="cpu")
        print(f"Loaded RL policy from {args.model}")
    else:
        print(f"Warning: RL policy file not found: {args.model}")
    
    # Generate task
    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)
    
    # Create hybrid policy
    hybrid_policy = HybridPolicy(
        observation_space=None,
        action_space=None,
        rl_policy=rl_policy,
        client_id=None,
        obstacle_ids=[],  # Will be set during environment creation
        planning_horizon=5.0,
        replan_threshold=1.0,
        max_iterations=500,
        step_size=0.3,
        goal_sample_rate=0.2,
        search_radius=2.0,
    )
    
    # Print task information
    print(f"Task start: {task.start}")
    print(f"Task goal: {task.goal}")
    print(f"Task horizon: {task.horizon}")
    
    # Run episode with debug
    print("Testing hybrid flight planning approach...")
    start_time = time.time()
    
    # Create a wrapper to fix the goal in the observation
    original_predict = hybrid_policy.predict
    
    def fixed_predict(observation, *args, **kwargs):
        # Исправляем формат наблюдения - преобразуем из 2D в 1D
        if len(observation.shape) > 1:
            # Если наблюдение двумерное, берем первую строку
            obs_flat = observation[0].copy()
        else:
            obs_flat = observation.copy()
            
        # Заменяем цель в наблюдении на реальную цель
        # Цель находится в последних 3 элементах наблюдения (12-15) или в индексах 6-9
        if len(obs_flat) >= 15:
            obs_flat[12:15] = (np.array(task.goal) - obs_flat[0:3]) / 10.0
        elif len(obs_flat) >= 9:
            obs_flat[6:9] = task.goal
        
        # Выводим отладочную информацию каждые 100 шагов
        if hybrid_policy.waypoint_control_count % 100 == 0:
            print(f"FIXED OBSERVATION:")
            print(f"  Position: {obs_flat[0:3]}")
            print(f"  Velocity: {obs_flat[3:6]}")
            if len(obs_flat) >= 15:
                goal_rel = obs_flat[12:15] * 10.0
                goal_abs = obs_flat[0:3] + goal_rel
                print(f"  Goal (rel): {goal_rel}")
                print(f"  Goal (abs): {goal_abs}")
            else:
                print(f"  Goal: {obs_flat[6:9]}")
            print(f"  Distance to goal: {np.linalg.norm(obs_flat[0:3] - np.array(task.goal)):.2f}")
            print(f"  Task goal: {task.goal}")
        
        # Вызываем оригинальный метод predict с исправленным наблюдением
        return original_predict(obs_flat, *args, **kwargs)
    
    hybrid_policy.predict = fixed_predict
    
    # Также передаем client_id в hybrid_policy для доступа к PyBullet
    if hasattr(task, 'env') and hasattr(task.env, 'CLIENT'):
        hybrid_policy.client_id = task.env.CLIENT
    elif hasattr(task, 'env') and hasattr(task.env, '_cli'):
        hybrid_policy.client_id = task.env._cli
        
    # Устанавливаем правильную цель для гибридной политики
    hybrid_policy.goal_position = np.array(task.goal)
    
    # Создаем собственную функцию для запуска эпизода
    def custom_run_episode(task, uid, model, gui=False):
        """
        Запускает эпизод с прямым управлением дроном.
        """
        # Создаем среду
        env = make_env(task, gui=gui)
        
        # Получаем начальное наблюдение
        try:
            obs = env._computeObs()
        except AttributeError:
            obs = env.get_observation()
            
        if isinstance(obs, dict):
            obs = obs[next(iter(obs))]
        
        # Устанавливаем правильную цель в модель
        model.goal_position = np.array(task.goal)
        
        # Инициализируем переменные
        pos0 = np.asarray(task.start, dtype=float)
        last_pos = pos0.copy()
        t_sim = 0.0
        energy = 0.0
        success = False
        step_count = 0
        
        # Основной цикл симуляции
        while t_sim < task.horizon:
            # Получаем действие от модели
            action = model.act(obs, t_sim)
            
            # Выводим отладочную информацию
            if step_count % 100 == 0:
                print(f"STEP {step_count}: t={t_sim:.2f}")
                print(f"  Position: {last_pos}")
                print(f"  Goal: {task.goal}")
                print(f"  Distance: {np.linalg.norm(last_pos - np.array(task.goal)):.2f}")
                print(f"  Action: {action}")
            
            # Применяем действие к среде
            obs, _r, terminated, truncated, info = env.step(action[None, :])
            
            # Обновляем время и энергию
            t_sim += SIM_DT
            energy += np.abs(action).sum() * SIM_DT
            
            # Получаем текущую позицию дрона
            if obs.ndim == 1:
                last_pos = obs[:3]
            else:
                last_pos = obs[0, :3]
                
            # Проверяем, достигли ли мы цели
            distance_to_goal = np.linalg.norm(last_pos - np.array(task.goal))
            if distance_to_goal < 0.3:  # Порог успеха
                success = True
                print(f"SUCCESS! Reached goal at t={t_sim:.2f}")
                break
                
            # Увеличиваем счетчик шагов
            step_count += 1
            
            # Пауза для визуализации
            if gui:
                try:
                    from swarm.core.drone import track_drone
                    if hasattr(env, 'CLIENT'):
                        client_id = env.CLIENT
                    elif hasattr(env, '_cli'):
                        client_id = env._cli
                    else:
                        client_id = 0
                        
                    if hasattr(env, 'DRONE_IDS'):
                        drone_id = env.DRONE_IDS[0]
                    else:
                        drone_id = 1
                        
                    track_drone(cli=client_id, drone_id=drone_id)
                except Exception as e:
                    print(f"Error tracking drone: {e}")
                time.sleep(SIM_DT)
                
        # Закрываем среду
        env.close()
        
        # Возвращаем результат
        return ValidationResult(
            uid=uid,
            success=success,
            time=t_sim,
            energy=energy,
            score=flight_reward(success, t_sim, energy)
        )
    
    # Запускаем эпизод с собственной функцией
    result = custom_run_episode(task=task, uid=0, model=hybrid_policy, gui=args.gui)
    test_time = time.time() - start_time
    
    # Get statistics
    stats = {}
    if hasattr(hybrid_policy, 'get_statistics'):
        stats = hybrid_policy.get_statistics()
    
    # Print results
    print("=" * 52)
    print("HYBRID FLIGHT PLANNING RESULTS")
    print("=" * 52)
    print(f"Success: {result.success}")
    print(f"Time    : {result.time_sec:.2f} s")
    print(f"Energy  : {result.energy:.1f} J")
    print(f"Score   : {result.score:.3f}")
    print("-" * 52)
    print("CONTROL STATISTICS:")
    if stats:
        for key, value in stats.items():
            print(f"{key:<20}: {value}")
    print("=" * 52)


if __name__ == "__main__":
    main()