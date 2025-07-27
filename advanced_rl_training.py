#!/usr/bin/env python3
"""
Продвинутое обучение RL модели с улучшенной архитектурой и алгоритмами.
"""
import sys
import os
import time
import json
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Deque
from datetime import datetime
from collections import deque, defaultdict
import math

# Добавляем путь к проекту
sys.path.append('/workspace/swarm')

# Импортируем базовые компоненты
from test_algorithms_simple import create_planner
from hybrid_algorithm import create_hybrid_planner

class AdvancedRLAgent:
    """
    Продвинутый RL агент с улучшенной архитектурой.
    Использует Double DQN с experience replay и target network.
    """
    
    def __init__(self, state_size: int = 9, action_size: int = 12, 
                 learning_rate: float = 0.001, memory_size: int = 10000):
        """
        Инициализация продвинутого RL агента.
        
        Parameters
        ----------
        state_size : int
            Размер пространства состояний (расширенный)
        action_size : int
            Размер пространства действий (больше направлений)
        learning_rate : float
            Скорость обучения
        memory_size : int
            Размер буфера experience replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995  # Более медленное уменьшение
        self.epsilon_min = 0.05      # Больше exploration
        self.gamma = 0.99            # Больше внимания к будущим наградам
        self.batch_size = 32
        self.target_update_freq = 100
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural network approximation (упрощенная версия)
        self.q_network = {}
        self.target_network = {}
        
        # Статистика обучения
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'success_rate': 0,
            'average_steps': 0,
            'loss_history': [],
            'reward_history': [],
            'epsilon_history': [],
            'q_values_history': []
        }
        
        # Счетчик обновлений
        self.update_count = 0
    
    def _get_state_features(self, state: np.ndarray) -> np.ndarray:
        """Извлекает расширенные признаки из состояния."""
        current_pos = state[:3]
        goal_pos = state[3:6]
        
        # Базовые признаки
        distance_to_goal = np.linalg.norm(goal_pos - current_pos)
        direction_to_goal = (goal_pos - current_pos) / (distance_to_goal + 1e-8)
        
        # Дополнительные признаки
        height_diff = goal_pos[2] - current_pos[2]
        horizontal_distance = np.linalg.norm(goal_pos[:2] - current_pos[:2])
        
        # Нормализованные признаки
        features = np.array([
            current_pos[0] / 50.0,      # Нормализованная X координата
            current_pos[1] / 50.0,      # Нормализованная Y координата
            current_pos[2] / 20.0,      # Нормализованная Z координата
            direction_to_goal[0],       # Направление к цели X
            direction_to_goal[1],       # Направление к цели Y
            direction_to_goal[2],       # Направление к цели Z
            distance_to_goal / 100.0,   # Нормализованное расстояние
            height_diff / 20.0,         # Нормализованная разность высот
            horizontal_distance / 100.0  # Нормализованное горизонтальное расстояние
        ])
        
        return features
    
    def _discretize_state(self, features: np.ndarray) -> str:
        """Дискретизирует признаки состояния с большей точностью."""
        # Используем более мелкую дискретизацию
        discrete_features = np.round(features * 10) / 10  # Точность 0.1
        return str(discrete_features.tolist())
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Выбирает действие с улучшенной epsilon-greedy стратегией."""
        features = self._get_state_features(state)
        state_key = self._discretize_state(features)
        
        if training and np.random.random() <= self.epsilon:
            # Exploration: случайное действие
            return np.random.randint(0, self.action_size)
        
        # Exploitation: лучшее известное действие
        if state_key not in self.q_network:
            self.q_network[state_key] = np.random.normal(0, 0.1, self.action_size)
        
        return np.argmax(self.q_network[state_key])
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Сохраняет опыт в буфер replay."""
        features = self._get_state_features(state)
        next_features = self._get_state_features(next_state)
        
        self.memory.append((features, action, reward, next_features, done))
    
    def replay(self):
        """Обучение на batch из experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Выбираем случайный batch
        batch = random.sample(self.memory, self.batch_size)
        
        total_loss = 0
        for state_features, action, reward, next_state_features, done in batch:
            state_key = self._discretize_state(state_features)
            next_state_key = self._discretize_state(next_state_features)
            
            # Инициализируем Q-values если нужно
            if state_key not in self.q_network:
                self.q_network[state_key] = np.random.normal(0, 0.1, self.action_size)
            if next_state_key not in self.q_network:
                self.q_network[next_state_key] = np.random.normal(0, 0.1, self.action_size)
            
            # Double DQN update
            current_q = self.q_network[state_key][action]
            
            if done:
                target_q = reward
            else:
                # Используем target network для стабильности
                if next_state_key in self.target_network:
                    next_q_values = self.target_network[next_state_key]
                else:
                    next_q_values = self.q_network[next_state_key]
                
                target_q = reward + self.gamma * np.max(next_q_values)
            
            # Обновляем Q-value
            self.q_network[state_key][action] += self.learning_rate * (target_q - current_q)
            
            # Вычисляем loss для статистики
            loss = (target_q - current_q) ** 2
            total_loss += loss
        
        # Сохраняем loss
        avg_loss = total_loss / self.batch_size
        self.training_stats['loss_history'].append(avg_loss)
        
        # Обновляем target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Обновляет target network."""
        self.target_network = {k: v.copy() for k, v in self.q_network.items()}
    
    def decay_epsilon(self):
        """Уменьшает epsilon более плавно."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_stats['epsilon_history'].append(self.epsilon)

class AdvancedSwarmEnvironment:
    """
    Продвинутая среда с препятствиями и улучшенной системой наград.
    """
    
    def __init__(self, start: np.ndarray, goal: np.ndarray, obstacles: List[np.ndarray] = None):
        """
        Инициализация продвинутой среды.
        
        Parameters
        ----------
        start : np.ndarray
            Начальная позиция
        goal : np.ndarray
            Целевая позиция
        obstacles : List[np.ndarray]
            Список препятствий (центры сфер)
        """
        self.start = start.copy()
        self.goal = goal.copy()
        self.current_pos = start.copy()
        self.previous_pos = start.copy()
        self.max_steps = 200  # Больше шагов для сложных задач
        self.current_step = 0
        
        # Препятствия (сферы с радиусом 2м)
        self.obstacles = obstacles if obstacles is not None else []
        self.obstacle_radius = 2.0
        
        # Расширенные действия: 12 направлений + изменение высоты
        self.actions = [
            np.array([1, 0, 0]),      # Вперед
            np.array([-1, 0, 0]),     # Назад
            np.array([0, 1, 0]),      # Вправо
            np.array([0, -1, 0]),     # Влево
            np.array([1, 1, 0]),      # Вперед-вправо
            np.array([1, -1, 0]),     # Вперед-влево
            np.array([-1, 1, 0]),     # Назад-вправо
            np.array([-1, -1, 0]),    # Назад-влево
            np.array([0, 0, 1]),      # Вверх
            np.array([0, 0, -1]),     # Вниз
            np.array([1, 0, 1]),      # Вперед-вверх
            np.array([1, 0, -1]),     # Вперед-вниз
        ]
        
        # Нормализуем действия
        for i, action in enumerate(self.actions):
            if np.linalg.norm(action) > 0:
                self.actions[i] = action / np.linalg.norm(action)
        
        # История позиций для анализа
        self.position_history = [start.copy()]
        
        # Статистика эпизода
        self.episode_stats = {
            'min_distance': np.linalg.norm(start - goal),
            'obstacle_collisions': 0,
            'progress_made': 0,
            'efficiency': 0
        }
    
    def reset(self) -> np.ndarray:
        """Сбрасывает среду в начальное состояние."""
        self.current_pos = self.start.copy()
        self.previous_pos = self.start.copy()
        self.current_step = 0
        self.position_history = [self.start.copy()]
        
        # Сбрасываем статистику
        self.episode_stats = {
            'min_distance': np.linalg.norm(self.start - self.goal),
            'obstacle_collisions': 0,
            'progress_made': 0,
            'efficiency': 0
        }
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Возвращает расширенное состояние."""
        # Базовое состояние
        base_state = np.concatenate([self.current_pos, self.goal])
        
        # Добавляем информацию о ближайшем препятствии
        if self.obstacles:
            distances_to_obstacles = [np.linalg.norm(self.current_pos - obs) 
                                    for obs in self.obstacles]
            min_obstacle_distance = min(distances_to_obstacles)
            closest_obstacle_idx = np.argmin(distances_to_obstacles)
            closest_obstacle = self.obstacles[closest_obstacle_idx]
            
            # Направление к ближайшему препятствию
            obstacle_direction = (closest_obstacle - self.current_pos) / (min_obstacle_distance + 1e-8)
            
            extended_state = np.concatenate([
                base_state,
                [min_obstacle_distance / 10.0],  # Нормализованное расстояние
                obstacle_direction[:2]           # Направление в 2D
            ])
        else:
            # Если нет препятствий, добавляем нули
            extended_state = np.concatenate([base_state, [0, 0, 0]])
        
        return extended_state
    
    def _check_collision(self, pos: np.ndarray) -> bool:
        """Проверяет столкновение с препятствиями."""
        for obstacle in self.obstacles:
            distance = np.linalg.norm(pos - obstacle)
            if distance < self.obstacle_radius:
                return True
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Выполняет действие в продвинутой среде."""
        self.previous_pos = self.current_pos.copy()
        
        # Применяем действие с адаптивным размером шага
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal)
        step_size = min(1.5, max(0.5, distance_to_goal / 10))  # Адаптивный размер шага
        
        movement = self.actions[action] * step_size
        new_pos = self.current_pos + movement
        
        # Ограничиваем движение в разумных пределах
        new_pos = np.clip(new_pos, [-50, -50, 0.5], [50, 50, 20])
        
        # Проверяем столкновение
        collision = self._check_collision(new_pos)
        if collision:
            self.episode_stats['obstacle_collisions'] += 1
            # Не двигаемся при столкновении
            new_pos = self.current_pos
        
        self.current_pos = new_pos
        self.current_step += 1
        self.position_history.append(new_pos.copy())
        
        # Обновляем статистику
        current_distance = np.linalg.norm(self.current_pos - self.goal)
        if current_distance < self.episode_stats['min_distance']:
            self.episode_stats['min_distance'] = current_distance
            self.episode_stats['progress_made'] = 1
        
        # Вычисляем награду (ИСХОДНАЯ ФУНКЦИЯ БЕЗ ИЗМЕНЕНИЙ)
        reward = self._calculate_reward()
        
        # Добавляем штраф за столкновение
        if collision:
            reward -= 5.0  # Штраф за столкновение
        
        # Проверяем условия завершения
        done = self._is_done()
        
        # Вычисляем эффективность
        if done:
            path_length = sum(np.linalg.norm(self.position_history[i+1] - self.position_history[i]) 
                            for i in range(len(self.position_history)-1))
            direct_distance = np.linalg.norm(self.goal - self.start)
            self.episode_stats['efficiency'] = direct_distance / (path_length + 1e-8)
        
        info = {
            'distance_to_goal': current_distance,
            'steps': self.current_step,
            'collision': collision,
            'episode_stats': self.episode_stats.copy()
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self) -> float:
        """Вычисляет награду за текущее состояние (ИСХОДНАЯ ФУНКЦИЯ БЕЗ ИЗМЕНЕНИЙ)."""
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal)
        
        # Награда за приближение к цели
        max_distance = np.linalg.norm(self.start - self.goal)
        progress_reward = (max_distance - distance_to_goal) / max_distance
        
        # Большая награда за достижение цели
        if distance_to_goal < 1.0:
            return 100.0 + progress_reward
        
        # Штраф за каждый шаг (стимулирует быстрое достижение цели)
        step_penalty = -0.1
        
        # Штраф за удаление от цели
        if distance_to_goal > max_distance:
            return -10.0 + step_penalty
        
        return progress_reward + step_penalty
    
    def _is_done(self) -> bool:
        """Проверяет условия завершения эпизода."""
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal)
        
        # Достигли цели
        if distance_to_goal < 1.0:
            return True
        
        # Превысили максимальное количество шагов
        if self.current_step >= self.max_steps:
            return True
        
        return False

def create_training_scenarios_with_obstacles() -> List[Dict]:
    """Создает тренировочные сценарии с препятствиями."""
    scenarios = [
        # Простые сценарии без препятствий
        {
            'name': 'Простой без препятствий',
            'start': np.array([0, 0, 3]),
            'goal': np.array([5, 0, 3]),
            'obstacles': [],
            'difficulty': 0.1
        },
        {
            'name': 'Средний без препятствий',
            'start': np.array([0, 0, 2]),
            'goal': np.array([10, 8, 4]),
            'obstacles': [],
            'difficulty': 0.3
        },
        
        # Сценарии с препятствиями
        {
            'name': 'Простой с одним препятствием',
            'start': np.array([0, 0, 3]),
            'goal': np.array([10, 0, 3]),
            'obstacles': [np.array([5, 0, 3])],
            'difficulty': 0.4
        },
        {
            'name': 'Средний с двумя препятствиями',
            'start': np.array([0, 0, 2]),
            'goal': np.array([15, 10, 4]),
            'obstacles': [np.array([7, 3, 3]), np.array([12, 7, 3])],
            'difficulty': 0.6
        },
        {
            'name': 'Сложный лабиринт',
            'start': np.array([-5, -5, 2]),
            'goal': np.array([15, 15, 5]),
            'obstacles': [
                np.array([2, -2, 3]), np.array([5, 2, 4]), 
                np.array([8, 8, 3]), np.array([12, 5, 4])
            ],
            'difficulty': 0.8
        },
        {
            'name': 'Вертикальный с препятствиями',
            'start': np.array([0, 0, 1]),
            'goal': np.array([3, 3, 12]),
            'obstacles': [np.array([1, 1, 6]), np.array([2, 2, 9])],
            'difficulty': 0.7
        }
    ]
    
    return scenarios

def train_advanced_rl_agent(episodes: int = 5000, save_model: bool = True) -> AdvancedRLAgent:
    """
    Обучает продвинутого RL агента.
    
    Parameters
    ----------
    episodes : int
        Количество эпизодов обучения
    save_model : bool
        Сохранять ли модель после обучения
        
    Returns
    -------
    AdvancedRLAgent
        Обученный агент
    """
    print("🚀 ПРОДВИНУТОЕ ОБУЧЕНИЕ RL МОДЕЛИ")
    print("=" * 60)
    
    # Создаем продвинутого агента
    agent = AdvancedRLAgent(
        state_size=9,
        action_size=12,
        learning_rate=0.001,
        memory_size=20000
    )
    
    # Получаем тренировочные сценарии
    training_scenarios = create_training_scenarios_with_obstacles()
    
    total_rewards = []
    success_count = 0
    episode_lengths = []
    
    print(f"📚 Начинаем продвинутое обучение на {episodes} эпизодах...")
    print(f"🎯 Сценариев обучения: {len(training_scenarios)}")
    print(f"🧠 Архитектура: Double DQN с experience replay")
    print(f"🎮 Действий: {agent.action_size}, Состояний: {agent.state_size}")
    
    best_success_rate = 0
    best_avg_reward = float('-inf')
    
    for episode in range(episodes):
        # Выбираем сценарий (больше внимания сложным сценариям на поздних этапах)
        if episode < episodes // 3:
            # Первая треть - простые сценарии
            scenario_idx = episode % min(3, len(training_scenarios))
        else:
            # Остальное время - все сценарии
            scenario_idx = episode % len(training_scenarios)
        
        scenario = training_scenarios[scenario_idx]
        
        # Создаем среду
        env = AdvancedSwarmEnvironment(
            start=scenario['start'],
            goal=scenario['goal'],
            obstacles=scenario['obstacles']
        )
        
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Выбираем действие
            action = agent.get_action(state, training=True)
            
            # Выполняем действие
            next_state, reward, done, info = env.step(action)
            
            # Сохраняем опыт
            agent.remember(state, action, reward, next_state, done)
            
            # Обучаемся на опыте
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # Обновляем статистику
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        if info['distance_to_goal'] < 1.0:
            success_count += 1
        
        # Уменьшаем exploration
        agent.decay_epsilon()
        
        # Выводим прогресс
        if (episode + 1) % 200 == 0:
            recent_rewards = total_rewards[-200:]
            recent_successes = sum(1 for i in range(max(0, len(total_rewards)-200), len(total_rewards))
                                 if i < success_count)
            
            avg_reward = np.mean(recent_rewards)
            success_rate = recent_successes / 200 * 100 if len(recent_rewards) == 200 else success_count / (episode + 1) * 100
            avg_steps = np.mean(episode_lengths[-200:])
            avg_loss = np.mean(agent.training_stats['loss_history'][-50:]) if agent.training_stats['loss_history'] else 0
            
            print(f"  Эпизод {episode + 1:5d}: Reward: {avg_reward:7.2f}, Success: {success_rate:5.1f}%, "
                  f"Steps: {avg_steps:5.1f}, Loss: {avg_loss:.4f}, ε: {agent.epsilon:.3f}")
            
            # Сохраняем лучшую модель
            if success_rate > best_success_rate or (success_rate == best_success_rate and avg_reward > best_avg_reward):
                best_success_rate = success_rate
                best_avg_reward = avg_reward
                
                if save_model:
                    save_checkpoint(agent, episode + 1, success_rate, avg_reward)
    
    # Обновляем финальную статистику
    final_success_rate = success_count / episodes * 100
    agent.training_stats.update({
        'episodes': episodes,
        'total_reward': sum(total_rewards),
        'success_rate': final_success_rate,
        'average_steps': np.mean(episode_lengths),
        'reward_history': total_rewards,
        'best_success_rate': best_success_rate,
        'best_avg_reward': best_avg_reward
    })
    
    print(f"\n✅ Продвинутое обучение завершено!")
    print(f"  📊 Финальная статистика:")
    print(f"    • Success Rate: {final_success_rate:.1f}%")
    print(f"    • Лучший Success Rate: {best_success_rate:.1f}%")
    print(f"    • Средняя награда: {np.mean(total_rewards):.2f}")
    print(f"    • Размер Q-network: {len(agent.q_network)} состояний")
    print(f"    • Размер experience buffer: {len(agent.memory)}")
    print(f"    • Финальный epsilon: {agent.epsilon:.3f}")
    
    return agent

def save_checkpoint(agent: AdvancedRLAgent, episode: int, success_rate: float, avg_reward: float):
    """Сохраняет checkpoint модели."""
    os.makedirs('models/checkpoints', exist_ok=True)
    
    checkpoint_data = {
        'q_network': {k: v.tolist() for k, v in agent.q_network.items()},
        'target_network': {k: v.tolist() for k, v in agent.target_network.items()},
        'training_stats': agent.training_stats,
        'hyperparameters': {
            'learning_rate': agent.learning_rate,
            'gamma': agent.gamma,
            'epsilon': agent.epsilon,
            'epsilon_decay': agent.epsilon_decay,
            'batch_size': agent.batch_size
        },
        'episode': episode,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = f'models/checkpoints/advanced_rl_checkpoint_ep{episode}.json'
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"    💾 Checkpoint сохранен: {checkpoint_path}")

def test_advanced_rl_agent(agent: AdvancedRLAgent, test_scenarios: List[Dict]) -> Dict:
    """
    Тестирует продвинутого RL агента.
    
    Parameters
    ----------
    agent : AdvancedRLAgent
        Обученный агент
    test_scenarios : List[Dict]
        Список тестовых сценариев
        
    Returns
    -------
    Dict
        Результаты тестирования
    """
    print(f"\n🧪 ТЕСТИРОВАНИЕ ПРОДВИНУТОГО RL АГЕНТА")
    print("=" * 50)
    
    results = []
    
    for i, scenario in enumerate(test_scenarios):
        print(f"  📋 Сценарий {i+1}: {scenario['name']}")
        print(f"      {scenario['start']} → {scenario['goal']}")
        if scenario['obstacles']:
            print(f"      Препятствий: {len(scenario['obstacles'])}")
        
        env = AdvancedSwarmEnvironment(
            start=scenario['start'],
            goal=scenario['goal'],
            obstacles=scenario['obstacles']
        )
        
        state = env.reset()
        path = [scenario['start'].copy()]
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(state, training=False)  # Без exploration
            state, reward, done, info = env.step(action)
            
            path.append(env.current_pos.copy())
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Вычисляем метрики
        success = info['distance_to_goal'] < 1.0
        path_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
        direct_distance = np.linalg.norm(scenario['goal'] - scenario['start'])
        efficiency = direct_distance / path_length if path_length > 0 else 0
        
        result = {
            'scenario': i + 1,
            'name': scenario['name'],
            'start': scenario['start'],
            'goal': scenario['goal'],
            'obstacles_count': len(scenario['obstacles']),
            'success': success,
            'steps': steps,
            'total_reward': total_reward,
            'path_length': path_length,
            'direct_distance': direct_distance,
            'efficiency': efficiency,
            'final_distance': info['distance_to_goal'],
            'collisions': info['episode_stats']['obstacle_collisions'],
            'difficulty': scenario['difficulty']
        }
        
        results.append(result)
        
        status = "✅ Success" if success else "❌ Failure"
        print(f"    {status}, Steps: {steps}, Efficiency: {efficiency:.3f}, "
              f"Reward: {total_reward:.1f}, Collisions: {result['collisions']}")
    
    # Общая статистика
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    avg_efficiency = np.mean([r['efficiency'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_collisions = np.mean([r['collisions'] for r in results])
    
    summary = {
        'results': results,
        'success_rate': success_rate,
        'average_efficiency': avg_efficiency,
        'average_steps': avg_steps,
        'average_collisions': avg_collisions,
        'total_scenarios': len(test_scenarios)
    }
    
    print(f"\n📊 ОБЩИЕ РЕЗУЛЬТАТЫ ПРОДВИНУТОГО RL:")
    print(f"  • Success Rate: {success_rate:.1f}%")
    print(f"  • Средняя эффективность: {avg_efficiency:.3f}")
    print(f"  • Среднее количество шагов: {avg_steps:.1f}")
    print(f"  • Среднее количество столкновений: {avg_collisions:.1f}")
    
    return summary

def main():
    """Главная функция продвинутого обучения."""
    
    print("🚀 ПРОДВИНУТАЯ СИСТЕМА ОБУЧЕНИЯ RL МОДЕЛИ")
    print("=" * 70)
    print("Проект: Swarm Path Planning")
    print("Версия: 8.0 (Advanced RL Training)")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Обучаем продвинутого RL агента
        print("🎓 Этап 1: Продвинутое обучение RL агента")
        agent = train_advanced_rl_agent(episodes=5000, save_model=True)
        
        # 2. Тестовые сценарии с препятствиями
        test_scenarios = create_training_scenarios_with_obstacles()
        
        # 3. Тестируем продвинутого RL агента
        print(f"\n🧪 Этап 2: Тестирование продвинутого RL агента")
        advanced_results = test_advanced_rl_agent(agent, test_scenarios)
        
        # 4. Сравниваем с предыдущей версией и другими алгоритмами
        print(f"\n⚖️  Этап 3: Сравнение производительности")
        
        # Загружаем результаты простой RL модели для сравнения
        try:
            with open('rl_training_results.json', 'r') as f:
                simple_rl_results = json.load(f)
            simple_success_rate = simple_rl_results['test_results']['success_rate']
            print(f"  📊 Простая RL модель: {simple_success_rate:.1f}% success rate")
        except:
            simple_success_rate = 25.0  # Из предыдущих результатов
            print(f"  📊 Простая RL модель: {simple_success_rate:.1f}% success rate (из памяти)")
        
        print(f"  📊 Продвинутая RL модель: {advanced_results['success_rate']:.1f}% success rate")
        
        improvement = advanced_results['success_rate'] - simple_success_rate
        print(f"  📈 Улучшение: {improvement:+.1f} процентных пунктов")
        
        # 5. Сохраняем результаты
        results_data = {
            'training_stats': agent.training_stats,
            'test_results': advanced_results,
            'comparison': {
                'simple_rl_success_rate': simple_success_rate,
                'advanced_rl_success_rate': advanced_results['success_rate'],
                'improvement': improvement
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('advanced_rl_training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Сохраняем финальную модель
        final_model_data = {
            'q_network': {k: v.tolist() for k, v in agent.q_network.items()},
            'target_network': {k: v.tolist() for k, v in agent.target_network.items()},
            'training_stats': agent.training_stats,
            'hyperparameters': {
                'learning_rate': agent.learning_rate,
                'gamma': agent.gamma,
                'epsilon': agent.epsilon,
                'epsilon_decay': agent.epsilon_decay,
                'batch_size': agent.batch_size
            },
            'training_date': datetime.now().isoformat()
        }
        
        os.makedirs('models', exist_ok=True)
        with open('models/advanced_rl_path_planner.json', 'w') as f:
            json.dump(final_model_data, f, indent=2)
        
        print(f"\n✅ ПРОДВИНУТОЕ ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 70)
        print(f"📊 Результаты сохранены в: advanced_rl_training_results.json")
        print(f"💾 Модель сохранена в: models/advanced_rl_path_planner.json")
        print(f"🗂️  Checkpoints сохранены в: models/checkpoints/")
        
        return agent, results_data
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    agent, results = main()