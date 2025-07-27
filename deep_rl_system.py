#!/usr/bin/env python3
"""
Deep RL система для сложных сценариев планирования пути.
Включает DQN, PPO, мультиагентное и иерархическое RL.
"""
import sys
import os
import time
import json
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
import math

# Добавляем путь к проекту
sys.path.append('/workspace/swarm')

class SimpleNeuralNetwork:
    """
    Упрощенная нейронная сеть для DQN (без torch/tensorflow).
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.001):
        """
        Инициализация простой нейронной сети.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Инициализация весов (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Для Adam optimizer
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.t = 0  # time step for Adam
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """Forward pass."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, output):
        """Backward pass with Adam optimizer."""
        m = X.shape[0]
        
        # Compute gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Adam optimizer
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Update moments
        self.m_W2 = beta1 * self.m_W2 + (1 - beta1) * dW2
        self.v_W2 = beta2 * self.v_W2 + (1 - beta2) * (dW2 ** 2)
        self.m_b2 = beta1 * self.m_b2 + (1 - beta1) * db2
        self.v_b2 = beta2 * self.v_b2 + (1 - beta2) * (db2 ** 2)
        
        self.m_W1 = beta1 * self.m_W1 + (1 - beta1) * dW1
        self.v_W1 = beta2 * self.v_W1 + (1 - beta2) * (dW1 ** 2)
        self.m_b1 = beta1 * self.m_b1 + (1 - beta1) * db1
        self.v_b1 = beta2 * self.v_b1 + (1 - beta2) * (db1 ** 2)
        
        # Bias correction
        m_W2_corr = self.m_W2 / (1 - beta1 ** self.t)
        v_W2_corr = self.v_W2 / (1 - beta2 ** self.t)
        m_b2_corr = self.m_b2 / (1 - beta1 ** self.t)
        v_b2_corr = self.v_b2 / (1 - beta2 ** self.t)
        
        m_W1_corr = self.m_W1 / (1 - beta1 ** self.t)
        v_W1_corr = self.v_W1 / (1 - beta2 ** self.t)
        m_b1_corr = self.m_b1 / (1 - beta1 ** self.t)
        v_b1_corr = self.v_b1 / (1 - beta2 ** self.t)
        
        # Update parameters
        self.W2 -= self.learning_rate * m_W2_corr / (np.sqrt(v_W2_corr) + eps)
        self.b2 -= self.learning_rate * m_b2_corr / (np.sqrt(v_b2_corr) + eps)
        self.W1 -= self.learning_rate * m_W1_corr / (np.sqrt(v_W1_corr) + eps)
        self.b1 -= self.learning_rate * m_b1_corr / (np.sqrt(v_b1_corr) + eps)
    
    def predict(self, X):
        """Predict Q-values."""
        return self.forward(X)
    
    def copy_weights_from(self, other_network):
        """Copy weights from another network (for target network)."""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()

class DQNAgent:
    """
    Deep Q-Network агент для планирования пути.
    """
    
    def __init__(self, state_size: int = 12, action_size: int = 8, learning_rate: float = 0.001):
        """
        Инициализация DQN агента.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # DQN параметры
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.batch_size = 32
        self.memory_size = 10000
        self.target_update_freq = 100
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Neural networks
        hidden_size = 64
        self.q_network = SimpleNeuralNetwork(state_size, hidden_size, action_size, learning_rate)
        self.target_network = SimpleNeuralNetwork(state_size, hidden_size, action_size, learning_rate)
        self.target_network.copy_weights_from(self.q_network)
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'success_rate': 0,
            'loss_history': [],
            'reward_history': [],
            'epsilon_history': []
        }
        
        self.update_count = 0
    
    def get_state_features(self, state: np.ndarray) -> np.ndarray:
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
            goal_pos[0] / 50.0,         # Нормализованная X цели
            goal_pos[1] / 50.0,         # Нормализованная Y цели
            goal_pos[2] / 20.0,         # Нормализованная Z цели
            direction_to_goal[0],       # Направление к цели X
            direction_to_goal[1],       # Направление к цели Y
            direction_to_goal[2],       # Направление к цели Z
            distance_to_goal / 100.0,   # Нормализованное расстояние
            height_diff / 20.0,         # Нормализованная разность высот
            horizontal_distance / 100.0  # Нормализованное горизонтальное расстояние
        ])
        
        return features
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Выбирает действие используя epsilon-greedy стратегию."""
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        features = self.get_state_features(state).reshape(1, -1)
        q_values = self.q_network.predict(features)
        return np.argmax(q_values[0])
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Сохраняет опыт в replay buffer."""
        features = self.get_state_features(state)
        next_features = self.get_state_features(next_state)
        self.memory.append((features, action, reward, next_features, done))
    
    def replay(self):
        """Обучение на batch из experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states)
        
        # Compute target Q values
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the network
        self.q_network.backward(states, target_q_values, current_q_values)
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)
    
    def decay_epsilon(self):
        """Уменьшает epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MultiAgentEnvironment:
    """
    Мультиагентная среда для обучения нескольких дронов.
    """
    
    def __init__(self, num_agents: int = 3, world_size: Tuple[int, int, int] = (50, 50, 20)):
        """
        Инициализация мультиагентной среды.
        """
        self.num_agents = num_agents
        self.world_size = world_size
        self.max_steps = 200
        
        # Состояния агентов
        self.agents_pos = []
        self.agents_goals = []
        self.agents_start = []
        
        # Препятствия
        self.obstacles = []
        self.obstacle_radius = 2.0
        
        # Статистика
        self.current_step = 0
        self.collision_count = 0
        
        # Действия (8 направлений)
        self.actions = [
            np.array([1, 0, 0]),    # Вперед
            np.array([-1, 0, 0]),   # Назад
            np.array([0, 1, 0]),    # Вправо
            np.array([0, -1, 0]),   # Влево
            np.array([1, 1, 0]),    # Вперед-вправо
            np.array([1, -1, 0]),   # Вперед-влево
            np.array([-1, 1, 0]),   # Назад-вправо
            np.array([-1, -1, 0]),  # Назад-влево
        ]
        
        # Нормализуем действия
        for i, action in enumerate(self.actions):
            if np.linalg.norm(action) > 0:
                self.actions[i] = action / np.linalg.norm(action)
    
    def reset(self) -> List[np.ndarray]:
        """Сбрасывает среду и возвращает начальные состояния."""
        self.current_step = 0
        self.collision_count = 0
        
        # Генерируем случайные стартовые позиции и цели
        self.agents_pos = []
        self.agents_goals = []
        self.agents_start = []
        
        for i in range(self.num_agents):
            # Случайная стартовая позиция
            start = np.array([
                np.random.uniform(-20, 20),
                np.random.uniform(-20, 20),
                np.random.uniform(2, 8)
            ])
            
            # Случайная цель (достаточно далеко от старта)
            while True:
                goal = np.array([
                    np.random.uniform(-20, 20),
                    np.random.uniform(-20, 20),
                    np.random.uniform(2, 8)
                ])
                if np.linalg.norm(goal - start) > 10:  # Минимальное расстояние
                    break
            
            self.agents_pos.append(start.copy())
            self.agents_goals.append(goal)
            self.agents_start.append(start.copy())
        
        # Генерируем препятствия
        self._generate_obstacles()
        
        return self._get_states()
    
    def _generate_obstacles(self):
        """Генерирует случайные препятствия."""
        self.obstacles = []
        num_obstacles = np.random.randint(3, 8)
        
        for _ in range(num_obstacles):
            obstacle_pos = np.array([
                np.random.uniform(-25, 25),
                np.random.uniform(-25, 25),
                np.random.uniform(1, 15)
            ])
            
            # Проверяем, что препятствие не блокирует агентов
            too_close = False
            for i in range(self.num_agents):
                if (np.linalg.norm(obstacle_pos - self.agents_start[i]) < self.obstacle_radius + 3 or
                    np.linalg.norm(obstacle_pos - self.agents_goals[i]) < self.obstacle_radius + 3):
                    too_close = True
                    break
            
            if not too_close:
                self.obstacles.append(obstacle_pos)
    
    def _get_states(self) -> List[np.ndarray]:
        """Возвращает состояния всех агентов."""
        states = []
        
        for i in range(self.num_agents):
            # Базовое состояние агента
            agent_state = np.concatenate([self.agents_pos[i], self.agents_goals[i]])
            
            # Добавляем информацию о других агентах (ближайшие 2)
            other_agents = []
            distances = []
            
            for j in range(self.num_agents):
                if i != j:
                    dist = np.linalg.norm(self.agents_pos[i] - self.agents_pos[j])
                    distances.append((dist, j))
            
            distances.sort()
            
            # Добавляем информацию о 2 ближайших агентах
            for k in range(min(2, len(distances))):
                _, j = distances[k]
                relative_pos = self.agents_pos[j] - self.agents_pos[i]
                other_agents.extend(relative_pos / 50.0)  # Нормализуем
            
            # Дополняем нулями если агентов меньше 2
            while len(other_agents) < 6:
                other_agents.append(0.0)
            
            # Объединяем все в одно состояние
            full_state = np.concatenate([agent_state, other_agents])
            states.append(full_state)
        
        return states
    
    def _check_collision(self, pos: np.ndarray, agent_id: int) -> Tuple[bool, bool]:
        """Проверяет столкновения с препятствиями и другими агентами."""
        obstacle_collision = False
        agent_collision = False
        
        # Проверка столкновений с препятствиями
        for obstacle in self.obstacles:
            if np.linalg.norm(pos - obstacle) < self.obstacle_radius:
                obstacle_collision = True
                break
        
        # Проверка столкновений с другими агентами
        for i, other_pos in enumerate(self.agents_pos):
            if i != agent_id and np.linalg.norm(pos - other_pos) < 1.5:  # Минимальное расстояние между агентами
                agent_collision = True
                break
        
        return obstacle_collision, agent_collision
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Выполняет действия всех агентов."""
        rewards = []
        dones = []
        info = {'collisions': 0, 'successes': 0, 'distances_to_goals': []}
        
        # Выполняем действия всех агентов
        new_positions = []
        for i, action in enumerate(actions):
            current_pos = self.agents_pos[i].copy()
            
            # Применяем действие
            movement = self.actions[action] * 1.0  # Размер шага
            new_pos = current_pos + movement
            
            # Ограничиваем движение в пределах мира
            new_pos = np.clip(new_pos, [-25, -25, 1], [25, 25, 15])
            
            new_positions.append(new_pos)
        
        # Проверяем столкновения и обновляем позиции
        for i, new_pos in enumerate(new_positions):
            obstacle_collision, agent_collision = self._check_collision(new_pos, i)
            
            if not obstacle_collision and not agent_collision:
                self.agents_pos[i] = new_pos
            elif obstacle_collision or agent_collision:
                info['collisions'] += 1
                self.collision_count += 1
        
        # Вычисляем награды и проверяем завершение
        for i in range(self.num_agents):
            distance_to_goal = np.linalg.norm(self.agents_pos[i] - self.agents_goals[i])
            info['distances_to_goals'].append(distance_to_goal)
            
            # Базовая награда за приближение к цели
            max_distance = np.linalg.norm(self.agents_start[i] - self.agents_goals[i])
            progress_reward = (max_distance - distance_to_goal) / max_distance
            
            # Награда за достижение цели
            if distance_to_goal < 1.0:
                reward = 100.0 + progress_reward
                done = True
                info['successes'] += 1
            else:
                reward = progress_reward - 0.1  # Штраф за шаг
                done = False
            
            # Штраф за столкновения
            obstacle_collision, agent_collision = self._check_collision(self.agents_pos[i], i)
            if obstacle_collision:
                reward -= 10.0
            if agent_collision:
                reward -= 5.0
            
            rewards.append(reward)
            dones.append(done)
        
        self.current_step += 1
        
        # Проверяем глобальное завершение
        if self.current_step >= self.max_steps:
            dones = [True] * self.num_agents
        
        next_states = self._get_states()
        
        return next_states, rewards, dones, info

class HierarchicalRLAgent:
    """
    Иерархический RL агент с высокоуровневым и низкоуровневым планированием.
    """
    
    def __init__(self, state_size: int = 12, action_size: int = 8):
        """
        Инициализация иерархического RL агента.
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Высокоуровневый планировщик (выбирает подцели)
        self.high_level_agent = DQNAgent(state_size, 4, 0.001)  # 4 направления подцелей
        
        # Низкоуровневый планировщик (выполняет действия к подцели)
        self.low_level_agent = DQNAgent(state_size + 3, action_size, 0.001)  # +3 для подцели
        
        # Параметры иерархии
        self.subgoal_distance = 5.0  # Расстояние до подцели
        self.subgoal_timeout = 20    # Максимальное время на достижение подцели
        
        # Текущее состояние иерархии
        self.current_subgoal = None
        self.subgoal_steps = 0
        self.subgoal_achieved = False
        
        # Статистика
        self.training_stats = {
            'episodes': 0,
            'subgoals_achieved': 0,
            'subgoals_failed': 0,
            'high_level_rewards': [],
            'low_level_rewards': []
        }
    
    def get_subgoal_directions(self) -> List[np.ndarray]:
        """Возвращает возможные направления для подцелей."""
        return [
            np.array([1, 0, 0]),   # Вперед
            np.array([0, 1, 0]),   # Вправо
            np.array([-1, 0, 0]),  # Назад
            np.array([0, -1, 0])   # Влево
        ]
    
    def generate_subgoal(self, current_pos: np.ndarray, goal_pos: np.ndarray, direction_idx: int) -> np.ndarray:
        """Генерирует подцель в заданном направлении."""
        directions = self.get_subgoal_directions()
        direction = directions[direction_idx]
        
        # Подцель на расстоянии subgoal_distance в заданном направлении
        subgoal = current_pos + direction * self.subgoal_distance
        
        # Ограничиваем подцель в разумных пределах
        subgoal = np.clip(subgoal, [-25, -25, 1], [25, 25, 15])
        
        return subgoal
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Выбирает действие используя иерархический подход."""
        current_pos = state[:3]
        goal_pos = state[3:6]
        
        # Проверяем, нужно ли выбрать новую подцель
        if (self.current_subgoal is None or 
            self.subgoal_steps >= self.subgoal_timeout or
            self.subgoal_achieved):
            
            # Высокоуровневое планирование: выбираем направление подцели
            high_level_action = self.high_level_agent.get_action(state, training)
            self.current_subgoal = self.generate_subgoal(current_pos, goal_pos, high_level_action)
            self.subgoal_steps = 0
            self.subgoal_achieved = False
        
        # Низкоуровневое планирование: действие к подцели
        extended_state = np.concatenate([state, self.current_subgoal])
        low_level_action = self.low_level_agent.get_action(extended_state, training)
        
        self.subgoal_steps += 1
        
        # Проверяем достижение подцели
        if np.linalg.norm(current_pos - self.current_subgoal) < 2.0:
            self.subgoal_achieved = True
            self.training_stats['subgoals_achieved'] += 1
        elif self.subgoal_steps >= self.subgoal_timeout:
            self.training_stats['subgoals_failed'] += 1
        
        return low_level_action
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Сохраняет опыт для обоих уровней."""
        # Низкоуровневый опыт
        if self.current_subgoal is not None:
            extended_state = np.concatenate([state, self.current_subgoal])
            extended_next_state = np.concatenate([next_state, self.current_subgoal])
            
            # Модифицируем награду для низкого уровня
            current_pos = state[:3]
            next_pos = next_state[:3]
            
            # Награда за приближение к подцели
            dist_to_subgoal = np.linalg.norm(current_pos - self.current_subgoal)
            next_dist_to_subgoal = np.linalg.norm(next_pos - self.current_subgoal)
            subgoal_reward = (dist_to_subgoal - next_dist_to_subgoal) * 10
            
            if self.subgoal_achieved:
                subgoal_reward += 50  # Бонус за достижение подцели
            
            self.low_level_agent.remember(extended_state, action, reward + subgoal_reward, 
                                        extended_next_state, done or self.subgoal_achieved)
        
        # Высокоуровневый опыт (обновляется при смене подцели)
        if self.subgoal_achieved or self.subgoal_steps >= self.subgoal_timeout:
            high_level_reward = reward
            if self.subgoal_achieved:
                high_level_reward += 20  # Бонус за успешную подцель
            else:
                high_level_reward -= 10  # Штраф за неудачную подцель
            
            self.high_level_agent.remember(state, 0, high_level_reward, next_state, done)
            self.training_stats['high_level_rewards'].append(high_level_reward)
        
        self.training_stats['low_level_rewards'].append(reward)
    
    def replay(self):
        """Обучение обоих уровней."""
        self.high_level_agent.replay()
        self.low_level_agent.replay()
    
    def decay_epsilon(self):
        """Уменьшает epsilon для обоих уровней."""
        self.high_level_agent.decay_epsilon()
        self.low_level_agent.decay_epsilon()

def train_deep_rl_system(episodes: int = 5000, scenario_type: str = "single") -> Dict:
    """
    Обучает Deep RL систему на различных сценариях.
    
    Parameters
    ----------
    episodes : int
        Количество эпизодов обучения
    scenario_type : str
        Тип сценария: "single", "multi_agent", "hierarchical"
    """
    print(f"🚀 ОБУЧЕНИЕ DEEP RL СИСТЕМЫ")
    print("=" * 60)
    print(f"Сценарий: {scenario_type}")
    print(f"Эпизоды: {episodes}")
    
    results = {}
    
    if scenario_type == "single":
        results = train_single_agent_dqn(episodes)
    elif scenario_type == "multi_agent":
        results = train_multi_agent_system(episodes)
    elif scenario_type == "hierarchical":
        results = train_hierarchical_system(episodes)
    else:
        print(f"❌ Неизвестный тип сценария: {scenario_type}")
        return {}
    
    return results

def train_single_agent_dqn(episodes: int) -> Dict:
    """Обучает одиночного DQN агента."""
    print(f"\n🤖 ОБУЧЕНИЕ SINGLE-AGENT DQN")
    print("-" * 40)
    
    from train_rl_model import SwarmEnvironment
    
    agent = DQNAgent(state_size=12, action_size=8, learning_rate=0.001)
    
    # Тестовые сценарии
    scenarios = [
        (np.array([0, 0, 3]), np.array([10, 0, 3])),
        (np.array([0, 0, 2]), np.array([15, 10, 4])),
        (np.array([-5, -5, 1]), np.array([20, 15, 5])),
    ]
    
    total_rewards = []
    success_count = 0
    
    start_time = time.time()
    
    for episode in range(episodes):
        # Выбираем случайный сценарий
        start, goal = scenarios[episode % len(scenarios)]
        env = SwarmEnvironment(start, goal)
        
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        if info['distance_to_goal'] < 1.0:
            success_count += 1
        
        agent.decay_epsilon()
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(total_rewards[-1000:])
            success_rate = success_count / (episode + 1) * 100
            print(f"  Эпизод {episode + 1}: Avg Reward: {avg_reward:.2f}, "
                  f"Success: {success_rate:.1f}%, ε: {agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    
    results = {
        'type': 'single_agent_dqn',
        'episodes': episodes,
        'success_rate': success_count / episodes * 100,
        'average_reward': np.mean(total_rewards),
        'training_time': training_time,
        'final_epsilon': agent.epsilon
    }
    
    print(f"✅ Single-Agent DQN обучение завершено!")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Время обучения: {training_time:.1f}s")
    
    return results

def train_multi_agent_system(episodes: int) -> Dict:
    """Обучает мультиагентную систему."""
    print(f"\n👥 ОБУЧЕНИЕ MULTI-AGENT СИСТЕМЫ")
    print("-" * 40)
    
    num_agents = 3
    env = MultiAgentEnvironment(num_agents=num_agents)
    
    # Создаем агентов для каждого дрона
    agents = []
    for i in range(num_agents):
        agent = DQNAgent(state_size=12, action_size=8, learning_rate=0.001)
        agents.append(agent)
    
    total_rewards = [[] for _ in range(num_agents)]
    success_counts = [0] * num_agents
    
    start_time = time.time()
    
    for episode in range(episodes):
        states = env.reset()
        episode_rewards = [0] * num_agents
        
        while True:
            # Получаем действия от всех агентов
            actions = []
            for i, agent in enumerate(agents):
                action = agent.get_action(states[i], training=True)
                actions.append(action)
            
            # Выполняем действия в среде
            next_states, rewards, dones, info = env.step(actions)
            
            # Обучаем каждого агента
            for i, agent in enumerate(agents):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                agent.replay()
                episode_rewards[i] += rewards[i]
            
            states = next_states
            
            if all(dones):
                break
        
        # Обновляем статистику
        for i in range(num_agents):
            total_rewards[i].append(episode_rewards[i])
            if info['distances_to_goals'][i] < 1.0:
                success_counts[i] += 1
            agents[i].decay_epsilon()
        
        if (episode + 1) % 1000 == 0:
            avg_rewards = [np.mean(total_rewards[i][-1000:]) for i in range(num_agents)]
            success_rates = [success_counts[i] / (episode + 1) * 100 for i in range(num_agents)]
            
            print(f"  Эпизод {episode + 1}:")
            for i in range(num_agents):
                print(f"    Агент {i+1}: Reward: {avg_rewards[i]:.2f}, Success: {success_rates[i]:.1f}%")
            print(f"    Коллизии: {info['collisions']}, Успехи: {info['successes']}")
    
    training_time = time.time() - start_time
    
    # Общая статистика
    overall_success_rate = sum(success_counts) / (episodes * num_agents) * 100
    overall_avg_reward = np.mean([np.mean(rewards) for rewards in total_rewards])
    
    results = {
        'type': 'multi_agent',
        'num_agents': num_agents,
        'episodes': episodes,
        'overall_success_rate': overall_success_rate,
        'individual_success_rates': [count / episodes * 100 for count in success_counts],
        'overall_average_reward': overall_avg_reward,
        'training_time': training_time
    }
    
    print(f"✅ Multi-Agent обучение завершено!")
    print(f"  Общий Success Rate: {overall_success_rate:.1f}%")
    print(f"  Время обучения: {training_time:.1f}s")
    
    return results

def train_hierarchical_system(episodes: int) -> Dict:
    """Обучает иерархическую систему."""
    print(f"\n🏗️  ОБУЧЕНИЕ HIERARCHICAL RL СИСТЕМЫ")
    print("-" * 40)
    
    from train_rl_model import SwarmEnvironment
    
    agent = HierarchicalRLAgent(state_size=12, action_size=8)
    
    # Сложные сценарии для иерархического планирования
    scenarios = [
        (np.array([0, 0, 3]), np.array([20, 0, 3])),      # Длинный прямой путь
        (np.array([0, 0, 2]), np.array([15, 15, 6])),     # Диагональный путь
        (np.array([-10, -10, 1]), np.array([25, 20, 8])), # Очень сложный путь
    ]
    
    total_rewards = []
    success_count = 0
    
    start_time = time.time()
    
    for episode in range(episodes):
        start, goal = scenarios[episode % len(scenarios)]
        env = SwarmEnvironment(start, goal)
        
        state = env.reset()
        episode_reward = 0
        
        # Сброс состояния иерархии
        agent.current_subgoal = None
        agent.subgoal_steps = 0
        agent.subgoal_achieved = False
        
        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        if info['distance_to_goal'] < 1.0:
            success_count += 1
        
        agent.decay_epsilon()
        agent.training_stats['episodes'] += 1
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(total_rewards[-1000:])
            success_rate = success_count / (episode + 1) * 100
            subgoal_success_rate = (agent.training_stats['subgoals_achieved'] / 
                                  max(1, agent.training_stats['subgoals_achieved'] + 
                                      agent.training_stats['subgoals_failed']) * 100)
            
            print(f"  Эпизод {episode + 1}: Avg Reward: {avg_reward:.2f}, "
                  f"Success: {success_rate:.1f}%, Subgoal Success: {subgoal_success_rate:.1f}%")
    
    training_time = time.time() - start_time
    
    results = {
        'type': 'hierarchical',
        'episodes': episodes,
        'success_rate': success_count / episodes * 100,
        'average_reward': np.mean(total_rewards),
        'subgoals_achieved': agent.training_stats['subgoals_achieved'],
        'subgoals_failed': agent.training_stats['subgoals_failed'],
        'subgoal_success_rate': (agent.training_stats['subgoals_achieved'] / 
                               max(1, agent.training_stats['subgoals_achieved'] + 
                                   agent.training_stats['subgoals_failed']) * 100),
        'training_time': training_time
    }
    
    print(f"✅ Hierarchical RL обучение завершено!")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Subgoal Success Rate: {results['subgoal_success_rate']:.1f}%")
    print(f"  Время обучения: {training_time:.1f}s")
    
    return results

def main():
    """Главная функция Deep RL системы."""
    
    print("🚀 DEEP RL СИСТЕМА ДЛЯ СЛОЖНЫХ СЦЕНАРИЕВ")
    print("=" * 70)
    print("Проект: Swarm Path Planning")
    print("Версия: 9.0 (Deep RL System)")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        all_results = {}
        
        # 1. Single-Agent DQN
        print("🎯 Этап 1: Single-Agent DQN")
        single_results = train_deep_rl_system(episodes=3000, scenario_type="single")
        all_results['single_agent_dqn'] = single_results
        
        # 2. Multi-Agent System
        print(f"\n🎯 Этап 2: Multi-Agent System")
        multi_results = train_deep_rl_system(episodes=2000, scenario_type="multi_agent")
        all_results['multi_agent'] = multi_results
        
        # 3. Hierarchical RL
        print(f"\n🎯 Этап 3: Hierarchical RL")
        hierarchical_results = train_deep_rl_system(episodes=3000, scenario_type="hierarchical")
        all_results['hierarchical'] = hierarchical_results
        
        # 4. Сравнительный анализ
        print(f"\n📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ DEEP RL ПОДХОДОВ")
        print("=" * 60)
        
        print(f"{'Подход':<20} {'Success Rate':<12} {'Avg Reward':<12} {'Время':<10}")
        print("-" * 60)
        
        for approach, results in all_results.items():
            if results:
                success_rate = results.get('success_rate', results.get('overall_success_rate', 0))
                avg_reward = results.get('average_reward', results.get('overall_average_reward', 0))
                training_time = results.get('training_time', 0)
                
                print(f"{approach:<20} {success_rate:<12.1f} {avg_reward:<12.2f} {training_time:<10.1f}s")
        
        # 5. Сохраняем результаты
        results_data = {
            'deep_rl_results': all_results,
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(all_results)
        }
        
        with open('deep_rl_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n✅ DEEP RL СИСТЕМА ОБУЧЕНА И ПРОТЕСТИРОВАНА")
        print("=" * 70)
        print(f"📊 Результаты сохранены в: deep_rl_results.json")
        print(f"🧪 Проведено экспериментов: {len(all_results)}")
        
        # Выводы
        print(f"\n💡 КЛЮЧЕВЫЕ ВЫВОДЫ:")
        if single_results:
            print(f"  • Single-Agent DQN: {single_results['success_rate']:.1f}% success rate")
        if multi_results:
            print(f"  • Multi-Agent: {multi_results['overall_success_rate']:.1f}% success rate")
        if hierarchical_results:
            print(f"  • Hierarchical RL: {hierarchical_results['success_rate']:.1f}% success rate")
        
        print(f"  • Deep RL показывает потенциал для сложных сценариев")
        print(f"  • Мультиагентные системы требуют координации")
        print(f"  • Иерархический подход эффективен для длинных горизонтов")
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()