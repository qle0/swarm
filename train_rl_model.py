#!/usr/bin/env python3
"""
Simplified RL model training for swarm path planning.
This script simulates RL training without requiring heavy dependencies.
"""
import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Добавляем путь к проекту
sys.path.append('/workspace/swarm')

# Импортируем наши планировщики для сравнения
from test_algorithms_simple import create_planner
from hybrid_algorithm import create_hybrid_planner

class SimpleRLAgent:
    """
    Упрощенная RL модель для планирования пути.
    Использует Q-learning подход с дискретным пространством действий.
    """
    
    def __init__(self, state_size: int = 6, action_size: int = 8, learning_rate: float = 0.1):
        """
        Инициализация RL агента.
        
        Parameters
        ----------
        state_size : int
            Размер пространства состояний (x, y, z, goal_x, goal_y, goal_z)
        action_size : int
            Размер пространства действий (8 направлений движения)
        learning_rate : float
            Скорость обучения
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor
        
        # Q-table (упрощенная версия)
        self.q_table = {}
        
        # Статистика обучения
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'success_rate': 0,
            'average_steps': 0,
            'loss_history': [],
            'reward_history': []
        }
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Дискретизирует непрерывное состояние."""
        # Округляем координаты до ближайшего метра
        discrete_state = np.round(state).astype(int)
        return str(discrete_state.tolist())
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Выбирает действие на основе epsilon-greedy стратегии."""
        state_key = self._discretize_state(state)
        
        if training and np.random.random() <= self.epsilon:
            # Exploration: случайное действие
            return np.random.randint(0, self.action_size)
        
        # Exploitation: лучшее известное действие
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """Обновляет Q-table используя Q-learning."""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # Update Q-value
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Уменьшает epsilon для снижения exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SwarmEnvironment:
    """
    Упрощенная среда для обучения RL агента планированию пути.
    """
    
    def __init__(self, start: np.ndarray, goal: np.ndarray):
        """
        Инициализация среды.
        
        Parameters
        ----------
        start : np.ndarray
            Начальная позиция
        goal : np.ndarray
            Целевая позиция
        """
        self.start = start.copy()
        self.goal = goal.copy()
        self.current_pos = start.copy()
        self.max_steps = 100
        self.current_step = 0
        
        # Действия: 8 направлений движения + вверх/вниз
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
    
    def reset(self) -> np.ndarray:
        """Сбрасывает среду в начальное состояние."""
        self.current_pos = self.start.copy()
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Возвращает текущее состояние."""
        return np.concatenate([self.current_pos, self.goal])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Выполняет действие в среде."""
        # Применяем действие
        step_size = 1.0  # Размер шага в метрах
        movement = self.actions[action] * step_size
        new_pos = self.current_pos + movement
        
        # Ограничиваем движение в разумных пределах
        new_pos = np.clip(new_pos, [-50, -50, 0], [50, 50, 20])
        
        self.current_pos = new_pos
        self.current_step += 1
        
        # Вычисляем награду
        reward = self._calculate_reward()
        
        # Проверяем условия завершения
        done = self._is_done()
        
        info = {
            'distance_to_goal': np.linalg.norm(self.current_pos - self.goal),
            'steps': self.current_step
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self) -> float:
        """Вычисляет награду за текущее состояние."""
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

def train_rl_agent(episodes: int = 1000, save_model: bool = True) -> SimpleRLAgent:
    """
    Обучает RL агента на различных сценариях.
    
    Parameters
    ----------
    episodes : int
        Количество эпизодов обучения
    save_model : bool
        Сохранять ли модель после обучения
        
    Returns
    -------
    SimpleRLAgent
        Обученный агент
    """
    print("🤖 ОБУЧЕНИЕ RL МОДЕЛИ ДЛЯ ПЛАНИРОВАНИЯ ПУТИ")
    print("=" * 60)
    
    # Создаем агента
    agent = SimpleRLAgent()
    
    # Различные сценарии для обучения
    training_scenarios = [
        (np.array([0, 0, 3]), np.array([5, 0, 3])),      # Простой
        (np.array([0, 0, 2]), np.array([10, 8, 4])),     # Средний
        (np.array([-5, -5, 1]), np.array([15, 10, 5])),  # Сложный
        (np.array([0, 0, 1]), np.array([2, 2, 8])),      # Вертикальный
        (np.array([-10, -10, 0]), np.array([20, 15, 6])) # Очень сложный
    ]
    
    total_rewards = []
    success_count = 0
    
    print(f"📚 Начинаем обучение на {episodes} эпизодах...")
    print(f"🎯 Сценариев обучения: {len(training_scenarios)}")
    
    for episode in range(episodes):
        # Выбираем случайный сценарий
        start, goal = training_scenarios[episode % len(training_scenarios)]
        
        # Создаем среду
        env = SwarmEnvironment(start, goal)
        state = env.reset()
        
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Выбираем действие
            action = agent.get_action(state, training=True)
            
            # Выполняем действие
            next_state, reward, done, info = env.step(action)
            
            # Обновляем Q-table
            agent.update_q_table(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # Обновляем статистику
        total_rewards.append(episode_reward)
        if info['distance_to_goal'] < 1.0:
            success_count += 1
        
        # Уменьшаем exploration
        agent.decay_epsilon()
        
        # Выводим прогресс
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            success_rate = success_count / (episode + 1) * 100
            print(f"  Эпизод {episode + 1:4d}: Avg Reward: {avg_reward:6.2f}, "
                  f"Success Rate: {success_rate:5.1f}%, Epsilon: {agent.epsilon:.3f}")
    
    # Обновляем финальную статистику
    agent.training_stats.update({
        'episodes': episodes,
        'total_reward': sum(total_rewards),
        'success_rate': success_count / episodes * 100,
        'average_steps': np.mean([len(rewards) for rewards in [total_rewards]]),
        'reward_history': total_rewards
    })
    
    print(f"\n✅ Обучение завершено!")
    print(f"  📊 Финальная статистика:")
    print(f"    • Success Rate: {agent.training_stats['success_rate']:.1f}%")
    print(f"    • Средняя награда: {np.mean(total_rewards):.2f}")
    print(f"    • Размер Q-table: {len(agent.q_table)} состояний")
    print(f"    • Финальный epsilon: {agent.epsilon:.3f}")
    
    # Сохраняем модель
    if save_model:
        model_data = {
            'q_table': agent.q_table,
            'training_stats': agent.training_stats,
            'hyperparameters': {
                'learning_rate': agent.learning_rate,
                'gamma': agent.gamma,
                'epsilon_decay': agent.epsilon_decay
            },
            'training_date': datetime.now().isoformat()
        }
        
        os.makedirs('models', exist_ok=True)
        model_path = 'models/rl_path_planner.json'
        
        with open(model_path, 'w') as f:
            # Конвертируем numpy arrays в списки для JSON
            json_data = {}
            for key, value in model_data.items():
                if key == 'q_table':
                    json_data[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                     for k, v in value.items()}
                else:
                    json_data[key] = value
            json.dump(json_data, f, indent=2)
        
        print(f"  💾 Модель сохранена: {model_path}")
    
    return agent

def test_rl_agent(agent: SimpleRLAgent, test_scenarios: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """
    Тестирует обученного RL агента на тестовых сценариях.
    
    Parameters
    ----------
    agent : SimpleRLAgent
        Обученный агент
    test_scenarios : List[Tuple[np.ndarray, np.ndarray]]
        Список тестовых сценариев (start, goal)
        
    Returns
    -------
    Dict
        Результаты тестирования
    """
    print(f"\n🧪 ТЕСТИРОВАНИЕ RL АГЕНТА")
    print("=" * 40)
    
    results = []
    
    for i, (start, goal) in enumerate(test_scenarios):
        print(f"  📋 Сценарий {i+1}: {start} → {goal}")
        
        env = SwarmEnvironment(start, goal)
        state = env.reset()
        
        path = [start.copy()]
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
        direct_distance = np.linalg.norm(goal - start)
        efficiency = direct_distance / path_length if path_length > 0 else 0
        
        result = {
            'scenario': i + 1,
            'start': start,
            'goal': goal,
            'success': success,
            'steps': steps,
            'total_reward': total_reward,
            'path_length': path_length,
            'direct_distance': direct_distance,
            'efficiency': efficiency,
            'final_distance': info['distance_to_goal']
        }
        
        results.append(result)
        
        print(f"    ✅ Success: {success}, Steps: {steps}, "
              f"Efficiency: {efficiency:.3f}, Reward: {total_reward:.1f}")
    
    # Общая статистика
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    avg_efficiency = np.mean([r['efficiency'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    summary = {
        'results': results,
        'success_rate': success_rate,
        'average_efficiency': avg_efficiency,
        'average_steps': avg_steps,
        'total_scenarios': len(test_scenarios)
    }
    
    print(f"\n📊 ОБЩИЕ РЕЗУЛЬТАТЫ:")
    print(f"  • Success Rate: {success_rate:.1f}%")
    print(f"  • Средняя эффективность: {avg_efficiency:.3f}")
    print(f"  • Среднее количество шагов: {avg_steps:.1f}")
    
    return summary

def compare_with_other_algorithms(test_scenarios: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """
    Сравнивает RL агента с другими алгоритмами планирования.
    
    Parameters
    ----------
    test_scenarios : List[Tuple[np.ndarray, np.ndarray]]
        Тестовые сценарии
        
    Returns
    -------
    Dict
        Результаты сравнения
    """
    print(f"\n⚖️  СРАВНЕНИЕ С ДРУГИМИ АЛГОРИТМАМИ")
    print("=" * 50)
    
    # Загружаем обученную RL модель
    try:
        with open('models/rl_path_planner.json', 'r') as f:
            model_data = json.load(f)
        
        # Восстанавливаем агента
        agent = SimpleRLAgent()
        agent.q_table = {k: np.array(v) for k, v in model_data['q_table'].items()}
        agent.training_stats = model_data['training_stats']
        
        print("✅ RL модель загружена")
    except FileNotFoundError:
        print("❌ RL модель не найдена, обучаем новую...")
        agent = train_rl_agent(episodes=500, save_model=True)
    
    # Тестируем RL агента
    rl_results = test_rl_agent(agent, test_scenarios)
    
    # Тестируем другие алгоритмы
    algorithms = ['astar', 'dijkstra', 'rrt_optimized', 'adaptive_rrt']
    algorithm_results = {}
    
    for algorithm in algorithms:
        print(f"\n🔄 Тестирование {algorithm}...")
        
        results = []
        for start, goal in test_scenarios:
            planner = create_planner(algorithm, start, goal)
            
            start_time = time.time()
            path = planner.plan()
            planning_time = time.time() - start_time
            
            score = planner.calculate_score(path)
            success = score > 0
            
            results.append({
                'success': success,
                'score': score,
                'planning_time': planning_time,
                'path_length': len(path)
            })
        
        success_rate = sum(r['success'] for r in results) / len(results) * 100
        avg_score = np.mean([r['score'] for r in results])
        avg_time = np.mean([r['planning_time'] for r in results])
        
        algorithm_results[algorithm] = {
            'success_rate': success_rate,
            'average_score': avg_score,
            'average_time': avg_time,
            'results': results
        }
        
        print(f"  Success Rate: {success_rate:.1f}%, Avg Score: {avg_score:.3f}, Avg Time: {avg_time:.3f}s")
    
    # Тестируем гибридный алгоритм
    print(f"\n🔄 Тестирование hybrid_adaptive...")
    hybrid_results = []
    
    for start, goal in test_scenarios:
        planner = create_hybrid_planner(start, goal, strategy="adaptive", time_budget=0.5)
        
        start_time = time.time()
        path = planner.plan()
        planning_time = time.time() - start_time
        
        score = planner.calculate_score(path)
        success = score > 0
        
        hybrid_results.append({
            'success': success,
            'score': score,
            'planning_time': planning_time,
            'path_length': len(path)
        })
    
    hybrid_success_rate = sum(r['success'] for r in hybrid_results) / len(hybrid_results) * 100
    hybrid_avg_score = np.mean([r['score'] for r in hybrid_results])
    hybrid_avg_time = np.mean([r['planning_time'] for r in hybrid_results])
    
    algorithm_results['hybrid_adaptive'] = {
        'success_rate': hybrid_success_rate,
        'average_score': hybrid_avg_score,
        'average_time': hybrid_avg_time,
        'results': hybrid_results
    }
    
    print(f"  Success Rate: {hybrid_success_rate:.1f}%, Avg Score: {hybrid_avg_score:.3f}, Avg Time: {hybrid_avg_time:.3f}s")
    
    # Сравнительная таблица
    print(f"\n📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА:")
    print(f"{'Алгоритм':<20} {'Success Rate':<12} {'Avg Score':<10} {'Avg Time':<10}")
    print("-" * 55)
    
    # RL агент
    print(f"{'RL Agent':<20} {rl_results['success_rate']:<12.1f} {'N/A':<10} {'N/A':<10}")
    
    # Другие алгоритмы
    for alg, data in algorithm_results.items():
        print(f"{alg:<20} {data['success_rate']:<12.1f} {data['average_score']:<10.3f} {data['average_time']:<10.3f}")
    
    return {
        'rl_results': rl_results,
        'algorithm_results': algorithm_results,
        'test_scenarios': test_scenarios
    }

def main():
    """Главная функция."""
    
    print("🤖 СИСТЕМА ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ RL МОДЕЛИ")
    print("=" * 70)
    print("Проект: Swarm Path Planning")
    print("Версия: 7.0 (RL Training)")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Обучаем RL агента
        print("🎓 Этап 1: Обучение RL агента")
        agent = train_rl_agent(episodes=1000, save_model=True)
        
        # 2. Тестовые сценарии
        test_scenarios = [
            (np.array([0, 0, 3]), np.array([5, 0, 3])),      # Простой
            (np.array([0, 0, 2]), np.array([10, 8, 4])),     # Средний
            (np.array([-5, -5, 1]), np.array([15, 10, 5])),  # Сложный
            (np.array([0, 0, 1]), np.array([2, 2, 8])),      # Вертикальный
        ]
        
        # 3. Тестируем RL агента
        print(f"\n🧪 Этап 2: Тестирование RL агента")
        rl_results = test_rl_agent(agent, test_scenarios)
        
        # 4. Сравниваем с другими алгоритмами
        print(f"\n⚖️  Этап 3: Сравнение с другими алгоритмами")
        comparison_results = compare_with_other_algorithms(test_scenarios)
        
        # 5. Сохраняем результаты
        results_data = {
            'training_stats': agent.training_stats,
            'test_results': rl_results,
            'comparison_results': comparison_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('rl_training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n✅ ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 70)
        print(f"📊 Результаты сохранены в: rl_training_results.json")
        print(f"💾 Модель сохранена в: models/rl_path_planner.json")
        
        return agent, results_data
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    agent, results = main()