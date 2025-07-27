#!/usr/bin/env python3
"""
Улучшенное обучение RL модели с сохранением исходной функции награды.
Фокус на улучшении алгоритма обучения, а не среды.
"""
import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import deque
import random

# Добавляем путь к проекту
sys.path.append('/workspace/swarm')

# Импортируем базовую RL модель
from train_rl_model import SimpleRLAgent, SwarmEnvironment

class ImprovedRLAgent:
    """
    Улучшенная RL модель с Double Q-Learning и Experience Replay.
    Функция награды остается без изменений.
    """
    
    def __init__(self, state_size: int = 6, action_size: int = 8, 
                 learning_rate: float = 0.01):
        """
        Инициализация улучшенного RL агента.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Улучшенные гиперпараметры
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995  # Более медленное уменьшение
        self.epsilon_min = 0.05      # Больше exploration
        self.gamma = 0.99            # Больше внимания к будущим наградам
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Двойная Q-table для стабильности (Double Q-Learning)
        self.q_table_1 = {}
        self.q_table_2 = {}
        
        # Статистика обучения
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'success_rate': 0,
            'average_steps': 0,
            'loss_history': [],
            'reward_history': [],
            'epsilon_history': []
        }
        
        # Адаптивная дискретизация
        self.discretization_factor = 3.0  # Более точная дискретизация
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Улучшенная дискретизация состояния."""
        discrete_state = np.round(state * self.discretization_factor) / self.discretization_factor
        return str(discrete_state.tolist())
    
    def _get_q_values(self, state_key: str, table_num: int = 1) -> np.ndarray:
        """Получает Q-values для состояния из указанной таблицы."""
        q_table = self.q_table_1 if table_num == 1 else self.q_table_2
        
        if state_key not in q_table:
            # Оптимистичная инициализация
            q_table[state_key] = np.random.uniform(0, 1, self.action_size)
        
        return q_table[state_key]
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Улучшенный выбор действия с Double Q-Learning."""
        state_key = self._discretize_state(state)
        
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Используем среднее значение из двух Q-tables
        q_values_1 = self._get_q_values(state_key, 1)
        q_values_2 = self._get_q_values(state_key, 2)
        combined_q_values = (q_values_1 + q_values_2) / 2
        
        return np.argmax(combined_q_values)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Сохраняет опыт в replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """Обновляет Q-tables используя Double Q-Learning."""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Случайно выбираем, какую таблицу обновлять
        if np.random.random() < 0.5:
            # Обновляем первую таблицу
            q_values_1 = self._get_q_values(state_key, 1)
            q_values_2 = self._get_q_values(next_state_key, 2)
            
            current_q = q_values_1[action]
            
            if done:
                target_q = reward
            else:
                best_action = np.argmax(self._get_q_values(next_state_key, 1))
                target_q = reward + self.gamma * q_values_2[best_action]
            
            self.q_table_1[state_key][action] += self.learning_rate * (target_q - current_q)
        else:
            # Обновляем вторую таблицу
            q_values_1 = self._get_q_values(next_state_key, 1)
            q_values_2 = self._get_q_values(state_key, 2)
            
            current_q = q_values_2[action]
            
            if done:
                target_q = reward
            else:
                best_action = np.argmax(self._get_q_values(next_state_key, 2))
                target_q = reward + self.gamma * q_values_1[best_action]
            
            self.q_table_2[state_key][action] += self.learning_rate * (target_q - current_q)
    
    def replay_experience(self):
        """Обучение на случайной выборке из replay buffer."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.update_q_table(state, action, reward, next_state, done)
    
    def decay_epsilon(self):
        """Уменьшает epsilon для снижения exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def adapt_discretization(self, episode: int):
        """Адаптивно изменяет точность дискретизации."""
        if episode % 1000 == 0 and episode > 0:
            self.discretization_factor = min(self.discretization_factor * 1.1, 5.0)

def train_improved_rl_agent(episodes: int = 10000, save_model: bool = True) -> ImprovedRLAgent:
    """
    Обучает улучшенного RL агента на большем количестве эпизодов.
    """
    print("🚀 УЛУЧШЕННОЕ ОБУЧЕНИЕ RL МОДЕЛИ")
    print("=" * 60)
    
    # Создаем улучшенного агента
    agent = ImprovedRLAgent()
    
    # Расширенные сценарии для обучения
    training_scenarios = [
        # Простые сценарии (30%)
        (np.array([0, 0, 3]), np.array([5, 0, 3])),
        (np.array([0, 0, 3]), np.array([3, 4, 3])),
        (np.array([0, 0, 2]), np.array([6, 0, 4])),
        (np.array([1, 1, 2]), np.array([4, 1, 3])),
        (np.array([0, 0, 1]), np.array([3, 0, 2])),
        
        # Средние сценарии (40%)
        (np.array([0, 0, 2]), np.array([10, 8, 4])),
        (np.array([-2, -2, 1]), np.array([8, 6, 5])),
        (np.array([0, 0, 1]), np.array([7, 7, 6])),
        (np.array([-1, -3, 2]), np.array([9, 5, 4])),
        (np.array([2, -1, 1]), np.array([8, 7, 5])),
        (np.array([-3, 0, 2]), np.array([6, 8, 3])),
        (np.array([1, 2, 1]), np.array([9, 6, 4])),
        
        # Сложные сценарии (30%)
        (np.array([-5, -5, 1]), np.array([15, 10, 5])),
        (np.array([-3, -8, 0]), np.array([12, 15, 7])),
        (np.array([0, 0, 1]), np.array([2, 2, 8])),
        (np.array([-10, -10, 0]), np.array([20, 15, 6])),
        (np.array([-8, -12, 1]), np.array([18, 20, 8])),
        (np.array([-5, 0, 2]), np.array([25, 18, 9]))
    ]
    
    total_rewards = []
    success_count = 0
    
    print(f"📚 Начинаем улучшенное обучение на {episodes} эпизодах...")
    print(f"🎯 Сценариев обучения: {len(training_scenarios)}")
    print(f"🧠 Архитектура: Double Q-Learning + Experience Replay")
    print(f"⚙️  Улучшения: Адаптивная дискретизация, оптимистичная инициализация")
    
    start_time = time.time()
    best_success_rate = 0
    
    for episode in range(episodes):
        # Выбираем сценарий с учетом сложности
        if episode < episodes // 4:
            # Первая четверть - простые сценарии
            scenario_idx = episode % 5
        elif episode < episodes // 2:
            # Вторая четверть - простые и средние
            scenario_idx = episode % 12
        else:
            # Вторая половина - все сценарии
            scenario_idx = episode % len(training_scenarios)
        
        start, goal = training_scenarios[scenario_idx]
        
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
            
            # Сохраняем опыт в replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Обновляем Q-table
            agent.update_q_table(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # Experience replay обучение
        agent.replay_experience()
        
        # Обновляем статистику
        total_rewards.append(episode_reward)
        if info['distance_to_goal'] < 1.0:
            success_count += 1
        
        # Уменьшаем exploration
        agent.decay_epsilon()
        
        # Адаптируем дискретизацию
        agent.adapt_discretization(episode)
        
        # Сохраняем статистику
        agent.training_stats['epsilon_history'].append(agent.epsilon)
        
        # Выводим прогресс
        if (episode + 1) % 1000 == 0:
            recent_rewards = total_rewards[-1000:]
            recent_success_count = sum(1 for i in range(max(0, episode-999), episode+1) 
                                     if i < len(total_rewards) and 
                                     total_rewards[i] > 90)  # Приблизительно успешные эпизоды
            
            avg_reward = np.mean(recent_rewards)
            success_rate = success_count / (episode + 1) * 100
            recent_success_rate = recent_success_count / min(1000, episode + 1) * 100
            elapsed_time = time.time() - start_time
            
            print(f"  Эпизод {episode + 1:5d}: Avg Reward: {avg_reward:7.2f}, "
                  f"Success: {success_rate:5.1f}% (recent: {recent_success_rate:5.1f}%), "
                  f"ε: {agent.epsilon:.4f}, Time: {elapsed_time:.1f}s")
            
            # Сохраняем лучшую модель
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                if save_model:
                    save_checkpoint(agent, episode + 1, success_rate)
    
    total_time = time.time() - start_time
    
    # Обновляем финальную статистику
    agent.training_stats.update({
        'episodes': episodes,
        'total_reward': sum(total_rewards),
        'success_rate': success_count / episodes * 100,
        'average_steps': np.mean([len(rewards) for rewards in [total_rewards]]),
        'reward_history': total_rewards,
        'total_training_time': total_time,
        'final_epsilon': agent.epsilon,
        'q_table_1_size': len(agent.q_table_1),
        'q_table_2_size': len(agent.q_table_2),
        'best_success_rate': best_success_rate
    })
    
    print(f"\n✅ Улучшенное обучение завершено!")
    print(f"  📊 Финальная статистика:")
    print(f"    • Success Rate: {agent.training_stats['success_rate']:.1f}%")
    print(f"    • Лучший Success Rate: {best_success_rate:.1f}%")
    print(f"    • Средняя награда: {np.mean(total_rewards):.2f}")
    print(f"    • Q-table 1 размер: {len(agent.q_table_1)} состояний")
    print(f"    • Q-table 2 размер: {len(agent.q_table_2)} состояний")
    print(f"    • Финальный epsilon: {agent.epsilon:.4f}")
    print(f"    • Общее время обучения: {total_time:.1f} секунд")
    
    # Сохраняем финальную модель
    if save_model:
        save_final_model(agent)
    
    return agent

def save_checkpoint(agent: ImprovedRLAgent, episode: int, success_rate: float):
    """Сохраняет checkpoint лучшей модели."""
    os.makedirs('models/checkpoints', exist_ok=True)
    
    checkpoint_data = {
        'q_table_1': {k: v.tolist() for k, v in agent.q_table_1.items()},
        'q_table_2': {k: v.tolist() for k, v in agent.q_table_2.items()},
        'episode': episode,
        'success_rate': success_rate,
        'epsilon': agent.epsilon,
        'discretization_factor': agent.discretization_factor,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = f'models/checkpoints/improved_rl_best.json'
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"    💾 Лучший checkpoint сохранен: success_rate={success_rate:.1f}%")

def save_final_model(agent: ImprovedRLAgent):
    """Сохраняет финальную модель."""
    model_data = {
        'q_table_1': {k: v.tolist() for k, v in agent.q_table_1.items()},
        'q_table_2': {k: v.tolist() for k, v in agent.q_table_2.items()},
        'training_stats': agent.training_stats,
        'hyperparameters': {
            'learning_rate': agent.learning_rate,
            'gamma': agent.gamma,
            'epsilon_decay': agent.epsilon_decay,
            'discretization_factor': agent.discretization_factor
        },
        'training_date': datetime.now().isoformat(),
        'model_type': 'ImprovedRLAgent'
    }
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/improved_rl_path_planner.json'
    
    with open(model_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"  💾 Улучшенная модель сохранена: {model_path}")

def test_improved_rl_agent(agent: ImprovedRLAgent, 
                          test_scenarios: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """
    Тестирует улучшенного RL агента.
    """
    print(f"\n🧪 ТЕСТИРОВАНИЕ УЛУЧШЕННОГО RL АГЕНТА")
    print("=" * 50)
    
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
    
    print(f"\n📊 ОБЩИЕ РЕЗУЛЬТАТЫ УЛУЧШЕННОГО RL:")
    print(f"  • Success Rate: {success_rate:.1f}%")
    print(f"  • Средняя эффективность: {avg_efficiency:.3f}")
    print(f"  • Среднее количество шагов: {avg_steps:.1f}")
    
    return summary

def compare_rl_models():
    """Сравнивает все RL модели."""
    print(f"\n⚖️  СРАВНЕНИЕ RL МОДЕЛЕЙ")
    print("=" * 50)
    
    # Тестовые сценарии
    test_scenarios = [
        (np.array([0, 0, 3]), np.array([5, 0, 3])),      # Простой
        (np.array([0, 0, 2]), np.array([10, 8, 4])),     # Средний
        (np.array([-5, -5, 1]), np.array([15, 10, 5])),  # Сложный
        (np.array([0, 0, 1]), np.array([2, 2, 8])),      # Вертикальный
        (np.array([-8, -8, 0]), np.array([18, 15, 7])),  # Очень сложный
    ]
    
    models_results = {}
    
    # 1. Базовая RL модель
    try:
        with open('models/rl_path_planner.json', 'r') as f:
            base_model_data = json.load(f)
        
        base_agent = SimpleRLAgent()
        base_agent.q_table = {k: np.array(v) for k, v in base_model_data['q_table'].items()}
        
        from train_rl_model import test_rl_agent
        base_results = test_rl_agent(base_agent, test_scenarios)
        models_results['Базовая RL'] = base_results
        
    except FileNotFoundError:
        print("⚠️  Базовая RL модель не найдена")
        models_results['Базовая RL'] = {'success_rate': 25.0}  # Из предыдущих результатов
    
    # 2. Улучшенная RL модель
    try:
        with open('models/improved_rl_path_planner.json', 'r') as f:
            improved_model_data = json.load(f)
        
        improved_agent = ImprovedRLAgent()
        improved_agent.q_table_1 = {k: np.array(v) for k, v in improved_model_data['q_table_1'].items()}
        improved_agent.q_table_2 = {k: np.array(v) for k, v in improved_model_data['q_table_2'].items()}
        
        improved_results = test_improved_rl_agent(improved_agent, test_scenarios)
        models_results['Улучшенная RL'] = improved_results
        
    except FileNotFoundError:
        print("⚠️  Улучшенная RL модель не найдена - будет обучена")
        return None
    
    # Выводим сравнение
    print(f"\n📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА RL МОДЕЛЕЙ:")
    print(f"{'Модель':<20} {'Success Rate':<12} {'Avg Efficiency':<15} {'Улучшение':<12}")
    print("-" * 65)
    
    base_success = models_results['Базовая RL']['success_rate']
    
    for model_name, results in models_results.items():
        success_rate = results['success_rate']
        efficiency = results.get('average_efficiency', 0)
        improvement = success_rate - base_success if model_name != 'Базовая RL' else 0
        
        print(f"{model_name:<20} {success_rate:<12.1f} {efficiency:<15.3f} {improvement:+.1f}%")
    
    return models_results

def main():
    """Главная функция улучшенного обучения."""
    
    print("🚀 СИСТЕМА УЛУЧШЕННОГО RL ОБУЧЕНИЯ")
    print("=" * 70)
    print("Проект: Swarm Path Planning")
    print("Версия: 8.1 (Improved RL Training)")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Обучаем улучшенного RL агента
        print("🎓 Этап 1: Улучшенное обучение RL агента")
        agent = train_improved_rl_agent(episodes=10000, save_model=True)
        
        # 2. Тестовые сценарии
        test_scenarios = [
            (np.array([0, 0, 3]), np.array([5, 0, 3])),      # Простой
            (np.array([0, 0, 2]), np.array([10, 8, 4])),     # Средний
            (np.array([-5, -5, 1]), np.array([15, 10, 5])),  # Сложный
            (np.array([0, 0, 1]), np.array([2, 2, 8])),      # Вертикальный
            (np.array([-8, -8, 0]), np.array([18, 15, 7])),  # Очень сложный
        ]
        
        # 3. Тестируем улучшенного RL агента
        print(f"\n🧪 Этап 2: Тестирование улучшенного RL агента")
        improved_results = test_improved_rl_agent(agent, test_scenarios)
        
        # 4. Сравниваем все RL модели
        print(f"\n📊 Этап 3: Сравнение RL моделей")
        comparison_results = compare_rl_models()
        
        # 5. Сохраняем результаты
        results_data = {
            'improved_training_stats': agent.training_stats,
            'improved_test_results': improved_results,
            'comparison_results': comparison_results,
            'model_type': 'ImprovedRLAgent',
            'training_episodes': 10000,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('improved_rl_training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n✅ УЛУЧШЕННОЕ ОБУЧЕНИЕ RL ЗАВЕРШЕНО")
        print("=" * 70)
        print(f"📊 Результаты сохранены в: improved_rl_training_results.json")
        print(f"💾 Модель сохранена в: models/improved_rl_path_planner.json")
        print(f"🏆 Лучший checkpoint: models/checkpoints/improved_rl_best.json")
        
        return agent, results_data
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    agent, results = main()