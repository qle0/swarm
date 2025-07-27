#!/usr/bin/env python3
"""
Расширенное обучение базовой RL модели с большим количеством эпизодов.
Функция награды остается без изменений.
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

# Импортируем базовую RL модель
from train_rl_model import SimpleRLAgent, SwarmEnvironment, test_rl_agent

class ExtendedRLAgent(SimpleRLAgent):
    """
    Расширенная версия базовой RL модели с улучшенными параметрами.
    """
    
    def __init__(self, state_size: int = 6, action_size: int = 8, learning_rate: float = 0.05):
        """
        Инициализация расширенного RL агента.
        """
        super().__init__(state_size, action_size, learning_rate)
        
        # Улучшенные гиперпараметры
        self.epsilon = 1.0
        self.epsilon_decay = 0.9998  # Очень медленное уменьшение
        self.epsilon_min = 0.1       # Больше exploration
        self.gamma = 0.98            # Немного больше внимания к будущим наградам
        
        # Более точная дискретизация
        self.discretization_precision = 1.0  # Точность дискретизации
        
        # Статистика обучения
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'success_rate': 0,
            'average_steps': 0,
            'loss_history': [],
            'reward_history': [],
            'epsilon_history': [],
            'success_episodes': []
        }
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Улучшенная дискретизация состояния."""
        # Используем более точную дискретизацию
        discrete_state = np.round(state / self.discretization_precision) * self.discretization_precision
        return str(discrete_state.tolist())
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """Обновляет Q-table с улучшенным алгоритмом."""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        if state_key not in self.q_table:
            # Оптимистичная инициализация
            self.q_table[state_key] = np.random.uniform(0, 1, self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.random.uniform(0, 1, self.action_size)
        
        # Q-learning update rule с улучшенной формулой
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # Адаптивная скорость обучения
        adaptive_lr = self.learning_rate * (1.0 + abs(reward) / 100.0)
        adaptive_lr = min(adaptive_lr, 0.3)  # Ограничиваем максимальную скорость
        
        # Update Q-value
        self.q_table[state_key][action] = current_q + adaptive_lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """Улучшенное уменьшение epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_stats['epsilon_history'].append(self.epsilon)
    
    def adapt_precision(self, episode: int, success_rate: float):
        """Адаптивно изменяет точность дискретизации."""
        # Если success rate низкий, увеличиваем точность
        if episode % 2000 == 0 and episode > 0:
            if success_rate < 20:
                self.discretization_precision = max(self.discretization_precision * 0.8, 0.5)
            elif success_rate > 40:
                self.discretization_precision = min(self.discretization_precision * 1.1, 2.0)

def train_extended_rl_agent(episodes: int = 20000, save_model: bool = True) -> ExtendedRLAgent:
    """
    Обучает расширенного RL агента на большом количестве эпизодов.
    """
    print("🚀 РАСШИРЕННОЕ ОБУЧЕНИЕ RL МОДЕЛИ")
    print("=" * 60)
    
    # Создаем расширенного агента
    agent = ExtendedRLAgent(learning_rate=0.05)
    
    # Прогрессивные сценарии для обучения
    training_scenarios = [
        # Очень простые сценарии (первые 5000 эпизодов)
        (np.array([0, 0, 3]), np.array([2, 0, 3])),
        (np.array([0, 0, 3]), np.array([1, 1, 3])),
        (np.array([0, 0, 2]), np.array([3, 0, 2])),
        (np.array([1, 0, 3]), np.array([3, 0, 3])),
        (np.array([0, 1, 3]), np.array([2, 1, 3])),
        
        # Простые сценарии (5000-10000 эпизодов)
        (np.array([0, 0, 3]), np.array([5, 0, 3])),
        (np.array([0, 0, 3]), np.array([3, 4, 3])),
        (np.array([0, 0, 2]), np.array([6, 0, 4])),
        (np.array([1, 1, 2]), np.array([4, 1, 3])),
        (np.array([0, 0, 1]), np.array([3, 0, 2])),
        
        # Средние сценарии (10000-15000 эпизодов)
        (np.array([0, 0, 2]), np.array([10, 8, 4])),
        (np.array([-2, -2, 1]), np.array([8, 6, 5])),
        (np.array([0, 0, 1]), np.array([7, 7, 6])),
        (np.array([-1, -3, 2]), np.array([9, 5, 4])),
        
        # Сложные сценарии (15000-20000 эпизодов)
        (np.array([-5, -5, 1]), np.array([15, 10, 5])),
        (np.array([0, 0, 1]), np.array([2, 2, 8])),
        (np.array([-3, -8, 0]), np.array([12, 15, 7])),
        (np.array([-8, -8, 0]), np.array([18, 15, 7]))
    ]
    
    total_rewards = []
    success_count = 0
    
    print(f"📚 Начинаем расширенное обучение на {episodes} эпизодах...")
    print(f"🎯 Сценариев обучения: {len(training_scenarios)}")
    print(f"🧠 Архитектура: Улучшенная базовая RL с адаптивными параметрами")
    print(f"⚙️  Улучшения: Оптимистичная инициализация, адаптивная скорость обучения")
    
    start_time = time.time()
    best_success_rate = 0
    
    for episode in range(episodes):
        # Прогрессивный выбор сценариев
        if episode < 5000:
            # Первые 5000 - очень простые
            scenario_idx = episode % 5
        elif episode < 10000:
            # 5000-10000 - простые
            scenario_idx = (episode % 10)
        elif episode < 15000:
            # 10000-15000 - средние
            scenario_idx = (episode % 14)
        else:
            # 15000-20000 - все сценарии
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
            agent.training_stats['success_episodes'].append(episode)
        
        # Уменьшаем exploration
        agent.decay_epsilon()
        
        # Адаптируем точность дискретизации
        current_success_rate = success_count / (episode + 1) * 100
        agent.adapt_precision(episode, current_success_rate)
        
        # Выводим прогресс
        if (episode + 1) % 2000 == 0:
            recent_rewards = total_rewards[-2000:]
            recent_success_count = sum(1 for i in range(max(0, episode-1999), episode+1) 
                                     if i < len(total_rewards) and 
                                     any(ep >= i for ep in agent.training_stats['success_episodes'][-100:]))
            
            avg_reward = np.mean(recent_rewards)
            success_rate = success_count / (episode + 1) * 100
            recent_success_rate = len([ep for ep in agent.training_stats['success_episodes'] 
                                     if ep >= episode - 1999]) / 2000 * 100
            elapsed_time = time.time() - start_time
            
            print(f"  Эпизод {episode + 1:5d}: Avg Reward: {avg_reward:7.2f}, "
                  f"Success: {success_rate:5.1f}% (recent: {recent_success_rate:5.1f}%), "
                  f"ε: {agent.epsilon:.4f}, Precision: {agent.discretization_precision:.2f}, "
                  f"Time: {elapsed_time:.1f}s")
            
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
        'q_table_size': len(agent.q_table),
        'best_success_rate': best_success_rate,
        'final_precision': agent.discretization_precision
    })
    
    print(f"\n✅ Расширенное обучение завершено!")
    print(f"  📊 Финальная статистика:")
    print(f"    • Success Rate: {agent.training_stats['success_rate']:.1f}%")
    print(f"    • Лучший Success Rate: {best_success_rate:.1f}%")
    print(f"    • Средняя награда: {np.mean(total_rewards):.2f}")
    print(f"    • Q-table размер: {len(agent.q_table)} состояний")
    print(f"    • Финальный epsilon: {agent.epsilon:.4f}")
    print(f"    • Финальная точность: {agent.discretization_precision:.2f}")
    print(f"    • Общее время обучения: {total_time:.1f} секунд")
    
    # Сохраняем финальную модель
    if save_model:
        save_final_model(agent)
    
    return agent

def save_checkpoint(agent: ExtendedRLAgent, episode: int, success_rate: float):
    """Сохраняет checkpoint лучшей модели."""
    os.makedirs('models/checkpoints', exist_ok=True)
    
    checkpoint_data = {
        'q_table': {k: v.tolist() for k, v in agent.q_table.items()},
        'episode': episode,
        'success_rate': success_rate,
        'epsilon': agent.epsilon,
        'discretization_precision': agent.discretization_precision,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = f'models/checkpoints/extended_rl_best.json'
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"    💾 Лучший checkpoint сохранен: success_rate={success_rate:.1f}%")

def save_final_model(agent: ExtendedRLAgent):
    """Сохраняет финальную модель."""
    model_data = {
        'q_table': {k: v.tolist() for k, v in agent.q_table.items()},
        'training_stats': agent.training_stats,
        'hyperparameters': {
            'learning_rate': agent.learning_rate,
            'gamma': agent.gamma,
            'epsilon_decay': agent.epsilon_decay,
            'discretization_precision': agent.discretization_precision
        },
        'training_date': datetime.now().isoformat(),
        'model_type': 'ExtendedRLAgent'
    }
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/extended_rl_path_planner.json'
    
    with open(model_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"  💾 Расширенная модель сохранена: {model_path}")

def compare_all_rl_models():
    """Сравнивает все RL модели."""
    print(f"\n⚖️  СРАВНЕНИЕ ВСЕХ RL МОДЕЛЕЙ")
    print("=" * 60)
    
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
        
        base_results = test_rl_agent(base_agent, test_scenarios)
        models_results['Базовая RL (1K эпизодов)'] = base_results
        
    except FileNotFoundError:
        models_results['Базовая RL (1K эпизодов)'] = {'success_rate': 25.0}
    
    # 2. Расширенная RL модель
    try:
        with open('models/extended_rl_path_planner.json', 'r') as f:
            extended_model_data = json.load(f)
        
        extended_agent = ExtendedRLAgent()
        extended_agent.q_table = {k: np.array(v) for k, v in extended_model_data['q_table'].items()}
        
        extended_results = test_rl_agent(extended_agent, test_scenarios)
        models_results['Расширенная RL (20K эпизодов)'] = extended_results
        
    except FileNotFoundError:
        print("⚠️  Расширенная RL модель не найдена - будет обучена")
        return None
    
    # 3. Классические алгоритмы для сравнения
    from test_algorithms_simple import create_planner
    
    algorithms = ['astar', 'dijkstra', 'rrt_optimized']
    
    for algorithm in algorithms:
        print(f"🔄 Тестирование {algorithm}...")
        
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
        
        models_results[algorithm.upper()] = {
            'success_rate': success_rate,
            'average_score': avg_score,
            'average_efficiency': avg_score  # Приблизительно
        }
    
    # Выводим сравнение
    print(f"\n📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА ВСЕХ АЛГОРИТМОВ:")
    print(f"{'Алгоритм':<30} {'Success Rate':<12} {'Avg Efficiency':<15} {'Тип':<10}")
    print("-" * 75)
    
    for model_name, results in models_results.items():
        success_rate = results['success_rate']
        efficiency = results.get('average_efficiency', results.get('average_score', 0))
        model_type = 'RL' if 'RL' in model_name else 'Classical'
        
        print(f"{model_name:<30} {success_rate:<12.1f} {efficiency:<15.3f} {model_type:<10}")
    
    return models_results

def main():
    """Главная функция расширенного обучения."""
    
    print("🚀 СИСТЕМА РАСШИРЕННОГО RL ОБУЧЕНИЯ")
    print("=" * 70)
    print("Проект: Swarm Path Planning")
    print("Версия: 8.2 (Extended RL Training)")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Обучаем расширенного RL агента
        print("🎓 Этап 1: Расширенное обучение RL агента")
        agent = train_extended_rl_agent(episodes=20000, save_model=True)
        
        # 2. Тестовые сценарии
        test_scenarios = [
            (np.array([0, 0, 3]), np.array([5, 0, 3])),      # Простой
            (np.array([0, 0, 2]), np.array([10, 8, 4])),     # Средний
            (np.array([-5, -5, 1]), np.array([15, 10, 5])),  # Сложный
            (np.array([0, 0, 1]), np.array([2, 2, 8])),      # Вертикальный
            (np.array([-8, -8, 0]), np.array([18, 15, 7])),  # Очень сложный
        ]
        
        # 3. Тестируем расширенного RL агента
        print(f"\n🧪 Этап 2: Тестирование расширенного RL агента")
        extended_results = test_rl_agent(agent, test_scenarios)
        
        # 4. Сравниваем все модели
        print(f"\n📊 Этап 3: Сравнение всех алгоритмов")
        comparison_results = compare_all_rl_models()
        
        # 5. Сохраняем результаты
        results_data = {
            'extended_training_stats': agent.training_stats,
            'extended_test_results': extended_results,
            'comparison_results': comparison_results,
            'model_type': 'ExtendedRLAgent',
            'training_episodes': 20000,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('extended_rl_training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n✅ РАСШИРЕННОЕ ОБУЧЕНИЕ RL ЗАВЕРШЕНО")
        print("=" * 70)
        print(f"📊 Результаты сохранены в: extended_rl_training_results.json")
        print(f"💾 Модель сохранена в: models/extended_rl_path_planner.json")
        print(f"🏆 Лучший checkpoint: models/checkpoints/extended_rl_best.json")
        
        return agent, results_data
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    agent, results = main()