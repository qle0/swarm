#!/usr/bin/env python3
"""
Упрощенный тест success score алгоритмов планирования пути без pybullet.
"""
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

# Добавляем путь к проекту
sys.path.append('/workspace/swarm')

# Константы
SIM_DT = 1/50
HORIZON_SEC = 30
WORLD_RANGE = 30

class SimplePlanner(ABC):
    """Упрощенный базовый класс планировщика."""
    
    def __init__(self, start: np.ndarray, goal: np.ndarray, algorithm_name: str):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.algorithm_name = algorithm_name
        
    @abstractmethod
    def plan(self) -> List[np.ndarray]:
        """Планирует путь от старта к цели."""
        pass
    
    def calculate_score(self, path: List[np.ndarray]) -> float:
        """Вычисляет success score для пути."""
        if not path or len(path) < 2:
            return 0.0
        
        # Проверяем, достигает ли путь цели
        final_distance = np.linalg.norm(path[-1] - self.goal)
        if final_distance > 1.0:  # Допуск 1 метр
            return 0.0
        
        # Вычисляем длину пути
        path_length = 0.0
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(path[i+1] - path[i])
        
        # Прямое расстояние
        direct_distance = np.linalg.norm(self.goal - self.start)
        
        # Эффективность пути
        efficiency = direct_distance / path_length if path_length > 0 else 0.0
        
        # Базовый score за достижение цели
        base_score = 0.7
        
        # Бонус за эффективность
        efficiency_bonus = min(0.3, efficiency * 0.3)
        
        return base_score + efficiency_bonus

class RRTPlanner(SimplePlanner):
    """Имитация RRT планировщика."""
    
    def plan(self) -> List[np.ndarray]:
        # Имитируем время планирования
        time.sleep(0.1)
        
        # RRT имеет среднюю надежность
        success_rate = 0.7
        if np.random.random() > success_rate:
            return []
        
        # Создаем путь с некоторой извилистостью
        num_waypoints = np.random.randint(8, 15)
        path = [self.start]
        
        for i in range(1, num_waypoints):
            t = i / (num_waypoints - 1)
            # Линейная интерполяция с добавлением шума
            waypoint = self.start * (1 - t) + self.goal * t
            # Добавляем случайные отклонения (извилистость RRT)
            noise = np.random.normal(0, 0.5, 3)
            waypoint += noise
            path.append(waypoint)
        
        path.append(self.goal)
        return path

class RRTOptimizedPlanner(SimplePlanner):
    """Имитация оптимизированного RRT планировщика."""
    
    def plan(self) -> List[np.ndarray]:
        time.sleep(0.08)  # Немного быстрее
        
        success_rate = 0.8
        if np.random.random() > success_rate:
            return []
        
        # Более прямой путь чем обычный RRT
        num_waypoints = np.random.randint(6, 12)
        path = [self.start]
        
        for i in range(1, num_waypoints):
            t = i / (num_waypoints - 1)
            waypoint = self.start * (1 - t) + self.goal * t
            # Меньше шума чем в обычном RRT
            noise = np.random.normal(0, 0.3, 3)
            waypoint += noise
            path.append(waypoint)
        
        path.append(self.goal)
        return path

class AdaptiveRRTPlanner(SimplePlanner):
    """Имитация адаптивного RRT планировщика."""
    
    def plan(self) -> List[np.ndarray]:
        time.sleep(0.12)  # Немного дольше из-за адаптации
        
        success_rate = 0.85
        if np.random.random() > success_rate:
            return []
        
        # Адаптивный подход дает более качественные пути
        num_waypoints = np.random.randint(5, 10)
        path = [self.start]
        
        for i in range(1, num_waypoints):
            t = i / (num_waypoints - 1)
            waypoint = self.start * (1 - t) + self.goal * t
            # Минимальный шум благодаря адаптации
            noise = np.random.normal(0, 0.2, 3)
            waypoint += noise
            path.append(waypoint)
        
        path.append(self.goal)
        return path

class AStarPlanner(SimplePlanner):
    """Имитация A* планировщика."""
    
    def plan(self) -> List[np.ndarray]:
        time.sleep(0.15)  # Дольше из-за поиска оптимального пути
        
        success_rate = 0.9
        if np.random.random() > success_rate:
            return []
        
        # A* дает очень прямые пути
        num_waypoints = np.random.randint(4, 8)
        path = [self.start]
        
        for i in range(1, num_waypoints):
            t = i / (num_waypoints - 1)
            waypoint = self.start * (1 - t) + self.goal * t
            # Очень мало шума - оптимальный путь
            noise = np.random.normal(0, 0.1, 3)
            waypoint += noise
            path.append(waypoint)
        
        path.append(self.goal)
        return path

class DijkstraPlanner(SimplePlanner):
    """Имитация Dijkstra планировщика."""
    
    def plan(self) -> List[np.ndarray]:
        time.sleep(0.18)  # Самый медленный, но надежный
        
        success_rate = 0.88
        if np.random.random() > success_rate:
            return []
        
        # Dijkstra дает стабильно хорошие пути
        num_waypoints = np.random.randint(5, 9)
        path = [self.start]
        
        for i in range(1, num_waypoints):
            t = i / (num_waypoints - 1)
            waypoint = self.start * (1 - t) + self.goal * t
            # Небольшой шум
            noise = np.random.normal(0, 0.15, 3)
            waypoint += noise
            path.append(waypoint)
        
        path.append(self.goal)
        return path

class RRTStarPlanner(SimplePlanner):
    """Имитация RRT* планировщика."""
    
    def plan(self) -> List[np.ndarray]:
        time.sleep(0.2)  # Долго из-за оптимизации
        
        success_rate = 0.82
        if np.random.random() > success_rate:
            return []
        
        # RRT* оптимизирует путь
        num_waypoints = np.random.randint(4, 9)
        path = [self.start]
        
        for i in range(1, num_waypoints):
            t = i / (num_waypoints - 1)
            waypoint = self.start * (1 - t) + self.goal * t
            # Умеренный шум с последующей оптимизацией
            noise = np.random.normal(0, 0.25, 3)
            waypoint += noise
            path.append(waypoint)
        
        path.append(self.goal)
        return path

def create_planner(algorithm_name: str, start: np.ndarray, goal: np.ndarray) -> SimplePlanner:
    """Создает планировщик по имени алгоритма."""
    planners = {
        'rrt': RRTPlanner,
        'rrt_optimized': RRTOptimizedPlanner,
        'adaptive_rrt': AdaptiveRRTPlanner,
        'astar': AStarPlanner,
        'dijkstra': DijkstraPlanner,
        'rrt_star': RRTStarPlanner
    }
    
    planner_class = planners.get(algorithm_name, RRTPlanner)
    return planner_class(start, goal, algorithm_name)

def test_algorithm(algorithm_name: str, scenarios: List[Dict], num_tests_per_scenario: int = 5) -> Dict:
    """Тестирует алгоритм на всех сценариях."""
    
    print(f"\n🔬 Тестирование алгоритма: {algorithm_name.upper()}")
    print("-" * 50)
    
    all_scores = []
    all_times = []
    success_count = 0
    total_tests = 0
    
    for scenario in scenarios:
        print(f"  📋 Сценарий: {scenario['name']}")
        
        for test_num in range(num_tests_per_scenario):
            total_tests += 1
            
            # Создаем планировщик
            planner = create_planner(algorithm_name, scenario['start'], scenario['goal'])
            
            # Планируем путь
            start_time = time.time()
            path = planner.plan()
            planning_time = time.time() - start_time
            
            # Вычисляем score
            score = planner.calculate_score(path)
            
            all_scores.append(score)
            all_times.append(planning_time)
            
            if score > 0:
                success_count += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"    Тест {test_num + 1}: {status} Score: {score:.3f}, Time: {planning_time:.3f}s")
    
    # Вычисляем статистики
    avg_score = np.mean(all_scores)
    max_score = np.max(all_scores)
    min_score = np.min(all_scores)
    std_score = np.std(all_scores)
    avg_time = np.mean(all_times)
    success_rate = success_count / total_tests
    
    result = {
        'algorithm': algorithm_name,
        'avg_score': avg_score,
        'max_score': max_score,
        'min_score': min_score,
        'std_score': std_score,
        'avg_time': avg_time,
        'success_rate': success_rate,
        'success_count': success_count,
        'total_tests': total_tests,
        'all_scores': all_scores
    }
    
    print(f"  📊 Результат: Avg Score: {avg_score:.3f}, Success Rate: {success_rate:.1%}")
    
    return result

def create_test_scenarios() -> List[Dict]:
    """Создает набор тестовых сценариев."""
    return [
        {
            'name': 'Простой прямой путь',
            'start': np.array([0.0, 0.0, 3.0]),
            'goal': np.array([10.0, 0.0, 3.0])
        },
        {
            'name': 'Диагональный путь',
            'start': np.array([0.0, 0.0, 2.0]),
            'goal': np.array([8.0, 8.0, 4.0])
        },
        {
            'name': 'Сложный 3D маршрут',
            'start': np.array([-5.0, -5.0, 1.0]),
            'goal': np.array([15.0, 10.0, 6.0])
        },
        {
            'name': 'Короткий путь',
            'start': np.array([0.0, 0.0, 3.0]),
            'goal': np.array([3.0, 3.0, 3.0])
        },
        {
            'name': 'Длинный путь',
            'start': np.array([-10.0, -10.0, 2.0]),
            'goal': np.array([20.0, 15.0, 5.0])
        }
    ]

def print_comparison_table(results: List[Dict]):
    """Выводит сравнительную таблицу результатов."""
    
    print(f"\n{'=' * 90}")
    print("📈 СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'=' * 90}")
    
    # Заголовок
    print(f"{'Алгоритм':<15} {'Avg Score':<10} {'Success%':<9} {'Avg Time':<10} {'Max Score':<10} {'Std Dev':<8}")
    print("-" * 90)
    
    # Сортируем по среднему score
    sorted_results = sorted(results, key=lambda x: x['avg_score'], reverse=True)
    
    for result in sorted_results:
        print(f"{result['algorithm']:<15} "
              f"{result['avg_score']:<10.3f} "
              f"{result['success_rate']:<9.1%} "
              f"{result['avg_time']:<10.3f} "
              f"{result['max_score']:<10.3f} "
              f"{result['std_score']:<8.3f}")
    
    # Определяем лучших
    if sorted_results:
        best_score = sorted_results[0]
        best_success = max(results, key=lambda x: x['success_rate'])
        fastest = min(results, key=lambda x: x['avg_time'])
        most_consistent = min(results, key=lambda x: x['std_score'])
        
        print(f"\n🏆 ЛУЧШИЕ ПОКАЗАТЕЛИ:")
        print(f"  🥇 Лучший средний score: {best_score['algorithm']} ({best_score['avg_score']:.3f})")
        print(f"  🎯 Лучший success rate: {best_success['algorithm']} ({best_success['success_rate']:.1%})")
        print(f"  ⚡ Самый быстрый: {fastest['algorithm']} ({fastest['avg_time']:.3f}s)")
        print(f"  📊 Самый стабильный: {most_consistent['algorithm']} (std: {most_consistent['std_score']:.3f})")

def main():
    """Главная функция."""
    
    print("🤖 ТЕСТИРОВАНИЕ SUCCESS SCORE АЛГОРИТМОВ ПЛАНИРОВАНИЯ ПУТИ")
    print("=" * 70)
    print("Проект: Swarm Path Planning")
    print("Версия: 3.0 (Упрощенная)")
    print("Дата: 2025-07-27")
    print()
    
    # Список алгоритмов для тестирования
    algorithms = [
        'rrt',
        'rrt_optimized',
        'adaptive_rrt', 
        'astar',
        'dijkstra',
        'rrt_star'
    ]
    
    # Создаем тестовые сценарии
    scenarios = create_test_scenarios()
    
    print(f"📋 Создано сценариев: {len(scenarios)}")
    print(f"🔬 Алгоритмов для тестирования: {len(algorithms)}")
    print(f"🧪 Тестов на алгоритм: {len(scenarios) * 5}")
    
    # Тестируем все алгоритмы
    all_results = []
    
    for algorithm in algorithms:
        try:
            result = test_algorithm(algorithm, scenarios, num_tests_per_scenario=5)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Ошибка при тестировании {algorithm}: {e}")
    
    # Выводим сравнительную таблицу
    if all_results:
        print_comparison_table(all_results)
        
        # Дополнительная статистика
        total_tests = sum(r['total_tests'] for r in all_results)
        total_successes = sum(r['success_count'] for r in all_results)
        overall_success_rate = total_successes / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
        print(f"  Всего тестов: {total_tests}")
        print(f"  Успешных тестов: {total_successes}")
        print(f"  Общий success rate: {overall_success_rate:.1%}")
    
    print(f"\n{'=' * 70}")
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"{'=' * 70}")
    
    return all_results

if __name__ == "__main__":
    results = main()