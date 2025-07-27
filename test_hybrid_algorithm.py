#!/usr/bin/env python3
"""
Комплексное тестирование гибридного алгоритма планирования пути.
"""
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Добавляем путь к проекту
sys.path.append('/workspace/swarm')

# Импортируем базовые планировщики и гибридный алгоритм
from test_algorithms_simple import (
    RRTPlanner, RRTOptimizedPlanner, AdaptiveRRTPlanner,
    AStarPlanner, DijkstraPlanner, RRTStarPlanner, create_planner
)
from hybrid_algorithm import create_hybrid_planner, PlanningStrategy

def create_test_scenarios() -> List[Dict]:
    """Создает разнообразные тестовые сценарии."""
    return [
        {
            'name': 'Простой короткий путь',
            'start': np.array([0.0, 0.0, 3.0]),
            'goal': np.array([5.0, 0.0, 3.0]),
            'difficulty': 0.1,
            'expected_time': 0.1
        },
        {
            'name': 'Средний диагональный путь',
            'start': np.array([0.0, 0.0, 2.0]),
            'goal': np.array([10.0, 8.0, 4.0]),
            'difficulty': 0.4,
            'expected_time': 0.2
        },
        {
            'name': 'Сложный длинный путь',
            'start': np.array([-5.0, -5.0, 1.0]),
            'goal': np.array([20.0, 15.0, 6.0]),
            'difficulty': 0.7,
            'expected_time': 0.3
        },
        {
            'name': 'Экстремально сложный путь',
            'start': np.array([-10.0, -10.0, 0.5]),
            'goal': np.array([25.0, 20.0, 8.0]),
            'difficulty': 0.9,
            'expected_time': 0.5
        },
        {
            'name': 'Вертикальный подъем',
            'start': np.array([0.0, 0.0, 1.0]),
            'goal': np.array([2.0, 2.0, 10.0]),
            'difficulty': 0.6,
            'expected_time': 0.25
        }
    ]

def test_hybrid_strategies(scenarios: List[Dict], num_tests: int = 5) -> Dict:
    """Тестирует все стратегии гибридного алгоритма."""
    
    print("🔬 ТЕСТИРОВАНИЕ ГИБРИДНЫХ СТРАТЕГИЙ")
    print("=" * 70)
    
    strategies = [
        'adaptive',
        'fast_first', 
        'quality_first',
        'parallel',
        'hierarchical',
        'multi_stage'
    ]
    
    results = defaultdict(lambda: defaultdict(list))
    
    for scenario in scenarios:
        print(f"\n📋 Сценарий: {scenario['name']} (сложность: {scenario['difficulty']:.1f})")
        print("-" * 50)
        
        for strategy in strategies:
            print(f"  🔄 {strategy}...", end=" ")
            
            scores = []
            times = []
            successes = []
            details_list = []
            
            for test_num in range(num_tests):
                # Создаем гибридный планировщик
                planner = create_hybrid_planner(
                    start=scenario['start'],
                    goal=scenario['goal'],
                    strategy=strategy,
                    time_budget=scenario['expected_time'] * 2,  # Даем больше времени
                    quality_threshold=0.8
                )
                
                # Планируем путь
                start_time = time.time()
                path = planner.plan()
                planning_time = time.time() - start_time
                
                # Вычисляем метрики
                score = planner.calculate_score(path)
                success = score > 0
                details = planner.get_planning_details()
                
                scores.append(score)
                times.append(planning_time)
                successes.append(success)
                details_list.append(details)
            
            # Сохраняем результаты
            avg_score = np.mean(scores)
            avg_time = np.mean(times)
            success_rate = np.mean(successes)
            
            results[strategy][scenario['name']] = {
                'scores': scores,
                'times': times,
                'successes': successes,
                'details': details_list,
                'avg_score': avg_score,
                'avg_time': avg_time,
                'success_rate': success_rate,
                'scenario_difficulty': scenario['difficulty']
            }
            
            # Анализируем использованные планировщики
            all_planners = []
            for detail in details_list:
                all_planners.extend(detail.get('planners_used', []))
            unique_planners = list(set(all_planners))
            
            print(f"Score: {avg_score:.3f}, Time: {avg_time:.3f}s, Success: {success_rate:.1%}")
            print(f"    Планировщики: {', '.join(unique_planners)}")
    
    return results

def compare_with_baseline(scenarios: List[Dict], num_tests: int = 5) -> Dict:
    """Сравнивает гибридные алгоритмы с базовыми."""
    
    print(f"\n{'=' * 70}")
    print("⚖️  СРАВНЕНИЕ С БАЗОВЫМИ АЛГОРИТМАМИ")
    print(f"{'=' * 70}")
    
    # Базовые алгоритмы для сравнения
    baseline_algorithms = ['rrt', 'rrt_optimized', 'adaptive_rrt', 'astar', 'dijkstra', 'rrt_star']
    
    # Лучшие гибридные стратегии (выберем несколько)
    hybrid_strategies = ['adaptive', 'parallel', 'multi_stage']
    
    comparison_results = {}
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}")
        print("-" * 40)
        
        scenario_results = {}
        
        # Тестируем базовые алгоритмы
        print("  Базовые алгоритмы:")
        for algorithm in baseline_algorithms:
            scores = []
            times = []
            
            for _ in range(num_tests):
                planner = create_planner(algorithm, scenario['start'], scenario['goal'])
                
                start_time = time.time()
                path = planner.plan()
                planning_time = time.time() - start_time
                
                score = planner.calculate_score(path)
                scores.append(score)
                times.append(planning_time)
            
            avg_score = np.mean(scores)
            avg_time = np.mean(times)
            success_rate = np.mean([s > 0 for s in scores])
            
            scenario_results[algorithm] = {
                'type': 'baseline',
                'avg_score': avg_score,
                'avg_time': avg_time,
                'success_rate': success_rate
            }
            
            print(f"    {algorithm:<15}: Score {avg_score:.3f}, Time {avg_time:.3f}s, Success {success_rate:.1%}")
        
        # Тестируем гибридные стратегии
        print("  Гибридные стратегии:")
        for strategy in hybrid_strategies:
            scores = []
            times = []
            planners_used_counts = defaultdict(int)
            
            for _ in range(num_tests):
                planner = create_hybrid_planner(
                    start=scenario['start'],
                    goal=scenario['goal'],
                    strategy=strategy,
                    time_budget=scenario['expected_time'] * 2,
                    quality_threshold=0.8
                )
                
                start_time = time.time()
                path = planner.plan()
                planning_time = time.time() - start_time
                
                score = planner.calculate_score(path)
                details = planner.get_planning_details()
                
                scores.append(score)
                times.append(planning_time)
                
                # Подсчитываем использованные планировщики
                for planner_name in details.get('planners_used', []):
                    planners_used_counts[planner_name] += 1
            
            avg_score = np.mean(scores)
            avg_time = np.mean(times)
            success_rate = np.mean([s > 0 for s in scores])
            
            scenario_results[f"hybrid_{strategy}"] = {
                'type': 'hybrid',
                'avg_score': avg_score,
                'avg_time': avg_time,
                'success_rate': success_rate,
                'planners_used': dict(planners_used_counts)
            }
            
            most_used = max(planners_used_counts.items(), key=lambda x: x[1]) if planners_used_counts else ("none", 0)
            print(f"    hybrid_{strategy:<10}: Score {avg_score:.3f}, Time {avg_time:.3f}s, Success {success_rate:.1%}")
            print(f"                        Most used: {most_used[0]} ({most_used[1]}/{num_tests})")
        
        comparison_results[scenario['name']] = scenario_results
    
    return comparison_results

def analyze_hybrid_performance(hybrid_results: Dict, comparison_results: Dict):
    """Анализирует производительность гибридных алгоритмов."""
    
    print(f"\n{'=' * 70}")
    print("📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ГИБРИДНЫХ АЛГОРИТМОВ")
    print(f"{'=' * 70}")
    
    # 1. Общая производительность гибридных стратегий
    print("\n1️⃣ РЕЙТИНГ ГИБРИДНЫХ СТРАТЕГИЙ:")
    
    strategy_stats = {}
    for strategy in hybrid_results.keys():
        all_scores = []
        all_times = []
        all_success_rates = []
        
        for scenario_data in hybrid_results[strategy].values():
            all_scores.append(scenario_data['avg_score'])
            all_times.append(scenario_data['avg_time'])
            all_success_rates.append(scenario_data['success_rate'])
        
        strategy_stats[strategy] = {
            'avg_score': np.mean(all_scores),
            'avg_time': np.mean(all_times),
            'avg_success_rate': np.mean(all_success_rates)
        }
    
    # Сортируем по среднему score
    sorted_strategies = sorted(strategy_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    print(f"{'Стратегия':<15} {'Avg Score':<10} {'Avg Time':<10} {'Success Rate':<12}")
    print("-" * 50)
    for strategy, stats in sorted_strategies:
        print(f"{strategy:<15} {stats['avg_score']:<10.3f} {stats['avg_time']:<10.3f} {stats['avg_success_rate']:<12.1%}")
    
    # 2. Сравнение с лучшими базовыми алгоритмами
    print(f"\n2️⃣ СРАВНЕНИЕ С БАЗОВЫМИ АЛГОРИТМАМИ:")
    
    for scenario_name, scenario_data in comparison_results.items():
        print(f"\n📋 {scenario_name}:")
        
        # Находим лучший базовый алгоритм
        baseline_results = {k: v for k, v in scenario_data.items() if v['type'] == 'baseline'}
        best_baseline = max(baseline_results.items(), key=lambda x: x[1]['avg_score'])
        
        # Находим лучший гибридный алгоритм
        hybrid_results_scenario = {k: v for k, v in scenario_data.items() if v['type'] == 'hybrid'}
        best_hybrid = max(hybrid_results_scenario.items(), key=lambda x: x[1]['avg_score'])
        
        print(f"  🥇 Лучший базовый: {best_baseline[0]} (score: {best_baseline[1]['avg_score']:.3f})")
        print(f"  🏆 Лучший гибридный: {best_hybrid[0]} (score: {best_hybrid[1]['avg_score']:.3f})")
        
        # Вычисляем улучшение
        improvement = ((best_hybrid[1]['avg_score'] - best_baseline[1]['avg_score']) / 
                      best_baseline[1]['avg_score'] * 100) if best_baseline[1]['avg_score'] > 0 else 0
        
        if improvement > 0:
            print(f"  📈 Улучшение: +{improvement:.1f}%")
        else:
            print(f"  📉 Ухудшение: {improvement:.1f}%")
    
    # 3. Анализ адаптивности
    print(f"\n3️⃣ АНАЛИЗ АДАПТИВНОСТИ:")
    
    if 'adaptive' in hybrid_results:
        adaptive_data = hybrid_results['adaptive']
        
        print("Адаптивная стратегия по сложности:")
        for scenario_name, data in adaptive_data.items():
            difficulty = data['scenario_difficulty']
            score = data['avg_score']
            
            # Анализируем использованные планировщики
            planners_used = []
            for detail in data['details']:
                planners_used.extend(detail.get('planners_used', []))
            
            most_common = max(set(planners_used), key=planners_used.count) if planners_used else "unknown"
            
            print(f"  Сложность {difficulty:.1f}: score {score:.3f}, чаще всего использовался {most_common}")
    
    # 4. Рекомендации
    print(f"\n4️⃣ РЕКОМЕНДАЦИИ:")
    
    best_overall = sorted_strategies[0]
    fastest_strategy = min(strategy_stats.items(), key=lambda x: x[1]['avg_time'])
    most_reliable = max(strategy_stats.items(), key=lambda x: x[1]['avg_success_rate'])
    
    print(f"  🏆 Лучшая общая производительность: {best_overall[0]}")
    print(f"  ⚡ Самая быстрая стратегия: {fastest_strategy[0]}")
    print(f"  🛡️  Самая надежная стратегия: {most_reliable[0]}")
    
    # Специфические рекомендации
    print(f"\n  💡 Специфические рекомендации:")
    print(f"    • Для критичных по времени задач: {fastest_strategy[0]}")
    print(f"    • Для максимального качества: {best_overall[0]}")
    print(f"    • Для неопределенных условий: adaptive")
    print(f"    • Для комплексных задач: parallel или multi_stage")

def main():
    """Главная функция."""
    
    print("🤖 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ГИБРИДНОГО АЛГОРИТМА")
    print("=" * 70)
    print("Проект: Swarm Path Planning")
    print("Версия: 6.0 (Гибридный алгоритм)")
    print(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Создаем тестовые сценарии
        scenarios = create_test_scenarios()
        print(f"📋 Создано сценариев: {len(scenarios)}")
        
        # Тестируем гибридные стратегии
        hybrid_results = test_hybrid_strategies(scenarios, num_tests=3)
        
        # Сравниваем с базовыми алгоритмами
        comparison_results = compare_with_baseline(scenarios, num_tests=3)
        
        # Анализируем результаты
        analyze_hybrid_performance(hybrid_results, comparison_results)
        
        print(f"\n{'=' * 70}")
        print("✅ ТЕСТИРОВАНИЕ ГИБРИДНОГО АЛГОРИТМА ЗАВЕРШЕНО")
        print(f"{'=' * 70}")
        
        return hybrid_results, comparison_results
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    hybrid_results, comparison_results = main()