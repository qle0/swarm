#!/usr/bin/env python3
"""
Детальный анализ производительности алгоритмов планирования пути.
"""
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Добавляем путь к проекту
sys.path.append('/workspace/swarm')

# Импортируем наши планировщики из предыдущего теста
from test_algorithms_simple import (
    RRTPlanner, RRTOptimizedPlanner, AdaptiveRRTPlanner,
    AStarPlanner, DijkstraPlanner, RRTStarPlanner, create_planner
)

def create_difficulty_scenarios() -> List[Dict]:
    """Создает сценарии различной сложности."""
    return [
        {
            'name': 'Очень легкий',
            'difficulty': 0.1,
            'start': np.array([0.0, 0.0, 3.0]),
            'goal': np.array([5.0, 0.0, 3.0]),
            'description': 'Короткий прямой путь без препятствий'
        },
        {
            'name': 'Легкий',
            'difficulty': 0.3,
            'start': np.array([0.0, 0.0, 3.0]),
            'goal': np.array([10.0, 5.0, 3.0]),
            'description': 'Средний путь с небольшим отклонением'
        },
        {
            'name': 'Средний',
            'difficulty': 0.5,
            'start': np.array([0.0, 0.0, 2.0]),
            'goal': np.array([15.0, 10.0, 4.0]),
            'description': 'Длинный путь с изменением высоты'
        },
        {
            'name': 'Сложный',
            'difficulty': 0.7,
            'start': np.array([-5.0, -5.0, 1.0]),
            'goal': np.array([20.0, 15.0, 6.0]),
            'description': 'Очень длинный путь с большим изменением высоты'
        },
        {
            'name': 'Очень сложный',
            'difficulty': 0.9,
            'start': np.array([-10.0, -10.0, 0.5]),
            'goal': np.array([25.0, 20.0, 8.0]),
            'description': 'Экстремально длинный путь'
        }
    ]

def run_performance_analysis():
    """Запускает детальный анализ производительности."""
    
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ АЛГОРИТМОВ")
    print("=" * 70)
    
    algorithms = ['rrt', 'rrt_optimized', 'adaptive_rrt', 'astar', 'dijkstra', 'rrt_star']
    scenarios = create_difficulty_scenarios()
    
    # Структура для хранения результатов
    results = defaultdict(lambda: defaultdict(list))
    
    print(f"📊 Тестирование {len(algorithms)} алгоритмов на {len(scenarios)} сценариях")
    print(f"🧪 Количество тестов на комбинацию: 10")
    print()
    
    total_tests = len(algorithms) * len(scenarios) * 10
    current_test = 0
    
    for scenario in scenarios:
        print(f"\n📋 Сценарий: {scenario['name']} (сложность: {scenario['difficulty']:.1f})")
        print(f"   {scenario['description']}")
        print("-" * 50)
        
        for algorithm in algorithms:
            print(f"  🔄 {algorithm}...", end=" ")
            
            scores = []
            times = []
            successes = []
            
            for test_num in range(10):
                current_test += 1
                progress = (current_test / total_tests) * 100
                
                # Создаем планировщик
                planner = create_planner(algorithm, scenario['start'], scenario['goal'])
                
                # Планируем путь
                start_time = time.time()
                path = planner.plan()
                planning_time = time.time() - start_time
                
                # Вычисляем score
                score = planner.calculate_score(path)
                success = score > 0
                
                scores.append(score)
                times.append(planning_time)
                successes.append(success)
            
            # Сохраняем результаты
            results[algorithm][scenario['name']] = {
                'scores': scores,
                'times': times,
                'successes': successes,
                'avg_score': np.mean(scores),
                'avg_time': np.mean(times),
                'success_rate': np.mean(successes),
                'score_std': np.std(scores),
                'difficulty': scenario['difficulty']
            }
            
            avg_score = np.mean(scores)
            success_rate = np.mean(successes)
            avg_time = np.mean(times)
            
            print(f"Score: {avg_score:.3f}, Success: {success_rate:.1%}, Time: {avg_time:.3f}s [{progress:.1f}%]")
    
    return results

def analyze_difficulty_scaling(results: Dict):
    """Анализирует, как алгоритмы справляются с увеличением сложности."""
    
    print(f"\n{'=' * 70}")
    print("📈 АНАЛИЗ МАСШТАБИРОВАНИЯ ПО СЛОЖНОСТИ")
    print(f"{'=' * 70}")
    
    algorithms = list(results.keys())
    scenarios = ['Очень легкий', 'Легкий', 'Средний', 'Сложный', 'Очень сложный']
    
    print(f"{'Алгоритм':<15} {'Очень легкий':<12} {'Легкий':<8} {'Средний':<8} {'Сложный':<8} {'Очень сложный':<13}")
    print("-" * 70)
    
    for algorithm in algorithms:
        row = f"{algorithm:<15} "
        for scenario in scenarios:
            if scenario in results[algorithm]:
                score = results[algorithm][scenario]['avg_score']
                row += f"{score:<12.3f} " if scenario == 'Очень легкий' else f"{score:<8.3f} "
            else:
                row += f"{'N/A':<12} " if scenario == 'Очень легкий' else f"{'N/A':<8} "
        print(row)
    
    # Анализ деградации производительности
    print(f"\n📉 ДЕГРАДАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ:")
    for algorithm in algorithms:
        if 'Очень легкий' in results[algorithm] and 'Очень сложный' in results[algorithm]:
            easy_score = results[algorithm]['Очень легкий']['avg_score']
            hard_score = results[algorithm]['Очень сложный']['avg_score']
            degradation = (easy_score - hard_score) / easy_score * 100 if easy_score > 0 else 0
            print(f"  {algorithm:<15}: {degradation:>6.1f}% снижение")

def analyze_time_complexity(results: Dict):
    """Анализирует временную сложность алгоритмов."""
    
    print(f"\n{'=' * 70}")
    print("⏱️  АНАЛИЗ ВРЕМЕННОЙ СЛОЖНОСТИ")
    print(f"{'=' * 70}")
    
    algorithms = list(results.keys())
    scenarios = ['Очень легкий', 'Легкий', 'Средний', 'Сложный', 'Очень сложный']
    
    print(f"{'Алгоритм':<15} {'Мин время':<10} {'Макс время':<11} {'Рост времени':<12}")
    print("-" * 50)
    
    for algorithm in algorithms:
        times = []
        for scenario in scenarios:
            if scenario in results[algorithm]:
                times.append(results[algorithm][scenario]['avg_time'])
        
        if times:
            min_time = min(times)
            max_time = max(times)
            time_growth = (max_time / min_time) if min_time > 0 else 0
            
            print(f"{algorithm:<15} {min_time:<10.3f} {max_time:<11.3f} {time_growth:<12.2f}x")

def find_optimal_algorithm(results: Dict):
    """Определяет оптимальный алгоритм для каждого типа задач."""
    
    print(f"\n{'=' * 70}")
    print("🎯 РЕКОМЕНДАЦИИ ПО ВЫБОРУ АЛГОРИТМА")
    print(f"{'=' * 70}")
    
    scenarios = ['Очень легкий', 'Легкий', 'Средний', 'Сложный', 'Очень сложный']
    
    for scenario in scenarios:
        print(f"\n📋 {scenario}:")
        
        # Собираем данные для сценария
        scenario_data = []
        for algorithm in results.keys():
            if scenario in results[algorithm]:
                data = results[algorithm][scenario]
                scenario_data.append({
                    'algorithm': algorithm,
                    'score': data['avg_score'],
                    'time': data['avg_time'],
                    'success_rate': data['success_rate'],
                    'stability': 1 / (data['score_std'] + 0.001)  # Обратная величина стандартного отклонения
                })
        
        if not scenario_data:
            continue
        
        # Лучший по score
        best_score = max(scenario_data, key=lambda x: x['score'])
        print(f"  🥇 Лучший score: {best_score['algorithm']} ({best_score['score']:.3f})")
        
        # Самый быстрый
        fastest = min(scenario_data, key=lambda x: x['time'])
        print(f"  ⚡ Самый быстрый: {fastest['algorithm']} ({fastest['time']:.3f}s)")
        
        # Самый надежный
        most_reliable = max(scenario_data, key=lambda x: x['success_rate'])
        print(f"  🛡️  Самый надежный: {most_reliable['algorithm']} ({most_reliable['success_rate']:.1%})")
        
        # Самый стабильный
        most_stable = max(scenario_data, key=lambda x: x['stability'])
        print(f"  📊 Самый стабильный: {most_stable['algorithm']}")
        
        # Комплексная оценка (взвешенная сумма)
        for data in scenario_data:
            # Нормализуем метрики
            max_score = max(scenario_data, key=lambda x: x['score'])['score']
            min_time = min(scenario_data, key=lambda x: x['time'])['time']
            max_success = max(scenario_data, key=lambda x: x['success_rate'])['success_rate']
            max_stability = max(scenario_data, key=lambda x: x['stability'])['stability']
            
            # Взвешенная оценка
            score_weight = 0.4
            time_weight = 0.2
            success_weight = 0.3
            stability_weight = 0.1
            
            normalized_score = data['score'] / max_score if max_score > 0 else 0
            normalized_time = min_time / data['time'] if data['time'] > 0 else 0
            normalized_success = data['success_rate'] / max_success if max_success > 0 else 0
            normalized_stability = data['stability'] / max_stability if max_stability > 0 else 0
            
            composite_score = (
                normalized_score * score_weight +
                normalized_time * time_weight +
                normalized_success * success_weight +
                normalized_stability * stability_weight
            )
            
            data['composite_score'] = composite_score
        
        best_overall = max(scenario_data, key=lambda x: x['composite_score'])
        print(f"  🏆 Лучший общий: {best_overall['algorithm']} (композитный score: {best_overall['composite_score']:.3f})")

def generate_summary_report(results: Dict):
    """Генерирует итоговый отчет."""
    
    print(f"\n{'=' * 70}")
    print("📋 ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'=' * 70}")
    
    algorithms = list(results.keys())
    total_tests = sum(len(results[alg]) * 10 for alg in algorithms)
    
    print(f"📊 Общая статистика:")
    print(f"  • Протестировано алгоритмов: {len(algorithms)}")
    print(f"  • Общее количество тестов: {total_tests}")
    print(f"  • Сценариев сложности: 5")
    
    # Общие рейтинги
    overall_scores = {}
    overall_times = {}
    overall_success_rates = {}
    
    for algorithm in algorithms:
        all_scores = []
        all_times = []
        all_successes = []
        
        for scenario_data in results[algorithm].values():
            all_scores.extend(scenario_data['scores'])
            all_times.extend(scenario_data['times'])
            all_successes.extend(scenario_data['successes'])
        
        overall_scores[algorithm] = np.mean(all_scores)
        overall_times[algorithm] = np.mean(all_times)
        overall_success_rates[algorithm] = np.mean(all_successes)
    
    print(f"\n🏆 ОБЩИЕ РЕЙТИНГИ:")
    
    # Рейтинг по score
    sorted_by_score = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  📈 По среднему score:")
    for i, (alg, score) in enumerate(sorted_by_score, 1):
        print(f"    {i}. {alg}: {score:.3f}")
    
    # Рейтинг по времени
    sorted_by_time = sorted(overall_times.items(), key=lambda x: x[1])
    print(f"  ⚡ По скорости:")
    for i, (alg, time) in enumerate(sorted_by_time, 1):
        print(f"    {i}. {alg}: {time:.3f}s")
    
    # Рейтинг по надежности
    sorted_by_success = sorted(overall_success_rates.items(), key=lambda x: x[1], reverse=True)
    print(f"  🛡️  По надежности:")
    for i, (alg, rate) in enumerate(sorted_by_success, 1):
        print(f"    {i}. {alg}: {rate:.1%}")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print(f"  • Для максимального качества: {sorted_by_score[0][0]}")
    print(f"  • Для максимальной скорости: {sorted_by_time[0][0]}")
    print(f"  • Для максимальной надежности: {sorted_by_success[0][0]}")
    
    # Универсальная рекомендация
    composite_scores = {}
    for alg in algorithms:
        score_rank = next(i for i, (a, _) in enumerate(sorted_by_score) if a == alg)
        time_rank = next(i for i, (a, _) in enumerate(sorted_by_time) if a == alg)
        success_rank = next(i for i, (a, _) in enumerate(sorted_by_success) if a == alg)
        
        # Чем меньше сумма рангов, тем лучше
        composite_scores[alg] = score_rank + time_rank + success_rank
    
    best_universal = min(composite_scores.items(), key=lambda x: x[1])
    print(f"  • Универсальный выбор: {best_universal[0]}")

def main():
    """Главная функция."""
    
    print("🤖 ДЕТАЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ АЛГОРИТМОВ")
    print("Проект: Swarm Path Planning")
    print("Версия: 4.0 (Детальный анализ)")
    print("Дата: 2025-07-27")
    print()
    
    try:
        # Запускаем анализ производительности
        print("🚀 Запуск тестирования...")
        results = run_performance_analysis()
        
        # Различные виды анализа
        analyze_difficulty_scaling(results)
        analyze_time_complexity(results)
        find_optimal_algorithm(results)
        generate_summary_report(results)
        
        print(f"\n{'=' * 70}")
        print("✅ ДЕТАЛЬНЫЙ АНАЛИЗ ЗАВЕРШЕН")
        print(f"{'=' * 70}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()