#!/usr/bin/env python3
"""
Гибридный алгоритм планирования пути, комбинирующий различные подходы.
"""
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum

# Добавляем путь к проекту
sys.path.append('/workspace/swarm')

# Импортируем базовые планировщики
from test_algorithms_simple import (
    SimplePlanner, RRTPlanner, RRTOptimizedPlanner, AdaptiveRRTPlanner,
    AStarPlanner, DijkstraPlanner, RRTStarPlanner
)

class PlanningStrategy(Enum):
    """Стратегии планирования для гибридного алгоритма."""
    FAST_FIRST = "fast_first"           # Сначала быстрый, потом точный
    QUALITY_FIRST = "quality_first"     # Сначала качественный
    ADAPTIVE = "adaptive"               # Адаптивный выбор
    PARALLEL = "parallel"               # Параллельное планирование
    HIERARCHICAL = "hierarchical"       # Иерархическое планирование
    MULTI_STAGE = "multi_stage"         # Многоэтапное планирование

class HybridPlanner(SimplePlanner):
    """
    Гибридный планировщик, комбинирующий различные алгоритмы.
    """
    
    def __init__(self, start: np.ndarray, goal: np.ndarray, 
                 strategy: PlanningStrategy = PlanningStrategy.ADAPTIVE,
                 time_budget: float = 1.0,
                 quality_threshold: float = 0.8):
        """
        Инициализация гибридного планировщика.
        
        Parameters
        ----------
        start : np.ndarray
            Начальная позиция
        goal : np.ndarray  
            Целевая позиция
        strategy : PlanningStrategy
            Стратегия планирования
        time_budget : float
            Бюджет времени на планирование (секунды)
        quality_threshold : float
            Пороговое значение качества для переключения алгоритмов
        """
        super().__init__(start, goal, "hybrid")
        self.strategy = strategy
        self.time_budget = time_budget
        self.quality_threshold = quality_threshold
        
        # Доступные планировщики
        self.planners = {
            'rrt_fast': RRTOptimizedPlanner,
            'rrt': RRTPlanner,
            'adaptive_rrt': AdaptiveRRTPlanner,
            'astar': AStarPlanner,
            'dijkstra': DijkstraPlanner,
            'rrt_star': RRTStarPlanner
        }
        
        # История планирования
        self.planning_history = []
        
    def _estimate_difficulty(self) -> float:
        """Оценивает сложность задачи планирования."""
        distance = np.linalg.norm(self.goal - self.start)
        height_change = abs(self.goal[2] - self.start[2])
        
        # Нормализуем метрики
        distance_factor = min(distance / 30.0, 1.0)  # 30м = максимальная сложность
        height_factor = min(height_change / 10.0, 1.0)  # 10м = максимальная сложность
        
        # Комбинированная сложность
        difficulty = 0.7 * distance_factor + 0.3 * height_factor
        return min(difficulty, 1.0)
    
    def _select_planner_adaptive(self, difficulty: float, remaining_time: float) -> str:
        """Адаптивный выбор планировщика на основе сложности и времени."""
        
        if remaining_time < 0.1:  # Очень мало времени
            return 'rrt_fast'
        elif remaining_time < 0.2:  # Мало времени
            return 'rrt' if difficulty < 0.5 else 'rrt_fast'
        elif difficulty < 0.3:  # Простая задача
            return 'astar'
        elif difficulty < 0.6:  # Средняя сложность
            return 'adaptive_rrt'
        elif difficulty < 0.8:  # Сложная задача
            return 'dijkstra'
        else:  # Очень сложная задача
            return 'rrt_star'
    
    def _plan_fast_first(self) -> List[np.ndarray]:
        """Стратегия: сначала быстрый алгоритм, потом улучшение."""
        start_time = time.time()
        
        # Этап 1: Быстрое планирование
        fast_planner = self.planners['rrt_fast'](self.start, self.goal, 'rrt_fast')
        fast_path = fast_planner.plan()
        fast_score = fast_planner.calculate_score(fast_path)
        
        elapsed = time.time() - start_time
        self.planning_history.append({
            'planner': 'rrt_fast',
            'score': fast_score,
            'time': elapsed,
            'path_length': len(fast_path)
        })
        
        # Если путь хороший или времени мало, возвращаем его
        if fast_score >= self.quality_threshold or elapsed >= self.time_budget * 0.8:
            return fast_path
        
        # Этап 2: Улучшение качества
        remaining_time = self.time_budget - elapsed
        if remaining_time > 0.1:
            quality_planner = self.planners['astar'](self.start, self.goal, 'astar')
            quality_path = quality_planner.plan()
            quality_score = quality_planner.calculate_score(quality_path)
            
            elapsed_total = time.time() - start_time
            self.planning_history.append({
                'planner': 'astar',
                'score': quality_score,
                'time': elapsed_total - elapsed,
                'path_length': len(quality_path)
            })
            
            # Возвращаем лучший путь
            return quality_path if quality_score > fast_score else fast_path
        
        return fast_path
    
    def _plan_quality_first(self) -> List[np.ndarray]:
        """Стратегия: сначала качественный алгоритм."""
        start_time = time.time()
        
        # Пробуем качественный алгоритм
        quality_planner = self.planners['dijkstra'](self.start, self.goal, 'dijkstra')
        quality_path = quality_planner.plan()
        quality_score = quality_planner.calculate_score(quality_path)
        
        elapsed = time.time() - start_time
        self.planning_history.append({
            'planner': 'dijkstra',
            'score': quality_score,
            'time': elapsed,
            'path_length': len(quality_path)
        })
        
        # Если получили хороший результат, возвращаем его
        if quality_score > 0 and elapsed <= self.time_budget:
            return quality_path
        
        # Иначе используем быстрый алгоритм как fallback
        remaining_time = self.time_budget - elapsed
        if remaining_time > 0.05:
            fast_planner = self.planners['rrt_fast'](self.start, self.goal, 'rrt_fast')
            fast_path = fast_planner.plan()
            fast_score = fast_planner.calculate_score(fast_path)
            
            elapsed_total = time.time() - start_time
            self.planning_history.append({
                'planner': 'rrt_fast',
                'score': fast_score,
                'time': elapsed_total - elapsed,
                'path_length': len(fast_path)
            })
            
            return fast_path if fast_score > 0 else quality_path
        
        return quality_path
    
    def _plan_adaptive(self) -> List[np.ndarray]:
        """Адаптивная стратегия планирования."""
        start_time = time.time()
        difficulty = self._estimate_difficulty()
        
        # Выбираем планировщик на основе сложности
        planner_name = self._select_planner_adaptive(difficulty, self.time_budget)
        planner = self.planners[planner_name](self.start, self.goal, planner_name)
        
        path = planner.plan()
        score = planner.calculate_score(path)
        
        elapsed = time.time() - start_time
        self.planning_history.append({
            'planner': planner_name,
            'score': score,
            'time': elapsed,
            'path_length': len(path),
            'difficulty': difficulty
        })
        
        return path
    
    def _plan_parallel(self) -> List[np.ndarray]:
        """Параллельная стратегия (имитация)."""
        start_time = time.time()
        
        # Имитируем параллельное выполнение нескольких алгоритмов
        planners_to_try = ['rrt_fast', 'astar', 'adaptive_rrt']
        results = []
        
        for planner_name in planners_to_try:
            planner_start = time.time()
            planner = self.planners[planner_name](self.start, self.goal, planner_name)
            path = planner.plan()
            score = planner.calculate_score(path)
            planner_time = time.time() - planner_start
            
            results.append({
                'planner': planner_name,
                'path': path,
                'score': score,
                'time': planner_time
            })
            
            self.planning_history.append({
                'planner': planner_name,
                'score': score,
                'time': planner_time,
                'path_length': len(path)
            })
            
            # Прерываем, если превысили бюджет времени
            if time.time() - start_time >= self.time_budget:
                break
        
        # Выбираем лучший результат
        if results:
            best_result = max(results, key=lambda x: x['score'])
            return best_result['path']
        
        return []
    
    def _plan_hierarchical(self) -> List[np.ndarray]:
        """Иерархическая стратегия планирования."""
        start_time = time.time()
        
        # Уровень 1: Грубое планирование (быстро)
        coarse_planner = self.planners['rrt_fast'](self.start, self.goal, 'rrt_fast')
        coarse_path = coarse_planner.plan()
        coarse_score = coarse_planner.calculate_score(coarse_path)
        
        elapsed = time.time() - start_time
        self.planning_history.append({
            'planner': 'rrt_fast_coarse',
            'score': coarse_score,
            'time': elapsed,
            'path_length': len(coarse_path)
        })
        
        # Если времени мало или результат хороший, возвращаем грубый план
        if elapsed >= self.time_budget * 0.7 or coarse_score >= self.quality_threshold:
            return coarse_path
        
        # Уровень 2: Детальное планирование
        remaining_time = self.time_budget - elapsed
        if remaining_time > 0.1 and len(coarse_path) > 2:
            # Используем промежуточные точки грубого пути как waypoints
            detail_planner = self.planners['astar'](self.start, self.goal, 'astar')
            detail_path = detail_planner.plan()
            detail_score = detail_planner.calculate_score(detail_path)
            
            elapsed_total = time.time() - start_time
            self.planning_history.append({
                'planner': 'astar_detail',
                'score': detail_score,
                'time': elapsed_total - elapsed,
                'path_length': len(detail_path)
            })
            
            return detail_path if detail_score > coarse_score else coarse_path
        
        return coarse_path
    
    def _plan_multi_stage(self) -> List[np.ndarray]:
        """Многоэтапная стратегия планирования."""
        start_time = time.time()
        best_path = []
        best_score = 0.0
        
        # Этапы планирования в порядке приоритета
        stages = [
            ('rrt_fast', 0.3),      # 30% времени на быстрое планирование
            ('adaptive_rrt', 0.4),  # 40% времени на адаптивное планирование
            ('astar', 0.3)          # 30% времени на точное планирование
        ]
        
        for stage_name, time_fraction in stages:
            stage_budget = self.time_budget * time_fraction
            stage_start = time.time()
            
            planner = self.planners[stage_name](self.start, self.goal, stage_name)
            path = planner.plan()
            score = planner.calculate_score(path)
            
            stage_time = time.time() - stage_start
            self.planning_history.append({
                'planner': f"{stage_name}_stage",
                'score': score,
                'time': stage_time,
                'path_length': len(path)
            })
            
            # Обновляем лучший результат
            if score > best_score:
                best_path = path
                best_score = score
            
            # Прерываем, если превысили общий бюджет времени
            if time.time() - start_time >= self.time_budget:
                break
            
            # Если получили отличный результат, можем остановиться
            if score >= 0.95:
                break
        
        return best_path
    
    def plan(self) -> List[np.ndarray]:
        """Основной метод планирования."""
        planning_start = time.time()
        
        # Выбираем стратегию планирования
        if self.strategy == PlanningStrategy.FAST_FIRST:
            path = self._plan_fast_first()
        elif self.strategy == PlanningStrategy.QUALITY_FIRST:
            path = self._plan_quality_first()
        elif self.strategy == PlanningStrategy.ADAPTIVE:
            path = self._plan_adaptive()
        elif self.strategy == PlanningStrategy.PARALLEL:
            path = self._plan_parallel()
        elif self.strategy == PlanningStrategy.HIERARCHICAL:
            path = self._plan_hierarchical()
        elif self.strategy == PlanningStrategy.MULTI_STAGE:
            path = self._plan_multi_stage()
        else:
            # Fallback к адаптивной стратегии
            path = self._plan_adaptive()
        
        total_time = time.time() - planning_start
        
        # Добавляем общую информацию о планировании
        self.planning_history.append({
            'type': 'summary',
            'strategy': self.strategy.value,
            'total_time': total_time,
            'total_score': self.calculate_score(path),
            'stages_used': len([h for h in self.planning_history if h.get('type') != 'summary'])
        })
        
        return path
    
    def get_planning_details(self) -> Dict:
        """Возвращает детальную информацию о процессе планирования."""
        if not self.planning_history:
            return {}
        
        summary = next((h for h in self.planning_history if h.get('type') == 'summary'), {})
        stages = [h for h in self.planning_history if h.get('type') != 'summary']
        
        return {
            'strategy': self.strategy.value,
            'total_time': summary.get('total_time', 0),
            'total_score': summary.get('total_score', 0),
            'stages_count': len(stages),
            'stages': stages,
            'planners_used': list(set(s['planner'] for s in stages)),
            'best_stage': max(stages, key=lambda x: x['score']) if stages else None
        }

def create_hybrid_planner(start: np.ndarray, goal: np.ndarray, 
                         strategy: str = "adaptive", 
                         time_budget: float = 1.0,
                         quality_threshold: float = 0.8) -> HybridPlanner:
    """Создает гибридный планировщик с заданными параметрами."""
    
    strategy_map = {
        'fast_first': PlanningStrategy.FAST_FIRST,
        'quality_first': PlanningStrategy.QUALITY_FIRST,
        'adaptive': PlanningStrategy.ADAPTIVE,
        'parallel': PlanningStrategy.PARALLEL,
        'hierarchical': PlanningStrategy.HIERARCHICAL,
        'multi_stage': PlanningStrategy.MULTI_STAGE
    }
    
    strategy_enum = strategy_map.get(strategy, PlanningStrategy.ADAPTIVE)
    
    return HybridPlanner(
        start=start,
        goal=goal,
        strategy=strategy_enum,
        time_budget=time_budget,
        quality_threshold=quality_threshold
    )

if __name__ == "__main__":
    # Простой тест гибридного планировщика
    print("🔬 Тест гибридного планировщика")
    
    start = np.array([0.0, 0.0, 3.0])
    goal = np.array([10.0, 10.0, 3.0])
    
    for strategy in ['adaptive', 'fast_first', 'quality_first', 'parallel', 'hierarchical', 'multi_stage']:
        print(f"\n📋 Тестирование стратегии: {strategy}")
        
        planner = create_hybrid_planner(start, goal, strategy=strategy, time_budget=0.5)
        
        start_time = time.time()
        path = planner.plan()
        total_time = time.time() - start_time
        
        score = planner.calculate_score(path)
        details = planner.get_planning_details()
        
        print(f"  ✅ Score: {score:.3f}")
        print(f"  ⏱️ Time: {total_time:.3f}s")
        print(f"  🔧 Planners used: {', '.join(details.get('planners_used', []))}")
        print(f"  📊 Stages: {details.get('stages_count', 0)}")