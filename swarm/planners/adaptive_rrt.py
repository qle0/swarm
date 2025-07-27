"""
Adaptive RRT planner that adjusts parameters based on environment complexity.
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
import random
from dataclasses import dataclass
import time

from swarm.planners.rrt_optimized import OptimizedRRTPlanner, RRTNode


class EnvironmentAnalyzer:
    """Analyzes environment complexity to adapt planning parameters."""
    
    def __init__(self, client_id: Optional[int] = None, obstacle_ids: Optional[List[int]] = None):
        """
        Initialize the environment analyzer.
        
        Parameters
        ----------
        client_id : Optional[int]
            PyBullet client ID
        obstacle_ids : Optional[List[int]]
            List of obstacle IDs
        """
        self.client_id = client_id
        self.obstacle_ids = obstacle_ids or []
        
    def analyze_complexity(self, start: np.ndarray, goal: np.ndarray, 
                          sample_count: int = 100) -> Dict[str, float]:
        """
        Analyze environment complexity between start and goal.
        
        Parameters
        ----------
        start : np.ndarray
            Start position
        goal : np.ndarray
            Goal position
        sample_count : int
            Number of samples for analysis
            
        Returns
        -------
        Dict[str, float]
            Complexity metrics
        """
        # Sample points in the environment
        bounds = self._get_bounds(start, goal)
        collision_count = 0
        narrow_passage_count = 0
        
        # Check collision density
        for _ in range(sample_count):
            point = self._sample_in_bounds(bounds)
            
            if self._check_collision(point):
                collision_count += 1
                
                # Check for narrow passages
                if self._is_narrow_passage(point):
                    narrow_passage_count += 1
        
        collision_density = collision_count / sample_count
        narrow_passage_density = narrow_passage_count / max(1, collision_count)
        
        # Analyze path directness
        direct_distance = np.linalg.norm(goal - start)
        path_complexity = self._estimate_path_complexity(start, goal)
        
        # Calculate overall complexity score
        complexity_score = (
            collision_density * 0.4 +
            narrow_passage_density * 0.3 +
            path_complexity * 0.3
        )
        
        return {
            'collision_density': collision_density,
            'narrow_passage_density': narrow_passage_density,
            'path_complexity': path_complexity,
            'complexity_score': complexity_score,
            'direct_distance': direct_distance
        }
    
    def _get_bounds(self, start: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get sampling bounds around start and goal."""
        margin = np.array([2.0, 2.0, 1.0])
        min_bounds = np.minimum(start, goal) - margin
        max_bounds = np.maximum(start, goal) + margin
        min_bounds[2] = max(min_bounds[2], 0.5)
        return min_bounds, max_bounds
    
    def _sample_in_bounds(self, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Sample a random point within bounds."""
        min_bounds, max_bounds = bounds
        return np.array([
            random.uniform(min_bounds[0], max_bounds[0]),
            random.uniform(min_bounds[1], max_bounds[1]),
            random.uniform(min_bounds[2], max_bounds[2])
        ])
    
    def _check_collision(self, point: np.ndarray) -> bool:
        """Check if a point is in collision."""
        if not self.client_id or not self.obstacle_ids:
            return False
        
        try:
            import pybullet as p
            
            # Create a small sphere at the point to check collision
            sphere_radius = 0.1
            sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius, physicsClientId=self.client_id)
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_id, 
                                       basePosition=point, physicsClientId=self.client_id)
            
            # Check for collisions
            collision = False
            for obstacle_id in self.obstacle_ids:
                contacts = p.getContactPoints(bodyA=body_id, bodyB=obstacle_id, physicsClientId=self.client_id)
                if contacts:
                    collision = True
                    break
            
            # Clean up
            p.removeBody(body_id, physicsClientId=self.client_id)
            
            return collision
        except:
            return False
    
    def _is_narrow_passage(self, point: np.ndarray, check_radius: float = 0.5) -> bool:
        """Check if a point is in a narrow passage."""
        # Sample points around the given point
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1])
        ]
        
        free_directions = 0
        for direction in directions:
            test_point = point + direction * check_radius
            if not self._check_collision(test_point):
                free_directions += 1
        
        # Narrow passage if only 1-2 directions are free
        return free_directions <= 2
    
    def _estimate_path_complexity(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Estimate path complexity by sampling along direct path."""
        direction = goal - start
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return 0.0
        
        direction = direction / distance
        collision_count = 0
        sample_count = max(10, int(distance / 0.5))
        
        for i in range(sample_count):
            t = i / (sample_count - 1)
            point = start + t * distance * direction
            if self._check_collision(point):
                collision_count += 1
        
        return collision_count / sample_count


class AdaptiveRRTPlanner(OptimizedRRTPlanner):
    """
    Adaptive RRT planner that adjusts parameters based on environment complexity.
    
    This planner analyzes the environment and adapts its parameters to:
    - Use smaller step sizes in complex environments
    - Increase sampling iterations for difficult problems
    - Adjust goal sampling rate based on path complexity
    - Modify search strategies based on obstacle density
    """
    
    def __init__(self, 
                 start: np.ndarray, 
                 goal: np.ndarray,
                 client_id: Optional[int] = None,
                 obstacle_ids: Optional[List[int]] = None,
                 base_max_iterations: int = 1000,
                 base_step_size: float = 0.5,
                 base_goal_sample_rate: float = 0.1,
                 search_radius: float = 1.0,
                 adaptation_enabled: bool = True,
                 complexity_analysis_samples: int = 50):
        """
        Initialize the adaptive RRT planner.
        
        Parameters
        ----------
        start : np.ndarray
            Start position (x, y, z)
        goal : np.ndarray
            Goal position (x, y, z)
        client_id : Optional[int]
            PyBullet client ID for collision checking
        obstacle_ids : Optional[List[int]]
            List of obstacle IDs for collision checking
        base_max_iterations : int
            Base maximum number of iterations
        base_step_size : float
            Base step size for tree expansion
        base_goal_sample_rate : float
            Base probability of sampling the goal
        search_radius : float
            Radius for goal region
        adaptation_enabled : bool
            Whether to enable parameter adaptation
        complexity_analysis_samples : int
            Number of samples for complexity analysis
        """
        # Store base parameters
        self.base_max_iterations = base_max_iterations
        self.base_step_size = base_step_size
        self.base_goal_sample_rate = base_goal_sample_rate
        self.adaptation_enabled = adaptation_enabled
        self.complexity_analysis_samples = complexity_analysis_samples
        
        # Analyze environment complexity
        self.analyzer = EnvironmentAnalyzer(client_id, obstacle_ids)
        self.complexity_metrics = {}
        
        if adaptation_enabled:
            print("Analyzing environment complexity...")
            start_time = time.time()
            self.complexity_metrics = self.analyzer.analyze_complexity(
                start, goal, complexity_analysis_samples
            )
            analysis_time = time.time() - start_time
            print(f"Environment analysis completed in {analysis_time:.2f}s")
            print(f"Complexity metrics: {self.complexity_metrics}")
            
            # Adapt parameters based on complexity
            adapted_params = self._adapt_parameters()
            print(f"Adapted parameters: {adapted_params}")
        else:
            adapted_params = {
                'max_iterations': base_max_iterations,
                'step_size': base_step_size,
                'goal_sample_rate': base_goal_sample_rate
            }
        
        # Initialize parent class with adapted parameters
        super().__init__(
            start=start,
            goal=goal,
            client_id=client_id,
            obstacle_ids=obstacle_ids,
            max_iterations=adapted_params['max_iterations'],
            step_size=adapted_params['step_size'],
            goal_sample_rate=adapted_params['goal_sample_rate'],
            search_radius=search_radius,
            adaptive_sampling=True,
            early_termination=True
        )
        
        # Store adapted parameters for statistics
        self.adapted_params = adapted_params
    
    def _adapt_parameters(self) -> Dict[str, float]:
        """
        Adapt planning parameters based on environment complexity.
        
        Returns
        -------
        Dict[str, float]
            Adapted parameters
        """
        complexity_score = self.complexity_metrics.get('complexity_score', 0.5)
        collision_density = self.complexity_metrics.get('collision_density', 0.5)
        path_complexity = self.complexity_metrics.get('path_complexity', 0.5)
        direct_distance = self.complexity_metrics.get('direct_distance', 5.0)
        
        # Adapt max iterations based on complexity
        iteration_multiplier = 1.0 + complexity_score * 2.0  # 1.0 to 3.0
        if collision_density > 0.7:  # Very dense environment
            iteration_multiplier *= 1.5
        
        max_iterations = int(self.base_max_iterations * iteration_multiplier)
        max_iterations = min(max_iterations, 10000)  # Cap at 10k iterations
        
        # Adapt step size based on obstacle density and path complexity
        step_multiplier = 1.0 - collision_density * 0.5  # Smaller steps in dense environments
        if path_complexity > 0.6:  # Complex path
            step_multiplier *= 0.7
        
        step_size = self.base_step_size * step_multiplier
        step_size = max(step_size, 0.1)  # Minimum step size
        
        # Adapt goal sampling rate based on path complexity
        if path_complexity > 0.7:  # Very complex path
            goal_sample_rate = self.base_goal_sample_rate * 0.5  # Less goal sampling
        elif path_complexity < 0.3:  # Simple path
            goal_sample_rate = self.base_goal_sample_rate * 2.0  # More goal sampling
        else:
            goal_sample_rate = self.base_goal_sample_rate
        
        goal_sample_rate = min(goal_sample_rate, 0.5)  # Cap at 50%
        
        # Adjust based on distance
        if direct_distance > 10.0:  # Long distance
            max_iterations = int(max_iterations * 1.2)
            goal_sample_rate *= 0.8
        elif direct_distance < 2.0:  # Short distance
            max_iterations = int(max_iterations * 0.8)
            goal_sample_rate *= 1.5
        
        return {
            'max_iterations': max_iterations,
            'step_size': step_size,
            'goal_sample_rate': goal_sample_rate,
            'iteration_multiplier': iteration_multiplier,
            'step_multiplier': step_multiplier
        }
    
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path using adaptive parameters.
        
        Returns
        -------
        List[np.ndarray]
            List of waypoints from start to goal
        """
        if self.adaptation_enabled:
            print(f"Planning with adaptive parameters:")
            print(f"  Max iterations: {self.max_iterations}")
            print(f"  Step size: {self.step_size:.3f}")
            print(f"  Goal sample rate: {self.goal_sample_rate:.3f}")
        
        # Use parent's planning method
        return super().plan()
    
    def get_statistics(self) -> dict:
        """
        Get planning statistics including adaptation info.
        
        Returns
        -------
        dict
            Statistics dictionary
        """
        stats = super().get_statistics()
        
        # Add adaptation-specific statistics
        stats.update({
            'adaptation_enabled': self.adaptation_enabled,
            'complexity_metrics': self.complexity_metrics,
            'adapted_params': self.adapted_params if hasattr(self, 'adapted_params') else {},
            'base_max_iterations': self.base_max_iterations,
            'base_step_size': self.base_step_size,
            'base_goal_sample_rate': self.base_goal_sample_rate,
        })
        
        return stats


class MultiStrategyRRTPlanner:
    """
    Planner that tries multiple RRT strategies and selects the best result.
    
    This planner runs multiple RRT variants in parallel or sequentially
    and returns the best path found.
    """
    
    def __init__(self, 
                 start: np.ndarray, 
                 goal: np.ndarray,
                 client_id: Optional[int] = None,
                 obstacle_ids: Optional[List[int]] = None,
                 strategies: Optional[List[str]] = None,
                 time_budget: float = 10.0,
                 parallel_execution: bool = False):
        """
        Initialize the multi-strategy planner.
        
        Parameters
        ----------
        start : np.ndarray
            Start position
        goal : np.ndarray
            Goal position
        client_id : Optional[int]
            PyBullet client ID
        obstacle_ids : Optional[List[int]]
            List of obstacle IDs
        strategies : Optional[List[str]]
            List of strategies to try
        time_budget : float
            Total time budget for planning
        parallel_execution : bool
            Whether to run strategies in parallel
        """
        self.start = start
        self.goal = goal
        self.client_id = client_id
        self.obstacle_ids = obstacle_ids or []
        self.time_budget = time_budget
        self.parallel_execution = parallel_execution
        
        # Default strategies
        if strategies is None:
            self.strategies = ['adaptive', 'optimized', 'basic']
        else:
            self.strategies = strategies
        
        self.results = []
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan using multiple strategies and return the best path.
        
        Returns
        -------
        List[np.ndarray]
            Best path found
        """
        if self.parallel_execution:
            return self._plan_parallel()
        else:
            return self._plan_sequential()
    
    def _plan_sequential(self) -> List[np.ndarray]:
        """Plan using strategies sequentially."""
        start_time = time.time()
        time_per_strategy = self.time_budget / len(self.strategies)
        
        best_path = []
        best_cost = float('inf')
        
        for strategy in self.strategies:
            if time.time() - start_time >= self.time_budget:
                break
            
            print(f"Trying strategy: {strategy}")
            strategy_start = time.time()
            
            try:
                planner = self._create_planner(strategy, time_per_strategy)
                path = planner.plan()
                
                if path and len(path) > 1:
                    cost = self._calculate_path_cost(path)
                    planning_time = time.time() - strategy_start
                    
                    result = {
                        'strategy': strategy,
                        'path': path,
                        'cost': cost,
                        'time': planning_time,
                        'success': True
                    }
                    
                    if hasattr(planner, 'get_statistics'):
                        result['stats'] = planner.get_statistics()
                    
                    self.results.append(result)
                    
                    if cost < best_cost:
                        best_path = path
                        best_cost = cost
                        print(f"  New best path found with cost {cost:.2f}")
                    else:
                        print(f"  Path found with cost {cost:.2f}")
                else:
                    print(f"  No path found")
                    
            except Exception as e:
                print(f"  Strategy failed: {e}")
                self.results.append({
                    'strategy': strategy,
                    'path': [],
                    'cost': float('inf'),
                    'time': time.time() - strategy_start,
                    'success': False,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        print(f"Multi-strategy planning completed in {total_time:.2f}s")
        print(f"Best path cost: {best_cost:.2f}")
        
        return best_path
    
    def _plan_parallel(self) -> List[np.ndarray]:
        """Plan using strategies in parallel (simplified version)."""
        # For now, implement as sequential with shorter time budgets
        # True parallel execution would require threading/multiprocessing
        print("Parallel execution not fully implemented, using sequential with shorter budgets")
        original_budget = self.time_budget
        self.time_budget = original_budget * 0.8  # Slightly less time per strategy
        return self._plan_sequential()
    
    def _create_planner(self, strategy: str, time_budget: float):
        """Create a planner for the given strategy."""
        max_iterations = max(100, int(time_budget * 200))  # Rough estimate
        
        if strategy == 'adaptive':
            return AdaptiveRRTPlanner(
                start=self.start,
                goal=self.goal,
                client_id=self.client_id,
                obstacle_ids=self.obstacle_ids,
                base_max_iterations=max_iterations,
                adaptation_enabled=True
            )
        elif strategy == 'optimized':
            return OptimizedRRTPlanner(
                start=self.start,
                goal=self.goal,
                client_id=self.client_id,
                obstacle_ids=self.obstacle_ids,
                max_iterations=max_iterations,
                adaptive_sampling=True,
                early_termination=True
            )
        elif strategy == 'basic':
            from swarm.planners.rrt import RRTPlanner
            return RRTPlanner(
                start=self.start,
                goal=self.goal,
                client_id=self.client_id,
                obstacle_ids=self.obstacle_ids,
                max_iterations=max_iterations
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _calculate_path_cost(self, path: List[np.ndarray]) -> float:
        """Calculate the total cost of a path."""
        if len(path) < 2:
            return float('inf')
        
        total_cost = 0.0
        for i in range(1, len(path)):
            total_cost += np.linalg.norm(path[i] - path[i-1])
        
        return total_cost
    
    def get_statistics(self) -> dict:
        """Get statistics for all strategies tried."""
        return {
            'strategies_tried': len(self.results),
            'successful_strategies': len([r for r in self.results if r['success']]),
            'results': self.results,
            'time_budget': self.time_budget,
            'parallel_execution': self.parallel_execution
        }