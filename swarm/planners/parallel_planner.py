"""
Parallel path planner that runs multiple planning algorithms in parallel
and selects the best result.
"""
from typing import List, Tuple, Optional, Dict, Any, Type
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from swarm.planners.base_planner import BasePlanner
from swarm.planners.rrt import RRTPlanner
from swarm.planners.rrt_optimized import OptimizedRRTPlanner
from swarm.planners.adaptive_rrt import AdaptiveRRTPlanner
from swarm.planners.rrt_dijkstra import RRTDijkstraPlanner
from swarm.planners.rrt_astar import RRTAStarPlanner
from swarm.planners.rrt_star import RRTStarPlanner
from swarm.planners.astar import AStarPlanner
from swarm.planners.dijkstra import DijkstraPlanner


class PlannerResult:
    """Result of a path planning algorithm."""
    
    def __init__(self, 
                 planner_name: str, 
                 path: List[np.ndarray], 
                 planning_time: float,
                 path_length: float,
                 path_quality: float):
        """
        Initialize a planner result.
        
        Parameters
        ----------
        planner_name : str
            Name of the planner
        path : List[np.ndarray]
            Planned path
        planning_time : float
            Time taken to plan the path (seconds)
        path_length : float
            Length of the path
        path_quality : float
            Quality of the path (0-1)
        """
        self.planner_name = planner_name
        self.path = path
        self.planning_time = planning_time
        self.path_length = path_length
        self.path_quality = path_quality
        
        # Calculate a score based on path quality and planning time
        # Higher is better
        self.score = self._calculate_score()
        
    def _calculate_score(self) -> float:
        """
        Calculate a score for the planner result.
        
        Returns
        -------
        float
            Score for the planner result
        """
        # Weight factors for different metrics
        quality_weight = 0.7
        time_weight = 0.2
        length_weight = 0.1
        
        # Normalize planning time (lower is better)
        # Assume planning time is between 0 and 10 seconds
        normalized_time = max(0, 1.0 - self.planning_time / 10.0)
        
        # Normalize path length (lower is better)
        # Use path quality as a proxy for normalized length
        normalized_length = self.path_quality
        
        # Calculate the score
        score = (quality_weight * self.path_quality + 
                 time_weight * normalized_time + 
                 length_weight * normalized_length)
                 
        return score


class ParallelPlanner(BasePlanner):
    """
    Parallel path planner that runs multiple planning algorithms in parallel
    and selects the best result.
    """
    
    def __init__(self, 
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 client_id: int,
                 obstacle_ids: Optional[List[int]] = None,
                 max_iterations: int = 1000,
                 step_size: float = 0.2,
                 goal_sample_rate: float = 0.2,
                 search_radius: float = 1.5,
                 timeout: float = 5.0,
                 planners: Optional[List[str]] = None):
        """
        Initialize the parallel planner.
        
        Parameters
        ----------
        start : Tuple[float, float, float]
            Start position (x, y, z)
        goal : Tuple[float, float, float]
            Goal position (x, y, z)
        client_id : int
            PyBullet client ID
        obstacle_ids : Optional[List[int]]
            List of obstacle IDs in the PyBullet simulation
        max_iterations : int
            Maximum number of iterations for each planner
        step_size : float
            Step size for each planner
        goal_sample_rate : float
            Goal sample rate for each planner
        search_radius : float
            Search radius for each planner
        timeout : float
            Timeout for planning (seconds)
        planners : Optional[List[str]]
            List of planners to use. If None, all available planners are used.
        """
        super().__init__(start, goal, client_id, obstacle_ids)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.timeout = timeout
        
        # Available planners
        self.available_planners = {
            'rrt': RRTPlanner,
            'rrt_dijkstra': RRTDijkstraPlanner,
            'rrt_astar': RRTAStarPlanner,
            'rrt_star': RRTStarPlanner
        }
        
        # Select planners to use
        if planners is None:
            self.planners = list(self.available_planners.keys())
        else:
            self.planners = [p for p in planners if p in self.available_planners]
            
        # Results from each planner
        self.results: Dict[str, PlannerResult] = {}
        
        # Best result
        self.best_result: Optional[PlannerResult] = None
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal using multiple planners in parallel.
        
        Returns
        -------
        List[np.ndarray]
            List of waypoints (3D positions) from start to goal
        """
        # Check if we can directly connect start to goal
        if not self._is_collision_free(self.start, self.goal):
            # If direct path is possible, return it immediately
            return [self.start, self.goal]
            
        # Run planners in parallel
        with ThreadPoolExecutor(max_workers=len(self.planners)) as executor:
            # Submit planning tasks
            futures = {}
            for planner_name in self.planners:
                future = executor.submit(self._run_planner, planner_name)
                futures[future] = planner_name
                
            # Wait for results with timeout
            start_time = time.time()
            completed_futures = []
            
            for future in as_completed(futures):
                # Check if we've exceeded the timeout
                if time.time() - start_time > self.timeout:
                    break
                    
                completed_futures.append(future)
                
            # Process results
            for future in completed_futures:
                planner_name = futures[future]
                try:
                    result = future.result()
                    self.results[planner_name] = result
                    
                    # Update the best result
                    if self.best_result is None or result.score > self.best_result.score:
                        self.best_result = result
                except Exception as e:
                    print(f"Error running planner {planner_name}: {e}")
        
        # If we have a best result, return its path
        if self.best_result is not None:
            print(f"Selected planner: {self.best_result.planner_name}")
            print(f"Path quality: {self.best_result.path_quality:.4f}")
            print(f"Planning time: {self.best_result.planning_time:.4f} s")
            print(f"Path length: {self.best_result.path_length:.4f} m")
            return self.best_result.path
            
        # If no planner succeeded, return a direct path
        print("No planner succeeded, using direct path")
        return [self.start, self.goal]
    
    def _run_planner(self, planner_name: str) -> PlannerResult:
        """
        Run a specific planner.
        
        Parameters
        ----------
        planner_name : str
            Name of the planner to run
            
        Returns
        -------
        PlannerResult
            Result of the planner
        """
        # Get the planner class
        planner_class = self.available_planners[planner_name]
        
        # Create the planner
        planner = planner_class(
            start=self.start,
            goal=self.goal,
            client_id=self.client_id,
            obstacle_ids=self.obstacle_ids,
            max_iterations=self.max_iterations,
            step_size=self.step_size,
            goal_sample_rate=self.goal_sample_rate,
            search_radius=self.search_radius
        )
        
        # Measure planning time
        start_time = time.time()
        
        # Plan the path
        path = planner.plan()
        
        # Calculate planning time
        planning_time = time.time() - start_time
        
        # Calculate path length
        path_length = self._calculate_path_length(path)
        
        # Calculate path quality
        path_quality = self._calculate_path_quality(path)
        
        # Create and return the result
        return PlannerResult(
            planner_name=planner_name,
            path=path,
            planning_time=planning_time,
            path_length=path_length,
            path_quality=path_quality
        )
    
    def _calculate_path_length(self, path: List[np.ndarray]) -> float:
        """
        Calculate the length of a path.
        
        Parameters
        ----------
        path : List[np.ndarray]
            Path to calculate the length of
            
        Returns
        -------
        float
            Length of the path
        """
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length
    
    def _calculate_path_quality(self, path: List[np.ndarray]) -> float:
        """
        Calculate the quality of a path.
        
        Parameters
        ----------
        path : List[np.ndarray]
            Path to calculate the quality of
            
        Returns
        -------
        float
            Quality of the path (0-1)
        """
        # Calculate the direct distance from start to goal
        direct_distance = np.linalg.norm(self.goal - self.start)
        
        # Calculate the path length
        path_length = self._calculate_path_length(path)
        
        # Calculate the path quality (ratio of direct distance to path length)
        # A perfect path would have a ratio of 1.0
        path_quality = direct_distance / path_length if path_length > 0 else 0.0
        
        # Normalize to [0, 1]
        return min(1.0, path_quality)
    
    def _is_collision_free(self, from_point: np.ndarray, to_point: np.ndarray) -> bool:
        """
        Check if a straight line between two points is collision-free.
        
        Parameters
        ----------
        from_point : np.ndarray
            Starting point
        to_point : np.ndarray
            End point
            
        Returns
        -------
        bool
            True if there is a collision, False otherwise
        """
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:  # Points are very close
            return False
            
        direction = direction / distance
        
        # Check for collisions along the path
        num_steps = max(3, int(distance / (self.step_size / 2)) + 1)
        for i in range(1, num_steps):
            point = from_point + direction * (i * distance / num_steps)
            if self.check_collision(point):
                return True
                
        return False
    
    def visualize_path(self, path: List[np.ndarray], color: Tuple[float, float, float] = (0, 1, 0), line_width: float = 3.0):
        """
        Visualize the path in PyBullet.
        
        Parameters
        ----------
        path : List[np.ndarray]
            List of waypoints
        color : Tuple[float, float, float]
            RGB color for the path
        line_width : float
            Line width for the path
        """
        if self.client_id is None:
            return
            
        import pybullet as p
        
        # Draw lines between consecutive waypoints
        for i in range(len(path) - 1):
            p.addUserDebugLine(
                path[i],
                path[i + 1],
                color,
                lineWidth=line_width,
                lifeTime=0,
                physicsClientId=self.client_id
            )