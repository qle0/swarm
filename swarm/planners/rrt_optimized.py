"""
Optimized RRT (Rapidly-exploring Random Tree) path planner with adaptive sampling.
"""
from typing import List, Tuple, Optional
import numpy as np
import random
from dataclasses import dataclass

from swarm.planners.base_planner import BasePlanner


@dataclass
class RRTNode:
    """Node in the RRT tree."""
    position: np.ndarray
    parent: Optional['RRTNode'] = None
    cost: float = 0.0
    children: List['RRTNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class OptimizedRRTPlanner(BasePlanner):
    """
    Optimized RRT planner with adaptive sampling and improved exploration.
    
    Improvements:
    - Adaptive step size based on obstacle density
    - Biased sampling towards unexplored regions
    - Early termination when path is found
    - Dynamic goal sampling rate
    """
    
    def __init__(self, 
                 start: np.ndarray, 
                 goal: np.ndarray,
                 client_id: Optional[int] = None,
                 obstacle_ids: Optional[List[int]] = None,
                 max_iterations: int = 2000,
                 step_size: float = 0.5,
                 goal_sample_rate: float = 0.1,
                 search_radius: float = 1.0,
                 adaptive_sampling: bool = True,
                 early_termination: bool = True):
        """
        Initialize the optimized RRT planner.
        
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
        max_iterations : int
            Maximum number of iterations
        step_size : float
            Maximum step size for tree expansion
        goal_sample_rate : float
            Initial probability of sampling the goal
        search_radius : float
            Radius for goal region
        adaptive_sampling : bool
            Whether to use adaptive sampling
        early_termination : bool
            Whether to terminate early when path is found
        """
        super().__init__(client_id, obstacle_ids)
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.max_iterations = max_iterations
        self.base_step_size = step_size
        self.step_size = step_size
        self.initial_goal_sample_rate = goal_sample_rate
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.adaptive_sampling = adaptive_sampling
        self.early_termination = early_termination
        self.nodes = []
        
        # Adaptive parameters
        self.obstacle_density_map = {}
        self.exploration_bias = 0.1
        self.min_step_size = 0.1
        self.max_step_size = 1.0
        
        # Statistics
        self.collision_checks = 0
        self.successful_expansions = 0
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal using optimized RRT.
        
        Returns
        -------
        List[np.ndarray]
            List of waypoints (3D positions) from start to goal
        """
        # Initialize the tree with the start node
        start_node = RRTNode(self.start)
        self.nodes = [start_node]
        
        # Check if we can directly connect start to goal
        if not self._is_collision_free(self.start, self.goal):
            return [self.start, self.goal]
        
        # Main RRT loop
        for iteration in range(self.max_iterations):
            # Adaptive goal sampling rate
            if self.adaptive_sampling:
                progress = iteration / self.max_iterations
                self.goal_sample_rate = self.initial_goal_sample_rate * (1 + progress * 2)
                self.goal_sample_rate = min(self.goal_sample_rate, 0.5)
            
            # Sample a random point
            if random.random() < self.goal_sample_rate:
                random_point = self.goal
            else:
                random_point = self._adaptive_sample()
            
            # Find the nearest node in the tree
            nearest_node = self._find_nearest_node(random_point)
            
            # Adaptive step size based on local obstacle density
            if self.adaptive_sampling:
                self.step_size = self._adaptive_step_size(nearest_node.position, random_point)
            
            # Steer towards the random point
            new_position = self._steer(nearest_node.position, random_point)
            
            # Check if the new position is collision-free
            self.collision_checks += 1
            if self.check_collision(new_position):
                continue
            
            # Check if the path to the new position is collision-free
            if self._is_collision_free(nearest_node.position, new_position):
                continue
            
            # Create a new node and add it to the tree
            new_node = RRTNode(new_position)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)
            nearest_node.children.append(new_node)
            self.nodes.append(new_node)
            self.successful_expansions += 1
            
            # Update obstacle density map
            if self.adaptive_sampling:
                self._update_obstacle_density(new_position)
            
            # Check if we can reach the goal from the new node
            distance_to_goal = np.linalg.norm(new_position - self.goal)
            if distance_to_goal < self.search_radius:
                # Try to connect to the goal
                if not self._is_collision_free(new_position, self.goal):
                    # Create the goal node and add it to the tree
                    goal_node = RRTNode(self.goal)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + distance_to_goal
                    new_node.children.append(goal_node)
                    self.nodes.append(goal_node)
                    
                    # Extract and return the path
                    path = self._extract_path(goal_node)
                    if self.early_termination:
                        print(f"RRT found path in {iteration + 1} iterations")
                        return path
        
        # If we reach here, try to find the closest node to goal
        closest_node = min(self.nodes, key=lambda n: np.linalg.norm(n.position - self.goal))
        closest_distance = np.linalg.norm(closest_node.position - self.goal)
        
        if closest_distance < self.search_radius * 2:
            # Try to connect the closest node to the goal
            if not self._is_collision_free(closest_node.position, self.goal):
                goal_node = RRTNode(self.goal)
                goal_node.parent = closest_node
                goal_node.cost = closest_node.cost + closest_distance
                path = self._extract_path(goal_node)
                print(f"RRT found approximate path (distance to goal: {closest_distance:.2f})")
                return path
        
        print("RRT: No path found, using direct path")
        return [self.start, self.goal]
    
    def _adaptive_sample(self) -> np.ndarray:
        """
        Sample a point with bias towards unexplored regions.
        
        Returns
        -------
        np.ndarray
            Sampled 3D point
        """
        if random.random() < self.exploration_bias and len(self.nodes) > 10:
            # Sample in unexplored regions
            return self._sample_unexplored_region()
        else:
            # Regular random sampling
            return self._sample_random_point()
    
    def _sample_unexplored_region(self) -> np.ndarray:
        """
        Sample a point in regions with low node density.
        
        Returns
        -------
        np.ndarray
            Sampled 3D point
        """
        # Find regions with low node density
        bounds = self._get_sampling_bounds()
        
        # Divide space into grid cells and find least populated ones
        grid_size = 1.0
        cell_counts = {}
        
        for node in self.nodes:
            cell = tuple(np.floor(node.position / grid_size).astype(int))
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
        
        # Sample from a cell with low density
        if cell_counts:
            min_count = min(cell_counts.values())
            low_density_cells = [cell for cell, count in cell_counts.items() if count == min_count]
            
            if low_density_cells:
                # Choose a random low-density cell
                chosen_cell = random.choice(low_density_cells)
                cell_center = np.array(chosen_cell) * grid_size + grid_size / 2
                
                # Sample within the cell
                offset = (np.random.random(3) - 0.5) * grid_size
                sample = cell_center + offset
                
                # Ensure sample is within bounds
                sample = np.clip(sample, bounds[0], bounds[1])
                return sample
        
        # Fallback to regular sampling
        return self._sample_random_point()
    
    def _adaptive_step_size(self, from_point: np.ndarray, to_point: np.ndarray) -> float:
        """
        Compute adaptive step size based on local obstacle density.
        
        Parameters
        ----------
        from_point : np.ndarray
            Starting point
        to_point : np.ndarray
            Target point
            
        Returns
        -------
        float
            Adaptive step size
        """
        # Get obstacle density around the from_point
        density = self._get_obstacle_density(from_point)
        
        # Adjust step size inversely to density
        if density > 0.5:
            step_size = self.min_step_size
        elif density > 0.2:
            step_size = self.base_step_size * 0.5
        else:
            step_size = min(self.max_step_size, self.base_step_size * 1.5)
        
        return step_size
    
    def _get_obstacle_density(self, position: np.ndarray, radius: float = 1.0) -> float:
        """
        Estimate obstacle density around a position.
        
        Parameters
        ----------
        position : np.ndarray
            Position to check
        radius : float
            Radius for density estimation
            
        Returns
        -------
        float
            Obstacle density (0.0 to 1.0)
        """
        # Use cached density if available
        cell = tuple(np.floor(position).astype(int))
        if cell in self.obstacle_density_map:
            return self.obstacle_density_map[cell]
        
        # Sample points around the position and check for collisions
        num_samples = 20
        collision_count = 0
        
        for _ in range(num_samples):
            # Sample a point within the radius
            angle = random.uniform(0, 2 * np.pi)
            elevation = random.uniform(-np.pi/4, np.pi/4)
            r = random.uniform(0, radius)
            
            sample_point = position + r * np.array([
                np.cos(elevation) * np.cos(angle),
                np.cos(elevation) * np.sin(angle),
                np.sin(elevation)
            ])
            
            if self.check_collision(sample_point):
                collision_count += 1
        
        density = collision_count / num_samples
        self.obstacle_density_map[cell] = density
        return density
    
    def _update_obstacle_density(self, position: np.ndarray):
        """
        Update obstacle density map based on successful expansion.
        
        Parameters
        ----------
        position : np.ndarray
            Position of successful expansion
        """
        cell = tuple(np.floor(position).astype(int))
        # Successful expansion indicates lower obstacle density
        current_density = self.obstacle_density_map.get(cell, 0.5)
        self.obstacle_density_map[cell] = current_density * 0.9
    
    def _get_sampling_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sampling bounds based on start and goal positions.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Min and max bounds for sampling
        """
        start = np.array(self.start).flatten()
        goal = np.array(self.goal).flatten()
        
        if start.size != 3 or goal.size != 3:
            return np.array([-5.0, -5.0, 0.5]), np.array([5.0, 5.0, 5.0])
        
        min_bounds = np.minimum(start, goal) - np.array([3.0, 3.0, 1.0])
        max_bounds = np.maximum(start, goal) + np.array([3.0, 3.0, 1.0])
        min_bounds[2] = max(min_bounds[2], 0.5)
        
        return min_bounds, max_bounds
    
    def _sample_random_point(self) -> np.ndarray:
        """
        Sample a random point in the 3D space.
        
        Returns
        -------
        np.ndarray
            Random 3D point
        """
        min_bounds, max_bounds = self._get_sampling_bounds()
        
        return np.array([
            random.uniform(min_bounds[0], max_bounds[0]),
            random.uniform(min_bounds[1], max_bounds[1]),
            random.uniform(min_bounds[2], max_bounds[2])
        ])
    
    def _find_nearest_node(self, point: np.ndarray) -> RRTNode:
        """
        Find the nearest node in the tree to a given point.
        
        Parameters
        ----------
        point : np.ndarray
            Target point
            
        Returns
        -------
        RRTNode
            Nearest node in the tree
        """
        distances = [np.linalg.norm(node.position - point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """
        Steer from one point towards another with the current step size.
        
        Parameters
        ----------
        from_point : np.ndarray
            Starting point
        to_point : np.ndarray
            Target point
            
        Returns
        -------
        np.ndarray
            New position after steering
        """
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_point
        else:
            return from_point + (direction / distance) * self.step_size
    
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
        
        if distance < 1e-6:
            return False
            
        direction = direction / distance
        
        # Check for collisions along the path
        num_steps = max(3, int(distance / (self.step_size / 3)) + 1)
        for i in range(1, num_steps):
            point = from_point + direction * (i * distance / num_steps)
            if self.check_collision(point):
                return True
                
        return False
    
    def _extract_path(self, goal_node: RRTNode) -> List[np.ndarray]:
        """
        Extract the path from the start to the goal node.

        Parameters
        ----------
        goal_node : RRTNode
            Goal node in the tree

        Returns
        -------
        List[np.ndarray]
            Path from start to goal
        """
        path = []
        current_node = goal_node
        
        while current_node is not None:
            path.append(current_node.position.copy())
            current_node = current_node.parent
        
        path.reverse()
        return path
    
    def get_statistics(self) -> dict:
        """
        Get planning statistics.
        
        Returns
        -------
        dict
            Statistics dictionary
        """
        return {
            'nodes_created': len(self.nodes),
            'collision_checks': self.collision_checks,
            'successful_expansions': self.successful_expansions,
            'obstacle_density_cells': len(self.obstacle_density_map),
            'final_step_size': self.step_size,
            'final_goal_sample_rate': self.goal_sample_rate,
        }