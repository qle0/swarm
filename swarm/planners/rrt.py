"""
RRT (Rapidly-exploring Random Tree) path planner implementation.
"""
from typing import List, Tuple, Optional
import numpy as np
import random

from swarm.planners.base_planner import BasePlanner


class RRTNode:
    """Node in the RRT tree."""
    
    def __init__(self, position: np.ndarray):
        """
        Initialize an RRT node.
        
        Parameters
        ----------
        position : np.ndarray
            3D position of the node
        """
        self.position = position
        self.parent = None
        self.cost = 0.0


class RRTPlanner(BasePlanner):
    """
    RRT (Rapidly-exploring Random Tree) path planner.
    
    This algorithm builds a tree by randomly sampling points in the space
    and connecting them to the closest node in the tree if the connection
    is collision-free.
    """
    
    def __init__(self, 
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 client_id: int,
                 obstacle_ids: Optional[List[int]] = None,
                 max_iterations: int = 1000,
                 step_size: float = 0.2,
                 goal_sample_rate: float = 0.1,
                 search_radius: float = 5.0):
        """
        Initialize the RRT planner.
        
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
            Maximum number of iterations
        step_size : float
            Maximum distance between nodes
        goal_sample_rate : float
            Probability of sampling the goal position
        search_radius : float
            Maximum distance to search for the goal
        """
        super().__init__(start, goal, client_id, obstacle_ids)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.nodes = []
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal using RRT.
        
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
            # If direct path is possible, return it immediately
            goal_node = RRTNode(self.goal)
            goal_node.parent = start_node
            goal_node.cost = np.linalg.norm(self.goal - self.start)
            return [self.start, self.goal]
        
        # Limit iterations to prevent hanging
        max_iter = min(self.max_iterations, 1000)
        
        # Main RRT loop
        for i in range(max_iter):
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"RRT planning: iteration {i}/{max_iter}")
                
            # Sample a random point
            if random.random() < self.goal_sample_rate:
                random_point = self.goal
            else:
                random_point = self._sample_random_point()
            
            # Find the nearest node in the tree
            nearest_node = self._find_nearest_node(random_point)
            
            # Steer towards the random point
            new_position = self._steer(nearest_node.position, random_point)
            
            # Check if the new position is collision-free
            if self.check_collision(new_position):
                continue
            
            # Check if the path to the new position is collision-free
            if self._is_collision_free(nearest_node.position, new_position):
                continue
            
            # Create a new node and add it to the tree
            new_node = RRTNode(new_position)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)
            self.nodes.append(new_node)
            
            # Check if we can reach the goal from the new node
            distance_to_goal = np.linalg.norm(new_position - self.goal)
            if distance_to_goal < self.search_radius:
                # Try to connect to the goal
                if not self._is_collision_free(new_position, self.goal):
                    # Create the goal node and add it to the tree
                    goal_node = RRTNode(self.goal)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + distance_to_goal
                    self.nodes.append(goal_node)
                    
                    # Extract the path
                    return self._extract_path(goal_node)
        
        # If we reach here, no path was found
        # Return a simple direct path as fallback
        print("RRT: No path found after maximum iterations, using direct path")
        return [self.start, self.goal]
    
    def _sample_random_point(self) -> np.ndarray:
        """
        Sample a random point in the 3D space.
        
        Returns
        -------
        np.ndarray
            Random 3D point
        """
        # Make sure start and goal are numpy arrays with the right shape
        start = np.array(self.start).flatten()
        goal = np.array(self.goal).flatten()
        
        if start.size != 3 or goal.size != 3:
            # If we have invalid start/goal, use default bounds
            min_bounds = np.array([-5.0, -5.0, 0.5])
            max_bounds = np.array([5.0, 5.0, 5.0])
        else:
            # Define the sampling bounds based on the start and goal positions
            min_bounds = np.minimum(start, goal) - np.array([5.0, 5.0, 2.0])
            max_bounds = np.maximum(start, goal) + np.array([5.0, 5.0, 2.0])
            
            # Ensure minimum z-coordinate is above ground
            min_bounds[2] = max(min_bounds[2], 0.5)
        
        # Sample a random point within the bounds
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
            Point to find the nearest node to
            
        Returns
        -------
        RRTNode
            Nearest node in the tree
        """
        distances = [np.linalg.norm(node.position - point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """
        Steer from one point towards another, limited by the step size.
        
        Parameters
        ----------
        from_point : np.ndarray
            Starting point
        to_point : np.ndarray
            Target point
            
        Returns
        -------
        np.ndarray
            New point after steering
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
    
    def _extract_path(self, goal_node: RRTNode) -> List[np.ndarray]:
        """
        Extract the path from the start to the goal node.
        
        Parameters
        ----------
        goal_node : RRTNode
            Goal node
            
        Returns
        -------
        List[np.ndarray]
            List of waypoints from start to goal
        """
        path = []
        current_node = goal_node
        
        # Traverse the tree from the goal to the start
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
            
        # Reverse the path to get it from start to goal
        return path[::-1]