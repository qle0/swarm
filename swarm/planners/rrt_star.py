"""
RRT* (Rapidly-exploring Random Tree Star) path planner implementation.

RRT* is an asymptotically optimal variant of RRT that rewires the tree
to improve path quality as more samples are added.
"""
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import time
import math

from swarm.planners.base_planner import BasePlanner


class RRTStarNode:
    """Node in the RRT* tree."""
    
    def __init__(self, position: np.ndarray):
        """
        Initialize an RRT* node.
        
        Parameters
        ----------
        position : np.ndarray
            3D position of the node
        """
        self.position = position
        self.parent = None
        self.cost = float('inf')  # Cost from start
        self.children = []  # List of child nodes


class RRTStarPlanner(BasePlanner):
    """
    RRT* (Rapidly-exploring Random Tree Star) path planner.
    
    This algorithm builds a tree by randomly sampling points in the space
    and connecting them to the closest node in the tree if the connection
    is collision-free. It then rewires the tree to improve path quality.
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
                 rewire_radius: float = 0.5,
                 adaptive_params: bool = True):
        """
        Initialize the RRT* planner.
        
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
        rewire_radius : float
            Radius for rewiring the tree
        adaptive_params : bool
            Whether to adapt parameters based on environment complexity
        """
        super().__init__(start, goal, client_id, obstacle_ids)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.rewire_radius = rewire_radius
        self.adaptive_params = adaptive_params
        self.nodes = []
        self.goal_node = None
        
        # Metrics for adaptive parameters
        self.collision_count = 0
        self.sample_count = 0
        self.environment_complexity = 0.0
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal using RRT*.
        
        Returns
        -------
        List[np.ndarray]
            List of waypoints (3D positions) from start to goal
        """
        # Initialize the tree with the start node
        start_node = RRTStarNode(self.start)
        start_node.cost = 0.0
        self.nodes = [start_node]
        
        # Check if we can directly connect start to goal
        if not self._is_collision_free(self.start, self.goal):
            # If direct path is possible, return it immediately
            goal_node = RRTStarNode(self.goal)
            goal_node.parent = start_node
            goal_node.cost = np.linalg.norm(self.goal - self.start)
            start_node.children.append(goal_node)
            self.goal_node = goal_node
            return [self.start, self.goal]
        
        # Limit iterations to prevent hanging
        max_iter = min(self.max_iterations, 2000)
        
        # Calculate the dimension of the space
        dim = 3  # 3D space
        
        # Main RRT* loop
        for i in range(max_iter):
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"RRT* planning: iteration {i}/{max_iter}")
                
                # Adapt parameters based on environment complexity if enabled
                if self.adaptive_params and i > 0:
                    self._adapt_parameters()
                
            # Sample a random point
            if np.random.random() < self.goal_sample_rate:
                random_point = self.goal
            else:
                random_point = self._sample_random_point()
                self.sample_count += 1
            
            # Find the nearest node in the tree
            nearest_node_idx = self._find_nearest_node(random_point)
            nearest_node = self.nodes[nearest_node_idx]
            
            # Steer towards the random point
            new_position = self._steer(nearest_node.position, random_point)
            
            # Check if the new position is collision-free
            if self.check_collision(new_position):
                self.collision_count += 1
                continue
            
            # Check if the path to the new position is collision-free
            if self._is_collision_free(nearest_node.position, new_position):
                self.collision_count += 1
                continue
            
            # Create a new node
            new_node = RRTStarNode(new_position)
            new_node_idx = len(self.nodes)
            
            # Calculate the cost to reach the new node
            new_cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)
            
            # Find nearby nodes for potential rewiring
            nearby_indices = self._find_nearby_nodes(new_position, self.rewire_radius)
            
            # Connect the new node to the best parent
            min_cost = new_cost
            min_idx = nearest_node_idx
            
            for idx in nearby_indices:
                node = self.nodes[idx]
                
                # Calculate potential cost through this node
                potential_cost = node.cost + np.linalg.norm(new_position - node.position)
                
                # Check if this path is better and collision-free
                if potential_cost < min_cost and not self._is_collision_free(node.position, new_position):
                    min_cost = potential_cost
                    min_idx = idx
            
            # Set the parent and cost of the new node
            new_node.parent = self.nodes[min_idx]
            new_node.cost = min_cost
            self.nodes[min_idx].children.append(new_node)
            
            # Add the new node to the tree
            self.nodes.append(new_node)
            
            # Rewire the tree
            self._rewire_tree(new_node_idx, nearby_indices)
            
            # Check if we can reach the goal from the new node
            distance_to_goal = np.linalg.norm(new_position - self.goal)
            if distance_to_goal < self.search_radius:
                # Try to connect to the goal
                if not self._is_collision_free(new_position, self.goal):
                    # Calculate the cost to reach the goal
                    goal_cost = new_node.cost + distance_to_goal
                    
                    # Create the goal node if it doesn't exist or if we found a better path
                    if self.goal_node is None or goal_cost < self.goal_node.cost:
                        if self.goal_node is not None and self.goal_node.parent is not None:
                            # Remove the goal node from its parent's children
                            self.goal_node.parent.children.remove(self.goal_node)
                        
                        # Create a new goal node or update the existing one
                        if self.goal_node is None:
                            self.goal_node = RRTStarNode(self.goal)
                            self.nodes.append(self.goal_node)
                        
                        # Update the goal node
                        self.goal_node.parent = new_node
                        self.goal_node.cost = goal_cost
                        new_node.children.append(self.goal_node)
            
            # If we've found a path to the goal and have enough iterations, we can stop
            if self.goal_node is not None and i > min(500, max_iter // 2):
                # Check if the path quality is good enough
                path_quality = self._evaluate_path_quality()
                if path_quality > 0.8:  # 80% quality threshold
                    break
        
        # Extract the path if we found one
        if self.goal_node is not None:
            path = self._extract_path(self.goal_node)
            return path
        
        # If we reach here, no path was found
        # Return a simple direct path as fallback
        print("RRT*: No path found after maximum iterations, using direct path")
        return [self.start, self.goal]
    
    def _adapt_parameters(self):
        """Adapt parameters based on environment complexity."""
        if self.sample_count == 0:
            return
            
        # Calculate environment complexity based on collision rate
        self.environment_complexity = self.collision_count / self.sample_count
        
        # Adjust parameters based on complexity
        if self.environment_complexity > 0.7:  # High complexity
            self.step_size = max(0.1, self.step_size * 0.9)  # Decrease step size
            self.goal_sample_rate = min(0.3, self.goal_sample_rate * 1.1)  # Increase goal bias
            self.rewire_radius = max(0.3, self.rewire_radius * 0.9)  # Decrease rewire radius
        elif self.environment_complexity < 0.3:  # Low complexity
            self.step_size = min(0.5, self.step_size * 1.1)  # Increase step size
            self.goal_sample_rate = max(0.1, self.goal_sample_rate * 0.9)  # Decrease goal bias
            self.rewire_radius = min(1.0, self.rewire_radius * 1.1)  # Increase rewire radius
    
    def _evaluate_path_quality(self) -> float:
        """
        Evaluate the quality of the current path to the goal.
        
        Returns
        -------
        float
            Path quality score between 0 and 1
        """
        if self.goal_node is None:
            return 0.0
            
        # Calculate the direct distance from start to goal
        direct_distance = np.linalg.norm(self.goal - self.start)
        
        # Calculate the path length
        path = self._extract_path(self.goal_node)
        path_length = 0.0
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(path[i+1] - path[i])
        
        # Calculate the path quality (ratio of direct distance to path length)
        # A perfect path would have a ratio of 1.0
        path_quality = direct_distance / path_length if path_length > 0 else 0.0
        
        # Normalize to [0, 1]
        return min(1.0, path_quality)
    
    def _find_nearby_nodes(self, position: np.ndarray, radius: float) -> List[int]:
        """
        Find nodes within a certain radius of a position.
        
        Parameters
        ----------
        position : np.ndarray
            Position to search around
        radius : float
            Search radius
            
        Returns
        -------
        List[int]
            Indices of nearby nodes
        """
        nearby_indices = []
        
        for i, node in enumerate(self.nodes):
            distance = np.linalg.norm(node.position - position)
            if distance < radius:
                nearby_indices.append(i)
                
        return nearby_indices
    
    def _rewire_tree(self, new_node_idx: int, nearby_indices: List[int]):
        """
        Rewire the tree to improve path quality.
        
        Parameters
        ----------
        new_node_idx : int
            Index of the new node
        nearby_indices : List[int]
            Indices of nearby nodes
        """
        new_node = self.nodes[new_node_idx]
        
        for idx in nearby_indices:
            # Skip the parent of the new node
            if self.nodes[idx] is new_node.parent:
                continue
                
            node = self.nodes[idx]
            
            # Calculate the potential new cost for this node
            potential_cost = new_node.cost + np.linalg.norm(node.position - new_node.position)
            
            # Check if rewiring would improve the cost
            if potential_cost < node.cost:
                # Check if the path is collision-free
                if not self._is_collision_free(new_node.position, node.position):
                    # Remove the node from its parent's children
                    if node.parent is not None:
                        node.parent.children.remove(node)
                    
                    # Rewire the node
                    node.parent = new_node
                    node.cost = potential_cost
                    new_node.children.append(node)
                    
                    # Update the costs of all descendants
                    self._update_descendants_cost(node)
    
    def _update_descendants_cost(self, node: RRTStarNode):
        """
        Update the costs of all descendants of a node.
        
        Parameters
        ----------
        node : RRTStarNode
            Node whose descendants need to be updated
        """
        for child in node.children:
            # Update the child's cost
            child.cost = node.cost + np.linalg.norm(child.position - node.position)
            
            # Recursively update the costs of all descendants
            self._update_descendants_cost(child)
    
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
            np.random.uniform(min_bounds[0], max_bounds[0]),
            np.random.uniform(min_bounds[1], max_bounds[1]),
            np.random.uniform(min_bounds[2], max_bounds[2])
        ])
    
    def _find_nearest_node(self, point: np.ndarray) -> int:
        """
        Find the index of the nearest node in the tree to a given point.
        
        Parameters
        ----------
        point : np.ndarray
            Point to find the nearest node to
            
        Returns
        -------
        int
            Index of the nearest node in the tree
        """
        distances = [np.linalg.norm(node.position - point) for node in self.nodes]
        return np.argmin(distances)
    
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
    
    def _extract_path(self, goal_node: RRTStarNode) -> List[np.ndarray]:
        """
        Extract the path from the start to the goal node.
        
        Parameters
        ----------
        goal_node : RRTStarNode
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