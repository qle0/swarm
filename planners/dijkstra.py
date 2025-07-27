"""
Dijkstra's algorithm for path planning.
"""
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import heapq

from swarm.planners.base_planner import BasePlanner


class DijkstraNode:
    """Node in the Dijkstra graph."""
    
    def __init__(self, position: np.ndarray, index: Tuple[int, int, int]):
        """
        Initialize a Dijkstra node.
        
        Parameters
        ----------
        position : np.ndarray
            3D position of the node
        index : Tuple[int, int, int]
            Grid index of the node (i, j, k)
        """
        self.position = position
        self.index = index
        self.g_cost = float('inf')  # Cost from start
        self.parent = None
        
    def __lt__(self, other):
        """
        Comparison operator for priority queue.
        
        Parameters
        ----------
        other : DijkstraNode
            Other node to compare with
            
        Returns
        -------
        bool
            True if this node has lower g_cost than the other
        """
        return self.g_cost < other.g_cost


class DijkstraPlanner(BasePlanner):
    """
    Dijkstra's algorithm for path planning.
    
    This algorithm creates a grid of nodes and finds the shortest path
    from the start to the goal by exploring nodes in order of increasing
    cost from the start.
    """
    
    def __init__(self, 
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 client_id: int,
                 obstacle_ids: Optional[List[int]] = None,
                 resolution: float = 0.2):
        """
        Initialize the Dijkstra planner.
        
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
        resolution : float
            Grid resolution (cell size)
        """
        super().__init__(start, goal, client_id, obstacle_ids)
        self.resolution = resolution
        self.grid = {}  # Dictionary mapping grid indices to nodes
        
        # Define the grid bounds
        self.min_bounds = np.minimum(self.start, self.goal) - np.array([5.0, 5.0, 2.0])
        self.max_bounds = np.maximum(self.start, self.goal) + np.array([5.0, 5.0, 2.0])
        
        # Ensure minimum z-coordinate is above ground
        self.min_bounds[2] = max(self.min_bounds[2], 0.5)
        
        # Calculate grid dimensions
        self.grid_size = np.ceil((self.max_bounds - self.min_bounds) / self.resolution).astype(int)
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal using Dijkstra's algorithm.
        
        Returns
        -------
        List[np.ndarray]
            List of waypoints (3D positions) from start to goal
        """
        # Create the grid
        self._create_grid()
        
        # Get the start and goal nodes
        start_index = self._position_to_index(self.start)
        goal_index = self._position_to_index(self.goal)
        
        if start_index not in self.grid:
            print(f"Dijkstra: Start position {self.start} is not in the grid")
            return []
            
        if goal_index not in self.grid:
            print(f"Dijkstra: Goal position {self.goal} is not in the grid")
            return []
            
        start_node = self.grid[start_index]
        goal_node = self.grid[goal_index]
        
        # Initialize the start node
        start_node.g_cost = 0
        
        # Initialize the open and closed sets
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        
        # Main Dijkstra loop
        while open_set:
            # Get the node with the lowest cost
            current_node = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current_node.index == goal_node.index:
                return self._extract_path(current_node)
                
            # Add the current node to the closed set
            closed_set.add(current_node.index)
            
            # Explore neighbors
            for neighbor_node in self._get_neighbors(current_node):
                # Skip if the neighbor is in the closed set
                if neighbor_node.index in closed_set:
                    continue
                    
                # Calculate the cost to reach the neighbor
                tentative_g_cost = current_node.g_cost + np.linalg.norm(
                    neighbor_node.position - current_node.position)
                
                # If we found a better path to the neighbor
                if tentative_g_cost < neighbor_node.g_cost:
                    neighbor_node.g_cost = tentative_g_cost
                    neighbor_node.parent = current_node
                    
                    # Add the neighbor to the open set
                    if neighbor_node.index not in [node.index for node in open_set]:
                        heapq.heappush(open_set, neighbor_node)
        
        # If we reach here, no path was found
        print("Dijkstra: No path found")
        return []
    
    def _create_grid(self):
        """Create a grid of nodes."""
        # Create nodes for all grid cells
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    index = (i, j, k)
                    position = self._index_to_position(index)
                    
                    # Skip if the position is in collision
                    if self.check_collision(position):
                        continue
                        
                    # Create a node for this grid cell
                    self.grid[index] = DijkstraNode(position, index)
    
    def _position_to_index(self, position: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert a 3D position to a grid index.
        
        Parameters
        ----------
        position : np.ndarray
            3D position
            
        Returns
        -------
        Tuple[int, int, int]
            Grid index (i, j, k)
        """
        index = np.floor((position - self.min_bounds) / self.resolution).astype(int)
        return tuple(index)
    
    def _index_to_position(self, index: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert a grid index to a 3D position.
        
        Parameters
        ----------
        index : Tuple[int, int, int]
            Grid index (i, j, k)
            
        Returns
        -------
        np.ndarray
            3D position
        """
        return self.min_bounds + (np.array(index) + 0.5) * self.resolution
    
    def _get_neighbors(self, node: DijkstraNode) -> List[DijkstraNode]:
        """
        Get the neighbors of a node.
        
        Parameters
        ----------
        node : DijkstraNode
            Node to get neighbors for
            
        Returns
        -------
        List[DijkstraNode]
            List of neighbor nodes
        """
        neighbors = []
        i, j, k = node.index
        
        # Check all 26 neighbors (3x3x3 cube minus the center)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    # Skip the center
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                        
                    neighbor_index = (i + di, j + dj, k + dk)
                    
                    # Skip if the neighbor is outside the grid
                    if (neighbor_index[0] < 0 or neighbor_index[0] >= self.grid_size[0] or
                        neighbor_index[1] < 0 or neighbor_index[1] >= self.grid_size[1] or
                        neighbor_index[2] < 0 or neighbor_index[2] >= self.grid_size[2]):
                        continue
                        
                    # Skip if the neighbor is not in the grid (collision)
                    if neighbor_index not in self.grid:
                        continue
                        
                    neighbors.append(self.grid[neighbor_index])
                    
        return neighbors
    
    def _extract_path(self, goal_node: DijkstraNode) -> List[np.ndarray]:
        """
        Extract the path from the start to the goal node.
        
        Parameters
        ----------
        goal_node : DijkstraNode
            Goal node
            
        Returns
        -------
        List[np.ndarray]
            List of waypoints from start to goal
        """
        path = []
        current_node = goal_node
        
        # Traverse the path from the goal to the start
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
            
        # Reverse the path to get it from start to goal
        return path[::-1]