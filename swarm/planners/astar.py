"""
A* path planning algorithm for 3D drone navigation.
"""
from typing import List, Tuple, Optional, Set
import numpy as np
import heapq
from dataclasses import dataclass, field

from swarm.planners.base_planner import BasePlanner


@dataclass
class AStarNode:
    """Node for A* algorithm."""
    position: Tuple[int, int, int]
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = 0.0           # Heuristic cost to goal
    f_cost: float = float('inf')  # Total cost (g + h)
    parent: Optional['AStarNode'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class AStarPlanner(BasePlanner):
    """
    A* path planner for 3D drone navigation.
    
    A* is more efficient than Dijkstra as it uses a heuristic to guide
    the search towards the goal, reducing the number of nodes explored.
    """
    
    def __init__(self, 
                 start: np.ndarray, 
                 goal: np.ndarray,
                 client_id: Optional[int] = None,
                 obstacle_ids: Optional[List[int]] = None,
                 grid_resolution: float = 0.5,
                 heuristic_weight: float = 1.0,
                 allow_diagonal: bool = True):
        """
        Initialize the A* planner.
        
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
        grid_resolution : float
            Resolution of the 3D grid
        heuristic_weight : float
            Weight for the heuristic function (1.0 = optimal, >1.0 = faster but suboptimal)
        allow_diagonal : bool
            Whether to allow diagonal movements
        """
        super().__init__(client_id, obstacle_ids)
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.grid_resolution = grid_resolution
        self.heuristic_weight = heuristic_weight
        self.allow_diagonal = allow_diagonal
        
        # Convert positions to grid coordinates
        self.start_grid = self._world_to_grid(self.start)
        self.goal_grid = self._world_to_grid(self.goal)
        
        # Define movement directions
        if allow_diagonal:
            # 26-connectivity (all adjacent cells in 3D)
            self.directions = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        self.directions.append((dx, dy, dz))
        else:
            # 6-connectivity (face-adjacent cells only)
            self.directions = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1)
            ]
        
        # Statistics
        self.nodes_explored = 0
        self.nodes_in_open = 0
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal using A*.
        
        Returns
        -------
        List[np.ndarray]
            List of waypoints (3D positions) from start to goal
        """
        # Check if start and goal are valid
        if self.check_collision(self.start):
            print("A*: Start position is in collision")
            return []
        
        if self.check_collision(self.goal):
            print("A*: Goal position is in collision")
            return []
        
        # Initialize open and closed sets
        open_set = []
        closed_set: Set[Tuple[int, int, int]] = set()
        
        # Create start node
        start_node = AStarNode(
            position=self.start_grid,
            g_cost=0.0,
            h_cost=self._heuristic(self.start_grid, self.goal_grid),
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        heapq.heappush(open_set, start_node)
        nodes_dict = {self.start_grid: start_node}
        
        while open_set:
            # Get the node with lowest f_cost
            current_node = heapq.heappop(open_set)
            self.nodes_explored += 1
            
            # Check if we reached the goal
            if current_node.position == self.goal_grid:
                path = self._reconstruct_path(current_node)
                print(f"A* found path with {len(path)} waypoints, explored {self.nodes_explored} nodes")
                return path
            
            # Add current node to closed set
            closed_set.add(current_node.position)
            
            # Explore neighbors
            for direction in self.directions:
                neighbor_pos = (
                    current_node.position[0] + direction[0],
                    current_node.position[1] + direction[1],
                    current_node.position[2] + direction[2]
                )
                
                # Skip if already in closed set
                if neighbor_pos in closed_set:
                    continue
                
                # Check if neighbor is collision-free
                neighbor_world = self._grid_to_world(neighbor_pos)
                if self.check_collision(neighbor_world):
                    continue
                
                # Calculate movement cost
                movement_cost = self._movement_cost(direction)
                tentative_g_cost = current_node.g_cost + movement_cost
                
                # Get or create neighbor node
                if neighbor_pos in nodes_dict:
                    neighbor_node = nodes_dict[neighbor_pos]
                else:
                    neighbor_node = AStarNode(
                        position=neighbor_pos,
                        h_cost=self._heuristic(neighbor_pos, self.goal_grid)
                    )
                    nodes_dict[neighbor_pos] = neighbor_node
                
                # Update neighbor if we found a better path
                if tentative_g_cost < neighbor_node.g_cost:
                    neighbor_node.parent = current_node
                    neighbor_node.g_cost = tentative_g_cost
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                    
                    # Add to open set if not already there
                    if neighbor_node not in open_set:
                        heapq.heappush(open_set, neighbor_node)
                        self.nodes_in_open += 1
        
        print(f"A*: No path found after exploring {self.nodes_explored} nodes")
        return []
    
    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert world coordinates to grid coordinates.
        
        Parameters
        ----------
        world_pos : np.ndarray
            World position
            
        Returns
        -------
        Tuple[int, int, int]
            Grid coordinates
        """
        grid_pos = np.round(world_pos / self.grid_resolution).astype(int)
        return tuple(grid_pos)
    
    def _grid_to_world(self, grid_pos: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert grid coordinates to world coordinates.
        
        Parameters
        ----------
        grid_pos : Tuple[int, int, int]
            Grid coordinates
            
        Returns
        -------
        np.ndarray
            World position
        """
        return np.array(grid_pos) * self.grid_resolution
    
    def _heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """
        Calculate heuristic distance between two grid positions.
        
        Parameters
        ----------
        pos1 : Tuple[int, int, int]
            First position
        pos2 : Tuple[int, int, int]
            Second position
            
        Returns
        -------
        float
            Heuristic distance
        """
        # Use Euclidean distance as heuristic
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        
        euclidean_dist = np.sqrt(dx*dx + dy*dy + dz*dz) * self.grid_resolution
        return self.heuristic_weight * euclidean_dist
    
    def _movement_cost(self, direction: Tuple[int, int, int]) -> float:
        """
        Calculate the cost of moving in a given direction.
        
        Parameters
        ----------
        direction : Tuple[int, int, int]
            Movement direction
            
        Returns
        -------
        float
            Movement cost
        """
        # Calculate Euclidean distance for the movement
        dx, dy, dz = direction
        distance = np.sqrt(dx*dx + dy*dy + dz*dz) * self.grid_resolution
        
        # Add penalty for vertical movement (drones prefer horizontal movement)
        if dz != 0:
            distance *= 1.1
        
        return distance
    
    def _reconstruct_path(self, goal_node: AStarNode) -> List[np.ndarray]:
        """
        Reconstruct the path from start to goal.
        
        Parameters
        ----------
        goal_node : AStarNode
            Goal node
            
        Returns
        -------
        List[np.ndarray]
            Path from start to goal
        """
        path = []
        current_node = goal_node
        
        while current_node is not None:
            world_pos = self._grid_to_world(current_node.position)
            path.append(world_pos)
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
            'nodes_explored': self.nodes_explored,
            'nodes_in_open': self.nodes_in_open,
            'grid_resolution': self.grid_resolution,
            'heuristic_weight': self.heuristic_weight,
            'allow_diagonal': self.allow_diagonal,
        }