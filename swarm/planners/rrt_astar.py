"""
Hybrid RRT+A* path planner implementation.
"""
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import heapq

from swarm.planners.base_planner import BasePlanner
from swarm.planners.rrt import RRTPlanner, RRTNode


class RRTAStarPlanner(BasePlanner):
    """
    Hybrid RRT+A* path planner.
    
    This planner uses RRT to explore the space and build a graph,
    then applies A* algorithm to find the shortest path in that graph.
    """
    
    def __init__(self, 
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 client_id: int,
                 obstacle_ids: Optional[List[int]] = None,
                 max_iterations: int = 1000,
                 step_size: float = 0.2,
                 goal_sample_rate: float = 0.2,  # Increased from 0.1 to 0.2
                 search_radius: float = 1.5,     # Increased from 1.0 to 1.5
                 smoothing_iterations: int = 10):
        """
        Initialize the RRT+A* planner.
        
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
            Maximum number of iterations for RRT
        step_size : float
            Maximum distance between nodes for RRT
        goal_sample_rate : float
            Probability of sampling the goal position for RRT
        search_radius : float
            Maximum distance to search for the goal for RRT
        smoothing_iterations : int
            Number of iterations for path smoothing
        """
        super().__init__(start, goal, client_id, obstacle_ids)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.smoothing_iterations = smoothing_iterations
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal using RRT+A*.
        
        Returns
        -------
        List[np.ndarray]
            List of waypoints (3D positions) from start to goal
        """
        # First, use RRT to build a graph
        rrt_planner = RRTPlanner(
            start=self.start,
            goal=self.goal,
            client_id=self.client_id,
            obstacle_ids=self.obstacle_ids,
            max_iterations=self.max_iterations,
            step_size=self.step_size,
            goal_sample_rate=self.goal_sample_rate,
            search_radius=self.search_radius
        )
        
        # Run RRT to build the graph
        rrt_path = rrt_planner.plan()
        
        # If RRT failed or found a direct path, return it
        if len(rrt_path) <= 2:
            return rrt_path
            
        # Get the RRT nodes and build a graph
        rrt_nodes = rrt_planner.nodes
        
        # If we don't have enough nodes, return the RRT path
        if len(rrt_nodes) < 3:
            return rrt_path
            
        # Build an adjacency list for A*
        adjacency_list = self._build_adjacency_list(rrt_nodes)
        
        # Find the start and goal indices
        start_idx = 0  # First node is always the start
        goal_idx = None
        
        # Find the goal node index
        for i, node in enumerate(rrt_nodes):
            if np.array_equal(node.position, self.goal):
                goal_idx = i
                break
                
        # If we couldn't find the goal node, use the last node as the goal
        if goal_idx is None:
            goal_idx = len(rrt_nodes) - 1
            
        # Run A* to find the shortest path
        shortest_path = self._astar_shortest_path(
            rrt_nodes, adjacency_list, start_idx, goal_idx)
            
        # If A* failed, return the RRT path
        if not shortest_path:
            return rrt_path
            
        # Apply path smoothing
        smoothed_path = self._smooth_path(shortest_path)
            
        # Return the smoothed path
        return smoothed_path
        
    def _build_adjacency_list(self, nodes: List[RRTNode]) -> Dict[int, List[Tuple[int, float]]]:
        """
        Build an adjacency list from RRT nodes.
        
        Parameters
        ----------
        nodes : List[RRTNode]
            List of RRT nodes
            
        Returns
        -------
        Dict[int, List[Tuple[int, float]]]
            Adjacency list representation of the graph
        """
        adjacency_list = {}
        
        # Initialize the adjacency list
        for i in range(len(nodes)):
            adjacency_list[i] = []
            
        # Add edges between nodes and their parents
        for i, node in enumerate(nodes):
            if node.parent is not None:
                # Find the parent index
                for j, potential_parent in enumerate(nodes):
                    if node.parent is potential_parent:
                        # Add an edge from the node to its parent
                        distance = np.linalg.norm(node.position - potential_parent.position)
                        adjacency_list[i].append((j, distance))
                        
                        # Add the reverse edge (undirected graph)
                        adjacency_list[j].append((i, distance))
                        break
                        
        return adjacency_list
        
    def _astar_shortest_path(self, 
                            nodes: List[RRTNode], 
                            adjacency_list: Dict[int, List[Tuple[int, float]]],
                            start_idx: int, 
                            goal_idx: int) -> List[np.ndarray]:
        """
        Find the shortest path from start to goal using A* algorithm.
        
        Parameters
        ----------
        nodes : List[RRTNode]
            List of RRT nodes
        adjacency_list : Dict[int, List[Tuple[int, float]]]
            Adjacency list representation of the graph
        start_idx : int
            Index of the start node
        goal_idx : int
            Index of the goal node
            
        Returns
        -------
        List[np.ndarray]
            List of waypoints (node positions) from start to goal
        """
        # Initialize g_scores (cost from start) with infinity for all nodes except the start node
        g_scores = {i: float('inf') for i in range(len(nodes))}
        g_scores[start_idx] = 0
        
        # Initialize f_scores (estimated total cost) with infinity for all nodes
        f_scores = {i: float('inf') for i in range(len(nodes))}
        
        # Calculate the heuristic (Euclidean distance to goal) for the start node
        h_start = np.linalg.norm(nodes[start_idx].position - nodes[goal_idx].position)
        f_scores[start_idx] = h_start
        
        # Initialize previous node dictionary for path reconstruction
        previous = {i: None for i in range(len(nodes))}
        
        # Priority queue for A* algorithm
        # Format: (f_score, node_index)
        open_set = [(f_scores[start_idx], start_idx)]
        
        # Set of visited nodes
        closed_set: Set[int] = set()
        
        while open_set:
            # Get the node with the smallest f_score
            _, current_idx = heapq.heappop(open_set)
            
            # If we've reached the goal, we can stop
            if current_idx == goal_idx:
                break
                
            # Skip if we've already processed this node
            if current_idx in closed_set:
                continue
                
            # Mark the node as visited
            closed_set.add(current_idx)
            
            # Check all neighbors of the current node
            for neighbor_idx, edge_distance in adjacency_list.get(current_idx, []):
                # Skip if we've already visited this neighbor
                if neighbor_idx in closed_set:
                    continue
                    
                # Calculate the tentative g_score
                tentative_g_score = g_scores[current_idx] + edge_distance
                
                # If we found a better path to the neighbor, update it
                if tentative_g_score < g_scores[neighbor_idx]:
                    # Update the path
                    previous[neighbor_idx] = current_idx
                    g_scores[neighbor_idx] = tentative_g_score
                    
                    # Calculate the heuristic (Euclidean distance to goal)
                    h_neighbor = np.linalg.norm(nodes[neighbor_idx].position - nodes[goal_idx].position)
                    
                    # Update the f_score
                    f_scores[neighbor_idx] = g_scores[neighbor_idx] + h_neighbor
                    
                    # Add to the open set
                    heapq.heappush(open_set, (f_scores[neighbor_idx], neighbor_idx))
        
        # Reconstruct the path from goal to start
        path = []
        current_idx = goal_idx
        
        # If there's no path to the goal, return an empty list
        if previous[goal_idx] is None and goal_idx != start_idx:
            return []
            
        # Traverse from goal to start
        while current_idx is not None:
            path.append(nodes[current_idx].position)
            current_idx = previous[current_idx]
            
        # Reverse to get path from start to goal
        return path[::-1]
    
    def _smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """
        Smooth the path using a simple path smoothing algorithm.
        
        Parameters
        ----------
        path : List[np.ndarray]
            List of waypoints
            
        Returns
        -------
        List[np.ndarray]
            Smoothed path
        """
        if len(path) <= 2:
            return path
            
        # Make a copy of the path
        smoothed_path = path.copy()
        
        # Perform multiple iterations of smoothing
        for _ in range(self.smoothing_iterations):
            # Start from the second point and end at the second-to-last point
            i = 1
            while i < len(smoothed_path) - 1:
                # Get the current point and its neighbors
                prev_point = smoothed_path[i - 1]
                current_point = smoothed_path[i]
                next_point = smoothed_path[i + 1]
                
                # Calculate the new position (average of neighbors)
                new_position = (prev_point + next_point) / 2
                
                # Check if the new position is collision-free
                if not self.check_collision(new_position) and not self._is_collision_free(prev_point, new_position) and not self._is_collision_free(new_position, next_point):
                    # Update the point
                    smoothed_path[i] = new_position
                
                i += 1
                
        # Remove redundant waypoints
        i = 1
        while i < len(smoothed_path) - 1:
            # Check if we can skip this waypoint
            if not self._is_collision_free(smoothed_path[i - 1], smoothed_path[i + 1]):
                # Remove the waypoint
                smoothed_path.pop(i)
            else:
                i += 1
                
        return smoothed_path
    
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