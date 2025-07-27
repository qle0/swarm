"""
Hybrid RRT+Dijkstra path planner implementation.
"""
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import heapq

from swarm.planners.base_planner import BasePlanner
from swarm.planners.rrt import RRTPlanner, RRTNode


class RRTDijkstraPlanner(BasePlanner):
    """
    Hybrid RRT+Dijkstra path planner.
    
    This planner uses RRT to explore the space and build a graph,
    then applies Dijkstra's algorithm to find the shortest path in that graph.
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
        Initialize the RRT+Dijkstra planner.
        
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
        """
        super().__init__(start, goal, client_id, obstacle_ids)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal using RRT+Dijkstra.
        
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
            
        # Build an adjacency list for Dijkstra
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
            
        # Run Dijkstra to find the shortest path
        shortest_path = self._dijkstra_shortest_path(
            rrt_nodes, adjacency_list, start_idx, goal_idx)
            
        # If Dijkstra failed, return the RRT path
        if not shortest_path:
            return rrt_path
            
        # Return the shortest path
        return shortest_path
        
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
        
    def _dijkstra_shortest_path(self, 
                               nodes: List[RRTNode], 
                               adjacency_list: Dict[int, List[Tuple[int, float]]],
                               start_idx: int, 
                               goal_idx: int) -> List[np.ndarray]:
        """
        Find the shortest path from start to goal using Dijkstra's algorithm.
        
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
        # Initialize distances with infinity for all nodes except the start node
        distances = {i: float('inf') for i in range(len(nodes))}
        distances[start_idx] = 0
        
        # Initialize previous node dictionary for path reconstruction
        previous = {i: None for i in range(len(nodes))}
        
        # Priority queue for Dijkstra's algorithm
        # Format: (distance, node_index)
        priority_queue = [(0, start_idx)]
        
        # Set of visited nodes
        visited: Set[int] = set()
        
        while priority_queue:
            # Get the node with the smallest distance
            current_distance, current_idx = heapq.heappop(priority_queue)
            
            # If we've reached the goal, we can stop
            if current_idx == goal_idx:
                break
                
            # Skip if we've already processed this node
            if current_idx in visited:
                continue
                
            # Mark the node as visited
            visited.add(current_idx)
            
            # Check all neighbors of the current node
            for neighbor_idx, edge_distance in adjacency_list.get(current_idx, []):
                # Skip if we've already visited this neighbor
                if neighbor_idx in visited:
                    continue
                    
                # Calculate the distance to the neighbor through the current node
                distance = current_distance + edge_distance
                
                # If we found a shorter path to the neighbor, update it
                if distance < distances[neighbor_idx]:
                    distances[neighbor_idx] = distance
                    previous[neighbor_idx] = current_idx
                    heapq.heappush(priority_queue, (distance, neighbor_idx))
        
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