"""
Base class for path planners.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
import pybullet as p


class BasePlanner(ABC):
    """Base class for all path planners."""
    
    def __init__(self, 
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 client_id: int,
                 obstacle_ids: Optional[List[int]] = None):
        """
        Initialize the path planner.
        
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
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.client_id = client_id
        self.obstacle_ids = obstacle_ids or []
        
    @abstractmethod
    def plan(self) -> List[np.ndarray]:
        """
        Plan a path from start to goal.
        
        Returns
        -------
        List[np.ndarray]
            List of waypoints (3D positions) from start to goal
        """
        pass
    
    def check_collision(self, position: np.ndarray, radius: float = 0.1) -> bool:
        """
        Check if a position is in collision with any obstacle.
        
        Parameters
        ----------
        position : np.ndarray
            Position to check (x, y, z)
        radius : float
            Collision radius
            
        Returns
        -------
        bool
            True if in collision, False otherwise
        """
        for obstacle_id in self.obstacle_ids:
            closest_points = p.getClosestPoints(
                bodyA=-1,  # Use a ray
                bodyB=obstacle_id,
                distance=radius,
                linkIndexA=-1,
                linkIndexB=-1,
                pointA=position,
                pointB=[0, 0, 0],  # Not used
                physicsClientId=self.client_id
            )
            
            if len(closest_points) > 0:
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
            RGB color (0-1 range)
        line_width : float
            Line width
        """
        if len(path) < 2:
            return
            
        for i in range(len(path) - 1):
            p.addUserDebugLine(
                lineFromXYZ=path[i],
                lineToXYZ=path[i + 1],
                lineColorRGB=color,
                lineWidth=line_width,
                physicsClientId=self.client_id
            )