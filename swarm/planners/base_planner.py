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
                 client_id: Optional[int] = None,
                 obstacle_ids: Optional[List[int]] = None):
        """
        Initialize the path planner.
        
        Parameters
        ----------
        client_id : Optional[int]
            PyBullet client ID for collision checking
        obstacle_ids : Optional[List[int]]
            List of obstacle IDs for collision checking
        """
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
        # Simple collision check
        # If there are no obstacles, return False
        if not self.obstacle_ids:
            return False
            
        # Check if the position is too close to the ground
        if position[2] < 0.1:
            return True
            
        # Check collision with obstacles
        for obstacle_id in self.obstacle_ids:
            try:
                # Get the AABB of the obstacle
                aabb_min, aabb_max = p.getAABB(obstacle_id, physicsClientId=self.client_id)
                
                # Expand the AABB by the radius
                aabb_min = np.array(aabb_min) - radius
                aabb_max = np.array(aabb_max) + radius
                
                # Check if the position is inside the expanded AABB
                if (position[0] >= aabb_min[0] and position[0] <= aabb_max[0] and
                    position[1] >= aabb_min[1] and position[1] <= aabb_max[1] and
                    position[2] >= aabb_min[2] and position[2] <= aabb_max[2]):
                    return True
            except:
                # If there's an error, skip this obstacle
                continue
                
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