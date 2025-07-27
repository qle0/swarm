"""
Policy wrapper for path planners.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym

from swarm.planners.base_planner import BasePlanner
from swarm.planners.rrt import RRTPlanner
from swarm.planners.rrt_optimized import OptimizedRRTPlanner
from swarm.planners.dijkstra import DijkstraPlanner
from swarm.planners.astar import AStarPlanner
from swarm.planners.path_smoother import PathSmoother


class PlannerPolicy:
    """
    Policy wrapper for path planners.
    
    This class implements the same interface as Stable Baselines 3 policies,
    allowing it to be used as a drop-in replacement for RL policies.
    """
    
    def __init__(self, 
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 planner_type: str = "rrt",
                 client_id: Optional[int] = None,
                 obstacle_ids: Optional[List[int]] = None,
                 use_path_smoothing: bool = True,
                 smoothing_method: str = "spline",
                 **kwargs):
        """
        Initialize the planner policy.
        
        Parameters
        ----------
        observation_space : gym.spaces.Space
            Observation space
        action_space : gym.spaces.Space
            Action space
        planner_type : str
            Type of planner to use ("rrt", "rrt_optimized", "dijkstra", "astar")
        client_id : Optional[int]
            PyBullet client ID
        obstacle_ids : Optional[List[int]]
            List of obstacle IDs in the PyBullet simulation
        use_path_smoothing : bool
            Whether to apply path smoothing
        smoothing_method : str
            Path smoothing method ("spline", "shortcut", "gradient", "bezier")
        **kwargs : Dict[str, Any]
            Additional arguments for the planner
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.planner_type = planner_type
        self.client_id = client_id
        self.obstacle_ids = obstacle_ids or []
        self.kwargs = kwargs
        self.use_path_smoothing = use_path_smoothing
        self.smoothing_method = smoothing_method
        
        # Path planning variables
        self.start = None
        self.goal = None
        self.planner = None
        self.path = []
        self.raw_path = []  # Store original path before smoothing
        self.current_waypoint_idx = 0
        
        # Control parameters
        self.max_speed = 1.0  # m/s
        self.waypoint_threshold = 0.2  # m
        
        # Path smoother
        if self.use_path_smoothing:
            collision_checker = None
            if self.client_id is not None and self.obstacle_ids:
                # Create a collision checker function
                def check_collision(pos):
                    from swarm.planners.base_planner import BasePlanner
                    temp_planner = BasePlanner(self.client_id, self.obstacle_ids)
                    return temp_planner.check_collision(pos)
                collision_checker = check_collision
            
            self.path_smoother = PathSmoother(collision_checker)
        else:
            self.path_smoother = None
        
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        """
        Predict the action based on the observation.
        
        Parameters
        ----------
        observation : np.ndarray
            Current observation
        deterministic : bool
            Whether to use deterministic actions
            
        Returns
        -------
        Tuple[np.ndarray, None]
            Action and dummy state
        """
        # Make sure observation is a numpy array
        observation = np.array(observation).flatten()
        
        # Check if observation has the expected format
        if observation.size < 9:
            # If observation is invalid, return zero action
            return np.zeros(3), None
        
        # Extract the current position and goal from the observation
        # Assuming observation format: [x, y, z, vx, vy, vz, goal_x, goal_y, goal_z, ...]
        current_position = observation[:3]
        goal_position = observation[6:9]
        
        # Initialize the planner if needed
        if self.planner is None or self.goal is None or not np.array_equal(self.goal, goal_position):
            self.start = current_position
            self.goal = goal_position
            try:
                self._initialize_planner()
            except Exception as e:
                print(f"Error initializing planner: {e}")
                # If planner initialization fails, use direct control
                return self._direct_control(current_position, goal_position), None
            
        # If we have a path, follow it
        if len(self.path) > 0:
            action = self._follow_path(current_position)
        else:
            # If no path is found, move directly towards the goal
            action = self._direct_control(current_position, goal_position)
            
        return action, None
    
    def _initialize_planner(self):
        """Initialize the path planner and compute a path."""
        # Create the appropriate planner
        if self.planner_type == "rrt":
            self.planner = RRTPlanner(
                start=self.start,
                goal=self.goal,
                client_id=self.client_id,
                obstacle_ids=self.obstacle_ids,
                **self.kwargs
            )
        elif self.planner_type == "rrt_optimized":
            self.planner = OptimizedRRTPlanner(
                start=self.start,
                goal=self.goal,
                client_id=self.client_id,
                obstacle_ids=self.obstacle_ids,
                **self.kwargs
            )
        elif self.planner_type == "dijkstra":
            self.planner = DijkstraPlanner(
                start=self.start,
                goal=self.goal,
                client_id=self.client_id,
                obstacle_ids=self.obstacle_ids,
                **self.kwargs
            )
        elif self.planner_type == "astar":
            self.planner = AStarPlanner(
                start=self.start,
                goal=self.goal,
                client_id=self.client_id,
                obstacle_ids=self.obstacle_ids,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown planner type: {self.planner_type}")
            
        # Plan a path
        self.raw_path = self.planner.plan()
        
        # Apply path smoothing if enabled
        if self.use_path_smoothing and self.path_smoother and len(self.raw_path) > 2:
            try:
                self.path = self.path_smoother.smooth_path(
                    self.raw_path, 
                    method=self.smoothing_method
                )
                print(f"Path smoothed: {len(self.raw_path)} -> {len(self.path)} waypoints")
            except Exception as e:
                print(f"Path smoothing failed: {e}, using raw path")
                self.path = self.raw_path
        else:
            self.path = self.raw_path
            
        self.current_waypoint_idx = 0
        
        # Visualize the path if a client ID is provided
        if self.client_id is not None and len(self.path) > 0:
            # Visualize raw path in red
            if hasattr(self.planner, 'visualize_path') and len(self.raw_path) > 0:
                self.planner.visualize_path(self.raw_path, color=(1, 0, 0), line_width=1.0)
            
            # Visualize smoothed path in green
            if len(self.path) != len(self.raw_path):
                self.planner.visualize_path(self.path, color=(0, 1, 0), line_width=3.0)
    
    def _follow_path(self, current_position: np.ndarray) -> np.ndarray:
        """
        Follow the planned path.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current position
            
        Returns
        -------
        np.ndarray
            Action (velocity command)
        """
        # Check if we reached the current waypoint
        if self.current_waypoint_idx < len(self.path):
            waypoint = self.path[self.current_waypoint_idx]
            distance = np.linalg.norm(current_position - waypoint)
            
            if distance < self.waypoint_threshold:
                # Move to the next waypoint
                self.current_waypoint_idx += 1
                
                # If we reached the end of the path, target the goal directly
                if self.current_waypoint_idx >= len(self.path):
                    return self._direct_control(current_position, self.goal)
                    
                waypoint = self.path[self.current_waypoint_idx]
                
            # Compute the velocity command to reach the waypoint
            direction = waypoint - current_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                velocity = direction / distance * self.max_speed
            else:
                velocity = np.zeros(3)
                
            # Convert to RPM for the drone
            # This is a simple mapping from velocity to RPM
            # In a real system, this would be more complex
            rpm = np.array([
                velocity[0] * 1000,  # x velocity to RPM
                velocity[1] * 1000,  # y velocity to RPM
                velocity[2] * 1000,  # z velocity to RPM
                0.0  # yaw rate (not used)
            ])
            
            return rpm
        else:
            # If we reached the end of the path, target the goal directly
            return self._direct_control(current_position, self.goal)
    
    def _direct_control(self, current_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
        """
        Direct control towards a target position.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current position
        target_position : np.ndarray
            Target position
            
        Returns
        -------
        np.ndarray
            Action (RPM command)
        """
        direction = target_position - current_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            velocity = direction / distance * self.max_speed
        else:
            velocity = np.zeros(3)
        
        # Convert to RPM for the drone
        # This is a simple mapping from velocity to RPM
        rpm = np.array([
            velocity[0] * 1000,  # x velocity to RPM
            velocity[1] * 1000,  # y velocity to RPM
            velocity[2] * 1000,  # z velocity to RPM
            0.0  # yaw rate (not used)
        ])
        
        return rpm