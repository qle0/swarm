"""
Hybrid policy combining advanced path planning with RL local control.

This policy uses a parallel planner that runs multiple path planning algorithms
(RRT, RRT+Dijkstra, RRT+A*, RRT*) and selects the best result based on path quality,
planning time, and path length.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import time

from swarm.planners.rrt import RRTPlanner
from swarm.planners.rrt_dijkstra import RRTDijkstraPlanner
from swarm.planners.rrt_astar import RRTAStarPlanner
from swarm.planners.rrt_star import RRTStarPlanner
from swarm.planners.parallel_planner import ParallelPlanner


class HybridPolicy:
    """
    Hybrid policy that combines global path planning with local RL control.
    
    This policy uses a parallel planner for global path planning to generate optimal waypoints,
    and then uses an RL policy for local control to follow the path while
    avoiding dynamic obstacles. The parallel planner runs multiple algorithms
    (RRT, RRT+Dijkstra, RRT+A*, RRT*) in parallel and selects the best result
    based on path quality, planning time, and path length.
    
    The policy also adapts to the environment complexity by adjusting the planning
    parameters based on the collision rate and planning success.
    """
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 rl_policy=None,
                 client_id: Optional[int] = None,
                 obstacle_ids: Optional[List[int]] = None,
                 planning_horizon: float = 5.0,
                 replan_threshold: float = 1.0,
                 **rrt_kwargs):
        """
        Initialize the hybrid policy.
        
        Parameters
        ----------
        observation_space
            Observation space
        action_space
            Action space
        rl_policy
            Trained RL policy for local control
        client_id : Optional[int]
            PyBullet client ID
        obstacle_ids : Optional[List[int]]
            List of obstacle IDs in the PyBullet simulation
        planning_horizon : float
            Time horizon for planning in seconds
        replan_threshold : float
            Distance threshold for replanning
        **rrt_kwargs
            Additional arguments for RRT planner
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.rl_policy = rl_policy
        self.client_id = client_id
        self.obstacle_ids = obstacle_ids or []
        self.planning_horizon = planning_horizon
        self.replan_threshold = replan_threshold
        
        # Initialize planning history for adaptive parameter tuning
        self.planning_history = []
        self.rrt_kwargs = rrt_kwargs
        
        # Path planning variables
        self.global_path = []
        self.current_waypoint_idx = 0
        self.last_plan_time = 0
        self.last_plan_position = None
        self.goal_position = None
        
        # Control parameters
        self.max_speed = 5.0  # m/s - увеличиваем скорость для быстрого достижения цели
        self.waypoint_threshold = 0.25  # m - увеличиваем порог для более быстрого прохождения точек
        self.use_rl_threshold = 1.0  # Distance threshold to switch to RL control
        
        # Statistics
        self.planning_count = 0
        self.rl_control_count = 0
        self.waypoint_control_count = 0
        
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        """
        Predict the action using hybrid approach.
        
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
            print(f"WARNING: Invalid observation size: {observation.size}")
            return np.zeros(4), None
        
        # Extract the current position and goal from the observation
        current_position = observation[:3]
        velocity = observation[3:6]
        goal_position = observation[6:9]
        
        # Debug information
        distance_to_goal = np.linalg.norm(current_position - goal_position)
        if self.waypoint_control_count % 100 == 0 or distance_to_goal < 1.0:
            print(f"DEBUG: Step {self.waypoint_control_count}")
            print(f"  Position: {current_position}, Velocity: {velocity}")
            print(f"  Goal: {goal_position}, Distance: {distance_to_goal:.2f}")
            
            if len(self.global_path) > 0:
                current_idx = min(self.current_waypoint_idx, len(self.global_path)-1)
                current_target = self.global_path[current_idx]
                print(f"  Current waypoint: {current_target}, idx={current_idx}/{len(self.global_path)}")
                print(f"  Distance to waypoint: {np.linalg.norm(current_position - current_target):.2f}")
            else:
                print("  No path available")
        
        # Check if we need to replan
        current_time = time.time()
        need_replan = self._should_replan(current_position, goal_position, current_time)
        
        if need_replan:
            print(f"PLANNING: from {current_position} to {goal_position}")
            self._plan_global_path(current_position, goal_position)
            
        # Decide between waypoint following and RL control
        action = self._get_hybrid_action(observation, current_position, goal_position)
        
        # Debug action
        if self.waypoint_control_count % 100 == 0 or distance_to_goal < 1.0:
            print(f"  Action: {action}")
        
        return action, None
    
    def _should_replan(self, current_position: np.ndarray, goal_position: np.ndarray, current_time: float) -> bool:
        """
        Determine if we should replan the global path.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current position
        goal_position : np.ndarray
            Goal position
        current_time : float
            Current time
            
        Returns
        -------
        bool
            True if replanning is needed
        """
        # Always plan if we don't have a path
        if len(self.global_path) == 0:
            return True
            
        # Replan if goal has changed
        if self.goal_position is None or not np.allclose(self.goal_position, goal_position, atol=0.1):
            return True
            
        # Replan if we've moved too far from the last planning position
        if (self.last_plan_position is not None and 
            np.linalg.norm(current_position - self.last_plan_position) > self.replan_threshold):
            return True
            
        # Replan if too much time has passed
        if current_time - self.last_plan_time > self.planning_horizon:
            return True
            
        # Replan if we've reached the end of the current path
        if self.current_waypoint_idx >= len(self.global_path):
            return True
            
        return False
    
    def _plan_global_path(self, start_position: np.ndarray, goal_position: np.ndarray):
        """
        Plan a global path using RRT+A* with path optimization.
        
        Parameters
        ----------
        start_position : np.ndarray
            Start position
        goal_position : np.ndarray
            Goal position
        """
        try:
            # First try direct path if possible
            direct_path_possible = True
            
            # Check if there are obstacles between start and goal
            if self.client_id is not None and self.obstacle_ids:
                import pybullet as p
                
                # Check for collisions along direct path
                direction = goal_position - start_position
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    direction = direction / distance
                    num_steps = max(10, int(distance / 0.2))
                    
                    for i in range(1, num_steps):
                        point = start_position + direction * (i * distance / num_steps)
                        
                        # Check for collision at this point
                        for obstacle_id in self.obstacle_ids:
                            try:
                                aabb_min, aabb_max = p.getAABB(obstacle_id, physicsClientId=self.client_id)
                                
                                # Expand AABB by safety margin
                                aabb_min = np.array(aabb_min) - 0.3
                                aabb_max = np.array(aabb_max) + 0.3
                                
                                # Check if point is inside AABB
                                if (point[0] >= aabb_min[0] and point[0] <= aabb_max[0] and
                                    point[1] >= aabb_min[1] and point[1] <= aabb_max[1] and
                                    point[2] >= aabb_min[2] and point[2] <= aabb_max[2]):
                                    direct_path_possible = False
                                    break
                            except:
                                continue
                        
                        if not direct_path_possible:
                            break
            
            # If direct path is possible, use it
            if direct_path_possible:
                # Оптимизируем путь для более прямого полета
                # Вычисляем вектор от старта к цели
                direction = goal_position - start_position
                distance = np.linalg.norm(direction)
                
                # Создаем более прямой путь с минимальным количеством точек
                if distance > 3.0:
                    # Для длинных дистанций добавляем одну промежуточную точку
                    midpoint = start_position + direction * 0.5
                    # Немного поднимаем промежуточную точку для избежания препятствий
                    midpoint[2] = max(midpoint[2], start_position[2] + 0.3, goal_position[2] + 0.3)
                    self.global_path = [start_position, midpoint, goal_position]
                else:
                    # Для коротких дистанций летим напрямую
                    self.global_path = [start_position, goal_position]
                self.current_waypoint_idx = 0
                self.last_plan_time = time.time()
                self.last_plan_position = start_position.copy()
                self.goal_position = goal_position.copy()
                self.planning_count += 1
                
                # Visualize the direct path if possible
                if self.client_id is not None:
                    import pybullet as p
                    for i in range(len(self.global_path) - 1):
                        p.addUserDebugLine(
                            self.global_path[i], 
                            self.global_path[i+1],
                            lineColorRGB=(0, 1, 0),  # Green for direct path
                            lineWidth=3.0,
                            lifeTime=0,
                            physicsClientId=self.client_id
                        )
                
                return
            
            # If direct path is not possible, use parallel planner
            # Create parallel planner that runs multiple algorithms
            print("Using parallel planner for path planning")
            
            # Adapt parameters based on environment complexity
            max_iterations = self.rrt_kwargs.get('max_iterations', 1000)
            step_size = self.rrt_kwargs.get('step_size', 0.2)
            goal_sample_rate = self.rrt_kwargs.get('goal_sample_rate', 0.2)
            search_radius = self.rrt_kwargs.get('search_radius', 1.5)
            
            # If we have planning history, adapt parameters
            if hasattr(self, 'planning_history') and len(self.planning_history) > 0:
                # Calculate success rate
                success_rate = sum(1 for h in self.planning_history if h['success']) / len(self.planning_history)
                
                # Calculate average planning time
                avg_planning_time = sum(h['planning_time'] for h in self.planning_history) / len(self.planning_history)
                
                # Adapt parameters based on success rate and planning time
                if success_rate < 0.7:  # Low success rate
                    max_iterations = int(max_iterations * 1.2)  # Increase iterations
                    step_size = max(0.1, step_size * 0.9)  # Decrease step size
                    goal_sample_rate = min(0.3, goal_sample_rate * 1.1)  # Increase goal bias
                elif avg_planning_time > 0.5:  # Planning takes too long
                    max_iterations = int(max_iterations * 0.9)  # Decrease iterations
                    step_size = min(0.3, step_size * 1.1)  # Increase step size
                    search_radius = min(2.0, search_radius * 1.1)  # Increase search radius
            
            # Create the parallel planner
            planner = ParallelPlanner(
                start=start_position,
                goal=goal_position,
                client_id=self.client_id,
                obstacle_ids=self.obstacle_ids,
                max_iterations=max_iterations,
                step_size=step_size,
                goal_sample_rate=goal_sample_rate,
                search_radius=search_radius,
                timeout=2.0,  # 2 second timeout for planning
                planners=['rrt', 'rrt_dijkstra', 'rrt_astar', 'rrt_star']
            )
            
            # Plan path (but limit iterations to prevent hanging)
            max_iter = min(self.rrt_kwargs.get('max_iterations', 500), 500)
            planner.max_iterations = max_iter
            
            # Disable verbose output for planning
            old_print = print
            def silent_print(*args, **kwargs):
                pass
            
            import builtins
            builtins.print = silent_print
            
            try:
                start_time = time.time()
                path = planner.plan()
                planning_time = time.time() - start_time
            finally:
                builtins.print = old_print
            
            # Record planning history for adaptive parameter tuning
            planning_success = len(path) > 1
            planning_record = {
                'success': planning_success,
                'planning_time': planning_time,
                'path_length': len(path),
                'parameters': {
                    'max_iterations': max_iterations,
                    'step_size': step_size,
                    'goal_sample_rate': goal_sample_rate,
                    'search_radius': search_radius
                }
            }
            self.planning_history.append(planning_record)
            
            # Keep only the last 10 planning records
            if len(self.planning_history) > 10:
                self.planning_history = self.planning_history[-10:]
            
            print(f"Path planning completed in {planning_time:.4f} seconds")
            print(f"Path length: {len(path)}")
            
            if len(path) > 0:
                # Optimize the path by removing unnecessary waypoints
                optimized_path = self._optimize_path(path)
                
                self.global_path = optimized_path
                self.current_waypoint_idx = 0
                self.last_plan_time = time.time()
                self.last_plan_position = start_position.copy()
                self.goal_position = goal_position.copy()
                self.planning_count += 1
                
                # Visualize the path if possible
                if self.client_id is not None:
                    planner.visualize_path(optimized_path, color=(0, 0, 1), line_width=2.0)  # Blue path
            else:
                # Оптимизированный запасной путь при неудачном планировании
                direction = goal_position - start_position
                distance = np.linalg.norm(direction)
                
                # Создаем более эффективный путь
                if distance > 2.0:
                    # Для длинных дистанций используем две промежуточные точки
                    # Первая точка - вверх от старта
                    first_point = start_position.copy()
                    first_point[2] += 0.5
                    
                    # Вторая точка - над целью
                    last_point = goal_position.copy()
                    last_point[2] += 0.5
                    
                    self.global_path = [start_position, first_point, last_point, goal_position]
                else:
                    # Для коротких дистанций - одна промежуточная точка повыше
                    midpoint = (start_position + goal_position) / 2
                    midpoint[2] = max(midpoint[2] + 0.8, start_position[2] + 0.8, goal_position[2] + 0.8)
                    self.global_path = [start_position, midpoint, goal_position]
                
                self.current_waypoint_idx = 0
                
        except Exception as e:
            print(f"Planning failed: {e}")
            # Оптимизированный запасной путь при ошибке
            direction = goal_position - start_position
            distance = np.linalg.norm(direction)
            
            # Создаем более эффективный путь
            if distance > 2.0:
                # Для длинных дистанций используем две промежуточные точки
                # Первая точка - вверх от старта
                first_point = start_position.copy()
                first_point[2] += 0.5
                
                # Вторая точка - над целью
                last_point = goal_position.copy()
                last_point[2] += 0.5
                
                self.global_path = [start_position, first_point, last_point, goal_position]
            else:
                # Для коротких дистанций - одна промежуточная точка повыше
                midpoint = (start_position + goal_position) / 2
                midpoint[2] = max(midpoint[2] + 0.8, start_position[2] + 0.8, goal_position[2] + 0.8)
                self.global_path = [start_position, midpoint, goal_position]
            
            self.current_waypoint_idx = 0
    
    def _optimize_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimize a path by removing unnecessary waypoints.
        
        Parameters
        ----------
        path : List[np.ndarray]
            Original path
            
        Returns
        -------
        List[np.ndarray]
            Optimized path
        """
        if len(path) <= 2:
            return path
            
        # Always keep start and goal
        optimized_path = [path[0]]
        
        # Add intermediate points only if they change direction significantly
        for i in range(1, len(path) - 1):
            prev_dir = path[i] - path[i-1]
            next_dir = path[i+1] - path[i]
            
            prev_dir_norm = np.linalg.norm(prev_dir)
            next_dir_norm = np.linalg.norm(next_dir)
            
            if prev_dir_norm > 0 and next_dir_norm > 0:
                # Normalize directions
                prev_dir = prev_dir / prev_dir_norm
                next_dir = next_dir / next_dir_norm
                
                # Calculate dot product to measure direction change
                dot_product = np.dot(prev_dir, next_dir)
                
                # If direction changes significantly (dot product < 0.9), keep the waypoint
                if dot_product < 0.9:
                    optimized_path.append(path[i])
            else:
                # If either segment is zero length, keep the waypoint
                optimized_path.append(path[i])
        
        # Add goal
        optimized_path.append(path[-1])
        
        return optimized_path
    
    def _get_hybrid_action(self, observation: np.ndarray, current_position: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
        """
        Get action using hybrid approach with improved goal targeting and direct goal approach.
        
        Parameters
        ----------
        observation : np.ndarray
            Full observation
        current_position : np.ndarray
            Current position
        goal_position : np.ndarray
            Goal position
            
        Returns
        -------
        np.ndarray
            Action
        """
        # Проверяем, что цель установлена правильно
        if goal_position is None or np.all(goal_position == 0):
            # Если цель не установлена или равна нулю, используем цель из self.goal_position
            if self.goal_position is not None:
                goal_position = self.goal_position
            else:
                # Если и self.goal_position не установлен, используем цель из наблюдения
                if observation.size >= 9:
                    goal_position = observation[6:9]
                else:
                    # Если нет цели, просто зависаем
                    return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(current_position - goal_position)
        
        # Выводим отладочную информацию
        if self.waypoint_control_count % 100 == 0:
            print(f"HYBRID ACTION:")
            print(f"  Current position: {current_position}")
            print(f"  Goal position: {goal_position}")
            print(f"  Distance to goal: {distance_to_goal:.2f}")
        
        # DIRECT GOAL APPROACH: If we're very close to the goal, target it directly
        if distance_to_goal < 0.5:  # Увеличиваем порог для более раннего перехода к точному управлению
            self.waypoint_control_count += 1
            return self._precise_goal_control(current_position, goal_position)
        
        # EMERGENCY TIMEOUT: If we're running out of time, go directly to goal
        elapsed_time = self.waypoint_control_count * 0.02  # Assuming 50Hz control
        if elapsed_time > 15.0 and distance_to_goal < 5.0:  # Уменьшаем порог времени для более раннего перехода к прямому управлению
            # Emergency direct approach to goal
            self.waypoint_control_count += 1
            
            # Create a direct path with intermediate point above
            midpoint = (current_position + goal_position) / 2
            midpoint[2] += 1.0  # Add height to avoid obstacles
            
            # If we're closer to midpoint, target goal directly
            if np.linalg.norm(current_position - midpoint) < 1.0:
                return self._precise_goal_control(current_position, goal_position)
            else:
                return self._waypoint_control(current_position, midpoint)
        
        # PATH FOLLOWING: If we have a path, follow it
        if len(self.global_path) > 0:
            # Get the current target waypoint
            if self.current_waypoint_idx < len(self.global_path):
                target_waypoint = self.global_path[self.current_waypoint_idx]
            else:
                # If we've reached the end of the path, target the goal directly
                self.waypoint_control_count += 1
                return self._precise_goal_control(current_position, goal_position)
            
            # Check if we're close to the current waypoint
            distance_to_waypoint = np.linalg.norm(current_position - target_waypoint)
            
            if distance_to_waypoint < self.waypoint_threshold:
                # Move to next waypoint
                self.current_waypoint_idx += 1
                
                # Check if we've reached the end of the path
                if self.current_waypoint_idx >= len(self.global_path):
                    # If we're at the last waypoint, target the goal directly
                    self.waypoint_control_count += 1
                    return self._precise_goal_control(current_position, goal_position)
                else:
                    # Otherwise, target the next waypoint
                    target_waypoint = self.global_path[self.current_waypoint_idx]
            
            # SHORTCUT: Check if we're close to the goal (even if not following the exact path)
            if distance_to_goal < 1.2:  # Увеличиваем порог для более раннего перехода к прямому управлению
                # When close to goal, target it directly
                self.waypoint_control_count += 1
                return self._precise_goal_control(current_position, goal_position)
            
            # OBSTACLE AVOIDANCE: Decide whether to use RL policy or waypoint following
            obstacles_nearby = self._detect_nearby_obstacles(current_position)
            
            if (self.rl_policy is not None and obstacles_nearby):
                # Use RL policy for local control when close to obstacles
                try:
                    action, _ = self.rl_policy.predict(observation, deterministic=True)
                    self.rl_control_count += 1
                    return action
                except Exception as e:
                    print(f"RL policy failed: {e}")
                    # Fallback to waypoint following if RL fails
                    pass
            
            # Use waypoint following
            self.waypoint_control_count += 1
            return self._waypoint_control(current_position, target_waypoint)
        
        # NO PATH: If we have no path, use direct control with intermediate point
        # Создаем прямой путь к цели через промежуточную точку
        self._plan_global_path(current_position, goal_position)
        
        # Если путь создан успешно, следуем по нему
        if len(self.global_path) > 0:
            self.waypoint_control_count += 1
            return self._waypoint_control(current_position, self.global_path[0])
        
        # Если не удалось создать путь, идем напрямую к цели
        self.waypoint_control_count += 1
        return self._precise_goal_control(current_position, goal_position)
        
    def _precise_goal_control(self, current_position: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
        """
        Precise control for final approach to goal with aggressive targeting.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current position
        goal_position : np.ndarray
            Goal position
            
        Returns
        -------
        np.ndarray
            Action (velocity command [vx, vy, vz, yaw_rate])
        """
        # Проверяем, что цель не нулевая
        if np.all(goal_position == 0):
            print("WARNING: Goal position is [0,0,0], using self.goal_position instead")
            if hasattr(self, 'goal_position') and self.goal_position is not None:
                goal_position = self.goal_position
            else:
                print("ERROR: No valid goal position found!")
                
        # Вычисляем вектор направления к цели
        direction = goal_position - current_position
        distance = np.linalg.norm(direction)
        
        # Преобразуем в команды для дрона в формате [vx, vy, vz, yaw_rate]
        if distance > 0:
            # Нормализуем направление
            norm_direction = direction / distance
            
            # Адаптивная скорость в зависимости от расстояния - оптимизируем для более точного приближения
            if distance < 0.15:
                # Очень близко - очень точное, медленное движение
                speed = min(0.3, distance * 2.0)
            elif distance < 0.5:
                # Близко - точное движение
                speed = min(1.2, distance * 2.0)
            else:
                # Дальше - более быстрое движение
                speed = min(3.0, distance * 1.5)
            
            # Применяем скорость к направлению
            vx = norm_direction[0] * speed
            vy = norm_direction[1] * speed
            vz = norm_direction[2] * speed
            
            # Добавляем небольшое смещение вверх, чтобы избежать столкновений с землей
            if current_position[2] < 0.5:
                vz += 0.2
        else:
            # Если мы в цели, зависаем
            vx, vy, vz = 0.0, 0.0, 0.0
        
        # Скорость вращения (не используется)
        yaw_rate = 0.0
        
        # Формируем команду в формате [vx, vy, vz, yaw_rate]
        action = np.array([vx, vy, vz, yaw_rate])
        
        # Выводим отладочную информацию
        if self.waypoint_control_count % 100 == 0:
            print(f"PRECISE GOAL CONTROL:")
            print(f"  Current position: {current_position}")
            print(f"  Goal position: {goal_position}")
            print(f"  Distance to goal: {distance:.2f}")
            print(f"  Direction: {direction}")
            print(f"  Speed: {speed if distance > 0 else 0}")
            print(f"  Action (velocity): {action}")
        
        return action
    
    def _detect_nearby_obstacles(self, position: np.ndarray, radius: float = 1.0) -> bool:
        """
        Detect if there are obstacles nearby with improved detection.
        
        Parameters
        ----------
        position : np.ndarray
            Current position
        radius : float
            Detection radius
            
        Returns
        -------
        bool
            True if obstacles are detected nearby
        """
        if not self.obstacle_ids or self.client_id is None:
            return False
            
        import pybullet as p
        
        # Check ground proximity first (treat ground as an obstacle)
        if position[2] < 0.3:  # If close to ground
            return True
            
        for obstacle_id in self.obstacle_ids:
            try:
                # Get the AABB of the obstacle
                aabb_min, aabb_max = p.getAABB(obstacle_id, physicsClientId=self.client_id)
                
                # Уменьшаем запас безопасности для более оптимального пути
                aabb_min = np.array(aabb_min) - 0.05
                aabb_max = np.array(aabb_max) + 0.05
                
                # Calculate obstacle center
                obstacle_center = np.array([(aabb_min[i] + aabb_max[i]) / 2 for i in range(3)])
                
                # Calculate distance to obstacle center
                distance_to_center = np.linalg.norm(position - obstacle_center)
                
                # Quick check based on center distance
                if distance_to_center < radius + 1.0:  # Add 1.0 to account for obstacle size
                    # More precise check using AABB
                    # Calculate closest point on AABB to position
                    closest_point = np.array([
                        max(aabb_min[0], min(position[0], aabb_max[0])),
                        max(aabb_min[1], min(position[1], aabb_max[1])),
                        max(aabb_min[2], min(position[2], aabb_max[2]))
                    ])
                    
                    # Calculate distance to closest point
                    distance_to_surface = np.linalg.norm(position - closest_point)
                    
                    if distance_to_surface < radius:
                        return True
            except Exception as e:
                # If there's an error, be conservative and assume obstacle is nearby
                return True
                
        return False
    
    def _waypoint_control(self, current_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
        """
        Advanced waypoint following control with adaptive speed.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current position
        target_position : np.ndarray
            Target position
            
        Returns
        -------
        np.ndarray
            Action in format [vx, vy, vz, yaw_rate]
        """
        # Вычисляем вектор направления к цели
        direction = target_position - current_position
        distance = np.linalg.norm(direction)
        
        # Преобразуем в команды для дрона в формате [vx, vy, vz, yaw_rate]
        if distance > 0:
            # Нормализуем направление
            norm_direction = direction / distance
            
            # Адаптивная скорость в зависимости от расстояния - оптимизируем для более быстрого полета
            if distance < 0.2:
                # Медленно при приближении к цели
                speed = min(self.max_speed * 0.3, distance * 1.5)
            elif distance < 0.6:
                # Средняя скорость на среднем расстоянии
                speed = min(self.max_speed * 0.9, distance * 1.8)
            else:
                # Полная скорость на большом расстоянии
                speed = min(self.max_speed, distance * 1.5)
            
            # Применяем скорость к направлению
            vx = norm_direction[0] * speed
            vy = norm_direction[1] * speed
            vz = norm_direction[2] * speed
            
            # Добавляем небольшое смещение вверх, чтобы избежать столкновений с землей
            if current_position[2] < 0.5:
                vz += 0.2
        else:
            # Если мы в цели, зависаем
            vx, vy, vz = 0.0, 0.0, 0.0
        
        # Скорость вращения (не используется)
        yaw_rate = 0.0
        
        # Формируем команду в формате [vx, vy, vz, yaw_rate]
        action = np.array([vx, vy, vz, yaw_rate])
        
        return action
    
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
            Action in format [vx, vy, vz, yaw_rate]
        """
        # Используем waypoint_control для прямого управления
        return self._waypoint_control(current_position, target_position)
    
    def act(self, observation: np.ndarray, t: float) -> np.ndarray:
        """
        Act method for compatibility with the validator's _run_episode function.
        
        Parameters
        ----------
        observation : np.ndarray
            Current observation
        t : float
            Current time
            
        Returns
        -------
        np.ndarray
            Action (RPM command)
        """
        # Получаем текущую позицию из наблюдения
        if observation.ndim > 1:
            current_position = observation[0, :3]
        else:
            current_position = observation[:3]
            
        # Используем self.goal_position, если он установлен
        if hasattr(self, 'goal_position') and self.goal_position is not None:
            goal_position = self.goal_position
        else:
            # Иначе пытаемся получить цель из наблюдения
            if observation.size >= 9:
                if observation.ndim > 1:
                    goal_position = observation[0, 6:9]
                else:
                    goal_position = observation[6:9]
            else:
                # Если нет цели, используем нулевую цель
                goal_position = np.zeros(3)
                
        # Вычисляем действие напрямую, минуя predict
        action = self._get_hybrid_action(observation, current_position, goal_position)
        
        # Ensure action is in the correct format (1D array)
        if len(action.shape) > 1:
            action = action.flatten()
            
        # Ensure action values are reasonable (clip if necessary)
        action = np.clip(action, -10, 10)
        
        # Debug output every 50 steps
        if self.waypoint_control_count % 50 == 0:
            print(f"ACT: t={t:.2f}, action={action}")
            print(f"  Position: {current_position}")
            print(f"  Goal: {goal_position}")
            print(f"  Distance: {np.linalg.norm(current_position - goal_position):.2f}")
        
        # Return action as a 1D array (required by env.step)
        return action
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the hybrid policy usage.
        
        Returns
        -------
        Dict[str, int]
            Statistics dictionary
        """
        return {
            'planning_count': self.planning_count,
            'rl_control_count': self.rl_control_count,
            'waypoint_control_count': self.waypoint_control_count,
        }