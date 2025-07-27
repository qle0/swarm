# swarm/envs/moving_drone.py
from __future__ import annotations

from typing import Optional, Tuple, Iterable
import numpy as np
import gymnasium.spaces as spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)
import pybullet as p

# ── project‑level utilities ────────────────────────────────────────────────
from swarm.validator.reward import flight_reward          # 3‑term scorer
from swarm.constants        import GOAL_TOL, HOVER_SEC


class MovingDroneAviary(BaseRLAviary):
    """
    Single‑drone environment whose *start*, *goal* and *horizon* are supplied
    via an external `MapTask`.

    The per‑step reward is the **increment** of `flight_reward`, so it can be
    fed directly to PPO/TD3/etc. without extra shaping.
    """
    MAX_TILT_RAD: float = 0.7          # safety cut‑off for roll / pitch (rad)

    # --------------------------------------------------------------------- #
    # 1. constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        task,
        drone_model : DroneModel   = DroneModel.CF2X,
        physics     : Physics      = Physics.PYB,
        pyb_freq    : int          = 240,
        ctrl_freq   : int          = 30,
        gui         : bool         = False,
        record      : bool         = False,
        obs         : ObservationType = ObservationType.KIN,
        act         : ActionType      = ActionType.RPM,
    ):
        """
        Parameters
        ----------
        task : MapTask
            Must expose `.start`, `.goal`, `.horizon`, `.sim_dt`.
        Remaining arguments are forwarded to ``BaseRLAviary`` unchanged.
        """
        self.task       = task
        self.GOAL_POS   = np.asarray(task.goal, dtype=float)
        self.EP_LEN_SEC = float(task.horizon)

        # internal book‑keeping
        self._time_alive   = 0.0
        self._hover_sec    = 0.0
        self._success      = False
        self._t_to_goal    = None
        self._prev_score   = 0.0

        # Let BaseRLAviary set up the PyBullet world
        super().__init__(
            drone_model  = drone_model,
            num_drones   = 1,
            initial_xyzs = np.asarray([task.start]),
            initial_rpys = None,
            physics      = physics,
            pyb_freq     = pyb_freq,
            ctrl_freq    = ctrl_freq,
            gui          = gui,
            record       = record,
            obs          = obs,
            act          = act,
        )

        # ‑‑‑ extend observation with relative goal vector (x, y, z)/10 ‑‑‑
        old_low,  old_high  = self.observation_space.low, self.observation_space.high
        pad_low  = -np.ones((old_low.shape[0], 3), dtype=np.float32) * np.inf
        pad_high = +np.ones((old_high.shape[0], 3), dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(
            low   = np.concatenate([old_low,  pad_low ], axis=1),
            high  = np.concatenate([old_high, pad_high], axis=1),
            dtype = np.float32,
        )

    # --------------------------------------------------------------------- #
    # 2. low‑level helpers
    # --------------------------------------------------------------------- #
    @property
    def _sim_dt(self) -> float:
        """Physics step in seconds (1 / CTRL_FREQ)."""
        return 1.0 / self.CTRL_FREQ

    # --------------------------------------------------------------------- #
    # 3. OpenAI‑Gym API overrides
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        """
        Resets the underlying simulator and internal counters,
        returns initial observation and info as usual.
        """
        obs, info = super().reset(**kwargs)

        self._time_alive = 0.0
        self._hover_sec  = 0.0
        self._success    = False
        self._t_to_goal  = None

        # baseline score (t = 0, e = 0)
        self._prev_score = flight_reward(
            success = False,
            t       = 0.0,
            e       = 0.0,
            horizon = self.EP_LEN_SEC,
        )

        return obs, info

    # -------- reward ----------------------------------------------------- #
    def _computeReward(self) -> float:
        """
        **Incremental** reward based on the three‑term `flight_reward`.
        """
        # current distance to goal
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(state[0:3] - self.GOAL_POS))

        # ── success detection: remain inside GOAL_TOL for HOVER_SEC seconds ──
        reached = dist < GOAL_TOL
        if reached:
            self._hover_sec += self._sim_dt
            if self._hover_sec >= HOVER_SEC and not self._success:
                self._success   = True
                self._t_to_goal = self._time_alive
        else:
            self._hover_sec = 0.0

        # ── clock update ────────────────────────────────────────────────────
        self._time_alive += self._sim_dt

        # ── call new reward function ───────────────────────────────────────
        score = flight_reward(
            success = self._success,
            t       = (self._t_to_goal if self._success else self._time_alive),
            e       = 0.0,                        # energy not tracked inside env
            horizon = self.EP_LEN_SEC,
        )

        r_t              = score - self._prev_score
        self._prev_score = score
        return float(r_t)

    # -------- termination ------------------------------------------------ #
    def _computeTerminated(self) -> bool:
        """
        Episode ends only when the success condition is definitely met *or*
        PyBullet flags a fatal collision (handled upstream).
        """
        # TODO: re‑enable collision handling (if desired)
        return bool(self._success)

    # -------- truncation (timeout / safety) ------------------------------ #
    def _computeTruncated(self) -> bool:
        """
        Early termination on excessive tilt or elapsed horizon.
        """
        # safety cut‑off
        state = self._getDroneStateVector(0)
        roll, pitch = state[7], state[8]
        if abs(roll) > self.MAX_TILT_RAD or abs(pitch) > self.MAX_TILT_RAD:
            return True

        # timeout
        return self._time_alive >= self.EP_LEN_SEC

    # -------- extra logging --------------------------------------------- #
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(state[0:3] - self.GOAL_POS))
        return {
            "distance_to_goal": dist,
            "score"           : self._prev_score,
            "success"         : self._success,
            "t_to_goal"       : self._t_to_goal,
        }

    # -------- action preprocessing ---------------------------------------- #
    def _preprocessAction(self, action):
        """
        Переопределение метода для правильной обработки команд скорости.
        
        В оригинальном методе для ActionType.VEL скорость масштабируется значением
        target[3] (yaw_rate), что приводит к нулевой скорости, если yaw_rate=0.
        
        Эта реализация использует непосредственно значения vx, vy, vz без масштабирования.
        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        
        for k in range(action.shape[0]):
            target = action[k, :]
            
            if self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                
                # Используем непосредственно компоненты скорости без масштабирования yaw_rate
                target_velocity = target[0:3]
                
                # Ограничиваем скорость, если она превышает лимит
                speed = np.linalg.norm(target_velocity)
                if speed > self.SPEED_LIMIT and speed > 0:
                    target_velocity = target_velocity * (self.SPEED_LIMIT / speed)
                
                # Вычисляем RPM через контроллер
                temp, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=state[0:3],  # текущая позиция
                    target_rpy=np.array([0, 0, state[9]]),  # сохраняем текущий yaw
                    target_vel=target_velocity  # целевая скорость напрямую
                )
                rpm[k, :] = temp
            else:
                # Для других типов действий используем оригинальную реализацию
                rpm[k, :] = super()._preprocessAction(action[k:k+1, :])[0]
                
        return rpm
        
    # -------- observation extension -------------------------------------- #
    def _computeObs(self) -> np.ndarray:
        """
        Default kinematics (12‑D) plus scaled relative goal vector (3‑D) → 15‑D.
        """
        kin = super()._computeObs()                       # shape (1, 12)
        rel = ((self.GOAL_POS - kin[0, :3]) / 10.0).reshape(1, 3)
        return np.concatenate([kin, rel], axis=1).astype(np.float32)
