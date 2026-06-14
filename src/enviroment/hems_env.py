
import gym
import numpy as np
import pandas as pd
from gym import spaces
from copy import deepcopy


EPS = 1e-8


class HEMSEnvironment(gym.Env):
    """
    Gym environment for Home Energy Management Systems (HEMS)
    using continuous control actions.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config, energy_env):
        super().__init__()

        # ---------------------------
        # External data handler
        # ---------------------------
        self.env_data = energy_env

        # ---------------------------
        # Configuration
        # ---------------------------
        self.window_length = getattr(config, "window_length", 24)
        self.dt = getattr(config, "delta_t", 1.0)

        # Battery parameters
        self.bat_capacity = getattr(config, "battery_capacity", 10.0)  # kWh
        self.p_bat_max = getattr(config, "p_bat_max", 5.0)
        self.soc_min = getattr(config, "soc_min", 0.1)
        self.soc_max = getattr(config, "soc_max", 0.9)

        # Reward weights
        self.lambda_T = getattr(config, "lambda_T", 2.0)
        self.lambda_deg = getattr(config, "lambda_deg", 0.05)
        self.lambda_uns = getattr(config, "lambda_uns", 10.0)
        self.lambda_sw = getattr(config, "lambda_sw", 0.1)

        # ---------------------------
        # Action space (Option A)
        # ---------------------------
        self.action_space = spaces.Box(
            low=np.array([-1, -1,  0, -1,  0], dtype=np.float32),
            high=np.array([ 1,  1,  1,  1,  1], dtype=np.float32),
        )

        # ---------------------------
        # Observation space
        # ---------------------------
        num_features = self.env_data.num_features
        obs_dim = self.window_length * num_features + 1 + 5 + 4

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ---------------------------
        # Internal states
        # ---------------------------
        self.reset()

    # ====================================================
    #                    RESET
    # ====================================================
    def reset(self):
        self.current_step = 0
        self.done = False

        # Battery
        self.soc = 0.5

        # Previous action
        self.prev_action = np.zeros(5, dtype=np.float32)

        obs = self._get_state()
        return obs

    # ====================================================
    #                    STEP
    # ====================================================
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Decode actions
        p_bat, p_grid, p_pv, p_load, idle = action

        # Battery SOC update
        self.soc += (p_bat * self.dt) / self.bat_capacity
        soc_violation = max(0, self.soc - self.soc_max) + max(0, self.soc_min - self.soc)
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)

        # Get data
        obs_now, obs_next, done_flag = self.env_data.step()
        price = obs_now[-1, self.env_data.feature_names.index("price")]
        load = obs_now[-1, self.env_data.feature_names.index("load")]
        temp_in = obs_now[-1, self.env_data.feature_names.index("temp_out")]

        # ---------------------------
        # Cost components
        # ---------------------------
        energy_cost = price * p_grid * self.dt
        comfort_cost = self.lambda_T * (temp_in - 22.0) ** 2
        battery_cost = self.lambda_deg * abs(p_bat)
        unmet_load_cost = self.lambda_uns * max(0, load - p_load)
        switching_cost = self.lambda_sw * np.sum(np.abs(action - self.prev_action))

        reward = -(
            energy_cost
            + comfort_cost
            + battery_cost
            + unmet_load_cost
            + switching_cost
            + 100.0 * soc_violation
        )

        self.prev_action = action.copy()
        self.current_step += 1
        self.done = done_flag

        obs = self._get_state()

        info = {
            "energy_cost": energy_cost,
            "comfort_cost": comfort_cost,
            "battery_cost": battery_cost,
            "soc": self.soc,
        }

        return obs, float(reward), self.done, info

    # ====================================================
    #                STATE CONSTRUCTION
    # ====================================================
    def _get_state(self):
        obs = self.env_data.get_observation(self.current_step)
        obs_flat = obs.flatten()

        time_idx = self.env_data.step_to_time(self.current_step)
        hour = time_idx.hour
        day = time_idx.weekday()

        time_features = np.array(
            [
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * day / 7),
                np.cos(2 * np.pi * day / 7),
            ],
            dtype=np.float32,
        )

        state = np.concatenate(
            [
                obs_flat,
                np.array([self.soc], dtype=np.float32),
                self.prev_action,
                time_features,
            ]
        )

        return state.astype(np.float32)

    # ====================================================
    #                  RENDER
    # ====================================================
    def render(self, mode="human"):
        print(
            f"Step={self.current_step}, SoC={self.soc:.2f}, PrevAction={self.prev_action}"
        )
