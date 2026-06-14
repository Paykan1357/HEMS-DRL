
import gym
import numpy as np
from gym import spaces
from copy import deepcopy

EPS = 1e-8


class HEMSEnvironment(gym.Env):
    """
    Physically consistent HEMS environment (CORRECTED VERSION).
    """

    def __init__(self, config, energy_env):
        super().__init__()

        self.env_data = energy_env
        self.window_length = getattr(config, "window_length", 24)
        self.delta_t = getattr(config, "delta_t", 1.0)

        # Battery parameters
        self.bat_capacity = getattr(config, "battery_capacity", 10.0)  # kWh
        self.p_bat_max = getattr(config, "p_bat_max", 6.0)
        self.soc_min = getattr(config, "soc_min", 0.05)
        self.soc_max = getattr(config, "soc_max", 0.95)

        # Grid parameters
        self.p_grid_import_max = getattr(config, "p_grid_import_max", 10.0)  # kW
        self.p_grid_export_max = getattr(config, "p_grid_export_max", 5.0)   # kW

        # Reward weights (FIXED: no duplicate)
        self.lambda_T = getattr(config, "lambda_T", 0.05)
        self.lambda_deg = getattr(config, "lambda_deg", 0.002)
        self.lambda_uns = getattr(config, "lambda_uns", 10.0)
        self.lambda_sw = getattr(config, "lambda_sw", 0.0001)
        self.lambda_pv = getattr(config, "lambda_pv", 0.05)  # ← SINGLE DEFINITION

        # Comfort parameters
        self.T_set = getattr(config, "T_set", 22.0)           # °C
        self.dead_band = getattr(config, "dead_band", 0.5)    # °C

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        # State space
        num_features = self.env_data.num_features
        obs_dim = self.window_length * num_features + 2 + 4 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.feature_map = {n: i for i, n in enumerate(self.env_data.feature_names)}
        required_features = ["price", "P_load", "P_pv", "T_in", "T_out", "occ"]
        for k in required_features:
            if k not in self.feature_map:
                raise KeyError(f"Missing feature {k}")

        self.reset()

    def reset(self):
        self.current_step = 0
        self.soc = 0.5
        self.action_prev = np.zeros(4, dtype=np.float32)
        self.reward_info = {}
        return self._get_state()

    def _denormalize_action(self, normalized_action):
        p_bat = normalized_action[0] * self.p_bat_max
        
        if normalized_action[1] >= 0:
            p_grid = normalized_action[1] * self.p_grid_import_max
        else:
            p_grid = normalized_action[1] * self.p_grid_export_max
        
        pv_frac = (normalized_action[2] + 1) / 2
        load_multiplier = (normalized_action[3] + 1) / 2
        
        return {
            'bat': p_bat,
            'grid': p_grid,
            'pv_frac': pv_frac,
            'load_multiplier': load_multiplier
        }

    def _compute_comfort_penalty(self, T_in):
        deviation = abs(T_in - self.T_set) - self.dead_band
        if deviation > 0:
            return self.lambda_T * (deviation ** 2)
        return 0.0

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Denormalize action
        phys_action = self._denormalize_action(action)
        P_bat_cmd = phys_action['bat']
        P_grid_cmd = phys_action['grid']
        pv_frac = phys_action['pv_frac']
        load_multiplier = phys_action['load_multiplier']

        # Get current observations
        obs_now, _, done = self.env_data.step()
        f = self.feature_map

        price = obs_now[-1, f["price"]]
        P_load_requested = obs_now[-1, f["P_load"]]
        P_pv_avail = obs_now[-1, f["P_pv"]]
        T_in = obs_now[-1, f["T_in"]]

        # ==========================================================
        # APPLY CONTROLS
        # ==========================================================
        
        P_load_served = P_load_requested * load_multiplier
        P_pv_used = min(pv_frac * P_pv_avail, P_pv_avail)
        P_bat = np.clip(P_bat_cmd, -self.p_bat_max, self.p_bat_max)
        
        # Battery SoC update
        delta_soc = (P_bat * self.delta_t) / self.bat_capacity
        self.soc = np.clip(self.soc + delta_soc, self.soc_min, self.soc_max)
        
        if self.soc <= self.soc_min and P_bat < 0:
            P_bat = 0.0
        elif self.soc >= self.soc_max and P_bat > 0:
            P_bat = 0.0
        
        # Power balance
        P_grid_required = P_load_served - P_pv_used - P_bat
        P_grid = np.clip(P_grid_required, -self.p_grid_export_max, self.p_grid_import_max)
        
        # Unmet load
        L_uns = max(0.0, P_grid_required - P_grid)
        
        P_grid_import = max(0.0, P_grid)
        P_grid_export = abs(min(0.0, P_grid))

        # ==========================================================
        # REWARD CALCULATION - CORRECTED
        # ==========================================================
        
        # FIX 1: Separate import cost and export revenue
        # Import cost (always >= 0)
        import_cost = price * P_grid_import * self.delta_t
        
        # Export revenue (always <= 0, reduces cost)
        # Using feed-in tariff (80% of import price as in your paper)
        feed_in_tariff = price * 0.8
        export_revenue = feed_in_tariff * P_grid_export * self.delta_t
        
        # Net cost (positive = paying, negative = earning)
        C_t = import_cost - export_revenue
        
        # FIX 2: PV reward should be small positive (incentive, not profit center)
        # Changed from 0.05 to 0.01 to avoid unrealistic profits
        pv_incentive = 0.01  # Small incentive for using PV
        R_t = pv_incentive * P_pv_used * self.delta_t
        
        # Other penalties
        U_t = self._compute_comfort_penalty(T_in)
        D_t = self.lambda_deg * abs(P_bat) * self.delta_t
        S_t = self.lambda_uns * L_uns * self.delta_t
        W_t = self.lambda_sw * np.sum(np.abs(action - self.action_prev))
        
        # FIX 3: Correct reward calculation (no double negative confusion)
        # Total cost to minimize = all penalties + net energy cost - PV incentive
        total_cost = C_t + U_t + D_t + S_t + W_t - R_t
        
        # Reward is negative cost (maximize reward = minimize cost)
        reward = -total_cost
        
        # Clip for stability
        reward = np.clip(reward, -10.0, 10.0)

        # Logging
        self.reward_info = {
            "energy_cost": C_t,
            "import_cost": import_cost,
            "export_revenue": export_revenue,
            "grid_import": P_grid_import,
            "grid_export": P_grid_export,
            "pv_used": P_pv_used,
            "pv_available": P_pv_avail,
            "load_served": P_load_served,
            "load_requested": P_load_requested,
            "unmet_load": L_uns,
            "soc": self.soc,
            "battery_power": P_bat,
            "temperature_in": T_in,
            "comfort_penalty": U_t,
            "smoothing_penalty": W_t,
            "degradation_penalty": D_t,
            "pv_reward": R_t,
            "total_cost": total_cost,
            "reward": reward
        }

        self.action_prev = action.copy()
        self.current_step += 1

        return self._get_state(), float(reward), done, self.reward_info

    def _get_state(self):
        obs = self.env_data.get_observation(self.current_step).flatten()
        
        current_obs = self.env_data.get_current_observation(self.current_step)
        f = self.feature_map
        P_pv_avail = current_obs[f["P_pv"]] if current_obs is not None else 0.0
        device_state = np.array([self.soc, P_pv_avail], dtype=np.float32)
        
        prev_action = self.action_prev.copy()
        
        time_idx = self.env_data.step_to_time(self.current_step)
        h = time_idx.hour
        d = time_idx.weekday()
        time_features = np.array([
            np.sin(2 * np.pi * h / 24),
            np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * d / 7),
            np.cos(2 * np.pi * d / 7),
        ], dtype=np.float32)
        
        state = np.concatenate([
            obs, device_state, prev_action, time_features
        ]).astype(np.float32)
        
        return state
