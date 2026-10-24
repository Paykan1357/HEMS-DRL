
import gym
import numpy as np
from gym import spaces

EPS = 1e-8
DEBUG = False


class HEMSEnvironment(gym.Env):
    """
    HEMS Environment — FINAL FIXED VERSION.

    Fixes applied:
    - Episode length = 96 steps
    - Reward clipped to [-10, 10]
    - Comfort penalty clipped to max 5.0
    - Indoor temperature clipped to [15, 35] °C
    - Battery SoC clipping corrected
    - reset() resets underlying data pointer
    - Handles both "occ" and "occupancy" column names
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config, energy_env):
        super().__init__()

        self.env_data = energy_env
        self.window_length = getattr(config, "window_length", 24)
        self.delta_t = getattr(config, "delta_t", 1.0)
        self.episode_length = int(getattr(config, "episode_length", 96))

        # Battery
        self.bat_capacity = getattr(config, "battery_capacity", 10.0)
        self.p_bat_max    = getattr(config, "p_bat_max", 3.0)
        self.soc_min      = getattr(config, "soc_min", 0.05)
        self.soc_max      = getattr(config, "soc_max", 0.95)
        self.eta_ch       = getattr(config, "eta_ch", 0.95)
        self.eta_dis      = getattr(config, "eta_dis", 0.95)

        # Grid
        self.p_grid_import_max = getattr(config, "p_grid_import_max", 15.0)
        self.p_grid_export_max = getattr(config, "p_grid_export_max", 5.0)

        # HVAC
        self.T_set     = getattr(config, "T_set", 22.0)
        self.dead_band = getattr(config, "dead_band", 0.5)
        self.T_min     = getattr(config, "T_min", 20.0)
        self.T_max     = getattr(config, "T_max", 26.0)
        self.R_thermal = getattr(config, "R_thermal", 5.0)
        self.C_thermal = getattr(config, "C_thermal", 4.0)
        self.Q_max     = getattr(config, "Q_max", 10.0)
        self.COP_cool  = getattr(config, "COP_cool", 3.0)
        self.COP_heat  = getattr(config, "COP_heat", 3.5)
        self.q_occup_per_person = getattr(config, "q_occup_per_person", 0.2)

        # ⭐ Comfort penalty clip (NEW)
        self.comfort_penalty_max = getattr(config, "comfort_penalty_max", 5.0)

        # Reward weights
        self.lambda_T   = getattr(config, "lambda_T", 0.05)
        self.lambda_deg = getattr(config, "lambda_deg", 0.002)
        self.lambda_uns = getattr(config, "lambda_uns", 2.0)
        self.lambda_sw  = getattr(config, "lambda_sw", 0.0001)
        self.lambda_pv  = getattr(config, "lambda_pv", 0.05)

        # Reward scaling
        self.reward_scale = getattr(config, "reward_scale", 10.0)
        self.reward_clip  = getattr(config, "reward_clip", 10.0)

        # Action space
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)

        # State space
        num_features = self.env_data.num_features
        obs_dim = self.window_length * num_features + 3 + 5 + 4
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.feature_map = {n: i for i, n in enumerate(self.env_data.feature_names)}

        # ⭐ Handle both "occ" and "occupancy"
        # Handle both "occ" and "occupancy"
        if "occ" in self.feature_map and "occupancy" not in self.feature_map:
            self.feature_map["occupancy"] = self.feature_map["occ"]
        
        required = ['P_load', 'P_pv', 'price', 'T_out', 'occupancy']
        for k in required:
            if k not in self.feature_map:
                raise KeyError(f"Missing feature '{k}'. Available: {list(self.feature_map)}")

        # Alias: 'occ' refers to 'occupancy'
        if 'occupancy' in self.feature_map and 'occ' not in self.feature_map:
            self.feature_map['occ'] = self.feature_map['occupancy']
        
        for k in required:
            if k not in self.feature_map:
                raise KeyError(f"Missing feature '{k}'. Available: {list(self.feature_map)}")

        self._np_random = np.random.RandomState()
        self._episode_steps = 0

        self.reset()

    def seed(self, seed=None):
        self._np_random = np.random.RandomState(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        """Reset environment AND underlying data for new episode."""
        self.env_data.current_step = 0
        try:
            self.env_data.reset()
        except Exception:
            pass

        self.current_step = 0
        self._episode_steps = 0
        self.soc = 0.5

        try:
            self.T_in = float(self.env_data.data[0, self.feature_map["T_out"]])
        except Exception:
            self.T_in = 22.0

        self.action_prev = np.zeros(5, dtype=np.float32)
        self.reward_info = {}
        return self._get_state()

    def _denormalize_action(self, a):
        a = np.clip(a, -1.0, 1.0)
        p_bat = a[0] * self.p_bat_max
        p_grid = a[1] * (self.p_grid_import_max if a[1] >= 0 else self.p_grid_export_max)
        pv_frac   = (a[2] + 1) / 2
        load_mult = (a[3] + 1) / 2
        q_hvac    = a[4] * self.Q_max
        return p_bat, p_grid, pv_frac, load_mult, q_hvac

    def _comfort_penalty(self, T_in):
        """Comfort penalty, clipped to prevent explosion."""
        dev = abs(T_in - self.T_set) - self.dead_band
        if dev > 0:
            penalty = self.lambda_T * (dev ** 2)
            return min(penalty, self.comfort_penalty_max)   # ⭐ CLIP
        return 0.0

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        p_bat_cmd, _, pv_frac, load_mult, q_hvac_cmd = self._denormalize_action(action)

        # --- read current obs ---
        obs_now, _, data_done = self.env_data.step()
        f = self.feature_map
        price      = float(obs_now[-1, f["price"]])
        P_load_req = float(obs_now[-1, f["P_load"]])
        P_pv_avail = float(obs_now[-1, f["P_pv"]])
        T_out      = float(obs_now[-1, f["T_out"]])
        occ        = float(obs_now[-1, f["occ"]])

        # --- apply controls ---
        P_load_served = P_load_req * load_mult
        P_pv_used     = float(np.clip(pv_frac * P_pv_avail, 0.0, P_pv_avail))

        # --- Battery ---
        P_bat = float(np.clip(p_bat_cmd, -self.p_bat_max, self.p_bat_max))
        P_ch  = max(0.0, -P_bat)
        P_dis = max(0.0,  P_bat)

        delta_soc = (self.eta_ch * P_ch - P_dis / self.eta_dis) * self.delta_t / self.bat_capacity
        soc_new = self.soc + delta_soc

        if soc_new > self.soc_max:
            soc_new = self.soc_max
            P_bat = P_dis
        elif soc_new < self.soc_min:
            soc_new = self.soc_min
            P_bat = -P_ch

        self.soc = float(np.clip(soc_new, self.soc_min, self.soc_max))

        # --- HVAC electrical ---
        if q_hvac_cmd >= 0:
            P_hvac_elec = q_hvac_cmd / self.COP_cool
        else:
            P_hvac_elec = abs(q_hvac_cmd) / self.COP_heat

        # --- Thermal dynamics (1R1C) ---
        Q_gains = self.q_occup_per_person * occ
        T_in_new = self.T_in + (self.delta_t / self.C_thermal) * (
            (T_out - self.T_in) / self.R_thermal - q_hvac_cmd + Q_gains
        )
        self.T_in = float(np.clip(T_in_new, 15.0, 35.0))   # ⭐ CLIP T_in

        # --- Power balance ---
        P_grid_req = P_load_served + P_hvac_elec - P_pv_used - P_bat
        P_grid = float(np.clip(P_grid_req, -self.p_grid_export_max, self.p_grid_import_max))
        L_uns = max(0.0, P_grid_req - P_grid)
        P_imp = max(0.0, P_grid)
        P_exp = abs(min(0.0, P_grid))

        # --- Reward components ---
        C_t = price * P_imp * self.delta_t - 0.8 * price * P_exp * self.delta_t
        R_t = self.lambda_pv * P_pv_used * self.delta_t
        U_t = self._comfort_penalty(self.T_in)
        D_t = self.lambda_deg * abs(P_bat) * self.delta_t
        S_t = self.lambda_uns * L_uns * self.delta_t
        W_t = self.lambda_sw * np.sum(np.abs(action - self.action_prev))

        total_cost = C_t + U_t + D_t + S_t + W_t - R_t
        raw_reward = -total_cost / self.reward_scale
        reward = float(np.clip(raw_reward, -self.reward_clip, self.reward_clip))

        # --- Log ---
        self.reward_info = {
            "energy_cost": C_t,               # ⭐ pure electricity cost
            "grid_import": P_imp,
            "grid_export": P_exp,
            "pv_used": P_pv_used,
            "pv_available": P_pv_avail,
            "load_served": P_load_served,
            "load_requested": P_load_req,
            "unmet_load": L_uns,
            "soc": self.soc,
            "battery_power": P_bat,
            "temperature_in": self.T_in,
            "temperature_out": T_out,
            "q_hvac": q_hvac_cmd,
            "p_hvac_elec": P_hvac_elec,
            "comfort_penalty": U_t,
            "degradation_penalty": D_t,
            "smoothing_penalty": W_t,
            "pv_reward": R_t,
            "total_cost": total_cost,
            "reward": reward,
        }

        self.action_prev = action.copy()
        self.current_step   += 1
        self._episode_steps += 1

        episode_done = (self._episode_steps >= self.episode_length)
        done = bool(episode_done or data_done)

        return self._get_state(), float(reward), done, self.reward_info

    def _get_state(self):
        obs = self.env_data.get_observation(self.current_step).flatten()
        cur = self.env_data.get_current_observation(self.current_step)
        f = self.feature_map
        P_pv_avail = float(cur[f["P_pv"]]) if cur is not None else 0.0

        device_state = np.array([self.soc, self.T_in, P_pv_avail], dtype=np.float32)
        prev_action = self.action_prev.astype(np.float32)

        t = self.env_data.step_to_time(self.current_step)
        h, d = t.hour, t.weekday()
        time_feat = np.array([
            np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
            np.sin(2*np.pi*d/7),  np.cos(2*np.pi*d/7),
        ], dtype=np.float32)

        state = np.concatenate([obs, device_state, prev_action, time_feat]).astype(np.float32)
        return state
