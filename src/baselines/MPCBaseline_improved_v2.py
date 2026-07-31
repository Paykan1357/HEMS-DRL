
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPC Baseline with Full Thermal-HVAC Control
-------------------------------------------
This version replaces the battery-only MPC with a full MILP that optimises
both battery scheduling and HVAC thermal power over a 12-hour horizon.
It uses a 1R1C RC network for building thermal dynamics and enforces
soft comfort constraints with large penalties.

Usage:
    from MPCBaseline_improved_v2 import MPCBaselineImprovedV2
    mpc = MPCBaselineImprovedV2(horizon=12)
    bat_action, hvac_action, slack = mpc.solve(
        current_SoC, current_T_in,
        load_forecast, pv_forecast,
        temp_forecast, internal_gains_forecast,
        price_forecast,
        prev_P_ch=0.0, prev_P_dis=0.0, prev_Q_HVAC=0.0
    )
"""

import numpy as np
from pulp import *
import warnings
warnings.filterwarnings("ignore")


class MPCBaselineImprovedV2:
    """
    MPC with battery and HVAC control using a 1R1C thermal model.
    Solves a mixed-integer linear program (MILP) with binary variables
    for battery charge/discharge and HVAC ON/OFF status.
    """

    def __init__(self, horizon=12):
        self.H = horizon

        # ---------- Battery parameters ----------
        self.E_max = 10.0          # kWh
        self.P_bat_max = 6.0       # kW
        self.eta_ch = 0.95
        self.eta_dis = 0.95
        self.SoC_min = 0.05 * self.E_max   # 0.5 kWh
        self.SoC_max = 0.95 * self.E_max   # 9.5 kWh

        # ---------- HVAC / thermal parameters ----------
        # These must match the values used in the DRL environment.
        self.R = 5.0               # °C/kW   thermal resistance
        self.C = 4.0               # kWh/°C  thermal capacitance
        self.Q_max = 10.0          # kW      max thermal power
        self.COP = 3.0             # coefficient of performance
        self.T_min = 20.0          # °C      lower comfort bound
        self.T_max = 26.0          # °C      upper comfort bound
        self.lambda_slack = 1e3    # penalty for comfort violations

        # ---------- Grid / economic ----------
        self.c_export = 0.08       # $/kWh   feed‑in tariff

        # ---------- Forecast improvement (optional) ----------
        self.pv_history = []
        self.load_history = []
        self.temp_history = []

    def solve(self, current_SoC, current_T_in,
              P_load_forecast, P_pv_forecast,
              T_out_forecast, Q_internal_forecast,
              price_forecast,
              prev_P_ch=0.0, prev_P_dis=0.0, prev_Q_HVAC=0.0):
        """
        Solve the MPC optimisation problem.

        Parameters
        ----------
        current_SoC : float
            Current battery state of charge (kWh)
        current_T_in : float
            Current indoor temperature (°C)
        P_load_forecast : list of float OR float
            Forecasted non‑HVAC load (kW) for next H hours.
            If a scalar is given, it is repeated H times.
        P_pv_forecast : list of float OR float
            Forecasted PV generation (kW)
        T_out_forecast : list of float OR float
            Forecasted outdoor temperature (°C)
        Q_internal_forecast : list of float OR float
            Forecasted internal heat gains (kW) from occupants + solar
        price_forecast : list of float OR float
            Electricity prices ($/kWh)
        prev_P_ch, prev_P_dis, prev_Q_HVAC : float
            Previous actions (used for smoothness, optional)

        Returns
        -------
        bat_action : float
            Battery power (kW), positive = discharge, negative = charge
        hvac_action : float
            HVAC thermal power (kW), positive = cooling
        slack_total : float
            Total slack used (comfort violations) – for monitoring
        """
        H = self.H

        # ---------- Helper to convert scalar to list and pad ----------
        def to_list_and_pad(value, length):
            """Convert scalar to list of length `length`, or pad existing list."""
            if not isinstance(value, (list, tuple, np.ndarray)):
                # Scalar: repeat it `length` times
                return [value] * length
            # It's already a list-like object
            if len(value) < length:
                # Pad with last value
                return list(value) + [value[-1]] * (length - len(value))
            return list(value[:length])

        # Convert all forecast inputs to lists of length H
        P_load_forecast = to_list_and_pad(P_load_forecast, H)
        P_pv_forecast   = to_list_and_pad(P_pv_forecast, H)
        T_out_forecast  = to_list_and_pad(T_out_forecast, H)
        Q_internal_forecast = to_list_and_pad(Q_internal_forecast, H)
        price_forecast  = to_list_and_pad(price_forecast, H)

        # ---------- Optional forecast improvement ----------
        self.load_history.append(P_load_forecast[0])
        self.pv_history.append(P_pv_forecast[0])
        self.temp_history.append(T_out_forecast[0])
        if len(self.load_history) > 48:
            self.load_history.pop(0)
            self.pv_history.pop(0)
            self.temp_history.pop(0)

        improved_load = self._improve_forecast(P_load_forecast, self.load_history)
        improved_pv   = self._improve_forecast(P_pv_forecast, self.pv_history)

        # ---------- Build the MILP model ----------
        prob = LpProblem("MPC_HVAC", LpMinimize)

        # ----- Decision variables -----
        # Battery
        P_ch = [LpVariable(f"ch_{t}", lowBound=0, upBound=self.P_bat_max) for t in range(H)]
        P_dis = [LpVariable(f"dis_{t}", lowBound=0, upBound=self.P_bat_max) for t in range(H)]
        SoC = [LpVariable(f"soc_{t}", lowBound=self.SoC_min, upBound=self.SoC_max) for t in range(H+1)]
        z_ch = [LpVariable(f"zch_{t}", cat='Binary') for t in range(H)]
        z_dis = [LpVariable(f"zdis_{t}", cat='Binary') for t in range(H)]

        # Grid
        P_import = [LpVariable(f"imp_{t}", lowBound=0) for t in range(H)]
        P_export = [LpVariable(f"exp_{t}", lowBound=0) for t in range(H)]

        # HVAC / thermal
        Q_hvac = [LpVariable(f"qhv_{t}", lowBound=0, upBound=self.Q_max) for t in range(H)]
        z_on = [LpVariable(f"zon_{t}", cat='Binary') for t in range(H)]
        P_hvac_elec = [LpVariable(f"phe_{t}", lowBound=0) for t in range(H)]

        # Indoor temperature (H+1 so we have T_in[H] for the final step)
        T_in = [LpVariable(f"tin_{t}", lowBound=0, upBound=40) for t in range(H+1)]

        # Slack variables for comfort (soft constraints)
        slack_cool = [LpVariable(f"slack_cool_{t}", lowBound=0) for t in range(H)]
        slack_heat = [LpVariable(f"slack_heat_{t}", lowBound=0) for t in range(H)]

        # HVAC smoothness (linearisation of absolute differences)
        delta_pos = [LpVariable(f"delta_pos_{t}", lowBound=0) for t in range(1, H)]
        delta_neg = [LpVariable(f"delta_neg_{t}", lowBound=0) for t in range(1, H)]

        # ----- Initial conditions -----
        prob += SoC[0] == current_SoC
        prob += T_in[0] == current_T_in

        # ----- Constraints for each time step -----
        for t in range(H):
            # Power balance
            prob += (P_import[t] - P_export[t] + improved_pv[t] + P_dis[t] ==
                     improved_load[t] + P_ch[t] + P_hvac_elec[t])

            # Battery complementarity
            prob += P_ch[t] <= z_ch[t] * self.P_bat_max
            prob += P_dis[t] <= z_dis[t] * self.P_bat_max
            prob += z_ch[t] + z_dis[t] <= 1

            # Battery dynamics
            prob += SoC[t+1] == SoC[t] + (self.eta_ch * P_ch[t] - P_dis[t] / self.eta_dis) / self.E_max

            # HVAC electrical power
            prob += P_hvac_elec[t] == Q_hvac[t] / self.COP

            # HVAC ON/OFF coupling
            prob += Q_hvac[t] <= self.Q_max * z_on[t]

            # Thermal dynamics (1R1C, forward Euler)
            dt = 1.0  # hourly
            prob += T_in[t+1] == T_in[t] + (dt / self.C) * (
                        (T_out_forecast[t] - T_in[t]) / self.R
                        + Q_hvac[t]
                        + Q_internal_forecast[t]
                    )

            # Soft comfort constraints
            prob += T_in[t] >= self.T_min - slack_cool[t]
            prob += T_in[t] <= self.T_max + slack_heat[t]

        # ----- Constraints for HVAC smoothness (linearised) -----
        for t in range(1, H):
            prob += Q_hvac[t] - Q_hvac[t-1] == delta_pos[t-1] - delta_neg[t-1]

        # ----- Objective -----
        obj = 0
        for t in range(H):
            # Electricity cost
            obj += price_forecast[t] * P_import[t] - self.c_export * P_export[t]
            # Battery degradation (small penalty to encourage smooth operation)
            obj += 0.0005 * (P_ch[t] + P_dis[t])
            # Comfort slack penalties (large weight)
            obj += self.lambda_slack * (slack_cool[t] + slack_heat[t])

        # HVAC smoothness penalty (linear absolute difference)
        for t in range(1, H):
            obj += 0.001 * (delta_pos[t-1] + delta_neg[t-1])

        prob += obj

        # ----- Solve -----
        solver = PULP_CBC_CMD(msg=False, timeLimit=60)
        prob.solve(solver)

        # ----- Extract results -----
        if prob.status == 1:  # optimal solution found
            bat_action = P_dis[0].varValue - P_ch[0].varValue
            hvac_action = Q_hvac[0].varValue
            slack_total = sum(slack_cool[t].varValue + slack_heat[t].varValue for t in range(H))
            return bat_action, hvac_action, slack_total
        else:
            # Fallback heuristic if solver fails
            return self._smart_fallback(current_SoC, current_T_in,
                                        P_load_forecast[0], P_pv_forecast[0],
                                        T_out_forecast[0], Q_internal_forecast[0],
                                        price_forecast[0])

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _improve_forecast(self, forecast, history):
        """Improve forecast using historical pattern (optional)."""
        improved = list(forecast)
        if len(history) >= 24:
            history_array = np.array(history[-48:])
            if len(history_array) >= 24:
                avg_pattern = []
                for hour in range(24):
                    indices = [i for i in range(len(history_array)) if i % 24 == hour]
                    if indices:
                        avg_pattern.append(np.mean(history_array[indices]))
                    else:
                        avg_pattern.append(1.0)
                for t in range(len(improved)):
                    pattern_value = avg_pattern[t % 24]
                    if pattern_value > 0:
                        scale = np.clip(improved[t] / pattern_value, 0.5, 2.0)
                        improved[t] = 0.7 * improved[t] + 0.3 * pattern_value * scale
        return improved

    def _smart_fallback(self, SoC, T_in, P_load, P_pv, T_out, Q_internal, price):
        """Simple heuristic fallback when MILP solver fails."""
        net = P_load - P_pv
        bat_action = 0.0
        hvac_action = 0.0

        # Battery heuristic
        if price > 0.20 and net > 0 and SoC > self.SoC_min + 0.5:
            bat_action = -min(net, self.P_bat_max)   # discharge
        elif price < 0.12 and net < 0 and SoC < self.SoC_max - 0.5:
            bat_action = min(-net, self.P_bat_max)   # charge

        # HVAC heuristic: simple proportional control around 22°C
        T_set = 22.0
        if T_out > T_set + 2.0:
            hvac_action = min(self.Q_max, max(0, (T_out - T_set) * 0.5))
        elif T_out < T_set - 2.0:
            hvac_action = min(self.Q_max, max(0, (T_set - T_out) * 0.5))
        else:
            hvac_action = 0.0

        return bat_action, hvac_action, 0.0
