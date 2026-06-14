

import numpy as np
from pulp import *

class MPCBaseline:
    """
    Improved MPC baseline
    """
    
    def __init__(self, horizon=6):
        self.H = horizon
        self.E_max = 10.0
        self.P_bat_max = 6.0
        self.eta_ch = 0.95
        self.eta_dis = 0.95
        self.SoC_min = 1.0
        self.SoC_max = 9.0
        self.c_export = 0.08
        
    def solve(self, current_SoC, P_load_forecast, P_pv_forecast, 
              T_out_forecast, Q_occup_forecast, price_forecast,
              prev_P_ch, prev_P_dis, prev_Q_HVAC):
        """
        Solve MPC optimization - all 9 arguments required
        """
        # Create problem
        prob = LpProblem("MPC_Problem", LpMinimize)
        
        # Variables for each time step
        P_ch = []
        P_dis = []
        P_grid_buy = []
        P_grid_sell = []
        SoC = []
        z_ch = []
        z_dis = []
        
        for t in range(self.H):
            P_ch.append(LpVariable(f"P_ch_{t}", lowBound=0, upBound=self.P_bat_max))
            P_dis.append(LpVariable(f"P_dis_{t}", lowBound=0, upBound=self.P_bat_max))
            P_grid_buy.append(LpVariable(f"P_grid_buy_{t}", lowBound=0))
            P_grid_sell.append(LpVariable(f"P_grid_sell_{t}", lowBound=0))
            z_ch.append(LpVariable(f"z_ch_{t}", cat='Binary'))
            z_dis.append(LpVariable(f"z_dis_{t}", cat='Binary'))
        
        # SoC variables (H+1)
        for t in range(self.H + 1):
            SoC.append(LpVariable(f"SoC_{t}", lowBound=self.SoC_min, upBound=self.SoC_max))
        
        # Initial SoC constraint
        prob += SoC[0] == current_SoC
        
        # Constraints for each time step
        for t in range(self.H):
            # Power balance
            prob += (P_grid_buy[t] - P_grid_sell[t] + P_pv_forecast[t] + P_dis[t] == 
                     P_load_forecast[t] + P_ch[t])
            
            # Battery complementarity (cannot charge and discharge at same time)
            prob += P_ch[t] <= z_ch[t] * self.P_bat_max
            prob += P_dis[t] <= z_dis[t] * self.P_bat_max
            prob += z_ch[t] + z_dis[t] <= 1
            
            # Battery dynamics
            prob += SoC[t+1] == SoC[t] + (self.eta_ch * P_ch[t] - P_dis[t] / self.eta_dis) / self.E_max
        
        # Objective: minimize electricity cost
        objective = 0
        for t in range(self.H):
            objective += price_forecast[t] * P_grid_buy[t] - self.c_export * P_grid_sell[t]
            # Small penalty for battery use to encourage smoothness
            objective += 0.001 * (P_ch[t] + P_dis[t])
        
        prob += objective
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=10))
        
        # Return results
        if prob.status == 1:
            bat_action = P_dis[0].varValue - P_ch[0].varValue
            return bat_action, 0.0, 0.0
        else:
            # Fallback: simple heuristic
            return self._fallback_action(current_SoC, P_load_forecast[0], P_pv_forecast[0], price_forecast[0])
    
    def _fallback_action(self, SoC, P_load, P_pv, price):
        """Simple fallback heuristic"""
        net_load = P_load - P_pv
        
        if price > 0.15:  # Peak price
            if net_load > 0 and SoC > 1.0:
                return -min(net_load, self.P_bat_max), 0.0, 0.0
        else:  # Off-peak
            if net_load < 0 and SoC < 9.0:
                return min(-net_load, self.P_bat_max), 0.0, 0.0
        
        return 0.0, 0.0, 0.0
