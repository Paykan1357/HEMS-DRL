
class RuleBasedBaseline:
    def __init__(self, battery_params, hvac_params):
        self.E_max = 10.0
        self.P_bat_max = 3.0
        self.SoC_min = 0.1
        self.SoC_max = 0.9
        self.T_set = 22.0
        self.delta = 0.5
        
    def get_action(self, SoC, P_load, P_pv, T_in, occupancy):
        """
        Rule-based controller - completely deterministic.
        """
        net_load = P_load - P_pv
        SoC_kWh = SoC * self.E_max
        
        # BATTERY RULES
        if net_load > 0:  # Deficit
            if SoC_kWh > self.SoC_min * self.E_max:
                # Discharge to cover deficit, limited by battery power
                discharge = min(net_load, self.P_bat_max, 
                               (SoC_kWh - self.SoC_min*self.E_max))
                bat_action = -discharge  # negative = discharge
            else:
                bat_action = 0.0
        else:  # Excess PV
            excess = -net_load
            if SoC_kWh < self.SoC_max * self.E_max:
                # Charge with excess, limited by battery power and capacity
                charge = min(excess, self.P_bat_max,
                           (self.SoC_max*self.E_max - SoC_kWh))
                bat_action = charge  # positive = charge
            else:
                bat_action = 0.0
        
        # HVAC RULES (simple dead-band)
        if T_in < self.T_set - self.delta:
            hvac_action = 2.0  # kW heating
        elif T_in > self.T_set + self.delta:
            hvac_action = -2.0  # kW cooling
        else:
            hvac_action = 0.0
        
        # PV utilization rule: use all available PV if possible
        pv_used = min(P_pv, P_load + abs(bat_action) if bat_action > 0 else P_load)
        
        return bat_action, hvac_action, pv_used
