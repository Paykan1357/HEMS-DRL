
import numpy as np

EPS = 1e-8

# ==========================================================
#               SAFETY & FALLBACK CONTROLLERS
# ==========================================================

def safe_idle_action():
    """
    Safe fallback action:
    No charging, no discharging, no load shedding, no grid exchange.
    All actions set to zero.
    """
    return np.zeros(4, dtype=np.float32)


def grid_only_action(load_demand):
    """
    Baseline controller:
    Supply all load from the grid.
    Action: [bat, grid, pv, load]
    """
    action = np.zeros(4, dtype=np.float32)
    action[1] = load_demand  # grid power
    return action


# ==========================================================
#               CONSTRAINT & PENALTY UTILITIES
# ==========================================================

def battery_soc_violation(soc, soc_min, soc_max):
    """
    Computes SoC constraint violation penalty.
    """
    return max(0.0, soc - soc_max) + max(0.0, soc_min - soc)


def ramping_penalty(action, prev_action):
    """
    Penalize large changes between consecutive actions.
    """
    return np.sum(np.abs(action - prev_action))


def unmet_load(load_demand, supplied_power):
    """
    Compute unmet load (load shedding).
    supplied_power = grid_power + pv_used + battery_discharge
    """
    return max(0.0, load_demand - supplied_power)


# ==========================================================
#               ENERGY PERFORMANCE METRICS
# ==========================================================

def energy_cost(grid_power, price, delta_t):
    """
    Electricity cost from grid.
    """
    return price * grid_power * delta_t


def comfort_penalty(temp_in, temp_set, lambda_T, delta=0.5):
    """
    Quadratic thermal comfort penalty with dead-band.
    Matches the CVI evaluation metric.
    
    Parameters:
    - temp_in: indoor temperature (°C)
    - temp_set: setpoint temperature (°C)
    - lambda_T: comfort penalty weight
    - delta: dead-band tolerance (°C), default 0.5
    """
    deviation = abs(temp_in - temp_set) - delta
    if deviation > 0:
        return lambda_T * deviation ** 2
    return 0.0


def battery_degradation(p_bat, lambda_deg):
    """
    Linear battery degradation proxy.
    """
    return lambda_deg * abs(p_bat)


# ==========================================================
#               EPISODE-LEVEL METRICS
# ==========================================================

def compute_episode_metrics(log):
    """
    Compute evaluation metrics over one episode.
    """
    metrics = {}

    metrics["total_energy_cost"] = np.sum(log["energy_cost"])
    metrics["avg_comfort_violation"] = np.mean(log["comfort_penalty"])
    metrics["total_unmet_load"] = np.sum(log["unmet_load"])
    metrics["avg_soc"] = np.mean(log["soc"])
    metrics["pv_utilization"] = (
        np.sum(log["pv_used"]) / (np.sum(log["pv_available"]) + EPS)
    )
    metrics["cvi"] = np.sum(log["comfort_violation_raw"])  # Cumulative CVI

    return metrics


# ==========================================================
#               RULE-BASED BASELINE (OPTIONAL)
# ==========================================================

def rule_based_controller(price, pv_power, load, soc, soc_max):
    """
    Simple heuristic baseline:
    Action: [bat, grid, pv, load]
    - Use PV first
    - Charge battery if excess PV
    - Use grid only if necessary
    """
    action = np.zeros(4, dtype=np.float32)

    pv_used = min(pv_power, load)
    action[2] = pv_used  # pv component

    remaining_load = load - pv_used

    if pv_power > load and soc < soc_max:
        action[0] = min(pv_power - load, 1.0)  # charge battery
        action[1] = 0.0  # no grid needed
    else:
        action[0] = 0.0  # no battery discharge by default
        action[1] = remaining_load  # grid supply

    return action


# ==========================================================
#               ACTION DENORMALIZATION UTILITY
# ==========================================================

def denormalize_action(normalized_action, bounds):
    """
    Convert normalized action [-1, 1] to physical values.
    
    Parameters:
    - normalized_action: array of shape (4,) in [-1, 1]
    - bounds: dict with keys 'bat', 'grid', 'pv', 'load'
             each containing (min, max) tuple
    
    Returns:
    - physical_action: array of shape (4,)
    """
    physical_action = np.zeros(4, dtype=np.float32)
    
    # Action indices: 0=bat, 1=grid, 2=pv, 3=load
    action_names = ['bat', 'grid', 'pv', 'load']
    
    for i, name in enumerate(action_names):
        min_val, max_val = bounds[name]
        # Denormalize from [-1, 1] to [min, max]
        physical_action[i] = min_val + (normalized_action[i] + 1) / 2 * (max_val - min_val)
    
    return physical_action


def normalize_action(physical_action, bounds):
    """
    Convert physical action values to normalized [-1, 1].
    
    Parameters:
    - physical_action: array of shape (4,)
    - bounds: dict with keys 'bat', 'grid', 'pv', 'load'
    
    Returns:
    - normalized_action: array of shape (4,)
    """
    normalized_action = np.zeros(4, dtype=np.float32)
    
    action_names = ['bat', 'grid', 'pv', 'load']
    
    for i, name in enumerate(action_names):
        min_val, max_val = bounds[name]
        # Normalize from [min, max] to [-1, 1]
        normalized_action[i] = 2 * (physical_action[i] - min_val) / (max_val - min_val) - 1
    
    return normalized_action
