
import numpy as np

EPS = 1e-8

# ==========================================================
#               SAFETY & FALLBACK CONTROLLERS
# ==========================================================

def safe_idle_action():
    """
    Safe fallback action:
    No charging, no discharging, no load shedding.
    """
    return np.zeros(5, dtype=np.float32)


def grid_only_action(load_demand):
    """
    Baseline controller:
    Supply all load from the grid.
    """
    action = np.zeros(5, dtype=np.float32)
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


def comfort_penalty(temp_in, temp_set, lambda_T):
    """
    Quadratic thermal comfort penalty.
    """
    return lambda_T * (temp_in - temp_set) ** 2


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

    return metrics


# ==========================================================
#               RULE-BASED BASELINE (OPTIONAL)
# ==========================================================

def rule_based_controller(price, pv_power, load, soc, soc_max):
    """
    Simple heuristic baseline:
    - Use PV first
    - Charge battery if excess PV
    - Use grid only if necessary
    """
    action = np.zeros(5, dtype=np.float32)

    pv_used = min(pv_power, load)
    action[2] = pv_used

    remaining_load = load - pv_used

    if pv_power > load and soc < soc_max:
        action[0] = min(pv_power - load, 1.0)  # charge battery
    else:
        action[1] = remaining_load  # grid supply

    return action
