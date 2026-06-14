"""
Core preprocessing utilities for HEMS state construction.
Purpose: transformations + standardization utilities
✔ Domain-agnostic
✔ Used by loader and controller

State definition (from paper):
    s_t = (X_{t-W+1:t}, d_t, a_{t-1}, tau_t)

Action space (4-dimensional):
    a_t = [a_t^bat, a_t^grid, a_t^pv, a_t^load]
"""

# ============================================================
# data_utils.py
#
# Data cleaning, normalization, and standardization utilities
# for Home Energy Management Systems (HEMS).
#
# Aligned with the state definition:
#   s_t = (X_{t-W+1:t}, d_t, a_{t-1}, tau_t)
#
# Suitable for SAC, TD3, ACKTR, and ensemble DRL agents.
# ============================================================

import numpy as np
import torch

# ============================================================
# Constants
# ============================================================
FLOAT = torch.float32
EPS = 1e-8

# Action dimension (removed idle action)
ACTION_DIM = 4


# ============================================================
#                 DATA CLEANING
# ============================================================
def clean_timeseries(data: np.ndarray):
    """
    Clean raw time-series data.

    Operations:
    - Replace NaN / Inf
    - Forward-fill then backward-fill
    - Clip extreme outliers (robust)

    Parameters
    ----------
    data : np.ndarray
        Shape (T, F)

    Returns
    -------
    cleaned : np.ndarray
    """
    data = data.copy()

    # Replace inf with nan
    data[~np.isfinite(data)] = np.nan

    # Forward fill
    for t in range(1, data.shape[0]):
        nan_mask = np.isnan(data[t])
        data[t, nan_mask] = data[t - 1, nan_mask]

    # Backward fill (if first rows had NaNs)
    for t in range(data.shape[0] - 2, -1, -1):
        nan_mask = np.isnan(data[t])
        data[t, nan_mask] = data[t + 1, nan_mask]

    # Robust clipping (1st–99th percentile)
    lower = np.percentile(data, 1, axis=0)
    upper = np.percentile(data, 99, axis=0)
    data = np.clip(data, lower, upper)

    return data


# ============================================================
#         TRAINING STATISTICS (STANDARDIZATION)
# ============================================================
def compute_standardization_stats(data: np.ndarray):
    """
    Compute mean and std from TRAIN data only.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, F)

    Returns
    -------
    stats : dict
        {"mean": (F,), "std": (F,)}
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    std = np.maximum(std, EPS)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def standardize(data: np.ndarray, stats: dict):
    """
    Apply standardization using precomputed TRAIN stats.

    Parameters
    ----------
    data : np.ndarray
        Shape (..., F)
    stats : dict

    Returns
    -------
    standardized : np.ndarray
    """
    return (data - stats["mean"]) / (stats["std"] + EPS)


# ============================================================
#        TIME-SERIES WINDOW NORMALIZATION (X_t)
# ============================================================
def normalize_window_last_step(window: np.ndarray):
    """
    Normalize time-series window by its last step.

    Used for:
      X_{t-W+1:t}

    Parameters
    ----------
    window : np.ndarray
        Shape (W, F)

    Returns
    -------
    normalized : np.ndarray
    """
    reference = window[-1] + EPS
    return window / reference


def normalize_window_minmax(window: np.ndarray):
    """
    Min-max normalization across time window.

    Parameters
    ----------
    window : np.ndarray
        Shape (W, F)

    Returns
    -------
    normalized : np.ndarray
    """
    min_v = window.min(axis=0, keepdims=True)
    max_v = window.max(axis=0, keepdims=True)
    return (window - min_v) / (max_v - min_v + EPS)


# ============================================================
#            DEVICE STATE NORMALIZATION (d_t)
# ============================================================
def normalize_device_state(
    soc,
    p_bat_max,
    p_bat_min,
    p_pv,
    p_pv_max,
):
    """
    Normalize device-related states.

    Parameters
    ----------
    soc : float
        State of charge in [0, 1]
    p_bat_max : float
        Maximum battery discharge power (kW)
    p_bat_min : float
        Minimum battery charge power (negative, kW)
    p_pv : float
        Current available PV power (kW)
    p_pv_max : float
        Maximum possible PV power (kW)

    Returns
    -------
    d_t : np.ndarray
        Normalized device state (shape: 4)
    """
    return np.array(
        [
            np.clip(soc, 0.0, 1.0),                 # already normalized
            p_bat_max / (p_bat_max + EPS),          # -> ~1
            p_bat_min / (abs(p_bat_min) + EPS),     # -> -1
            p_pv / (p_pv_max + EPS),                # [0,1]
        ],
        dtype=np.float32,
    )


# ============================================================
#          PREVIOUS ACTION NORMALIZATION (a_{t-1})
# ============================================================
def normalize_action(action: np.ndarray, action_low, action_high):
    """
    Normalize previous action to [-1, 1].

    Parameters
    ----------
    action : np.ndarray
        Physical action values (shape: ACTION_DIM)
    action_low : np.ndarray
        Lower bounds for each action dimension
    action_high : np.ndarray
        Upper bounds for each action dimension

    Returns
    -------
    normalized_action : np.ndarray
        Normalized action in [-1, 1] (shape: ACTION_DIM)
    """
    return 2.0 * (action - action_low) / (action_high - action_low + EPS) - 1.0


def denormalize_action(normalized_action: np.ndarray, action_low, action_high):
    """
    Denormalize action from [-1, 1] to physical values.

    Parameters
    ----------
    normalized_action : np.ndarray
        Normalized action in [-1, 1] (shape: ACTION_DIM)
    action_low : np.ndarray
        Lower bounds for each action dimension
    action_high : np.ndarray
        Upper bounds for each action dimension

    Returns
    -------
    physical_action : np.ndarray
        Physical action values
    """
    return action_low + (normalized_action + 1.0) / 2.0 * (action_high - action_low)


# ============================================================
#          COMFORT PENALTY (Dead-band Quadratic)
# ============================================================
def comfort_penalty_deadband(
    temp_in: float,
    temp_set: float,
    lambda_T: float,
    dead_band: float = 0.5,
):
    """
    Quadratic dead-band comfort penalty.
    Matches CVI evaluation metric defined in the paper.

    U_t = lambda_T * max(0, |T_in - T_set| - dead_band)^2

    Parameters
    ----------
    temp_in : float
        Indoor temperature (°C)
    temp_set : float
        Setpoint temperature (°C)
    lambda_T : float
        Comfort penalty weight
    dead_band : float
        Tolerance dead-band (°C), default 0.5

    Returns
    -------
    penalty : float
        Comfort penalty value
    """
    deviation = abs(temp_in - temp_set) - dead_band
    if deviation > 0:
        return lambda_T * (deviation ** 2)
    return 0.0


# ============================================================
#               TIME ENCODING (tau_t)
# ============================================================
def encode_time(hour: int, day: int):
    """
    Cyclic time encoding.

    Parameters
    ----------
    hour : int (0–23)
    day : int (0–6)

    Returns
    -------
    tau_t : np.ndarray (4,)
    """
    return np.array(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * day / 7.0),
            np.cos(2 * np.pi * day / 7.0),
        ],
        dtype=np.float32,
    )


# ============================================================
#          FINAL STATE ASSEMBLY (OPTIONAL)
# ============================================================
def build_state(
    window_X,
    device_state,
    prev_action,
    time_encoding,
):
    """
    Assemble full state vector s_t.

    Parameters
    ----------
    window_X : np.ndarray
        Shape (W, F)
    device_state : np.ndarray
        Shape (4,)
    prev_action : np.ndarray
        Shape (ACTION_DIM,)
    time_encoding : np.ndarray
        Shape (4,)

    Returns
    -------
    state : np.ndarray
        Flat state vector
    """
    return np.concatenate(
        [
            window_X.flatten(),
            device_state,
            prev_action,
            time_encoding,
        ]
    ).astype(np.float32)


# ============================================================
#        Time-Series Transformations
# ============================================================
def compute_differences(data: np.ndarray):
    """
    Compute first-order temporal differences.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, F)

    Returns
    -------
    diffs : np.ndarray
        Shape (T, F)
    """
    diffs = np.diff(data, axis=0)
    pad = np.zeros((1, data.shape[1]), dtype=data.dtype)
    return np.vstack([pad, diffs])


# ============================================================
#        Tensor Reshaping (for Neural Networks)
# ============================================================
def to_rnn_format(window: np.ndarray):
    """
    RNN-compatible format.

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, features)

    Returns
    -------
    Same shape (window_length, features)
    """
    return window


def to_cnn_format(window: np.ndarray):
    """
    CNN-compatible format.

    From: (window_length, features)
    To:   (features, 1, window_length)

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, features)

    Returns
    -------
    cnn_input : np.ndarray
        Shape (features, 1, window_length)
    """
    return np.transpose(window, (1, 0))[..., None]


def to_cnn_rnn_format(window: np.ndarray):
    """
    Hybrid CNN-RNN format.

    From: (window_length, features)
    To:   (features, window_length)

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length, features)

    Returns
    -------
    hybrid_input : np.ndarray
        Shape (features, window_length)
    """
    return np.transpose(window, (1, 0))
