
import os
import yaml
import numpy as np
from typing import Dict, Any


# ==========================================================
#                  CONFIGURATION UTILITIES
# ==========================================================

class Config:
    """
    Lightweight configuration object.
    Allows attribute-style access to dictionary entries.
    """
    def __init__(self, **entries):
        for key, value in entries.items():
            setattr(self, key, value)


def load_yaml_config(file_path: str) -> Config:
    """
    Load a YAML configuration file and return a Config object.

    Parameters
    ----------
    file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Configuration object with attribute access.
    """
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("YAML configuration must define a dictionary.")

    return Config(**data)


# ==========================================================
#                  FILESYSTEM UTILITIES
# ==========================================================

def ensure_directory(path: str) -> None:
    """
    Create directory if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path.
    """
    os.makedirs(path, exist_ok=True)


# ==========================================================
#                  STATE PREPROCESSING
# ==========================================================

def flatten_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten a structured HEMS state into a 1D vector.

    Expected state format:
    {
        "features": ndarray (...),
        "action_prev": ndarray (...)
    }

    Parameters
    ----------
    state : dict
        Structured environment state.

    Returns
    -------
    np.ndarray
        Flattened 1D state vector.
    """
    if not isinstance(state, dict):
        raise TypeError("State must be a dictionary.")

    flat_components = []
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            flat_components.append(value.flatten())
        else:
            raise TypeError(f"State element '{key}' must be a NumPy array.")

    return np.concatenate(flat_components)
