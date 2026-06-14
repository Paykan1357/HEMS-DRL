
"""
Utility functions for PyTorch-based DRL agents.

This module provides lightweight helpers for:
  - parameter synchronization
  - soft target network updates

Designed for continuous-control algorithms such as:
  SAC, TD3, ACKTR, and ensemble-based DRL methods.
"""

import torch


# ====================================================
#                  Global Settings
# ====================================================
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
FLOAT = torch.float32


# ====================================================
#              Parameter Copy Utilities
# ====================================================
def hard_update(target_net: torch.nn.Module, source_net: torch.nn.Module):
    """
    Copy parameters from source network to target network.

    Used for:
      - target network initialization
      - hard synchronization steps

    Parameters
    ----------
    target_net : torch.nn.Module
        Network to be updated.
    source_net : torch.nn.Module
        Network providing parameters.
    """
    for target_param, source_param in zip(
        target_net.parameters(), source_net.parameters()
    ):
        target_param.data.copy_(source_param.data)


def soft_update(
    target_net: torch.nn.Module,
    source_net: torch.nn.Module,
    tau: float,
):
    """
    Perform Polyak (soft) update of target network parameters.

    target ← (1 − τ)·target + τ·source

    Commonly used in:
      - SAC
      - TD3

    Parameters
    ----------
    target_net : torch.nn.Module
        Target network.
    source_net : torch.nn.Module
        Online (learned) network.
    tau : float
        Soft update coefficient (0 < tau ≤ 1).
    """
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must be in (0, 1].")

    with torch.no_grad():
        for target_param, source_param in zip(
            target_net.parameters(), source_net.parameters()
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)
