
"""
Gaussian Actor network for HEMS Deep Reinforcement Learning.

Implements a stochastic policy with Gaussian distribution for continuous
action spaces. Suitable for SAC and ACKTR algorithms.

Action space (4-dimensional):
    - a_t^bat: battery power (continuous, range: [-1, 1] normalized)
    - a_t^grid: grid power exchange (continuous, range: [-1, 1] normalized)
    - a_t^pv: PV utilization fraction (range: [0, 1])
    - a_t^load: controllable load multiplier (range: [0, 1])
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class GaussianActor(nn.Module):
    """
    Gaussian policy network for HEMS continuous control.
    
    Outputs mean and log_std for a diagonal Gaussian distribution,
    then applies tanh squashing to bound actions to [-1, 1].
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        action_limit: float = 1.0,
    ):
        """
        Args:
            state_dim: Dimension of flattened state vector
            action_dim: Dimension of action space (default: 4)
            hidden_dim: Number of hidden units per layer
            num_layers: Number of hidden layers
            action_limit: Maximum absolute action value (default: 1.0)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        
        # Build shared feature extractor
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Mean and log_std output layers
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Initialize mean layer with small weights for stable initial actions
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> tuple:
        """
        Forward pass to compute mean and log_std.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            
        Returns:
            mean: Mean of Gaussian distribution (batch, action_dim)
            log_std: Log standard deviation (batch, action_dim)
        """
        x = self.shared_layers(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False, with_logprob: bool = True) -> tuple:
        """
        Sample an action from the policy.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            deterministic: If True, use mean instead of sampling
            with_logprob: If True, compute log probability of the sampled action
            
        Returns:
            action: Sampled action tensor of shape (batch, action_dim) in [-1, 1]
            log_prob: Log probability of the action (if with_logprob=True)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Create distribution
        pi_distribution = Normal(mean, std)
        
        if deterministic:
            # Use mean for deterministic evaluation
            x = mean
            log_prob = None
        else:
            # Sample using reparameterization trick
            x = pi_distribution.rsample()
            
            # Compute log probability with tanh squashing correction
            if with_logprob:
                log_prob = pi_distribution.log_prob(x).sum(dim=-1, keepdim=True)
                # Apply tanh squashing correction
                log_prob = log_prob - torch.log(1 - torch.tanh(x).pow(2) + EPS).sum(dim=-1, keepdim=True)
            else:
                log_prob = None
        
        # Apply tanh squashing to bound actions to [-1, 1]
        action = torch.tanh(x)
        
        # Note: Further denormalization will be done by the environment
        # to map [-1, 1] to physical action ranges
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of a given action under the current policy.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            action: Action tensor of shape (batch, action_dim) in [-1, 1]
            
        Returns:
            log_prob: Log probability of the action (batch, 1)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Un-tanh the action for log probability computation
        x = torch.atanh(torch.clamp(action, -1 + EPS, 1 - EPS))
        
        # Compute log probability
        pi_distribution = Normal(mean, std)
        log_prob = pi_distribution.log_prob(x).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing correction
        log_prob = log_prob - torch.log(1 - action.pow(2) + EPS).sum(dim=-1, keepdim=True)
        
        return log_prob
    
    def requires_grad(self, req: bool):
        """Enable/disable gradient computation for all parameters."""
        for param in self.parameters():
            param.requires_grad = req
    
    def freeze(self):
        """Freeze all parameters (useful for target networks)."""
        self.requires_grad(False)
    
    def unfreeze(self):
        """Unfreeze all parameters."""
        self.requires_grad(True)
