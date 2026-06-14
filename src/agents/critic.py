
"""
Critic networks for HEMS Deep Reinforcement Learning.

Provides Q-function approximators for continuous control algorithms
such as TD3, SAC, and ACKTR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Critic network (Q-function) for HEMS.
    
    Takes state and action as input, outputs Q-value estimate.
    """
    
    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dim: int = 256, num_layers: int = 2):
        """
        Args:
            state_dim: Dimension of flattened state vector
            action_dim: Dimension of action space (default: 4)
            hidden_dim: Number of hidden units per layer
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        
        # Input layer: state + action
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (single Q-value)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-value.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            action: Action tensor of shape (batch, action_dim)
            
        Returns:
            Q-value tensor of shape (batch, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
    def requires_grad(self, req: bool):
        """Enable/disable gradient computation."""
        for param in self.parameters():
            param.requires_grad = req


class DoubleCritic(nn.Module):
    """
    Double Critic network for algorithms like SAC and TD3.
    
    Maintains two separate Q-networks to reduce overestimation bias.
    """
    
    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dim: int = 256, num_layers: int = 2):
        """
        Args:
            state_dim: Dimension of flattened state vector
            action_dim: Dimension of action space (default: 4)
            hidden_dim: Number of hidden units per layer
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.Q1 = Critic(state_dim, action_dim, hidden_dim, num_layers)
        self.Q2 = Critic(state_dim, action_dim, hidden_dim, num_layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Forward pass through both Q-networks.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            action: Action tensor of shape (batch, action_dim)
            
        Returns:
            Tuple of (Q1, Q2) values, each of shape (batch, 1)
        """
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2
    
    def Q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through only Q1 (for actor update in TD3/SAC).
        
        Args:
            state: State tensor of shape (batch, state_dim)
            action: Action tensor of shape (batch, action_dim)
            
        Returns:
            Q1 value tensor of shape (batch, 1)
        """
        return self.Q1(state, action)
    
    def requires_grad(self, req: bool):
        """Enable/disable gradient computation for both networks."""
        self.Q1.requires_grad(req)
        self.Q2.requires_grad(req)
