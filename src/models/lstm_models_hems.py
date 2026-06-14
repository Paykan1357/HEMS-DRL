
"""
LSTM-based Actor–Critic models for Home Energy Management Systems (HEMS)

State:
s_t = (X_{t-W+1:t}, d_t, a_{t-1}, tau_t)

Action:
a_t = (a_bat, a_grid, a_pv, a_load) ∈ R^4

Compatible with:
- SAC
- TD3
- ACKTR
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from base_models import BaseModel
from data_utils import EPS


# =====================================================
#               LSTM Time-Series Encoder
# =====================================================
class HEMSLSTMEncoder(nn.Module):
    """
    Encodes historical time-series:
    X_{t-W+1:t} ∈ R^{W × F}
    """

    def __init__(self, num_features, window_length, d_model, num_layers):
        super().__init__()

        self.input_fc = nn.Linear(num_features, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        """
        x : [batch, W, F]
        """
        x = self.activation(self.input_fc(x))
        x, _ = self.lstm(x)
        return x[:, -1, :]   # last time step embedding


# =====================================================
#              Deterministic Actor (TD3)
# =====================================================
class DeterministicHEMSActor(BaseModel):
    """
    Deterministic policy for TD3.
    Outputs actions in [-1, 1]^4.
    """
    
    def __init__(
        self,
        num_features,
        window_length,
        device_dim,
        action_dim=4,
        time_dim=4,
        d_model=128,
        num_layers=3,
    ):
        super().__init__()
        
        self.encoder = HEMSLSTMEncoder(
            num_features, window_length, d_model, num_layers
        )
        
        # Input: state embedding + device state + prev action + time features
        input_dim = d_model + device_dim + action_dim + time_dim
        
        self.policy = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh(),  # All actions normalized to [-1, 1]
        )
        
        self.action_dim = action_dim

    def forward(self, X, d_t, a_prev, tau_t):
        """
        Args:
            X: Historical window [batch, W, F]
            d_t: Device state [batch, device_dim]
            a_prev: Previous action [batch, action_dim]
            tau_t: Time features [batch, time_dim]
            
        Returns:
            actions: Normalized actions in [-1, 1], shape [batch, action_dim]
        """
        z = self.encoder(X)
        z = torch.cat([z, d_t, a_prev, tau_t], dim=-1)
        return self.policy(z)


# =====================================================
#               Gaussian Actor (SAC)
# =====================================================
class GaussianHEMSActor(BaseModel):
    """
    Stochastic policy for SAC.
    Outputs actions in [-1, 1]^4 with log probability.
    """
    
    def __init__(
        self,
        num_features,
        window_length,
        device_dim,
        action_dim=4,
        time_dim=4,
        d_model=128,
        num_layers=3,
    ):
        super().__init__()
        
        self.encoder = HEMSLSTMEncoder(
            num_features, window_length, d_model, num_layers
        )
        
        input_dim = d_model + device_dim + action_dim + time_dim
        
        self.fc_mu = nn.Linear(input_dim, action_dim)
        self.fc_logstd = nn.Linear(input_dim, action_dim)
        
        self.action_dim = action_dim

    def forward(self, X, d_t, a_prev, tau_t, sample=True):
        """
        Returns:
        - action ∈ [-1,1]^4
        - log_prob (for SAC)
        """
        z = self.encoder(X)
        z = torch.cat([z, d_t, a_prev, tau_t], dim=-1)

        mu = self.fc_mu(z)
        log_std = torch.clamp(self.fc_logstd(z), -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mu, std)

        x = dist.rsample() if sample else mu
        action = torch.tanh(x)

        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


# =====================================================
#                     Critic (Q)
# =====================================================
class HEMSCritic(BaseModel):
    """
    Q(s,a) critic network.
    """
    
    def __init__(
        self,
        num_features,
        window_length,
        device_dim,
        action_dim=4,
        time_dim=4,
        d_model=128,
        num_layers=3,
    ):
        super().__init__()
        
        self.encoder = HEMSLSTMEncoder(
            num_features, window_length, d_model, num_layers
        )
        
        # Input: state embedding + device state + prev action + time features + current action
        input_dim = d_model + device_dim + action_dim + time_dim + action_dim
        
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, X, d_t, a_prev, tau_t, action):
        """
        Args:
            X: Historical window [batch, W, F]
            d_t: Device state [batch, device_dim]
            a_prev: Previous action [batch, action_dim]
            tau_t: Time features [batch, time_dim]
            action: Current action [batch, action_dim]
            
        Returns:
            q_value: Q-value estimate [batch]
        """
        z = self.encoder(X)
        z = torch.cat([z, d_t, a_prev, tau_t, action], dim=-1)
        return self.q_net(z).squeeze(-1)


# =====================================================
#                Double Critic (TD3/SAC)
# =====================================================
class DoubleHEMSCritic(BaseModel):
    """
    Double Q-network to reduce overestimation bias.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.Q1 = HEMSCritic(*args, **kwargs)
        self.Q2 = HEMSCritic(*args, **kwargs)

    def forward(self, X, d_t, a_prev, tau_t, action):
        """
        Returns both Q-values.
        """
        return (
            self.Q1(X, d_t, a_prev, tau_t, action),
            self.Q2(X, d_t, a_prev, tau_t, action),
        )
    
    def Q1_only(self, X, d_t, a_prev, tau_t, action):
        """
        Return only Q1 (for actor update in TD3).
        """
        return self.Q1(X, d_t, a_prev, tau_t, action)
