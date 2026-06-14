"""
ACKTR (Actor-Critic) Agent for HEMS.

Proper on-policy implementation with:
- No replay buffer
- Generalized Advantage Estimation (GAE)
- Stable training with LayerNorm handling both 1D and 2D inputs
"""

"""
ACKTR (Actor-Critic) Agent for HEMS - CORRECTED VERSION
Fixes: Action clipping, proper denormalization tracking
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical limits for denormalization
BAT_MAX = 6.0        # kW
GRID_IMPORT_MAX = 10.0   # kW
GRID_EXPORT_MAX = 3.0    # kW


class GaussianActor(nn.Module):
    """Gaussian policy for continuous action spaces."""
    
    def __init__(self, state_dim, action_dim=4, hidden_dim=256, log_std_init=-0.5):
        super().__init__()
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)
    
    def forward(self, state):
        original_shape = state.shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        x = self.net(state)
        mean = torch.tanh(self.mean_layer(x))
        
        log_std = torch.clamp(self.log_std, -5, 1)
        std = torch.exp(log_std)
        
        if len(original_shape) == 1:
            mean = mean.squeeze(0)
        
        return mean, std
    
    def sample(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            if mean.dim() == 1:
                dist = Normal(mean, std)
                action = dist.rsample()
            else:
                dist = Normal(mean, std.unsqueeze(0).expand_as(mean))
                action = dist.rsample()
        
        # FIX 1: Clamp normalized action to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        return action
    
    def log_prob(self, state, action):
        mean, std = self.forward(state)
        
        if mean.dim() == 1:
            dist = Normal(mean, std)
        else:
            dist = Normal(mean, std.unsqueeze(0).expand_as(mean))
        
        log_prob = dist.log_prob(action)
        
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return log_prob
    
    def entropy(self, state):
        _, std = self.forward(state)
        
        if isinstance(std, torch.Tensor) and std.dim() == 0:
            std = std.unsqueeze(0)
        
        entropy = 0.5 * (np.log(2 * np.pi * np.e) + 2 * torch.log(std)).sum()
        
        if entropy.dim() > 0:
            entropy = entropy.mean()
        
        return entropy.unsqueeze(0)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
    
    def forward(self, state):
        original_shape = state.shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        value = self.net(state)
        
        if len(original_shape) == 1:
            value = value.squeeze(0).squeeze(-1)
        else:
            value = value.squeeze(-1)
        
        return value


def denormalize_action(normalized_action):
    """
    Convert normalized action [-1, 1] to physical units (kW).
    
    FIX 2: Proper denormalization with correct limits.
    """
    action = np.array(normalized_action).flatten()
    
    # Battery: [-BAT_MAX, BAT_MAX]
    bat_physical = action[0] * BAT_MAX
    
    # Grid: if positive -> import (up to GRID_IMPORT_MAX)
    #       if negative -> export (down to -GRID_EXPORT_MAX)
    if action[1] >= 0:
        grid_physical = action[1] * GRID_IMPORT_MAX
    else:
        grid_physical = action[1] * GRID_EXPORT_MAX
    
    # PV: [0, 1] (normalized already between -1 and 1)
    pv_physical = (action[2] + 1) / 2
    
    # Load: [0, 1]
    load_physical = (action[3] + 1) / 2
    
    # FIX 3: Clamp to physical limits (final safety)
    bat_physical = np.clip(bat_physical, -BAT_MAX, BAT_MAX)
    grid_physical = np.clip(grid_physical, -GRID_EXPORT_MAX, GRID_IMPORT_MAX)
    pv_physical = np.clip(pv_physical, 0, 1)
    load_physical = np.clip(load_physical, 0, 1)
    
    return np.array([bat_physical, grid_physical, pv_physical, load_physical], dtype=np.float32)


class ACKTRAgent:
    """
    Corrected ACKTR agent with proper action handling.
    """
    
    def __init__(self, name, env, seed, args):
        self.name = name
        self.env = env
        self.device = DEVICE
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.gamma = getattr(args, "gamma", 0.99)
        self.gae_lambda = getattr(args, "gae_lambda", 0.95)
        self.actor_lr = getattr(args, "lr_actor", 3e-4)
        self.critic_lr = getattr(args, "lr_critic", 3e-4)
        self.clip_grad = getattr(args, "clip_grad", 0.5)
        self.value_loss_coef = getattr(args, "value_loss_coef", 0.5)
        self.entropy_coef = getattr(args, "entropy_coef", 0.01)
        
        self.actor = GaussianActor(self.state_dim, self.action_dim).to(DEVICE)
        self.critic = ValueNetwork(self.state_dim).to(DEVICE)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        num_episodes = getattr(args, "num_episodes", 500)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=num_episodes)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=num_episodes)
        
        self.checkpoint_folder = getattr(args, "checkpoint_dir", f"./checkpoints/{name}")
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        
        self.total_steps = 0
        
        print(f"[ACKTR] Initialized with state_dim={self.state_dim}, action_dim={self.action_dim}")
        print(f"[ACKTR] Physical limits: Battery ±{BAT_MAX} kW, Grid [-{GRID_EXPORT_MAX}, {GRID_IMPORT_MAX}] kW")
    
    def select_action(self, state, deterministic=False, return_physical=True):
        """
        Select action.
        
        Args:
            state: observation
            deterministic: use deterministic policy
            return_physical: if True, return physical actions (kW); if False, return normalized [-1,1]
        """
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state, dict):
            state = state.get("prices", state)
        
        state = np.asarray(state, dtype=np.float32).flatten()
        
        if np.any(np.isnan(state)):
            print("[WARNING] NaN in state! Returning zero action.")
            return np.zeros(self.action_dim, dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        
        with torch.no_grad():
            normalized_action = self.actor.sample(state_tensor, deterministic=deterministic)
        
        normalized_action_np = normalized_action.cpu().numpy()
        
        if normalized_action_np.ndim == 0:
            normalized_action_np = np.array([normalized_action_np] * self.action_dim)
        
        if return_physical:
            # FIX 4: Convert to physical units
            physical_action = denormalize_action(normalized_action_np)
            return physical_action
        else:
            return normalized_action_np.astype(np.float32)
    
    def act(self, obs, exploration=True):
        """Alias for select_action."""
        return self.select_action(obs, deterministic=not exploration, return_physical=True)
    
    def select_action_normalized(self, state, deterministic=False):
        """Return normalized action (for training statistics)."""
        return self.select_action(state, deterministic=deterministic, return_physical=False)
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        returns = []
        
        gae = 0
        next_value_tensor = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
                next_value_tensor = 0
            else:
                delta = rewards[t] + self.gamma * next_value_tensor - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value_tensor = values[t]
        
        return advantages, returns
    
    def update(self, trajectory):
        states = np.array(trajectory['states'], dtype=np.float32)
        actions = np.array(trajectory['actions'], dtype=np.float32)
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        
        if np.any(np.isnan(states)) or np.any(np.isnan(actions)):
            print("[WARNING] NaN in trajectory! Skipping update.")
            return {"actor_loss": 0, "critic_loss": 0, "entropy": 0}
        
        states_tensor = torch.FloatTensor(states).to(DEVICE)
        actions_tensor = torch.FloatTensor(actions).to(DEVICE)
        
        with torch.no_grad():
            values = self.critic(states_tensor).cpu().numpy()
            
            if len(trajectory['next_states']) > 0:
                last_state = np.array(trajectory['next_states'][-1], dtype=np.float32)
                last_state_tensor = torch.FloatTensor(last_state).to(DEVICE)
                next_value = self.critic(last_state_tensor).cpu().numpy()
            else:
                next_value = 0
        
        values = np.array(values).flatten()
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(-1).to(DEVICE)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(-1).to(DEVICE)
        
        if advantages_tensor.std() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        else:
            advantages_tensor = advantages_tensor - advantages_tensor.mean()
        
        log_probs = self.actor.log_prob(states_tensor, actions_tensor)
        policy_loss = -(log_probs * advantages_tensor).mean()
        entropy = self.actor.entropy(states_tensor)
        entropy_loss = -self.entropy_coef * entropy
        actor_loss = policy_loss + entropy_loss
        
        if not torch.isnan(actor_loss):
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
            self.actor_optimizer.step()
        
        current_values = self.critic(states_tensor)
        critic_loss = self.value_loss_coef * F.mse_loss(current_values, returns_tensor.squeeze(-1))
        
        if not torch.isnan(critic_loss):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
            self.critic_optimizer.step()
        
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        self.total_steps += 1
        
        return {
            "actor_loss": actor_loss.item() if not torch.isnan(actor_loss) else 0,
            "critic_loss": critic_loss.item() if not torch.isnan(critic_loss) else 0,
            "entropy": entropy.item() if not torch.isnan(entropy) else 0,
            "policy_loss": policy_loss.item() if not torch.isnan(policy_loss) else 0
        }
    
    def save_models(self, episode=None):
        suffix = f"_ep{episode}" if episode is not None else ""
        torch.save(self.actor.state_dict(), os.path.join(self.checkpoint_folder, f"actor{suffix}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(self.checkpoint_folder, f"critic{suffix}.pth"))
    
    def load_models(self, episode=None):
        suffix = f"_ep{episode}" if episode is not None else ""
        actor_path = os.path.join(self.checkpoint_folder, f"actor{suffix}.pth")
        critic_path = os.path.join(self.checkpoint_folder, f"critic{suffix}.pth")
        
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))


