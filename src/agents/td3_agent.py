"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent for HEMS.

Fixed version with:
- Tanh output for continuous actions (bat, grid, pv, load)
- No softmax (actions don't need to sum to 1)
- Gradient clipping
- NaN protection
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Utility
# ----------------------------
def ensure_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# ----------------------------
# Actor Network (HEMS-specific)
# ----------------------------
class Actor(nn.Module):
    """
    Deterministic actor for HEMS.
    Outputs actions in [-1, 1] for all 4 dimensions.
    No softmax – actions are independent continuous values.
    """
    def __init__(self, state_dim, action_dim=4, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Initialize weights for stability
        self.apply(self._init_weights)
        # Small final layer weights
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Critic Network (HEMS-specific)
# ----------------------------
class Critic(nn.Module):
    """
    Twin Q-network for TD3.
    """
    def __init__(self, state_dim, action_dim=4, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # Check for NaN before adding
        if np.any(np.isnan(state)) or np.any(np.isnan(action)):
            return
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )

    def size(self):
        return len(self.buffer)
    
    def __len__(self):
        return len(self.buffer)


# ----------------------------
# TD3 Agent
# ----------------------------
class TD3Agent:
    def __init__(self, name, env, seed, args, state_dim, action_dim):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim  # Should be 4
        self.env = env
        self.device = DEVICE

        # Hyperparameters
        self.gamma = getattr(args, "gamma", 0.99)
        self.tau = getattr(args, "tau", 0.005)
        self.policy_noise = getattr(args, "policy_noise", 0.2)
        self.noise_clip = getattr(args, "noise_clip", 0.5)
        self.policy_delay = getattr(args, "policy_delay", 2)
        self.batch_size = getattr(args, "batch_size", 256)
        self.max_action = 1.0

        # Networks
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target_2 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=getattr(args, "lr_actor", 1e-4))
        self.critic_opt = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=getattr(args, "lr_critic", 1e-4)  # Lower LR for stability
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=getattr(args, "buffer_size", 1000000))
        self.total_it = 0
        
        # Checkpoint dir
        self.checkpoint_dir = getattr(args, "checkpoint_dir", "./checkpoints_TD3")
        ensure_dir(self.checkpoint_dir)
        
        print(f"[TD3] Initialized with state_dim={state_dim}, action_dim={action_dim}")

    # ----------------------------
    # Select action (for environment)
    # ----------------------------
    def select_action(self, state, deterministic=False):
        """Select action for HEMS."""
        # Handle different state formats
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state, dict):
            state = state.get("prices", state)
        
        # Convert to numpy and flatten
        state = np.asarray(state, dtype=np.float32).flatten()
        
        # Check for NaN in state
        if np.any(np.isnan(state)):
            print("[WARNING] NaN in state! Returning zero action.")
            return np.zeros(self.action_dim, dtype=np.float32)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        # Add exploration noise
        if not deterministic:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        # Final NaN check
        if np.any(np.isnan(action)):
            print("[WARNING] NaN in action! Returning zero action.")
            return np.zeros(self.action_dim, dtype=np.float32)
        
        return action.astype(np.float32)
    
    # ----------------------------
    # Predict action (alias)
    # ----------------------------
    def predict_action(self, state_vec, exploration=True):
        """Alias for select_action."""
        return self.select_action(state_vec, deterministic=not exploration)

    # ----------------------------
    # TD3 update
    # ----------------------------
    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        self.total_it += 1
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        # ---------- Critic update ----------
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # Target Q-values
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_value = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)
        critic_loss = nn.MSELoss()(current_Q1, target_value) + nn.MSELoss()(current_Q2, target_value)

        # Check for NaN loss
        if torch.isnan(critic_loss):
            print("[WARNING] Critic loss is NaN! Skipping update.")
            return

        self.critic_opt.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_opt.step()

        # ---------- Delayed actor update ----------
        if self.total_it % self.policy_delay == 0:
            actor_actions = self.actor(states)
            actor_loss = -self.critic_1(states, actor_actions).mean()
            
            if not torch.isnan(actor_loss):
                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()

                # Soft update target networks
                self._soft_update()

    def _soft_update(self):
        """Soft update target networks."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # ----------------------------
    # Save / Load models
    # ----------------------------
    def save_models(self, episode=None):
        """Save model checkpoints."""
        folder_path = self.checkpoint_dir
        ensure_dir(folder_path)
        
        suffix = f"_ep{episode}" if episode is not None else ""
        
        torch.save(self.actor.state_dict(), os.path.join(folder_path, f"actor{suffix}.pth"))
        torch.save(self.critic_1.state_dict(), os.path.join(folder_path, f"critic_1{suffix}.pth"))
        torch.save(self.critic_2.state_dict(), os.path.join(folder_path, f"critic_2{suffix}.pth"))
        
        print(f"[INFO] Models saved to {folder_path}")

    def load_models(self, folder_path, episode=None):
        """Load model checkpoints."""
        suffix = f"_ep{episode}" if episode is not None else ""
        
        actor_path = os.path.join(folder_path, f"actor{suffix}.pth")
        critic1_path = os.path.join(folder_path, f"critic_1{suffix}.pth")
        critic2_path = os.path.join(folder_path, f"critic_2{suffix}.pth")
        
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
            self.actor_target.load_state_dict(self.actor.state_dict())
        if os.path.exists(critic1_path):
            self.critic_1.load_state_dict(torch.load(critic1_path, map_location=DEVICE))
            self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        if os.path.exists(critic2_path):
            self.critic_2.load_state_dict(torch.load(critic2_path, map_location=DEVICE))
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        print(f"[INFO] Models loaded from {folder_path}")
