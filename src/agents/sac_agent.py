"""
SAC (Soft Actor-Critic) Agent for HEMS.

Implements the complete SAC algorithm with automatic entropy tuning.
Action space: 4-dimensional continuous (bat, grid, pv, load)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer


# -----------------------------
# Actor Network (Gaussian Policy)
# -----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, x, deterministic=False):
        mu, std = self.forward(x)
        
        if deterministic:
            action = torch.tanh(mu)
            log_prob = None
        else:
            dist = torch.distributions.Normal(mu, std)
            z = dist.rsample()
            action = torch.tanh(z)
            # Log probability with tanh squashing correction
            log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


# -----------------------------
# Critic Network (Q-function)
# -----------------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# -----------------------------
# SAC Agent (Complete)
# -----------------------------
class SACAgent:
    def __init__(self, env, args=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]  # Should be 4

        # Hyperparameters
        self.batch_size = getattr(args, "batch_size", 256)
        self.gamma = getattr(args, "gamma", 0.99)
        self.tau = getattr(args, "tau", 0.005)
        self.lr_actor = getattr(args, "lr_actor", 3e-4)
        self.lr_critic = getattr(args, "lr_critic", 3e-4)
        self.automatic_entropy_tuning = getattr(args, "alpha_tuning", True)
        self.alpha = getattr(args, "alpha", 0.2)

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic2 = Critic(self.state_dim, self.action_dim).to(self.device)

        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.lr_critic)

        # Replay buffer
        buffer_capacity = getattr(args, "buffer_size", 1000000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, device=self.device)

        # Entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(self.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=getattr(args, "lr_alpha", 3e-4))
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = getattr(args, "alpha", 0.2)

        # Checkpoint dir
        self.checkpoint_dir = getattr(args, "checkpoint_dir", "./checkpoints_SAC")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # -----------------------------
    # Select action
    # -----------------------------
    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic=deterministic)
        return action.cpu().numpy()[0]

    # -----------------------------
    # Soft update target networks
    # -----------------------------
    def soft_update_target(self):
        for target_param, source_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
        
        for target_param, source_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    # -----------------------------
    # Update networks
    # -----------------------------
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch(self.batch_size)
        
        # ---------- Critic update ----------
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_value = rewards + self.gamma * (1 - dones) * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        # ---------- Actor update ----------
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # ---------- Alpha (entropy) update ----------
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
        
        # Soft update target networks
        self.soft_update_target()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "alpha": self.alpha.item()
        }

    # -----------------------------
    # Save models
    # -----------------------------
    def save_models(self, episode=None):
        path = self.checkpoint_dir
        os.makedirs(path, exist_ok=True)
        
        suffix = f"_ep{episode}" if episode is not None else ""
        
        torch.save(self.actor.state_dict(), f"{path}/actor{suffix}.pth")
        torch.save(self.critic1.state_dict(), f"{path}/critic1{suffix}.pth")
        torch.save(self.critic2.state_dict(), f"{path}/critic2{suffix}.pth")
        torch.save(self.target_critic1.state_dict(), f"{path}/tcritic1{suffix}.pth")
        torch.save(self.target_critic2.state_dict(), f"{path}/tcritic2{suffix}.pth")
        
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, f"{path}/alpha{suffix}.pth")
        
        print(f"[INFO] Models saved to {path}")

    # -----------------------------
    # Load models
    # -----------------------------
    def load_models(self, episode=None):
        path = self.checkpoint_dir
        suffix = f"_ep{episode}" if episode is not None else ""
        
        actor_path = os.path.join(path, f"actor{suffix}.pth")
        if not os.path.exists(actor_path):
            print(f"[WARNING] No pretrained models found in {path}. Training from scratch.")
            return
        
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic1.load_state_dict(torch.load(f"{path}/critic1{suffix}.pth", map_location=self.device))
        self.critic2.load_state_dict(torch.load(f"{path}/critic2{suffix}.pth", map_location=self.device))
        self.target_critic1.load_state_dict(torch.load(f"{path}/tcritic1{suffix}.pth", map_location=self.device))
        self.target_critic2.load_state_dict(torch.load(f"{path}/tcritic2{suffix}.pth", map_location=self.device))
        
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load(f"{path}/alpha{suffix}.pth", map_location=self.device)
            self.alpha = self.log_alpha.exp().detach()
        
        print(f"[INFO] Models loaded from {path}")
