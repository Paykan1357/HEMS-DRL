
"""
Replay Buffer for HEMS Deep Reinforcement Learning.

Stores transitions (state, action, reward, done, next_state) for
off-policy algorithms like DQN, DDPG, TD3, and SAC.

Action dimension: 4 (bat, grid, pv, load)
State dimension: Varies (flattened state vector)
"""

import random
import numpy as np
import torch


class ReplayBuffer:
    """
    Generic replay buffer for HEMS (continuous control).
    Stores flat state vectors for efficient sampling.
    """
    
    def __init__(self, capacity=1000000, device=None):
        """
        Args:
            capacity: Maximum number of transitions to store
            device: PyTorch device (CPU or CUDA)
        """
        self.capacity = int(capacity)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Circular buffer storage
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.next_obs_buf = []

        self.pos = 0  # Current write position

    def __len__(self):
        """Return current number of stored transitions."""
        return len(self.obs_buf)

    def push(self, obs, action, reward, done, next_obs):
        """
        Add a transition to the buffer (alternative name for add()).
        
        Args:
            obs: Current state (flattened)
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            next_obs: Next state (flattened)
        """
        self.add(obs, action, reward, done, next_obs)

    def add(self, obs, action, reward, done, next_obs):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Current state (flattened)
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            next_obs: Next state (flattened)
        """
        # Ensure obs/next_obs are always flat 1D vectors
        obs = np.asarray(obs, dtype=np.float32).flatten()
        next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
        action = np.asarray(action, dtype=np.float32)

        if len(self.obs_buf) < self.capacity:
            # Buffer not full yet: append
            self.obs_buf.append(obs)
            self.act_buf.append(action)
            self.rew_buf.append(reward)
            self.done_buf.append(done)
            self.next_obs_buf.append(next_obs)
        else:
            # Buffer full: overwrite at current position
            self.obs_buf[self.pos] = obs
            self.act_buf[self.pos] = action
            self.rew_buf[self.pos] = reward
            self.done_buf[self.pos] = done
            self.next_obs_buf[self.pos] = next_obs
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (obs, actions, rewards, dones, next_obs) as PyTorch tensors
        """
        return self.sample_batch(batch_size)

    def sample_batch(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            obs: State tensor [batch, state_dim]
            actions: Action tensor [batch, action_dim]
            rewards: Reward tensor [batch, 1]
            dones: Done flag tensor [batch, 1]
            next_obs: Next state tensor [batch, state_dim]
        """
        batch_size = int(batch_size)
        
        # Random sample indices
        idxs = random.sample(range(len(self.obs_buf)), batch_size)

        # Stack and convert to tensors
        obs = torch.tensor(
            np.stack([self.obs_buf[i] for i in idxs]),
            dtype=torch.float32, device=self.device
        )

        actions = torch.tensor(
            np.stack([self.act_buf[i] for i in idxs]),
            dtype=torch.float32, device=self.device
        )

        rewards = torch.tensor(
            np.array([self.rew_buf[i] for i in idxs], dtype=np.float32),
            device=self.device
        ).unsqueeze(-1)

        dones = torch.tensor(
            np.array([self.done_buf[i] for i in idxs], dtype=np.float32),
            device=self.device
        ).unsqueeze(-1)

        next_obs = torch.tensor(
            np.stack([self.next_obs_buf[i] for i in idxs]),
            dtype=torch.float32, device=self.device
        )

        return obs, actions, rewards, dones, next_obs

    def size(self):
        """Return current number of stored transitions (alias for __len__)."""
        return len(self.obs_buf)

    def clear(self):
        """Clear all transitions from the buffer."""
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.next_obs_buf = []
        self.pos = 0

    def is_ready(self, batch_size):
        """
        Check if buffer has enough samples for a batch.
        
        Args:
            batch_size: Required batch size
            
        Returns:
            True if buffer size >= batch_size
        """
        return len(self.obs_buf) >= batch_size
