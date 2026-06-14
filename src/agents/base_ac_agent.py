"""
Base Actor-Critic Agent for HEMS Deep Reinforcement Learning.

This module provides the foundational class for continuous-action DRL agents
such as TD3, SAC, and ACKTR applied to Home Energy Management Systems.

Action space: 4-dimensional continuous
    [a_t^bat, a_t^grid, a_t^pv, a_t^load]
"""

import numpy as np
import torch
from copy import deepcopy
from replay_buffer import ReplayBuffer


class BaseACAgent:
    """
    Base class for Actor-Critic agents in HEMS.
    
    Attributes:
        name (str): Agent identifier
        env (gym.Env): HEMS environment
        device (torch.device): CUDA or CPU
        state_dim (tuple): Observation space shape
        action_dim (int): Action dimension (4 for HEMS)
        batch_size (int): Training batch size
        buffer_size (int): Replay buffer capacity
        gamma (float): Discount factor
        tau (float): Soft update coefficient
        buffer (ReplayBuffer): Experience replay buffer
    """
    
    def __init__(self, name, env, seed, args):
        self.name = name
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Environment-derived dimensions
        self.state_dim = env.observation_space.shape[0]  # Flattened state vector
        self.action_dim = env.action_space.shape[0]      # Should be 4 for HEMS

        # Hyperparameters (with HEMS-appropriate defaults)
        self.batch_size = getattr(args, "batch_size", 256)
        self.buffer_size = getattr(args, "buffer_size", 1000000)
        self.gamma = getattr(args, "gamma", 0.99)
        self.tau = getattr(args, "tau", 0.005)
        
        # Learning rates (can be overridden by subclasses)
        self.actor_lr = getattr(args, "actor_lr", 3e-4)
        self.critic_lr = getattr(args, "critic_lr", 3e-4)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=self.buffer_size)

        # Placeholders to be set by subclasses (TD3, SAC, etc.)
        self.actor = None
        self.actor_target = None
        self.critic_1 = None
        self.critic_2 = None
        self.critic_target_1 = None
        self.critic_target_2 = None
        
        # Optimizers (to be initialized after networks are created)
        self.actor_optimizer = None
        self.critic_optimizer_1 = None
        self.critic_optimizer_2 = None

    def define_actors_critics(self, args):
        """
        Create actor and critic networks.
        Must be implemented by subclass (TD3, SAC, etc.).
        
        Args:
            args: Configuration arguments containing network architecture parameters
        """
        raise NotImplementedError("Subclass must implement define_actors_critics()")

    def copy_params_to_target(self):
        """
        Initialize target networks with the same parameters as source networks.
        """
        if self.actor is not None:
            self.actor_target = deepcopy(self.actor)
        if self.critic_1 is not None:
            self.critic_target_1 = deepcopy(self.critic_1)
        if self.critic_2 is not None:
            self.critic_target_2 = deepcopy(self.critic_2)

    def soft_update_target(self):
        """
        Perform Polyak averaging for target networks.
        target = tau * source + (1 - tau) * target
        
        This method should be called after each training step.
        """
        if self.actor is not None and self.actor_target is not None:
            for target_param, source_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
            ):
                target_param.data.copy_(
                    self.tau * source_param.data + (1.0 - self.tau) * target_param.data
                )
        
        if self.critic_target_1 is not None and self.critic_1 is not None:
            for target_param, source_param in zip(
                self.critic_target_1.parameters(), self.critic_1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * source_param.data + (1.0 - self.tau) * target_param.data
                )
        
        if self.critic_target_2 is not None and self.critic_2 is not None:
            for target_param, source_param in zip(
                self.critic_target_2.parameters(), self.critic_2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * source_param.data + (1.0 - self.tau) * target_param.data
                )

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Episode termination flag
        """
        self.buffer.push(state, action, reward, next_state, done)

    def sample_batch(self):
        """
        Sample a random batch of transitions from the replay buffer.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        return self.buffer.sample(self.batch_size)

    def save_checkpoint(self, filepath):
        """
        Save agent checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict() if self.actor else None,
            'critic_1_state_dict': self.critic_1.state_dict() if self.critic_1 else None,
            'critic_2_state_dict': self.critic_2.state_dict() if self.critic_2 else None,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict() if self.actor_optimizer else None,
            'critic_optimizer_1_state_dict': self.critic_optimizer_1.state_dict() if self.critic_optimizer_1 else None,
            'critic_optimizer_2_state_dict': self.critic_optimizer_2.state_dict() if self.critic_optimizer_2 else None,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load agent checkpoint.
        
        Args:
            filepath: Path to load the checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.actor and checkpoint['actor_state_dict']:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
        if self.critic_1 and checkpoint['critic_1_state_dict']:
            self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        if self.critic_2 and checkpoint['critic_2_state_dict']:
            self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        if self.actor_optimizer and checkpoint['actor_optimizer_state_dict']:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        if self.critic_optimizer_1 and checkpoint['critic_optimizer_1_state_dict']:
            self.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1_state_dict'])
        if self.critic_optimizer_2 and checkpoint['critic_optimizer_2_state_dict']:
            self.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2_state_dict'])
        
        print(f"Checkpoint loaded from {filepath}")
