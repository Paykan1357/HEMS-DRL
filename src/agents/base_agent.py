"""
Base Agent for HEMS Deep Reinforcement Learning.

This module provides the foundational class for DRL agents in
Home Energy Management Systems.
"""

import numpy as np


class BaseAgent:
    """
    Base class for all HEMS DRL agents.
    
    Attributes:
        name (str): Agent identifier
        env (gym.Env): HEMS environment
        seed (int): Random seed for reproducibility
        training (bool): Training mode flag
    """
    
    def __init__(self, name, env, seed=0, reward_type=None):
        self.name = name
        self.env = env
        self.seed = seed
        self.training = True
        self.reward_type = reward_type  # Optional: different reward variants

    def set_eval(self):
        """Set agent to evaluation mode (no exploration)."""
        self.training = False

    def set_train(self):
        """Set agent to training mode (with exploration)."""
        self.training = True

    def act(self, obs, exploration=False):
        """
        Select an action given the current observation.
        
        Args:
            obs: Current state observation
            exploration: Whether to add exploration noise
        
        Returns:
            action: Action array (4-dimensional for HEMS)
        """
        raise NotImplementedError("Subclass must implement act()")

    def evaluate(self, env, exploration=False, render=False):
        """
        Evaluate agent in the given environment.
        
        Args:
            env: HEMS environment
            exploration: Whether to use exploration during evaluation
            render: Whether to render (if supported)
        
        Returns:
            total_reward (float): Cumulative reward over episode
            infos (list): List of step information dicts
            final_metrics (dict): Final performance metrics
        """
        self.set_eval()
        obs = env.reset()
        done = False
        
        rewards = []
        step_infos = []
        
        while not done:
            # Select action
            action = self.act(obs, exploration=exploration)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store step data
            rewards.append(reward)
            step_infos.append(info)
            
            # Update observation
            obs = next_obs
        
        # Compute episode metrics
        total_reward = sum(rewards)
        
        # Aggregate final metrics from step infos
        final_metrics = self._aggregate_metrics(step_infos)
        
        return total_reward, step_infos, final_metrics
    
    def _aggregate_metrics(self, step_infos):
        """
        Aggregate step-wise metrics into episode-level metrics.
        
        Args:
            step_infos: List of info dicts from each step
        
        Returns:
            metrics: Dict of episode-level metrics
        """
        if not step_infos:
            return {}
        
        metrics = {
            "total_energy_cost": sum(info.get("energy_cost", 0) for info in step_infos),
            "total_unmet_load": sum(info.get("unmet_load", 0) for info in step_infos),
            "avg_soc": np.mean([info.get("soc", 0) for info in step_infos]),
            "avg_comfort_penalty": np.mean([info.get("comfort_penalty", 0) for info in step_infos]),
        }
        
        # Add PV utilization if available
        pv_used = [info.get("pv_used", 0) for info in step_infos]
        pv_available = [info.get("pv_available", 1) for info in step_infos]
        metrics["pv_utilization"] = sum(pv_used) / (sum(pv_available) + 1e-8)
        
        return metrics
