
"""
Action noise functions for HEMS Deep Reinforcement Learning.

Used for exploration in continuous action spaces (TD3, DDPG, etc.).
Action dimension: 4 (bat, grid, pv, load)
"""

import numpy as np


class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    
    OU noise is well-suited for physical systems like HEMS because it
    produces smooth, temporally correlated noise rather than white noise.
    """
    
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1, x0=None):
        """
        Args:
            mu: Mean of the noise process (usually zero)
            sigma: Volatility / scale of the noise
            theta: Speed of mean reversion
            dt: Time step
            x0: Initial value (default: zeros)
        """
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """
        Reset the noise process to initial state.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})'


class NormalActionNoise:
    """
    Independent Gaussian white noise for action exploration.
    
    Simpler than OU noise, adds i.i.d. Gaussian noise to each action dimension.
    """
    
    def __init__(self, size, sigma=0.2):
        """
        Args:
            size: Action dimension (should be 4 for HEMS)
            sigma: Standard deviation of the noise
        """
        self.size = size
        self.sigma = sigma

    def __call__(self):
        """Generate and return noise."""
        return np.random.randn(self.size) * self.sigma

    def reset(self):
        """Reset the noise process (no-op for independent Gaussian)."""
        pass


class ClippedNormalActionNoise(NormalActionNoise):
    """
    Clipped Gaussian noise for action exploration.
    Clips noise to [-clip, clip] to prevent extreme actions.
    """
    
    def __init__(self, size, sigma=0.2, clip=0.5):
        super().__init__(size, sigma)
        self.clip = clip
    
    def __call__(self):
        noise = super().__call__()
        return np.clip(noise, -self.clip, self.clip)


class AdaptiveActionNoise:
    """
    Adaptive noise that decays over time.
    Useful for annealing exploration during training.
    """
    
    def __init__(self, size, initial_sigma=0.3, final_sigma=0.05, decay_steps=100000):
        """
        Args:
            size: Action dimension
            initial_sigma: Starting noise standard deviation
            final_sigma: Final noise standard deviation
            decay_steps: Number of steps over which to decay
        """
        self.size = size
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def __call__(self):
        """Generate noise with current sigma."""
        sigma = self._current_sigma()
        return np.random.randn(self.size) * sigma
    
    def _current_sigma(self):
        """Compute current sigma based on decay schedule."""
        if self.current_step >= self.decay_steps:
            return self.final_sigma
        
        progress = self.current_step / self.decay_steps
        sigma = self.initial_sigma * (1 - progress) + self.final_sigma * progress
        return sigma
    
    def step(self):
        """Increment step counter for decay."""
        self.current_step += 1
    
    def reset(self):
        """Reset step counter."""
        self.current_step = 0
