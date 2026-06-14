
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor network for Home Energy Management System (HEMS).

    Outputs continuous control actions (4-dimensional):
    a_t = (a_t^bat, a_t^grid, a_t^pv, a_t^load)
    
    - a_t^bat: battery power (continuous, -1 to 1 → scaled to physical bounds)
    - a_t^grid: grid power exchange (continuous, -1 to 1 → scaled to import/export bounds)
    - a_t^pv: PV utilization fraction (0 to 1)
    - a_t^load: controllable load multiplier (0 to 1)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layer with small weights for stability
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate actions.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            
        Returns:
            actions: Tensor of shape (batch, action_dim=4) in range [-1, 1]
                     To be denormalized by the environment.
        """
        x = self.shared_layers(state)
        z = self.output_layer(x)
        
        # Action-specific activations
        a_bat = torch.tanh(z[:, 0])       # battery: -1 to 1 (charge/discharge)
        a_grid = torch.tanh(z[:, 1])      # grid: -1 to 1 (import/export)
        a_pv = torch.sigmoid(z[:, 2])     # PV: 0 to 1 (utilization fraction)
        a_load = torch.sigmoid(z[:, 3])   # load: 0 to 1 (load multiplier)
        
        actions = torch.stack([a_bat, a_grid, a_pv, a_load], dim=1)
        
        return actions

    def freeze(self):
        """Freeze actor parameters (useful for target networks)."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze actor parameters."""
        for p in self.parameters():
            p.requires_grad = True
