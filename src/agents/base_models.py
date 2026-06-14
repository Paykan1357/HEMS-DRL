"""
Base model utilities for HEMS deep reinforcement learning.

Purpose:
- Common parent class for all neural networks
- Weight initialization utilities
- Layer freezing for transfer learning / pretraining

Supported layers:
- nn.Linear
- nn.LSTM
- nn.GRU (optional)
"""

import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all HEMS neural network models.
    """

    def __init__(self):
        super().__init__()

    # ====================================================
    #            FREEZE / UNFREEZE PARAMETERS
    # ====================================================
    def requires_grad(self, requires_grad: bool, pretrained: bool = False):
        """
        Enable or disable gradient computation.

        Parameters
        ----------
        requires_grad : bool
            Whether parameters should be trainable.
        pretrained : bool
            If True, keeps shared 'base' layers frozen.
        """
        for name, param in self.named_parameters():
            if requires_grad and pretrained and "base." in name:
                continue
            param.requires_grad = requires_grad

    # ====================================================
    #               WEIGHT INITIALIZATION
    # ====================================================
    def init_weights(self, nonlinearity: str = "leaky_relu", a: float = 0.1):
        """
        Initialize weights for Linear and LSTM layers.

        Parameters
        ----------
        nonlinearity : str
            Activation function used ('relu' or 'leaky_relu')
        a : float
            Negative slope for LeakyReLU
        """
        for module in self.modules():

            # ---------- Linear Layers ----------
            if isinstance(module, nn.Linear):
                if nonlinearity == "relu":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                elif nonlinearity == "leaky_relu":
                    nn.init.kaiming_normal_(
                        module.weight, nonlinearity="leaky_relu", a=a
                    )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            # ---------- LSTM Layers ----------
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "bias" in name:
                        nn.init.constant_(param, 0.0)
                    elif "weight" in name:
                        if nonlinearity == "relu":
                            nn.init.kaiming_normal_(param, nonlinearity="relu")
                        elif nonlinearity == "leaky_relu":
                            nn.init.kaiming_normal_(
                                param, nonlinearity="leaky_relu", a=a
                            )
