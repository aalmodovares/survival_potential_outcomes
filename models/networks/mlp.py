import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from typing import List, Union

from models.utils import get_activation

class MLP(nn.Module):
    def __init__(self, input_dim:int , hidden_sizes: List[int],
                 output_dim: int,
                 dropout: float, activations: Union[List[str], str], device:str,
                 weight_init:str):
        """
                weight_init: str or None
                    - 'xavier_uniform' -> Glorot Uniform
                    - 'xavier_normal'  -> Glorot Normal
                    - None -> Use default PyTorch initialization
                """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device
        self.weight_init = weight_init  # Store initialization method


        if isinstance(activations, str):
            activations = [activations] * (len(hidden_sizes))
        assert len(activations) == len(hidden_sizes)
        self.activations = nn.ModuleList([get_activation(act) for act in activations])
        self.layers = nn.ModuleList()
        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_sizes[0]))
            for i in range(1, len(hidden_sizes)):
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        # Pass all layers to the device
        self.layers.to(self.device)
        # Apply custom weight initialization if specified
        if self.weight_init is not None:
            self._init_weights()

    def _init_weights(self):
        """Apply custom initialization to linear layers."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if self.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif self.weight_init == "xavier_normal":
                    nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)  # Set bias to zero

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activations[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

