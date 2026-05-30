import torch
import warnings
from torch import nn

from models.networks.mlp import MLP
from models.networks.kan import KANnet

from torch.distributions import Normal, Independent, Weibull,  Distribution
import typing
from typing import Tuple

from models.utils import get_default_dict

class Encoder(nn.Module):
    def __init__(self, network_name,
                 input_dim, hidden_sizes,
                 latent_dim, latent_dist,
                 dropout, activation, device,
                 weight_init,
                 min_std=None, max_std=None, min_shape=None, max_shape=None, min_scale=None, max_scale=None,
                 fixed_std=None,
                 limit_variance=False, fix_variance=True, limit_shape=False, limit_scale=False):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.activation = activation
        self.device = device
        self.limit_variance = limit_variance
        self.latent_dist = latent_dist
        self.weight_init = weight_init

        if (fix_variance and limit_variance):
            warnings.warn(
                'fix_variance and limit_variance are both True. This is not possible. limit_variance will be set to False')
            self.limit_variance = False
        if limit_variance:
            if min_std is None:
                # warning
                warnings.warn('min_std must be provided if limit_variance is True')
                min_std = get_default_dict()['min_std']
            if max_std is None:
                # warning
                warnings.warn('max_std must be provided if limit_variance is True')
                max_std = get_default_dict()['max_std']

        if limit_shape:
            if min_shape is None:
                # warning
                warnings.warn('min_shape must be provided if limit_shape is True')
                min_shape = get_default_dict()['min_shape']
            if max_shape is None:
                # warning
                warnings.warn('max_shape must be provided if limit_shape is True')
                max_shape = get_default_dict()['max_shape']
        if limit_scale:
            if min_scale is None:
                # warning
                warnings.warn('min_scale must be provided if limit_scale is True')
                min_scale = get_default_dict()['min_scale']
            if max_scale is None:
                # warning
                warnings.warn('max_scale must be provided if limit_scale is True')
                max_scale = get_default_dict()['max_scale']
        if fix_variance:
            if fixed_std is None:
                # warning
                warnings.warn('fixed_std must be provided if fix_variance is True')
                fixed_std = get_default_dict()['fixed_std']
        
        self.min_std = min_std
        self.max_std = max_std
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fixed_std = fixed_std
        self.limit_shape = limit_shape
        self.limit_scale = limit_scale
        self.fix_variance = fix_variance
        
        
        if self.latent_dist == 'normal' or self.latent_dist == 'gaussian' or self.latent_dist =='weibull':
            if self.fix_variance:
                self.output_dim = self.latent_dim
            else:
                self.output_dim = self.latent_dim * 2
        
        elif self.latent_dist == 'flow':
            raise NotImplementedError('Flows not implemented yet')
            
        else:
            raise ValueError(f'Latent distribution {self.latent_dist} not recognized')


        # todo: add flows (NSF)

        if network_name == 'mlp':
            self.net = MLP(input_dim=self.input_dim,
                       hidden_sizes=self.hidden_sizes,
                       output_dim=self.output_dim,
                       dropout=self.dropout,
                       activations=self.activation,
                       device=self.device,
                       weight_init=self.weight_init)
        elif network_name == 'KAN':

            self.net = KANnet(self.input_dim, self.hidden_sizes, self.output_dim, self.device)

        elif network_name == 'linear':
            self.net = nn.Linear(self.input_dim, self.output_dim)
        
        elif network_name == 'NSF' or network_name == 'flow':
            raise NotImplementedError('Flows not implemented yet')

        else:
            raise ValueError(f'Network name {network_name} not recognized')

    def forward(self, x:torch.Tensor) -> Distribution:
        output = self.net(x)
        if self.latent_dist == 'normal' or self.latent_dist == 'gaussian':
            if self.fix_variance:
                mu = self.net(x)
                std = torch.tensor(self.fixed_std)
            else:
                mu, log_var = torch.chunk(self.net(x), 2, dim=-1)
                if self.limit_variance:
                    std = self.min_variance + (self.max_variance - self.min_variance) * torch.sigmoid(log_var)
                else:
                    std = torch.exp(0.5 * log_var)

            # var = torch.nn.functional.softplus(log_var)
            return Independent(Normal(loc=mu, scale=std), 1) #Independent Gaussian distribution

        elif self.latent_dist == 'weibull':
            log_alpha, log_k = torch.chunk(self.net(x), 2, dim=-1)  # both positive parameters
            if self.limit_shape:
                k = self.min_shape + (self.max_shape - self.min_shape) * torch.sigmoid(log_k)
            else:
                k = torch.exp(log_k)
            if self.limit_scale:
                alpha = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(log_alpha)
            else:
                alpha = torch.exp(log_alpha)

            return Independent(Weibull(scale=alpha, concentration=k), reinterpreted_batch_ndims=1)
        
        elif self.latent_dist == 'flow':
            return output


    # get the mean and log var of the latent space
    @torch.no_grad()
    def get_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.forward(x)
        return q.mean, q.variance

    @torch.no_grad()
    def get_samples(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        # sample n_samples from the latent space without gradients
        q = self.forward(x)
        z = q.sample((n_samples,))
        return z

    # sample from the latent space
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        q = self.forward(x)
        z = q.rsample()
        return z
