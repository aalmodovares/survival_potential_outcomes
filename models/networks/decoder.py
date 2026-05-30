from importlib.metadata import distribution

import torch
from torch import nn
import warnings

from models.networks.mlp import MLP
from models.networks.kan import KANnet
from models.utils import get_default_dict

import typing
from typing import Tuple, List
from torch.distributions import Normal, Independent, Bernoulli, Categorical, Distribution, Weibull, Exponential, NegativeBinomial

class Decoder(nn.Module):
    def __init__(self, network_name,
                 input_dim, hidden_sizes, data_types, dropout, activation, device,
                 weight_init,
                 min_std=None, max_std=None, min_shape=None, max_shape=None, min_scale=None, max_scale=None,
                 fixed_std=None,
                 limit_variance=False, fix_variance=True, limit_shape=False, limit_scale=False):
        super(Decoder, self).__init__()

        self.network_name = network_name
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.data_types = data_types
        self.dropout = dropout
        self.activation = activation
        self.device = device

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
        self.limit_variance = limit_variance
        self.weight_init = weight_init

        output_dim = 0
        for data_type in data_types:
            if data_type=='continuous' or data_type == 'gaussian' or data_type=='normal' or data_type == 'weibull' or data_type=='negative-binomial':
                if self.fix_variance:
                    output_dim += 1
                else:
                    output_dim += 2
            elif data_type=='binary' or data_type=='bernoulli' or data_type=='exponential':
                output_dim +=1
            elif 'categorical' in data_type:
                k = data_type.split('_')[1]
                output_dim += int(k)

        self.output_dim = output_dim
        # todo: add option to add flows (that allows to include splines)
        
        if self.network_name == 'mlp':
            self.net = MLP(input_dim=self.input_dim,
                           hidden_sizes=self.hidden_sizes,
                           output_dim=self.output_dim,
                           dropout=self.dropout,
                           activations=self.activation,
                           device=self.device,
                           weight_init=self.weight_init)
        elif self.network_name == 'KAN':
            self.net = KANnet(self.input_dim, self.hidden_sizes, self.output_dim, self.device)

        elif self.network_name == 'linear':
            self.net = nn.Linear(self.input_dim, self.output_dim)

        elif self.network_name == 'NSF' or self.network_name == 'flow':
            raise NotImplementedError('Flows not implemented yet')

        else:
            raise ValueError(f'Network name {self.network_name} not recognized')

    def forward(self, z: torch.Tensor) -> List[Distribution]:
        params = self.net(z)
        param_count = 0
        distribution_list = []
        for data_type in self.data_types:
            if data_type=='continuous' or data_type == 'gaussian' or data_type=='normal' or data_type=='numerical':
                if self.fix_variance:
                    mu = params[:, param_count]
                    std = self.fixed_std
                    param_count += 1
                else:
                    mu = params[:, param_count]
                    log_var = params[:, param_count+1]
                    param_count += 2

                    if self.limit_variance:
                        std = self.min_variance + (self.max_variance - self.min_variance) * torch.sigmoid(log_var)
                    else:
                        std = torch.exp(0.5 * log_var)

                distribution_list.append(Normal(loc=mu, scale=std))


            elif data_type == 'weibull':
                log_alpha = params[:, param_count]
                log_k =  params[:, param_count+1]
                param_count += 2
                if self.limit_shape:
                    k = self.min_shape + (self.max_shape - self.min_shape) * torch.sigmoid(log_k)
                else:
                    k = torch.exp(log_k)
                if self.limit_scale:
                    alpha = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(log_alpha)
                else:
                    alpha = torch.exp(log_alpha)

                distribution_list.append(Weibull(scale=alpha, concentration=k))

            elif data_type == 'negative-binomial':
                mu = params[:, param_count]
                log_r = params[:, param_count+1]
                param_count += 2
                r = torch.exp(log_r)
                p = r / (r + mu)
                distribution_list.append(NegativeBinomial(total_count=r, probs=p))

            elif data_type == 'exponential':
                rate = torch.exp(params[:, param_count])
                param_count += 1
                distribution_list.append(Exponential(rate=rate))

            elif data_type == 'binary' or data_type=='bernoulli':
                p = torch.sigmoid(params[:, param_count])
                param_count += 1
                distribution_list.append(Bernoulli(probs=p))
            elif 'categorical' in data_type:
                k = int(data_type.split('_')[1])
                p = torch.nn.functional.softmax(params[:, param_count:param_count+k], dim=-1)
                param_count += k
                distribution_list.append(Categorical(probs=p))
            else:
                raise NotImplementedError('Data type not implemented')
        return distribution_list

    def reconstruction_log_prob(self, distribution_list: List[Distribution], x_true: torch.Tensor, mask_true: torch.Tensor=None):
        x = x_true.clone()
        rec_log_prob = torch.zeros((x.shape[0], len(distribution_list)))
        # x_count = 0
        for i, (data_type, distribution) in enumerate(zip(self.data_types, distribution_list)):
            if (data_type=='continuous') or (data_type=='binary'):
                x_true_i =  x[:, i]
                # x_count+=1
            elif 'categorical' in data_type:
                # k = int(data_type.split('_')[1])
                x_true_i = x[:, i]
                # x_count+=k
            else:
                raise NotImplementedError('Data type not implemented')

            # rec_log_prob += distribution.log_prob(x_true_i).mean()
            rec_log_prob[:, i] = distribution.log_prob(x_true_i)

        if mask_true is not None:
            rec_log_prob = rec_log_prob * (1-mask_true) # mask is true if missing

        return rec_log_prob

    def reconstruct(self, z: torch.Tensor):
        distribution_list = self.forward(z)
        x_reconstructed = []
        for distribution in distribution_list:
            x_reconstructed.append(distribution.sample())
        return torch.cat(x_reconstructed, dim=-1)

    @torch.no_grad()
    def reconstruct_mean(self, z: torch.Tensor):
        distribution_list = self.forward(z)
        x_reconstructed = []
        for data_type, distribution in zip(self.data_types, distribution_list):
            if data_type=='continuous':
                x_reconstructed.append(distribution.mean)
            elif data_type=='binary':
                p = distribution.probs
                x_i_hat = int(p>0.5)
                x_reconstructed.append(x_i_hat)
            elif 'categorical' in data_type:
                probs = distribution.probs
                x_i_hat = torch.zeros(probs.shape)
                i_max = torch.argmax(probs, dim=-1)
                x_i_hat[torch.arange(probs.shape[0]), i_max] = 1
                x_reconstructed.append(x_i_hat)
            else:
                raise NotImplementedError('Data type not implemented')
        return torch.cat(x_reconstructed, dim=-1)

    def sample(self, z_dist):
        z = z_dist.sample()
        return self.reconstruct(z)


