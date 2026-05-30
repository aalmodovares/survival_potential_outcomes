import torch
from torch import nn
from kan import KAN, ex_round
from typing import List, Optional
import numpy as np
import os
import sympy
from sympy.printing.latex import latex
import matplotlib.pyplot as plt
import datetime

from models.utils import get_activation


class KANnet(nn.Module):
    def __init__(self, input_dim:int , hidden_sizes: List[int],
                 output_dim: int, device:str, kan_params: Optional[dict]=None):
        super(KANnet, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.device = device
        self.kan_params = kan_params

        self.kan_params = {
            'k': 3,
            'grid': 5,
            'seed': 10,
            'sparse_init': False,
            'reg_metric': 'edge_forward_spline_n',
            'lamb': 0.01,
            'lamb_l1': 0.01,
            'lamb_entropy': 0.01,
            'lamb_coef': 0.,
            'lamb_coefdiff': 0.,
            'grid_update_num': 10,
            'stop_grid_update_step': 50,
            'start_grid_update_step': -1,
            'update_grid': True
        }
        # Update kan_params with the provided ones
        if kan_params is not None:
            self.kan_params.update(kan_params)

        if len(self.hidden_sizes) == 0:
            self.dims = [self.input_dim] + [self.output_dim]
        else:
            self.dims = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        # We need a random ID to distingush different runs with the same parameters, even when parallelizing: use the current time
        directory_created = False
        folder_set = False
        while not folder_set:
            r_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
            self.model_id = f"{kan_params['k']}_{kan_params['grid']}_{kan_params['lamb']}_{kan_params['lamb_entropy']}_{r_id}"
            self.ckpt_path = os.path.join(os.getcwd(), 'kan_ckpts', f'kan_{str(self.model_id)}') + os.sep
            if not os.path.exists(self.ckpt_path):  # To prevent overwriting if two runs start at the same time
                try:
                    os.makedirs(self.ckpt_path)
                    folder_set = True
                except:
                    print('Error creating folder, trying again with a different seed...')
        self.img_folder = os.path.join(self.ckpt_path, 'video') + os.sep
        self.plot_folder = os.path.join(self.ckpt_path, 'plots') + os.sep
        self.model = KAN(width=self.dims, grid=self.kan_params['grid'], k=self.kan_params['k'],
                         device=torch.device(self.device), sparse_init=self.kan_params['sparse_init'],
                         ckpt_path=self.ckpt_path)

        self.old_save_act = None
        self.old_symbolic_enabled = None
        self.grid_update_freq = None

    def forward(self, x):
        return self.model(x)

    def on_training_start(self):
        self.old_save_act, self.old_symbolic_enabled = self.model.disable_symbolic_in_fit(self.kan_params['lamb'])
        self.grid_update_freq = int(self.kan_params['stop_grid_update_step'] / self.kan_params['grid_update_num'])

    def on_epoch_start(self, epoch, n_epochs, x):
        if epoch == n_epochs - 1 and self.old_save_act:
            self.model.save_act = True

        if epoch % self.grid_update_freq == 0 and epoch < self.kan_params['stop_grid_update_step'] and epoch > self.kan_params['start_grid_update_step'] and self.kan_params['update_grid']:
            self.model.update_grid({'train_input': x})

    def on_epoch_end(self):
        self.model.symbolic_enabled = self.old_symbolic_enabled

    def compute_regularization(self):
        if self.model.save_act:
            reg_ = self.model.get_reg(self.kan_params['reg_metric'], self.kan_params['lamb_l1'],
                                      self.kan_params['lamb_entropy'], self.kan_params['lamb_coef'], self.kan_params['lamb_coefdiff'])
        else:
            reg_ = torch.tensor(0.)
        return reg_
