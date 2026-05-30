import torch
from torch import nn

from models.wrapper import Wrapper
from models.networks.mlp import MLP
from models.networks.kan import KANnet
from models.networks.regressors import outcome_regressor

from models.utils import inverse_probability_metric

class CFR(Wrapper):

    def __init__(self, network_name,
                 input_dim, hidden_sizes_phi, hidden_sizes_y, phi_dim,
                 outcome_dist, num_treatments,
                 divergence_loss, beta_div,
                 experiment, model_path,
                 device,
                 weight_init_phi=None, weight_init_y=None, dropout=0.0, activation='relu',
                 min_std=None, max_std=None, min_shape=None, max_shape=None, min_scale=None, max_scale=None,
                 fixed_std=None,
                 limit_variance=False, fix_variance=False, limit_shape=False, limit_scale=False,
                 min_outcome=None,
                 kan_params_y=None, kan_params_phi=None,
                 save=True, optim_config=None, wandblog=False, is_subnetwork=False
                 ):

        '''
        :param network_name: str, name of the network; either mlp or KAN
        :param input_dim:
        :param hidden_sizes_phi:
        :param hidden_sizes_y:
        :param phi_dim:
        :param outcome_dist:
        :param num_treatments:
        :param divergence_loss: str, selection of Inverse probability metric; either wasserstein, mmd or wasserstein2
        :param beta_div: float, weight of the divergence loss
        :param experiment:
        :param model_path:
        :param dropout:
        :param activation:
        :param device:
        :param min_std:
        :param max_std:
        :param min_shape:
        :param max_shape:
        :param min_scale:
        :param max_scale:
        :param fixed_std:
        :param limit_variance:
        :param fix_variance:
        :param limit_shape:
        :param limit_scale:
        :param min_outcome:
        :param save:
        :param optim_config:
        :param wandblog:
        '''

        if beta_div == 0:
            name = 'tarnet'
        else:
            name = 'cfrnet'

        super(CFR, self).__init__(name, device, experiment, model_path, save, optim_config, wandblog)

        self.network_name = network_name

        self.input_dim = input_dim
        self.hidden_sizes_phi = hidden_sizes_phi
        self.hidden_sizes_y = hidden_sizes_y
        self.phi_dim = phi_dim
        self.outcome_dist = outcome_dist
        self.num_treatments = num_treatments
        self.dropout = dropout
        self.activation = activation

        self.divergence_loss = divergence_loss #options: mmd, wasserstein or wasserstein2
        self.beta_div = beta_div

        self.weight_init_phi = weight_init_phi
        self.weight_init_y = weight_init_y

        self.net = nn.ModuleList()
        # shared network
        if network_name == 'mlp':
            self.net.append(MLP(input_dim = self.input_dim,
                                hidden_sizes= self.hidden_sizes_phi,
                                output_dim = phi_dim,
                                dropout=self.dropout,
                                activations=self.activation,
                                device=self.device,
                                weight_init=self.weight_init_phi
                                ))
        elif network_name == 'KAN':
            self.net.append(KANnet(input_dim = self.input_dim,
                                   hidden_sizes= self.hidden_sizes_phi,
                                   output_dim = phi_dim,
                                   device=self.device,
                                   kan_params=kan_params_phi))
        for _ in range(self.num_treatments):
            self.net.append(outcome_regressor(network_name=self.network_name,
                                              input_dim = self.phi_dim,
                                              hidden_sizes = self.hidden_sizes_y,
                                              outcome_dist = self.outcome_dist,
                                              dropout = self.dropout,
                                              activation = self.activation,
                                              device = self.device,
                                              min_std=min_std,
                                              max_std=max_std,
                                              min_shape=min_shape,
                                              max_shape=max_shape,
                                              min_scale=min_scale,
                                              max_scale=max_scale,
                                              fixed_std=fixed_std,
                                              limit_variance=limit_variance,
                                              fix_variance=fix_variance,
                                              limit_shape = limit_shape,
                                              limit_scale = limit_scale,
                                              min_outcome=min_outcome,
                                              weight_init= self.weight_init_y,
                                              kan_params=kan_params_y
                                              ))
        if not is_subnetwork:
            self.get_optim()

    def forward(self, x_dict):
        x = x_dict['x']
        phi = self.net[0](x)
        return [phi] + [self.net[i+1](phi) for i in range(self.num_treatments)]

    def loss(self, output, true_values):
        y_true = true_values['y']
        t_true = true_values['t']
        c_true = true_values['c']
        assert t_true is not None, 'treatment is None'
        log_prob = torch.zeros(len(t_true), 1)
        phi = output[0]
        for i, y_dist in enumerate(output[1:]):
            log_prob_i = self.net[i+1].log_prob(y_dist, y_true, c_true)
            log_prob_i = log_prob_i*torch.reshape((t_true == i).int(), (-1, 1))
            if torch.sum(torch.isnan(log_prob_i)): # TODO: this is for debugging, can be erased later
                print(f'NaN in log_prob_i in positions {torch.nonzero(torch.isnan(log_prob_i))}')
            log_prob += log_prob_i
        reconstruction_loss = -log_prob.mean()

        # divergence loss between representation groups
        if self.divergence_loss is None:
            return {'loss':reconstruction_loss}
        else:
            div_loss = inverse_probability_metric(phi, t_true, self.divergence_loss)

        loss = reconstruction_loss + self.beta_div*div_loss
        loss_dict = {'loss': loss, 'log_prob': log_prob.mean(),'reconstruction_loss': reconstruction_loss, 'divergence_loss': div_loss}

        return loss_dict

    @torch.no_grad()
    def predict(self, X_test, load_model: str):
        self.load(load_model)
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
        y_dist_list = self({'x':X_test})[1:] # the first element is phi


        return {'y_dist_list': y_dist_list}



