import torch
from torch import nn
from tqdm import tqdm
import wandb
from models.networks.regressors import outcome_regressor, treatment_regressor
from models.wrapper import Wrapper
from models.utils import get_activation

class Outcome_Model(Wrapper):
    def __init__(self, network_name,
                 input_dim, hidden_sizes, outcome_dist,
                 experiment, model_path,
                 dropout, activation, device,
                 weight_init=None,
                 save=True, limit_variance=False, limit_shape=False, optim_config=None,
                 wandblog=False, is_subnetwork=False):

        super(Outcome_Model, self).__init__('outcome-model',
                                            device, experiment, model_path, save, optim_config, wandblog)

        self.network_name = network_name
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.outcome_dist = outcome_dist
        self.dropout = dropout
        self.activation = activation
        self.limit_variance = limit_variance
        self.limit_shape = limit_shape
        self.weight_init = weight_init

        if self.outcome_dist=='not-specified':
            self.criterion = 'mse'
        else:
            self.criterion = 'log_prob'

        self.net = outcome_regressor(network_name=self.network_name,
                                     input_dim = self.input_dim,
                                     hidden_sizes = self.hidden_sizes,
                                     outcome_dist = self.outcome_dist,
                                     activation=self.activation,
                                     dropout = self.dropout,
                                     device = self.device,
                                     weight_init=self.weight_init)

        if not is_subnetwork:
            self.get_optim()

    def loss(self, output, true_values):
        y_dist_pred = output
        y_true = true_values['y']
        c_true = true_values['c']
        if self.criterion == 'mse':
            loss = self.net.mse(y_dist_pred, y_true, c_true)
        else:
            loss = -self.net.log_prob(y_dist_pred, y_true, c_true).mean()
            loss = loss.mean()

        loss_dict = {'loss': loss}

        return loss_dict

    @torch.no_grad()
    def predict(self, X_test, load_model: str='best'):
        self.load(load_model)
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
        y_dist = self({'x':X_test})

        return {'y_dist': y_dist}

    def summary(self, X_df, y):
        if self.network_name != 'linear':
            raise NotImplementedError('Summary only implemented for linear models')
        results_df = self.net.summary(X_df, y)
        print(results_df)

class Treatment_Model(Wrapper):
    def __init__(self, network_name,
                 input_dim, hidden_sizes, num_treatments,
                 experiment, model_path,
                 dropout, activation, device,
                 treatment_name='t',
                 weight_init=None,
                 save=True, optim_config=None, wandblog=False, is_subnetwork=False):

        super(Treatment_Model, self).__init__('treatment-model',
                                              device, experiment, model_path, save, optim_config, wandblog)

        self.network_name = network_name
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.num_treatments = num_treatments
        if self.num_treatments == 2:
            self.output_dim = 1
        else:
            self.output_dim = self.num_treatments
        self.device = device
        self.dropout = dropout
        self.activation = activation
        self.weight_init = weight_init
        self.treatment_name = treatment_name

        self.net = treatment_regressor(network_name=network_name,
                                       input_dim=input_dim,
                                       hidden_sizes=hidden_sizes,
                                       num_treatments=num_treatments,
                                       dropout=dropout,
                                       activation=activation,
                                       device = device,
                                       weight_init=weight_init)

        if not is_subnetwork:
            self.get_optim()

    def loss(self, output, true_values):
        t_pred = output
        t_true = true_values[self.treatment_name]

        assert t_true is not None, 'treatment is None'
        loss = self.net.loss(t_pred, t_true)
        loss = loss.mean()
        loss_dict = {'loss': loss}

        return loss_dict

    def predict(self, X_test, load_model: str='best'):
        self.load(load_model)
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
        p_t_pred = self({'x':X_test}).probs
        if self.num_treatments == 2:
            t_pred = p_t_pred > 0.5
        else:
            t_pred = p_t_pred.argmax(dim=1)
        return {'p_t_pred': p_t_pred, 't_pred': t_pred}

    def summary(self, X_df, y):
        if self.network_name != 'linear':
            raise NotImplementedError('Summary only implemented for linear models')
        results_df = self.net.summary(X_df, y)
        print(results_df)

