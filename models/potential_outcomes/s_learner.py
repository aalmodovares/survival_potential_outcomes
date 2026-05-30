import pandas as pd
import torch
from torch import nn
from models.wrapper import Wrapper
from models.networks.regressors import outcome_regressor


class S_learner(Wrapper):
        def __init__(self, network_name,
                     input_dim, hidden_sizes, outcome_dist, num_treatments,
                     experiment, model_path,
                     device,
                     weight_init=None, dropout=0.0, activation='relu',
                     save=True, limit_variance=False, fix_variance=False, limit_shape=False, limit_scale=False, min_shape=None, max_shape=None, min_scale=None, max_scale=None,
                     kan_params=None,
                     optim_config=None,
                     wandblog=False, is_subnetwork=False):
            super(S_learner, self).__init__('s-learner', device, experiment, model_path, save, optim_config, wandblog)

            self.network_name = network_name
            self.input_dim = input_dim + 1 #including the treatment
            self.hidden_sizes = hidden_sizes
            self.outcome_dist = outcome_dist
            self.num_treatments = num_treatments
            self.dropout = dropout
            self.activation = activation
            self.limit_variance = limit_variance
            self.limit_shape = limit_shape
            self.min_shape = min_shape
            self.max_shape = max_shape
            self.limit_scale = limit_scale
            self.min_scale = min_scale
            self.max_scale = max_scale
            self.fix_variance = fix_variance
            self.weight_init = weight_init

            if outcome_dist == 'not-specified':
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
                                         limit_variance = self.limit_variance,
                                         fix_variance = self.fix_variance,
                                         limit_shape = self.limit_shape,
                                         min_shape = self.min_shape,
                                         max_shape = self.max_shape,
                                         limit_scale = self.limit_scale,
                                         min_scale = self.min_scale,
                                         max_scale = self.max_scale,
                                         weight_init = self.weight_init,
                                         kan_params=kan_params)
            if not is_subnetwork:
                self.get_optim()


        def forward(self, x_dict):
            x = x_dict['x']
            t = x_dict['t']
            t = t.reshape(-1, 1) if len(t.shape) == 1 else t
            xt = torch.cat([x, t], dim=1)
            return self.net(xt)

        def loss(self, output, true_values):
            y_true = true_values['y']
            c_true = true_values['c']
            y_dist = output

            if self.criterion == 'log_prob':
                log_prob = self.net.log_prob(y_dist, y_true, c_true)
                loss = -log_prob.mean()
            else:
                loss = self.net.mse(y_dist, y_true, c_true)

            loss_dict = {'loss': loss}
            return loss_dict

        def on_training_start(self):
            self.net.on_training_start()

        def on_epoch_start(self, n_epochs, train_dict):
            x = train_dict['x']
            t = train_dict['t']
            t = t.reshape(-1, 1) if len(t.shape) == 1 else t
            xt = torch.cat([x, t], dim=1)
            self.net.on_epoch_start(self.current_epoch, n_epochs, xt)

        def on_epoch_end(self):
            self.net.on_epoch_end()

        def compute_regularization(self):
            return self.net.compute_regularization()

        @torch.no_grad()
        def predict(self, X_test, load_model: str='best'):
            self.load(load_model)
            self.eval()
            X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
            y_dist_list = []
            for i in range(self.num_treatments):
                t = torch.ones(X_test.shape[0], 1)*i
                t = t.to(self.device)
                y_dist = self({'x': X_test, 't': t})
                if self.outcome_dist == 'not-specified':
                    y_dist_list.append(y_dist)
                else:
                    y_dist_list.append(y_dist)
            return {'y_dist_list': y_dist_list}


        def summary(self, x_train, t_train, y_train, x_columns):
            # print the summary of the network, only for linear networks
            if self.network_name != 'linear':
                raise ValueError('Summary is only available for linear networks')

            if self.outcome_dist != 'not-specified' and self.outcome_dist != 'binary' and self.outcome_dist != 'bernoulli':
                raise ValueError('Summary is only available for linear networks with not specified or binary outcome distribution')


            X_df = pd.DataFrame(x_train, columns=x_columns)
            X_df['treatment'] = t_train # treatment should be in the last column
            results_df = self.net.net.summary(X_df, y_train)

            print(results_df)

        @torch.no_grad()
        def predict_params(self, X_test, load_model: str = 'best'):

            if load_model is not None:
                self.load(load_model)

            if self.outcome_dist != 'weibull':
                raise ValueError('Survival curves can only be computed for Weibull outcome distribution')
            self.eval()

            X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)

            y_dist_list = []
            for i in range(self.num_treatments):
                t = torch.ones(X_test.shape[0], 1)*i
                y_dist = self({'x': X_test, 't': t})

                y_dist_list.append(y_dist)

            shape = [y.concentration for y in y_dist_list]
            scale = [y.scale for y in y_dist_list]

            return shape, scale






