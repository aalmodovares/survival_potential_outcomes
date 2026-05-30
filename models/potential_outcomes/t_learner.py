import torch
from torch import nn
from models.wrapper import Wrapper
from models.networks.regressors import outcome_regressor
class T_learner(Wrapper):
    def __init__(self, network_name,
                 input_dim, hidden_sizes, outcome_dist, num_treatments,
                 experiment, model_path,
                 device,
                 weight_init=None, dropout=0.0, activation='relu',
                 save=True, limit_variance=False, min_std=None, max_std=None,
                 fix_variance=False, fixed_std=None,
                 limit_shape=False, min_shape=None, max_shape=None,
                 limit_scale=False, min_scale=None, max_scale=None,
                 kan_params=None,
                 optim_config=None, wandblog=False,
                 is_subnetwork=False):

        super(T_learner, self).__init__('t-learner', device, experiment, model_path, save, optim_config, wandblog)

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.outcome_dist = outcome_dist
        self.num_treatments = num_treatments
        self.dropout = dropout
        self.activation = activation
        self.limit_variance = limit_variance
        self.limit_shape = limit_shape
        self.limit_scale = limit_scale
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fix_variance = fix_variance
        self.weight_init = weight_init

        if self.outcome_dist == 'not-specified':
            self.criterion = 'mse'
        else:
            self.criterion = 'log_prob'

        self.net = nn.ModuleList()
        for _ in range(num_treatments):
            self.net.append(outcome_regressor(network_name=network_name,
                                              input_dim=input_dim,
                                              hidden_sizes=hidden_sizes,
                                              outcome_dist=self.outcome_dist,
                                              dropout=dropout,
                                              activation=activation,
                                              device=device,
                                              limit_variance=limit_variance,
                                              min_std=min_std,
                                              max_std=max_std,
                                              fix_variance=fix_variance,
                                              fixed_std=fixed_std,
                                              limit_shape = limit_shape,
                                              min_shape = min_shape,
                                              max_shape = max_shape,
                                              limit_scale = limit_scale,
                                              min_scale = min_scale,
                                              max_scale = max_scale,
                                              weight_init=self.weight_init,
                                              kan_params=kan_params))
        if not is_subnetwork:
            self.get_optim()

    def forward(self, x_dict):
        x = x_dict['x']
        return [net(x) for net in self.net]

    def loss(self, output, true_values):
        # true_values = {'y': y, 't':t, 'c':c}
        y_true = true_values['y']
        t_true = true_values['t']
        c_true = true_values['c']
        assert t_true is not None, 'treatment is None'
        loss = 0.0
        for i, y_dist in enumerate(output):
            if self.criterion =='log_prob':
                log_prob_i = self.net[i].log_prob(y_dist, y_true, c_true)
                log_prob_i = log_prob_i*(t_true.view(-1,1) == i).int()
                if torch.sum(torch.isnan(log_prob_i)):  # TODO: THIS IS FOR DEBUGGING, REMOVE THIS LATER
                    print(f'NaN in log_prob_i {torch.nonzero(torch.isnan(log_prob_i))}')
                loss -= log_prob_i.mean()
            else:
                se = self.net[i].squared_error(y_dist, y_true)
                se = se*(t_true.view(-1,1) == i).int()
                loss += se.mean()

        loss_dict = {'loss': loss}
        return loss_dict

    @torch.no_grad()
    def predict(self, X_test, load_model: str='best'):
        self.load(load_model)
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
        y_dist_list = self({'x':X_test})

        return {'y_dist_list': y_dist_list}
