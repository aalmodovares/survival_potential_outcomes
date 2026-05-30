import pytest

import numpy as np
import torch

from models.potential_outcomes.regressors_models import Outcome_Model, Treatment_Model

# @pytest.mark.parametrize("network_name", ['mlp', 'linear', 'KAN'])
# @pytest.mark.parametrize("num_treatments", [2, 3])
# def test_treatment_regressor(network_name, num_treatments):
#
#     x_values = torch.tensor(np.random.rand(10, 2), dtype=torch.float32)
#     treatment_values = torch.tensor(np.random.randint(0, num_treatments-1, 10), dtype=torch.float32)
#     y_values = torch.tensor(np.random.rand(10, 1), dtype=torch.float32)
#
#     t_model = Treatment_Model(network_name=network_name,
#                                 input_dim=2,
#                                 hidden_sizes=[10, 10],
#                                 num_treatments=num_treatments,
#                                 experiment='none',
#                                 model_path='../models/checkpoints',
#                                 activation='relu',
#                                 dropout=0.1,
#                                 device='cpu',
#                                 save=False)
#
#     output = t_model({'x': x_values, 't': treatment_values})
#
#     # assert output type is a distribution
#     assert hasattr(output, 'mean')
#
#     #loss
#     loss = t_model.loss(output, {'t': treatment_values})
#     assert 'loss' in loss.keys()
#
#
#     # train
#     t_model.fit(X_train=x_values, y_train = y_values, t_train = treatment_values)
#
#
#     # predict
#     t_pred = t_model.predict(x_values)


@pytest.mark.parametrize("network_name", ['mlp', 'linear', 'KAN'])
@pytest.mark.parametrize("outcome_dist", ['gaussian', 'bernoulli', 'negative-binomial'])
@pytest.mark.parametrize("censoring", [True, False])
def test_outcome_regressor(network_name, outcome_dist, censoring):

    x_values = torch.tensor(np.random.rand(10, 2), dtype=torch.float32)
    t_values = torch.tensor(np.random.randint(0, 1, 10), dtype=torch.float32)
    if outcome_dist == 'bernoulli':
        y_values = torch.tensor(np.random.randint(0, 1, 10), dtype=torch.float32)
    elif outcome_dist == 'gaussian':
        y_values = torch.tensor(np.random.rand(10, 1), dtype=torch.float32)
    elif outcome_dist == 'weibull':
        y_values = torch.tensor(np.random.rand(10, 2), dtype=torch.float32)
    elif outcome_dist == 'gumbel':
        y_values = torch.tensor(np.random.rand(10, 2), dtype=torch.float32)
    elif outcome_dist == 'not-specified':
        y_values = torch.tensor(np.random.rand(10, 1), dtype=torch.float32)

    else:
        y_values = torch.tensor(np.random.randint(0, 5, 10), dtype=torch.float32)

    c_values = torch.tensor(np.random.randint(0, 1, 10), dtype=torch.float32) if censoring else None

    o_model = Outcome_Model(network_name=network_name,
                            input_dim = 2,
                            hidden_sizes=[10, 10],
                            outcome_dist = outcome_dist,
                            experiment = 'none_o',
                            model_path = '../models/checkpoints',
                            dropout=0.1,
                            activation = 'relu',
                            device='cpu',
                            save=False)

    # train
    if outcome_dist == 'bernoulli':
        c_values = None
    o_model.fit(X_train = x_values, y_train = y_values, t_train = t_values, c_train = c_values, epochs=10)

    # predict
    y_pred = o_model.predict(x_values)