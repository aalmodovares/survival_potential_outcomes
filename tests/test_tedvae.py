import pytest
import numpy as np
from models.potential_outcomes import TEDVAE
from sklearn.model_selection import train_test_split
import torch
import os
# CREATE DATA FOR TESTING
from tests.synthetic_data import get_synthetic_test_dataset

@pytest.mark.parametrize("learner", ['tedvae'])
@pytest.mark.parametrize("outcome_dist", ['gaussian'])
@pytest.mark.parametrize("network_name", ['mlp'])
def test_learner(learner, outcome_dist, network_name):
    input_dim = 2
    hidden_sizes = [10, 10]
    outcome_dist = outcome_dist
    num_treatments = 2
    experiment = 'test'
    model_path = '/test_models/checkpoints'
    os.makedirs(model_path, exist_ok=True)
    dropout = 0.1
    activation = 'relu'
    device = 'cpu'
    save = False  # TODO: also set to False, to prevent test failure due to model path error (see test_potential_outcomes.py)
    limit_variance = False
    fix_variance = True
    limit_shape = False
    optim_config = None
    wandblog = False

    data = get_synthetic_test_dataset()
    mask = np.zeros_like(data[['x1', 'x2']].values)
    # define random datapoints missing (1 in the mask)
    mask = np.random.binomial(1, 0.1, size=mask.shape)
    data_types = ['continuous', 'continuous']
    train_data, test_data, train_mask, test_mask = train_test_split(data, mask, test_size=0.2)
    test_data, val_data, test_mask, val_mask = train_test_split(test_data, test_mask,  test_size=0.5)

    X_train, y_train, t_train = train_data[['x1', 'x2']].values, train_data['y'].values, train_data['t'].values
    X_val, y_val, t_val = val_data[['x1', 'x2']].values, val_data['y'].values, val_data['t'].values
    X_test, y_test, t_test = test_data[['x1', 'x2']].values, test_data['y'].values, test_data['t'].values

    model = TEDVAE(input_dim = input_dim,
                       encoder_name=network_name, decoder_name =network_name,
                       treatment_regressor_name = network_name, outcome_regressor_name = network_name,
                       latent_dim_t = 10, latent_dim_y = 10, latent_dim_c= 10,
                       hidden_sizes_decoder=hidden_sizes, hidden_sizes_encoder=hidden_sizes,
                       hidden_sizes_treatment_regressor=hidden_sizes, hidden_sizes_outcome_regressor=hidden_sizes,
                       outcome_dist=outcome_dist, num_treatments=num_treatments,
                       latent_dist='gaussian', data_types = data_types,
                       loss_weights = None,
                       experiment=experiment, model_path=model_path,
                       dropout=0.0,
                       activation_decoder='relu',
                       activation_treatment_regressor='relu',
                       activation_outcome_regressor='relu',
                       activation_encoder='relu', device=device,
                       save=save,
                       optim_config=optim_config, wandblog=wandblog
                       )

    model.fit(X_train=X_train, y_train=y_train, t_train=t_train, mask_train = train_mask,
                  X_val=X_val, y_val=y_val, t_val=t_val, mask_val = val_mask,
                  epochs=10)

    #assert that the output of the forwards are


    output = model({'x': torch.tensor(X_test, dtype=torch.float)})
    assert isinstance(output, dict)


    # check that the model can predict
    model.predict(X_test)
    # check that the prediction is a dict, in which y_dist_list is a list of distributions of size (N, 1), where N is the number of samples
    pred = model.predict(X_test)
    assert isinstance(pred, dict)
    assert 'y_dist_list' in pred
    assert isinstance(pred['y_dist_list'], list)
    assert pred['y_dist_list'][0].mean.shape[0] == X_test.shape[0]

    # check that the model can be loaded
    model.load()

# test_learner('t-learner')