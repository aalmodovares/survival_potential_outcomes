import sys
import os
import optuna
import json
import wandb
import pickle
import time
from joblib import Parallel, delayed
from torch.distributions import Bernoulli

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from evaluation.metrics import *
from models.potential_outcomes import CFR, S_learner, T_learner, TARNet, TEDVAE
from models.utils import ite


def load_data(i, setting):

    assert setting in ['no_censor', 'nic', 'ic'], f'Setting {setting} not recognized'

    data_path = os.path.join(parent_dir, 'data', 'use_cases')

    with open(os.path.join(data_path, 'preprocessed_crc_data.pkl'), 'rb') as f:
        loaded_data = pickle.load(f)

    data = loaded_data[setting][str(i)]
    return data

def load_hyperparameters(model_name, setting, best=True, path='results_CRC_censoring/hyperparameters'):
    if best:
        file_name = os.path.join(path, f'{model_name}_{setting}_best.json')
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                best_params = json.load(f)
            return best_params
        else:
            print('Best hyperparameters not found, loading default hyperparameters')
    with open(os.path.join(path, f'{model_name}.json'), 'rb') as f:
        default_params = json.load(f)
    return default_params

def train_and_validate_model(model_name, model_obj, setting, i, hyperparameters, model_path, save_model, train_anyway, WANDB_MODE, disable_tqdm=False):

    data = load_data(i, setting)
    X_train, t_train, y_train, c_train = data['X_train'], data['t_train'], data['y_train'], data['c_train']
    X_val, t_val, y_val, c_val = data['X_val'], data['t_val'], data['y_val'], data['c_val']
    data_types, input_dim, num_treatments = data['data_types'], data['input_dim'], data['num_treatments']

    experiment = f'{model_name}_crc_censoring_{setting}_{i}'

    hyperparameters.update({'input_dim': input_dim,
                            'num_treatments': num_treatments,
                            'model_path': model_path,
                            'wandblog': True,
                            'save': save_model,
                            'experiment': experiment,
                            })
    if 'tedvae' in model_name:
        hyperparameters.update({'data_types': data_types})  # TEDVAE needs the data types for the input features

    train_flag = not 'last_model_' + experiment + '.pth' in os.listdir(model_path)
    train_flag = True if train_anyway else train_flag

    model = model_obj(**hyperparameters)
    model.disable_tqdm = disable_tqdm

    if train_flag:
        run = wandb.init(entity='causalgaps',
                         group='causalgaps',
                         project='crc_censoring',
                         mode=WANDB_MODE,
                         config={'setting': setting, 'model': model_name, **hyperparameters})

        if model_name == 'x-learner':
            model.fit(X_train, y_train, t_train, c_train, epochs_outcome=1000, epochs_ite=1000, epochs_propensity=1000,
                      batch_size=256)
        else:
            model.fit(X_train, y_train, t_train, c_train,
                      X_val=X_val, y_val=y_val, t_val=t_val, c_val=c_val,
                      epochs=1000, batch_size=1000)
        run.finish()
    else:
        print(f'Model {model_name} is already trained, skipping training...')

    return model

def train_individual_model(i, setting, model_name, model_obj, hyperparameters, disable_tqdm):  # Ancillary function for hyperparameter optimization, to train a model on a single dataset and return the target loss on the test set

    data = load_data(i, setting)
    X_test, t_test, y_test, c_test = data['X_test'], data['t_test'], data['y_test'], data['c_test']

    model = train_and_validate_model(model_name, model_obj, setting, i, hyperparameters, model_path=None, save_model=False, train_anyway=True, WANDB_MODE='disabled', disable_tqdm=disable_tqdm)

    test_dict = {'x': torch.tensor(X_test, dtype=torch.float), 'y': torch.tensor(y_test, dtype=torch.float), 't': torch.tensor(t_test, dtype=torch.float), 'c': torch.tensor(c_test, dtype=torch.float)}
    test_loss_dict = model.loss(model(test_dict), test_dict)

    if 'tedvae' in model_name:
        target_loss = test_loss_dict['loss_p_y']  # For tedvae, we optimize the loss of the outcome model
    elif model_name == 'cfrnet':
        target_loss = - test_loss_dict['log_prob']
    elif model_name == 't-learner' or model_name == 's-learner' or model_name == 'tarnet':
        target_loss = test_loss_dict['loss']
    else:
        raise ValueError(f'Model {model_name} not recognized for hyperparameter optimization')
    if torch.isnan(target_loss):
        return 1e6 # High enough if something goes wrong
    else:
        return target_loss.item()

def optimize_hyperparameters(model_name, model_obj, setting, save_path_hyperparameters, n_trials, n_jobs):

    if model_name not in ['s-learner', 't-learner', 'tarnet', 'cfrnet', 'sa-tedvae-m0', 'sa-tedvae-m1', 'sa-tedvae-m2']:
        print(f'Hyperparameter optimization not implemented for model {model_name}, skipping')
        return

    n_datasets = 10 # Number of datasets to be used per setting for hyperparameter optimization (we use only 10 to reduce the computational burden)

    def objective(trial):
        # Trial to be done
        hyperparameters = load_hyperparameters(model_name, setting, best=False, path=save_path_hyperparameters)
        # Prepare the hyperparameters to be optimized by Optuna, which is model-dependent!
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        hyperparameters['optim_config']['lr'] = learning_rate
        hyperparameters['dropout'] = dropout
        if model_name == 's-learner' or model_name == 't-learner':
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128])
            activation = trial.suggest_categorical('activation', ['relu', 'elu', 'tanh'])
            hyperparameters['hidden_sizes'] = [hidden_size] * n_layers
            hyperparameters['activation'] = activation
        if model_name == 'tarnet' or model_name == 'cfrnet':
            n_layers_phi = trial.suggest_int('n_layers_phi', 1, 3)
            hidden_size_phi = trial.suggest_categorical('hidden_size_phi', [16, 32, 64, 128])
            n_layers_y = trial.suggest_int('n_layers_y', 1, 3)
            hidden_size_y = trial.suggest_categorical('hidden_size_y', [16, 32, 64, 128])
            phi_dim = trial.suggest_categorical('phi_dim', [16, 32, 64, 128])
            activation = trial.suggest_categorical('activation', ['relu', 'elu', 'tanh'])
            hyperparameters['hidden_sizes_phi'] = [hidden_size_phi] * n_layers_phi
            hyperparameters['hidden_sizes_y'] = [hidden_size_y] * n_layers_y
            hyperparameters['phi_dim'] = phi_dim
            hyperparameters['activation'] = activation
        if model_name == 'cfrnet':
            beta_div = trial.suggest_float('beta_div', 0.0, 1.0)
            hyperparameters['beta_div'] = beta_div
        if 'tedvae' in model_name:
            latent_dim_t = trial.suggest_categorical('latent_dim_t', [5, 10, 20])
            latent_dim_c = trial.suggest_categorical('latent_dim_c', [5, 10, 20, 30])
            latent_dim_y = trial.suggest_categorical('latent_dim_y', [5, 10, 20])
            n_layers_encoder = trial.suggest_int('n_layers_encoder', 1, 3)
            hidden_size_encoder = trial.suggest_categorical('hidden_size_encoder', [16, 32, 64, 128])
            n_layers_decoder = trial.suggest_int('n_layers_decoder', 1, 3)
            hidden_size_decoder = trial.suggest_categorical('hidden_size_decoder', [16, 32, 64, 128])
            n_layers_treatment_regressor = trial.suggest_int('n_layers_treatment_regressor', 0, 3)
            hidden_size_treatment_regressor = trial.suggest_categorical('hidden_size_treatment_regressor', [16, 32, 64, 128])
            n_layers_outcome_regressor = trial.suggest_int('n_layers_outcome_regressor', 1, 3)
            hidden_size_outcome_regressor = trial.suggest_categorical('hidden_size_outcome_regressor', [16, 32, 64, 128])
            w_disentanglement_t = trial.suggest_float('w_disentanglement_t', 0.1, 10.0, log=True)
            w_disentanglement_y = trial.suggest_float('w_disentanglement_y', 0.1, 10.0, log=True)
            w_kl = trial.suggest_float('w_kl', 0.1, 10.0, log=True)
            w_reconstruction = trial.suggest_float('w_reconstruction', 0.1, 10.0, log=True)
            w_model_t = trial.suggest_float('w_model_t', 0.1, 10.0, log=True)
            w_model_y = trial.suggest_float('w_model_y', 0.1, 10.0, log=True)
            activation = trial.suggest_categorical('activation', ['relu', 'elu', 'tanh'])

            if model_name in ['sa-tedvae-m1', 'sa-tedvae-m2']:  # Model the censoring mechanism
                n_layers_censoring_model = trial.suggest_int('n_layers_censoring_model', 1, 3)
                hidden_size_censoring_model = trial.suggest_categorical('hidden_size_censoring_model', [16, 32, 64, 128])
                w_disentanglement_c = trial.suggest_float('w_disentanglement_c', 0.1, 10.0, log=True)
                w_model_c = trial.suggest_float('w_model_c', 0.1, 10.0, log=True)

            hyperparameters['latent_dim_t'] = latent_dim_t
            hyperparameters['latent_dim_c'] = latent_dim_c
            hyperparameters['latent_dim_y'] = latent_dim_y
            hyperparameters['hidden_sizes_encoder'] = [hidden_size_encoder] * n_layers_encoder
            hyperparameters['hidden_sizes_decoder'] = [hidden_size_decoder] * n_layers_decoder
            hyperparameters['hidden_sizes_treatment_regressor'] = [hidden_size_treatment_regressor] * n_layers_treatment_regressor
            hyperparameters['hidden_sizes_outcome_regressor'] = [hidden_size_outcome_regressor] * n_layers_outcome_regressor
            hyperparameters['loss_weights']['w_disentanglement_t'] = w_disentanglement_t
            hyperparameters['loss_weights']['w_disentanglement_y'] = w_disentanglement_y
            hyperparameters['loss_weights']['w_kl'] = w_kl
            hyperparameters['loss_weights']['w_reconstruction'] = w_reconstruction
            hyperparameters['loss_weights']['w_model_t'] = w_model_t
            hyperparameters['loss_weights']['w_model_y'] = w_model_y
            hyperparameters['activation_encoder'] = activation
            hyperparameters['activation_decoder'] = activation
            hyperparameters['activation_treatment_regressor'] = activation
            hyperparameters['activation_outcome_regressor'] = activation

            if model_name in ['sa-tedvae-m1', 'sa-tedvae-m2']:
                hyperparameters['hidden_sizes_censoring_model'] = [hidden_size_censoring_model] * n_layers_censoring_model
                hyperparameters['loss_weights']['w_disentanglement_c'] = w_disentanglement_c
                hyperparameters['loss_weights']['w_model_c'] = w_model_c
                hyperparameters['activation_censoring_model'] = activation

        # Do the optimization loop with the current hyperparameters and return the average loss across the datasets and settings
        losses = Parallel(verbose=10, n_jobs=n_jobs)(delayed(train_individual_model)(i=i, setting=setting, model_name=model_name, model_obj=model_obj, hyperparameters=hyperparameters, disable_tqdm=True) for i in range(n_datasets))
        avg_loss = np.mean(losses)
        if np.isnan(avg_loss):
            avg_loss = np.inf # High enough if something goes wrong
        return avg_loss

    # For hyperparamenter optimization, we use Optuna.
    study = optuna.create_study(direction='minimize') # Minimize because it is a loss
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))  # to prevent a valueerror from interrupting the optimization, which can happen for some hyperparameter combinations

    # Get the best hyperparameters and update the corresponding json file
    best_hyperparameters = study.best_params
    hyperparameters = load_hyperparameters(model_name, setting, best=False, path=save_path_hyperparameters)
    hyperparameters['dropout'] = best_hyperparameters['dropout']
    hyperparameters['optim_config']['lr'] = best_hyperparameters['learning_rate']
    if model_name == 's-learner' or model_name == 't-learner':
        hyperparameters['hidden_sizes'] = [best_hyperparameters['hidden_size']] * best_hyperparameters['n_layers']
        hyperparameters['activation'] = best_hyperparameters['activation']
    if model_name == 'tarnet' or model_name == 'cfrnet':
        hyperparameters['hidden_sizes_phi'] = [best_hyperparameters['hidden_size_phi']] * best_hyperparameters['n_layers_phi']
        hyperparameters['hidden_sizes_y'] = [best_hyperparameters['hidden_size_y']] * best_hyperparameters['n_layers_y']
        hyperparameters['phi_dim'] = best_hyperparameters['phi_dim']
        hyperparameters['activation'] = best_hyperparameters['activation']
    if model_name == 'cfrnet':
        hyperparameters['beta_div'] = best_hyperparameters['beta_div']
    if 'tedvae' in model_name:
        hyperparameters['latent_dim_t'] = best_hyperparameters['latent_dim_t']
        hyperparameters['latent_dim_c'] = best_hyperparameters['latent_dim_c']
        hyperparameters['latent_dim_y'] = best_hyperparameters['latent_dim_y']
        hyperparameters['hidden_sizes_encoder'] = [best_hyperparameters['hidden_size_encoder']] * best_hyperparameters['n_layers_encoder']
        hyperparameters['hidden_sizes_decoder'] = [best_hyperparameters['hidden_size_decoder']] * best_hyperparameters['n_layers_decoder']
        hyperparameters['hidden_sizes_treatment_regressor'] = [best_hyperparameters['hidden_size_treatment_regressor']] * best_hyperparameters['n_layers_treatment_regressor']
        hyperparameters['hidden_sizes_outcome_regressor'] = [best_hyperparameters['hidden_size_outcome_regressor']] * best_hyperparameters['n_layers_outcome_regressor']
        hyperparameters['loss_weights']['w_disentanglement_t'] = best_hyperparameters['w_disentanglement_t']
        hyperparameters['loss_weights']['w_disentanglement_y'] = best_hyperparameters['w_disentanglement_y']
        hyperparameters['loss_weights']['w_kl'] = best_hyperparameters['w_kl']
        hyperparameters['loss_weights']['w_reconstruction'] = best_hyperparameters['w_reconstruction']
        hyperparameters['loss_weights']['w_model_t'] = best_hyperparameters['w_model_t']
        hyperparameters['loss_weights']['w_model_y'] = best_hyperparameters['w_model_y']
        hyperparameters['activation_encoder'] = best_hyperparameters['activation']
        hyperparameters['activation_decoder'] = best_hyperparameters['activation']
        hyperparameters['activation_treatment_regressor'] = best_hyperparameters['activation']
        hyperparameters['activation_outcome_regressor'] = best_hyperparameters['activation']

        if model_name in ['sa-tedvae-m1', 'sa-tedvae-m2']:
            hyperparameters['hidden_sizes_censoring_model'] = [best_hyperparameters['hidden_size_censoring_model']] * best_hyperparameters['n_layers_censoring_model']
            hyperparameters['loss_weights']['w_disentanglement_c'] = best_hyperparameters['w_disentanglement_c']
            hyperparameters['loss_weights']['w_model_c'] = best_hyperparameters['w_model_c']
            hyperparameters['activation_censoring_model'] = best_hyperparameters['activation']

    # Save these best hyperparameters in a json file
    with open(os.path.join(save_path_hyperparameters, f'{model_name}_{setting}_best.json'), 'w') as f:
        json.dump(hyperparameters, f, ensure_ascii=False, indent=4)

def validate_model_on_dataset(setting, i, model_name, model_obj, save_path_hyperparameters, model_path, compute_tmle=False):
    # Load the model
    hyperparameters = load_hyperparameters(model_name, setting, best=True, path=save_path_hyperparameters)
    # We set train_anyway to False, because if the model has been trained already, we load its weights only
    model = train_and_validate_model(model_name, model_obj, setting, i, hyperparameters,
                                                      model_path=model_path, save_model=True, train_anyway=False,
                                                      WANDB_MODE='disabled', disable_tqdm=True)
    # VALIDATION AND TESTING
    data = load_data(i, setting)
    X_train, t_train, y_train, c_train = data['X_train'], data['t_train'], data['y_train'], data['c_train']
    X_test, t_test, y_test, c_test = data['X_test'], data['t_test'], data['y_test'], data['c_test']

    input_is_dist = True  # Flag to be used later for ITE computation (depends on whether we have a distribution or a point estimate)
    if model_name == 's-learner-linear':
        input_is_dist = False

    if 'tedvae' in model_name:
        n_latent_samples = 11  # Odd samples, for median selection
        p_trains = [model.predict(X_train, load_model='best') for _ in range(n_latent_samples)]  # Take 10 latent samples
        p_tests = [model.predict(X_test, load_model='best') for _ in range(n_latent_samples)]
        n_samples_train = X_train.shape[0]
        n_samples_test = X_test.shape[0]
        # Now, for each patient, we take the distribution that is the median of the means taking into account all treatments (i.e., we select a single latent sample)
        means_train = [p_train['y_dist_list'][0].mean.detach().cpu().numpy().squeeze() + p_train['y_dist_list'][1].mean.detach().cpu().numpy().squeeze() for p_train in p_trains]
        means_test = [p_test['y_dist_list'][0].mean.detach().cpu().numpy().squeeze() + p_test['y_dist_list'][1].mean.detach().cpu().numpy().squeeze() for p_test in p_tests]
        outcome_scale_train = [torch.zeros((n_samples_train, 1)), torch.zeros((n_samples_train, 1))]
        outcome_concentration_train = [torch.zeros((n_samples_train, 1)), torch.zeros((n_samples_train, 1))]
        outcome_scale_test = [torch.zeros((n_samples_test, 1)), torch.zeros((n_samples_test, 1))]
        outcome_concentration_test = [torch.zeros((n_samples_test, 1)), torch.zeros((n_samples_test, 1))]
        propensity_score_train = torch.zeros((n_samples_train, 1))
        propensity_score_test = torch.zeros((n_samples_test, 1))
        if model_name == 'sa-tedvae-m1':
            censor_train = [torch.zeros((n_samples_train, 1)), torch.zeros((n_samples_train, 1))]
            censor_test = [torch.zeros((n_samples_test, 1)), torch.zeros((n_samples_test, 1))]
        elif model_name == 'sa-tedvae-m2':
            censor_scale_train = [torch.zeros((n_samples_train, 1)), torch.zeros((n_samples_train, 1))]
            censor_concentration_train = [torch.zeros((n_samples_train, 1)), torch.zeros((n_samples_train, 1))]
            censor_scale_test = [torch.zeros((n_samples_test, 1)), torch.zeros((n_samples_test, 1))]
            censor_concentration_test = [torch.zeros((n_samples_test, 1)), torch.zeros((n_samples_test, 1))]
        for i in range(len(X_train)):
            median_index = np.argsort([m[i] for m in means_train])[n_latent_samples // 2]
            outcome_scale_train[0][i] = p_trains[median_index]['y_dist_list'][0].scale[i]
            outcome_concentration_train[0][i] = p_trains[median_index]['y_dist_list'][0].concentration[i]
            outcome_scale_train[1][i] = p_trains[median_index]['y_dist_list'][1].scale[i]
            outcome_concentration_train[1][i] = p_trains[median_index]['y_dist_list'][1].concentration[i]
            propensity_score_train[i] = p_trains[median_index]['propensity_score_pred'][i]
            if model_name == 'sa-tedvae-m1':
                censor_train[0][i] = p_trains[median_index]['c_dist_list'][0].probs[i]
                censor_train[1][i] = p_trains[median_index]['c_dist_list'][1].probs[i]
            elif model_name == 'sa-tedvae-m2':
                censor_scale_train[0][i] = p_trains[median_index]['c_dist_list'][0].scale[i]
                censor_concentration_train[0][i] = p_trains[median_index]['c_dist_list'][0].concentration[i]
                censor_scale_train[1][i] = p_trains[median_index]['c_dist_list'][1].scale[i]
                censor_concentration_train[1][i] = p_trains[median_index]['c_dist_list'][1].concentration[i]
        for i in range(len(X_test)):
            median_index = np.argsort([m[i] for m in means_test])[n_latent_samples // 2]
            outcome_scale_test[0][i] = p_tests[median_index]['y_dist_list'][0].scale[i]
            outcome_concentration_test[0][i] = p_tests[median_index]['y_dist_list'][0].concentration[i]
            outcome_scale_test[1][i] = p_tests[median_index]['y_dist_list'][1].scale[i]
            outcome_concentration_test[1][i] = p_tests[median_index]['y_dist_list'][1].concentration[i]
            propensity_score_test[i] = p_tests[median_index]['propensity_score_pred'][i]
            if model_name == 'sa-tedvae-m1':
                censor_test[0][i] = p_tests[median_index]['c_dist_list'][0].probs[i]
                censor_test[1][i] = p_tests[median_index]['c_dist_list'][1].probs[i]
            elif model_name == 'sa-tedvae-m2':
                censor_scale_test[0][i] = p_tests[median_index]['c_dist_list'][0].scale[i]
                censor_concentration_test[0][i] = p_tests[median_index]['c_dist_list'][0].concentration[i]
                censor_scale_test[1][i] = p_tests[median_index]['c_dist_list'][1].scale[i]
                censor_concentration_test[1][i] = p_tests[median_index]['c_dist_list'][1].concentration[i]

        predictions_train = {'y_dist_list': [Weibull(concentration=outcome_concentration_train[0], scale=outcome_scale_train[0]),
                                             Weibull(concentration=outcome_concentration_train[1], scale=outcome_scale_train[1])],
                             'propensity_score_pred': propensity_score_train}
        predictions_test = {'y_dist_list': [Weibull(concentration=outcome_concentration_test[0], scale=outcome_scale_test[0]),
                                            Weibull(concentration=outcome_concentration_test[1], scale=outcome_scale_test[1])],
                            'propensity_score_pred': propensity_score_test}

        if model_name == 'sa-tedvae-m1':
            predictions_train['c_dist_list'] = [Bernoulli(probs=censor_train[0]), Bernoulli(probs=censor_train[1])]
            predictions_test['c_dist_list'] = [Bernoulli(probs=censor_test[0]), Bernoulli(probs=censor_test[1])]

        elif model_name == 'sa-tedvae-m2':
            predictions_train['c_dist_list'] = [Weibull(concentration=censor_concentration_train[0], scale=censor_scale_train[0]),
                                             Weibull(concentration=censor_concentration_train[1], scale=censor_scale_train[1])]
            predictions_test['c_dist_list'] = [Weibull(concentration=censor_concentration_test[0], scale=censor_scale_test[0]),
                                            Weibull(concentration=censor_concentration_test[1], scale=censor_scale_test[1])]

        metrics = {'outcome_scale_train': [o.detach().cpu().numpy() for o in outcome_scale_train],
                   'outcome_concentration_train': [o.detach().cpu().numpy() for o in outcome_concentration_train],
                   'outcome_scale_test': [o.detach().cpu().numpy() for o in outcome_scale_test],
                   'outcome_concentration_test': [o.detach().cpu().numpy() for o in outcome_concentration_test],
                   'propensity_score_train': propensity_score_train.detach().cpu().numpy(),
                   'propensity_score_test': propensity_score_test.detach().cpu().numpy()}
        if model_name == 'sa-tedvae-m1':
            metrics['c_train'] = [c.detach().cpu().numpy() for c in censor_train]
            metrics['c_test'] = [c.detach().cpu().numpy() for c in censor_test]
        elif model_name == 'sa-tedvae-m2':
            metrics['censor_scale_train'] = [c.detach().cpu().numpy() for c in censor_scale_train]
            metrics['censor_concentration_train'] = [c.detach().cpu().numpy() for c in censor_concentration_train]
            metrics['censor_concentration_test'] = [c.detach().cpu().numpy() for c in censor_concentration_test]
            metrics['censor_scale_test'] = [c.detach().cpu().numpy() for c in censor_scale_test]
    else:
        predictions_train = model.predict(X_train, load_model='best')
        predictions_test = model.predict(X_test, load_model='best')

    if model_name == 'x-learner':
        ite_train = predictions_train['ite_pred']
        ite_test = predictions_test['ite_pred']
        metrics = {}
    else:
        ite_train = ite(predictions_train['y_dist_list'], scaler=None, distribution_name='weibull', input_is_dist=input_is_dist)
        ite_test = ite(predictions_test['y_dist_list'], scaler=None, distribution_name='weibull', input_is_dist=input_is_dist)
        if model_name not in ['sa-tedvae-m0', 'sa-tedvae-m1', 'sa-tedvae-m2', 's-learner-linear']:
            metrics = {'outcome_scale_train': [p.scale.detach().cpu().numpy() for p in predictions_train['y_dist_list']],
                       'outcome_concentration_train': [p.concentration.detach().cpu().numpy() for p in predictions_train['y_dist_list']],
                       'outcome_scale_test': [p.scale.detach().cpu().numpy() for p in predictions_test['y_dist_list']],
                       'outcome_concentration_test': [p.concentration.detach().cpu().numpy() for p in predictions_test['y_dist_list']]}

    if model_name == 's-learner-linear': # Only model that does not predict a distribution...
        metrics = {}
    else:
        met_sur = get_survival_metrics(metrics, dist='weibull',
                                       t_train=t_train, t_test=t_test,
                                       y_train=y_train, y_test=y_test,
                                       c_train=c_train, c_test=c_test)
        metrics.update(met_sur)

    metrics['ate_predicted_train'] = np.mean(ite_train)
    metrics['ate_predicted_test'] = np.mean(ite_test)
    metrics['ite_predicted_train'] = ite_train
    metrics['ite_predicted_test'] = ite_test
    metrics['ate_gt'] = data['ate_ground_truth']
    metrics['ate_obs'] = data['ate_obs']
    metrics['ate_error_train'] = np.abs(metrics['ate_predicted_train'] - metrics['ate_gt'])
    metrics['ate_error_test'] = np.abs(metrics['ate_predicted_test'] - metrics['ate_gt'])

    metrics['censor_prop_train'] = np.mean(c_train)
    metrics['censor_prop_test'] = np.mean(c_test)

    if compute_tmle:
        max_y = min((500, np.percentile(y_train, 95)*1.2))  # We cap the time to 500 or the 95th percentile of the observed times, to avoid very long times that can cause numerical issues in TMLE estimation
        pred_tmle, pred_tmle_g_comp, t_tmle = obtain_tmle(X_train, t_train, y_train, c_train, max_y=max_y, alpha=0.05, n_points=25, verbose=0)
        metrics['pred_tmle'] = pred_tmle
        metrics['pred_tmle_g_comp'] = pred_tmle_g_comp
        metrics['t_tmle'] = t_tmle
    else:
        metrics['pred_tmle'] = None
        metrics['pred_tmle_g_comp'] = None
        metrics['t_tmle'] = None

    if model_name in ['sa-tedvae-m1', 'sa-tedvae-m2']:
        metrics['censor_prop_train_model'] = [p.mean.detach().cpu().numpy() for p in
                                              predictions_train['c_dist_list']]
        metrics['censor_prop_test_model'] = [p.mean.detach().cpu().numpy() for p in predictions_test['c_dist_list']]

    return metrics

def get_survival_metrics(input_dict, t_train, t_test, y_train, y_test, c_train, c_test, dist='weibull',
                         max_time=100, n_time=100):
    assert dist == 'weibull', 'Currently, only the Weibull distribution is supported for survival metrics computation'
    time_vector = np.linspace(0, max_time, n_time)
    survival_metrics = obtain_c_index(input_dict, t_train=t_train, t_test=t_test, y_train=y_train, y_test=y_test,
                                      c_train=c_train, c_test=c_test)
    survival_metrics['time_vector'] = time_vector.flatten()
    return survival_metrics


if __name__== '__main__':

    train_anyway = not True  # If set to True, it trains all models even if the trained model is already in the model_path. If False, it only trains the models that are not already in the model_path, and loads the ones that are already there.
    find_best_hyperparams = not True # Careful with this, as this takes a lot of time! If set to True, it performs hyperparameter optimization for all models before training and evaluating them. If parameters have already been optimized, it is not needed to rerun this, as they are stored
    compute_tmle = True  # Careful as this takes a lot of time
    WANDB_MODE = 'disabled' # options: 'disabled', 'online', 'offline'
    n_bootstraps = 100  # Number of bootstrapping splits
    num_jobs = 10  # Number of parallel jobs for hyperparameter optimization and training (set to 1 for no paralelization)
    smetrics = ['ate_error_train', 'ate_error_test', 'ate_predicted_train', 'ate_predicted_test', 'ci_td', 'ci_ipcw', 'ci_censored', 'ibs', 'rcll']  # Metrics to be plotted in the final results

    # This is done to visualize the full output of the metrics in the validation step
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Build the results path
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_CRC_censoring')
    save_path_hyperparameters = os.path.join(save_path, 'hyperparameters')
    model_path = os.path.join(save_path, 'models')
    plots_path = os.path.join(save_path, 'plots')
    results_path = os.path.join(save_path, 'results')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_hyperparameters, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # Models and settings be used
    models = {
        's-learner': S_learner,
        't-learner': T_learner,
        'tarnet': TARNet,
        'cfrnet': CFR,
        'sa-tedvae-m0': TEDVAE,
        'sa-tedvae-m1': TEDVAE,
        'sa-tedvae-m2': TEDVAE,
    }
    settings = ['no_censor', 'ic']  # Two censoring paradigms

    # HYPERPARAMETER TUNING: CAREFUL WITH THIS, AS THIS CAN TAKE A LOT OF TIME
    if find_best_hyperparams:
        for model_name, model_obj in models.items():
            for setting in settings:
                optimize_hyperparameters(model_name, model_obj, setting, save_path_hyperparameters, n_trials=200, n_jobs=num_jobs)

    # TRAIN MODELS AND SAVE THEIR WEIGHTS: DO THIS ONLY ONCE BEFORE EVALUATION (WE CAN EVALUATE USING THE SAVED MODELS)
    if train_anyway:
        def train_model_f(setting, i, model_name, model_obj, max_trials=5):
            hyperparameters = load_hyperparameters(model_name, setting, best=True, path=save_path_hyperparameters)
            trial = 0
            t_elapsed = None
            while trial < max_trials: # In case convergence issues appear during training, retry (sometimes training fails... retry, as it tends to be due to random factors)
                try:
                    t0 = time.time()
                    model = train_and_validate_model(model_name, model_obj, setting, i, hyperparameters,
                                                     model_path=model_path, save_model=True, train_anyway=train_anyway, WANDB_MODE=WANDB_MODE, disable_tqdm=True)
                    t_elapsed = time.time() - t0
                    trial = max_trials + 1  # If we reach this line, it means that the training was successful, so we can exit the loop by setting the trial to a value higher than max_trials

                except Exception as e:
                    print(f'Error while training model {model_name} on dataset {i} and setting {setting}, trial {trial}, retrying. Exception was {e}')
                    trial += 1
            return t_elapsed
        time_data = {}
        for model_name, model_obj in models.items():
            print(f'Training model {model_name} on {n_bootstraps} datasets and settings {settings}...')
            all_times = Parallel(verbose=10, n_jobs=num_jobs)(delayed(train_model_f)(setting=setting, i=i, model_name=model_name, model_obj=model_obj) for setting in settings for i in range(n_bootstraps))
            time_data[model_name] = all_times
        with open(os.path.join(results_path, 'training_times.pkl'), 'wb') as f:
            pickle.dump(time_data, f, protocol=pickle.HIGHEST_PROTOCOL)


    # VALIDATE MODELS AND COMPUTE METRICS
    results = {setting: {} for setting in settings}
    list_for_csv_results = []
    for setting in settings:
        for j, (model_name, model_obj) in enumerate(models.items()):
            print(f'Validating for model {model_name} on setting {setting}...')
            results[setting][model_name] = {}
            o = Parallel(verbose=10, n_jobs=num_jobs)(delayed(validate_model_on_dataset)(setting=setting,
                                                                                         i=i,
                                                                                         model_name=model_name,
                                                                                         model_obj=model_obj,
                                                                                         save_path_hyperparameters=save_path_hyperparameters,
                                                                                         model_path=model_path,
                                                                                         compute_tmle=(j == 0) if compute_tmle else False) # We compute TMLE only on the first 10 datasets to reduce the computational burden, as it is very time-consuming
                                                      for i in range(n_bootstraps))
            for r in o:
                if len(results[setting][model_name]) == 0:
                    for key in r.keys():
                        results[setting][model_name][key] = [] # Add the keys of r to the results dictionary
                for key in r.keys():
                    results[setting][model_name][key].append(r[key])

            model_results_dict = {'Model': model_name,
                                  'Setting': setting,
                                  }

            print(f'Results for model {model_name} on setting {setting}...')

            for m in smetrics:
                if m in results[setting][model_name].keys():
                    print(f'{m} for model {model_name} on setting {setting} (median - mean - std): {np.median(results[setting][model_name][m])}, {np.mean(results[setting][model_name][m])}, {np.std(results[setting][model_name][m])}')
                    model_results_dict[m + '_median'] = np.median(results[setting][model_name][m])
                    model_results_dict[m + '_mean'] = np.mean(results[setting][model_name][m])
                    model_results_dict[m + '_std'] = np.std(results[setting][model_name][m])
                else:
                    print(f'{m} for model {model_name} on setting {setting} not found in results, skipping...')
                    model_results_dict[m + '_median'] = None
                    model_results_dict[m + '_mean'] = None
                    model_results_dict[m + '_std'] = None

            list_for_csv_results.append(model_results_dict)


        # Formatting function
        def custom_format(x):
            if isinstance(x, (int, float)):
                if abs(x) >= 1000:
                    return f"{x:.2e}"  # Scientific notation for large numbers
                else:
                    return f"{x:.2f}"  # Fixed-point notation for small numbers
            else:
                return str(x)

        df = pd.DataFrame(list_for_csv_results)
        df_formatted = df.map(custom_format)
        df.to_csv(os.path.join(results_path, f'results_{setting}.csv'), index=False)

    # Save the results to be used later
    with open(os.path.join(results_path, 'results_tmle.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(df_formatted)  # Print it once with all results
