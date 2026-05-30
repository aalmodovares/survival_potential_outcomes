import sys
import os
import optuna
from joblib import Parallel, delayed
import json
import wandb
import pickle
import time
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from scipy.stats import wilcoxon
from torch.distributions import Normal, Bernoulli, Exponential, Weibull

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from evaluation.metrics import *
from models.potential_outcomes import CFR, S_learner, T_learner, TARNet, TEDVAE
from models.utils import ite

def load_data(i, setting):

    data_path = os.path.join(parent_dir, 'data', 'IHDP', 'IHDP')

    train_df = pd.read_csv(f'{data_path}{setting}/ihdp_npci_train_{i}.csv')
    test_df = pd.read_csv(f'{data_path}{setting}/ihdp_npci_test_{i}.csv')

    with open(f'{data_path}{setting}/ihdp_npci_params_{i}.pkl', 'rb') as f:
        ground_truth_params = pickle.load(f)

    # DATA PREPROCESSING
    num_val_data = 100  # keep 100 data points for internal validation (train + val, test). Note that the scalers are based on the training data only
    # rest 1 to x14 because it is coded as 1 and 2 instead of 0 and 1
    train_df.loc[:, 'x14'] = train_df.loc[:, 'x14'] - 1
    test_df.loc[:, 'x14'] = test_df.loc[:, 'x14'] - 1
    x_columns = [f'x{i}' for i in range(1, 26)]
    data_types = ['continuous'] * 6 + ['binary'] * 19  # Needed for TEDVAE
    std_time = train_df.loc[:, 'survival_factual'].std()
    train_df.loc[:, 'survival_factual'] = train_df.loc[:, 'survival_factual'] / std_time
    test_df.loc[:, 'survival_factual'] = test_df.loc[:, 'survival_factual'] / std_time

    X_train = train_df[x_columns].values[:-num_val_data]
    X_val = train_df[x_columns].values[-num_val_data:]
    X_test = test_df[x_columns].values

    t_train = train_df['treatment'].values[:-num_val_data]
    t_val = train_df['treatment'].values[-num_val_data:]
    t_test = test_df['treatment'].values

    y_train = train_df['y_factual'].values[:-num_val_data]
    y_val = train_df['y_factual'].values[-num_val_data:]
    y_test = test_df['y_factual'].values
    c_train = train_df['censoring_indicator'].values[:-num_val_data] if 'survival' in setting else None
    c_val = train_df['censoring_indicator'].values[-num_val_data:] if 'survival' in setting else None
    c_test = test_df['censoring_indicator'].values if 'survival' in setting else None

    # scale continous features
    scaler = StandardScaler()
    X_train[:, :6] = scaler.fit_transform(X_train[:, :6])
    X_val[:, :6] = scaler.transform(X_val[:, :6])
    X_test[:, :6] = scaler.transform(X_test[:, :6])

    return {'X_train': X_train, 't_train': t_train, 'y_train': y_train, 'c_train': c_train,
            'X_val': X_val, 't_val': t_val, 'y_val': y_val, 'c_val': c_val,
            'X_test': X_test, 't_test': t_test, 'y_test': y_test, 'c_test': c_test,
            'data_types': data_types, 'input_dim': 25, 'num_treatments': 2,
            'train_df': train_df, 'test_df': test_df, 'num_val_data': num_val_data, 'x_columns': x_columns,
            'ground_truth_params': ground_truth_params}

def load_hyperparameters(model_name, setting, best=True, path='results_IHDP/hyperparameters'):
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

    experiment = f'{model_name}_ihdp_{setting}_{i}'

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
                         project='ihdp_survival',
                         mode=WANDB_MODE,
                         config={'setting': setting, 'model': model_name, **hyperparameters})

        if model_name == 'x-learner':
            model.fit(X_train, y_train, t_train, c_train, epochs_outcome=1000, epochs_ite=1000, epochs_propensity=1000,
                      batch_size=256)
        else:
            model.fit(X_train, y_train, t_train, c_train,
                      X_val=X_val, y_val=y_val, t_val=t_val, c_val=c_val,
                      epochs=1000, batch_size=1000)
    else:
        print(f'Model {model_name} is already trained, skipping training...')
        run = None
    return model, train_flag, run

def train_individual_model(i, setting, model_name, model_obj, hyperparameters, disable_tqdm):  # Ancillary function for hyperparameter optimization, to train a model on a single dataset and return the target loss on the test set

    data = load_data(i, setting)
    X_test, t_test, y_test, c_test = data['X_test'], data['t_test'], data['y_test'], data['c_test']

    model, _, run = train_and_validate_model(model_name, model_obj, setting, i, hyperparameters, model_path=None, save_model=False, train_anyway=True, WANDB_MODE='disabled', disable_tqdm=disable_tqdm)
    run.finish()

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
        losses = Parallel(verbose=10, n_jobs=n_jobs)(delayed(train_individual_model)(i=i, setting=setting, model_name=model_name, model_obj=model_obj, hyperparameters=hyperparameters, disable_tqdm=True) for i in range(1, n_datasets + 1))
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
    model, train_flag, run = train_and_validate_model(model_name, model_obj, setting, i, hyperparameters,
                                                      model_path=model_path, save_model=True, train_anyway=False,
                                                      WANDB_MODE='disabled', disable_tqdm=True)
    if train_flag:
        run.finish()  # Clowe WANDB if training took place

    # VALIDATION AND TESTING
    data = load_data(i, setting)
    X_train, t_train, y_train, c_train = data['X_train'], data['t_train'], data['y_train'], data['c_train']
    X_test, t_test, y_test, c_test = data['X_test'], data['t_test'], data['y_test'], data['c_test']
    train_df, test_df, num_val_data, x_columns = data['train_df'], data['test_df'], data['num_val_data'], data['x_columns']
    ground_truth_params = data['ground_truth_params']
    train_df = train_df[:-num_val_data]  # remove the validation data from the training dataframe, to be able to compute the real ite on the training data
    for key in ground_truth_params.keys():
        if 'train' in key:
            ground_truth_params[key] = ground_truth_params[key][:-num_val_data]  # remove the validation data from the ground truth params as well

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
    # real ite
    ite_real_train = (train_df['mu1'] - train_df['mu0']).values
    ite_real_test = (test_df['mu1'] - test_df['mu0']).values

    if model_name == 's-learner-linear': # Only model that does not predict a distribution...
        metrics = {}
    else:
        met_sur = get_survival_metrics(metrics,
                                       t_train=t_train, t_test=t_test,
                                       y_train=y_train, y_test=y_test,
                                       c_train=c_train, c_test=c_test,
                                       setting=setting, ground_truth_params=ground_truth_params, dist='weibull')
        metrics.update(met_sur)

    metrics['ate_error_train'] = ate_error_from_ite(ite_train, ite_real_train)
    metrics['ate_error_test'] = ate_error_from_ite(ite_test, ite_real_test)
    metrics['pehe_train'] = pehe(ite_train, ite_real_train)
    metrics['pehe_test'] = pehe(ite_test, ite_real_test)
    metrics['ate_predicted_train'] = np.mean(ite_train)
    metrics['ate_predicted_test'] = np.mean(ite_test)
    metrics['ite_predicted_train'] = ite_train
    metrics['ite_predicted_test'] = ite_test
    metrics['ite_real_train'] = ite_real_train
    metrics['ite_real_test'] = ite_real_test

    # compute the pehe and ate error for the survival setting using the samples
    y1_samples_train, y0_samples_train = train_df['y1'], train_df['y0']
    y1_samples_test, y0_samples_test = test_df['y1'], test_df['y0']

    ite_samples_train = y1_samples_train - y0_samples_train
    ite_samples_test = y1_samples_test - y0_samples_test

    # compute a Wilcoxon signed rank test to see if there are significant differences in the distributions of the ITEs
    res = wilcoxon(ite_train.squeeze(), y=ite_real_train, alternative='two-sided')
    print(f'Wilcoxon test for ite samples train and ite real train: p-value is {res.pvalue}')
    metrics['p_value_wilcoxon_train'] = res.pvalue
    res = wilcoxon(ite_test.squeeze(), y=ite_real_test, alternative='two-sided')
    print(f'Wilconxon test for ite samples test and ite real test: p-value is {res.pvalue}')
    metrics['p_value_wilcoxon_test'] = res.pvalue

    metrics['pehe_samples_train'] = pehe(ite_samples_train, ite_real_train)
    metrics['ate_error_samples_train'] = ate_error_from_ite(ite_samples_train, ite_real_train)
    metrics['pehe_samples_test'] = pehe(ite_samples_test, ite_real_test)
    metrics['ate_error_samples_test'] = ate_error_from_ite(ite_samples_test, ite_real_test)
    metrics['censor_prop_train'] = np.mean(c_train)
    metrics['censor_prop_test'] = np.mean(c_test)

    if compute_tmle:
        max_y = min((500, np.percentile(y_train, 99)))  # We cap the time to 500 or the 90th percentile of the observed times, to avoid very long times that can cause numerical issues in TMLE estimation
        pred_tmle, pred_tmle_g_comp, t_tmle = obtain_tmle(X_train, t_train, y_train, c_train, max_y=max_y, alpha=0.05, n_points=50, verbose=0)
        metrics['pred_tmle'] = pred_tmle
        metrics['pred_tmle_g_comp'] = pred_tmle_g_comp
        metrics['t_tmle'] = t_tmle
    else:
        metrics['pred_tmle'] = None
        metrics['pred_tmle_g_comp'] = None
        metrics['t_tmle'] = None

    if model_name in ['sa-tedvae-m1', 'sa-tedvae-m2']:
        metrics['censor_prop_train_model'] = [p.mean.detach().cpu().numpy() for p in predictions_train['c_dist_list']]
        metrics['censor_prop_test_model'] = [p.mean.detach().cpu().numpy() for p in predictions_train['c_dist_list']]

    return metrics


def get_gt_ihdp_curves(setting, ground_truth_params, time_vector):
    time_vector = torch.from_numpy(time_vector).reshape((len(time_vector), 1))
    survival_train_0_censor_gt = 1.0 - Exponential(1 / ground_truth_params['c0_train']).cdf(time_vector).numpy()
    survival_test_0_censor_gt = 1.0 - Exponential(1 / ground_truth_params['c0_test']).cdf(time_vector).numpy()
    survival_train_1_censor_gt = 1.0 - Exponential(1 / ground_truth_params['c1_train']).cdf(time_vector).numpy()
    survival_test_1_censor_gt = 1.0 - Exponential(1 / ground_truth_params['c1_test']).cdf(time_vector).numpy()
    if setting == '_survival_a':
        survival_train_0_outcome_gt = 1.0 - Exponential(1 / ground_truth_params['mu_0_train']).cdf(time_vector).numpy()  # ntime x npatients
        survival_test_0_outcome_gt = 1.0 - Exponential(1 / ground_truth_params['mu_0_test']).cdf(time_vector).numpy()
        survival_train_1_outcome_gt = 1.0 - Exponential(1 / ground_truth_params['mu_1_train']).cdf(time_vector).numpy()
        survival_test_1_outcome_gt = 1.0 - Exponential(1 / ground_truth_params['mu_1_test']).cdf(time_vector).numpy()
    elif setting == '_survival_b':
        survival_train_0_outcome_gt = 1.0 - Weibull(ground_truth_params['scale_0_train'], ground_truth_params['shape']).cdf(time_vector).numpy()
        survival_test_0_outcome_gt = 1.0 - Weibull(ground_truth_params['scale_0_test'], ground_truth_params['shape']).cdf(time_vector).numpy()
        survival_train_1_outcome_gt = 1.0 - Weibull(ground_truth_params['scale_1_train'], ground_truth_params['shape']).cdf(time_vector).numpy()
        survival_test_1_outcome_gt = 1.0 - Weibull(ground_truth_params['scale_1_test'], ground_truth_params['shape']).cdf(time_vector).numpy()
    elif setting == '_survival_c':
        survival_train_0_outcome_gt = 1.0 - Normal(ground_truth_params['mean_0_train'], ground_truth_params['var'] / 2).cdf(torch.log(time_vector)).numpy()
        survival_test_0_outcome_gt = 1.0 - Normal(ground_truth_params['mean_0_test'], ground_truth_params['var'] / 2).cdf(torch.log(time_vector)).numpy()
        survival_train_1_outcome_gt = 1.0 - Normal(ground_truth_params['mean_1_train'], ground_truth_params['var'] / 2).cdf(torch.log(time_vector)).numpy()
        survival_test_1_outcome_gt = 1.0 - Normal(ground_truth_params['mean_1_test'], ground_truth_params['var'] / 2).cdf(torch.log(time_vector)).numpy()
    else:
        raise ValueError(f'Setting {setting} not recognized for survival metrics computation')
    return survival_train_0_censor_gt, survival_test_0_censor_gt, survival_train_1_censor_gt, survival_test_1_censor_gt, survival_train_0_outcome_gt, survival_test_0_outcome_gt, survival_train_1_outcome_gt, survival_test_1_outcome_gt


def get_pred_ihdp_curves(input_dict, time_vector):
    survival_train_0_outcome_pred = get_weibull_survival(concentration=input_dict['outcome_concentration_train'][0], scale=input_dict['outcome_scale_train'][0], time_vector=time_vector)
    survival_train_1_outcome_pred = get_weibull_survival(concentration=input_dict['outcome_concentration_train'][1], scale=input_dict['outcome_scale_train'][1], time_vector=time_vector)
    survival_test_0_outcome_pred = get_weibull_survival(concentration=input_dict['outcome_concentration_test'][0], scale=input_dict['outcome_scale_test'][0], time_vector=time_vector)
    survival_test_1_outcome_pred = get_weibull_survival(concentration=input_dict['outcome_concentration_test'][1], scale=input_dict['outcome_scale_test'][1], time_vector=time_vector)
    if 'censor_scale_train' in input_dict:
        survival_train_0_censor_pred = get_weibull_survival(concentration=input_dict['censor_concentration_train'][0], scale=input_dict['censor_scale_train'][0], time_vector=time_vector)
        survival_train_1_censor_pred = get_weibull_survival(concentration=input_dict['censor_concentration_train'][1], scale=input_dict['censor_scale_train'][1], time_vector=time_vector)
        survival_test_0_censor_pred = get_weibull_survival(concentration=input_dict['censor_concentration_test'][0], scale=input_dict['censor_scale_test'][0], time_vector=time_vector)
        survival_test_1_censor_pred = get_weibull_survival(concentration=input_dict['censor_concentration_test'][1], scale=input_dict['censor_scale_test'][1], time_vector=time_vector)
    else:
        survival_train_0_censor_pred, survival_train_1_censor_pred = None, None
        survival_test_0_censor_pred, survival_test_1_censor_pred = None, None

    return survival_train_0_censor_pred, survival_test_0_censor_pred, survival_train_1_censor_pred, survival_test_1_censor_pred, survival_train_0_outcome_pred, survival_test_0_outcome_pred, survival_train_1_outcome_pred, survival_test_1_outcome_pred


def get_survival_metrics(input_dict, t_train, t_test, y_train, y_test, c_train, c_test, setting, ground_truth_params, dist='weibull', max_time=100, n_time=100):
    assert dist == 'weibull', 'Currently, only the Weibull distribution is supported for survival metrics computation'
    time_vector = np.linspace(0, max_time, n_time)
    survival_metrics = obtain_c_index(input_dict, t_train=t_train, t_test=t_test, y_train=y_train, y_test=y_test, c_train=c_train, c_test=c_test)
    # For each treatment, compute the ground truth curves and the predicted curves
    survival_train_0_censor_gt, survival_test_0_censor_gt, survival_train_1_censor_gt, survival_test_1_censor_gt, survival_train_0_outcome_gt, survival_test_0_outcome_gt, survival_train_1_outcome_gt, survival_test_1_outcome_gt = get_gt_ihdp_curves(setting, ground_truth_params, time_vector)
    survival_train_0_censor_pred, survival_test_0_censor_pred, survival_train_1_censor_pred, survival_test_1_censor_pred, survival_train_0_outcome_pred, survival_test_0_outcome_pred, survival_train_1_outcome_pred, survival_test_1_outcome_pred = get_pred_ihdp_curves(input_dict, time_vector)
    # Compute the metrics
    survival_metrics['rmst_train_0_outcome'] = np.mean(np.abs(survival_train_0_outcome_pred - survival_train_0_outcome_gt))
    survival_metrics['rmst_train_1_outcome'] = np.mean(np.abs(survival_train_1_outcome_pred - survival_train_1_outcome_gt))
    survival_metrics['rmst_train_outcome'] = (survival_metrics['rmst_train_0_outcome'] + survival_metrics['rmst_train_1_outcome']) / 2
    survival_metrics['rmst_test_0_outcome'] = np.mean(np.abs(survival_test_0_outcome_pred - survival_test_0_outcome_gt))
    survival_metrics['rmst_test_1_outcome'] = np.mean(np.abs(survival_test_1_outcome_pred - survival_test_1_outcome_gt))
    survival_metrics['rmst_test_outcome'] = (survival_metrics['rmst_test_0_outcome'] + survival_metrics['rmst_test_1_outcome']) / 2
    if survival_train_0_censor_pred is not None:
        survival_metrics['rmst_train_0_censor_pred'] = np.mean(np.abs(survival_train_0_censor_pred - survival_train_0_censor_gt))
        survival_metrics['rmst_train_1_censor_pred'] = np.mean(np.abs(survival_train_1_censor_pred - survival_train_1_censor_gt))
        survival_metrics['rmst_train_censor_pred'] = (survival_metrics['rmst_train_0_censor_pred'] + survival_metrics['rmst_train_1_censor_pred']) / 2
    else:
        survival_metrics['rmst_train_0_censor_pred'] = None
        survival_metrics['rmst_train_1_censor_pred'] = None
        survival_metrics['rmst_train_censor_pred'] = None

    if survival_test_0_censor_pred is not None:
        survival_metrics['rmst_test_0_censor_pred'] = np.mean(np.abs(survival_test_0_censor_pred - survival_test_0_censor_gt))
        survival_metrics['rmst_test_1_censor_pred'] = np.mean(np.abs(survival_test_1_censor_pred - survival_test_1_censor_gt))
        survival_metrics['rmst_test_censor_pred'] = (survival_metrics['rmst_test_0_censor_pred'] + survival_metrics['rmst_test_1_censor_pred']) / 2
    else:
        survival_metrics['rmst_test_0_censor_pred'] = None
        survival_metrics['rmst_test_1_censor_pred'] = None
        survival_metrics['rmst_test_censor_pred'] = None
    survival_metrics['time_vector'] = time_vector.flatten()

    return survival_metrics


if __name__== '__main__':

    train_anyway = not True  # If set to True, it trains all models even if the trained model is already in the model_path. If False, it only trains the models that are not already in the model_path, and loads the ones that are already there.
    find_best_hyperparams = not True # Careful with this, as this takes a lot of time! If set to True, it performs hyperparameter optimization for all models before training and evaluating them. If parameters have already been optimized, it is not needed to rerun this, as they are stored
    plot_bool = False
    compute_tmle = True  # Careful as this takes a lot of time
    WANDB_MODE = 'disabled' # options: 'disabled', 'online', 'offline'
    NUM_DATASETS = 100  # Number of datasets to be used for training and evaluation (for hyperparameter search, they are hard-wired inside)
    num_jobs = 10  # Number of parallel jobs for hyperparameter optimization and training (set to 1 for no paralelization)
    smetrics = ['ate_error_train', 'ate_error_test', 'pehe_train', 'pehe_test', 'rmst_train_0_outcome', 'rmst_train_1_outcome', 'rmst_test_0_outcome', 'rmst_test_1_outcome', 'ci_td', 'ci_ipcw', 'ci_censored', 'ibs']  # Metrics to be plotted in the final results
    # This is done to visualize the full output of the metrics in the validation step
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Build the results path
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_IHDP')
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
    # models = {'s-learner': S_learner, 'sa-tedvae-m0': TEDVAE, 'sa-tedvae-m1': TEDVAE, 'sa-tedvae-m2': TEDVAE, 't-learner': T_learner, 'tarnet': TARNet, 'cfrnet': CFR}
    models = {
        's-learner': S_learner,
        't-learner': T_learner,
        'tarnet': TARNet,
        'cfrnet': CFR,
        'sa-tedvae-m0': TEDVAE,
        'sa-tedvae-m1': TEDVAE,
        'sa-tedvae-m2': TEDVAE,
    }
    settings = ['_survival_a', '_survival_b', '_survival_c']

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
                    model, train_flag, run = train_and_validate_model(model_name, model_obj, setting, i, hyperparameters,
                                                                      model_path=model_path, save_model=True, train_anyway=train_anyway, WANDB_MODE=WANDB_MODE, disable_tqdm=True)
                    t_elapsed = time.time() - t0
                    if train_flag:
                        run.finish()
                    trial = max_trials + 1  # If we reach this line, it means that the training was successful, so we can exit the loop by setting the trial to a value higher than max_trials

                except Exception as e:
                    print(f'Error while training model {model_name} on dataset {i} and setting {setting}, trial {trial}, retrying. Exception was {e}')
                    trial += 1
            return t_elapsed  # To save the training time for each model and setting, if needed
        time_data = {}
        for model_name, model_obj in models.items():
            print(f'Training model {model_name} on {NUM_DATASETS} datasets and settings {settings}...')
            all_times = Parallel(verbose=10, n_jobs=num_jobs)(delayed(train_model_f)(setting=setting, i=i, model_name=model_name, model_obj=model_obj) for setting in settings for i in range(1, NUM_DATASETS + 1))
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
                                                                                         compute_tmle=(j == 0) if compute_tmle else False)  # Only compute TMLE for the first 10 datasets and once, as it is very time-consuming
                                                      for i in range(1, NUM_DATASETS + 1))
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
            if plot_bool:
                plot_censored_vs_pehe(results[setting][model_name]['censor_prop_train'],
                                      results[setting][model_name]['pehe_train'],
                                      results[setting][model_name]['ate_error_train'],
                                      path=os.path.join(plots_path, f'censoring_vs_pehe_{model_name}_{setting}_train.png'))
                plot_censored_vs_pehe(results[setting][model_name]['censor_prop_test'],
                                      results[setting][model_name]['pehe_test'],
                                      results[setting][model_name]['ate_error_test'],
                                      path=os.path.join(plots_path, f'censoring_vs_pehe_{model_name}_{setting}_test.png'))

        # compute statistics of each metric using ground truth samples (data not available in real data!)
        model_results_dict = {'Model': model_name,
                              'Setting': setting,
                              }

        for m in smetrics:
            for m in smetrics:
                candidate_found = False
                for cm in results[setting][model_name].keys(): #check if the difference between the smetric strings and the candidate metrics is '_samples'
                    if '_samples' in cm:
                        cm_name = cm.replace('_samples', '')
                        if cm_name == m:
                            candidate_found = True
                            print(f'{cm} for model {model_name} on setting {setting} (median - mean - std): {np.median(results[setting][model_name][cm])}, {np.mean(results[setting][model_name][cm])}, {np.std(results[setting][model_name][cm])}')
                            model_results_dict[m + '_median'] = np.median(results[setting][model_name][cm])
                            model_results_dict[m + '_mean'] = np.mean(results[setting][model_name][cm])
                            model_results_dict[m + '_std'] = np.std(results[setting][model_name][cm])
                if not candidate_found:
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
    with open(os.path.join(results_path, 'results_tmle_2.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(df_formatted)  # Print it once with all results
