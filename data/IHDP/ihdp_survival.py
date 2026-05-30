# creation of IHDP semisynthetic dataset for survival, following an approach similar to Hill (2011)

# Hill, J. L. (2011).
# Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics,
# 20(1), 217-240.
import numpy as np
import torch
import os
import pickle
from matplotlib.pyplot import title
from torch.distributions import Normal, Uniform, Exponential, Weibull
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.nn import Softplus
import seaborn as sns
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_survival_datasets(setting: str, data_train:pd.DataFrame, data_test:pd.DataFrame, scaler: str, seed: int):
    '''
    :param setting: 'A' (linear, exponential) or 'B' (nonlinear, weibull) or 'C' (linear log normal)
    :return: the mean and the samples of the potential outcomes
    '''
    params_to_store = {}  # To be used later for ground truth evaluation, if needed
    # seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train = data_train.iloc[:,1:]
    x_test = data_test.iloc[:,1:]

    t_train = data_train.iloc[:,0]
    t_test = data_test.iloc[:,0]

    N = x_train.shape[0]  # number of samples
    D = x_train.shape[1]  # number of covariates

    if scaler == 'standard':
        scaler = StandardScaler()
        continuous_features = [f'x{i}' for i in range(1, 7)]

        x_train_norm = x_train.copy()
        x_test_norm = x_test.copy()

        x_train_norm.loc[:,continuous_features] = scaler.fit_transform(x_train.loc[:,continuous_features])
        x_test_norm.loc[:,continuous_features] = scaler.transform(x_test.loc[:,continuous_features])

    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        continuous_features = [f'x{i}' for i in range(1, 7)]
        epsilon = 1e-3

        x_train_norm = x_train.copy()
        x_test_norm = x_test.copy()

        x_train_norm.loc[:,continuous_features] = scaler.fit_transform(x_train.loc[:,continuous_features]) + epsilon
        x_test_norm.loc[:,continuous_features] = scaler.transform(x_test.loc[:,continuous_features]) + epsilon
    else:
        x_train_norm = x_train
        x_test_norm = x_test

    if setting == 'A' or setting == 'a':
        params_survival = Uniform(0,10).sample((D, 1))

        # causal effect:
        causal_effect = 200

        # potential_outcomes

        mu0_train = torch.tensor((x_train_norm  @ params_survival).squeeze())
        y0_train = Exponential(1/mu0_train).sample().numpy()

        mu1_train = torch.tensor((x_train_norm  @ params_survival).squeeze()) + causal_effect
        y1_train = Exponential(1/mu1_train).sample().numpy()

        mu0_test = torch.tensor((x_test_norm  @ params_survival).squeeze())
        y0_test = Exponential(1/mu0_test).sample().numpy()

        mu1_test = torch.tensor((x_test_norm  @ params_survival).squeeze()) + causal_effect
        y1_test = Exponential(1/mu1_test).sample().numpy()


        params_censoring = Uniform(5, 10).sample((D, 1))
        more_time_censoring = 200

        c0_train = torch.tensor((x_train_norm  @ params_censoring).squeeze())
        c1_train = torch.tensor((x_train_norm  @ params_censoring).squeeze()) + more_time_censoring

        c0_train_samples = Exponential(1/c0_train).sample().numpy()
        c1_train_samples = Exponential(1/c1_train).sample().numpy()

        c0_test = torch.tensor((x_test_norm  @ params_censoring).squeeze())
        c1_test = torch.tensor((x_test_norm  @ params_censoring).squeeze()) + more_time_censoring

        c0_test_samples = Exponential(1/c0_test).sample().numpy()
        c1_test_samples = Exponential(1/c1_test).sample().numpy()

        params_to_store = {'params_survival': params_survival,
                           'causal_effect': causal_effect,
                           'mu_0_train': mu0_train, 'mu_1_train': mu1_train,
                           'mu_0_test': mu0_test, 'mu_1_test': mu1_test,
                           'params_censoring': params_censoring,
                           'more_time_censoring': more_time_censoring,
                           'c0_train': c0_train, 'c1_train': c1_train,
                           'c0_test': c0_test, 'c1_test': c1_test}

    elif setting == 'B' or setting =='b':

        dim_product = 8
        causal_effect = 4

        weights_1 = Uniform(-1,1).sample((D, dim_product))
        weights_2 = Uniform(5,15).sample((1, dim_product))
        shape = Uniform(1, 2.5).sample((1,))

        print(f'shape (k): {shape}')

        scale0_train = (weights_2 @ Softplus()(torch.tensor((x_train_norm  @ weights_1).T.values, dtype=torch.float32))).t().squeeze()
        y0_train = Weibull(scale0_train, shape).sample().numpy()
        # mu0_train = Weibull(scale0_train, shape).mean
        mu0_train =  scale0_train * torch.exp(torch.lgamma(1 + shape.reciprocal()))

        scale1_train = (weights_2 @ Softplus()(torch.tensor((x_train_norm  @ weights_1).T.values + causal_effect , dtype=torch.float32))).t().squeeze()
        y1_train = Weibull(scale1_train, shape).sample().numpy()
        # mu1_train = Weibull(scale1_train, shape).mean
        mu1_train =  scale1_train * torch.exp(torch.lgamma(1 + shape.reciprocal()))

        scale0_test = (weights_2 @ Softplus()(torch.tensor((x_test_norm  @ weights_1).T.values, dtype=torch.float32))).t().squeeze()
        y0_test = Weibull(scale0_test, shape).sample().numpy()
        mu0_test = Weibull(scale0_test, shape).mean

        scale1_test = (weights_2 @ Softplus()(torch.tensor((x_test_norm  @ weights_1).T.values + causal_effect, dtype=torch.float32)) + causal_effect).t().squeeze()
        y1_test = Weibull(scale1_test, shape).sample().numpy()
        mu1_test = Weibull(scale1_test, shape).mean


        params_censoring = Uniform(0, 20).sample((D, 1))
        more_time_censoring = 200

        c0_train = torch.tensor((x_train_norm  @ params_censoring).squeeze())
        c1_train = torch.tensor((x_train_norm  @ params_censoring).squeeze()) + more_time_censoring

        c0_train_samples = Exponential(1/c0_train).sample().numpy()
        c1_train_samples = Exponential(1/c1_train).sample().numpy()

        c0_test = torch.tensor((x_test_norm  @ params_censoring).squeeze())
        c1_test = torch.tensor((x_test_norm  @ params_censoring).squeeze()) + more_time_censoring

        c0_test_samples = Exponential(1/c0_test).sample().numpy()
        c1_test_samples = Exponential(1/c1_test).sample().numpy()

        params_to_store = {'weights_1': weights_1, 'weights_2': weights_2, 'shape': shape, 'causal_effect': causal_effect,
                           'scale_0_train': scale0_train, 'scale_1_train': scale1_train, 'scale_0_test': scale0_test, 'scale_1_test': scale1_test,
                           'params_censoring': params_censoring, 'more_time_censoring': more_time_censoring,
                           'c0_train': c0_train, 'c1_train': c1_train, 'c0_test': c0_test, 'c1_test': c1_test}


    elif setting == 'C' or setting == 'c':
        # log normal, with a linear relationship between the covariates and the mean of the log normal
        # the scale is fixed to 1
        causal_effect_days = 200
        var = 1
        params_survival = Uniform(0.2,0.6).sample((D, 1))
        mean_0_train = torch.tensor((x_train_norm  @ params_survival).squeeze())
        mu0_train = torch.exp(mean_0_train + var/2)
        y0_train = torch.exp(Normal(loc=mean_0_train, scale=var).sample()).numpy()

        causal_effect = torch.log(1 + causal_effect_days/torch.mean(mu0_train))

        mean_1_train = torch.tensor((x_train_norm  @ params_survival).squeeze()) + causal_effect
        mu1_train = torch.exp(mean_1_train + var/2)
        y1_train = torch.exp(Normal(loc=mean_1_train, scale=var).sample()).numpy()

        mean_0_test = torch.tensor((x_test_norm  @ params_survival).squeeze())
        mu0_test = torch.exp(mean_0_test + var/2)
        y0_test = torch.exp(Normal(loc=mean_0_test, scale=var).sample()).numpy()

        mean_1_test = torch.tensor((x_test_norm  @ params_survival).squeeze()) + causal_effect
        mu1_test = torch.exp(mean_1_test + var/2)
        y1_test = torch.exp(Normal(loc=mean_1_test, scale=var).sample()).numpy()

        params_censoring = Uniform(10, 25).sample((D, 1))
        more_time_censoring = 250

        c0_train = torch.tensor((x_train_norm  @ params_censoring).squeeze())
        c1_train = torch.tensor((x_train_norm  @ params_censoring).squeeze()) + more_time_censoring

        c0_train_samples = Exponential(1/c0_train).sample().numpy()
        c1_train_samples = Exponential(1/c1_train).sample().numpy()

        c0_test = torch.tensor((x_test_norm  @ params_censoring).squeeze())
        c1_test = torch.tensor((x_test_norm  @ params_censoring).squeeze()) + more_time_censoring

        c0_test_samples = Exponential(1/c0_test).sample().numpy()
        c1_test_samples = Exponential(1/c1_test).sample().numpy()

        params_to_store = {'params_survival': params_survival, 'causal_effect': causal_effect, 'var': var,
                           'mean_0_train': mean_0_train, 'mean_1_train': mean_1_train,
                           'mean_0_test': mean_0_test, 'mean_1_test': mean_1_test,
                           'params_censoring': params_censoring, 'more_time_censoring': more_time_censoring,
                           'c0_train': c0_train, 'c1_train': c1_train, 'c0_test': c0_test, 'c1_test': c1_test}

    else:
        raise ValueError('setting must be A, B or C')

    ## TRAIN
    survival_data_train = x_train.copy()
    survival_data_train['treatment'] = t_train
    survival_data_train['y0'] = y0_train
    survival_data_train['y1'] = y1_train
    survival_data_train['mu0'] = mu0_train
    survival_data_train['mu1'] = mu1_train
    survival_data_train['c0_mean'] = c0_train
    survival_data_train['c1_mean'] = c1_train
    survival_data_train['c0'] = c0_train_samples
    survival_data_train['c1'] = c1_train_samples

    survival_data_train['ite_samples'] = y1_train - y0_train
    survival_data_train['ite_mean'] = mu1_train - mu0_train
    survival_data_train['ate_samples'] = survival_data_train['ite_samples'].mean()
    survival_data_train['ate_mean'] = survival_data_train['ite_mean'].mean()

    survival_data_train['censoring_factual'] = t_train * c1_train_samples + (1 - t_train) * c0_train_samples
    survival_data_train['censoring_cfactual'] = (1 - t_train) * c1_train_samples + t_train * c0_train_samples
    survival_data_train['survival_factual'] = t_train * y1_train + (1 - t_train) * y0_train
    survival_data_train['survival_cfactual'] = (1 - t_train) * y1_train + t_train * y0_train

    survival_data_train['y_factual'] = np.minimum(survival_data_train['survival_factual'], survival_data_train['censoring_factual'])
    survival_data_train['y_cfactual'] = np.minimum(survival_data_train['survival_cfactual'], survival_data_train['censoring_cfactual'])
    survival_data_train['censoring_indicator'] = (survival_data_train['survival_factual'] >= survival_data_train['censoring_factual']).astype(int)
    survival_data_train['censoring_indicator_cfactual']= (survival_data_train['survival_cfactual'] >= survival_data_train['censoring_cfactual']).astype(int)
    ## TEST
    survival_data_test = x_test.copy()
    survival_data_test['treatment'] = t_test
    survival_data_test['y0'] = y0_test
    survival_data_test['y1'] = y1_test
    survival_data_test['mu0'] = mu0_test
    survival_data_test['mu1'] = mu1_test
    survival_data_test['c0_mean'] = c0_test
    survival_data_test['c1_mean'] = c1_test
    survival_data_test['c0'] = c0_test_samples
    survival_data_test['c1'] = c1_test_samples

    survival_data_test['ite_samples'] = y1_test - y0_test
    survival_data_test['ite_mean'] = mu1_test - mu0_test
    survival_data_test['ate_samples'] = survival_data_test['ite_samples'].mean()
    survival_data_test['ate_mean'] = survival_data_test['ite_mean'].mean()

    survival_data_test['censoring_factual'] = t_test * c1_test_samples + (1 - t_test) * c0_test_samples
    survival_data_test['censoring_cfactual'] = (1 - t_test) * c1_test_samples + t_test * c0_test_samples
    survival_data_test['survival_factual'] = t_test * y1_test + (1 - t_test) * y0_test
    survival_data_test['survival_cfactual'] = (1 - t_test) * y1_test + t_test * y0_test

    survival_data_test['y_factual'] = np.minimum(survival_data_test['survival_factual'], survival_data_test['censoring_factual'])
    survival_data_test['y_cfactual'] = np.minimum(survival_data_test['survival_cfactual'], survival_data_test['censoring_cfactual'])
    survival_data_test['censoring_indicator'] = (survival_data_test['survival_factual'] >= survival_data_test['censoring_factual']).astype(int)
    survival_data_test['censoring_indicator_cfactual']= (survival_data_test['survival_cfactual'] >= survival_data_test['censoring_cfactual']).astype(int)

    if setting == 'B':
        survival_data_train['shape'] = shape.item()

    return survival_data_train, survival_data_test, params_to_store

def plot_histograms(survival_data, setting, include_mean=True, survival_data_test=None):
    from tueplots import bundles, axes, figsizes

    n_bins = 30
    plt.rcParams.update({"figure.dpi": 300})
    with plt.rc_context({**bundles.icml2024(column='full'), **figsizes.icml2024_full(ncols=3, nrows=3), **axes.lines()}):

        fig, ax = plt.subplots(3, 3)
        sns.histplot(survival_data, x='y0', color='deepskyblue', kde=True, ax=ax[0, 0], stat='density',
                     label=r'$\tau_s(0)$', bins=n_bins)
        sns.histplot(survival_data, x='y1', color='orange', kde=True, ax=ax[0, 1], stat='density',
                     label=r'$\tau_s(1)$', bins=n_bins)
        # kde of mu0 and mu1
        # sns.kdeplot(survival_data['mu0'], color='blue', ax=ax[0, 0], label ='mean 0')
        # sns.kdeplot(survival_data['mu1'], color='orange', ax=ax[0, 1], label='mean 1')
        if include_mean:
            sns.histplot(survival_data, x='mu0', color='blue', kde=True, ax=ax[0, 0], stat='density', label='$\mu_0$',
                         bins=n_bins)
            sns.histplot(survival_data, x='mu1', color='chocolate', kde=True, ax=ax[0, 1], stat='density',
                     label='$\mu_1$', bins=n_bins)

        # ites
        sns.histplot(survival_data['ite_samples'], kde=True, ax=ax[0, 2], stat='density', label='Noisy ITE')
        print('ATE samples:', survival_data['ate_samples'][0])
        print('ATE ideal:', survival_data['ate_mean'][0])
        y_lim = ax[0, 2].get_ylim()[1]
        if np.std(survival_data['ite_mean']) > 0.1:
            # sns.kdeplot(ite_ideal, color='orange', ax=ax[0, 2], label='ideal ite')
            if include_mean:
                sns.histplot(survival_data['ite_mean'], color='orange', kde=True, ax=ax[0, 2], stat='density',
                         label='ITE (mean)', bins=n_bins)
            else:
                std_ite = np.std(survival_data['ite_samples'])
                # plot an horizontal line with the std of the ite at ate_mean, 0.2*y_lim
                ax[0, 2].plot([survival_data['ate_mean'][0] - std_ite, survival_data['ate_mean'][0] + std_ite], [0.2 * y_lim, 0.2 * y_lim], color='orange')

        ax[0, 2].axvline(survival_data['ate_samples'][0], color='blue', linestyle='--', label='Noisy ATE')
        ax[0, 2].axvline(survival_data['ate_mean'][0], color='orange', linestyle='--', label='ITE (mean)')

        ax[0, 2].text(survival_data['ate_samples'][0], 2 / 3 * y_lim,
                      f"{survival_data['ate_samples'][0]:.2f}", color='blue')
        ax[0, 2].text(survival_data['ate_mean'][0], 1 / 3 * y_lim, f"{survival_data['ate_mean'][0]:.2f}",
                      color='orange')

        # censoring times in the training set
        sns.histplot(survival_data, x='c0', color='green', kde=False, ax=ax[1, 0], stat='density',
                     label=r'$\tau_c(0)$', bins=n_bins)
        sns.histplot(survival_data, x='c1', color='red', kde=False, ax=ax[1, 1], stat='density', label=r'$\tau_c(1)$',
                     bins=n_bins)
        if include_mean:
            sns.kdeplot(survival_data, x='c0_mean', color='green', ax=ax[1, 0], label=r'$\tau_c(0)$ (mean)')
            sns.kdeplot(survival_data, x='c1_mean', color='red', ax=ax[1, 1], label=r'$\tau_c(1)$ (mean)')

        if survival_data_test is not None:
            # Determine common bin edges
            ite_train = survival_data['ite_samples']#samples
            ite_test = survival_data_test['ite_samples']
            # Define number of bins
            # num_bins = 20  # Adjust this value as needed

            # Compute bin edges with equal width
            bin_range = (min(ite_train.min(), ite_test.min()), max(ite_train.max(), ite_test.max()))
            bin_width = (bin_range[1] - bin_range[0]) / n_bins
            bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)  # Create bin edges
            sns.histplot(ite_train, bins=bins, label='train', stat='density', alpha=0.3, ax=ax[1,2])
            sns.histplot(ite_test, bins=bins, label='test', stat='density', alpha=0.3, ax=ax[1,2])
            ax[1,2].set_xlabel('ITE samples train/test (days)')
            ax[1,2].legend()

        # plot the censoring times and the factual outcome
        sns.histplot(survival_data, x='censoring_factual', kde=True, ax=ax[2, 0], stat='density', label=r'$\tau_c$',
                     bins=n_bins)
        sns.histplot(survival_data, x='survival_factual', kde=True, ax=ax[2, 0], stat='density', label=r'$\tau_s$',
                     bins=n_bins)

        sns.histplot(survival_data, x='y_factual', hue='censoring_indicator',
                     kde=True, stat='density', ax=ax[2, 1], bins=n_bins)

        # plot finally the y_factual
        sns.histplot(survival_data, x='y_factual', kde=True, ax=ax[2, 2], stat='density', label='Observed time (y)',
                     bins=n_bins)

        if setting == 'B':
            # write the shape parameter of the weibull
            title = f'Setting {setting} - seed: {i} - shape: {survival_data["shape"][0]:.2f}'
        else:
            title = f'Setting {setting} - seed: {i}'


        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[0, 2].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()
        # ax[1, 2].legend()
        ax[2, 0].legend()
        ax[2, 2].legend()

        leg_21 = ax[2, 1].get_legend()
        leg_21.set_title('Censoring indicator')

        ax[0, 0].set_xlabel('')
        ax[0, 1].set_xlabel('')
        ax[0, 2].set_xlabel('ITE (days)')
        ax[1, 0].set_xlabel('time  (days)')
        ax[1, 1].set_xlabel('time  (days)')
        ax[2, 0].set_xlabel('time of factual survival/censoring (for T=t)')
        ax[2, 1].set_xlabel('y factual (time in days)')
        ax[2, 2].set_xlabel('y factual (time in days)')

        # ax[1, 2].set_visible(False)
        if survival_data_test is None:
            ax[1, 2].set_frame_on(False)
            ax[1, 2].set_xticks([])
            ax[1, 2].set_yticks([])

        xlim_0 = list(ax[0, 0].get_xlim())
        xlim_1 = list(ax[0, 1].get_xlim())
        #
        # xlim_0[1] = 0.8 * xlim_0[1]
        # xlim_1[1] = 0.8 * xlim_1[1]

        ax[1, 0].set_xlim(xlim_0)
        ax[1, 1].set_xlim(xlim_1)

        ax[2, 0].set_xlim(0, 1/2*max([xlim_0[1], xlim_1[1]]))
        ax[2, 1].set_xlim(0, 1/4*max([xlim_0[1], xlim_1[1]]))
        ax[2, 2].set_xlim(0, 1/4 * max([xlim_0[1], xlim_1[1]]))

        n_censored = survival_data['censoring_indicator'].sum()
        prop_censored = n_censored / survival_data.shape[0] * 100

        ylim_censor = ax[2, 1].get_ylim()
        xlim_censor = ax[2, 1].get_xlim()
        ax[2, 1].text(xlim_censor[1] * 0.35, ylim_censor[1] * 0.3, f'Censored: {n_censored} ({prop_censored:.2f}$\%$)', color='black')

        ax[0, 1].set_ylabel('')
        ax[0, 2].set_ylabel('')
        ax[1, 1].set_ylabel('')
        ax[2, 1].set_ylabel('')
        ax[2, 2].set_ylabel('')


        # ax[1, 0].sharex(ax[0, 0])
        # ax[1, 1].sharex(ax[0, 1])

        fig.suptitle(title)

        # plt.tight_layout()
        hist_dir = os.path.join(CURRENT_DIR, 'histograms')
        os.makedirs(hist_dir, exist_ok=True)
        plt.savefig(os.path.join(hist_dir, f'setting_{setting}_seed_{i}_means_{include_mean}.pdf'))
        plt.show()


# get the covariates
# i = 1
settings = ['A', 'B', 'C']
scaling = {'A':'minmax', 'B':'minmax', 'C':'minmax'}

save_hists = False  # if True, save the histograms of the potential outcomes for each seed and setting, otherwise only print the mean and std of the censored proportion across seeds

data_train = pd.read_csv(os.path.join(CURRENT_DIR, 'IHDP', 'ihdp_npci_train_1.csv'))
data_test = pd.read_csv(os.path.join(CURRENT_DIR, 'IHDP', 'ihdp_npci_test_1.csv'))

# keep only the treatment and the covariates
data_train = data_train.loc[:,['treatment'] + [f'x{i}' for i in range(1, 26)]]
data_test = data_test.loc[:,['treatment'] + [f'x{i}' for i in range(1, 26)]]

path_a = os.path.join(CURRENT_DIR, 'IHDP_survival_a')
path_b = os.path.join(CURRENT_DIR, 'IHDP_survival_b')
path_c = os.path.join(CURRENT_DIR, 'IHDP_survival_c')
paths = {'A': path_a, 'B': path_b, 'C': path_c}

for setting in settings:
    censored  = []
    for i in range(1, 101):

        scaling_i = scaling[setting]
        path = paths[setting]
        os.makedirs(path, exist_ok=True)

        survival_data_train_, survival_data_test_, params_to_store = generate_survival_datasets(setting, data_train, data_test, scaling_i, i)
        # plot histograms of the potential outcomes, with kde of mu0 and mu1
        # save the data in a csv
        survival_data_train_.to_csv(f'{path}/ihdp_npci_train_{i}.csv', index=False)
        survival_data_test_.to_csv(f'{path}/ihdp_npci_test_{i}.csv', index=False)

        # Save the parameters used to generate the data in a pickle file
        with open(f'{path}/ihdp_npci_params_{i}.pkl', 'wb') as f:
            pickle.dump(params_to_store, f, protocol=pickle.HIGHEST_PROTOCOL)

        if save_hists:
            plot_histograms(survival_data_train_, setting, False, survival_data_test_)

        censored.append(survival_data_train_['censoring_indicator'].mean())

    print(f'Setting {setting} - Censored: {np.mean(censored)} +- {np.std(censored)}')

        
