import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
from models.utils import aggregate_all
from pytmle.plotting import initialize_subplots
from typing import Optional
from matplotlib.lines import Line2D
from pytmle import PyTMLE
import torchtuples as tt
from pycox.models import CoxPH
from sksurv.ensemble import RandomSurvivalForest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
import statsmodels.stats.multitest as multitest
from tabulate import tabulate
from statsmodels.stats.proportion import proportion_confint
from pycox.evaluation import EvalSurv
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw, concordance_index_censored
from torch.distributions import Weibull
from SurvivalEVAL.Evaluator import SurvivalEvaluator
from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def get_weibull_survival(concentration, scale, time_vector):
    time_vector = torch.from_numpy(time_vector).reshape((len(time_vector), 1))
    return 1.0 - Weibull(concentration=torch.from_numpy(concentration), scale=torch.from_numpy(scale)).cdf(time_vector[:, 0]).T.numpy()

def get_weibull_right_censored_log_likelihood(concentration, scale, time_vector, censoring_vector, min_log=1e-10):
    dist = Weibull(concentration=torch.from_numpy(concentration), scale=torch.from_numpy(scale))
    time_vector = torch.from_numpy(time_vector).reshape((len(time_vector), 1))
    log_survival = np.log((torch.clamp(1.0 - dist.cdf(time_vector[:, 0]), min=min_log).numpy()))
    log_likelihood = dist.log_prob(time_vector[:, 0]).numpy()
    # Keep only the main diagonals (we want a single value for each patient's time)
    log_survival = log_survival[np.arange(len(time_vector)), np.arange(len(time_vector))]
    log_likelihood = log_likelihood[np.arange(len(time_vector)), np.arange(len(time_vector))]
    # We assume in censoring vector that 1 means censoring and 0 means event
    rcll_all = (1 - censoring_vector.squeeze()) * log_likelihood + censoring_vector.squeeze() * log_survival
    return np.mean(rcll_all)


def get_weibull_mean(concentration, scale):
    dist = Weibull(concentration=torch.from_numpy(concentration), scale=torch.from_numpy(scale))
    return dist.mean.detach().cpu().numpy().flatten()


def bern_conf_interval(n, mean, ibs=False):
    # Confidence interval
    ci_bot, ci_top = proportion_confint(count=mean * n, nobs=n, alpha=0.1, method='beta')
    if mean < 0.5 and not ibs:
        ci_bot_2 = 1 - ci_top
        ci_top = 1 - ci_bot
        ci_bot = ci_bot_2
        mean = 1 - mean

    return np.round(ci_bot, 4), mean, np.round(ci_top, 4)


def obtain_c_index(input_dict, t_train, t_test, y_train, y_test, c_train, c_test, return_ci=True):

    # VERY IMPORTANT WARNING: IN OUR CODE, C=1 MEANS CENSORED AND C=0 MEANS EVENT, WHILE THE CONVENTION IS USUALLY DIFFERENT FOR SURVIVAL LIBRARIES!!

    # SurvEVAL metrics
    time_bins = np.linspace(0, np.ceil(max((np.max(y_train), np.max(y_test)))), 100)  # Time bins at which the model outputs are evaluated, we take 100 points until the max time in the training set
    survival_outputs = np.zeros([len(c_test), len(time_bins)])
    pred_surv_test_0 = get_weibull_survival(concentration=input_dict['outcome_concentration_test'][0], scale=input_dict['outcome_scale_test'][0], time_vector=time_bins) # returns ntime x patients
    pred_surv_test_1 = get_weibull_survival(concentration=input_dict['outcome_concentration_test'][1], scale=input_dict['outcome_scale_test'][1], time_vector=time_bins)
    for i in range(len(c_test)):
        survival_outputs[i] = pred_surv_test_0[:, i] if t_test[i] < 0.5 else pred_surv_test_1[:, i]

    evl = SurvivalEvaluator(pred_survs=survival_outputs, time_coordinates=time_bins, event_times=y_test, event_indicators=1 - c_test, train_event_times=y_train, train_event_indicators=1 - c_train)
    d_calibration = evl.d_calibration(num_bins=100)[0]  # Keep only the p-value: if higher than alpha, then it is well calibrated

    # PyCox Concordance index
    time_test = np.unique(y_test).flatten()  # Time points at which we want to compute the risk predictions, we take the unique time points in the test set
    n_samples = len(c_test)  # Number of patients
    pred_risk = np.zeros([n_samples, len(time_test)])
    pred_risk_test_0 = 1.0 - get_weibull_survival(concentration=input_dict['outcome_concentration_test'][0], scale=input_dict['outcome_scale_test'][0], time_vector=time_test) # returns ntime x patients
    pred_risk_test_1 = 1.0 - get_weibull_survival(concentration=input_dict['outcome_concentration_test'][1], scale=input_dict['outcome_scale_test'][1], time_vector=time_test)
    for i in range(n_samples):
        pred_risk[i] = pred_risk_test_0[:, i] if t_test[i] < 0.5 else pred_risk_test_1[:, i]
    surv_f = pd.DataFrame(1 - pred_risk.T, index=time_test)  # In the format expected by pycox
    # Evaluate using PyCox c-index
    ev = EvalSurv(surv_f, y_test, 1 - c_test, censor_surv='km')  # We use c_test=1 as censored, we need to invert to pycox format
    ci_pycox = ev.concordance_td()

    # Right censored log-likelihood, see https://arxiv.org/pdf/2103.14755
    concentration_0 = input_dict['outcome_concentration_test'][0].squeeze()
    scale_0 = input_dict['outcome_scale_test'][0].squeeze()
    concentration_1 = input_dict['outcome_concentration_test'][1].squeeze()
    scale_1 = input_dict['outcome_scale_test'][1].squeeze()
    concentration_factual = t_test * concentration_1 + (1 - t_test) * concentration_0
    scale_factual = t_test * scale_1 + (1 - t_test) * scale_0
    rcll = get_weibull_right_censored_log_likelihood(concentration=concentration_factual.reshape(-1, 1), scale=scale_factual.reshape(-1, 1), time_vector=y_test, censoring_vector=c_test)

    # Evaluate using sksurv c-index, both with and without ipcw

    idx_to_erase = np.where(y_test > np.max(y_train))[0]  # We need to remove patients in the test set with a larger time than the max in the training set to prevent errors in the estimators
    if len(idx_to_erase) > 0:
        y_test = np.delete(y_test, idx_to_erase)
        t_test = np.delete(t_test, idx_to_erase)
        c_test = np.delete(c_test, idx_to_erase)
    survival_train = np.empty(dtype=[('event', np.bool), ('time', np.float64)], shape=len(y_train))
    survival_train['event'] = np.bool(1 - c_train)  # Note that in our setting, c=1 means censored, and c=0 means event, so we need to invert it for the sksurv format, where event=True means that the event happened (not censored)
    survival_train['time'] = y_train.flatten()
    survival_test = np.empty(dtype=[('event', np.bool), ('time', np.float64)], shape=len(y_test))  # To prevent errors, add a dummy patient censored at the end (weirdness of the library used)
    survival_test['event'] = np.bool(1 - c_test)
    survival_test['time'] = y_test.flatten()
    # As risk, we use the 1/mean of the distributions (as higher value indicates higher risk)
    risk0 = 1.0 / get_weibull_mean(concentration=input_dict['outcome_concentration_test'][0], scale=input_dict['outcome_scale_test'][0])
    risk1 = 1.0 / get_weibull_mean(concentration=input_dict['outcome_concentration_test'][1], scale=input_dict['outcome_scale_test'][1])
    if len(idx_to_erase) > 0:
        risk0 = np.delete(risk0, idx_to_erase)
        risk1 = np.delete(risk1, idx_to_erase)
    risk = t_test * risk1 + (1 - t_test) * risk0
    ci_ipcw = concordance_index_ipcw(survival_train, survival_test, risk)[0]
    ci_censored = concordance_index_censored(np.bool(1 - c_test), y_test, risk)[0]

    # Integrated brier score
    ibs_times = np.linspace(np.ceil(np.min(y_test)) + 1, np.floor(np.max(y_test)) - 1, 100) # The +1 and -1 are to prevent errors in ibs computation later
    pred_surv_test = np.zeros([len(y_test), len(ibs_times)])
    surv_test_0 = get_weibull_survival(concentration=input_dict['outcome_concentration_test'][0], scale=input_dict['outcome_scale_test'][0], time_vector=ibs_times)
    surv_test_1 = get_weibull_survival(concentration=input_dict['outcome_concentration_test'][1], scale=input_dict['outcome_scale_test'][1], time_vector=ibs_times)
    if len(idx_to_erase) > 0:
        surv_test_0 = np.delete(surv_test_0, idx_to_erase, axis=1)
        surv_test_1 = np.delete(surv_test_1, idx_to_erase, axis=1)
    for i in range(len(y_test)):
        pred_surv_test[i] = surv_test_0[:, i] if t_test[i] < 0.5 else surv_test_1[:, i]
    ibs = integrated_brier_score(survival_train, survival_test, pred_surv_test, ibs_times)

    metrics = {'ci_td': ci_pycox,
               'ci_ipcw': ci_ipcw,
               'ci_censored': ci_censored,
               'ibs': ibs,
               'd_calibration': d_calibration,
               'rcll': rcll}

    if return_ci:
        keys = list(metrics.keys())
        for key in keys:
            if key not in ['rcll', 'd_calibration']:  # TODO: CI for rcll could be computed with bootstrapping or assuming gaussianity? Not implemented by now...
                ci_bot, mean, ci_top = bern_conf_interval(n_samples, metrics[key], ibs=(key == 'ibs'))
                metrics[key + '_ci'] = (ci_bot, mean, ci_top)

    return metrics

def obtain_tmle(x_train, t_train, y_train, c_train, max_y, alpha=0.05, n_points=100, verbose=0):
    # we need a df to use tmle, with X, t, y and 1-c
    # x_train, t_train and y_ytain are assumed to be numpy arrays
    train_df = pd.DataFrame(x_train)
    train_df['t'] = t_train
    train_df['y'] = y_train
    train_df['event'] = (1 - c_train).astype(int)
    '''
    in_features = x_train.shape[1] + 1  # len(x) + len(t)
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)

    model = CoxPH(net, tt.optim.Adam)

    target_times = list(np.linspace(0, np.amax(y_train), n_points))  # evaluate each 20 days until 3000 days

    print("Fitting TMLE model...")
    tmle = PyTMLE(train_df,
                  col_event_times="y",
                  col_event_indicator="event",
                  col_group="t",
                  target_times=target_times,
                  g_comp=True,
                  verbose=verbose)
    tmle.fit(cv_folds=5,
             max_updates=350,
             save_models=False,
             #models=[model, RandomSurvivalForest()],
             #propensity_score_models=[SVC(), RandomForestClassifier()],
             labtrans=None,
             bootstrap=False)
    '''
    target_times = list(np.linspace(0, max_y, n_points))
    print("Fitting TMLE model...")

    surv_models = [
        ExtraSurvivalTrees(
            n_estimators=100,
            min_samples_leaf=30,
            max_features="sqrt",
            random_state=1,
            n_jobs=-1,
        ),
        RandomSurvivalForest(
            n_estimators=100,
            min_samples_leaf=30,
            max_features="sqrt",
            random_state=2,
            n_jobs=-1,
        ),
    ]

    ps_models = [
        HistGradientBoostingClassifier(
            max_iter=100,
            min_samples_leaf=30,
            random_state=1,
        ),
        RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=30,
            max_features="sqrt",
            random_state=2,
            n_jobs=-1,
        ),
    ]

    tmle = PyTMLE(train_df,
                  col_event_times="y",
                  col_event_indicator="event",
                  col_group="t",
                  target_times=target_times,
                  g_comp=True,
                  verbose=verbose)
    t0 = time.time()
    tmle.fit(cv_folds=3, min_nuisance=0.1, max_updates=50, models=surv_models, propensity_score_models=ps_models, bootstrap=False)
    t_elapsed = time.time() - t0

    pred = tmle.predict(type='risks', alpha=alpha)
    pred_g_comp = tmle.predict(type='risks', alpha=alpha, g_comp=True)
    return pred, pred_g_comp, t_elapsed

def ate_error_from_ite(ite_pred, ite_true):
    return np.abs(np.mean(ite_pred) - np.mean(ite_true))

def ate_error(ate_pred, ate_true):
    return np.abs(ate_pred - ate_true)

def relative_ate_error(ate_pred, ate_true):
    return np.abs(ate_pred - ate_true) / np.abs(ate_true)

def pehe(ite_pred, ite_true):
    return np.sqrt(np.mean(np.square(ite_pred - ite_true)))

def relative_pehe(ite_pred, ite_true):
    return np.sqrt(np.mean(np.square(ite_pred - ite_true))) / np.sqrt(np.mean(np.square(ite_true)))

def plot_ites_distribution(ite_pred, ite_true, n_bins=20, path=None, show=True, title=None, labels=None):
    if labels is None:
        labels = ['ITE pred', 'ITE true']
    sns.histplot(ite_pred, alpha=0.3,  color='blue', label=labels[0], kde=True, stat='density', bins=n_bins)
    ylim = plt.gca().get_ylim()
    if ite_true.std() <0.01:
        plt.axvline(ite_true.mean(), 0, ylim[1], color='red', label=labels[1])
    else:
        sns.histplot(ite_true, alpha=0.3, color='red', label=labels[1], kde=True, stat='density', bins=n_bins)
    plt.legend()
    plt.xlabel('ITE')
    plt.ylabel('Density')
    if title is not None:
        plt.title(title)
    if path is not None:
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()
    else:
        plt.show()

def plot_outcomes(t_true, y_true, y_pred_list,
                  y_pred_list_std = None,
                  y_counterfactual_true = None,
                  survival_true=None, c_true=None,
                  t_true_test=None, y_true_test=None, y_pred_list_test=None,
                  y_pred_list_std_test=None,
                  y_counterfactual_true_test=None,
                  survival_true_test=None, c_true_test=None,
                  outcome_name = 'outcome', sort= True, n_points=100):

    treatments = np.unique(t_true)
    num_treatments = len(treatments)
    df = pd.DataFrame({'treatment': t_true, 'outcome': y_true})

    if y_counterfactual_true is not None:
        if num_treatments != 2:
            raise ValueError('Counterfactual outcomes only available for binary treatments')
        df['counterfactual'] = y_counterfactual_true

    if c_true is not None:
        df['censored'] = c_true

    if t_true_test is not None:
        df['split'] = ['train']*len(t_true)
        df_test = pd.DataFrame({'treatment': t_true_test, 'outcome': y_true_test})
        if y_counterfactual_true_test is not None:
            df_test['counterfactual'] = y_counterfactual_true_test
        df_test['split'] = ['test']*len(t_true_test)
        if c_true_test is not None:
            df_test['censored'] = c_true_test
        df = pd.concat([df, df_test])

    for i, treatment in enumerate(treatments):
        df['pred_'+str(treatment)] = y_pred_list[i]
        if y_pred_list_std is not None:
            df['std_'+str(treatment)] = y_pred_list_std[i]

        if t_true_test is not None:
            df['pred_'+str(treatment)+'_test'] = y_pred_list_test[i]
            if y_pred_list_std_test is not None:
                df['std_'+str(treatment)+'_test'] = y_pred_list_std_test[i]
            df['split'] = ['test']*len(t_true_test)

    if c_true is not None:
        if survival_true is None:
            df = df[df['censored'] == 0]
        else:
            if survival_true_test is not None:
                survival = np.concatenate([survival_true, survival_true_test])
            else:
                survival = survival_true
            df['outcome'] = survival
    if sort:
        df = df.sort_values(by='outcome', ascending=False).reset_index(drop=True)

    df_treatment_list = [df[df['treatment'] == i].reset_index(drop=True) for i in treatments]
    fig, axs = plt.subplots(1, num_treatments, figsize=(5 * num_treatments, 5))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, (treatment, df_t) in enumerate(zip(treatments, df_treatment_list)):
        ax = axs[i]
        x_ax = df_t.index
        # sample random n_points from x_ax
        if len(x_ax) > n_points:
            x_ax = pd.Index(np.random.choice(x_ax, n_points, replace=False)).sort_values()
            df_t = df_t.loc[x_ax]

        ax.plot(x_ax, df_t[f'pred_'+str(treatment)], label='Predicted factual Outcome', color=colors[i], alpha=1, linewidth=0.5)
        ax.fill_between(x_ax, df_t[f'pred_'+str(treatment)] - df_t[f'std_'+str(treatment)], df_t[f'pred_'+str(treatment)] + df_t[f'std_'+str(treatment)], alpha=0.3, color=colors[i])

        ax.set_title(f'Treatment {treatment}')
        ax.set_xlabel('Sample')

        #potential outcomes
        other_treatments = [t for t in treatments if t != treatment]
        for t in other_treatments:
            ax.plot(x_ax, df_t[f'pred_'+str(t)], label=f'Counterfactual T={t}', color='grey', alpha=0.5, linewidth=0.5)
            ax.fill_between(x_ax, df_t[f'pred_'+str(t)] - df_t[f'std_'+str(t)], df_t[f'pred_'+str(t)] + df_t[f'std_'+str(t)], alpha=0.1, color='grey')

        if 'split' in df_t.columns:
            df_t_train = df_t[df_t['split'] == 'train']
            df_t_test = df_t[df_t['split'] == 'test']
            x_ax = df_t.index
            x_train = df_t_train.index
            x_test = df_t_test.index

            ax.scatter(x_train, df_t_train['outcome'], label='Factual Outcome Train', color='red', s=10, marker='x',
                       linestyle='-',
                       linewidth=1, zorder=10)
            ax.scatter(x_test, df_t_test['outcome'], label='Factual Outcome Test', color='green', s=10, marker='x',
                       linestyle='-',
                       linewidth=1, zorder=10, alpha=0.6)

            if y_counterfactual_true is not None:
                ax.scatter(x_ax, df_t['counterfactual'], label='CF Outcome', color='purple', s=10, marker='x', linestyle='-',
                           linewidth=1, zorder=10, alpha=0.6)
        else:
            ax.scatter(x_ax, df_t['counterfactual'], label='Factual Outcome', color='red', s=10, marker='x', linestyle='-',
                       linewidth=1, zorder=10)
            if y_counterfactual_true is not None:
                ax.scatter(x_ax, df_t['y_cf'], label='CF Outcome', color='purple', s=10, marker='x', linestyle='-')
        ax.legend()
        axs[0].set_ylabel(outcome_name)

        return fig, axs




def plot_outcomes_dist(t_true, y_0_true, y_1_true,
                       y_0_pred, y_1_pred,
                       c_true=None, path=None, show=True):
    if c_true is not None:
        t_true = t_true[c_true==0]
        y_0_true = y_0_true[c_true==0]
        y_1_true = y_1_true[c_true==0]
        y_0_pred = y_0_pred[c_true==0]
        y_1_pred = y_1_pred[c_true==0]

    figure, axs = plt.subplots(1,2)
    # potential outcomes
    sns.histplot(y_0_true, alpha=0.3, color='red', label='Y_0 true', kde=True, stat='density', ax=axs[0])
    sns.histplot(y_1_true, alpha=0.3, color='blue', label='Y_1 true', kde=True, stat='density', ax=axs[0])
    sns.histplot(y_0_pred, alpha=0.3, color='green', label='Y_0 pred', kde=True, stat='density', ax=axs[0])
    sns.histplot(y_1_pred, alpha=0.3, color='orange', label='Y_1 pred', kde=True, stat='density', ax=axs[0])

    # factual outcomes and predictions
    y_factual_true = y_0_true*(1-t_true) + y_1_true*t_true
    y_factual_pred = y_0_pred*(1-t_true) + y_1_pred*t_true

    sns.histplot(y_factual_true, alpha=0.3, color='red', label='Y factual true', kde=True, stat='density', ax=axs[1])
    sns.histplot(y_factual_pred, alpha=0.3, color='green', label='Y factual pred', kde=True, stat='density', ax=axs[1])

    axs[0].set_xlabel('Potential outcomes')
    axs[1].set_xlabel('Factual outcomes')
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()
    else:
        plt.show()

def plot_censored_dist(t_true, y_true, y_pred_0, y_pred_1, c_true, path=None):
    y_factual_pred = y_pred_0*(1-t_true) + y_pred_1*t_true

    y_censored = y_true[c_true==1]
    y_factual_pred_censored = y_factual_pred[c_true==1]

    #distribution of outcomes, for censored, predictions must be greater than true values

    sns.histplot(y_censored, alpha=0.3, color='red', label='Y true', kde=True, stat='density')
    sns.histplot(y_factual_pred_censored, alpha=0.3, color='green', label='Y pred', kde=True, stat='density')
    plt.title('censored outcomes y_pred>y_true')
    plt.legend()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_censored_vs_pehe(censored_prop, pehe, ate_error=None, path=None, show=True, log=True):
    # plot scatterplot of PEHE and ATE error vs censored proportion
    '''

    :param censored_prop: list or np.array, is going to be the x-axis
    :param pehe: list or np.array, is going to be the y-axis
    :param ate_error: optional(list, np.array), is going to be the y-axis (copy)
    :param path: saving path
    :return: None
    '''

    fig, ax = plt.subplots()
    ax.scatter(censored_prop, pehe, label='PEHE')
    if ate_error is not None:
        # second y-axis
        ax2 = ax.twinx()
        ax2.scatter(censored_prop, ate_error, label='ATE error', color='red')
    ax.set_xlabel('Censored proportion')
    ax.set_ylabel('PEHE')
    ax.legend(loc='upper left')
    if log:
        ax.set_yscale('log')
    if ate_error is not None:
        ax2.legend(loc='upper right')
        ax2.set_ylabel('ATE error')
        if log:
            ax2.set_yscale('log')
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()
    else:
        plt.show()

def plot_losses_fedavg(losses_dict, path='./', key='loss', log_yscale=False, log_xscale=False, starting_epoch=0, show=True):
    """
    Plot training losses for each model in a federated learning setting.
    :param losses_dict: is a dict with keys as models and the values are other dicts where the keys
    are the components of the losses. Each value of the inner dict is a list with as many values as epochs
    :param path:
    :param key: string to select the loss component
    :return:
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, losses in losses_dict.items():
        if key in losses:
            ax.plot(np.array(losses[key])[starting_epoch:], label=model_name)
        else:
            print(f"Warning: {key} not found in losses for model {model_name}")

    if log_yscale:
        ax.set_yscale('log')
    if log_xscale:
        ax.set_xscale('log')

    ax.set_xlabel('Epochs')
    ax.set_ylabel(key)
    ax.set_title(f'{key} per Epoch for Federated Models')
    ax.legend()
    plt.tight_layout()

    if path:
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_ite_error_cdf(ite_pred_dict, ite_true, path=None, show=True, xlim=None):
    # this method computes the error of the computed ites for each individual and plots the cdf of each


    if isinstance(ite_true, list):
        ite_true_list = ite_true
    else:
        ite_true_list = [ite_true] * len(ite_pred_dict)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (model_name, ite_pred) in enumerate(ite_pred_dict.items()):
        ite_error = np.abs(ite_pred - ite_true_list[i])
        sns.ecdfplot(ite_error, label=model_name, ax=ax)

    ax.set_xlabel('ITE Error')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF of ITE Errors')
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_ite_cdf(ite_pred_dict, path=None, show=True, xlim=None):
    # this method computes the error of the computed ites for each individual and plots the cdf of each


    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (model_name, ite_pred) in enumerate(ite_pred_dict.items()):
        ite = np.abs(ite_pred)
        sns.ecdfplot(ite, label=model_name, ax=ax)

    ax.set_xlabel('ITE')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF of ITEs')
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


# helper for plotting survival probabilities with tmle estimates, since the original library only plots risks
def plot_survival_probabilities(
    tmle_est: pd.DataFrame,
    g_comp_est: Optional[pd.DataFrame] = None,
    color_1: Optional[str] = None,
    color_0: Optional[str] = None,
    use_bootstrap: bool = False,
    ax=None,
) -> tuple:
    target_events = np.unique(tmle_est["Event"])
    mean_key = "mean_bootstrap" if use_bootstrap else "Pt Est"
    ci_lower_key = "CI_lower_bootstrap" if use_bootstrap else "CI_lower"
    ci_upper_key = "CI_upper_bootstrap" if use_bootstrap else "CI_upper"

    all_ci_upper_surv = []
    all_ci_lower_surv = []

    groups = np.unique(tmle_est["Group"])
    assert len(groups) == 2, "Only two groups are supported for survival plotting."

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for i, event in enumerate(target_events):
        used_colors = []
        for group, color in zip(groups, [color_0, color_1]):
            mask_tmle = (tmle_est["Event"] == event) & (tmle_est["Group"] == group)

            time = tmle_est.loc[mask_tmle, "Time"].values
            pt_est_risk = tmle_est.loc[mask_tmle, "Pt Est"].values
            mean_risk = tmle_est.loc[mask_tmle, mean_key].values
            ci_lower_risk = tmle_est.loc[mask_tmle, ci_lower_key].values
            ci_upper_risk = tmle_est.loc[mask_tmle, ci_upper_key].values

            # survival = 1 - risk
            pt_est_surv = 1.0 - pt_est_risk
            mean_surv = 1.0 - mean_risk
            ci_lower_surv = 1.0 - ci_upper_risk  # note flip
            ci_upper_surv = 1.0 - ci_lower_risk  # note flip

            all_ci_upper_surv.append(ci_upper_surv)
            all_ci_lower_surv.append(ci_lower_surv)

            yerr = [mean_surv - ci_lower_surv, ci_upper_surv - mean_surv]

            container = ax.plot(
                time, pt_est_surv, linestyle="--", color=color, marker="o", label='TMLE T=' + str(group)
            )
            used_colors.append(container[0].get_color())
            ax.errorbar(
                time,
                mean_surv,
                yerr=yerr,
                capsize=13,
                color=used_colors[-1],
                linestyle="",
                alpha=0.2
            )

            if g_comp_est is not None:
                mask_g = (g_comp_est["Event"] == event) & (g_comp_est["Group"] == group)
                assert all(
                    time == g_comp_est.loc[mask_g, "Time"].values
                ), "Target times do not match for TMLE and g-computation."
                g_comp_risk = g_comp_est.loc[mask_g, "Pt Est"].values
                g_comp_surv = 1.0 - g_comp_risk
                ax.scatter(
                    time, g_comp_surv, color=used_colors[-1], marker="x", s=100
                )

def plot_survival_curves_with_ci(
    results_dir: str,
    model_name: str,
    M: int,
    K: int,
    starting_seed: int = 0,
    timepoints=None,
    scaler=None,
    outcome_dist = None,
    k_mode: str = "mean",          # "mean" or "all"
    alpha: float = 0.05,
    patient_idx: int | None = None,
    plot_population: bool = True,
    plot_individual: bool = False,
    show_natural_course: bool = False,  # only if you saved it and your aggregator computes it
    max_realisations: int | None = None,
    figsize=(10, 6),
    title: str | None = None,
    save_path: str | None = None,
    show_kaplan_meier: bool = False,
    y_true: torch.Tensor | None = None,
    c_true: torch.Tensor | None = None,
    weight_censored: bool = False,
    tmle_predictions=None,
    over_set: str = "test",
    tau: float | None = None,  # unused here but keeps signature compatibility if you want
):
    if k_mode not in {"mean", "all"}:
        raise ValueError("k_mode must be in {'mean','all'}")
    if plot_individual and patient_idx is None:
        raise ValueError("patient_idx must be provided when plot_individual=True")

    M_load = M if max_realisations is None else min(M, max_realisations)

    agg = aggregate_all(
        results_dir=results_dir,
        model_name=model_name,
        M=M_load,
        K=K,
        starting_seed=starting_seed,
        timepoints=timepoints,
        scaler=scaler,
        tau=0.0 if tau is None else tau,  # required by aggregate_all; not used for curves
        q_ps=(),
        outcome_dist = outcome_dist,
        k_mode=k_mode,
        alpha=alpha,
        over_set=over_set,
        per_individual=plot_individual,
        population_summary=plot_population,
        use_ipcw=weight_censored,
        only_survival=True,
    )

    tp = agg["timepoints"].detach().cpu().numpy()

    def _plot_band(ax, summary_stats, label, linestyle="-", band_alpha=0.2):
        m = summary_stats["mean"].detach().cpu().numpy()
        lo = summary_stats["q_low"].detach().cpu().numpy()
        hi = summary_stats["q_high"].detach().cpu().numpy()
        ax.plot(tp, m, linestyle=linestyle, label=label)
        ax.fill_between(tp, lo, hi, alpha=band_alpha)

    def _select_patient(stats_dict, idx):
        return {k: v[idx] for k, v in stats_dict.items()}

    fig, ax = plt.subplots(figsize=figsize)

    # Population curves
    if plot_population:
        pop = agg["population"]
        _plot_band(ax, pop["survival_t0"], label="Population: Treatment 0")
        _plot_band(ax, pop["survival_t1"], label="Population: Treatment 1")

        if show_natural_course and "survival_natural_course" in pop:
            _plot_band(ax, pop["survival_natural_course"], label="Population: Natural course")

        if show_kaplan_meier:
            if y_true is None or c_true is None:
                raise ValueError("y_true and c_true must be provided to plot Kaplan-Meier estimate")
            from lifelines import KaplanMeierFitter
            kmf = KaplanMeierFitter()
            kmf.fit(y_true, event_observed=(1 - c_true))
            kmf.plot_survival_function()




        if tmle_predictions is not None:
            plot_survival_probabilities(
                tmle_est=tmle_predictions,
                color_1="C1",
                color_0="C0",
                use_bootstrap=False,
                ax = ax,
            )

    # Individual curves
    if plot_individual:
        indiv = agg["individual"]
        i0 = _select_patient(indiv["survival_t0"], patient_idx)
        i1 = _select_patient(indiv["survival_t1"], patient_idx)

        _plot_band(ax, i0, label=f"Individual {patient_idx}: Treatment 0", linestyle="--", band_alpha=0.15)
        _plot_band(ax, i1, label=f"Individual {patient_idx}: Treatment 1", linestyle="--", band_alpha=0.15)
        #
        # if show_natural_course and "survival_natural" in indiv:
        #     iN = _select_patient(indiv["survival_natural"], patient_idx)
        #     _plot_band(ax, iN, label=f"Individual {patient_idx}: Natural course", linestyle="--", band_alpha=0.12)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0.0, 1.0)

    if title is None:
        title = f"Survival curves with {(1-alpha)*100:.0f}% CI (k_mode={k_mode})"
        if weight_censored:
            title += " (IPCW)"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    return fig, ax

## STATISTICAL ANALYSIS

def friedman_test(all_data, comp_index, alpha, higher_is_better):
    """
    Perform the Friedman test on the provided data. Based on Demsar06.
    :param all_data: 2D numpy array of shape (n_methods, n_datasets) where each row is a method and each column is a dataset.
    :param comp_index: Method to set as baseline for post-hoc tests, as in Demsar06. Should be the best performing metric...
    :param alpha: significance level for the test.
    :return: Friedman test p_value, davenport p-value and pairwise to the best baseline post-hoc p-values.
    """
    # Check that comp_index gives the best performing method (double check just in case...)
    avg_performance = np.mean(all_data, axis=1)  # Average performance across datasets for each method
    if higher_is_better:
        assert comp_index == np.argmax(avg_performance), "comp_index must be the index of the best performing method."
    else:
        assert comp_index == np.argmin(avg_performance), "comp_index must be the index of the best performing method."
    # Manual implementation of the Friedman test--to compute post-hoc metrics later on
    n_methods, n_reps = all_data.shape
    ranking_matrix = np.zeros_like(all_data)

    for k in range(n_reps):
        # Rank the methods for each dataset/fold
        if higher_is_better:
            ranking_matrix[:, k] = stats.rankdata(-all_data[:, k], method='average')  # Average ranks for ties
        else:
            ranking_matrix[:, k] = stats.rankdata(all_data[:, k], method='average')  # Average ranks for ties

    # Calculate the Friedman test statistic
    average_rank = np.mean(ranking_matrix, axis=1)
    friedman_stat = (12 * n_reps/ (n_methods * (n_methods + 1))) * (np.sum(np.square(average_rank)) - (n_methods * (n_methods + 1) ** 2 / 4))  # Friedman test statistic
    friedman_p_value = stats.chi2.sf(friedman_stat, df=n_methods - 1)  # p-value for the Friedman test
    davenport_stat = friedman_stat * (n_reps - 1) / (n_reps * (n_methods - 1))  # Davenport's statistic
    davenport_p_value = stats.f.sf(davenport_stat, dfn=n_methods - 1, dfd=(n_methods - 1) * (n_methods) * (n_reps - 1))

    # If we reject, we can perform post-hoc tests here. # TODO: Unsure if this is OK, need to account for higher is better in the p-values!!
    z_stat = np.zeros(n_methods)
    for j in range(n_methods):
        z_stat[j] = (average_rank[comp_index] - average_rank[j]) / np.sqrt((n_methods * (n_methods + 1)) / (6 * n_reps))  # Z-statistic for post-hoc tests
    p_values_post_hoc = stats.norm.cdf(z_stat)
    _, p_values_adjusted_post_hoc, _, _ = multitest.multipletests(p_values_post_hoc, alpha=alpha, method='holm')  # Holm-Bonferroni correction
    return friedman_p_value, davenport_p_value, p_values_post_hoc, p_values_adjusted_post_hoc

def get_p_values_from_table_data(data, alpha=0.05, higher_is_better=True, output_latex=True, list_of_methods=None, list_of_metrics=None, show_data=True):
    """
    Function to get p-values from a table of data in a structured way, automatically comparing with the best method for each metric.
    :param data: Organized as a numpy array: methods_to_compare x metrics x datasets/folds. Note that all datasets/folds need to have the same ordering: we use paired tests!!
    :param alpha: float, significance level for the hypothesis test.
    :param higher_is_better: bool or list of bool, if True, higher values are better, otherwise lower values are better.
    :param output_latex: bool, if True, outputs the table in LaTeX format, to copy and paste into a LaTeX document.
    :param list_of_methods: List of method names, if None, uses the default names.
    :param list_of_metrics: List of metric names, if None, uses the default names.
    :return: Outputs a p-value table comparing each method to the specified comparison method.
    """

    assert isinstance(data, np.ndarray), "Data must be a numpy array."
    assert data.ndim == 3, "Data must be a 3D numpy array with shape (n_methods, n_metrics, n_reps)."

    n_methods, n_metrics, n_reps = data.shape
    average_results = np.mean(data, axis=2)  # Average over repetitions, we have an array of shape (n_methods, n_metrics)

    if list_of_methods is None:
        list_of_methods = [f'Method {i+1}' for i in range(data.shape[0])]
    if list_of_metrics is None:
        list_of_metrics = [f'Metric {i+1}' for i in range(data.shape[1])]

    if not isinstance(higher_is_better, bool):
        assert len(higher_is_better) == n_metrics, "If higher_is_better is a list, it must have the same length as the number of metrics."
    else:
        higher_is_better = [higher_is_better] * data.shape[1]  # If it's a single bool, replicate it for all metrics

    max_idxs = np.argmax(average_results, axis=0)
    min_idxs = np.argmin(average_results, axis=0)
    comp_index = [max_idxs[i] if higher_is_better[i] else min_idxs[i] for i in range(n_metrics)]

    for i in range(n_metrics):

        # Print the data for complete reference
        print(f'\nData for metric {list_of_metrics[i]}, where higher_is_better is {higher_is_better[i]}:')
        if show_data:
            for j in range(n_methods):
                print(f'{list_of_methods[j]}: {data[j, i, :]}:.3f / avg: {np.mean(data[j, i, :]):.3f}')
        table_metrics = ['Average metric'] + [f"{np.mean(data[j, i, :]):.3f}" for j in range(n_methods)]
        # First method: use paired Wilcoxon signed-rank test to obtain p-values, and correct them using Holm-Bonferroni method. This is done per-metric, so if we have many metrics, we will have many p-values.

        baseline_values = data[comp_index[i], i, :]  # Baseline values for the metric
        p_values = []
        for j in range(n_methods):
            test_values = data[j, i, :]  # Test values for the metric
            if comp_index[i] == j: # If we are comparing the baseline method with itself, we skip this comparison, as the Wilcoxon test will throw an error
                p_values.append(1.0)  # No difference, p-value is 1
                continue
            if higher_is_better[i]:
                # If higher is better, we want to test if the test values are significantly lower than the baseline values (i.e., significantly worse)
                _, p_value = stats.wilcoxon(test_values, baseline_values, alternative='less')
            else:
                # If lower is better, we want to test if the test values are significantly higher than the baseline values (i.e., significantly worse)
                _, p_value = stats.wilcoxon(test_values, baseline_values, alternative='greater')
            p_values.append(p_value)
        # Apply Holm-Bonferroni correction
        p_values = np.array(p_values)
        _, corrected_p_vals, _, _ = multitest.multipletests(np.array(p_values), alpha=alpha, method='holm')
        # Prepare a table to store all data for this metric
        table_wilcoxon_corr = ['Paired Wilcoxon tests (corrected)']
        table_wilcoxon_unc = ['Paired Wilcoxon tests (uncorrected)']
        for j in range(n_methods):
            p_val_str = f"{corrected_p_vals[j]:.3f}" if corrected_p_vals[j] >= 1e-3 else "<1e-3"  # Format p-values
            if corrected_p_vals[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            table_wilcoxon_corr.append(p_val_str)

            p_val_str = f"{p_values[j]:.3f}" if p_values[j] >= 1e-3 else "<1e-3"  # Format small p-values
            if p_values[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            table_wilcoxon_unc.append(p_val_str)

        # Second method: the Friedman test, which is a non-parametric test for repeated measures done on all metrics at once. Blocks = methods, treatments = datasets / folds (we could also implement one on datasets * metrics, a general one, later on). We rely on Demsar06 for this implementation.
        friedman_p_value, davenport_p_value, p_values_post_hoc_unc, p_values_post_hoc_corr = friedman_test(data[:, i, :], comp_index[i], alpha, higher_is_better[i])

        # Prepare this for the table
        friedman_post_hoc_table_corr = ['Friedman post-hoc tests (Corrected)']
        friedman_post_hoc_table_unc = ['Friedman post-hoc tests (Uncorrected)']
        for j in range(n_methods):
            p_val_str = f"{p_values_post_hoc_corr[j]:.3f}" if p_values_post_hoc_corr[j] >= 1e-3 else "<1e-3"  # Format p-values
            if p_values_post_hoc_corr[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            friedman_post_hoc_table_corr.append(p_val_str)

            p_val_str = f"{p_values_post_hoc_unc[j]:.3f}" if p_values_post_hoc_unc[j] >= 1e-3 else "<1e-3"  # Format small p-values
            if p_values_post_hoc_unc[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            friedman_post_hoc_table_unc.append(p_val_str)
        if friedman_p_value < 1e-3:
            friedman_p_value_str = "<1e-3"  # Format small p-values
        else:
            friedman_p_value_str = f"{friedman_p_value:.3f}"
        if davenport_p_value < 1e-3:
            davenport_p_value_str = "<1e-3"  # Format small p-values
        else:
            davenport_p_value_str = f"{davenport_p_value:.3f}"
        print(f'Friedman p-value: {friedman_p_value_str}, Davenport p-value: {davenport_p_value_str} for metric {list_of_metrics[i]}')
        if n_reps <= 10 or n_methods <=5:
            print('Since the number of data points is small, the Friedman test may not be reliable. Consider using a larger dataset or a different test.')

        table_data = [table_metrics, table_wilcoxon_unc, table_wilcoxon_corr, friedman_post_hoc_table_unc, friedman_post_hoc_table_corr]

        if output_latex:
            print(tabulate(table_data, headers=[f'Metric {list_of_metrics[i]}'] + list_of_methods, tablefmt='latex'))
        else:
            print(tabulate(table_data, headers=[f'Metric {list_of_metrics[i]}'] + list_of_methods, tablefmt='grid'))

    # Finally, run a Friedman test on all metrics at once
    all_data = data.copy()
    # For all metrics where lower is better, we need to invert the data so that higher is better always
    for j in range(n_metrics):
        if not higher_is_better[j]:
            all_data[:, j, :] = -all_data[:, j, :]  # Invert the data for lower is better metrics
    # Now, reshape the data to have shape (n_methods, n_metrics * n_reps)
    all_data = all_data.reshape(n_methods, n_metrics * n_reps)
    avg_metrics = np.mean(all_data, axis=1)  # Average over repetitions and metrics
    best_method = np.argmax(avg_metrics)  # Best method across all metrics (remember, higher is better now!)
    friedman_p_value, davenport_p_value, p_values_post_hoc_unc, p_values_post_hoc_corr = friedman_test(all_data, best_method, alpha, higher_is_better=True)
    print(f'Friedman test on all metrics: p-value: {friedman_p_value:.4f}, Davenport p-value: {davenport_p_value:.4f}')
    # Prepare this for the table
    friedman_post_hoc_table_unc = ['Friedman post-hoc tests (all metrics, uncorrected)']
    friedman_post_hoc_table_corr = ['Friedman post-hoc tests (all metrics, corrected)']
    for j in range(n_methods):
        p_val_str = f"{p_values_post_hoc_corr[j]:.3f}" if p_values_post_hoc_corr[j] >= 1e-3 else "<1e-3"  # Format p-values
        if p_values_post_hoc_corr[j] >= alpha:
            p_val_str += '*'  # Mark best values
        if j == best_method:
            p_val_str += ' (baseline)'  # Mark the baseline method
        friedman_post_hoc_table_corr.append(p_val_str)

        p_val_str = f"{p_values_post_hoc_unc[j]:.3f}" if p_values_post_hoc_unc[j] >= 1e-3 else "<1e-3"  # Format small p-values
        if p_values_post_hoc_unc[j] >= alpha:
            p_val_str += '*'  # Mark best values
        if j == best_method:
            p_val_str += ' (baseline)'  # Mark the baseline method
        friedman_post_hoc_table_unc.append(p_val_str)
    table_data = [friedman_post_hoc_table_unc, friedman_post_hoc_table_corr]
    if output_latex:
        print(tabulate(table_data, headers=['All metrics'] + list_of_methods, tablefmt='latex'))
    else:
        print(tabulate(table_data, headers=['All metrics'] + list_of_methods, tablefmt='grid'))

