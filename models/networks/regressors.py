import numpy as np
import torch
import pandas as pd
import scipy.stats as stats

from torch import nn
from models.networks.mlp import MLP
from models.networks.kan import KANnet

from torch.distributions import Normal, Weibull, Bernoulli, Categorical, Poisson, NegativeBinomial, Exponential, Gumbel
from typing import Union

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings

from models.utils import get_activation, get_default_dict

class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

    def summary(self, X_df, y):
        """
        Computes a summary similar to sklearn's linear regression output.
        X: torch.Tensor, shape (n_samples, n_features)
        y: torch.Tensor, shape (n_samples, 1)
        """
        X = torch.tensor(X_df.values, dtype=torch.float32)
        variable_names = list(X_df.columns)
        n, p = X.shape  # Number of samples and features
        X_ext = torch.cat([X, torch.ones(n, 1)], dim=1)  # Add bias term
        variable_names.append("Intercept")  # Name for the bias term

        # Get parameters
        weights = self.linear.weight.detach().cpu().numpy().flatten()
        bias = self.linear.bias.detach().cpu().numpy()
        beta = np.append(weights, bias)

        # Predictions & Residuals
        y_pred = self.forward(X).detach().cpu().numpy().flatten()
        y_true = y.flatten()
        residuals = y_true - y_pred

        # Variance & Standard Errors
        rss = np.sum(residuals ** 2)  # Residual sum of squares
        mse = rss / (n - p - 1)  # Mean Squared Error
        XTX_inv = np.linalg.inv(X_ext.T @ X_ext)  # (X'X)^(-1)
        std_errors = np.sqrt(np.diagonal(mse * XTX_inv))  # Standard errors

        # t-values and p-values
        t_values = beta / std_errors
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - p - 1)) for t in t_values]

        r2 = r2_score(y_true, y_pred)  # R² (coefficient of determination)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))  # Adjusted R²
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
        mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error

        # Store results in a DataFrame
        results_df = pd.DataFrame({
            "Variable": variable_names,
            "Coefficient": beta,
            "Std. Error": std_errors,
            "t-value": t_values,
            "p-value": p_values
        })

        # Print additional performance metrics
        print(f"\nPerformance Metrics:")
        print(f"R²: {r2:.4f}")
        print(f"Adjusted R²: {adj_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return results_df


class LogisticRegressor(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)  # Linear layer

    def forward(self, x):
        return self.linear(x)  # Apply logistic function

    def summary(self, X_df, y):
        """
        Computes a summary similar to sklearn's logistic regression output.
        X: torch.Tensor, shape (n_samples, n_features)
        y: torch.Tensor, shape (n_samples, 1) (binary labels 0 or 1)
        """
        X = torch.tensor(X_df.values, dtype=torch.float32)
        variable_names = list(X_df.columns)
        n, p = X.shape  # Number of samples, number of features
        X_ext = torch.cat([X, torch.ones(n, 1)], dim=1)  # Add bias term
        variable_names.append("Intercept")  # Name for the bias term

        # Get model parameters
        weights = self.linear.weight.detach().cpu().numpy().flatten()
        bias = self.linear.bias.detach().cpu().numpy()
        beta = np.append(weights, bias)  # Append intercept to coefficients

        # Compute predicted probabilities
        logits = self.forward(X).detach().cpu()
        probs = torch.sigmoid(logits).numpy().flatten()
        probs = np.clip(probs, 1e-5, 1 - 1e-5)  # Avoid log(0) issues

        # Compute the diagonal weight matrix W (n x n)
        W = np.diag(probs * (1 - probs))

        # Compute the variance-covariance matrix: (X'WX)^(-1)
        try:
            XTWX_inv = np.linalg.inv(X_ext.T @ W @ X_ext)  # Fisher Information matrix inverse
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Standard errors may be unreliable.")
            XTWX_inv = np.linalg.pinv(X_ext.T @ W @ X_ext)  # Use pseudo-inverse if singular
        std_errors = np.sqrt(np.diagonal(XTWX_inv))  # Standard errors

        # Compute Wald test statistics (t-values) and p-values
        t_values = beta / std_errors
        p_values = [2 * (1 - stats.norm.cdf(np.abs(t))) for t in t_values]

        # Compute log-likelihood values for Likelihood Ratio Test
        log_likelihood_full = np.sum(
            y.flatten() * np.log(probs) + (1 - y.flatten()) * np.log(1 - probs))
        mean_y = y.mean()
        log_likelihood_null = np.sum(
            y.flatten() * np.log(mean_y) + (1 - y.flatten()) * np.log(1 - mean_y))

        # Compute Likelihood Ratio Statistic
        likelihood_ratio_stat = -2 * (log_likelihood_null - log_likelihood_full)
        p_value_lrt = 1 - stats.chi2.cdf(likelihood_ratio_stat, df=p)

        # Compute McFadden's Pseudo-R²
        pseudo_r2 = 1 - (log_likelihood_full / log_likelihood_null)

        # Store results in a DataFrame
        results_df = pd.DataFrame({
            "Variable": variable_names,
            "Coefficient": beta,
            "Std. Error": std_errors,
            "t-value": t_values,
            "p-value": p_values
        })


        print(f"\nLog-Likelihood (Full Model): {log_likelihood_full:.4f}")
        print(f"Log-Likelihood (Null Model): {log_likelihood_null:.4f}")
        print(f"Likelihood Ratio Test Statistic: {likelihood_ratio_stat:.4f}")
        print(f"LRT p-value: {p_value_lrt:.4f}")
        print(f"McFadden’s Pseudo-R²: {pseudo_r2:.4f}")

        return results_df


class treatment_regressor(nn.Module):
    def __init__(self, network_name, input_dim, hidden_sizes, num_treatments, dropout, activation, device,
                 weight_init):
        super(treatment_regressor, self).__init__()

        self.network_name=network_name
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

        if network_name == 'mlp':
            self.net = MLP(input_dim, hidden_sizes, self.output_dim, self.dropout, self.activation, self.device, self.weight_init)
        elif network_name == 'linear':
            self.net = nn.Linear(input_dim, self.output_dim)
        elif network_name == 'KAN':
            self.net = KANnet(input_dim, hidden_sizes, self.output_dim, self.device)
        else:
            raise ValueError('Invalid network name')

    def forward(self, x):
        if self.num_treatments == 2:
            p = nn.Sigmoid()(self.net(x))
            return Bernoulli(probs=p)
        p = nn.Softmax()(self.net(x))
        return Categorical(probs=p)

    def loss(self, p_y_pred, y_true):
        y_true = y_true.view(-1, 1)
        return -p_y_pred.log_prob(y_true)

    def predict(self, x):
        # return the most probable treatment
        if self.num_treatments == 2:
            return self.net(x) > 0.5
        else:
            aux = torch.zeros((x.shape[0], self.num_treatments))
            aux[torch.arange(x.shape[0]), self.net(x).argmax(dim=1)] = 1
            return aux




class outcome_regressor(nn.Module):
    def __init__(self, network_name,
                 input_dim, hidden_sizes, outcome_dist,
                 device,
                 weight_init=None, dropout=0.0, activation='relu',
                 min_std=None, max_std=None, min_shape=None, max_shape=None, min_scale=None, max_scale=None, fixed_std=None,
                 limit_variance=False, fix_variance=True, limit_shape=False, limit_scale=False,
                 min_outcome=None, kan_params=None):
        super(outcome_regressor, self).__init__()
        '''
        :param min_std: float, minimum standard deviation REQUIRED IF limit_variance is True // default value 1e-3
        :param max_std: float, maximum standard deviation REQUIRED IF limit_variance is True // default value 2.0
        :param min_shape: float, minimum shape parameter REQUIRED IF limit_shape is True // default value 0.5
        :param max_shape: float, maximum shape parameter REQUIRED IF limit_shape is True // default value 10.0
        :param min_scale: float, minimum scale parameter REQUIRED IF limit_scale is True // default value 0.5
        :param max_scale: float, maximum scale parameter REQUIRED IF limit_scale is True // default value 2.0
        :param fixed_std: float, fixed standard deviation REQUIRED IF fix_variance is True // default value 1.0
        '''
        self.network_name = network_name
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.outcome_dist = outcome_dist
        self.device = device
        self.dropout = dropout
        self.activation = activation
        self.weight_init = weight_init

        self.limit_variance = limit_variance
        self.limit_shape = limit_shape
        self.limit_scale = limit_scale
        self.fix_variance = fix_variance

        if (fix_variance and limit_variance):
            warnings.warn(
                'fix_variance and limit_variance are both True. This is not possible. limit_variance will be set to False')
            self.limit_variance = False
        if limit_variance:
            if min_std is None:
                # warning
                warnings.warn('min_std must be provided if limit_variance is True')
                min_std = get_default_dict()['min_std']
            if max_std is None:
                # warning
                warnings.warn('max_std must be provided if limit_variance is True')
                max_std = get_default_dict()['max_std']

        if limit_shape:
            if min_shape is None:
                # warning
                warnings.warn('min_shape must be provided if limit_shape is True')
                min_shape = get_default_dict()['min_shape']
            if max_shape is None:
                # warning
                warnings.warn('max_shape must be provided if limit_shape is True')
                max_shape = get_default_dict()['max_shape']
        if limit_scale:
            if min_scale is None:
                # warning
                warnings.warn('min_scale must be provided if limit_scale is True')
                min_scale = get_default_dict()['min_scale']
            if max_scale is None:
                # warning
                warnings.warn('max_scale must be provided if limit_scale is True')
                max_scale = get_default_dict()['max_scale']
        if fix_variance:
            if fixed_std is None:
                # warning
                warnings.warn('fixed_std must be provided if fix_variance is True')
                fixed_std = get_default_dict()['fixed_std']


        self.min_std = min_std
        self.max_std = max_std
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fixed_std = fixed_std
        self.min_outcome = min_outcome
        
        if self.outcome_dist == 'not-specified':
            self.output_dim = 1
        elif self.outcome_dist == 'binary' or self.outcome_dist == 'bernoulli':
            self.output_dim = 1
        elif self.outcome_dist == 'gaussian' or self.outcome_dist=='normal' or self.outcome_dist == 'continuous' or self.outcome_dist == 'log-normal':
            if self.fix_variance:
                self.output_dim = 1
            else:
                self.output_dim = 2
        elif self.outcome_dist == 'exponential':
            self.output_dim = 1
        elif self.outcome_dist == 'weibull':
            self.output_dim = 2
        elif self.outcome_dist == 'gumbel':
            self.output_dim = 2
        elif 'categorical' in self.outcome_dist:  # todo: weibull categorical, as in DeepHit
            k = int(self.outcome_dist.split('_')[1])
            self.output_dim = k
        elif self.outcome_dist == 'negative-binomial':
            self.output_dim = 2
        else:
            raise ValueError('Invalid outcome distribution')

        if network_name == 'mlp':
            self.net = MLP(input_dim, hidden_sizes, self.output_dim, self.dropout, self.activation, self.device, self.weight_init)
        elif network_name == 'linear':
            if self.outcome_dist == 'not-specified':
                self.net = LinearRegressor(input_dim)
            elif self.outcome_dist == 'binary' or self.outcome_dist == 'bernoulli':
                self.net = LogisticRegressor(input_dim)
            else:
                self.net = nn.Linear(input_dim, self.output_dim)
        elif network_name == 'KAN':
            self.net = KANnet(input_dim, hidden_sizes, self.output_dim, self.device, kan_params=kan_params)
        else:
            raise ValueError('Invalid network name')


    def forward(self, x: torch.Tensor) -> Union[Bernoulli, Normal, Weibull, Categorical,
                                                NegativeBinomial, Exponential, Gumbel]:
        if self.outcome_dist == 'not-specified':
            return self.net(x)
        if self.outcome_dist == 'binary' or self.outcome_dist == 'bernoulli':
            p = torch.sigmoid(self.net(x))
            return Bernoulli(probs=p)
        elif self.outcome_dist == 'gaussian' or self.outcome_dist=='normal' or self.outcome_dist == 'continuous' or self.outcome_dist=='log-normal':
            if self.fix_variance:
                mu = self.net(x)
                std = torch.tensor(self.fixed_std)
            else:
                mu, log_var = torch.chunk(self.net(x), 2, dim=-1)
                if self.limit_variance:
                    std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(log_var)
                else:
                    std = torch.exp(0.5 * log_var)
            if self.min_outcome is not None:
                mu = nn.Softplus()(mu) + self.min_outcome
            return Normal(loc=mu, scale=std)
        elif 'categorical' in self.outcome_dist:
            k = int(self.outcome_dist.split('_')[1])
            logits = self.net(x)
            return Categorical(logits=logits)
        elif self.outcome_dist == 'weibull':
            log_alpha, log_k = torch.chunk(self.net(x), 2, dim=-1)  # both positive parameters
            if self.limit_shape:
                k = self.min_shape + (self.max_shape - self.min_shape) * torch.sigmoid(log_k) # In case of limiting, the output has the meaning of shape
            else:
                k = torch.exp(log_k)  # In case of not limiting, the output has the meaning of log(shape)
            if self.limit_scale: # It is important to limit the shape to avoid instability...
                alpha = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(log_alpha)  # In Case of limiting, the output has the meaning of scale
            else:
                alpha = torch.exp(log_alpha)  # In case of not limiting, the output has the meaning of log(scale)

            return Weibull(scale=alpha, concentration=k)

        elif self.outcome_dist == 'gumbel':
            mu, log_beta = torch.chunk(self.net(x), 2, dim=-1)  # both parameters
            if self.limit_scale:
                beta = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(log_beta)
            else:
                beta = torch.nn.Softplus()(log_beta)
            return Gumbel(loc=mu, scale=beta)
        elif self.outcome_dist == 'exponential':
            log_rate = self.net(x)
            rate = torch.exp(log_rate)
            return Exponential(rate=rate)
        elif self.outcome_dist == 'negative-binomial':
            total_count, logits = torch.chunk(self.net(x), 2, dim=-1)
            total_count = torch.nn.Softplus()(total_count) + 1
            return NegativeBinomial(total_count=total_count, logits=logits)
        else:
            raise ValueError('Invalid outcome distribution')

    def log_prob(self, y_dist_pred: Union[Bernoulli, Normal, Weibull, Categorical,
                                    Gumbel, Exponential, NegativeBinomial], y_true: torch.Tensor,
                 c_true: torch.Tensor = None) -> torch.Tensor:
        # c_true is the censoring indicator
        # c_true = 1 if y_true is observed, c_true = 0 if y_true is censored

        if self.outcome_dist == 'not-specified':
            raise ValueError('Outcome distribution not specified, log likelihood cannot be computed')

        if 'categorical' not in self.outcome_dist:
            y_true = y_true.view(-1, 1)  # y true must be a tensor [-1,1]
        log_prob = y_dist_pred.log_prob(y_true)
        if c_true is None:
            return log_prob
        else:
            c_true = c_true.view(-1, 1)
            if self.outcome_dist == 'weibull':
                k, alpha = y_dist_pred.concentration, y_dist_pred.scale
                # survival_function = torch.exp(-((y_true/beta)**alpha))
                log_survival_function = -((y_true / alpha) ** k)  # simpler than torch cdf
            elif self.outcome_dist == 'gumbel':
                # assuming that gumbel distribution is used because we are modeling log-times
                # gumbel over log-time: G = log T ~ Gumbel(loc=mu, scale=beta)
                g = torch.log(y_true.clamp_min(1e-12))

                mu = y_dist_pred.loc  # location
                beta = y_dist_pred.scale  # scale > 0

                z = -(g - mu) / beta  # z = -(g - mu)/beta
                # F = exp(-exp(z))
                # logF = -exp(z)
                logF = -torch.expm1(z)

                # logS = log(1 - F) = log(1 - exp(logF))
                # stable: logS = log(-expm1(logF))
                cdf = -torch.expm1(logF)
                survival_function = 1 - cdf
                log_survival_function = torch.log(survival_function.clamp_min(1e-12))
            elif self.outcome_dist == 'gaussian' or self.outcome_dist == 'normal' or self.outcome_dist == 'continuous' or self.outcome_dist == 'log-normal':
                log_survival_function = torch.log(torch.clamp(1 - y_dist_pred.cdf(y_true), min=0.0001))
            elif self.outcome_dist == 'exponential':
                # log_survival_function =  torch.log(torch.clamp(1 - y_dist_pred.cdf(y_true), min=0.001))
                log_survival_function = -y_true * y_dist_pred.rate
            elif self.outcome_dist == 'negative-binomial':
                # ensure y_true is a tensor of integers
                y_true = y_true.view(-1).long()
                cdf = torch.zeros_like(y_true)
                for i in range(y_true.shape[0]):
                    dist_i = NegativeBinomial(total_count=y_dist_pred.total_count[i], logits=y_dist_pred.logits[i])
                    cdf[i] = torch.sum(torch.exp(dist_i.log_prob(torch.arange(0, y_true[i] + 1).to(self.device))), dim=0)
                # cdf = torch.cumsum(torch.exp(y_dist_pred.log_prob(torch.arange(y_true, max_value))), dim=0)
                # cdf = y_dist_pred.cdf(y_true)
                log_survival_function = torch.log(1 - cdf[y_true.long() - 1])
            else:
                # for discrete distributions, S(t) = \prod_{1}^{k} 1 - h(k), where h(t) is the hazard function
                # todo: implement this
                raise NotImplementedError('Censoring not implemented for this distribution')

            if self.outcome_dist != 'gumbel':
                survival_log_prob = (c_true * log_survival_function + (
                            1 - c_true) * log_prob)  # c is censoring, 1 is censored, 0 is
            else:
                survival_log_prob = (c_true * log_survival_function + (
                            1 - c_true) * (log_prob - torch.log(y_true)))  # c is censoring, 1 is censored, 0 is observed

            return survival_log_prob

    def mse(self, y_dist_pred: Union[Bernoulli, Normal, Weibull, Categorical, Exponential, NegativeBinomial, Gumbel, torch.Tensor], y_true: torch.Tensor,
            c_true: torch.Tensor = None) -> torch.Tensor:
        se = self.squared_error(y_dist_pred, y_true, c_true)
        return torch.mean(se)

    def squared_error(self, y_dist_pred: Union[Bernoulli, Normal, Weibull, Categorical, Exponential, NegativeBinomial, Gumbel, torch.Tensor], y_true: torch.Tensor,
                      c_true: torch.Tensor = None) -> torch.Tensor:
        y_true = y_true.view(-1, 1)
        if self.outcome_dist == 'not-specified':
            y_pred = y_dist_pred
        else:
            y_pred = y_dist_pred.mean

        se = ((y_true - y_pred) ** 2)
        if c_true is not None:
            c_true = c_true.view(-1, 1)
            se = se[c_true == 1]
        return se

    def compute_regularization(self):
        if self.network_name == 'KAN':
            return self.net.compute_regularization()
        else:
            return 0

    def on_training_start(self):
        if self.network_name == 'KAN':
            self.net.on_training_start()
        else:
            pass

    def on_epoch_start(self, epoch, n_epochs, x):
        if self.network_name == 'KAN':
            self.net.on_epoch_start(epoch, n_epochs, x)
        else:
            pass
    def on_epoch_end(self):
        if self.network_name == 'KAN':
            self.net.on_epoch_end()
        else:
            pass