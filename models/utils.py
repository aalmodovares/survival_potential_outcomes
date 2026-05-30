
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import os
import pickle
from torch.distributions import Normal

# dafault values for limiting statistics
def get_default_dict():
    '''
    :param min_std: float, minimum standard deviation REQUIRED IF limit_variance is True // default value 1e-3
    :param max_std: float, maximum standard deviation REQUIRED IF limit_variance is True // default value 2.0
    :param min_shape: float, minimum shape parameter REQUIRED IF limit_shape is True // default value 0.5
    :param max_shape: float, maximum shape parameter REQUIRED IF limit_shape is True // default value 10.0
    :param min_scale: float, minimum scale parameter REQUIRED IF limit_scale is True // default value 0.5
    :param max_scale: float, maximum scale parameter REQUIRED IF limit_scale is True // default value 2.0
    :param fixed_std: float, fixed standard deviation REQUIRED IF fix_variance is True // default value 1.0
    '''
    default_dict = {
        'min_std': 1e-3,
        'max_std': 2.0,
        'min_shape': 0.5,
        'max_shape': 10.0,
        'min_scale': 0.5,
        'max_scale': 2.0,
        'fixed_std': 1.0
    }
    return default_dict

def get_scaler(outcome_dist):
    if outcome_dist == 'log-normal':
        log_transformer = FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1,
            validate=True
        )

        # 2. Build pipeline: log → standardize
        log_standard_scaler = Pipeline([
            ('log', log_transformer),
            ('std', StandardScaler())
        ])
        y_scaler = log_standard_scaler
        scaler_ = log_standard_scaler[1]

    elif outcome_dist == 'gaussian':
        y_scaler = StandardScaler()
        scaler_ = y_scaler

    elif outcome_dist == 'weibull':
        y_scaler = FunctionTransformer(lambda x: x + 1.e-3) #no scaling
        scaler_ = None

    else:
        raise ValueError(f"Unsupported outcome distribution: {outcome_dist}")

    return y_scaler, scaler_

def get_activation(activation_str):
    activation_modules = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "softmax": nn.Softmax(dim=-1),
        "softplus": nn.Softplus(),
        # Add more activation functions if required
        'linear': nn.Identity()
    }

    if activation_str not in activation_modules:
        raise ValueError(f"Unsupported activation function: {activation_str}")

    return activation_modules[activation_str]

def inverse_probability_metric(phi, t, name):
    # todo: adapt to multiple treatments
    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]
    Xc = phi[ic]
    Xt = phi[it]

    if name == 'mmd':
        dist = mmd(Xc,Xt)
    elif name == 'wasserstein':
        dist, imb_mat = wasserstein(Xc, Xt, p=0.5, backpropT=True) # todo: hardcoded p to 0.5
    elif name == 'wasserstein2':
        dist, imb_mat = wasserstein(Xc, Xt, p=0.5, sq=True, backpropT=True)
    else:
        raise ValueError('Invalid divergence loss')
    return dist


def compute_kernel(x, y, kernel_type="rbf", sigma=None):
    if kernel_type == "rbf":
        if sigma is None:
            sigma = torch.median(torch.pdist(x)) + torch.median(torch.pdist(y))
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")


def mmd(x, y, kernel_type="rbf", sigma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples x and y.

    Args:
        x (Tensor): A PyTorch tensor of shape (n_x, d), where n_x is the number of samples in x and d is the dimension.
        y (Tensor): A PyTorch tensor of shape (n_y, d), where n_y is the number of samples in y and d is the dimension.
        kernel_type (str): The type of kernel to use. Currently, only 'rbf' (Radial Basis Function) is supported.
        sigma (float, optional): The bandwidth parameter for the RBF kernel. If None, it will be estimated using the median heuristic.

    Returns:
        float: The MMD value between x and y.
    """
    k_xx = compute_kernel(x, x, kernel_type, sigma)
    k_yy = compute_kernel(y, y, kernel_type, sigma)
    k_xy = compute_kernel(x, y, kernel_type, sigma)

    mmd = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
    return mmd

def wasserstein(Xc, Xt, p, lam=10, its=10, sq=False, backpropT=True):
    """ Returns the Wasserstein distance between treatment groups """
    nc = float(Xc.shape[0])
    nt = float(Xt.shape[0])


    # Compute distance matrix
    if sq:
        M = torch.cdist(Xt, Xc, p=2) ** 2
    else:
        M = torch.cdist(Xt, Xc, p=2)

    # Estimate lambda and delta
    M_mean = torch.mean(M)
    delta = M.max().detach()
    eff_lam = (lam / M_mean).detach()

    # Compute new distance matrix
    Mt = M
    row = delta * torch.ones((1, M.shape[1]))
    col = torch.cat([delta * torch.ones((M.shape[0], 1)), torch.zeros((1, 1))], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)

    # Compute marginal vectors
    a = torch.cat([p * torch.ones(int(nt), 1) / nt, (1 - p) * torch.ones((1, 1))], dim=0)
    b = torch.cat([(1 - p) * torch.ones(int(nc), 1) / nc, p * torch.ones((1, 1))], dim=0)

    # Compute kernel matrix
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + 1e-6  # added constant to avoid NaN
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(its):
        u = 1.0 / (ainvK @ (b / (u.T @ K).T))
    v = b / (u.T @ K).T

    T = u * (v.T * K)

    if not backpropT:
        T = T.detach()

    E = T * Mt
    D = 2 * torch.sum(E)

    return D, Mlam

import torch
from torch.optim import Adam, AdamW

class ClippedAdam(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, clip_value=1.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.clip_value = clip_value

    def step(self, closure=None):
        # Clip gradients before updating parameters
        torch.nn.utils.clip_grad_value_(self.param_groups[0]['params'], self.clip_value)
        # Call the original Adam optimizer step
        return super().step(closure)

class ClippedAdamW(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, clip_value=1.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.clip_value = clip_value

    def step(self, closure=None):
        # Clip gradients before updating parameters
        torch.nn.utils.clip_grad_value_(self.param_groups[0]['params'], self.clip_value)
        # Call the original Adam optimizer step
        return super().step(closure)


def _as_1d_timegrid(time_points: torch.Tensor) -> torch.Tensor:
    """
    Accepts time_points shaped (T,) or (1,T) or (N,T) and returns (T,) on same device/dtype.
    Assumes all rows are identical when (N,T).
    """
    if time_points.dim() == 1:
        return time_points
    if time_points.dim() == 2:
        return time_points[0]
    raise ValueError(f"time_points must have dim 1 or 2, got {time_points.shape}")


def _interp_batch(grid_t: torch.Tensor, values: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation over a common 1D grid for batched values.
    grid_t: (T,)
    values: (N,T)
    query_t: scalar or (K,) or (N,) or (N,K)
    Returns:
      (N,) if query_t is scalar
      (N,K) if query_t is (K,) or (N,K)
      (N,) if query_t is (N,) (pointwise per patient)
    """
    device = values.device
    grid_t = grid_t.to(device)

    # Make query_t into shape (N,K)
    if not torch.is_tensor(query_t):
        query_t = torch.tensor(query_t, device=device, dtype=grid_t.dtype)

    query_t = query_t.to(device=device, dtype=grid_t.dtype)

    N, T = values.shape

    if query_t.dim() == 0:
        q = query_t.view(1, 1).expand(N, 1)
    elif query_t.dim() == 1:
        if query_t.shape[0] == N:
            q = query_t.view(N, 1)
        else:
            q = query_t.view(1, -1).expand(N, -1)
    elif query_t.dim() == 2:
        if query_t.shape[0] != N:
            raise ValueError(f"query_t with dim=2 must have shape (N,K). Got {query_t.shape}, N={N}")
        q = query_t
    else:
        raise ValueError(f"query_t must have dim 0/1/2, got {query_t.dim()}")

    # Clamp queries to grid bounds
    q = torch.clamp(q, grid_t[0], grid_t[-1])

    # Find right indices: idx in [1..T-1]
    idx = torch.bucketize(q, grid_t)  # returns in [0..T]
    idx = torch.clamp(idx, 1, T - 1)

    t0 = grid_t[idx - 1]      # (N,K)
    t1 = grid_t[idx]          # (N,K)

    v0 = values.gather(1, idx - 1)  # (N,K)
    v1 = values.gather(1, idx)      # (N,K)

    denom = (t1 - t0).clamp_min(torch.finfo(grid_t.dtype).eps)
    w = (q - t0) / denom
    out = v0 + w * (v1 - v0)

    # Squeeze back if scalar query
    if query_t.dim() == 0:
        return out[:, 0]
    if query_t.dim() == 1 and query_t.shape[0] == N:
        return out[:, 0]
    return out


def _rmst_from_survival(time_grid: torch.Tensor, survival: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """
    RMST_i(tau) = \int_0^tau S_i(u) du computed by trapezoidal rule on a shared grid.
    time_grid: (T,)
    survival: (N,T)
    tau: scalar or (N,) allowed
    Returns: (N,) RMST
    """
    device = survival.device
    time_grid = time_grid.to(device=device, dtype=survival.dtype)

    if not torch.is_tensor(tau):
        tau = torch.tensor(tau, device=device, dtype=time_grid.dtype)
    tau = tau.to(device=device, dtype=time_grid.dtype)

    # If tau varies per patient, compute per patient by interpolation and masking.
    # For speed and simplicity, support scalar tau as the typical case.
    if tau.dim() == 0:
        # Restrict grid to <= tau plus one interpolated point at tau if needed
        if tau <= time_grid[0]:
            # RMST is approx 0 if tau at/under grid start
            return torch.zeros(survival.shape[0], device=device, dtype=survival.dtype)

        # Indices where time <= tau
        mask = time_grid <= tau
        t_in = time_grid[mask]
        s_in = survival[:, mask]

        # Ensure tau is included as last point
        if t_in.numel() == 0:
            # Should not happen due to tau > time_grid[0], but safe
            s_tau = _interp_batch(time_grid, survival, tau)  # (N,)
            return tau * s_tau  # rough fallback
        if t_in[-1] < tau:
            s_tau = _interp_batch(time_grid, survival, tau).unsqueeze(1)  # (N,1)
            t_in = torch.cat([t_in, tau.view(1)], dim=0)                  # (T'+1,)
            s_in = torch.cat([s_in, s_tau], dim=1)                        # (N,T'+1)

        # Trapezoid: sum 0.5*(S_j+S_{j+1})*(t_{j+1}-t_j)
        dt = (t_in[1:] - t_in[:-1]).unsqueeze(0)         # (1,T'-1)
        s_avg = 0.5 * (s_in[:, 1:] + s_in[:, :-1])       # (N,T'-1)
        return torch.sum(s_avg * dt, dim=1)              # (N,)

    # tau is (N,) case (less common)
    if tau.dim() == 1 and tau.shape[0] == survival.shape[0]:
        # Compute RMST per patient by integrating on full grid, with last segment clipped at tau
        N, T = survival.shape
        # Compute survival at tau for each patient
        s_tau = _interp_batch(time_grid, survival, tau)  # (N,)
        # Build an "effective" time grid per patient by clipping each grid point at tau_i
        # This is heavier; use a simple loop if needed.
        rmst = torch.zeros(N, device=device, dtype=survival.dtype)
        for i in range(N):
            rmst[i] = _rmst_from_survival(time_grid, survival[i:i+1, :], tau[i]).squeeze(0)
        return rmst

    raise ValueError(f"tau must be scalar or shape (N,), got {tau.shape}")


def _quantile_from_survival(time_grid: torch.Tensor, survival: torch.Tensor, p: float) -> torch.Tensor:
    """
    Returns Q_i(p) where P(T <= Q_i(p)) = p, from a discrete survival curve S_i(t_j).

    time_grid: (T,) increasing
    survival:  (N,T) assumed non-increasing in t (up to numerical noise)
    p: float in (0,1)
    Returns: (N,) quantiles in the same units as time_grid
    """
    device = survival.device
    dtype = survival.dtype

    time_grid = time_grid.to(device=device, dtype=dtype).flatten()
    if time_grid.dim() != 1:
        raise ValueError("time_grid must be 1D (T,)")

    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    N, T = survival.shape
    eps = torch.finfo(dtype).eps

    # CDF is non-decreasing: F(t) = 1 - S(t)
    cdf = 1.0 - survival  # (N,T)

    # Find first index j where cdf[:, j] >= p.
    # mask is True starting at the crossing point.
    mask = cdf >= p  # (N,T) boolean

    # If never crosses (all False), argmax returns 0, so handle separately.
    has_cross = mask.any(dim=1)  # (N,)

    # First crossing index: for boolean mask, argmax gives first True because True=1, False=0.
    j = mask.float().argmax(dim=1)  # (N,)

    # For rows with no crossing, set j to last index (quantile beyond grid -> clamp to t_last).
    j = torch.where(has_cross, j, torch.full_like(j, T - 1))

    # If crossing at j=0, quantile is at/before first grid point.
    j0 = torch.clamp(j - 1, 0, T - 1)
    j1 = j

    t0 = time_grid[j0]  # (N,)
    t1 = time_grid[j1]  # (N,)

    c0 = cdf[torch.arange(N, device=device), j0]  # (N,)
    c1 = cdf[torch.arange(N, device=device), j1]  # (N,)

    # Linear interpolation between (t0, c0) and (t1, c1) to solve c(t)=p.
    # If j0 == j1, or c1 == c0, fall back to t1.
    denom = (c1 - c0).abs().clamp_min(eps)
    w = (p - c0) / denom
    w = torch.clamp(w, 0.0, 1.0)

    q = t0 + w * (t1 - t0)

    # Clamp to grid bounds (important when p is outside achieved range due to truncation)
    q = torch.clamp(q, time_grid[0], time_grid[-1])
    return q

def _to_tensor(x):
    if torch.is_tensor(x):
        return x
    return torch.tensor(x)

#     return {"individual": indiv, "population": pop}

def ensure_results_dir(path="./results"):
    os.makedirs(path, exist_ok=True)
    return path


def ite(y_pred_list, scaler=None, load_model='best', distribution_name='gaussian', treatment=None, limits=None, input_is_dist=True):

    if len(y_pred_list)!=2 and treatment is None:
        raise ValueError('ITES can only be computed for binary treatments')

    if treatment is None:
        treatment = 1

    distribution_name = distribution_name.lower() if distribution_name is not None else y_pred_list[0].__class__.__name__.lower()

    # to numpy
    if distribution_name == 'gumbel':
        # shape, scale = self.predict_params(X_test, load_model=load_model)
        location, scale = [y.loc for y in y_pred_list], [y.scale for y in y_pred_list]
        shape, scale = [1 / scale_i for scale_i in scale], [torch.exp(location_i) for location_i in location]
        y_pred_list = [torch.distributions.Weibull(scale=scale[i], concentration=shape[i]) for i in range(len(scale))]
        # y_pred_mean = [y_pred_list[i].mean.detach().cpu().numpy() for i in range(self.num_treatments)]
    # else:
    #     y_pred_mean = [y_pred.detach().cpu().numpy() for y_pred in y_pred_mean]
    if input_is_dist:
        y_pred_mean = [y_pred_list_i.mean.detach().cpu().numpy() for y_pred_list_i in y_pred_list]
    else:
        y_pred_mean = [y_pred.detach().cpu().numpy() for y_pred in y_pred_list]  # Note that this is a point estimate, not a distribution!

    if limits is not None:
        y_pred_mean = [np.clip(y_pred, limits[0], limits[1]) for y_pred in y_pred_mean]

    if distribution_name == 'log-normal' or distribution_name=='lognormal':
        y_pred_std = [y.sttdev.detach().cpu().numpy() for y in y_pred_list]
        # y_pred_std = [y_std.detach().cpu().numpy() for y_std in y_pred_std]

    if scaler is not None:
        y_pred_mean = [scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten() for y_pred in y_pred_mean]
        if distribution_name == 'log-normal':
            y_pred_std = [(y_std*scaler.scale_).flatten() for y_std in y_pred_std]

    if distribution_name == 'log-normal':
        ites = np.exp(y_pred_mean[treatment] + 0.5 * y_pred_std[1]**2) - np.exp(y_pred_mean[0] + 0.5*y_pred_std[0]**2)
    else:
        ites = y_pred_mean[treatment] - y_pred_mean[0]

    return ites

def ate(self, y_pred_list, scaler=None, treatment=None, limits=None):
    if self.num_treatments != 2 and treatment is None:
        raise NotImplementedError('ATEs only implemented for binary treatments')
    # ite = self.ite(X_test, scaler, treatment=treatment, limits=limits)
    ite = self.ite(y_pred_list, scaler, treatment=treatment, limits=limits)
    return ite.mean()

def get_survival_curves(y_dist_list,
                        distribution_name=None,
                        # X_test, load_model='best',
                        scaler=None,
                        time_points=None):
    """
    Get survival function from model parameters.
    Compute individual Weibull survival curves.

    Args:
        y_dist_list, a list with with len=num_treatments. Contains the distribution of the model predicted for each treatment.
                    [dist_t0(N,), dist_t1(N,)]
        time_points: Optional list or tensor of time points at which to compute the survival function. If None, a default range is used.

    Returns:
        Tensor of shape (N, T) with survival probabilities S_i(t_j) = exp(- (t_j / λ_i)^{k_i}).
    """
    if time_points is not None:
        time_points = _to_tensor(time_points)

    distribution_name = y_dist_list[0].__class__.__name__.upper() if distribution_name is None else distribution_name.upper()
    if distribution_name == 'WEIBULL' or distribution_name == 'GUMBEL':

        # shape, scale = self.predict_params(X_test, load_model=load_model)
        if distribution_name == 'GUMBEL':
            location = [y.loc for y in y_dist_list]
            scale = [y.scale for y in y_dist_list]

            shape = [1 / scale_i for scale_i in scale]  # Gumble shape parameter is the inverse of the scale
            scale = [torch.exp(location_i) for location_i in location]  # Gumble scale parameter must be positive
        else:
            shape = [y.concentration for y in y_dist_list]
            scale = [y.scale for y in y_dist_list]

        N = shape[0].shape[0]  # number of samples

        if time_points is None:
            max_mean = 10 * scale
            time_points = torch.linspace(0, 10 * max_mean, 1000).reshape(1, -1)  # (1, T)

        T = time_points.shape[1]
        # expand vectors
        shape = [torch.tensor(shapei).expand(N, T) for shapei in shape]
        scale = [torch.tensor(scalei).expand(N, T) for scalei in scale]
        time_points = time_points.expand(N, T)

        # return torch.exp(-(time_points / scale) ** shape)  # (N,T)
        return [torch.exp(-(time_points / scale[i]) ** shape[i]) for i in range(len(scale))]

    elif (distribution_name == 'NORMAL' or distribution_name == 'LOGNORMAL'
          or distribution_name == 'LOG-NORMAL' or distribution_name == 'GAUSSIAN'):
        # mu, sigma = self.predict_params(X_test, load_model=load_model)
        mu= [y.loc for y in y_dist_list]
        sigma= [y.scale for y in y_dist_list]

        if scaler is not None:
            scaler_mean = torch.tensor(scaler.mean_)
            scaler_std = torch.tensor(scaler.scale_)

            mu = [(scaler_mean + mui * scaler_std)for mui in mu]
            sigma = [(sigmai * scaler_std) for sigmai in sigma]

        N = mu[0].shape[0]  # number of samples

        if time_points is None:
            max_mean = torch.exp(mu[0] + 3 * sigma[0])
            time_points = torch.linspace(0.01, max_mean.max(), 1000).reshape(1, -1)  # (1, T)

        if time_points.dim() == 1:
            time_points = time_points.reshape(1, -1)

        T = time_points.shape[1]
        # expand vectors
        mu = [mui.clone().detach().expand(N, T) for mui in mu]
        sigma = [sigmai.clone().detach().expand(N, T) for sigmai in sigma]
        time_points = time_points.expand(N, T)


        # survival function for log-normal: S(t) = 1 - Φ((ln(t) - mu) / sigma)
        standard_normal = Normal(0, 1)
        survival_curves = []
        for i in range(len(mu)):
            if distribution_name == 'GAUSSIAN' or distribution_name == 'NORMAL':
                # for gaussian, we assume that the time to event is modeled as exp(X), so we compute survival for exp(X)
                z = (time_points - mu[i]) / sigma[i]
            else:
                z = (torch.log(time_points) - mu[i]) / sigma[i]
            survival_curve = 1 - standard_normal.cdf(z)
            survival_curves.append(survival_curve)


        return survival_curves, time_points
    else:
        raise NotImplementedError('Survival curves only implemented for Weibull (including exponential with shape=1) and Log-Normal outcomes')

def get_hazard_curves(survival_curves, time_points):
    """
    Get hazard function from survival curves.
    Compute individual hazard curves.
    Args:
        survival_curves: list with len=num_treatments. Each element is a tensor of shape (N, T) with survival probabilities S_i(t_j).
        time_points: Tensor of shape (T,) with time points corresponding to the survival curves.
    Returns:
        List with len=num_treatments. Each element is a tensor of shape (N, T) with hazard probabilities h_i(t_j).
    """
    grid = _as_1d_timegrid(time_points)
    hazard_curves = []
    for S in survival_curves: #(N,T)
        # compute hazard h(t) = -d/dt log S(t) = f(t) / S(t)
        # approximate derivative using finite differences
        dt = grid[1:] - grid[:-1]  # (T-1,)
        dS = S[:, 1:] - S[:, :-1]  # (N, T-1)
        f = -dS / dt  # (N, T-1)
        S_mid = (S[:, 1:] + S[:, :-1]) / 2  # (N, T-1)
        h = f / S_mid  # (N, T-1)
        # pad the first column with zeros
        h = torch.cat([torch.zeros((S.shape[0], 1)), h], dim=1)  # (N, T)
        hazard_curves.append(h)
    return hazard_curves

def get_cumulative_hazard_curves(survival_curves, time_points):
    """
    Get cumulative hazard function from survival curves.
    Compute individual cumulative hazard curves.
    Args:
        survival_curves: list with len=num_treatments. Each element is a tensor of shape (N, T) with survival probabilities S_i(t_j).
        time_points: Tensor of shape (T,) with time points corresponding to the survival curves.
    Returns:
        List with len=num_treatments. Each element is a tensor of shape (N, T) with cumulative hazard probabilities H_i(t_j).
    """
    grid = _as_1d_timegrid(time_points)
    cumulative_hazard_curves = []
    for S in survival_curves:
        H = -torch.log(S + 1e-10)  # (N, T)
        cumulative_hazard_curves.append(H)
    return cumulative_hazard_curves


def delta_s(survival_curves, time_points,
            # X_test,
            tau,
            # load_model="best",time_points=None,
            # scaler=None, return_individual=True
            ):
    """
    Delta survival at time t: ΔS_i(t) = S_i(t | A=1) - S_i(t | A=0)

    Returns:
      if return_individual: (delta_individual, time_points_used)
        delta_individual: (N,) if t is scalar or (N,K) if t is a list/tensor of K times
      else: (delta_mean, time_points_used)
        delta_mean: scalar if t scalar else (K,)
    """
    grid = _as_1d_timegrid(time_points)

    S0_t = _interp_batch(grid, survival_curves[0], tau)
    S1_t = _interp_batch(grid, survival_curves[1], tau)
    delta = S1_t - S0_t

    return delta

def delta_rmst(survival_curves, time_points,
               tau,
               scaler=None):
    """
    Delta RMST up to tau: ΔRMST_i(τ) = RMST_i(τ|A=1) - RMST_i(τ|A=0)
    RMST_i(τ) = ∫_0^τ S_i(u) du

    Returns:
      if return_individual: (delta_individual, time_points_used)
        delta_individual: (N,)
      else: (delta_mean, time_points_used)
        delta_mean: scalar
    """
    # survival_curves, tp = self.get_survival_curves(
    #     X_test, load_model=load_model, time_points=time_points, scaler=scaler
    # )
    # grid = _as_1d_timegrid(tp)
    grid = _as_1d_timegrid(time_points)

    rmst0 = _rmst_from_survival(grid, survival_curves[0], tau)  # (N,)
    rmst1 = _rmst_from_survival(grid, survival_curves[1], tau)  # (N,)
    delta = rmst1 - rmst0

    return delta


def delta_q(survival_curves, time_points,
            p,
            scaler=None):
    """
    Delta quantile at probability p: ΔQ_i(p) = Q_i(p|A=1) - Q_i(p|A=0)
    Quantile defined by P(T <= Q(p)) = p.

    Returns:
      if return_individual: (delta_individual, time_points_used)
        delta_individual: (N,)
      else: (delta_mean, time_points_used)
        delta_mean: scalar
    """
    grid = _as_1d_timegrid(time_points)

    q0 = _quantile_from_survival(grid, survival_curves[0], p)  # (N,)
    q1 = _quantile_from_survival(grid, survival_curves[1], p)  # (N,)
    delta = q1 - q0

    return delta

def odds_ratio(survival_curves, time_points,
               tau=None, scaler=None):
    """
    Get odds ratio at time tau from model parameters.
    Compute individual Weibull survival curves and odds ratios.

    Args:
        X_test: Input features for the test set.
        load_model: Model to load, default is 'best'.
        tau: Time point at which to evaluate the odds ratio.
    Returns:
        odds_ratios: Tensor of shape (N,) with odds ratios OR_i(tau)
    """

    grid = _as_1d_timegrid(time_points)

    if tau is not None:
        S0_t = _interp_batch(grid, survival_curves[0], tau)
        S1_t = _interp_batch(grid, survival_curves[1], tau)
    else:
        S0_t = survival_curves[0]
        S1_t = survival_curves[1]

    # odds ratio OR_i(t) = (S_i(t|A=1) / (1 - S_i(t|A=1))) / (S_i(t|A=0) / (1 - S_i(t|A=0)))
    odds_ratios = (S1_t / (1 - S1_t)) / (S0_t / (1 - S0_t))

    return odds_ratios

def risk_difference(survival_curves, time_points,
                    # X_test,
                    tau = None,
                    # load_model='best',
                    scaler=None):
    """
    Get risk difference at time tau from model parameters.
    Compute individual Weibull survival curves and risk differences

    Note that risk difference is defined here as RD_i(t) = S_i(t|A=0) - S_i(t|A=1) = 1-delta_S_i(t)

    Args:
        X_test: Input features for the test set.
        load_model: Model to load, default is 'best'.
        tau: Time point at which to evaluate the risk difference.
    Returns:
        risk_differences: Tensor of shape (N,) with risk differences RD_i(tau)
    """

    grid = _as_1d_timegrid(time_points)

    if tau is not None:
        S0_t = _interp_batch(grid, survival_curves[0], tau)
        S1_t = _interp_batch(grid, survival_curves[1], tau)
    else:
        S0_t = survival_curves[0]
        S1_t = survival_curves[1]

    # risk difference RD_i(t) = [1-S_i(t|A=1)] - [1-S_i(t|A=0)] =  S_i(t|A=0) - S_i(t|A=1)
    risk_differences = S0_t - S1_t

    return risk_differences, time_points

def hazard_ratio(curve, time_points, tau=None, type_input='hazard'):
    """
    Get hazard ratios at time tau from model parameters.
    Args:
        curve: list with len=num_treatments. Each element is a tensor of shape (N, T) with hazard probabilities h_i(t_j).
        time_points: Tensor of shape (T,) with time points corresponding to the hazard curves.
        type_input: 'hazard' or 'survival', whether to compute hazard ratio from hazard or survival curves.
    Returns:
        hazard_ratios: Tensor of shape (N,) with hazard ratios HR_i(tau)
    """

    grid = _as_1d_timegrid(time_points)

    if type_input == 'hazard':
        h0_t = curve[0]
        h1_t = curve[1]
    elif type_input == 'survival':
        S0_t = curve[0]
        S1_t = curve[1]
        delta_time = grid[1] - grid[0]
        h0_t = - (S0_t[:,1:] - S0_t[:,:-1]) / (delta_time * S0_t[:, :-1] + 1e-10)
        h0_t = torch.cat([h0_t[:,0:1], h0_t], dim=1)  # pad first column
        h1_t = - (S1_t[:,1:] - S1_t[:,:-1]) / (delta_time * S1_t[:, :-1] + 1e-10)
        h1_t = torch.cat([h1_t[:,0:1], h1_t], dim=1)  # pad first column
    else:
        raise ValueError("type_input must be 'hazard' or 'survival'")

    if tau is not None:
        h0_t = _interp_batch(grid, h0_t, tau)
        h1_t = _interp_batch(grid, h1_t, tau)


    # hazard ratio HR_i(t) = h_i(t|A=1) / h_i(t|A=0)
    hazard_ratios = h1_t / (h0_t + 1e-10)

    return hazard_ratios


import os
import pickle
import torch


def _load_realisation_pkl(results_dir, model_name, seed, over_set="test"):
    if over_set != "train":
        pkl_path = os.path.join(results_dir, f"realisation_{seed}_{model_name}.pkl")
    else:
        pkl_path = os.path.join(results_dir, f"realisation_{seed}_{model_name}_train.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)  # list[K] of dicts


def _ensure_timegrid(timepoints):
    if torch.is_tensor(timepoints):
        return timepoints.float()
    return torch.tensor(timepoints, dtype=torch.float32)


def _ipcw_weights(p_cens, eps=1e-6, normalized=True):
    # p_cens is "probability of censoring" per individual
    weights = 1.0 / (1.0 - p_cens + eps)
    if normalized:
        weights = weights / weights.mean()
    return weights


def summarize_draws(draws, alpha=0.05, dim=0, stats=("mean", "median", "std", "q_low", "q_high")):
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    out = {}
    if "mean" in stats:
        out["mean"] = draws.mean(dim=dim)
    if "median" in stats:
        out["median"] = draws.median(dim=dim).values
    if "std" in stats:
        out["std"] = draws.std(dim=dim, unbiased=True)
    if "q_low" in stats or "q_high" in stats:
        qs = torch.quantile(draws, torch.tensor([q_low, q_high], device=draws.device, dtype=draws.dtype), dim=dim)
        out["q_low"] = qs[0]
        out["q_high"] = qs[1]
    return out


def _reduce_k(draws, k_mode="mean"):
    if k_mode == "mean":
        return draws.mean(dim=1)  # (M,...)
    if k_mode == "all":
        M, K = draws.shape[0], draws.shape[1]
        return draws.reshape(M * K, *draws.shape[2:])
    raise ValueError("k_mode must be in {'mean','all'}")


def _compute_metrics_from_draw(d, timepoints, tau, q_ps=(0.25, 0.5, 0.75), outcome_dist=None, scaler=None, only_survival=False):
    """
    d has keys:
      propensity_score_pred: (N, ...)
      probability_censoring_pred: (N,)
      y_dist_list: [dist0(N,), dist1(N,)]
    returns dict[str, tensor] with per-individual tensors, and curves as (N,T).
    """
    y_dist_list = d["y_dist_list"]
    survival_out = get_survival_curves(y_dist_list, time_points=timepoints, distribution_name=outcome_dist, scaler=scaler)
    if isinstance(survival_out, tuple):
        survival_curves, tp_used = survival_out
    else:
        survival_curves, tp_used = survival_out, timepoints

    hazard_curves = get_hazard_curves(survival_curves, tp_used)
    cumhaz_curves = get_cumulative_hazard_curves(survival_curves, tp_used)

    out = {}
    out["propensity_score_pred"] = d["propensity_score_pred"]
    out["probability_censoring_pred"] = d["probability_censoring_pred"]

    out["survival_t0"] = survival_curves[0]
    out["survival_t1"] = survival_curves[1]

    t_true = torch.tensor(d["t_true"], dtype=torch.float32).view(-1,1)

    survival_natural_course = torch.where(t_true == 1, survival_curves[1], survival_curves[0])
    out["survival_natural_course"] = survival_natural_course

    if not only_survival:
        out["hazard_t0"] = hazard_curves[0]
        out["hazard_t1"] = hazard_curves[1]

        out['hazard_natural_course'] = torch.where(t_true == 1, hazard_curves[1], hazard_curves[0])


        out["cumhaz_t0"] = cumhaz_curves[0]
        out["cumhaz_t1"] = cumhaz_curves[1]

        out['cumhaz_natural_course'] = torch.where(t_true == 1, cumhaz_curves[1], cumhaz_curves[0])

        out["delta_s_tau"] = delta_s(survival_curves, tp_used, tau=tau)
        out["delta_rmst_tau"] = delta_rmst(survival_curves, tp_used, tau=tau)
        out["odds_ratio_tau"] = odds_ratio(survival_curves, tp_used, tau=tau)
        rd, _ = risk_difference(survival_curves, tp_used, tau=tau)
        out["risk_difference_tau"] = rd
        out["hazard_ratio_tau"] = hazard_ratio(hazard_curves, tp_used, tau=tau, type_input="hazard")

        for p in q_ps:
            out[f"delta_q_p{str(p).replace('.', '_')}"] = delta_q(survival_curves, tp_used, p=p)

    out["_timepoints_used"] = tp_used
    return out


def _stack_metric(reals_metrics, key):
    # reals_metrics: list[M] of list[K] of dict metrics
    return torch.stack([torch.stack([mk[key] for mk in reals_metrics[m]], dim=0) for m in range(len(reals_metrics))], dim=0)


def _summarize_population_from_draws(x_draws, p_cens_draws=None, alpha=0.05, eps=1e-6):
    """
    x_draws: (M*K,N,...) per-individual values
    p_cens_draws: (M*K,N) censoring probabilities for IPCW
    returns summarize_draws over draws of population-aggregated values, i.e, over M*K*N
    """
    if p_cens_draws is None:
        w_draws = torch.ones_like(x_draws[:, :, 0])  if x_draws.dim()==3 else torch.ones_like(x_draws)# (D,N)
    else:
        w_draws = torch.stack([_ipcw_weights(p_cens_draws[d], eps=eps) for d in range(p_cens_draws.shape[0])], dim=0)  # (D,N)

    # For quantiles/median/std at population-level with IPCW, do per-draw weighted summaries then summarize across draws.
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0

    # reshape x_draws into (D*N, ...) and w_draws into (D*N,)
    D, N = x_draws.shape[0], x_draws.shape[1]

    # first, mean over N with weights per draw
    xw_draws = (x_draws * w_draws.view(D, N, *[1]*(x_draws.ndim - 2))).sum(dim=1) / w_draws.sum(dim=1).clamp_min(eps).view(D, *[1]*(x_draws.ndim - 2))  # (D,...)

    # statistics over D,N
    pop_mean = xw_draws.mean(dim=0)
    pop_median = torch.quantile(xw_draws, 0.5, dim=0)
    pop_std = torch.std(xw_draws, dim=0, unbiased=True)
    pop_q_low = torch.quantile(xw_draws, q_low, dim=0)
    pop_q_high = torch.quantile(xw_draws, q_high, dim=0)


    return {
        "mean": pop_mean,
        "median": pop_median,
        "std": pop_std,
        "q_low": pop_q_low,
        "q_high": pop_q_high,
    }


def aggregate_all(
    results_dir,
    model_name,
    M,
    K,
    timepoints,
    tau,
    scaler=None,
    outcome_dist=None,
    starting_seed=0,
    k_mode="mean",
    alpha=0.05,
    over_set="test",
    q_ps=(0.25, 0.5, 0.75),
    per_individual=False,
    population_summary=True,
    use_ipcw=False,
    eps=1e-6,
    only_survival=False,
):
    """
    Returns a dict with:
      - "individual": summaries over draws for each individual (if per_individual)
      - "population": summaries over draws after aggregating over N (if population_summary)
      - "timepoints": time grid tensor
    """
    if timepoints is not None:
        tp = _ensure_timegrid(timepoints)
    else:
        tp = timepoints

    # Load and compute metrics for each (m,k)
    reals_metrics = []
    for m in range(M):
        real = _load_realisation_pkl(results_dir, model_name, starting_seed + m, over_set=over_set)  # list[K]
        if len(real) > K:
            raise ValueError(f"Expected K={K} draws in each realisation, got {len(real)}.")
        if K < len(real):
            real = real[:K]
        reals_metrics.append([_compute_metrics_from_draw(real[k], tp, tau, q_ps=q_ps, outcome_dist=outcome_dist, scaler=scaler, only_survival=only_survival) for k in range(K)])

    # Keys to aggregate (exclude helper)
    keys = [k for k in reals_metrics[0][0].keys() if k != "_timepoints_used"]

    # Build tensors (M,K,...) then reduce K -> draws tensor (D,...)
    tensors = {}
    for key in keys:
        mk = _stack_metric(reals_metrics, key)          # (M,K,...)
        tensors[key] = _reduce_k(mk, k_mode=k_mode)     # (D,...), D=M or M*K

    D = tensors[keys[0]].shape[0]
    p_cens_draws = tensors["probability_censoring_pred"]
    if p_cens_draws.ndim != 2:
        # enforce (D,N) for weights; if your p_cens has extra dims, adapt here
        p_cens_draws = p_cens_draws.reshape(D, -1)

    out = {"timepoints": tp}

    if per_individual:
        indiv = {}
        for key in keys:
            indiv[key] = summarize_draws(tensors[key], alpha=alpha, dim=0)
        out["individual"] = indiv

    if population_summary:
        pop = {}
        for key in keys:
            if key == "propensity_score_pred" or key == "probability_censoring_pred":
                # these are not per-individual quantities to aggregate over N
                continue
            x = tensors[key]
            # Only apply IPCW when x is per-individual, that is draws shape (D,N,...) or (D,N)
            apply_w = use_ipcw and (x.ndim >= 2) and (x.shape[1] == p_cens_draws.shape[1])
            pop[key] = _summarize_population_from_draws(
                x_draws=x,
                p_cens_draws=p_cens_draws if apply_w else None,
                alpha=alpha,
                eps=eps,
            )
        out["population"] = pop

    return out

