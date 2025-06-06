import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from typing import Optional, Dict, List, Tuple, Union
import warnings
from tqdm import tqdm
import multiprocessing as mp

from .core import MDSVProcess
from .models import MDSVFit, MDSVFilter, MDSVBoot, MDSVRoll, MDSVSim

try:
    from . import mdsv_cpp
except ImportError:
    warnings.warn("C++ extension not available. Falling back to Python implementation.")
    mdsv_cpp = None


def fit_mdsv(data: Union[np.ndarray, pd.DataFrame],
             N: int = 2,
             K: int = 3,
             model_type: int = 0,
             leverage: bool = False,
             start_params: Optional[np.ndarray] = None,
             fixed_params: Optional[Dict[int, float]] = None,
             optimizer_options: Optional[Dict] = None,
             use_cpp: bool = True,
             verbose: bool = True) -> MDSVFit:
    """
    Fit MDSV model to data

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Univariate or bivariate data. If bivariate, first column is returns, second is RV
    N : int
        Number of components
    K : int
        Number of states per component
    model_type : int
        0: univariate returns, 1: univariate RV, 2: joint
    leverage : bool
        Whether to include leverage effect
    start_params : np.ndarray, optional
        Starting parameters
    fixed_params : dict, optional
        Dictionary of parameter indices and values to fix
    optimizer_options : dict, optional
        Options for optimizer
    use_cpp : bool
        Whether to use C++ implementation
    verbose : bool
        Whether to print progress

    Returns
    -------
    MDSVFit
        Fitted model results
    """
    # Convert data to numpy array
    if isinstance(data, pd.DataFrame):
        dates = data.index if isinstance(data.index, pd.DatetimeIndex) else None
        data_array = data.values
    else:
        dates = None
        data_array = np.asarray(data)

    # Validate data dimensions
    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)

    n_obs, n_cols = data_array.shape

    # Validate model type and data consistency
    if model_type == 0 and n_cols != 1:
        raise ValueError("Univariate returns model requires 1 column of data")
    elif model_type == 1:
        if n_cols == 1:
            # Add dummy returns column for RV-only model
            data_array = np.column_stack([np.ones(n_obs), data_array])
        elif n_cols != 2:
            raise ValueError("Univariate RV model requires 1 or 2 columns")
    elif model_type == 2 and n_cols != 2:
        raise ValueError("Joint model requires 2 columns of data")

    # Model type names
    model_names = {
        0: "Univariate log-return",
        1: "Univariate realized variances",
        2: "Joint log-return and realized variances"
    }

    # Get starting parameters
    if start_params is None:
        start_params = _get_default_start_params(data_array, model_type, leverage)

    # Define objective function
    def objective(params_tilde):
        if use_cpp and mdsv_cpp is not None:
            # Use C++ implementation
            if fixed_params:
                fixed_pars = list(fixed_params.keys())
                fixed_vals = list(fixed_params.values())
            else:
                fixed_pars = None
                fixed_vals = None

            return mdsv_cpp.logLik(params_tilde, data_array, model_type,
                                   leverage, K, N, 70, fixed_pars, fixed_vals, "lognormal")
        else:
            # Python implementation
            return _log_likelihood_py(params_tilde, data_array, model_type,
                                      leverage, K, N, fixed_params)

    # Transform parameters to unconstrained space
    if use_cpp and mdsv_cpp is not None:
        params_tilde = mdsv_cpp.natWork(start_params, leverage, model_type)
    else:
        params_tilde = _nat_to_work_py(start_params, leverage, model_type)

    # Optimization
    if optimizer_options is None:
        optimizer_options = {'method': 'L-BFGS-B', 'options': {'disp': verbose}}

    if verbose:
        print(f"Fitting MDSV({N},{K}) model...")

    result = minimize(objective, params_tilde, **optimizer_options)

    # Transform back to natural parameters
    if use_cpp and mdsv_cpp is not None:
        final_params = mdsv_cpp.workNat(result.x, leverage, model_type,
                                        fixed_params.keys() if fixed_params else None,
                                        fixed_params.values() if fixed_params else None)
    else:
        final_params = _work_to_nat_py(result.x, leverage, model_type, fixed_params)

    # Check convergence
    convergence = 0 if result.success else 1

    # Create result object
    return MDSVFit(
        model_type=model_names[model_type],
        leverage=leverage,
        N=N,
        K=K,
        data=data_array,
        estimates=final_params,
        log_likelihood=-result.fun,
        convergence=convergence,
        dates=dates
    )


def filter_mdsv(data: Union[np.ndarray, pd.DataFrame],
                params: np.ndarray,
                N: int = 2,
                K: int = 3,
                model_type: int = 0,
                leverage: bool = False,
                calculate_var: bool = True,
                var_alpha: List[float] = None,
                use_cpp: bool = True) -> MDSVFilter:
    """
    Filter MDSV model with given parameters

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Data to filter
    params : np.ndarray
        Model parameters
    N : int
        Number of components
    K : int
        Number of states per component
    model_type : int
        0: univariate returns, 1: univariate RV, 2: joint
    leverage : bool
        Whether model includes leverage
    calculate_var : bool
        Whether to calculate VaR
    var_alpha : list of float
        VaR significance levels
    use_cpp : bool
        Whether to use C++ implementation

    Returns
    -------
    MDSVFilter
        Filtering results
    """
    if var_alpha is None:
        var_alpha = [0.01, 0.05]

    # Convert data
    if isinstance(data, pd.DataFrame):
        dates = data.index if isinstance(data.index, pd.DatetimeIndex) else None
        data_array = data.values
    else:
        dates = None
        data_array = np.asarray(data)

    # Ensure proper dimensions
    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)

    # Model type names
    model_names = {
        0: "Univariate log-return",
        1: "Univariate realized variances",
        2: "Joint log-return and realized variances"
    }

    # Run filtering
    if use_cpp and mdsv_cpp is not None:
        results = mdsv_cpp.logLik2(data_array, params, model_type, leverage,
                                   K, N, 0, 2, 70, "lognormal")
    else:
        results = _filter_py(data_array, params, model_type, leverage, K, N)

    # Calculate VaR if requested
    var_values = {}
    if calculate_var and model_type != 1:
        process = MDSVProcess(N, K)
        process.update_from_vector(params, leverage, model_type)
        vol_vector = process.volatility_vector()
        pi_0 = results['w_hat']

        if leverage:
            params[5 + j] = np.exp(params_tilde[5 + j])  # l
        params[6 + j] = 1 / (1 + np.exp(params_tilde[6 + j]))  # theta

    # Apply fixed parameters
    if fixed_params:
        for idx, value in fixed_params.items():
            params[idx - 1] = value  # R uses 1-based indexing

    return params


def _simulate_markov_chain(pi_0: np.ndarray, P: np.ndarray,
                           n_steps: int, n_chains: int) -> np.ndarray:
    """Simulate Markov chain paths"""
    n_states = len(pi_0)
    chains = np.zeros((n_chains, n_steps), dtype=int)

    # Initial states
    chains[:, 0] = np.random.choice(n_states, size=n_chains, p=pi_0)

    # Simulate transitions
    for t in range(1, n_steps):
        for i in range(n_chains):
            current_state = chains[i, t - 1]
            chains[i, t] = np.random.choice(n_states, p=P[current_state, :])

    return chains


def _calculate_var(vol_vector: np.ndarray, weights: np.ndarray,
                   alpha: float) -> float:
    """Calculate Value-at-Risk"""
    # Create mixture distribution
    # For normal mixture, use weighted quantile approximation

    # Sort volatilities and weights
    idx = np.argsort(vol_vector)
    sorted_vols = vol_vector[idx]
    sorted_weights = weights[idx]

    # Find quantile by cumulative probability
    cum_prob = np.cumsum(sorted_weights)
    idx_var = np.searchsorted(cum_prob, alpha)

    if idx_var == 0:
        sigma_var = np.sqrt(sorted_vols[0])
    elif idx_var >= len(sorted_vols):
        sigma_var = np.sqrt(sorted_vols[-1])
    else:
        # Linear interpolation
        p1 = cum_prob[idx_var - 1]
        p2 = cum_prob[idx_var]
        w = (alpha - p1) / (p2 - p1)
        sigma_var = np.sqrt((1 - w) * sorted_vols[idx_var - 1] + w * sorted_vols[idx_var])

    # Return normal quantile
    return stats.norm.ppf(alpha) * sigma_var


def _log_likelihood_py(params_tilde: np.ndarray, data: np.ndarray,
                       model_type: int, leverage: bool, K: int, N: int,
                       fixed_params: Optional[Dict] = None) -> float:
    """Python implementation of log-likelihood calculation"""
    # This is a simplified version - full implementation would be complex
    # In practice, you'd want to use the C++ version

    params = _work_to_nat_py(params_tilde, leverage, model_type, fixed_params)

    # Create process
    process = MDSVProcess(N, K)
    process.update_from_vector(params[:5])

    # Get volatility vector and transition matrix
    vol_vector = process.volatility_vector()
    trans_matrix = process.full_transition_matrix()
    pi_0 = process.stationary_distribution()

    # Initialize log-likelihood
    log_lik = 0.0

    # Current state probabilities
    alpha = pi_0.copy()

    # Forward algorithm
    for t in range(len(data)):
        # Observation likelihood
        if model_type == 0:
            # Returns only
            obs_lik = stats.norm.pdf(data[t, 0], 0, np.sqrt(vol_vector))
        elif model_type == 1:
            # RV only
            shape = params[5]
            obs_lik = stats.gamma.pdf(data[t, 1] / vol_vector, shape, scale=1 / shape) / vol_vector
        else:
            # Joint model - simplified
            obs_lik = stats.norm.pdf(data[t, 0], 0, np.sqrt(vol_vector))
            # Would need to add RV likelihood component

        # Update likelihood
        alpha_new = obs_lik * alpha
        log_lik += np.log(np.sum(alpha_new))

        # Normalize and transition
        alpha = alpha_new / np.sum(alpha_new)
        alpha = trans_matrix.T @ alpha

    return -log_lik  # Return negative for minimization


def _filter_py(data: np.ndarray, params: np.ndarray, model_type: int,
               leverage: bool, K: int, N: int) -> Dict:
    """Python implementation of filtering"""
    # Simplified implementation
    n_obs = len(data)
    n_states = K ** N

    # Create process
    process = MDSVProcess(N, K)
    process.update_from_vector(params[:5])

    # Initialize
    filtered_proba = np.zeros((n_states, n_obs))
    smoothed_proba = np.zeros((n_states, n_obs))

    # Forward filtering
    vol_vector = process.volatility_vector()
    trans_matrix = process.full_transition_matrix()
    alpha = process.stationary_distribution()

    log_lik = 0.0

    for t in range(n_obs):
        # Observation likelihood
        if model_type == 0:
            obs_lik = stats.norm.pdf(data[t, 0], 0, np.sqrt(vol_vector))
        elif model_type == 1:
            shape = params[5]
            obs_lik = stats.gamma.pdf(data[t, 1] / vol_vector, shape, scale=1 / shape) / vol_vector
        else:
            # Joint - simplified
            obs_lik = stats.norm.pdf(data[t, 0], 0, np.sqrt(vol_vector))

        # Update
        alpha_new = obs_lik * alpha
        normalizer = np.sum(alpha_new)
        log_lik += np.log(normalizer)

        alpha = alpha_new / normalizer
        filtered_proba[:, t] = alpha

        # Transition
        if t < n_obs - 1:
            alpha = trans_matrix.T @ alpha

    # Backward smoothing (simplified - just use filtered for now)
    smoothed_proba = filtered_proba

    # Leverage calculation
    levier = np.ones(n_obs)
    if leverage:
        # Simplified leverage calculation
        if model_type == 0:
            l1, theta = params[5:7]
        elif model_type == 2:
            l1, theta = params[10:12]

        for t in range(1, min(70, n_obs)):
            if data[t - 1, 0] < 0:
                # Simplified - would need full history
                levier[t] = 1 + l1 * np.abs(data[t - 1, 0])

    return {
        'loglik': -log_lik,
        'filtred_proba': filtered_proba,
        'smoothed_proba': smoothed_proba,
        'Levier': levier,
        'w_hat': filtered_proba[:, -1],
        'Marg_loglik': -log_lik if model_type == 2 else None
    }


def calculate_loss_functions(roll_result: MDSVRoll, horizons: List[int],
                             window: int) -> Dict[str, pd.DataFrame]:
    """Calculate loss functions for rolling forecasts"""
    # Extract relevant data
    estimates = roll_result.estimates.iloc[-window:]
    prevision = roll_result.prevision.iloc[-window:]

    results = {}

    # For returns
    if roll_result.model_type != "Univariate realized variances":
        # Calculate losses for cumulative forecasts
        cum_results = []
        marg_results = []

        for h in horizons:
            if h > roll_result.n_ahead:
                continue

            # Cumulative forecast columns
            forecast_cols = [f'rt2p{i}' for i in range(1, h + 1)]
            if all(col in prevision.columns for col in forecast_cols):
                # Get forecasts and actuals
                forecasts = prevision[forecast_cols].sum(axis=1).values[:-h]
                actuals = []
                for i in range(len(estimates) - h):
                    actual = sum(estimates.iloc[i:i + h]['rt'].values ** 2)
                    actuals.append(actual)
                actuals = np.array(actuals)

                # Calculate metrics
                valid_mask = ~(np.isnan(forecasts) | np.isnan(actuals))
                if np.sum(valid_mask) > 0:
                    f = forecasts[valid_mask]
                    a = actuals[valid_mask]

                    qlik = np.mean(np.log(f) + a / f)
                    rmse = np.sqrt(np.mean((f - a) ** 2)) / h
                    mae = np.mean(np.abs(f - a)) / h

                    cum_results.append({
                        'horizon': h,
                        'QLIK': qlik,
                        'RMSE': rmse,
                        'MAE': mae
                    })

            # Marginal forecast
            if f'rt2p{h}' in prevision.columns:
                forecasts = prevision[f'rt2p{h}'].values[:-h]
                actuals = estimates.iloc[h:]['rt'].values ** 2

                valid_mask = ~(np.isnan(forecasts) | np.isnan(actuals))
                if np.sum(valid_mask) > 0:
                    f = forecasts[valid_mask]
                    a = actuals[valid_mask]

                    qlik = np.mean(np.log(f) + a / f)
                    rmse = np.sqrt(np.mean((f - a) ** 2))
                    mae = np.mean(np.abs(f - a))

                    marg_results.append({
                        'horizon': h,
                        'QLIK': qlik,
                        'RMSE': rmse,
                        'MAE': mae
                    })

        if cum_results:
            results['Cumulative Loss Functions:\nLog-returns:'] = pd.DataFrame(cum_results).set_index('horizon').T
        if marg_results:
            results['Marginal Loss Functions:\nLog-returns:'] = pd.DataFrame(marg_results).set_index('horizon').T

    # For RV
    if roll_result.model_type != "Univariate log-return":
        # Similar calculations for RV
        cum_results = []
        marg_results = []

        for h in horizons:
            if h > roll_result.n_ahead:
                continue

            # Cumulative forecast columns
            forecast_cols = [f'rvtp{i}' for i in range(1, h + 1)]
            if all(col in prevision.columns for col in forecast_cols):
                # Get forecasts and actuals
                forecasts = prevision[forecast_cols].sum(axis=1).values[:-h]
                actuals = []
                for i in range(len(estimates) - h):
                    actual = sum(estimates.iloc[i:i + h]['rvt'].values)
                    actuals.append(actual)
                actuals = np.array(actuals)

                # Calculate metrics
                valid_mask = ~(np.isnan(forecasts) | np.isnan(actuals))
                if np.sum(valid_mask) > 0:
                    f = forecasts[valid_mask]
                    a = actuals[valid_mask]

                    qlik = np.mean(np.log(f) + a / f)
                    rmse = np.sqrt(np.mean((f - a) ** 2)) / h
                    mae = np.mean(np.abs(f - a)) / h

                    cum_results.append({
                        'horizon': h,
                        'QLIK': qlik,
                        'RMSE': rmse,
                        'MAE': mae
                    })

            # Marginal forecast
            if f'rvtp{h}' in prevision.columns:
                forecasts = prevision[f'rvtp{h}'].values[:-h]
                actuals = estimates.iloc[h:]['rvt'].values

                valid_mask = ~(np.isnan(forecasts) | np.isnan(actuals))
                if np.sum(valid_mask) > 0:
                    f = forecasts[valid_mask]
                    a = actuals[valid_mask]

                    qlik = np.mean(np.log(f) + a / f)
                    rmse = np.sqrt(np.mean((f - a) ** 2))
                    mae = np.mean(np.abs(f - a))

                    marg_results.append({
                        'horizon': h,
                        'QLIK': qlik,
                        'RMSE': rmse,
                        'MAE': mae
                    })

        if cum_results:
            results['Cumulative Loss Functions:\nRealized Variances:'] = pd.DataFrame(cum_results).set_index(
                'horizon').T
        if marg_results:
            results['Marginal Loss Functions:\nRealized Variances:'] = pd.DataFrame(marg_results).set_index('horizon').T

    return results


def perform_var_tests(roll_result: MDSVRoll) -> Dict[float, Dict]:
    """Perform VaR backtesting"""
    results = {}

    for alpha in roll_result.var_alpha:
        var_col = f'VaR{int(100 * (1 - alpha))}'
        viol_col = f'I{int(100 * (1 - alpha))}'

        if var_col in roll_result.estimates.columns:
            violations = roll_result.estimates[viol_col].values
            n_viol = np.sum(violations)
            n_obs = len(violations)

            # Kupiec test (unconditional coverage)
            expected_viol = alpha * n_obs
            if n_viol > 0:
                lr_uc = -2 * np.log((alpha ** n_viol * (1 - alpha) ** (n_obs - n_viol)) /
                                    ((n_viol / n_obs) ** n_viol * (1 - n_viol / n_obs) ** (n_obs - n_viol)))
            else:
                lr_uc = -2 * np.log(alpha ** n_viol * (1 - alpha) ** (n_obs - n_viol))

            # Christoffersen test (independence)
            # Create transition matrix
            transitions = np.zeros((2, 2))
            for i in range(1, len(violations)):
                transitions[int(violations[i - 1]), int(violations[i])] += 1

            # Calculate probabilities
            if np.sum(transitions) > 0:
                pi = np.sum(transitions[:, 1]) / np.sum(transitions)
                pi0 = transitions[0, 1] / np.sum(transitions[0, :]) if np.sum(transitions[0, :]) > 0 else 0
                pi1 = transitions[1, 1] / np.sum(transitions[1, :]) if np.sum(transitions[1, :]) > 0 else 0

                # LR statistic
                if pi > 0 and pi < 1 and pi0 > 0 and pi0 < 1 and pi1 > 0 and pi1 < 1:
                    lr_ind = -2 * np.log(
                        ((1 - pi) ** (transitions[0, 0] + transitions[1, 0]) * pi ** (
                                    transitions[0, 1] + transitions[1, 1])) /
                        (((1 - pi0) ** transitions[0, 0] * pi0 ** transitions[0, 1]) *
                         ((1 - pi1) ** transitions[1, 0] * pi1 ** transitions[1, 1]))
                    )
                else:
                    lr_ind = 0.0
            else:
                lr_ind = 0.0

            results[alpha] = {
                'expected': expected_viol,
                'actual': n_viol,
                'actual_pct': n_viol / n_obs,
                'lr_uc': lr_uc,
                'lr_uc_crit': stats.chi2.ppf(1 - alpha, 1),
                'p_uc': 1 - stats.chi2.cdf(lr_uc, 1),
                'reject_uc': lr_uc > stats.chi2.ppf(1 - alpha, 1),
                'lr_ind': lr_ind,
                'lr_ind_crit': stats.chi2.ppf(1 - alpha, 1),
                'p_ind': 1 - stats.chi2.cdf(lr_ind, 1),
                'reject_ind': lr_ind > stats.chi2.ppf(1 - alpha, 1),
                'lr_cc': lr_uc + lr_ind,
                'lr_cc_crit': stats.chi2.ppf(1 - alpha, 2),
                'p_cc': 1 - stats.chi2.cdf(lr_uc + lr_ind, 2),
                'reject_cc': (lr_uc + lr_ind) > stats.chi2.ppf(1 - alpha, 2)
            }

    return results


def _simulate_with_leverage(V_t: np.ndarray, params: np.ndarray,
                            model_type: int, n_start: int) -> Dict:
    """Simulate with leverage effect"""
    n_total = len(V_t)

    # Add burn-in for leverage
    if n_start == 0:
        n_start = 70
        V_t = np.concatenate([V_t[:70], V_t])
        n_total += 70

    # Initialize with some history
    r_t = np.zeros(n_total)
    r_t[:70] = np.random.randn(70) * np.sqrt(V_t[:70])

    # Leverage parameters
    if model_type == 0:
        l1, theta = params[5:7]
    elif model_type == 2:
        l1, theta = params[10:12]
    else:
        return {}  # No leverage for pure RV model

    # Generate with leverage
    for t in range(70, n_total):
        # Calculate leverage multiplier
        leverage = 1.0
        for i in range(min(70, t)):
            if r_t[t - i - 1] < 0:
                li = l1 * (theta ** i)
                leverage *= (1 + li * np.abs(r_t[t - i - 1]) / np.sqrt(V_t[t - i - 1]))

        r_t[t] = np.random.randn() * np.sqrt(V_t[t] * leverage)

    result = {'r_t': r_t[n_start:]}

    # Add RV if joint model
    if model_type == 2:
        xi, varphi, delta1, delta2, shape = params[5:10]
        e_t = r_t / np.sqrt(V_t)
        log_RV = (xi + varphi * np.log(V_t) + delta1 * e_t +
                  delta2 * (e_t ** 2 - 1) + shape * np.random.randn(n_total))
        result['RV_t'] = np.exp(log_RV)[n_start:]

    return result


def _bootstrap_with_leverage_py(fit, pi_0, sig, matP, mc_sim, z_t, n_ahead):
    """Python implementation of bootstrap with leverage"""
    n_boot = len(mc_sim)
    model_type_map = {
        "Univariate log-return": 0,
        "Joint log-return and realized variances": 2
    }
    model_type = model_type_map.get(fit.model_type, 0)

    # Get historical data for leverage initialization
    n_hist = min(200, len(fit.data))
    hist_returns = fit.data[-n_hist:, 0]

    # Leverage parameters
    if model_type == 0:
        l1, theta = fit.estimates[5:7]
    else:
        l1, theta = fit.estimates[10:12]

    # Simulate paths
    rt_sim = np.zeros((n_boot, n_ahead))

    for i in range(n_boot):
        # Get volatility path
        vol_path = sig[mc_sim[i]]

        # Initialize leverage calculation with history
        recent_returns = list(hist_returns[-70:])
        recent_vols = [1.0] * 70  # Simplified

        # Apply leverage and generate returns
        for t in range(n_ahead):
            # Calculate leverage
            leverage = 1.0
            for j in range(min(70, len(recent_returns))):
                if recent_returns[-(j + 1)] < 0:
                    lj = l1 * (theta ** j)
                    leverage *= (1 + lj * np.abs(recent_returns[-(j + 1)]) / np.sqrt(recent_vols[-(j + 1)]))

            # Generate return
            rt_sim[i, t] = z_t[i, t] * np.sqrt(vol_path[t] * leverage)

            # Update history
            recent_returns.append(rt_sim[i, t])
            recent_vols.append(vol_path[t] * leverage)
            if len(recent_returns) > 70:
                recent_returns.pop(0)
                recent_vols.pop(0)

    results = {
        'rt_sim': rt_sim,
        'rt2': np.mean(rt_sim ** 2, axis=0)
    }

    # Add RV if joint model
    if model_type == 2:
        xi, varphi, delta1, delta2, shape = fit.estimates[5:10]
        rvt_sim = np.zeros((n_boot, n_ahead))

        for i in range(n_boot):
            vol_path = sig[mc_sim[i]]
            e_t = rt_sim[i] / np.sqrt(vol_path)
            log_rv = (xi + varphi * np.log(vol_path) + delta1 * e_t +
                      delta2 * (e_t ** 2 - 1) + shape * np.random.randn(n_ahead))
            rvt_sim[i] = np.exp(log_rv)

        results['rvt_sim'] = rvt_sim
        results['rvt'] = np.mean(rvt_sim, axis=0)

    return results


def _analytical_forecast_py(fit, pi_0, sig, matP, n_ahead):
    """Python implementation of analytical forecast"""
    model_type_map = {
        "Univariate log-return": 0,
        "Univariate realized variances": 1,
        "Joint log-return and realized variances": 2
    }
    model_type = model_type_map[fit.model_type]

    forecasts = {'rt2': [], 'rvt': []}

    # Current distribution
    pi_t = pi_0.copy()

    for h in range(n_ahead):
        # Expected variance
        expected_var = np.sum(pi_t * sig)
        forecasts['rt2'].append(expected_var)

        # For RV models
        if model_type == 1:
            # Simple RV forecast
            forecasts['rvt'].append(expected_var)
        elif model_type == 2:
            # Joint model forecast
            xi, varphi, delta1, delta2, shape = fit.estimates[5:10]
            # Expected log RV
            expected_log_rv = xi + varphi * np.sum(pi_t * np.log(sig))
            # Adjustment for log-normal
            expected_rv = np.exp(expected_log_rv + 0.5 * shape ** 2)
            forecasts['rvt'].append(expected_rv)

        # Update distribution
        pi_t = matP.T @ pi_t

    return {
        'rt2': np.array(forecasts['rt2']),
        'rvt': np.array(forecasts['rvt']) if forecasts['rvt'] else None
    }
    leverage_val = results['Levier'][-1]
    vol_vector = vol_vector * leverage_val


for alpha in var_alpha:
    var_values[alpha] = _calculate_var(vol_vector, pi_0, alpha)

return MDSVFilter(
    model_type=model_names[model_type],
    leverage=leverage,
    N=N,
    K=K,
    data=data_array,
    estimates=params,
    log_likelihood=-results['loglik'],
    leverage_values=results['Levier'],
    filtered_proba=results['filtred_proba'],
    smoothed_proba=results['smoothed_proba'],
    dates=dates,
    marg_loglik=results.get('Marg_loglik'),
    var_values=var_values
)


def bootstrap_forecast(fit: Union[MDSVFit, MDSVFilter],
                       n_ahead: int = 100,
                       n_bootpred: int = 10000,
                       rseed: Optional[int] = None,
                       use_cpp: bool = True) -> MDSVBoot:
    """
    Bootstrap forecasting for MDSV model

    Parameters
    ----------
    fit : MDSVFit or MDSVFilter
        Fitted model
    n_ahead : int
        Forecast horizon
    n_bootpred : int
        Number of bootstrap predictions
    rseed : int, optional
        Random seed
    use_cpp : bool
        Whether to use C++ implementation

    Returns
    -------
    MDSVBoot
        Bootstrap forecast results
    """
    if rseed is not None:
        np.random.seed(rseed)

    # Extract model info
    model_type_map = {
        "Univariate log-return": 0,
        "Univariate realized variances": 1,
        "Joint log-return and realized variances": 2
    }
    model_type = model_type_map[fit.model_type]

    # Get filtering results
    if isinstance(fit, MDSVFit):
        # Need to filter first
        filter_results = filter_mdsv(fit.data, fit.estimates, fit.N, fit.K,
                                     model_type, fit.leverage, False, use_cpp=use_cpp)
    else:
        filter_results = fit

    # Extract final filtered distribution
    pi_0 = filter_results.filtered_proba[:, -1]

    # Create process object
    process = MDSVProcess(fit.N, fit.K)
    process.update_from_vector(fit.estimates, fit.leverage, model_type)

    # Get volatility vector and transition matrix
    sig = process.volatility_vector()
    matP = process.full_transition_matrix()

    # Simulate Markov chain paths
    mc_sim = _simulate_markov_chain(pi_0, matP, n_ahead, n_bootpred)

    # Generate standard normal innovations
    z_t = np.random.randn(n_bootpred, n_ahead)

    # Run bootstrap
    if fit.leverage:
        # With leverage - need to simulate paths
        if use_cpp and mdsv_cpp is not None:
            # Prepare leverage matrix
            n_hist = min(200, len(fit.data))
            hist_data = fit.data[-n_hist:, 0]
            levier_results = mdsv_cpp.levierVolatility(hist_data, fit.estimates, 70, model_type)
            levier_mat = np.tile(levier_results['Levier'], (n_bootpred, 1)).T

            results = mdsv_cpp.R_hat(n_ahead, hist_data, mc_sim.T, z_t.T,
                                     levier_mat, sig, fit.estimates, model_type, fit.N, 70)
        else:
            results = _bootstrap_with_leverage_py(fit, pi_0, sig, matP, mc_sim, z_t, n_ahead)

        # Extract relevant results
        rt_sim = results.get('rt_sim')
        rt2 = results.get('rt2')
        rvt_sim = results.get('rvt_sim')
        rvt = results.get('rvt')
    else:
        # Without leverage - analytical calculation
        if use_cpp and mdsv_cpp is not None:
            xi = fit.estimates[5] if model_type == 2 else 0
            varphi = fit.estimates[6] if model_type == 2 else 0
            delta1 = fit.estimates[7] if model_type == 2 else 0
            delta2 = fit.estimates[8] if model_type == 2 else 0
            shape = fit.estimates[9] if model_type == 2 else fit.estimates[5] if model_type == 1 else 0

            results = mdsv_cpp.f_sim(n_ahead, sig, pi_0, matP, varphi, xi, shape, delta1, delta2)
        else:
            results = _analytical_forecast_py(fit, pi_0, sig, matP, n_ahead)

        rt_sim = None
        rt2 = results.get('rt2')
        rvt_sim = None
        rvt = results.get('rvt')

    return MDSVBoot(
        model_type=fit.model_type,
        leverage=fit.leverage,
        N=fit.N,
        K=fit.K,
        data=fit.data,
        estimates=fit.estimates,
        log_likelihood=fit.log_likelihood if isinstance(fit, MDSVFit) else filter_results.log_likelihood,
        n_ahead=n_ahead,
        n_bootpred=n_bootpred,
        dates=fit.dates,
        rt_sim=rt_sim,
        rt2=rt2,
        rvt_sim=rvt_sim,
        rvt=rvt
    )


def rolling_forecast(data: Union[np.ndarray, pd.DataFrame],
                     N: int = 2,
                     K: int = 3,
                     model_type: int = 0,
                     leverage: bool = False,
                     n_ahead: int = 1,
                     n_bootpred: int = 10000,
                     forecast_length: int = 500,
                     refit_every: int = 25,
                     refit_window: str = "recursive",
                     window_size: Optional[int] = None,
                     calculate_var: bool = True,
                     var_alpha: List[float] = None,
                     n_jobs: int = 1,
                     rseed: Optional[int] = None,
                     verbose: bool = True,
                     use_cpp: bool = True) -> MDSVRoll:
    """
    Rolling estimation and forecasting

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Full dataset
    N, K : int
        Model dimensions
    model_type : int
        0: returns, 1: RV, 2: joint
    leverage : bool
        Include leverage effect
    n_ahead : int
        Forecast horizon
    n_bootpred : int
        Bootstrap replications
    forecast_length : int
        Out-of-sample period length
    refit_every : int
        Refit frequency
    refit_window : str
        "recursive" or "moving"
    window_size : int, optional
        For moving window
    calculate_var : bool
        Calculate VaR
    var_alpha : list
        VaR levels
    n_jobs : int
        Parallel jobs
    rseed : int
        Random seed
    verbose : bool
        Print progress
    use_cpp : bool
        Use C++ code

    Returns
    -------
    MDSVRoll
        Rolling forecast results
    """
    if var_alpha is None:
        var_alpha = [0.01, 0.05]

    if rseed is not None:
        np.random.seed(rseed)

    # Convert data
    if isinstance(data, pd.DataFrame):
        dates = data.index if isinstance(data.index, pd.DatetimeIndex) else None
        data_array = data.values
    else:
        dates = None
        data_array = np.asarray(data)

    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)

    n_total = len(data_array)

    # Determine window size
    if window_size is None:
        window_size = n_total - forecast_length - 1

    # Model type names
    model_names = {
        0: "Univariate log-return",
        1: "Univariate realized variances",
        2: "Joint log-return and realized variances"
    }

    # Initialize results storage
    estimates_list = []
    prevision_list = []

    # Determine refit points
    refit_points = list(range(0, forecast_length, refit_every))
    if forecast_length not in refit_points:
        refit_points.append(forecast_length)

    # Progress bar
    if verbose:
        pbar = tqdm(total=forecast_length, desc="Rolling forecast")

    # Current parameter estimates
    current_params = None

    for t in range(forecast_length):
        # Determine data window
        if refit_window == "recursive":
            start_idx = 0
        else:  # moving
            start_idx = max(0, t - window_size + 1)

        end_idx = n_total - forecast_length + t
        window_data = data_array[start_idx:end_idx]

        # Refit if necessary
        if t in refit_points:
            if verbose and t > 0:
                pbar.set_description(f"Refitting at t={t}")

            fit_result = fit_mdsv(window_data, N, K, model_type, leverage,
                                  start_params=current_params, use_cpp=use_cpp,
                                  verbose=False)
            current_params = fit_result.estimates

        # Filter with current parameters
        filter_result = filter_mdsv(window_data, current_params, N, K,
                                    model_type, leverage, calculate_var,
                                    var_alpha, use_cpp)

        # Store estimates
        est_dict = {
            'date': dates[end_idx] if dates is not None else end_idx,
            'rt': data_array[end_idx, 0] if model_type != 1 else np.nan,
            'rvt': data_array[end_idx, 1] if model_type != 0 else np.nan,
            'model': 'MDSV',
            'N': N,
            'K': K,
            'Levier': leverage,
            'ModelType': model_names[model_type],
            'predict_loglik': filter_result.log_likelihood,
            'loglik': filter_result.log_likelihood,
            'AIC': filter_result.aic,
            'BIC': filter_result.bic
        }

        # Add parameter estimates
        param_names = _get_param_names(model_type, leverage)
        for name, value in zip(param_names, current_params):
            est_dict[name] = value

        # Add VaR estimates if calculated
        if calculate_var and model_type != 1:
            for alpha in var_alpha:
                if alpha in filter_result.var_values:
                    est_dict[f'VaR{int(100 * (1 - alpha))}'] = filter_result.var_values[alpha]
                    # Check VaR violation
                    est_dict[f'I{int(100 * (1 - alpha))}'] = (
                            data_array[end_idx, 0] < filter_result.var_values[alpha]
                    )

        estimates_list.append(est_dict)

        # Generate forecasts
        if n_ahead > 1 or leverage:
            # Need bootstrap
            boot_result = bootstrap_forecast(filter_result, n_ahead,
                                             n_bootpred, use_cpp=use_cpp)

            prev_dict = {'date': est_dict['date']}
            if model_type != 1:
                prev_dict.update({f'rt2p{h}': boot_result.rt2[h - 1]
                                  for h in range(1, n_ahead + 1)})
            if model_type != 0:
                prev_dict.update({f'rvtp{h}': boot_result.rvt[h - 1]
                                  for h in range(1, n_ahead + 1)})
        else:
            # Analytical one-step ahead
            prev_dict = {'date': est_dict['date']}
            # Simplified - would need proper implementation
            if model_type != 1:
                prev_dict['rt2p1'] = current_params[3] ** 2  # sigma^2
            if model_type != 0:
                prev_dict['rvtp1'] = current_params[3] ** 2

        prevision_list.append(prev_dict)

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    # Convert to DataFrames
    estimates_df = pd.DataFrame(estimates_list)
    prevision_df = pd.DataFrame(prevision_list)

    return MDSVRoll(
        model_type=model_names[model_type],
        leverage=leverage,
        N=N,
        K=K,
        data=data_array,
        n_ahead=n_ahead,
        forecast_length=forecast_length,
        refit_every=refit_every,
        refit_window=refit_window,
        window_size=window_size,
        calculate_var=calculate_var,
        var_alpha=np.array(var_alpha),
        estimates=estimates_df,
        prevision=prevision_df,
        dates=dates
    )


def simulate_mdsv(N: int = 2,
                  K: int = 3,
                  params: np.ndarray = None,
                  model_type: int = 0,
                  leverage: bool = False,
                  n_sim: int = 1000,
                  n_start: int = 0,
                  m_sim: int = 1,
                  rseed: Optional[int] = None,
                  use_cpp: bool = True) -> MDSVSim:
    """
    Simulate from MDSV model

    Parameters
    ----------
    N, K : int
        Model dimensions
    params : np.ndarray
        Model parameters
    model_type : int
        0: returns, 1: RV, 2: joint
    leverage : bool
        Include leverage
    n_sim : int
        Simulation length
    n_start : int
        Burn-in period
    m_sim : int
        Number of simulations
    rseed : int
        Random seed
    use_cpp : bool
        Use C++ code

    Returns
    -------
    MDSVSim
        Simulation results
    """
    if rseed is not None:
        np.random.seed(rseed)

    # Default parameters if not provided
    if params is None:
        params = _get_default_params(model_type, leverage)

    # Model type names
    model_names = {
        0: "Univariate log-return",
        1: "Univariate realized variances",
        2: "Joint log-return and realized variances"
    }

    # Create process
    process = MDSVProcess(N, K)
    process.update_from_vector(params[:5])

    # Get volatility vector and transition matrix
    vol_vector = process.volatility_vector()
    trans_matrix = process.full_transition_matrix()
    stationary_dist = process.stationary_distribution()

    # Storage for simulations
    simulations = {}

    for sim_num in range(m_sim):
        # Simulate Markov chain
        states = _simulate_markov_chain(stationary_dist, trans_matrix,
                                        n_sim + n_start, 1)[0]

        # Get volatilities
        V_t = vol_vector[states]

        if leverage:
            # With leverage effect
            sim_data = _simulate_with_leverage(V_t, params, model_type, n_start)
        else:
            # Without leverage
            if model_type == 0:
                # Returns only
                r_t = np.random.randn(n_sim + n_start) * np.sqrt(V_t)
                sim_data = {'r_t': r_t[n_start:]}
            elif model_type == 1:
                # RV only
                shape = params[5]
                RV_t = V_t * np.random.gamma(shape, 1 / shape, size=n_sim + n_start)
                sim_data = {'RV_t': RV_t[n_start:]}
            else:
                # Joint
                xi, varphi, delta1, delta2, shape = params[5:10]
                r_t = np.random.randn(n_sim + n_start) * np.sqrt(V_t)
                e_t = r_t / np.sqrt(V_t)

                log_RV = (xi + varphi * np.log(V_t) + delta1 * e_t +
                          delta2 * (e_t ** 2 - 1) + shape * np.random.randn(n_sim + n_start))
                RV_t = np.exp(log_RV)

                sim_data = {
                    'r_t': r_t[n_start:],
                    'RV_t': RV_t[n_start:]
                }

        simulations[sim_num] = sim_data

    return MDSVSim(
        model_type=model_names[model_type],
        leverage=leverage,
        N=N,
        K=K,
        parameters=params,
        n_sim=n_sim,
        n_start=n_start,
        m_sim=m_sim,
        simulations=simulations
    )


# Helper functions

def _get_default_start_params(data: np.ndarray, model_type: int,
                              leverage: bool) -> np.ndarray:
    """Get default starting parameters"""
    params = [0.52, 0.85, 2.77, np.sqrt(np.var(data[:, 0])), 0.72]

    if model_type == 1:
        params.append(2.10)  # shape
    elif model_type == 2:
        params.extend([-1.5, 0.72, -0.09, 0.04, 2.10])

    if leverage:
        params.extend([1.5, 0.87568])

    return np.array(params)


def _get_default_params(model_type: int, leverage: bool) -> np.ndarray:
    """Get default parameters for simulation"""
    params = [0.52, 0.99, 2.77, 1.95, 0.72]

    if model_type == 1:
        params.append(2.10)
    elif model_type == 2:
        params.extend([-0.5, 0.93, 0.93, 0.04, 2.10])

    if leverage:
        params.extend([0.78, 0.876])

    return np.array(params)


def _get_param_names(model_type: int, leverage: bool) -> List[str]:
    """Get parameter names"""
    names = ["omega", "a", "b", "sigma", "v0"]

    if model_type == 1:
        names.append("shape")
    elif model_type == 2:
        names.extend(["xi", "varphi", "delta1", "delta2", "shape"])

    if leverage:
        names.extend(["l", "theta"])

    return names


def _nat_to_work_py(params: np.ndarray, leverage: bool,
                    model_type: int) -> np.ndarray:
    """Transform natural to working parameters (Python)"""
    params_tilde = params.copy()

    params_tilde[0] = np.log((1 / params[0]) - 1)  # omega
    params_tilde[1] = np.log((1 / params[1]) - 1)  # a
    params_tilde[2] = np.log(params[2] - 1)  # b
    params_tilde[3] = np.log(params[3])  # sigma
    params_tilde[4] = np.log((1 / params[4]) - 1)  # v0

    j = 0
    if model_type == 1:
        params_tilde[5] = np.log(params[5])  # shape
        j = 1
    elif model_type == 2:
        # xi, varphi, delta1, delta2 unchanged
        params_tilde[9] = np.log(params[9])  # shape
        j = 5

    if leverage:
        params_tilde[5 + j] = np.log(params[5 + j])  # l
        params_tilde[6 + j] = np.log((1 / params[6 + j]) - 1)  # theta

    return params_tilde


def _work_to_nat_py(params_tilde: np.ndarray, leverage: bool,
                    model_type: int, fixed_params: Optional[Dict] = None) -> np.ndarray:
    """Transform working to natural parameters (Python)"""
    params = params_tilde.copy()

    params[0] = 1 / (1 + np.exp(params_tilde[0]))  # omega
    params[1] = 1 / (1 + np.exp(params_tilde[1]))  # a
    params[2] = 1 + np.exp(params_tilde[2])  # b
    params[3] = np.exp(params_tilde[3])  # sigma
    params[4] = 1 / (1 + np.exp(params_tilde[4]))  # v0

    j = 0
    if model_type == 1:
        params[5] = np.exp(params_tilde[5])  # shape
        j = 1
    elif model_type == 2:
        # xi, varphi, delta1, delta2 unchanged
        params[9] = np.exp(params_tilde[9])  # shape
        j = 5

    if leverage: