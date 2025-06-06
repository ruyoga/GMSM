import numpy as np
import pandas as pd

from gmsm.modules.legacy.likelihood import msm_likelihood, msm_ll, msm_smooth
from gmsm.utils import msm_A, msm_clustermat, msm_states, msm_mat_power, msm_marginals

def msm_likelihood2(para, kbar, dat, n_vol):
    """
    Calculate log-likelihood and filtered probability values for MSM(k) model.
    :param para: vector of parameter values [m0, b, gamma_k, sigma]
    :param kbar: n freq components
    :param dat: matrix of returns
    :param n_vol: number of trading days in a year
    :return: dict of log likelihood and filtered probability values, A and g_m
    """
    m0 = para[0]
    b = para[1]
    gamma_k = para[2]
    sigma = para[3] / np.sqrt(n_vol)

    k2 = 2 ** kbar
    A = msm_A(b, gamma_k, kbar)
    g_m = msm_states(m0, kbar)

    N = len(dat)

    # Initial stateprobs
    pimat0 = np.ones(k2) / k2

    # Calculate conditional densities
    sig_mat = np.tile(sigma * g_m, (N, 1))
    omega_t = np.tile(dat, (1, k2))

    # Normal densities
    omega_t = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * ((omega_t / sig_mat) ** 2)) / sig_mat
    omega_t = omega_t + 1e-16  # Add small constant to avoid numerical issues

    # Calculate likelihood
    likelihood = msm_likelihood(pimat0, omega_t, A)

    # Prepare results
    result = {
        "LL": likelihood["LL"],
        "LLs": likelihood["LLs"],
        "filtered": likelihood["pmat"][1:],  # Remove initial state
        "A": A,
        "g_m": g_m
    }

    return result

def msm_ll2(para, kbar, dat, n_vol):
    """
    Calculate log-likelihood for optimization
    :param para: vector of parameter values [m0, b, gamma_k, sigma]
    :param kbar: number of freq components
    :param dat:  matrix of returns
    :param n_vol: number of trading days in a year
    :return: negative log-likelihood
    """
    m0 = para[0]
    b = para[1]
    gamma_k = para[2]
    sigma = para[3] / np.sqrt(n_vol)

    k2 = 2 ** kbar
    A = msm_A(b, gamma_k, kbar)
    g_m = msm_states(m0, kbar)

    N = len(dat)

    # Initial state probabilities
    pimat0 = np.ones(k2) / k2

    # Calculate conditional densities
    sig_mat = np.tile(sigma * g_m, (N, 1))
    omega_t = np.tile(dat, (1, k2))

    # Normal densities
    omega_t = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * ((omega_t / sig_mat) ** 2)) / sig_mat
    omega_t = omega_t + 1e-16  # Add small constant to avoid numerical issues

    # Calculate likelihood
    LL = msm_ll(pimat0, omega_t, A)

    if not np.isfinite(LL):
        print('Warning: Log-likelihood is inf. Probably due to all zeros in conditional probability.')

    return LL

def msm_predict(g_m, sigma, n, P, A, h=None):
    """
    :param g_m: vector of possible msm states
    :param sigma: unconditional volatility
    :param n: n trading days in a year
    :param P: matrix of state probs
    :param A: transmat
    :param h: forecast horizon
    :return: dict of volatility & squared volatility predictions
    """
    if h is not None:
        if h < 1:
            raise ValueError("h must be a non-zero integer")
        h = int(h)

    sigma = sigma / np.sqrt(n)

    if h is not None:
        # h-step ahead forecast
        p_hat = P[-1:, :] @ msm_mat_power(A, h)
        vol_sq = sigma**2 * (p_hat @ (g_m**2).T)
    else:
        # Fitted values
        vol_sq = sigma**2 * (P @ (g_m**2).T)

    # Calculate volatility
    vol = np.sqrt(vol_sq)

    return {
        "vol": vol,
        "vol_sq": vol_sq
    }

def msm_parameter_check(dat, kbar, x0=None):
    """
    Check and validate parameters for MSM model.
    """

    if isinstance(dat, pd.DataFrame) or isinstance(dat, pd.Series):
        dat = dat.values

    if not isinstance(dat, np.ndarray):
        dat = np.array(dat)

    if dat.ndim == 1:
        dat = dat.reshape(-1, 1)
    elif dat.shape[1] > 1:
        dat = dat.T

    if kbar < 1:
        raise ValueError('kbar (number of volatility clusters) must be a positive integer.')

    if x0 is not None:
        if len(x0) != 4:
            raise ValueError('Initial values must be of length 4 in the form [m0, b, gammak, sigma]')

        m0, b, gamma_k, sigma = x0

        if m0 < 1 or m0 > 1.99:
            raise ValueError("m0 must be between (1,1.99]")
        if b < 1:
            raise ValueError("b must be greater than 1")
        if gamma_k < 0.0001 or gamma_k > 0.9999:
            raise ValueError("gamma_k must be between [0,1]")
        if sigma < 0.00001:
            raise ValueError("sigma must be a positive (non-zero) value")
    else:
        x0 = [1.5, 2.5, 0.9, np.std(dat)]

    lb = [1, 1, 0.0001, 0.0001]
    ub = [1.9999, 50, 0.9999, 50]

    return {
        "dat": dat,
        "kbar": kbar,
        "start_value": x0,
        "lb": lb,
        "ub": ub
    }

def msm_grad(para, kbar, ret, n_vol):
    """
    Calculate gradient of MSM model.
    :param para: parameter vector
    :param kbar: number of freq components
    :param ret: return data
    :param n_vol: n trading days in a year
    :return: grad matrix
    """
    def check_para(x):
        if x[0] >= 2:
            x[0] = 1.9999
        if x[2] >= 1:
            x[2] = 0.9999
        return x

    para_size = len(para)
    para = np.array(para).reshape(-1, 1)
    para_abs = np.abs(para)

    # Calculate step size for finite difference
    h = 1e-8 * np.maximum(para_abs, 1e-2) * np.sign(para)

    # Create parameter vectors for forward and backward steps
    ll_1 = np.zeros((len(ret), para_size))
    ll_2 = np.zeros((len(ret), para_size))

    for i in range(para_size):
        # Forward step
        x_temp1 = para.copy()
        x_temp1[i] += h[i]
        # Backward step
        x_temp2 = para.copy()
        x_temp2[i] -= h[i]

        # Calculate likelihoods
        ll_1[:, i] = msm_likelihood2(check_para(x_temp1.flatten()), kbar, ret, n_vol)["LLs"]
        ll_2[:, i] = msm_likelihood2(check_para(x_temp2.flatten()), kbar, ret, n_vol)["LLs"]

    # Calculate derivatives
    der = (ll_1 - ll_2) / (2 * h.T)

    return der

def msm_std_err(para, kbar, ret, n_vol, lag=0):
    """
    Calculate standard errors for MSM model parameters.
    """
    if kbar == 1:
        grad = msm_grad(para, kbar, ret, n_vol)
        grad = np.delete(grad, 1, axis=1)  # Remove column for b (not used when kbar=1)
        J = grad.T @ grad
        s = np.sqrt(np.diag(np.linalg.inv(J)))
        se = np.array([s[0], np.nan, s[1], s[2]]).reshape(-1, 1)
    else:
        # For kbar>1, use numerical approximation
        grad = msm_grad(para, kbar, ret, n_vol)
        J = grad.T @ grad
        se = np.sqrt(np.diag(np.linalg.inv(J))).reshape(-1, 1)

    # Adjust sigma standard error for annualization
    se[3] = se[3] / np.sqrt(n_vol)

    return se