import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import warnings



class MSM:
    """
    MSM from the 2004 Calvert & Fisher Paper
    -----------
    ret : numpy.ndarray
        Input time series of returns (T x 1).
    kbar : int
        Number of frequency components (volatility cascades).
    n_vol : int
        Number of trading periods in a year (for annualization).
    nw_lag : int
        Number of lags for Newey-West standard errors (currently affects gradient calc).
    parameters : numpy.ndarray
        Estimated model parameters [m0, b, gamma_k, sigma_unannualized].
    std_errors : numpy.ndarray
        Standard errors of the estimated parameters.
    log_likelihood : float
        The maximized log-likelihood value of the fitted model.
    results : dict
        - LL: Log-likelihood.
        - LLs: Vector of log-likelihood contributions per time step.
        - filtered_probabilities: P(state_t | data_{1:t}).
        - smoothed_probabilities: P(state_t | data_{1:T}).
        - transition_matrix: Estimated state transition matrix A.
        - state_vol_multipliers: Estimated state volatility multipliers g_m.
        - component_matrix: Matrix mapping states to component values (Mmat).
        - optim_message: Message from the optimizer.
        - optim_convergence: Boolean indicating optimizer convergence.
        - optim_iter: Number of optimizer iterations.
        - parameters: Estimated parameters [m0, b, gamma_k, sigma_unannualized].
        - std_errors: Standard errors.
        - coefficients: Coefficients ready for display (b is NaN if kbar=1, sigma is annualized).
    """
    def __init__(self, ret, kbar=1, n_vol=252, para0=None, nw_lag=0):
        """
        Parameters:
        -----------
        ret : array-like
            Vector or Series of financial returns.
        kbar : int, optional
            Number of frequency components (volatility cascades), default is 1.
            Determines the number of states (2^kbar).
        n_vol : int, optional
            Number of trading periods in a year (e.g., 252 for daily data),
            used for annualizing sigma. Default is 252.
        para0 : list or tuple, optional
            Initial parameter values [m0, b, gammak, sigma_annualized] for optimization.
            If None, default starting values are used.
        nw_lag : int, optional
            Number of lags for Newey-West adjustment in standard error calculation.
            Default is 0 (no adjustment). Note: Implementation currently uses OPG standard errors.
        """
        self.n_vol = n_vol
        self.nw_lag = nw_lag

        checked_inputs = self._check_and_prepare_inputs(ret, kbar, para0)
        self.ret = checked_inputs["dat"]
        self.kbar = checked_inputs["kbar"]
        self.k_states = 2**self.kbar
        self._para0 = checked_inputs["start_value"]
        self._lb = checked_inputs["lb"]
        self._ub = checked_inputs["ub"]

        # De-annualize initial sigma guess
        self._para0[3] = self._para0[3] / np.sqrt(self.n_vol)

        self.parameters = None
        self.std_errors = None
        self.log_likelihood = None
        self.results = {}
        self._fit()

    def _matrix_power(self, A, power):
        """
        Calculate matrix power
        -----------
        A : numpy.ndarray
            Square matrix.
        power : int
            Integer power to raise the matrix to.

        Returns:
        --------
        numpy.ndarray
            A raised to the power `power`.
        """
        if not isinstance(power, int) or power < 0:
            raise ValueError("power must be a non-negative integer.")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix.")
        return np.linalg.matrix_power(A, power)

    def _check_and_prepare_inputs(self, dat, kbar, x0):
        """
        Validate input data and parameters, return processed inputs.
        """
        if isinstance(dat, (pd.DataFrame, pd.Series)):
            dat = dat.values

        if not isinstance(dat, np.ndarray):
            dat = np.array(dat)

        if dat.ndim == 1:
            dat = dat.reshape(-1, 1)  # Ensure column vector
        elif dat.shape[1] > 1:
            warnings.warn("Input data has multiple columns. Using the first column for returns.", UserWarning)
            dat = dat[:, 0].reshape(-1, 1)

        if np.any(np.isnan(dat)):
            raise ValueError("Input data contains NaN values.")

        if kbar < 1 or not isinstance(kbar, int):
            raise ValueError('kbar (number of volatility components) must be a positive integer.')

        # Parameter bounds (internal sigma is unannualized)
        lb = [1.0 + 1e-6, 1.0 + 1e-6, 1e-6, 1e-6]  # m0>1, b>1, gamma>0, sigma>0
        ub = [2.0 - 1e-6, 50.0, 1.0 - 1e-6, 50.0]  # m0<2, b large, gamma<1, sigma large

        if x0 is not None:
            if len(x0) != 4:
                raise ValueError('Initial values (para0) must be of length 4: [m0, b, gammak, sigma_annualized]')

            m0, b_init, gamma_k_init, sigma_ann_init = x0

            if not (lb[0] <= m0 <= ub[0]):
                warnings.warn(
                    f"Initial m0 ({m0}) outside typical bounds ({lb[0]:.2f}, {ub[0]:.2f}). Clamping to bounds.",
                    UserWarning)
                m0 = np.clip(m0, lb[0], ub[0])

            # b only relevant if kbar > 1
            if kbar > 1 and not (lb[1] <= b_init <= ub[1]):
                warnings.warn(f"Initial b ({b_init}) outside bounds ({lb[1]:.2f}, {ub[1]:.2f}). Clamping to bounds.",
                              UserWarning)
                b_init = np.clip(b_init, lb[1], ub[1])
            elif kbar == 1:
                b_init = 1.5  # Assign a default valid value, won't be used in estimation
            if not (lb[2] <= gamma_k_init <= ub[2]):
                warnings.warn(
                    f"Initial gammak ({gamma_k_init}) outside bounds ({lb[2]:.4f}, {ub[2]:.4f}). Clamping to bounds.",
                    UserWarning)
                gamma_k_init = np.clip(gamma_k_init, lb[2], ub[2])
            if sigma_ann_init <= 0:
                warnings.warn(f"Initial annualized sigma ({sigma_ann_init}) must be positive. Using default.",
                              UserWarning)
                sigma_ann_init = np.std(dat) * np.sqrt(self.n_vol) * 100  # Default guess (annualized %)
                if sigma_ann_init <= lb[3]:  # default is positive
                    sigma_ann_init = lb[3] * np.sqrt(self.n_vol) * 1.1  # Slightly above lower bound

            start_value = [m0, b_init, gamma_k_init, sigma_ann_init]

        else:
            # Default initial values if none provided
            m0_def = 1.5
            b_def = 2.5 if kbar > 1 else 1.5  # Default b > 1 only if needed
            gamma_k_def = 0.9
            # Calculate annualized std dev from data as initial sigma guess
            sigma_ann_def = np.std(dat) * np.sqrt(self.n_vol)
            # Ensure default sigma is positive and within bounds
            if sigma_ann_def <= lb[3] / np.sqrt(self.n_vol):
                sigma_ann_def = lb[3] * np.sqrt(self.n_vol) * 1.1  # Use lower bound if std dev is too small

            start_value = [m0_def, b_def, gamma_k_def, sigma_ann_def]
            print(f"Using default starting values: {start_value}")

        # Adjust bounds if kbar == 1 (b is irrelevant)
        if kbar == 1:
            lb[1] = 1.5  # Fix b to an arbitrary valid value
            ub[1] = 1.5
            start_value[1] = 1.5  # Fix initial b as well

        return {
            "dat": dat,
            "kbar": kbar,
            "start_value": start_value,  # sigma is annualized here
            "lb": lb,  # sigma bound is unannualized
            "ub": ub  # sigma bound is unannualized
        }

    def _calculate_transition_matrix(self, b, gamma_kbar):
        """
        Parameters:
        -----------
        b : float
            Growth rate parameter.
        gamma_kbar : float
            Transition probability of the highest frequency component.

        Returns:
        --------
        numpy.ndarray
            The transition matrix A (k_states x k_states).
        """
        if not (self._lb[2] <= gamma_kbar <= self._ub[2]):
            gamma_kbar = np.clip(gamma_kbar, self._lb[2], self._ub[2])
            warnings.warn(f"gamma_kbar clipped to [{self._lb[2]:.4f}, {self._ub[2]:.4f}]", RuntimeWarning)

        if self.kbar > 1 and not (self._lb[1] <= b <= self._ub[1]):
            b = np.clip(b, self._lb[1], self._ub[1])
            warnings.warn(f"b clipped to [{self._lb[1]:.2f}, {self._ub[1]:.2f}]", RuntimeWarning)

        gamma_k = np.zeros(self.kbar)

        power_base = 1.0 - gamma_kbar
        if power_base <= 0: power_base = 1e-10  # Avoid log(<=0) or 0^neg

        if self.kbar > 1:
            exponent = 1.0 / (b ** (self.kbar - 1))
        else:
            exponent = 1.0

        try:
            gamma_k[0] = 1.0 - power_base ** exponent
        except ValueError:  # Handle potential complex result if base is negative (unlikely with clip)
            gamma_k[0] = 1e-10  # Assign small positive value
            warnings.warn("Numerical issue calculating gamma_k[0]. Setting to smol value.", RuntimeWarning)

        # Ensure gamma_k[0] is within valid probs range
        gamma_k[0] = np.clip(gamma_k[0], 1e-10, 1.0 - 1e-10)

        # Base 2x2 transition matrix for the first component (k=1)
        A_comp1 = np.array([
            [1.0 - 0.5 * gamma_k[0], 0.5 * gamma_k[0]],
            [0.5 * gamma_k[0], 1.0 - 0.5 * gamma_k[0]]
        ])
        A = A_comp1

        # Build the full transition matrix using Kronecker product for kbar > 1
        if self.kbar > 1:
            base_gamma0 = 1.0 - gamma_k[0]
            if base_gamma0 <= 0: base_gamma0 = 1e-10  # Avoid issues

            for i in range(1, self.kbar):
                try:
                    gamma_k[i] = 1.0 - base_gamma0 ** (b ** i)
                except ValueError:
                    gamma_k[i] = 1e-10
                    warnings.warn(f"Numerical issue calculating gamma_k[{i}]. Setting to small value.", RuntimeWarning)

                gamma_k[i] = np.clip(gamma_k[i], 1e-10, 1.0 - 1e-10)

                # Transition matrix for component i+1
                a_i = np.array([
                    [1.0 - 0.5 * gamma_k[i], 0.5 * gamma_k[i]],
                    [0.5 * gamma_k[i], 1.0 - 0.5 * gamma_k[i]]
                ])
                A = np.kron(A, a_i)

        # Normalize rows to ensure they sum to 1
        row_sums = np.sum(A, axis=1, keepdims=True)
        # Avoid division by zero if a row sum is somehow zero
        row_sums[row_sums == 0] = 1.0
        A = A / row_sums

        return A

    def _calculate_component_matrix(self, m0):
        """
        Parameters:
        -----------
        m0 : float
            The base multiplier value.

        Returns:
        --------
        numpy.ndarray
            Matrix Mmat (k_states x kbar). Mmat[i, k] is the value
            of component k+1 in state i.
        """
        if not (self._lb[0] <= m0 <= self._ub[0]):
            m0 = np.clip(m0, self._lb[0], self._ub[0])
            warnings.warn(f"m0 clipped to [{self._lb[0]:.2f}, {self._ub[0]:.2f}]", RuntimeWarning)

        m1 = 2.0 - m0
        Mmat = np.zeros((self.k_states, self.kbar))

        for i in range(self.k_states):  # Iterate through states 0 to k_states - 1
            for j in range(self.kbar):  # Iterate through components 0 to kbar - 1
                # Check if the j-th bit of state i is 1
                if (i >> j) & 1:
                    Mmat[i, j] = m1  # If component j is 1, component j+1 value is m1
                else:
                    Mmat[i, j] = m0  # If component j is 0, component j+1 value is m0
        return Mmat

    def _calculate_state_vol_multipliers(self, m0):
        """
        Calculate all possible state volatility multipliers (g_m).
        g_m = sqrt(product of component values for that state).

        Parameters:
        -----------
        m0 : float
            The base multiplier value.

        Returns:
        --------
        numpy.ndarray
            Vector g_m of state volatility multipliers (1 x k_states).
        """
        if not (self._lb[0] <= m0 <= self._ub[0]):
            m0 = np.clip(m0, self._lb[0], self._ub[0])
            warnings.warn(f"m0 clipped to [{self._lb[0]:.2f}, {self._ub[0]:.2f}]", RuntimeWarning)

        m1 = 2.0 - m0
        g_m_sq = np.ones(self.k_states)  # Start with 1.0 for product

        for i in range(self.k_states):  # Iterate through states
            for j in range(self.kbar):  # Iterate through components/bits
                if (i >> j) & 1:  # Check j-th bit
                    g_m_sq[i] *= m1
                else:
                    g_m_sq[i] *= m0

        # Ensure non-negativity before sqrt, although m0, m1 should be positive
        g_m_sq = np.maximum(g_m_sq, 0)
        g_m = np.sqrt(g_m_sq)

        return g_m.reshape(1, -1)  # Return as row vector

    def _calculate_conditional_densities(self, params):
        """
        Calculate the conditional densities p(ret_t | state_j) for all t and j.

        Parameters:
        -----------
        params : array-like
            Parameter vector [m0, b, gamma_k, sigma_unannualized].

        Returns:
        --------
        numpy.ndarray
            Matrix omega_t (T x k_states) of conditional normal densities.
        """
        m0, _, _, sigma_unann = params  # Extract needed parameters
        g_m = self._calculate_state_vol_multipliers(m0)  # Shape (1, k_states)

        if sigma_unann <= 0:
            sigma_unann = 1e-6  # Ensure positive sigma
            warnings.warn("Sigma <= 0 encountered in density calculation. Setting to small positive.", RuntimeWarning)

        state_sigmas = sigma_unann * g_m  # Shape (1, k_states)

        T = self.ret.shape[0]
        sig_mat = np.tile(state_sigmas, (T, 1))
        ret_mat = np.tile(self.ret, (1, self.k_states))

        # Calculate normal PDF: N(ret_t; 0, state_sigma_j^2)
        # pdf = (1 / (sqrt(2*pi) * sigma)) * exp(-0.5 * (x / sigma)^2)
        # Avoid division by zero or very small sigma
        sig_mat = np.maximum(sig_mat, 1e-16)  # Floor sigma values

        norm_const = 1.0 / (np.sqrt(2 * np.pi) * sig_mat)
        exponent = -0.5 * np.square(ret_mat / sig_mat)

        omega_t = norm_const * np.exp(exponent)

        # Add small constant to prevent zero probabilities (important for log-likelihood)
        omega_t = omega_t + 1e-16

        return omega_t

    def _calculate_likelihood(self, params, return_details=False):
        """
        Calculate log-likelihood and optionally filtered probabilities using Hamilton filter.

        Parameters:
        -----------
        params : array-like
            Parameter vector [m0, b, gamma_k, sigma_unannualized].
        return_details : bool, optional
            If True, return dict with pmat, LL, LLs, A, g_m.
            If False, return only the negative log-likelihood value (for optimization).

        Returns:
        --------
        float or dict :
            If return_details is False: Negative log-likelihood (-sum(LLs)).
            If return_details is True: Dictionary containing:
                'pmat': Filtered probabilities P(state_t | data_{1:t}) (T+1 x k_states).
                        Includes initial state t=0.
                'LL': Negative log-likelihood.
                'LLs': Vector of log-likelihood contributions log(p(data_t|data_{1:t-1})) (T x 1).
                'A': Transition matrix.
                'g_m': State volatility multipliers.
        """
        m0, b, gamma_k, sigma_unann = params

        # --- Calculate model components ---
        A = self._calculate_transition_matrix(b, gamma_k)  # (k_states x k_states)
        g_m = self._calculate_state_vol_multipliers(m0)  # (1 x k_states)

        # Calculate conditional densities p(ret_t | state_j)
        omega_t = self._calculate_conditional_densities(params)
        # omega_t has shape (T, k_states) where T = len(self.ret)
        T = omega_t.shape[0]  # Number of observations/returns

        pmat = np.zeros((T + 1, self.k_states))  # Filtered probs P(S_t | data_{1:t})
        # Row 0 is P(S_0), Rows 1 to T for data
        LLs = np.zeros(T)  # Log-likelihood contributions log p(y_t | Y_{t-1})

        pmat[0, :] = 1.0 / self.k_states

        for t in range(T):
            predicted_prob_st = pmat[t, :] @ A  # Shape (1, k_states)

            # Likelihood of observation y_t: p(y_t | data_{1:t-1})
            likelihood_yt = np.sum(predicted_prob_st * omega_t[t, :])  # Scalar

            # Store log-likelihood contribution
            if likelihood_yt <= 1e-100:
                warnings.warn(f"Likelihood near zero at t={t + 1}. Check model/data.", RuntimeWarning)
                LLs[t] = -100.0  # Avoid log(0), assign large penalty
                pmat[t + 1, :] = 1.0 / self.k_states
            else:
                LLs[t] = np.log(likelihood_yt)
                # Update step: P(S_t | data_{1:t}) = P(S_t | data_{1:t-1}) * p(y_t | S_t) / p(y_t | data_{1:t-1})
                numerator = predicted_prob_st * omega_t[t, :]  # Element-wise product
                pmat[t + 1, :] = numerator / likelihood_yt

            pmat[t + 1, :] = pmat[t + 1, :] / np.sum(pmat[t + 1, :])

        neg_ll = -np.sum(LLs)

        if not np.isfinite(neg_ll):
            warnings.warn(f"Log-likelihood is not finite ({neg_ll}). Optimization might fail.", RuntimeWarning)
            neg_ll = 1e12  # Large number

        if return_details:
            return {
                "pmat": pmat,  # Shape (T+1, k_states)
                "LL": neg_ll,
                "LLs": LLs,  # Shape (T,)
                "A": A,
                "g_m": g_m
            }
        else:
            return neg_ll

    def _log_likelihood_objective(self, params, *args):
        """
        Objective function for the optimizer. Returns negative log-likelihood.

        Parameters:
        -----------
        params : array-like
            Parameter vector [m0, b, gamma_k, sigma_unannualized].
        *args : tuple
            Placeholder for any additional arguments (currently none).

        Returns:
        --------
        float
            Negative log-likelihood.
        """
        return self._calculate_likelihood(params, return_details=False)

    def _smooth_probabilities(self, A, filtered_pmat):
        """
        Calculate smoothed probabilities P(state_t | data_{1:T}) using Kim's smoother.

        Parameters:
        -----------
        A : numpy.ndarray
            Transition matrix (k_states x k_states).
        filtered_pmat : numpy.ndarray
            Filtered probabilities P(state_t | data_{1:t}) from Hamilton filter.
            Shape (T+1, k_states), where T is number of observations.

        Returns:
        --------
        numpy.ndarray
            Smoothed probabilities (T x k_states), corresponding to observations 1 to T.
        """
        T_plus_1, k = filtered_pmat.shape  # T_plus_1 = T_obs + 1
        T_obs = T_plus_1 - 1

        smoothed_p = np.zeros((T_obs, k))  # Smoothed probs for t=1...T_obs

        smoothed_p[T_obs - 1, :] = filtered_pmat[T_obs, :]

        predicted_p = np.zeros((T_obs, k))
        for t in range(T_obs):
            predicted_p[t, :] = filtered_pmat[t, :] @ A  # P(S_{t+1} | data_{1:t})

        for t in range(T_obs - 2, -1, -1):
            pred_next = predicted_p[t + 1, :]
            pred_next[pred_next < 1e-100] = 1e-100  # Floor denominator
            ratio = smoothed_p[t + 1, :] / pred_next

            update_factor = A.T @ ratio  # Shape (k,)
            smoothed_p[t, :] = filtered_pmat[t + 1, :] * update_factor

            smoothed_p[t, :] = smoothed_p[t, :] / np.sum(smoothed_p[t, :])

        return smoothed_p  # Shape (T_obs, k_states)

    def _predict_volatility(self, P, A, g_m, sigma_unann, h=None):
        """
        Calculate predicted or fitted volatility/variance.

        Parameters:
        -----------
        P : numpy.ndarray
            Matrix of state probabilities (filtered or smoothed) (T x k_states).
        A : numpy.ndarray
            Transition matrix (k_states x k_states).
        g_m : numpy.ndarray
            State volatility multipliers (1 x k_states).
        sigma_unann : float
            Unannualized sigma parameter.
        h : int, optional
            Forecast horizon. If None, calculates fitted values. If >= 1,
            calculates h-step ahead forecast based on the last probability in P.

        Returns:
        --------
        dict :
            'vol': Annualized volatility forecast/fit (T x 1 or 1 x 1).
            'vol_sq': Annualized variance forecast/fit (T x 1 or 1 x 1).
        """
        if h is not None:
            if not isinstance(h, int) or h < 1:
                raise ValueError("Forecast horizon h must be a positive integer.")

            # h-step ahead forecast uses last probability P[T-1, :]
            A_h = self._matrix_power(A, h)
            p_hat = P[-1:, :] @ A_h  # Shape (1, k_states)
        else:
            p_hat = P  # Shape (T, k_states)

        g_m_squared = np.square(g_m)  # Shape (1, k_states)

        expected_g_m_sq = p_hat @ g_m_squared.T

        # Unannualized variance
        vol_sq_unann = (sigma_unann ** 2) * expected_g_m_sq

        # Annualize
        vol_sq_ann = vol_sq_unann * self.n_vol
        vol_ann = np.sqrt(vol_sq_ann)

        return {
            "vol": vol_ann,  # Annualized Volatility
            "vol_sq": vol_sq_ann  # Annualized Variance
        }

    def _calculate_gradient(self, params):
        """
        Calculate the gradient (score) of the log-likelihood function
        using numerical finite differences.

        Parameters:
        -----------
        params : array-like
            Parameter vector [m0, b, gamma_k, sigma_unannualized] at which
            to evaluate the gradient.

        Returns:
        --------
        numpy.ndarray
            Gradient matrix (T x n_params), where T is number of observations
            and n_params is the number of parameters (4).
        """
        n_params = len(params)
        T = self.ret.shape[0]
        grad = np.zeros((T, n_params))

        h = 1e-7  # Adjust step size if needed
        h_vec = np.maximum(np.abs(params) * h, h)  # Relative step size, minimum h

        for i in range(n_params):
            # Central difference: (f(x+h) - f(x-h)) / (2h)
            params_fwd = params.copy()
            params_bwd = params.copy()

            params_fwd[i] += h_vec[i]
            params_bwd[i] -= h_vec[i]

            # Clip parameters to bounds if step takes them outside
            params_fwd = np.clip(params_fwd, self._lb, self._ub)
            params_bwd = np.clip(params_bwd, self._lb, self._ub)

            try:
                ll_fwd = self._calculate_likelihood(params_fwd, return_details=True)["LLs"]
            except Exception as e:
                warnings.warn(f"Error calculating forward LL for grad param {i}: {e}. Setting LLs to zeros.",
                              RuntimeWarning)
                ll_fwd = np.zeros(T)

            try:
                ll_bwd = self._calculate_likelihood(params_bwd, return_details=True)["LLs"]
            except Exception as e:
                warnings.warn(f"Error calculating backward LL for grad param {i}: {e}. Setting LLs to zeros.",
                              RuntimeWarning)
                ll_bwd = np.zeros(T)

            # Calculate derivative for parameter i
            delta_ll = ll_fwd - ll_bwd
            delta_h = params_fwd[i] - params_bwd[i]  # Actual difference used

            if delta_h == 0:
                grad[:, i] = 0
            else:
                grad[:, i] = delta_ll / delta_h

        # Special handling for kbar=1: gradient w.r.t 'b' is zero
        if self.kbar == 1:
            grad[:, 1] = 0.0

        return grad  # Shape (T, n_params)

    def _calculate_std_errors(self, params):
        """
        Calculate standard errors for model parameters using OPG estimate.
        SE = sqrt(diag(inv(J))) where J = sum(g_t * g_t') and g_t is gradient at time t.

        Parameters:
        -----------
        params : array-like
            Estimated parameter vector [m0, b, gamma_k, sigma_unannualized].

        Returns:
        --------
        numpy.ndarray
            Vector of standard errors (n_params x 1).
        """
        n_params = len(params)
        # Calculate gradient matrix g (T x n_params)
        g = self._calculate_gradient(params)  # T x n_params

        # Handle potential NaNs or Infs in gradient
        if not np.all(np.isfinite(g)):
            warnings.warn("Non-finite values found in gradient matrix. Standard errors may be unreliable.",
                          RuntimeWarning)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)  # Replace non-finite with 0

        J = g.T @ g  # Shape (n_params x n_params)

        try:
            cov_matrix = np.linalg.inv(J + np.eye(n_params) * 1e-8)
            variances = np.diag(cov_matrix)
            variances = np.maximum(variances, 0)
            se = np.sqrt(variances)
        except np.linalg.LinAlgError:
            warnings.warn("Could not invert information matrix J. Standard errors set to NaN.", RuntimeWarning)
            se = np.full(n_params, np.nan)

        if self.kbar == 1:
            se[1] = np.nan

        # Note: SEs calculated here are for the unannualized sigma.

        return se.reshape(-1, 1)

    def _calculate_marginal_probabilities(self, p, m0, Mmat):
        """
        Calculate marginal probabilities P(M_k=m0 | data) for each component k.

        Parameters:
        -----------
        p : numpy.ndarray
            Matrix of smoothed or filtered state probabilities (T x k_states).
        m0 : float
            The base multiplier value.
        Mmat : numpy.ndarray
            Matrix of volatility component values for each state (k_states x kbar).

        Returns:
        --------
        numpy.ndarray
            Matrix of marginal probabilities (T x kbar).
        """
        if p.shape[1] != self.k_states:
            raise ValueError("Number of columns in p must match k_states.")
        if Mmat.shape[0] != self.k_states or Mmat.shape[1] != self.kbar:
            raise ValueError("Dimensions of Mmat are incorrect.")
        if not (self._lb[0] <= m0 <= self._ub[0]):
            m0 = np.clip(m0, self._lb[0], self._ub[0])
            warnings.warn(f"m0 clipped to [{self._lb[0]:.2f}, {self._ub[0]:.2f}]", RuntimeWarning)

        T = p.shape[0]
        m_marginals = np.zeros((T, self.kbar))


        for k in range(self.kbar):
            states_where_comp_k_is_m0 = np.isclose(Mmat[:, k], m0)
            m_marginals[:, k] = np.sum(p[:, states_where_comp_k_is_m0], axis=1)

        return m_marginals


    def _fit(self):
        bounds = list(zip(self._lb, self._ub))

        objective = self._log_likelihood_objective

        print("Starting optimization...")
        opt_result = minimize(
            objective,
            self._para0,  # Initial guess (sigma unannualized)
            args=(),  # No extra args needed for objective
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False, 'maxiter': 500}
        )
        print(f"Optimization finished: Success={opt_result.success}, Message={opt_result.message}")

        if not opt_result.success:
            warnings.warn(f"Optimization did not converge: {opt_result.message}", RuntimeWarning)
            self.parameters = opt_result.x
        else:
            self.parameters = opt_result.x  # Store estimated parameters [m0, b, gammak, sigma_unann]

        # Recalculate likelihood details with final parameters
        final_likelihood_details = self._calculate_likelihood(self.parameters, return_details=True)

        self.log_likelihood = -final_likelihood_details["LL"]  # Store positive LL
        self.std_errors = self._calculate_std_errors(self.parameters)

        # Calculate smoothed probabilities
        # smoothed_p should be (T, k)
        smoothed_p = self._smooth_probabilities(
            final_likelihood_details["A"],
            final_likelihood_details["pmat"]
        )

        coef = self.parameters.copy()
        se_display = self.std_errors.copy()

        if self.kbar == 1:
            coef[1] = np.nan
            se_display[1] = np.nan

        coef[3] = coef[3] * np.sqrt(self.n_vol)
        se_display[3] = se_display[3] * np.sqrt(self.n_vol)

        self.results = {
            "LL": self.log_likelihood,
            "LLs": final_likelihood_details["LLs"],  # Shape (T,)
            "filtered_probabilities": final_likelihood_details["pmat"][1:, :],
            "smoothed_probabilities": smoothed_p,  # Shape (T, k_states)
            "transition_matrix": final_likelihood_details["A"],
            "state_vol_multipliers": final_likelihood_details["g_m"],
            "component_matrix": self._calculate_component_matrix(self.parameters[0]),
            "optim_message": opt_result.message,
            "optim_convergence": opt_result.success,
            "optim_iter": opt_result.nit,
            "parameters": self.parameters,  # [m0, b, gammak, sigma_unann]
            "std_errors": self.std_errors,  # SEs for unannualized sigma
            "coefficients": coef,  # Coefs for display (annualized sigma)
            "se_display": se_display  # SEs for display (annualized sigma)
        }
        self.coef_names = ["m0", "b", "gammak", "sigma_annualized"]

    def summary(self):
        if self.parameters is None:
            print("Model has not been fitted yet.")
            return None

        coef = self.results["coefficients"]
        se = self.results["se_display"].flatten()  # Use SEs for annualized sigma

        # Calculate t-values and p-values
        tval = np.full_like(coef, np.nan)
        valid_se = (se != 0) & ~np.isnan(se)
        tval[valid_se] = coef[valid_se] / se[valid_se]

        n_estimated_params = np.sum(~np.isnan(coef))
        df = len(self.ret) - n_estimated_params
        if df <= 0:
            warnings.warn("Degrees of freedom <= 0. P-values may be unreliable.", RuntimeWarning)
            pval = np.full_like(coef, np.nan)
        else:
            pval = 2 * (1 - stats.t.cdf(np.abs(tval), df=df))

        print("*" * 76)
        print(f"  Markov Switching Multifractal Model (MSM) - kbar={self.kbar}")
        print("*" * 76)
        print(f"  Log-Likelihood: {self.log_likelihood:.4f}")
        print(f"  Observations:   {len(self.ret)}")
        print(
            f"  Optimization:   Converged={self.results['optim_convergence']}, Iterations={self.results['optim_iter']}")
        print("-" * 76)

        summary_data = {
            "Estimate": coef,
            "Std Error": se,
            "t-value": tval,
            "p-value": pval
        }
        summary_df = pd.DataFrame(summary_data, index=self.coef_names)

        pd.options.display.float_format = '{:,.4f}'.format
        print(summary_df)
        print("-" * 76)
        pd.options.display.float_format = None

        return summary_df

    def predict(self, h=None):
        """
        Parameters:
        -----------
        h : int, optional
            Forecast horizon in periods (matching data frequency).
            - If None (default): Returns fitted conditional volatility based on
              smoothed probabilities.
            - If h >= 1: Returns h-step ahead volatility forecast based on the
              last available filtered probability.
        Returns:
        --------
        dict :
            'vol': numpy.ndarray
                Annualized conditional volatility (fitted or forecast).
                Shape (T, 1) for fitted, (1, 1) for forecast.
            'vol_sq': numpy.ndarray
                Annualized conditional variance (fitted or forecast).
                Shape (T, 1) for fitted, (1, 1) for forecast.
        """
        if self.parameters is None:
            print("Model has not been fitted yet.")
            return None

        sigma_unann = self.parameters[3]
        A = self.results["transition_matrix"]
        g_m = self.results["state_vol_multipliers"]

        if h is None:
            P = self.results["smoothed_probabilities"]  # Shape (T, k)
        else:
            P = self.results["filtered_probabilities"]  # Shape (T, k)

        return self._predict_volatility(P, A, g_m, sigma_unann, h)

    def plot(self, plot_type="vol", use_smoothed=True):
        if self.parameters is None:
            print("Model has not been fitted yet.")
            return

        if plot_type not in ["vol", "volsq"]:
            raise ValueError("plot_type must be either 'vol' or 'volsq'")

        sigma_unann = self.parameters[3]
        A = self.results["transition_matrix"]
        g_m = self.results["state_vol_multipliers"]

        if use_smoothed:
            P = self.results["smoothed_probabilities"]
            title_suffix = "(Smoothed)"
        else:
            P = self.results["filtered_probabilities"]
            title_suffix = "(Filtered)"

        pred = self._predict_volatility(P, A, g_m, sigma_unann, h=None)

        plt.figure(figsize=(12, 6))

        if plot_type == "vol":
            fitted_vol = pred["vol"][:len(self.ret)]  # Match length just in case
            plt.plot(fitted_vol, label=f"Fitted Annualized Volatility {title_suffix}", color='blue')
            plt.plot(np.abs(self.ret * np.sqrt(self.n_vol)), label="Annualized Abs Returns", alpha=0.6, color='orange',
                     linestyle=':')
            plt.title("Fitted Conditional Volatility vs Absolute Returns")
            plt.ylabel("Annualized Volatility")
        else:  # volsq
            fitted_var = pred["vol_sq"][:len(self.ret)]
            plt.plot(fitted_var, label=f"Fitted Annualized Variance {title_suffix}", color='blue')
            plt.plot(np.square(self.ret * np.sqrt(self.n_vol)), label="Annualized Squared Returns", alpha=0.6,
                     color='orange', linestyle=':')
            plt.title("Fitted Conditional Variance vs Squared Returns")
            plt.ylabel("Annualized Variance")

        plt.xlabel("Time Period")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def decompose(self, use_smoothed=True):
        """
        Decompose volatility into contributions from each frequency component M_k.
        Calculates E[M_k | data].

        Parameters:
        -----------
        use_smoothed : bool, optional
            If True (default), use smoothed probabilities for decomposition.
            If False, use filtered probabilities.

        Returns:
        --------
        numpy.ndarray
            Matrix (T x kbar) where each column represents the expected
            value of a volatility component M_k over time.
            E[M_k | data] = m0 * P(M_k=m0 | data) + (2-m0) * P(M_k=2-m0 | data)
        """
        if self.parameters is None:
            print("Model has not been fitted yet.")
            return None
        if self.kbar < 2:
            print("Decomposition requires kbar >= 2.")
            return None

        m0 = self.parameters[0]
        Mmat = self.results["component_matrix"]

        if use_smoothed:
            P = self.results["smoothed_probabilities"]
        else:
            P = self.results["filtered_probabilities"]

        p_m0_marginals = self._calculate_marginal_probabilities(P, m0, Mmat)

        m1 = 2.0 - m0
        expected_M_k = m1 + (2.0 * m0 - 2.0) * p_m0_marginals

        return expected_M_k  # Shape (T, kbar)

    def plot_components(self, use_smoothed=True):
        """
        Plot the decomposed volatility components E[M_k | data] over time.

        Parameters:
        -----------
        use_smoothed : bool, optional
            Passed to the decompose method. If True (default), uses smoothed
            probabilities. If False, uses filtered probabilities.
        """
        if self.parameters is None:
            print("Model has not been fitted yet.")
            return
        if self.kbar < 2:
            print("Component plotting requires kbar >= 2.")
            return

        expected_M_k = self.decompose(use_smoothed=use_smoothed)
        if expected_M_k is None:
            return

        prob_type = "Smoothed" if use_smoothed else "Filtered"
        fig, axes = plt.subplots(self.kbar, 1, figsize=(12, 2.5 * self.kbar), sharex=True)
        fig.suptitle(f"Expected Volatility Components E[M_k | Data] ({prob_type})", fontsize=14)

        for k in range(self.kbar):
            ax = axes[k] if self.kbar > 1 else axes
            ax.plot(expected_M_k[:, k], label=f"E[M_{k + 1} | Data]")
            m0 = self.parameters[0]
            ax.axhline(m0, color='r', linestyle='--', alpha=0.7, label=f'm0={m0:.3f}')
            ax.axhline(2 - m0, color='g', linestyle='--', alpha=0.7, label=f'2-m0={2 - m0:.3f}')

            ax.set_title(f"Component M_{k + 1}")
            ax.set_ylabel("E[M_k]")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='best')

        axes[-1].set_xlabel("Time Period")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

