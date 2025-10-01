"""
MDSV - Multifractal Discrete Stochastic Volatility Model
Python implementation based on Augustyniak et al. (2024)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, gamma, lognorm
from scipy.special import binom
import warnings
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass

# Try to import C++ extensions, fall back to pure Python if not available
try:
    import mdsv_cpp

    USE_CPP = True
except ImportError:
    warnings.warn("C++ extensions not available, using pure Python implementation")
    USE_CPP = False


@dataclass
class MDSVResult:
    """Container for MDSV estimation results"""
    parameters: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    volatility_states: np.ndarray
    transition_matrix: np.ndarray
    filtered_probabilities: Optional[np.ndarray] = None
    smoothed_probabilities: Optional[np.ndarray] = None
    convergence: bool = True
    n_iterations: int = 0


class MDSV:
    """
    Multifractal Discrete Stochastic Volatility Model

    Parameters
    ----------
    N : int
        Number of volatility components
    D : int
        Number of states per component (K in the original paper)
    model_type : str or int
        Type of model: 'returns' (0), 'rv' (1), or 'joint' (2)
    leverage : bool
        Whether to include leverage effect
    """

    def __init__(self, N: int = 3, D: int = 10,
                 model_type: Union[str, int] = 'returns',
                 leverage: bool = False):
        self.N = N
        self.D = D  # Using D for dimension per component (K in paper)
        self.leverage = leverage

        # Convert model type to integer
        if isinstance(model_type, str):
            model_type_map = {'returns': 0, 'rv': 1, 'joint': 2}
            self.model_type = model_type_map.get(model_type.lower(), 0)
        else:
            self.model_type = model_type

        self.n_states = D ** N
        self._fitted = False
        self.params_ = None

        # Initialize C++ core if available
        if USE_CPP:
            self.cpp_core = mdsv_cpp.MDSVCore(D, N)

    def _initialize_params(self, data: np.ndarray) -> np.ndarray:
        """Initialize parameters for optimization"""
        n_params = self._get_n_params()
        params = np.zeros(n_params)

        # Basic parameters
        params[0] = 0.5  # omega
        params[1] = 0.99  # a
        params[2] = 2.5  # b
        params[3] = np.std(data[:, 0]) if data.ndim > 1 else np.std(data)  # sigma
        params[4] = 0.7  # v0

        # Model-specific parameters
        idx = 5
        if self.model_type == 1:
            params[idx] = 2.0  # shape
            idx += 1
        elif self.model_type == 2:
            params[idx] = -0.5  # xi
            params[idx + 1] = 0.9  # varphi
            params[idx + 2] = -0.1  # delta1
            params[idx + 3] = 0.05  # delta2
            params[idx + 4] = 2.0  # shape
            idx += 5

        # Leverage parameters
        if self.leverage:
            params[idx] = 0.8  # l
            params[idx + 1] = 0.9  # theta

        # Transform to working parameters
        return self._nat_to_work(params)

    def _get_n_params(self) -> int:
        """Get number of parameters for current model configuration"""
        n = 5  # Base parameters
        if self.model_type == 1:
            n += 1
        elif self.model_type == 2:
            n += 5
        if self.leverage:
            n += 2
        return n

    def _work_to_nat(self, para_tilde: np.ndarray) -> np.ndarray:
        """Transform working parameters to natural parameters"""
        if USE_CPP:
            return self.cpp_core.work_nat(para_tilde, self.leverage, self.model_type)
        else:
            return self._work_to_nat_python(para_tilde)

    def _nat_to_work(self, para: np.ndarray) -> np.ndarray:
        """Transform natural parameters to working parameters"""
        if USE_CPP:
            return self.cpp_core.nat_work(para, self.leverage, self.model_type)
        else:
            return self._nat_to_work_python(para)

    def _work_to_nat_python(self, para_tilde: np.ndarray) -> np.ndarray:
        """Pure Python implementation of parameter transformation"""
        para = para_tilde.copy()

        para[0] = 1 / (1 + np.exp(para_tilde[0]))  # omega
        para[1] = 1 / (1 + np.exp(para_tilde[1]))  # a
        para[2] = 1 + np.exp(para_tilde[2])  # b
        para[3] = np.exp(para_tilde[3])  # sigma
        para[4] = 1 / (1 + np.exp(para_tilde[4]))  # v0

        j = 0
        if self.model_type == 1:
            para[5] = np.exp(para_tilde[5])  # shape
            j = 1
        elif self.model_type == 2:
            para[5] = para_tilde[5]  # xi
            para[6] = para_tilde[6]  # varphi
            para[7] = para_tilde[7]  # delta1
            para[8] = para_tilde[8]  # delta2
            para[9] = np.exp(para_tilde[9])  # shape
            j = 5

        if self.leverage:
            para[5 + j] = np.exp(para_tilde[5 + j])  # l
            para[6 + j] = 1 / (1 + np.exp(para_tilde[6 + j]))  # theta

        return para

    def _nat_to_work_python(self, para: np.ndarray) -> np.ndarray:
        """Pure Python implementation of inverse parameter transformation"""
        para_tilde = para.copy()

        para_tilde[0] = np.log((1 / para[0]) - 1)  # omega
        para_tilde[1] = np.log((1 / para[1]) - 1)  # a
        para_tilde[2] = np.log(para[2] - 1)  # b
        para_tilde[3] = np.log(para[3])  # sigma
        para_tilde[4] = np.log((1 / para[4]) - 1)  # v0

        j = 0
        if self.model_type == 1:
            para_tilde[5] = np.log(para[5])  # shape
            j = 1
        elif self.model_type == 2:
            para_tilde[5] = para[5]  # xi
            para_tilde[6] = para[6]  # varphi
            para_tilde[7] = para[7]  # delta1
            para_tilde[8] = para[8]  # delta2
            para_tilde[9] = np.log(para[9])  # shape
            j = 5

        if self.leverage:
            para_tilde[5 + j] = np.log(para[5 + j])  # l
            para_tilde[6 + j] = np.log((1 / para[6 + j]) - 1)  # theta

        return para_tilde

    def _compute_volatility_vector(self, para: np.ndarray) -> np.ndarray:
        """Compute volatility state vector"""
        if USE_CPP:
            return self.cpp_core.volatility_vector(para, self.D, self.N)
        else:
            return self._compute_volatility_vector_python(para)

    def _compute_volatility_vector_python(self, para: np.ndarray) -> np.ndarray:
        """Pure Python implementation of volatility vector computation"""
        omega = para[0]
        sigma2 = para[3]
        v0 = para[4]

        # Create state values for single component
        sigma_i = np.zeros(self.D)
        for k in range(self.D):
            sigma_i[k] = v0 * ((2 - v0) / v0) ** k

        # Compute stationary probabilities
        proba_pi = np.array([binom(self.D - 1, k) * omega ** k * (1 - omega) ** (self.D - 1 - k)
                             for k in range(self.D)])

        # Normalize
        e_i = np.sum(proba_pi * sigma_i)
        sigma_i = sigma_i / e_i

        # Kronecker product for N components
        sigma = np.array([1.0])
        for _ in range(self.N):
            sigma = np.kron(sigma, sigma_i)

        return sigma2 * sigma

    def _compute_transition_matrix(self, para: np.ndarray) -> np.ndarray:
        """Compute transition matrix"""
        if USE_CPP:
            return self.cpp_core.transition_matrix(para, self.D, self.N)
        else:
            return self._compute_transition_matrix_python(para)

    def _compute_transition_matrix_python(self, para: np.ndarray) -> np.ndarray:
        """Pure Python implementation of transition matrix computation"""
        omega = para[0]
        a = para[1]
        b = para[2]

        # Persistence levels for each component
        phi = np.array([a ** (b ** i) for i in range(self.N)])

        # Stationary probabilities
        proba_pi = np.array([binom(self.D - 1, k) * omega ** k * (1 - omega) ** (self.D - 1 - k)
                             for k in range(self.D)])

        # Build transition matrices for each component
        P_components = []
        for i in range(self.N):
            P_i = phi[i] * np.eye(self.D) + (1 - phi[i]) * np.outer(np.ones(self.D), proba_pi)
            P_components.append(P_i)

        # Kronecker product of all components
        P = P_components[0]
        for i in range(1, self.N):
            P = np.kron(P, P_components[i])

        return P

    def _log_likelihood(self, para_tilde: np.ndarray, data: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization"""
        if USE_CPP:
            return self.cpp_core.log_likelihood(para_tilde, data,
                                                self.model_type, self.leverage)
        else:
            return self._log_likelihood_python(para_tilde, data)

    def _log_likelihood_python(self, para_tilde: np.ndarray, data: np.ndarray) -> float:
        """Pure Python implementation of log-likelihood computation"""
        para = self._work_to_nat(para_tilde)
        n = len(data)

        # Get volatility states and transition matrix
        sigma = self._compute_volatility_vector(para)
        P = self._compute_transition_matrix(para)

        # Initial state probabilities
        p0 = self._compute_stationary_dist(para)

        # Forward filtering
        log_lik = 0.0
        filter_probs = p0.copy()

        for t in range(n):
            # Compute likelihood for each state
            likelihood_t = self._compute_observation_likelihood(data[t], sigma, para)

            # Joint probability
            joint_probs = filter_probs * likelihood_t
            normalizer = np.sum(joint_probs)

            if normalizer > 0:
                log_lik += np.log(normalizer)
                filter_probs = joint_probs / normalizer
            else:
                return np.inf  # Invalid likelihood

            # Predict next state
            if t < n - 1:
                filter_probs = P.T @ filter_probs

        return -log_lik  # Return negative for minimization

    def _compute_stationary_dist(self, para: np.ndarray) -> np.ndarray:
        """Compute stationary distribution"""
        omega = para[0]

        # Single component probabilities
        proba_single = np.array([binom(self.D - 1, k) * omega ** k * (1 - omega) ** (self.D - 1 - k)
                                 for k in range(self.D)])

        # Kronecker product for N components
        proba = proba_single.copy()
        for _ in range(1, self.N):
            proba = np.kron(proba, proba_single)

        return proba

    def _compute_observation_likelihood(self, obs: np.ndarray,
                                        sigma: np.ndarray,
                                        para: np.ndarray) -> np.ndarray:
        """Compute likelihood of observation for each volatility state"""
        n_states = len(sigma)
        likelihood = np.zeros(n_states)

        if self.model_type == 0:
            # Univariate returns
            r = obs if np.isscalar(obs) else obs[0]
            for i in range(n_states):
                std_dev = np.sqrt(sigma[i])
                likelihood[i] = norm.pdf(r, 0, std_dev)

        elif self.model_type == 1:
            # Univariate realized variance
            rv = obs if np.isscalar(obs) else obs[0]
            shape = para[5]
            for i in range(n_states):
                # Using log-normal distribution as in the paper
                likelihood[i] = lognorm.pdf(rv / sigma[i], s=np.sqrt(shape),
                                            scale=np.exp(-shape / 2)) / sigma[i]

        elif self.model_type == 2:
            # Joint model
            r = obs[0]
            rv = obs[1]
            xi = para[5]
            varphi = para[6]
            delta1 = para[7]
            delta2 = para[8]
            shape = para[9]

            for i in range(n_states):
                std_dev = np.sqrt(sigma[i])
                epsilon = r / std_dev

                # Mean of log RV given return
                mu_rv = xi + varphi * np.log(sigma[i]) + delta1 * epsilon + delta2 * (epsilon ** 2 - 1)

                # Joint likelihood
                likelihood[i] = (norm.pdf(r, 0, std_dev) *
                                 lognorm.pdf(rv, s=np.sqrt(shape), scale=np.exp(mu_rv)))

        return likelihood

    def fit(self, data: np.ndarray,
            initial_params: Optional[np.ndarray] = None,
            method: str = 'L-BFGS-B',
            options: Optional[Dict] = None,
            verbose: bool = False) -> MDSVResult:
        """
        Fit MDSV model to data

        Parameters
        ----------
        data : np.ndarray
            Time series data. Shape (T,) for univariate or (T, 2) for joint model
        initial_params : np.ndarray, optional
            Initial parameter values (in natural scale)
        method : str
            Optimization method for scipy.optimize.minimize
        options : dict, optional
            Options for optimizer
        verbose : bool
            Whether to print optimization progress

        Returns
        -------
        MDSVResult
            Fitted model results
        """
        # Prepare data
        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T
        n_obs = len(data)

        # Initialize parameters
        if initial_params is None:
            x0 = self._initialize_params(data)
        else:
            x0 = self._nat_to_work(initial_params)

        # Set up optimization
        if options is None:
            options = {'maxiter': 1000, 'disp': verbose}

        # Optimize
        result = minimize(
            fun=self._log_likelihood,
            x0=x0,
            args=(data,),
            method=method,
            options=options
        )

        # Extract fitted parameters
        para_tilde = result.x
        para = self._work_to_nat(para_tilde)

        # Store parameters
        self.params_ = self._create_param_dict(para)
        self._fitted = True

        # Compute model components
        volatility_states = self._compute_volatility_vector(para)
        transition_matrix = self._compute_transition_matrix(para)

        # Compute information criteria
        log_lik = -result.fun
        n_params = len(para)
        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(n_obs)

        # Create result object
        return MDSVResult(
            parameters=self.params_,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            volatility_states=volatility_states,
            transition_matrix=transition_matrix,
            convergence=result.success,
            n_iterations=result.nit if hasattr(result, 'nit') else 0
        )

    def filter(self, data: np.ndarray, params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run forward filtering to get filtered probabilities

        Parameters
        ----------
        data : np.ndarray
            Time series data
        params : dict, optional
            Parameters to use (if not provided, uses fitted parameters)

        Returns
        -------
        filtered_probs : np.ndarray
            Filtered probabilities P(S_t | Y_1:t) of shape (T, n_states)
        log_likelihood : float
            Log-likelihood of the data
        """
        if params is None:
            if not self._fitted:
                raise ValueError("Model must be fitted first or parameters provided")
            para = self._param_dict_to_array(self.params_)
        else:
            para = self._param_dict_to_array(params)

        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T
        n = len(data)

        # Get model components
        sigma = self._compute_volatility_vector(para)
        P = self._compute_transition_matrix(para)
        p0 = self._compute_stationary_dist(para)

        # Storage for filtered probabilities
        filtered_probs = np.zeros((n, self.n_states))
        log_lik = 0.0

        # Forward filtering
        filter_probs = p0.copy()
        for t in range(n):
            # Observation likelihood
            likelihood_t = self._compute_observation_likelihood(data[t], sigma, para)

            # Update
            joint_probs = filter_probs * likelihood_t
            normalizer = np.sum(joint_probs)
            log_lik += np.log(normalizer)

            filter_probs = joint_probs / normalizer
            filtered_probs[t, :] = filter_probs

            # Predict
            if t < n - 1:
                filter_probs = P.T @ filter_probs

        return filtered_probs, log_lik

    def smooth(self, data: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
        """
        Run forward-backward algorithm to get smoothed probabilities

        Parameters
        ----------
        data : np.ndarray
            Time series data
        params : dict, optional
            Parameters to use

        Returns
        -------
        smoothed_probs : np.ndarray
            Smoothed probabilities P(S_t | Y_1:T) of shape (T, n_states)
        """
        # First run forward filtering
        filtered_probs, _ = self.filter(data, params)
        n = len(data)

        # Get transition matrix
        if params is None:
            para = self._param_dict_to_array(self.params_)
        else:
            para = self._param_dict_to_array(params)
        P = self._compute_transition_matrix(para)

        # Backward pass
        smoothed_probs = np.zeros_like(filtered_probs)
        smoothed_probs[-1, :] = filtered_probs[-1, :]

        for t in range(n - 2, -1, -1):
            # Backward recursion
            pred_probs = P.T @ filtered_probs[t, :]
            pred_probs[pred_probs == 0] = 1e-10  # Avoid division by zero

            ratio = smoothed_probs[t + 1, :] / pred_probs
            smoothed_probs[t, :] = filtered_probs[t, :] * (P @ ratio)

        return smoothed_probs

    def _create_param_dict(self, para: np.ndarray) -> Dict[str, float]:
        """Create parameter dictionary with named parameters"""
        param_dict = {
            'omega': para[0],
            'a': para[1],
            'b': para[2],
            'sigma': para[3],
            'v0': para[4]
        }

        idx = 5
        if self.model_type == 1:
            param_dict['shape'] = para[idx]
            idx += 1
        elif self.model_type == 2:
            param_dict['xi'] = para[idx]
            param_dict['varphi'] = para[idx + 1]
            param_dict['delta1'] = para[idx + 2]
            param_dict['delta2'] = para[idx + 3]
            param_dict['shape'] = para[idx + 4]
            idx += 5

        if self.leverage:
            param_dict['l'] = para[idx]
            param_dict['theta'] = para[idx + 1]

        return param_dict

    def _param_dict_to_array(self, param_dict: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array"""
        para = [
            param_dict['omega'],
            param_dict['a'],
            param_dict['b'],
            param_dict['sigma'],
            param_dict['v0']
        ]

        if self.model_type == 1:
            para.append(param_dict['shape'])
        elif self.model_type == 2:
            para.extend([
                param_dict['xi'],
                param_dict['varphi'],
                param_dict['delta1'],
                param_dict['delta2'],
                param_dict['shape']
            ])

        if self.leverage:
            para.extend([param_dict['l'], param_dict['theta']])

        return np.array(para)