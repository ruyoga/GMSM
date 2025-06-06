import numpy as np
from scipy import stats
from typing import Optional, Tuple, Dict, Union
import warnings

try:
    from . import mdsv_cpp
except ImportError:
    warnings.warn("C++ extension not available. Some functions will be slower.")
    mdsv_cpp = None


class MDSVProcess:
    """
    Multifractal Discrete Stochastic Volatility Process

    Parameters
    ----------
    N : int
        Number of components for the MDSV process
    K : int
        Number of states of each MDSV process component
    omega : float
        Probability of success of the stationary distribution (0 < omega < 1)
    a : float
        Highest persistence of the component (0 < a < 1)
    b : float
        Decay rate of persistences (b > 1)
    sigma : float
        Unconditional standard deviation
    v0 : float
        States defined parameter (0 < v0 < 1)
    """

    def __init__(self, N: int = 2, K: int = 3, omega: float = 0.5,
                 a: float = 0.99, b: float = 2.77, sigma: float = 1.0,
                 v0: float = 0.72):
        self.N = N
        self.K = K
        self.omega = omega
        self.a = a
        self.b = b
        self.sigma = sigma
        self.v0 = v0

        # Validate parameters
        self._validate_parameters()

        # Calculate derived quantities
        self._update_derived_quantities()

    def _validate_parameters(self):
        """Validate parameter constraints"""
        if self.N < 1:
            raise ValueError("N must be positive")
        if self.K < 2:
            raise ValueError("K must be greater than 1")
        if not 0 < self.omega < 1:
            raise ValueError("omega must be between 0 and 1")
        if not 0 < self.a < 1:
            raise ValueError("a must be between 0 and 1")
        if self.b <= 1:
            raise ValueError("b must be greater than 1")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not 0 < self.v0 < 1:
            raise ValueError("v0 must be between 0 and 1")

    def _update_derived_quantities(self):
        """Update derived quantities after parameter changes"""
        # Persistence levels
        self.phi = np.array([self.a ** (self.b ** (i - 1)) for i in range(1, self.N + 1)])

        # State space
        self.nu = np.array([self.v0 * ((2 - self.v0) / self.v0) ** (j - 1)
                            for j in range(1, self.K + 1)])

        # Stationary probabilities
        self.pi = np.array([stats.binom.pmf(j - 1, self.K - 1, self.omega)
                            for j in range(1, self.K + 1)])

    def volatility_vector(self, use_cpp: bool = True) -> np.ndarray:
        """
        Calculate the volatility vector

        Parameters
        ----------
        use_cpp : bool
            Whether to use C++ implementation if available

        Returns
        -------
        np.ndarray
            Volatility vector of dimension K^N
        """
        if use_cpp and mdsv_cpp is not None:
            para = np.array([self.omega, self.a, self.b, self.sigma, self.v0])
            return mdsv_cpp.volatilityVector(para, self.K, self.N)
        else:
            # Python implementation
            return self._volatility_vector_py()

    def _volatility_vector_py(self) -> np.ndarray:
        """Python implementation of volatility vector calculation"""
        # Calculate E[V_i] for normalization
        e_vi = np.sum(self.pi * self.nu)

        # Create all possible combinations of states
        from itertools import product
        states = list(product(range(self.K), repeat=self.N))

        # Calculate volatility for each state combination
        vol_vector = np.zeros(len(states))
        for i, state in enumerate(states):
            vol = self.sigma ** 2
            for j, s in enumerate(state):
                vol *= self.nu[s] / e_vi
            vol_vector[i] = vol

        return vol_vector

    def transition_matrix(self, component: int = 0, use_cpp: bool = True) -> np.ndarray:
        """
        Calculate transition matrix for a component

        Parameters
        ----------
        component : int
            Component index (0 to N-1)
        use_cpp : bool
            Whether to use C++ implementation if available

        Returns
        -------
        np.ndarray
            K x K transition matrix
        """
        if component >= self.N:
            raise ValueError(f"Component index {component} out of range [0, {self.N - 1}]")

        phi_i = self.phi[component]
        P_i = phi_i * np.eye(self.K) + (1 - phi_i) * np.outer(np.ones(self.K), self.pi)

        return P_i

    def full_transition_matrix(self, use_cpp: bool = True) -> np.ndarray:
        """
        Calculate full transition matrix for the MDSV process

        Parameters
        ----------
        use_cpp : bool
            Whether to use C++ implementation if available

        Returns
        -------
        np.ndarray
            K^N x K^N transition matrix
        """
        if use_cpp and mdsv_cpp is not None:
            para = np.array([self.omega, self.a, self.b, self.sigma, self.v0])
            return mdsv_cpp.P(para, self.K, self.N)
        else:
            # Python implementation using Kronecker product
            P = self.transition_matrix(0, use_cpp=False)
            for i in range(1, self.N):
                P_i = self.transition_matrix(i, use_cpp=False)
                P = np.kron(P, P_i)
            return P

    def stationary_distribution(self, use_cpp: bool = True) -> np.ndarray:
        """
        Calculate stationary distribution of the full process

        Parameters
        ----------
        use_cpp : bool
            Whether to use C++ implementation if available

        Returns
        -------
        np.ndarray
            Stationary distribution vector of dimension K^N
        """
        if use_cpp and mdsv_cpp is not None:
            return mdsv_cpp.probapi(self.omega, self.K, self.N)
        else:
            # Python implementation
            pi_full = self.pi.copy()
            for _ in range(1, self.N):
                pi_full = np.kron(pi_full, self.pi)
            return pi_full

    def parameters_to_vector(self, leverage: bool = False, model_type: int = 0,
                             l: float = 1.5, theta: float = 0.87) -> np.ndarray:
        """
        Convert parameters to vector format for estimation

        Parameters
        ----------
        leverage : bool
            Whether to include leverage parameters
        model_type : int
            0: univariate returns, 1: univariate RV, 2: joint
        l : float
            Leverage parameter l
        theta : float
            Leverage parameter theta

        Returns
        -------
        np.ndarray
            Parameter vector
        """
        params = [self.omega, self.a, self.b, self.sigma, self.v0]

        if model_type == 1:
            params.append(2.1)  # shape parameter for RV
        elif model_type == 2:
            params.extend([-1.5, 0.72, -0.09, 0.04, 2.1])  # xi, varphi, delta1, delta2, shape

        if leverage:
            params.extend([l, theta])

        return np.array(params)

    def update_from_vector(self, params: np.ndarray, leverage: bool = False,
                           model_type: int = 0):
        """
        Update parameters from vector format

        Parameters
        ----------
        params : np.ndarray
            Parameter vector
        leverage : bool
            Whether leverage parameters are included
        model_type : int
            0: univariate returns, 1: univariate RV, 2: joint
        """
        self.omega = params[0]
        self.a = params[1]
        self.b = params[2]
        self.sigma = params[3]
        self.v0 = params[4]

        self._validate_parameters()
        self._update_derived_quantities()

    def __repr__(self):
        return (f"MDSVProcess(N={self.N}, K={self.K}, omega={self.omega:.3f}, "
                f"a={self.a:.3f}, b={self.b:.3f}, sigma={self.sigma:.3f}, "
                f"v0={self.v0:.3f})")