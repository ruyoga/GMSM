"""
MDSV Simulation Module
Generate synthetic data from MDSV models
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import gamma, lognorm


class MDSVSimulator:
    """
    Simulate data from MDSV models

    Useful for:
    - Model validation
    - Monte Carlo studies
    - Forecasting via simulation
    - Testing estimation procedures
    """

    def __init__(self, N: int = 3, D: int = 10, model_type: int = 0, leverage: bool = False):
        """
        Initialize simulator

        Parameters
        ----------
        N : int
            Number of volatility components
        D : int
            States per component
        model_type : int
            0: returns, 1: RV, 2: joint
        leverage : bool
            Whether to include leverage effect
        """
        self.N = N
        self.D = D
        self.model_type = model_type
        self.leverage = leverage
        self.n_states = D ** N

    def simulate(self, params: Dict, n_periods: int = 1000,
                 burn_in: int = 200, seed: Optional[int] = None) -> Dict:
        """
        Simulate data from MDSV model

        Parameters
        ----------
        params : dict
            Model parameters
        n_periods : int
            Number of periods to simulate
        burn_in : int
            Burn-in period to discard
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary containing:
            - 'returns': simulated returns (if applicable)
            - 'rv': simulated realized variance (if applicable)
            - 'volatility': true volatility path
            - 'states': hidden state sequence
            - 'leverage': leverage multipliers (if applicable)
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert params dict to array
        param_array = self._params_to_array(params)

        # Generate volatility states and transition matrix
        sigma_states = self._compute_volatility_vector(param_array)
        P = self._compute_transition_matrix(param_array)

        # Total simulation length
        total_periods = n_periods + burn_in

        # Simulate Markov chain
        states = self._simulate_markov_chain(P, param_array, total_periods)

        # Get volatility path
        volatility = sigma_states[states]

        # Initialize output
        results = {
            'states': states[burn_in:],
            'volatility': volatility[burn_in:]
        }

        # Apply leverage if needed
        if self.leverage:
            leverage_mult = np.ones(total_periods)
            if self.model_type in [0, 2]:  # Need returns for leverage
                # First generate returns without leverage to bootstrap
                returns_temp = np.random.normal(0, np.sqrt(volatility), total_periods)
                leverage_mult = self._compute_leverage_path(returns_temp, param_array)
                volatility = volatility * leverage_mult
                results['leverage'] = leverage_mult[burn_in:]

        # Generate observations based on model type
        if self.model_type == 0:  # Returns only
            returns = np.random.normal(0, np.sqrt(volatility), total_periods)
            results['returns'] = returns[burn_in:]

        elif self.model_type == 1:  # RV only
            shape = param_array[5]
            rv = self._simulate_rv(volatility, shape, total_periods)
            results['rv'] = rv[burn_in:]

        elif self.model_type == 2:  # Joint
            returns, rv = self._simulate_joint(volatility, param_array, total_periods)
            results['returns'] = returns[burn_in:]
            results['rv'] = rv[burn_in:]

        # Update volatility in results
        results['volatility'] = volatility[burn_in:]

        return results

    def simulate_multiple_paths(self, params: Dict, n_paths: int = 100,
                                n_periods: int = 1000,
                                seed: Optional[int] = None) -> Dict:
        """
        Simulate multiple independent paths

        Parameters
        ----------
        params : dict
            Model parameters
        n_paths : int
            Number of paths to simulate
        n_periods : int
            Length of each path
        seed : int, optional
            Random seed

        Returns
        -------
        dict
            Dictionary with arrays of shape (n_paths, n_periods)
        """
        if seed is not None:
            np.random.seed(seed)

        # Storage
        all_results = {}

        for i in range(n_paths):
            path_results = self.simulate(params, n_periods, burn_in=200)

            if i == 0:
                # Initialize storage
                for key, value in path_results.items():
                    all_results[key] = np.zeros((n_paths, len(value)))

            # Store this path
            for key, value in path_results.items():
                all_results[key][i] = value

        return all_results

    def _simulate_markov_chain(self, P: np.ndarray, params: np.ndarray,
                               n_periods: int) -> np.ndarray:
        """Simulate Markov chain path"""
        states = np.zeros(n_periods, dtype=int)

        # Initial state from stationary distribution
        p0 = self._compute_stationary_dist(params)
        states[0] = np.random.choice(self.n_states, p=p0)

        # Simulate transitions
        for t in range(1, n_periods):
            trans_probs = P[states[t - 1], :]
            states[t] = np.random.choice(self.n_states, p=trans_probs)

        return states

    def _simulate_rv(self, volatility: np.ndarray, shape: float,
                     n_periods: int) -> np.ndarray:
        """Simulate realized variance"""
        # Using log-normal distribution as in the paper
        rv = np.zeros(n_periods)
        for t in range(n_periods):
            rv[t] = volatility[t] * np.random.lognormal(-shape / 2, np.sqrt(shape))
        return rv

    def _simulate_joint(self, volatility: np.ndarray, params: np.ndarray,
                        n_periods: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate joint returns and RV"""
        xi = params[5]
        varphi = params[6]
        delta1 = params[7]
        delta2 = params[8]
        shape = params[9]

        returns = np.zeros(n_periods)
        rv = np.zeros(n_periods)

        for t in range(n_periods):
            # Generate return
            epsilon = np.random.normal(0, 1)
            returns[t] = np.sqrt(volatility[t]) * epsilon

            # Generate RV conditional on return
            mu_rv = xi + varphi * np.log(volatility[t]) + \
                    delta1 * epsilon + delta2 * (epsilon ** 2 - 1)
            rv[t] = np.random.lognormal(mu_rv, np.sqrt(shape))

        return returns, rv

    def _compute_leverage_path(self, returns: np.ndarray,
                               params: np.ndarray) -> np.ndarray:
        """Compute leverage multipliers for entire path"""
        n = len(returns)
        leverage = np.ones(n)

        # Get leverage parameters
        j = 0
        if self.model_type == 1:
            j = 1
        elif self.model_type == 2:
            j = 5

        if self.leverage:
            l1 = params[5 + j]
            theta = params[6 + j]

            # Compute leverage effect
            for t in range(1, n):
                lev_t = 1.0
                for i in range(min(t, 70)):  # Max 70 lags
                    if returns[t - i - 1] < 0:
                        li = l1 * (theta ** i)
                        lev_t *= (1 + li * abs(returns[t - i - 1]) /
                                  np.sqrt(leverage[t - i - 1]))
                leverage[t] = lev_t

        return leverage

    def _params_to_array(self, params: Dict) -> np.ndarray:
        """Convert parameter dictionary to array"""
        param_list = [
            params['omega'],
            params['a'],
            params['b'],
            params['sigma'],
            params['v0']
        ]

        if self.model_type == 1:
            param_list.append(params['shape'])
        elif self.model_type == 2:
            param_list.extend([
                params['xi'],
                params['varphi'],
                params['delta1'],
                params['delta2'],
                params['shape']
            ])

        if self.leverage:
            param_list.extend([params['l'], params['theta']])

        return np.array(param_list)

    def _compute_volatility_vector(self, params: np.ndarray) -> np.ndarray:
        """Compute volatility state vector"""
        omega = params[0]
        sigma2 = params[3]
        v0 = params[4]

        # Create state values for single component
        sigma_i = np.array([v0 * ((2 - v0) / v0) ** k for k in range(self.D)])

        # Compute stationary probabilities
        from scipy.special import binom
        proba_pi = np.array([
            binom(self.D - 1, k) * omega ** k * (1 - omega) ** (self.D - 1 - k)
            for k in range(self.D)
        ])

        # Normalize
        e_i = np.sum(proba_pi * sigma_i)
        sigma_i = sigma_i / e_i

        # Kronecker product for N components
        sigma = np.array([1.0])
        for _ in range(self.N):
            sigma = np.kron(sigma, sigma_i)

        return sigma2 * sigma

    def _compute_transition_matrix(self, params: np.ndarray) -> np.ndarray:
        """Compute transition matrix"""
        omega = params[0]
        a = params[1]
        b = params[2]

        # Persistence levels
        phi = np.array([a ** (b ** i) for i in range(self.N)])

        # Stationary probabilities
        from scipy.special import binom
        proba_pi = np.array([
            binom(self.D - 1, k) * omega ** k * (1 - omega) ** (self.D - 1 - k)
            for k in range(self.D)
        ])

        # Build transition matrices for each component
        P_components = []
        for i in range(self.N):
            P_i = phi[i] * np.eye(self.D) + (1 - phi[i]) * np.outer(np.ones(self.D), proba_pi)
            P_components.append(P_i)

        # Kronecker product
        P = P_components[0]
        for i in range(1, self.N):
            P = np.kron(P, P_components[i])

        return P

    def _compute_stationary_dist(self, params: np.ndarray) -> np.ndarray:
        """Compute stationary distribution"""
        omega = params[0]

        from scipy.special import binom
        proba_single = np.array([
            binom(self.D - 1, k) * omega ** k * (1 - omega) ** (self.D - 1 - k)
            for k in range(self.D)
        ])

        proba = proba_single.copy()
        for _ in range(1, self.N):
            proba = np.kron(proba, proba_single)

        return proba