"""
MDSV Filtering Module
Implements forward filtering and backward smoothing algorithms
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.special import binom


class MDSVFilter:
    """
    Filtering algorithms for MDSV models

    Implements:
    - Forward filtering (Hamilton filter)
    - Backward smoothing
    - Viterbi algorithm for most likely state sequence
    """

    def __init__(self, model):
        """
        Initialize filter with MDSV model

        Parameters
        ----------
        model : MDSV
            MDSV model instance (can be fitted or unfitted)
        """
        self.model = model
        self.K = model.D  # States per component
        self.N = model.N  # Number of components
        self.n_states = model.n_states

    def forward_filter(self, data: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward filtering algorithm (Hamilton filter)

        Parameters
        ----------
        data : np.ndarray
            Observation data of shape (T,) or (T, 2)
        params : np.ndarray
            Model parameters

        Returns
        -------
        filtered_probs : np.ndarray
            Filtered probabilities P(S_t | Y_1:t) of shape (T, n_states)
        log_likelihood : float
            Log-likelihood of the data
        """
        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T
        T = len(data)

        # Get model components
        sigma = self.model._compute_volatility_vector(params)
        P = self.model._compute_transition_matrix(params)
        p0 = self._compute_initial_distribution(params)

        # Storage
        filtered_probs = np.zeros((T, self.n_states))
        log_lik = 0.0

        # Initialize
        alpha = p0.copy()

        for t in range(T):
            # Compute observation likelihood
            likelihood = self._observation_likelihood(data[t], sigma, params)

            # Update step
            alpha = alpha * likelihood
            c = np.sum(alpha)

            if c > 0:
                alpha = alpha / c
                log_lik += np.log(c)
            else:
                # Handle numerical issues
                alpha = np.ones(self.n_states) / self.n_states
                log_lik -= 100  # Penalty for numerical issues

            filtered_probs[t] = alpha

            # Prediction step for next time
            if t < T - 1:
                alpha = P.T @ alpha

        return filtered_probs, log_lik

    def backward_smooth(self, data: np.ndarray, params: np.ndarray,
                        filtered_probs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward-backward smoothing algorithm

        Parameters
        ----------
        data : np.ndarray
            Observation data
        params : np.ndarray
            Model parameters
        filtered_probs : np.ndarray, optional
            Pre-computed filtered probabilities

        Returns
        -------
        smoothed_probs : np.ndarray
            Smoothed probabilities P(S_t | Y_1:T) of shape (T, n_states)
        """
        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T
        T = len(data)

        # Get filtered probabilities if not provided
        if filtered_probs is None:
            filtered_probs, _ = self.forward_filter(data, params)

        # Get transition matrix
        P = self.model._compute_transition_matrix(params)
        sigma = self.model._compute_volatility_vector(params)

        # Initialize backward recursion
        smoothed_probs = np.zeros_like(filtered_probs)
        smoothed_probs[-1] = filtered_probs[-1]

        # Backward pass
        beta = np.ones(self.n_states)

        for t in range(T - 2, -1, -1):
            # Get likelihood for next observation
            likelihood_next = self._observation_likelihood(data[t + 1], sigma, params)

            # Backward recursion
            beta = P @ (likelihood_next * beta)
            beta = beta / np.sum(beta)  # Normalize to prevent underflow

            # Combine with filtered probabilities
            smoothed_probs[t] = filtered_probs[t] * beta
            smoothed_probs[t] = smoothed_probs[t] / np.sum(smoothed_probs[t])

        return smoothed_probs

    def viterbi(self, data: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm for most likely state sequence

        Parameters
        ----------
        data : np.ndarray
            Observation data
        params : np.ndarray
            Model parameters

        Returns
        -------
        states : np.ndarray
            Most likely state sequence of shape (T,)
        """
        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T
        T = len(data)

        # Get model components
        sigma = self.model._compute_volatility_vector(params)
        P = self.model._compute_transition_matrix(params)
        p0 = self._compute_initial_distribution(params)

        # Initialize Viterbi arrays
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        likelihood = self._observation_likelihood(data[0], sigma, params)
        delta[0] = np.log(p0 + 1e-10) + np.log(likelihood + 1e-10)

        # Forward pass
        for t in range(1, T):
            likelihood = self._observation_likelihood(data[t], sigma, params)

            for j in range(self.n_states):
                # Find best previous state
                transitions = delta[t - 1] + np.log(P[:, j] + 1e-10)
                psi[t, j] = np.argmax(transitions)
                delta[t, j] = np.max(transitions) + np.log(likelihood[j] + 1e-10)

        # Backward pass - trace back best path
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def compute_predictive_distribution(self, filtered_probs: np.ndarray,
                                        params: np.ndarray,
                                        n_ahead: int = 1) -> np.ndarray:
        """
        Compute predictive distribution for future states

        Parameters
        ----------
        filtered_probs : np.ndarray
            Last filtered probability distribution
        params : np.ndarray
            Model parameters
        n_ahead : int
            Number of steps ahead

        Returns
        -------
        predictive_probs : np.ndarray
            Predictive probabilities of shape (n_ahead, n_states)
        """
        P = self.model._compute_transition_matrix(params)

        predictive_probs = np.zeros((n_ahead, self.n_states))
        current_probs = filtered_probs.copy()

        for h in range(n_ahead):
            current_probs = P.T @ current_probs
            predictive_probs[h] = current_probs

        return predictive_probs

    def _compute_initial_distribution(self, params: np.ndarray) -> np.ndarray:
        """Compute initial state distribution (stationary distribution)"""
        omega = params[0]

        # Single component probabilities (binomial)
        proba_single = np.array([
            binom(self.K - 1, k) * omega ** k * (1 - omega) ** (self.K - 1 - k)
            for k in range(self.K)
        ])

        # Kronecker product for N components
        proba = proba_single.copy()
        for _ in range(1, self.N):
            proba = np.kron(proba, proba_single)

        return proba

    def _observation_likelihood(self, obs: np.ndarray, sigma: np.ndarray,
                                params: np.ndarray) -> np.ndarray:
        """Compute observation likelihood for each state"""
        return self.model._compute_observation_likelihood(obs, sigma, params)

    def compute_state_dependent_moments(self, params: np.ndarray) -> Dict:
        """
        Compute moments for each state

        Returns
        -------
        dict
            Dictionary with 'mean' and 'variance' arrays for each state
        """
        sigma = self.model._compute_volatility_vector(params)

        moments = {
            'volatility': np.sqrt(sigma),
            'variance': sigma
        }

        if self.model.model_type == 2:  # Joint model
            # Add RV-specific moments
            xi = params[5]
            varphi = params[6]
            shape = params[9]

            moments['rv_mean'] = np.exp(xi + varphi * np.log(sigma) + shape / 2)
            moments['rv_variance'] = moments['rv_mean'] ** 2 * (np.exp(shape) - 1)

        return moments