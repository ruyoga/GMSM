"""
MDSV Forecasting Module
Provides forecasting functionality for MDSV models
"""

import numpy as np
from typing import Optional, Dict, Tuple
from scipy.stats import norm, lognorm
import warnings


class MDSVForecaster:
    """
    Forecasting for MDSV models

    This class handles multi-step ahead forecasting for fitted MDSV models,
    including bootstrap methods for models with leverage effects.
    """

    def __init__(self, model):
        """
        Initialize forecaster with a fitted MDSV model

        Parameters
        ----------
        model : MDSV
            Fitted MDSV model instance
        """
        if not model._fitted:
            raise ValueError("Model must be fitted before forecasting")

        self.model = model
        self.params = model._param_dict_to_array(model.params_)
        self.sigma = model._compute_volatility_vector(self.params)
        self.P = model._compute_transition_matrix(self.params)

    def forecast(self, n_ahead: int,
                 last_obs: Optional[np.ndarray] = None,
                 filtered_probs: Optional[np.ndarray] = None,
                 n_simulations: int = 10000,
                 return_simulations: bool = False) -> Dict:
        """
        Generate forecasts for n_ahead periods

        Parameters
        ----------
        n_ahead : int
            Number of periods to forecast
        last_obs : np.ndarray, optional
            Last observation(s) to condition on
        filtered_probs : np.ndarray, optional
            Filtered probability distribution at last time point
        n_simulations : int
            Number of simulations for bootstrap forecasting (if leverage)
        return_simulations : bool
            Whether to return full simulation paths

        Returns
        -------
        dict
            Dictionary containing forecast results:
            - 'volatility': Expected volatility forecast
            - 'returns_var': Forecast variance of returns (if applicable)
            - 'rv': Forecast realized variance (if applicable)
            - 'quantiles': Forecast quantiles
            - 'simulations': Full simulation paths (if requested)
        """

        # Get initial state distribution
        if filtered_probs is None:
            filtered_probs = self.model._compute_stationary_dist(self.params)

        results = {}

        if self.model.leverage:
            # Use bootstrap for models with leverage
            results = self._forecast_bootstrap(
                n_ahead, filtered_probs, last_obs, n_simulations, return_simulations
            )
        else:
            # Analytical forecasting for models without leverage
            results = self._forecast_analytical(n_ahead, filtered_probs)

        return results

    def _forecast_analytical(self, n_ahead: int,
                             initial_probs: np.ndarray) -> Dict:
        """
        Analytical forecasting for models without leverage effect

        Uses the Markov chain structure to compute exact forecast distributions
        """
        forecasts = {
            'volatility': np.zeros(n_ahead),
            'state_probs': np.zeros((n_ahead, self.model.n_states))
        }

        # Propagate state probabilities forward
        current_probs = initial_probs.copy()
        for h in range(n_ahead):
            # Predict next state distribution
            current_probs = self.P.T @ current_probs
            forecasts['state_probs'][h, :] = current_probs

            # Expected volatility
            forecasts['volatility'][h] = np.sum(current_probs * self.sigma)

        # Add model-specific forecasts
        if self.model.model_type == 0:
            # Returns model - forecast variance
            forecasts['returns_var'] = forecasts['volatility']

        elif self.model.model_type == 1:
            # RV model
            shape = self.params[5]
            forecasts['rv'] = forecasts['volatility'] * np.exp(shape / 2)

        elif self.model.model_type == 2:
            # Joint model
            xi = self.params[5]
            varphi = self.params[6]
            delta2 = self.params[8]
            shape = self.params[9]

            # Expected RV (approximation)
            forecasts['returns_var'] = forecasts['volatility']
            forecasts['rv'] = np.zeros(n_ahead)

            for h in range(n_ahead):
                expected_log_v = np.sum(current_probs * np.log(self.sigma))
                forecasts['rv'][h] = np.exp(xi + varphi * expected_log_v +
                                            shape / 2 - delta2) / np.sqrt(1 - 2 * delta2)

        return forecasts

    def _forecast_bootstrap(self, n_ahead: int,
                            initial_probs: np.ndarray,
                            last_obs: Optional[np.ndarray],
                            n_simulations: int,
                            return_simulations: bool) -> Dict:
        """
        Bootstrap forecasting for models with leverage effect

        Simulates future paths accounting for leverage dynamics
        """
        # Initialize storage
        if self.model.model_type in [0, 2]:
            returns_sim = np.zeros((n_simulations, n_ahead))
        if self.model.model_type in [1, 2]:
            rv_sim = np.zeros((n_simulations, n_ahead))

        volatility_sim = np.zeros((n_simulations, n_ahead))

        # Get leverage parameters
        if self.model.leverage:
            j = 0
            if self.model.model_type == 1:
                j = 1
            elif self.model.model_type == 2:
                j = 5
            l1 = self.params[5 + j]
            theta_l = self.params[6 + j]

            # Initialize leverage history
            n_lags = 70  # As in the R code
            if last_obs is not None and len(last_obs) >= n_lags:
                leverage_history = self._compute_leverage(last_obs[-n_lags:], l1, theta_l)
            else:
                leverage_history = np.ones(n_lags)

        # Simulate paths
        for sim in range(n_simulations):
            # Sample initial state
            state = np.random.choice(self.model.n_states, p=initial_probs)

            # Storage for this simulation
            returns_path = []

            for h in range(n_ahead):
                # Get current volatility
                vol = self.sigma[state]

                # Apply leverage if applicable
                if self.model.leverage and h > 0:
                    # Update leverage based on past returns
                    if len(returns_path) > 0:
                        recent_returns = np.array(returns_path[-min(n_lags, len(returns_path)):])
                        leverage = self._compute_leverage_scalar(recent_returns, l1, theta_l)
                        vol *= leverage

                volatility_sim[sim, h] = vol

                # Generate returns
                if self.model.model_type in [0, 2]:
                    r = np.random.normal(0, np.sqrt(vol))
                    returns_sim[sim, h] = r
                    returns_path.append(r)

                # Generate RV
                if self.model.model_type == 1:
                    shape = self.params[5]
                    rv_sim[sim, h] = vol * np.random.lognormal(-shape / 2, np.sqrt(shape))

                elif self.model.model_type == 2:
                    xi = self.params[5]
                    varphi = self.params[6]
                    delta1 = self.params[7]
                    delta2 = self.params[8]
                    shape = self.params[9]

                    epsilon = r / np.sqrt(vol)
                    mu_rv = xi + varphi * np.log(vol) + delta1 * epsilon + delta2 * (epsilon ** 2 - 1)
                    rv_sim[sim, h] = np.random.lognormal(mu_rv, np.sqrt(shape))

                # Transition to next state
                trans_probs = self.P[state, :]
                state = np.random.choice(self.model.n_states, p=trans_probs)

        # Compute summary statistics
        results = {
            'volatility': np.mean(volatility_sim, axis=0),
            'volatility_std': np.std(volatility_sim, axis=0)
        }

        if self.model.model_type in [0, 2]:
            results['returns_var'] = np.mean(returns_sim ** 2, axis=0)
            results['returns_quantiles'] = np.percentile(returns_sim, [5, 25, 50, 75, 95], axis=0)

        if self.model.model_type in [1, 2]:
            results['rv'] = np.mean(rv_sim, axis=0)
            results['rv_std'] = np.std(rv_sim, axis=0)
            results['rv_quantiles'] = np.percentile(rv_sim, [5, 25, 50, 75, 95], axis=0)

        if return_simulations:
            results['simulations'] = {
                'volatility': volatility_sim
            }
            if self.model.model_type in [0, 2]:
                results['simulations']['returns'] = returns_sim
            if self.model.model_type in [1, 2]:
                results['simulations']['rv'] = rv_sim

        return results

    def _compute_leverage(self, returns: np.ndarray, l1: float, theta_l: float) -> np.ndarray:
        """
        Compute leverage effect based on past returns

        Parameters
        ----------
        returns : np.ndarray
            Past returns
        l1 : float
            Leverage intensity parameter
        theta_l : float
            Leverage decay parameter

        Returns
        -------
        np.ndarray
            Leverage multipliers
        """
        n = len(returns)
        leverage = np.ones(n)

        for t in range(1, n):
            lev_t = 1.0
            for i in range(min(t, 70)):  # Max 70 lags as in R code
                li = l1 * (theta_l ** i)
                if returns[t - i - 1] < 0:
                    lev_t *= (1 + li * abs(returns[t - i - 1]) / np.sqrt(leverage[t - i - 1]))
            leverage[t] = lev_t

        return leverage

    def _compute_leverage_scalar(self, recent_returns: np.ndarray,
                                 l1: float, theta_l: float) -> float:
        """
        Compute single leverage value based on recent returns
        """
        leverage = 1.0
        n = len(recent_returns)

        for i in range(min(n, 70)):
            if i < n and recent_returns[-(i + 1)] < 0:
                li = l1 * (theta_l ** i)
                leverage *= (1 + li * abs(recent_returns[-(i + 1)]))

        return leverage

    def compute_value_at_risk(self, n_ahead: int = 1,
                              alpha: float = 0.05,
                              n_simulations: int = 10000) -> np.ndarray:
        """
        Compute Value at Risk for returns

        Parameters
        ----------
        n_ahead : int
            Forecast horizon
        alpha : float
            VaR level (e.g., 0.05 for 95% VaR)
        n_simulations : int
            Number of simulations

        Returns
        -------
        np.ndarray
            VaR values for each horizon
        """
        if self.model.model_type not in [0, 2]:
            raise ValueError("VaR only available for models with returns")

        # Get forecast with simulations
        forecast = self.forecast(n_ahead, n_simulations=n_simulations,
                                 return_simulations=True)

        if 'simulations' in forecast and 'returns' in forecast['simulations']:
            returns_sim = forecast['simulations']['returns']
            var = np.percentile(returns_sim, alpha * 100, axis=0)
        else:
            # Analytical approximation using normal distribution
            volatility = forecast['volatility']
            var = norm.ppf(alpha) * np.sqrt(volatility)

        return var