"""
MDSV Forecasting Module - CORRECTED to match Augustyniak et al. (2024) Equation 22
Provides forecasting functionality for MDSV models
"""

import numpy as np
from typing import Optional, Dict, Tuple
from scipy.stats import norm, lognorm
import warnings


class MDSVForecaster:
    """
    Forecasting for MDSV models

    CRITICAL FIX: Implements Equation 22 from Augustyniak et al. (2024) correctly
    with leverage effect L_t+1^φ multiplier
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
        self.sigma = model._compute_volatility_vector(self.params)  # Base volatility states (V_t, not V_t^(L))
        self.P = model._compute_transition_matrix(self.params)

    def forecast(self, n_ahead: int,
                 last_obs: Optional[np.ndarray] = None,
                 filtered_probs: Optional[np.ndarray] = None,
                 n_simulations: int = 10000,
                 return_simulations: bool = False,
                 return_history: Optional[np.ndarray] = None) -> Dict:
        """
        Generate forecasts for n_ahead periods using CORRECT Equation 22

        Parameters
        ----------
        n_ahead : int
            Number of periods to forecast
        last_obs : np.ndarray, optional
            Last observation(s) to condition on
        filtered_probs : np.ndarray, optional
            Filtered probability distribution at last time point
        n_simulations : int
            Number of simulations for bootstrap forecasting (multi-step)
        return_simulations : bool
            Whether to return full simulation paths
        return_history : np.ndarray, optional
            Historical returns for computing leverage effect (at least 70 lags)

        Returns
        -------
        dict
            Dictionary containing forecast results
        """
        # Get initial state distribution
        if filtered_probs is None:
            filtered_probs = self.model._compute_stationary_dist(self.params)

        # For models with leverage, we need return history
        if self.model.leverage and return_history is None:
            warnings.warn("Leverage model requires return_history for accurate forecasts. "
                         "Using L_t = 1.0 (no leverage effect)")

        # Use analytical formula for one-step forecast (Equation 22)
        if n_ahead == 1:
            results = self._forecast_one_step_analytical(
                filtered_probs, return_history
            )
        else:
            # Multi-step: use simulations
            results = self._forecast_bootstrap(
                n_ahead, filtered_probs, last_obs, n_simulations,
                return_simulations, return_history
            )

        return results

    def _forecast_one_step_analytical(self, filtered_probs: np.ndarray,
                                     return_history: Optional[np.ndarray] = None) -> Dict:
        """
        CRITICAL FIX: Implement Equation 22 from Augustyniak et al. (2024) EXACTLY

        Equation 22:
        RV^_t+1 = L_{t+1}^φ * exp(ξ + δ₁²/(2(1-2δ₂)) - δ₂ + γ²/2) / sqrt(1-2δ₂)
                  * Σⱼ p(V_{t+1} = υⱼ | Y₁:ₜ) * υⱼ^φ
        """
        results = {}

        # Propagate state probabilities one step ahead
        next_probs = self.P.T @ filtered_probs

        if self.model.model_type == 0:
            # Returns only model
            expected_vol = np.sum(next_probs * self.sigma)
            results['volatility'] = expected_vol
            results['returns_var'] = expected_vol

        elif self.model.model_type == 1:
            # RV only model
            shape = self.params[5]
            expected_vol = np.sum(next_probs * self.sigma)
            results['volatility'] = expected_vol

            # For RV-only, no leverage, simpler formula
            results['rv'] = expected_vol * np.exp(shape / 2)

        elif self.model.model_type == 2:
            # Joint model - USE EQUATION 22 EXACTLY
            xi = self.params[5]
            varphi = self.params[6]
            delta1 = self.params[7]
            delta2 = self.params[8]
            shape = self.params[9]  # gamma in paper

            # Get leverage parameters
            if self.model.leverage:
                l1 = self.params[10]
                theta_l = self.params[11]

                # Compute L_{t+1} from return history
                if return_history is not None and len(return_history) > 0:
                    L_t_plus_1 = self._compute_leverage_forward(
                        return_history, l1, theta_l
                    )
                else:
                    L_t_plus_1 = 1.0
                    warnings.warn("No return history provided, using L_t+1 = 1.0")
            else:
                L_t_plus_1 = 1.0

            # CRITICAL: Compute Σⱼ p_j * υⱼ^φ (NOT exp(φ * E[log(V)]))
            sum_prob_vol_phi = np.sum(next_probs * np.power(self.sigma, varphi))

            # Compute constant term from Equation 22
            const_term = np.exp(
                xi +
                delta1**2 / (2 * (1 - 2*delta2)) -
                delta2 +
                shape**2 / 2
            ) / np.sqrt(1 - 2*delta2)

            # EQUATION 22: Complete formula
            rv_forecast = (L_t_plus_1 ** varphi) * const_term * sum_prob_vol_phi

            results['rv'] = rv_forecast
            results['volatility'] = np.sum(next_probs * self.sigma)
            results['returns_var'] = results['volatility'] * L_t_plus_1
            results['leverage_multiplier'] = L_t_plus_1
            results['state_probs'] = next_probs

        return results

    def _forecast_bootstrap(self, n_ahead: int,
                           initial_probs: np.ndarray,
                           last_obs: Optional[np.ndarray],
                           n_simulations: int,
                           return_simulations: bool,
                           return_history: Optional[np.ndarray]) -> Dict:
        """
        Bootstrap forecasting for multi-step forecasts with leverage effect
        """
        # Initialize storage
        if self.model.model_type in [0, 2]:
            returns_sim = np.zeros((n_simulations, n_ahead))
        if self.model.model_type in [1, 2]:
            rv_sim = np.zeros((n_simulations, n_ahead))

        volatility_sim = np.zeros((n_simulations, n_ahead))
        leverage_sim = np.zeros((n_simulations, n_ahead))

        # Get leverage parameters
        if self.model.leverage:
            if self.model.model_type == 1:
                l1 = self.params[6]
                theta_l = self.params[7]
            elif self.model.model_type == 2:
                l1 = self.params[10]
                theta_l = self.params[11]
            else:
                l1 = self.params[5]
                theta_l = self.params[6]

        # Simulate paths
        for sim in range(n_simulations):
            # Sample initial state
            state = np.random.choice(self.model.n_states, p=initial_probs)

            # Initialize return history for leverage
            if self.model.leverage and return_history is not None:
                returns_history = list(return_history[-70:])  # Keep last 70
            else:
                returns_history = []

            for h in range(n_ahead):
                # Get current base volatility (without leverage)
                vol = self.sigma[state]

                # Compute leverage multiplier from history
                if self.model.leverage and len(returns_history) > 0:
                    L_t = self._compute_leverage_forward(
                        np.array(returns_history), l1, theta_l
                    )
                else:
                    L_t = 1.0

                # Apply leverage to get V^(L)_t = V_t * L_t
                vol_leveraged = vol * L_t
                leverage_sim[sim, h] = L_t
                volatility_sim[sim, h] = vol_leveraged

                # Generate returns if needed
                if self.model.model_type in [0, 2]:
                    r = np.random.normal(0, np.sqrt(vol_leveraged))
                    returns_sim[sim, h] = r
                    returns_history.append(r)
                    if len(returns_history) > 70:
                        returns_history.pop(0)

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

                    if self.model.model_type in [0, 2]:
                        epsilon = r / np.sqrt(vol_leveraged)
                    else:
                        epsilon = 0

                    # Use leveraged volatility in log(RV) equation
                    mu_rv = xi + varphi * np.log(vol_leveraged) + delta1 * epsilon + delta2 * (epsilon ** 2 - 1)
                    rv_sim[sim, h] = np.random.lognormal(mu_rv, np.sqrt(shape))

                # Transition to next state
                trans_probs = self.P[state, :]
                state = np.random.choice(self.model.n_states, p=trans_probs)

        # Compute summary statistics
        results = {
            'volatility': np.mean(volatility_sim, axis=0),
            'volatility_std': np.std(volatility_sim, axis=0),
            'leverage_multiplier': np.mean(leverage_sim, axis=0)
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
                'volatility': volatility_sim,
                'leverage': leverage_sim
            }
            if self.model.model_type in [0, 2]:
                results['simulations']['returns'] = returns_sim
            if self.model.model_type in [1, 2]:
                results['simulations']['rv'] = rv_sim

        return results

    def _compute_leverage_forward(self, returns: np.ndarray,
                                  l1: float, theta_l: float,
                                  n_lags: int = 70) -> float:
        """
        Compute forward leverage multiplier L_{t+1} from past returns

        CORRECT implementation of Equation 19:
        L_t = ∏ᵢ₌₁^NL [1 + lᵢ * |r_{t-i}| / √(L_{t-i})] * 1{r_{t-i} < 0}
        where lᵢ = θₗ^(i-1) * l₁

        For forecasting t+1, we use returns up to time t
        """
        if len(returns) == 0:
            return 1.0

        n = len(returns)
        max_lags = min(n, n_lags)

        # We need to compute L values iteratively for accuracy
        L_history = np.ones(n + 1)  # L_0, L_1, ..., L_n

        for t in range(1, n + 1):
            L_t = 1.0
            for i in range(1, min(t, n_lags) + 1):
                if returns[t - i] < 0:
                    li = l1 * (theta_l ** (i - 1))
                    L_t *= (1.0 + li * abs(returns[t - i]) / np.sqrt(L_history[t - i]))
            L_history[t] = L_t

        # Return the last computed leverage (which is L_t, used for forecasting t+1)
        return L_history[-1]

    def _compute_leverage_scalar(self, recent_returns: np.ndarray,
                                 l1: float, theta_l: float) -> float:
        """
        Simplified leverage computation for recent returns only
        """
        return self._compute_leverage_forward(recent_returns, l1, theta_l)

    def compute_value_at_risk(self, n_ahead: int = 1,
                              alpha: float = 0.05,
                              n_simulations: int = 10000,
                              return_history: Optional[np.ndarray] = None) -> np.ndarray:
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
        return_history : np.ndarray, optional
            Historical returns for leverage computation

        Returns
        -------
        np.ndarray
            VaR values for each horizon
        """
        if self.model.model_type not in [0, 2]:
            raise ValueError("VaR only available for models with returns")

        # Get forecast with simulations
        forecast = self.forecast(
            n_ahead,
            n_simulations=n_simulations,
            return_simulations=True,
            return_history=return_history
        )

        if 'simulations' in forecast and 'returns' in forecast['simulations']:
            returns_sim = forecast['simulations']['returns']
            var = np.percentile(returns_sim, alpha * 100, axis=0)
        else:
            # Analytical approximation
            volatility = forecast['volatility']
            var = norm.ppf(alpha) * np.sqrt(volatility)

        return var