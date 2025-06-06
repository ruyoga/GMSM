import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass
import warnings
from gmsm.modules.visualization.mdsv import plot_mdsv_fit, plot_mdsv_filter
from gmsm.modules.networks.mdsv import MDSVProcess

try:
    from gmsm.modules.networks.cpp import mdsv
except ImportError:
    mdsv_cpp = None


@dataclass
class MDSVResult:
    """Base class for MDSV results"""
    model_type: str
    leverage: bool
    N: int
    K: int
    data: np.ndarray
    dates: Optional[pd.DatetimeIndex] = None


class MDSVFit(MDSVResult):
    """
    Results from MDSV model fitting

    Attributes
    ----------
    estimates : np.ndarray
        Estimated parameters
    log_likelihood : float
        Log-likelihood value
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    convergence : int
        Convergence status (0 if converged)
    """

    def __init__(self, model_type: str, leverage: bool, N: int, K: int,
                 data: np.ndarray, estimates: np.ndarray, log_likelihood: float,
                 convergence: int, dates: Optional[pd.DatetimeIndex] = None):
        super().__init__(model_type, leverage, N, K, data, dates)
        self.estimates = estimates
        self.log_likelihood = log_likelihood
        self.convergence = convergence

        # Calculate information criteria
        n_params = len(estimates)
        n_obs = len(data)
        self.aic = -log_likelihood - n_params
        self.bic = -log_likelihood - 0.5 * n_params * np.log(n_obs)

    def summary(self):
        """Print summary of fitting results"""
        print("=" * 49)
        print("================= MDSV fitting ==================")
        print("=" * 49)
        print(f"Model   : MDSV({self.N},{self.K})")
        print(f"Data    : {self.model_type}")
        print(f"Leverage: {self.leverage}")
        print()
        print("Optimal Parameters")
        print("-" * 49)

        convergence_msg = "Convergence." if self.convergence == 0 else "No Convergence. Return the best result."
        print(f"Convergence : {convergence_msg}")

        # Parameter names based on model type
        param_names = self._get_parameter_names()
        for name, value in zip(param_names, self.estimates):
            print(f"{name:8s}: {value:10.6f}")

        print()
        print(f"LogLikelihood : {self.log_likelihood:10.2f}")
        print()
        print("Information Criteria")
        print("-" * 49)
        print(f"AIC : {self.aic:10.2f}")
        print(f"BIC : {self.bic:10.2f}")

    def _get_parameter_names(self) -> List[str]:
        """Get parameter names based on model configuration"""
        names = ["omega", "a", "b", "sigma", "v0"]

        if self.model_type == "Univariate realized variances":
            names.append("shape")
        elif self.model_type == "Joint log-return and realized variances":
            names.extend(["xi", "varphi", "delta1", "delta2", "shape"])

        if self.leverage:
            names.extend(["l", "theta"])

        return names

    def plot(self, plot_type: Union[str, List[str]] = ["dis", "nic"], **kwargs):
        """
        Plot results

        Parameters
        ----------
        plot_type : str or list of str
            Type of plot: "dis" for stationary distribution, "nic" for news impact curve
        """
        plot_mdsv_fit(self, plot_type, **kwargs)


class MDSVFilter(MDSVResult):
    """
    Results from MDSV filtering

    Attributes
    ----------
    estimates : np.ndarray
        Input parameters
    log_likelihood : float
        Log-likelihood value
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    leverage_values : np.ndarray
        Leverage effect at each time point
    filtered_proba : np.ndarray
        Filtered probabilities
    smoothed_proba : np.ndarray
        Smoothed probabilities
    marg_loglik : Optional[float]
        Marginal log-likelihood (for joint models)
    var_values : Optional[Dict[str, float]]
        Value-at-Risk estimates
    """

    def __init__(self, model_type: str, leverage: bool, N: int, K: int,
                 data: np.ndarray, estimates: np.ndarray, log_likelihood: float,
                 leverage_values: np.ndarray, filtered_proba: np.ndarray,
                 smoothed_proba: np.ndarray, dates: Optional[pd.DatetimeIndex] = None,
                 marg_loglik: Optional[float] = None, var_values: Optional[Dict[str, float]] = None):
        super().__init__(model_type, leverage, N, K, data, dates)
        self.estimates = estimates
        self.log_likelihood = log_likelihood
        self.leverage_values = leverage_values
        self.filtered_proba = filtered_proba
        self.smoothed_proba = smoothed_proba
        self.marg_loglik = marg_loglik
        self.var_values = var_values or {}

        # Calculate information criteria
        n_params = len(estimates)
        n_obs = len(data)
        self.aic = -log_likelihood - n_params
        self.bic = -log_likelihood - 0.5 * n_params * np.log(n_obs)

    def get_filtered_volatility(self, process: MDSVProcess) -> np.ndarray:
        """Get filtered volatility path"""
        vol_vector = process.volatility_vector()

        # Get most likely state at each time
        state_sequence = np.argmax(self.smoothed_proba, axis=0)

        # Extract volatilities
        filtered_vol = np.sqrt(vol_vector[state_sequence])

        if self.leverage:
            filtered_vol *= np.sqrt(self.leverage_values)

        return filtered_vol

    def summary(self):
        """Print summary of filtering results"""
        print("=" * 49)
        print("================ MDSV Filtering =================")
        print("=" * 49)
        print(f"Model   : MDSV({self.N},{self.K})")
        print(f"Data    : {self.model_type}")
        print(f"Leverage: {self.leverage}")
        print()
        print("Optimal Parameters")
        print("-" * 49)

        # Parameter names
        param_names = self._get_parameter_names()
        for name, value in zip(param_names, self.estimates):
            print(f"{name:8s}: {value:10.6f}")

        print()
        print(f"LogLikelihood : {self.log_likelihood:10.2f}")

        if self.marg_loglik is not None:
            print(f"Marginal LogLikelihood : {self.marg_loglik:10.2f}")

        print()
        print("Information Criteria")
        print("-" * 49)
        print(f"AIC : {self.aic:10.2f}")
        print(f"BIC : {self.bic:10.2f}")

        if self.var_values:
            print()
            print("Value at Risk")
            print("-" * 49)
            for alpha, var in sorted(self.var_values.items()):
                print(f"{int((1 - alpha) * 100)}%  : {var:10.6f}")

    def _get_parameter_names(self) -> List[str]:
        """Get parameter names based on model configuration"""
        names = ["omega", "a", "b", "sigma", "v0"]

        if self.model_type == "Univariate realized variances":
            names.append("shape")
        elif self.model_type == "Joint log-return and realized variances":
            names.extend(["xi", "varphi", "delta1", "delta2", "shape"])

        if self.leverage:
            names.extend(["l", "theta"])

        return names

    def plot(self, **kwargs):
        """Plot filtered results"""
        plot_mdsv_filter(self, **kwargs)


class MDSVBoot(MDSVResult):
    """
    Results from MDSV bootstrap forecasting

    Attributes
    ----------
    estimates : np.ndarray
        Model parameters
    log_likelihood : float
        Log-likelihood value
    n_ahead : int
        Forecast horizon
    n_bootpred : int
        Number of bootstrap predictions
    rt_sim : Optional[np.ndarray]
        Simulated returns (if applicable)
    rt2 : Optional[np.ndarray]
        Forecasted squared returns
    rvt_sim : Optional[np.ndarray]
        Simulated realized variances (if applicable)
    rvt : Optional[np.ndarray]
        Forecasted realized variances
    """

    def __init__(self, model_type: str, leverage: bool, N: int, K: int,
                 data: np.ndarray, estimates: np.ndarray, log_likelihood: float,
                 n_ahead: int, n_bootpred: int, dates: Optional[pd.DatetimeIndex] = None,
                 rt_sim: Optional[np.ndarray] = None, rt2: Optional[np.ndarray] = None,
                 rvt_sim: Optional[np.ndarray] = None, rvt: Optional[np.ndarray] = None):
        super().__init__(model_type, leverage, N, K, data, dates)
        self.estimates = estimates
        self.log_likelihood = log_likelihood
        self.n_ahead = n_ahead
        self.n_bootpred = n_bootpred
        self.rt_sim = rt_sim
        self.rt2 = rt2
        self.rvt_sim = rvt_sim
        self.rvt = rvt

        # Calculate AIC/BIC
        n_params = len(estimates)
        n_obs = len(data)
        self.aic = -log_likelihood - n_params
        self.bic = -log_likelihood - 0.5 * n_params * np.log(n_obs)

    def summary(self, max_display: int = 10):
        """Print summary of bootstrap results"""
        print("=" * 49)
        print("========== MDSV Bootstrap Forecasting ===========")
        print("=" * 49)
        print(f"Model       : MDSV({self.N},{self.K})")
        print(f"Data        : {self.model_type}")
        print(f"Leverage    : {self.leverage}")
        print(f"n.ahead     : {self.n_ahead}")

        if self.dates is not None and len(self.dates) > 0:
            print(f"Date (T[0]) : {self.dates[-1]}")

        print()

        # Display forecasts
        if self.leverage and self.rt_sim is not None:
            # With leverage, show distribution summaries
            self._print_forecast_summary("Log-returns", self.rt_sim, max_display)

            if self.model_type == "Univariate log-return":
                rt2_sim = self.rt_sim ** 2
                self._print_forecast_summary("Sigma", np.sqrt(rt2_sim), max_display)

            if self.rvt_sim is not None:
                self._print_forecast_summary("Realized Variances", self.rvt_sim, max_display)
        else:
            # Without leverage, show point forecasts
            if self.rt2 is not None:
                self._print_point_forecast("Sigma", np.sqrt(self.rt2), max_display)

            if self.rvt is not None:
                self._print_point_forecast("Realized Variances", self.rvt, max_display)

    def _print_forecast_summary(self, title: str, data: np.ndarray, max_display: int):
        """Print summary statistics for forecast distribution"""
        print(f"{title} (summary) :")

        # Calculate quantiles
        quantiles = np.percentile(data, [0, 25, 50, 75, 100], axis=0)
        means = np.mean(data, axis=0)

        # Create summary dataframe
        summary = pd.DataFrame({
            'min': quantiles[0],
            'q.25': quantiles[1],
            'mean': means,
            'median': quantiles[2],
            'q.75': quantiles[3],
            'max': quantiles[4]
        }, index=[f"t+{i + 1}" for i in range(len(means))])

        # Print first few rows
        print(summary.head(min(max_display, len(summary))).round(6))
        if len(summary) > max_display:
            print(".........................")

    def _print_point_forecast(self, title: str, data: np.ndarray, max_display: int):
        """Print point forecasts"""
        print(f"{title} (summary) :")

        # Create series
        forecast = pd.Series(data, index=[f"t+{i + 1}" for i in range(len(data))])

        # Print first few values
        print(forecast.head(min(max_display, len(forecast))).round(6))
        if len(forecast) > max_display:
            print(".........................")


class MDSVRoll(MDSVResult):
    """
    Results from MDSV rolling estimation and forecasting

    Attributes
    ----------
    n_ahead : int
        Forecast horizon
    forecast_length : int
        Total length of forecast period
    refit_every : int
        Refit frequency
    refit_window : str
        "recursive" or "moving"
    window_size : Optional[int]
        Size of moving window
    calculate_var : bool
        Whether VaR was calculated
    var_alpha : np.ndarray
        VaR significance levels
    estimates : pd.DataFrame
        Parameter estimates at each time
    prevision : pd.DataFrame
        Forecast results
    """

    def __init__(self, model_type: str, leverage: bool, N: int, K: int,
                 data: np.ndarray, n_ahead: int, forecast_length: int,
                 refit_every: int, refit_window: str, window_size: Optional[int],
                 calculate_var: bool, var_alpha: np.ndarray,
                 estimates: pd.DataFrame, prevision: pd.DataFrame,
                 dates: Optional[pd.DatetimeIndex] = None):
        super().__init__(model_type, leverage, N, K, data, dates)
        self.n_ahead = n_ahead
        self.forecast_length = forecast_length
        self.refit_every = refit_every
        self.refit_window = refit_window
        self.window_size = window_size
        self.calculate_var = calculate_var
        self.var_alpha = var_alpha
        self.estimates = estimates
        self.prevision = prevision

    def summary(self, var_test: bool = True, loss_horizon: List[int] = None,
                loss_window: int = 756):
        """
        Print summary of rolling results

        Parameters
        ----------
        var_test : bool
            Whether to perform VaR backtesting
        loss_horizon : List[int]
            Horizons for loss function calculation
        loss_window : int
            Window for loss calculation
        """
        if loss_horizon is None:
            loss_horizon = [1, 5, 10, 25, 50, 75, 100]

        # Calculate summary statistics
        from .estimation import calculate_loss_functions, perform_var_tests

        print("=" * 49)
        print("=== MDSV Rolling Estimation and Forecasting =====")
        print("=" * 49)
        print(f"Model              : MDSV({self.N},{self.K})")
        print(f"Data               : {self.model_type}")
        print(f"Leverage           : {self.leverage}")
        print(f"No.refit           : {len(self.estimates) // self.refit_every}")
        print(f"Refit Horizon      : {self.refit_every}")
        print(f"No.Forecasts       : {self.forecast_length}")
        print(f"n.ahead            : {self.n_ahead}")

        if self.dates is not None:
            print(f"Date (T[0])        : {self.dates[-self.forecast_length]}")

        print()
        print("Forecasting performances")
        print("-" * 49)

        # Predictive density
        pred_lik = self.estimates['predict_loglik'].iloc[-loss_window:].sum()
        print(f"Predictive density : {pred_lik:.2f}")
        print("--------------------")
        print()

        # Calculate and display loss functions
        loss_results = calculate_loss_functions(self, loss_horizon, loss_window)

        for metric_type, results in loss_results.items():
            print(f"{metric_type}")
            print("-" * 28)
            print(results.round(3))
            print()

        # VaR tests if applicable
        if var_test and self.calculate_var and self.model_type != "Univariate realized variances":
            print("\nVaR Tests")
            var_test_results = perform_var_tests(self)
            for alpha, tests in var_test_results.items():
                print("-" * 49)
                print(f"alpha              : {alpha:.2%}")
                print(f"Expected Exceed    : {tests['expected']:.1f}")
                print(f"Actual VaR Exceed  : {tests['actual']}")
                print(f"Actual %           : {tests['actual_pct']:.2%}")
                print()

                # Unconditional coverage test
                print("Unconditional Coverage (Kupiec)")
                print("Null-Hypothesis    : Correct exceedances")
                print(f"LR.uc Statistic    : {tests['lr_uc']:.3f}")
                print(f"LR.uc Critical     : {tests['lr_uc_crit']:.3f}")
                print(f"LR.uc p-value      : {tests['p_uc']:.3f}")
                print(f"Reject Null        : {tests['reject_uc']}")
                print()

                # Independence test
                print("Independence (Christoffersen)")
                print("Null-Hypothesis    : Independence of failures")
                print(f"LR.ind Statistic   : {tests['lr_ind']:.3f}")
                print(f"LR.ind Critical    : {tests['lr_ind_crit']:.3f}")
                print(f"LR.ind p-value     : {tests['p_ind']:.3f}")
                print(f"Reject Null        : {tests['reject_ind']}")
                print()

                # Conditional coverage test
                print("Conditional Coverage (Christoffersen)")
                print("Null-Hypothesis    : Correct exceedances and Independence of failures")
                print(f"LR.cc Statistic    : {tests['lr_cc']:.3f}")
                print(f"LR.cc Critical     : {tests['lr_cc_crit']:.3f}")
                print(f"LR.cc p-value      : {tests['p_cc']:.3f}")
                print(f"Reject Null        : {tests['reject_cc']}")
                print()

    def plot(self, plot_type: Union[str, List[str]] = ["sigma", "VaR", "dens"], **kwargs):
        """
        Plot rolling results

        Parameters
        ----------
        plot_type : str or list of str
            Types of plots: "sigma", "VaR", "dens"
        """
        from .visualization import plot_mdsv_roll
        plot_mdsv_roll(self, plot_type, **kwargs)


class MDSVSim(MDSVResult):
    """
    Results from MDSV simulation

    Attributes
    ----------
    parameters : np.ndarray
        Parameters used for simulation
    n_sim : int
        Simulation horizon
    n_start : int
        Burn-in samples
    m_sim : int
        Number of simulations
    simulations : Dict[int, Dict[str, np.ndarray]]
        Simulated data for each simulation run
    """

    def __init__(self, model_type: str, leverage: bool, N: int, K: int,
                 parameters: np.ndarray, n_sim: int, n_start: int, m_sim: int,
                 simulations: Dict[int, Dict[str, np.ndarray]]):
        # Create dummy data array for base class
        super().__init__(model_type, leverage, N, K, np.array([]), None)
        self.parameters = parameters
        self.n_sim = n_sim
        self.n_start = n_start
        self.m_sim = m_sim
        self.simulations = simulations

    def get_simulation(self, sim_num: int = 1) -> Dict[str, np.ndarray]:
        """
        Get specific simulation results

        Parameters
        ----------
        sim_num : int
            Simulation number (1 to m_sim)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'r_t' and/or 'RV_t' arrays
        """
        if sim_num < 1 or sim_num > self.m_sim:
            raise ValueError(f"sim_num must be between 1 and {self.m_sim}")
        return self.simulations[sim_num - 1]

    def plot(self, sim_num: int = 1, **kwargs):
        """Plot simulation results"""
        from .visualization import plot_mdsv_sim
        plot_mdsv_sim(self, sim_num, **kwargs)