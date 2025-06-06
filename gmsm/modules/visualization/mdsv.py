import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Optional
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from gmsm.modules.models.mdsv import MDSVFit, MDSVFilter, MDSVRoll, MDSVSim
from .core import MDSVProcess


def plot_mdsv_fit(fit: MDSVFit, plot_type: Union[str, List[str]] = ["dis", "nic"],
                  figsize: tuple = (12, 5), **kwargs):
    """
    Plot MDSV fit results

    Parameters
    ----------
    fit : MDSVFit
        Fitted model
    plot_type : str or list
        "dis" for stationary distribution, "nic" for news impact curve
    figsize : tuple
        Figure size
    """
    if isinstance(plot_type, str):
        plot_type = [plot_type]

    n_plots = len(plot_type)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Create process object
    process = MDSVProcess(fit.N, fit.K)
    process.update_from_vector(fit.estimates[:5])

    for i, ptype in enumerate(plot_type):
        ax = axes[i]

        if ptype == "dis":
            # Stationary distribution plot
            vol_vector = np.sqrt(process.volatility_vector())
            prob_vector = process.stationary_distribution()

            # Aggregate by unique volatility values
            unique_vols = np.unique(np.round(vol_vector, 4))
            aggregated_probs = []

            for vol in unique_vols:
                mask = np.abs(vol_vector - vol) < 1e-4
                aggregated_probs.append(np.sum(prob_vector[mask]))

            ax.plot(unique_vols, aggregated_probs, 'b-', linewidth=2)
            ax.set_xlabel('Volatilities')
            ax.set_ylabel('Probabilities')
            ax.set_title('Density plot: Stationary\ndistribution of the volatilities')
            ax.grid(True, alpha=0.3)

        elif ptype == "nic" and fit.model_type != "Univariate realized variances":
            # News Impact Curve
            # Need to filter the data first to get state probabilities
            from .estimation import filter_mdsv

            model_type_map = {
                "Univariate log-return": 0,
                "Joint log-return and realized variances": 2
            }
            model_type = model_type_map[fit.model_type]

            filter_result = filter_mdsv(fit.data, fit.estimates, fit.N, fit.K,
                                        model_type, fit.leverage, False)

            # Get smoothed volatilities
            vol_vector = process.volatility_vector()
            smoothed_probs = filter_result.smoothed_proba

            n_obs = fit.data.shape[0]
            V_t = np.zeros(n_obs)
            for t in range(n_obs):
                state_idx = np.argmax(smoothed_probs[:, t])
                V_t[t] = vol_vector[state_idx]

            if fit.leverage:
                V_t *= filter_result.leverage_values

            V_t = np.sqrt(V_t)

            # Create NIC using lowess
            returns = fit.data[:-1, 0]
            next_vol = V_t[1:]

            # Sort by returns for plotting
            sorted_idx = np.argsort(returns)

            # Apply lowess smoothing
            smoothed = lowess(next_vol[sorted_idx], returns[sorted_idx],
                              frac=0.2, return_sorted=False)

            ax.plot(returns[sorted_idx], smoothed, 'b-', linewidth=2)
            ax.set_xlabel('log-returns (rt-1)')
            ax.set_ylabel('volatilities (Vt)')
            ax.set_title('News Impact Curve')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_mdsv_filter(filter_result: MDSVFilter, figsize: tuple = (12, 8), **kwargs):
    """
    Plot MDSV filtering results

    Parameters
    ----------
    filter_result : MDSVFilter
        Filtering results
    figsize : tuple
        Figure size
    """
    model_type_map = {
        "Univariate log-return": 0,
        "Univariate realized variances": 1,
        "Joint log-return and realized variances": 2
    }
    model_type = model_type_map[filter_result.model_type]

    # Create process to get filtered volatility
    process = MDSVProcess(filter_result.N, filter_result.K)
    process.update_from_vector(filter_result.estimates[:5])

    V_t = filter_result.get_filtered_volatility(process)

    # Setup subplots based on model type
    if model_type == 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [1, 1]})
    elif model_type == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [1, 1]})
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize,
                                            gridspec_kw={'height_ratios': [1.15, 1, 1.15]})

    # Use dates if available
    x_data = filter_result.dates if filter_result.dates is not None else np.arange(len(V_t))

    # Plot filtered volatilities
    ax1.plot(x_data, V_t, 'b-', linewidth=1)
    ax1.set_ylabel('Filtered Volatilities')
    ax1.set_title('Filtered Volatilities')
    ax1.grid(True, alpha=0.3)
    if filter_result.dates is None:
        ax1.set_xlabel('Time')
    else:
        ax1.set_xticklabels([])

    # Plot data based on model type
    if model_type == 0:
        # Log-returns
        ax2.plot(x_data, filter_result.data[:, 0], 'k-', linewidth=0.5)
        ax2.set_ylabel('Log-returns')
        ax2.set_xlabel('Date' if filter_result.dates is not None else 'Time')
        ax2.grid(True, alpha=0.3)

    elif model_type == 1:
        # Realized volatilities
        ax2.plot(x_data, np.sqrt(filter_result.data[:, 1]), 'k-', linewidth=0.5)
        ax2.set_ylabel('Realized Volatilities')
        ax2.set_xlabel('Date' if filter_result.dates is not None else 'Time')
        ax2.grid(True, alpha=0.3)

    else:
        # Joint model
        # RV
        ax2.plot(x_data, np.sqrt(filter_result.data[:, 1]), 'k-', linewidth=0.5)
        ax2.set_ylabel('Realized Volatilities')
        ax2.grid(True, alpha=0.3)
        if filter_result.dates is not None:
            ax2.set_xticklabels([])

        # Returns
        ax3.plot(x_data, filter_result.data[:, 0], 'k-', linewidth=0.5)
        ax3.set_ylabel('Log-returns')
        ax3.set_xlabel('Date' if filter_result.dates is not None else 'Time')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_mdsv_roll(roll_result: MDSVRoll,
                   plot_type: Union[str, List[str]] = ["sigma", "VaR", "dens"],
                   figsize: tuple = (12, 5), **kwargs):
    """
    Plot rolling forecast results

    Parameters
    ----------
    roll_result : MDSVRoll
        Rolling forecast results
    plot_type : str or list
        Types of plots: "sigma", "VaR", "dens"
    figsize : tuple
        Figure size per plot
    """
    if isinstance(plot_type, str):
        plot_type = [plot_type]

    # Filter valid plot types
    valid_types = []
    for ptype in plot_type:
        if ptype == "VaR" and roll_result.model_type == "Univariate realized variances":
            print("Warning: VaR plot not available for RV-only model")
            continue
        valid_types.append(ptype)

    if not valid_types:
        return

    # Create plots
    for ptype in valid_types:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if ptype == "sigma":
            # Plot realized vs predicted volatility
            dates = roll_result.estimates['date'].values

            if roll_result.model_type != "Univariate realized variances":
                # Plot squared returns
                realized = roll_result.estimates['rt'].values ** 2

                # Get 1-ahead forecasts
                if 'rt2p1' in roll_result.prevision.columns:
                    predicted = roll_result.prevision['rt2p1'].values

                    ax.plot(dates, realized, 'gray', linewidth=0.5,
                            label='Realized values', alpha=0.7)
                    ax.plot(dates, predicted, 'b-', linewidth=1.5,
                            label='1-ahead forecast')

                    ax.set_ylabel('Squared Returns')
                    ax.set_title('Log-returns square: 1-ahead forecast vs realized values')

            if roll_result.model_type != "Univariate log-return":
                # Plot RV
                realized = roll_result.estimates['rvt'].values

                if 'rvtp1' in roll_result.prevision.columns:
                    predicted = roll_result.prevision['rvtp1'].values

                    ax.clear()  # Clear if we're switching from returns to RV
                    ax.plot(dates, realized, 'gray', linewidth=0.5,
                            label='Realized values', alpha=0.7)
                    ax.plot(dates, predicted, 'b-', linewidth=1.5,
                            label='1-ahead forecast')

                    ax.set_ylabel('Realized Variance')
                    ax.set_title('Realized Variances: 1-ahead forecast vs realized values')

            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True, alpha=0.3)

        elif ptype == "VaR" and roll_result.calculate_var:
            # VaR exceedance plot
            for alpha in roll_result.var_alpha:
                var_col = f'VaR{int(100 * (1 - alpha))}'
                viol_col = f'I{int(100 * (1 - alpha))}'

                if var_col in roll_result.estimates.columns:
                    dates = roll_result.estimates['date'].values
                    returns = roll_result.estimates['rt'].values
                    var_values = roll_result.estimates[var_col].values
                    violations = roll_result.estimates[viol_col].values

                    # Plot returns
                    ax.scatter(dates[~violations], returns[~violations],
                               color='gray', s=10, alpha=0.5, label='Returns')

                    # Plot VaR line
                    ax.plot(dates, var_values, 'k-', linewidth=1.5, label='VaR')

                    # Highlight violations
                    ax.scatter(dates[violations], returns[violations],
                               color='red', s=20, label='VaR violations')

                    ax.set_xlabel('Date')
                    ax.set_ylabel('Returns')
                    ax.set_title(f'Log-returns and Value-at-Risk Exceedances (alpha = {alpha})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

                    # Create new figure for next alpha
                    if alpha != roll_result.var_alpha[-1]:
                        fig, ax = plt.subplots(1, 1, figsize=figsize)

        elif ptype == "dens":
            # Predictive density plot
            returns = roll_result.estimates['rt'].values
            pred_loglik = roll_result.estimates['predict_loglik'].values

            # Sort returns and apply lowess smoothing
            sorted_idx = np.argsort(returns)
            smoothed = lowess(pred_loglik[sorted_idx], returns[sorted_idx],
                              frac=0.2, return_sorted=False)

            ax.plot(returns[sorted_idx], smoothed, 'b-', linewidth=2)
            ax.set_xlabel('Log-returns')
            ax.set_ylabel('Densities')
            ax.set_title('Density forecasts')
            ax.grid(True, alpha=0.3)

        if ptype != "VaR":  # VaR creates its own figures
            plt.tight_layout()
            plt.show()


def plot_mdsv_sim(sim_result: MDSVSim, sim_num: int = 1,
                  figsize: tuple = (12, 8), **kwargs):
    """
    Plot simulation results

    Parameters
    ----------
    sim_result : MDSVSim
        Simulation results
    sim_num : int
        Which simulation to plot
    figsize : tuple
        Figure size
    """
    sim_data = sim_result.get_simulation(sim_num)

    # Determine number of subplots based on available data
    n_plots = len(sim_data)

    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

    plot_idx = 0

    # Plot returns if available
    if 'r_t' in sim_data:
        ax = axes[plot_idx]
        r_t = sim_data['r_t']

        ax.plot(r_t, 'b-', linewidth=0.5)
        ax.set_ylabel('Returns')
        ax.set_xlabel('Time')
        ax.set_title(f'Simulated Returns (Simulation {sim_num})')
        ax.grid(True, alpha=0.3)

        # Add summary statistics
        ax.text(0.02, 0.98, f'Mean: {np.mean(r_t):.4f}\nStd: {np.std(r_t):.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_idx += 1

    # Plot RV if available
    if 'RV_t' in sim_data:
        ax = axes[plot_idx]
        RV_t = sim_data['RV_t']

        ax.plot(RV_t, 'r-', linewidth=0.5)
        ax.set_ylabel('Realized Variance')
        ax.set_xlabel('Time')
        ax.set_title(f'Simulated Realized Variance (Simulation {sim_num})')
        ax.grid(True, alpha=0.3)

        # Add summary statistics
        ax.text(0.02, 0.98, f'Mean: {np.mean(RV_t):.4f}\nStd: {np.std(RV_t):.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_acf_comparison(data: Union[np.ndarray, pd.Series],
                        fitted_models: dict,
                        max_lag: int = 300,
                        title: str = "ACF Comparison",
                        figsize: tuple = (10, 6)):
    """
    Plot empirical ACF against fitted model ACFs

    Parameters
    ----------
    data : array-like
        Empirical data (e.g., realized variances)
    fitted_models : dict
        Dictionary of {label: MDSVFit/process} pairs
    max_lag : int
        Maximum lag to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    from statsmodels.tsa.stattools import acf

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Calculate empirical ACF
    empirical_acf = acf(data, nlags=max_lag, fft=True)
    lags = np.arange(max_lag + 1)

    # Plot empirical ACF
    ax.plot(lags, empirical_acf, 'k-', linewidth=2, label='Empirical')

    # Plot model ACFs
    colors = plt.cm.tab10(np.linspace(0, 1, len(fitted_models)))

    for (label, model), color in zip(fitted_models.items(), colors):
        if isinstance(model, MDSVProcess):
            process = model
        elif isinstance(model, MDSVFit):
            process = MDSVProcess(model.N, model.K)
            process.update_from_vector(model.estimates[:5])
        else:
            continue

        # Calculate theoretical ACF
        # This is simplified - would need full implementation
        model_acf = []
        for lag in range(max_lag + 1):
            if lag == 0:
                model_acf.append(1.0)
            else:
                # Use the autocorrelation formula from the paper
                phi = process.phi
                corr = 1.0
                for i in range(process.N):
                    corr *= (1 + phi[i] ** lag * ((process.K - 1) - 1)) / ((process.K - 1) - 1)
                model_acf.append(corr)

        ax.plot(lags, model_acf, '--', color=color, linewidth=1.5, label=label)

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.show()