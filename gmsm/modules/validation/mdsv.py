import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf
import warnings

warnings.filterwarnings('ignore')


class MDSVValidation:
    """
    Comprehensive validation framework for MDSV variance predictions
    """

    def __init__(self, returns, variance_predictions, frequency='1min'):
        """
        Initialize validation framework

        Parameters:
        -----------
        returns : pd.Series
            Observed returns with datetime index
        variance_predictions : pd.Series
            MDSV variance predictions with datetime index
        frequency : str
            Data frequency ('1min', '5min', etc.)
        """
        self.returns = returns
        self.variance_predictions = variance_predictions
        self.frequency = frequency
        self.annualization_factor = self._get_annualization_factor()

        # Align data
        common_index = returns.index.intersection(variance_predictions.index)
        self.returns = returns.loc[common_index]
        self.variance_predictions = variance_predictions.loc[common_index]

    def _get_annualization_factor(self):
        """Calculate annualization factor based on frequency"""
        if self.frequency == '1min':
            return 252 * 24 * 60  # Trading days * hours * minutes
        elif self.frequency == '5min':
            return 252 * 24 * 12
        elif self.frequency == '15min':
            return 252 * 24 * 4
        elif self.frequency == '1H':
            return 252 * 24
        elif self.frequency == '1D':
            return 252
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")

    def compute_realized_variance_rolling(self, window_minutes=60):
        """
        ROLLING WINDOW METHOD FOR VARIANCE VALIDATION

        Rationale: Since we cannot observe the "true" instantaneous variance at each
        1-minute interval (variance is latent), we use rolling windows to create
        a proxy for realized variance that can be compared against MDSV predictions.

        The rolling window approach:
        1. Takes a window of past returns (e.g., 60 minutes)
        2. Computes realized variance as sum of squared returns in that window
        3. Rolls this window forward to create a time series of realized variance
        4. This gives us an observable "ground truth" to validate predictions against

        Mathematical foundation:
        - True variance is σ²(t) (unobservable)
        - MDSV predicts E[σ²(t+1)|Ω(t)] where Ω(t) is info at time t
        - Rolling realized variance: RV(t) = Σ[r(t-i)²] for i=0 to window_size-1
        - Under certain conditions: E[RV(t)] ≈ σ²(t) for the window period

        This is the STANDARD approach in volatility forecasting literature.
        """
        window_size = window_minutes  # For 1-minute data

        print(f"Computing rolling realized variance with {window_size}-minute windows...")
        print("This creates observable targets to validate against MDSV predictions.")

        # Rolling realized variance (sum of squared returns)
        # This is the key method your professor wants you to implement
        realized_var = self.returns.rolling(
            window=window_size,
            min_periods=window_size // 2
        ).apply(lambda x: np.sum(x ** 2), raw=True)

        return realized_var.dropna()

    def compute_multiple_horizon_rolling_variance(self, windows=[30, 60, 120, 240]):
        """
        Compute rolling realized variance at multiple time horizons
        This addresses the horizon matching problem in variance prediction validation

        Different horizons help validate:
        - Short-term: 30-min windows (high frequency, noisy)
        - Medium-term: 60-120 min windows (balanced signal/noise)
        - Long-term: 240+ min windows (smoother, less noisy)
        """
        rolling_variances = {}

        for window in windows:
            print(f"Computing {window}-minute rolling realized variance...")
            rv = self.returns.rolling(window=window, min_periods=window // 2).apply(
                lambda x: np.sum(x ** 2), raw=True
            ).dropna()
            rolling_variances[f'RV_{window}min'] = rv

        return pd.DataFrame(rolling_variances)

    def rolling_window_forecast_validation(self, prediction_horizon=1, validation_window=60):
        """
        CORE ROLLING WINDOW VALIDATION METHOD

        This implements the exact rolling window validation your professor wants:

        1. At each time t, use past data to make variance prediction for t+h
        2. Wait until t+h and compute realized variance in window [t+h-w, t+h]
        3. Compare prediction vs realized variance
        4. Roll forward and repeat

        This simulates real-time forecasting and validation
        """
        print("Implementing Rolling Window Forecast Validation...")
        print("=" * 60)

        validation_results = []
        start_idx = validation_window + prediction_horizon

        for i in range(start_idx, len(self.returns) - prediction_horizon):
            current_time = self.returns.index[i]
            forecast_time = self.returns.index[i + prediction_horizon]

            # Get MDSV prediction made at current_time for forecast_time
            if current_time in self.variance_predictions.index:
                mdsv_prediction = self.variance_predictions.loc[current_time]

                # Compute realized variance in window ending at forecast_time
                end_idx = i + prediction_horizon
                start_idx_window = max(0, end_idx - validation_window)

                window_returns = self.returns.iloc[start_idx_window:end_idx]
                realized_variance = np.sum(window_returns ** 2)

                validation_results.append({
                    'prediction_time': current_time,
                    'forecast_time': forecast_time,
                    'mdsv_prediction': mdsv_prediction,
                    'realized_variance': realized_variance,
                    'prediction_error': mdsv_prediction - realized_variance,
                    'squared_error': (mdsv_prediction - realized_variance) ** 2,
                    'absolute_error': abs(mdsv_prediction - realized_variance)
                })

        validation_df = pd.DataFrame(validation_results)

        # Compute validation statistics
        mse = validation_df['squared_error'].mean()
        mae = validation_df['absolute_error'].mean()
        correlation = validation_df['mdsv_prediction'].corr(validation_df['realized_variance'])
        bias = validation_df['prediction_error'].mean()

        print(f"Rolling Window Validation Results:")
        print(f"  Prediction horizon: {prediction_horizon} periods")
        print(f"  Validation window: {validation_window} periods")
        print(f"  Number of forecasts: {len(validation_df)}")
        print(f"  Mean Squared Error: {mse:.8f}")
        print(f"  Mean Absolute Error: {mae:.8f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Bias: {bias:.8f}")

        return validation_df

    def annualize_predictions(self):
        """Convert variance predictions to annualized volatility"""
        annualized_vol = np.sqrt(self.variance_predictions * self.annualization_factor)
        return annualized_vol

    def mincer_zarnowitz_test(self, realized_var):
        """
        Mincer-Zarnowitz regression test for unbiased predictions
        Regress realized variance on predicted variance
        """
        # Align data
        common_idx = realized_var.index.intersection(self.variance_predictions.index)
        y = realized_var.loc[common_idx].values
        x = self.variance_predictions.loc[common_idx].values

        # Remove any infinite or NaN values
        mask = np.isfinite(y) & np.isfinite(x)
        y, x = y[mask], x[mask]

        if len(y) < 30:
            print("Warning: Insufficient data for Mincer-Zarnowitz test")
            return None

        # OLS regression: realized = alpha + beta * predicted + error
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Test H0: alpha=0, beta=1 (unbiased predictions)
        n = len(y)
        mse = np.mean((y - (intercept + slope * x)) ** 2)

        # t-tests
        t_alpha = intercept / (np.sqrt(mse * (1 / n + np.mean(x ** 2) / (n * np.var(x)))))
        t_beta = (slope - 1) / (np.sqrt(mse / (n * np.var(x))))

        results = {
            'alpha': intercept,
            'beta': slope,
            'r_squared': r_value ** 2,
            't_alpha': t_alpha,
            't_beta': t_beta,
            'p_alpha': 2 * (1 - stats.t.cdf(abs(t_alpha), n - 2)),
            'p_beta': 2 * (1 - stats.t.cdf(abs(t_beta), n - 2)),
            'unbiased': (abs(t_alpha) < 1.96) and (abs(t_beta) < 1.96)
        }

        return results

    def diebold_mariano_test(self, realized_var, benchmark_predictions=None):
        """
        Diebold-Mariano test for forecast superiority
        Compare MDSV against benchmark (e.g., historical volatility)
        """
        if benchmark_predictions is None:
            # Use rolling historical volatility as benchmark
            benchmark_predictions = self.returns.rolling(window=60).var()

        # Align all series
        common_idx = realized_var.index.intersection(
            self.variance_predictions.index
        ).intersection(benchmark_predictions.index)

        y = realized_var.loc[common_idx].values
        pred1 = self.variance_predictions.loc[common_idx].values  # MDSV
        pred2 = benchmark_predictions.loc[common_idx].values  # Benchmark

        # Remove NaN/inf values
        mask = np.isfinite(y) & np.isfinite(pred1) & np.isfinite(pred2)
        y, pred1, pred2 = y[mask], pred1[mask], pred2[mask]

        if len(y) < 30:
            return None

        # Loss differences (using squared errors)
        loss1 = (y - pred1) ** 2
        loss2 = (y - pred2) ** 2
        loss_diff = loss1 - loss2

        # DM statistic
        d_bar = np.mean(loss_diff)
        d_var = np.var(loss_diff, ddof=1)
        dm_stat = d_bar / np.sqrt(d_var / len(loss_diff))

        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'mdsv_superior': dm_stat < -1.96,  # MDSV has lower loss
            'mean_loss_diff': d_bar
        }

    def compute_validation_metrics(self, realized_var):
        """Compute comprehensive validation metrics"""
        # Align data
        common_idx = realized_var.index.intersection(self.variance_predictions.index)
        y = realized_var.loc[common_idx]
        pred = self.variance_predictions.loc[common_idx]

        # Remove NaN/inf values
        mask = np.isfinite(y) & np.isfinite(pred)
        y, pred = y[mask], pred[mask]

        if len(y) < 10:
            return {}

        metrics = {
            'mse': mean_squared_error(y, pred),
            'rmse': np.sqrt(mean_squared_error(y, pred)),
            'mae': mean_absolute_error(y, pred),
            'mape': np.mean(np.abs((y - pred) / y)) * 100,
            'correlation': np.corrcoef(y, pred)[0, 1],
            'hit_rate': np.mean((np.sign(y.diff().dropna()) == np.sign(pred.diff().dropna()))),
        }

        return metrics

    def plot_validation_results(self, realized_var, save_path=None):
        """Create comprehensive validation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Align data for plotting
        common_idx = realized_var.index.intersection(self.variance_predictions.index)
        y = realized_var.loc[common_idx]
        pred = self.variance_predictions.loc[common_idx]

        # 1. Time series plot
        axes[0, 0].plot(y.index, y, label='Realized Variance', alpha=0.7)
        axes[0, 0].plot(pred.index, pred, label='MDSV Predictions', alpha=0.7)
        axes[0, 0].set_title('Realized vs Predicted Variance')
        axes[0, 0].legend()
        axes[0, 0].set_ylabel('Variance')

        # 2. Scatter plot
        axes[0, 1].scatter(pred, y, alpha=0.5)
        axes[0, 1].plot([pred.min(), pred.max()], [pred.min(), pred.max()], 'r--', label='45° line')
        axes[0, 1].set_xlabel('Predicted Variance')
        axes[0, 1].set_ylabel('Realized Variance')
        axes[0, 1].set_title('Predicted vs Realized')
        axes[0, 1].legend()

        # 3. Residuals plot
        residuals = y - pred
        axes[1, 0].plot(residuals.index, residuals)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Prediction Residuals')
        axes[1, 0].set_ylabel('Residuals')

        # 4. QQ plot of residuals
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('QQ Plot of Residuals')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def run_full_validation(self, window_minutes=60, benchmark_predictions=None):
        """Run complete validation suite WITH ROLLING WINDOW METHOD"""
        print("MDSV Model Validation Report")
        print("=" * 50)

        # 1. ROLLING WINDOW FORECAST VALIDATION (Primary Method)
        print("\n🎯 ROLLING WINDOW FORECAST VALIDATION (Professor's Method)")
        rolling_validation = self.rolling_window_forecast_validation(
            prediction_horizon=1,
            validation_window=window_minutes
        )

        # 2. Multiple horizon rolling variance
        print(f"\n📊 MULTIPLE HORIZON ANALYSIS")
        multi_horizon_rv = self.compute_multiple_horizon_rolling_variance()

        # 3. Standard rolling realized variance (for comparison)
        print(f"\n📈 STANDARD ROLLING REALIZED VARIANCE")
        realized_var = self.compute_realized_variance_rolling(window_minutes)

        # 4. Basic metrics on standard approach
        print(f"\n📋 BASIC VALIDATION METRICS (Standard Approach):")
        metrics = self.compute_validation_metrics(realized_var)
        for key, value in metrics.items():
            print(f"   {key.upper()}: {value:.4f}")

        # 5. Mincer-Zarnowitz test
        print(f"\n🧪 MINCER-ZARNOWITZ UNBIASEDNESS TEST:")
        mz_results = self.mincer_zarnowitz_test(realized_var)
        if mz_results:
            print(f"   Alpha (intercept): {mz_results['alpha']:.6f}")
            print(f"   Beta (slope): {mz_results['beta']:.6f}")
            print(f"   R-squared: {mz_results['r_squared']:.4f}")
            print(f"   Unbiased predictions: {mz_results['unbiased']}")
            print(f"   p-value (alpha=0): {mz_results['p_alpha']:.4f}")
            print(f"   p-value (beta=1): {mz_results['p_beta']:.4f}")

        # 6. Diebold-Mariano test
        print(f"\n⚖️  DIEBOLD-MARIANO FORECAST COMPARISON:")
        dm_results = self.diebold_mariano_test(realized_var, benchmark_predictions)
        if dm_results:
            print(f"   DM Statistic: {dm_results['dm_statistic']:.4f}")
            print(f"   p-value: {dm_results['p_value']:.4f}")
            print(f"   MDSV Superior: {dm_results['mdsv_superior']}")

        # 7. Annualized volatility
        print(f"\n📊 ANNUALIZED VOLATILITY STATISTICS:")
        ann_vol = self.annualize_predictions()
        print(f"   Mean Annualized Vol: {ann_vol.mean():.2f}%")
        print(f"   Std Annualized Vol: {ann_vol.std():.2f}%")
        print(f"   Min/Max: {ann_vol.min():.2f}% / {ann_vol.max():.2f}%")

        # 8. Create plots
        print(f"\n📈 GENERATING VALIDATION PLOTS...")
        self.plot_validation_results(realized_var)
        self.plot_rolling_window_validation(rolling_validation)

        return {
            'rolling_window_validation': rolling_validation,  # Primary result
            'multi_horizon_rv': multi_horizon_rv,
            'standard_metrics': metrics,
            'mincer_zarnowitz': mz_results,
            'diebold_mariano': dm_results,
            'realized_variance': realized_var,
            'annualized_volatility': ann_vol
        }

    def plot_rolling_window_validation(self, rolling_validation_df):
        """
        Plot rolling window validation results
        This visualizes the core validation method your professor requested
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Predictions vs Realized over time
        axes[0, 0].plot(rolling_validation_df['forecast_time'],
                        rolling_validation_df['realized_variance'],
                        label='Realized Variance', alpha=0.7, linewidth=1)
        axes[0, 0].plot(rolling_validation_df['forecast_time'],
                        rolling_validation_df['mdsv_prediction'],
                        label='MDSV Predictions', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Rolling Window Validation: Predictions vs Realized')
        axes[0, 0].legend()
        axes[0, 0].set_ylabel('Variance')

        # 2. Scatter plot of predictions vs realized
        axes[0, 1].scatter(rolling_validation_df['mdsv_prediction'],
                           rolling_validation_df['realized_variance'],
                           alpha=0.5, s=20)

        # Add 45-degree line
        min_val = min(rolling_validation_df['mdsv_prediction'].min(),
                      rolling_validation_df['realized_variance'].min())
        max_val = max(rolling_validation_df['mdsv_prediction'].max(),
                      rolling_validation_df['realized_variance'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        axes[0, 1].set_xlabel('MDSV Predictions')
        axes[0, 1].set_ylabel('Realized Variance')
        axes[0, 1].set_title('Rolling Window: Predicted vs Realized')
        axes[0, 1].legend()

        # 3. Prediction errors over time
        axes[1, 0].plot(rolling_validation_df['forecast_time'],
                        rolling_validation_df['prediction_error'])
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Rolling Window: Prediction Errors Over Time')
        axes[1, 0].set_ylabel('Prediction Error')

        # 4. Error distribution
        axes[1, 1].hist(rolling_validation_df['prediction_error'], bins=30, alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Rolling Window: Error Distribution')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        # Print rolling window specific statistics
        print("\nRolling Window Validation Statistics:")
        print(f"  Mean Prediction Error: {rolling_validation_df['prediction_error'].mean():.8f}")
        print(f"  Std Prediction Error: {rolling_validation_df['prediction_error'].std():.8f}")
        print(f"  RMSE: {np.sqrt(rolling_validation_df['squared_error'].mean()):.8f}")
        print(f"  MAE: {rolling_validation_df['absolute_error'].mean():.8f}")

        # Test for serial correlation in errors (should be white noise)
        errors = rolling_validation_df['prediction_error'].dropna()
        if len(errors) > 10:
            autocorr_1 = errors.autocorr(lag=1)
            print(f"  Error Autocorrelation (lag 1): {autocorr_1:.4f}")
            print(f"  Errors appear random: {abs(autocorr_1) < 0.1}")  # Rule of thumb


# Example usage and additional helper functions
def load_and_prepare_data(returns_file, predictions_file):
    """
    Helper function to load and prepare data
    Assumes CSV files with datetime index
    """
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True).squeeze()
    predictions = pd.read_csv(predictions_file, index_col=0, parse_dates=True).squeeze()

    return returns, predictions


def compare_with_implied_volatility(mdsv_vol, spx_options_data):
    """
    Compare MDSV predictions with implied volatility from options
    """
    # This function would compare your annualized MDSV volatility
    # with implied volatility from your SPX options data
    # Implementation depends on your options data structure
    pass


# Example implementation
if __name__ == "__main__":
    # Example with synthetic data - replace with your actual data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10000, freq='1min')

    # Synthetic returns (replace with your S&P 1-minute returns)
    returns = pd.Series(np.random.normal(0, 0.001, len(dates)), index=dates)

    # Synthetic MDSV predictions (replace with your model predictions)
    true_var = 0.001 ** 2 * (1 + 0.5 * np.sin(np.arange(len(dates)) / 100))
    predictions = pd.Series(true_var + np.random.normal(0, 0.0001, len(dates)), index=dates)

    # Run validation
    validator = MDSVValidation(returns, predictions, frequency='1min')
    results = validator.run_full_validation(window_minutes=60)

    print("\nValidation complete! Use results for your options trading strategy.")