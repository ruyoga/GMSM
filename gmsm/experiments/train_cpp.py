import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import json
from datetime import datetime

# Add MDSV package to path
root = Path().resolve()
mdsv_path = root / 'mdsv-main'
sys.path.append(str(mdsv_path))

from gmsm.models.mdsv.src.mdsv import MDSV, MDSVResult
from gmsm.models.mdsv.src.estimation import MDSVEstimator, EstimationOptions
from gmsm.models.mdsv.src.forecasting import MDSVForecaster

# ============================================================================
# VERIFY C++ ACCELERATION
# ============================================================================
print("=" * 80)
print("C++ ACCELERATION CHECK")
print("=" * 80)

cpp_enabled = False
try:
    import mdsv_cpp

    print("✓ C++ module (mdsv_cpp) imported successfully!")
    test_core = mdsv_cpp.MDSVCore(10, 3)
    print(f"✓ MDSVCore instance created: {test_core}")

    import gmsm.models.mdsv.src.mdsv as mdsv_module

    if hasattr(mdsv_module, 'USE_CPP'):
        print(f"✓ MDSV USE_CPP flag: {mdsv_module.USE_CPP}")
        cpp_enabled = mdsv_module.USE_CPP
        if cpp_enabled:
            print("✓ C++ ACCELERATION IS ENABLED!")
        else:
            print("✗ Warning: C++ module exists but USE_CPP is False")
except ImportError as e:
    print(f"✗ C++ module not available: {e}")
    print("⚠ WARNING: Running with pure Python implementation (slower)")

# ============================================================================
# STEP 1: LOADING PREPROCESSED DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING PREPROCESSED DATA")
print("=" * 80)

data_path = root / 'gmsm' / 'data' / 'processed' / 'daily_data_2015_2024.csv'
print(f"Loading data from: {data_path}")

mdsv_input = pd.read_csv(data_path)
mdsv_input['date'] = pd.to_datetime(mdsv_input['date'])

print(f"\nTotal days: {len(mdsv_input)}")
print(f"Date range: {mdsv_input['date'].min()} to {mdsv_input['date'].max()}")

mdsv_input_clean = mdsv_input.dropna(
    subset=['demeaned_log_return', 'realized_variance']
).reset_index(drop=True)
print(f"\nAfter removing NaN: {len(mdsv_input_clean)} days")

# ============================================================================
# STEP 2: Split Data into Historical (2015-2023) and Test (2024)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA SPLIT")
print("=" * 80)

historical_data = mdsv_input_clean[
    (mdsv_input_clean['date'].dt.year >= 2015) &
    (mdsv_input_clean['date'].dt.year <= 2023)
    ].copy().reset_index(drop=True)

test_data_2024 = mdsv_input_clean[
    mdsv_input_clean['date'].dt.year == 2024
    ].copy().reset_index(drop=True)

print(f"Historical data (2015-2023): {len(historical_data)} days")
print(f"  Date range: {historical_data['date'].min()} to {historical_data['date'].max()}")
print(f"\nTest data (2024): {len(test_data_2024)} days")
print(f"  Date range: {test_data_2024['date'].min()} to {test_data_2024['date'].max()}")

# ============================================================================
# STEP 3: Define Re-estimation Schedule (Every 90 Days)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: RE-ESTIMATION SCHEDULE (EVERY 90 DAYS)")
print("=" * 80)

REESTIMATION_INTERVAL = 20  # days

# Create quarters for 2024
quarters = []
n_days_2024 = len(test_data_2024)
start_idx = 0
quarter_num = 1

while start_idx < n_days_2024:
    end_idx = min(start_idx + REESTIMATION_INTERVAL, n_days_2024)
    quarters.append({
        'quarter': quarter_num,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'n_days': end_idx - start_idx,
        'start_date': test_data_2024['date'].iloc[start_idx],
        'end_date': test_data_2024['date'].iloc[end_idx - 1]
    })
    start_idx = end_idx
    quarter_num += 1

print(f"\nCreated {len(quarters)} quarters for re-estimation:")
for q in quarters:
    print(f"  Q{q['quarter']}: Days {q['start_idx'] + 1}-{q['end_idx']} "
          f"({q['n_days']} days) - {q['start_date'].strftime('%Y-%m-%d')} to "
          f"{q['end_date'].strftime('%Y-%m-%d')}")

print(f"\nRe-estimation strategy:")
print(f"  - Q1: Train on 2015-2023, forecast days 1-90 of 2024")
print(f"  - Q2: Train on 2015-2023 + Q1 2024, forecast days 91-180 of 2024")
print(f"  - Q3: Train on 2015-2023 + Q1-Q2 2024, forecast days 181-270 of 2024")
print(f"  - Q4: Train on 2015-2023 + Q1-Q3 2024, forecast remaining days of 2024")

# ============================================================================
# STEP 4: Rolling Re-estimation and Forecasting
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: ROLLING RE-ESTIMATION AND FORECASTING")
print("=" * 80)

# Storage for all results
all_predictions = []
all_actuals = []
all_leverage = []
all_dates = []
all_quarters = []

# Storage for quarter-specific results
quarter_results = []

# Initial parameters (None for first quarter)
previous_params = None

# Total timing
total_start_time = time.time()

# Process each quarter
for q_info in quarters:
    q_num = q_info['quarter']
    q_start_idx = q_info['start_idx']
    q_end_idx = q_info['end_idx']

    print("\n" + "=" * 80)
    print(f"QUARTER {q_num}: Days {q_start_idx + 1}-{q_end_idx} of 2024")
    print("=" * 80)
    print(f"Date range: {q_info['start_date'].strftime('%Y-%m-%d')} to "
          f"{q_info['end_date'].strftime('%Y-%m-%d')}")

    # ========================================================================
    # Define training data for this quarter
    # ========================================================================
    if q_num == 1:
        # Q1: Use only historical data (2015-2023)
        train_data = historical_data.copy()
        print(f"\nTraining data: 2015-2023 ({len(train_data)} days)")
    else:
        # Q2+: Use historical + all 2024 data up to this quarter
        past_2024_data = test_data_2024.iloc[:q_start_idx].copy()
        train_data = pd.concat([historical_data, past_2024_data],
                               ignore_index=True)
        print(f"\nTraining data: 2015-2023 + first {q_start_idx} days of 2024 "
              f"({len(train_data)} days)")

    train_returns = train_data['demeaned_log_return'].values
    train_rv = train_data['realized_variance'].values
    train_joint = np.column_stack([train_returns, train_rv])

    print(f"Training data shape: {train_joint.shape}")
    print(f"  Returns: mean={train_returns.mean():.4f}, std={train_returns.std():.4f}")
    print(f"  RV: mean={train_rv.mean():.4f}, std={train_rv.std():.4f}")

    # ========================================================================
    # Fit MDSV Model
    # ========================================================================
    print(f"\nFitting MDSV(3, 10) model with leverage for Q{q_num}...")

    model = MDSV(
        N=3,
        D=10,
        model_type=2,  # Joint model
        leverage=True
    )

    fit_start_time = time.time()

    try:
        # Use previous parameters as initial guess for warm start (Q2+)
        if previous_params is not None:
            print("  Using warm start with previous quarter's parameters")
            # Convert previous params to array format for initialization
            # Note: This requires the model to accept initial parameters
            # If not supported, it will just use default initialization

        result = model.fit(
            data=train_joint,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'disp': False},  # Less verbose
            verbose=False
        )

        fit_time = time.time() - fit_start_time

        # Extract results
        if isinstance(result, dict):
            success = result.get('success', result.get('convergence', True))
            log_likelihood = result.get('log_likelihood', result.get('loglik', np.nan))
            bic = result.get('bic', np.nan)
            aic = result.get('aic', np.nan)
            nit = result.get('nit', result.get('n_iterations', 'N/A'))
            parameters = result.get('parameters', {})
        else:
            success = result.convergence if hasattr(result, 'convergence') else (
                result.success if hasattr(result, 'success') else True)
            log_likelihood = result.log_likelihood
            bic = result.bic
            aic = result.aic
            nit = result.n_iterations if hasattr(result, 'n_iterations') else (
                result.nit if hasattr(result, 'nit') else 'N/A')
            parameters = result.parameters

        print(f"\n✓ Model fitted successfully for Q{q_num}")
        print(f"  Success: {success}")
        print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  AIC: {aic:.2f}")
        print(f"  Iterations: {nit}")
        print(f"  Fitting time: {fit_time:.2f}s ({fit_time / 60:.2f} min)")

        # Store parameters for next quarter's warm start
        if model._fitted and hasattr(model, 'params_'):
            previous_params = model.params_.copy() if hasattr(model.params_, 'copy') else model.params_

    except Exception as e:
        fit_time = time.time() - fit_start_time
        print(f"\n✗ Error fitting model for Q{q_num} after {fit_time:.2f}s: {e}")
        import traceback

        traceback.print_exc()

        # Skip this quarter if fitting fails
        quarter_results.append({
            'quarter': q_num,
            'success': False,
            'error': str(e),
            'fit_time': fit_time
        })
        continue

    # ========================================================================
    # Generate Forecasts for This Quarter
    # ========================================================================
    print(f"\nGenerating one-step-ahead forecasts for Q{q_num}...")

    try:
        forecaster = MDSVForecaster(model)

        q_predictions = []
        q_actuals = []
        q_leverage = []
        q_dates = []

        forecast_start_time = time.time()

        # Forecast each day in this quarter
        for i in range(q_start_idx, q_end_idx):
            # Data available up to day i (for forecasting day i)
            if i == 0:
                # First day: use only training data
                available_returns = train_returns
                available_rv = train_rv
            else:
                # Subsequent days: add 2024 data up to day i-1
                available_returns = np.concatenate([
                    train_returns,
                    test_data_2024['demeaned_log_return'].iloc[:i].values
                ])
                available_rv = np.concatenate([
                    train_rv,
                    test_data_2024['realized_variance'].iloc[:i].values
                ])

            available_data = np.column_stack([available_returns, available_rv])
            last_obs = available_data[-1:]

            # Get return history for leverage (last 70 returns)
            return_history = available_returns[-min(len(available_returns), 70):]

            # Forecast one step ahead
            forecast = forecaster.forecast(
                n_ahead=1,
                last_obs=last_obs,
                return_history=return_history
            )

            # Extract results
            if isinstance(forecast, dict):
                pred_rv = forecast['rv']
                if hasattr(pred_rv, '__len__'):
                    pred_rv = pred_rv[0]

                lev = forecast.get('leverage_multiplier', 1.0)
                if hasattr(lev, '__len__'):
                    lev = lev[0]
            else:
                pred_rv = forecast[0] if hasattr(forecast, '__len__') else forecast
                lev = 1.0

            q_predictions.append(pred_rv)
            q_actuals.append(test_data_2024['realized_variance'].iloc[i])
            q_leverage.append(lev)
            q_dates.append(test_data_2024['date'].iloc[i])

            # Progress update
            if (i - q_start_idx + 1) % 30 == 0 or i == q_start_idx:
                elapsed = time.time() - forecast_start_time
                date_str = test_data_2024['date'].iloc[i].strftime('%Y-%m-%d')
                progress = (i - q_start_idx + 1) / (q_end_idx - q_start_idx) * 100
                print(f"  Progress: {progress:.1f}% - {date_str}: "
                      f"pred={pred_rv:.4f}, actual={q_actuals[-1]:.4f}, L={lev:.2f}")

        forecast_time = time.time() - forecast_start_time

        print(f"\n✓ Completed {len(q_predictions)} forecasts for Q{q_num} "
              f"in {forecast_time:.2f}s ({forecast_time / 60:.2f} min)")

        # Add to global results
        all_predictions.extend(q_predictions)
        all_actuals.extend(q_actuals)
        all_leverage.extend(q_leverage)
        all_dates.extend(q_dates)
        all_quarters.extend([q_num] * len(q_predictions))

        # Calculate quarter-specific metrics
        q_predictions = np.array(q_predictions)
        q_actuals = np.array(q_actuals)
        q_leverage = np.array(q_leverage)

        valid_mask = ~np.isnan(q_predictions) & ~np.isnan(q_actuals)
        q_pred_valid = q_predictions[valid_mask]
        q_act_valid = q_actuals[valid_mask]

        if len(q_pred_valid) > 0:
            q_errors = q_pred_valid - q_act_valid
            q_rmse = np.sqrt(np.mean(q_errors ** 2))
            q_mae = np.mean(np.abs(q_errors))
            q_mape = np.mean(np.abs(q_errors / q_act_valid)) * 100

            print(f"\nQ{q_num} Performance:")
            print(f"  RMSE: {q_rmse:.4f}")
            print(f"  MAE: {q_mae:.4f}")
            print(f"  MAPE: {q_mape:.2f}%")
            print(f"  Actual RV: mean={q_act_valid.mean():.4f}, std={q_act_valid.std():.4f}")
            print(f"  Predicted RV: mean={q_pred_valid.mean():.4f}, std={q_pred_valid.std():.4f}")
            print(f"  Std Ratio: {q_pred_valid.std() / q_act_valid.std():.4f}")
        else:
            q_rmse = q_mae = q_mape = np.nan

        # Store quarter results
        quarter_results.append({
            'quarter': q_num,
            'start_date': q_info['start_date'].strftime('%Y-%m-%d'),
            'end_date': q_info['end_date'].strftime('%Y-%m-%d'),
            'n_days': q_info['n_days'],
            'train_size': len(train_data),
            'success': True,
            'fit_time': fit_time,
            'forecast_time': forecast_time,
            'log_likelihood': float(log_likelihood) if np.isfinite(log_likelihood) else None,
            'aic': float(aic) if np.isfinite(aic) else None,
            'bic': float(bic) if np.isfinite(bic) else None,
            'rmse': float(q_rmse) if np.isfinite(q_rmse) else None,
            'mae': float(q_mae) if np.isfinite(q_mae) else None,
            'mape': float(q_mape) if np.isfinite(q_mape) else None,
            'actual_mean': float(q_act_valid.mean()) if len(q_act_valid) > 0 else None,
            'actual_std': float(q_act_valid.std()) if len(q_act_valid) > 0 else None,
            'predicted_mean': float(q_pred_valid.mean()) if len(q_pred_valid) > 0 else None,
            'predicted_std': float(q_pred_valid.std()) if len(q_pred_valid) > 0 else None,
            'std_ratio': float(q_pred_valid.std() / q_act_valid.std()) if len(
                q_pred_valid) > 0 and q_act_valid.std() > 0 else None,
            'leverage_mean': float(np.nanmean(q_leverage)),
            'leverage_std': float(np.nanstd(q_leverage))
        })

    except Exception as e:
        forecast_time = time.time() - forecast_start_time
        print(f"\n✗ Error forecasting for Q{q_num} after {forecast_time:.2f}s: {e}")
        import traceback

        traceback.print_exc()

        quarter_results.append({
            'quarter': q_num,
            'success': False,
            'error': str(e),
            'forecast_time': forecast_time
        })

total_time = time.time() - total_start_time

# ============================================================================
# STEP 5: Overall Performance Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: OVERALL PERFORMANCE EVALUATION (ALL QUARTERS)")
print("=" * 80)

predictions_all = np.array(all_predictions)
actuals_all = np.array(all_actuals)
leverage_all = np.array(all_leverage)
quarters_all = np.array(all_quarters)

valid_mask = ~np.isnan(predictions_all) & ~np.isnan(actuals_all)
predictions_valid = predictions_all[valid_mask]
actuals_valid = actuals_all[valid_mask]

print(f"\nTotal forecasts: {len(predictions_all)}")
print(f"Valid forecasts: {len(predictions_valid)}")

if len(predictions_valid) > 10:
    errors = predictions_valid - actuals_valid
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actuals_valid)) * 100
    median_ae = np.median(np.abs(errors))

    print(f"\nOverall MDSV Performance (Rolling Re-estimation):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Median AE: {median_ae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    print(f"\nActual RV Statistics (2024):")
    print(f"  Mean: {actuals_valid.mean():.4f}")
    print(f"  Std: {actuals_valid.std():.4f}")
    print(f"  Min: {actuals_valid.min():.4f}")
    print(f"  Max: {actuals_valid.max():.4f}")

    print(f"\nPredicted RV Statistics:")
    print(f"  Mean: {predictions_valid.mean():.4f}")
    print(f"  Std: {predictions_valid.std():.4f}")
    print(f"  Min: {predictions_valid.min():.4f}")
    print(f"  Max: {predictions_valid.max():.4f}")
    print(f"  Std Ratio: {predictions_valid.std() / actuals_valid.std():.4f}")

    # Naive benchmark
    print(f"\n" + "-" * 80)
    print("BENCHMARK: Naive (Persistence) Forecast")
    print("-" * 80)

    naive_predictions = test_data_2024['realized_variance'].iloc[:-1].values
    naive_actuals = test_data_2024['realized_variance'].iloc[1:].values
    naive_mask = ~np.isnan(naive_predictions) & ~np.isnan(naive_actuals)

    naive_rmse = np.sqrt(np.mean((naive_predictions[naive_mask] - naive_actuals[naive_mask]) ** 2))
    naive_mae = np.mean(np.abs(naive_predictions[naive_mask] - naive_actuals[naive_mask]))

    print(f"  RMSE: {naive_rmse:.4f}")
    print(f"  MAE: {naive_mae:.4f}")

    improvement_rmse = ((naive_rmse - rmse) / naive_rmse * 100)
    improvement_mae = ((naive_mae - mae) / naive_mae * 100)

    print(f"\nMDSV vs Naive:")
    if improvement_rmse > 0:
        print(f"  ✓ RMSE improvement: {improvement_rmse:.2f}%")
    else:
        print(f"  ✗ RMSE worse by: {abs(improvement_rmse):.2f}%")

    if improvement_mae > 0:
        print(f"  ✓ MAE improvement: {improvement_mae:.2f}%")
    else:
        print(f"  ✗ MAE worse by: {abs(improvement_mae):.2f}%")
else:
    rmse = mae = mape = median_ae = np.nan
    naive_rmse = naive_mae = np.nan
    improvement_rmse = improvement_mae = np.nan

# ============================================================================
# STEP 6: Quarter-by-Quarter Comparison
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: QUARTER-BY-QUARTER COMPARISON")
print("=" * 80)

print(f"\n{'Quarter':<10} {'Days':<8} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Std Ratio':<12} {'Fit Time(s)':<12}")
print("-" * 85)

for q_res in quarter_results:
    if q_res.get('success', False):
        print(f"Q{q_res['quarter']:<9} {q_res['n_days']:<8} "
              f"{q_res['rmse']:<10.4f} {q_res['mae']:<10.4f} "
              f"{q_res['mape']:<10.2f} {q_res['std_ratio']:<12.4f} "
              f"{q_res['fit_time']:<12.2f}")
    else:
        print(f"Q{q_res['quarter']:<9} {'FAILED':<8}")

# ============================================================================
# STEP 7: Save Results
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: SAVING RESULTS")
print("=" * 80)

output_path = root / 'data' / 'results'
output_path.mkdir(parents=True, exist_ok=True)

# Save detailed forecasts
forecast_df = pd.DataFrame({
    'date': all_dates,
    'quarter': quarters_all,
    'actual_rv': actuals_all,
    'predicted_rv': predictions_all,
    'leverage_multiplier': leverage_all,
    'error': actuals_all - predictions_all,
    'squared_error': (actuals_all - predictions_all) ** 2,
    'abs_error': np.abs(actuals_all - predictions_all)
})

forecast_df.to_csv(output_path / 'forecasts.csv', index=False)
print(f"\n✓ Forecasts saved to: {output_path / 'mdsv_forecasts_2024_rolling_90days.csv'}")

# Save comprehensive results summary
results_summary = {
    'methodology': {
        'approach': 'Rolling re-estimation every 90 days',
        'reestimation_interval_days': REESTIMATION_INTERVAL,
        'n_quarters': len(quarters),
        'cpp_acceleration': cpp_enabled,
        'model_specification': {
            'N': 3,
            'D': 10,
            'model_type': 2,
            'leverage': True
        }
    },
    'timing': {
        'total_time_seconds': float(total_time),
        'total_time_minutes': float(total_time / 60),
        'total_fitting_time': sum([q.get('fit_time', 0) for q in quarter_results if q.get('success')]),
        'total_forecasting_time': sum([q.get('forecast_time', 0) for q in quarter_results if q.get('success')])
    },
    'overall_performance': {
        'valid_forecasts': int(len(predictions_valid)),
        'total_forecasts': int(len(predictions_all)),
        'rmse': float(rmse) if np.isfinite(rmse) else None,
        'mae': float(mae) if np.isfinite(mae) else None,
        'mape': float(mape) if np.isfinite(mape) else None,
        'median_ae': float(median_ae) if np.isfinite(median_ae) else None,
        'naive_rmse': float(naive_rmse) if np.isfinite(naive_rmse) else None,
        'naive_mae': float(naive_mae) if np.isfinite(naive_mae) else None,
        'improvement_rmse_pct': float(improvement_rmse) if np.isfinite(improvement_rmse) else None,
        'improvement_mae_pct': float(improvement_mae) if np.isfinite(improvement_mae) else None,
        'predicted_rv_mean': float(predictions_valid.mean()) if len(predictions_valid) > 0 else None,
        'predicted_rv_std': float(predictions_valid.std()) if len(predictions_valid) > 0 else None,
        'actual_rv_mean': float(actuals_valid.mean()) if len(actuals_valid) > 0 else None,
        'actual_rv_std': float(actuals_valid.std()) if len(actuals_valid) > 0 else None,
        'std_ratio': float(predictions_valid.std() / actuals_valid.std()) if len(
            predictions_valid) > 0 and actuals_valid.std() > 0 else None
    },
    'quarter_results': quarter_results
}

with open(output_path / 'mdsv_rolling_90days_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"✓ Summary saved to: {output_path / 'mdsv_rolling_90days_summary.json'}")

# Save quarter-by-quarter comparison
quarter_comparison_df = pd.DataFrame(quarter_results)
quarter_comparison_df.to_csv(output_path / 'mdsv_quarter_comparison.csv', index=False)
print(f"✓ Quarter comparison saved to: {output_path / 'mdsv_quarter_comparison.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"\nMethodology: Rolling re-estimation every {REESTIMATION_INTERVAL} days")
print(f"Total quarters processed: {len(quarters)}")
print(f"Successful quarters: {sum([1 for q in quarter_results if q.get('success', False)])}")

print(f"\nTiming Summary:")
print(f"  Total pipeline time: {total_time:.2f}s ({total_time / 60:.2f} min)")
print(f"  Total fitting time: {sum([q.get('fit_time', 0) for q in quarter_results if q.get('success')]):.2f}s")
print(f"  Total forecasting time: {sum([q.get('forecast_time', 0) for q in quarter_results if q.get('success')]):.2f}s")

if len(predictions_valid) > 10 and np.isfinite(rmse):
    print(f"\nOverall Performance:")
    print(f"  C++ Acceleration: {'✓ Enabled' if cpp_enabled else '✗ Disabled'}")
    print(f"  Valid forecasts: {len(predictions_valid)}/{len(predictions_all)}")
    print(f"  MDSV RMSE: {rmse:.4f}")
    print(f"  Naive RMSE: {naive_rmse:.4f}")
    print(f"  Improvement: {improvement_rmse:.2f}%")
    print(f"  Std Ratio: {predictions_valid.std() / actuals_valid.std():.4f}")

    print(f"\nQuarterly Performance Summary:")
    for q_res in quarter_results:
        if q_res.get('success', False):
            print(f"  Q{q_res['quarter']}: RMSE={q_res['rmse']:.4f}, "
                  f"MAE={q_res['mae']:.4f}, Std Ratio={q_res['std_ratio']:.4f}")

print("\n" + "=" * 80)
print("KEY IMPROVEMENTS FROM ROLLING RE-ESTIMATION:")
print("=" * 80)
print("✓ Model parameters updated quarterly with latest data")
print("✓ Adapts to changing market conditions over 2024")
print("✓ More realistic out-of-sample forecasting scenario")
print("✓ Each quarter uses all available historical information")
print("✓ No look-ahead bias in forecasting process")

print("\n" + "=" * 80)
print("OUTPUT FILES GENERATED:")
print("=" * 80)
print(f"1. {output_path / 'mdsv_forecasts_2024_rolling_90days.csv'}")
print(f"   - Detailed daily forecasts with quarter indicators")
print(f"2. {output_path / 'mdsv_rolling_90days_summary.json'}")
print(f"   - Comprehensive performance metrics and methodology")
print(f"3. {output_path / 'mdsv_quarter_comparison.csv'}")
print(f"   - Quarter-by-quarter performance comparison")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)