import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add MDSV package to path
root = Path().resolve()
mdsv_path = root / 'mdsv-main'
sys.path.append(str(mdsv_path))

from gmsm.models.mdsv.src.mdsv import MDSV, MDSVResult
from gmsm.models.mdsv.src.estimation import MDSVEstimator, EstimationOptions
from gmsm.models.mdsv.src.forecasting import MDSVForecaster

print("="*80)
print("STEP 1: LOADING PREPROCESSED DATA")
print("="*80)

data_path = root / 'gmsm' / 'data' / 'processed' / 'daily_data_2015_2024.csv'
print(f"Loading data from: {data_path}")

mdsv_input = pd.read_csv(data_path)
mdsv_input['date'] = pd.to_datetime(mdsv_input['date'])

print(f"\nTotal days: {len(mdsv_input)}")
print(f"Date range: {mdsv_input['date'].min()} to {mdsv_input['date'].max()}")

# Check for NaN values
print(f"\nChecking for missing values:")
print(mdsv_input[['demeaned_log_return', 'realized_variance']].isna().sum())

# Remove any rows with NaN values
mdsv_input_clean = mdsv_input.dropna(subset=['demeaned_log_return', 'realized_variance']).reset_index(drop=True)
print(f"\nAfter removing NaN: {len(mdsv_input_clean)} days")
print(f"Date range: {mdsv_input_clean['date'].min()} to {mdsv_input_clean['date'].max()}")

# ============================================================================
# STEP 2: Train/Test Split
# ============================================================================
print("\n" + "="*80)
print("STEP 2: TRAIN/TEST SPLIT")
print("="*80)

# Use last year as test set
test_size = 252  # Approximately 1 trading year
train_data = mdsv_input_clean.iloc[:-test_size].copy()
test_data = mdsv_input_clean.iloc[-test_size:].copy()

print(f"Train set: {len(train_data)} days ({train_data['date'].min()} to {train_data['date'].max()})")
print(f"Test set: {len(test_data)} days ({test_data['date'].min()} to {test_data['date'].max()})")

# ============================================================================
# STEP 3: Fit MDSV Model using MDSVEstimator
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FITTING MDSV MODEL")
print("="*80)

# Prepare training data: returns and RV as columns
train_returns = train_data['demeaned_log_return'].values
train_rv = train_data['realized_variance'].values
train_joint = np.column_stack([train_returns, train_rv])

print(f"\nTraining data shape: {train_joint.shape}")
print(f"  Column 0 (returns): mean={train_returns.mean():.4f}, std={train_returns.std():.4f}")
print(f"  Column 1 (RV): mean={train_rv.mean():.4f}, std={train_rv.std():.4f}")
print(f"  Returns range: [{train_returns.min():.4f}, {train_returns.max():.4f}]")
print(f"  RV range: [{train_rv.min():.4f}, {train_rv.max():.4f}]")

# Initialize MDSV model with smaller dimensions for faster fitting
print("\nInitializing MDSV(2, 5) model with leverage...")
model = MDSV(
    N=2,              # Number of components
    D=5,              # States per component
    model_type=2,     # Joint model (0=returns only, 1=RV only, 2=joint)
    leverage=True     # Include leverage effect
)

# Use MDSVEstimator for proper estimation (following the example)
print("\nSetting up MDSVEstimator with options...")
estimator = MDSVEstimator(model)

# Create estimation options
options = EstimationOptions(
    method='L-BFGS-B',
    maxiter=1000,
    verbose=True,
)

print("\nFitting model using MDSVEstimator (this will take several minutes)...")
print("This is the proper way to fit the model - please be patient...\n")

try:
    result = estimator.estimate(
        data=train_joint,
        options=options
    )

    print("\n" + "="*80)
    print("MODEL FITTING RESULTS")
    print("="*80)
    print(f"Success: {result.success}")
    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"BIC: {result.bic:.2f}")
    print(f"AIC: {result.aic:.2f}")
    print(f"Number of iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")

    print("\nEstimated Parameters:")
    for param_name, param_value in result.parameters.items():
        if isinstance(param_value, (int, float, np.floating)):
            print(f"  {param_name}: {param_value:.6f}")
        elif isinstance(param_value, np.ndarray):
            if param_value.size <= 10:
                print(f"  {param_name}: {param_value}")
            else:
                print(f"  {param_name}: shape={param_value.shape}, mean={param_value.mean():.6f}")
        else:
            print(f"  {param_name}: {type(param_value)}")

    # Check if model fitted successfully
    if not np.isfinite(result.log_likelihood):
        raise ValueError("Model fitting failed - log-likelihood is not finite")

except Exception as e:
    print(f"\nError during fitting: {e}")
    import traceback
    traceback.print_exc()
    print("\nTrying simpler model (no leverage)...")

    # Fallback: simpler model without leverage
    model = MDSV(N=2, D=5, model_type=2, leverage=False)
    estimator = MDSVEstimator(model)

    result = estimator.estimate(
        data=train_joint,
        options=options
    )
    print(f"\nFallback model results:")
    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"BIC: {result.bic:.2f}")

# ============================================================================
# STEP 4: One-Step Ahead Forecasting
# ============================================================================
print("\n" + "="*80)
print("STEP 4: ONE-STEP AHEAD FORECASTING")
print("="*80)

# Create forecaster with the fitted model
print("\nInitializing forecaster with fitted model...")
forecaster = MDSVForecaster(model)

# Generate rolling forecasts
print("\nGenerating one-step ahead forecasts...")
print("Using expanding window approach (re-estimating is too slow)...\n")

predictions_rv = []
actuals_rv = []
prediction_errors = []

# For each test point, use all data up to that point to forecast
for i in range(len(test_data)):
    try:
        # Data available up to time t (for forecasting t+1)
        if i == 0:
            # First forecast: use only training data
            available_returns = train_returns
            available_rv = train_rv
        else:
            # Subsequent forecasts: add observed test data up to (but not including) current point
            available_returns = np.concatenate([
                train_returns,
                test_data['demeaned_log_return'].iloc[:i].values
            ])
            available_rv = np.concatenate([
                train_rv,
                test_data['realized_variance'].iloc[:i].values
            ])

        available_data = np.column_stack([available_returns, available_rv])

        # Get last observation for conditioning (leverage effect)
        last_obs = available_data[-1:]

        # Forecast 1-step ahead
        forecast = forecaster.forecast(
            n_ahead=1,
            last_obs=last_obs,
            n_simulations=10000  # More simulations for better accuracy
        )

        # Extract RV forecast (handle different return formats)
        if isinstance(forecast, dict):
            if 'rv' in forecast:
                pred_rv = forecast['rv'][0] if hasattr(forecast['rv'], '__len__') else forecast['rv']
            elif 'volatility' in forecast:
                pred_rv = forecast['volatility'][0] if hasattr(forecast['volatility'], '__len__') else forecast['volatility']
            else:
                # Try to get first value from forecast dict
                pred_rv = list(forecast.values())[0]
                if hasattr(pred_rv, '__len__'):
                    pred_rv = pred_rv[0]
        else:
            pred_rv = forecast[0] if hasattr(forecast, '__len__') else forecast

        predictions_rv.append(pred_rv)
        actuals_rv.append(test_data['realized_variance'].iloc[i])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Forecast {i+1}/{len(test_data)}: predicted={pred_rv:.4f}, actual={test_data['realized_variance'].iloc[i]:.4f}")

    except Exception as e:
        print(f"  Error at forecast {i+1}: {e}")
        predictions_rv.append(np.nan)
        actuals_rv.append(test_data['realized_variance'].iloc[i])
        prediction_errors.append(str(e))

predictions_rv = np.array(predictions_rv)
actuals_rv = np.array(actuals_rv)

print(f"\nCompleted {len(predictions_rv)} forecasts")

# ============================================================================
# STEP 5: Evaluate Performance
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FORECAST EVALUATION")
print("="*80)

# Remove NaN predictions
valid_mask = ~np.isnan(predictions_rv) & ~np.isnan(actuals_rv)
predictions_valid = predictions_rv[valid_mask]
actuals_valid = actuals_rv[valid_mask]

print(f"\nValid forecasts: {len(predictions_valid)}/{len(predictions_rv)}")

if len(predictions_valid) > 10:  # Need reasonable sample size
    # Calculate error metrics
    errors = predictions_valid - actuals_valid
    squared_errors = errors ** 2
    abs_errors = np.abs(errors)

    rmse = np.sqrt(np.mean(squared_errors))
    mae = np.mean(abs_errors)
    mape = np.mean(np.abs(errors / actuals_valid)) * 100

    # Median metrics (more robust to outliers)
    median_ae = np.median(abs_errors)

    print(f"\nMDSV Forecast Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Median AE: {median_ae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    print(f"\nActual RV statistics (test period):")
    print(f"  Mean: {actuals_valid.mean():.4f}")
    print(f"  Std: {actuals_valid.std():.4f}")
    print(f"  Median: {np.median(actuals_valid):.4f}")

    print(f"\nPredicted RV statistics:")
    print(f"  Mean: {predictions_valid.mean():.4f}")
    print(f"  Std: {predictions_valid.std():.4f}")
    print(f"  Median: {np.median(predictions_valid):.4f}")

    # Naive benchmark (persistence/random walk model)
    print(f"\n" + "-"*80)
    print("BENCHMARK COMPARISON")
    print("-"*80)

    # Persistence: RV_t predicts RV_{t+1}
    naive_predictions = test_data['realized_variance'].iloc[:-1].values
    naive_actuals = test_data['realized_variance'].iloc[1:].values
    naive_mask = ~np.isnan(naive_predictions) & ~np.isnan(naive_actuals)

    naive_rmse = np.sqrt(np.mean((naive_predictions[naive_mask] - naive_actuals[naive_mask]) ** 2))
    naive_mae = np.mean(np.abs(naive_predictions[naive_mask] - naive_actuals[naive_mask]))

    print(f"\nNaive (Persistence) Forecast:")
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
    print("\n✗ Insufficient valid forecasts for evaluation!")
    if len(prediction_errors) > 0:
        print(f"\nSample errors encountered:")
        for err in prediction_errors[:5]:
            print(f"  - {err}")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 6: SAVING RESULTS")
print("="*80)

output_path = root / 'data' / 'results'
output_path.mkdir(parents=True, exist_ok=True)

# Save predictions
forecast_df = pd.DataFrame({
    'date': test_data['date'].values,
    'actual_rv': actuals_rv,
    'predicted_rv': predictions_rv,
    'error': actuals_rv - predictions_rv,
    'squared_error': (actuals_rv - predictions_rv) ** 2,
    'abs_error': np.abs(actuals_rv - predictions_rv)
})

forecast_df.to_csv(output_path / 'mdsv_forecasts_2015_2024.csv', index=False)
print(f"\n✓ Forecasts saved to: {output_path / 'mdsv_forecasts_2015_2024.csv'}")

# Save model info
model_info = {
    'model_specification': {
        'N': model.N,
        'D': model.D,
        'model_type': model.model_type,
        'leverage': model.leverage
    },
    'estimation_results': {
        'log_likelihood': float(result.log_likelihood),
        'aic': float(result.aic),
        'bic': float(result.bic),
        'success': result.success if hasattr(result, 'success') else None
    },
    'data_info': {
        'train_size': len(train_data),
        'test_size': len(test_data),
        'train_period': f"{train_data['date'].min()} to {train_data['date'].max()}",
        'test_period': f"{test_data['date'].min()} to {test_data['date'].max()}"
    },
    'forecast_performance': {
        'valid_forecasts': int(len(predictions_valid)),
        'total_forecasts': int(len(predictions_rv)),
        'rmse': float(rmse) if len(predictions_valid) > 10 else None,
        'mae': float(mae) if len(predictions_valid) > 10 else None,
        'mape': float(mape) if len(predictions_valid) > 10 else None,
        'naive_rmse': float(naive_rmse) if len(predictions_valid) > 10 else None,
        'improvement_pct': float(improvement_rmse) if len(predictions_valid) > 10 else None
    }
}

import json
with open(output_path / 'mdsv_model_info_2015_2024.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"✓ Model info saved to: {output_path / 'mdsv_model_info_2015_2024.json'}")

print("\n" + "="*80)
print("PIPELINE COMPLETED")
print("="*80)
#%%
