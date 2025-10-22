import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path

# Setup paths
root = Path().resolve().parent
cboe_path = root / 'data' / 'cboe'

# Define the years you want to process
years = range(2015, 2025)  # 2015 to 2024

# List to store daily data from each year
all_daily_data = []
previous_year_last_price = None  # Track last price from previous year

print("Processing data from 2015 to 2024...")

for year in years:
    year_path = cboe_path / str(year)

    # Check if the year directory exists
    if not year_path.exists():
        print(f"Warning: Directory for {year} not found, skipping...")
        continue

    print(f"\nProcessing year {year}...")

    try:
        # Load data with Polars and convert to pandas
        lazy_df = pl.scan_parquet(year_path / '*.gzip.parquet')
        df = lazy_df.collect().to_pandas()

        print(f"  Loaded {len(df)} rows for {year}")

        # Step 1: Get underlying prices at each unique datetime
        underlying_prices = (
            df[['quote_datetime', 'active_underlying_price']]
            .drop_duplicates(subset=['quote_datetime'])
            .sort_values('quote_datetime')
            .reset_index(drop=True)
        )

        # Free up memory
        del df

        # Step 2: Calculate intraday log returns
        underlying_prices['log_return'] = (
            np.log(underlying_prices['active_underlying_price']) -
            np.log(underlying_prices['active_underlying_price'].shift(1))
        )

        # Calculate squared log returns for realized variance
        underlying_prices['squared_log_return'] = underlying_prices['log_return'] ** 2

        # Step 3: Extract date for grouping
        underlying_prices['date'] = pd.to_datetime(underlying_prices['quote_datetime']).dt.date

        # Step 4: Calculate DAILY realized variance
        daily_data = underlying_prices.groupby('date').agg({
            'squared_log_return': 'sum',
            'quote_datetime': 'count'
        }).reset_index()

        daily_data.columns = ['date', 'sum_squared_returns', 'n_obs']
        daily_data['realized_variance'] = daily_data['sum_squared_returns'] * (100 ** 2)

        # Step 5: Calculate DAILY close-to-close log returns
        daily_close_prices = underlying_prices.groupby('date').agg({
            'active_underlying_price': 'last'
        }).reset_index()

        daily_close_prices.columns = ['date', 'close_price']

        # Calculate daily returns, handling the boundary with previous year
        daily_close_prices['daily_log_return'] = (
            np.log(daily_close_prices['close_price']) -
            np.log(daily_close_prices['close_price'].shift(1))
        )

        # Fill the first day's return using previous year's last price
        if previous_year_last_price is not None:
            first_day_return = (
                np.log(daily_close_prices.loc[0, 'close_price']) -
                np.log(previous_year_last_price)
            )
            daily_close_prices.loc[0, 'daily_log_return'] = first_day_return

        # Store last price for next year
        previous_year_last_price = daily_close_prices.loc[len(daily_close_prices)-1, 'close_price']

        daily_close_prices['daily_log_return_pct'] = daily_close_prices['daily_log_return'] * 100

        # Step 6: Merge daily returns with realized variance
        daily_data = daily_data.merge(
            daily_close_prices[['date', 'daily_log_return_pct']],
            on='date',
            how='left'
        )

        # Add year column for reference
        daily_data['year'] = year

        # Append to list
        all_daily_data.append(daily_data)

        print(f"  Processed {len(daily_data)} daily observations for {year}")

        # Free up memory
        del underlying_prices, daily_close_prices

    except Exception as e:
        print(f"  Error processing {year}: {str(e)}")
        continue

# Concatenate all years
print("\nCombining all years...")
combined_daily_data = pd.concat(all_daily_data, ignore_index=True)

# Free up memory
del all_daily_data

# Sort by date
combined_daily_data = combined_daily_data.sort_values('date').reset_index(drop=True)

print(f"\nTotal daily observations across all years: {len(combined_daily_data)}")
print(f"Date range: {combined_daily_data['date'].min()} to {combined_daily_data['date'].max()}")

# Step 7: Demean the daily returns across ALL years (skip NaN if first row of first year)
mean_return = combined_daily_data['daily_log_return_pct'].mean()
combined_daily_data['demeaned_log_return'] = combined_daily_data['daily_log_return_pct'] - mean_return

print(f"Mean return across all years: {mean_return:.6f}%")

# Step 8: Create forward-looking realized variance
combined_daily_data['forward_realized_variance'] = combined_daily_data['realized_variance'].shift(-1)

# Step 9: Remove first row if it has NaN return (first day of 2015) and last row (no forward RV)
if pd.isna(combined_daily_data.loc[0, 'daily_log_return_pct']):
    combined_daily_data = combined_daily_data.iloc[1:-1].reset_index(drop=True)
else:
    combined_daily_data = combined_daily_data.iloc[:-1].reset_index(drop=True)

# Step 10: Save to CSV
output_path = root / 'data' / 'processed' / 'daily_data_2015_2024.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
combined_daily_data.to_csv(output_path, index=False)

print(f"\nData saved to: {output_path}")
print(f"Final shape: {combined_daily_data.shape}")
print(f"\nChecking for NaN values:")
print(combined_daily_data.isnull().sum())
print("\nSample of final data:")
print(combined_daily_data.head())