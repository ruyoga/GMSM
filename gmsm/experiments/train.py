import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import time

from gmsm.models.mdsv.src.mdsv import MDSV, MDSVResult
from gmsm.models.mdsv.src.forecasting import MDSVForecaster
import mdsv_cpp

REESTIMATION_INTERVAL = 90

root = Path.resolve()

data_path = root / 'gmsm' / 'data' / 'processed' / 'daily_data_2015_2024'
input_data = pd.read_csv(data_path)
input_data['date'] = pd.to_datetime(input_data['date'])

input_clean = input_data.dropna(
    subset=['demeaned_log_return', 'realized_variance']
).reset_index(drop=True)

train_data = input_clean[
    (input_clean['date'].dt.year >= 2015) &
    (input_clean['date'].dt.year <= 2024)
].copy().reset_index(drop=True)

test_data = input_clean[
    (input_clean['date'].dt.year == 2024)
    ].copy().reset_index(drop=True)

periods = []
n_days = len(test_data)
start_idx = 0
period_num = 1

while start_idx < n_days:
    end_idx = min(start_idx + REESTIMATION_INTERVAL, n_days)
    periods.append({
        'period': period_num,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'n_days': end_idx - start_idx,
        'start_date': test_data['date'].iloc[start_idx],
        'end_date': test_data['date'].iloc[end_idx-1]
    })
    start_idx = end_idx
    period_num += 1

print(f'Created {len(periods)}, periods for re estmation')
for q in periods:
    print(f"  Q{q['quarter']}: Days {q['start_idx'] + 1}-{q['end_idx']} "
          f"({q['n_days']} days) - {q['start_date'].strftime('%Y-%m-%d')} to "
          f"{q['end_date'].strftime('%Y-%m-%d')}")

all_preds = []
all_actuals = []
all_leverage = []
all_dates = []
all_periods = []

period_results = []

previous_parans = None

total_start_time = time.time()

for p_info in periods:
    p_num = p_info['quarter']
    p_start_idx = p_info['start_idx']
    p_end_idx = p_info['end_idx']