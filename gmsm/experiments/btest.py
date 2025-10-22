"""
Test script for the optimized long-only Polars backtester
Demonstrates performance improvements and includes IV tracking
"""

import polars as pl
from pathlib import Path
import time
from gmsm.models.mdsv.src.backtester import OptionsBacktesterPolars


def load_data_polars(data_dir: Path, use_sample: bool = False):
    """Load predictions and options data using Polars"""
    print("Loading data with Polars...")

    # Load predictions
    print(f"Data directory: {data_dir}")
    forecast_path = data_dir / 'results' / 'forecasts.csv'

    if not forecast_path.exists():
        # Try alternative path
        forecast_path = data_dir / 'mdsv_forecasts_2024_rolling_90days_nolev.csv'

    if not forecast_path.exists():
        raise FileNotFoundError(f"Could not find forecast file at {forecast_path}")

    forecast_df = pl.read_csv(str(forecast_path))
    print(f"✓ Loaded {len(forecast_df):,} forecast rows")

    # Load options data
    if use_sample:
        # For testing, use sample data
        sample_path = data_dir / 'options_example.csv'
        if not sample_path.exists():
            sample_path = data_dir.parent / 'options_example.csv'

        options_df = pl.read_csv(str(sample_path))
        print(f"✓ Loaded {len(options_df):,} sample option rows")
        return forecast_df, options_df
    else:
        # Full dataset - use lazy loading for efficiency
        root = data_dir.parent
        cboe_path = root / 'gmsm' / 'data' / 'cboe' / '2024'

        if not cboe_path.exists():
            print(f"⚠️  Full options data not found at {cboe_path}")
            print("Falling back to sample data...")
            return load_data_polars(data_dir, use_sample=True)

        pattern = str(cboe_path / '*.gzip.parquet')
        options_lf = pl.scan_parquet(pattern)
        print(f"✓ Connected to options data at: {cboe_path}")
        return forecast_df, options_lf


def run_backtest_suite():
    """Run backtests with different configurations"""

    # Setup paths
    root = Path().resolve()
    data_dir = root / 'data'

    if not data_dir.exists():
        data_dir = root  # Try current directory

    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    # Load data using Polars
    print("\n" + "="*70)
    print("DATA LOADING")
    print("="*70)

    start_time = time.time()
    try:
        forecast_df, options_df = load_data_polars(data_dir, use_sample=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Attempting to load sample data...")
        forecast_df, options_df = load_data_polars(data_dir, use_sample=True)

    load_time = time.time() - start_time
    print(f"\n✓ Data loading time: {load_time:.2f} seconds")

    # Test configurations - all LONG ONLY
    configs = [
        {
            'name': 'Conservative_1M_Straddles',
            'strategy': 'straddle',
            'threshold': 0.03,  # 3% signal required
            'maturity': '1M',
            'description': 'Buy straddles when predicted vol > market IV by 3%+'
        },
        {
            'name': 'Moderate_1M_Strangles',
            'strategy': 'strangle',
            'threshold': 0.02,  # 2% signal required
            'maturity': '1M',
            'description': 'Buy strangles when predicted vol > market IV by 2%+'
        },
        {
            'name': 'Aggressive_2W_Both',
            'strategy': 'both',
            'threshold': 0.01,  # 1% signal required
            'maturity': '2W',
            'description': 'Buy both straddles & strangles on 1%+ signals (2-week options)'
        },
        {
            'name': 'AllMaturities_Straddles',
            'strategy': 'straddle',
            'threshold': 0.025,  # 2.5% signal required
            'maturity': None,  # All maturities
            'description': 'Buy straddles across all maturities with 2.5%+ signal'
        }
    ]

    results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*70}")

        start_time = time.time()

        # Initialize backtester
        backtester = OptionsBacktesterPolars(
            predictions_df=forecast_df,
            options_df=options_df,
            initial_capital=100000,
            strategy_type=config['strategy'],
            threshold=config['threshold'],
            base_position_size=10000,
            signal_multiplier=2.0,
            maturity_filter=config.get('maturity'),
            output_dir=output_dir
        )

        # Run backtest
        try:
            trades_df, equity_df, iv_data_df = backtester.run_backtest()
            backtest_time = time.time() - start_time

            # Store results
            result = {
                'config': config['name'],
                'strategy': config['strategy'],
                'threshold': config['threshold'],
                'maturity': config.get('maturity', 'all'),
                'trades': len(trades_df) if len(trades_df) > 0 else 0,
                'time_seconds': backtest_time
            }

            if len(trades_df) > 0:
                final_equity = equity_df['equity'][-1]
                result['final_capital'] = final_equity
                result['total_return_pct'] = (final_equity / 100000 - 1) * 100
                result['avg_return_per_trade'] = trades_df['return_pct'].mean()
                result['win_rate'] = (trades_df['net_pnl'] > 0).mean() * 100
                result['best_trade'] = trades_df['net_pnl'].max()
                result['worst_trade'] = trades_df['net_pnl'].min()
                result['sharpe_ratio'] = (trades_df['return_pct'].mean() /
                                        trades_df['return_pct'].std()) if len(trades_df) > 1 else 0
            else:
                result['final_capital'] = 100000
                result['total_return_pct'] = 0
                result['avg_return_per_trade'] = 0
                result['win_rate'] = 0
                result['best_trade'] = 0
                result['worst_trade'] = 0
                result['sharpe_ratio'] = 0

            results.append(result)

            print(f"\n✓ Backtest completed in {backtest_time:.2f} seconds")
            print(f"  Trades executed: {result['trades']}")
            print(f"  Total return: {result['total_return_pct']:.2f}%")
            print(f"  Win rate: {result['win_rate']:.1f}%")

            if len(iv_data_df) > 0:
                print(f"  IV data points: {len(iv_data_df)}")

        except Exception as e:
            print(f"\n✗ Error in backtest: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    if len(results) > 0:
        print(f"\n{'='*70}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*70}")

        results_df = pl.DataFrame(results)
        print(results_df)

        # Save summary results
        summary_path = output_dir / 'backtest_comparison.csv'
        results_df.write_csv(str(summary_path))
        print(f"\n✓ Results saved to: {summary_path}")

        # Print best performing strategy
        best_idx = results_df['total_return_pct'].arg_max()
        best_config = results_df[best_idx]
        print(f"\n🏆 Best Strategy: {best_config['config'][0]}")
        print(f"   Return: {best_config['total_return_pct'][0]:.2f}%")
        print(f"   Win Rate: {best_config['win_rate'][0]:.1f}%")
        print(f"   Sharpe: {best_config['sharpe_ratio'][0]:.2f}")


def analyze_iv_data(output_dir: Path = Path('results')):
    """Analyze the IV data saved by the backtester"""
    print(f"\n{'='*70}")
    print("ANALYZING IMPLIED VOLATILITY DATA")
    print(f"{'='*70}")

    # Find all IV files
    iv_files = list(output_dir.glob('iv_data_*.csv'))

    if not iv_files:
        print("No IV data files found. Run backtests first.")
        return

    for iv_file in iv_files:
        print(f"\nAnalyzing: {iv_file.name}")

        iv_df = pl.read_csv(str(iv_file))

        if len(iv_df) == 0:
            print("  No data in file")
            continue

        print(f"  Data points: {len(iv_df)}")
        print(f"  Date range: {iv_df['date'].min()} to {iv_df['date'].max()}")

        # Calculate statistics
        print("\n  Volatility Statistics:")
        print(f"    Predicted vol: {iv_df['predicted_vol'].mean():.4f} ± {iv_df['predicted_vol'].std():.4f}")
        print(f"    Market IV:     {iv_df['market_iv'].mean():.4f} ± {iv_df['market_iv'].std():.4f}")
        print(f"    Signal:        {iv_df['signal_strength'].mean():.4f} ± {iv_df['signal_strength'].std():.4f}")

        # Signal distribution
        positive_signals = (iv_df['signal_strength'] > 0).sum()
        print(f"\n  Signal Distribution:")
        print(f"    Positive signals: {positive_signals} ({positive_signals/len(iv_df)*100:.1f}%)")
        print(f"    Negative signals: {len(iv_df) - positive_signals} ({(1-positive_signals/len(iv_df))*100:.1f}%)")

        # When we actually traded
        if 'traded' in iv_df.columns:
            traded = iv_df['traded'].sum()
            print(f"\n  Trading Activity:")
            print(f"    Days traded: {traded} ({traded/len(iv_df)*100:.1f}%)")

        # Correlation between predicted and market
        corr = iv_df['predicted_vol'].corr(iv_df['market_iv'])
        print(f"\n  Correlation (predicted vs market): {corr:.3f}")

if __name__ == '__main__':
    # Run the backtest suite
    print("\n" + "="*70)
    print("RUNNING BACKTEST SUITE")
    print("="*70)

    try:
        run_backtest_suite()
    except Exception as e:
        print(f"\n✗ Error running backtests: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure you have the required data files:")
        print("  - Forecasts: data/results/forecasts.csv or mdsv_forecasts_2024_rolling_90days_nolev.csv")
        print("  - Options: options_example.csv or gmsm/data/cboe/2024/*.gzip.parquet")

    # Analyze IV data if available
    try:
        analyze_iv_data()
    except Exception as e:
        print(f"\n⚠️  Could not analyze IV data: {e}")

    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print("\nOutput files in 'results/' directory:")
    print("  - trades_*.csv: Individual trade details")
    print("  - equity_*.csv: Portfolio value over time")
    print("  - iv_data_*.csv: Market IV vs predictions")
    print("  - backtest_comparison.csv: Summary of all configurations")