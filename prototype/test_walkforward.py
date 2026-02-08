import pandas as pd

from prototype.walkforward import WalkForward


def print_full(s):
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.max_colwidth', None,
        'display.width', None,  # Ensures the line doesn't break
        'display.expand_frame_repr', False  # Prevents wrapping to a new line
    ):
        print(s)


'''
@cli.command()
@click.option('--ticker', '-t', required=True, help='Stock ticker symbol.')
@click.option('--strategy-type', '-s', required=True, type=click.Choice(list(STRATEGY_CLASS_MAP.keys())),
              help='The type of strategy to train.')
@click.option('--tune/--no-tune', default=True, help='Enable/disable hyperparameter tuning. Default is enabled.')
@click.option('--plot/--no-plot', default=True,
              help='Plot results immediately after training. Default is disabled.')
@click.option('--start-date', default='2000-01-01', help='Start date for backtest (YYYY-MM-DD).')
@click.option('--end-date', default=None, help='End date for backtest (YYYY-MM-DD). Defaults to yesterday.')
@click.option('--use-tuned-strategy-params/--no-use-tuned-strategy-params', default=True,
              help='Use best parameters found by tune-strategy.')
@click.option('--override-param', '-o', 'override_params', multiple=True,
              help='Override a specific strategy parameter (e.g., -o risk_per_trade_pct=0.02).')
@click.option('--commission', default=0.0, help='Commission cost per share (e.g., 0.005).')
'''
if __name__ == '__main__':
    """Runs the walk-forward analysis and saves the final model."""
    #print(STRATEGY_CLASS_MAP)

    walkforward = WalkForward()

    tickers = ['SPY', 'WMT', 'T', 'JPM', 'BAC', 'C', 'CAT', 'FDX',
               'PFE', 'COST', 'AMZN', 'AAPL', 'INTC', 'DIS', 'HD',
               'NFLX', 'UNH', 'PG', 'KO', 'CSCO', 'BA']

    result, all_quality_scores = walkforward.run_pybroker_walkforward(
        portfolio_name='20_stocks', # prefix of result files
        tickers=tickers,
        strategy_type='structure_liquidity', #'donchian_breakout',  # 'structure_liquidity'
        start_date='2000-01-01',
        end_date='2020-01-01',
        tune_hyperparameters=True,
        plot_results=False,
        save_assets=True,
    )
    print(f"all_quality_scores={all_quality_scores}")
    print_full(result.metrics_df)
