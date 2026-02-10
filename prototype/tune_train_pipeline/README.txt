
All of those py files are extracted from quant_engine.py, and first refactored to Walkforward class.
The new Workforward class are able to accept a portfolio (multiple tickers) to tune and train,
the class structure is easier/more flexible to extend.
In theory, training a single model on multiple tickers (often called Global Modeling or Cross-Sectional Training)
is a common practice to give the model more data points and help it learn general market patterns.

In the pipeline,
def run_pipeline_tune_train_backtest(
        portfolio_name,
        tickers, strategy_type, start_date, end_date,
        backtest_portfolio_name,
        backtest_tickers, backtest_start_date, backtest_end_date):
    start = datetime.now()
    # 1. tune strategy parameters and copy  _best_strategy_params.json to strategy_configs directory
    run_tune_strategy(portfolio_name, tickers, strategy_type, start_date, end_date)
    copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}_best_strategy_params.json", f"{strategy_type}.json")

    # 2. tune hyperparameters and copy _best_params.json to strategy_configs directory
    run_tune_hyper_params(portfolio_name, tickers, strategy_type, start_date, end_date)
    copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}_best_params.json", f"{strategy_type}_hyper_params.json")

    # 3. train model and copy .pkl file to strategy_configs directory
    run_train_model(portfolio_name, tickers, strategy_type, start_date, end_date)
    copy_to_strategy_configs(f"{portfolio_name}_{strategy_type}.pkl", f"{strategy_type}_model.pkl")

    # 4. use the tuned strategy_parameter, trained model to run backtest out of train window to see result
    run_backtest_portfolio(backtest_portfolio_name, backtest_tickers, strategy_type, backtest_start_date, backtest_end_date)


The above provide a structure/framework/pipeline as a starting point, we can try difference portfolios, model configurations(scaler),
strategy/features parameters and ... more. and use the results/metrix to evaluate and improve overall ML performance.

You can run run_backtest(use TRAINED model), and run_backtest_passthrough(use PASSTHROUGH model) to compare results.

or you can run run_backtest_compare.py, if you have already trained 'structure_liquidity' and 'donchian_breakout',
it will compare TRAINED and PASSTHROUGH with combination of two strategies, three tickers set, print out concat metrix_df result.
