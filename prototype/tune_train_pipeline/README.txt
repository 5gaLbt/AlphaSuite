
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

    tickers1 = ['SPY', 'WMT', 'T', 'JPM', 'BAC', 'C', 'CAT', 'FDX',
               'PFE', 'COST', 'AMZN', 'AAPL', 'INTC', 'DIS', 'HD',
               'NFLX', 'UNH', 'PG', 'KO', 'CSCO', 'BA']

    tickers2 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX', 'ADBE',
     'CRM', 'NOW', 'ASML', 'TSM', 'CAT', 'DE', 'UNP', 'JPM', 'GS', 'AAL', 'JNJ',
     'UNH', 'PG', 'KO', 'NEE', 'DUK', 'PLUG', 'RIVN', 'HOOD', 'DNN']

    tickers3 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'NOW', 'ASML', 'TSM',
     'CAT', 'DE', 'UNP', 'JPM', 'GS', 'AAL', 'JNJ', 'UNH', 'PG', 'KO', 'NEE', 'DUK', 'PLUG', 'RIVN', 'HOOD', 'DNN',
     'AMD', 'INTC', 'CSCO', 'ORCL', 'PFE', 'MRK', 'DIS', 'BAC', 'WMT', 'HD', 'XOM', 'CVX', 'NKE', 'MCD', 'T', 'SOFI',
     'CCL', 'GM', 'PYPL']

==========================================================================

====== structure_liquidity-ticker1-2021-01-01-2025-12-31 ========
                      name        trained_value    passthrough value
0              trade_count                   70                   89
1     initial_market_value             100000.0             100000.0
2         end_market_value            161675.21            142976.67
3                total_pnl             62175.35             43456.27
4           unrealized_pnl              -500.14               -479.6
5         total_return_pct             62.17535             43.45627
6             total_profit             92489.27             85022.27
7               total_loss            -30313.92             -41566.0
8               total_fees                  0.0                  0.0
9             max_drawdown            -17724.33            -19225.88
10        max_drawdown_pct            -11.62925           -14.320044
11       max_drawdown_date  2025-04-08 00:00:00  2025-04-08 00:00:00
12                win_rate            44.285714             44.94382
13               loss_rate            55.714286             55.05618
14          winning_trades                   31                   40
15           losing_trades                   39                   49
16                 avg_pnl           888.219286           488.272697
17          avg_return_pct                7.297             6.265056
18          avg_trade_bars            69.028571            60.483146
19              avg_profit          2983.524839           2125.55675
20          avg_profit_pct            23.997097              21.7055
21  avg_winning_trade_bars           121.967742               110.05
22                avg_loss              -777.28          -848.285714
23            avg_loss_pct            -5.977436            -6.339388
24   avg_losing_trade_bars            26.948718            20.020408
25             largest_win             16125.12              6546.77
26         largest_win_pct                76.97                51.66
27        largest_win_bars                  268                  112
28            largest_loss             -1721.83             -1951.27
29        largest_loss_pct                -5.25               -13.59
30       largest_loss_bars                   17                   17
31                max_wins                    4                    5
32              max_losses                    7                   11
33                  sharpe             0.114535              0.07858
34                 sortino             0.164318              0.10859
35           profit_factor              1.33579             1.219767
36             ulcer_index             1.704923             2.068908
37                     upi             0.046432             0.029026
38               equity_r2             0.958851             0.889431
39               std_error         23193.831375         14936.523312

====== structure_liquidity-ticker2-2021-01-01-2025-12-31 ========
                      name        trained_value    passthrough value
0              trade_count                   95                   94
1     initial_market_value             100000.0             100000.0
2         end_market_value            182173.32            194620.32
3                total_pnl             82720.24             95261.87
4           unrealized_pnl              -546.92              -641.55
5         total_return_pct             82.72024             95.26187
6             total_profit            134613.03             133504.2
7               total_loss            -51892.79            -38242.33
8               total_fees                  0.0                  0.0
9             max_drawdown             -26491.9            -21753.25
10        max_drawdown_pct           -16.490695           -12.392073
11       max_drawdown_date  2025-04-08 00:00:00  2025-04-08 00:00:00
12                win_rate            48.421053             52.12766
13               loss_rate            51.578947             47.87234
14          winning_trades                   46                   49
15           losing_trades                   49                   45
16                 avg_pnl           870.739368          1013.424149
17          avg_return_pct            11.599053            10.858404
18          avg_trade_bars            61.094737            63.670213
19              avg_profit          2926.370217           2724.57551
20          avg_profit_pct            34.664783            29.357143
21  avg_winning_trade_bars           105.543478           102.673469
22                avg_loss         -1059.036531          -849.829556
23            avg_loss_pct            -10.05449            -9.284667
24   avg_losing_trade_bars            19.367347                 21.2
25             largest_win             12449.58             12163.53
26         largest_win_pct               104.05               267.91
27        largest_win_bars                  182                  133
28            largest_loss             -1896.83             -2048.97
29        largest_loss_pct               -18.44                -9.68
30       largest_loss_bars                    8                   23
31                max_wins                   10                    8
32              max_losses                    8                   10
33                  sharpe             0.104875               0.1216
34                 sortino             0.151361             0.186158
35           profit_factor             1.310329             1.359149
36             ulcer_index             2.381198              2.11853
37                     upi             0.042182             0.052184
38               equity_r2              0.92618              0.97236
39               std_error         25853.328641         31889.348519

====== structure_liquidity-ticker3-2021-01-01-2025-12-31 ========
                      name        trained_value    passthrough value
0              trade_count                   92                   98
1     initial_market_value             100000.0             100000.0
2         end_market_value            204400.18             205918.4
3                total_pnl            104896.48            106340.89
4           unrealized_pnl               -496.3              -422.49
5         total_return_pct            104.89648            106.34089
6             total_profit            150919.76            153122.21
7               total_loss            -46023.28            -46781.32
8               total_fees                  0.0                  0.0
9             max_drawdown            -26417.55            -25451.95
10        max_drawdown_pct           -14.805578           -13.924341
11       max_drawdown_date  2025-04-08 00:00:00  2025-04-08 00:00:00
12                win_rate                 50.0            54.081633
13               loss_rate                 50.0            45.918367
14          winning_trades                   46                   53
15           losing_trades                   46                   45
16                 avg_pnl           1140.17913          1085.111122
17          avg_return_pct            12.535761            13.115204
18          avg_trade_bars            65.032609            62.081633
19              avg_profit          3280.864348          2889.098302
20          avg_profit_pct            34.041957            30.363962
21  avg_winning_trade_bars                110.0                 98.0
22                avg_loss         -1000.506087         -1039.584889
23            avg_loss_pct            -8.970435                 -7.2
24   avg_losing_trade_bars            20.065217            19.777778
25             largest_win              12068.5             12543.64
26         largest_win_pct               267.91               267.91
27        largest_win_bars                  133                  133
28            largest_loss             -1869.07              -2134.6
29        largest_loss_pct               -10.05                -9.26
30       largest_loss_bars                    9                   20
31                max_wins                    7                   10
32              max_losses                    7                    7
33                  sharpe             0.122914             0.124178
34                 sortino             0.179124             0.181363
35           profit_factor             1.373397             1.368972
36             ulcer_index             2.352581             2.262631
37                     upi             0.050557             0.053091
38               equity_r2             0.906434             0.965075
39               std_error         30141.107589         33592.105544

====== donchian_breakout-ticker1-2021-01-01-2025-12-31 ========
                      name        trained_value    passthrough value
0              trade_count                   84                  154
1     initial_market_value             100000.0             100000.0
2         end_market_value            174010.09            163134.22
3                total_pnl             74172.54              63648.9
4           unrealized_pnl              -162.45              -514.68
5         total_return_pct             74.17254              63.6489
6             total_profit            105809.33            120890.18
7               total_loss            -31636.79            -57241.28
8               total_fees                  0.0                  0.0
9             max_drawdown            -11005.33            -18018.13
10        max_drawdown_pct             -6.84639           -11.475447
11       max_drawdown_date  2025-09-02 00:00:00  2025-04-07 00:00:00
12                win_rate            53.571429            49.350649
13               loss_rate            46.428571            50.649351
14          winning_trades                   45                   76
15           losing_trades                   39                   78
16                 avg_pnl           883.006429           413.304545
17          avg_return_pct             2.114762             1.214156
18          avg_trade_bars            15.916667            14.987013
19              avg_profit          2351.318444          1590.660263
20          avg_profit_pct             6.283333             5.366711
21  avg_winning_trade_bars            21.755556            21.815789
22                avg_loss          -811.199744          -733.862564
23            avg_loss_pct            -2.695128            -2.831923
24   avg_losing_trade_bars             9.179487             8.333333
25             largest_win              8380.19              8038.61
26         largest_win_pct                14.02                 16.3
27        largest_win_bars                   48                   55
28            largest_loss             -3990.62             -3650.57
29        largest_loss_pct                 -4.9                -5.49
30       largest_loss_bars                    4                    1
31                max_wins                    5                   11
32              max_losses                    5                    9
33                  sharpe             0.120388             0.094412
34                 sortino             0.173316              0.14649
35           profit_factor             1.433595             1.272593
36             ulcer_index             1.724817             1.985173
37                     upi             0.052987             0.041291
38               equity_r2             0.949144             0.940547
39               std_error         21494.187406         22656.995562

====== donchian_breakout-ticker2-2021-01-01-2025-12-31 ========
                      name        trained_value    passthrough value
0              trade_count                  119                  206
1     initial_market_value             100000.0             100000.0
2         end_market_value            176626.15             163499.3
3                total_pnl             77246.16             63933.68
4           unrealized_pnl              -620.01              -434.38
5         total_return_pct             77.24616             63.93368
6             total_profit             140242.5            151060.76
7               total_loss            -62996.34            -87127.08
8               total_fees                  0.0                  0.0
9             max_drawdown            -14795.22            -17042.95
10        max_drawdown_pct           -11.426331           -16.161813
11       max_drawdown_date  2024-11-01 00:00:00  2023-10-27 00:00:00
12                win_rate             47.89916            47.087379
13               loss_rate             52.10084            52.912621
14          winning_trades                   57                   97
15           losing_trades                   62                  109
16                 avg_pnl           649.127395            310.35767
17          avg_return_pct             2.018151             1.914272
18          avg_trade_bars            14.806723            13.684466
19              avg_profit          2460.394737          1557.327423
20          avg_profit_pct              8.67193             9.017526
21  avg_winning_trade_bars            22.192982            20.597938
22                avg_loss             -1016.07          -799.331009
23            avg_loss_pct            -4.099032            -4.406972
24   avg_losing_trade_bars             8.016129              7.53211
25             largest_win             13692.66             10220.83
26         largest_win_pct                27.91                85.42
27        largest_win_bars                   21                   48
28            largest_loss             -3758.04              -3087.3
29        largest_loss_pct                -9.75               -11.33
30       largest_loss_bars                    4                    3
31                max_wins                    6                    8
32              max_losses                    6                    8
33                  sharpe             0.099761             0.078721
34                 sortino             0.150412             0.113764
35           profit_factor             1.336522             1.242003
36             ulcer_index             2.338363             2.612604
37                     upi             0.040833             0.032264
38               equity_r2             0.829438             0.860425
39               std_error         20619.714303         21564.148409

====== donchian_breakout-ticker3-2021-01-01-2025-12-31 ========
                      name        trained_value    passthrough value
0              trade_count                  152                  267
1     initial_market_value             100000.0             100000.0
2         end_market_value            164437.48            145939.96
3                total_pnl             65055.01             46594.99
4           unrealized_pnl              -617.53              -655.03
5         total_return_pct             65.05501             46.59499
6             total_profit            145317.36            125860.92
7               total_loss            -80262.35            -79265.93
8               total_fees                  0.0                  0.0
9             max_drawdown            -18344.91            -23237.22
10        max_drawdown_pct           -13.527678           -18.492323
11       max_drawdown_date  2025-05-06 00:00:00  2023-10-27 00:00:00
12                win_rate            44.078947            41.947566
13               loss_rate            55.921053            58.052434
14          winning_trades                   67                  112
15           losing_trades                   85                  155
16                 avg_pnl           427.993487           174.513071
17          avg_return_pct             1.610526             1.151348
18          avg_trade_bars            13.335526            13.018727
19              avg_profit          2168.915821          1123.758214
20          avg_profit_pct              8.92403             8.254107
21  avg_winning_trade_bars            20.567164            20.348214
22                avg_loss          -944.262941          -511.393097
23            avg_loss_pct            -4.154235            -3.980968
24   avg_losing_trade_bars             7.635294             7.722581
25             largest_win             12367.57              7402.09
26         largest_win_pct                27.91                51.78
27        largest_win_bars                   21                   45
28            largest_loss             -3758.04             -2432.88
29        largest_loss_pct                -9.75                -8.93
30       largest_loss_bars                    4                    3
31                max_wins                    6                   10
32              max_losses                    7                    9
33                  sharpe             0.085609             0.060861
34                 sortino             0.141744             0.092273
35           profit_factor             1.266521             1.186196
36             ulcer_index             2.527915             2.793468
37                     upi             0.033349             0.023744
38               equity_r2             0.701475              0.72581
39               std_error         16429.104749          17167.90985

==========================================================================
