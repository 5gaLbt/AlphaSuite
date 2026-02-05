import json
import logging
import os
import pickle
import traceback

import numpy as np
import pandas as pd
import pybroker
from pybroker.strategy import WalkforwardWindow
from quant_engine import NumpyEncoder
from tools.file_wrapper import convert_to_json_serializable

logger = logging.getLogger(__name__)


# --- NEW: Custom Strategy class for Expanding Window Walk-Forward ---
class ExpandingWindowStrategy(pybroker.Strategy):
    """
    A custom Strategy class that overrides the default walk-forward split logic
    to implement an expanding window instead of a sliding one. This is necessary
    for versions of pybroker that do not have a `rolling` parameter.
    """

    def walkforward_split(
            self,
            df: pd.DataFrame,
            windows: int,
            lookahead: int,
            train_size: float,  # This parameter is ignored in this implementation
            shuffle: bool = False,
    ) -> iter:
        """
        Generates train/test splits for an expanding window walk-forward analysis.
        The training data grows with each window to include the previous test set.
        """
        logger.info("Using custom ExpandingWindowStrategy to generate expanding window splits...")

        date_col = 'date'  # pybroker lowercases column names
        unique_dates = np.unique(df[date_col])
        n_dates = len(unique_dates)

        if windows <= 0 or n_dates == 0: return

        # The data is divided into `windows + 1` chunks. The first is for initial training.
        test_size_days = n_dates // (windows + 1)
        if test_size_days < 1:
            logger.error(f"Not enough data for {windows} windows. Each test set would have < 1 day.")
            return

        train_end_date_idx = n_dates - (windows * test_size_days) - 1

        for i in range(windows):
            test_start_date_idx = train_end_date_idx + lookahead

            # For the last window, extend the test set to the end of the data to include any remainder.
            if i == windows - 1:
                test_end_date_idx = n_dates - 1
            else:
                test_end_date_idx = test_start_date_idx + test_size_days - 1
            if test_end_date_idx >= n_dates: test_end_date_idx = n_dates - 1
            if test_start_date_idx > test_end_date_idx: break

            train_end_date = unique_dates[train_end_date_idx]
            test_start_date = unique_dates[test_start_date_idx]
            test_end_date = unique_dates[test_end_date_idx]

            train_indices = df.index[df[date_col] <= train_end_date].to_numpy()
            test_indices = df.index[(df[date_col] >= test_start_date) & (df[date_col] <= test_end_date)].to_numpy()
            if shuffle: np.random.shuffle(train_indices)
            yield WalkforwardWindow(train_indices, test_indices)
            train_end_date_idx = test_end_date_idx


def log_walkforward_split_dates(data_df, start_date, end_date, strategy_config, windows, train_size_prop):
    # --- Log the walk-forward split dates for clarity ---
    try:
        # Use the same ExpandingWindowStrategy to visualize the splits that will actually be used.
        temp_strategy_for_split = ExpandingWindowStrategy(data_source=data_df, start_date=start_date,
                                                          end_date=end_date, config=strategy_config)
        logger.info("--- Walk-Forward Analysis Splits ---")
        for i, (train_idx, test_idx) in enumerate(
                temp_strategy_for_split.walkforward_split(data_df, windows=windows, train_size=train_size_prop,
                                                          lookahead=1)):
            try:
                # --- Add a more robust guard against invalid splits from the generator ---
                if len(train_idx) == 0 or len(test_idx) == 0:
                    logger.warning(f"Skipping display for Fold {i + 1} as it contains an empty train/test split.")
                    continue

                train_start_date = data_df.loc[train_idx[0]]['date'].date()
                train_end_date = data_df.loc[train_idx[-1]]['date'].date()
                test_start_date = data_df.loc[test_idx[0]]['date'].date()
                test_end_date = data_df.loc[test_idx[-1]]['date'].date()
                logger.info(f"Fold {i + 1}:")
                logger.info(f"  Train: {train_start_date} to {train_end_date} ({len(train_idx)} bars)")
                logger.info(f"  Test:  {test_start_date} to {test_end_date} ({len(test_idx)} bars)")
            except IndexError:
                logger.warning(
                    f"Skipping display for Fold {i + 1} due to out-of-bounds indices from pybroker splitter.")
                traceback.print_exc()
                continue
        logger.info("------------------------------------")
    except Exception as e:
        logger.warning(f"Could not display walk-forward splits due to an error: {e}")
        traceback.print_exc()


def save_walkforward_artifacts(wf, portfolio_name, result):
    """Helper function to save all assets from a walk-forward run."""
    logger.info(f"--- Saving artifacts for {portfolio_name} - {wf._strategy_type} ---")
    model_dir = os.path.join('../pybroker_trainer', 'artifacts')
    os.makedirs(model_dir, exist_ok=True)

    # --- Always save core backtest artifacts ---
    # 1. Save the full backtest result object
    results_filename = os.path.join(model_dir, f'{portfolio_name}_{wf._strategy_type}_results.pkl')
    with open(results_filename, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"Saved backtest results to {results_filename}")

    # 2. Save the strategy parameters used for the run
    strategy_params_filename = os.path.join(model_dir, f'{portfolio_name}_{wf._strategy_type}_strategy_params.json')
    with open(strategy_params_filename, 'w') as f:
        # --- Add a versioning flag to indicate ratios are annualized ---
        params_to_save = wf._strategy_params.copy()
        params_to_save['ratios_annualized'] = True
        json.dump(params_to_save, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Saved strategy parameters to {strategy_params_filename}")

    if not wf._is_ml or not wf._last_trained_model:
        return

    # 1. Save the model object
    # --- Save ML-specific artifacts only if applicable ---
    model_filename = os.path.join(model_dir, f'{portfolio_name}_{wf._strategy_type}.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(wf._last_trained_model, f)
    logger.info(f"Saved final trained model to {model_filename}")

    # 2. Save the features list
    features_filename = os.path.join(model_dir, f'{portfolio_name}_{wf._strategy_type}_features.json')
    with open(features_filename, 'w') as f:
        json.dump(wf._features, f, indent=4)
    logger.info(f"Saved feature list to {features_filename}")

    # 3. Save feature importances
    importances_filename = os.path.join(model_dir, f'{portfolio_name}_{wf._strategy_type}_importances.pkl')
    with open(importances_filename, 'wb') as f:
        pickle.dump(wf._all_feature_importances, f)
    logger.info(f"Saved feature importances to {importances_filename}")

    # 4. Save the best model hyperparameters if tuning was enabled
    if wf._tune_hyperparameters and wf._last_best_params:
        params_filename = os.path.join(model_dir, f'{portfolio_name}_{wf._strategy_type}_best_params.json')
        with open(params_filename, 'w') as f:
            json.dump(convert_to_json_serializable(wf._last_best_params), f, indent=4)
        logger.info(f"Saved best hyperparameters from last fold to {params_filename}")
