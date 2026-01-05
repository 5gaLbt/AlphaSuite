"""
A self-contained module for the 'Structure + Liquidity' trading strategy.

This strategy identifies setups where price "sweeps" a major liquidity zone
(previous swing high or low) and then reclaims the level. It combines this
with market structure context (Higher Lows or Lower Highs) to find high-probability
reversals or trend continuations, which are then qualified by an ML model.
"""

import numpy as np
import pandas as pd

from pybroker_trainer.strategy_sdk import BaseStrategy

class StructureLiquidityStrategy(BaseStrategy):
    """
    Structure + Liquidity Strategy (Sweep & Reclaim).
    
    Core Logic:
    1. Market Structure (Relaxed): 
       - Bullish: Higher Swing Lows.
       - Bearish: Lower Swing Highs.
    2. Liquidity Zones: 
       - The last confirmed swing high and swing low.
    3. Setup (Decoupled from Structure): 
       - Long: Price sweeps below the last swing low and closes back above it.
       - Short: Price sweeps above the last swing high and closes back below it.
    4. ML Filter:
       - Uses Macro Context (SMA 200 distance) and Micro Structure to qualify setups.
    """

    description = "Identifies liquidity sweeps of major swing points followed by a reclaim, filtered by an ML model."

    @staticmethod
    def define_parameters():
        """Defines parameters, their types, defaults, and tuning ranges."""
        return {
            'swing_lookback': {'type': 'int', 'default': 2, 'tuning_range': (2, 7)},
            'atr_period': {'type': 'int', 'default': 14, 'tuning_range': (10, 30)},
            'initial_stop_atr_multiplier': {'type': 'float', 'default': 2.0, 'tuning_range': (1.5, 4.0)},
            'trailing_stop_atr_multiplier': {'type': 'float', 'default': 3.0, 'tuning_range': (2.5, 6.0)},
            'stop_out_window': {'type': 'int', 'default': 60, 'tuning_range': (20, 120)},
            'probability_threshold': {'type': 'float', 'default': 0.52, 'tuning_range': (0.50, 0.75)},
            'risk_per_trade_pct': {'type': 'float', 'default': 0.02, 'tuning_range': (0.01, 0.05)},
            'is_long_only': {'type': 'bool', 'default': True, 'tuning_range': None},
        }

    def get_feature_list(self) -> list[str]:
        """Returns the list of feature column names required by the model."""
        return [
            'structure_bullish', 'structure_bearish',
            'dist_to_swing_high', 'dist_to_swing_low',
            'atr', 'volume', 'rsi', 'roc',
            'dist_to_sma_200', 'is_above_sma_200'
        ]

    def add_strategy_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates swing points, market structure, and features for the ML model.
        """
        lookback = self.params.get('swing_lookback', 2)

        # --- 1. Swing Detection (Lagged) ---
        # Using centered window, confirmed at t + lookback
        roll_max = data['high'].rolling(window=lookback*2+1, center=True).max()
        roll_min = data['low'].rolling(window=lookback*2+1, center=True).min()
        
        is_swing_high_raw = (data['high'] == roll_max)
        is_swing_low_raw = (data['low'] == roll_min)
        
        # Shift to confirmation time
        swing_high_confirmed = is_swing_high_raw.shift(lookback).fillna(False)
        swing_low_confirmed = is_swing_low_raw.shift(lookback).fillna(False)
        
        # --- 2. Track Last Two Swing Prices ---
        # Current Swing Price (Forward Filled)
        data['last_swing_high_price'] = data['high'].shift(lookback).where(swing_high_confirmed).ffill()
        data['last_swing_low_price'] = data['low'].shift(lookback).where(swing_low_confirmed).ffill()
        
        # Previous Swing Price (Value before the current one updated)
        # We detect when the swing price changes, and grab the value just before that.
        swing_high_changed = data['last_swing_high_price'] != data['last_swing_high_price'].shift(1)
        swing_low_changed = data['last_swing_low_price'] != data['last_swing_low_price'].shift(1)
        
        data['prev_swing_high_price'] = data['last_swing_high_price'].shift(1).where(swing_high_changed).ffill()
        data['prev_swing_low_price'] = data['last_swing_low_price'].shift(1).where(swing_low_changed).ffill()
        
        # --- 3. Relaxed Market Structure (for ML feature) ---
        # Bullish: Higher Lows | Bearish: Lower Highs
        data['structure_bullish'] = (data['last_swing_low_price'] > data['prev_swing_low_price']).astype(int)
        data['structure_bearish'] = (data['last_swing_high_price'] < data['prev_swing_high_price']).astype(int)
        
        # --- 4. Distances (for ML) ---
        if 'atr' not in data.columns:
             data['atr'] = (data['high'] - data['low']).rolling(14).mean()
             
        data['dist_to_swing_high'] = (data['close'] - data['last_swing_high_price']) / data['atr']
        data['dist_to_swing_low'] = (data['close'] - data['last_swing_low_price']) / data['atr']

        # --- 5. Macro Trend Context ---
        data['sma_200'] = data['close'].rolling(200).mean()
        data['dist_to_sma_200'] = (data['close'] - data['sma_200']) / data['atr']
        data['is_above_sma_200'] = (data['close'] > data['sma_200']).astype(int)

        return data

    def get_setup_mask(self, data: pd.DataFrame) -> pd.Series:
        """
        Returns a boolean Series for the 'Sweep & Reclaim' setup.
        The setup is decoupled from structure; the ML model uses structure as a feature.
        """
        # Ensure columns exist
        if 'last_swing_low_price' not in data.columns or 'last_swing_high_price' not in data.columns:
            return pd.Series(False, index=data.index)

        # Long Setup: Price dips below last swing LOW, but closes ABOVE it.
        long_setup = (data['low'] < data['last_swing_low_price']) & (data['close'] > data['last_swing_low_price'])
        
        # Short Setup: Price rallies above last swing HIGH, but closes BELOW it.
        short_setup = (data['high'] > data['last_swing_high_price']) & (data['close'] < data['last_swing_high_price'])
        
        if self.params.get('is_long_only', True):
            # Prevent re-entry on consecutive bars
            return long_setup & ~long_setup.shift(1).fillna(False)

        raw_setup_mask = long_setup | short_setup
        # Prevent re-entry on consecutive bars
        return raw_setup_mask & ~raw_setup_mask.shift(1).fillna(False)