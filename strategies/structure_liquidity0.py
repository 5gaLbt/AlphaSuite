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
    Structure + Liquidity Strategy (Original Concept).
    
    Core Logic:
    1. Market Structure: 
       - Bullish: Strict Higher Swing Highs AND Higher Swing Lows.
       - Bearish: Strict Lower Swing Lows AND Lower Swing Highs.
    2. Liquidity Zones: 
       - Identified by 'Equal Highs' or 'Equal Lows' (consecutive bars within tolerance).
    3. Setup: 
       - Long: Bullish Structure + Equal Lows formed (Liquidity Test).
       - Short: Bearish Structure + Equal Highs formed.
    """

    description = "Original implementation: Strict structure rules and Equal Highs/Lows detection."

    @staticmethod
    def define_parameters():
        """Defines parameters, their types, defaults, and tuning ranges."""
        return {
            'swing_lookback': {'type': 'int', 'default': 5, 'tuning_range': (3, 10)},
            'liquidity_tolerance': {'type': 'float', 'default': 0.001, 'tuning_range': (0.0005, 0.005)},
            'atr_period': {'type': 'int', 'default': 14, 'tuning_range': (10, 30)},
            'initial_stop_atr_multiplier': {'type': 'float', 'default': 2.0, 'tuning_range': (1.5, 4.0)},
            'trailing_stop_atr_multiplier': {'type': 'float', 'default': 3.0, 'tuning_range': (2.5, 6.0)},
            'stop_out_window': {'type': 'int', 'default': 60, 'tuning_range': (20, 120)},
            'probability_threshold': {'type': 'float', 'default': 0.60, 'tuning_range': (0.55, 0.80)},
            'risk_per_trade_pct': {'type': 'float', 'default': 0.02, 'tuning_range': (0.01, 0.05)},
            'is_long_only': {'type': 'bool', 'default': True, 'tuning_range': None},
        }

    def get_feature_list(self) -> list[str]:
        """Returns the list of feature column names required by the model."""
        return [
            'structure_bullish', 'structure_bearish',
            'is_equal_high', 'is_equal_low',
            'dist_to_swing_high', 'dist_to_swing_low',
            'atr', 'volume', 'rsi', 'roc'
        ]

    def add_strategy_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates swing points, market structure, and features for the ML model.
        """
        lookback = self.params.get('swing_lookback', 5)
        tolerance = self.params.get('liquidity_tolerance', 0.001)

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
        
        # --- 3. Strict Market Structure ---
        # Bullish: Higher High AND Higher Low
        data['structure_bullish'] = (
            (data['last_swing_high_price'] > data['prev_swing_high_price']) & 
            (data['last_swing_low_price'] > data['prev_swing_low_price'])
        ).astype(int)

        # Bearish: Lower Low AND Lower High
        data['structure_bearish'] = (
            (data['last_swing_low_price'] < data['prev_swing_low_price']) & 
            (data['last_swing_high_price'] < data['prev_swing_high_price'])
        ).astype(int)
        
        # --- 4. Liquidity Zones (Equal Highs/Lows) ---
        # Defined as consecutive bars with high/low within tolerance
        data['is_equal_high'] = ((data['high'] - data['high'].shift(1)).abs() / data['high'] < tolerance).astype(int)
        data['is_equal_low'] = ((data['low'] - data['low'].shift(1)).abs() / data['low'] < tolerance).astype(int)
        
        # --- 5. Distances (for ML) ---
        if 'atr' not in data.columns:
             data['atr'] = (data['high'] - data['low']).rolling(14).mean()
             
        data['dist_to_swing_high'] = (data['close'] - data['last_swing_high_price']) / data['atr']
        data['dist_to_swing_low'] = (data['close'] - data['last_swing_low_price']) / data['atr']

        return data

    def get_setup_mask(self, data: pd.DataFrame) -> pd.Series:
        """
        Returns setup mask based on Original Concept:
        Long: Bullish Structure + Equal Lows
        Short: Bearish Structure + Equal Highs
        """
        # Ensure columns exist
        if 'structure_bullish' not in data.columns:
            return pd.Series(False, index=data.index)

        # Long Setup
        # Structure is bullish AND we just formed an equal low (potential support test)
        long_setup = (data['structure_bullish'] == 1) & (data['is_equal_low'] == 1)
        
        # Short Setup
        short_setup = (data['structure_bearish'] == 1) & (data['is_equal_high'] == 1)
        
        if self.params.get('is_long_only', True):
            return long_setup

        return long_setup | short_setup