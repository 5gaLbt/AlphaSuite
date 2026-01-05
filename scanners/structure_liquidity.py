import pandas as pd

from scanners.scanner_sdk import BaseScanner

class StructureLiquidityScanner(BaseScanner):
    """
    Scans for 'Structure + Liquidity' setups (Sweep & Reclaim).

    Goal:
        Identify stocks that have just swept a major liquidity zone (Swing High/Low)
        and reclaimed the level, signaling a potential reversal or trend continuation.

    Criteria:
        - Long: Price dips below the last Swing Low but closes above it.
        - Short: Price rallies above the last Swing High but closes below it.
        - Context: Reports whether the underlying market structure is Bullish (Higher Lows)
          or Bearish (Lower Highs).
    """
    @staticmethod
    def define_parameters():
        return [
            {"name": "min_avg_volume", "type": "int", "default": 250000, "label": "Min. Avg. Volume"},
            {"name": "volume_lookback_days", "type": "int", "default": 50, "label": "Avg. Volume Lookback"},
            {"name": "min_market_cap", "type": "int", "default": 1000000000, "label": "Min. Market Cap"},
            {"name": "swing_lookback", "type": "int", "default": 3, "label": "Swing Lookback"},
        ]

    @staticmethod
    def get_leading_columns():
        return ['symbol', 'setup_type', 'structure', 'last_swing_price', 'marketcap']

    @staticmethod
    def get_sort_info():
        return {'by': 'marketcap', 'ascending': False}

    def scan_company(self, group: pd.DataFrame, company_info: dict) -> dict | None:
        lookback = self.params.get('swing_lookback', 3)

        # Need enough data to calculate swings and confirm them
        if len(group) < (lookback * 2) + 50:
            return None

        # --- 1. Swing Detection (Lagged) ---
        # Calculate rolling min/max to identify swing points
        roll_max = group['high'].rolling(window=lookback*2+1, center=True).max()
        roll_min = group['low'].rolling(window=lookback*2+1, center=True).min()
        
        is_swing_high_raw = (group['high'] == roll_max)
        is_swing_low_raw = (group['low'] == roll_min)
        
        # Shift to get confirmation time
        swing_high_confirmed = is_swing_high_raw.shift(lookback).fillna(False)
        swing_low_confirmed = is_swing_low_raw.shift(lookback).fillna(False)
        
        # --- 2. Track Last Swing Prices ---
        confirmed_high_prices = group['high'].shift(lookback).where(swing_high_confirmed)
        confirmed_low_prices = group['low'].shift(lookback).where(swing_low_confirmed)
        
        last_swing_high = confirmed_high_prices.ffill()
        last_swing_low = confirmed_low_prices.ffill()
        
        # --- 3. Determine Structure ---
        # Get the swing price *before* the current one to compare
        prev_swing_high = last_swing_high.shift(1).where(swing_high_confirmed).ffill()
        prev_swing_low = last_swing_low.shift(1).where(swing_low_confirmed).ffill()
        
        # --- 4. Check for Setup on the Latest Bar ---
        # We look at the very last row of data
        current_row = group.iloc[-1]
        curr_idx = group.index[-1]
        
        # Ensure we have valid swing levels for the current bar
        if pd.isna(last_swing_low.iloc[-1]) or pd.isna(last_swing_high.iloc[-1]):
            return None

        current_swing_low = last_swing_low.iloc[-1]
        current_swing_high = last_swing_high.iloc[-1]

        # Long Setup: Sweep Low & Reclaim
        is_long = (current_row['low'] < current_swing_low) and (current_row['close'] > current_swing_low)
        
        # Short Setup: Sweep High & Reclaim
        is_short = (current_row['high'] > current_swing_high) and (current_row['close'] < current_swing_high)

        if is_long or is_short:
            # Determine Structure Context
            # Bullish if Higher Lows, Bearish if Lower Highs
            is_bullish_struct = current_swing_low > prev_swing_low.iloc[-1] if pd.notna(prev_swing_low.iloc[-1]) else False
            is_bearish_struct = current_swing_high < prev_swing_high.iloc[-1] if pd.notna(prev_swing_high.iloc[-1]) else False
            
            structure_str = "Bullish" if is_bullish_struct else ("Bearish" if is_bearish_struct else "Neutral")

            for key in ['id', 'isactive', 'longbusinesssummary', 'bookvalue']:
                if key in company_info: del company_info[key]
            
            company_info['setup_type'] = 'Long (Sweep Low)' if is_long else 'Short (Sweep High)'
            company_info['structure'] = structure_str
            company_info['last_swing_price'] = current_swing_low if is_long else current_swing_high
            return company_info

        return None
