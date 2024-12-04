# src/features/order_book_features.py
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import logging

@dataclass
class OrderBookFeatures:
    # Simplified features we can calculate from historical OHLCV data
    volatility: float
    volume_profile: float
    price_trend: float
    liquidity_estimate: float

class OrderBookAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, historical_data: pd.DataFrame) -> OrderBookFeatures:
        """
        Calculate features from historical OHLCV data instead of order book data
        Args:
            historical_data: DataFrame with OHLCV data
        Returns:
            OrderBookFeatures object
        """
        if historical_data is None or historical_data.empty:
            self.logger.warning("No historical data available, returning default features")
            return self._get_default_features()

        try:
            # Calculate volatility (using close prices)
            returns = historical_data['close'].pct_change()
            volatility = returns.std()

            # Volume profile (normalized volume)
            volume_profile = (historical_data['volume'] - historical_data['volume'].mean()) / historical_data['volume'].std()
            volume_profile = volume_profile.iloc[-1]  # Get most recent

            # Price trend (simple moving average comparison)
            sma_short = historical_data['close'].rolling(5).mean()
            sma_long = historical_data['close'].rolling(20).mean()
            price_trend = (sma_short.iloc[-1] / sma_long.iloc[-1]) - 1

            # Liquidity estimate (using volume and price range)
            typical_price_range = (historical_data['high'] - historical_data['low']).mean()
            avg_volume = historical_data['volume'].mean()
            liquidity_estimate = avg_volume / typical_price_range if typical_price_range != 0 else 0

            return OrderBookFeatures(
                volatility=float(volatility),
                volume_profile=float(volume_profile),
                price_trend=float(price_trend),
                liquidity_estimate=float(liquidity_estimate)
            )

        except Exception as e:
            self.logger.error(f"Error calculating historical features: {str(e)}")
            return self._get_default_features()

    def _get_default_features(self) -> OrderBookFeatures:
        """Return default features when data is unavailable"""
        return OrderBookFeatures(
            volatility=0.0,
            volume_profile=0.0,
            price_trend=0.0,
            liquidity_estimate=0.0
        )