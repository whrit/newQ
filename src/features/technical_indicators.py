# src/features/technical_indicators.py
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass
import talib
import logging
from datetime import datetime
from src.data.data_manager import DataManager

@dataclass
class TechnicalFeatures:
    trend_indicators: Dict[str, float]
    momentum_indicators: Dict[str, float]
    volatility_indicators: Dict[str, float]
    volume_indicators: Dict[str, float]
    cycle_indicators: Dict[str, float]

class TechnicalAnalyzer:
    """
    Advanced technical analysis with multiple indicator categories
    """
    def __init__(self, data_manager: DataManager, lookback_periods: Dict[str, int] = None):
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.lookback_periods = lookback_periods or {
            'short': 10,
            'medium': 20,
            'long': 50
        }

    def calculate_features(self, data: pd.DataFrame, start_date: datetime, end_date: datetime) -> TechnicalFeatures:
        """Calculate technical features based on historical data."""
        try:
            if not self._validate_market_data(data):
                self.logger.warning(f"No valid data for the specified date range: {start_date} to {end_date}")
                return self._get_default_features()

            # Proceed with calculations
            return self._calculate_features_from_data(data)

        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            return self._get_default_features()

    def _calculate_features_from_data(self, data: pd.DataFrame) -> TechnicalFeatures:
        """Calculate all technical features from market data."""
        try:
            # Ensure data types are correct and handle NaN values
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

            # Fill any remaining NaN values
            data = data.ffill().bfill()

            if data.empty:
                self.logger.warning("No valid data after cleaning")
                return self._get_default_features()

            # Compute indicators
            trend_indicators = self._calculate_trend_indicators(data)
            momentum_indicators = self._calculate_momentum_indicators(data)
            volatility_indicators = self._calculate_volatility_indicators(data)
            volume_indicators = self._calculate_volume_indicators(data)
            cycle_indicators = self._calculate_cycle_indicators(data)

            # Return aggregated features
            return TechnicalFeatures(
                trend_indicators=trend_indicators,
                momentum_indicators=momentum_indicators,
                volatility_indicators=volatility_indicators,
                volume_indicators=volume_indicators,
                cycle_indicators=cycle_indicators
            )
        except Exception as e:
            self.logger.error(f"Error in _calculate_features_from_data: {str(e)}")
            return self._get_default_features()

    def _get_default_features(self) -> TechnicalFeatures:
        """Return default feature values."""
        return TechnicalFeatures(
            trend_indicators={'sma_short': 0, 'sma_medium': 0, 'sma_long': 0, 'macd': 0, 'adx': 50},
            momentum_indicators={
                'rsi': 50, 
                'stoch_k': 50, 
                'stoch_d': 50, 
                'roc': 0,
                'cci': 0  # Add default CCI value
            },
            volatility_indicators={'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0, 'atr': 0, 'historical_volatility': 0},
            volume_indicators={'obv': 0, 'vpt': 0, 'cmf': 0, 'volume_rsi': 50, 'volume_ma_ratio': 1},
            cycle_indicators={'ht_sine': 0, 'ht_leadsine': 0, 'dc_period': 0, 'dc_phase': 0}
        )

    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        try:
            # Ensure data is in float64 format
            close = data['close'].astype(float).values
            high = data['high'].astype(float).values
            low = data['low'].astype(float).values

            # Adjust periods based on available data
            short_period = min(self.lookback_periods['short'], len(close) - 1)
            medium_period = min(self.lookback_periods['medium'], len(close) - 1)
            long_period = min(self.lookback_periods['long'], len(close) - 1)

            # Calculate moving averages
            sma_short = talib.SMA(close, timeperiod=short_period)[-1]
            sma_medium = talib.SMA(close, timeperiod=medium_period)[-1]
            sma_long = talib.SMA(close, timeperiod=long_period)[-1]

            # MACD with adjusted periods
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=min(12, len(close) - 1),
                slowperiod=min(26, len(close) - 1),
                signalperiod=min(9, len(close) - 1)
            )

            # ADX with adjusted period
            adx = talib.ADX(high, low, close, timeperiod=min(14, len(close) - 1))[-1]

            return {
                'sma_short': sma_short,
                'sma_medium': sma_medium,
                'sma_long': sma_long,
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'macd_hist': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
                'adx': adx if not np.isnan(adx) else 50
            }
        except Exception as e:
            self.logger.error(f"Error in trend calculation: {str(e)}")
            return self._get_default_features().trend_indicators

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        try:
            # Convert data to float64
            close = data['close'].astype('float64').values
            high = data['high'].astype('float64').values
            low = data['low'].values.astype('float64')

            period = min(14, len(close)-1)
            
            # Calculate RSI
            rsi = talib.RSI(close, timeperiod=period)[-1]
            
            # Calculate Stochastic
            slowk, slowd = talib.STOCH(high, low, close,
                                     fastk_period=period,
                                     slowk_period=3,
                                     slowd_period=3)
            
            # Calculate ROC
            roc = talib.ROC(close, timeperiod=min(10, len(close)-1))[-1]
            
            # Calculate CCI
            cci = talib.CCI(high, low, close, timeperiod=period)[-1]

            return {
                'rsi': rsi if not np.isnan(rsi) else 50,
                'stoch_k': slowk[-1] if not np.isnan(slowk[-1]) else 50,
                'stoch_d': slowd[-1] if not np.isnan(slowd[-1]) else 50,
                'roc': roc if not np.isnan(roc) else 0,
                'cci': cci if not np.isnan(cci) else 0  # Add CCI to returned dict
            }
        except Exception as e:
            self.logger.error(f"Error in momentum calculation: {str(e)}")
            return self._get_default_features().momentum_indicators

    def _validate_market_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if data is None or data.empty:
            self.logger.warning("No data available")
            return False
            
        if len(data) < 2:
            self.logger.warning("Insufficient data points")
            return False
            
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
            return False
            
        return True
        
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility indicators"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        
        # ATR
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        
        # Historical Volatility
        close_returns = np.log(close[1:] / close[:-1])
        hist_vol = np.std(close_returns) * np.sqrt(252)
        
        return {
            'bb_upper': upper[-1],
            'bb_middle': middle[-1],
            'bb_lower': lower[-1],
            'bb_width': (upper[-1] - lower[-1]) / middle[-1],
            'atr': atr,
            'historical_volatility': hist_vol
        }
        
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume indicators"""
        close = data['close'].values
        volume = data['volume'].values
        
        # On-Balance Volume
        obv = talib.OBV(close, volume)[-1]
        
        # Volume-price trend
        vpt = self._calculate_vpt(close, volume)
        
        # Chaikin Money Flow
        cmf = self._calculate_cmf(data)
        
        # Volume RSI
        volume_rsi = talib.RSI(volume, timeperiod=14)[-1]
        
        return {
            'obv': obv,
            'vpt': vpt,
            'cmf': cmf,
            'volume_rsi': volume_rsi,
            'volume_ma_ratio': volume[-1] / np.mean(volume[-20:])
        }
        
    def _calculate_cycle_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate cycle indicators"""
        close = data['close'].values
        
        # Hilbert Transform
        ht_sine, ht_leadsine = talib.HT_SINE(close)
        
        # DCPeriod and DCPhase
        dc_period = talib.HT_DCPERIOD(close)[-1]
        dc_phase = talib.HT_DCPHASE(close)[-1]
        
        return {
            'ht_sine': ht_sine[-1],
            'ht_leadsine': ht_leadsine[-1],
            'dc_period': dc_period,
            'dc_phase': dc_phase
        }
        
    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict[str, Union[float, bool]]:
        """Calculate Ichimoku Cloud indicators"""
        try:
            high_9 = data['high'].rolling(window=9).max()
            low_9 = data['low'].rolling(window=9).min()
            high_26 = data['high'].rolling(window=26).max()
            low_26 = data['low'].rolling(window=26).min()
            high_52 = data['high'].rolling(window=52).max()
            low_52 = data['low'].rolling(window=52).min()

            conversion = (high_9 + low_9) / 2
            base = (high_26 + low_26) / 2
            leading_span_a = (conversion + base) / 2
            leading_span_b = (high_52 + low_52) / 2

            current_price = data['close'].iloc[-1]
            cloud_top = max(leading_span_a.iloc[-1], leading_span_b.iloc[-1])
            cloud_bottom = min(leading_span_a.iloc[-1], leading_span_b.iloc[-1])

            return {
                'conversion': conversion.iloc[-1],
                'base': base.iloc[-1],
                'leading_span_a': leading_span_a.iloc[-1],
                'leading_span_b': leading_span_b.iloc[-1],
                'price_above_cloud': current_price > cloud_top
            }
        except Exception as e:
            self.logger.error(f"Error in Ichimoku calculation: {str(e)}")
            return {
                'conversion': 0,
                'base': 0,
                'leading_span_a': 0,
                'leading_span_b': 0,
                'price_above_cloud': False
            }
        
    def _calculate_vpt(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate Volume-Price Trend"""
        close_diff = np.diff(close)
        close_prev = close[:-1]
        vpt = volume[1:] * (close_diff / close_prev)
        return np.sum(vpt)
        
    def _calculate_cmf(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate Chaikin Money Flow"""
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfv = mfm * data['volume']
        cmf = mfv.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        return cmf.iloc[-1]
    
    def get_trading_signals(self, features: TechnicalFeatures) -> Dict[str, float]:
        """
        Generate trading signals from technical features
        Returns:
            Dictionary with signal strengths (-1 to 1)
        """
        signals = {}
        
        # Trend signals
        trend_signal = self._calculate_trend_signal(features.trend_indicators)
        signals['trend'] = trend_signal
        
        # Momentum signals
        momentum_signal = self._calculate_momentum_signal(features.momentum_indicators)
        signals['momentum'] = momentum_signal
        
        # Volatility signals
        volatility_signal = self._calculate_volatility_signal(features.volatility_indicators)
        signals['volatility'] = volatility_signal
        
        # Volume signals
        volume_signal = self._calculate_volume_signal(features.volume_indicators)
        signals['volume'] = volume_signal
        
        # Combined signal
        signals['combined'] = np.mean([
            trend_signal,
            momentum_signal,
            volatility_signal,
            volume_signal
        ])
        
        return signals
        
    def _calculate_trend_signal(self, indicators: Dict[str, float]) -> float:
        """Calculate trend signal"""
        signals = [
            1 if indicators.get('sma_short', 0) > indicators.get('sma_medium', 0) else -1,
            1 if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else -1,
            1 if indicators.get('adx', 0) > 25 else 0,
            1 if indicators.get('price_above_cloud', False) else -1
        ]
        return np.mean(signals)
        
    def _calculate_momentum_signal(self, indicators: Dict[str, float]) -> float:
        """Calculate momentum signal"""
        signals = [
            1 if indicators['rsi'] > 50 else -1,
            1 if indicators['stoch_k'] > indicators['stoch_d'] else -1,
            1 if indicators['cci'] > 0 else -1
        ]
        return np.mean(signals)
        
    def _calculate_volatility_signal(self, indicators: Dict[str, float]) -> float:
        """Calculate volatility signal"""
        bb_position = (indicators['bb_middle'] - indicators['bb_lower']) / \
                     (indicators['bb_upper'] - indicators['bb_lower'])
        signals = [
            1 if bb_position > 0.5 else -1,
            1 if indicators['historical_volatility'] < 0.2 else -1
        ]
        return np.mean(signals)
        
    def _calculate_volume_signal(self, indicators: Dict[str, float]) -> float:
        """Calculate volume signal"""
        signals = [
            1 if indicators['volume_ma_ratio'] > 1 else -1,
            1 if indicators['cmf'] > 0 else -1,
            1 if indicators['volume_rsi'] > 50 else -1
        ]
        return np.mean(signals)