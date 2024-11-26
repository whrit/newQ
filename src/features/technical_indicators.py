# src/features/technical_indicators.py
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass
import talib

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
    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.lookback_periods = lookback_periods or {
            'short': 14,
            'medium': 50,
            'long': 200
        }
        
    def calculate_features(self, data: pd.DataFrame) -> TechnicalFeatures:
        """
        Calculate comprehensive technical features
        Args:
            data: DataFrame with OHLCV data
        Returns:
            TechnicalFeatures object with all indicators
        """
        trend = self._calculate_trend_indicators(data)
        momentum = self._calculate_momentum_indicators(data)
        volatility = self._calculate_volatility_indicators(data)
        volume = self._calculate_volume_indicators(data)
        cycle = self._calculate_cycle_indicators(data)
        
        return TechnicalFeatures(
            trend_indicators=trend,
            momentum_indicators=momentum,
            volatility_indicators=volatility,
            volume_indicators=volume,
            cycle_indicators=cycle
        )
        
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend indicators"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Moving averages
        sma_short = talib.SMA(close, timeperiod=self.lookback_periods['short'])[-1]
        sma_medium = talib.SMA(close, timeperiod=self.lookback_periods['medium'])[-1]
        sma_long = talib.SMA(close, timeperiod=self.lookback_periods['long'])[-1]
        
        # Exponential moving averages
        ema_short = talib.EMA(close, timeperiod=self.lookback_periods['short'])[-1]
        ema_medium = talib.EMA(close, timeperiod=self.lookback_periods['medium'])[-1]
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        
        # ADX
        adx = talib.ADX(high, low, close, timeperiod=14)[-1]
        
        # Ichimoku Cloud
        ichimoku = self._calculate_ichimoku(data)
        
        return {
            'sma_short': sma_short,
            'sma_medium': sma_medium,
            'sma_long': sma_long,
            'ema_short': ema_short,
            'ema_medium': ema_medium,
            'macd': macd[-1],
            'macd_signal': macd_signal[-1],
            'macd_hist': macd_hist[-1],
            'adx': adx,
            'ichimoku_conversion': ichimoku['conversion'],
            'ichimoku_base': ichimoku['base'],
            'price_above_cloud': ichimoku['above_cloud']
        }
        
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)[-1]
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        
        # ROC
        roc = talib.ROC(close, timeperiod=10)[-1]
        
        # CCI
        cci = talib.CCI(high, low, close, timeperiod=14)[-1]
        
        # Williams %R
        willr = talib.WILLR(high, low, close, timeperiod=14)[-1]
        
        return {
            'rsi': rsi,
            'stoch_k': slowk[-1],
            'stoch_d': slowd[-1],
            'roc': roc,
            'cci': cci,
            'willr': willr
        }
        
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
            'above_cloud': current_price > cloud_top
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
            1 if indicators['sma_short'] > indicators['sma_medium'] else -1,
            1 if indicators['macd'] > indicators['macd_signal'] else -1,
            1 if indicators['adx'] > 25 else 0,
            1 if indicators['price_above_cloud'] else -1
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