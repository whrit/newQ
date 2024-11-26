# src/features/sector_analysis.py
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor

@dataclass
class SectorMetrics:
    correlation: float
    relative_strength: float
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    sector_rotation: Dict[str, float]

class SectorAnalyzer:
    """
    Advanced sector analysis for market context
    """
    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.lookback_periods = lookback_periods or {
            'short': 5,
            'medium': 20,
            'long': 60
        }
        self.logger = logging.getLogger(__name__)
        
    def analyze_sector(self, symbol: str, sector_data: pd.DataFrame, 
                      market_data: pd.DataFrame) -> SectorMetrics:
        """
        Perform comprehensive sector analysis
        Args:
            symbol: Stock symbol
            sector_data: DataFrame with sector price data
            market_data: DataFrame with market index data
        Returns:
            SectorMetrics object with analysis results
        """
        try:
            # Calculate various metrics
            correlation = self._calculate_correlation(symbol, sector_data)
            relative_strength = self._calculate_relative_strength(sector_data, market_data)
            performance = self._calculate_performance_metrics(sector_data)
            risk = self._calculate_risk_metrics(sector_data)
            rotation = self._analyze_sector_rotation(sector_data, market_data)
            
            return SectorMetrics(
                correlation=correlation,
                relative_strength=relative_strength,
                performance_metrics=performance,
                risk_metrics=risk,
                sector_rotation=rotation
            )
            
        except Exception as e:
            self.logger.error(f"Error in sector analysis: {str(e)}")
            raise
            
    def _calculate_correlation(self, symbol: str, sector_data: pd.DataFrame) -> float:
        """Calculate correlation with sector"""
        symbol_returns = sector_data[symbol].pct_change().dropna()
        sector_returns = sector_data.mean(axis=1).pct_change().dropna()
        return symbol_returns.corr(sector_returns)
        
# Continuation of src/features/sector_analysis.py

    def _calculate_relative_strength(self, sector_data: pd.DataFrame,
                                   market_data: pd.DataFrame) -> float:
        """
        Calculate sector's relative strength against market
        """
        # Calculate returns
        sector_returns = sector_data.mean(axis=1).pct_change().dropna()
        market_returns = market_data['close'].pct_change().dropna()
        
        # Calculate relative strength over different periods
        rs_scores = {}
        for period_name, period in self.lookback_periods.items():
            sector_perf = (1 + sector_returns.tail(period)).prod() - 1
            market_perf = (1 + market_returns.tail(period)).prod() - 1
            rs_scores[period_name] = sector_perf / market_perf if market_perf != 0 else 0
            
        # Weighted average of different timeframes
        weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
        return sum(rs_scores[period] * weights[period] for period in weights)
        
    def _calculate_performance_metrics(self, sector_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for sector
        """
        sector_returns = sector_data.mean(axis=1).pct_change().dropna()
        
        performance_metrics = {
            'daily_returns': sector_returns.mean(),
            'volatility': sector_returns.std() * np.sqrt(252),
            'sharpe_ratio': self._calculate_sharpe_ratio(sector_returns),
            'sortino_ratio': self._calculate_sortino_ratio(sector_returns),
            'max_drawdown': self._calculate_max_drawdown(sector_returns),
            'win_rate': len(sector_returns[sector_returns > 0]) / len(sector_returns),
            'up_capture': self._calculate_up_capture(sector_returns),
            'down_capture': self._calculate_down_capture(sector_returns)
        }
        
        return performance_metrics
        
    def _calculate_risk_metrics(self, sector_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for sector
        """
        sector_returns = sector_data.mean(axis=1).pct_change().dropna()
        
        risk_metrics = {
            'var_95': self._calculate_var(sector_returns, 0.95),
            'cvar_95': self._calculate_cvar(sector_returns, 0.95),
            'beta': self._calculate_beta(sector_returns),
            'tracking_error': self._calculate_tracking_error(sector_returns),
            'information_ratio': self._calculate_information_ratio(sector_returns),
            'concentration_risk': self._calculate_concentration_risk(sector_data),
            'tail_risk': self._calculate_tail_risk(sector_returns)
        }
        
        return risk_metrics
        
    def _analyze_sector_rotation(self, sector_data: pd.DataFrame,
                               market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze sector rotation patterns and market cycle positioning
        """
        rotation_metrics = {}
        
        # Calculate momentum scores
        rotation_metrics.update(
            self._calculate_momentum_scores(sector_data, market_data)
        )
        
        # Calculate sector cycle position
        rotation_metrics.update(
            self._calculate_cycle_position(sector_data, market_data)
        )
        
        # Calculate sector leadership
        rotation_metrics.update(
            self._calculate_sector_leadership(sector_data)
        )
        
        return rotation_metrics
        
    def _calculate_momentum_scores(self, sector_data: pd.DataFrame,
                                 market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate sector momentum scores"""
        sector_returns = sector_data.mean(axis=1).pct_change()
        
        momentum_scores = {}
        for period_name, period in self.lookback_periods.items():
            momentum = sector_returns.rolling(period).mean().iloc[-1]
            momentum_scores[f'momentum_{period_name}'] = momentum
            
        return momentum_scores
        
    def _calculate_cycle_position(self, sector_data: pd.DataFrame,
                                market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Determine sector's position in market cycle
        Returns values between 0 and 1 for different cycle phases
        """
        # Calculate various cycle indicators
        sector_ma_ratios = self._calculate_moving_average_ratios(sector_data)
        volatility_regime = self._calculate_volatility_regime(sector_data)
        trend_strength = self._calculate_trend_strength(sector_data)
        
        cycle_metrics = {
            'early_cycle_score': self._early_cycle_indicator(
                sector_ma_ratios, volatility_regime
            ),
            'mid_cycle_score': self._mid_cycle_indicator(
                sector_ma_ratios, trend_strength
            ),
            'late_cycle_score': self._late_cycle_indicator(
                sector_ma_ratios, volatility_regime, trend_strength
            ),
            'recession_score': self._recession_indicator(
                sector_ma_ratios, volatility_regime
            )
        }
        
        return cycle_metrics
        
    def _calculate_sector_leadership(self, sector_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate sector leadership metrics"""
        returns = sector_data.mean(axis=1).pct_change()
        
        leadership_metrics = {
            'relative_momentum': self._calculate_relative_momentum(returns),
            'strength_index': self._calculate_strength_index(returns),
            'leadership_score': self._calculate_leadership_score(returns),
            'consistency_score': self._calculate_consistency_score(returns)
        }
        
        return leadership_metrics
        
    def _calculate_moving_average_ratios(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate ratios of different moving averages"""
        prices = data.mean(axis=1)
        
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_50 = prices.rolling(50).mean().iloc[-1]
        ma_200 = prices.rolling(200).mean().iloc[-1]
        
        return {
            'ma20_50': ma_20 / ma_50,
            'ma50_200': ma_50 / ma_200,
            'price_ma200': prices.iloc[-1] / ma_200
        }
        
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identify volatility regime"""
        returns = data.mean(axis=1).pct_change()
        current_vol = returns.tail(20).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)
        
        return {
            'current_vol': current_vol,
            'vol_ratio': current_vol / historical_vol,
            'vol_regime': 1 if current_vol > historical_vol else 0
        }
        
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength indicator"""
        prices = data.mean(axis=1)
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        
        trend_points = 0
        if prices.iloc[-1] > ma_20.iloc[-1]:
            trend_points += 1
        if ma_20.iloc[-1] > ma_50.iloc[-1]:
            trend_points += 1
        if np.all(np.diff(ma_20.tail(10)) > 0):
            trend_points += 1
            
        return trend_points / 3
        
    def _early_cycle_indicator(self, ma_ratios: Dict[str, float],
                             volatility: Dict[str, float]) -> float:
        """Calculate early cycle probability"""
        score = 0
        if ma_ratios['ma20_50'] > 1 and ma_ratios['ma50_200'] < 1:
            score += 0.5
        if volatility['vol_ratio'] < 0.8:
            score += 0.5
        return score
        
    def _mid_cycle_indicator(self, ma_ratios: Dict[str, float],
                           trend_strength: float) -> float:
        """Calculate mid cycle probability"""
        score = 0
        if all(ratio > 1 for ratio in ma_ratios.values()):
            score += 0.6
        score += 0.4 * trend_strength
        return score
        
    def _late_cycle_indicator(self, ma_ratios: Dict[str, float],
                            volatility: Dict[str, float],
                            trend_strength: float) -> float:
        """Calculate late cycle probability"""
        score = 0
        if ma_ratios['ma20_50'] < 1 and ma_ratios['ma50_200'] > 1:
            score += 0.4
        if volatility['vol_ratio'] > 1.2:
            score += 0.3
        if trend_strength < 0.3:
            score += 0.3
        return score
        
    def _recession_indicator(self, ma_ratios: Dict[str, float],
                           volatility: Dict[str, float]) -> float:
        """Calculate recession probability"""
        score = 0
        if all(ratio < 1 for ratio in ma_ratios.values()):
            score += 0.6
        if volatility['vol_ratio'] > 1.5:
            score += 0.4
        return score
        
    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    @staticmethod
    def _calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0
        
    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()
        
    def get_sector_summary(self, metrics: SectorMetrics) -> Dict[str, str]:
        """
        Generate human-readable summary of sector analysis
        """
        summary = {
            'overall_strength': self._interpret_strength(
                metrics.relative_strength,
                metrics.performance_metrics['sharpe_ratio']
            ),
            'risk_assessment': self._interpret_risk(
                metrics.risk_metrics
            ),
            'cycle_position': self._interpret_cycle(
                metrics.sector_rotation
            ),
            'recommendation': self._generate_recommendation(metrics)
        }
        
        return summary