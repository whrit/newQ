# src/utils.py
import numpy as np
from typing import Dict, List, Union, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import yaml
import pytz
from dataclasses import dataclass, asdict

@dataclass
class TradingMetrics:
    """Container for trading metrics"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    volatility: float
    var_95: float
    trades_count: int

class MetricsCalculator:
    """Calculate trading performance metrics"""
    @staticmethod
    def calculate_metrics(returns: np.ndarray,
                         risk_free_rate: float = 0.02) -> TradingMetrics:
        """Calculate comprehensive trading metrics"""
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() \
                      if returns.std() > 0 else 0
                      
        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Trade statistics
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) \
                       if len(losing_trades) > 0 and losing_trades.sum() != 0 else 0
                       
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        
        return TradingMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=returns.mean(),
            volatility=volatility,
            var_95=var_95,
            trades_count=len(returns)
        )

class DataProcessor:
    """Process and prepare data for analysis"""
    @staticmethod
    def prepare_market_data(data: pd.DataFrame,
                          features: List[str]) -> np.ndarray:
        """Prepare market data for model input"""
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaled_data = DataProcessor.scale_features(data[features])
        
        return scaled_data.values
        
    @staticmethod
    def scale_features(data: pd.DataFrame) -> pd.DataFrame:
        """Scale features to [0, 1] range"""
        return (data - data.min()) / (data.max() - data.min())
        
    @staticmethod
    def create_sequences(data: np.ndarray,
                        sequence_length: int) -> np.ndarray:
        """Create sequences for time series analysis"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

class ConfigManager:
    """Manage configuration settings"""
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load configuration from file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def save_config(self, config: Dict):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
            
    def update_config(self, updates: Dict):
        """Update configuration with new values"""
        self.config.update(updates)
        self.save_config(self.config)

class Logger:
    """Custom logger for trading system"""
    def __init__(self, name: str, log_path: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def log_trade(self, trade_info: Dict):
        """Log trade information"""
        self.logger.info(f"Trade executed: {json.dumps(trade_info, indent=2)}")
        
    def log_error(self, error_info: Dict):
        """Log error information"""
        self.logger.error(f"Error occurred: {json.dumps(error_info, indent=2)}")

class TimeManager:
    """Manage time-related functionality"""
    def __init__(self, timezone: str = 'America/New_York'):
        self.timezone = pytz.timezone(timezone)
        
    def is_market_hours(self, time: Optional[datetime] = None) -> bool:
        """Check if within market hours"""
        if time is None:
            time = datetime.now(self.timezone)
        return (
            time.weekday() < 5 and  # Monday to Friday
            time.hour >= 9 and  # After market open
            (time.hour < 16 or (time.hour == 16 and time.minute == 0))  # Before market close
        )
        
    def time_to_market_open(self, time: Optional[datetime] = None) -> timedelta:
        """Calculate time until market open"""
        if time is None:
            time = datetime.now(self.timezone)
            
        if self.is_market_hours(time):
            return timedelta(0)
            
        # Find next market open
        while True:
            time += timedelta(days=1)
            time = time.replace(hour=9, minute=30, second=0, microsecond=0)
            if self.is_market_hours(time):
                break
                
        return time - datetime.now(self.timezone)

class PositionManager:
    """Manage trading positions"""
    def __init__(self, max_position_size: float):
        self.max_position_size = max_position_size
        self.positions = {}
        
    def can_open_position(self, symbol: str, size: float) -> bool:
        """Check if position can be opened"""
        current_size = self.positions.get(symbol, 0)
        return abs(current_size + size) <= self.max_position_size
        
    def update_position(self, symbol: str, size: float):
        """Update position size"""
        self.positions[symbol] = self.positions.get(symbol, 0) + size
        
    def get_position(self, symbol: str) -> float:
        """Get current position size"""
        return self.positions.get(symbol, 0)

class RiskManager:
    """Manage trading risk"""
    def __init__(self, config: Dict):
        self.max_drawdown = config['max_drawdown']
        self.position_limit = config['position_limit']
        self.var_limit = config['var_limit']
        self.concentration_limit = config['concentration_limit']
        self.correlation_limit = config['correlation_limit']
        
    def check_risk_limits(self, metrics: TradingMetrics) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if risk limits are breached
        Returns:
            Tuple of (overall_ok, individual_checks)
        """
        checks = {
            'drawdown': metrics.max_drawdown <= self.max_drawdown,
            'var': metrics.var_95 >= self.var_limit,
            'volatility': metrics.volatility <= self.concentration_limit
        }
        
        return all(checks.values()), checks
        
    def calculate_position_size(self, 
                              price: float,
                              volatility: float,
                              account_value: float) -> float:
        """
        Calculate safe position size using Kelly Criterion and risk limits
        """
        # Kelly position size
        kelly_fraction = self.calculate_kelly_fraction(volatility)
        kelly_size = account_value * kelly_fraction / price
        
        # Risk-adjusted size
        risk_adjusted_size = self.adjust_for_risk(kelly_size, volatility)
        
        # Apply limits
        final_size = min(
            risk_adjusted_size,
            self.position_limit,
            account_value * self.concentration_limit / price
        )
        
        return final_size
        
    def calculate_kelly_fraction(self, volatility: float) -> float:
        """Calculate Kelly Criterion fraction"""
        win_rate = 0.5  # Can be adjusted based on model predictions
        win_loss_ratio = 1.5  # Can be adjusted based on historical performance
        
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Adjust kelly fraction based on volatility
        volatility_adjustment = 1 / (1 + volatility)
        
        return kelly * volatility_adjustment * 0.5  # Using half-kelly for safety
        
    def adjust_for_risk(self, size: float, volatility: float) -> float:
        """Adjust position size based on risk factors"""
        vol_factor = 1 / (1 + volatility)
        return size * vol_factor

class PerformanceAnalyzer:
    """Analyze trading performance"""
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        
    def analyze_performance(self, 
                          trades: List[Dict],
                          positions: Dict[str, float],
                          equity_curve: np.ndarray) -> Dict:
        """
        Comprehensive performance analysis
        """
        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Get base metrics
        metrics = self.metrics_calculator.calculate_metrics(returns)
        
        # Additional analysis
        additional_metrics = {
            'trade_analysis': self.analyze_trades(trades),
            'position_analysis': self.analyze_positions(positions),
            'risk_analysis': self.analyze_risk(returns, equity_curve)
        }
        
        return {**asdict(metrics), **additional_metrics}
        
    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """Analyze trading patterns and performance"""
        if not trades:
            return {}
            
        trade_returns = [t['return'] for t in trades]
        holding_times = [t['holding_time'] for t in trades]
        
        return {
            'avg_holding_time': np.mean(holding_times),
            'best_trade': max(trade_returns),
            'worst_trade': min(trade_returns),
            'avg_win': np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0,
            'avg_loss': np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0,
            'trade_size_analysis': self._analyze_trade_sizes(trades)
        }
        
    def analyze_positions(self, positions: Dict[str, float]) -> Dict:
        """Analyze position characteristics"""
        position_values = list(positions.values())
        
        return {
            'total_exposure': sum(abs(v) for v in position_values),
            'net_exposure': sum(position_values),
            'largest_position': max(abs(v) for v in position_values) if position_values else 0,
            'position_concentration': self._calculate_concentration(position_values),
            'long_short_ratio': self._calculate_long_short_ratio(position_values)
        }
        
    def analyze_risk(self, returns: np.ndarray, equity: np.ndarray) -> Dict:
        """Detailed risk analysis"""
        return {
            'daily_var': self._calculate_var(returns, 0.95),
            'daily_cvar': self._calculate_cvar(returns, 0.95),
            'tail_ratio': self._calculate_tail_ratio(returns),
            'downside_deviation': self._calculate_downside_deviation(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns, equity)
        }
        
    def _analyze_trade_sizes(self, trades: List[Dict]) -> Dict:
        """Analyze trade size patterns"""
        sizes = [t['size'] for t in trades]
        
        return {
            'avg_size': np.mean(sizes),
            'size_stddev': np.std(sizes),
            'size_skew': self._calculate_skew(sizes)
        }
        
    def _calculate_concentration(self, positions: List[float]) -> float:
        """Calculate position concentration (Herfindahl index)"""
        if not positions:
            return 0
        total = sum(abs(p) for p in positions)
        if total == 0:
            return 0
        weights = [abs(p)/total for p in positions]
        return sum(w*w for w in weights)
        
    def _calculate_long_short_ratio(self, positions: List[float]) -> float:
        """Calculate long/short exposure ratio"""
        long_exposure = sum(p for p in positions if p > 0)
        short_exposure = abs(sum(p for p in positions if p < 0))
        return long_exposure / short_exposure if short_exposure != 0 else float('inf')
        
    @staticmethod
    def _calculate_var(returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
        
    @staticmethod
    def _calculate_cvar(returns: np.ndarray, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        var = np.percentile(returns, (1 - confidence) * 100)
        return np.mean(returns[returns <= var])
        
    @staticmethod
    def _calculate_tail_ratio(returns: np.ndarray) -> float:
        """Calculate ratio of upside to downside tail risk"""
        return abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))
        
    @staticmethod
    def _calculate_downside_deviation(returns: np.ndarray) -> float:
        """Calculate downside deviation"""
        negative_returns = returns[returns < 0]
        return np.std(negative_returns) if len(negative_returns) > 0 else 0
        
    @staticmethod
    def _calculate_sortino_ratio(returns: np.ndarray, 
                               risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_std = np.std(returns[returns < 0])
        return np.mean(excess_returns) / downside_std if downside_std != 0 else 0
        
    @staticmethod
    def _calculate_calmar_ratio(returns: np.ndarray, 
                              equity: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
        max_dd = np.min(equity / np.maximum.accumulate(equity) - 1)
        return annual_return / abs(max_dd) if max_dd != 0 else 0
        
    @staticmethod
    def _calculate_skew(data: List[float]) -> float:
        """Calculate skewness of distribution"""
        if not data:
            return 0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return (sum((x - mean) ** 3 for x in data) / 
                ((n - 1) * std ** 3))

class ValidationUtils:
    """Utilities for data validation and verification"""
    @staticmethod
    def validate_trade_data(trade_data: Dict) -> bool:
        """Validate trade data completeness and consistency"""
        required_fields = ['symbol', 'size', 'price', 'timestamp', 'side']
        return all(field in trade_data for field in required_fields)
        
    @staticmethod
    def validate_market_data(market_data: pd.DataFrame) -> bool:
        """Validate market data quality"""
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        has_required = all(col in market_data.columns for col in required_columns)
        
        # Check for missing values
        no_missing = not market_data[required_columns].isnull().any().any()
        
        # Check for data consistency
        price_consistent = all(
            market_data['high'] >= market_data['low'],
            market_data['high'] >= market_data['open'],
            market_data['high'] >= market_data['close'],
            market_data['low'] <= market_data['open'],
            market_data['low'] <= market_data['close']
        )
        
        return has_required and no_missing and price_consistent
        
    @staticmethod
    def validate_config(config: Dict) -> Tuple[bool, List[str]]:
        """Validate configuration settings"""
        required_settings = {
            'trading': ['max_position_size', 'max_drawdown', 'risk_limits'],
            'execution': ['order_timeout', 'max_slippage'],
            'risk': ['var_limit', 'correlation_limit']
        }
        
        missing = []
        for category, settings in required_settings.items():
            if category not in config:
                missing.append(f"Missing category: {category}")
                continue
            for setting in settings:
                if setting not in config[category]:
                    missing.append(f"Missing setting: {category}.{setting}")
                    
        return len(missing) == 0, missing