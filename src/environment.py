# src/environment.py
import numpy as np
from typing import Dict, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .features.technical_indicators import TechnicalAnalyzer
from .features.market_sentiment import MarketSentimentAnalyzer
from .features.order_book_features import OrderBookAnalyzer
from .features.sector_analysis import SectorAnalyzer

@dataclass
class State:
    price_data: Dict[str, float]
    technical_indicators: Dict[str, float]
    sentiment_data: Dict[str, float]
    order_book_features: Dict[str, float]
    sector_metrics: Dict[str, float]
    position_info: Dict[str, float]
    account_info: Dict[str, float]
    market_context: Dict[str, float]

class TradingEnvironment:
    """
    Advanced trading environment integrating multiple data sources and analysis
    """
    def __init__(self, 
                 trading_client: TradingClient,
                 data_client: StockHistoricalDataClient,
                 config: Dict[str, Any]):
        self.trading_client = trading_client
        self.data_client = data_client
        self.config = config
        
        # Initialize analyzers
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = MarketSentimentAnalyzer(config.get('sentiment_config', {}))
        self.order_book_analyzer = OrderBookAnalyzer()
        self.sector_analyzer = SectorAnalyzer()
        
        # Initialize tracking variables
        self.current_step = 0
        self.current_position = 0
        self.position_history = []
        self.reward_history = []
        self.state_history = []
        
    async def get_state(self, symbol: str) -> State:
        """
        Get comprehensive current market state
        """
        try:
            # Fetch raw market data
            price_data = await self._fetch_price_data(symbol)
            order_book = await self._fetch_order_book(symbol)
            position_info = self._get_position_info(symbol)
            account_info = self._get_account_info()
            
            # Calculate features
            technical_features = self.technical_analyzer.calculate_features(
                pd.DataFrame(price_data)
            )
            
            sentiment_data = await self.sentiment_analyzer.analyze_sentiment(symbol)
            
            order_book_features = self.order_book_analyzer.calculate_features(
                order_book['bids'],
                order_book['asks']
            )
            
            sector_metrics = await self._get_sector_metrics(symbol)
            
            market_context = await self._get_market_context()
            
            # Combine into state
            return State(
                price_data=self._normalize_price_data(price_data),
                technical_indicators=technical_features.to_dict(),
                sentiment_data=sentiment_data.to_dict(),
                order_book_features=order_book_features.to_dict(),
                sector_metrics=sector_metrics,
                position_info=position_info,
                account_info=account_info,
                market_context=market_context
            )
            
        except Exception as e:
            self.logger.error(f"Error getting state: {str(e)}")
            raise
            
    async def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        """
        Execute one environment step
        Args:
            action: Trading action to take
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        try:
            # Execute action
            execution_info = await self._execute_action(action)
            
            # Get new state
            new_state = await self.get_state(self.config['symbol'])
            
            # Calculate reward
            reward = self._calculate_reward(
                execution_info,
                new_state
            )
            
            # Check if episode is done
            done = self._is_episode_done()
            
            # Additional info
            info = {
                'execution': execution_info,
                'metrics': self._calculate_metrics(),
                'position_changes': self._get_position_changes()
            }
            
            # Update tracking
            self.current_step += 1
            self.state_history.append(new_state)
            self.reward_history.append(reward)
            
            return new_state, reward, done, info
            
        except Exception as e:
            self.logger.error(f"Error in environment step: {str(e)}")
            raise
            
    async def _fetch_price_data(self, symbol: str) -> Dict[str, np.ndarray]:
        """Fetch historical price data"""
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=(datetime.now() - timedelta(days=5)),
            end=datetime.now()
        )
        
        bars = await self.data_client.get_stock_bars(bars_request)
        return self._process_bars(bars)
        
    async def _fetch_order_book(self, symbol: str) -> Dict[str, List]:
        """Fetch current order book"""
        # Implementation depends on your data source
        pass
        
    def _get_position_info(self, symbol: str) -> Dict[str, float]:
        """Get current position information"""
        try:
            position = self.trading_client.get_position(symbol)
            return {
                'quantity': float(position.qty),
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'cost_basis': float(position.cost_basis)
            }
        except:
            return {
                'quantity': 0.0,
                'market_value': 0.0,
                'avg_entry_price': 0.0,
                'unrealized_pl': 0.0,
                'unrealized_plpc': 0.0,
                'cost_basis': 0.0
            }
            
    def _get_account_info(self) -> Dict[str, float]:
        """Get current account information"""
        account = self.trading_client.get_account()
        return {
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'daytrading_buying_power': float(account.daytrading_buying_power),
            'regt_buying_power': float(account.regt_buying_power),
            'day_trade_count': float(account.daytrade_count)
        }
        
    async def _get_sector_metrics(self, symbol: str) -> Dict[str, float]:
        """Get sector analysis metrics"""
        # Fetch sector data
        sector_data = await self._fetch_sector_data(symbol)
        market_data = await self._fetch_market_data()
        
        metrics = self.sector_analyzer.analyze_sector(
            symbol,
            sector_data,
            market_data
        )
        
        return metrics.to_dict()
        
    async def _get_market_context(self) -> Dict[str, float]:
        """Get broader market context"""
        # Market indices
        indices = await self._fetch_market_indices()
        
        # Volatility indicators
        vix = await self._fetch_vix()
        
        # Market breadth
        breadth = await self._fetch_market_breadth()
        
        return {
            'indices': indices,
            'volatility': vix,
            'breadth': breadth
        }
        
    def _calculate_reward(self, execution_info: Dict, 
                         new_state: State) -> float:
        """
        Calculate reward for the action taken
        Combines multiple factors:
        - P&L
        - Transaction costs
        - Risk-adjusted returns
        - Position alignment with signals
        """
        # Calculate basic P&L
        pnl = execution_info.get('realized_pnl', 0) + \
              execution_info.get('unrealized_pnl', 0)
              
        # Calculate transaction costs
        transaction_cost = execution_info.get('commission', 0) + \
                         execution_info.get('slippage', 0)
                         
        # Calculate risk-adjusted component
        risk_adjustment = self._calculate_risk_adjustment(new_state)
        
        # Calculate signal alignment
        signal_alignment = self._calculate_signal_alignment(
            execution_info['position'],
            new_state
        )
        
        # Combine components
        reward = (pnl - transaction_cost) * risk_adjustment * signal_alignment
        
        return reward
        
    def _calculate_risk_adjustment(self, state: State) -> float:
        """Calculate risk adjustment factor"""
        volatility = state.technical_indicators.get('historical_volatility', 1.0)
        var = state.technical_indicators.get('value_at_risk', 0.0)
        
        return 1.0 / (1.0 + volatility * abs(var))
        
    def _calculate_signal_alignment(self, position: float, 
                                  state: State) -> float:
        """Calculate how well position aligns with signals"""
        # Combine various signals
        technical_signal = self._get_technical_signal(state)
        sentiment_signal = self._get_sentiment_signal(state)
        flow_signal = self._get_order_flow_signal(state)
        
        # Weighted combination of signals
        combined_signal = (
            0.4 * technical_signal +
            0.3 * sentiment_signal +
            0.3 * flow_signal
        )
        
        # Calculate alignment
        alignment = np.sign(position) * np.sign(combined_signal)
        return max(0.1, alignment)
        
    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        # Check various conditions
        time_done = self.current_step >= self.config['max_steps']
        capital_done = self._check_capital_threshold()
        risk_done = self._check_risk_limits()
        
        return time_done or capital_done or risk_done
        
    def reset(self) -> State:
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_position = 0
        self.position_history = []
        self.reward_history = []
        self.state_history = []
        
        return self.get_state(self.config['symbol'])