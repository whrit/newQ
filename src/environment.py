# src/environment.py
import numpy as np
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass
import pandas as pd
import logging
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf

from src.data.data_manager import DataManager
from .features.technical_indicators import TechnicalAnalyzer
from .features.market_sentiment import MarketSentimentAnalyzer
from .features.order_book_features import OrderBookAnalyzer
from .features.sector_analysis import SectorAnalyzer
from .risk_management.position_sizer import PositionSizer

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

class StateNormalizer:
    """Normalize state values for better learning"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running_mean = None
        self.running_std = None
        self.epsilon = 1e-8
        self.logged_dimensions = False  # Track if we've logged dimensions

    def update_stats(self, state: State) -> None:
        """Update running statistics"""
        state_vector = self._state_to_vector(state)
        
        if self.running_mean is None:
            self.running_mean = state_vector
            self.running_std = np.ones_like(state_vector)
        else:
            # Update running mean and std using exponential moving average
            self.running_mean = 0.99 * self.running_mean + 0.01 * state_vector
            self.running_std = 0.99 * self.running_std + 0.01 * np.abs(state_vector - self.running_mean)

    def normalize(self, state: State) -> np.ndarray:
        """Normalize state values"""
        state_vector = self._state_to_vector(state)
        
        if self.running_mean is None:
            self.update_stats(state)
            
        normalized = (state_vector - self.running_mean) / (self.running_std + self.epsilon)
        return normalized

    def _state_to_vector(self, state: State) -> np.ndarray:
        """Convert state to vector representation with detailed logging"""
        components = []
        
        # Price data
        if hasattr(state, 'price_data'):
            # Check if price_data is a DataFrame
            if isinstance(state.price_data, pd.DataFrame):
                price_values = state.price_data.values.flatten()
            # Check if price_data is a dict
            elif isinstance(state.price_data, dict):
                price_values = list(state.price_data.values())
            # If it's a numpy array
            else:
                price_values = state.price_data.flatten()
            components.extend(price_values)
            if not self.logged_dimensions:
                if isinstance(state.price_data, dict):
                    self.logger.info(f"Price features ({len(price_values)}): {list(state.price_data.keys())}")
                else:
                    self.logger.info(f"Price features ({len(price_values)})")

        # Technical indicators
        if hasattr(state, 'technical_indicators'):
            tech_values = list(state.technical_indicators.values())
            components.extend(tech_values)
            if not self.logged_dimensions:
                self.logger.info(f"Technical features ({len(tech_values)}): {list(state.technical_indicators.keys())}")

        # Sentiment data
        if hasattr(state, 'sentiment_data'):
            # Convert MarketSentiment object to dictionary if needed
            if hasattr(state.sentiment_data, '__dict__'):  # Check if it's an object
                sentiment_dict = state.sentiment_data.__dict__
            elif hasattr(state.sentiment_data, 'to_dict'):  # Check if it has to_dict method
                sentiment_dict = state.sentiment_data.to_dict()
            else:
                sentiment_dict = state.sentiment_data  # Assume it's already a dict
            
            # Filter out non-numeric values
            sent_values = [v for v in sentiment_dict.values() if isinstance(v, (int, float))]
            components.extend(sent_values)
            if not self.logged_dimensions:
                self.logger.info(f"Sentiment features ({len(sent_values)}): {list(sentiment_dict.keys())}")

        # Position info
        if hasattr(state, 'position_info'):
            pos_values = list(state.position_info.values())
            components.extend(pos_values)
            if not self.logged_dimensions:
                self.logger.info(f"Position features ({len(pos_values)}): {list(state.position_info.keys())}")

        # Account info
        if hasattr(state, 'account_info'):
            acc_values = list(state.account_info.values())
            components.extend(acc_values)
            if not self.logged_dimensions:
                self.logger.info(f"Account features ({len(acc_values)}): {list(state.account_info.keys())}")

        state_vector = np.array(components, dtype=np.float32)
        
        if not self.logged_dimensions:
            self.logger.info(f"Total state dimension: {len(state_vector)}")
            self.logged_dimensions = True  # Only log once
        
        # Check for NaN or infinite values
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            self.logger.warning("State vector contains NaN or infinite values!")
            state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return state_vector

class TradingEnvironment:
    def __init__(self, 
                trading_client: TradingClient,
                data_client: StockHistoricalDataClient,
                config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.trading_client = trading_client
        self.data_client = data_client
        
        # Initialize default position size settings if not provided
        if 'position_size' not in config:
            config['position_size'] = {
                'max_position_size': 0.1,  # 10% of portfolio
                'risk_limit': 0.02,        # 2% risk per trade
                'min_size': 100,           # Minimum trade size
                'max_leverage': 1.0,       # Maximum leverage
                'method': 'risk_based'     # Position sizing method
            }
        
        # If config exists but needs parameter alignment
        elif 'position_size' in config:
            position_size_config = config['position_size']
            # Ensure all required parameters exist
            defaults = {
                'max_position_size': 0.1,
                'risk_limit': 0.02,
                'min_size': 100,
                'max_leverage': 1.0,
                'method': 'risk_based'
            }
            # Update with defaults for any missing parameters
            for key, value in defaults.items():
                if key not in position_size_config:
                    position_size_config[key] = value

        self.config = config
        
        # Initialize components
        self.data_manager = DataManager(config=self.config.get('data_config', {}))
        self.technical_analyzer = TechnicalAnalyzer(data_manager=self.data_manager)
        self.sentiment_analyzer = MarketSentimentAnalyzer(
            config=config.get('sentiment_config', {}),
            data_manager=self.data_manager
        )
        self.order_book_analyzer = OrderBookAnalyzer()
        self.sector_analyzer = SectorAnalyzer(data_manager=self.data_manager)
        self.position_sizer = PositionSizer(config)

        # Initialize metrics tracking
        self.state_normalizer = StateNormalizer()
        self.current_step = 0
        self.current_position = 0
        self.position_history = []
        self.reward_history = []
        self.state_history = []
        self.returns_history = []
        self.volatility_history = []
        self.episode_memory = []
        self.max_drawdown = 0.0
        self.best_return = -float('inf')

    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        """Execute one environment step - synchronous version"""
        try:
            # Execute action
            execution_info = self._execute_action(action)
            
            # Get new state
            new_state = self.get_state(self.config['symbol'])
            
            # Calculate reward
            reward = self._calculate_reward(execution_info, new_state)
            
            # Update metrics
            self._update_metrics(new_state, reward)
            
            # Check if episode is done
            done = self._is_episode_done()
            
            return new_state, reward, done, {
                'execution': execution_info,
                'metrics': self._get_metrics(),
                'position_changes': self._get_position_changes()
            }
            
        except Exception as e:
            self.logger.error(f"Error in environment step: {str(e)}")
            raise

    def _calculate_reward(self, execution_info: Dict, new_state: State) -> float:
        """Calculate comprehensive reward"""
        try:
            # Base P&L reward
            pnl = execution_info.get('realized_pnl', 0) + execution_info.get('unrealized_pnl', 0)
            
            # Risk-adjusted returns
            sharpe = self._calculate_risk_adjusted_returns(pnl, new_state)
            
            # Position alignment
            alignment = self._calculate_signal_alignment(execution_info['position'], new_state)
            
            # Transaction costs
            costs = execution_info.get('commission', 0) + execution_info.get('slippage', 0)
            
            # Market impact
            impact = self._calculate_market_impact(execution_info)
            
            # Drawdown penalty
            drawdown = self._calculate_drawdown_penalty()
            
            # Exploration bonus
            exploration = self._calculate_exploration_bonus(new_state)
            
            # Combine all components
            reward = (
                sharpe * 0.3 +
                alignment * 0.2 +
                exploration * 0.1 -
                costs * 0.1 -
                impact * 0.1 -
                drawdown * 0.2
            )
            
            return float(np.clip(reward, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0

    def _calculate_risk_adjusted_returns(self, pnl: float, state: State) -> float:
        """Calculate Sharpe-like ratio"""
        try:
            volatility = state.technical_indicators.get('historical_volatility', 1.0)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_return = pnl - risk_free_rate
            
            # Avoid division by zero
            if volatility < 1e-6:
                volatility = 1e-6
                
            sharpe = excess_return / volatility
            return float(np.clip(sharpe, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted returns: {str(e)}")
            return 0.0

    def _calculate_signal_alignment(self, position: float, state: State) -> float:
        """Calculate alignment with trading signals"""
        try:
            # Get various trading signals
            technical = self._get_technical_signal(state)
            sentiment = self._get_sentiment_signal(state)
            flow = self._get_order_flow_signal(state)
            market = self._get_market_signal(state)
            
            # Combine signals with weights
            combined_signal = (
                technical * 0.4 +
                sentiment * 0.3 +
                flow * 0.2 +
                market * 0.1
            )
            
            # Calculate alignment
            position_sign = np.sign(position) if position != 0 else 0
            alignment = position_sign * combined_signal
            
            return float(np.clip(alignment, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating signal alignment: {str(e)}")
            return 0.0

    def _calculate_market_impact(self, execution_info: Dict) -> float:
        """Calculate market impact of trades"""
        try:
            volume = execution_info.get('volume', 0)
            avg_volume = execution_info.get('avg_volume', 1)
            
            if avg_volume < 1e-6:
                return 0.0
                
            return float(min(1.0, volume / avg_volume))
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact: {str(e)}")
            return 0.0

    def _calculate_drawdown_penalty(self) -> float:
        """Calculate drawdown penalty"""
        try:
            if not self.returns_history:
                return 0.0
                
            returns = np.array(self.returns_history)
            cumulative_returns = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / (np.abs(peak) + 1e-6)
            
            return float(np.clip(drawdown[-1], 0, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown penalty: {str(e)}")
            return 0.0

    def _calculate_exploration_bonus(self, state: State) -> float:
        """Calculate exploration bonus based on state novelty"""
        try:
            state_vector = self.state_normalizer.normalize(state)
            
            if len(self.episode_memory) == 0:
                self.episode_memory.append(state_vector)
                return 1.0
                
            # Calculate distance to recent states
            recent_states = np.array(self.episode_memory[-100:])
            distances = np.linalg.norm(recent_states - state_vector, axis=1)
            min_distance = np.min(distances)
            
            # Convert to novelty score
            novelty = 1.0 - np.exp(-min_distance)
            
            # Update memory
            self.episode_memory.append(state_vector)
            
            return float(novelty)
            
        except Exception as e:
            self.logger.error(f"Error calculating exploration bonus: {str(e)}")
            return 0.0

    def reset(self) -> State:
        """Reset environment"""
        self.current_step = 0
        self.current_position = 0
        self.position_history = []
        self.reward_history = []
        self.state_history = []
        self.episode_memory = []
        self.returns_history = []
        self.volatility_history = []
        self.max_drawdown = 0.0
        
        # Add await here
        initial_state = self.get_state(self.config['symbol'])
        self.state_normalizer.update_stats(initial_state)
        
        return initial_state

    def _fetch_sector_data(self, symbol: str) -> pd.DataFrame:
        """Fetch sector-related data synchronously"""
        try:
            ticker = yf.Ticker(symbol)
            sector = ticker.info.get('sector', '')
            sector_stocks = self._get_sector_stocks(sector)
            
            # Fetch data for all sector stocks
            sector_data = pd.DataFrame()
            start_date = datetime.now() - timedelta(days=90)
            
            for stock in sector_stocks[:10]:
                try:
                    ticker = yf.Ticker(stock)
                    data = ticker.history(start=start_date)
                    if not data.empty:
                        # Ensure consistent column naming
                        if 'Close' in data.columns:
                            data = data.rename(columns={'Close': 'close'})
                        sector_data[stock] = data['close']
                except Exception as e:
                    self.logger.warning(f"Error fetching data for {stock}: {str(e)}")
                    continue
                    
            return sector_data
            
        except Exception as e:
            self.logger.error(f"Error fetching sector data: {str(e)}")
            return pd.DataFrame()

    def _get_sector_stocks(self, sector: str) -> List[str]:
        """Get list of stocks in the same sector"""
        sector_etfs = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO'],
            'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Materials': ['LIN', 'APD', 'ECL', 'DD', 'NEM'],
            'Industrials': ['UPS', 'HON', 'UNP', 'BA', 'CAT'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
            'Communications': ['GOOGL', 'META', 'NFLX', 'CMCSA', 'VZ']
        }
        return sector_etfs.get(sector, ['SPY'])  # Default to SPY if sector not found

    def _fetch_market_indices(self) -> Dict[str, float]:
        """Fetch major market indices data synchronously"""
        try:
            indices = ['SPY', 'QQQ', 'DIA']
            index_data = {}
            
            for index in indices:
                ticker = yf.Ticker(index)
                data = ticker.history(period="1d")
                if not data.empty:
                    index_data[index] = float(data['Close'].iloc[-1])
                    
            return index_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market indices: {str(e)}")
            return {'SPY': 0.0, 'QQQ': 0.0, 'DIA': 0.0}

    async def _fetch_vix(self) -> float:
        """Fetch VIX data using yfinance"""
        try:
            ticker = yf.Ticker("^VIX")
            data = ticker.history(period="1d")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return 20.0  # Default VIX value
            
        except Exception as e:
            self.logger.error(f"Error fetching VIX: {str(e)}")
            return 20.0  # Default VIX value

    async def _fetch_market_breadth(self) -> Dict[str, float]:
        """Fetch market breadth indicators"""
        try:
            # Using advance-decline data from NYSE
            adv_dec_ratio = 1.0  # Default value
            new_highs = 0
            new_lows = 0
            
            return {
                'adv_dec_ratio': adv_dec_ratio,
                'new_highs': new_highs,
                'new_lows': new_lows
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching market breadth: {str(e)}")
            return {
                'adv_dec_ratio': 1.0,
                'new_highs': 0,
                'new_lows': 0
            }
                
    def _fetch_price_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical price data synchronously"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Increased from 5 days to ensure sufficient data
            
            # Try to get data from data manager first
            data = self.data_manager.get_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not data.empty:
                return data
                
            # Fallback to yfinance if needed
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if not data.empty:
                # Standardize column names
                data.columns = data.columns.str.lower()
                return data
                
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()
        
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
        
    def _get_sector_metrics(self, symbol: str) -> Dict[str, float]:
        """Get sector analysis metrics synchronously"""
        try:
            # Fetch sector data
            sector_data = self._fetch_sector_data(symbol)
            
            # Get market data
            market_symbols = ['SPY', 'QQQ', 'DIA']
            market_data = pd.DataFrame()
            
            for market_symbol in market_symbols:
                ticker = yf.Ticker(market_symbol)
                data = ticker.history(period='3mo')
                if not data.empty:
                    market_data[market_symbol] = data['Close']
            
            metrics = self.sector_analyzer.analyze_sector(
                symbol,
                sector_data,
                market_data
            )
            
            return metrics.to_dict() if metrics else {}
            
        except Exception as e:
            self.logger.error(f"Error getting sector metrics: {str(e)}")
            return {}
        
    def _get_market_context(self) -> Dict[str, float]:
        """Get broader market context synchronously"""
        try:
            # Market indices
            indices = self._fetch_market_indices()
            
            # Volatility indicators
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d")
            vix = float(vix_data['Close'].iloc[-1]) if not vix_data.empty else 20.0
            
            # Market breadth
            breadth = {
                'adv_dec_ratio': 1.0,
                'new_highs': 0,
                'new_lows': 0
            }
            
            return {
                'indices': indices,
                'volatility': vix,
                'breadth': breadth
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}")
            return {
                'indices': {'SPY': 0.0, 'QQQ': 0.0, 'DIA': 0.0},
                'volatility': 20.0,
                'breadth': {'adv_dec_ratio': 1.0, 'new_highs': 0, 'new_lows': 0}
            }
        
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
        
    def get_state(self, symbol: str) -> State:
        """Get current environment state - synchronous version"""
        try:
            # Fetch price data
            price_data = self._fetch_price_data(symbol)
            
            # Ensure price_data is a DataFrame
            if isinstance(price_data, dict):
                price_df = pd.DataFrame.from_dict(price_data)
            else:
                price_df = price_data
            
            # Calculate technical features
            technical_features = self.technical_analyzer.calculate_features(
                data=price_df,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            technical_data = {
                **technical_features.trend_indicators,
                **technical_features.momentum_indicators,
                **technical_features.volatility_indicators,
                **technical_features.volume_indicators,
                **technical_features.cycle_indicators
            }
            
            # Get other components synchronously
            sentiment_data = self.sentiment_analyzer.get_sentiment(symbol)
            order_book_features = self.order_book_analyzer.analyze(price_df)
            sector_metrics = self._get_sector_metrics(symbol)
            position_info = self._get_position_info(symbol)
            account_info = self._get_account_info()
            market_context = self._get_market_context()

            return State(
                price_data=price_df.to_dict('records')[0] if not price_df.empty else {
                    'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0.0
                },
                technical_indicators=technical_data,
                sentiment_data=sentiment_data,
                order_book_features=order_book_features,
                sector_metrics=sector_metrics,
                position_info=position_info,
                account_info=account_info,
                market_context=market_context
            )

        except Exception as e:
            self.logger.error(f"Error getting state: {str(e)}")
            raise
            
    def _process_bars(self, bars) -> pd.DataFrame:
        """Process bar data into required format"""
        try:
            # Handle empty bars
            if not bars or not hasattr(bars, '__getitem__'):
                return pd.DataFrame({
                    'open': [0.0],
                    'high': [0.0],
                    'low': [0.0],
                    'close': [0.0],
                    'volume': [0.0]
                }, index=[pd.Timestamp.now()])  # Add timestamp index

            symbol = self.config['symbol']
            
            if symbol in bars:
                bars_list = []
                for bar in bars[symbol]:
                    bars_list.append({
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': float(bar.volume),
                        'timestamp': pd.Timestamp(bar.timestamp)
                    })
                    
                if bars_list:
                    bars_df = pd.DataFrame(bars_list)
                    if not bars_df.empty:
                        bars_df.set_index('timestamp', inplace=True)
                        return bars_df.astype(float)
            
            # Return default DataFrame with timestamp index
            return pd.DataFrame({
                'open': [0.0],
                'high': [0.0],
                'low': [0.0],
                'close': [0.0],
                'volume': [0.0]
            }, index=[pd.Timestamp.now()])
            
        except Exception as e:
            self.logger.error(f"Error processing bars: {str(e)}")
            return pd.DataFrame({
                'open': [0.0],
                'high': [0.0],
                'low': [0.0],
                'close': [0.0],
                'volume': [0.0]
            }, index=[pd.Timestamp.now()])
        
    def _execute_action(self, action: int) -> Dict[str, Any]:
        """Execute trading action synchronously"""
        try:
            symbol = self.config['symbol']
            position = self._get_position_info(symbol)
            current_position = position['quantity']
            
            # Get current price and account information
            price_data = self._fetch_price_data(symbol)
            current_price = price_data['close'].iloc[-1] if not price_data.empty else 0.0
            account = self._get_account_info()
            
            if action == 0:  # Hold
                return {
                    'action': 'hold',
                    'position': current_position,
                    'realized_pnl': 0.0,
                    'unrealized_pnl': position['unrealized_pl'],
                    'commission': 0.0,
                    'slippage': 0.0,
                    'volume': 0.0,
                    'avg_volume': self.data_manager.get_average_volume(symbol)
                }
            
            # Calculate dynamic position size
            position_size = self._calculate_position_size(account['equity'])
            
            if action == 1:  # Buy
                new_position = current_position + position_size
                return {
                    'action': 'buy',
                    'position': new_position,
                    'realized_pnl': 0.0,
                    'unrealized_pnl': position['unrealized_pl'],
                    'commission': position_size * current_price * 0.001,
                    'slippage': position_size * current_price * 0.0005,
                    'volume': position_size,
                    'avg_volume': self.data_manager.get_average_volume(symbol)
                }
                
            elif action == -1:  # Sell
                new_position = current_position - position_size
                return {
                    'action': 'sell',
                    'position': new_position,
                    'realized_pnl': position['unrealized_pl'],
                    'unrealized_pnl': 0.0,
                    'commission': position_size * current_price * 0.001,
                    'slippage': position_size * current_price * 0.0005,
                    'volume': position_size,
                    'avg_volume': self.data_manager.get_average_volume(symbol)
                }
                
        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")
            return {
                'action': 'error',
                'position': current_position,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'commission': 0.0,
                'slippage': 0.0,
                'volume': 0.0,
                'avg_volume': 1.0
            }

    def _calculate_position_size(self, equity: float) -> float:
        """Calculate position size using PositionSizer"""
        try:
            symbol = self.config['symbol']
            
            # Get current price and historical data
            price_df = self.data_manager.get_market_data(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            if price_df.empty:
                return 0.0
                
            current_price = float(price_df['close'].iloc[-1])
            
            # Calculate technical features properly
            technical_features = self.technical_analyzer.calculate_features(
                data=price_df,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            # Get volatility
            volatility = technical_features.volatility_indicators.get('historical_volatility', 0.02)
            
            # Use position sizer
            position_size = self.position_sizer.calculate_position_size(
                price=current_price,
                volatility=volatility,
                account_value=equity,
                win_rate=None,
                profit_factor=None
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _update_metrics(self, state: State, reward: float) -> None:
        """Update environment metrics"""
        try:
            # Update histories
            self.state_history.append(state)
            self.reward_history.append(reward)
            
            # Calculate returns if possible
            if len(self.state_history) > 1:
                prev_price = float(self.state_history[-2].price_data.get('close', 0))
                curr_price = float(state.price_data.get('close', 0))
                
                if prev_price > 0 and curr_price > 0:
                    returns = (curr_price - prev_price) / prev_price
                    self.returns_history.append(returns)
                    
                    # Update max drawdown
                    if len(self.returns_history) > 1:
                        cumulative_returns = np.cumsum(self.returns_history)
                        peak = np.maximum.accumulate(cumulative_returns)
                        drawdown = (peak - cumulative_returns) / (np.abs(peak) + 1e-6)
                        self.max_drawdown = max(self.max_drawdown, drawdown[-1])
            
            # Update position history
            pos_qty = state.position_info.get('quantity', 0) if isinstance(state.position_info, dict) else 0
            self.position_history.append(pos_qty)
            
            # Update volatility history safely
            if isinstance(state.technical_indicators, dict):
                vol = state.technical_indicators.get('historical_volatility', 0)
                self.volatility_history.append(vol)
                
            self.current_step += 1
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def _get_metrics(self) -> Dict:
        """Get current metrics"""
        try:
            return {
                'total_reward': sum(self.reward_history),
                'average_reward': np.mean(self.reward_history) if self.reward_history else 0,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'volatility': np.std(self.returns_history) if self.returns_history else 0,
                'position_changes': len([i for i in range(1, len(self.position_history)) 
                                    if self.position_history[i] != self.position_history[i-1]])
            }
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {}

    def _get_position_changes(self) -> List[Tuple[int, float]]:
        """Get history of position changes"""
        try:
            changes = []
            for i in range(1, len(self.position_history)):
                if self.position_history[i] != self.position_history[i-1]:
                    changes.append((i, self.position_history[i]))
            return changes
        except Exception as e:
            self.logger.error(f"Error getting position changes: {str(e)}")
            return []

    def _get_technical_signal(self, state: State) -> float:
        """Get technical analysis signal"""
        try:
            signals = []
            tech_indicators = state.technical_indicators
            
            # Trend signals
            if all(k in tech_indicators for k in ['sma_short', 'sma_medium']):
                signals.append(1 if tech_indicators['sma_short'] > tech_indicators['sma_medium'] else -1)
                
            if 'macd' in tech_indicators:
                signals.append(1 if tech_indicators['macd'] > 0 else -1)
                
            # Momentum signals
            if 'rsi' in tech_indicators:
                rsi = tech_indicators['rsi']
                if rsi > 70:
                    signals.append(-1)
                elif rsi < 30:
                    signals.append(1)
                else:
                    signals.append(0)
                    
            return np.mean(signals) if signals else 0
            
        except Exception as e:
            self.logger.error(f"Error getting technical signal: {str(e)}")
            return 0

    def _get_sentiment_signal(self, state: State) -> float:
        """Get sentiment signal"""
        try:
            if isinstance(state.sentiment_data, dict):
                return state.sentiment_data.get('composite_score', 0)
            return 0
        except Exception as e:
            self.logger.error(f"Error getting sentiment signal: {str(e)}")
            return 0

    def _get_order_flow_signal(self, state: State) -> float:
        """Get order flow signal"""
        try:
            if isinstance(state.order_book_features, dict):
                signals = [
                    state.order_book_features.get('price_trend', 0),
                    state.order_book_features.get('volume_profile', 0)
                ]
                return np.mean([s for s in signals if s is not None])
            return 0
        except Exception as e:
            self.logger.error(f"Error getting order flow signal: {str(e)}")
            return 0

    def _get_market_signal(self, state: State) -> float:
        """Get overall market signal"""
        try:
            if isinstance(state.market_context, dict):
                # Get VIX signal (inverse relationship)
                vix = state.market_context.get('volatility', 20)
                vix_signal = -1 if vix > 30 else 1 if vix < 15 else 0
                
                # Get market breadth signal
                breadth = state.market_context.get('breadth', {})
                breadth_signal = 1 if breadth.get('adv_dec_ratio', 1) > 1.5 else -1
                
                return np.mean([vix_signal, breadth_signal])
            return 0
        except Exception as e:
            self.logger.error(f"Error getting market signal: {str(e)}")
            return 0

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not self.returns_history:
                return 0.0
                
            returns = np.array(self.returns_history)
            excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
            
            if len(returns) < 2:
                return 0.0
                
            sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-6) * np.sqrt(252)
            return float(sharpe)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _check_capital_threshold(self) -> bool:
        """Check if capital has fallen below threshold"""
        try:
            if not self.state_history:
                return False
                
            current_equity = self.state_history[-1].account_info['equity']
            initial_equity = self.state_history[0].account_info['equity']
            
            return current_equity < initial_equity * 0.8  # 20% drawdown threshold
            
        except Exception as e:
            self.logger.error(f"Error checking capital threshold: {str(e)}")
            return False

    def _check_risk_limits(self) -> bool:
        """Check if risk limits have been exceeded"""
        try:
            if len(self.volatility_history) < 2:
                return False
                
            current_vol = self.volatility_history[-1]
            vol_threshold = 0.4  # 40% annualized volatility threshold
            
            return current_vol > vol_threshold
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return False