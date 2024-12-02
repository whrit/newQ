# src/risk_management/portfolio_manager.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import logging
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

@dataclass
class PortfolioMetrics:
    total_value: float
    cash: float
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    leverage: float
    sector_exposure: Dict[str, float]
    position_concentration: float
    beta: float
    correlation_matrix: pd.DataFrame
    sharpe_ratio: float  # Added
    volatility: float    # Added

@dataclass
class PortfolioConstraints:
    max_position_size: float
    max_sector_exposure: float
    max_leverage: float
    max_concentration: float
    max_correlation: float
    min_cash_buffer: float

class PortfolioManager:
    """
    Advanced portfolio management with risk controls and optimization
    Using Alpaca and yfinance for market data
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.positions = {}
        self.sector_mappings = {}
        
        # Set default constraints if not provided in config
        default_constraints = {
            'max_position_size': 0.2,
            'max_sector_exposure': 0.3,
            'max_leverage': 1.5,
            'max_concentration': 0.25,
            'max_correlation': 0.7,
            'min_cash_buffer': 0.05
        }
        
        # Use provided constraints or defaults
        constraints_config = config.get('constraints', default_constraints)
        self.constraints = PortfolioConstraints(**constraints_config)
        
        self.historical_data = {}
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            config['alpaca_api_key'],
            config['alpaca_secret_key']
        )
        self.data_client = StockHistoricalDataClient(
            config['alpaca_api_key'],
            config['alpaca_secret_key']
        )
        
        # Cache for sector data
        self._sector_cache = {}
        self._sector_cache_expiry = {}
        self.SECTOR_CACHE_DURATION = timedelta(days=1)
        
    async def update_portfolio(self, positions: Optional[Dict[str, float]] = None) -> PortfolioMetrics:
        """
        Update portfolio state and calculate metrics using Alpaca data
        """
        try:
            # If positions not provided, get from Alpaca
            if positions is None:
                positions = await self._get_alpaca_positions()
            
            self.positions = positions
            
            # Get current prices from Alpaca
            prices = await self._get_current_prices(list(positions.keys()))
            account = self.trading_client.get_account()
            account_value = float(account.equity)
            
            # Calculate metrics
            metrics = await self._calculate_portfolio_metrics(positions, prices, account_value)
            
            # Check constraints
            violations = self._check_constraints(metrics)
            if violations:
                self.logger.warning(f"Portfolio constraints violated: {violations}")
                
            # Update historical data
            await self._update_historical_data(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {str(e)}")
            raise
            
    async def _get_alpaca_positions(self) -> Dict[str, float]:
        """Get current positions from Alpaca"""
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            return {
                pos.symbol: float(pos.qty)
                for pos in alpaca_positions
            }
        except Exception as e:
            self.logger.error(f"Error getting Alpaca positions: {str(e)}")
            return {}

    async def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices using Alpaca"""
        try:
            bars_request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                limit=1
            )
            bars = self.data_client.get_stock_bars(bars_request)
            
            return {
                symbol: bars[symbol][-1].close
                for symbol in symbols
                if symbol in bars
            }
        except Exception as e:
            self.logger.error(f"Error getting prices from Alpaca: {str(e)}")
            # Fallback to yfinance
            return self._get_yfinance_prices(symbols)

    def _get_yfinance_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fallback to yfinance for prices"""
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    # Use standardized column name
                    prices[symbol] = data['Close'].iloc[-1]  # Keep 'Close' capitalized for direct access
            except Exception as e:
                self.logger.error(f"Error getting yfinance price for {symbol}: {str(e)}")
        return prices

    def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for symbol and standardize column names"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo")
            
            # Standardize column names to lowercase
            data.columns = data.columns.str.lower()
            
            return data
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return pd.DataFrame()
        
    async def _get_historical_returns(self, symbols: List[str]) -> pd.DataFrame:
        """Get historical returns using Alpaca with yfinance fallback"""
        try:
            # Get 1 year of daily data from Alpaca
            bars_request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=(datetime.now() - timedelta(days=365))
            )
            bars = self.data_client.get_stock_bars(bars_request)
            
            # Convert to DataFrame
            dfs = []
            for symbol in symbols:
                if symbol in bars:
                    symbol_bars = bars[symbol]
                    df = pd.DataFrame([{
                        'date': bar.timestamp,
                        'close': bar.close,
                        'symbol': symbol
                    } for bar in symbol_bars])
                    dfs.append(df)
                else:
                    # Fallback to yfinance
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period="1y")
                    # Standardize column names
                    df = df.reset_index()
                    df.columns = df.columns.str.lower()
                    df = df.rename(columns={'index': 'date', 'close': 'close'})
                    df['symbol'] = symbol
                    dfs.append(df[['date', 'close', 'symbol']])
            
            if not dfs:
                return pd.DataFrame()
                
            # Combine and calculate returns
            combined_df = pd.concat(dfs, ignore_index=True)
            pivot_df = combined_df.pivot(
                index='date',
                columns='symbol',
                values='close'
            )
            
            return pivot_df.pct_change().dropna()
                
        except Exception as e:
            self.logger.error(f"Error getting historical returns: {str(e)}")
            return pd.DataFrame()

    async def _get_asset_betas(self) -> Dict[str, float]:
        """Calculate asset betas using market data"""
        try:
            symbols = list(self.positions.keys())
            
            # Get market (SPY) data
            spy_data = await self._get_historical_returns(['SPY'])
            
            # Get asset data
            asset_data = await self._get_historical_returns(symbols)
            
            betas = {}
            for symbol in symbols:
                if symbol in asset_data.columns:
                    cov = asset_data[symbol].cov(spy_data['SPY'])
                    var = spy_data['SPY'].var()
                    betas[symbol] = cov / var if var != 0 else 1.0
                else:
                    betas[symbol] = 1.0
                    
            return betas
            
        except Exception as e:
            self.logger.error(f"Error calculating betas: {str(e)}")
            return {symbol: 1.0 for symbol in self.positions.keys()}

    async def _get_sector_mappings(self, symbol: str) -> str:
        """Get sector information from yfinance"""
        try:
            if symbol in self._sector_cache:
                if datetime.now() - self._sector_cache_expiry[symbol] < self.SECTOR_CACHE_DURATION:
                    return self._sector_cache[symbol]
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            
            # Update cache
            self._sector_cache[symbol] = sector
            self._sector_cache_expiry[symbol] = datetime.now()
            
            return sector
            
        except Exception as e:
            self.logger.error(f"Error getting sector for {symbol}: {str(e)}")
            return 'Unknown'

    def _get_nav(self) -> float:
        """Get current portfolio NAV from Alpaca"""
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            self.logger.error(f"Error getting NAV: {str(e)}")
            return sum(self.positions.values())
        
    async def _calculate_portfolio_metrics(self, positions: Dict[str, float],
                                        prices: Dict[str, float],
                                        account_value: float) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            # Calculate exposures
            position_values = {symbol: size * prices.get(symbol, 0) 
                            for symbol, size in positions.items()}
            
            long_exposure = sum(v for v in position_values.values() if v > 0)
            short_exposure = sum(v for v in position_values.values() if v < 0)
            net_exposure = long_exposure + short_exposure
            gross_exposure = long_exposure - short_exposure
            
            # Calculate leverage
            leverage = gross_exposure / account_value if account_value > 0 else 0
            
            # Calculate sector exposures
            sector_exposure = {}
            for symbol, value in position_values.items():
                sector = await self._get_sector_mappings(symbol)
                sector_exposure[sector] = sector_exposure.get(sector, 0) + value
                
            # Normalize sector exposure by total value
            sector_exposure = {k: v/account_value for k, v in sector_exposure.items()}
            
            # Calculate concentration
            position_concentration = self._calculate_concentration(position_values, account_value)
            
            # Calculate beta and correlation - Fix the await statements
            betas = await self._get_asset_betas()
            beta = sum(betas.values()) / len(positions) if positions else 0
            
            returns = await self._get_historical_returns(list(positions.keys()))
            correlation_matrix = returns.corr() if not returns.empty else pd.DataFrame()
            
            # Calculate performance metrics
            volatility = returns.std().mean() * np.sqrt(252) if not returns.empty else 0
            
            # Calculate Sharpe ratio
            if not returns.empty:
                excess_returns = returns.mean() - 0.02/252  # Assuming 2% risk-free rate
                sharpe_ratio = np.sqrt(252) * excess_returns.mean() / volatility if volatility > 0 else 0
            else:
                sharpe_ratio = 0
            
            return PortfolioMetrics(
                total_value=account_value,
                cash=account_value - net_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                leverage=leverage,
                sector_exposure=sector_exposure,
                position_concentration=position_concentration,
                beta=beta,
                correlation_matrix=correlation_matrix,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            raise

    def _check_constraints(self, metrics: PortfolioMetrics) -> List[str]:
        """Check portfolio constraints and return list of violations"""
        violations = []
        
        try:
            # Check leverage
            if metrics.leverage > self.constraints.max_leverage:
                violations.append(
                    f"Leverage {metrics.leverage:.2f} exceeds max {self.constraints.max_leverage}"
                )
                
            # Check position concentration
            if metrics.position_concentration > self.constraints.max_concentration:
                violations.append(
                    f"Concentration {metrics.position_concentration:.2f} exceeds max {self.constraints.max_concentration}"
                )
                
            # Check sector exposure
            for sector, exposure in metrics.sector_exposure.items():
                if abs(exposure) > self.constraints.max_sector_exposure:
                    violations.append(
                        f"Sector {sector} exposure {exposure:.2f} exceeds max {self.constraints.max_sector_exposure}"
                    )
                    
            # Check cash buffer
            cash_ratio = metrics.cash / metrics.total_value
            if cash_ratio < self.constraints.min_cash_buffer:
                violations.append(
                    f"Cash buffer {cash_ratio:.2f} below minimum {self.constraints.min_cash_buffer}"
                )
                
            # Check correlation constraints
            if len(metrics.correlation_matrix) > 1:  # Only check if we have multiple positions
                correlations = metrics.correlation_matrix.values
                np.fill_diagonal(correlations, 0)  # Ignore self-correlations
                max_correlation = np.max(np.abs(correlations))
                
                if max_correlation > self.constraints.max_correlation:
                    violations.append(
                        f"Maximum correlation {max_correlation:.2f} exceeds limit {self.constraints.max_correlation}"
                    )
                    
            return violations
            
        except Exception as e:
            self.logger.error(f"Error checking constraints: {str(e)}")
            return [f"Error checking constraints: {str(e)}"]

    def _calculate_concentration(self, position_values: Dict[str, float], 
                            total_value: float) -> float:
        """Calculate portfolio concentration (Herfindahl index)"""
        try:
            if not position_values or total_value == 0:
                return 0
                
            # Calculate position weights
            weights = [abs(v)/total_value for v in position_values.values()]
            
            # Calculate Herfindahl index
            return sum(w*w for w in weights)
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration: {str(e)}")
            return 0

    async def _update_historical_data(self, metrics: PortfolioMetrics):
        """Update historical data for portfolio analysis"""
        try:
            timestamp = datetime.now()
            
            self.historical_data[timestamp] = {
                'total_value': metrics.total_value,
                'cash': metrics.cash,
                'leverage': metrics.leverage,
                'position_concentration': metrics.position_concentration,
                'sharpe_ratio': metrics.sharpe_ratio,
                'volatility': metrics.volatility
            }
            
            # Keep only last 30 days of data
            cutoff = timestamp - timedelta(days=30)
            self.historical_data = {
                k: v for k, v in self.historical_data.items()
                if k > cutoff
            }
            
        except Exception as e:
            self.logger.error(f"Error updating historical data: {str(e)}")

    async def optimize_portfolio(self) -> Dict[str, float]:
        """
        Enhanced portfolio optimization with real market data
        """
        try:
            # Get current portfolio state
            metrics = await self.update_portfolio()
            
            # Get historical data for optimization
            returns = await self._get_historical_returns(list(self.positions.keys()))
            
            # Calculate optimization inputs
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            sharpe = returns.mean() / volatility if volatility.any() != 0 else 0
            
            # Calculate target weights
            target_weights = self._calculate_target_weights_with_constraints(
                returns, 
                metrics,
                max_weight=self.constraints.max_position_size
            )
            
            # Calculate adjustments
            adjustments = {}
            for symbol, current_position in self.positions.items():
                target_position = target_weights.get(symbol, 0) * self._get_nav()
                if abs(target_position - current_position) > self.config['min_adjustment']:
                    adjustments[symbol] = target_position - current_position
                    
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {str(e)}")
            return {}

    def _calculate_target_weights_with_constraints(self,
                                                returns: pd.DataFrame,
                                                metrics: PortfolioMetrics,
                                                max_weight: float) -> Dict[str, float]:
        """Calculate target weights with constraints using risk-adjusted returns"""
        try:
            # Calculate risk metrics
            vol = returns.std() * np.sqrt(252)
            sharpe = returns.mean() / vol
            
            # Initial weights based on Sharpe ratio
            raw_weights = {symbol: max(0, s) for symbol, s in sharpe.items()}
            
            # Apply maximum weight constraint
            for symbol in raw_weights:
                raw_weights[symbol] = min(raw_weights[symbol], max_weight)
            
            # Normalize weights
            total = sum(raw_weights.values())
            if total > 0:
                weights = {k: v/total for k, v in raw_weights.items()}
            else:
                weights = {k: 1/len(raw_weights) for k in raw_weights}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating target weights: {str(e)}")
            return {symbol: 1/len(self.positions) for symbol in self.positions}