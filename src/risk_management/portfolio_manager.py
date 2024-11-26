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
        self.constraints = PortfolioConstraints(**config['constraints'])
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
                    prices[symbol] = data['Close'].iloc[-1]
            except Exception as e:
                self.logger.error(f"Error getting yfinance price for {symbol}: {str(e)}")
        return prices

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
                    df['symbol'] = symbol
                    dfs.append(df[['Close']].reset_index())
            
            # Combine and calculate returns
            combined_df = pd.concat(dfs)
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