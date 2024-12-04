# src/data/data_manager.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import time
from typing import Dict, List, Optional

class DataManager:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize clients using config
        self.trading_client = TradingClient(
            config['alpaca_api_key'],
            config['alpaca_secret_key']
        )
        self.data_client = StockHistoricalDataClient(
            config['alpaca_api_key'],
            config['alpaca_secret_key']
        )
        
        # Initialize database
        self.db_path = 'market_data.db'
        self._initialize_database()
        
        # Cache settings
        self.cache_duration = timedelta(minutes=15)
        self._last_update = {}

    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Create market data table
            c.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    Date TIMESTAMP,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    dividends REAL,
                    source TEXT
                )
            ''')
            
            # Create fundamental data table
            c.execute('''
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    symbol TEXT,
                    timestamp DATETIME,
                    pe_ratio REAL,
                    peg_ratio REAL,
                    profit_margins REAL,
                    revenue_growth REAL,
                    debt_to_equity REAL,
                    current_ratio REAL,
                    return_on_equity REAL,
                    beta REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise

    def get_market_data(self, symbol: str, start_date: Optional[datetime], end_date: Optional[datetime], period: str = "6mo") -> pd.DataFrame:
        """Get market data from database or APIs."""
        try:
            # If no new data is needed, use the cache
            if not self._needs_update(symbol):
                data = self._get_cached_data(symbol, start_date, end_date)
                if not data.empty:
                    return data
            
            # Fetch new data from APIs
            data = self._fetch_alpaca_data(symbol, start_date, end_date)
            if data.empty:
                data = self._fetch_yfinance_data(symbol, period)
            
            # Save and cache new data
            if not data.empty:
                self._save_market_data(symbol, data)
                self._last_update[symbol] = datetime.now()
            
            return data

        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _needs_update(self, symbol: str) -> bool:
        """Check if data needs to be updated"""
        if symbol not in self._last_update:
            return True
        return datetime.now() - self._last_update[symbol] > self.cache_duration

    def _get_cached_data(self, 
                        symbol: str, 
                        start_date: Optional[datetime],
                        end_date: Optional[datetime]) -> pd.DataFrame:
        """Get data from SQLite database"""
        try:
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND Date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND Date <= ?"
                params.append(end_date)
                
            query += " ORDER BY Date"
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting cached data: {str(e)}")
            return pd.DataFrame()

    def _fetch_alpaca_data(self, 
                          symbol: str,
                          start_date: Optional[datetime],
                          end_date: Optional[datetime]) -> pd.DataFrame:
        """Fetch data from Alpaca"""
        try:
            # For free plan, don't set end date to get data up to 15 minutes ago
            request_params = {
                'symbol_or_symbols': symbol,
                'timeframe': TimeFrame.Day,
                'start': start_date if start_date else datetime.now() - timedelta(days=30)
            }
            
            bars_request = StockBarsRequest(**request_params)
            bars = self.data_client.get_stock_bars(bars_request)
            
            if symbol in bars:
                df = pd.DataFrame([{
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume),
                    'source': 'alpaca'
                } for bar in bars[symbol]])
                
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                return df
                
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpaca data: {str(e)}")
            return pd.DataFrame()

    def _fetch_yfinance_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                # Standardize column names
                data.columns = data.columns.str.lower()
                data['source'] = 'yfinance'
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching yfinance data: {str(e)}")
            return pd.DataFrame()

    def _save_market_data(self, symbol: str, data: pd.DataFrame):
        """Save market data to database."""
        try:
            if data.empty:
                return

            conn = sqlite3.connect(self.db_path)
            
            # Reset index for SQL compatibility
            data = data.reset_index()
            data['symbol'] = symbol
            
            # Rename timestamp column to Date if needed
            if 'timestamp' in data.columns:
                data = data.rename(columns={'timestamp': 'Date'})
            
            # Drop unwanted columns (e.g., stock splits)
            if 'capital gains' in data.columns:
                data = data.drop(columns=['capital gains'])
            if 'stock splits' in data.columns:
                data = data.drop(columns=['stock splits'])
            
            # Ensure all required columns exist
            required_cols = ['symbol', 'Date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'source']
            for col in required_cols:
                if col not in data.columns:
                    if col in ['dividends']:
                        data[col] = 0.0
                    elif col == 'source':
                        data[col] = 'unknown'

            # Insert data into the database
            data.to_sql('market_data', conn, if_exists='append', index=False)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving market data for {symbol}: {str(e)}")

    def get_fundamental_data(self, symbol: str) -> Dict:
        """Get fundamental data from database or yfinance"""
        try:
            # Check cached data first
            cached_data = self._get_cached_fundamental_data(symbol)
            if not self._needs_update(f"{symbol}_fundamental"):
                return cached_data
            
            # Fetch new data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamental_data = {
                'pe_ratio': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'profit_margins': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'beta': info.get('beta', 1)
            }
            
            # Save to database
            self._save_fundamental_data(symbol, fundamental_data)
            self._last_update[f"{symbol}_fundamental"] = datetime.now()
            
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"Error getting fundamental data: {str(e)}")
            return {}

    def _get_cached_fundamental_data(self, symbol: str) -> Dict:
        """Get fundamental data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT * FROM fundamental_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            df = pd.read_sql_query(query, conn, params=[symbol])
            conn.close()
            
            if not df.empty:
                return df.iloc[0].to_dict()
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting cached fundamental data: {str(e)}")
            return {}

    def _save_fundamental_data(self, symbol: str, data: Dict):
        """Save fundamental data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            data['symbol'] = symbol
            data['timestamp'] = datetime.now()
            
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            
            query = f"""
                INSERT INTO fundamental_data ({columns})
                VALUES ({placeholders})
            """
            
            c.execute(query, list(data.values()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving fundamental data: {str(e)}")

    def get_average_volume(self, symbol: str) -> float:
        """Get average trading volume for a symbol"""
        try:
            # Get last 20 days of data
            data = self.get_market_data(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=20),
                end_date=datetime.now()
            )
            
            if data.empty:
                return 0.0
                
            return float(data['volume'].mean())
            
        except Exception as e:
            self.logger.error(f"Error getting average volume: {str(e)}")
            return 0.0

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol"""
        try:
            data = self.get_market_data(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            
            if data.empty:
                return 0.0
                
            return float(data['close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error getting latest price: {str(e)}")
            return 0.0