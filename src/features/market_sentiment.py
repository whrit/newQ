# src/features/market_sentiment.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta, timezone
import logging
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from src.features.technical_indicators import TechnicalAnalyzer
import yfinance as yf
from src.data.data_manager import DataManager

@dataclass
class SentimentScore:
    value: float  # -1 to 1
    confidence: float  # 0 to 1
    source: str
    timestamp: datetime

@dataclass
class NewsItem:
    title: str
    url: str
    time_published: datetime
    summary: str
    source: str
    sentiment_score: float
    relevance_score: float
    topics: List[str]

@dataclass
class MarketSentiment:
    composite_score: float
    news_sentiment: Dict[str, SentimentScore]
    technical_sentiment: Dict[str, float]
    fundamental_sentiment: Dict[str, float]
    recent_news: List[NewsItem]

class AlphaVantageNewsClient:
    """Client for Alpha Vantage News Sentiment API"""
    def __init__(self, api_key: str, base_url: str = "https://www.alphavantage.co/query"):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    async def get_market_news(self, 
                            symbol: str, 
                            topics: Optional[List[str]] = None,
                            time_from: Optional[str] = None) -> List[NewsItem]:
        """Fetch market news from Alpha Vantage"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": symbol,
                    "apikey": self.api_key,
                    "sort": "RELEVANCE",
                    "limit": 50
                }
                
                if topics:
                    params["topics"] = ",".join(topics)
                if time_from:
                    params["time_from"] = time_from

                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_news_response(data)
                    else:
                        self.logger.error(f"Alpha Vantage API error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error fetching news: {str(e)}")
            return []

    def _parse_news_response(self, data: Dict) -> List[NewsItem]:
        """Parse Alpha Vantage news response"""
        news_items = []
        for item in data.get("feed", []):
            try:
                news_items.append(NewsItem(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    time_published=datetime.strptime(
                        item.get("time_published", ""), 
                        "%Y%m%dT%H%M%S"
                    ),
                    summary=item.get("summary", ""),
                    source=item.get("source", ""),
                    sentiment_score=float(item.get("overall_sentiment_score", 0)),
                    relevance_score=float(item.get("relevance_score", 0)),
                    topics=item.get("topics", [])
                ))
            except Exception as e:
                self.logger.error(f"Error parsing news item: {str(e)}")
                continue
                
        return news_items

class MarketSentimentAnalyzer:
    """
    Market sentiment analysis using news, technical, and fundamental data.
    """
    def __init__(self, config: Dict, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Initialize Alpha Vantage client
        self.alpha_vantage = AlphaVantageNewsClient(config.get("alpha_vantage_api_key"))
        
        if not data_manager:
            raise ValueError("A valid DataManager instance is required.")
        
        # Initialize TechnicalAnalyzer with DataManager
        self.technical_analyzer = TechnicalAnalyzer(data_manager=data_manager)

    def _check_cache(self, symbol: str) -> bool:
        """Check if we have valid cached data for the symbol"""
        if symbol not in self.cache:
            return False

        timestamp, _ = self.cache[symbol]
        return datetime.now() - timestamp < self.cache_duration

    def _update_cache(self, symbol: str, data: Dict) -> None:
        """Update cache with new sentiment data"""
        self.cache[symbol] = (datetime.now(), data)

    def analyze_sentiment(self, symbol: str) -> Dict:
        """Analyze market sentiment with synchronous calls"""
        if self._check_cache(symbol):
            return self.cache[symbol][1]
                
        try:
            # Gather sentiment from different sources
            news_sentiment = self._analyze_news_sentiment(symbol)
            technical_sentiment = self._analyze_technical_sentiment(symbol)
            fundamental_sentiment = self._analyze_fundamental_sentiment(symbol)
                
            # Calculate composite score with adjusted weights
            composite_score = self._calculate_composite_score(
                news_sentiment["sentiment_scores"],
                technical_sentiment,
                fundamental_sentiment
            )
                
            sentiment = {
                "composite_score": composite_score,
                "news_sentiment": news_sentiment["sentiment_scores"],
                "technical_sentiment": technical_sentiment,
                "fundamental_sentiment": fundamental_sentiment,
                "recent_news": news_sentiment["news_items"]
            }
                
            self._update_cache(symbol, sentiment)
            return sentiment
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            raise

    def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment with improved error handling - synchronous version"""
        if not symbol:
            self.logger.warning("Symbol cannot be None or empty")
            return {"sentiment_scores": {}, "news_items": []}
            
        try:
            time_from = (datetime.now() - timedelta(days=1)).strftime("%Y%m%dT%H%M")
            
            topics = [
                "earnings",
                "financial_markets",
                "economy_macro",
                "mergers_and_acquisitions",
                "technology"
            ]
            
            # Use requests instead of aiohttp for synchronous calls
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.alpha_vantage.api_key,
                "sort": "RELEVANCE",
                "limit": 50,
                "topics": ",".join(topics),
                "time_from": time_from
            }
            
            response = requests.get(self.alpha_vantage.base_url, params=params)
            
            if response.status_code != 200:
                self.logger.error(f"Alpha Vantage API error: {response.status_code}")
                return {"sentiment_scores": {}, "news_items": []}
                
            data = response.json()
            news_items = self.alpha_vantage._parse_news_response(data)
            
            if not news_items:
                return {
                    "sentiment_scores": {},
                    "news_items": []
                }
                
            # Calculate weighted sentiment scores
            source_sentiments = {}
            for source in set(item.source for item in news_items if item.source):
                source_items = [item for item in news_items if item.source == source]
                if source_items:
                    weighted_sentiment = self._calculate_weighted_sentiment(source_items)
                    confidence = np.mean([item.relevance_score for item in source_items 
                                    if isinstance(item.relevance_score, (int, float))])
                    
                    source_sentiments[source] = SentimentScore(
                        value=weighted_sentiment,
                        confidence=confidence,
                        source=source,
                        timestamp=datetime.now()
                    )
                    
            return {
                "sentiment_scores": source_sentiments,
                "news_items": news_items
            }
            
        except Exception as e:
            self.logger.error(f"Error in news sentiment analysis: {str(e)}")
            return {"sentiment_scores": {}, "news_items": []}
            
    def _calculate_weighted_sentiment(self, news_items: List[NewsItem]) -> float:
        """Calculate weighted sentiment score with improved error handling"""
        if not news_items:
            return 0.0
            
        try:
            weights = []
            sentiments = []
            
            for item in news_items:
                if not isinstance(item, NewsItem):
                    continue
                    
                # Handle potential missing or invalid values
                sentiment = item.sentiment_score
                if not isinstance(sentiment, (int, float)) or np.isnan(sentiment):
                    continue
                    
                relevance = item.relevance_score
                if not isinstance(relevance, (int, float)) or np.isnan(relevance):
                    relevance = 1.0
                    
                time_weight = self._calculate_time_weight(item.time_published)
                weight = max(relevance * time_weight, 0.0001)  # Ensure non-zero weight
                
                weights.append(weight)
                sentiments.append(sentiment)
                
            if not weights or not sentiments:
                return 0.0
                
            # Use np.average with explicit handling of edge cases
            weighted_sum = np.sum(np.array(weights) * np.array(sentiments))
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                return float(weighted_sum / total_weight)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted sentiment: {str(e)}")
            return 0.0
        
    def _calculate_time_weight(self, published_time: datetime) -> float:
        """Calculate time-based weight with exponential decay and timezone handling"""
        try:
            # Ensure both times are timezone aware
            now = datetime.now(timezone.utc)
            if published_time.tzinfo is None:
                published_time = published_time.replace(tzinfo=timezone.utc)
                
            age = now - published_time
            hours_old = max(age.total_seconds() / 3600, 0)  # Ensure non-negative
            return np.exp(-hours_old / 24)  # 24-hour half-life
            
        except Exception as e:
            self.logger.error(f"Error calculating time weight: {str(e)}")
            return 0.0
        
    def _analyze_technical_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze sentiment from technical indicators with improved error handling"""
        try:
            if not symbol:
                self.logger.warning("Symbol cannot be None or empty")
                return {}

            # Define the date range with timezone-aware datetimes
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=90)  # Last 3 months

            # Get market data from data manager instead of yfinance
            data = self.data_manager.get_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                period="3mo"
            )
            
            if data.empty:
                self.logger.warning(f"No market data available for {symbol}")
                return {}
            
            # Convert column names to lowercase and ensure float type
            data.columns = data.columns.str.lower()
            
            # Handle zero or negative values to prevent division by zero
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].replace({0: np.nan})
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                if data[col].isna().any():
                    data[col] = data[col].fillna(1.0)  # Last resort default

            # Calculate technical features using TechnicalAnalyzer
            features = self.technical_analyzer.calculate_features(
                data=data, 
                start_date=start_date,
                end_date=end_date
            )

            # Get trading signals
            signals = self.technical_analyzer.get_trading_signals(features)

            # Ensure all values are float and handle NaN/inf values
            sanitized_signals = {}
            for k, v in signals.items():
                if isinstance(v, (int, float)):
                    # Replace inf/-inf with max/min valid values
                    if np.isinf(v):
                        v = np.sign(v) * 1.0
                    # Replace NaN with neutral value
                    if np.isnan(v):
                        v = 0.0
                    sanitized_signals[k] = float(v)
                else:
                    sanitized_signals[k] = 0.0

            return sanitized_signals

        except Exception as e:
            self.logger.error(f"Error in technical sentiment: {str(e)}")
            return {}

    def _get_fundamental_data(self, symbol: str) -> Dict:
        """Get fundamental data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'pe_ratio': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'profit_margins': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'beta': info.get('beta', 1)
            }
        except Exception as e:
            self.logger.error(f"Error getting fundamental data: {str(e)}")
            return {}

    def _calculate_earnings_sentiment(self, fundamentals: Dict) -> float:
        """Calculate sentiment based on earnings metrics"""
        try:
            profit_margin_score = np.clip(fundamentals.get('profit_margins', 0) * 5, -1, 1)
            roe_score = np.clip(fundamentals.get('return_on_equity', 0) * 2, -1, 1)
            
            weights = [0.6, 0.4]  # Giving more weight to profit margins
            scores = [profit_margin_score, roe_score]
            
            return np.average(scores, weights=weights)
        except Exception:
            return 0.0

    def _calculate_valuation_sentiment(self, fundamentals: Dict) -> float:
        """Calculate sentiment based on valuation metrics"""
        try:
            pe_ratio = fundamentals.get('pe_ratio', 0)
            peg_ratio = fundamentals.get('peg_ratio', 0)
            
            # Convert PE and PEG ratios to sentiment scores (-1 to 1)
            pe_score = 1 - (np.clip(pe_ratio, 0, 50) / 25)  # PE of 25 gives score of 0
            peg_score = 1 - (np.clip(peg_ratio, 0, 3) / 1.5)  # PEG of 1.5 gives score of 0
            
            weights = [0.5, 0.5]
            scores = [pe_score, peg_score]
            
            return np.average(scores, weights=weights)
        except Exception:
            return 0.0

    def _calculate_growth_sentiment(self, fundamentals: Dict) -> float:
        """Calculate sentiment based on growth metrics"""
        try:
            revenue_growth = fundamentals.get('revenue_growth', 0)
            growth_score = np.clip(revenue_growth * 2, -1, 1)  # 50% growth = max score
            
            beta = fundamentals.get('beta', 1)
            beta_score = 1 - abs(beta - 1)  # Score is higher when beta is closer to 1
            
            weights = [0.7, 0.3]  # More weight on growth than beta
            scores = [growth_score, beta_score]
            
            return np.average(scores, weights=weights)
        except Exception:
            return 0.0

    def _analyze_fundamental_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze sentiment from fundamental data"""
        try:
            fundamentals = self._get_fundamental_data(symbol)
            
            if not fundamentals:
                return {
                    'earnings_sentiment': 0.0,
                    'valuation_sentiment': 0.0,
                    'growth_sentiment': 0.0
                }
            
            sentiment = {
                'earnings_sentiment': self._calculate_earnings_sentiment(fundamentals),
                'valuation_sentiment': self._calculate_valuation_sentiment(fundamentals),
                'growth_sentiment': self._calculate_growth_sentiment(fundamentals)
            }
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error in fundamental sentiment: {str(e)}")
            return {
                'earnings_sentiment': 0.0,
                'valuation_sentiment': 0.0,
                'growth_sentiment': 0.0
            }
            
    def _calculate_composite_score(self, news_sentiment: Dict[str, SentimentScore],
                                technical_sentiment: Dict[str, float],
                                fundamental_sentiment: Dict[str, float]) -> float:
        """Calculate weighted composite sentiment score with improved error handling"""
        try:
            weights = {
                'news': 0.4,
                'technical': 0.3,
                'fundamental': 0.3
            }
            
            # Handle news sentiment safely
            news_scores = []
            if news_sentiment:
                news_scores = [
                    s.value * s.confidence 
                    for s in news_sentiment.values() 
                    if isinstance(s, SentimentScore) and 
                    isinstance(s.value, (int, float)) and 
                    isinstance(s.confidence, (int, float)) and 
                    not np.isnan(s.value) and 
                    not np.isnan(s.confidence)
                ]
            news_score = np.mean(news_scores) if news_scores else 0.0
            
            # Handle technical sentiment
            tech_scores = []
            if technical_sentiment:
                tech_scores = [
                    v for v in technical_sentiment.values() 
                    if isinstance(v, (int, float)) and not np.isnan(v)
                ]
            tech_score = np.mean(tech_scores) if tech_scores else 0.0
            
            # Handle fundamental sentiment
            fund_scores = []
            if fundamental_sentiment:
                fund_scores = [
                    v for v in fundamental_sentiment.values() 
                    if isinstance(v, (int, float)) and not np.isnan(v)
                ]
            fund_score = np.mean(fund_scores) if fund_scores else 0.0
            
            # Combine scores with weights
            scores = {
                'news': news_score,
                'technical': tech_score,
                'fundamental': fund_score
            }
            
            # Calculate weighted sum with validation
            composite_score = sum(
                scores[k] * weights[k] 
                for k in weights.keys() 
                if isinstance(scores[k], (int, float)) and not np.isnan(scores[k])
            )
            
            return float(np.clip(composite_score, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {str(e)}")
            return 0.0

    # Add an alias method for compatibility
    def get_sentiment(self, symbol: str) -> Dict:
        """Synchronous alias for analyze_sentiment"""
        return self.analyze_sentiment(symbol)