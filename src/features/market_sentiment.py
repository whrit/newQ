# src/features/market_sentiment.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import aiohttp

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
    Market sentiment analysis using news, technical, and fundamental data
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Initialize Alpha Vantage client
        self.alpha_vantage = AlphaVantageNewsClient(config["alpha_vantage_api_key"])
        
    async def analyze_sentiment(self, symbol: str) -> MarketSentiment:
        """
        Analyze market sentiment from multiple sources
        """
        if self._check_cache(symbol):
            return self.cache[symbol]['data']
            
        try:
            # Gather sentiment from different sources concurrently
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                technical_future = executor.submit(self._analyze_technical_sentiment, symbol)
                fundamental_future = executor.submit(self._analyze_fundamental_sentiment, symbol)
                
                technical_sentiment = technical_future.result()
                fundamental_sentiment = fundamental_future.result()
            
            # Calculate composite score with adjusted weights
            composite_score = self._calculate_composite_score(
                news_sentiment["sentiment_scores"],
                technical_sentiment,
                fundamental_sentiment
            )
            
            sentiment = MarketSentiment(
                composite_score=composite_score,
                news_sentiment=news_sentiment["sentiment_scores"],
                technical_sentiment=technical_sentiment,
                fundamental_sentiment=fundamental_sentiment,
                recent_news=news_sentiment["news_items"]
            )
            
            self._update_cache(symbol, sentiment)
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            raise

    async def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """Enhanced news sentiment analysis using Alpha Vantage"""
        try:
            time_from = (datetime.now() - timedelta(days=1)).strftime("%Y%m%dT%H%M")
            
            topics = [
                "earnings",
                "financial_markets",
                "economy_macro",
                "mergers_and_acquisitions",
                "technology"
            ]
            
            news_items = await self.alpha_vantage.get_market_news(
                symbol=symbol,
                topics=topics,
                time_from=time_from
            )
            
            if not news_items:
                return {
                    "sentiment_scores": {},
                    "news_items": []
                }
            
            # Calculate weighted sentiment scores
            source_sentiments = {}
            for source in set(item.source for item in news_items):
                source_items = [item for item in news_items if item.source == source]
                if source_items:
                    weighted_sentiment = self._calculate_weighted_sentiment(source_items)
                    source_sentiments[source] = SentimentScore(
                        value=weighted_sentiment,
                        confidence=np.mean([item.relevance_score for item in source_items]),
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
        """Calculate weighted sentiment score for news items"""
        if not news_items:
            return 0.0
            
        weights = []
        sentiments = []
        
        for item in news_items:
            time_weight = self._calculate_time_weight(item.time_published)
            weight = item.relevance_score * time_weight
            
            weights.append(weight)
            sentiments.append(item.sentiment_score)
            
        return np.average(sentiments, weights=weights) if weights else 0.0
        
    def _calculate_time_weight(self, published_time: datetime) -> float:
        """Calculate time-based weight with exponential decay"""
        age = datetime.now() - published_time
        hours_old = age.total_seconds() / 3600
        return np.exp(-hours_old / 24)  # 24-hour half-life
        
    def _analyze_technical_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze sentiment from technical indicators"""
        try:
            # Get technical indicators
            technical_indicators = self._get_technical_indicators(symbol)
            
            # Calculate sentiment scores
            sentiment = {
                'trend_strength': self._calculate_trend_strength(technical_indicators),
                'momentum': self._calculate_momentum_sentiment(technical_indicators),
                'volatility': self._calculate_volatility_sentiment(technical_indicators),
                'support_resistance': self._calculate_support_resistance_sentiment(technical_indicators)
            }
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error in technical sentiment: {str(e)}")
            return {}
            
    def _analyze_fundamental_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze sentiment from fundamental data"""
        try:
            fundamentals = self._get_fundamental_data(symbol)
            
            sentiment = {
                'earnings_sentiment': self._calculate_earnings_sentiment(fundamentals),
                'valuation_sentiment': self._calculate_valuation_sentiment(fundamentals),
                'growth_sentiment': self._calculate_growth_sentiment(fundamentals),
                'institutional_sentiment': self._calculate_institutional_sentiment(fundamentals)
            }
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error in fundamental sentiment: {str(e)}")
            return {}
            
    def _calculate_composite_score(self, news_sentiment: Dict[str, SentimentScore],
                                 technical_sentiment: Dict[str, float],
                                 fundamental_sentiment: Dict[str, float]) -> float:
        """Calculate weighted composite sentiment score"""
        # Adjusted weights to distribute social sentiment weight
        weights = {
            'news': 0.4,  # Increased from 0.3
            'technical': 0.3,  # Increased from 0.25
            'fundamental': 0.3  # Increased from 0.25
        }
        
        scores = {
            'news': np.mean([s.value * s.confidence for s in news_sentiment.values()]),
            'technical': np.mean(list(technical_sentiment.values())),
            'fundamental': np.mean(list(fundamental_sentiment.values()))
        }
        
        composite_score = sum(scores[k] * weights[k] for k in weights.keys())
        return np.clip(composite_score, -1, 1)