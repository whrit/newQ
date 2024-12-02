# src/models/sentiment_analyzer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union, Optional
import numpy as np
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
import logging
import requests
from dataclasses import dataclass
from src.features.technical_indicators import TechnicalAnalyzer
import yfinance as yf
from config.trading_config import ConfigManager

@dataclass
class NewsArticle:
    title: str
    url: str
    time_published: datetime
    summary: str
    source: str
    sentiment: float
    relevance_score: float
    topics: List[str]

class AlphaVantageClient:
    """Client for Alpha Vantage News Sentiment API"""
    def __init__(self, api_key: str, base_url: str = "https://www.alphavantage.co/query"):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    async def get_news_sentiment(self, 
                               symbol: str, 
                               topics: Optional[List[str]] = None,
                               time_from: Optional[str] = None,
                               time_to: Optional[str] = None,
                               limit: int = 50) -> List[NewsArticle]:
        """Fetch news sentiment data from Alpha Vantage"""
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.api_key,
                "limit": limit,
                "sort": "RELEVANCE"
            }
            
            if topics:
                params["topics"] = ",".join(topics)
            if time_from:
                params["time_from"] = time_from
            if time_to:
                params["time_to"] = time_to
                
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get("feed", []):
                article = NewsArticle(
                    title=item.get("title"),
                    url=item.get("url"),
                    time_published=datetime.strptime(item.get("time_published"), 
                                                   "%Y%m%dT%H%M%S"),
                    summary=item.get("summary"),
                    source=item.get("source"),
                    sentiment=self._process_sentiment_score(item.get("overall_sentiment_score")),
                    relevance_score=float(item.get("relevance_score", 0)),
                    topics=item.get("topics", [])
                )
                articles.append(article)
                
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news sentiment: {str(e)}")
            return []
            
    def _process_sentiment_score(self, score: Optional[str]) -> float:
        """Convert sentiment score to float and normalize to [-1, 1]"""
        try:
            if score is None:
                return 0.0
            return float(score)
        except ValueError:
            return 0.0

class SentimentAnalyzer:
    def __init__(self, custom_config: Dict = None):
        # Load configuration
        config_manager = ConfigManager()
        base_config = config_manager.get_api_config()
        
        # Merge with custom config if provided
        self.config = {**base_config, **(custom_config or {})}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Alpha Vantage client with API key from env
        self.alpha_vantage = AlphaVantageClient(
            self.config["alpha_vantage_api_key"]
        )
        
        # Rest of initialization remains the same
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.sentiment_cache = {}
        self.cache_expiry = timedelta(minutes=30)

        self.technical_analyzer = TechnicalAnalyzer(
        lookback_periods={
            'short': 14,
            'medium': 50,
            'long': 200
        }
    )
        
    async def _analyze_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze sentiment from news sources using Alpha Vantage"""
        try:
            topics = [
                "earnings", 
                "financial_markets",
                "economy_macro",
                "mergers_and_acquisitions"
            ]
            
            time_from = (datetime.now() - timedelta(days=1)).strftime("%Y%m%dT%H%M")
            
            articles = await self.alpha_vantage.get_news_sentiment(
                symbol=symbol,
                topics=topics,
                time_from=time_from,
                limit=100
            )
            
            if not articles:
                return {'news': 0.0}
                
            total_weight = 0
            weighted_sentiment = 0
            
            for article in articles:
                time_weight = self._calculate_time_weight(article.time_published)
                weight = article.relevance_score * time_weight
                
                weighted_sentiment += article.sentiment * weight
                total_weight += weight
                
            avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
            
            return {
                'news': avg_sentiment,
                'article_count': len(articles),
                'latest_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in news sentiment analysis: {str(e)}")
            return {'news': 0.0}
            
    def _calculate_time_weight(self, published_time: datetime) -> float:
        """Calculate time-based weight with exponential decay"""
        age = datetime.now() - published_time
        hours_old = age.total_seconds() / 3600
        return 0.5 ** (hours_old / 24)  # 24-hour half-life
        
    def analyze_text(self, texts: Union[str, List[str]]) -> Dict[str, float]:
        """Analyze sentiment using FinBERT model"""
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                  return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
            scores = []
            for probs in probabilities:
                score = probs[2] - probs[0]  # positive - negative
                scores.append(score.item())
                
            return {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'scores': [0.0] * len(texts), 'mean': 0.0, 'std': 0.0}

    async def analyze_market_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze market sentiment using news and technical data"""
        cache_key = f"{symbol}_sentiment"
        if cache_key in self.sentiment_cache:
            timestamp, sentiment = self.sentiment_cache[cache_key]
            if datetime.now() - timestamp < self.cache_expiry:
                return sentiment
                
        try:
            # Get news sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            # Get technical sentiment
            technical_sentiment = await self._analyze_technical_sentiment(symbol)
            
            # Combine sentiments with adjusted weights
            weights = {
                'news': 0.7,      # Increased weight for news
                'technical': 0.3   # Reduced weight for technical
            }
            
            sentiments = {
                'news': news_sentiment.get('news', 0),
                'technical': technical_sentiment.get('technical', 0)
            }
            
            combined_sentiment = sum(sentiments[source] * weights[source] 
                                   for source in weights.keys())
            
            result = {
                'combined': combined_sentiment,
                'news_sentiment': news_sentiment,
                'technical_sentiment': technical_sentiment,
                'timestamp': datetime.now()
            }
            
            self.sentiment_cache[cache_key] = (datetime.now(), result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in market sentiment analysis: {str(e)}")
            return {
                'combined': 0.0,
                'news_sentiment': {'news': 0.0},
                'technical_sentiment': {'technical': 0.0},
                'timestamp': datetime.now()
            }

    async def _analyze_technical_sentiment(self, symbol: str) -> Dict[str, float]:
        """Analyze sentiment from technical indicators using TechnicalAnalyzer"""
        try:
            # Get historical data using yfinance
            ticker = yf.Ticker(symbol)
            historical_data = ticker.history(period="1y")
            historical_data.columns = historical_data.columns.str.lower()
            
            # Calculate technical features
            features = self.technical_analyzer.calculate_features(historical_data)
            
            # Get trading signals
            signals = self.technical_analyzer.get_trading_signals(features)
            
            # Calculate sentiment scores
            trend_sentiment = signals.get('trend', 0)
            momentum_sentiment = signals.get('momentum', 0)
            volatility_sentiment = -1 * signals.get('volatility', 0)  # Inverse volatility
            volume_sentiment = signals.get('volume', 0)
            
            # Weight the components
            weights = {
                'trend': 0.35,
                'momentum': 0.25,
                'volatility': 0.20,
                'volume': 0.20
            }
            
            sentiment_components = {
                'trend': trend_sentiment,
                'momentum': momentum_sentiment,
                'volatility': volatility_sentiment,
                'volume': volume_sentiment
            }
            
            # Calculate final technical sentiment
            technical_sentiment = sum(
                sentiment * weights[component]
                for component, sentiment in sentiment_components.items()
            )
            
            # Normalize to [-1, 1] range
            technical_sentiment = np.clip(technical_sentiment, -1, 1)
            
            return {
                'technical': technical_sentiment,
                'components': sentiment_components,
                'signals': signals,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in technical sentiment analysis: {str(e)}")
            return {'technical': 0.0}