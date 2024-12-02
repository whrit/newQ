# tests/validate_system.py
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict

from config.trading_config import ConfigManager
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.agents.deep_q_agent import DeepQAgent
from src.features.technical_indicators import TechnicalAnalyzer
from src.features.market_sentiment import MarketSentimentAnalyzer
from src.risk_management.portfolio_manager import PortfolioManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemValidator:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Test with major stocks
        
    async def run_all_checks(self):
        """Run all system validation checks"""
        try:
            logger.info("Starting system validation...")
            
            # Check configuration
            await self.validate_configuration()
            
            # Test API connections
            await self.test_api_connections()
            
            # Validate data pipeline
            await self.validate_data_pipeline()
            
            # Test trading components
            await self.test_trading_components()
            
            # Run simulation
            await self.run_test_simulation()
            
            logger.info("All system checks completed successfully!")
            
        except Exception as e:
            logger.error(f"System validation failed: {str(e)}")
            raise
            
    async def validate_configuration(self):
        """Validate configuration settings"""
        logger.info("Validating configuration...")
        
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY',
            'ALPHA_VANTAGE_API_KEY',
            'TRADING_MODE'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not self.config_manager.get(var):
                missing_vars.append(var)
                
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
        logger.info("Configuration validation passed")
        
    async def test_api_connections(self):
        """Test API connections"""
        logger.info("Testing API connections...")
        
        # Test Alpha Vantage
        try:
            sentiment_analyzer = SentimentAnalyzer()
            sentiment = await sentiment_analyzer.analyze_market_sentiment("AAPL")
            logger.info(f"Alpha Vantage API test successful: {sentiment['combined']}")
        except Exception as e:
            logger.error(f"Alpha Vantage API test failed: {str(e)}")
            raise
            
        # Test Alpaca
        try:
            portfolio_manager = PortfolioManager(self.config_manager.get_api_config())
            metrics = await portfolio_manager.update_portfolio()
            logger.info(f"Alpaca API test successful: Portfolio Value {metrics.total_value}")
        except Exception as e:
            logger.error(f"Alpaca API test failed: {str(e)}")
            raise
            
    async def validate_data_pipeline(self):
        """Validate data pipeline components"""
        logger.info("Validating data pipeline...")
        
        for symbol in self.test_symbols:
            try:
                # Test technical analysis
                tech_analyzer = TechnicalAnalyzer()
                data = self._get_test_data(symbol)
                features = tech_analyzer.calculate_features(data)
                logger.info(f"Technical analysis for {symbol} successful")
                
                # Test sentiment analysis
                sentiment_analyzer = MarketSentimentAnalyzer(
                    self.config_manager.get_api_config()
                )
                sentiment = await sentiment_analyzer.analyze_sentiment(symbol)
                logger.info(f"Sentiment analysis for {symbol} successful")
                
            except Exception as e:
                logger.error(f"Data pipeline validation failed for {symbol}: {str(e)}")
                raise
                
    async def test_trading_components(self):
        """Test trading components"""
        logger.info("Testing trading components...")
        
        try:
            # Initialize agent
            state_dim = 10  # Example state dimension
            action_dim = 3  # Example action dimension (buy, hold, sell)
            agent = DeepQAgent(state_dim=state_dim, action_dim=action_dim)
            
            # Test action selection
            test_state = np.random.random(state_dim)
            action = agent.select_action(test_state)
            logger.info(f"Agent action selection test passed: {action}")
            
            # Test training step
            for _ in range(5):  # Add some test experiences
                agent.update_memory(
                    test_state,
                    action,
                    reward=0.1,
                    next_state=np.random.random(state_dim),
                    done=False
                )
            
            metrics = agent.train_step()
            logger.info(f"Agent training test passed: {metrics}")
            
        except Exception as e:
            logger.error(f"Trading components test failed: {str(e)}")
            raise
            
    async def run_test_simulation(self):
        """Run a short test simulation"""
        logger.info("Running test simulation...")
        
        try:
            # Initialize components
            symbol = self.test_symbols[0]
            sentiment_analyzer = MarketSentimentAnalyzer(
                self.config_manager.get_api_config()
            )
            portfolio_manager = PortfolioManager(
                self.config_manager.get_api_config()
            )
            
            # Run simulation for a few steps
            for i in range(3):
                # Get market state
                sentiment = await sentiment_analyzer.analyze_sentiment(symbol)
                portfolio = await portfolio_manager.update_portfolio()
                
                logger.info(f"Simulation step {i+1}:")
                logger.info(f"Sentiment: {sentiment.composite_score:.2f}")
                logger.info(f"Portfolio value: {portfolio.total_value:.2f}")
                
                await asyncio.sleep(1)  # Avoid rate limits
                
            logger.info("Test simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Test simulation failed: {str(e)}")
            raise
            
    def _get_test_data(self, symbol: str) -> pd.DataFrame:
        """Get test data for a symbol"""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1mo")
        return data

def main():
    """Run system validation"""
    validator = SystemValidator()
    asyncio.run(validator.run_all_checks())

if __name__ == "__main__":
    main()