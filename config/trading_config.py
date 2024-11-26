# src/utils/config.py
import os
from dotenv import load_dotenv
import ast
from typing import Any, Dict

class ConfigManager:
    """Manage configuration settings from environment variables"""
    def __init__(self):
        load_dotenv()  # Load .env file
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with type conversion"""
        value = os.getenv(key, default)
        
        if value is None:
            return default
            
        try:
            # Try to evaluate as literal (for lists, dicts, etc)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If not a literal, return as string
            return value
            
    def get_deep_q_config(self) -> Dict:
        """Get Deep Q-Learning configuration"""
        return {
            'learning_rate': float(self.get('LEARNING_RATE', 0.001)),
            'gamma': float(self.get('GAMMA', 0.95)),
            'initial_epsilon': float(self.get('EPSILON_START', 1.0)),
            'final_epsilon': float(self.get('EPSILON_MIN', 0.01)),
            'epsilon_decay': float(self.get('EPSILON_DECAY', 0.995)),
            'batch_size': int(self.get('BATCH_SIZE', 64)),
            'buffer_size': int(self.get('MEMORY_SIZE', 10000)),
            'update_frequency': int(self.get('TARGET_UPDATE', 100)),
            'hidden_dims': self.get('HIDDEN_DIMS', [256, 128]),
            'device': "cuda" if torch.cuda.is_available() else "cpu"
        }
        
    def get_api_config(self) -> Dict:
        """Get API configuration"""
        return {
            'alpaca_api_key': self.get('ALPACA_API_KEY'),
            'alpaca_secret_key': self.get('ALPACA_SECRET_KEY'),
            'alpha_vantage_api_key': self.get('ALPHA_VANTAGE_API_KEY'),
            'trading_mode': self.get('TRADING_MODE', 'paper')
        }