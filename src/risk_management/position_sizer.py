# src/risk_management/position_sizer.py
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

@dataclass
class PositionSizeParams:
    kelly_fraction: float
    max_position_size: float
    volatility_adjustment: float
    account_risk_fraction: float

class PositionSizer:
    """
    Advanced position sizing with multiple methodologies
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.params = PositionSizeParams(**config['position_size'])
        
    def calculate_position_size(self,
                              price: float,
                              volatility: float,
                              account_value: float,
                              win_rate: Optional[float] = None,
                              profit_factor: Optional[float] = None) -> float:
        """
        Calculate optimal position size using multiple methods
        """
        try:
            # Kelly Criterion size
            kelly_size = self._kelly_position_size(
                win_rate or 0.5,
                profit_factor or 1.5,
                account_value
            )
            
            # Risk-based size
            risk_size = self._risk_based_size(
                price,
                volatility,
                account_value
            )
            
            # Volatility-adjusted size
            vol_adjusted = self._volatility_adjusted_size(
                min(kelly_size, risk_size),
                volatility
            )
            
            # Apply global constraints
            final_size = self._apply_constraints(
                vol_adjusted,
                price,
                account_value
            )
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0
            
    def _kelly_position_size(self, win_rate: float,
                           profit_factor: float,
                           account_value: float) -> float:
        """Calculate position size using Kelly Criterion"""
        # Kelly formula: f = (p(b+1) - 1)/b
        # where p is win rate, b is profit/loss ratio
        kelly = (win_rate * (profit_factor + 1) - 1) / profit_factor
        
        # Apply kelly fraction for safety
        kelly *= self.params.kelly_fraction
        
        return kelly * account_value
        
    def _risk_based_size(self, price: float,
                        volatility: float,
                        account_value: float) -> float:
        """Calculate position size based on risk limits"""
        # Risk per trade
        risk_amount = account_value * self.params.account_risk_fraction
        
        # Use volatility to estimate potential loss
        max_loss = volatility * price
        
        if max_loss > 0:
            return risk_amount / max_loss
        return 0
        
    def _volatility_adjusted_size(self, base_size: float,
                                volatility: float) -> float:
        """Adjust position size based on volatility"""
        vol_factor = 1 / (1 + volatility * self.params.volatility_adjustment)
        return base_size * vol_factor
        
    def _apply_constraints(self, size: float,
                         price: float,
                         account_value: float) -> float:
        """Apply position size constraints"""
        # Maximum position size constraint
        max_size = account_value * self.params.max_position_size / price
        
        # Round to appropriate precision
        size = min(size, max_size)
        
        # Round to nearest lot size
        lot_size = self.config.get('lot_size', 1)
        size = round(size / lot_size) * lot_size
        
        return size