# src/risk_management/position_sizer.py
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

@dataclass
class PositionSizeParams:
    max_position_size: float  # Maximum allowed position size as fraction of portfolio
    risk_limit: float        # Maximum risk per trade
    min_size: float         # Minimum trade size
    max_leverage: float     # Maximum allowed leverage
    method: str            # Position sizing method (e.g., 'risk_based', 'kelly')

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
        Calculate optimal position size using selected method
        """
        try:
            if self.params.method == 'risk_based':
                size = self._risk_based_size(price, volatility, account_value)
            elif self.params.method == 'kelly':
                size = self._kelly_position_size(
                    win_rate or 0.5,
                    profit_factor or 1.5,
                    account_value
                )
            else:
                self.logger.warning(f"Unknown position sizing method {self.params.method}, defaulting to risk_based")
                size = self._risk_based_size(price, volatility, account_value)
            
            # Apply global constraints
            size = self._apply_constraints(size, price, account_value)
            
            return size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0
            
    def _kelly_position_size(self, win_rate: float,
                           profit_factor: float,
                           account_value: float) -> float:
        """Calculate position size using Kelly Criterion"""
        kelly = (win_rate * (profit_factor + 1) - 1) / profit_factor
        kelly = max(0, min(kelly, self.params.max_position_size))  # Apply maximum size constraint
        return kelly * account_value
        
    def _risk_based_size(self, price: float,
                        volatility: float,
                        account_value: float) -> float:
        """Calculate position size based on risk limits"""
        # Calculate maximum loss we're willing to take based on risk_limit
        max_risk = account_value * self.params.risk_limit
        
        # Use volatility to estimate potential loss per share
        potential_loss_per_share = volatility * price
        
        if potential_loss_per_share > 0:
            # Calculate size that would result in our maximum allowed risk
            size = max_risk / potential_loss_per_share
        else:
            size = 0
            
        return size
        
    def _apply_constraints(self, size: float,
                         price: float,
                         account_value: float) -> float:
        """Apply position size constraints"""
        # Check minimum size
        if size < self.params.min_size:
            return 0
            
        # Check maximum position size
        max_size = (account_value * self.params.max_position_size) / price
        size = min(size, max_size)
        
        # Check leverage constraint
        if price * size > account_value * self.params.max_leverage:
            size = (account_value * self.params.max_leverage) / price
            
        # Round to nearest lot size
        lot_size = self.config.get('lot_size', 1)
        size = round(size / lot_size) * lot_size
        
        return size