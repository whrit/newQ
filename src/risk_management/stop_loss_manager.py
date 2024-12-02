# src/risk_management/stop_loss_manager.py
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

@dataclass
class StopLevels:
    stop_loss: float
    trailing_stop: Optional[float]
    take_profit: Optional[float]
    time_stop: Optional[float]
    smart_stop: Optional[float]

class StopLossManager:
    """
    Advanced stop loss management with multiple stop types
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_stop_levels(self,
                            entry_price: float,
                            position_size: float,
                            volatility: float,
                            atr: float,
                            support_resistance: Optional[Dict] = None) -> StopLevels:
        """
        Calculate comprehensive stop levels
        """
        try:
            # Fixed stop loss
            fixed_stop = self._calculate_fixed_stop(
                entry_price,
                position_size,
                volatility
            )
            
            # Trailing stop
            trailing_stop = self._calculate_trailing_stop(
                entry_price,
                atr
            )
            
            # Take profit
            take_profit = self._calculate_take_profit(
                entry_price,
                fixed_stop,
                support_resistance
            )
            
            # Time-based stop
            time_stop = self._calculate_time_stop(
                entry_price,
                volatility
            )
            
            # Smart stop using multiple factors
            smart_stop = self._calculate_smart_stop(
                entry_price,
                volatility,
                atr,
                support_resistance
            )
            
            return StopLevels(
                stop_loss=fixed_stop,
                trailing_stop=trailing_stop,
                take_profit=take_profit,
                time_stop=time_stop,
                smart_stop=smart_stop
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating stop levels: {str(e)}")
            raise
            
    def _calculate_fixed_stop(self, entry_price: float,
                            position_size: float,
                            volatility: float) -> float:
        """Calculate fixed stop loss level"""
        # Risk-based stop
        risk_amount = self.config['max_risk_per_trade']
        risk_distance = risk_amount / position_size
        
        # Volatility-based stop
        vol_distance = volatility * self.config['volatility_multiplier']
        
        # Use the larger of the two
        stop_distance = max(risk_distance, vol_distance)
        
        return entry_price * (1 - stop_distance)
        
    def _calculate_trailing_stop(self, entry_price: float,
                               atr: float) -> float:
        """Calculate trailing stop level based on ATR"""
        # Use ATR for dynamic stop distance
        atr_multiplier = self.config['atr_multiplier']
        stop_distance = atr * atr_multiplier
        
        return entry_price * (1 - stop_distance)
        
    def _calculate_take_profit(self, entry_price: float,
                             stop_loss: float,
                             support_resistance: Optional[Dict] = None) -> float:
        """Calculate take profit level"""
        # Basic risk/reward ratio
        risk = entry_price - stop_loss
        reward = risk * self.config['risk_reward_ratio']
        
        # Adjust based on support/resistance if available
        if support_resistance:
            next_resistance = support_resistance.get('next_resistance')
            if next_resistance:
                reward = min(reward, next_resistance - entry_price)
                
        return entry_price + reward
        
    def _calculate_time_stop(self, entry_price: float,
                           volatility: float) -> float:
        """Calculate time-based stop level"""
        # More aggressive stop based on holding period
        time_factor = self.config['time_decay_factor']
        time_distance = volatility * time_factor
        
        return entry_price * (1 - time_distance)
        
    def _calculate_smart_stop(self, entry_price: float,
                            volatility: float,
                            atr: float,
                            support_resistance: Optional[Dict] = None) -> float:
        """
        Calculate smart stop using multiple factors
        """
        factors = []
        
        # Volatility component
        vol_stop = entry_price * (1 - volatility * self.config['smart_vol_multiplier'])
        factors.append(vol_stop)
        
        # ATR component
        atr_stop = entry_price * (1 - atr * self.config['smart_atr_multiplier'])
        factors.append(atr_stop)
        
        # Support level component if available
        if support_resistance and 'support_levels' in support_resistance:
            closest_support = self._find_closest_support(
                entry_price,
                support_resistance['support_levels']
            )
            if closest_support:
                factors.append(closest_support)
                
        # Weight and combine factors
        weights = self._calculate_stop_weights(volatility, atr)
        smart_stop = sum(f * w for f, w in zip(factors, weights))
        
        return smart_stop
        
    def adjust_stops(self, current_price: float,
                    stop_levels: StopLevels,
                    position_data: Dict) -> StopLevels:
        """
        Adjust stop levels based on price movement
        """
        try:
            # Update trailing stop
            new_trailing_stop = self._update_trailing_stop(
                current_price,
                stop_levels.trailing_stop,
                position_data
            )
            
            # Update smart stop
            new_smart_stop = self._update_smart_stop(
                current_price,
                stop_levels.smart_stop,
                position_data
            )
            
            # Update time stop
            new_time_stop = self._update_time_stop(
                current_price,
                stop_levels.time_stop,
                position_data
            )
            
            return StopLevels(
                stop_loss=stop_levels.stop_loss,  # Fixed stop doesn't change
                trailing_stop=new_trailing_stop,
                take_profit=stop_levels.take_profit,  # Take profit doesn't change
                time_stop=new_time_stop,
                smart_stop=new_smart_stop
            )
            
        except Exception as e:
            self.logger.error(f"Error adjusting stops: {str(e)}")
            raise
            
    def _update_trailing_stop(self, current_price: float,
                            trailing_stop: float,
                            position_data: Dict) -> float:
        """Update trailing stop level"""
        if not trailing_stop:
            return None
            
        profit = current_price - position_data['entry_price']
        if profit > 0:
            # Calculate new stop based on profit
            trail_distance = profit * self.config['trail_percentage']
            new_stop = current_price - trail_distance
            # Only move stop up, never down
            return max(trailing_stop, new_stop)
            
        return trailing_stop
        
    def _update_smart_stop(self, current_price: float,
                          smart_stop: float,
                          position_data: Dict) -> float:
        """Update smart stop level"""
        if not smart_stop:
            return None
            
        # Calculate new factors
        volatility = position_data.get('current_volatility', 0)
        profit_factor = self._calculate_profit_factor(
            current_price,
            position_data['entry_price']
        )
        
        # Adjust stop based on profit and volatility
        if profit_factor > 1:
            # In profit - tighten stop
            adjustment = volatility * self.config['smart_stop_tighten_factor']
            new_stop = current_price * (1 - adjustment)
            return max(smart_stop, new_stop)
            
        return smart_stop
        
    def _update_time_stop(self, current_price: float,
                         time_stop: float,
                         position_data: Dict) -> float:
        """Update time-based stop level"""
        if not time_stop:
            return None
            
        # Make stop more aggressive over time
        time_held = position_data.get('time_held', 0)
        time_decay = np.exp(-time_held * self.config['time_decay_rate'])
        
        new_stop = current_price * (1 - self.config['base_time_stop'] * time_decay)
        return max(time_stop, new_stop)
        
    def _find_closest_support(self, price: float,
                            support_levels: List[float]) -> Optional[float]:
        """Find closest support level below price"""
        valid_supports = [s for s in support_levels if s < price]
        if valid_supports:
            return max(valid_supports)
        return None
        
    def _calculate_stop_weights(self, volatility: float,
                              atr: float) -> List[float]:
        """Calculate weights for different stop factors"""
        # Base weights
        weights = [
            self.config['volatility_weight'],
            self.config['atr_weight'],
            self.config['support_weight']
        ]
        
        # Adjust based on market conditions
        if volatility > self.config['high_volatility_threshold']:
            # Increase weight of volatility-based stop in high volatility
            weights[0] *= 1.5
            weights[1] *= 0.75
            
        # Normalize weights
        total = sum(weights)
        return [w/total for w in weights]
        
    def _calculate_profit_factor(self, current_price: float,
                               entry_price: float) -> float:
        """Calculate profit factor for position"""
        return (current_price - entry_price) / entry_price
        
    def check_stop_triggers(self, current_price: float,
                          stop_levels: StopLevels,
                          position_data: Dict) -> Tuple[bool, str]:
        """
        Check if any stop loss conditions are triggered
        Returns:
            Tuple of (triggered: bool, reason: str)
        """
        triggers = []
        
        # Check fixed stop
        if current_price <= stop_levels.stop_loss:
            triggers.append(('Fixed stop loss', stop_levels.stop_loss))
            
        # Check trailing stop
        if stop_levels.trailing_stop and current_price <= stop_levels.trailing_stop:
            triggers.append(('Trailing stop', stop_levels.trailing_stop))
            
        # Check take profit
        if stop_levels.take_profit and current_price >= stop_levels.take_profit:
            triggers.append(('Take profit', stop_levels.take_profit))
            
        # Check time stop
        if stop_levels.time_stop and position_data.get('time_held', 0) >= stop_levels.time_stop:
            triggers.append(('Time stop', position_data.get('time_held')))
            
        # Check smart stop
        if stop_levels.smart_stop and current_price <= stop_levels.smart_stop:
            triggers.append(('Smart stop', stop_levels.smart_stop))
            
        if triggers:
            # Return the most conservative stop (highest for long positions)
            best_trigger = max(triggers, key=lambda x: x[1])
            return True, best_trigger[0]
            
        return False, ''
        
    def get_exit_parameters(self, trigger_type: str,
                          position_data: Dict) -> Dict:
        """
        Get exit parameters based on stop type
        """
        params = {
            'order_type': 'market',  # Default to market order
            'time_in_force': 'immediate_or_cancel',
            'reduce_only': True
        }
        
        if trigger_type == 'Take profit':
            params.update({
                'order_type': 'limit',
                'time_in_force': 'good_till_cancelled',
                'limit_price': position_data['take_profit']
            })
            
        elif trigger_type == 'Trailing stop':
            params.update({
                'order_type': 'stop_market',
                'stop_price': position_data['trailing_stop']
            })
            
        return params