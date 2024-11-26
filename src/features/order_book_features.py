# src/features/order_book_features.py
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd

@dataclass
class OrderBookFeatures:
    spread: float
    bid_ask_imbalance: float
    depth_imbalance: float
    price_impact: Dict[str, float]
    volume_imbalance: float
    order_flow: float
    liquidity_score: float
    volatility_indicators: Dict[str, float]

class OrderBookAnalyzer:
    """
    Advanced order book analysis with multiple feature calculations
    """
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    def calculate_features(self, 
                         bids: List[Tuple[float, float]], 
                         asks: List[Tuple[float, float]],
                         recent_trades: List[Dict] = None) -> OrderBookFeatures:
        """
        Calculate comprehensive order book features
        Args:
            bids: List of (price, size) tuples for bids
            asks: List of (price, size) tuples for asks
            recent_trades: List of recent trades with price and size
        Returns:
            OrderBookFeatures object
        """
        # Basic calculations
        spread = self._calculate_spread(bids[0][0], asks[0][0])
        bid_ask_imbalance = self._calculate_bid_ask_imbalance(bids, asks)
        depth_imbalance = self._calculate_depth_imbalance(bids, asks)
        
        # Advanced calculations
        price_impact = self._calculate_price_impact(bids, asks)
        volume_imbalance = self._calculate_volume_imbalance(bids, asks)
        order_flow = self._calculate_order_flow(recent_trades) if recent_trades else 0
        liquidity_score = self._calculate_liquidity_score(bids, asks)
        volatility = self._calculate_volatility_indicators(bids, asks)
        
        return OrderBookFeatures(
            spread=spread,
            bid_ask_imbalance=bid_ask_imbalance,
            depth_imbalance=depth_imbalance,
            price_impact=price_impact,
            volume_imbalance=volume_imbalance,
            order_flow=order_flow,
            liquidity_score=liquidity_score,
            volatility_indicators=volatility
        )
        
    def _calculate_spread(self, best_bid: float, best_ask: float) -> float:
        """Calculate bid-ask spread"""
        return (best_ask - best_bid) / best_bid
        
    def _calculate_bid_ask_imbalance(self, bids: List[Tuple[float, float]], 
                                   asks: List[Tuple[float, float]]) -> float:
        """Calculate bid-ask imbalance at each level"""
        bid_volume = sum(size for _, size in bids[:self.depth_levels])
        ask_volume = sum(size for _, size in asks[:self.depth_levels])
        
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
    def _calculate_depth_imbalance(self, bids: List[Tuple[float, float]], 
                                 asks: List[Tuple[float, float]]) -> float:
        """Calculate depth imbalance with decay factor"""
        depth_imbalance = 0
        decay_factor = 0.95
        
        for level in range(min(self.depth_levels, len(bids), len(asks))):
            level_weight = decay_factor ** level
            bid_size = bids[level][1]
            ask_size = asks[level][1]
            
            level_imbalance = (bid_size - ask_size) / (bid_size + ask_size)
            depth_imbalance += level_weight * level_imbalance
            
        return depth_imbalance
        
    def _calculate_price_impact(self, bids: List[Tuple[float, float]], 
                              asks: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate price impact for different order sizes"""
        sizes = [100, 1000, 10000]  # Example order sizes
        impacts = {}
        
        for size in sizes:
            # Calculate impact for buying
            buy_impact = self._calculate_side_impact(asks, size)
            # Calculate impact for selling
            sell_impact = self._calculate_side_impact(bids, size)
            
            impacts[f'buy_impact_{size}'] = buy_impact
            impacts[f'sell_impact_{size}'] = sell_impact
            
        return impacts
        
    def _calculate_side_impact(self, orders: List[Tuple[float, float]], 
                             target_size: float) -> float:
        """Calculate price impact for one side"""
        remaining_size = target_size
        weighted_price = 0
        total_filled = 0
        
        for price, size in orders:
            filled = min(remaining_size, size)
            weighted_price += price * filled
            total_filled += filled
            remaining_size -= filled
            
            if remaining_size <= 0:
                break
                
        return (weighted_price / total_filled) if total_filled > 0 else 0
        
    def _calculate_volume_imbalance(self, bids: List[Tuple[float, float]], 
                                  asks: List[Tuple[float, float]]) -> float:
        """Calculate volume imbalance with price sensitivity"""
        bid_volume = 0
        ask_volume = 0
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        for price, size in bids[:self.depth_levels]:
            price_distance = abs(price - mid_price) / mid_price
            bid_volume += size * np.exp(-price_distance)
            
        for price, size in asks[:self.depth_levels]:
            price_distance = abs(price - mid_price) / mid_price
            ask_volume += size * np.exp(-price_distance)
            
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
    def _calculate_order_flow(self, recent_trades: List[Dict]) -> float:
        """Calculate order flow imbalance"""
        buy_volume = sum(trade['size'] for trade in recent_trades 
                        if trade.get('side') == 'buy')
        sell_volume = sum(trade['size'] for trade in recent_trades 
                         if trade.get('side') == 'sell')
                         
        total_volume = buy_volume + sell_volume
        return (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
    def _calculate_liquidity_score(self, bids: List[Tuple[float, float]], 
                                 asks: List[Tuple[float, float]]) -> float:
        """Calculate overall liquidity score"""
        spread = self._calculate_spread(bids[0][0], asks[0][0])
        depth = min(sum(size for _, size in bids[:self.depth_levels]),
                   sum(size for _, size in asks[:self.depth_levels]))
        
        # Normalize components
        spread_score = 1 / (1 + spread)
        depth_score = np.log1p(depth) / 10  # Normalize large depth values
        
        return 0.5 * spread_score + 0.5 * depth_score
        
    def _calculate_volatility_indicators(self, bids: List[Tuple[float, float]], 
                                      asks: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate volatility indicators from order book"""
        mid_prices = [(bid[0] + ask[0]) / 2 
                     for bid, ask in zip(bids[:self.depth_levels], 
                                       asks[:self.depth_levels])]
                                       
        price_diffs = np.diff(mid_prices)
        
        return {
            'price_variance': np.var(mid_prices),
            'price_range': max(mid_prices) - min(mid_prices),
            'mean_price_impact': np.mean(np.abs(price_diffs))
        }