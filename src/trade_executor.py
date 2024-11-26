# src/trade_executor.py
from typing import Dict, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import logging
from datetime import datetime
import asyncio

@dataclass
class ExecutionResult:
    order_id: str
    filled_qty: float
    filled_price: float
    commission: float
    slippage: float
    success: bool
    info: Dict

class TradeExecutor:
    """
    Advanced trade execution with smart order routing and execution optimization
    """
    def __init__(self, trading_client: TradingClient, config: Dict):
        self.trading_client = trading_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def execute_trade(self, 
                          symbol: str,
                          side: OrderSide,
                          quantity: float,
                          order_type: str = 'market',
                          price: Optional[float] = None,
                          time_in_force: TimeInForce = TimeInForce.DAY) -> ExecutionResult:
        """
        Execute trade with smart order routing
        """
        try:
            # Validate order parameters
            self._validate_order_params(symbol, quantity, price)
            
            # Split order if necessary
            if self._should_split_order(quantity):
                return await self._execute_split_order(
                    symbol, side, quantity, order_type, price, time_in_force
                )
                
            # Get execution strategy
            strategy = self._get_execution_strategy(
                symbol, side, quantity, order_type
            )
            
            # Execute order
            order_result = await self._execute_order(
                symbol, side, quantity, order_type, price, 
                time_in_force, strategy
            )
            
            # Monitor and adjust if necessary
            final_result = await self._monitor_and_adjust(order_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            raise
            
    def _validate_order_params(self, symbol: str, quantity: float, 
                             price: Optional[float]) -> None:
        """Validate order parameters"""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        if price is not None and price <= 0:
            raise ValueError("Price must be positive")
            
        # Check position limits
        current_position = self._get_current_position(symbol)
        if not self._check_position_limits(current_position, quantity):
            raise ValueError("Order exceeds position limits")
            
    def _should_split_order(self, quantity: float) -> bool:
        """Determine if order should be split"""
        return quantity > self.config['max_single_order_size']
        
    async def _execute_split_order(self, 
                                 symbol: str,
                                 side: OrderSide,
                                 quantity: float,
                                 order_type: str,
                                 price: Optional[float],
                                 time_in_force: TimeInForce) -> ExecutionResult:
        """Execute large order in smaller chunks"""
        chunk_size = self.config['max_single_order_size']
        num_chunks = int(np.ceil(quantity / chunk_size))
        
        results = []
        remaining_qty = quantity
        
        for i in range(num_chunks):
            chunk_qty = min(chunk_size, remaining_qty)
            
            # Execute chunk
            result = await self._execute_order(
                symbol, side, chunk_qty, order_type, 
                price, time_in_force, None
            )
            
            results.append(result)
            remaining_qty -= chunk_qty
            
            # Wait between chunks
            await self._smart_delay(i, results)
            
        # Combine results
        return self._combine_execution_results(results)
        
    def _get_execution_strategy(self, symbol: str, side: OrderSide,
                              quantity: float, order_type: str) -> Dict:
        """Get optimal execution strategy based on market conditions"""
        strategy = {
            'use_iceberg': quantity > self.config['iceberg_threshold'],
            'time_horizon': self._calculate_time_horizon(quantity),
            'price_limits': self._calculate_price_limits(symbol, side),
            'participation_rate': self._calculate_participation_rate(quantity),
            'use_smart_routing': True
        }
        
        return strategy
        
    async def _execute_order(self, 
                           symbol: str,
                           side: OrderSide,
                           quantity: float,
                           order_type: str,
                           price: Optional[float],
                           time_in_force: TimeInForce,
                           strategy: Optional[Dict]) -> ExecutionResult:
        """Execute single order with given strategy"""
        try:
            # Prepare order request based on type
            if order_type == 'market':
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=time_in_force
                )
            elif order_type == 'limit':
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=time_in_force,
                    limit_price=price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
                
            # Apply execution strategy
            if strategy:
                self._apply_execution_strategy(order_request, strategy)
                
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            # Wait for fill
            filled_order = await self._wait_for_fill(order.id)
            
            # Calculate execution metrics
            execution_metrics = self._calculate_execution_metrics(
                filled_order,
                price if price else filled_order.filled_price
            )
            
            return ExecutionResult(
                order_id=filled_order.id,
                filled_qty=float(filled_order.filled_qty),
                filled_price=float(filled_order.filled_price),
                commission=execution_metrics['commission'],
                slippage=execution_metrics['slippage'],
                success=True,
                info=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {str(e)}")
            return ExecutionResult(
                order_id="",
                filled_qty=0,
                filled_price=0,
                commission=0,
                slippage=0,
                success=False,
                info={'error': str(e)}
            )
            
    async def _wait_for_fill(self, order_id: str, timeout: float = 60.0) -> Dict:
        """Wait for order to be filled with timeout"""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            order = self.trading_client.get_order_by_id(order_id)
            if order.status in ['filled', 'cancelled', 'expired']:
                return order
            await asyncio.sleep(0.5)
            
        raise TimeoutError("Order fill timeout")
        
    def _calculate_execution_metrics(self, filled_order: Dict,
                                  reference_price: float) -> Dict:
        """Calculate execution metrics including slippage and costs"""
        # Calculate basic metrics
        filled_price = float(filled_order.filled_price)
        qty = float(filled_order.filled_qty)
        
        # Calculate slippage
        slippage = abs(filled_price - reference_price) / reference_price
        
        # Calculate commission
        commission = self._calculate_commission(filled_price, qty)
        
        # Calculate implementation shortfall
        shortfall = self._calculate_implementation_shortfall(
            filled_order,
            reference_price
        )
        
        return {
            'commission': commission,
            'slippage': slippage,
            'shortfall': shortfall,
            'execution_price': filled_price,
            'reference_price': reference_price,
            'filled_qty': qty,
            'execution_time': filled_order.filled_at
        }
        
    def _calculate_implementation_shortfall(self, filled_order: Dict,
                                         reference_price: float) -> float:
        """Calculate implementation shortfall"""
        filled_price = float(filled_order.filled_price)
        qty = float(filled_order.filled_qty)
        
        # Calculate trading costs
        explicit_costs = self._calculate_commission(filled_price, qty)
        price_impact = abs(filled_price - reference_price) * qty
        opportunity_cost = self._calculate_opportunity_cost(
            filled_order,
            reference_price
        )
        
        return explicit_costs + price_impact + opportunity_cost
        
    def _apply_execution_strategy(self, order_request: Dict,
                                strategy: Dict) -> None:
        """Apply execution strategy parameters to order"""
        if strategy.get('use_iceberg'):
            order_request.iceberg_qty = self._calculate_iceberg_qty(
                float(order_request.qty)
            )
            
        if strategy.get('price_limits'):
            order_request.limit_price = strategy['price_limits']['limit']
            order_request.stop_price = strategy['price_limits']['stop']
            
    def _combine_execution_results(self, 
                                 results: List[ExecutionResult]) -> ExecutionResult:
        """Combine multiple execution results into one"""
        total_qty = sum(r.filled_qty for r in results)
        weighted_price = sum(r.filled_qty * r.filled_price for r in results) / total_qty
        total_commission = sum(r.commission for r in results)
        avg_slippage = np.mean([r.slippage for r in results])
        
        return ExecutionResult(
            order_id=",".join(r.order_id for r in results),
            filled_qty=total_qty,
            filled_price=weighted_price,
            commission=total_commission,
            slippage=avg_slippage,
            success=all(r.success for r in results),
            info={'individual_results': [r.info for r in results]}
        )
        
    async def _smart_delay(self, chunk_index: int,
                          previous_results: List[ExecutionResult]) -> None:
        """Calculate smart delay between order chunks"""
        if not previous_results:
            await asyncio.sleep(1)
            return
            
        # Analyze market impact
        market_impact = self._analyze_market_impact(previous_results)
        
        # Calculate adaptive delay
        delay = self._calculate_adaptive_delay(
            chunk_index,
            market_impact,
            len(previous_results)
        )
        
        await asyncio.sleep(delay)
        
    def _analyze_market_impact(self, 
                             previous_results: List[ExecutionResult]) -> float:
        """Analyze market impact of previous executions"""
        if not previous_results:
            return 0.0
            
        impacts = []
        for result in previous_results:
            ref_price = result.info.get('reference_price', result.filled_price)
            impact = abs(result.filled_price - ref_price) / ref_price
            impacts.append(impact)
            
        return np.mean(impacts)
        
    def _calculate_adaptive_delay(self, chunk_index: int,
                                market_impact: float,
                                num_previous: int) -> float:
        """Calculate adaptive delay based on market impact"""
        base_delay = self.config['base_chunk_delay']
        impact_factor = 1 + market_impact * 10  # Increase delay with higher impact
        sequence_factor = 1 + (chunk_index / 10)  # Increase delay as sequence progresses
        
        return base_delay * impact_factor * sequence_factor