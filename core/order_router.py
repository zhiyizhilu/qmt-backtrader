from typing import Dict, Optional, Any
import logging

from core.strategy_logic import OrderInfo, TradeInfo


class OrderRouter:
    """订单路由 - 维护订单ID与策略实例的映射关系

    将 QMT 的委托/成交回调正确分发到对应的策略实例，
    支持同账户多策略场景下的订单归属追踪。
    """

    def __init__(self):
        self._order_instance_map: Dict[str, str] = {}
        self._instance_callbacks: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def register_order(self, order_id: str, instance_id: str):
        """下单后注册订单归属

        Args:
            order_id: 订单ID
            instance_id: 策略实例ID
        """
        self._order_instance_map[order_id] = instance_id
        self.logger.debug(f'订单注册: {order_id} → {instance_id}')

    def register_instance(self, instance_id: str, strategy):
        """注册策略实例的回调对象

        Args:
            instance_id: 策略实例ID
            strategy: 策略实例（StrategyLogic 子类）
        """
        self._instance_callbacks[instance_id] = strategy
        self.logger.debug(f'策略实例注册: {instance_id}')

    def route_order(self, order_id: str, order_info: OrderInfo):
        """将委托回调路由到正确的策略实例

        Args:
            order_id: 订单ID
            order_info: 委托信息
        """
        instance_id = self._order_instance_map.get(order_id)
        if instance_id and instance_id in self._instance_callbacks:
            self._instance_callbacks[instance_id].on_order(order_info)
            self.logger.debug(
                f'委托路由: {order_id} → {instance_id}, '
                f'{order_info.symbol} {order_info.direction} {order_info.status}'
            )
        else:
            self.logger.warning(f'委托路由失败: 订单 {order_id} 未找到归属策略')

    def route_trade(self, order_id: str, trade_info: TradeInfo):
        """将成交回调路由到正确的策略实例

        Args:
            order_id: 订单ID
            trade_info: 成交信息
        """
        instance_id = self._order_instance_map.get(order_id)
        if instance_id and instance_id in self._instance_callbacks:
            self._instance_callbacks[instance_id].on_trade(trade_info)
            self.logger.debug(
                f'成交路由: {order_id} → {instance_id}, '
                f'{trade_info.symbol} {trade_info.direction} x{trade_info.volume}@{trade_info.price}'
            )
        else:
            self.logger.warning(f'成交路由失败: 订单 {order_id} 未找到归属策略')

    def cleanup_order(self, order_id: str):
        """订单终态后清理映射

        Args:
            order_id: 订单ID
        """
        if order_id in self._order_instance_map:
            instance_id = self._order_instance_map.pop(order_id)
            self.logger.debug(f'订单清理: {order_id} (原归属: {instance_id})')

    def has_order(self, order_id: str) -> bool:
        """检查订单是否已注册"""
        return order_id in self._order_instance_map

    def get_instance_id(self, order_id: str) -> Optional[str]:
        """获取订单归属的策略实例ID"""
        return self._order_instance_map.get(order_id)

    def get_all_registered_instances(self) -> list:
        """获取所有已注册的策略实例ID"""
        return list(self._instance_callbacks.keys())

    def get_pending_order_count(self) -> int:
        """获取待确认订单数量"""
        return len(self._order_instance_map)
