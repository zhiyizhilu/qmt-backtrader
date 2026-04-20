from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging


class StrategyExecutor(ABC):
    """策略执行器接口 - 定义统一的交易执行接口"""

    @abstractmethod
    def execute_buy(self, symbol: str, price: float, volume: int) -> Any:
        """执行买入操作"""
        pass

    @abstractmethod
    def execute_sell(self, symbol: str, price: float, volume: int) -> Any:
        """执行卖出操作"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        pass

    @abstractmethod
    def get_position(self, symbol: str = None) -> Any:
        """获取持仓"""
        pass

    @abstractmethod
    def get_account(self) -> Any:
        """获取账户信息"""
        pass

    @abstractmethod
    def get_cash(self) -> float:
        """获取可用现金"""
        pass

    @abstractmethod
    def get_position_size(self, symbol: str) -> int:
        """获取指定标的的持仓数量"""
        pass


class BacktestExecutor(StrategyExecutor):
    """回测执行器 - 通过backtrader执行交易"""

    def __init__(self, strategy):
        """初始化回测执行器"""
        self.strategy = strategy
        self._symbol_data_map: Dict[str, Any] = {}

    def register_data(self, symbol: str, data_feed) -> None:
        """注册标的与数据源的映射"""
        self._symbol_data_map[symbol] = data_feed

    def execute_buy(self, symbol: str, price: float, volume: int) -> Any:
        """执行买入操作"""
        data = self._symbol_data_map.get(symbol)
        if data:
            return self.strategy.buy(data=data, size=volume)
        return self.strategy.buy(size=volume)

    def execute_sell(self, symbol: str, price: float, volume: int) -> Any:
        """执行卖出操作"""
        data = self._symbol_data_map.get(symbol)
        if data:
            return self.strategy.sell(data=data, size=volume)
        return self.strategy.sell(size=volume)

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        try:
            if hasattr(self.strategy, 'cancel'):
                self.strategy.cancel(order_id)
                return True
        except Exception:
            pass
        return False

    def get_position(self, symbol: str = None) -> Any:
        """获取持仓"""
        if symbol:
            data = self._symbol_data_map.get(symbol)
            if data:
                return self.strategy.getposition(data)
        return self.strategy.position

    def get_account(self) -> Any:
        """获取账户信息"""
        return self.strategy.broker

    def get_cash(self) -> float:
        """获取可用现金"""
        return self.strategy.broker.getcash()

    def get_position_size(self, symbol: str) -> int:
        """获取指定标的的持仓数量"""
        data = self._symbol_data_map.get(symbol)
        if data:
            return self.strategy.getposition(data).size
        return 0


class QMTExecutor(StrategyExecutor):
    """QMT执行器 - 通过QMT接口执行交易"""

    def __init__(self, qmt_api):
        """初始化QMT执行器"""
        self.qmt_api = qmt_api
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def execute_buy(self, symbol: str, price: float, volume: int) -> Any:
        """执行买入操作"""
        result = self.qmt_api.buy(symbol, price, volume)
        if result is None:
            self.logger.error(f'买入失败: {symbol}, 价格: {price}, 数量: {volume}')
        return result

    def execute_sell(self, symbol: str, price: float, volume: int) -> Any:
        """执行卖出操作"""
        result = self.qmt_api.sell(symbol, price, volume)
        if result is None:
            self.logger.error(f'卖出失败: {symbol}, 价格: {price}, 数量: {volume}')
        return result

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        return self.qmt_api.cancel_order(order_id)

    def get_position(self, symbol: str = None) -> Any:
        """获取持仓"""
        return self.qmt_api.get_position(symbol)

    def get_account(self) -> Any:
        """获取账户信息"""
        return self.qmt_api.get_account()

    def get_cash(self) -> float:
        """获取可用现金"""
        account = self.qmt_api.get_account()
        if account and hasattr(account, 'cash'):
            return account.cash
        return 0.0

    def get_position_size(self, symbol: str) -> int:
        """获取指定标的的持仓数量"""
        position = self.qmt_api.get_position(symbol)
        if position and hasattr(position, 'volume'):
            return position.volume
        return 0
