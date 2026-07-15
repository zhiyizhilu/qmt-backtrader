from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging
import math

if TYPE_CHECKING:
    from core.virtual_book import VirtualBook
    from core.order_router import OrderRouter
    from core.data_adapter import MarketDataAdapter


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

    def __init__(self, strategy, data_adapter=None):
        """初始化回测执行器

        Args:
            strategy: backtrader Strategy 实例
            data_adapter: BacktraderDataAdapter 实例，用于涨跌停/停牌检查
        """
        self.strategy = strategy
        self._symbol_data_map: Dict[str, Any] = {}
        self._data_adapter = data_adapter

    def register_data(self, symbol: str, data_feed) -> None:
        """注册标的与数据源的映射"""
        self._symbol_data_map[symbol] = data_feed

    def execute_buy(self, symbol: str, price: float, volume: int) -> Any:
        data = self._symbol_data_map.get(symbol)
        if data:
            current_close = data.close[0]
            if math.isnan(current_close):
                logging.getLogger(__name__).warning(
                    f'[BacktestExecutor] 买入拒绝-数据为NaN: {symbol}'
                )
                return None
            if self._data_adapter and self._data_adapter.is_suspended(symbol):
                logging.getLogger(__name__).warning(
                    f'[BacktestExecutor] 买入拒绝-停牌(成交量为0): {symbol}'
                )
                return None
            if self._data_adapter and self._data_adapter.is_limit_up(symbol):
                logging.getLogger(__name__).warning(
                    f'[BacktestExecutor] 买入拒绝-涨停: {symbol}, '
                    f'收盘价={current_close:.2f}'
                )
                return None
            result = self.strategy.buy(data=data, size=volume)
            logging.getLogger(__name__).debug(
                f'[BacktestExecutor] 买入: {symbol}, 价格={price:.2f}, 数量={volume}, '
                f'数据源={data._name if hasattr(data, "_name") else "N/A"}'
            )
            return result
        logging.getLogger(__name__).warning(
            f'[BacktestExecutor] 买入失败-未找到数据源: {symbol}, 可用={list(self._symbol_data_map.keys())}'
        )
        return self.strategy.buy(size=volume)

    def execute_sell(self, symbol: str, price: float, volume: int) -> Any:
        data = self._symbol_data_map.get(symbol)
        if data:
            current_close = data.close[0]
            if math.isnan(current_close):
                logging.getLogger(__name__).warning(
                    f'[BacktestExecutor] 卖出拒绝-数据为NaN: {symbol}'
                )
                return None
            if self._data_adapter and self._data_adapter.is_suspended(symbol):
                logging.getLogger(__name__).warning(
                    f'[BacktestExecutor] 卖出拒绝-停牌(成交量为0): {symbol}'
                )
                return None
            if self._data_adapter and self._data_adapter.is_limit_down(symbol):
                logging.getLogger(__name__).warning(
                    f'[BacktestExecutor] 卖出拒绝-跌停: {symbol}, '
                    f'收盘价={current_close:.2f}'
                )
                return None
            result = self.strategy.sell(data=data, size=volume)
            logging.getLogger(__name__).debug(
                f'[BacktestExecutor] 卖出: {symbol}, 价格={price:.2f}, 数量={volume}, '
                f'数据源={data._name if hasattr(data, "_name") else "N/A"}'
            )
            return result
        logging.getLogger(__name__).warning(
            f'[BacktestExecutor] 卖出失败-未找到数据源: {symbol}, 可用={list(self._symbol_data_map.keys())}'
        )
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
    """QMT执行器 - 通过QMT接口执行交易

    支持虚拟簿记模式：当传入 virtual_book 时，持仓和资金查询
    走 VirtualBook，而非直接查账户，实现策略级隔离。
    """

    def __init__(self, qmt_api, virtual_book: 'VirtualBook' = None,
                 data_adapter: 'MarketDataAdapter' = None):
        """初始化QMT执行器

        Args:
            qmt_api: QMTTrader 实例，用于执行实际交易操作
            virtual_book: 虚拟持仓簿，可选。传入后持仓/资金查询走簿记
            data_adapter: 数据适配器，可选。传入后启用涨跌停/停牌校验
        """
        self.qmt_api = qmt_api
        self.virtual_book = virtual_book
        self._data_adapter = data_adapter
        self._order_router: Optional['OrderRouter'] = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def set_order_router(self, order_router: 'OrderRouter'):
        """设置订单路由器"""
        self._order_router = order_router

    def set_data_adapter(self, data_adapter: 'MarketDataAdapter'):
        """设置数据适配器，用于涨跌停/停牌校验"""
        self._data_adapter = data_adapter

    def execute_buy(self, symbol: str, price: float, volume: int) -> Any:
        """执行买入操作"""
        if self._data_adapter:
            if self._data_adapter.is_suspended(symbol):
                self.logger.warning(f'买入拒绝-停牌: {symbol}')
                return None
            if self._data_adapter.is_limit_up(symbol):
                self.logger.warning(f'买入拒绝-涨停: {symbol}')
                return None

        # VirtualBook 交易前校验：检查可用现金
        if self.virtual_book:
            # ETF佣金0.02%，股票0.02%+印花税0.1%，取0.05%预留佣金+滑点
            estimated_cost = price * volume * 1.0005
            available_cash = self.virtual_book.get_cash()
            # 扣减待确认买入订单占用资金
            for order in self.virtual_book._pending_orders.values():
                if order['action'] == 'buy':
                    available_cash -= order['price'] * order['volume'] * 1.0005
            if available_cash < estimated_cost:
                self.logger.warning(
                    f'买入拒绝-虚拟簿记资金不足: {symbol}, '
                    f'需={estimated_cost:.2f}, 可用={available_cash:.2f}'
                )
                return None

        strategy_name = self.virtual_book.strategy_id if self.virtual_book else ''
        result = self.qmt_api.buy(symbol, price, volume, strategy_name=strategy_name)
        if result is None:
            self.logger.error(f'买入失败: {symbol}, 价格: {price}, 数量: {volume}')
        else:
            order_id = str(result)
            if self.virtual_book:
                self.virtual_book.on_buy_submitted(symbol, price, volume, order_id)
            if self._order_router and self.virtual_book:
                self._order_router.register_order(order_id, self.virtual_book.strategy_id)
        return result

    def execute_sell(self, symbol: str, price: float, volume: int) -> Any:
        """执行卖出操作"""
        if self._data_adapter:
            if self._data_adapter.is_suspended(symbol):
                self.logger.warning(f'卖出拒绝-停牌: {symbol}')
                return None
            if self._data_adapter.is_limit_down(symbol):
                self.logger.warning(f'卖出拒绝-跌停: {symbol}')
                return None

        # VirtualBook 交易前校验：检查可用持仓
        if self.virtual_book:
            available_vol = self.virtual_book.get_position_size(symbol)
            # 扣减待确认卖出订单占用持仓
            for order in self.virtual_book._pending_orders.values():
                if order['action'] == 'sell' and order['symbol'] == symbol:
                    available_vol -= order['volume']
            if available_vol < volume:
                self.logger.warning(
                    f'卖出拒绝-虚拟簿记持仓不足: {symbol}, '
                    f'需={volume}, 可用={available_vol}'
                )
                return None

        strategy_name = self.virtual_book.strategy_id if self.virtual_book else ''
        result = self.qmt_api.sell(symbol, price, volume, strategy_name=strategy_name)
        if result is None:
            self.logger.error(f'卖出失败: {symbol}, 价格: {price}, 数量: {volume}')
        else:
            order_id = str(result)
            if self.virtual_book:
                self.virtual_book.on_sell_submitted(symbol, price, volume, order_id)
            if self._order_router and self.virtual_book:
                self._order_router.register_order(order_id, self.virtual_book.strategy_id)
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
        """获取可用现金 - 优先走 VirtualBook"""
        if self.virtual_book:
            return self.virtual_book.get_cash()
        account = self.qmt_api.get_account()
        if account and hasattr(account, 'cash'):
            return account.cash
        return 0.0

    def get_position_size(self, symbol: str) -> int:
        """获取指定标的的持仓数量 - 优先走 VirtualBook"""
        if self.virtual_book:
            return self.virtual_book.get_position_size(symbol)
        position = self.qmt_api.get_position(symbol)
        if position and hasattr(position, 'volume'):
            return position.volume
        return 0
