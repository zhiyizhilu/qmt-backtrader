import time
from typing import Dict, List, Optional, Callable
import logging


class VirtualBook:
    """虚拟持仓簿 - 维护策略实例的独立持仓和资金视图

    核心原则：
    - VirtualBook 是策略持仓的唯一真相
    - 数据来源是每一笔交易的记录，不是从账户反推
    - 对账只校验总和一致性，不负责拆分归属
    """

    def __init__(self, strategy_id: str, initial_capital: float = 0):
        self.strategy_id = strategy_id
        self.initial_capital = initial_capital
        self._positions: Dict[str, int] = {}
        self._cash: float = initial_capital
        self._pending_orders: Dict[str, dict] = {}
        self._last_sync_time: Optional[float] = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def on_buy_submitted(self, symbol: str, price: float, volume: int, order_id: str):
        """买入下单，记录待确认订单"""
        self._pending_orders[order_id] = {
            'action': 'buy',
            'symbol': symbol,
            'price': price,
            'volume': volume,
        }
        self.logger.debug(
            f'[{self.strategy_id}] 买入待确认: {symbol}, '
            f'价格={price:.2f}, 数量={volume}, 订单={order_id}'
        )

    def on_sell_submitted(self, symbol: str, price: float, volume: int, order_id: str):
        """卖出下单，记录待确认订单"""
        self._pending_orders[order_id] = {
            'action': 'sell',
            'symbol': symbol,
            'price': price,
            'volume': volume,
        }
        self.logger.debug(
            f'[{self.strategy_id}] 卖出待确认: {symbol}, '
            f'价格={price:.2f}, 数量={volume}, 订单={order_id}'
        )

    def on_buy_filled(self, symbol: str, price: float, volume: int, commission: float = 0):
        """买入成交，更新持仓和现金"""
        self._positions[symbol] = self._positions.get(symbol, 0) + volume
        self._cash -= price * volume + commission
        self.logger.info(
            f'[{self.strategy_id}] 买入成交: {symbol}, '
            f'价格={price:.2f}, 数量={volume}, 手续费={commission:.2f}'
        )

    def on_sell_filled(self, symbol: str, price: float, volume: int, commission: float = 0):
        """卖出成交，更新持仓和现金"""
        self._positions[symbol] = self._positions.get(symbol, 0) - volume
        if self._positions[symbol] <= 0:
            self._positions.pop(symbol, None)
        self._cash += price * volume - commission
        self.logger.info(
            f'[{self.strategy_id}] 卖出成交: {symbol}, '
            f'价格={price:.2f}, 数量={volume}, 手续费={commission:.2f}'
        )

    def on_order_completed(self, order_id: str):
        """订单完成（含取消/拒绝），从待确认列表移除"""
        if order_id in self._pending_orders:
            pending = self._pending_orders.pop(order_id)
            self.logger.debug(
                f'[{self.strategy_id}] 订单完成: {order_id}, '
                f'动作={pending["action"]}, 标的={pending["symbol"]}'
            )

    def get_position_size(self, symbol: str) -> int:
        """查询策略级持仓数量"""
        return self._positions.get(symbol, 0)

    def get_cash(self) -> float:
        """查询策略级可用现金"""
        return self._cash

    def get_total_value(self, price_func: Callable[[str], Optional[float]]) -> float:
        """查询策略总市值

        Args:
            price_func: 获取标的当前价格的函数，接受symbol返回价格
        """
        pos_value = 0.0
        for symbol, volume in self._positions.items():
            if volume > 0:
                price = price_func(symbol)
                if price and price > 0:
                    pos_value += price * volume
        return self._cash + pos_value

    def get_symbols(self) -> List[str]:
        """获取所有持仓标的"""
        return [s for s, v in self._positions.items() if v > 0]

    def get_positions(self) -> Dict[str, int]:
        """获取所有持仓的副本"""
        return dict(self._positions)

    def has_pending_orders(self, symbol: str = None) -> bool:
        """是否有待确认订单

        Args:
            symbol: 可选，检查特定标的是否有待确认订单
        """
        if symbol is None:
            return len(self._pending_orders) > 0
        return any(o['symbol'] == symbol for o in self._pending_orders.values())

    def initialize_from_account(
        self,
        account_positions: Dict[str, int],
        account_cash: float,
        claimed_symbols: set
    ):
        """从账户实际状态初始化虚拟持仓

        Args:
            account_positions: 账户实际持仓 {symbol: volume}
            account_cash: 账户实际现金
            claimed_symbols: 已被其他策略认领的标的集合
        """
        for symbol, volume in account_positions.items():
            if symbol not in claimed_symbols and volume > 0:
                self._positions[symbol] = volume

        self._cash = min(self.initial_capital, account_cash) if self.initial_capital > 0 else account_cash
        self._last_sync_time = time.time()

        self.logger.info(
            f'[{self.strategy_id}] 初始化完成: '
            f'持仓={len(self._positions)}只, 现金={self._cash:.2f}'
        )

    def sync_with_account(
        self,
        account_positions: Dict[str, int],
        account_cash: float,
        holders_map: Dict[str, List['VirtualBook']]
    ):
        """定期对账 - 用账户实际数据校准 VirtualBook

        Args:
            account_positions: 账户实际持仓
            account_cash: 账户实际现金
            holders_map: 标的 → 持有该标的的 VirtualBook 列表
        """
        for symbol in set(list(self._positions.keys()) + list(account_positions.keys())):
            book_vol = self._positions.get(symbol, 0)
            actual_vol = account_positions.get(symbol, 0)

            if symbol not in self._positions and actual_vol > 0:
                continue

            if book_vol == actual_vol:
                continue

            if self.has_pending_orders(symbol):
                continue

            holders = holders_map.get(symbol, [])
            if len(holders) == 1 and holders[0].strategy_id == self.strategy_id:
                self._positions[symbol] = actual_vol
                if actual_vol <= 0:
                    self._positions.pop(symbol, None)
                self.logger.info(
                    f'[{self.strategy_id}] 对账校准: {symbol} '
                    f'簿记={book_vol}, 实际={actual_vol}'
                )

        if abs(account_cash - self._cash) > 1.0:
            if not self.has_pending_orders():
                self._cash = account_cash
                self.logger.info(
                    f'[{self.strategy_id}] 现金校准: 簿记={self._cash:.2f}, 实际={account_cash:.2f}'
                )

        self._last_sync_time = time.time()

    def __repr__(self):
        return (
            f'VirtualBook(id={self.strategy_id}, '
            f'持仓={len(self._positions)}只, '
            f'现金={self._cash:.2f}, '
            f'待确认={len(self._pending_orders)})'
        )
