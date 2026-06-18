import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from engine.data_feed import ArrayDataFeed


@dataclass
class Position:
    symbol: str = ''
    size: int = 0
    avg_price: float = 0.0

    @property
    def value(self) -> float:
        return self.size * self.avg_price


@dataclass
class Order:
    order_id: str = ''
    symbol: str = ''
    direction: str = ''
    size: int = 0
    price: float = 0.0
    executed_size: int = 0
    executed_price: float = 0.0
    commission: float = 0.0
    status: str = ''
    datetime: Any = None
    order_type: str = 'market'
    limit_price: Optional[float] = None

    STATUS_SUBMITTED = 'submitted'
    STATUS_COMPLETED = 'completed'
    STATUS_CANCELED = 'canceled'
    STATUS_REJECTED = 'rejected'
    STATUS_MARGIN = 'margin'
    STATUS_PENDING = 'pending'

    @property
    def is_buy(self) -> bool:
        return self.direction == 'buy'

    @property
    def is_completed(self) -> bool:
        return self.status == self.STATUS_COMPLETED


@dataclass
class Trade:
    trade_id: str = ''
    order_id: str = ''
    symbol: str = ''
    direction: str = ''
    size: int = 0
    price: float = 0.0
    commission: float = 0.0
    pnl: float = 0.0
    pnlcomm: float = 0.0
    datetime: Any = None


class SimulatedBroker:
    """模拟经纪商 - 实现回测场景下的订单执行

    复现 Backtrader 的 COC (cheat-on-close) 模式行为：
    - 订单以当前 bar 收盘价立即成交
    - 支持佣金和滑点
    - set_checksubmit(False) 允许卖出回款用于买入
    - 支持限价单（order_type='limit'）
    - 支持大单冲击成本模型（impact_model='linear'）
    - 支持成交量限制（max_participation_rate）
    - 支持聚宽式佣金结构：买入/卖出分离费率 + 印花税 + 最低佣金
    """

    def __init__(self, cash: float = 200000.0, commission: float = 0.0001,
                 slippage: float = 0.0, coc: bool = True,
                 impact_model: Optional[str] = None, impact_k: float = 0.1,
                 max_participation_rate: float = 0.0):
        self._cash = cash
        self._starting_cash = cash
        self._commission = commission
        self._slippage = slippage
        self._coc = coc
        self._check_submit = True
        self._impact_model = impact_model
        self._impact_k = impact_k
        self._max_participation_rate = max_participation_rate
        self._positions: Dict[str, Position] = {}
        self._orders: List[Order] = []
        self._trades: List[Trade] = []
        self._pending_orders: List[Order] = []
        self._order_counter = 0
        self._trade_counter = 0
        self._closed_trade_pnl: Dict[str, float] = {}
        # 聚宽式佣金参数
        self._open_commission: Optional[float] = None   # 买入佣金率
        self._close_commission: Optional[float] = None  # 卖出佣金率
        self._close_tax: float = 0.0                    # 卖出印花税率
        self._min_commission: float = 0.0               # 最低佣金(元)
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    @property
    def startingcash(self) -> float:
        return self._starting_cash

    def setcash(self, cash: float):
        self._cash = cash
        self._starting_cash = cash

    def setcommission(self, commission: float):
        self._commission = commission

    def set_order_cost(self, open_commission: float = 0.0,
                       close_commission: float = 0.0,
                       close_tax: float = 0.0,
                       min_commission: float = 0.0):
        """设置聚宽式佣金结构

        Args:
            open_commission: 买入佣金率（如 2.5/10000 = 0.00025）
            close_commission: 卖出佣金率（如 2.5/10000 = 0.00025）
            close_tax: 卖出印花税率（如 0.001 = 千1）
            min_commission: 最低佣金（元，如 5）
        """
        self._open_commission = open_commission
        self._close_commission = close_commission
        self._close_tax = close_tax
        self._min_commission = min_commission

    @staticmethod
    def _is_stamp_tax_exempt(symbol: str) -> bool:
        """判断标的是否免征印花税（ETF、债券等）

        A股印花税规则：仅股票卖出征收万5印花税，ETF/债券/基金免征
        """
        if not symbol or '.' not in symbol:
            return False
        code, suffix = symbol.rsplit('.', 1)
        # 沪市基金: 50xxxx(ETF)、51xxxx(ETF)、52xxxx(ETF)、56xxxx(ETF)、58xxxx(ETF)
        if suffix == 'SH' and code[:2] in ('50', '51', '52', '56', '58'):
            return True
        # 深市基金: 15xxxx(ETF/LOF)、16xxxx(LOF)、18xxxx(封闭式基金)
        if suffix == 'SZ' and code[:2] in ('15', '16', '18'):
            return True
        return False

    def _calc_commission(self, amount: float, direction: str,
                         symbol: str = '') -> float:
        """计算佣金

        如果设置了聚宽式佣金参数(_open_commission)，则使用聚宽式计算：
        - 买入: max(amount * open_commission, min_commission)
        - 卖出: max(amount * close_commission, min_commission) + amount * close_tax (股票)
                ETF/基金卖出免征印花税
        否则使用统一的 _commission 费率
        """
        if self._open_commission is not None:
            if direction == 'buy':
                comm = max(amount * self._open_commission, self._min_commission)
            else:
                comm = max(amount * self._close_commission, self._min_commission)
                # 仅股票卖出征收印花税，ETF/基金免征
                if not self._is_stamp_tax_exempt(symbol):
                    comm += amount * self._close_tax
            return comm
        else:
            return amount * self._commission

    def set_slippage_perc(self, slippage: float):
        self._slippage = slippage

    def set_coc(self, coc: bool):
        self._coc = coc

    def set_checksubmit(self, check: bool):
        self._check_submit = check

    def getcash(self) -> float:
        return self._cash

    def getvalue(self, data_feeds: Dict[str, ArrayDataFeed] = None,
                 current_indices: Dict[str, int] = None,
                 lazy_feeds: Dict = None,
                 current_date: str = None) -> float:
        total = self._cash
        for symbol, pos in self._positions.items():
            if pos.size == 0:
                continue
            price = None
            # 优先从预加载feed获取价格
            if data_feeds and current_indices:
                feed = data_feeds.get(symbol)
                idx = current_indices.get(symbol, -1)
                if feed and idx >= 0:
                    p = feed.get_close(idx)
                    if not math.isnan(p):
                        price = p
            # fallback到lazy feed
            if price is None and lazy_feeds and current_date:
                lazy = lazy_feeds.get(symbol)
                if lazy:
                    price = lazy.get_close_by_date(current_date)
            # 最终fallback到持仓均价
            if price is None or math.isnan(price) if price else True:
                price = pos.avg_price
            total += pos.size * price
        return total

    def get_position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol=symbol))

    def get_position_size(self, symbol: str) -> int:
        return self._positions.get(symbol, Position(symbol=symbol)).size

    def _apply_volume_limit(self, size: int, data_feed: ArrayDataFeed,
                            current_idx: int) -> int:
        if self._max_participation_rate <= 0:
            return size
        daily_volume = data_feed.get_volume(current_idx)
        if math.isnan(daily_volume) or daily_volume <= 0:
            return size
        max_size = int(daily_volume * self._max_participation_rate)
        if max_size <= 0:
            max_size = 1
        return min(size, max_size)

    def _apply_impact_cost(self, exec_price: float, size: int,
                           data_feed: ArrayDataFeed, current_idx: int,
                           direction: str) -> float:
        if self._impact_model is None:
            return exec_price
        if self._impact_model != 'linear':
            return exec_price
        daily_volume = data_feed.get_volume(current_idx)
        if math.isnan(daily_volume) or daily_volume <= 0:
            return exec_price
        impact = self._impact_k * (size / daily_volume) * exec_price
        if direction == 'buy':
            return exec_price + impact
        else:
            return exec_price - impact

    def submit_buy(self, symbol: str, size: int, data_feed: ArrayDataFeed,
                   current_idx: int, order_datetime=None, exec_price: float = None,
                   order_type: str = 'market', limit_price: Optional[float] = None) -> Optional[Order]:
        if current_idx < 0 or current_idx >= data_feed.length:
            return None

        close_price = data_feed.get_close(current_idx)
        if math.isnan(close_price):
            return None

        if order_type == 'limit' and limit_price is not None:
            if close_price > limit_price:
                return self._create_pending_order(
                    symbol, 'buy', size, close_price, order_datetime,
                    order_type='limit', limit_price=limit_price
                )

        size = self._apply_volume_limit(size, data_feed, current_idx)
        if size <= 0:
            return None

        if exec_price is None or exec_price <= 0:
            exec_price = close_price
        if self._slippage > 0:
            exec_price = exec_price * (1 + self._slippage)

        exec_price = self._apply_impact_cost(exec_price, size, data_feed, current_idx, 'buy')

        comm = self._calc_commission(exec_price * size, 'buy', symbol)

        cost = exec_price * size + comm
        if self._check_submit:
            if cost > self._cash:
                return self._create_rejected_order(symbol, 'buy', size, close_price, order_datetime)

        self._cash -= cost

        pos = self._positions.get(symbol, Position(symbol=symbol))
        old_size = pos.size
        old_avg = pos.avg_price

        if old_size >= 0:
            new_size = old_size + size
            new_avg = exec_price if new_size > 0 else 0.0
            if old_size > 0 and new_size > 0:
                new_avg = (old_size * old_avg + size * exec_price) / new_size
        else:
            new_size = old_size + size
            if new_size >= 0:
                close_size = min(size, abs(old_size))
                pnl = close_size * (old_avg - exec_price)
                self._record_trade(symbol, 'buy', close_size, exec_price, comm, pnl, order_datetime)
                if new_size > 0:
                    new_avg = exec_price
                else:
                    new_avg = 0.0
            else:
                new_avg = old_avg

        pos.size = new_size
        pos.avg_price = new_avg
        self._positions[symbol] = pos

        order = self._create_completed_order(symbol, 'buy', size, close_price,
                                              exec_price, size, comm, order_datetime)
        return order

    def submit_sell(self, symbol: str, size: int, data_feed: ArrayDataFeed,
                    current_idx: int, order_datetime=None, exec_price: float = None,
                    order_type: str = 'market', limit_price: Optional[float] = None) -> Optional[Order]:
        if current_idx < 0 or current_idx >= data_feed.length:
            return None

        close_price = data_feed.get_close(current_idx)
        if math.isnan(close_price):
            return None

        if order_type == 'limit' and limit_price is not None:
            if close_price < limit_price:
                return self._create_pending_order(
                    symbol, 'sell', size, close_price, order_datetime,
                    order_type='limit', limit_price=limit_price
                )

        size = self._apply_volume_limit(size, data_feed, current_idx)
        if size <= 0:
            return None

        if exec_price is None or exec_price <= 0:
            exec_price = close_price
        if self._slippage > 0:
            exec_price = exec_price * (1 - self._slippage)

        exec_price = self._apply_impact_cost(exec_price, size, data_feed, current_idx, 'sell')

        comm = self._calc_commission(exec_price * size, 'sell', symbol)

        pos = self._positions.get(symbol, Position(symbol=symbol))
        if pos.size <= 0:
            return None

        actual_size = min(size, pos.size)

        self._cash += (exec_price * actual_size - comm)

        close_size = actual_size
        pnl = close_size * (exec_price - pos.avg_price)
        self._record_trade(symbol, 'sell', close_size, exec_price, comm, pnl, order_datetime)

        new_size = pos.size - actual_size
        pos.size = new_size
        pos.avg_price = pos.avg_price if new_size > 0 else 0.0
        self._positions[symbol] = pos

        order = self._create_completed_order(symbol, 'sell', actual_size, close_price,
                                              exec_price, actual_size, comm, order_datetime)
        return order

    def submit_buy_lazy(self, symbol: str, size: int, current_price: float,
                         order_datetime=None) -> Optional[Order]:
        """买入lazy feed的股票（不需要data_feed和current_idx）"""
        if current_price is None or current_price <= 0 or math.isnan(current_price):
            return None

        exec_price = current_price
        if self._slippage > 0:
            exec_price = exec_price * (1 + self._slippage)

        comm = self._calc_commission(exec_price * size, 'buy', symbol)
        cost = exec_price * size + comm

        if self._check_submit:
            if cost > self._cash:
                return self._create_rejected_order(symbol, 'buy', size, current_price, order_datetime)

        self._cash -= cost

        pos = self._positions.get(symbol, Position(symbol=symbol))
        old_size = pos.size
        old_avg = pos.avg_price

        if old_size >= 0:
            new_size = old_size + size
            new_avg = exec_price if new_size > 0 else 0.0
            if old_size > 0 and new_size > 0:
                new_avg = (old_size * old_avg + size * exec_price) / new_size
        else:
            new_size = old_size + size
            if new_size >= 0:
                close_size = min(size, abs(old_size))
                pnl = close_size * (old_avg - exec_price)
                self._record_trade(symbol, 'buy', close_size, exec_price, comm, pnl, order_datetime)
                if new_size > 0:
                    new_avg = exec_price
                else:
                    new_avg = 0.0
            else:
                new_avg = old_avg

        pos.size = new_size
        pos.avg_price = new_avg
        self._positions[symbol] = pos

        order = self._create_completed_order(symbol, 'buy', size, current_price,
                                              exec_price, size, comm, order_datetime)
        return order

    def submit_sell_lazy(self, symbol: str, size: int, current_price: float,
                          order_datetime=None) -> Optional[Order]:
        """卖出lazy feed的股票（不需要data_feed和current_idx）"""
        if current_price is None or current_price <= 0 or math.isnan(current_price):
            return None

        exec_price = current_price
        if self._slippage > 0:
            exec_price = exec_price * (1 - self._slippage)

        comm = self._calc_commission(exec_price * size, 'sell', symbol)

        pos = self._positions.get(symbol, Position(symbol=symbol))
        if pos.size <= 0:
            return None

        actual_size = min(size, pos.size)
        self._cash += (exec_price * actual_size - comm)

        close_size = actual_size
        pnl = close_size * (exec_price - pos.avg_price)
        self._record_trade(symbol, 'sell', close_size, exec_price, comm, pnl, order_datetime)

        new_size = pos.size - actual_size
        pos.size = new_size
        pos.avg_price = pos.avg_price if new_size > 0 else 0.0
        self._positions[symbol] = pos

        order = self._create_completed_order(symbol, 'sell', actual_size, current_price,
                                              exec_price, actual_size, comm, order_datetime)
        return order

    def check_pending_orders(self, data_feeds: Dict[str, ArrayDataFeed],
                             current_indices: Dict[str, int],
                             order_datetime=None) -> List[Order]:
        """检查限价单是否触发成交

        在每个bar调用，检查所有pending订单是否满足限价条件。
        买单：当前价格 <= limit_price 时触发
        卖单：当前价格 >= limit_price 时触发

        Args:
            data_feeds: 数据源字典
            current_indices: 当前索引字典
            order_datetime: 订单时间

        Returns:
            本次成交的订单列表
        """
        if not self._pending_orders:
            return []

        executed = []
        remaining = []

        for order in self._pending_orders:
            feed = data_feeds.get(order.symbol)
            idx = current_indices.get(order.symbol, -1)

            if feed is None or idx < 0 or idx >= feed.length:
                remaining.append(order)
                continue

            close_price = feed.get_close(idx)
            if math.isnan(close_price):
                remaining.append(order)
                continue

            triggered = False
            if order.is_buy and close_price <= order.limit_price:
                triggered = True
            elif not order.is_buy and close_price >= order.limit_price:
                triggered = True

            if triggered:
                size = self._apply_volume_limit(order.size, feed, idx)
                if size <= 0:
                    remaining.append(order)
                    continue

                exec_price = order.limit_price
                if self._slippage > 0:
                    if order.is_buy:
                        exec_price = exec_price * (1 + self._slippage)
                    else:
                        exec_price = exec_price * (1 - self._slippage)

                exec_price = self._apply_impact_cost(exec_price, size, feed, idx, order.direction)

                comm = self._calc_commission(exec_price * size, 'buy' if order.is_buy else 'sell', order.symbol)

                if order.is_buy:
                    cost = exec_price * size + comm
                    if self._check_submit and cost > self._cash:
                        remaining.append(order)
                        continue

                    self._cash -= cost
                    pos = self._positions.get(order.symbol, Position(symbol=order.symbol))
                    old_size = pos.size
                    old_avg = pos.avg_price

                    if old_size >= 0:
                        new_size = old_size + size
                        new_avg = exec_price if new_size > 0 else 0.0
                        if old_size > 0 and new_size > 0:
                            new_avg = (old_size * old_avg + size * exec_price) / new_size
                    else:
                        new_size = old_size + size
                        if new_size >= 0:
                            close_size = min(size, abs(old_size))
                            pnl = close_size * (old_avg - exec_price)
                            self._record_trade(order.symbol, 'buy', close_size, exec_price, comm, pnl, order_datetime)
                            if new_size > 0:
                                new_avg = exec_price
                            else:
                                new_avg = 0.0
                        else:
                            new_avg = old_avg

                    pos.size = new_size
                    pos.avg_price = new_avg
                    self._positions[order.symbol] = pos

                    order.executed_size = size
                    order.executed_price = exec_price
                    order.commission = comm
                    order.status = Order.STATUS_COMPLETED
                else:
                    pos = self._positions.get(order.symbol, Position(symbol=order.symbol))
                    if pos.size <= 0:
                        remaining.append(order)
                        continue

                    actual_size = min(size, pos.size)
                    self._cash += (exec_price * actual_size - comm)

                    pnl = actual_size * (exec_price - pos.avg_price)
                    self._record_trade(order.symbol, 'sell', actual_size, exec_price, comm, pnl, order_datetime)

                    new_size = pos.size - actual_size
                    pos.size = new_size
                    pos.avg_price = pos.avg_price if new_size > 0 else 0.0
                    self._positions[order.symbol] = pos

                    order.executed_size = actual_size
                    order.executed_price = exec_price
                    order.commission = comm
                    order.status = Order.STATUS_COMPLETED

                executed.append(order)
            else:
                remaining.append(order)

        self._pending_orders = remaining
        return executed

    def cancel_pending_order(self, order_id: str) -> bool:
        """撤销限价挂单

        Args:
            order_id: 订单ID

        Returns:
            是否撤销成功
        """
        for i, order in enumerate(self._pending_orders):
            if order.order_id == order_id:
                order.status = Order.STATUS_CANCELED
                self._pending_orders.pop(i)
                return True
        return False

    def get_pending_orders(self) -> List[Order]:
        """获取所有挂单"""
        return list(self._pending_orders)

    def get_orders(self) -> List[Order]:
        return list(self._orders)

    def get_trades(self) -> List[Trade]:
        return list(self._trades)

    def _create_completed_order(self, symbol: str, direction: str, size: int,
                                 price: float, exec_price: float, exec_size: int,
                                 commission: float, order_datetime) -> Order:
        self._order_counter += 1
        order = Order(
            order_id=str(self._order_counter),
            symbol=symbol,
            direction=direction,
            size=size,
            price=price,
            executed_size=exec_size,
            executed_price=exec_price,
            commission=commission,
            status=Order.STATUS_COMPLETED,
            datetime=order_datetime,
        )
        self._orders.append(order)
        return order

    def _create_pending_order(self, symbol: str, direction: str, size: int,
                               price: float, order_datetime,
                               order_type: str = 'limit',
                               limit_price: Optional[float] = None) -> Order:
        self._order_counter += 1
        order = Order(
            order_id=str(self._order_counter),
            symbol=symbol,
            direction=direction,
            size=size,
            price=price,
            status=Order.STATUS_PENDING,
            datetime=order_datetime,
            order_type=order_type,
            limit_price=limit_price,
        )
        self._orders.append(order)
        self._pending_orders.append(order)
        return order

    def _create_rejected_order(self, symbol: str, direction: str, size: int,
                                price: float, order_datetime) -> Order:
        self._order_counter += 1
        order = Order(
            order_id=str(self._order_counter),
            symbol=symbol,
            direction=direction,
            size=size,
            price=price,
            status=Order.STATUS_REJECTED,
            datetime=order_datetime,
        )
        self._orders.append(order)
        return order

    def _record_trade(self, symbol: str, direction: str, size: int,
                       price: float, commission: float, pnl: float,
                       trade_datetime):
        self._trade_counter += 1
        trade = Trade(
            trade_id=str(self._trade_counter),
            symbol=symbol,
            direction=direction,
            size=size,
            price=price,
            commission=commission,
            pnl=pnl,
            pnlcomm=pnl - commission,
            datetime=trade_datetime,
        )
        self._trades.append(trade)
