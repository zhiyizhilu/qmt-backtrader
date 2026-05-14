import datetime
import math
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Any

from core.data_adapter import MarketDataAdapter, get_limit_ratio
from core.executor import StrategyExecutor
from core.strategy_logic import OrderInfo
from engine.data_feed import ArrayDataFeed
from engine.broker import SimulatedBroker
from engine.timeline import Timeline


class EngineDataAdapter(MarketDataAdapter):
    """自研引擎数据适配器 - 从 ArrayDataFeed 读取数据

    替代 BacktraderDataAdapter，实现 MarketDataAdapter 接口。
    数据直接从 numpy 数组读取，无 Python 对象包装。
    涨跌停/停牌判断逻辑与 BacktraderDataAdapter 完全一致。
    """

    DAILY_PERIODS = {'1d', 'day', 'daily'}
    MAX_CLOSE_PRICES = 5000

    def __init__(self, data_feeds: Dict[str, ArrayDataFeed],
                 timeline: Timeline, period: str = '1d'):
        self._data_feeds = data_feeds
        self._timeline = timeline
        self._period = period
        self._current_global_idx: int = -1
        self._current_local_indices: Dict[str, int] = {}
        self._close_prices: Dict[str, deque] = {}
        self._daily_close_prices: Dict[str, deque] = {}
        self._ohlcv_data: Dict[str, deque] = {}
        self._last_daily_date: Dict[str, Optional[datetime.date]] = {}
        self._current_day_close: Dict[str, Optional[float]] = {}

        for symbol in data_feeds:
            self._close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
            self._daily_close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
            self._ohlcv_data[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
            self._last_daily_date[symbol] = None
            self._current_day_close[symbol] = None

    @property
    def period(self) -> str:
        return self._period

    @period.setter
    def period(self, value: str) -> None:
        self._period = value

    def _is_daily(self) -> bool:
        return self._period in self.DAILY_PERIODS

    def update(self, global_idx: int):
        """更新当前全局索引，计算每个数据源的本地索引并缓存价格数据"""
        self._current_global_idx = global_idx
        self._current_local_indices = self._timeline.get_all_feed_bar_indices(global_idx)

        for symbol, local_idx in self._current_local_indices.items():
            feed = self._data_feeds.get(symbol)
            if feed is None:
                continue

            close = feed.get_close(local_idx)
            if math.isnan(close):
                continue

            self._close_prices[symbol].append(close)

            if self._is_daily():
                self._daily_close_prices[symbol].append(close)
                o = feed.get_open(local_idx)
                h = feed.get_high(local_idx)
                l = feed.get_low(local_idx)
                v = feed.get_volume(local_idx)
                if not (math.isnan(o) or math.isnan(h) or math.isnan(l)):
                    self._ohlcv_data[symbol].append({
                        'open': o,
                        'high': h,
                        'low': l,
                        'close': close,
                        'volume': 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v),
                    })
            else:
                current_date = feed.get_date(local_idx)
                if self._last_daily_date.get(symbol) != current_date:
                    if self._last_daily_date.get(symbol) is not None and self._current_day_close.get(symbol) is not None:
                        self._daily_close_prices[symbol].append(self._current_day_close[symbol])
                    self._last_daily_date[symbol] = current_date
                self._current_day_close[symbol] = close

    def finalize_daily_bars(self):
        for symbol, feed in self._data_feeds.items():
            if not self._is_daily() and self._last_daily_date.get(symbol) is not None:
                local_idx = self._current_local_indices.get(symbol, -1)
                if local_idx >= 0:
                    close = feed.get_close(local_idx)
                    if not math.isnan(close):
                        self._daily_close_prices[symbol].append(close)

    def get_current_price(self, symbol: str) -> Optional[float]:
        local_idx = self._current_local_indices.get(symbol, -1)
        if local_idx < 0:
            return None
        feed = self._data_feeds.get(symbol)
        if feed is None:
            return None
        price = feed.get_close(local_idx)
        if math.isnan(price):
            return None
        return price

    def get_close_prices(self, symbol: str, period: int = None) -> List[float]:
        if self._is_daily():
            prices = list(self._close_prices.get(symbol, []))
        else:
            prices = list(self._daily_close_prices.get(symbol, []))
            if self._current_day_close.get(symbol) is not None:
                prices.append(self._current_day_close[symbol])
        if period is not None:
            result = prices[-period:] if len(prices) >= period else prices
            return result
        return prices

    def get_current_date(self) -> Optional[datetime.date]:
        return self._timeline.get_date(self._current_global_idx)

    def get_current_datetime(self) -> Optional[datetime.datetime]:
        return self._timeline.get_datetime(self._current_global_idx)

    def get_next_datetime(self) -> Optional[datetime.datetime]:
        next_idx = self._current_global_idx + 1
        dt = self._timeline.get_datetime(next_idx)
        if dt is None:
            dt = self._timeline.get_datetime(self._current_global_idx)
        if dt is not None and self._is_daily() and dt.hour == 0 and dt.minute == 0:
            dt = dt.replace(hour=15, minute=0)
        return dt

    def get_symbols(self) -> List[str]:
        return list(self._data_feeds.keys())

    def is_suspended(self, symbol: str) -> bool:
        local_idx = self._current_local_indices.get(symbol, -1)
        if local_idx < 0:
            return True
        feed = self._data_feeds.get(symbol)
        if feed is None:
            return True
        return feed.is_volume_zero_or_nan(local_idx)

    def _get_prev_daily_close(self, symbol: str) -> Optional[float]:
        if self._is_daily():
            local_idx = self._current_local_indices.get(symbol, -1)
            if local_idx is not None and local_idx >= 1:
                feed = self._data_feeds.get(symbol)
                if feed:
                    prev = feed.get_close(local_idx - 1)
                    if not math.isnan(prev) and prev > 0:
                        return prev
            return None
        else:
            daily_closes = list(self._daily_close_prices.get(symbol, []))
            if daily_closes:
                return daily_closes[-1]
            return None

    def is_limit_up(self, symbol: str) -> bool:
        local_idx = self._current_local_indices.get(symbol, -1)
        if local_idx < 0:
            return False
        feed = self._data_feeds.get(symbol)
        if feed is None:
            return False
        try:
            close = feed.get_close(local_idx)
            if math.isnan(close):
                return False
            prev_close = self._get_prev_daily_close(symbol)
            if prev_close is None or prev_close <= 0:
                return False
        except (AttributeError, IndexError):
            return False
        limit_ratio = get_limit_ratio(symbol)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        return close >= limit_price - 0.005

    def is_limit_down(self, symbol: str) -> bool:
        local_idx = self._current_local_indices.get(symbol, -1)
        if local_idx < 0:
            return False
        feed = self._data_feeds.get(symbol)
        if feed is None:
            return False
        try:
            close = feed.get_close(local_idx)
            if math.isnan(close):
                return False
            prev_close = self._get_prev_daily_close(symbol)
            if prev_close is None or prev_close <= 0:
                return False
        except (AttributeError, IndexError):
            return False
        limit_ratio = get_limit_ratio(symbol)
        limit_price = round(prev_close * (1 - limit_ratio), 2)
        return close <= limit_price + 0.005

    def get_ohlcv_data(self, symbol: str, period: int = None) -> List[Dict[str, float]]:
        data_list = list(self._ohlcv_data.get(symbol, []))
        if period is not None:
            return data_list[-period:] if len(data_list) >= period else data_list
        return data_list


class EngineExecutor(StrategyExecutor):
    """自研引擎执行器 - 通过 SimulatedBroker 执行交易

    替代 BacktestExecutor，实现 StrategyExecutor 接口。
    """

    def __init__(self, broker: SimulatedBroker, data_adapter: EngineDataAdapter,
                 data_feeds: Dict[str, ArrayDataFeed]):
        self._broker = broker
        self._data_adapter = data_adapter
        self._data_feeds = data_feeds
        self._bar_start_cash: float = broker.getcash()
        self._bar_start_positions: Dict[str, int] = {}
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def execute_buy(self, symbol: str, price: float, volume: int) -> Any:
        feed = self._data_feeds.get(symbol)
        if feed is None:
            self.logger.warning(f'[EngineExecutor] 买入失败-未找到数据源: {symbol}')
            return None

        local_idx = self._data_adapter._current_local_indices.get(symbol, -1)
        if local_idx < 0:
            self.logger.warning(f'[EngineExecutor] 买入失败-无当前索引: {symbol}')
            return None

        current_close = feed.get_close(local_idx)
        if math.isnan(current_close):
            self.logger.warning(f'[EngineExecutor] 买入拒绝-数据为NaN: {symbol}')
            return None

        if self._data_adapter.is_suspended(symbol):
            self.logger.warning(f'[EngineExecutor] 买入拒绝-停牌(成交量为0): {symbol}')
            return None

        if self._data_adapter.is_limit_up(symbol):
            self.logger.warning(
                f'[EngineExecutor] 买入拒绝-涨停: {symbol}, 收盘价={current_close:.2f}'
            )
            return None

        order_datetime = self._data_adapter.get_next_datetime()
        order = self._broker.submit_buy(symbol, volume, feed, local_idx, order_datetime)

        if order is not None and order.status == OrderInfo.STATUS_COMPLETED:
            order_info = OrderInfo(
                order_id=order.order_id,
                symbol=symbol,
                direction='buy',
                price=order.price,
                volume=order.size,
                status=OrderInfo.STATUS_COMPLETED,
                executed_volume=order.executed_size,
                executed_price=order.executed_price,
                commission=order.commission,
                datetime=order_datetime,
            )
            logic = self._get_strategy_logic()
            if logic:
                logic.on_order(order_info)
                logic.on_trade(self._convert_to_trade_info(order, order_datetime))

        return order

    def execute_sell(self, symbol: str, price: float, volume: int) -> Any:
        feed = self._data_feeds.get(symbol)
        if feed is None:
            self.logger.warning(f'[EngineExecutor] 卖出失败-未找到数据源: {symbol}')
            return None

        local_idx = self._data_adapter._current_local_indices.get(symbol, -1)
        if local_idx < 0:
            self.logger.warning(f'[EngineExecutor] 卖出失败-无当前索引: {symbol}')
            return None

        current_close = feed.get_close(local_idx)
        if math.isnan(current_close):
            self.logger.warning(f'[EngineExecutor] 卖出拒绝-数据为NaN: {symbol}')
            return None

        if self._data_adapter.is_suspended(symbol):
            self.logger.warning(f'[EngineExecutor] 卖出拒绝-停牌(成交量为0): {symbol}')
            return None

        if self._data_adapter.is_limit_down(symbol):
            self.logger.warning(
                f'[EngineExecutor] 卖出拒绝-跌停: {symbol}, 收盘价={current_close:.2f}'
            )
            return None

        order_datetime = self._data_adapter.get_next_datetime()
        order = self._broker.submit_sell(symbol, volume, feed, local_idx, order_datetime)

        if order is not None and order.status == OrderInfo.STATUS_COMPLETED:
            order_info = OrderInfo(
                order_id=order.order_id,
                symbol=symbol,
                direction='sell',
                price=order.price,
                volume=order.size,
                status=OrderInfo.STATUS_COMPLETED,
                executed_volume=order.executed_size,
                executed_price=order.executed_price,
                commission=order.commission,
                datetime=order_datetime,
            )
            logic = self._get_strategy_logic()
            if logic:
                logic.on_order(order_info)
                logic.on_trade(self._convert_to_trade_info(order, order_datetime))

        return order

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_position(self, symbol: str = None) -> Any:
        if symbol:
            return self._broker.get_position(symbol)
        return None

    def get_account(self) -> Any:
        return self._broker

    def get_cash(self) -> float:
        return self._bar_start_cash

    def get_position_size(self, symbol: str) -> int:
        if symbol in self._bar_start_positions:
            return self._bar_start_positions[symbol]
        return self._broker.get_position_size(symbol)

    def snapshot_cash(self):
        self._bar_start_cash = self._broker.getcash()
        self._bar_start_positions = {
            symbol: self._broker.get_position_size(symbol)
            for symbol in self._data_feeds
        }

    def _get_strategy_logic(self):
        return getattr(self, '_strategy_logic', None)

    def set_strategy_logic(self, logic):
        self._strategy_logic = logic

    def _convert_to_trade_info(self, order, order_datetime):
        from core.strategy_logic import TradeInfo
        return TradeInfo(
            trade_id=order.order_id,
            order_id=order.order_id,
            symbol=order.symbol,
            direction=order.direction,
            price=order.executed_price,
            volume=order.executed_size,
            commission=order.commission,
            pnl=0.0,
        )
