import datetime
import math
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Any

from core.data_adapter import MarketDataAdapter, get_limit_ratio
from core.executor import StrategyExecutor
from core.strategy_logic import OrderInfo
from engine.data_feed import ArrayDataFeed, LazyDataFeed
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
                 timeline: Timeline, period: str = '1d',
                 lazy_feeds: Dict[str, LazyDataFeed] = None):
        self._data_feeds = data_feeds
        self._timeline = timeline
        self._period = period
        self._lazy_feeds: Dict[str, LazyDataFeed] = lazy_feeds or {}
        self._current_global_idx: int = -1
        self._current_local_indices: Dict[str, int] = {}
        self._close_prices: Dict[str, deque] = {}
        self._daily_close_prices: Dict[str, deque] = {}
        self._ohlcv_data: Dict[str, deque] = {}
        self._last_daily_date: Dict[str, Optional[datetime.date]] = {}
        self._current_day_close: Dict[str, Optional[float]] = {}
        self._current_day_open: Dict[str, Optional[float]] = {}
        self._current_day_high: Dict[str, Optional[float]] = {}
        self._current_day_low: Dict[str, Optional[float]] = {}
        self._current_day_volume: Dict[str, float] = {}

        for symbol in data_feeds:
            self._close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
            self._daily_close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
            self._ohlcv_data[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
            self._last_daily_date[symbol] = None
            self._current_day_close[symbol] = None
            self._current_day_open[symbol] = None
            self._current_day_high[symbol] = None
            self._current_day_low[symbol] = None
            self._current_day_volume[symbol] = 0.0

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
        indices_column = self._timeline.get_feed_local_indices_column(global_idx)

        self._current_local_indices.clear()
        for feed_idx in range(len(indices_column)):
            local_idx = int(indices_column[feed_idx])
            if local_idx >= 0:
                self._current_local_indices[self._timeline._feed_symbols[feed_idx]] = local_idx

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
                o = feed.get_open(local_idx)
                h = feed.get_high(local_idx)
                l = feed.get_low(local_idx)
                v = feed.get_volume(local_idx)
                v = 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)

                if self._last_daily_date.get(symbol) != current_date:
                    if self._last_daily_date.get(symbol) is not None and self._current_day_close.get(symbol) is not None:
                        self._daily_close_prices[symbol].append(self._current_day_close[symbol])
                        day_open = self._current_day_open.get(symbol)
                        day_high = self._current_day_high.get(symbol)
                        day_low = self._current_day_low.get(symbol)
                        day_vol = self._current_day_volume.get(symbol, 0.0)
                        if day_open is not None and day_high is not None and day_low is not None:
                            self._ohlcv_data[symbol].append({
                                'open': day_open,
                                'high': day_high,
                                'low': day_low,
                                'close': self._current_day_close[symbol],
                                'volume': day_vol,
                            })
                    self._last_daily_date[symbol] = current_date
                    self._current_day_open[symbol] = o if not math.isnan(o) else close
                    self._current_day_high[symbol] = h if not math.isnan(h) else close
                    self._current_day_low[symbol] = l if not math.isnan(l) else close
                    self._current_day_volume[symbol] = v
                else:
                    if not math.isnan(h) and self._current_day_high.get(symbol) is not None:
                        self._current_day_high[symbol] = max(self._current_day_high[symbol], h)
                    if not math.isnan(l) and self._current_day_low.get(symbol) is not None:
                        self._current_day_low[symbol] = min(self._current_day_low[symbol], l)
                    self._current_day_volume[symbol] = self._current_day_volume.get(symbol, 0.0) + v

                self._current_day_close[symbol] = close

    def finalize_daily_bars(self):
        for symbol, feed in self._data_feeds.items():
            if not self._is_daily() and self._last_daily_date.get(symbol) is not None:
                local_idx = self._current_local_indices.get(symbol, -1)
                if local_idx >= 0:
                    close = feed.get_close(local_idx)
                    if not math.isnan(close):
                        self._daily_close_prices[symbol].append(close)
                        day_open = self._current_day_open.get(symbol)
                        day_high = self._current_day_high.get(symbol)
                        day_low = self._current_day_low.get(symbol)
                        day_vol = self._current_day_volume.get(symbol, 0.0)
                        if day_open is not None and day_high is not None and day_low is not None:
                            self._ohlcv_data[symbol].append({
                                'open': day_open,
                                'high': day_high,
                                'low': day_low,
                                'close': close,
                                'volume': day_vol,
                            })

    def get_current_price(self, symbol: str) -> Optional[float]:
        # 优先从预加载feed获取
        local_idx = self._current_local_indices.get(symbol, -1)
        if local_idx >= 0:
            feed = self._data_feeds.get(symbol)
            if feed is not None:
                price = feed.get_close(local_idx)
                if not math.isnan(price):
                    return price
        # 从lazy feed按需获取
        lazy = self._lazy_feeds.get(symbol)
        if lazy is not None:
            current_date = self.get_current_date()
            if current_date:
                return lazy.get_close_by_date(current_date.strftime('%Y-%m-%d'))
        return None

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
        # 优先从预加载feed判断
        local_idx = self._current_local_indices.get(symbol, -1)
        if local_idx >= 0:
            feed = self._data_feeds.get(symbol)
            if feed is not None:
                return feed.is_volume_zero_or_nan(local_idx)
        # 从lazy feed判断
        lazy = self._lazy_feeds.get(symbol)
        if lazy is not None:
            current_date = self.get_current_date()
            if current_date:
                return lazy.is_suspended(current_date.strftime('%Y-%m-%d'))
        # 未知symbol：如果策略自行管理数据源（如1m策略通过QMT直接获取），
        # 不应因框架未加载该symbol就视为停牌，返回False让策略自行判断
        return False

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
        # 优先从预加载feed判断
        local_idx = self._current_local_indices.get(symbol, -1)
        if local_idx >= 0:
            feed = self._data_feeds.get(symbol)
            if feed is not None:
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
        # 从lazy feed判断
        lazy = self._lazy_feeds.get(symbol)
        if lazy is not None:
            current_date = self.get_current_date()
            if current_date:
                date_str = current_date.strftime('%Y-%m-%d')
                bar = lazy.get_bar_by_date(date_str)
                if bar and bar.close > 0:
                    prev_close = lazy.get_prev_close(date_str)
                    if prev_close and prev_close > 0:
                        limit_ratio = get_limit_ratio(symbol)
                        limit_price = round(prev_close * (1 + limit_ratio), 2)
                        return bar.close >= limit_price - 0.005
        return False

    def is_limit_down(self, symbol: str) -> bool:
        # 优先从预加载feed判断
        local_idx = self._current_local_indices.get(symbol, -1)
        if local_idx >= 0:
            feed = self._data_feeds.get(symbol)
            if feed is not None:
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
        # 从lazy feed判断
        lazy = self._lazy_feeds.get(symbol)
        if lazy is not None:
            current_date = self.get_current_date()
            if current_date:
                date_str = current_date.strftime('%Y-%m-%d')
                bar = lazy.get_bar_by_date(date_str)
                if bar and bar.close > 0:
                    prev_close = lazy.get_prev_close(date_str)
                    if prev_close and prev_close > 0:
                        limit_ratio = get_limit_ratio(symbol)
                        limit_price = round(prev_close * (1 - limit_ratio), 2)
                        return bar.close <= limit_price + 0.005
        return False

    def get_ohlcv_data(self, symbol: str, period: int = None) -> List[Dict[str, float]]:
        data_list = list(self._ohlcv_data.get(symbol, []))
        if period is not None:
            return data_list[-period:] if len(data_list) >= period else data_list
        return data_list

    def get_return_over_days(self, symbol: str, num_days: int) -> Optional[Dict[str, Any]]:
        """基于统一时间轴计算N个交易日收益率

        使用 Timeline 确定参考日期，再从数据源查找该日期的收盘价，
        保证不同数据源对比的是同一个日历日期。
        """
        current_idx = self._current_global_idx
        ref_idx = current_idx - num_days
        if ref_idx < 0:
            return None

        current_price = self.get_current_price(symbol)
        if current_price is None:
            return None

        feed_idx = self._timeline.get_feed_index_by_symbol(symbol)
        if feed_idx < 0:
            return None

        feed = self._data_feeds.get(symbol)
        if feed is None:
            return None

        ref_price = None
        ref_date = None
        for idx in range(ref_idx, -1, -1):
            local_idx = self._timeline.get_feed_bar_index(feed_idx, idx)
            if local_idx >= 0:
                close = feed.get_close(local_idx)
                if not math.isnan(close) and close > 0:
                    ref_price = close
                    ref_date = self._timeline.get_date(idx)
                    break

        if ref_price is None:
            return None

        return {
            'rate': (current_price - ref_price) / ref_price,
            'start_price': ref_price,
            'end_price': current_price,
            'start_date': ref_date,
        }

    def get_close_prices_for_days(self, symbol: str, num_days: int) -> List[float]:
        """基于统一时间轴获取最近N个交易日的收盘价序列

        使用 Timeline 确定日期范围，再从数据源逐日查找收盘价，
        保证不同数据源返回的是同一组日历日期的收盘价。
        缺失数据的日期使用最近可用价格前向填充。
        """
        feed_idx = self._timeline.get_feed_index_by_symbol(symbol)
        if feed_idx < 0:
            return []

        feed = self._data_feeds.get(symbol)
        if feed is None:
            return []

        prices = []
        last_valid_price = None

        start_idx = max(0, self._current_global_idx - num_days)
        for idx in range(start_idx, self._current_global_idx + 1):
            local_idx = self._timeline.get_feed_bar_index(feed_idx, idx)
            if local_idx >= 0:
                close = feed.get_close(local_idx)
                if not math.isnan(close) and close > 0:
                    prices.append(close)
                    last_valid_price = close
                    continue
            if last_valid_price is not None:
                prices.append(last_valid_price)

        return prices


class EngineExecutor(StrategyExecutor):
    """自研引擎执行器 - 通过 SimulatedBroker 执行交易

    替代 BacktestExecutor，实现 StrategyExecutor 接口。
    """

    def __init__(self, broker: SimulatedBroker, data_adapter: EngineDataAdapter,
                 data_feeds: Dict[str, ArrayDataFeed],
                 lazy_feeds: Dict[str, LazyDataFeed] = None):
        self._broker = broker
        self._data_adapter = data_adapter
        self._data_feeds = data_feeds
        self._lazy_feeds: Dict[str, LazyDataFeed] = lazy_feeds or {}
        self._bar_start_cash: float = broker.getcash()
        self._bar_start_positions: Dict[str, int] = {}
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def execute_buy(self, symbol: str, price: float, volume: int) -> Any:
        # 预加载feed - 原有逻辑
        feed = self._data_feeds.get(symbol)
        if feed is not None:
            return self._execute_buy_preloaded(symbol, price, volume, feed)

        # lazy feed
        lazy = self._lazy_feeds.get(symbol)
        if lazy is None:
            self.logger.warning(f'[EngineExecutor] 买入失败-未找到数据源: {symbol}')
            return None

        if self._data_adapter.is_suspended(symbol):
            self.logger.warning(f'[EngineExecutor] 买入拒绝-停牌: {symbol}')
            return None

        if self._data_adapter.is_limit_up(symbol):
            self.logger.warning(f'[EngineExecutor] 买入拒绝-涨停: {symbol}')
            return None

        current_price = self._data_adapter.get_current_price(symbol)
        if current_price is None:
            self.logger.warning(f'[EngineExecutor] 买入失败-无法获取价格: {symbol}')
            return None

        order_datetime = self._data_adapter.get_next_datetime()
        order = self._broker.submit_buy_lazy(symbol, volume, current_price, order_datetime)

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

    def _execute_buy_preloaded(self, symbol: str, price: float, volume: int,
                                feed: ArrayDataFeed) -> Any:
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
        # 预加载feed - 原有逻辑
        feed = self._data_feeds.get(symbol)
        if feed is not None:
            return self._execute_sell_preloaded(symbol, price, volume, feed)

        # lazy feed
        lazy = self._lazy_feeds.get(symbol)
        if lazy is None:
            self.logger.warning(f'[EngineExecutor] 卖出失败-未找到数据源: {symbol}')
            return None

        if self._data_adapter.is_suspended(symbol):
            self.logger.warning(f'[EngineExecutor] 卖出拒绝-停牌: {symbol}')
            return None

        if self._data_adapter.is_limit_down(symbol):
            self.logger.warning(f'[EngineExecutor] 卖出拒绝-跌停: {symbol}')
            return None

        current_price = self._data_adapter.get_current_price(symbol)
        if current_price is None:
            self.logger.warning(f'[EngineExecutor] 卖出失败-无法获取价格: {symbol}')
            return None

        order_datetime = self._data_adapter.get_next_datetime()
        order = self._broker.submit_sell_lazy(symbol, volume, current_price, order_datetime)

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

    def _execute_sell_preloaded(self, symbol: str, price: float, volume: int,
                                 feed: ArrayDataFeed) -> Any:
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
