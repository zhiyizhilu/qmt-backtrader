import datetime
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Any


class MarketDataAdapter(ABC):
    """市场数据适配器基类 - 提供统一的数据访问接口"""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        pass

    @abstractmethod
    def get_close_prices(self, symbol: str, period: int = None) -> List[float]:
        """获取收盘价序列"""
        pass

    @abstractmethod
    def get_current_date(self) -> Optional[datetime.date]:
        """获取当前日期"""
        pass

    def get_symbols(self) -> List[str]:
        """获取已注册的标的列表"""
        return []


class BacktraderDataAdapter(MarketDataAdapter):
    """Backtrader数据适配器 - 回测模式下使用

    支持多周期数据：
    - 日线(1d): 每个bar代表一天，close_prices直接存储日线收盘价
    - 分钟线(1m/5m等): 每个bar代表一分钟/五分钟，close_prices聚合为日线收盘价
    - tick: 逐笔数据，close_prices聚合为日线收盘价
    """

    DAILY_PERIODS = {'1d', 'day', 'daily'}
    MAX_CLOSE_PRICES = 5000

    def __init__(self, period: str = '1d'):
        self._period = period
        self._symbol_data_map: Dict[str, Any] = {}
        self._close_prices: Dict[str, deque] = {}
        self._daily_close_prices: Dict[str, deque] = {}
        self._last_daily_date: Dict[str, Optional[datetime.date]] = {}
        self._current_day_close: Dict[str, Optional[float]] = {}

    def register_data(self, symbol: str, data_feed) -> None:
        """注册标的与数据源的映射"""
        self._symbol_data_map[symbol] = data_feed
        self._close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
        self._daily_close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
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

    def update_from_backtrader(self) -> None:
        """从backtrader数据源更新当前bar的数据"""
        import math

        for symbol, data_feed in self._symbol_data_map.items():
            close = data_feed.close[0]

            # 跳过 NaN 数据（前置填充行），不加入收盘价队列
            if math.isnan(close):
                continue

            self._close_prices[symbol].append(close)

            if self._is_daily():
                self._daily_close_prices[symbol].append(close)
            else:
                current_date = data_feed.datetime.date(0)
                if self._last_daily_date[symbol] != current_date:
                    if self._last_daily_date[symbol] is not None and self._current_day_close[symbol] is not None:
                        self._daily_close_prices[symbol].append(self._current_day_close[symbol])
                    self._last_daily_date[symbol] = current_date
                self._current_day_close[symbol] = close

    def finalize_daily_bars(self) -> None:
        """在回测结束时调用，将最后一天的收盘价加入日线列表"""
        for symbol, data_feed in self._symbol_data_map.items():
            if not self._is_daily() and self._last_daily_date.get(symbol) is not None:
                close = data_feed.close[0]
                self._daily_close_prices[symbol].append(close)

    def get_current_price(self, symbol: str) -> Optional[float]:
        data = self._symbol_data_map.get(symbol)
        if data:
            price = data.close[0]
            # NaN 表示该日期尚无实际数据（前置填充行），返回 None
            if price != price:  # NaN != NaN is True
                return None
            return price
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
            # 调试日志：数据不足时输出
            if len(prices) < period and len(prices) > 0:
                import logging
                logging.getLogger(__name__).debug(
                    f'[DataAdapter] {symbol} 收盘价数量不足: 有{len(prices)}条, 需要{period}条'
                )
            return result
        return prices

    def get_current_date(self) -> Optional[datetime.date]:
        for symbol, data in self._symbol_data_map.items():
            return data.datetime.date(0)
        return None

    def get_current_datetime(self) -> Optional[datetime.datetime]:
        """获取当前完整的日期时间（含时分秒），分钟线/tick数据时有效"""
        for symbol, data in self._symbol_data_map.items():
            return data.datetime.datetime(0)
        return None

    def get_symbols(self) -> List[str]:
        return list(self._symbol_data_map.keys())


class LiveDataAdapter(MarketDataAdapter):
    """实时数据适配器 - 实盘/模拟盘模式下使用"""

    MAX_CLOSE_PRICES = 5000

    def __init__(self):
        self._close_prices: Dict[str, deque] = {}
        self._current_prices: Dict[str, float] = {}
        self._current_date: Optional[datetime.date] = None

    def load_history(self, symbol: str, close_prices: List[float]) -> None:
        """加载历史收盘价数据"""
        self._close_prices[symbol] = deque(close_prices, maxlen=self.MAX_CLOSE_PRICES)
        if close_prices:
            self._current_prices[symbol] = close_prices[-1]

    def update(self, data: Dict[str, Dict]) -> None:
        """更新实时数据"""
        for symbol, symbol_data in data.items():
            close = symbol_data['close'][-1]
            if symbol not in self._close_prices:
                self._close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
            self._close_prices[symbol].append(close)
            self._current_prices[symbol] = close

    def set_current_date(self, date: datetime.date) -> None:
        """设置当前日期"""
        self._current_date = date

    def get_current_price(self, symbol: str) -> Optional[float]:
        return self._current_prices.get(symbol)

    def get_close_prices(self, symbol: str, period: int = None) -> List[float]:
        prices = self._close_prices.get(symbol, [])
        if period is not None:
            return prices[-period:] if len(prices) >= period else list(prices)
        return list(prices)

    def get_current_date(self) -> Optional[datetime.date]:
        return self._current_date

    def get_symbols(self) -> List[str]:
        return list(self._close_prices.keys())
