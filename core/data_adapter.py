import datetime
import math
import time
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Any


def get_limit_ratio(symbol: str) -> float:
    code = symbol.split('.')[0] if '.' in symbol else symbol
    if code.startswith(('300', '301')):
        return 0.20
    if code.startswith('688'):
        return 0.20
    if code.startswith(('4', '8')):
        return 0.30
    return 0.10


def get_trade_unit(symbol: str) -> int:
    """获取股票的最小交易单位（股）
    
    A 股交易单位规则（2025年后）：
    - 科创板（688开头）：200股起
    - 其他（主板、创业板、北交所）：100股起
    """
    code = symbol.split('.')[0] if '.' in symbol else symbol
    if code.startswith('688'):
        return 200
    return 100


def validate_trade_volume(symbol: str, volume: int) -> tuple[bool, str]:
    """校验交易数量是否符合规则
    
    A 股交易数量规则：
    - 科创板（688开头）：>= 200股，超过后可1股1股递增（无需整数倍）
    - 其他：>= 100股，且为100股整数倍
    
    Returns:
        (is_valid, error_message)
    """
    code = symbol.split('.')[0] if '.' in symbol else symbol
    
    if volume <= 0:
        return False, "交易数量必须大于0"
    
    if code.startswith('688'):
        # 科创板：≥200股
        if volume < 200:
            return False, "科创板股票最少买200股"
        return True, ""
    else:
        # 其他：≥100股且为100股整数倍
        if volume < 100:
            return False, "最少买100股"
        if volume % 100 != 0:
            return False, f"数量必须为100股整数倍，当前: {volume}"
        return True, ""


def validate_stock_code(symbol: str) -> bool:
    """校验股票代码格式是否合法
    
    合法格式示例：
    - 000001.SZ
    - 600000.SH
    - 300001.SZ
    - 688001.SH
    """
    if not symbol or '.' not in symbol:
        return False
    try:
        code, exchange = symbol.split('.', 1)
        if exchange not in ('SZ', 'SH', 'BJ'):
            return False
        if not code.isdigit():
            return False
        if len(code) not in (6, 8):
            return False
        return True
    except Exception:
        return False


class MarketDataAdapter(ABC):
    """市场数据适配器基类 - 提供统一的数据访问接口"""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        pass

    def get_open_price(self, symbol: str) -> Optional[float]:
        """获取当前bar的开盘价，默认返回None，子类可覆盖"""
        return None

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

    def is_suspended(self, symbol: str) -> bool:
        return False

    def is_limit_up(self, symbol: str) -> bool:
        return False

    def is_limit_down(self, symbol: str) -> bool:
        return False

    def get_ohlcv_data(self, symbol: str, period: int = None) -> List[Dict[str, float]]:
        """获取OHLCV数据序列

        Args:
            symbol: 标的代码
            period: 获取的周期数，None表示全部

        Returns:
            [{'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}, ...]
        """
        return []

    def get_return_over_days(self, symbol: str, num_days: int) -> Optional[Dict[str, Any]]:
        """基于统一时间轴计算N个交易日的收益率

        确保不同数据源对比的是同一个日历日期的收盘价。
        默认实现使用 get_close_prices 的行偏移，子类可覆盖以实现日期对齐。

        Args:
            symbol: 标的代码
            num_days: 交易日数

        Returns:
            {'rate': 收益率, 'start_price': 起始价, 'end_price': 结束价, 'start_date': 起始日期}
            数据不足时返回 None
        """
        close_prices = self.get_close_prices(symbol)
        if len(close_prices) <= num_days:
            return None
        start_price = close_prices[-(num_days + 1)]
        end_price = close_prices[-1]
        return {
            'rate': (end_price - start_price) / start_price,
            'start_price': start_price,
            'end_price': end_price,
            'start_date': None,
        }

    def get_close_prices_for_days(self, symbol: str, num_days: int) -> List[float]:
        """基于统一时间轴获取最近N个交易日的收盘价序列

        确保不同数据源返回的是同一组日历日期的收盘价。
        默认实现使用 get_close_prices 的行偏移，子类可覆盖以实现日期对齐。

        Args:
            symbol: 标的代码
            num_days: 交易日数

        Returns:
            收盘价列表，长度为 num_days + 1（包含当前日）
        """
        prices = self.get_close_prices(symbol)
        if len(prices) <= num_days:
            return prices
        return prices[-(num_days + 1):]


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
        self._ohlcv_data: Dict[str, deque] = {}

    def register_data(self, symbol: str, data_feed) -> None:
        """注册标的与数据源的映射"""
        self._symbol_data_map[symbol] = data_feed
        self._close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
        self._daily_close_prices[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)
        self._last_daily_date[symbol] = None
        self._current_day_close[symbol] = None
        self._ohlcv_data[symbol] = deque(maxlen=self.MAX_CLOSE_PRICES)

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

            if math.isnan(close):
                continue

            self._close_prices[symbol].append(close)

            if self._is_daily():
                self._daily_close_prices[symbol].append(close)
                o = data_feed.open[0]
                h = data_feed.high[0]
                l = data_feed.low[0]
                v = data_feed.volume[0]
                if not (math.isnan(o) or math.isnan(h) or math.isnan(l)):
                    self._ohlcv_data[symbol].append({
                        'open': o,
                        'high': h,
                        'low': l,
                        'close': close,
                        'volume': 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v),
                    })
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

    def get_open_price(self, symbol: str) -> Optional[float]:
        """获取当前bar的开盘价"""
        data = self._symbol_data_map.get(symbol)
        if data:
            price = data.open[0]
            if price != price:  # NaN check
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
        best_date = None
        for symbol, data in self._symbol_data_map.items():
            try:
                if len(data) > 0:
                    d = data.datetime.date(0)
                    if best_date is None or d > best_date:
                        best_date = d
            except Exception:
                continue
        return best_date

    def get_current_datetime(self) -> Optional[datetime.datetime]:
        best_dt = None
        for symbol, data in self._symbol_data_map.items():
            try:
                if len(data) > 0:
                    d = data.datetime.datetime(0)
                    if best_dt is None or d > best_dt:
                        best_dt = d
            except Exception:
                continue
        if best_dt is not None and isinstance(best_dt, datetime.datetime) and best_dt.hour == 0 and best_dt.minute == 0:
            best_dt = best_dt.replace(hour=15, minute=0)
        return best_dt

    def get_symbols(self) -> List[str]:
        return list(self._symbol_data_map.keys())

    def is_suspended(self, symbol: str) -> bool:
        data = self._symbol_data_map.get(symbol)
        if data is None:
            return True
        try:
            vol = data.volume[0]
            if isinstance(vol, float) and math.isnan(vol):
                return True
            return float(vol) == 0
        except (AttributeError, IndexError):
            return True

    def is_limit_up(self, symbol: str) -> bool:
        data = self._symbol_data_map.get(symbol)
        if data is None:
            return False
        try:
            close = data.close[0]
            if math.isnan(close):
                return False
            prev_close = data.close[-1]
            if math.isnan(prev_close) or prev_close <= 0:
                return False
        except (AttributeError, IndexError):
            return False
        limit_ratio = get_limit_ratio(symbol)
        limit_price = round(prev_close * (1 + limit_ratio), 2)
        return close >= limit_price - 0.005

    def is_limit_down(self, symbol: str) -> bool:
        data = self._symbol_data_map.get(symbol)
        if data is None:
            return False
        try:
            close = data.close[0]
            if math.isnan(close):
                return False
            prev_close = data.close[-1]
            if math.isnan(prev_close) or prev_close <= 0:
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


class QMTLiveDataAdapter(MarketDataAdapter):
    """QMT实盘数据适配器 - 通过 get_full_tick / get_market_data_ex 按需获取

    与 LiveDataAdapter 的区别：
    - LiveDataAdapter: 依赖 subscribe_quote 推送缓存，受100只订阅限制
    - QMTLiveDataAdapter: 按需调用 get_full_tick / get_market_data_ex，不受订阅限制

    数据获取策略：
    - get_current_price: 调用 get_full_tick 获取最新快照价
    - get_close_prices: 调用 get_market_data_ex 获取历史K线
    - 持仓股价格: 优先从 subscribe_quote 推送缓存读取（实时性更好）
    """

    def __init__(self, data_processor=None):
        self._data_processor = data_processor
        self._current_date: Optional[datetime.date] = None
        self._subscribed_prices: Dict[str, float] = {}  # 持仓股订阅价格缓存
        self._kline_cache: Dict[str, List[float]] = {}  # K线缓存（调仓日有效）
        self._tick_cache: Dict[str, float] = {}         # get_full_tick 缓存
        self._tick_cache_time: float = 0                 # 缓存时间戳
        self._tick_cache_ttl: float = 3.0                # 缓存有效期（秒）
        self._logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def set_subscribed_price(self, symbol: str, price: float):
        """更新持仓股的订阅推送价格"""
        self._subscribed_prices[symbol] = price

    def remove_subscribed_price(self, symbol: str):
        """移除不再持仓的订阅价格"""
        self._subscribed_prices.pop(symbol, None)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格

        优先级：
        1. 持仓股 → 从 subscribe_quote 推送缓存读取（实时性最好）
        2. 非持仓股 → 调用 get_full_tick 获取（有3秒缓存）
        """
        if symbol in self._subscribed_prices:
            return self._subscribed_prices[symbol]
        return self._get_tick_price(symbol)

    def _get_tick_price(self, symbol: str) -> Optional[float]:
        """通过 get_full_tick 获取最新价（带缓存）

        如果 symbol 已在缓存中，检查 TTL 后刷新整个缓存；
        如果 symbol 不在缓存中，单独获取该标的。
        """
        if symbol in self._tick_cache:
            now = time.time()
            if now - self._tick_cache_time > self._tick_cache_ttl:
                self._refresh_tick_cache()
            return self._tick_cache.get(symbol)

        return self._fetch_single_tick(symbol)

    def _fetch_single_tick(self, symbol: str) -> Optional[float]:
        """单独获取一只股票的最新价"""
        try:
            from xtquant import xtdata
            tick_data = xtdata.get_full_tick([symbol])
            if tick_data and symbol in tick_data:
                price = self._parse_tick_price(tick_data[symbol])
                if price and price > 0:
                    self._tick_cache[symbol] = price
                    return price
        except Exception as e:
            self._logger.warning(f"获取{symbol} tick价格失败: {e}")
        return None

    def _refresh_tick_cache(self):
        """刷新全推快照缓存"""
        try:
            from xtquant import xtdata
            symbols_to_refresh = list(self._tick_cache.keys())
            if not symbols_to_refresh:
                self._tick_cache_time = time.time()
                return
            tick_data = xtdata.get_full_tick(symbols_to_refresh)
            if tick_data:
                for code, data in tick_data.items():
                    price = self._parse_tick_price(data)
                    if price and price > 0:
                        self._tick_cache[code] = price
            self._tick_cache_time = time.time()
        except Exception as e:
            self._logger.warning(f"刷新tick缓存失败: {e}")

    @staticmethod
    def _parse_tick_price(data) -> float:
        """解析tick数据中的最新价"""
        if isinstance(data, dict):
            return data.get('lastPrice', 0)
        return getattr(data, 'lastPrice', 0)

    def get_close_prices(self, symbol: str, period: int = None) -> List[float]:
        """获取历史收盘价序列

        通过 download_history_data + get_market_data_ex 按需获取，
        不受订阅数量限制。
        """
        if symbol not in self._kline_cache:
            prices = self._download_kline(symbol)
            self._kline_cache[symbol] = prices
        else:
            prices = self._kline_cache[symbol]

        if period is not None:
            return prices[-period:] if len(prices) >= period else prices
        return prices

    def _download_kline(self, symbol: str) -> List[float]:
        """下载并获取K线收盘价"""
        if not self._data_processor:
            return []
        try:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=120)).strftime('%Y-%m-%d')
            data = self._data_processor.get_data(symbol, start_date, end_date)
            if data is not None and not data.empty and 'close' in data.columns:
                return data['close'].tolist()
        except Exception as e:
            self._logger.warning(f"下载K线数据失败 {symbol}: {e}")
        return []

    def get_current_date(self) -> Optional[datetime.date]:
        return self._current_date

    def set_current_date(self, date: datetime.date):
        """设置当前日期"""
        self._current_date = date

    def get_symbols(self) -> List[str]:
        return list(set(list(self._subscribed_prices.keys()) + list(self._tick_cache.keys())))

    def invalidate_kline_cache(self):
        """清空K线缓存（调仓日重新下载）"""
        self._kline_cache.clear()

    def preload_tick_data(self, stock_list: List[str], batch_size: int = 100):
        """预加载股票池的快照数据（调仓日选股前调用）

        基于实测数据：
        - 100只/批: ~0.04秒
        - 1000只: ~0.4秒
        - 分批获取避免单批次过大导致阻塞
        """
        try:
            from xtquant import xtdata

            total = len(stock_list)
            for i in range(0, total, batch_size):
                batch = stock_list[i:i + batch_size]
                tick_data = xtdata.get_full_tick(batch)
                if tick_data:
                    for code, data in tick_data.items():
                        price = self._parse_tick_price(data)
                        if price and price > 0:
                            self._tick_cache[code] = price
                # 批次间短暂停顿，避免触发限制
                if i + batch_size < total:
                    time.sleep(0.05)

            self._tick_cache_time = time.time()
            self._logger.info(f"预加载完成: {len(self._tick_cache)} 只股票快照数据")
        except Exception as e:
            self._logger.warning(f"预加载tick数据失败: {e}")

    def get_ohlcv_data(self, symbol: str, period: int = None) -> List[Dict[str, float]]:
        if not self._data_processor:
            return []
        try:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            lookback = max(period or 60, 60) + 10
            start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback)).strftime('%Y-%m-%d')
            data = self._data_processor.get_data(symbol, start_date, end_date)
            if data is None or data.empty:
                return []
            result = []
            for _, row in data.iterrows():
                result.append({
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row.get('volume', 0)),
                })
            if period is not None:
                return result[-period:] if len(result) >= period else result
            return result
        except Exception as e:
            self._logger.warning(f"获取OHLCV数据失败 {symbol}: {e}")
            return []
