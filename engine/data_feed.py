import math
import datetime as dt_module
import logging
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from core.strategy_logic import BarData

logger = logging.getLogger(__name__)


class ArrayDataFeed:
    """基于 numpy 数组的高性能数据源

    替代 bt.feeds.PandasData，将 DataFrame 预加载为 numpy 数组，
    运行时通过索引直接访问，避免 Backtrader lines 对象的开销。
    """

    def __init__(self, symbol: str, dataframe: pd.DataFrame):
        self.symbol = symbol

        if dataframe.empty:
            self.dates = np.array([], dtype='datetime64[ns]')
            self.opens = np.array([], dtype=np.float64)
            self.highs = np.array([], dtype=np.float64)
            self.lows = np.array([], dtype=np.float64)
            self.closes = np.array([], dtype=np.float64)
            self.volumes = np.array([], dtype=np.float64)
            self.length = 0
            return

        if not isinstance(dataframe.index, pd.DatetimeIndex):
            if 'datetime' in dataframe.columns:
                dataframe = dataframe.set_index('datetime')
            else:
                dataframe.index = pd.to_datetime(dataframe.index)

        dataframe = dataframe.sort_index()

        self.dates = dataframe.index.values.astype('datetime64[ns]')
        self.opens = dataframe['open'].values.astype(np.float64)
        self.highs = dataframe['high'].values.astype(np.float64)
        self.lows = dataframe['low'].values.astype(np.float64)
        self.closes = dataframe['close'].values.astype(np.float64)
        self.volumes = dataframe['volume'].values.astype(np.float64)
        self.length = len(dataframe)

    def get_bar(self, idx: int) -> BarData:
        if idx < 0 or idx >= self.length:
            return BarData(symbol=self.symbol)

        close = self.closes[idx]
        if math.isnan(close):
            return BarData(symbol=self.symbol)

        bar_dt = self.get_datetime(idx)

        return BarData(
            symbol=self.symbol,
            open=float(self.opens[idx]),
            high=float(self.highs[idx]),
            low=float(self.lows[idx]),
            close=float(close),
            volume=float(self.volumes[idx]),
            datetime=bar_dt,
        )

    def get_close(self, idx: int) -> float:
        if idx < 0 or idx >= self.length:
            return float('nan')
        return float(self.closes[idx])

    def get_open(self, idx: int) -> float:
        if idx < 0 or idx >= self.length:
            return float('nan')
        return float(self.opens[idx])

    def get_high(self, idx: int) -> float:
        if idx < 0 or idx >= self.length:
            return float('nan')
        return float(self.highs[idx])

    def get_low(self, idx: int) -> float:
        if idx < 0 or idx >= self.length:
            return float('nan')
        return float(self.lows[idx])

    def get_volume(self, idx: int) -> float:
        if idx < 0 or idx >= self.length:
            return float('nan')
        return float(self.volumes[idx])

    def get_date(self, idx: int) -> Optional[dt_module.date]:
        if idx < 0 or idx >= self.length:
            return None
        ts = pd.Timestamp(self.dates[idx])
        return ts.date()

    def get_datetime(self, idx: int) -> Optional[dt_module.datetime]:
        if idx < 0 or idx >= self.length:
            return None
        ts = pd.Timestamp(self.dates[idx])
        return ts.to_pydatetime()

    def is_nan(self, idx: int) -> bool:
        if idx < 0 or idx >= self.length:
            return True
        return math.isnan(self.closes[idx])

    def is_volume_zero_or_nan(self, idx: int) -> bool:
        if idx < 0 or idx >= self.length:
            return True
        vol = self.volumes[idx]
        if math.isnan(vol):
            return True
        return vol == 0

    def __len__(self) -> int:
        return self.length

    def to_dataframe(self) -> pd.DataFrame:
        """将ArrayDataFeed转回DataFrame（供LazyDataFeed使用）"""
        if self.length == 0:
            return pd.DataFrame()
        return pd.DataFrame({
            'open': self.opens,
            'high': self.highs,
            'low': self.lows,
            'close': self.closes,
            'volume': self.volumes,
        }, index=pd.DatetimeIndex(self.dates))


class LazyDataFeed:
    """按需加载的数据源

    与ArrayDataFeed接口兼容，但数据不预加载到内存。
    复用QMTDataProcessor的smart_cache三级缓存机制：
    内存缓存 → 磁盘parquet(.cache/) → QMT API下载

    日线模式：首次访问时从smart_cache磁盘缓存读取整年数据，缓存到_cache
    分钟线模式：按日加载，缓存到_minute_cache[date_str]，每日清空
    """

    def __init__(self, symbol: str, data_processor, period: str = '1d',
                 start_date: str = '', end_date: str = ''):
        self.symbol = symbol
        self._data_processor = data_processor
        self._period = period
        self._start_date = start_date
        self._end_date = end_date
        self._cache: Optional[ArrayDataFeed] = None
        self._minute_cache: Dict[str, ArrayDataFeed] = {}
        self._date_index: Optional[pd.DatetimeIndex] = None
        self._daily_df: Optional[pd.DataFrame] = None  # 保留原始DataFrame供get_daily_df使用

    def _ensure_daily_loaded(self):
        """确保日线数据已加载（从smart_cache磁盘缓存读取）"""
        if self._cache is not None:
            return
        try:
            df = self._data_processor.get_data(
                self.symbol, self._start_date, self._end_date, period='1d'
            )
            if df is not None and not df.empty:
                self._cache = ArrayDataFeed(self.symbol, df)
                self._date_index = df.index if isinstance(df.index, pd.DatetimeIndex) else None
                self._daily_df = df
        except Exception as e:
            logger.debug(f"LazyDataFeed加载{self.symbol}日线数据失败: {e}")

    def _ensure_minute_loaded(self, date_str: str):
        """确保指定日期的1m数据已加载"""
        if date_str in self._minute_cache:
            return
        try:
            date = dt_module.datetime.strptime(date_str, '%Y-%m-%d')
            start = (date - dt_module.timedelta(days=1)).strftime('%Y-%m-%d')
            end = (date + dt_module.timedelta(days=1)).strftime('%Y-%m-%d')
            df = self._data_processor.get_data(
                self.symbol, start, end, period='1m'
            )
            if df is not None and not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    target_date = pd.Timestamp(date_str).date()
                    df = df[df.index.date == target_date]
                if not df.empty:
                    self._minute_cache[date_str] = ArrayDataFeed(self.symbol, df)
        except Exception as e:
            logger.debug(f"LazyDataFeed加载{self.symbol} {date_str} 1m数据失败: {e}")

    def clear_minute_cache(self):
        """清空分钟线缓存（每日调用，释放内存）"""
        self._minute_cache.clear()

    def _find_idx_by_date(self, date_str: str) -> int:
        """根据日期字符串找到对应的索引"""
        if self._date_index is None:
            return -1
        ts = pd.Timestamp(date_str)
        mask = self._date_index == ts
        if mask.any():
            return int(mask.argmax())
        return -1

    # ---- 与ArrayDataFeed兼容的接口（按索引访问） ----

    @property
    def length(self) -> int:
        self._ensure_daily_loaded()
        return self._cache.length if self._cache else 0

    def __len__(self) -> int:
        return self.length

    def get_bar(self, idx: int) -> BarData:
        self._ensure_daily_loaded()
        if self._cache is None:
            return BarData(symbol=self.symbol)
        return self._cache.get_bar(idx)

    def get_close(self, idx: int) -> float:
        self._ensure_daily_loaded()
        if self._cache is None:
            return float('nan')
        return self._cache.get_close(idx)

    def get_open(self, idx: int) -> float:
        self._ensure_daily_loaded()
        if self._cache is None:
            return float('nan')
        return self._cache.get_open(idx)

    def get_high(self, idx: int) -> float:
        self._ensure_daily_loaded()
        if self._cache is None:
            return float('nan')
        return self._cache.get_high(idx)

    def get_low(self, idx: int) -> float:
        self._ensure_daily_loaded()
        if self._cache is None:
            return float('nan')
        return self._cache.get_low(idx)

    def get_volume(self, idx: int) -> float:
        self._ensure_daily_loaded()
        if self._cache is None:
            return float('nan')
        return self._cache.get_volume(idx)

    def get_date(self, idx: int) -> Optional[dt_module.date]:
        self._ensure_daily_loaded()
        if self._cache is None:
            return None
        return self._cache.get_date(idx)

    def get_datetime(self, idx: int) -> Optional[dt_module.datetime]:
        self._ensure_daily_loaded()
        if self._cache is None:
            return None
        return self._cache.get_datetime(idx)

    def is_nan(self, idx: int) -> bool:
        self._ensure_daily_loaded()
        if self._cache is None:
            return True
        return self._cache.is_nan(idx)

    def is_volume_zero_or_nan(self, idx: int) -> bool:
        self._ensure_daily_loaded()
        if self._cache is None:
            return True
        return self._cache.is_volume_zero_or_nan(idx)

    # ---- 按日期访问的扩展接口 ----

    def get_close_by_date(self, date_str: str) -> Optional[float]:
        """获取指定日期的收盘价"""
        self._ensure_daily_loaded()
        if self._cache is None:
            return None
        idx = self._find_idx_by_date(date_str)
        if idx >= 0:
            close = self._cache.get_close(idx)
            return close if not math.isnan(close) else None
        return None

    def get_bar_by_date(self, date_str: str) -> Optional[BarData]:
        """获取指定日期的BarData"""
        self._ensure_daily_loaded()
        if self._cache is None:
            return None
        idx = self._find_idx_by_date(date_str)
        if idx >= 0:
            return self._cache.get_bar(idx)
        return None

    def is_suspended(self, date_str: str) -> bool:
        """判断是否停牌（成交量为0或无数据）"""
        self._ensure_daily_loaded()
        if self._cache is None:
            return True
        idx = self._find_idx_by_date(date_str)
        if idx >= 0:
            return self._cache.is_volume_zero_or_nan(idx)
        return True  # 无当日数据视为停牌

    def get_prev_close(self, date_str: str) -> Optional[float]:
        """获取前一日收盘价"""
        self._ensure_daily_loaded()
        if self._cache is None or self._date_index is None:
            return None
        idx = self._find_idx_by_date(date_str)
        if idx > 0:
            close = self._cache.get_close(idx - 1)
            return close if not math.isnan(close) else None
        return None

    def get_daily_df(self, end_date: str = None, n_days: int = 60) -> Optional[pd.DataFrame]:
        """获取日线DataFrame（供策略计算均线等指标）"""
        self._ensure_daily_loaded()
        if self._daily_df is None:
            return None
        df = self._daily_df.copy()
        if end_date and isinstance(df.index, pd.DatetimeIndex):
            cutoff = pd.Timestamp(end_date)
            df = df[df.index <= cutoff]
        if n_days and len(df) > n_days:
            df = df.iloc[-n_days:]
        return df

    def get_minute_df(self, date_str: str) -> Optional[pd.DataFrame]:
        """获取指定日期的1m DataFrame"""
        self._ensure_minute_loaded(date_str)
        feed = self._minute_cache.get(date_str)
        if feed:
            return feed.to_dataframe()
        return None
