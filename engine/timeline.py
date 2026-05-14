import datetime as dt_module
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine.data_feed import ArrayDataFeed


class Timeline:
    """统一时间轴 - 高效的多数据源时间对齐

    替代 Backtrader 的多数据源时间对齐机制。
    预计算统一时间轴和每个数据源的索引映射，
    将运行时对齐从 O(N*M) 优化为 O(N)。

    支持日线和分钟线两种粒度：
    - 日线模式：每个全局索引对应一个交易日
    - 分钟线模式：每个全局索引对应一根分钟K线
    """

    DAILY_PERIODS = {'1d', 'day', 'daily'}

    def __init__(self, data_feeds: List[ArrayDataFeed], period: str = '1d'):
        self._data_feeds = data_feeds
        self._feed_symbols = [df.symbol for df in data_feeds]
        self._period = period
        self._is_daily = period in self.DAILY_PERIODS

        self._timestamps: np.ndarray = np.array([], dtype='datetime64[ns]')
        self._ts_to_idx: Dict = {}
        self._feed_local_indices: List[np.ndarray] = []
        self._num_bars: int = 0

        self._build()

    def _build(self):
        all_ts_set = set()
        feed_date_arrays = []

        for feed in self._data_feeds:
            if feed.length == 0:
                feed_date_arrays.append(np.array([], dtype='datetime64[ns]'))
                continue
            dates = feed.dates
            feed_date_arrays.append(dates)
            for d in dates:
                ts = pd.Timestamp(d)
                if self._is_daily:
                    all_ts_set.add(ts.normalize())
                else:
                    all_ts_set.add(ts)

        sorted_ts = sorted(all_ts_set)
        self._timestamps = np.array(
            [ts.to_datetime64() for ts in sorted_ts],
            dtype='datetime64[ns]'
        )
        self._num_bars = len(sorted_ts)

        self._ts_to_idx = {}
        for i, ts in enumerate(sorted_ts):
            if self._is_daily:
                self._ts_to_idx[ts.date()] = i
            else:
                self._ts_to_idx[ts] = i

        self._feed_local_indices = []
        for feed_idx, feed in enumerate(self._data_feeds):
            if feed.length == 0:
                self._feed_local_indices.append(
                    np.full(self._num_bars, -1, dtype=np.int32)
                )
                continue

            local_indices = np.full(self._num_bars, -1, dtype=np.int32)
            feed_dates = feed_date_arrays[feed_idx]

            for local_idx in range(len(feed_dates)):
                ts = pd.Timestamp(feed_dates[local_idx])
                if self._is_daily:
                    key = ts.date()
                else:
                    key = ts
                global_idx = self._ts_to_idx.get(key)
                if global_idx is not None:
                    local_indices[global_idx] = local_idx

            self._feed_local_indices.append(local_indices)

    def get_trading_dates(self) -> np.ndarray:
        return self._timestamps

    def get_num_days(self) -> int:
        return self._num_bars

    def get_date(self, global_idx: int) -> Optional[dt_module.date]:
        if global_idx < 0 or global_idx >= self._num_bars:
            return None
        return pd.Timestamp(self._timestamps[global_idx]).date()

    def get_datetime(self, global_idx: int) -> Optional[dt_module.datetime]:
        if global_idx < 0 or global_idx >= self._num_bars:
            return None
        ts = pd.Timestamp(self._timestamps[global_idx])
        return ts.to_pydatetime()

    def get_feed_bar_index(self, feed_idx: int, global_idx: int) -> int:
        if feed_idx < 0 or feed_idx >= len(self._feed_local_indices):
            return -1
        if global_idx < 0 or global_idx >= self._num_bars:
            return -1
        return int(self._feed_local_indices[feed_idx][global_idx])

    def get_all_feed_bar_indices(self, global_idx: int) -> Dict[str, int]:
        result = {}
        for feed_idx, symbol in enumerate(self._feed_symbols):
            local_idx = self.get_feed_bar_index(feed_idx, global_idx)
            if local_idx >= 0:
                result[symbol] = local_idx
        return result

    def get_feed_index_by_symbol(self, symbol: str) -> int:
        try:
            return self._feed_symbols.index(symbol)
        except ValueError:
            return -1
