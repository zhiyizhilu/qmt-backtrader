import math
import datetime as dt_module
from typing import Optional

import numpy as np
import pandas as pd

from core.strategy_logic import BarData


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
