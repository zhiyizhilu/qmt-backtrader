"""富途数据处理器 - 从本地 .cache/FutuData 目录读取预存的行情数据

数据来源: 富途OpenD API (futu-api)，通过 save_futu_data.py 预先下载到本地。
存储格式: .cache/FutuData/market/{symbol}/{year}_{period}.parquet
          .cache/FutuData/market_raw/{symbol}/{year}_{period}.parquet
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Optional

from core.data.base import DataProcessor
from core.cache import cache_manager


class FutuDataProcessor(DataProcessor):
    """富途数据处理器

    从本地 .cache/FutuData 目录读取预存的行情数据（parquet格式）。
    数据需要通过 save_futu_data.py 预先下载。

    支持后复权（market/）和不复权（market_raw/）两种数据。
    """

    def __init__(self, fallback_to_simulated: bool = False, data_dir: str = ''):
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        if data_dir:
            self._data_dir = data_dir
        else:
            base = os.environ.get('QMT_CACHE_DIR', os.path.join(os.getcwd(), '.cache'))
            self._data_dir = os.path.join(base, 'FutuData')

        self._raw_fetcher = FutuDataProcessor_Raw(self)

    def get_data(self, symbol: str, start_date: str, end_date: str,
                 period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取后复权行情数据

        从 .cache/FutuData/market/{symbol}/ 目录按年份读取parquet文件并合并。

        Args:
            symbol: 股票代码，QMT格式如 '601398.SH'
            start_date: 起始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            period: 周期 '1d', '1m' 等
        """
        df = self._load_from_parquet(symbol, start_date, end_date, period, sub_dir='market')

        if df is not None and not df.empty:
            return df

        raise ValueError(f"{symbol} 在 FutuData 中没有 {start_date}~{end_date} 的数据")

    def get_raw_data(self, symbol: str, start_date: str, end_date: str,
                     period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取不复权行情数据"""
        return self._raw_fetcher.get_data(symbol, start_date, end_date, period, **kwargs)

    def _load_from_parquet(self, symbol: str, start_date: str, end_date: str,
                           period: str, sub_dir: str = 'market') -> Optional[pd.DataFrame]:
        """从parquet文件加载数据

        目录结构: {data_dir}/{sub_dir}/{symbol}/{year}_{period}.parquet
        """
        symbol_dir = os.path.join(self._data_dir, sub_dir, symbol)
        if not os.path.isdir(symbol_dir):
            self.logger.debug(f"目录不存在: {symbol_dir}")
            return None

        try:
            start_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year
        except Exception:
            return None

        period_suffix = self._map_period(period)

        dfs = []
        for year in range(start_year, end_year + 1):
            file_path = os.path.join(symbol_dir, f"{year}_{period_suffix}.parquet")
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    if df is not None and not df.empty:
                        dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"读取失败: {file_path}, {e}")

        if not dfs:
            return None

        df = pd.concat(dfs)

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                self.logger.warning(f"无法解析日期索引: {symbol}")
                return None

        df = df.sort_index()

        df = df[(df.index >= start_date) & (df.index <= end_date)]

        if df.empty:
            return None

        keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        if keep_cols:
            df = df[keep_cols]

        return self.preprocess_data(df)

    @staticmethod
    def _map_period(period: str) -> str:
        """映射周期格式: '1d' -> '1d', '1m' -> '1m' 等"""
        mapping = {
            '1d': '1d', 'day': '1d', 'daily': '1d',
            '1m': '1m', '1min': '1m', 'minute': '1m',
            '5m': '5m', '5min': '5m',
            '15m': '15m', '15min': '15m',
            '30m': '30m', '30min': '30m',
            '60m': '60m', '60min': '60m',
        }
        return mapping.get(period, period)

    def get_stock_list(self, sector: str = '沪深A股') -> list:
        """FutuData不支持获取股票列表，返回空列表"""
        self.logger.warning("FutuDataProcessor 不支持获取股票列表")
        return []

    def get_industry_mapping(self, level: int = 1, stock_pool=None) -> dict:
        """FutuData不支持获取行业映射"""
        self.logger.warning("FutuDataProcessor 不支持获取行业映射")
        return {}


class FutuDataProcessor_Raw:
    """不复权行情数据获取器（FutuData 数据源）"""

    def __init__(self, processor: FutuDataProcessor):
        self._processor = processor

    def get_data(self, symbol: str, start_date: str, end_date: str,
                 period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取不复权行情数据"""
        df = self._processor._load_from_parquet(
            symbol, start_date, end_date, period, sub_dir='market_raw'
        )
        if df is not None and not df.empty:
            return df

        raise ValueError(f"{symbol} 在 FutuData 中没有不复权数据 ({start_date}~{end_date})")
