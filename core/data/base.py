import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

from core.cache import cache_manager


# ── 合并缓存 dict ↔ DataFrame 互转工具 ──

def _merged_dict_to_parquet(data: dict, mode: str = 'dividend') -> Optional[pd.DataFrame]:
    """将合并缓存 dict 转为单个 DataFrame 以便 parquet 存储

    Args:
        data: {stock: DataFrame} 或 {stock: {table: DataFrame}}
        mode: 'dividend' → {stock: DataFrame}; 'financial' → {stock: {table: DataFrame}}

    Returns:
        合并后的 DataFrame，带 _stock_code (和 _table_name) 列
    """
    _logger = logging.getLogger('_merged_dict_to_parquet')
    try:
        frames = []
        if mode == 'financial':
            # 按 table_name 分组拼接，避免不同表列名不同导致 concat 失败
            table_frames: Dict[str, list] = {}
            for stock, tables in data.items():
                if not isinstance(tables, dict):
                    continue
                for table_name, df in tables.items():
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    df = df.copy()
                    if isinstance(df.index, pd.DatetimeIndex):
                        # 检查索引名是否已经是普通列，避免重复
                        if df.index.name in df.columns:
                            df = df.reset_index(drop=True)
                        else:
                            df = df.reset_index()
                    elif df.index.name and df.index.name != 'index':
                        # 检查索引名是否已经是普通列，避免重复
                        if df.index.name in df.columns:
                            df = df.reset_index(drop=True)
                        else:
                            df = df.reset_index()
                    df['_stock_code'] = stock
                    df['_table_name'] = table_name
                    table_frames.setdefault(table_name, []).append(df)
            for table_name, tframes in table_frames.items():
                if tframes:
                    try:
                        frames.append(pd.concat(tframes, ignore_index=True))
                    except Exception as e:
                        _logger.warning(f"合并表 {table_name} 失败，跳过: {e}")
        else:  # dividend
            for stock, df in data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                df = df.copy()
                if isinstance(df.index, pd.DatetimeIndex):
                    # 检查索引名是否已经是普通列，避免重复
                    if df.index.name in df.columns:
                        df = df.reset_index(drop=True)
                    else:
                        df = df.reset_index()
                elif df.index.name and df.index.name != 'index':
                    # 检查索引名是否已经是普通列，避免重复
                    if df.index.name in df.columns:
                        df = df.reset_index(drop=True)
                    else:
                        df = df.reset_index()
                df['_stock_code'] = stock
                frames.append(df)
        if not frames:
            _logger.warning(f"无有效数据可合并 (mode={mode})")
            return None
        try:
            result = pd.concat(frames, ignore_index=True)
            _logger.info(f"合并缓存构建成功 (mode={mode}): {len(result)} 行, {len(result.columns)} 列")
            return result
        except Exception as e:
            _logger.warning(f"最终 concat 失败: {e}, 尝试逐表拼接")
            # 最后的 fallback：用 outer join 确保不丢失数据
            try:
                return pd.concat(frames, ignore_index=True, sort=False)
            except Exception as e2:
                _logger.error(f"合并缓存构建彻底失败: {e2}")
                return None
    except Exception as e:
        _logger.error(f"合并缓存构建异常: {e}")
        return None


def _parquet_to_merged_dict(df: pd.DataFrame, mode: str = 'dividend') -> dict:
    """将 parquet 读回的 DataFrame 还原为合并缓存 dict

    Args:
        df: 带 _stock_code (和 _table_name) 列的 DataFrame
        mode: 'dividend' 或 'financial'

    Returns:
        {stock: DataFrame} 或 {stock: {table: DataFrame}}
    """
    result = {}
    if mode == 'financial':
        for (stock, table_name), group in df.groupby(['_stock_code', '_table_name']):
            group = group.drop(columns=['_stock_code', '_table_name'], errors='ignore')
            # 保留 DatetimeIndex（如果存在名为 index 或原始索引列）
            group = _restore_dataframe_index(group)
            result.setdefault(stock, {})[table_name] = group
    else:  # dividend
        for stock, group in df.groupby('_stock_code'):
            group = group.drop(columns=['_stock_code'], errors='ignore')
            group = _restore_dataframe_index(group)
            result[stock] = group
    return result


def _restore_dataframe_index(df: pd.DataFrame) -> pd.DataFrame:
    """还原 DataFrame 的 DatetimeIndex

    parquet 存储时 DatetimeIndex 会变成普通列（如 'index', 'pubDate', 'statDate' 等），
    此函数尝试从常见日期列中恢复 DatetimeIndex。
    """
    # 优先使用公告日期列（避免未来数据），其次使用报告期列
    # QMT: m_anntime(公告日期), m_timetag(报告期) — 格式为 YYYYMMDD 字符串
    for col in ['index', 'announce_date', 'm_anntime', 'pubDate', '公告日期',
                 'report_date', 'm_timetag', 'statDate', '报告期']:
        if col in df.columns:
            try:
                dt_values = pd.to_datetime(df[col], errors='coerce')
                valid = dt_values.notna()
                if valid.any():
                    df = df[valid].copy()
                    df = df.drop(columns=[col])
                    df.index = dt_values[valid]
                    df = df.sort_index()
                    return df
            except Exception:
                continue

    # QMT 分红数据: time 列是毫秒时间戳
    if 'time' in df.columns and df['time'].dtype in ('float64', 'int64'):
        try:
            dt_values = pd.to_datetime(df['time'], unit='ms', errors='coerce')
            valid = dt_values.notna()
            if valid.any():
                df = df[valid].copy()
                df = df.drop(columns=['time'])
                df.index = dt_values[valid]
                df = df.sort_index()
                return df
        except Exception:
            pass

    return df.reset_index(drop=True)


class DataProcessor(ABC):
    """数据处理器基类"""
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """获取数据"""
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        critical_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in data.columns]
        data = data.dropna(subset=critical_cols)
        return data
