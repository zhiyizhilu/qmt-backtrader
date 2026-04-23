import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
import time
import hashlib
from datetime import datetime
from tqdm import tqdm
from core.cache import smart_cache, cache_manager


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
                        df = df.reset_index()
                    elif df.index.name and df.index.name != 'index':
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
                    df = df.reset_index()
                elif df.index.name and df.index.name != 'index':
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


class QMTDataProcessor(DataProcessor):
    """QMT数据处理器
    
    当QMT不可用时，自动降级为模拟数据，方便策略测试。
    可通过 fallback_to_simulated=False 禁用降级行为。
    """
    
    FINANCIAL_TABLES = [
        'Balance', 'Income', 'CashFlow', 'Capital',
        'Holdernum', 'Top10holder', 'Top10flowholder', 'Pershareindex',
    ]

    def __init__(self, fallback_to_simulated: bool = False):
        try:
            from xtquant import xtdata
            self.xtdata = xtdata
        except ImportError:
            self.xtdata = None
            logging.getLogger(__name__).warning("xtquant not installed, using simulated data")
        self._fallback_to_simulated = fallback_to_simulated
        self._financial_data_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    def check_connection(self) -> bool:
        """检查QMT数据服务是否可用"""
        if not self.xtdata:
            return False
        try:
            self.xtdata.get_financial_index('000001.SZ')
            return True
        except Exception:
            return False

    @smart_cache(cache_type='market', incremental=True)
    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """从QMT获取数据，QMT不可用时降级为模拟数据"""
        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning(f"xtquant 未安装，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        try:
            start_time = start_date.replace('-', '')
            end_time = end_date.replace('-', '')

            self.logger.debug(f"正在下载 {symbol} 行情数据 ({start_date} ~ {end_date}, {period})...")
            self.xtdata.download_history_data(stock_code=symbol, period=period, start_time='', end_time='', incrementally=True)
            self.logger.debug(f"{symbol} 行情数据下载完成，正在读取...")

            history_data = self.xtdata.get_market_data_ex(
                [], [symbol], period=period,
                start_time=start_time, end_time=end_time,
                count=-1, dividend_type="front"  # 前复权，与akshare/baostock保持一致
            )

            if symbol in history_data:
                df = history_data[symbol]
                
                # Check if data is empty
                if df.empty:
                    if self._fallback_to_simulated:
                        self.logger.warning(f"{symbol} 数据为空，使用模拟数据")
                        return self._generate_simulated_data(start_date, end_date, symbol)
                    raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据，请检查QMT数据是否同步")

                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S')
                    except (ValueError, TypeError):
                        try:
                            df.index = pd.to_datetime(df.index, format='%Y%m%d')
                        except (ValueError, TypeError):
                            if self._fallback_to_simulated:
                                self.logger.warning(f"{symbol} 索引转换失败，使用模拟数据")
                                return self._generate_simulated_data(start_date, end_date, symbol)
                            raise ValueError(f"{symbol} 索引转换失败")

                df = df[(df.index >= start_date) & (df.index <= end_date)]

                if df.empty:
                    if self._fallback_to_simulated:
                        self.logger.warning(f"{symbol} 数据为空，使用模拟数据")
                        return self._generate_simulated_data(start_date, end_date, symbol)
                    raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据，请检查QMT数据是否同步")

                return self.preprocess_data(df)
            else:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} 未在返回数据中，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 未在返回数据中，请检查QMT数据是否同步")
        except RuntimeError:
            raise
        except ValueError:
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"获取QMT数据失败: {e}，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError(f"获取QMT数据失败: {e}，请检查QMT是否已启动并登录")
    
    def _normalize_qmt_financial_df(self, df: pd.DataFrame,
                                      report_type: str = 'announce_time') -> pd.DataFrame:
        """将 QMT xtdata 返回的财务 DataFrame 标准化

        QMT xtdata 返回的财务数据索引是 RangeIndex，日期信息存储在列中：
        - m_anntime: 公告日期 (YYYYMMDD 字符串)
        - m_timetag: 报告期 (YYYYMMDD 字符串)

        此方法将其转换为 DatetimeIndex，便于回测按日期过滤。
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # 选择日期列：优先使用公告日期避免未来数据
        if report_type == 'announce_time' and 'm_anntime' in df.columns:
            date_col = 'm_anntime'
        elif 'm_timetag' in df.columns:
            date_col = 'm_timetag'
        else:
            # 没有已知的日期列，返回原数据
            return df

        try:
            dt_values = pd.to_datetime(df[date_col], errors='coerce')
            valid = dt_values.notna()
            if valid.any():
                df = df[valid].copy()
                df = df.drop(columns=[date_col])
                df.index = dt_values[valid]
                df = df.sort_index()
        except Exception:
            pass

        return df

    def _normalize_qmt_dividend_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """将 QMT xtdata 返回的分红 DataFrame 标准化

        QMT xtdata 返回的分红数据 time 列是毫秒时间戳，
        索引是 RangeIndex。此方法将其转换为 DatetimeIndex。
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        if 'time' in df.columns and df['time'].dtype in ('float64', 'int64'):
            try:
                dt_values = pd.to_datetime(df['time'], unit='ms', errors='coerce')
                valid = dt_values.notna()
                if valid.any():
                    df = df[valid].copy()
                    df = df.drop(columns=['time'])
                    df.index = dt_values[valid]
                    df = df.sort_index()
            except Exception:
                pass

        return df

    def download_financial_data(self, stock_list: List[str],
                                table_list: Optional[List[str]] = None,
                                start_time: str = '', end_time: str = '') -> None:
        """下载财务数据到本地缓存

        Args:
            stock_list: 股票代码列表，如 ['000001.SZ', '600000.SH']
            table_list: 财务报表列表，如 ['Balance', 'Income']，为空则下载全部
            start_time: 起始时间，如 '20230101'
            end_time: 结束时间，如 '20240101'
        """
        import time

        if not self.xtdata:
            self.logger.warning("xtquant 未安装，无法下载财务数据")
            return

        tables = table_list or self.FINANCIAL_TABLES
        total = len(stock_list)

        try:
            start_ts = time.time()
            if total <= 5:
                for i, stock in enumerate(stock_list, 1):
                    self.logger.info(f"[ {i} / {total} ] 正在下载 {stock} 财务数据...")
                    self.xtdata.download_financial_data(stock, tables)
                    self.logger.info(f"[ {i} / {total} ] {stock} 财务数据下载完成")
            else:
                # 大批量下载：分批处理，每批50只，避免QMT超时
                batch_size = 50
                for batch_start in range(0, total, batch_size):
                    batch_end = min(batch_start + batch_size, total)
                    batch = stock_list[batch_start:batch_end]
                    self.logger.info(
                        f"[ {batch_start + 1}~{batch_end} / {total} ] "
                        f"正在下载财务数据 (表: {', '.join(tables)})..."
                    )
                    self.xtdata.download_financial_data2(
                        batch, tables,
                        start_time=start_time, end_time=end_time,
                    )
                    self.logger.info(
                        f"[ {batch_start + 1}~{batch_end} / {total} ] "
                        f"财务数据下载完成"
                    )
            elapsed = time.time() - start_ts
            self.logger.info(
                f"财务数据下载完成: {total} 只股票, {', '.join(tables)}, "
                f"耗时 {elapsed:.1f}秒"
            )
        except Exception as e:
            self.logger.error(f"财务数据下载失败: {e}")

    def _try_load_merged_financial_cache(self, stock_list: List[str],
                                          table_list: Optional[List[str]] = None,
                                          start_time: str = '', end_time: str = '',
                                          report_type: str = 'announce_time') -> Optional[Dict[str, Any]]:
        """尝试从合并缓存加载财务数据，命中则返回数据，未命中返回 None

        此方法用于在外部（如 backtest_api）判断是否可以跳过下载阶段。
        只检查合并缓存（单个文件快速加载），不做逐只缓存检查。
        """
        import hashlib as _hashlib
        namespace = 'QMTDataProcessor_Financial'
        sorted_stocks = sorted(stock_list)
        stocks_hash = _hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]

        # 1. 精确匹配
        merged_cache_key = f"merged_{stocks_hash}_{start_time}_{end_time}_{report_type}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='financial')
            if merged_cached and len(merged_cached) >= len(stock_list):
                self.logger.info(f"合并缓存(精确)命中: {len(merged_cached)} 只股票")
                return merged_cached

        # 2. 子集匹配
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.cache_dir / namespace
        if ns_dir.exists():
            for cache_file in ns_dir.glob('merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='financial')
                                result_subset = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if len(result_subset) == len(stock_list):
                                    self.logger.info(
                                        f"合并缓存(子集)命中: 缓存{len(cached_stocks)}只, 请求{len(stock_list)}只"
                                    )
                                    return result_subset
                except Exception:
                    continue

        return None

    def get_financial_data(self, stock_list: List[str],
                           table_list: Optional[List[str]] = None,
                           start_time: str = '', end_time: str = '',
                           report_type: str = 'announce_time') -> Dict[str, Any]:
        """获取财务数据

        逐只股票逐表查询并缓存为 parquet，避免中断后重复下载。
        全部下载完成后合并为总缓存(pkl)，提升后续加载速度。

        Args:
            stock_list: 股票代码列表
            table_list: 财务报表列表，为空则获取全部
            start_time: 起始时间
            end_time: 结束时间
            report_type: 报表筛选方式
                'report_time' - 按报告期（截止日期）
                'announce_time' - 按披露日期（回测时推荐，避免未来数据）

        Returns:
            dict: { stock1: { table1: DataFrame, ... }, ... }
        """
        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning("xtquant 未安装，返回空财务数据")
                return {}
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        tables = table_list or self.FINANCIAL_TABLES
        namespace = 'QMTDataProcessor_Financial'

        # 尝试从合并缓存加载（基于股票列表内容hash，支持子集匹配）
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"merged_{stocks_hash}_{start_time}_{end_time}_{report_type}"

        # 1. 精确匹配：股票列表完全相同
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='financial')
            self.logger.info(f"从合并缓存(精确)加载财务数据: {len(merged_cached)} 只股票")
            return merged_cached

        # 2. 子集匹配：已有缓存包含请求的全部股票
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.cache_dir / namespace
        if ns_dir.exists():
            for cache_file in ns_dir.glob('merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue  # 已尝试精确匹配
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='financial')
                                # 过滤只保留请求的股票
                                result_subset = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if len(result_subset) == len(stock_list):
                                    self.logger.info(
                                        f"从合并缓存(子集)加载财务数据: "
                                        f"缓存{len(cached_stocks)}只, 请求{len(stock_list)}只"
                                    )
                                    return result_subset
                except Exception:
                    continue

        # 逐只股票缓存（每只股票的每个表单独存为 parquet）
        result = {}
        cache_hits = 0
        download_hits = 0
        fail_count = 0
        time_suffix = f"_{start_time}_{end_time}" if start_time or end_time else ""
        total = len(stock_list)
        phase_start = time.time()

        self.logger.info(f"开始获取财务数据: {total} 只股票, 表: {', '.join(tables)}")

        for i, stock in enumerate(stock_list, 1):
            stock_data = {}
            all_cached = True

            # 逐表检查 parquet 缓存
            for table in tables:
                cache_key = f"{stock}{time_suffix}_{table}_{report_type}"
                cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
                if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                    stock_data[table] = cached
                else:
                    # 尝试旧格式 pkl（向后兼容）
                    cached_pkl = cache_manager.disk_cache.get(namespace, cache_key, 'pkl')
                    if cached_pkl is not None and isinstance(cached_pkl, pd.DataFrame) and not cached_pkl.empty:
                        stock_data[table] = cached_pkl
                        cache_manager.disk_cache.put(namespace, cache_key, cached_pkl, 'parquet')
                    else:
                        all_cached = False

            if all_cached and stock_data:
                result[stock] = stock_data
                cache_hits += 1
                if i % 20 == 0 or i == total:
                    self.logger.info(
                        f"[ {i} / {total} ] 进度: {cache_hits} 缓存命中, "
                        f"{download_hits} 已下载, {fail_count} 失败"
                    )
                continue

            # 有缺失的表，重新下载该股票全部财务数据
            try:
                dl_start = time.time()
                data = self.xtdata.get_financial_data(
                    [stock], tables,
                    start_time=start_time, end_time=end_time,
                    report_type=report_type,
                )
                dl_elapsed = time.time() - dl_start
                if data and stock in data and data[stock]:
                    stock_new_data = data[stock]
                    # 合并已缓存和新增数据
                    stock_data.update(stock_new_data)
                    result[stock] = stock_data
                    # 逐表写入 parquet 缓存
                    for table, df in stock_new_data.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # QMT xtdata 返回的财务数据索引是 RangeIndex，
                            # 需要将其转换为 DatetimeIndex 以便回测按日期查询
                            df = self._normalize_qmt_financial_df(df, report_type)
                            cache_key = f"{stock}{time_suffix}_{table}_{report_type}"
                            cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
                download_hits += 1
                self.logger.info(
                    f"[ {i} / {total} ] {stock} 财务数据已下载 "
                    f"({dl_elapsed:.1f}秒)"
                )
            except Exception as e:
                fail_count += 1
                # 下载失败但有部分缓存，仍使用缓存数据
                if stock_data:
                    result[stock] = stock_data
                self.logger.warning(
                    f"[ {i} / {total} ] {stock} 财务数据下载失败: {e}"
                )
                continue

        # 全部下载完成后，写入合并缓存（转为 DataFrame 存 parquet）
        phase_elapsed = time.time() - phase_start
        self.logger.info(
            f"财务数据获取完成: 缓存命中 {cache_hits}, 新下载 {download_hits}, "
            f"失败 {fail_count}, 耗时 {phase_elapsed:.1f}秒"
        )
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='financial')
            if merged_df is not None:
                cache_manager.disk_cache.put(namespace, merged_cache_key, merged_df, 'parquet')
                self.logger.info(f"财务数据合并缓存已创建: {len(result)} 只股票")

        return result

    def _get_stock_list_cache_key(self, sector: str) -> str:
        """生成成分股缓存key，按日期区分"""
        today = datetime.now().strftime('%Y%m%d')
        return f"sector_{sector}_{today}"

    def _load_stock_list_from_cache(self, sector: str) -> Optional[List[str]]:
        """尝试从本地缓存加载成分股列表"""
        namespace = 'QMTDataProcessor_Sector'
        cache_key = self._get_stock_list_cache_key(sector)
        cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            if 'stock_code' in cached.columns:
                stock_list = cached['stock_code'].astype(str).tolist()
                self.logger.info(f"从本地缓存加载 {sector} 成分股: {len(stock_list)} 只")
                return stock_list
        return None

    def _save_stock_list_to_cache(self, sector: str, stock_list: List[str]) -> None:
        """将成分股列表保存到本地缓存"""
        namespace = 'QMTDataProcessor_Sector'
        cache_key = self._get_stock_list_cache_key(sector)
        df = pd.DataFrame({'stock_code': stock_list})
        cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
        self.logger.info(f"成分股列表已缓存: {sector} ({len(stock_list)} 只)")

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        """获取板块成分股列表

        优先从本地缓存加载，缓存按日期区分。若缓存不存在且QMT可用，
        则从QMT获取并保存到本地缓存，供后续离线使用。

        Args:
            sector: 板块名称，如 '沪深A股', '上证50', '沪深300'

        Returns:
            股票代码列表
        """
        # 1. 尝试从本地缓存加载（支持离线回测）
        cached = self._load_stock_list_from_cache(sector)
        if cached is not None:
            return cached

        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning("xtquant 未安装，返回模拟股票列表")
                return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        try:
            stock_list = self.xtdata.get_stock_list_in_sector(sector)
            if stock_list:
                self._save_stock_list_to_cache(sector, stock_list)
                return stock_list
            try:
                self.xtdata.download_sector_data()
            except Exception:
                pass
            stock_list = self.xtdata.get_stock_list_in_sector(sector)
            if stock_list:
                self._save_stock_list_to_cache(sector, stock_list)
            return stock_list if stock_list else []
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"获取板块成分股失败: {e}，返回模拟列表")
                return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
            raise RuntimeError(f"获取板块成分股失败: {e}")

    def get_sector_list(self) -> List[str]:
        """获取所有板块列表"""
        if not self.xtdata:
            return []
        try:
            result = self.xtdata.get_sector_list() or []
            if result:
                return result
            try:
                self.xtdata.download_sector_data()
            except Exception:
                pass
            return self.xtdata.get_sector_list() or []
        except Exception:
            return []

    def _get_industry_mapping_cache_key(self, level: int, stock_pool: Optional[List[str]]) -> str:
        """生成行业映射缓存key，按日期和股票池区分"""
        today = datetime.now().strftime('%Y%m%d')
        if stock_pool:
            pool_hash = hashlib.md5(','.join(sorted(stock_pool)).encode()).hexdigest()[:8]
            return f"industry_sw{level}_{pool_hash}_{today}"
        return f"industry_sw{level}_all_{today}"

    def _load_industry_mapping_from_cache(self, level: int,
                                          stock_pool: Optional[List[str]]) -> Optional[Dict[str, str]]:
        """尝试从本地缓存加载行业映射"""
        namespace = 'QMTDataProcessor_Industry'
        cache_key = self._get_industry_mapping_cache_key(level, stock_pool)
        cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            if 'stock_code' in cached.columns and 'industry_name' in cached.columns:
                mapping = dict(zip(
                    cached['stock_code'].astype(str).tolist(),
                    cached['industry_name'].astype(str).tolist()
                ))
                self.logger.info(f"从本地缓存加载申万{level}级行业映射: {len(mapping)} 只股票")
                return mapping
        return None

    def _save_industry_mapping_to_cache(self, level: int,
                                        stock_pool: Optional[List[str]],
                                        mapping: Dict[str, str]) -> None:
        """将行业映射保存到本地缓存"""
        namespace = 'QMTDataProcessor_Industry'
        cache_key = self._get_industry_mapping_cache_key(level, stock_pool)
        df = pd.DataFrame({
            'stock_code': list(mapping.keys()),
            'industry_name': list(mapping.values())
        })
        cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
        self.logger.info(f"行业映射已缓存: 申万{level}级 ({len(mapping)} 只股票)")

    def get_industry_mapping(self, level: int = 1,
                             stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        """获取申万行业分类映射

        优先从本地缓存加载，缓存按日期和股票池区分。若缓存不存在且QMT可用，
        则从QMT获取并保存到本地缓存，供后续离线使用。

        Args:
            level: 行业级别，1=一级行业，2=二级行业，3=三级行业
            stock_pool: 股票池，为空则使用沪深A股

        Returns:
            { stock_code: industry_name, ... } 映射字典
        """
        # 1. 尝试从本地缓存加载（支持离线回测）
        cached = self._load_industry_mapping_from_cache(level, stock_pool)
        if cached is not None:
            return cached

        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning("xtquant 未安装，返回模拟行业映射")
                return {
                    '000001.SZ': '银行', '000002.SZ': '房地产',
                    '600000.SH': '银行', '600036.SH': '银行',
                }
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        try:
            self.logger.info("正在获取申万行业板块列表...")
            sector_list = self.get_sector_list()
            if not sector_list:
                self.logger.info("行业板块列表为空，尝试下载...")
                try:
                    client = self.xtdata.get_client()
                    client.down_all_sector_data()
                except Exception:
                    try:
                        self.xtdata.download_sector_data()
                    except Exception:
                        pass
                sector_list = self.xtdata.get_sector_list() or []
            sw_prefix = f'SW{level}'
            sw_sectors = [s for s in sector_list
                          if s.upper().startswith(sw_prefix.upper())
                          and '加权' not in s
                          and not s.upper().startswith('1000SW')
                          and not s.upper().startswith('300SW')
                          and not s.upper().startswith('500SW')
                          and not s.upper().startswith('HKSW')
                          and not s.upper().startswith('转债SW')]

            self.logger.info(f"找到 {len(sw_sectors)} 个申万{level}级行业板块")

            if stock_pool is None:
                stock_pool = self.get_stock_list('沪深A股')

            stock_set = set(stock_pool)
            mapping: Dict[str, str] = {}

            for idx, sw_sector in enumerate(sw_sectors, 1):
                industry_name = sw_sector[len(sw_prefix):]
                members = self.xtdata.get_stock_list_in_sector(sw_sector) or []
                for stock in members:
                    if stock in stock_set:
                        mapping[stock] = industry_name
                if idx % 10 == 0 or idx == len(sw_sectors):
                    self.logger.info(
                        f"[ {idx} / {len(sw_sectors)} ] 行业映射进度: "
                        f"已处理 {idx} 个行业, 累计匹配 {len(mapping)} 只股票"
                    )

            self.logger.info(f"申万{level}级行业映射加载完成: {len(sw_sectors)} 个行业, {len(mapping)} 只股票")
            # 保存到本地缓存供离线使用
            self._save_industry_mapping_to_cache(level, stock_pool, mapping)
            return mapping
        except RuntimeError:
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"获取行业映射失败: {e}，返回模拟数据")
                return {
                    '000001.SZ': '银行', '000002.SZ': '房地产',
                    '600000.SH': '银行', '600036.SH': '银行',
                }
            raise RuntimeError(f"获取行业映射失败: {e}")

    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        """获取股票分红数据

        逐只股票查询并缓存，避免中断后重复下载。
        全部下载完成后合并为总缓存，提升后续加载速度。

        Args:
            stock_list: 股票代码列表

        Returns:
            { stock_code: DataFrame(columns=[time, interest, ...]), ... }
            interest 列为每股派息金额
        """
        if not self.xtdata:
            if self._fallback_to_simulated:
                self.logger.warning("xtquant 未安装，返回空分红数据")
                return {}
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        namespace = 'QMTDataProcessor_Financial'

        # 尝试从合并缓存加载（基于股票列表内容hash，支持子集匹配）
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"dividend_merged_{stocks_hash}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='dividend')
            self.logger.info(f"从合并缓存(精确)加载分红数据: {len(merged_cached)} 只股票")
            return merged_cached

        # 子集匹配
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.cache_dir / namespace
        if ns_dir.exists():
            for cache_file in ns_dir.glob('dividend_merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='dividend')
                                result_subset = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if len(result_subset) == len(stock_list):
                                    self.logger.info(
                                        f"从合并缓存(子集)加载分红数据: "
                                        f"缓存{len(cached_stocks)}只, 请求{len(stock_list)}只"
                                    )
                                    return result_subset
                except Exception:
                    continue
        result = {}
        cache_hits = 0
        download_hits = 0
        fail_count = 0
        total = len(stock_list)
        phase_start = time.time()

        self.logger.info(f"开始获取分红数据: {total} 只股票")

        for i, stock in enumerate(stock_list, 1):
            cache_key = f"{stock}_dividend"
            cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
            if cached is not None:
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    result[stock] = cached
                cache_hits += 1
                if i % 20 == 0 or i == total:
                    self.logger.info(
                        f"[ {i} / {total} ] 分红数据进度: {cache_hits} 缓存, "
                        f"{download_hits} 已下载, {fail_count} 失败"
                    )
            else:
                # 尝试旧格式 pkl（向后兼容）
                cached_pkl = cache_manager.disk_cache.get(namespace, cache_key, 'pkl')
                if cached_pkl is not None and isinstance(cached_pkl, pd.DataFrame) and not cached_pkl.empty:
                    result[stock] = cached_pkl
                    # 迁移为 parquet 格式
                    cache_manager.disk_cache.put(namespace, cache_key, cached_pkl, 'parquet')
                    cache_hits += 1
                else:
                    try:
                        dl_start = time.time()
                        df = self.xtdata.get_divid_factors(stock)
                        dl_elapsed = time.time() - dl_start
                        if df is not None and not df.empty:
                            # QMT xtdata 返回的分红数据 time 列是毫秒时间戳，需要标准化
                            df = self._normalize_qmt_dividend_df(df)
                            result[stock] = df
                            cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
                        download_hits += 1
                        self.logger.info(
                            f"[ {i} / {total} ] {stock} 分红数据已下载 "
                            f"({dl_elapsed:.1f}秒)"
                        )
                    except Exception as e:
                        fail_count += 1
                        self.logger.warning(
                            f"[ {i} / {total} ] {stock} 分红数据下载失败: {e}"
                        )
                        continue

        # 全部下载完成后，写入合并缓存（转为 DataFrame 存 parquet）
        phase_elapsed = time.time() - phase_start
        self.logger.info(
            f"分红数据获取完成: 缓存命中 {cache_hits}, 新下载 {download_hits}, "
            f"失败 {fail_count}, 耗时 {phase_elapsed:.1f}秒"
        )
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='dividend')
            if merged_df is not None:
                cache_manager.disk_cache.put(namespace, merged_cache_key, merged_df, 'parquet')
                self.logger.info(f"分红数据合并缓存已创建: {len(result)} 只股票")

        self.logger.info(f"分红数据加载完成: {len(result)}/{len(stock_list)} 只股票有数据")
        return result

    def get_instrument_detail(self, stock_code: str) -> Optional[Dict]:
        """获取合约基础信息

        Args:
            stock_code: 股票代码，如 '000001.SZ'

        Returns:
            合约信息字典
        """
        if not self.xtdata:
            return None
        try:
            return self.xtdata.get_instrument_detail(stock_code)
        except Exception:
            return None

    def _generate_simulated_data(self, start_date: str, end_date: str, symbol: str = None) -> pd.DataFrame:
        """生成模拟数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        seed = 42
        if symbol:
            seed = hash(symbol) % 10000
        rng = np.random.default_rng(seed)
        
        base_price = 10.0 + rng.random() * 5.0
        prices = []
        for i in range(len(date_range)):
            price_change = rng.normal(0, 0.02)
            base_price *= (1 + price_change)
            prices.append(base_price)
        
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': rng.integers(10000, 100000, len(date_range)),
        }, index=date_range)
        
        return self.preprocess_data(df)


class CSVDataProcessor(DataProcessor):
    """CSV数据处理器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    def get_data(self, symbol: str, start_date: str, end_date: str, file_path: str, **kwargs) -> pd.DataFrame:
        """从CSV文件获取数据

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV文件不存在: {file_path}")

        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV文件为空: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV文件格式错误: {file_path}, {e}")
        except Exception as e:
            raise ValueError(f"读取CSV文件失败: {file_path}, {e}")

        required_cols = {'open', 'high', 'low', 'close'}
        missing = required_cols - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"CSV文件缺少必要列: {missing}, 文件: {file_path}")

        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return self.preprocess_data(df)


def _convert_symbol_to_qmt(symbol: str) -> str:
    """将各种格式的股票代码转换为QMT格式 (如 000001.SZ)

    支持输入格式:
    - QMT格式: 000001.SZ, 600000.SH (直接返回)
    - 纯数字: 000001, 600000
    - baostock格式: sz.000001, sh.600000
    """
    if '.' in symbol and not symbol.startswith(('sz.', 'sh.', 'SZ.', 'SH.')):
        return symbol  # 已经是QMT格式

    # baostock格式 -> 纯数字+市场
    if symbol.lower().startswith(('sz.', 'sh.')):
        parts = symbol.split('.')
        market = parts[0].upper()
        code = parts[1]
        return f"{code}.{market}"

    # 纯数字 -> 根据代码规则判断市场
    code = symbol.replace('.', '')
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def _convert_symbol_to_akshare(symbol: str) -> str:
    """将QMT格式股票代码转换为akshare行情格式 (纯数字, 如 000001)
    用于 stock_zh_a_hist 等行情接口
    """
    return symbol.split('.')[0]


def _convert_symbol_to_akshare_financial(symbol: str) -> str:
    """将QMT格式股票代码转换为akshare财务数据格式 (如 SZ000001, SH600519)
    用于 stock_balance_sheet_by_report_em 等财务接口
    """
    parts = symbol.split('.')
    if len(parts) == 2:
        code, market = parts
        return f"{market.upper()}{code}"
    # 纯数字，根据代码首字符判断市场
    code = symbol
    if code.startswith(('6', '9')):
        return f"SH{code}"
    else:
        return f"SZ{code}"


def _convert_symbol_to_baostock(symbol: str) -> str:
    """将QMT格式股票代码转换为baostock格式 (如 sz.000001)"""
    parts = symbol.split('.')
    if len(parts) == 2:
        code, market = parts
        return f"{market.lower()}.{code}"
    # 纯数字
    code = symbol
    if code.startswith(('6', '9')):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


class AKShareDataProcessor(DataProcessor):
    """AKShare数据处理器

    基于东方财富数据源，支持获取A股历史行情数据，数据范围远超QMT的一年限制。
    股票代码统一使用QMT格式 (000001.SZ)，内部自动转换。

    注意：akshare仅提供行情数据，财务/行业/分红等数据仍依赖QMT。
    """

    # 指数代码映射 (QMT格式 -> 中证指数代码)
    INDEX_CODE_MAP = {
        '000300.SH': '000300',  # 沪深300
        '000905.SH': '000905',  # 中证500
        '000852.SH': '000852',  # 中证1000
        '000016.SH': '000016',  # 上证50
    }

    PERIOD_MAP = {
        '1d': 'daily',
        'day': 'daily',
        'daily': 'daily',
        '1w': 'weekly',
        'week': 'weekly',
        'weekly': 'weekly',
        '1M': 'monthly',
        'month': 'monthly',
        'monthly': 'monthly',
    }

    FINANCIAL_TABLES = [
        'Balance', 'Income', 'CashFlow', 'Pershareindex',
    ]

    def __init__(self, fallback_to_simulated: bool = True):
        try:
            import akshare as ak
            self._ak = ak
        except ImportError:
            self._ak = None
            logging.getLogger(__name__).warning("akshare not installed, pip install akshare")
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    @smart_cache(cache_type='market', incremental=True)
    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """从AKShare获取行情数据

        Args:
            symbol: QMT格式股票代码, 如 '000001.SZ'
            start_date: 起始日期 '2016-01-01'
            end_date: 结束日期 '2026-04-17'
            period: 周期, 仅支持日线 '1d'
        """
        if not self._ak:
            if self._fallback_to_simulated:
                self.logger.warning(f"akshare 未安装，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("akshare 未安装，请 pip install akshare")

        try:
            ak_code = _convert_symbol_to_akshare(symbol)
            ak_period = self.PERIOD_MAP.get(period, 'daily')
            start_dt = start_date.replace('-', '')
            end_dt = end_date.replace('-', '')

            self.logger.info(f"AKShare获取数据: {symbol} ({ak_code}), {start_date} ~ {end_date}")

            # 带重试的数据获取
            max_retries = 3
            df = None
            for retry in range(max_retries):
                try:
                    df = self._ak.stock_zh_a_hist(
                        symbol=ak_code,
                        period=ak_period,
                        start_date=start_dt,
                        end_date=end_dt,
                        adjust="qfq",  # 前复权，回测推荐
                    )
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = 2 * (retry + 1)
                        self.logger.warning(f"AKShare请求失败(第{retry+1}次): {e}, {wait_time}秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise

            if df is None or df.empty:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} AKShare数据为空，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

            # 中文列名映射为英文
            col_map = {
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
                '成交额': 'amount', '振幅': 'amplitude', '涨跌幅': 'pct_change',
                '涨跌额': 'change', '换手率': 'turnover',
            }
            df = df.rename(columns=col_map)

            # 设置日期索引
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                df = df.set_index('datetime')
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 过滤日期范围
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            if df.empty:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} 数据过滤后为空，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

            # 确保必要列存在且类型正确
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 只保留回测需要的列
            keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[keep_cols]

            # 限频：akshare建议每次调用间隔
            time.sleep(1.0)

            return self.preprocess_data(df)

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"AKShare获取数据失败: {e}，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError(f"AKShare获取数据失败: {e}")

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        """获取板块成分股列表

        支持沪深300、中证500、上证50等主要指数成分股。
        其他板块名称尝试通过akshare获取。
        """
        if not self._ak:
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

        try:
            # 常见板块名映射到指数代码
            sector_map = {
                '沪深300': '000300',
                '中证500': '000905',
                '中证1000': '000852',
                '上证50': '000016',
            }
            index_code = sector_map.get(sector)
            if index_code:
                df = self._ak.index_stock_cons_csindex(symbol=index_code)
                if df is not None and not df.empty:
                    # 成分股代码列可能是 '品种代码' 或 '股票代码'
                    code_col = None
                    for col_name in ['品种代码', '股票代码', 'code', '成分券代码']:
                        if col_name in df.columns:
                            code_col = col_name
                            break
                    if code_col:
                        codes = df[code_col].astype(str).tolist()
                        result = [_convert_symbol_to_qmt(c) for c in codes]
                        self.logger.info(f"AKShare获取 {sector} 成分股: {len(result)} 只")
                        return result

            # 如果不是已知指数，尝试获取全部A股
            if sector in ('沪深A股', 'A股', '全部A股'):
                df = self._ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    code_col = None
                    for col_name in ['代码', '股票代码', 'code']:
                        if col_name in df.columns:
                            code_col = col_name
                            break
                    if code_col:
                        codes = df[code_col].astype(str).tolist()
                        result = [_convert_symbol_to_qmt(c) for c in codes]
                        self.logger.info(f"AKShare获取 {sector}: {len(result)} 只")
                        return result

            self.logger.warning(f"AKShare不支持板块 '{sector}'，返回默认列表")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        except Exception as e:
            self.logger.warning(f"AKShare获取板块成分股失败: {e}")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

    def download_financial_data(self, stock_list: List[str],
                                  table_list: Optional[List[str]] = None,
                                  start_time: str = '', end_time: str = '') -> None:
        """使用AKShare下载财务数据

        Args:
            stock_list: 股票代码列表，如 ['000001.SZ', '600000.SH']
            table_list: 财务报表列表，如 ['Balance', 'Income']，为空则下载全部
            start_time: 起始时间，如 '20230101'
            end_time: 结束时间，如 '20240101'
        """
        if not self._ak:
            self.logger.warning("akshare 未安装，无法下载财务数据")
            return

        self.logger.info(f"AKShare财务数据下载: {len(stock_list)} 只股票")
        # AKShare的财务数据是实时获取的，不需要预下载
        pass

    def get_financial_data(self, stock_list: List[str],
                           table_list: Optional[List[str]] = None,
                           start_time: str = '', end_time: str = '',
                           report_type: str = 'announce_time') -> Dict[str, Any]:
        """使用AKShare获取财务数据

        逐只股票逐表查询并缓存为 parquet，避免中断后重复下载。
        全部下载完成后合并为总缓存，提升后续加载速度。

        Args:
            stock_list: 股票代码列表
            table_list: 财务报表列表，为空则获取全部
            start_time: 起始时间
            end_time: 结束时间
            report_type: 报表筛选方式（AKShare暂不支持，忽略）

        Returns:
            dict: { stock1: { table1: DataFrame, ... }, ... }
        """
        if not self._ak:
            self.logger.warning("akshare 未安装，返回空财务数据")
            return {}

        tables = table_list or self.FINANCIAL_TABLES
        namespace = 'AKShareDataProcessor_Financial'

        # 尝试从合并缓存加载（基于股票列表内容hash，支持子集匹配）
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"merged_{stocks_hash}_{start_time}_{end_time}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='financial')
            self.logger.info(f"从合并缓存(精确)加载财务数据: {len(merged_cached)} 只股票")
            return merged_cached

        # 子集匹配
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.cache_dir / namespace
        if ns_dir.exists():
            for cache_file in ns_dir.glob('merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='financial')
                                result_subset = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if len(result_subset) == len(stock_list):
                                    self.logger.info(
                                        f"从合并缓存(子集)加载财务数据: "
                                        f"缓存{len(cached_stocks)}只, 请求{len(stock_list)}只"
                                    )
                                    return result_subset
                except Exception:
                    continue

        # 逐只股票逐表缓存
        result = {}
        cache_hits = 0
        total = len(stock_list)

        for i, symbol in enumerate(stock_list, 1):
            ak_code = _convert_symbol_to_akshare_financial(symbol)
            symbol_data = {}
            all_cached = True

            # 逐表检查 parquet 缓存
            for table in tables:
                cache_key = f"{symbol}_{table}"
                cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
                if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                    symbol_data[table] = cached
                else:
                    all_cached = False

            if all_cached and symbol_data:
                result[symbol] = symbol_data
                cache_hits += 1
                self.logger.info(f"[ {i} / {total} ] {symbol} 财务数据已缓存")
                continue

            # 有缺失的表，重新下载
            downloaded_tables = []
            failed_tables = []
            for table in tables:
                if table in symbol_data:
                    continue
                try:
                    df = self._get_akshare_financial_data(ak_code, table)
                    if df is not None and not df.empty:
                        symbol_data[table] = df
                        cache_key = f"{symbol}_{table}"
                        cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
                        downloaded_tables.append(table)
                    else:
                        failed_tables.append(table)
                except Exception as e:
                    failed_tables.append(table)
                    self.logger.debug(f"获取 {symbol} {table} 财务数据失败: {e}")

            if symbol_data:
                result[symbol] = symbol_data
            if downloaded_tables and not failed_tables:
                self.logger.info(f"[ {i} / {total} ] {symbol} 财务数据已下载 ({len(downloaded_tables)}表)")
            elif downloaded_tables:
                self.logger.info(f"[ {i} / {total} ] {symbol} 部分下载成功 ({len(downloaded_tables)}/{len(tables)}表), 失败: {failed_tables}")
            else:
                self.logger.info(f"[ {i} / {total} ] {symbol} 财务数据下载失败 (0/{len(tables)}表)")
                self.logger.warning(f"{symbol} 所有财务数据下载失败: {failed_tables}")

        # 全部下载完成后，写入合并缓存
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='financial')
            if merged_df is not None:
                cache_manager.disk_cache.put(namespace, merged_cache_key, merged_df, 'parquet')
                self.logger.info(f"财务数据合并缓存已创建: {len(result)} 只股票")

        return result

    def _get_akshare_financial_data(self, ak_code: str, table: str) -> Optional[pd.DataFrame]:
        """获取AKShare财务数据并转换为QMT格式

        Args:
            ak_code: AKShare财务接口格式的代码 (如 SZ000001, SH600519)
            table: 财务报表名 (Balance/Income/CashFlow/Pershareindex)
        """
        try:
            df = None

            if table == 'Balance':
                # 资产负债表
                df = self._ak.stock_balance_sheet_by_report_em(symbol=ak_code)
            elif table == 'Income':
                # 利润表
                df = self._ak.stock_profit_sheet_by_report_em(symbol=ak_code)
            elif table == 'CashFlow':
                # 现金流量表
                df = self._ak.stock_cash_flow_sheet_by_report_em(symbol=ak_code)
            elif table == 'Pershareindex':
                # 业绩报表（按报告期批量查询，再按股票过滤）
                df = self._get_pershareindex_data(ak_code)
            else:
                return None

            if df is None or df.empty:
                return None

            # 转换列名以兼容QMT格式
            df = self._convert_akshare_columns(df, table)

            # 设置 DatetimeIndex 用于回测按日期过滤
            # 优先使用公告日期（避免未来数据），其次使用报告期
            index_set = False
            if '公告日期' in df.columns:
                try:
                    dt_col = pd.to_datetime(df['公告日期'], errors='coerce')
                    df = df.dropna(subset=['公告日期'])
                    if not df.empty:
                        df.index = dt_col.loc[df.index]
                        df = df.sort_index()
                        index_set = True
                except Exception:
                    pass
            if not index_set and '报告期' in df.columns:
                try:
                    dt_col = pd.to_datetime(df['报告期'], errors='coerce')
                    df = df.dropna(subset=['报告期'])
                    if not df.empty:
                        df.index = dt_col.loc[df.index]
                        df = df.sort_index()
                        index_set = True
                except Exception:
                    pass

            return df

        except Exception as e:
            self.logger.debug(f"AKShare获取 {ak_code} {table} 数据失败: {e}")
            return None

    def _get_pershareindex_data(self, ak_code: str) -> Optional[pd.DataFrame]:
        """获取单只股票的每股指标数据

        stock_yjbb_em 是批量接口（按报告期），需要查询最近几个报告期再过滤。
        ak_code 格式如 SZ000001，需要提取纯数字代码用于过滤。
        """
        # 从 ak_code (如 SZ000001) 提取纯数字代码
        import re
        code = re.sub(r'^[A-Za-z]+', '', ak_code)

        # 查询最近4个报告期
        from datetime import datetime
        now = datetime.now()
        dates = []
        for year_offset in range(2):
            y = now.year - year_offset
            for period in ['1231', '0930', '0630', '0331']:
                date_str = f"{y}{period}"
                if date_str <= now.strftime('%Y%m%d'):
                    dates.append(date_str)

        all_dfs = []
        for date in dates[:4]:  # 只查最近4个报告期
            try:
                batch_df = self._ak.stock_yjbb_em(date=date)
                if batch_df is not None and not batch_df.empty:
                    # 按股票代码过滤
                    code_col = None
                    for col in ['股票代码', '代码', 'symbol', 'code']:
                        if col in batch_df.columns:
                            code_col = col
                            break
                    if code_col:
                        filtered = batch_df[batch_df[code_col].astype(str) == code]
                        if not filtered.empty:
                            all_dfs.append(filtered)
                time.sleep(0.5)
            except Exception as e:
                self.logger.debug(f"stock_yjbb_em(date={date}) 查询失败: {e}")
                continue

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        return None

    def _convert_akshare_columns(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """转换AKShare列名为QMT兼容格式"""
        # 常见列名映射
        column_map = {
            # 资产负债表
            '资产总计': 'total_assets',
            '负债合计': 'total_liabilities',
            '所有者权益合计': 'total_equity',
            '流动资产合计': 'total_current_assets',
            '非流动资产合计': 'total_noncurrent_assets',
            '流动负债合计': 'total_current_liabilities',
            '非流动负债合计': 'total_noncurrent_liabilities',
            '货币资金': 'cash_and_equivalents',
            '应收账款': 'accounts_receivable',
            '存货': 'inventory',
            # 利润表
            '营业总收入': 'total_operate_income',
            '营业收入': 'operate_income',
            '营业总成本': 'total_operate_cost',
            '营业成本': 'operate_cost',
            '营业利润': 'operate_profit',
            '利润总额': 'total_profit',
            '净利润': 'net_profit',
            '归属于母公司股东的净利润': 'net_profit_parent',
            '基本每股收益': 'eps_basic',
            '稀释每股收益': 'eps_diluted',
            # 现金流量表
            '经营活动产生的现金流量净额': 'net_operate_cash_flow',
            '投资活动产生的现金流量净额': 'net_invest_cash_flow',
            '筹资活动产生的现金流量净额': 'net_finance_cash_flow',
            '现金及现金等价物净增加额': 'net_cash_increase',
            # 每股指标
            '每股收益': 'eps',
            '每股净资产': 'bps',
            '每股经营现金流量': 'ocfps',
            '每股资本公积金': 'capital_reserve_ps',
            '每股未分配利润': 'undistributed_profit_ps',
            '净资产收益率': 'roe',
            '净资产收益率-加权': 'roe_weighted',
            '净资产收益率-摊薄': 'roe_diluted',
            '销售毛利率': 'gross_profit_margin',
            '营业收入同比增长率': 'inc_operate_income_rate',
            '净利润同比增长率': 'inc_net_profit_rate',
        }

        # 应用映射
        df = df.rename(columns=column_map)

        return df

    def get_industry_mapping(self, level: int = 1,
                               stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        """使用AKShare获取行业分类数据

        Args:
            level: 行业级别，1=一级行业，2=二级行业，3=三级行业（AKShare暂不支持级别区分）
            stock_pool: 股票池，为空则获取全部A股

        Returns:
            dict: { stock_code: industry_name, ... }
        """
        if not self._ak:
            self.logger.warning("akshare 未安装，返回空行业数据")
            return {}

        try:
            # 获取申万行业分类数据
            df = self._ak.stock_board_industry_name_em()
            if df is None or df.empty:
                return {}

            industry_map = {}
            for _, row in df.iterrows():
                industry_name = row.get('板块名称', '')
                if not industry_name:
                    continue

                # 获取该行业下的股票
                try:
                    stocks_df = self._ak.stock_board_industry_cons_em(symbol=industry_name)
                    if stocks_df is not None and not stocks_df.empty:
                        code_col = None
                        for col_name in ['代码', '股票代码', 'code']:
                            if col_name in stocks_df.columns:
                                code_col = col_name
                                break

                        if code_col:
                            for code in stocks_df[code_col].astype(str).tolist():
                                qmt_code = _convert_symbol_to_qmt(code)
                                if stock_pool is None or qmt_code in stock_pool:
                                    industry_map[qmt_code] = industry_name
                except Exception as e:
                    self.logger.debug(f"获取行业 {industry_name} 成分股失败: {e}")

            self.logger.info(f"AKShare获取行业数据: {len(industry_map)} 只股票")
            return industry_map

        except Exception as e:
            self.logger.warning(f"AKShare获取行业数据失败: {e}")
            return {}

    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        """使用AKShare获取分红数据

        逐只股票查询并缓存为 parquet，避免中断后重复下载。
        全部下载完成后合并为总缓存，提升后续加载速度。

        Args:
            stock_list: 股票代码列表

        Returns:
            dict: { stock_code: DataFrame, ... }
        """
        if not self._ak:
            self.logger.warning("akshare 未安装，返回空分红数据")
            return {}

        namespace = 'AKShareDataProcessor_Financial'

        # 尝试从合并缓存加载（基于股票列表内容hash，支持子集匹配）
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"dividend_merged_{stocks_hash}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='dividend')
            self.logger.info(f"从合并缓存(精确)加载分红数据: {len(merged_cached)} 只股票")
            return merged_cached

        # 子集匹配
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.cache_dir / namespace
        if ns_dir.exists():
            for cache_file in ns_dir.glob('dividend_merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='dividend')
                                result_subset = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if len(result_subset) == len(stock_list):
                                    self.logger.info(
                                        f"从合并缓存(子集)加载分红数据: "
                                        f"缓存{len(cached_stocks)}只, 请求{len(stock_list)}只"
                                    )
                                    return result_subset
                except Exception:
                    continue

        # 逐只股票缓存
        result = {}
        total = len(stock_list)

        for i, symbol in enumerate(stock_list, 1):
            cache_key = f"{symbol}_dividend"
            cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
            if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                result[symbol] = cached
                self.logger.info(f"[ {i} / {total} ] {symbol} 分红数据已缓存")
                continue

            try:
                ak_code = _convert_symbol_to_akshare(symbol)
                df = self._ak.stock_dividend_cninfo(symbol=ak_code)
                if df is not None and not df.empty:
                    # 转换列名
                    df = df.rename(columns={
                        '分红年度': 'dividend_year',
                        '派息日': 'dividend_date',
                        '每股派息': 'dividend_per_share',
                        '股权登记日': 'record_date',
                        '除权除息日': 'ex_dividend_date',
                    })
                    result[symbol] = df
                    cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
                self.logger.info(f"[ {i} / {total} ] {symbol} 分红数据已下载")
            except Exception as e:
                self.logger.debug(f"获取 {symbol} 分红数据失败: {e}")
                self.logger.info(f"[ {i} / {total} ] {symbol} 分红数据下载失败")

        # 全部下载完成后，写入合并缓存
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='dividend')
            if merged_df is not None:
                cache_manager.disk_cache.put(namespace, merged_cache_key, merged_df, 'parquet')
                self.logger.info(f"分红数据合并缓存已创建: {len(result)} 只股票")

        return result

    def _generate_simulated_data(self, start_date: str, end_date: str, symbol: str = None) -> pd.DataFrame:
        """生成模拟数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        seed = 42
        if symbol:
            seed = hash(symbol) % 10000
        rng = np.random.default_rng(seed)
        base_price = 10.0 + rng.random() * 5.0
        prices = []
        for i in range(len(date_range)):
            price_change = rng.normal(0, 0.02)
            base_price *= (1 + price_change)
            prices.append(base_price)
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': rng.integers(10000, 100000, len(date_range)),
        }, index=date_range)
        return self.preprocess_data(df)


class BaoStockDataProcessor(DataProcessor):
    """BaoStock数据处理器

    免费开源证券数据平台，支持1990年至今的A股历史数据。
    股票代码统一使用QMT格式 (000001.SZ)，内部自动转换。

    注意：baostock仅提供行情数据，财务/行业/分红等数据仍依赖QMT。
    """

    PERIOD_MAP = {
        '1d': 'd', 'day': 'd', 'daily': 'd',
        '1w': 'w', 'week': 'w', 'weekly': 'w',
        '1M': 'm', 'month': 'm', 'monthly': 'm',
        '5m': '5', '5min': '5',
        '15m': '15', '15min': '15',
        '30m': '30', '30min': '30',
        '60m': '60', '60min': '60',
    }

    def __init__(self, fallback_to_simulated: bool = True):
        self._bs = None
        self._logged_in = False
        try:
            import baostock as bs
            self._bs = bs
        except ImportError:
            logging.getLogger(__name__).warning("baostock not installed, pip install baostock")
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def _ensure_login(self):
        """确保baostock已登录"""
        if not self._bs:
            return False
        if not self._logged_in:
            return self._do_login()
        return True

    def _do_login(self):
        """执行baostock登录"""
        try:
            # 先尝试登出（防止重复登录）
            try:
                self._bs.logout()
            except Exception:
                pass
            lg = self._bs.login()
            if lg.error_code == '0':
                self._logged_in = True
                return True
            else:
                self.logger.error(f"baostock登录失败: {lg.error_msg}")
                self._logged_in = False
                return False
        except Exception as e:
            self.logger.error(f"baostock登录异常: {e}")
            self._logged_in = False
            return False

    def _reconnect(self):
        """重新连接baostock（连接超时后调用）"""
        self._logged_in = False
        return self._do_login()

    def __del__(self):
        """析构时登出baostock"""
        if self._bs and self._logged_in:
            try:
                self._bs.logout()
                self._logged_in = False
            except Exception:
                pass

    @smart_cache(cache_type='market', incremental=True)
    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """从BaoStock获取行情数据

        Args:
            symbol: QMT格式股票代码, 如 '000001.SZ'
            start_date: 起始日期 '2016-01-01'
            end_date: 结束日期 '2026-04-17'
            period: 周期, 支持 '1d', '1w', '1M', '5m', '15m', '30m', '60m'
        """
        if not self._bs:
            if self._fallback_to_simulated:
                self.logger.warning(f"baostock 未安装，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("baostock 未安装，请 pip install baostock")

        if not self._ensure_login():
            if self._fallback_to_simulated:
                self.logger.warning(f"baostock 登录失败，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("baostock 登录失败")

        try:
            bs_code = _convert_symbol_to_baostock(symbol)
            bs_freq = self.PERIOD_MAP.get(period, 'd')

            # 日线字段
            if bs_freq in ('d', 'w', 'm'):
                fields = "date,open,high,low,close,volume,amount,turn,pctChg"
            else:
                fields = "date,time,open,high,low,close,volume,amount"

            self.logger.info(f"BaoStock获取数据: {symbol} ({bs_code}), {start_date} ~ {end_date}, freq={bs_freq}")

            # 带重连重试的数据查询
            max_retries = 2
            rs = None
            for retry in range(max_retries + 1):
                rs = self._bs.query_history_k_data_plus(
                    code=bs_code,
                    fields=fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=bs_freq,
                    adjustflag="2",  # 前复权
                )
                if rs.error_code == '0':
                    break
                # 检测登录过期错误，自动重连
                if '未登录' in str(rs.error_msg) or 'login' in str(rs.error_msg).lower():
                    if retry < max_retries:
                        self.logger.warning(f"BaoStock连接超时，正在重连... (第{retry+1}次)")
                        time.sleep(1)
                        if self._reconnect():
                            continue
                # 其他错误直接退出
                break

            if rs.error_code != '0':
                if self._fallback_to_simulated:
                    self.logger.warning(f"BaoStock查询失败: {rs.error_msg}，使用模拟数据: {symbol}")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise RuntimeError(f"BaoStock查询失败: {rs.error_msg}")

            # 收集数据
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} BaoStock数据为空，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

            df = pd.DataFrame(data_list, columns=rs.fields)

            # 类型转换 (baostock返回的都是字符串)
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 处理日期索引
            if 'date' in df.columns:
                if 'time' in df.columns:
                    # 分钟线: 合并 date + time
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].str[:6].apply(
                        lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}" if len(x) >= 6 else x
                    ), errors='coerce')
                else:
                    df['datetime'] = pd.to_datetime(df['date'])
                df = df.set_index('datetime')

            # 去除空行
            df = df.dropna(subset=['close'])

            if df.empty:
                if self._fallback_to_simulated:
                    self.logger.warning(f"{symbol} 数据清洗后为空，使用模拟数据")
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 数据清洗后为空")

            # 只保留回测需要的列
            keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[keep_cols]

            # 限频
            time.sleep(0.1)

            return self.preprocess_data(df)

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                self.logger.warning(f"BaoStock获取数据失败: {e}，使用模拟数据: {symbol}")
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError(f"BaoStock获取数据失败: {e}")

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        """获取板块成分股列表

        支持沪深300、中证500等主要指数。
        当 BaoStock 无法连接时，尝试从本地缓存中提取股票列表。
        """
        if not self._bs:
            return self._get_stock_list_from_cache(sector) or ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

        if not self._ensure_login():
            return self._get_stock_list_from_cache(sector) or ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

        try:
            # 常见板块名映射到baostock指数代码
            sector_map = {
                '沪深300': 'sh.000300',
                '上证50': 'sh.000016',
                '中证500': 'sh.000905',
                '中证1000': 'sh.000852',
            }

            bs_index_code = sector_map.get(sector)
            if bs_index_code:
                # baostock没有通用的指数成分股查询，只有特定接口
                rs = None
                if sector == '沪深300':
                    rs = self._bs.query_hs300_stocks()
                elif sector == '上证50':
                    rs = self._bs.query_sz50_stocks()
                elif sector == '中证500':
                    rs = self._bs.query_zz500_stocks()

                # 检测登录过期并重连
                if rs and '未登录' in str(rs.error_msg):
                    self.logger.warning("BaoStock获取成分股时连接超时，正在重连...")
                    if self._reconnect():
                        if sector == '沪深300':
                            rs = self._bs.query_hs300_stocks()
                        elif sector == '上证50':
                            rs = self._bs.query_sz50_stocks()
                        elif sector == '中证500':
                            rs = self._bs.query_zz500_stocks()

                if rs and rs.error_code == '0':
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        code_col = 'code' if 'code' in df.columns else rs.fields[0]
                        codes = df[code_col].tolist()
                        result = [_convert_symbol_to_qmt(c) for c in codes]
                        self.logger.info(f"BaoStock获取 {sector} 成分股: {len(result)} 只")
                        return result

            # 全部A股
            if sector in ('沪深A股', 'A股', '全部A股'):
                rs = self._bs.query_stock_basic()
                if rs and rs.error_code == '0':
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        code_col = 'code' if 'code' in df.columns else rs.fields[0]
                        codes = df[code_col].tolist()
                        result = [_convert_symbol_to_qmt(c) for c in codes]
                        self.logger.info(f"BaoStock获取 {sector}: {len(result)} 只")
                        return result

            self.logger.warning(f"BaoStock不支持板块 '{sector}'，返回默认列表")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        except Exception as e:
            self.logger.warning(f"BaoStock获取板块成分股失败: {e}")
            return self._get_stock_list_from_cache(sector) or ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

    def _get_stock_list_from_cache(self, sector: str = '沪深A股') -> Optional[List[str]]:
        """当BaoStock离线时，从本地缓存中提取股票列表

        尝试从分红/财务合并缓存中提取股票代码。
        """
        namespace = 'BaoStockDataProcessor_Financial'

        # 尝试从合并缓存提取股票列表
        # 沪深300 对应约300只，遍历常见数量
        for try_len in [300, 500, 50, 800]:
            for mode in ['dividend', 'financial']:
                merged_cache_key = f"{mode}_merged_{try_len}"
                merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
                if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
                    try:
                        merged_dict = _parquet_to_merged_dict(merged_cached, mode=mode)
                        if merged_dict:
                            stocks = list(merged_dict.keys())
                            self.logger.info(f"从合并缓存({mode}_merged_{try_len})提取股票列表: {len(stocks)} 只")
                            return stocks
                    except Exception:
                        continue

        # 尝试从逐只缓存扫描
        import glob, os
        cache_dir = os.path.join('.cache', namespace)
        if os.path.exists(cache_dir):
            stocks = set()
            for f in glob.glob(os.path.join(cache_dir, '*_dividend.parquet')):
                basename = os.path.basename(f)
                stock = basename.replace('_dividend.parquet', '')
                stocks.add(stock)
            if stocks:
                result = sorted(stocks)
                self.logger.info(f"从逐只缓存扫描提取股票列表: {len(result)} 只")
                return result

        return None

    def _generate_simulated_data(self, start_date: str, end_date: str, symbol: str = None) -> pd.DataFrame:
        """生成模拟数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        seed = 42
        if symbol:
            seed = hash(symbol) % 10000
        rng = np.random.default_rng(seed)
        base_price = 10.0 + rng.random() * 5.0
        prices = []
        for i in range(len(date_range)):
            price_change = rng.normal(0, 0.02)
            base_price *= (1 + price_change)
            prices.append(base_price)
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': rng.integers(10000, 100000, len(date_range)),
        }, index=date_range)
        return self.preprocess_data(df)

    def download_financial_data(self, stock_list: List[str],
                                table_list: Optional[List[str]] = None,
                                start_time: str = '', end_time: str = '') -> None:
        """BaoStock财务数据为实时查询，无需预下载"""
        pass

    def _get_financial_data_single(self, stock: str, yq_list: List[tuple]) -> Optional[pd.DataFrame]:
        """获取单只股票的财务数据（用于逐只缓存）"""
        bs_code = _convert_symbol_to_baostock(stock)
        return self._fetch_baostock_financial_data(bs_code, yq_list)

    def get_financial_data(self, stock_list: List[str],
                           table_list: Optional[List[str]] = None,
                           start_time: str = '', end_time: str = '',
                           report_type: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """使用BaoStock季频接口获取财务数据

        将杜邦分析、成长能力、盈利能力、现金流量等数据合并到
        'Pershareindex' 表中，以兼容策略层的字段访问。
        逐只股票查询并缓存，避免中断后重复下载。
        全部下载完成后合并为总缓存，提升后续加载速度。
        即使 BaoStock 登录失败，也会尝试从缓存加载已有数据。
        """
        namespace = 'BaoStockDataProcessor_Financial'
        can_query = self._bs is not None and self._ensure_login()

        # 尝试从合并缓存加载（基于股票列表内容hash，支持子集匹配）
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"merged_{stocks_hash}_{start_time}_{end_time}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='financial')
            filtered = {k: v for k, v in merged_cached.items() if k in stock_list}
            if filtered:
                self.logger.info(f"从合并缓存(精确)加载财务数据: {len(filtered)}/{len(stock_list)} 只股票")
                if len(filtered) >= len(stock_list):
                    return filtered
                for stock in stock_list:
                    if stock not in filtered:
                        filtered[stock] = {}
                return filtered

        # 子集匹配：遍历已有合并缓存文件
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.cache_dir / namespace
        if ns_dir.exists():
            for cache_file in ns_dir.glob('merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='financial')
                                filtered = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if filtered:
                                    self.logger.info(
                                        f"从合并缓存(子集)加载财务数据: "
                                        f"缓存{len(cached_stocks)}只, 请求{len(stock_list)}只"
                                    )
                                    if len(filtered) >= len(stock_list):
                                        return filtered
                                    for stock in stock_list:
                                        if stock not in filtered:
                                            filtered[stock] = {}
                                    return filtered
                except Exception:
                    continue

        yq_list = self._get_year_quarter_range(start_time, end_time)
        if not yq_list:
            return {stock: {} for stock in stock_list}

        # 如果无法在线查询，尝试从逐只缓存获取
        if not can_query:
            self.logger.warning("BaoStock登录失败，仅从缓存加载财务数据")
            result = {}
            yq_key = f"{yq_list[0][0]}Q{yq_list[0][1]}_{yq_list[-1][0]}Q{yq_list[-1][1]}" if yq_list else "all"
            for stock in stock_list:
                cache_key = f"{stock}_{yq_key}"
                cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
                if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                    result[stock] = {'Pershareindex': cached}
            if result:
                self.logger.info(f"从逐只缓存加载财务数据: {len(result)}/{len(stock_list)} 只股票")
            else:
                self.logger.warning("缓存中无财务数据，策略基本面过滤将降级")
            return result

        result = {}
        cache_hits = 0
        # 使用 yq_list 生成缓存键，避免空 start_time/end_time 问题
        yq_key = f"{yq_list[0][0]}Q{yq_list[0][1]}_{yq_list[-1][0]}Q{yq_list[-1][1]}" if yq_list else "all"
        total = len(stock_list)
        for i, stock in enumerate(stock_list, 1):
            # 逐只股票使用缓存
            cache_key = f"{stock}_{yq_key}"
            cached = cache_manager.disk_cache.get(
                'BaoStockDataProcessor_Financial', cache_key, 'parquet'
            )
            if cached is not None:
                if cached is not None and not (isinstance(cached, pd.DataFrame) and cached.empty):
                    result[stock] = {'Pershareindex': cached}
                cache_hits += 1
                self.logger.info(f"[ {i} / {total} ] {cache_key} 已缓存")
            else:
                df = self._get_financial_data_single(stock, yq_list)
                if df is not None and not df.empty:
                    result[stock] = {'Pershareindex': df}
                    # 立即写入磁盘缓存（使用 parquet 格式）
                    cache_manager.disk_cache.put(
                        'BaoStockDataProcessor_Financial', cache_key, df, 'parquet'
                    )
                self.logger.info(f"[ {i} / {total} ] {cache_key} 已下载")

        # 全部下载完成后，写入合并缓存（转为 DataFrame 存 parquet）
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='financial')
            if merged_df is not None:
                save_cache_key = f"merged_{stocks_hash}_{start_time}_{end_time}"
                cache_manager.disk_cache.put(
                    'BaoStockDataProcessor_Financial', save_cache_key, merged_df, 'parquet'
                )
                self.logger.info(f"财务数据合并缓存已创建: {len(result)} 只股票")

        return result

    def _get_year_quarter_range(self, start_time: str, end_time: str) -> List[tuple]:
        """将时间范围转换为 (year, quarter) 列表"""
        try:
            if start_time:
                start_dt = pd.to_datetime(start_time, errors='coerce')
            else:
                start_dt = pd.Timestamp.now() - pd.DateOffset(years=5)
            if end_time:
                end_dt = pd.to_datetime(end_time, errors='coerce')
            else:
                end_dt = pd.Timestamp.now()
        except Exception:
            start_dt = pd.Timestamp.now() - pd.DateOffset(years=5)
            end_dt = pd.Timestamp.now()

        if pd.isna(start_dt):
            start_dt = pd.Timestamp.now() - pd.DateOffset(years=5)
        if pd.isna(end_dt):
            end_dt = pd.Timestamp.now()

        start_year, start_quarter = start_dt.year, (start_dt.month - 1) // 3 + 1
        end_year, end_quarter = end_dt.year, (end_dt.month - 1) // 3 + 1

        yq_list = []
        for y in range(start_year, end_year + 1):
            for q in range(1, 5):
                if (y == start_year and q < start_quarter) or (y == end_year and q > end_quarter):
                    continue
                yq_list.append((y, q))
        return yq_list

    def _fetch_baostock_financial_data(self, bs_code: str, yq_list: List[tuple]) -> Optional[pd.DataFrame]:
        """查询BaoStock季频财务数据并合并为DataFrame"""
        records_by_date: Dict[str, Dict[str, Any]] = {}

        query_funcs = [
            (self._bs.query_dupont_data, 'dupont'),
            (self._bs.query_growth_data, 'growth'),
            (self._bs.query_profit_data, 'profit'),
            (self._bs.query_cash_flow_data, 'cashflow'),
        ]

        for year, quarter in yq_list:
            for query_func, source in query_funcs:
                try:
                    rs = query_func(bs_code, year=year, quarter=quarter)
                    if rs.error_code != '0':
                        self.logger.debug(f"BaoStock {source} 查询失败 {bs_code} {year}Q{quarter}: {rs.error_msg}")
                        continue
                    row_count = 0
                    while rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        stat_date = row.get('statDate')
                        if not stat_date:
                            continue
                        if stat_date not in records_by_date:
                            records_by_date[stat_date] = {}
                        records_by_date[stat_date].update(row)
                        row_count += 1
                    if row_count > 0:
                        self.logger.debug(f"BaoStock {source} {bs_code} {year}Q{quarter}: {row_count} 条, 字段={rs.fields}")
                except Exception as e:
                    self.logger.debug(f"BaoStock {source} 异常 {bs_code} {year}Q{quarter}: {e}")
                    continue

        if not records_by_date:
            self.logger.debug(f"BaoStock 财务数据为空: {bs_code}, yq={yq_list}")
            return None

        df = pd.DataFrame.from_dict(records_by_date, orient='index')

        # 字段映射：兼容QMT风格的策略字段名
        column_map = {
            'dupontROE': 'du_return_on_equity',
            'YOYNI': 'inc_net_profit_rate',
        }
        df = df.rename(columns=column_map)

        # 数值转换
        for col in df.columns:
            if col not in ('code', 'pubDate', 'statDate'):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算每股经营现金流（若存在总额和总股本）
        if 'NCIOperAct' in df.columns and 'totalShare' in df.columns:
            df['s_fa_ocfps'] = df['NCIOperAct'] / df['totalShare']

        # 设置索引为公告日期，缺失则用统计日期
        date_col = 'pubDate' if 'pubDate' in df.columns else 'statDate'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df = df.sort_index()
        else:
            self.logger.debug(f"BaoStock 无日期列: {bs_code}, 列={list(df.columns)}")
            return None

        self.logger.debug(f"BaoStock 财务数据成功: {bs_code}, 行={len(df)}, 列={list(df.columns)}")
        return df

    def get_industry_mapping(self, level: int = 1,
                             stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        """获取行业分类映射

        使用 query_stock_industry 获取证监会行业分类。
        参考: https://www.baostock.com/mainContent?file=stockIndustry.md
        Args:
            level: 行业级别（BaoStock暂不支持多级，忽略）
            stock_pool: 股票池，为空则获取全部A股
        """
        if not self._bs:
            return {}
        if not self._ensure_login():
            return {}

        try:
            rs = self._bs.query_stock_industry()
            if rs.error_code != '0':
                self.logger.warning(f"BaoStock获取行业分类失败: {rs.error_msg}")
                return {}

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return {}

            df = pd.DataFrame(data_list, columns=rs.fields)
            mapping = {}

            code_col = 'code' if 'code' in df.columns else rs.fields[0]
            industry_col = 'industry' if 'industry' in df.columns else None

            if not industry_col:
                self.logger.warning("BaoStock返回数据中无industry字段，无法构建行业映射")
                return {}

            pool_set = set(stock_pool) if stock_pool else None

            for _, row in df.iterrows():
                bs_code = row.get(code_col)
                industry = row.get(industry_col)
                if not bs_code or not industry:
                    continue
                qmt_code = _convert_symbol_to_qmt(bs_code)
                if pool_set is not None and qmt_code not in pool_set:
                    continue
                mapping[qmt_code] = str(industry)

            self.logger.info(f"BaoStock行业映射加载完成: {len(mapping)} 只股票")
            return mapping
        except Exception as e:
            self.logger.warning(f"BaoStock获取行业映射失败: {e}")
            return {}

    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        """获取分红数据

        使用 query_dividend_data 获取除权除息信息。
        逐只股票查询并缓存为 parquet，避免中断后重复下载。
        即使 BaoStock 登录失败，也会尝试从缓存加载已有数据。
        """
        namespace = 'BaoStockDataProcessor_Financial'
        can_query = self._bs is not None and self._ensure_login()

        # 尝试从合并缓存加载（基于股票列表内容hash，支持子集匹配）
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"dividend_merged_{stocks_hash}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='dividend')
            filtered = {k: v for k, v in merged_cached.items() if k in stock_list}
            if filtered:
                self.logger.info(f"从合并缓存(精确)加载分红数据: {len(filtered)}/{len(stock_list)} 只股票")
                if len(filtered) >= len(stock_list):
                    return filtered
                if not can_query:
                    return filtered

        # 子集匹配：遍历已有合并缓存文件
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.cache_dir / namespace
        if ns_dir.exists():
            for cache_file in ns_dir.glob('dividend_merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='dividend')
                                filtered = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if filtered:
                                    self.logger.info(
                                        f"从合并缓存(子集)加载分红数据: "
                                        f"缓存{len(cached_stocks)}只, 请求{len(stock_list)}只"
                                    )
                                    if len(filtered) >= len(stock_list):
                                        return filtered
                                    if not can_query:
                                        return filtered
                                    break  # 部分命中，退出去逐只查询补充
                except Exception:
                    continue

        result = {}
        current_year = pd.Timestamp.now().year
        total = len(stock_list)

        # 如果无法在线查询，尝试从逐只缓存获取
        if not can_query:
            self.logger.warning("BaoStock登录失败，仅从缓存加载分红数据")
            for i, stock in enumerate(stock_list, 1):
                cache_key = f"{stock}_dividend"
                cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
                if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                    result[stock] = cached
            if result:
                self.logger.info(f"从逐只缓存加载分红数据: {len(result)}/{len(stock_list)} 只股票")
            else:
                self.logger.warning("缓存中无分红数据，策略将无法计算股息率")
            return result

        for i, stock in enumerate(stock_list, 1):
            cache_key = f"{stock}_dividend"
            cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
            if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                result[stock] = cached
                self.logger.info(f"[ {i} / {total} ] {stock} 分红数据已缓存")
                continue

            bs_code = _convert_symbol_to_baostock(stock)
            rs_list = []

            # 查询近10年分红数据
            for year in range(current_year - 9, current_year + 1):
                try:
                    rs = self._bs.query_dividend_data(
                        code=bs_code, year=str(year), yearType='report'
                    )
                    if rs.error_code == '0':
                        while rs.next():
                            rs_list.append(rs.get_row_data())
                except Exception:
                    continue

            if not rs_list:
                self.logger.info(f"[ {i} / {total} ] {stock} 无分红数据")
                continue

            try:
                df = pd.DataFrame(rs_list, columns=rs.fields)  # type: ignore[possibly-undefined]
                # BaoStock返回字符串，转换数值
                if 'dividCashPsBeforeTax' in df.columns:
                    df['interest'] = pd.to_numeric(df['dividCashPsBeforeTax'], errors='coerce')
                # 以除权除息日期作为索引
                date_col = 'dividOperateDate' if 'dividOperateDate' in df.columns else 'dividPayDate'
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col])
                    df = df.set_index(date_col)
                    df = df.sort_index()
                result[stock] = df
                cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
                self.logger.info(f"[ {i} / {total} ] {stock} 分红数据已下载")
            except Exception as e:
                self.logger.debug(f"处理 {stock} 分红数据失败: {e}")
                self.logger.info(f"[ {i} / {total} ] {stock} 分红数据处理失败")

        # 全部下载完成后，写入合并缓存（转为 DataFrame 存 parquet）
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='dividend')
            if merged_df is not None:
                save_cache_key = f"dividend_merged_{stocks_hash}"
                cache_manager.disk_cache.put(namespace, save_cache_key, merged_df, 'parquet')
                self.logger.info(f"分红数据合并缓存已创建: {len(result)} 只股票")

        self.logger.info(f"BaoStock分红数据加载完成: {len(result)}/{len(stock_list)} 只股票有数据")
        return result


def create_data_processor(data_source: str = 'qmt', fallback_to_simulated: bool = True) -> DataProcessor:
    """数据处理器工厂函数

    Args:
        data_source: 数据源，可选 'qmt', 'akshare', 'baostock'
        fallback_to_simulated: 数据获取失败时是否降级为模拟数据

    Returns:
        DataProcessor 实例
    """
    if data_source == 'akshare':
        return AKShareDataProcessor(fallback_to_simulated=fallback_to_simulated)
    elif data_source == 'baostock':
        return BaoStockDataProcessor(fallback_to_simulated=fallback_to_simulated)
    elif data_source == 'qmt':
        return QMTDataProcessor(fallback_to_simulated=fallback_to_simulated)
    else:
        raise ValueError(f"不支持的数据源: {data_source}，可选: 'qmt', 'akshare', 'baostock'")
