import pandas as pd
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.cache import smart_cache, cache_manager
from core.data.base import DataProcessor, _merged_dict_to_parquet, _parquet_to_merged_dict


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
        import numpy as np
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
