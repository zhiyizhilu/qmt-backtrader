import pandas as pd
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime

from core.cache import smart_cache, cache_manager
from core.data.base import DataProcessor, _merged_dict_to_parquet, _parquet_to_merged_dict


class QMTDataProcessor(DataProcessor):
    """QMT数据处理器
    
    纯QMT数据源，行情数据通过smart_cache缓存到本地parquet。
    缓存命中时直接读取，不调用QMT API；缓存缺失时才下载并保存。
    非行情数据（成分股、行业映射等）的OpenData兜底仍保留。
    """
    
    FINANCIAL_TABLES = [
        'Balance', 'Income', 'CashFlow', 'Capital',
        'HolderNum', 'Top10Holder', 'Top10FlowHolder', 'Pershareindex',
    ]

    def __init__(self, fallback_to_simulated: bool = False, use_opendata: bool = True):
        try:
            from xtquant import xtdata
            self.xtdata = xtdata
        except ImportError:
            self.xtdata = None
            logging.getLogger(__name__).warning("xtquant not installed, using simulated data")
        self._fallback_to_simulated = fallback_to_simulated
        self._financial_data_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        
        # 非行情数据的OpenData兜底（成分股、行业映射等），行情数据不再使用OpenData补充
        if use_opendata:
            try:
                from core.data.opendata import OpenDataProcessor
                self._opendata = OpenDataProcessor(fallback_to_simulated=fallback_to_simulated)
            except ImportError:
                self._opendata = None
        else:
            self._opendata = None

        # 增加本地CSV成分股管理器（聚宽历史数据）
        try:
            from core.data.index_constituent import IndexConstituentManager
            self._index_constituent_mgr = IndexConstituentManager(xtdata=self.xtdata)
        except Exception:
            self._index_constituent_mgr = None

        # 增加本地CSV行业成分股管理器（聚宽历史数据）
        try:
            from core.data.industry_constituent import IndustryConstituentManager
            self._industry_constituent_mgr = IndustryConstituentManager(xtdata=self.xtdata)
        except Exception:
            self._industry_constituent_mgr = None
    
    def check_connection(self) -> bool:
        """检查QMT数据服务是否可用"""
        if not self.xtdata:
            return False
        try:
            # 使用 get_market_data 检测连接，get_financial_index 不存在
            self.xtdata.get_market_data(['close'], ['000001.SZ'], period='1d', count=1)
            return True
        except Exception:
            return False

    def _guard_against_stale_data(self, symbol: str, start_date: str,
                                  end_date: str, df: pd.DataFrame,
                                  namespace: str) -> None:
        """防止 QMT 离线/陈旧时返回的部分数据覆盖已有的完整磁盘缓存。

        当 QMT 客户端关闭但 xtdata 仍可导入时，get_market_data_ex 可能只返回
        本地陈旧的局部数据（截止日期远早于请求结束日期，例如仅到上次同步的
        2024-12）。如果不加校验直接交给 smart_cache 缓存，就会把已下载的完整
        数据（含 2025/2026）覆盖成残缺数据，导致后续回测结果被截断到 2024 年。

        校验逻辑：若本次返回的数据截止日明显早于请求结束日，且磁盘中已存在
        更完整的同标的缓存，则判定为「QMT 离线返回的陈旧数据」，主动抛异常，
        让 smart_cache 回退使用磁盘上已有的完整缓存，而不是用残缺数据覆盖它。
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return
        if not isinstance(df.index, pd.DatetimeIndex):
            return

        req_end_ts = pd.Timestamp(end_date)
        data_end = df.index.max()

        # 返回数据截止日与请求结束日差距在容忍范围内（含周末/节假日/当日未收盘），视为正常
        if data_end >= req_end_ts - pd.Timedelta(days=5):
            return

        # 返回数据明显早于请求结束日 → 检查磁盘是否已有更完整的缓存
        try:
            from core.cache import cache_manager as _cm
            cached = _cm.disk_cache.get_yearly_range(
                namespace, symbol,
                list(range(pd.Timestamp(start_date).year, req_end_ts.year + 1)),
                '1d',
            )
            if (cached is not None and isinstance(cached.index, pd.DatetimeIndex)
                    and cached.index.max() > data_end):
                raise RuntimeError(
                    f"QMT返回数据陈旧(截止{data_end.date()})，磁盘已有更完整缓存"
                    f"(截止{cached.index.max().date()})，疑似QMT未连接/离线，"
                    f"放弃用残缺数据覆盖缓存: {symbol}"
                )
        except RuntimeError:
            raise
        except Exception:
            # 磁盘检查异常不应影响正常流程
            pass

    def _load_from_disk_if_complete(self, symbol: str, start_date: str,
                                    end_date: str, namespace: str) -> Optional[pd.DataFrame]:
        """当磁盘缓存已完整覆盖请求区间时，直接返回磁盘数据（无需 QMT）。

        用于实现「首次自动下载后，后续回测不依赖 QMT」：QMT 离线但缓存完整时，
        直接复用本地缓存，避免任何 QMT 调用（含离线时可能返回的陈旧局部数据）。
        缓存不完整时返回 None，交由上层走正常（可能失败的）QMT 获取流程。
        """
        from core.cache import cache_manager as _cm
        try:
            cached = _cm.disk_cache.get_yearly_range(
                namespace, symbol,
                list(range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1)),
                '1d',
            )
            if (cached is not None and isinstance(cached.index, pd.DatetimeIndex)
                    and cached.index.min() <= pd.Timestamp(start_date)
                    and cached.index.max() >= pd.Timestamp(end_date) - pd.Timedelta(days=5)):
                return self.preprocess_data(
                    cached[(cached.index >= pd.Timestamp(start_date))
                           & (cached.index <= pd.Timestamp(end_date))]
                )
        except Exception:
            pass
        return None

    @smart_cache(cache_type='market', incremental=True)
    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """从QMT获取后复权行情数据（带缓存）
        
        smart_cache 自动处理：内存缓存 → 磁盘缓存(年份分片parquet) → 缺失年份才调此函数体。
        命名空间 QMTDataProcessor → .cache/QMTData/market/{symbol}/{year}_{period}.parquet
        
        缓存命中时不调用QMT API，直接返回缓存数据；
        缓存缺失时调用 download_history_data + get_market_data_ex，下载后保存到缓存。
        
        如需不复权数据，请使用 get_raw_data()。
        """
        if not self.xtdata:
            raise RuntimeError("QMT (xtquant) 未安装或未连接")

        # QMT 离线但本地缓存已完整覆盖请求区间 → 直接返回缓存，实现回测零 QMT 依赖
        if not self.check_connection():
            disk_df = self._load_from_disk_if_complete(symbol, start_date, end_date, 'QMTDataProcessor')
            if disk_df is not None:
                return disk_df
            raise RuntimeError(
                f"QMT未连接且本地缓存未完整覆盖请求区间，无法获取行情: "
                f"{symbol} ({start_date}~{end_date})"
            )

        start_time = start_date.replace('-', '')
        end_time = end_date.replace('-', '')

        self.logger.debug(f"正在下载 {symbol} 行情数据 ({start_date} ~ {end_date}, {period})...")
        self.xtdata.download_history_data(
            stock_code=symbol, period=period,
            start_time='19900101', end_time='',
            incrementally=False
        )
        self.logger.debug(f"{symbol} 行情数据下载完成，正在读取...")

        history_data = self.xtdata.get_market_data_ex(
            [], [symbol], period=period,
            start_time=start_time, end_time=end_time,
            count=-1, dividend_type="back"
        )

        if symbol not in history_data:
            raise RuntimeError(f"QMT无数据: {symbol} ({start_date}~{end_date})")

        df = history_data[symbol]

        if df.empty:
            raise RuntimeError(f"QMT无数据: {symbol} ({start_date}~{end_date})")

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S')
            except (ValueError, TypeError):
                try:
                    df.index = pd.to_datetime(df.index, format='%Y%m%d')
                except (ValueError, TypeError):
                    raise RuntimeError(f"QMT日期解析失败: {symbol}")

        df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

        if df.empty:
            raise RuntimeError(f"QMT无数据: {symbol} ({start_date}~{end_date})")

        # 离线/陈旧防护：避免残缺数据覆盖已有完整缓存
        self._guard_against_stale_data(symbol, start_date, end_date, df, 'QMTDataProcessor')

        return self.preprocess_data(df)

    def _get_data_from_qmt(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
        """从QMT直接获取数据（内部方法，不走缓存）
        
        仅用于诊断/对比场景（如 verify_hfq_consistency）。
        正常回测请使用 get_data()，它会走 smart_cache 缓存。
        失败返回None而非抛异常。
        """
        if not self.xtdata:
            return None

        try:
            start_time = start_date.replace('-', '')
            end_time = end_date.replace('-', '')

            self.logger.debug(f"正在下载 {symbol} 行情数据 ({start_date} ~ {end_date}, {period})...")
            self.xtdata.download_history_data(
                stock_code=symbol, period=period,
                start_time='19900101', end_time='',
                incrementally=False
            )
            self.logger.debug(f"{symbol} 行情数据下载完成，正在读取...")

            history_data = self.xtdata.get_market_data_ex(
                [], [symbol], period=period,
                start_time=start_time, end_time=end_time,
                count=-1, dividend_type="back"
            )

            if symbol in history_data:
                df = history_data[symbol]
                
                # Check if data is empty
                if df.empty:
                    return None

                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S')
                    except (ValueError, TypeError):
                        try:
                            df.index = pd.to_datetime(df.index, format='%Y%m%d')
                        except (ValueError, TypeError):
                            return None

                df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

                if df.empty:
                    return None

                return self.preprocess_data(df)
            else:
                return None
        except Exception as e:
            self.logger.debug(f"QMT获取数据失败: {e}")
            return None

    @staticmethod
    def _parse_cache_date_range(cache_key: str):
        """从缓存 key 中解析日期范围

        支持的格式:
        - '000001.SZ_2025-03-12_2026-04-21_Balance_announce_time' → ('2025-03-12', '2026-04-21')
        - '000001.SZ_Balance_announce_time' → None (旧格式，无日期范围)

        Returns:
            (start_date, end_date) 或 None
        """
        import re
        m = re.match(r'^[A-Z0-9]+\.[A-Z]+_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_', cache_key)
        if m:
            return (m.group(1), m.group(2))
        return None

    def _find_best_financial_cache(self, stock: str, table: str,
                                    report_type: str,
                                    req_start: str, req_end: str) -> Optional[Tuple[pd.DataFrame, str]]:
        """查找能覆盖请求日期范围的最佳财务数据缓存

        查找逻辑:
        1. 精确匹配: {stock}_{req_start}_{req_end}_{table}_{report_type}
        2. 包含匹配: 同股票同表的所有缓存中，日期范围包含请求范围的
        3. 旧格式匹配: 无日期范围的旧缓存（视为全量数据，按索引日期截取）

        Args:
            stock: 股票代码
            table: 财务表名
            report_type: 报表筛选方式
            req_start: 请求起始日期
            req_end: 请求结束日期

        Returns:
            (DataFrame, cache_key) 或 None
        """
        namespace = 'QMTDataProcessor_Financial'
        time_suffix = f"_{req_start}_{req_end}" if req_start or req_end else ""

        # 1. 精确匹配
        exact_key = f"{stock}{time_suffix}_{table}_{report_type}"
        cached = cache_manager.disk_cache.get(namespace, exact_key, 'parquet')
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            return (cached, exact_key)

        # 2. 模式匹配: 查找同股票同表的所有缓存
        pattern = f"{stock}*_{table}_{report_type}"
        matches = cache_manager.disk_cache.find_by_pattern(namespace, pattern, 'parquet')

        best_match = None
        best_range = None

        req_start_ts = pd.Timestamp(req_start) if req_start else pd.Timestamp.min
        req_end_ts = pd.Timestamp(req_end) if req_end else pd.Timestamp.max

        for key, data in matches:
            if not isinstance(data, pd.DataFrame) or data.empty:
                continue

            date_range = self._parse_cache_date_range(key)

            if date_range is None:
                # 旧格式缓存（无日期范围），检查数据索引是否覆盖请求范围
                if isinstance(data.index, pd.DatetimeIndex) and not data.index.empty:
                    data_start = data.index.min()
                    data_end = data.index.max()
                    if data_start <= req_start_ts and data_end >= req_end_ts:
                        if best_match is None:
                            best_match = (data, key)
                            best_range = (data_start, data_end)
                        elif (data_end - data_start) < (best_range[1] - best_range[0]):
                            best_match = (data, key)
                            best_range = (data_start, data_end)
                continue

            cache_start, cache_end = date_range
            cache_start_ts = pd.Timestamp(cache_start)
            cache_end_ts = pd.Timestamp(cache_end)

            if cache_start_ts <= req_start_ts and cache_end_ts >= req_end_ts:
                if best_match is None:
                    best_match = (data, key)
                    best_range = (cache_start_ts, cache_end_ts)
                elif (cache_end_ts - cache_start_ts) < (best_range[1] - best_range[0]):
                    best_match = (data, key)
                    best_range = (cache_start_ts, cache_end_ts)

        if best_match:
            return best_match

        # 3. 尝试旧格式 pkl
        cached_pkl = cache_manager.disk_cache.get(namespace, exact_key, 'pkl')
        if cached_pkl is not None and isinstance(cached_pkl, pd.DataFrame) and not cached_pkl.empty:
            cache_manager.disk_cache.put(namespace, exact_key, cached_pkl, 'parquet')
            cache_manager.disk_cache.delete(namespace, exact_key, 'pkl')
            self.logger.info(f"旧pkl缓存已迁移为parquet: {exact_key}")
            return (cached_pkl, exact_key)

        return None

    def _cleanup_redundant_caches(self, stock: str, table: str,
                                   report_type: str,
                                   best_key: str, best_start: pd.Timestamp, best_end: pd.Timestamp) -> int:
        """清理被大范围缓存包含的小范围冗余缓存

        当有了大范围缓存（如 2020-2026）后，删除被包含的小范围缓存（如 2025-2026、2026-01~2026-04）

        Returns:
            删除的缓存数量
        """
        namespace = 'QMTDataProcessor_Financial'
        pattern = f"{stock}*_{table}_{report_type}"
        matches = cache_manager.disk_cache.find_by_pattern(namespace, pattern, 'parquet')

        deleted_count = 0
        for key, data in matches:
            if key == best_key:
                continue  # 跳过最佳缓存本身

            date_range = self._parse_cache_date_range(key)
            if date_range is None:
                continue  # 跳过旧格式缓存

            cache_start, cache_end = date_range
            cache_start_ts = pd.Timestamp(cache_start)
            cache_end_ts = pd.Timestamp(cache_end)

            # 如果小范围缓存被大范围缓存完全包含，则删除
            if cache_start_ts >= best_start and cache_end_ts <= best_end:
                cache_manager.disk_cache.delete(namespace, key, 'parquet')
                self.logger.debug(f"删除冗余小范围缓存: {key} (范围 {cache_start}~{cache_end} 被包含在 {best_start.date()}~{best_end.date()} 内)")
                deleted_count += 1

        return deleted_count

    def _find_overlapping_caches(self, stock: str, table: str,
                                  report_type: str,
                                  req_start: str, req_end: str) -> List[Tuple[pd.DataFrame, str, Optional[Tuple[str, str]]]]:
        """查找与请求日期范围有重叠的所有缓存

        用于合并策略：找到所有部分重叠的缓存，合并后覆盖更大范围。

        Returns:
            [(DataFrame, cache_key, date_range_or_none), ...] 按日期范围排序
        """
        namespace = 'QMTDataProcessor_Financial'
        pattern = f"{stock}*_{table}_{report_type}"
        matches = cache_manager.disk_cache.find_by_pattern(namespace, pattern, 'parquet')

        req_start_ts = pd.Timestamp(req_start) if req_start else pd.Timestamp.min
        req_end_ts = pd.Timestamp(req_end) if req_end else pd.Timestamp.max

        overlapping = []
        for key, data in matches:
            if not isinstance(data, pd.DataFrame) or data.empty:
                continue

            date_range = self._parse_cache_date_range(key)

            if date_range is None:
                if isinstance(data.index, pd.DatetimeIndex) and not data.index.empty:
                    data_start = data.index.min()
                    data_end = data.index.max()
                    if data_end >= req_start_ts and data_start <= req_end_ts:
                        overlapping.append((data, key, None))
                continue

            cache_start, cache_end = date_range
            cache_start_ts = pd.Timestamp(cache_start)
            cache_end_ts = pd.Timestamp(cache_end)

            if cache_end_ts >= req_start_ts and cache_start_ts <= req_end_ts:
                overlapping.append((data, key, date_range))

        overlapping.sort(key=lambda x: pd.Timestamp(x[2][0]) if x[2] else pd.Timestamp.min)
        return overlapping

    def _merge_financial_caches(self, existing_dfs: List[pd.DataFrame],
                                 new_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """合并多个财务数据缓存 DataFrame

        合并规则:
        - 按 DatetimeIndex 去重，保留最新数据
        - 列不一致时取交集
        """
        frames = [df for df in existing_dfs if isinstance(df, pd.DataFrame) and not df.empty]
        if new_df is not None and isinstance(new_df, pd.DataFrame) and not new_df.empty:
            frames.append(new_df)

        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]

        all_cols = [set(f.columns) for f in frames]
        common_cols = set.intersection(*all_cols) if all_cols else set()

        if common_cols:
            aligned = [f[list(common_cols)] for f in frames]
        else:
            self.logger.warning("合并缓存时列名不一致且无交集，使用 outer 合并")
            aligned = frames

        merged = pd.concat(aligned)
        if isinstance(merged.index, pd.DatetimeIndex):
            merged = merged[~merged.index.duplicated(keep='last')]
            merged = merged.sort_index()
        else:
            merged = merged.reset_index(drop=True)

        return merged

    def _normalize_qmt_financial_df(self, df: pd.DataFrame,
                                      report_type: str = 'announce_time') -> pd.DataFrame:
        """将 QMT xtdata 返回的财务 DataFrame 标准化

        QMT xtdata 返回的财务数据索引是 RangeIndex，日期信息存储在列中：
        - m_anntime: 公告日期 (YYYYMMDD 字符串)
        - m_timetag: 报告期 (YYYYMMDD 字符串)
        - declareDate: 公告日期 (HolderNum, Top10Holder, Top10FlowHolder 表)
        - endDate: 报告期截止日期 (HolderNum, Top10Holder, Top10FlowHolder 表)

        此方法将其转换为 DatetimeIndex，便于回测按日期过滤。
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # 选择日期列：优先使用公告日期避免未来数据
        if report_type == 'announce_time' and 'm_anntime' in df.columns:
            date_col = 'm_anntime'
        elif 'm_timetag' in df.columns:
            date_col = 'm_timetag'
        elif report_type == 'announce_time' and 'declareDate' in df.columns:
            # HolderNum, Top10Holder, Top10FlowHolder 表使用 declareDate
            date_col = 'declareDate'
        elif 'endDate' in df.columns:
            # HolderNum, Top10Holder, Top10FlowHolder 表使用 endDate
            date_col = 'endDate'
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

    def _check_data_ready(self, stock_list: List[str], tables: List[str],
                          start_time: str, end_time: str) -> Tuple[Set[str], Set[str]]:
        """检查哪些股票的数据已就绪

        Returns:
            (ready_stocks, missing_stocks): 已就绪的股票集合和缺失的股票集合
        """
        try:
            test_data = self.xtdata.get_financial_data(
                stock_list, tables,
                start_time=start_time, end_time=end_time
            )
        except UnicodeDecodeError as e:
            # QMT 返回的数据编码问题，尝试逐个股票检查
            self.logger.debug(f"批量检查数据编码错误，降级为单只检查: {e}")
            ready_stocks = set()
            missing_stocks = set()
            for stock in stock_list:
                try:
                    single_data = self.xtdata.get_financial_data([stock], tables, start_time=start_time, end_time=end_time)
                    if single_data and stock in single_data:
                        ready_stocks.add(stock)
                    else:
                        missing_stocks.add(stock)
                except Exception as se:
                    self.logger.debug(f"单只股票 {stock} 检查失败: {se}")
                    missing_stocks.add(stock)
            return ready_stocks, missing_stocks
        except Exception as e:
            self.logger.debug(f"检查数据就绪失败: {e}")
            return set(), set(stock_list)

        if not test_data:
            return set(), set(stock_list)

        ready_stocks = set()
        missing_stocks = set()

        for stock in stock_list:
            if stock not in test_data:
                missing_stocks.add(stock)
                continue

            stock_data_dict = test_data[stock]
            if not isinstance(stock_data_dict, dict):
                missing_stocks.add(stock)
                continue

            # 检查所有请求的表
            all_tables_ready = True
            for table in tables:
                if table not in stock_data_dict:
                    all_tables_ready = False
                    break
                df = stock_data_dict[table]
                if not isinstance(df, pd.DataFrame) or df.empty:
                    all_tables_ready = False
                    break

            if all_tables_ready:
                ready_stocks.add(stock)
            else:
                missing_stocks.add(stock)

        return ready_stocks, missing_stocks

    def _get_tables_ready_status(self, stock: str, tables: List[str],
                                  start_time: str, end_time: str) -> Dict[str, bool]:
        """获取某只股票各表的就绪状态

        Returns:
            {table_name: is_ready}
        """
        status = {table: False for table in tables}
        try:
            test_data = self.xtdata.get_financial_data(
                [stock], tables,
                start_time=start_time, end_time=end_time
            )
            if not test_data or stock not in test_data:
                return status

            stock_data_dict = test_data[stock]
            if not isinstance(stock_data_dict, dict):
                return status

            for table in tables:
                if table in stock_data_dict:
                    df = stock_data_dict[table]
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        status[table] = True
        except Exception:
            pass
        return status

    def _download_batch(self, batch: List[str], tables: List[str],
                       start_time: str, end_time: str,
                       batch_idx: int, total_batches: int) -> Set[str]:
        """下载一个批次的数据，返回成功下载的股票集合

        策略:
        1. 过滤退市股票
        2. 先检查哪些股票已有缓存（无需下载）
        3. 对缺失的股票调用 download_financial_data2
        4. 轮询等待新数据同步
        5. 如果批次下载失败或超时，降级为单只下载
        """
        batch_desc = f"[{batch_idx}/{total_batches}]"

        # 新增：过滤退市股票
        from core.cache import cache_manager
        active_batch = [s for s in batch if not cache_manager.index_manager.is_delisted(s)]
        delisted_in_batch = [s for s in batch if cache_manager.index_manager.is_delisted(s)]

        if delisted_in_batch:
            self.logger.info(f"{batch_desc} 跳过 {len(delisted_in_batch)} 只退市股票")

        if not active_batch:
            return set()

        # 步骤1: 检查已有缓存
        ready_before, missing_before = self._check_data_ready(
            active_batch, tables, start_time, end_time
        )

        if not missing_before:
            # 全部已有缓存
            self.logger.info(f"{batch_desc} {len(active_batch)}只股票已有缓存，跳过下载")
            return ready_before

        # 显示准备下载的股票代码（最多显示5只，超出显示数量）
        missing_list = sorted(list(missing_before))
        if len(missing_list) <= 5:
            stocks_str = ', '.join(missing_list)
        else:
            stocks_str = ', '.join(missing_list[:5]) + f' 等{len(missing_list)}只'

        # 显示请求的表
        tables_str = ', '.join(tables[:4])
        if len(tables) > 4:
            tables_str += f' 等{len(tables)}个表'

        if ready_before:
            ready_list = sorted(list(ready_before))
            if len(ready_list) <= 3:
                ready_str = f"缓存命中: {', '.join(ready_list)}"
            else:
                ready_str = f"缓存命中: {', '.join(ready_list[:3])} 等{len(ready_list)}只"
            self.logger.info(f"{batch_desc} {ready_str}，准备下载: {stocks_str} (表: {tables_str})")
        else:
            self.logger.info(f"{batch_desc} 准备下载: {stocks_str} (表: {tables_str})")

        # 步骤2: 下载缺失的股票
        stocks_to_download = list(missing_before)
        try:
            self.xtdata.download_financial_data2(
                stocks_to_download, tables,
                start_time=start_time, end_time=end_time,
            )
        except Exception as e:
            self.logger.warning(f"{batch_desc} 批次下载调用失败: {e}，将尝试单只下载")
            # 直接降级到单只下载
            return self._download_single_stocks(stocks_to_download, tables, start_time, end_time, batch_desc)

        # 步骤3: 轮询等待新数据同步
        # 优化：先等待一段时间让QMT同步数据，再开始轮询检查
        # 频繁调用get_financial_data会给QMT造成负载，导致数据同步变慢
        max_wait = 60  # 增加最大等待时间
        initial_wait = 5  # 首次等待5秒，让QMT有时间同步
        wait_interval = 2  # 轮询间隔从0.3秒增加到2秒
        waited = 0

        # 先等待initial_wait秒，不打扰QMT
        time.sleep(initial_wait)
        waited = initial_wait

        while waited <= max_wait:
            ready_now, missing_now = self._check_data_ready(
                stocks_to_download, tables, start_time, end_time
            )

            if not missing_now:
                # 全部下载成功
                all_ready = ready_before | ready_now
                if waited > 0:
                    self.logger.info(f"{batch_desc} 下载完成，等待{waited:.1f}秒")
                else:
                    self.logger.info(f"{batch_desc} 下载完成")
                return all_ready

            # 检查是否有股票数据为空（QMT中无数据）- 只在超时前检查，用于日志记录
            if waited >= 15:  # 等待15秒后检查
                for stock in list(missing_now):
                    try:
                        test_data = self.xtdata.get_financial_data([stock], tables, start_time=start_time, end_time=end_time)
                        if test_data and stock in test_data:
                            all_empty = True
                            for table in tables:
                                if table in test_data[stock]:
                                    df = test_data[stock][table]
                                    if isinstance(df, pd.DataFrame) and not df.empty:
                                        all_empty = False
                                        break
                            if all_empty:
                                self.logger.warning(f"{batch_desc} {stock} QMT中数据为空（可能正在下载）")
                    except Exception:
                        pass

            # 部分就绪，记录进度（每3秒或首次记录）
            progress_interval = 3  # 每3秒记录一次进度
            if waited == 0 and len(ready_now) > 0:
                self.logger.info(f"{batch_desc} 立即可用: {len(ready_now)}/{len(stocks_to_download)}只")
            elif waited > 0 and int(waited) % progress_interval == 0 and int(waited - wait_interval) % progress_interval != 0:
                ready_count = len(ready_now)
                total_count = len(stocks_to_download)
                if ready_count < total_count:
                    # 显示仍未就绪的股票及其表状态
                    still_missing = sorted(list(missing_now))[:3]
                    missing_details = []
                    for stock in still_missing:
                        table_status = self._get_tables_ready_status(stock, tables, start_time, end_time)
                        ready_tables = [t for t, ready in table_status.items() if ready]
                        if ready_tables:
                            missing_details.append(f"{stock}({len(ready_tables)}/{len(tables)}表)")
                        else:
                            missing_details.append(f"{stock}(无)")
                    missing_str = ', '.join(missing_details)
                    if len(missing_now) > 3:
                        missing_str += f' 等{len(missing_now)}只'
                    self.logger.info(f"{batch_desc} 下载中... {ready_count}/{total_count}只就绪，已等待{waited:.1f}秒，未就绪: {missing_str}")

            if waited < max_wait:
                time.sleep(wait_interval)
                waited += wait_interval
            else:
                break

        # 步骤4: 超时，降级为单只下载剩余股票
        if missing_now:
            timeout_stocks = sorted(list(missing_now))[:5]
            timeout_str = ', '.join(timeout_stocks)
            if len(missing_now) > 5:
                timeout_str += f' 等{len(missing_now)}只'
            self.logger.warning(
                f"{batch_desc} 批次下载超时，未就绪: {timeout_str}，降级为单只下载"
            )
            single_success = self._download_single_stocks(
                list(missing_now), tables, start_time, end_time, batch_desc
            )
            return ready_before | ready_now | single_success
        else:
            return ready_before | ready_now

    def _download_single_stocks(self, stock_list: List[str], tables: List[str],
                                start_time: str, end_time: str, batch_desc: str) -> Set[str]:
        """单只股票逐个下载，用于批次失败后的降级处理"""
        success_stocks = set()
        failed_stocks = []

        # 显示单只降级的股票列表
        stock_list_sorted = sorted(stock_list)
        if len(stock_list_sorted) <= 5:
            stocks_str = ', '.join(stock_list_sorted)
        else:
            stocks_str = ', '.join(stock_list_sorted[:5]) + f' 等{len(stock_list_sorted)}只'
        self.logger.info(f"{batch_desc} 单只降级下载: {stocks_str}")

        for i, stock in enumerate(stock_list, 1):
            try:
                # 单只下载
                self.xtdata.download_financial_data2([stock], tables, start_time=start_time, end_time=end_time)

                # 等待这只股票的数据就绪（最多15秒）
                # 先等3秒让QMT同步，再轮询检查
                time.sleep(3)
                max_wait = 15
                waited = 3
                stock_ready = False
                while waited <= max_wait:
                    ready, missing = self._check_data_ready([stock], tables, start_time, end_time)
                    if stock in ready:
                        success_stocks.add(stock)
                        stock_ready = True
                        break
                    if waited < max_wait:
                        time.sleep(2)
                        waited += 2
                    else:
                        break

                if not stock_ready:
                    failed_stocks.append(stock)
                    self.logger.debug(f"{batch_desc} 单只下载超时: {stock}")

            except Exception as e:
                failed_stocks.append(stock)
                self.logger.debug(f"{batch_desc} 单只下载失败 {stock}: {e}")

        # 汇总结果
        if success_stocks:
            if failed_stocks:
                self.logger.info(f"{batch_desc} 单只下载结果: 成功{len(success_stocks)}只，失败{len(failed_stocks)}只")
                if len(failed_stocks) <= 3:
                    self.logger.warning(f"{batch_desc} 下载失败: {', '.join(failed_stocks)}")
            else:
                self.logger.info(f"{batch_desc} 单只下载全部成功: {len(success_stocks)}只")

        return success_stocks

    def download_financial_data(self, stock_list: List[str],
                                table_list: Optional[List[str]] = None,
                                start_time: str = '', end_time: str = '') -> None:
        """下载财务数据到本地缓存（改进版，支持失败降级）

        Args:
            stock_list: 股票代码列表，如 ['000001.SZ', '600000.SH']
            table_list: 财务报表列表，如 ['Balance', 'Income']，为空则下载全部
            start_time: 起始时间，如 '20230101'
            end_time: 结束时间，如 '20240101'
        """
        if not self.xtdata:
            self.logger.warning("xtquant 未安装，无法下载财务数据")
            return

        # 新增：过滤退市股票
        from core.cache import cache_manager
        delisted = []
        active_stocks = []
        for stock in stock_list:
            if cache_manager.index_manager.is_delisted(stock):
                delisted.append(stock)
                continue
            active_stocks.append(stock)

        if delisted:
            delist_display = ', '.join(delisted[:5])
            if len(delisted) > 5:
                delist_display += f' 等{len(delisted)}只'
            self.logger.info(
                f"跳过 {len(delisted)} 只退市股票: {delist_display}"
            )

        if not active_stocks:
            self.logger.info("全部股票已退市，无需下载财务数据")
            return

        tables = table_list or self.FINANCIAL_TABLES
        total = len(active_stocks)

        start_ts = time.time()

        # 优化：先一次性发起所有下载请求，再统一等待和检查
        # 避免频繁轮询get_financial_data导致QMT负载过高
        batch_size = 50  # 增大批次，减少下载调用次数
        total_batches = (total + batch_size - 1) // batch_size

        self.logger.info(f"分 {total_batches} 批发起下载请求（每批{batch_size}只）...")

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = active_stocks[batch_start:batch_end]
            batch_idx = batch_start // batch_size + 1
            try:
                self.xtdata.download_financial_data2(
                    batch, tables, start_time=start_time, end_time=end_time
                )
                self.logger.info(f"[{batch_idx}/{total_batches}] 已发起下载请求: {len(batch)}只")
            except Exception as e:
                self.logger.warning(f"[{batch_idx}/{total_batches}] 下载请求失败: {e}")

        # 等待QMT同步数据
        wait_time = max(15, total // 10)  # 根据股票数量动态调整等待时间
        self.logger.info(f"等待 {wait_time} 秒让QMT同步数据...")
        time.sleep(wait_time)

        # 统一检查哪些股票数据已就绪
        all_success = set()
        check_batch_size = 50
        for check_start in range(0, total, check_batch_size):
            check_end = min(check_start + check_batch_size, total)
            check_batch = active_stocks[check_start:check_end]
            try:
                ready, missing = self._check_data_ready(check_batch, tables, start_time, end_time)
                all_success.update(ready)
            except Exception:
                pass

        # 如果还有缺失的，再等待一轮
        all_missing = set(active_stocks) - all_success
        if all_missing and len(all_missing) < total:
            self.logger.info(f"第一轮检查: {len(all_success)}/{total} 只就绪，等待15秒后重试 {len(all_missing)} 只...")
            time.sleep(15)
            try:
                ready2, missing2 = self._check_data_ready(list(all_missing), tables, start_time, end_time)
                all_success.update(ready2)
            except Exception:
                pass

        all_missing = set(active_stocks) - all_success
        if all_missing:
            self.logger.warning(f"仍有 {len(all_missing)} 只股票数据未就绪")

        # 下载完成后，对所有传入的股票调用 get_financial_data 保存到缓存
        # 注意：这里使用 active_stocks 而不是 all_success，因为即使 _download_batch
        # 认为某只股票已就绪（QMT中有数据），也可能需要保存到本地缓存
        self.logger.info(f"正在将 {len(active_stocks)} 只股票的数据保存到缓存...")
        try:
            # 使用 announce_time 作为默认 report_type
            self.get_financial_data(
                active_stocks, tables, start_time, end_time, report_type='announce_time'
            )
        except Exception as e:
            self.logger.warning(f"保存数据到缓存时出错: {e}")

        elapsed = time.time() - start_ts
        failed_count = total - len(all_success)

        if failed_count == 0:
            self.logger.info(
                f"财务数据下载完成: {total} 只股票全部成功, "
                f"耗时 {elapsed:.1f}秒"
            )
        else:
            self.logger.warning(
                f"财务数据下载完成: {len(all_success)}/{total} 只成功, "
                f"{failed_count} 只失败, 耗时 {elapsed:.1f}秒"
            )

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
        ns_dir = cache_manager.disk_cache.get_namespace_dir(namespace)
        if ns_dir.exists():
            for cache_file in ns_dir.glob('merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                candidate = cache_manager.disk_cache._try_read_cache_file(cache_file, 'parquet')
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

        return None

    def get_financial_data(self, stock_list: List[str],
                           table_list: Optional[List[str]] = None,
                           start_time: str = '', end_time: str = '',
                           report_type: str = 'announce_time') -> Dict[str, Any]:
        """获取财务数据

        逐只股票逐表查询并缓存为 parquet，避免中断后重复下载。

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
        result = self._get_financial_data_from_qmt(stock_list, table_list, start_time, end_time, report_type)
        return result

    def _get_financial_data_from_qmt(self, stock_list: List[str],
                                      table_list: Optional[List[str]] = None,
                                      start_time: str = '', end_time: str = '',
                                      report_type: str = 'announce_time') -> Dict[str, Any]:
        """从QMT获取财务数据 - V2按年份分片缓存

        优先使用按年份分片的新缓存格式:
          .cache/QMTData/financial/{symbol}/{year}_{table}.parquet

        向后兼容旧格式（合并缓存、逐只缓存）。
        """
        if not self.xtdata:
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        tables = table_list or self.FINANCIAL_TABLES
        namespace = 'QMTDataProcessor_Financial'

        req_years = self._parse_years_from_time_range(start_time, end_time)

        result = {}
        cache_hits = 0
        download_hits = 0
        fail_count = 0
        total = len(stock_list)
        phase_start = time.time()

        self.logger.info(f"开始获取财务数据: {total} 只股票, 表: {', '.join(tables)}, 请求年份={req_years or '全部'}")

        for i, stock in enumerate(stock_list, 1):
            # 新增：跳过退市股票
            from core.cache import cache_manager as _cm
            if _cm.index_manager.is_delisted(stock):
                self.logger.debug(f"跳过退市股票: {stock}")
                continue

            stock_data = {}
            tables_to_download = []
            missing_years_by_table = {}  # 记录每个表缺失的年份

            for table in tables:
                table_suffix = f"{table}_{report_type}"

                if req_years:
                    cached_years = cache_manager.index_manager.get_available_financial_years(stock, table_suffix)
                    if not cached_years:
                        cached_years = cache_manager.disk_cache.list_yearly_files(namespace, stock, table_suffix)

                    checked_years = cache_manager.index_manager.get_checked_financial_years(stock, table_suffix)
                    missing_years = set(req_years) - set(cached_years) - set(checked_years)

                    if not missing_years:
                        df = cache_manager.disk_cache.get_yearly_range(namespace, stock, sorted(req_years), table_suffix)
                        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                            stock_data[table] = df
                            continue

                    if missing_years:
                        missing_years_by_table[table] = sorted(missing_years)
                    tables_to_download.append(table)
                else:
                    available_years = cache_manager.index_manager.get_available_financial_years(stock, table_suffix)
                    if not available_years:
                        available_years = cache_manager.disk_cache.list_yearly_files(namespace, stock, table_suffix)

                    if available_years:
                        df = cache_manager.disk_cache.get_yearly_range(namespace, stock, sorted(available_years), table_suffix)
                        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                            stock_data[table] = df
                            continue

                    best = self._find_best_financial_cache(stock, table, report_type, start_time, end_time)
                    if best is not None:
                        stock_data[table] = best[0]
                        continue

                    tables_to_download.append(table)

            # 显示需要下载的详细信息
            if tables_to_download and i <= 5:  # 只显示前5只股票的详细信息，避免日志过多
                self.logger.info(f"[ {i} / {total} ] {stock} 需要下载:")
                for table in tables_to_download:
                    if table in missing_years_by_table:
                        years = missing_years_by_table[table]
                        years_str = ', '.join([str(y) for y in years[:5]])
                        if len(years) > 5:
                            years_str += f' 等{len(years)}年'
                        self.logger.info(f"  - {table}: 缺失年份 {years_str}")
                    else:
                        self.logger.info(f"  - {table}: 无缓存")

            if not tables_to_download and stock_data:
                result[stock] = stock_data
                cache_hits += 1
                if i % 20 == 0 or i == total:
                    self.logger.info(
                        f"[ {i} / {total} ] 进度: {cache_hits} 缓存命中, "
                        f"{download_hits} 已下载, {fail_count} 失败"
                    )
                continue

            if tables_to_download:
                try:
                    dl_start = time.time()
                    data = self.xtdata.get_financial_data(
                        [stock], tables_to_download,
                        start_time=start_time, end_time=end_time,
                        report_type=report_type,
                    )
                    dl_elapsed = time.time() - dl_start
                    if data and stock in data and data[stock]:
                        stock_new_data = data[stock]
                        for table, df in stock_new_data.items():
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                df = self._normalize_qmt_financial_df(df, report_type)
                                stock_data[table] = df

                                table_suffix = f"{table}_{report_type}"
                                if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                                    save_years = None
                                    if table in missing_years_by_table:
                                        save_years = missing_years_by_table[table]
                                    written = cache_manager.disk_cache.put_yearly_from_df(
                                        namespace, stock, table_suffix, df,
                                        only_years=save_years,
                                        skip_existing=True,
                                    )
                                    if written:
                                        years_str = ', '.join([str(y) for y in sorted(written)[:5]])
                                        if len(written) > 5:
                                            years_str += f' 等{len(written)}个年份'
                                        self.logger.debug(
                                            f"  保存 {stock} {table}: {years_str} -> "
                                            f"{stock}/{min(written)}_{table}_{report_type}.parquet ..."
                                        )
                                    for y in written:
                                        cache_manager.index_manager.update_financial_index(stock, table_suffix, y)

                                    if req_years and table in missing_years_by_table:
                                        empty_years = sorted(set(missing_years_by_table[table]) - set(written))
                                        if empty_years:
                                            cache_manager.index_manager.update_checked_financial_years(stock, table_suffix, empty_years)
                                            if i <= 5:
                                                self.logger.debug(
                                                    f"  记录 {stock} {table} 已检查无数据年份: {empty_years}"
                                                )
                                else:
                                    time_suffix = f"_{start_time}_{end_time}" if start_time or end_time else ""
                                    cache_key = f"{stock}{time_suffix}_{table}_{report_type}"
                                    cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
                                    self.logger.debug(
                                        f"  保存 {stock} {table}: "
                                        f"{cache_key}.parquet ({len(df)} 行)"
                                    )
                            else:
                                table_suffix = f"{table}_{report_type}"
                                if req_years and table in missing_years_by_table:
                                    cache_manager.index_manager.update_checked_financial_years(
                                        stock, table_suffix, missing_years_by_table[table]
                                    )
                    else:
                        for table in tables_to_download:
                            table_suffix = f"{table}_{report_type}"
                            if req_years and table in missing_years_by_table:
                                cache_manager.index_manager.update_checked_financial_years(
                                    stock, table_suffix, missing_years_by_table[table]
                                )

                    download_hits += 1
                    if i % 20 == 0:
                        cache_manager.index_manager.save_index()
                    # 显示下载的详细信息
                    if data and stock in data and data[stock]:
                        downloaded_tables = list(data[stock].keys())
                        tables_info = []
                        for table in downloaded_tables:
                            if table in data[stock]:
                                df = data[stock][table]
                                if isinstance(df, pd.DataFrame) and not df.empty:
                                    if isinstance(df.index, pd.DatetimeIndex):
                                        years = sorted(df.index.year.unique())
                                        years_str = f"{min(years)}~{max(years)}" if years else "N/A"
                                    else:
                                        years_str = f"{len(df)}行"
                                    tables_info.append(f"{table}({years_str})")
                                else:
                                    tables_info.append(f"{table}(空)")

                        tables_detail = ', '.join(tables_info[:4])
                        if len(tables_info) > 4:
                            tables_detail += f' 等{len(tables_info)}表'
                    else:
                        tables_detail = "无数据"

                    self.logger.info(
                        f"[ {i} / {total} ] {stock} 已下载: {tables_detail} "
                        f"({dl_elapsed:.1f}秒)"
                    )
                except Exception as e:
                    fail_count += 1
                    if stock_data:
                        result[stock] = stock_data
                    self.logger.warning(
                        f"[ {i} / {total} ] {stock} 财务数据下载失败: {e}"
                    )
                    continue

            if stock_data:
                result[stock] = stock_data

        cache_manager.index_manager.save_index()

        phase_elapsed = time.time() - phase_start
        self.logger.info(
            f"财务数据获取完成: 缓存命中 {cache_hits}, 新下载 {download_hits}, "
            f"失败 {fail_count}, 耗时 {phase_elapsed:.1f}秒"
        )

        return result

    def _parse_years_from_time_range(self, start_time: str, end_time: str) -> List[int]:
        """从时间范围解析出涉及的年份列表"""
        if not start_time and not end_time:
            return []
        try:
            start_year = pd.Timestamp(start_time).year if start_time else 2000
            end_year = pd.Timestamp(end_time).year if end_time else pd.Timestamp.now().year
            return list(range(start_year, end_year + 1))
        except Exception:
            return []

    def _cleanup_redundant_merged_caches(self, stock_list: List[str],
                                          start_time: str, end_time: str,
                                          report_type: str,
                                          best_key: str) -> int:
        """清理被大范围合并缓存包含的小范围冗余合并缓存

        合并缓存的key格式: merged_<hash>_<start>_<end>_<report_type>
        当有了新的合并缓存后，删除股票列表相同但范围更小的旧缓存
        """
        namespace = 'QMTDataProcessor_Financial'
        ns_dir = cache_manager.disk_cache.get_namespace_dir(namespace)
        if not ns_dir.exists():
            return 0

        # 计算当前股票列表的hash
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]

        req_start_ts = pd.Timestamp(start_time) if start_time else pd.Timestamp.min
        req_end_ts = pd.Timestamp(end_time) if end_time else pd.Timestamp.max

        deleted_count = 0
        for cache_file in ns_dir.glob('merged_*.parquet'):
            if cache_file.name == f"{best_key}.parquet":
                continue

            try:
                # 解析文件名: merged_<hash>_<start>_<end>_<report_type>.parquet
                parts = cache_file.stem.split('_')
                if len(parts) < 5:
                    continue

                cache_hash = parts[1]
                cache_start = parts[2]
                cache_end = parts[3]
                cache_report_type = '_'.join(parts[4:])

                # 只处理相同股票列表和相同report_type的缓存
                if cache_hash != stocks_hash or cache_report_type != report_type:
                    continue

                cache_start_ts = pd.Timestamp(cache_start)
                cache_end_ts = pd.Timestamp(cache_end)

                # 如果小范围缓存被大范围缓存完全包含，则删除
                if cache_start_ts >= req_start_ts and cache_end_ts <= req_end_ts:
                    cache_file.unlink()
                    self.logger.debug(f"删除冗余合并缓存: {cache_file.name} (范围 {cache_start}~{cache_end} 被包含在 {start_time}~{end_time} 内)")
                    deleted_count += 1
            except Exception:
                continue

        if deleted_count > 0:
            self.logger.info(f"清理了 {deleted_count} 个冗余合并缓存")

        return deleted_count



    def get_stock_list(self, sector: str = '沪深A股', date: Optional[str] = None) -> List[str]:
        """获取板块成分股列表

        优先从本地 CSV (JQData) 加载，无数据时从 QMT/OpenData 获取。

        Args:
            sector: 板块名称，如 '沪深A股', '上证50', '沪深300'
            date: 目标日期，格式 'YYYY-MM-DD'，为None时获取最新成分股

        Returns:
            股票代码列表
        """
        # 如果指定了日期，优先使用历史成分股
        if date:
            return self.get_historical_stock_list(sector, date)

        # 尝试从 JQData CSV 获取（如果有对应指数映射）
        if self._index_constituent_mgr:
            index_code = self._index_constituent_mgr.sector_to_index_code(sector)
            if index_code:
                result = self._index_constituent_mgr.get_constituent_stocks(index_code, datetime.now().strftime('%Y-%m-%d'))
                if result:
                    self.logger.info(f"本地CSV成分股: {sector}, 共 {len(result)} 只")
                    return result

        # 尝试从 QMT 获取
        if self.xtdata:
            try:
                stock_list = self.xtdata.get_stock_list_in_sector(sector)
                if stock_list:
                    return stock_list
                try:
                    self.xtdata.download_sector_data()
                except Exception:
                    pass
                stock_list = self.xtdata.get_stock_list_in_sector(sector)
                return stock_list if stock_list else []
            except Exception as e:
                self.logger.debug(f"QMT获取板块成分股失败: {e}")

        # QMT 失败，使用 OpenData 兜底
        if self._opendata:
            self.logger.info(f"QMT获取成分股失败，使用OpenData: {sector}")
            try:
                return self._opendata.get_stock_list(sector)
            except Exception as e:
                self.logger.warning(f"OpenData获取成分股也失败: {e}")

        # 全部失败
        raise RuntimeError(f"获取板块成分股失败: {sector}")

    def get_historical_stock_list(self, sector: str = '沪深A股',
                                   date: Optional[str] = None) -> List[str]:
        """获取指定日期的历史成分股列表

        数据源优先级:
        1. 本地CSV文件（聚宽历史成分股数据，最完整可靠）
        2. QMT 的 get_stock_list_in_sector(sector, timetag)
        3. OpenData 的纳入/剔除日期还原方式（可能不完整）

        当日期超出CSV范围时，自动从QMT获取最新成分股并更新CSV文件。

        Args:
            sector: 板块名称，如 '沪深300', '中证500'
            date: 目标日期，格式 'YYYY-MM-DD'，默认为当前日期

        Returns:
            股票代码列表
        """
        # 0. 优先使用本地CSV成分股数据（聚宽历史数据）
        if self._index_constituent_mgr:
            index_code = self._index_constituent_mgr.sector_to_index_code(sector)
            if index_code:
                result = self._index_constituent_mgr.get_constituent_stocks(index_code, date or datetime.now().strftime('%Y-%m-%d'))
                if result:
                    self.logger.info(f"本地CSV成分股: {sector}, date={date}, 共 {len(result)} 只")
                    return result

        # 1. 使用 QMT 历史成分股接口（get_stock_list_in_sector 支持 timetag）
        if self.xtdata and date:
            try:
                import time as _time
                dt_obj = datetime.strptime(date, '%Y-%m-%d')
                timetag = int(_time.mktime(dt_obj.timetuple()) * 1000)
                stock_list = self.xtdata.get_stock_list_in_sector(sector, timetag)
                if stock_list and len(stock_list) > 10:
                    self.logger.info(
                        f"QMT历史成分股: {sector}, date={date}, "
                        f"共 {len(stock_list)} 只"
                    )
                    return stock_list
            except Exception as e:
                self.logger.debug(f"QMT获取历史成分股失败: {e}")

        # 2. 使用 OpenData 获取历史成分股（基于纳入/剔除日期还原）
        if self._opendata:
            try:
                result = self._opendata.get_historical_stock_list(sector, date)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"OpenData获取历史成分股失败: {e}")

        # 3. 回退到当前成分股
        self.logger.warning(f"历史成分股获取失败，使用当前成分股: {sector}")
        return self.get_stock_list(sector)

    def get_all_historical_stocks_in_range(self, sector: str = '沪深A股',
                                            start_date: Optional[str] = None,
                                            end_date: Optional[str] = None) -> List[str]:
        """获取回测期间所有历史成分股的并集

        指数成分股会定期调整（如沪深300每半年调整一次），
        回测期间涉及的股票数量远超单次成分股数量。
        此方法收集整个回测期间所有变更记录的成分股并集，
        确保回测时能获取到所有曾经属于该指数的股票数据。

        Args:
            sector: 板块名称，如 '沪深300', '中证500'
            start_date: 回测起始日期，格式 'YYYY-MM-DD'
            end_date: 回测结束日期，格式 'YYYY-MM-DD'

        Returns:
            去重的股票代码列表
        """
        if self._index_constituent_mgr:
            index_code = self._index_constituent_mgr.sector_to_index_code(sector)
            if index_code:
                result = self._index_constituent_mgr.get_all_constituent_stocks_in_range(
                    index_code,
                    start_date or datetime.now().strftime('%Y-%m-%d'),
                    end_date or datetime.now().strftime('%Y-%m-%d')
                )
                if result:
                    self.logger.info(f"回测期间 {sector} 全部历史成分股: 共 {len(result)} 只")
                    return result

        # 回退：只用起始日期的成分股
        self.logger.warning(f"无法获取回测期间全部成分股，回退到起始日期成分股")
        return self.get_historical_stock_list(sector, date=start_date)

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



    def get_industry_mapping(self, level: int = 1,
                             stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        """获取申万行业分类映射

        数据源优先级:
        1. 本地CSV文件（聚宽历史行业成分股数据，最完整可靠）
        2. QMT 实时获取
        3. OpenData 兜底

        Args:
            level: 行业级别，1=一级行业，2=二级行业，3=三级行业
            stock_pool: 股票池，为空则使用沪深A股

        Returns:
            { stock_code: industry_name, ... } 映射字典
        """
        # 0. 优先使用本地CSV行业成分股数据（聚宽历史数据）
        if level == 1 and self._industry_constituent_mgr:
            if stock_pool is None:
                stock_pool = self.get_stock_list('沪深A股')
            mapping = self._industry_constituent_mgr.get_industry_mapping(stock_pool)
            if mapping:
                self.logger.info(f"本地CSV行业映射: {len(set(mapping.values()))} 个行业, {len(mapping)} 只股票")
                return mapping

        # 1. 尝试从 QMT 获取
        if self.xtdata:
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
                return mapping
            except Exception as e:
                self.logger.debug(f"QMT获取行业映射失败: {e}")

        # 2. QMT 失败，使用 OpenData 兜底
        if self._opendata:
            self.logger.info("QMT获取行业映射失败，使用OpenData")
            try:
                return self._opendata.get_industry_mapping(level=level, stock_pool=stock_pool)
            except Exception as e:
                self.logger.warning(f"OpenData获取行业映射也失败: {e}")

        # 3. 全部失败
        raise RuntimeError("获取行业映射失败")

    def get_historical_industry_mapping(self, stock_list: List[str],
                                        date: Optional[str] = None,
                                        classification: str = '申银万国行业分类标准') -> Dict[str, str]:
        """获取指定日期的历史行业分类映射

        数据源优先级:
        1. 本地CSV文件（聚宽历史行业成分股数据，最完整可靠）
        2. OpenData（巨潮资讯）
        3. 回退到当前行业映射

        Args:
            stock_list: 股票代码列表（QMT格式，如 ['000001.SZ', '600000.SH']）
            date: 目标日期，格式 'YYYY-MM-DD'，默认为当前日期
            classification: 行业分类标准，默认'申银万国行业分类标准'

        Returns:
            { stock_code: industry_name, ... } 映射字典
        """
        # 0. 优先使用本地CSV行业成分股数据（聚宽历史数据）
        if self._industry_constituent_mgr:
            target_date = date or datetime.now().strftime('%Y-%m-%d')
            mapping = self._industry_constituent_mgr.get_industry_mapping(stock_list, date=target_date)
            if mapping:
                self.logger.info(f"本地CSV历史行业映射: date={target_date}, "
                                 f"{len(set(mapping.values()))} 个行业, {len(mapping)} 只股票")
                return mapping

        # 使用 OpenData 获取历史行业映射
        if self._opendata:
            try:
                result = self._opendata.get_historical_industry_mapping(
                    stock_list=stock_list, date=date, classification=classification
                )
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"OpenData获取历史行业映射失败: {e}")

        # 回退到当前行业映射
        self.logger.info("历史行业映射获取失败，使用当前行业映射")
        return self.get_industry_mapping(level=1, stock_pool=stock_list)



    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        """获取股票分红数据

        使用单一合并缓存文件（dividend_all），避免产生大量单独文件。
        加载时从合并缓存读取已有数据，仅下载缺失的股票，下载后增量合并回缓存。
        对无分红数据的股票，记录到索引中避免重复下载（30天内有效）。

        Args:
            stock_list: 股票代码列表

        Returns:
            { stock_code: DataFrame(columns=[time, interest, ...]), ... }
            interest 列为每股派息金额
        """
        if not self.xtdata:
            raise RuntimeError("xtquant 未安装，请安装 xtquant 后重试")

        namespace = 'QMTDataProcessor_Financial'
        merged_cache_key = 'dividend_all'

        result = {}
        request_set = set(stock_list)

        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            cached_dict = _parquet_to_merged_dict(merged_cached, mode='dividend')
            cached_stocks = set(cached_dict.keys())
            for stock in stock_list:
                if stock in cached_dict and isinstance(cached_dict[stock], pd.DataFrame) and not cached_dict[stock].empty:
                    result[stock] = cached_dict[stock]
            missing_stocks = sorted(request_set - cached_stocks)
            self.logger.info(
                f"从合并缓存加载分红数据: 缓存有 {len(cached_stocks)} 只, "
                f"本次命中 {len(result)} 只, 缺失 {len(missing_stocks)} 只"
            )
        else:
            missing_stocks = sorted(stock_list)
            self.logger.info(f"无分红合并缓存, 需下载 {len(missing_stocks)} 只股票")

        checked_stocks = cache_manager.index_manager.get_checked_dividend_stocks()
        if checked_stocks:
            before = len(missing_stocks)
            missing_stocks = sorted(set(missing_stocks) - checked_stocks)
            skipped = before - len(missing_stocks)
            if skipped > 0:
                self.logger.info(
                    f"分红已检查跳过: {skipped} 只股票近期已确认无分红数据"
                )

        if missing_stocks:
            download_hits = 0
            empty_hits = 0
            fail_count = 0
            empty_stocks = []
            total = len(missing_stocks)
            phase_start = time.time()

            for i, stock in enumerate(missing_stocks, 1):
                try:
                    dl_start = time.time()
                    df = self.xtdata.get_divid_factors(stock)
                    dl_elapsed = time.time() - dl_start
                    if df is not None and not df.empty:
                        df = self._normalize_qmt_dividend_df(df)
                        result[stock] = df
                        download_hits += 1
                        self.logger.info(
                            f"[ {i} / {total} ] {stock} 分红数据已下载 "
                            f"({len(df)} 条, {dl_elapsed:.1f}秒)"
                        )
                    else:
                        empty_hits += 1
                        empty_stocks.append(stock)
                        self.logger.info(
                            f"[ {i} / {total} ] {stock} 无分红数据 "
                            f"({dl_elapsed:.1f}秒)"
                        )
                except Exception as e:
                    fail_count += 1
                    self.logger.warning(
                        f"[ {i} / {total} ] {stock} 分红数据下载失败: {e}"
                    )
                    continue

            if empty_stocks:
                cache_manager.index_manager.update_checked_dividend_stocks(empty_stocks)
                cache_manager.index_manager.save_index()

            phase_elapsed = time.time() - phase_start
            self.logger.info(
                f"分红数据下载完成: 有数据 {download_hits}, "
                f"无数据 {empty_hits}, 失败 {fail_count}, 耗时 {phase_elapsed:.1f}秒"
            )

            if result:
                merged_df = _merged_dict_to_parquet(result, mode='dividend')
                if merged_df is not None:
                    if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
                        try:
                            merged_df = pd.concat([merged_cached, merged_df], ignore_index=True)
                            merged_df = merged_df.drop_duplicates(subset=['_stock_code'], keep='last')
                        except Exception:
                            pass
                    cache_manager.disk_cache.put(namespace, merged_cache_key, merged_df, 'parquet')
                    self.logger.info(f"分红数据合并缓存已更新: {len(result)} 只股票")

            self._cleanup_dividend_individual_cache(namespace)

        self.logger.info(f"分红数据加载完成: {len(result)}/{len(stock_list)} 只股票有数据")
        return result

    def _cleanup_dividend_individual_cache(self, namespace: str):
        """清理旧的单独分红缓存文件"""
        ns_dir = cache_manager.disk_cache.get_namespace_dir(namespace)
        if not ns_dir.exists():
            return
        cleaned = 0
        for f in ns_dir.glob('*_dividend.parquet'):
            try:
                f.unlink()
                cleaned += 1
            except Exception:
                pass
        for f in ns_dir.glob('*_dividend.pkl'):
            try:
                f.unlink()
                cleaned += 1
            except Exception:
                pass
        for f in ns_dir.glob('dividend_merged_*.parquet'):
            try:
                f.unlink()
                cleaned += 1
            except Exception:
                pass
        if cleaned > 0:
            self.logger.info(f"已清理 {cleaned} 个冗余分红缓存文件")

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

    @smart_cache(cache_type='market', incremental=True, namespace_suffix='_Raw')
    def get_raw_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取不复权行情数据（带缓存），用于股息率等需要实际价格的计算

        与 get_data() 使用后复权数据不同，此方法获取不复权数据，
        独立缓存于 market_raw 命名空间，与后复权缓存互不干扰。
        命名空间 QMTDataProcessor_Raw → .cache/QMTData/market_raw/{symbol}/{year}_{period}.parquet
        """
        if not self.xtdata:
            raise RuntimeError("QMT (xtquant) 未安装或未连接")

        # QMT 离线但本地缓存已完整覆盖请求区间 → 直接返回缓存，实现回测零 QMT 依赖
        if not self.check_connection():
            disk_df = self._load_from_disk_if_complete(symbol, start_date, end_date, 'QMTDataProcessor_Raw')
            if disk_df is not None:
                return disk_df
            raise RuntimeError(
                f"QMT未连接且本地缓存未完整覆盖请求区间，无法获取不复权行情: "
                f"{symbol} ({start_date}~{end_date})"
            )

        start_time = start_date.replace('-', '')
        end_time = end_date.replace('-', '')

        self.xtdata.download_history_data(
            stock_code=symbol, period=period,
            start_time='19900101', end_time='',
            incrementally=False
        )

        history_data = self.xtdata.get_market_data_ex(
            [], [symbol], period=period,
            start_time=start_time, end_time=end_time,
            count=-1, dividend_type="none"
        )

        if symbol not in history_data:
            raise RuntimeError(f"QMT无不复权数据: {symbol} ({start_date}~{end_date})")

        df = history_data[symbol]

        if df.empty:
            raise RuntimeError(f"QMT无不复权数据: {symbol} ({start_date}~{end_date})")

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S')
            except (ValueError, TypeError):
                try:
                    df.index = pd.to_datetime(df.index, format='%Y%m%d')
                except (ValueError, TypeError):
                    raise RuntimeError(f"QMT日期解析失败: {symbol}")

        df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

        if df.empty:
            raise RuntimeError(f"QMT无不复权数据: {symbol} ({start_date}~{end_date})")

        # 离线/陈旧防护：避免残缺数据覆盖已有完整缓存
        self._guard_against_stale_data(symbol, start_date, end_date, df, 'QMTDataProcessor_Raw')

        return self.preprocess_data(df)

    def get_tick_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取tick逐笔数据（不走smart_cache，使用pickle按年存储）

        tick数据包含列表列(askPrice/bidPrice/askVol/bidVol)，无法写入parquet，
        因此绕过smart_cache，直接下载并按年保存为pickle文件。

        缓存路径: .cache/QMTData/market_tick/{symbol}/{year}_tick.pkl
        """
        import pickle as _pickle

        if not self.xtdata:
            raise RuntimeError("QMT (xtquant) 未安装或未连接")

        start_time = start_date.replace('-', '')
        end_time = end_date.replace('-', '')

        # 缓存目录
        cache_base = cache_manager.disk_cache.cache_dir / 'QMTData' / 'market_tick' / symbol
        cache_base.mkdir(parents=True, exist_ok=True)

        # 计算请求涉及的年份
        try:
            req_start_year = pd.Timestamp(start_date).year
            req_end_year = pd.Timestamp(end_date).year
        except Exception:
            req_start_year = 1990
            req_end_year = 2099
        req_years = list(range(req_start_year, req_end_year + 1))

        # 检查已有缓存，收集缺失年份
        cached_frames = []
        missing_years = []
        for year in req_years:
            cache_file = cache_base / f"{year}_tick.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        df = _pickle.load(f)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        cached_frames.append(df)
                        continue
                except Exception:
                    pass
            missing_years.append(year)

        # 只下载缺失年份的数据
        if missing_years:
            self.logger.info(f"{symbol} tick: 缓存缺失年份={sorted(missing_years)}, 正在下载...")
            self.xtdata.download_history_data(
                stock_code=symbol, period='tick',
                start_time=start_time, end_time=end_time,
                incrementally=False
            )

            history_data = self.xtdata.get_market_data_ex(
                [], [symbol], period='tick',
                start_time=start_time, end_time=end_time,
                count=-1, dividend_type="none"
            )

            if symbol not in history_data or history_data[symbol].empty:
                if not cached_frames:
                    raise RuntimeError(f"QMT无tick数据: {symbol} ({start_date}~{end_date})")
                self.logger.warning(f"{symbol} tick: 缺失年份无数据")
                result = pd.concat(cached_frames).sort_index()
                return result[~result.index.duplicated(keep='last')]

            df = history_data[symbol]

            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S')
                except (ValueError, TypeError):
                    try:
                        df.index = pd.to_datetime(df.index, format='%Y%m%d')
                    except (ValueError, TypeError):
                        raise RuntimeError(f"QMT日期解析失败: {symbol}")

            # 按年拆分并保存
            for year in sorted(df.index.year.unique()):
                if year not in missing_years:
                    continue
                year_df = df[df.index.year == year]
                if year_df.empty:
                    continue
                cache_file = cache_base / f"{year}_tick.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        _pickle.dump(year_df, f, protocol=_pickle.HIGHEST_PROTOCOL)
                    self.logger.debug(f"{symbol} tick: 保存 {year} 年数据 ({len(year_df)} 条)")
                except Exception as e:
                    self.logger.warning(f"{symbol} tick: 保存 {year} 年缓存失败: {e}")
                cached_frames.append(year_df)

        if not cached_frames:
            raise RuntimeError(f"QMT无tick数据: {symbol} ({start_date}~{end_date})")

        result = pd.concat(cached_frames).sort_index()
        result = result[~result.index.duplicated(keep='last')]
        result = result[(result.index >= pd.Timestamp(start_date)) & (result.index <= pd.Timestamp(end_date))]

        if result.empty:
            raise RuntimeError(f"QMT无tick数据: {symbol} ({start_date}~{end_date})")

        return result

