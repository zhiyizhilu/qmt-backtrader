import os
import re
import time
import hashlib
import threading
import logging
import inspect
from typing import Dict, Any, Optional, Callable, Tuple, List
from functools import wraps
from pathlib import Path
import pandas as pd

from core.cache.mem_cache import MemCache
from core.cache.disk_cache import DiskCache, HAS_PYARROW
from core.cache.index_manager import CacheIndexManager


class SmartCacheManager:
    """智能缓存核心调度器（支持内存+磁盘+增量合并+按年份分片）

    V2 核心变化:
    - 行情数据按年份分片存储，一个股票一年一个文件
    - 财报数据按年份分片存储，一个股票一年一个文件
    - 通过 CacheIndexManager 维护索引，快速判断缓存是否可用
    - 保持向后兼容，旧格式文件仍可读取
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                instance = super(SmartCacheManager, cls).__new__(cls)
                instance._init_done = False
                cls._instance = instance
            return cls._instance

    def __init__(self):
        if self._init_done:
            return
        self.base_cache_dir = os.environ.get('QMT_CACHE_DIR', os.path.join(os.getcwd(), '.cache'))
        self.mem_cache = MemCache(capacity=int(os.environ.get('QMT_MEM_CACHE_LIMIT', 500)))
        self.disk_cache = DiskCache(self.base_cache_dir)
        self.index_manager = CacheIndexManager(Path(self.base_cache_dir))
        self.stats = {
            'mem_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'incremental_merges': 0,
            'yearly_hits': 0,
            'total_load_time_ms': 0.0
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_done = True

    def configure(self, cache_dir: str, mem_limit: int = 500):
        self.base_cache_dir = cache_dir
        self.disk_cache = DiskCache(self.base_cache_dir)
        self.index_manager = CacheIndexManager(Path(self.base_cache_dir))
        self.mem_cache.capacity = mem_limit
        self.logger.info(f"缓存系统配置更新: 目录={cache_dir}, 内存限制={mem_limit}")

    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        generic_funcs = {'get_data', 'get_financial_data', 'get_dividend_data'}
        key_parts = [] if func_name in generic_funcs else [func_name]
        skip_keys = {'start_date', 'end_date', 'start_time', 'end_time'}
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')

        def _format_arg(v):
            if isinstance(v, (list, tuple)):
                if len(v) > 3:
                    s = str(sorted([str(x) for x in v]))
                    h = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
                    return f"list_{len(v)}_{h}"
                return str(v)
            if isinstance(v, dict):
                s = str(sorted(v.items()))
                h = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
                return f"dict_{len(v)}_{h}"
            return str(v)

        for arg in args:
            if isinstance(arg, (pd.DataFrame, pd.Series)):
                continue
            if not isinstance(arg, (str, int, float, bool, list, tuple, dict, type(None))):
                continue
            if isinstance(arg, str) and date_pattern.match(arg):
                continue
            key_parts.append(_format_arg(arg))

        for k, v in sorted(kwargs.items()):
            if k not in skip_keys and not isinstance(v, (pd.DataFrame, pd.Series)):
                key_parts.append(f"{k}={_format_arg(v)}")

        return "_".join(key_parts)

    def _get_param_value(self, func: Callable, args: tuple, kwargs: dict, param_name: str, default: Any = '') -> Any:
        if param_name in kwargs:
            return kwargs[param_name]
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if param_name in params:
                idx = params.index(param_name)
                if idx < len(args):
                    return args[idx]
        except (ValueError, TypeError):
            pass
        return default

    def _build_incremental_args(self, func: Callable, args: tuple, kwargs: dict,
                                 start_date_override: Optional[str] = None,
                                 end_date_override: Optional[str] = None) -> Tuple[tuple, dict]:
        new_kwargs = dict(kwargs)
        if start_date_override is not None:
            new_kwargs['start_date'] = start_date_override
        if end_date_override is not None:
            new_kwargs['end_date'] = end_date_override

        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
        except (ValueError, TypeError) as e:
            self.logger.warning(f"[增量参数] 无法获取函数签名: {e}, func={getattr(func, '__name__', func)}, 回退简单处理")
            return (), new_kwargs

        override_params = set()
        if start_date_override is not None:
            override_params.add('start_date')
        if end_date_override is not None:
            override_params.add('end_date')

        min_override_idx = None
        for i, name in enumerate(params):
            if name in override_params:
                min_override_idx = i
                break

        if min_override_idx is None:
            return args, new_kwargs

        if min_override_idx >= len(args):
            return args, new_kwargs

        new_args = args[:min_override_idx]
        for i in range(min_override_idx, len(args)):
            param_name = params[i] if i < len(params) else None
            if param_name and param_name not in new_kwargs:
                new_kwargs[param_name] = args[i]

        return new_args, new_kwargs

    def _merge_incremental_data(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        if old_df is None or old_df.empty:
            return new_df
        if new_df is None or new_df.empty:
            return old_df
        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)
        if old_cols != new_cols:
            common_cols = list(old_cols & new_cols)
            if common_cols:
                old_df = old_df[common_cols]
                new_df = new_df[common_cols]
            else:
                self.logger.warning(f"增量数据与缓存数据列完全不同，放弃合并: 旧={list(old_cols)}, 新={list(new_cols)}")
                return old_df
        merged = pd.concat([old_df, new_df])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged.sort_index(inplace=True)
        return merged

    def _parse_years_from_range(self, start_date: str, end_date: str) -> List[int]:
        """从日期范围解析出涉及的年份列表"""
        try:
            start_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year
            return list(range(start_year, end_year + 1))
        except Exception:
            return []

    def execute_with_cache(self, namespace: str, cache_type: str, incremental: bool,
                           func: Callable, args: tuple, kwargs: dict) -> Any:
        """缓存执行入口 - 优先使用按年份分片的新逻辑"""
        if cache_type == 'market' and incremental:
            return self._execute_market_cache_v2(namespace, func, args, kwargs)

        start_time = time.time()
        format_type = 'parquet' if HAS_PYARROW else 'pkl'
        base_key = self._get_cache_key(func.__name__, *args, **kwargs)
        full_key = f"{namespace}_{base_key}"

        if not incremental:
            time_params = []
            for tk in ['start_date', 'end_date', 'start_time', 'end_time']:
                if tk in kwargs:
                    time_params.append(f"{tk}={kwargs[tk]}")
            if time_params:
                base_key = f"{base_key}_" + "_".join(time_params)
                full_key = f"{namespace}_{base_key}"

        mem_data = self.mem_cache.get(full_key)
        if mem_data is not None:
            self.stats['mem_hits'] += 1
            self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
            if incremental and isinstance(mem_data, pd.DataFrame) and not mem_data.empty:
                req_start = self._get_param_value(func, args, kwargs, 'start_date', '')
                req_end = self._get_param_value(func, args, kwargs, 'end_date', '')
                mem_end = mem_data.index.max().strftime('%Y-%m-%d') if isinstance(mem_data.index, pd.DatetimeIndex) else str(mem_data.index.max())
                if req_end and mem_end >= req_end:
                    return self._filter_by_date(mem_data, req_start, req_end)
            else:
                return mem_data

        disk_data = None
        disk_key = base_key
        if incremental and cache_type == 'market':
            found = self.disk_cache.find_by_prefix(namespace, base_key, format_type)
            if found is not None:
                disk_key, disk_data = found
        else:
            disk_data = self.disk_cache.get(namespace, base_key, format_type)

        if disk_data is not None:
            self.mem_cache.put(full_key, disk_data)
            self.stats['disk_hits'] += 1

            if incremental and isinstance(disk_data, pd.DataFrame) and not disk_data.empty:
                req_start = self._get_param_value(func, args, kwargs, 'start_date', '')
                req_end = self._get_param_value(func, args, kwargs, 'end_date', '')
                disk_start = disk_data.index.min().strftime('%Y-%m-%d') if isinstance(disk_data.index, pd.DatetimeIndex) else str(disk_data.index.min())
                disk_end = disk_data.index.max().strftime('%Y-%m-%d') if isinstance(disk_data.index, pd.DatetimeIndex) else str(disk_data.index.max())

                if req_start and req_end and disk_start <= req_start and disk_end >= req_end:
                    self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                    return self._filter_by_date(disk_data, req_start, req_end)

                self.logger.info(f"[{namespace}] 触发智能增量更新: {base_key}, 本地范围 {disk_start}~{disk_end}, 需求范围 {req_start}~{req_end}")
                self.stats['incremental_merges'] += 1

                if pd.Timestamp(disk_start) > pd.Timestamp(req_start):
                    self.logger.debug(f"[{namespace}] 向前扩展: 从 {req_start} 到 {disk_start}")
                    inc_args, inc_kwargs = self._build_incremental_args(
                        func, args, kwargs,
                        start_date_override=req_start,
                        end_date_override=(pd.Timestamp(disk_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    )
                else:
                    next_day = (pd.Timestamp(disk_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    self.logger.debug(f"[{namespace}] 向后扩展: 从 {next_day} 到 {req_end}")
                    inc_args, inc_kwargs = self._build_incremental_args(
                        func, args, kwargs, start_date_override=next_day
                    )

                try:
                    new_data = func(*inc_args, **inc_kwargs)
                    merged_data = self._merge_incremental_data(disk_data, new_data)

                    new_disk_key = self._build_disk_key_with_dates(base_key, merged_data)
                    self.disk_cache.delete(namespace, disk_key, format_type)
                    self.disk_cache.put(namespace, new_disk_key, merged_data, format_type)
                    self.mem_cache.put(full_key, merged_data)

                    self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                    return self._filter_by_date(merged_data, req_start, req_end)
                except Exception as e:
                    self.logger.warning(f"增量更新失败，退回全量下载: {e}")
            elif not incremental or not (isinstance(disk_data, pd.DataFrame) and not disk_data.empty):
                self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                return disk_data

        self.stats['misses'] += 1
        self.logger.info(f"[{namespace}] 缓存未命中，全量获取: {base_key}")

        try:
            result = func(*args, **kwargs)
            if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                if incremental and cache_type == 'market' and isinstance(result, pd.DataFrame):
                    disk_write_key = self._build_disk_key_with_dates(base_key, result)
                else:
                    disk_write_key = base_key
                self.disk_cache.put(namespace, disk_write_key, result, format_type)
                self.mem_cache.put(full_key, result)

            self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
            return result
        except Exception as e:
            self.logger.error(f"[{namespace}] 获取数据失败: {e}")
            raise

    def _execute_market_cache_v2(self, namespace: str, func: Callable,
                                  args: tuple, kwargs: dict) -> Any:
        """行情数据按年份分片缓存逻辑

        核心流程:
        1. 从请求参数中提取 symbol, period, start_date, end_date
        2. 计算请求涉及的年份列表
        3. 查索引/磁盘，判断哪些年份已有缓存
        4. 对缺失年份调用原始函数获取数据
        5. 按年份写入磁盘，更新索引
        6. 合并所有年份的数据并返回
        """
        start_time = time.time()
        symbol = self._get_param_value(func, args, kwargs, 'symbol', '')
        period = self._get_param_value(func, args, kwargs, 'period', '1d')
        req_start = self._get_param_value(func, args, kwargs, 'start_date', '')
        req_end = self._get_param_value(func, args, kwargs, 'end_date', '')
        skip_current_year_refresh = kwargs.pop('skip_current_year_refresh', False)

        if not symbol or not req_start or not req_end:
            return func(*args, **kwargs)

        is_raw = namespace.endswith('_Raw')
        mem_key = f"{namespace}_market_{symbol}_{period}_{req_start}_{req_end}"

        mem_data = self.mem_cache.get(mem_key)
        if mem_data is not None:
            self.stats['mem_hits'] += 1
            self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
            return mem_data

        req_years = self._parse_years_from_range(req_start, req_end)
        if not req_years:
            return func(*args, **kwargs)

        cached_frames, truly_missing = self._check_cache_coverage(
            namespace, symbol, period, req_years, req_start, req_end, is_raw,
            skip_current_year_refresh
        )

        if truly_missing:
            new_frames = self._fetch_missing_years(
                namespace, func, args, kwargs, symbol, period,
                truly_missing, req_years, req_start, req_end, is_raw
            )
            cached_frames.extend(new_frames)

        if not cached_frames:
            result = self._full_fetch_fallback(
                namespace, func, args, kwargs, symbol, period, is_raw, mem_key
            )
            self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
            return result

        result = self._merge_and_return(cached_frames, req_start, req_end, mem_key)
        self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
        return result

    def _check_cache_coverage(self, namespace: str, symbol: str, period: str,
                               req_years: List[int], req_start: str,
                               req_end: str, is_raw: bool,
                               skip_current_year_refresh: bool = False) -> Tuple[list, set]:
        """检查缓存覆盖情况，返回 (cached_frames, truly_missing)

        从索引和磁盘判断哪些年份已有缓存，并验证已有缓存是否覆盖请求范围。
        如果缓存完全覆盖请求范围，truly_missing 为空集。
        """
        idx = self.index_manager

        available_years = (idx.get_available_market_raw_years(symbol, period) if is_raw
                          else idx.get_available_market_years(symbol, period))
        if not available_years:
            available_years = self.disk_cache.list_yearly_files(namespace, symbol, period)

        cached_years = set(available_years) & set(req_years)
        missing_years = set(req_years) - cached_years

        checked_years = set((idx.get_checked_market_raw_years(symbol, period) if is_raw
                             else idx.get_checked_market_years(symbol, period)))
        truly_missing = missing_years - checked_years

        cached_frames = []
        actually_cached_years = set()
        if cached_years:
            for year in sorted(cached_years):
                df = self.disk_cache.get_yearly(namespace, symbol, year, period)
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    cached_frames.append(df)
                    actually_cached_years.add(year)
                    self.stats['yearly_hits'] += 1
                else:
                    self.logger.info(f"[{namespace}] {symbol} 索引声称有{year}年缓存但文件缺失，将重新获取")

            stale_index_years = cached_years - actually_cached_years
            if stale_index_years:
                stale_index_years -= checked_years
                if stale_index_years:
                    truly_missing |= stale_index_years
                    self.logger.info(f"[{namespace}] {symbol} 索引与实际缓存不同步，缺失年份: {sorted(stale_index_years)}")

        if not truly_missing:
            if cached_frames:
                if skip_current_year_refresh and not (missing_years - checked_years):
                    pass
                else:
                    coverage_missing = self._verify_cache_date_coverage(
                        namespace, symbol, pd.concat(cached_frames), req_years, req_start, req_end, checked_years,
                        skip_current_year_refresh
                    )
                    if coverage_missing == set(req_years):
                        cached_frames = []
                    elif coverage_missing:
                        cached_frames = [f for f in cached_frames
                                         if not (isinstance(f.index, pd.DatetimeIndex) and
                                                 f.index.year.unique().tolist() and
                                                 set(f.index.year.unique()) & coverage_missing)]
                    truly_missing = coverage_missing
            else:
                if missing_years and missing_years == checked_years:
                    self.logger.info(f"[{namespace}] {symbol} 所有年份已检查且无数据，跳过: {sorted(missing_years)}")
                else:
                    truly_missing = set(req_years)
                    self.logger.info(f"[{namespace}] {symbol} 缓存数据为空，重新获取: {sorted(truly_missing)}")

        return cached_frames, truly_missing

    def _verify_cache_date_coverage(self, namespace: str, symbol: str,
                                     cached_df: pd.DataFrame, req_years: List[int],
                                     req_start: str, req_end: str,
                                     checked_years: set,
                                     skip_current_year_refresh: bool = False) -> set:
        """验证缓存数据是否覆盖请求的日期范围，返回仍需获取的年份集合

        返回空集表示缓存完全覆盖，无需增量获取。
        返回 set(req_years) 表示缓存不可用，需全量重新获取。
        """
        if not isinstance(cached_df.index, pd.DatetimeIndex):
            self.logger.info(f"[{namespace}] {symbol} 缓存索引非时间类型，重新获取: {sorted(req_years)}")
            return set(req_years)

        disk_start = cached_df.index.min().strftime('%Y-%m-%d')
        disk_end = cached_df.index.max().strftime('%Y-%m-%d')

        missing = set()
        if disk_start > req_start:
            first_cached_year = cached_df.index.min().year
            start_missing = set(y for y in req_years if y < first_cached_year)
            start_missing -= checked_years
            missing |= start_missing
            if start_missing:
                self.logger.info(
                    f"[{namespace}] {symbol} 缓存未覆盖起始日期: "
                    f"缓存起始={disk_start}, 请求起始={req_start}, 缺失年份={sorted(start_missing)}"
                )
            else:
                self.logger.debug(
                    f"[{namespace}] {symbol} 缓存未覆盖起始日期(已检查): "
                    f"缓存起始={disk_start}, 请求起始={req_start}"
                )

        if disk_end < req_end:
            last_cached_year = cached_df.index.max().year
            end_missing = set(y for y in req_years if y > last_cached_year)
            end_missing -= checked_years
            if not end_missing and not skip_current_year_refresh:
                end_missing = {max(req_years)} - checked_years
            missing |= end_missing
            if end_missing:
                self.logger.info(
                    f"[{namespace}] {symbol} 缓存未覆盖结束日期: "
                    f"缓存截止={disk_end}, 请求截止={req_end}, 增量获取年份={sorted(end_missing)}"
                )
            else:
                self.logger.debug(
                    f"[{namespace}] {symbol} 缓存未覆盖结束日期(已检查): "
                    f"缓存截止={disk_end}, 请求截止={req_end}"
                )

        if disk_end >= req_end and not skip_current_year_refresh:
            current_year = pd.Timestamp.now().year
            for y in req_years:
                if y == current_year and y not in missing and y not in checked_years:
                    today = pd.Timestamp.now().normalize()
                    if pd.Timestamp(disk_end) < today - pd.Timedelta(days=2):
                        missing |= {y}
                        self.logger.info(
                            f"[{namespace}] {symbol} 当年缓存落后于今天: "
                            f"缓存截止={disk_end}, 今天={today.strftime('%Y-%m-%d')}, 需刷新{y}年"
                        )

        if not skip_current_year_refresh:
            gap_missing = self._detect_date_gaps(namespace, symbol, cached_df, req_years, checked_years)
            if gap_missing:
                missing |= gap_missing

        return missing

    def _detect_date_gaps(self, namespace: str, symbol: str,
                          cached_df: pd.DataFrame, req_years: List[int],
                          checked_years: set) -> set:
        """检测缓存数据中的日期空洞，返回有空洞的年份集合

        正常交易日间隔不超过5天（含周末），超过15天视为异常空洞。
        空洞所在年份需要重新获取。
        """
        if len(cached_df) < 2:
            return set()

        missing = set()
        dates = cached_df.index
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i - 1]).days
            if gap_days > 15:
                gap_year_start = dates[i - 1].year
                gap_year_end = dates[i].year
                gap_years = set()
                for y in req_years:
                    if gap_year_start <= y <= gap_year_end:
                        gap_years.add(y)
                gap_years -= checked_years
                if gap_years:
                    missing |= gap_years
                    self.logger.info(
                        f"[{namespace}] {symbol} 检测到日期空洞: "
                        f"{dates[i-1].strftime('%Y-%m-%d')} ~ {dates[i].strftime('%Y-%m-%d')} "
                        f"(间隔{gap_days}天), 需重新获取年份={sorted(gap_years)}"
                    )

        return missing

    def _fetch_missing_years(self, namespace: str, func: Callable, args: tuple,
                              kwargs: dict, symbol: str, period: str,
                              truly_missing: set, req_years: List[int],
                              req_start: str, req_end: str,
                              is_raw: bool) -> list:
        """获取缺失年份数据，写入磁盘并更新索引，返回新数据帧列表

        对于后复权数据(is_raw=False)，采用重叠校验策略：
        1. 先获取缺失年份数据（含少量重叠日期）
        2. 对比重叠日期的后复权价，判断调整因子是否变化
        3. 如果调整因子未变 → 安全追加新数据
        4. 如果调整因子已变 → 全量重获取所有年份
        """
        idx = self.index_manager
        self.stats['incremental_merges'] += 1
        actual_cached = sorted(set(req_years) - truly_missing)

        if not is_raw and actual_cached:
            return self._fetch_hfq_with_overlap_check(
                namespace, func, args, kwargs, symbol, period,
                truly_missing, req_years, req_start, req_end
            )

        self.logger.info(
            f"[{namespace}] {symbol} 行情增量更新: "
            f"已有年份={actual_cached}, 缺失年份={sorted(truly_missing)}"
        )

        min_missing = min(truly_missing)
        max_missing = max(truly_missing)

        inc_start = f"{min_missing}-01-01"
        inc_end = f"{max_missing}-12-31"

        inc_args, inc_kwargs = self._build_incremental_args(
            func, args, kwargs,
            start_date_override=inc_start,
            end_date_override=inc_end
        )

        new_frames = []
        try:
            new_data = func(*inc_args, **inc_kwargs)
            if new_data is not None and isinstance(new_data, pd.DataFrame) and not new_data.empty:
                written = self.disk_cache.put_yearly_from_df(namespace, symbol, period, new_data)
                for y in written:
                    if is_raw:
                        idx.update_market_raw_index(symbol, period, y)
                    else:
                        idx.update_market_index(symbol, period, y)
                new_frames.append(new_data)
                fetched_years_with_data = (set(new_data.index.year.unique())
                                           if isinstance(new_data.index, pd.DatetimeIndex)
                                           else set())
                no_data_years = truly_missing - fetched_years_with_data
                checked_gap_years = truly_missing & fetched_years_with_data
                if checked_gap_years:
                    if is_raw:
                        idx.update_checked_market_raw_years(symbol, period, sorted(checked_gap_years))
                    else:
                        idx.update_checked_market_years(symbol, period, sorted(checked_gap_years))
            else:
                no_data_years = truly_missing
            if no_data_years:
                if is_raw:
                    idx.update_checked_market_raw_years(symbol, period, sorted(no_data_years))
                else:
                    idx.update_checked_market_years(symbol, period, sorted(no_data_years))
        except Exception as e:
            self.logger.warning(f"[{namespace}] {symbol} 增量获取失败: {e}")
            if is_raw:
                idx.update_checked_market_raw_years(symbol, period, sorted(truly_missing))
            else:
                idx.update_checked_market_years(symbol, period, sorted(truly_missing))

        return new_frames

    def _fetch_hfq_with_overlap_check(self, namespace: str, func: Callable, args: tuple,
                                       kwargs: dict, symbol: str, period: str,
                                       truly_missing: set, req_years: List[int],
                                       req_start: str, req_end: str) -> list:
        """后复权数据增量更新：重叠校验策略

        后复权价格 = 原始价格 × 累计调整因子，调整因子随除权除息事件变化。
        如果期间没有新的除权除息事件，调整因子不变，可以安全追加。
        如果有新的除权除息事件，调整因子变化，必须全量重获取。

        策略：
        1. 检查缺失年份是在已有年份前面还是后面
        2. 如果有缺失年份在已有年份前面 → 直接全量重获取
        3. 如果缺失年份只在已有年份后面 → 重叠校验
           a. 从已有缓存的最后一天的前几天开始获取新数据（制造重叠）
           b. 对比重叠日期的后复权价
           c. 一致 → 调整因子未变 → 只写入新年份数据
           d. 不一致 → 调整因子已变 → 全量重获取所有年份
        """
        idx = self.index_manager
        all_years = sorted(req_years)
        actual_cached = sorted(set(req_years) - truly_missing)

        if not actual_cached:
            return self._fetch_hfq_full_range(
                namespace, func, args, kwargs, symbol, period,
                truly_missing, req_years, req_start, req_end
            )

        min_cached = min(actual_cached)
        missing_before = any(y < min_cached for y in truly_missing)

        if missing_before:
            self.logger.info(
                f"[{namespace}] {symbol} 检测到缺失年份在已有年份前面，触发全量重获取: "
                f"已有年份={actual_cached}, 缺失年份={sorted(truly_missing)}"
            )
            return self._fetch_hfq_full_range(
                namespace, func, args, kwargs, symbol, period,
                truly_missing, req_years, req_start, req_end
            )

        overlap_start, cached_last_date = self._get_hfq_overlap_start(
            namespace, symbol, period, actual_cached
        )

        if overlap_start is None:
            return self._fetch_hfq_full_range(
                namespace, func, args, kwargs, symbol, period,
                truly_missing, req_years, req_start, req_end
            )

        min_missing = min(truly_missing)
        max_missing = max(truly_missing)
        inc_start = max(overlap_start, f"{min_missing}-01-01")
        inc_end = f"{max_missing}-12-31"

        self.logger.info(
            f"[{namespace}] {symbol} 后复权增量更新(含重叠校验): "
            f"已有年份={actual_cached}, 缺失年份={sorted(truly_missing)}, "
            f"获取范围={inc_start}~{inc_end}, 重叠校验日期<={cached_last_date}"
        )

        inc_args, inc_kwargs = self._build_incremental_args(
            func, args, kwargs,
            start_date_override=inc_start,
            end_date_override=inc_end
        )

        new_frames = []
        try:
            new_data = func(*inc_args, **inc_kwargs)
            if new_data is None or not isinstance(new_data, pd.DataFrame) or new_data.empty:
                idx.update_checked_market_years(symbol, period, sorted(truly_missing))
                return new_frames

            factor_changed = self._check_hfq_factor_change(
                namespace, symbol, period, new_data, cached_last_date
            )

            if factor_changed:
                self.logger.info(
                    f"[{namespace}] {symbol} 检测到调整因子变化，触发全量重获取"
                )
                written = self.disk_cache.put_yearly_from_df(
                    namespace, symbol, period, new_data,
                    only_years=sorted(new_data.index.year.unique())
                )
                for y in written:
                    idx.update_market_index(symbol, period, y)
                return self._fetch_hfq_full_range(
                    namespace, func, args, kwargs, symbol, period,
                    truly_missing, req_years, req_start, req_end
                )

            self.logger.info(
                f"[{namespace}] {symbol} 调整因子未变化，安全追加新数据"
            )
            new_only_years = sorted(truly_missing & set(new_data.index.year.unique()))
            written = self.disk_cache.put_yearly_from_df(
                namespace, symbol, period, new_data,
                only_years=new_only_years if new_only_years else None
            )
            for y in written:
                idx.update_market_index(symbol, period, y)
            new_frames.append(new_data)

            fetched_years_with_data = (set(new_data.index.year.unique())
                                       if isinstance(new_data.index, pd.DatetimeIndex)
                                       else set())
            checked_gap_years = truly_missing & fetched_years_with_data
            if checked_gap_years:
                idx.update_checked_market_years(symbol, period, sorted(checked_gap_years))
            no_data_years = truly_missing - fetched_years_with_data
            if no_data_years:
                idx.update_checked_market_years(symbol, period, sorted(no_data_years))

        except Exception as e:
            self.logger.warning(f"[{namespace}] {symbol} 后复权增量更新失败: {e}")
            idx.update_checked_market_years(symbol, period, sorted(truly_missing))

        return new_frames

    def _get_hfq_overlap_start(self, namespace: str, symbol: str, period: str,
                                cached_years: List[int]) -> Tuple[Optional[str], Optional[str]]:
        """获取后复权重叠校验的起始日期

        从已有缓存的最后一年中找到最后一个交易日，向前取5个交易日作为重叠起始。
        返回 (overlap_start, cached_last_date)，如果无法获取则返回 (None, None)。
        """
        if not cached_years:
            return None, None

        last_cached_year = max(cached_years)
        df = self.disk_cache.get_yearly(namespace, symbol, last_cached_year, period)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None, None
        if not isinstance(df.index, pd.DatetimeIndex):
            return None, None

        sorted_df = df.sort_index()
        if len(sorted_df) < 2:
            last_date = sorted_df.index[-1].strftime('%Y-%m-%d')
            overlap_start = (sorted_df.index[-1] - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
            return overlap_start, last_date

        overlap_idx = max(0, len(sorted_df) - 6)
        overlap_start = sorted_df.index[overlap_idx].strftime('%Y-%m-%d')
        cached_last_date = sorted_df.index[-1].strftime('%Y-%m-%d')
        return overlap_start, cached_last_date

    def _check_hfq_factor_change(self, namespace: str, symbol: str, period: str,
                                  new_data: pd.DataFrame,
                                  cached_last_date: str) -> bool:
        """检查后复权调整因子是否变化

        对比新获取数据和已有缓存中重叠日期的后复权价。
        如果差异超过 0.1%，认为调整因子已变化。
        """
        if not isinstance(new_data.index, pd.DatetimeIndex):
            return True

        overlap_data = new_data[new_data.index <= cached_last_date]
        if overlap_data.empty:
            self.logger.info(
                f"[{namespace}] {symbol} 无重叠数据可校验，保守判定调整因子已变化"
            )
            return True

        all_years = sorted(overlap_data.index.year.unique())
        cached_frames = []
        for year in all_years:
            df = self.disk_cache.get_yearly(namespace, symbol, year, period)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                cached_frames.append(df)

        if not cached_frames:
            return True

        cached_merged = pd.concat(cached_frames).sort_index()
        cached_merged = cached_merged[~cached_merged.index.duplicated(keep='last')]

        common_dates = overlap_data.index.intersection(cached_merged.index)
        if len(common_dates) == 0:
            self.logger.info(
                f"[{namespace}] {symbol} 无公共日期可校验，保守判定调整因子已变化"
            )
            return True

        for date in common_dates:
            old_close = float(cached_merged.loc[date, 'close'])
            new_close = float(overlap_data.loc[date, 'close'])
            if old_close <= 0:
                continue
            diff_pct = abs(new_close - old_close) / old_close
            if diff_pct > 0.001:
                self.logger.info(
                    f"[{namespace}] {symbol} 调整因子变化检测: "
                    f"日期={date.strftime('%Y-%m-%d')}, "
                    f"旧价={old_close:.4f}, 新价={new_close:.4f}, "
                    f"差异={diff_pct*100:.2f}%"
                )
                return True

        return False

    def _fetch_hfq_full_range(self, namespace: str, func: Callable, args: tuple,
                               kwargs: dict, symbol: str, period: str,
                               truly_missing: set, req_years: List[int],
                               req_start: str, req_end: str) -> list:
        """后复权数据全量重获取策略（调整因子变化时的兜底方案）

        从最早年份到最晚年份一次性获取全部后复权数据，覆盖旧缓存，
        确保所有年份使用同一个调整因子。
        """
        idx = self.index_manager
        all_years = sorted(req_years)
        self.logger.info(
            f"[{namespace}] {symbol} 后复权数据全量重获取: "
            f"获取范围={all_years[0]}~{all_years[-1]}"
        )

        inc_start = f"{all_years[0]}-01-01"
        inc_end = f"{all_years[-1]}-12-31"

        inc_args, inc_kwargs = self._build_incremental_args(
            func, args, kwargs,
            start_date_override=inc_start,
            end_date_override=inc_end
        )

        new_frames = []
        try:
            new_data = func(*inc_args, **inc_kwargs)
            if new_data is not None and isinstance(new_data, pd.DataFrame) and not new_data.empty:
                written = self.disk_cache.put_yearly_from_df(
                    namespace, symbol, period, new_data,
                    only_years=all_years
                )
                for y in written:
                    idx.update_market_index(symbol, period, y)
                new_frames.append(new_data)
                fetched_years_with_data = (set(new_data.index.year.unique())
                                           if isinstance(new_data.index, pd.DatetimeIndex)
                                           else set())
                no_data_years = truly_missing - fetched_years_with_data
                checked_gap_years = truly_missing & fetched_years_with_data
                if checked_gap_years:
                    idx.update_checked_market_years(symbol, period, sorted(checked_gap_years))
                all_checked = set(all_years) - fetched_years_with_data
                if all_checked:
                    idx.update_checked_market_years(symbol, period, sorted(all_checked))
            else:
                idx.update_checked_market_years(symbol, period, sorted(truly_missing))
        except Exception as e:
            self.logger.warning(f"[{namespace}] {symbol} 后复权全量获取失败: {e}")
            idx.update_checked_market_years(symbol, period, sorted(truly_missing))

        return new_frames

    def _full_fetch_fallback(self, namespace: str, func: Callable, args: tuple,
                              kwargs: dict, symbol: str, period: str,
                              is_raw: bool, mem_key: str) -> Any:
        """缓存完全未命中时的全量获取兜底"""
        idx = self.index_manager
        self.stats['misses'] += 1
        self.logger.info(f"[{namespace}] {symbol} 缓存未命中，全量获取")
        try:
            result = func(*args, **kwargs)
            if result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                written = self.disk_cache.put_yearly_from_df(namespace, symbol, period, result)
                for y in written:
                    if is_raw:
                        idx.update_market_raw_index(symbol, period, y)
                    else:
                        idx.update_market_index(symbol, period, y)
                self.mem_cache.put(mem_key, result)
            return result
        except Exception as e:
            self.logger.error(f"[{namespace}] 获取数据失败: {e}")
            raise

    def _merge_and_return(self, cached_frames: list, req_start: str,
                           req_end: str, mem_key: str) -> pd.DataFrame:
        """合并数据帧、按日期过滤、写入内存缓存并返回"""
        if len(cached_frames) == 1:
            merged = cached_frames[0]
        else:
            merged = self._merge_incremental_data(cached_frames[0], cached_frames[1])
            for i in range(2, len(cached_frames)):
                merged = self._merge_incremental_data(merged, cached_frames[i])

        result = self._filter_by_date(merged, req_start, req_end)
        self.mem_cache.put(mem_key, result)
        self.stats['disk_hits'] += 1
        return result

    def _build_disk_key_with_dates(self, base_key: str, df: pd.DataFrame) -> str:
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return base_key
        start = df.index.min().strftime('%Y%m%d')
        end = df.index.max().strftime('%Y%m%d')
        return f"{base_key}_{start}_{end}"

    def _filter_by_date(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        filtered = df
        if start_date:
            filtered = filtered[filtered.index >= start_date]
        if end_date:
            filtered = filtered[filtered.index <= end_date]
        return filtered

    def get_stats_report(self) -> str:
        total_requests = self.stats['mem_hits'] + self.stats['disk_hits'] + self.stats['misses']
        hit_rate = (self.stats['mem_hits'] + self.stats['disk_hits']) / total_requests if total_requests > 0 else 0
        avg_time = self.stats['total_load_time_ms'] / total_requests if total_requests > 0 else 0
        report = (
            f"=== 智能缓存系统性能报告 ===\n"
            f"总请求数: {total_requests}\n"
            f"内存命中: {self.stats['mem_hits']} 次\n"
            f"磁盘命中: {self.stats['disk_hits']} 次\n"
            f"年份命中: {self.stats['yearly_hits']} 次\n"
            f"未命中数: {self.stats['misses']} 次\n"
            f"增量合并: {self.stats['incremental_merges']} 次\n"
            f"综合命中率: {hit_rate:.2%}\n"
            f"平均加载耗时: {avg_time:.2f} ms/次\n"
            f"============================"
        )
        return report


cache_manager = SmartCacheManager()


def smart_cache(cache_type: str = 'market', incremental: bool = False):
    """智能数据缓存装饰器

    Args:
        cache_type: 'market' (行情数据) 或 'financial' (财务数据), 均优先使用Parquet
        incremental: 是否开启智能增量合并（仅针对带有 start_date, end_date 参数且返回 DataFrame 的函数有效）

    使用示例:
        @smart_cache(cache_type='market', incremental=True)
        def get_data(self, symbol, start_date, end_date):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            namespace = self.__class__.__name__
            if cache_type == 'financial':
                namespace = f"{namespace}_Financial"
            return cache_manager.execute_with_cache(
                namespace=namespace,
                cache_type=cache_type,
                incremental=incremental,
                func=func,
                args=(self,) + args,
                kwargs=kwargs
            )
        return wrapper
    return decorator
