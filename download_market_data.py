import argparse
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from core.cache import cache_manager
from core.data.index_constituent import IndexConstituentManager
from core.data.opendata import OpenDataProcessor
from core.data.qmt import QMTDataProcessor


def setup_logger(log_to_file: bool = False, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger('download_market_data')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    if log_to_file:
        if log_file is None:
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = f"{log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_download_market_data.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"日志将同时写入文件: {log_file}")

    return logger


logger = logging.getLogger('download_market_data')


class MissingType(Enum):
    NONE = 'none'
    CURRENT_YEAR = 'current_year'
    SUSPENSION = 'suspension'
    DATA_INCOMPLETE = 'data_incomplete'
    EXTRA_NON_TRADING = 'extra_non_trading'
    RAW_MARKET_MISMATCH = 'raw_market_mismatch'


@dataclass
class CheckResult:
    symbol: str
    year: str
    data_type: str
    missing_type: MissingType
    missing_dates: List[str] = field(default_factory=list)
    extra_dates: List[str] = field(default_factory=list)
    raw_market_mismatch: bool = False
    coverage: str = ""
    suspended_ranges: List[List[str]] = field(default_factory=list)
    raw_rows: int = 0
    market_rows: int = 0

class TradingDayCalendar:
    BENCHMARK_INDEX = '000300.SH'

    def __init__(self, opendata_processor: OpenDataProcessor):
        self._processor = opendata_processor
        self._trading_days: Dict[str, Set[str]] = {}
        self._trading_days_by_year: Dict[int, Set[str]] = {}

    def get_trading_days(self, start_date: str, end_date: str) -> Set[str]:
        cache_key = f"{start_date}_{end_date}"
        if cache_key in self._trading_days:
            return self._trading_days[cache_key]

        logger.info(f"获取交易日历基准: {self.BENCHMARK_INDEX} ({start_date} ~ {end_date})...")
        try:
            df = self._processor.get_data(
                self.BENCHMARK_INDEX, start_date, end_date, "1d"
            )
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                days = {d.strftime('%Y-%m-%d') for d in df.index}
                logger.info(f"交易日历获取成功: {len(days)} 个交易日 "
                            f"({sorted(days)[0]} ~ {sorted(days)[-1]})")
                self._trading_days[cache_key] = days
                self._build_by_year(days)
                return days
        except Exception as e:
            logger.warning(f"从OpenData获取交易日历失败: {e}")

        logger.info("回退到磁盘缓存获取交易日历...")
        days = self._load_trading_days_from_cache(start_date, end_date)
        if days:
            self._trading_days[cache_key] = days
            self._build_by_year(days)
            return days

        logger.error("无法获取交易日历，将跳过完整性校验")
        return set()

    def get_trading_days_by_year(self, start_year: int, end_year: int) -> Dict[str, Set[str]]:
        result = {}
        for year in range(start_year, end_year + 1):
            if year in self._trading_days_by_year:
                result[str(year)] = self._trading_days_by_year[year]
        return result

    def _build_by_year(self, days: Set[str]):
        for d in days:
            year = int(d[:4])
            if year not in self._trading_days_by_year:
                self._trading_days_by_year[year] = set()
            self._trading_days_by_year[year].add(d)

    def _load_trading_days_from_cache(self, start_date: str, end_date: str) -> Set[str]:
        namespace = 'OpenDataProcessor'
        req_years = cache_manager._parse_years_from_range(start_date, end_date)
        if not req_years:
            return set()

        frames = []
        for year in req_years:
            df = cache_manager.disk_cache.get_yearly(
                namespace, self.BENCHMARK_INDEX, year, '1d'
            )
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                frames.append(df)

        if not frames:
            return set()

        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        merged = merged[(merged.index >= start_ts) & (merged.index <= end_ts)]
        return {d.strftime('%Y-%m-%d') for d in merged.index}

class DataIntegrityChecker:
    SUSPENSION_GAP_THRESHOLD = 15
    MISSING_GAP_THRESHOLD = 8

    def __init__(self, trading_days: Set[str]):
        self._trading_days = sorted(trading_days)
        self._trading_day_set = trading_days

    def check_symbol(self, symbol: str, start_date: str, end_date: str,
                     data_type: str) -> Dict:
        namespace = 'OpenDataProcessor_Raw' if data_type == 'raw' else 'OpenDataProcessor'
        req_years = cache_manager._parse_years_from_range(start_date, end_date)
        if not req_years:
            return {'status': 'no_data', 'missing_years': set(req_years),
                    'gaps': [], 'suspended_ranges': []}

        frames = []
        for year in req_years:
            df = cache_manager.disk_cache.get_yearly(namespace, symbol, year, '1d')
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                frames.append(df)

        if not frames:
            return {'status': 'no_data', 'missing_years': set(req_years),
                    'gaps': [], 'suspended_ranges': []}

        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        merged = merged[(merged.index >= start_ts) & (merged.index <= end_ts)]

        if merged.empty:
            return {'status': 'no_data', 'missing_years': set(req_years),
                    'gaps': [], 'suspended_ranges': []}

        cached_dates = {d.strftime('%Y-%m-%d') for d in merged.index}
        missing_dates = self._trading_day_set - cached_dates

        suspended_ranges = cache_manager.index_manager.get_suspended_ranges(symbol)
        if suspended_ranges:
            valid_ranges = []
            changed = False
            for s_start, s_end in suspended_ranges:
                range_missing = sum(1 for d in missing_dates if s_start <= d <= s_end)
                if range_missing > 0:
                    valid_ranges.append([s_start, s_end])
                else:
                    changed = True
                    logger.debug(f"[{symbol}] 停牌区间 {s_start}~{s_end} 已有数据，清除标记")
            if changed:
                cache_manager.index_manager.mark_suspended(symbol, valid_ranges)

        if not missing_dates:
            return {'status': 'complete', 'missing_years': set(),
                    'gaps': [], 'suspended_ranges': []}

        gaps, suspended_ranges = self._classify_missing_dates(
            symbol, missing_dates, cached_dates
        )

        missing_years = set()
        for gap in gaps:
            for y in req_years:
                gap_start_year = pd.Timestamp(gap['start']).year
                gap_end_year = pd.Timestamp(gap['end']).year
                if gap_start_year <= y <= gap_end_year:
                    missing_years.add(y)

        return {
            'status': 'incomplete',
            'missing_years': missing_years,
            'gaps': gaps,
            'suspended_ranges': suspended_ranges,
            'total_missing': len(missing_dates),
            'total_trading_days': len(self._trading_day_set),
            'coverage': f"{len(cached_dates)}/{len(self._trading_day_set)}"
        }

    def check_symbol_year(self, symbol: str, year: str,
                          trading_days: Set[str],
                          market_dir: str, market_raw_dir: str) -> List[CheckResult]:
        results = []

        m_files = self._get_year_files(os.path.join(market_dir, symbol))
        r_files = self._get_year_files(os.path.join(market_raw_dir, symbol))

        df_m = self._read_parquet_safe(os.path.join(market_dir, symbol, m_files[year])) if year in m_files else None
        df_r = self._read_parquet_safe(os.path.join(market_raw_dir, symbol, r_files[year])) if year in r_files else None

        if df_r is not None:
            raw_dates = {d.strftime('%Y-%m-%d') for d in df_r.index}
            extra_in_raw = raw_dates - trading_days
            missing_in_raw = trading_days - raw_dates

            if extra_in_raw:
                results.append(CheckResult(
                    symbol=symbol, year=year, data_type='raw',
                    missing_type=MissingType.EXTRA_NON_TRADING,
                    extra_dates=sorted(extra_in_raw),
                    raw_rows=len(df_r), market_rows=len(df_m) if df_m is not None else 0,
                ))

            if missing_in_raw:
                mtype = self.classify_missing(sorted(missing_in_raw), trading_days, year)
                results.append(CheckResult(
                    symbol=symbol, year=year, data_type='raw',
                    missing_type=mtype,
                    missing_dates=sorted(missing_in_raw),
                    raw_rows=len(df_r), market_rows=len(df_m) if df_m is not None else 0,
                ))

        if df_m is not None:
            market_dates = {d.strftime('%Y-%m-%d') for d in df_m.index}
            extra_in_market = market_dates - trading_days
            missing_in_market = trading_days - market_dates

            if extra_in_market:
                results.append(CheckResult(
                    symbol=symbol, year=year, data_type='adjusted',
                    missing_type=MissingType.EXTRA_NON_TRADING,
                    extra_dates=sorted(extra_in_market),
                    raw_rows=len(df_r) if df_r is not None else 0, market_rows=len(df_m),
                ))

            if missing_in_market:
                mtype = self.classify_missing(sorted(missing_in_market), trading_days, year)
                results.append(CheckResult(
                    symbol=symbol, year=year, data_type='adjusted',
                    missing_type=mtype,
                    missing_dates=sorted(missing_in_market),
                    raw_rows=len(df_r) if df_r is not None else 0, market_rows=len(df_m),
                ))

        if df_m is not None and df_r is not None:
            raw_dates = {d.strftime('%Y-%m-%d') for d in df_r.index}
            market_dates = {d.strftime('%Y-%m-%d') for d in df_m.index}
            in_raw_not_market = raw_dates - market_dates
            in_market_not_raw = market_dates - raw_dates
            if in_raw_not_market or in_market_not_raw:
                results.append(CheckResult(
                    symbol=symbol, year=year, data_type='both',
                    missing_type=MissingType.RAW_MARKET_MISMATCH,
                    missing_dates=sorted(in_raw_not_market),
                    extra_dates=sorted(in_market_not_raw),
                    raw_market_mismatch=True,
                    raw_rows=len(df_r), market_rows=len(df_m),
                ))

        return results

    def classify_missing(self, missing_dates: list, trading_days: Set[str], year: str) -> MissingType:
        if not missing_dates:
            return MissingType.NONE
        current_year = str(datetime.now().year)
        if year == current_year:
            return MissingType.CURRENT_YEAR
        return MissingType.SUSPENSION

    def _classify_missing_dates(self, symbol: str, missing_dates: Set[str],
                                 cached_dates: Set[str]) -> Tuple[List, List]:
        if not missing_dates:
            return [], []

        sorted_missing = sorted(missing_dates)
        groups = self._group_consecutive_dates(sorted_missing)

        gaps = []
        suspended_ranges = []

        for group in groups:
            group_start = group[0]
            group_end = group[-1]
            group_trading_days = sum(1 for d in group if d in self._trading_day_set)
            has_data_before = any(d < group_start for d in cached_dates)
            has_data_after = any(d > group_end for d in cached_dates)

            if group_trading_days <= 3 and has_data_before and has_data_after:
                gaps.append({'start': group_start, 'end': group_end,
                             'trading_days': group_trading_days, 'type': 'missing'})
            elif group_trading_days >= self.SUSPENSION_GAP_THRESHOLD:
                suspended_ranges.append([group_start, group_end])
                if has_data_before and has_data_after:
                    gaps.append({'start': group_start, 'end': group_end,
                                 'trading_days': group_trading_days, 'type': 'suspended'})
            elif has_data_before and has_data_after:
                if group_trading_days <= self.MISSING_GAP_THRESHOLD:
                    gaps.append({'start': group_start, 'end': group_end,
                                 'trading_days': group_trading_days, 'type': 'missing'})
                else:
                    suspended_ranges.append([group_start, group_end])
                    gaps.append({'start': group_start, 'end': group_end,
                                 'trading_days': group_trading_days, 'type': 'suspended'})
            elif not has_data_before or not has_data_after:
                if group_trading_days <= self.MISSING_GAP_THRESHOLD:
                    gaps.append({'start': group_start, 'end': group_end,
                                 'trading_days': group_trading_days, 'type': 'missing'})
                else:
                    suspended_ranges.append([group_start, group_end])

        return gaps, suspended_ranges

    def _group_consecutive_dates(self, sorted_dates: List[str]) -> List[List[str]]:
        if not sorted_dates:
            return []
        groups = [[sorted_dates[0]]]
        for i in range(1, len(sorted_dates)):
            prev = pd.Timestamp(sorted_dates[i - 1])
            curr = pd.Timestamp(sorted_dates[i])
            if (curr - prev).days <= 3:
                groups[-1].append(sorted_dates[i])
            else:
                groups.append([sorted_dates[i]])
        return groups

    @staticmethod
    def _get_year_files(stock_dir: str) -> Dict[str, str]:
        if not os.path.exists(stock_dir):
            return {}
        return {f.split('_')[0]: f for f in os.listdir(stock_dir) if f.endswith('.parquet')}

    @staticmethod
    def _read_parquet_safe(filepath: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_parquet(filepath)
            if df.empty:
                return None
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.warning(f"读取文件失败 {filepath}: {e}")
            return None

class DownloadProgress:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.cached = 0
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self.repaired = 0
        self.suspended = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def record_cached(self):
        with self.lock:
            self.cached += 1
            self.completed += 1

    def record_downloaded(self):
        with self.lock:
            self.downloaded += 1
            self.completed += 1

    def record_failed(self):
        with self.lock:
            self.failed += 1
            self.completed += 1

    def record_skipped(self):
        with self.lock:
            self.skipped += 1
            self.completed += 1

    def record_repaired(self):
        with self.lock:
            self.repaired += 1
            self.completed += 1

    def record_suspended(self):
        with self.lock:
            self.suspended += 1
            self.completed += 1

    def report(self) -> str:
        elapsed = time.time() - self.start_time
        if self.completed > 0 and self.completed < self.total:
            eta = elapsed / self.completed * (self.total - self.completed)
            eta_str = f"{int(eta // 60)}m{int(eta % 60)}s"
        else:
            eta_str = "-"
        parts = [
            f"缓存命中={self.cached}",
            f"下载={self.downloaded}",
            f"修复={self.repaired}",
            f"停牌={self.suspended}",
            f"失败={self.failed}",
            f"跳过={self.skipped}",
        ]
        return (
            f"[ {self.completed} / {self.total} ] {', '.join(parts)} | "
            f"耗时={int(elapsed // 60)}m{int(elapsed % 60)}s ETA={eta_str}"
        )

class DataRepairEngine:
    def __init__(self, processor: OpenDataProcessor, checker: DataIntegrityChecker,
                 progress: DownloadProgress):
        self.processor = processor
        self.checker = checker
        self.progress = progress

    def repair_one(self, symbol: str, start_date: str, end_date: str,
                   period: str, data_type: str) -> str:
        try:
            result = self.checker.check_symbol(symbol, start_date, end_date, data_type)

            if result['status'] == 'complete':
                self.progress.record_cached()
                return f"complete:{symbol}"

            if result['status'] == 'no_data':
                logger.info(f"[{symbol}] 无缓存数据，执行完整下载...")
                if data_type == 'raw':
                    df = self.processor.get_raw_data(symbol, start_date, end_date, period)
                else:
                    df = self.processor.get_data(symbol, start_date, end_date, period)

                if df is not None and not df.empty:
                    self.progress.record_downloaded()
                    return f"ok:{symbol}"
                else:
                    self.progress.record_skipped()
                    return f"empty:{symbol}"

            if result['status'] == 'incomplete':
                missing_gaps = [g for g in result['gaps'] if g['type'] == 'missing']
                suspended_gaps = [g for g in result['gaps'] if g['type'] == 'suspended']

                if missing_gaps:
                    gap_desc = ", ".join(
                        f"{g['start']}~{g['end']}({g['trading_days']}天)"
                        for g in missing_gaps
                    )
                    logger.info(f"[{symbol}] 数据缺失需修复: {gap_desc}, 覆盖率={result['coverage']}")

                    _invalidate_missing_years_cache(symbol, period, data_type, result['missing_years'])

                    if data_type == 'raw':
                        df = self.processor.get_raw_data(symbol, start_date, end_date, period)
                    else:
                        df = self.processor.get_data(symbol, start_date, end_date, period)

                    if df is not None and not df.empty:
                        recheck = self.checker.check_symbol(symbol, start_date, end_date, data_type)
                        if recheck['status'] == 'complete':
                            old_suspended = cache_manager.index_manager.get_suspended_ranges(symbol)
                            if old_suspended:
                                cache_manager.index_manager.mark_suspended(symbol, [])
                                logger.debug(f"[{symbol}] 数据修复完毕，清除旧停牌标记: {old_suspended}")
                            self.progress.record_repaired()
                            return f"repaired:{symbol}:{result['total_missing']}"
                        else:
                            remaining_gaps = [g for g in recheck.get('gaps', []) if g['type'] == 'missing']
                            if remaining_gaps:
                                still_missing_ranges = [[g['start'], g['end']] for g in remaining_gaps]
                                _merge_suspended_ranges(symbol, still_missing_ranges)
                                still_desc = ", ".join(
                                    f"{g['start']}~{g['end']}({g['trading_days']}天)"
                                    for g in remaining_gaps
                                )
                                logger.info(f"[{symbol}] 修复后仍缺失(数据源无数据)，已标记为停牌: {still_desc}")

                            if recheck.get('suspended_ranges'):
                                _merge_suspended_ranges(symbol, recheck['suspended_ranges'])
                                logger.info(f"[{symbol}] 标记停牌区间: {recheck['suspended_ranges']}")

                            self.progress.record_repaired()
                            return f"repaired:{symbol}:{result['total_missing']}"
                    else:
                        self.progress.record_failed()
                        return f"err:{symbol}:修复下载失败"

                if suspended_gaps:
                    gap_desc = ", ".join(
                        f"{g['start']}~{g['end']}({g['trading_days']}天)"
                        for g in suspended_gaps
                    )
                    logger.info(f"[{symbol}] 疑似停牌区间，尝试从数据源修复: {gap_desc}, 覆盖率={result['coverage']}")

                    _invalidate_missing_years_cache(symbol, period, data_type, result['missing_years'])

                    if data_type == 'raw':
                        df = self.processor.get_raw_data(symbol, start_date, end_date, period)
                    else:
                        df = self.processor.get_data(symbol, start_date, end_date, period)

                    if df is not None and not df.empty:
                        recheck = self.checker.check_symbol(symbol, start_date, end_date, data_type)
                        if recheck['status'] == 'complete':
                            old_suspended = cache_manager.index_manager.get_suspended_ranges(symbol)
                            if old_suspended:
                                cache_manager.index_manager.mark_suspended(symbol, [])
                                logger.debug(f"[{symbol}] 数据修复完毕，清除旧停牌标记: {old_suspended}")
                            logger.info(f"[{symbol}] 疑似停牌区间修复成功（实际为数据源缺失）")
                            self.progress.record_repaired()
                            return f"repaired:{symbol}:{result['total_missing']}"
                        else:
                            if recheck.get('suspended_ranges'):
                                _merge_suspended_ranges(symbol, recheck['suspended_ranges'])
                                logger.info(f"[{symbol}] 修复后确认停牌区间: {recheck['suspended_ranges']}")
                            else:
                                remaining_missing = [g for g in recheck.get('gaps', []) if g['type'] == 'missing']
                                if remaining_missing:
                                    still_missing_ranges = [[g['start'], g['end']] for g in remaining_missing]
                                    _merge_suspended_ranges(symbol, still_missing_ranges)
                                    logger.info(f"[{symbol}] 修复后仍有缺失(数据源无数据)，已标记为停牌")

                            self.progress.record_suspended()
                            return f"suspended:{symbol}:{len(recheck.get('suspended_ranges', result.get('suspended_ranges', [])))}段"
                    else:
                        suspended_ranges_to_mark = result.get('suspended_ranges', [])
                        if suspended_ranges_to_mark:
                            _merge_suspended_ranges(symbol, suspended_ranges_to_mark)
                            logger.info(f"[{symbol}] 数据源无数据，确认停牌区间: {suspended_ranges_to_mark}")

                        self.progress.record_suspended()
                        return f"suspended:{symbol}:{len(suspended_ranges_to_mark)}段"

                self.progress.record_suspended()
                return f"suspended:{symbol}:{len(result.get('suspended_ranges', []))}段"

            self.progress.record_skipped()
            return f"unknown:{symbol}"
        except Exception as e:
            self.progress.record_failed()
            logger.error(f"[{symbol}] 校验修复失败: {e}")
            return f"err:{symbol}:{e}"

    def trim_non_trading_days(self, filepath: str, trading_days: Set[str]) -> bool:
        try:
            df = pd.read_parquet(filepath)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            original_len = len(df)
            mask = df.index.normalize().isin([pd.Timestamp(d) for d in trading_days])
            df = df[mask]

            if len(df) < original_len:
                df.to_parquet(filepath, index=True)
                logger.info(f"  清理非交易日: {original_len} -> {len(df)} 行 (删除{original_len - len(df)}行)")
                return True
            return False
        except Exception as e:
            logger.error(f"  清理非交易日失败: {e}")
            return False

    def fix_check_results(self, results: List[CheckResult],
                          calendar_by_year: Dict[str, Set[str]],
                          market_dir: str, market_raw_dir: str,
                          dry_run: bool = False) -> Dict:
        stats = {
            'trimmed': 0,
            'redownload_success': 0,
            'redownload_fail': 0,
            'post_trim_fail': 0,
            'skipped': 0,
        }

        extra_results = [r for r in results if r.missing_type == MissingType.EXTRA_NON_TRADING]
        if extra_results:
            logger.info(f"Step 1: 修复 {len(extra_results)} 个非交易日数据问题")
            for r in extra_results:
                if dry_run:
                    logger.info(f"  [DRY RUN] {r.symbol} {r.year}年 {r.data_type} - 含{len(r.extra_dates)}个非交易日")
                    stats['skipped'] += 1
                    continue

                base_dir = market_dir if r.data_type == 'adjusted' else market_raw_dir
                filepath = os.path.join(base_dir, r.symbol, f'{r.year}_1d.parquet')

                if os.path.exists(filepath):
                    trading_days = calendar_by_year.get(r.year, set())
                    if self.trim_non_trading_days(filepath, trading_days):
                        stats['trimmed'] += 1
                    else:
                        logger.debug(f"  {r.symbol} {r.year}年 {r.data_type} - 无需清理")

        redownload_tasks = {}

        for r in results:
            if r.missing_type == MissingType.EXTRA_NON_TRADING:
                continue
            if r.missing_type == MissingType.SUSPENSION:
                continue
            if r.missing_type == MissingType.CURRENT_YEAR:
                continue

            if r.missing_type == MissingType.DATA_INCOMPLETE:
                sub_dir = 'market_raw' if r.data_type == 'raw' else 'market'
                redownload_tasks[(r.symbol, sub_dir, r.year)] = f"{r.data_type}数据不完整"
                continue

            if r.missing_type == MissingType.RAW_MARKET_MISMATCH:
                redownload_tasks[(r.symbol, 'market', r.year)] = 'raw_market不一致'
                redownload_tasks[(r.symbol, 'market_raw', r.year)] = 'raw_market不一致'
                continue

        if not redownload_tasks:
            logger.info("Step 2: 无需重新下载")
            return stats

        logger.info(f"Step 2: 需要重新下载 {len(redownload_tasks)} 个任务")

        by_symbol = defaultdict(list)
        for (symbol, sub_dir, year), reason in redownload_tasks.items():
            by_symbol[symbol].append((sub_dir, year, reason))

        for idx, (symbol, tasks) in enumerate(sorted(by_symbol.items()), 1):
            for sub_dir, year, reason in tasks:
                logger.info(f"[{idx}/{len(by_symbol)}] {symbol} {sub_dir} {year}年 - {reason}")

                if dry_run:
                    stats['skipped'] += 1
                    continue

                cache_dir = market_dir if sub_dir == 'market' else market_raw_dir
                _delete_year_cache(cache_dir, symbol, year)

                year_start = f'{year}-01-01'
                year_end = f'{year}-12-31'
                try:
                    if sub_dir == 'market':
                        df = self.processor.get_data(symbol, year_start, year_end, '1d')
                    else:
                        df = self.processor.get_raw_data(symbol, year_start, year_end, '1d')

                    if df is not None and not df.empty:
                        if isinstance(df.index, pd.DatetimeIndex):
                            trading_dates = calendar_by_year.get(year, set())
                            mask = df.index.normalize().isin([pd.Timestamp(d) for d in trading_dates])
                            trimmed = df[mask]
                            if len(trimmed) < len(df):
                                filepath = os.path.join(cache_dir, symbol, f'{year}_1d.parquet')
                                if os.path.exists(filepath):
                                    trimmed.to_parquet(filepath, index=True)
                                    logger.info(f"  下载并清理非交易日: {len(df)} -> {len(trimmed)} 行")
                                    df = trimmed

                        if isinstance(df.index, pd.DatetimeIndex):
                            stock_dates = {d.strftime('%Y-%m-%d') for d in df.index}
                            trading_dates = calendar_by_year.get(year, set())
                            extra = stock_dates - trading_dates
                            if extra:
                                stats['post_trim_fail'] += 1
                                logger.warning(f"  修复后仍有{len(extra)}个非交易日")
                            else:
                                stats['redownload_success'] += 1
                                logger.info(f"  修复成功: {len(df)} 行")
                        else:
                            stats['redownload_success'] += 1
                            logger.info(f"  修复成功: {len(df)} 行")
                    else:
                        logger.warning(f"  修复失败: 无数据")
                        stats['redownload_fail'] += 1
                except Exception as e:
                    logger.warning(f"  修复失败: {e}")
                    stats['redownload_fail'] += 1

        return stats


class ReportGenerator:
    TYPE_LABELS = {
        'raw_extra_days': '不复权含非交易日',
        'raw_missing_days_incomplete': '不复权缺交易日(需修复)',
        'raw_missing_days_current': '不复权缺交易日(当年)',
        'raw_missing_days_suspension': '不复权缺交易日(停牌)',
        'market_extra_days': '后复权含非交易日',
        'market_missing_days_incomplete': '后复权缺交易日(需修复)',
        'market_missing_days_current': '后复权缺交易日(当年)',
        'market_missing_days_suspension': '后复权缺交易日(停牌)',
        'raw_market_mismatch': '不复权vs后复权不一致',
        'raw_empty': '不复权数据为空',
        'market_empty': '后复权数据为空',
    }

    TYPE_BADGE_CSS = {
        'raw_extra_days': 'extra',
        'raw_missing_days_incomplete': 'missing',
        'raw_missing_days_current': 'current',
        'raw_missing_days_suspension': 'suspension',
        'market_extra_days': 'extra',
        'market_missing_days_incomplete': 'missing',
        'market_missing_days_current': 'current',
        'market_missing_days_suspension': 'suspension',
        'raw_market_mismatch': 'mismatch',
        'raw_empty': 'empty',
        'market_empty': 'empty',
    }

    MISSING_TYPE_LABELS = {
        MissingType.NONE: '无问题',
        MissingType.CURRENT_YEAR: '当年数据未完',
        MissingType.SUSPENSION: '停牌/未上市/退市',
        MissingType.DATA_INCOMPLETE: '数据不完整(需修复)',
        MissingType.EXTRA_NON_TRADING: '含非交易日',
        MissingType.RAW_MARKET_MISMATCH: '不复权vs后复权不一致',
    }

    MISSING_TYPE_CSS = {
        MissingType.NONE: 'suspension',
        MissingType.CURRENT_YEAR: 'current',
        MissingType.SUSPENSION: 'suspension',
        MissingType.DATA_INCOMPLETE: 'missing',
        MissingType.EXTRA_NON_TRADING: 'extra',
        MissingType.RAW_MARKET_MISMATCH: 'mismatch',
    }

    CSS_TEMPLATE = (
        "body { font-family: 'Microsoft YaHei', sans-serif; margin: 20px; background: #f5f5f5; }"
        "h1 { color: #333; }"
        ".summary { display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap; }"
        ".card { background: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }"
        ".card h3 { margin: 0 0 5px; color: #666; font-size: 13px; }"
        ".card .value { font-size: 24px; font-weight: bold; }"
        ".card.green .value { color: #52c41a; }"
        ".card.red .value { color: #ff4d4f; }"
        ".card.blue .value { color: #1890ff; }"
        ".card.orange .value { color: #fa8c16; }"
        ".card.purple .value { color: #722ed1; }"
        ".calendar { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }"
        ".calendar h3 { margin: 0 0 10px; color: #333; }"
        ".calendar p { margin: 2px 0; font-size: 13px; color: #666; }"
        "table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; }"
        "th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }"
        "th { background: #fafafa; font-weight: bold; }"
        "tr:hover { background: #f0f7ff; }"
        ".badge { padding: 2px 6px; border-radius: 4px; font-size: 11px; white-space: nowrap; }"
        ".badge-extra { background: #fff1f0; color: #cf1322; }"
        ".badge-missing { background: #fff7e6; color: #d46b08; }"
        ".badge-current { background: #e6f7ff; color: #096dd9; }"
        ".badge-suspension { background: #f6ffed; color: #389e0d; }"
        ".badge-mismatch { background: #fff2e8; color: #d4380d; }"
        ".badge-empty { background: #f0f5ff; color: #2f54eb; }"
    )

    def generate_html(self, results, stats, calendar_by_year,
                      market_dir, market_raw_dir,
                      output_path, dry_run):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        type_counts = defaultdict(int)
        for r in results:
            type_key = self._checkresult_to_type_key(r)
            type_counts[type_key] += 1

        stocks_market = self._get_stock_dirs(market_dir)
        stocks_raw = self._get_stock_dirs(market_raw_dir)
        all_stocks = sorted(stocks_market | stocks_raw)
        INDEX_CODES = {'000300.SH', '000905.SH', '000852.SH', '000016.SH',
                       '000001.SH', '399001.SZ', '399006.SZ'}
        stock_count = len([s for s in all_stocks if s not in INDEX_CODES])

        calendar_info = '<br>'.join(
            f'{year}年: {len(days)}个交易日 ({min(days)} ~ {max(days)})'
            for year, days in sorted(calendar_by_year.items())
        )

        suspension_count = len(set(r.symbol for r in results if r.missing_type == MissingType.SUSPENSION))
        suspension_summary = ''
        if suspension_count:
            suspension_summary = '<div class="card orange"><h3>有停牌记录的股票</h3><div class="value">' + str(suspension_count) + '</div></div>'

        dry_run_label = '<b>(DRY RUN)</b>' if dry_run else ''
        issues_card_class = 'green' if not results else 'red'

        html_parts = []
        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html><head><meta charset="utf-8">')
        html_parts.append('<title>交易日一致性检查报告</title>')
        html_parts.append('<style>')
        html_parts.append(self.CSS_TEMPLATE)
        html_parts.append('</style></head><body>')
        html_parts.append('<h1>交易日一致性检查报告</h1>')
        html_parts.append(f'<p>生成时间: {now} {dry_run_label} | 基准: 沪深300交易日历</p>')
        html_parts.append('<div class="calendar">')
        html_parts.append('<h3>沪深300交易日历</h3>')
        html_parts.append(calendar_info)
        html_parts.append('</div>')
        html_parts.append('<div class="summary">')
        html_parts.append(f'<div class="card blue"><h3>股票总数</h3><div class="value">{stock_count}</div></div>')
        html_parts.append(f'<div class="card {issues_card_class}"><h3>问题数</h3><div class="value">{len(results)}</div></div>')
        html_parts.append(f'<div class="card purple"><h3>非交易日清理</h3><div class="value">{stats.get("trimmed", 0)}</div></div>')
        html_parts.append(f'<div class="card green"><h3>重新下载成功</h3><div class="value">{stats.get("redownload_success", 0)}</div></div>')
        html_parts.append(f'<div class="card red"><h3>重新下载失败</h3><div class="value">{stats.get("redownload_fail", 0)}</div></div>')
        html_parts.append(f'<div class="card orange"><h3>清理后仍不一致</h3><div class="value">{stats.get("post_trim_fail", 0)}</div></div>')
        html_parts.append(f'<div class="card blue"><h3>跳过(dry-run)</h3><div class="value">{stats.get("skipped", 0)}</div></div>')
        html_parts.append(suspension_summary)
        html_parts.append('</div>')
        html_parts.append('<h2>问题分类统计</h2>')
        html_parts.append('<table><tr><th>类型</th><th>数量</th></tr>')

        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            label = self.TYPE_LABELS.get(t, t)
            css = self.TYPE_BADGE_CSS.get(t, 'mismatch')
            html_parts.append(f'<tr><td><span class="badge badge-{css}">{label}</span></td><td>{count}</td></tr>')

        if results:
            html_parts.append('</table>')
            html_parts.append(f'<h2>问题详情（共 {len(results)} 条）</h2>')

            by_stock_year = defaultdict(list)
            for r in results:
                key = (r.symbol, r.year)
                by_stock_year[key].append(r)

            html_parts.append('<table><tr><th>股票</th><th>年份</th><th>问题</th><th>详情</th></tr>')
            for (symbol, year), stock_results in sorted(by_stock_year.items()):
                for j, r in enumerate(stock_results):
                    mtype_label = self.MISSING_TYPE_LABELS.get(r.missing_type, str(r.missing_type))
                    css = self.MISSING_TYPE_CSS.get(r.missing_type, 'mismatch')
                    symbol_cell = symbol if j == 0 else ''
                    year_cell = year if j == 0 else ''
                    detail = self._format_detail(r)
                    html_parts.append(
                        f'<tr><td>{symbol_cell}</td><td>{year_cell}</td>'
                        f'<td><span class="badge badge-{css}">{mtype_label}</span></td>'
                        f'<td>{detail}</td></tr>'
                    )

        suspension_results = [r for r in results if r.missing_type == MissingType.SUSPENSION]
        if suspension_results:
            by_symbol = defaultdict(list)
            for r in suspension_results:
                by_symbol[r.symbol].append(r)

            html_parts.append(f'<h2>停牌/数据缺失记录（共 {len(by_symbol)} 只股票）</h2>')
            html_parts.append('<table><tr><th>股票</th><th>年份</th><th>缺失交易日数</th><th>类型</th></tr>')
            for symbol, records in sorted(by_symbol.items()):
                for j, r in enumerate(records):
                    symbol_cell = symbol if j == 0 else ''
                    type_label = '不复权' if r.data_type == 'raw' else '后复权' if r.data_type == 'adjusted' else '两者'
                    html_parts.append(
                        f'<tr><td>{symbol_cell}</td><td>{r.year}</td>'
                        f'<td>{len(r.missing_dates)}</td><td>{type_label}</td></tr>'
                    )

        html_parts.append("</table></body></html>")

        html = '\n'.join(html_parts)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"报告已保存: {output_path}")

    def generate_summary(self, results):
        summary = defaultdict(int)
        for r in results:
            summary[r.missing_type.value] += 1
        return dict(summary)

    def _checkresult_to_type_key(self, r):
        prefix = 'raw' if r.data_type in ('raw', 'both') else 'market'
        if r.missing_type == MissingType.EXTRA_NON_TRADING:
            return f'{prefix}_extra_days'
        elif r.missing_type == MissingType.DATA_INCOMPLETE:
            return f'{prefix}_missing_days_incomplete'
        elif r.missing_type == MissingType.CURRENT_YEAR:
            return f'{prefix}_missing_days_current'
        elif r.missing_type == MissingType.SUSPENSION:
            return f'{prefix}_missing_days_suspension'
        elif r.missing_type == MissingType.RAW_MARKET_MISMATCH:
            return 'raw_market_mismatch'
        return f'{prefix}_unknown'

    def _format_detail(self, r):
        parts = []
        if r.extra_dates:
            parts.append(f'含{len(r.extra_dates)}个非交易日')
        if r.missing_dates:
            parts.append(f'缺{len(r.missing_dates)}个交易日')
        if r.raw_market_mismatch:
            parts.append(f'raw独有{len(r.missing_dates)}天, market独有{len(r.extra_dates)}天')
        if r.coverage:
            parts.append(f'覆盖率={r.coverage}')
        return ', '.join(parts) if parts else str(r.missing_type.value)

    @staticmethod
    def _get_stock_dirs(base_dir):
        if not os.path.exists(base_dir):
            return set()
        return {d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))}


def _needs_download(symbol, start_date, end_date, period, data_type):
    idx = cache_manager.index_manager
    req_years = cache_manager._parse_years_from_range(start_date, end_date)
    if not req_years:
        return True

    if data_type == 'raw':
        available = idx.get_available_market_raw_years(symbol, period)
        if not available:
            namespace = 'OpenDataProcessor_Raw'
            available = cache_manager.disk_cache.list_yearly_files(namespace, symbol, period)
        checked = set(idx.get_checked_market_raw_years(symbol, period))
    else:
        available = idx.get_available_market_years(symbol, period)
        if not available:
            namespace = 'OpenDataProcessor'
            available = cache_manager.disk_cache.list_yearly_files(namespace, symbol, period)
        checked = set(idx.get_checked_market_years(symbol, period))

    cached_years = set(available)
    missing = set(req_years) - cached_years - checked
    if missing:
        return True

    if not _check_cache_integrity(symbol, start_date, end_date, period, data_type):
        return True

    return False


def _check_cache_integrity(symbol, start_date, end_date, period, data_type):
    namespace = 'OpenDataProcessor_Raw' if data_type == 'raw' else 'OpenDataProcessor'
    req_years = cache_manager._parse_years_from_range(start_date, end_date)

    frames = []
    for year in req_years:
        df = cache_manager.disk_cache.get_yearly(namespace, symbol, year, period)
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            frames.append(df)

    if not frames:
        return False

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep='last')]

    if len(merged) < 2:
        return True

    dates = merged.index
    for i in range(1, len(dates)):
        gap_days = (dates[i] - dates[i - 1]).days
        if gap_days > 15:
            logger.debug(f"[{symbol}] 检测到日期空洞: "
                         f"{dates[i-1].strftime('%Y-%m-%d')} ~ {dates[i].strftime('%Y-%m-%d')} "
                         f"(间隔{gap_days}天)")
            return False

    return True


def download_one(symbol, start_date, end_date, period, data_type,
                  processor, progress, force):
    try:
        if not force and not _needs_download(symbol, start_date, end_date, period, data_type):
            progress.record_cached()
            logger.debug(f"[{symbol}] 缓存命中且数据完整，跳过下载")
            return f"cached:{symbol}"

        logger.debug(f"[{symbol}] 开始下载 {data_type} 数据...")
        if data_type == 'raw':
            df = processor.get_raw_data(symbol, start_date, end_date, period)
        else:
            df = processor.get_data(symbol, start_date, end_date, period)

        if df is not None and not df.empty:
            progress.record_downloaded()
            logger.debug(f"[{symbol}] 下载成功，共 {len(df)} 条数据")
            return f"ok:{symbol}"
        else:
            progress.record_skipped()
            logger.warning(f"[{symbol}] 数据为空，跳过")
            return f"empty:{symbol}"
    except Exception as e:
        progress.record_failed()
        logger.error(f"[{symbol}] 下载失败: {e}")
        return f"err:{symbol}:{e}"


def _invalidate_missing_years_cache(symbol, period, data_type, missing_years):
    if not missing_years:
        return

    namespace = 'OpenDataProcessor_Raw' if data_type == 'raw' else 'OpenDataProcessor'
    is_raw = data_type == 'raw'

    for year in missing_years:
        deleted = cache_manager.disk_cache.delete_yearly(namespace, symbol, year, period)
        if deleted:
            logger.debug(f"[{symbol}] 已删除 {year} 年缓存文件以强制重新下载")

    idx = cache_manager.index_manager
    if is_raw:
        available = idx.get_available_market_raw_years(symbol, period)
        new_years = [y for y in available if y not in missing_years]
        if available != new_years:
            with idx.lock:
                entry = idx._market_raw_index.get(symbol, {}).get(period, {})
                if entry:
                    entry['years'] = new_years
                    idx._dirty_flags['market_raw'] = True
    else:
        available = idx.get_available_market_years(symbol, period)
        new_years = [y for y in available if y not in missing_years]
        if available != new_years:
            with idx.lock:
                entry = idx._market_index.get(symbol, {}).get(period, {})
                if entry:
                    entry['years'] = new_years
                    idx._dirty_flags['market'] = True


def _merge_suspended_ranges(symbol, new_ranges):
    if not new_ranges:
        return
    existing = cache_manager.index_manager.get_suspended_ranges(symbol)
    merged = existing + new_ranges
    merged.sort(key=lambda x: x[0])
    result_ranges = [merged[0]]
    for start, end in merged[1:]:
        prev_start, prev_end = result_ranges[-1]
        if start <= prev_end:
            result_ranges[-1] = [prev_start, max(prev_end, end)]
        else:
            result_ranges.append([start, end])
    cache_manager.index_manager.mark_suspended(symbol, result_ranges)


def _delete_year_cache(cache_dir, symbol, year):
    stock_dir = os.path.join(cache_dir, symbol)
    if not os.path.exists(stock_dir):
        return
    for f in os.listdir(stock_dir):
        if f.startswith(f'{year}_') and f.endswith('.parquet'):
            filepath = os.path.join(stock_dir, f)
            os.remove(filepath)
            logger.debug(f"已删除: {filepath}")


def resolve_stock_list(opendata_processor, qmt_processor, pool, stocks,
                        start_date, end_date):
    if stocks:
        stock_list = []
        for s in stocks.split(','):
            s = s.strip()
            if '.' not in s:
                s = f"{s}.SH" if s.startswith(('6', '9')) else f"{s}.SZ"
            stock_list.append(s)
        logger.info(f"使用手动指定股票列表: {len(stock_list)} 只")
        return stock_list

    if pool:
        index_code = IndexConstituentManager.sector_to_index_code(pool)

        if pool in ('沪深A股', 'A股', '全部A股'):
            logger.info(f"获取 '{pool}' 全部股票列表（当前）...")
            stock_list = []

            if qmt_processor:
                try:
                    stock_list = qmt_processor.get_stock_list(pool)
                    if stock_list and len(stock_list) > 1000:
                        logger.info(f"从 QMT 获取到 {len(stock_list)} 只 {pool} 股票")
                        return stock_list
                    else:
                        logger.warning(f"QMT 返回股票数量过少 ({len(stock_list)} 只)")
                except Exception as e:
                    logger.warning(f"从 QMT 获取股票列表失败: {e}")

            stock_list = opendata_processor.get_stock_list(pool)
            if stock_list and len(stock_list) > 10:
                logger.info(f"从 OpenData 获取到 {len(stock_list)} 只 {pool} 股票")
                return stock_list
            else:
                logger.error(f"无法获取有效的 {pool} 股票列表")
                return []

        if start_date and end_date and index_code:
            logger.info(f"获取板块 '{pool}' 在 {start_date} ~ {end_date} 期间的历史成份股并集...")
            mgr = IndexConstituentManager()
            stock_list = mgr.get_all_constituent_stocks_in_range(
                index_code, start_date, end_date
            )
            if not stock_list:
                logger.warning("IndexConstituentManager 未获取到数据，回退到 OpenDataProcessor...")
                stock_list = opendata_processor.get_historical_stock_list(pool, date=start_date)
        elif start_date:
            logger.info(f"获取板块 '{pool}' 历史成份股(基准日期: {start_date})...")
            stock_list = opendata_processor.get_historical_stock_list(pool, date=start_date)
        else:
            logger.info(f"获取板块 '{pool}' 当前成份股...")
            if qmt_processor:
                try:
                    stock_list = qmt_processor.get_stock_list(pool)
                    if stock_list and len(stock_list) > 10:
                        logger.info(f"从 QMT 获取到 {len(stock_list)} 只 {pool} 成份股")
                        return stock_list
                except Exception as e:
                    logger.debug(f"QMT获取成份股失败: {e}")
            stock_list = opendata_processor.get_stock_list(pool)
        logger.info(f"板块 '{pool}' 共获取到 {len(stock_list)} 只股票")
        return stock_list

    logger.info("获取沪深A股全部股票列表...")
    stock_list = []

    if qmt_processor:
        try:
            stock_list = qmt_processor.get_stock_list('沪深A股')
            if stock_list and len(stock_list) > 1000:
                logger.info(f"从 QMT 获取到 {len(stock_list)} 只沪深A股")
                return stock_list
            else:
                logger.warning(f"QMT 返回股票数量过少 ({len(stock_list)} 只)，尝试其他数据源...")
        except Exception as e:
            logger.warning(f"从 QMT 获取股票列表失败: {e}")

    logger.info("尝试从 OpenData (AKShare) 获取...")
    stock_list = opendata_processor.get_stock_list('沪深A股')

    if not stock_list or len(stock_list) <= 10:
        logger.error("无法获取有效的全市场股票列表")
        return []

    logger.info(f"共获取到 {len(stock_list)} 只股票")
    return stock_list


def run_download(stock_list, start_date, end_date, period, data_type,
                  processor, workers, force):
    logger.info(f"{'='*60}")
    logger.info(f"开始下载 {data_type} 行情数据: {len(stock_list)} 只股票, "
                 f"日期范围 {start_date} ~ {end_date}, 周期 {period}")
    logger.info(f"{'='*60}")

    progress = DownloadProgress(len(stock_list))
    save_interval = max(1, len(stock_list) // 20)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for symbol in stock_list:
            future = executor.submit(
                download_one,
                symbol, start_date, end_date, period,
                data_type, processor, progress, force,
            )
            futures[future] = symbol

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                status = result.split(':')[0]

                if status == "cached":
                    logger.debug(f"[{progress.completed} / {progress.total}] {symbol} - 缓存命中")
                elif status == "ok":
                    logger.info(f"[{progress.completed} / {progress.total}] {symbol} - 下载成功")
                elif status == "empty":
                    logger.warning(f"[{progress.completed} / {progress.total}] {symbol} - 数据为空")
                elif status == "err":
                    logger.error(f"[{progress.completed} / {progress.total}] {symbol} - 下载失败: {result.split(':', 2)[-1]}")
            except Exception as e:
                logger.error(f"[{progress.completed} / {progress.total}] {symbol} - 异常: {e}")

            if progress.completed % save_interval == 0:
                cache_manager.index_manager.save_index()

            if progress.completed % 5 == 0 or progress.completed == progress.total:
                logger.info(progress.report())

    logger.info("正在保存索引文件...")
    cache_manager.index_manager.save_index()
    logger.info("索引文件保存完成")

    logger.info(f"{'='*60}")
    logger.info(f"{data_type} 行情数据下载完成: {progress.report()}")
    logger.info(f"{'='*60}")


def run_verify(stock_list, start_date, end_date, period, data_type,
                processor, checker, workers):
    logger.info(f"{'='*60}")
    logger.info(f"开始校验 {data_type} 行情数据完整性: {len(stock_list)} 只股票, "
                 f"日期范围 {start_date} ~ {end_date}")
    logger.info(f"{'='*60}")

    progress = DownloadProgress(len(stock_list))
    save_interval = max(1, len(stock_list) // 20)
    incomplete_symbols = []

    repair_engine = DataRepairEngine(processor, checker, progress)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for symbol in stock_list:
            future = executor.submit(
                repair_engine.repair_one,
                symbol, start_date, end_date, period, data_type,
            )
            futures[future] = symbol

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                status = result.split(':')[0]

                if status == "complete":
                    logger.debug(f"[{progress.completed} / {progress.total}] {symbol} - 数据完整")
                elif status == "repaired":
                    logger.info(f"[{progress.completed} / {progress.total}] {symbol} - 修复成功")
                elif status == "suspended":
                    seg_count = result.split(':')[2] if len(result.split(':')) > 2 else "?"
                    logger.info(f"[{progress.completed} / {progress.total}] {symbol} - 含停牌 {seg_count}段")
                elif status == "ok":
                    logger.info(f"[{progress.completed} / {progress.total}] {symbol} - 下载成功")
                elif status == "empty":
                    logger.warning(f"[{progress.completed} / {progress.total}] {symbol} - 数据为空")
                elif status == "err":
                    logger.error(f"[{progress.completed} / {progress.total}] {symbol} - 失败: {result.split(':', 2)[-1]}")

                if status in ('repaired', 'suspended', 'empty', 'err'):
                    incomplete_symbols.append(symbol)

            except Exception as e:
                logger.error(f"[{progress.completed} / {progress.total}] {symbol} - 异常: {e}")
                incomplete_symbols.append(symbol)

            if progress.completed % save_interval == 0:
                cache_manager.index_manager.save_index()

            if progress.completed % 5 == 0 or progress.completed == progress.total:
                logger.info(progress.report())

    logger.info("正在保存索引文件...")
    cache_manager.index_manager.save_index()
    logger.info("索引文件保存完成")

    logger.info(f"{'='*60}")
    logger.info(f"{data_type} 行情数据校验完成: {progress.report()}")
    if incomplete_symbols:
        logger.info(f"需关注的股票({len(incomplete_symbols)}): {incomplete_symbols[:50]}")
    logger.info(f"{'='*60}")


def run_check(start_date, end_date, processor, calendar,
              fix=False, dry_run=False, report_path=''):
    logger.info(f"{'='*60}")
    logger.info(f"开始检查交易日一致性: {start_date} ~ {end_date}")
    logger.info(f"{'='*60}")

    cache_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'OpenData')
    market_dir = os.path.join(cache_base, 'market')
    market_raw_dir = os.path.join(cache_base, 'market_raw')

    if not os.path.exists(market_dir) and not os.path.exists(market_raw_dir):
        logger.error(f"数据目录不存在: {cache_base}")
        return

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    logger.info(f"获取交易日历 ({start_date} ~ {end_date})...")
    trading_days_set = calendar.get_trading_days(start_date, end_date)

    if not trading_days_set:
        logger.error("无法获取交易日历，退出检查")
        return

    calendar_by_year = calendar.get_trading_days_by_year(start_year, end_year)

    years = sorted(calendar_by_year.keys())
    logger.info(f"交易日历年份: {years}")

    checker = DataIntegrityChecker(trading_days_set)

    stocks_market = set()
    stocks_raw = set()
    if os.path.exists(market_dir):
        stocks_market = {d for d in os.listdir(market_dir) if os.path.isdir(os.path.join(market_dir, d))}
    if os.path.exists(market_raw_dir):
        stocks_raw = {d for d in os.listdir(market_raw_dir) if os.path.isdir(os.path.join(market_raw_dir, d))}
    all_stocks = sorted(stocks_market | stocks_raw)

    INDEX_CODES = {'000300.SH', '000905.SH', '000852.SH', '000016.SH',
                   '000001.SH', '399001.SZ', '399006.SZ'}
    all_stocks = [s for s in all_stocks if s not in INDEX_CODES]

    total = len(all_stocks)
    logger.info(f"开始检查 {total} 只股票, {len(years)} 个年份")

    all_results = []
    for i, symbol in enumerate(all_stocks, 1):
        if i % 100 == 0:
            logger.info(f"进度: {i}/{total}")

        m_files = DataIntegrityChecker._get_year_files(os.path.join(market_dir, symbol))
        r_files = DataIntegrityChecker._get_year_files(os.path.join(market_raw_dir, symbol))
        stock_years = sorted(set(m_files.keys()) | set(r_files.keys()))

        for year in stock_years:
            if year not in calendar_by_year:
                continue

            trading_days = calendar_by_year[year]
            results = checker.check_symbol_year(symbol, year, trading_days, market_dir, market_raw_dir)
            all_results.extend(results)

    if not all_results:
        logger.info("所有数据与交易日历一致，无问题！")
    else:
        type_counts = defaultdict(int)
        for r in all_results:
            type_counts[r.missing_type.value] += 1
        logger.info(f"发现 {len(all_results)} 个问题: {dict(type_counts)}")

    stats = {
        'trimmed': 0, 'redownload_success': 0, 'redownload_fail': 0,
        'post_trim_fail': 0, 'skipped': 0,
    }

    if fix and all_results:
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}开始修复...")
        repair_engine = DataRepairEngine(processor, checker, DownloadProgress(0))
        stats = repair_engine.fix_check_results(
            all_results, calendar_by_year, market_dir, market_raw_dir, dry_run
        )
        logger.info(
            f"修复完成: 非交易日清理={stats['trimmed']}, "
            f"重新下载成功={stats['redownload_success']}, "
            f"重新下载失败={stats['redownload_fail']}, "
            f"清理后仍不一致={stats['post_trim_fail']}, "
            f"跳过={stats['skipped']}"
        )

    if not report_path:
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'check_trading_days_report.html')

    report_gen = ReportGenerator()
    report_gen.generate_html(
        all_results, stats, calendar_by_year,
        market_dir, market_raw_dir, report_path, dry_run
    )

    logger.info(f"{'='*60}")
    logger.info(f"交易日一致性检查完成")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='批量并发下载市场行情数据（含完整性校验与修复）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            '使用示例:',
            '  # 下载沪深300后复权行情数据',
            '  python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --workers 5',
            '',
            '  # 下载并校验数据完整性（以沪深300交易日为基准）',
            '  python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type all --workers 5 --verify',
            '',
            '  # 仅校验已有数据完整性（不下载新数据）',
            '  python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --verify-only',
            '',
            '  # 检查交易日一致性并生成HTML报告',
            '  python download_market_data.py --check --start 2020-01-01 --end 2026-12-31',
            '',
            '  # 检查并修复，生成HTML报告',
            '  python download_market_data.py --check --fix --start 2020-01-01 --end 2026-12-31',
            '',
            '  # dry-run模式：显示修复计划但不执行',
            '  python download_market_data.py --check --fix --dry-run --start 2020-01-01 --end 2026-12-31',
            '',
            '  # 强制重新下载 (忽略已有缓存)',
            '  python download_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-01-01 --type raw --force',
            '',
            '  # 指定股票代码下载',
            '  python download_market_data.py --stocks 000001.SZ,600000.SH,600519.SH --start 2020-01-01 --end 2026-01-01 --type all',
            '',
            '  # 将日志同时写入文件',
            '  python download_market_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-01-01 --type all --workers 20 --log --verify',
        ]),
    )
    parser.add_argument('--pool', type=str, default=None,
                        help='股票池板块名称: 沪深300, 中证500, 中证1000, 上证50, 沪深A股')
    parser.add_argument('--stocks', type=str, default=None,
                        help='手动指定股票代码，逗号分隔，如 000001.SZ,600000.SH')
    parser.add_argument('--start', type=str, default=None,
                        help='数据起始日期，如 2020-01-01')
    parser.add_argument('--end', type=str, default=None,
                        help='数据结束日期，如 2026-04-28')
    parser.add_argument('--period', type=str, default='1d',
                        choices=['1d', '1w', '1M'],
                        help='数据周期 (默认: 1d)')
    parser.add_argument('--type', type=str, default='all',
                        choices=['adjusted', 'raw', 'all'],
                        help='数据类型: adjusted=后复权 raw=不复权 all=两者都下载 (默认: all)')
    parser.add_argument('--workers', type=int, default=5,
                        help='并发线程数 (默认: 5，建议不超过10)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='强制重新下载，忽略已有缓存')
    parser.add_argument('--verify', action='store_true', default=False,
                        help='下载后校验数据完整性并修复缺失')
    parser.add_argument('--verify-only', action='store_true', default=False,
                        help='仅校验已有数据完整性并修复缺失，不执行新下载')
    parser.add_argument('--check', action='store_true', default=False,
                        help='检查数据一致性（以沪深300交易日为基准），需配合 --start/--end 或使用默认值')
    parser.add_argument('--fix', action='store_true', default=False,
                        help='与 --check 配合使用，自动修复不一致的数据')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='与 --fix 配合使用，仅报告不实际修复')
    parser.add_argument('--report', type=str, default='',
                        help='HTML报告输出路径（与 --check 配合使用）')
    parser.add_argument('--full-year', action='store_true', default=True,
                        dest='full_year',
                        help='自动将首尾年份扩展为全年下载（默认开启）')
    parser.add_argument('--no-full-year', action='store_false',
                        dest='full_year',
                        help='禁用全年扩展，严格按指定日期范围下载')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='指定缓存数据存储目录 (默认: 项目根目录下的 .cache 文件夹)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='启用详细日志输出，显示每个股票的下载状态')
    parser.add_argument('--log', action='store_true', default=False,
                        help='将日志同时写入到 logs/ 目录下的文件')
    parser.add_argument('--log-file', type=str, default=None,
                        help='指定日志文件路径 (需配合 --log 使用)')

    args = parser.parse_args()

    global logger
    logger = setup_logger(log_to_file=args.log, log_file=args.log_file)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if args.cache_dir:
        cache_manager.configure(cache_dir=args.cache_dir)

    opendata_processor = OpenDataProcessor(fallback_to_simulated=False)

    qmt_processor = None
    try:
        qmt_processor = QMTDataProcessor(fallback_to_simulated=False)
        logger.info("QMT 处理器初始化成功")
    except Exception as e:
        logger.warning(f"QMT 处理器初始化失败: {e}，将使用 OpenData 获取股票列表")

    if args.check:
        check_start = args.start or '2020-01-01'
        check_end = args.end or f'{datetime.now().year}-12-31'
        calendar = TradingDayCalendar(opendata_processor)
        run_check(
            check_start, check_end,
            opendata_processor, calendar,
            fix=args.fix, dry_run=args.dry_run,
            report_path=args.report,
        )
        return

    if not args.start or not args.end:
        parser.error("下载模式需要指定 --start 和 --end 参数")

    data_types = []
    if args.type == 'all':
        data_types = ['adjusted', 'raw']
    else:
        data_types = [args.type]

    start_date = args.start
    end_date = args.end

    if args.full_year:
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        logger.info(f"全年扩展已启用: 日期范围调整为 {start_date} ~ {end_date}")

    stock_list = resolve_stock_list(
        opendata_processor, qmt_processor,
        args.pool, args.stocks,
        start_date, end_date,
    )

    if not stock_list:
        logger.error("未获取到任何股票，退出")
        return

    for data_type in data_types:
        if args.verify_only:
            trading_days = None
            if data_type == 'adjusted':
                calendar = TradingDayCalendar(opendata_processor)
                trading_days = calendar.get_trading_days(start_date, end_date)

            if trading_days:
                checker = DataIntegrityChecker(trading_days)
            else:
                checker = DataIntegrityChecker(set())

            run_verify(
                stock_list, start_date, end_date,
                args.period, data_type, opendata_processor,
                checker, args.workers,
            )
        else:
            run_download(
                stock_list, start_date, end_date,
                args.period, data_type, opendata_processor,
                args.workers, args.force,
            )

            if args.verify:
                calendar = TradingDayCalendar(opendata_processor)
                trading_days = calendar.get_trading_days(start_date, end_date)
                checker = DataIntegrityChecker(trading_days)

                run_verify(
                    stock_list, start_date, end_date,
                    args.period, data_type, opendata_processor,
                    checker, args.workers,
                )


if __name__ == '__main__':
    main()
