import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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


class TradingDayCalendar:
    BENCHMARK_INDEX = '000300.SH'

    def __init__(self, opendata_processor: OpenDataProcessor):
        self._processor = opendata_processor
        self._trading_days: Dict[str, Set[str]] = {}

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
                return days
        except Exception as e:
            logger.warning(f"从OpenData获取交易日历失败: {e}")

        logger.info("回退到磁盘缓存获取交易日历...")
        days = self._load_trading_days_from_cache(start_date, end_date)
        if days:
            self._trading_days[cache_key] = days
            return days

        logger.error("无法获取交易日历，将跳过完整性校验")
        return set()

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

        # 清理已不再适用的停牌区间（修复后数据已完整但停牌标记未清除的情况）
        # 不再排除停牌区间内的缺失日期——这些日期仍然需要经过分类和修复流程，
        # 在 verify_and_repair_one 中会先尝试从数据源获取，只有确认数据源也没有时才标记为停牌
        suspended_ranges = cache_manager.index_manager.get_suspended_ranges(symbol)
        if suspended_ranges:
            valid_ranges = []
            changed = False
            for s_start, s_end in suspended_ranges:
                # 检查停牌区间内是否还有缺失的交易日
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
            group_len = len(group)
            group_trading_days = sum(1 for d in group if d in self._trading_day_set)

            has_data_before = any(d < group_start for d in cached_dates)
            has_data_after = any(d > group_end for d in cached_dates)

            if group_trading_days <= 3 and has_data_before and has_data_after:
                gaps.append({
                    'start': group_start,
                    'end': group_end,
                    'trading_days': group_trading_days,
                    'type': 'missing'
                })
            elif group_trading_days >= self.SUSPENSION_GAP_THRESHOLD:
                suspended_ranges.append([group_start, group_end])
                if has_data_before and has_data_after:
                    gaps.append({
                        'start': group_start,
                        'end': group_end,
                        'trading_days': group_trading_days,
                        'type': 'suspended'
                    })
            elif has_data_before and has_data_after:
                if group_trading_days <= self.MISSING_GAP_THRESHOLD:
                    gaps.append({
                        'start': group_start,
                        'end': group_end,
                        'trading_days': group_trading_days,
                        'type': 'missing'
                    })
                else:
                    suspended_ranges.append([group_start, group_end])
                    gaps.append({
                        'start': group_start,
                        'end': group_end,
                        'trading_days': group_trading_days,
                        'type': 'suspended'
                    })
            elif not has_data_before or not has_data_after:
                if group_trading_days <= self.MISSING_GAP_THRESHOLD:
                    gaps.append({
                        'start': group_start,
                        'end': group_end,
                        'trading_days': group_trading_days,
                        'type': 'missing'
                    })
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


def _needs_download(symbol: str, start_date: str, end_date: str,
                     period: str, data_type: str) -> bool:
    idx = cache_manager.index_manager
    req_years = cache_manager._parse_years_from_range(start_date, end_date)
    if not req_years:
        return True

    if data_type == 'raw':
        available = idx.get_available_market_raw_years(symbol, period)
        if not available:
            namespace = 'OpenDataProcessor_Raw'
            available = cache_manager.disk_cache.list_yearly_files(
                namespace, symbol, period
            )
        checked = set(idx.get_checked_market_raw_years(symbol, period))
    else:
        available = idx.get_available_market_years(symbol, period)
        if not available:
            namespace = 'OpenDataProcessor'
            available = cache_manager.disk_cache.list_yearly_files(
                namespace, symbol, period
            )
        checked = set(idx.get_checked_market_years(symbol, period))

    cached_years = set(available)
    missing = set(req_years) - cached_years - checked
    if missing:
        return True

    if not _check_cache_integrity(symbol, start_date, end_date, period, data_type):
        return True

    return False


def _check_cache_integrity(symbol: str, start_date: str, end_date: str,
                            period: str, data_type: str) -> bool:
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


def download_one(symbol: str, start_date: str, end_date: str,
                  period: str, data_type: str, processor: OpenDataProcessor,
                  progress: DownloadProgress, force: bool) -> str:
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


def _invalidate_missing_years_cache(symbol: str, period: str, data_type: str,
                                     missing_years: set) -> None:
    """删除受影响年份的缓存文件和索引条目，使下次 get_data 从数据源重新获取

    当校验发现数据缺失时，缓存系统（smart_cache）会认为年份文件已存在而直接返回旧数据，
    导致修复无效。本函数在修复前清除受影响年份的缓存，确保 get_data 能真正从数据源下载。
    """
    if not missing_years:
        return

    namespace = 'OpenDataProcessor_Raw' if data_type == 'raw' else 'OpenDataProcessor'
    is_raw = data_type == 'raw'

    for year in missing_years:
        deleted = cache_manager.disk_cache.delete_yearly(namespace, symbol, year, period)
        if deleted:
            logger.debug(f"[{symbol}] 已删除 {year} 年缓存文件以强制重新下载")

    # 清除受影响年份的索引条目
    idx = cache_manager.index_manager
    if is_raw:
        available = idx.get_available_market_raw_years(symbol, period)
        new_years = [y for y in available if y not in missing_years]
        if available != new_years:
            # 重建该 symbol+period 的索引（移除缺失年份）
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


def _merge_suspended_ranges(symbol: str, new_ranges: List[List[str]]) -> None:
    """合并新的停牌区间到已有停牌标记中（而非覆盖）

    避免修复成功后标记被误覆盖导致数据完整但仍被标记为停牌的问题。
    """
    if not new_ranges:
        return
    existing = cache_manager.index_manager.get_suspended_ranges(symbol)
    merged = existing + new_ranges
    # 按起始日期排序
    merged.sort(key=lambda x: x[0])
    # 合并重叠区间
    result_ranges = [merged[0]]
    for start, end in merged[1:]:
        prev_start, prev_end = result_ranges[-1]
        if start <= prev_end:
            result_ranges[-1] = [prev_start, max(prev_end, end)]
        else:
            result_ranges.append([start, end])
    cache_manager.index_manager.mark_suspended(symbol, result_ranges)


def verify_and_repair_one(symbol: str, start_date: str, end_date: str,
                           period: str, data_type: str,
                           processor: OpenDataProcessor,
                           checker: DataIntegrityChecker,
                           progress: DownloadProgress) -> str:
    try:
        result = checker.check_symbol(symbol, start_date, end_date, data_type)

        if result['status'] == 'complete':
            progress.record_cached()
            return f"complete:{symbol}"

        if result['status'] == 'no_data':
            logger.info(f"[{symbol}] 无缓存数据，执行完整下载...")
            if data_type == 'raw':
                df = processor.get_raw_data(symbol, start_date, end_date, period)
            else:
                df = processor.get_data(symbol, start_date, end_date, period)

            if df is not None and not df.empty:
                progress.record_downloaded()
                return f"ok:{symbol}"
            else:
                progress.record_skipped()
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

                # 删除受影响年份的缓存文件和索引，强制从数据源重新下载
                _invalidate_missing_years_cache(symbol, period, data_type, result['missing_years'])

                if data_type == 'raw':
                    df = processor.get_raw_data(symbol, start_date, end_date, period)
                else:
                    df = processor.get_data(symbol, start_date, end_date, period)

                if df is not None and not df.empty:
                    # 修复后重新校验，检查是否仍有缺失
                    recheck = checker.check_symbol(symbol, start_date, end_date, data_type)
                    if recheck['status'] == 'complete':
                        # 修复成功，清除之前可能误标的停牌区间
                        old_suspended = cache_manager.index_manager.get_suspended_ranges(symbol)
                        if old_suspended:
                            cache_manager.index_manager.mark_suspended(symbol, [])
                            logger.debug(f"[{symbol}] 数据修复完整，清除旧停牌标记: {old_suspended}")
                        progress.record_repaired()
                        return f"repaired:{symbol}:{result['total_missing']}"
                    else:
                        # 数据源本身就没有这些数据，将剩余缺失标记为停牌，避免无限重试
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

                        progress.record_repaired()
                        return f"repaired:{symbol}:{result['total_missing']}"
                else:
                    progress.record_failed()
                    return f"err:{symbol}:修复下载失败"

            # 有停牌型缺失，先尝试从数据源修复（可能是数据源缺失而非真正停牌）
            # 只有数据源也拿不到数据时，才真正标记为停牌
            if suspended_gaps:
                gap_desc = ", ".join(
                    f"{g['start']}~{g['end']}({g['trading_days']}天)"
                    for g in suspended_gaps
                )
                logger.info(f"[{symbol}] 疑似停牌区间，尝试从数据源修复: {gap_desc}, 覆盖率={result['coverage']}")

                # 删除受影响年份的缓存文件和索引，强制从数据源重新下载
                _invalidate_missing_years_cache(symbol, period, data_type, result['missing_years'])

                if data_type == 'raw':
                    df = processor.get_raw_data(symbol, start_date, end_date, period)
                else:
                    df = processor.get_data(symbol, start_date, end_date, period)

                if df is not None and not df.empty:
                    # 修复后重新校验
                    recheck = checker.check_symbol(symbol, start_date, end_date, data_type)
                    if recheck['status'] == 'complete':
                        # 修复成功！之前是误判为停牌
                        old_suspended = cache_manager.index_manager.get_suspended_ranges(symbol)
                        if old_suspended:
                            cache_manager.index_manager.mark_suspended(symbol, [])
                            logger.debug(f"[{symbol}] 数据修复完整，清除旧停牌标记: {old_suspended}")
                        logger.info(f"[{symbol}] 疑似停牌区间修复成功（实际为数据源缺失）")
                        progress.record_repaired()
                        return f"repaired:{symbol}:{result['total_missing']}"
                    else:
                        # 数据源下载后仍有缺失，确认为真正停牌
                        if recheck.get('suspended_ranges'):
                            _merge_suspended_ranges(symbol, recheck['suspended_ranges'])
                            logger.info(f"[{symbol}] 修复后确认停牌区间: {recheck['suspended_ranges']}")
                        else:
                            # 仍有一些小gap标记为missing，将它们也标记为停牌避免无限重试
                            remaining_missing = [g for g in recheck.get('gaps', []) if g['type'] == 'missing']
                            if remaining_missing:
                                still_missing_ranges = [[g['start'], g['end']] for g in remaining_missing]
                                _merge_suspended_ranges(symbol, still_missing_ranges)
                                logger.info(f"[{symbol}] 修复后仍有缺失(数据源无数据)，已标记为停牌")

                        progress.record_suspended()
                        return f"suspended:{symbol}:{len(recheck.get('suspended_ranges', result.get('suspended_ranges', [])))}段"
                else:
                    # 数据源完全拿不到数据，确认停牌
                    suspended_ranges_to_mark = result.get('suspended_ranges', [])
                    if suspended_ranges_to_mark:
                        _merge_suspended_ranges(symbol, suspended_ranges_to_mark)
                        logger.info(f"[{symbol}] 数据源无数据，确认停牌区间: {suspended_ranges_to_mark}")

                    progress.record_suspended()
                    return f"suspended:{symbol}:{len(suspended_ranges_to_mark)}段"

            progress.record_suspended()
            return f"suspended:{symbol}:{len(result.get('suspended_ranges', []))}段"

        progress.record_skipped()
        return f"unknown:{symbol}"

    except Exception as e:
        progress.record_failed()
        logger.error(f"[{symbol}] 校验修复失败: {e}")
        return f"err:{symbol}:{e}"


def resolve_stock_list(opendata_processor: OpenDataProcessor,
                        qmt_processor: Optional[QMTDataProcessor],
                        pool: Optional[str],
                        stocks: Optional[str],
                        start_date: Optional[str],
                        end_date: Optional[str]) -> List[str]:
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
                logger.error(f"无法获取有效的 {pool} 股票列表！")
                return []

        if start_date and end_date and index_code:
            logger.info(f"获取板块 '{pool}' 在 {start_date} ~ {end_date} 期间的历史成分股并集...")
            mgr = IndexConstituentManager()
            stock_list = mgr.get_all_constituent_stocks_in_range(
                index_code, start_date, end_date
            )
            if not stock_list:
                logger.warning("IndexConstituentManager 未获取到数据，回退到 OpenDataProcessor...")
                stock_list = opendata_processor.get_historical_stock_list(pool, date=start_date)
        elif start_date:
            logger.info(f"获取板块 '{pool}' 历史成分股 (基准日期: {start_date})...")
            stock_list = opendata_processor.get_historical_stock_list(pool, date=start_date)
        else:
            logger.info(f"获取板块 '{pool}' 当前成分股...")
            if qmt_processor:
                try:
                    stock_list = qmt_processor.get_stock_list(pool)
                    if stock_list and len(stock_list) > 10:
                        logger.info(f"从 QMT 获取到 {len(stock_list)} 只 {pool} 成分股")
                        return stock_list
                except Exception as e:
                    logger.debug(f"QMT获取成分股失败: {e}")
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
        logger.error("无法获取有效的全市场股票列表！")
        return []

    logger.info(f"共获取到 {len(stock_list)} 只股票")
    return stock_list


def run_download(stock_list: List[str], start_date: str, end_date: str,
                  period: str, data_type: str, processor: OpenDataProcessor,
                  workers: int, force: bool):
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


def run_verify(stock_list: List[str], start_date: str, end_date: str,
                period: str, data_type: str, processor: OpenDataProcessor,
                checker: DataIntegrityChecker, workers: int):
    logger.info(f"{'='*60}")
    logger.info(f"开始校验 {data_type} 行情数据完整性: {len(stock_list)} 只股票, "
                 f"日期范围 {start_date} ~ {end_date}")
    logger.info(f"{'='*60}")

    progress = DownloadProgress(len(stock_list))
    save_interval = max(1, len(stock_list) // 20)
    incomplete_symbols = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for symbol in stock_list:
            future = executor.submit(
                verify_and_repair_one,
                symbol, start_date, end_date, period,
                data_type, processor, checker, progress,
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
                    missing_count = result.split(':')[2] if len(result.split(':')) > 2 else "?"
                    logger.info(f"[{progress.completed} / {progress.total}] {symbol} - 已修复缺失({missing_count}天)")
                elif status == "suspended":
                    seg_count = result.split(':')[2] if len(result.split(':')) > 2 else "?"
                    logger.info(f"[{progress.completed} / {progress.total}] {symbol} - 含停牌({seg_count}段)")
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
        logger.info(f"需关注的股票 ({len(incomplete_symbols)}): {incomplete_symbols[:50]}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='批量并发下载市场行情数据（含完整性校验与修复）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载沪深300后复权行情数据
  python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --workers 5

  # 下载并校验数据完整性（以沪深300交易日为基准）
  python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type all --workers 5 --verify

  # 仅校验已有数据完整性（不下载新数据）
  python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --verify-only

  # 强制重新下载 (忽略已有缓存)
  python download_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-01-01 --type raw --force

  # 指定股票代码下载
  python download_market_data.py --stocks 000001.SZ,600000.SH,600519.SH --start 2020-01-01 --end 2026-01-01 --type all

  # 将日志同时写入文件
  python download_market_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-01-01 --type all --workers 20 --log --verify
        """,
    )
    parser.add_argument('--pool', type=str, default=None,
                        help='股票池板块名称: 沪深300, 中证500, 中证1000, 上证50, 沪深A股')
    parser.add_argument('--stocks', type=str, default=None,
                        help='手动指定股票代码，逗号分隔，如 000001.SZ,600000.SH')
    parser.add_argument('--start', type=str, required=True,
                        help='数据起始日期，如 2020-01-01')
    parser.add_argument('--end', type=str, required=True,
                        help='数据结束日期，如 2026-04-28')
    parser.add_argument('--period', type=str, default='1d',
                        choices=['1d', '1w', '1M'],
                        help='数据周期 (默认: 1d)')
    parser.add_argument('--type', type=str, default='all',
                        choices=['adjusted', 'raw', 'all'],
                        help='数据类型: adjusted=后复权, raw=不复权, all=两者都下载 (默认: all)')
    parser.add_argument('--workers', type=int, default=5,
                        help='并发线程数 (默认: 5，建议不超过10)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='强制重新下载，忽略已有缓存')
    parser.add_argument('--verify', action='store_true', default=False,
                        help='下载后校验数据完整性并修复缺失')
    parser.add_argument('--verify-only', action='store_true', default=False,
                        help='仅校验已有数据完整性并修复缺失，不执行新下载')
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

    data_types = []
    if args.type == 'all':
        data_types = ['adjusted', 'raw']
    else:
        data_types = [args.type]

    opendata_processor = OpenDataProcessor(fallback_to_simulated=False)

    qmt_processor = None
    try:
        qmt_processor = QMTDataProcessor(fallback_to_simulated=False)
        logger.info("QMT 处理器初始化成功")
    except Exception as e:
        logger.warning(f"QMT 处理器初始化失败: {e}，将使用 OpenData 获取股票列表")

    stock_list = resolve_stock_list(
        opendata_processor, qmt_processor, args.pool, args.stocks, args.start, args.end
    )
    if not stock_list:
        logger.error("未获取到任何股票，退出")
        sys.exit(1)

    logger.info(f"共 {len(stock_list)} 只股票，数据类型: {data_types}，并发线程: {args.workers}")

    trading_days = set()
    checker = None
    if args.verify or args.verify_only:
        calendar = TradingDayCalendar(opendata_processor)
        trading_days = calendar.get_trading_days(args.start, args.end)
        if trading_days:
            checker = DataIntegrityChecker(trading_days)
            logger.info(f"交易日历就绪: {len(trading_days)} 个交易日")
        else:
            logger.warning("无法获取交易日历，将跳过完整性校验")
            args.verify = False
            args.verify_only = False

    for dtype in data_types:
        if not args.verify_only:
            run_download(
                stock_list, args.start, args.end, args.period,
                dtype, opendata_processor, args.workers, args.force
            )

        if args.verify or args.verify_only:
            if checker:
                run_verify(
                    stock_list, args.start, args.end, args.period,
                    dtype, opendata_processor, checker, args.workers
                )
            else:
                logger.warning(f"跳过 {dtype} 数据校验（无交易日历）")

    logger.info("全部任务完成!")


if __name__ == '__main__':
    main()
