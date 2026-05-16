"""从富途 request_history_kline 获取行情数据（后复权/不复权）

参考 download_financial_data.py 的模式实现，支持：
  - 指定股票池板块或手动指定股票
  - 增量下载（跳过已有缓存）
  - 强制重新下载
  - 日志文件输出
  - 进度追踪

数据保存到 .cache/FutuData/market/ 和 .cache/FutuData/market_raw/ 目录，
格式与 FutuDataProcessor 一致: {symbol}/{year}_{period}.parquet
"""
import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.data.futu import FutuDataProcessor
from core.data.index_constituent import IndexConstituentManager
from core.data.opendata import OpenDataProcessor
from core.data.qmt import QMTDataProcessor


def setup_logger(log_to_file: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('download_futu_data')
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
            log_file = f"{log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_download_futu_data.log"

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


logger = logging.getLogger('download_futu_data')


class FutuDownloadProgress:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.cached = 0
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()

    def record_cached(self):
        self.cached += 1
        self.completed += 1

    def record_downloaded(self):
        self.downloaded += 1
        self.completed += 1

    def record_failed(self):
        self.failed += 1
        self.completed += 1

    def record_skipped(self):
        self.skipped += 1
        self.completed += 1

    def report(self) -> str:
        elapsed = time.time() - self.start_time
        if self.completed > 0 and self.completed < self.total:
            eta = elapsed / self.completed * (self.total - self.completed)
            eta_str = f"{int(eta // 60)}m{int(eta % 60)}s"
        else:
            eta_str = "-"
        return (
            f"[ {self.completed} / {self.total} ] 进度: {self.cached} 缓存命中, "
            f"{self.downloaded} 已下载, {self.failed} 失败, {self.skipped} 跳过 | "
            f"耗时={int(elapsed // 60)}m{int(elapsed % 60)}s ETA={eta_str}"
        )


def resolve_stock_list(qmt_processor: Optional[QMTDataProcessor],
                        opendata_processor: OpenDataProcessor,
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
    if qmt_processor:
        try:
            stock_list = qmt_processor.get_stock_list('沪深A股')
            if stock_list and len(stock_list) > 1000:
                logger.info(f"从 QMT 获取到 {len(stock_list)} 只沪深A股")
                return stock_list
            else:
                logger.warning(f"QMT 返回股票数量过少 ({len(stock_list)} 只)")
        except Exception as e:
            logger.warning(f"从 QMT 获取股票列表失败: {e}")

    stock_list = opendata_processor.get_stock_list('沪深A股')
    if not stock_list or len(stock_list) <= 10:
        logger.error("无法获取有效的全市场股票列表！")
        return []
    logger.info(f"共获取到 {len(stock_list)} 只股票")
    return stock_list


class FutuHFQChecker:
    HFQ_CHANGE_THRESHOLD = 0.25

    @staticmethod
    def get_limit_ratio(symbol: str) -> float:
        code = symbol.split('.')[0] if '.' in symbol else symbol
        if code.startswith(('300', '301')):
            return 0.20
        if code.startswith('688'):
            return 0.20
        if code.startswith(('4', '8')):
            return 0.30
        return 0.10

    @staticmethod
    def check_hfq_continuity(data_dir: str, symbol: str,
                              start_date: str, end_date: str,
                              period: str = '1d',
                              threshold: float = None,
                              year_boundary_only: bool = True) -> List[Dict]:
        market_dir = os.path.join(data_dir, 'market', symbol)
        if not os.path.isdir(market_dir):
            return []

        period_suffix = FutuDataProcessor._map_period(period)
        start_year = pd.Timestamp(start_date).year
        end_year = pd.Timestamp(end_date).year

        frames = []
        for year in range(start_year, end_year + 1):
            fpath = os.path.join(market_dir, f"{year}_{period_suffix}.parquet")
            if os.path.exists(fpath):
                try:
                    df = pd.read_parquet(fpath)
                    if df is not None and not df.empty:
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                        frames.append(df)
                except Exception:
                    pass

        if len(frames) < 2:
            return []

        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        merged = merged[(merged.index >= start_ts) & (merged.index <= end_ts)]

        if len(merged) < 2:
            return []

        raw_dir = os.path.join(data_dir, 'market_raw', symbol)
        raw_merged = None
        if os.path.isdir(raw_dir):
            raw_frames = []
            for year in range(start_year, end_year + 1):
                fpath = os.path.join(raw_dir, f"{year}_{period_suffix}.parquet")
                if os.path.exists(fpath):
                    try:
                        df = pd.read_parquet(fpath)
                        if df is not None and not df.empty:
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index)
                            raw_frames.append(df)
                    except Exception:
                        pass
            if raw_frames:
                raw_merged = pd.concat(raw_frames).sort_index()
                raw_merged = raw_merged[~raw_merged.index.duplicated(keep='last')]
                raw_merged = raw_merged[(raw_merged.index >= start_ts) & (raw_merged.index <= end_ts)]

        if threshold is None:
            limit_ratio = FutuHFQChecker.get_limit_ratio(symbol)
            threshold = max(FutuHFQChecker.HFQ_CHANGE_THRESHOLD, limit_ratio + 0.08)

        issues = []
        closes = merged['close'].values
        dates = merged.index

        for i in range(1, len(closes)):
            prev_close = closes[i - 1]
            curr_close = closes[i]

            if prev_close <= 0 or curr_close <= 0:
                continue
            if np.isnan(prev_close) or np.isnan(curr_close):
                continue

            pct_change = (curr_close / prev_close) - 1.0

            if year_boundary_only:
                if dates[i - 1].year == dates[i].year:
                    continue

            if abs(pct_change) > threshold:
                is_real_market_event = False
                if raw_merged is not None and not raw_merged.empty:
                    prev_date = dates[i - 1]
                    curr_date = dates[i]
                    prev_raw = raw_merged.loc[prev_date, 'close'] if prev_date in raw_merged.index else None
                    curr_raw = raw_merged.loc[curr_date, 'close'] if curr_date in raw_merged.index else None
                    if prev_raw is not None and curr_raw is not None:
                        try:
                            prev_raw_f = float(prev_raw)
                            curr_raw_f = float(curr_raw)
                            if prev_raw_f > 0 and curr_raw_f > 0:
                                raw_pct = abs((curr_raw_f / prev_raw_f) - 1.0)
                                if raw_pct > threshold * 0.5:
                                    is_real_market_event = True
                        except (ValueError, TypeError):
                            pass

                if not is_real_market_event:
                    issues.append({
                        'symbol': symbol,
                        'prev_date': dates[i - 1].strftime('%Y-%m-%d'),
                        'curr_date': dates[i].strftime('%Y-%m-%d'),
                        'prev_close': float(prev_close),
                        'curr_close': float(curr_close),
                        'pct_change': float(pct_change),
                        'is_year_boundary': dates[i - 1].year != dates[i].year,
                    })

        return issues


def run_check_hfq(processor: FutuDataProcessor,
                   start_date: str, end_date: str,
                   fix: bool = False, dry_run: bool = False,
                   year_boundary_only: bool = True,
                   threshold: float = 0.25,
                   stocks: Optional[str] = None,
                   report_path: str = ''):
    logger.info(f"{'='*60}")
    logger.info(f"开始检查后复权数据连续性 (Futu): {start_date} ~ {end_date}")
    if year_boundary_only:
        logger.info("检查模式: 仅年份边界")
    else:
        logger.info("检查模式: 全量（含日内异常）")
    logger.info(f"{'='*60}")

    data_dir = processor._data_dir
    market_dir = os.path.join(data_dir, 'market')

    if not os.path.exists(market_dir):
        logger.error(f"后复权数据目录不存在: {market_dir}")
        return

    stock_dirs = sorted([d for d in os.listdir(market_dir)
                         if os.path.isdir(os.path.join(market_dir, d))])

    if stocks:
        specified = set(s.strip() for s in stocks.split(','))
        stock_dirs = [s for s in stock_dirs if s in specified]

    total = len(stock_dirs)
    logger.info(f"扫描 {total} 只股票...")

    all_issues = []
    for i, symbol in enumerate(stock_dirs, 1):
        if i % 200 == 0:
            logger.info(f"进度: {i}/{total}, 已发现 {len(all_issues)} 处异常")

        issues = FutuHFQChecker.check_hfq_continuity(
            data_dir, symbol, start_date, end_date,
            threshold=threshold,
            year_boundary_only=year_boundary_only,
        )
        all_issues.extend(issues)

    if not all_issues:
        logger.info("未发现后复权数据连续性问题！")
        return

    by_symbol = defaultdict(list)
    for issue in all_issues:
        by_symbol[issue['symbol']].append(issue)

    logger.info(f"\n{'='*60}")
    logger.info(f"检测结果: {len(by_symbol)} 只股票存在 {len(all_issues)} 处异常")
    logger.info(f"{'='*60}")

    for symbol in sorted(by_symbol.keys()):
        issues = by_symbol[symbol]
        boundary_count = sum(1 for iss in issues if iss['is_year_boundary'])
        intraday_count = len(issues) - boundary_count
        parts = []
        if boundary_count:
            parts.append(f"年份边界断裂{boundary_count}处")
        if intraday_count:
            parts.append(f"日内异常{intraday_count}处")
        logger.info(f"  {symbol}: {', '.join(parts)}")
        for issue in issues:
            pct = issue['pct_change'] * 100
            tag = "[边界]" if issue['is_year_boundary'] else "[日内]"
            logger.info(
                f"    {tag} {issue['prev_date']} -> {issue['curr_date']}: "
                f"{issue['prev_close']:.4f} -> {issue['curr_close']:.4f} ({pct:+.2f}%)"
            )

    if fix:
        logger.info(f"\n{'='*60}")
        logger.info(f"开始修复 {'(DRY RUN)' if dry_run else ''}")
        logger.info(f"{'='*60}")

        progress = FutuDownloadProgress(len(by_symbol))
        period_suffix = FutuDataProcessor._map_period('1d')

        for symbol in sorted(by_symbol.keys()):
            if dry_run:
                logger.info(f"  [DRY RUN] {symbol}: 将删除旧缓存并重新获取 {start_date} ~ {end_date}")
                progress.record_skipped()
                continue

            logger.info(f"  修复 {symbol}...")

            market_sym_dir = os.path.join(data_dir, 'market', symbol)
            start_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year

            for year in range(start_year, end_year + 1):
                fpath = os.path.join(market_sym_dir, f"{year}_{period_suffix}.parquet")
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        logger.warning(f"  删除失败 {fpath}: {e}")

            success = processor._download_missing_data(
                symbol, start_date, end_date, '1d', 'market'
            )

            if success:
                verify_issues = FutuHFQChecker.check_hfq_continuity(
                    data_dir, symbol, start_date, end_date,
                    year_boundary_only=True,
                )
                if verify_issues:
                    logger.warning(f"  {symbol}: 修复后仍存在 {len(verify_issues)} 处异常")
                    progress.record_failed()
                else:
                    logger.info(f"  {symbol}: 修复成功")
                    progress.record_downloaded()
            else:
                logger.error(f"  {symbol}: 修复失败")
                progress.record_failed()

        logger.info(f"\n修复完成: {progress.report()}")

    if not report_path:
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'check_futu_hfq_report.html')

    _generate_hfq_report(all_issues, report_path, dry_run)

    logger.info(f"{'='*60}")
    logger.info(f"后复权连续性检查完成")
    logger.info(f"{'='*60}")


def _generate_hfq_report(issues, output_path, dry_run):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    by_symbol = defaultdict(list)
    for issue in issues:
        by_symbol[issue['symbol']].append(issue)

    type_counts = defaultdict(int)
    for issue in issues:
        if issue['is_year_boundary']:
            type_counts['年份边界断裂'] += 1
        else:
            type_counts['日内异常波动'] += 1

    css = (
        "body { font-family: 'Microsoft YaHei', sans-serif; margin: 20px; background: #f5f5f5; }"
        "h1 { color: #333; }"
        ".summary { display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap; }"
        ".card { background: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }"
        ".card h3 { margin: 0 0 5px; color: #666; font-size: 13px; }"
        ".card .value { font-size: 24px; font-weight: bold; }"
        ".card.red .value { color: #ff4d4f; }"
        ".card.orange .value { color: #fa8c16; }"
        ".card.blue .value { color: #1890ff; }"
        "table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; }"
        "th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }"
        "th { background: #fafafa; font-weight: bold; }"
        "tr:hover { background: #f0f7ff; }"
        ".badge { padding: 2px 6px; border-radius: 4px; font-size: 11px; white-space: nowrap; }"
        ".badge-boundary { background: #fff1f0; color: #cf1322; }"
        ".badge-intraday { background: #fff7e6; color: #d46b08; }"
    )

    parts = []
    parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    parts.append('<title>富途后复权数据连续性检查报告</title>')
    parts.append(f'<style>{css}</style></head><body>')
    parts.append('<h1>富途后复权数据连续性检查报告</h1>')
    parts.append(f'<p>生成时间: {now} {"(DRY RUN)" if dry_run else ""}</p>')
    parts.append('<div class="summary">')
    parts.append(f'<div class="card blue"><h3>受影响股票数</h3><div class="value">{len(by_symbol)}</div></div>')
    parts.append(f'<div class="card red"><h3>异常点总数</h3><div class="value">{len(issues)}</div></div>')
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        parts.append(f'<div class="card orange"><h3>{t}</h3><div class="value">{count}</div></div>')
    parts.append('</div>')
    parts.append('<h2>问题详情</h2>')
    parts.append('<table><tr><th>股票</th><th>前一交易日</th><th>当前交易日</th><th>前收盘价</th><th>当前收盘价</th><th>涨跌幅</th><th>类型</th></tr>')
    for symbol in sorted(by_symbol.keys()):
        symbol_issues = by_symbol[symbol]
        for j, issue in enumerate(symbol_issues):
            symbol_cell = symbol if j == 0 else ''
            badge = '年份边界断裂' if issue['is_year_boundary'] else '日内异常波动'
            badge_css = 'boundary' if issue['is_year_boundary'] else 'intraday'
            pct = issue['pct_change'] * 100
            pct_color = 'red' if pct < 0 else 'green'
            parts.append(
                f'<tr><td>{symbol_cell}</td>'
                f'<td>{issue["prev_date"]}</td>'
                f'<td>{issue["curr_date"]}</td>'
                f'<td>{issue["prev_close"]:.4f}</td>'
                f'<td>{issue["curr_close"]:.4f}</td>'
                f'<td style="color:{pct_color}">{pct:+.2f}%</td>'
                f'<td><span class="badge badge-{badge_css}">{badge}</span></td></tr>'
            )
    parts.append('</table></body></html>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    logger.info(f"报告已保存: {output_path}")


def _needs_futu_download(processor: FutuDataProcessor, symbol: str,
                          start_date: str, end_date: str,
                          period: str, sub_dir: str) -> bool:
    """检查是否有缺失年份需要下载"""
    missing_years = processor._get_missing_years(symbol, start_date, end_date, period, sub_dir)
    return len(missing_years) > 0


def download_futu_batch(processor: FutuDataProcessor,
                        stock_list: List[str],
                        start_date: str, end_date: str,
                        period: str, sub_dirs: List[str],
                        force: bool) -> FutuDownloadProgress:
    progress = FutuDownloadProgress(len(stock_list) * len(sub_dirs))
    total_tasks = len(stock_list) * len(sub_dirs)
    task_idx = 0

    for symbol in stock_list:
        for sub_dir in sub_dirs:
            task_idx += 1
            data_type = '后复权' if sub_dir == 'market' else '不复权'

            try:
                if not force and not _needs_futu_download(
                    processor, symbol, start_date, end_date, period, sub_dir
                ):
                    progress.record_cached()
                    logger.debug(
                        f"[ {task_idx} / {total_tasks} ] {symbol} {data_type} - 缓存命中"
                    )
                    if task_idx % 5 == 0 or task_idx == total_tasks:
                        logger.info(progress.report())
                    continue

                logger.debug(f"[{symbol}] 开始下载 {data_type} 数据...")
                success = processor._download_missing_data(
                    symbol, start_date, end_date, period, sub_dir
                )

                if success:
                    # 下载后再次检查，确认数据确实存在
                    if _needs_futu_download(processor, symbol, start_date, end_date, period, sub_dir):
                        progress.record_skipped()
                        logger.warning(
                            f"[ {task_idx} / {total_tasks} ] {symbol} {data_type} - "
                            f"下载完成但数据仍缺失（可能未上市或已退市）"
                        )
                    else:
                        progress.record_downloaded()
                        logger.info(
                            f"[ {task_idx} / {total_tasks} ] {symbol} {data_type} - 下载成功"
                        )
                else:
                    progress.record_failed()
                    logger.error(
                        f"[ {task_idx} / {total_tasks} ] {symbol} {data_type} - 下载失败"
                    )

            except Exception as e:
                progress.record_failed()
                logger.error(
                    f"[ {task_idx} / {total_tasks} ] {symbol} {data_type} - 下载失败: {e}"
                )

            if task_idx % 5 == 0 or task_idx == total_tasks:
                logger.info(progress.report())

    return progress


def main():
    parser = argparse.ArgumentParser(
        description='批量下载富途行情数据（支持增量下载与跳过已缓存）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载沪深300后复权日线行情数据
  python download_futu_data.py --pool 沪深300 --start 2020-01-01 --end 2026-12-31

  # 下载指定股票的后复权和不复权数据
  python download_futu_data.py --stocks 601398.SH,600036.SH --start 2020-01-01 --end 2026-12-31 --type all

  # 下载1分钟数据
  python download_futu_data.py --stocks 601398.SH --start 2025-01-01 --end 2026-12-31 --period 1m

  # 强制重新下载（忽略已有缓存）
  python download_futu_data.py --pool 沪深300 --start 2020-01-01 --end 2026-12-31 --force

  # 启用详细日志 + 写入日志文件
  python download_futu_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-12-31 --verbose --log
        """,
    )
    parser.add_argument('--pool', type=str, default=None,
                        help='股票池板块名称: 沪深300, 中证500, 中证1000, 上证50, 沪深A股')
    parser.add_argument('--stocks', type=str, default=None,
                        help='手动指定股票代码，逗号分隔，如 601398.SH,600036.SH')
    parser.add_argument('--start', type=str, required=True,
                        help='数据起始日期，如 2020-01-01')
    parser.add_argument('--end', type=str, required=True,
                        help='数据结束日期，如 2026-12-31')
    parser.add_argument('--period', type=str, default='1d',
                        choices=['1d', '1m', '5m', '15m', '30m', '60m'],
                        help='数据周期 (默认: 1d)')
    parser.add_argument('--type', type=str, default='adjusted',
                        choices=['adjusted', 'raw', 'all'],
                        help='数据类型: adjusted=后复权, raw=不复权, all=两者都下载 (默认: adjusted)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='强制重新下载，忽略已有缓存')
    parser.add_argument('--check-hfq', action='store_true', default=False,
                        help='检查后复权数据连续性（年份边界断裂检测）')
    parser.add_argument('--full-scan', action='store_true', default=False,
                        help='与 --check-hfq 配合，检查所有日期（不仅年份边界）')
    parser.add_argument('--hfq-threshold', type=float, default=0.25,
                        help='后复权连续性检查的涨跌幅异常阈值（默认0.25即25%%）')
    parser.add_argument('--fix', action='store_true', default=False,
                        help='与 --check-hfq 配合，自动修复断裂数据')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='与 --fix 配合，只显示修复计划不执行')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='指定缓存数据存储目录 (默认: 项目根目录下的 .cache 文件夹)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='启用详细日志输出')
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

    # 初始化富途处理器
    processor = FutuDataProcessor(
        data_dir=args.cache_dir if args.cache_dir else ''
    )

    # 检查富途OpenD服务是否可用
    if not processor.check_futu_service():
        logger.error("富途OpenD服务未开启，请启动 OpenD 后重试")
        sys.exit(1)

    logger.info(f"富途OpenD服务已连接 ({processor._futu_host}:{processor._futu_port})")

    if args.check_hfq:
        check_start = args.start or '2014-01-01'
        check_end = args.end or f'{datetime.now().year}-12-31'
        run_check_hfq(
            processor,
            check_start, check_end,
            fix=args.fix, dry_run=args.dry_run,
            year_boundary_only=not args.full_scan,
            threshold=args.hfq_threshold,
            stocks=args.stocks,
        )
        return

    # 初始化 QMT 和 OpenData 处理器（用于获取股票列表）
    qmt_processor = None
    try:
        qmt_processor = QMTDataProcessor(fallback_to_simulated=False)
        logger.info("QMT 处理器初始化成功")
    except Exception as e:
        logger.warning(f"QMT 处理器初始化失败: {e}，将使用 OpenData 获取股票列表")

    opendata_processor = OpenDataProcessor(fallback_to_simulated=False)

    # 获取股票列表
    stock_list = resolve_stock_list(
        qmt_processor, opendata_processor, args.pool, args.stocks,
        args.start, args.end
    )
    if not stock_list:
        logger.error("未获取到任何股票，退出")
        sys.exit(1)

    # 确定下载的数据类型
    sub_dirs = []
    if args.type == 'all':
        sub_dirs = ['market', 'market_raw']
    elif args.type == 'adjusted':
        sub_dirs = ['market']
    else:
        sub_dirs = ['market_raw']

    type_desc = ', '.join(
        '后复权' if d == 'market' else '不复权' for d in sub_dirs
    )

    logger.info(f"{'='*60}")
    logger.info(f"开始下载富途行情数据: {len(stock_list)} 只股票")
    logger.info(f"  日期范围: {args.start} ~ {args.end}")
    logger.info(f"  周期: {args.period}")
    logger.info(f"  数据类型: {type_desc}")
    logger.info(f"  强制重新下载: {'是' if args.force else '否'}")
    logger.info(f"{'='*60}")

    # 批量下载
    progress = download_futu_batch(
        processor, stock_list,
        args.start, args.end, args.period,
        sub_dirs, args.force,
    )

    logger.info(f"{'='*60}")
    logger.info(f"富途行情数据下载完成: {progress.report()}")
    logger.info(f"{'='*60}")

    # 验证
    logger.info("\n验证保存的数据...")
    for sub_dir in sub_dirs:
        cache_root = os.path.join(processor._data_dir, sub_dir)
        data_type = '后复权' if sub_dir == 'market' else '不复权'
        sample_count = 0
        for symbol in stock_list[:5]:
            symbol_dir = os.path.join(cache_root, symbol)
            if not os.path.exists(symbol_dir):
                continue
            period_suffix = FutuDataProcessor._map_period(args.period)
            files = sorted([
                f for f in os.listdir(symbol_dir)
                if f.endswith(f'_{period_suffix}.parquet')
            ])
            if files:
                sample_count += 1
                logger.info(f"  {symbol} [{data_type}]: {len(files)} 个文件 - {files[-1]}")

        if sample_count == 0:
            logger.info(f"  [{data_type}] 无有效数据文件")

    logger.info(f"\n全部完成！数据保存在 {processor._data_dir}")


if __name__ == '__main__':
    main()
