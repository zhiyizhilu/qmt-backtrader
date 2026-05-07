import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

import pandas as pd

from core.cache import cache_manager
from core.data.index_constituent import IndexConstituentManager
from core.data.opendata import OpenDataProcessor
from core.data.qmt import QMTDataProcessor


def setup_logger(log_to_file: bool = False, log_file: str = None) -> logging.Logger:
    """设置日志记录器，支持输出到控制台和文件"""
    logger = logging.getLogger('download_market_data')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_to_file:
        if log_file is None:
            # 默认日志文件路径: logs/YYYYMMDD_HHMMSS_download_market_data.log
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


# 全局logger，将在main中初始化
logger = logging.getLogger('download_market_data')


class DownloadProgress:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.cached = 0
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
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
    return bool(missing)


def download_one(symbol: str, start_date: str, end_date: str,
                  period: str, data_type: str, processor: OpenDataProcessor,
                  progress: DownloadProgress, force: bool) -> str:
    try:
        if not force and not _needs_download(symbol, start_date, end_date, period, data_type):
            progress.record_cached()
            logger.debug(f"[{symbol}] 缓存命中，跳过下载")
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
        
        # 对于全市场股票（沪深A股），始终获取当前全部股票列表
        if pool in ('沪深A股', 'A股', '全部A股'):
            logger.info(f"获取 '{pool}' 全部股票列表（当前）...")
            stock_list = []
            
            # 优先使用 QMT
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
            
            # 回退到 OpenData
            stock_list = opendata_processor.get_stock_list(pool)
            if stock_list and len(stock_list) > 10:
                logger.info(f"从 OpenData 获取到 {len(stock_list)} 只 {pool} 股票")
                return stock_list
            else:
                logger.error(f"无法获取有效的 {pool} 股票列表！")
                return []
        
        # 对于指数成分股，使用历史成分股逻辑
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
            # 优先使用 QMT 获取当前成分股
            if qmt_processor:
                try:
                    stock_list = qmt_processor.get_stock_list(pool)
                    if stock_list and len(stock_list) > 10:  # 确保获取到足够多的股票
                        logger.info(f"从 QMT 获取到 {len(stock_list)} 只 {pool} 成分股")
                        return stock_list
                except Exception as e:
                    logger.debug(f"QMT获取成分股失败: {e}")
            # QMT 失败，使用 OpenData
            stock_list = opendata_processor.get_stock_list(pool)
        logger.info(f"板块 '{pool}' 共获取到 {len(stock_list)} 只股票")
        return stock_list

    # 获取全市场股票列表 - 优先使用 QMT
    logger.info("获取沪深A股全部股票列表...")
    stock_list = []
    
    if qmt_processor:
        try:
            stock_list = qmt_processor.get_stock_list('沪深A股')
            if stock_list and len(stock_list) > 1000:  # 确保获取到足够多的股票
                logger.info(f"从 QMT 获取到 {len(stock_list)} 只沪深A股")
                return stock_list
            else:
                logger.warning(f"QMT 返回股票数量过少 ({len(stock_list)} 只)，尝试其他数据源...")
        except Exception as e:
            logger.warning(f"从 QMT 获取股票列表失败: {e}")
    
    # 尝试使用 OpenData (AKShare)
    logger.info("尝试从 OpenData (AKShare) 获取...")
    stock_list = opendata_processor.get_stock_list('沪深A股')
    
    if not stock_list or len(stock_list) <= 10:
        logger.error("无法获取有效的全市场股票列表！")
        return []
        
    logger.info(f"共获取到 {len(stock_list)} 只股票")
    return stock_list


def main():
    parser = argparse.ArgumentParser(
        description='批量并发下载市场行情数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载中证1000不复权行情数据 (5线程并发)
  python download_market_data.py --pool 中证1000 --start 2020-04-28 --end 2026-04-28 --type raw --workers 5

  # 下载沪深300后复权行情数据
  python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --workers 5

  # 同时下载后复权和不复权数据
  python download_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-01-01 --type all --workers 5

  # 指定股票代码下载
  python download_market_data.py --stocks 000001.SZ,600000.SH,600519.SH --start 2020-01-01 --end 2026-01-01 --type all

  # 强制重新下载 (忽略已有缓存)
  python download_market_data.py --pool 中证1000 --start 2020-01-01 --end 2026-01-01 --type raw --force

  # 启用详细日志输出 (显示每个股票的下载状态)
  python download_market_data.py --pool 沪深300 --start 2020-01-01 --end 2026-01-01 --type adjusted --workers 5 --verbose

  # 将日志同时写入文件
  python download_market_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-01-01 --type all --workers 20 --log

  # 指定日志文件路径
  python download_market_data.py --pool 沪深A股 --start 2020-01-01 --end 2026-01-01 --type all --workers 20 --log --log-file logs/my_download.log
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
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='指定缓存数据存储目录 (默认: 项目根目录下的 .cache 文件夹)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='启用详细日志输出，显示每个股票的下载状态')
    parser.add_argument('--log', action='store_true', default=False,
                        help='将日志同时写入到 logs/ 目录下的文件')
    parser.add_argument('--log-file', type=str, default=None,
                        help='指定日志文件路径 (需配合 --log 使用)')

    args = parser.parse_args()

    # 初始化logger
    global logger
    logger = setup_logger(log_to_file=args.log, log_file=args.log_file)

    # 根据 verbose 参数设置日志级别
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

    # 初始化数据处理器
    opendata_processor = OpenDataProcessor(fallback_to_simulated=False)
    
    # 尝试初始化 QMT 处理器（用于获取股票列表）
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

    for dtype in data_types:
        logger.info(f"{'='*60}")
        logger.info(f"开始下载 {dtype} 行情数据: {len(stock_list)} 只股票, "
                     f"日期范围 {args.start} ~ {args.end}, 周期 {args.period}")
        logger.info(f"{'='*60}")

        progress = DownloadProgress(len(stock_list))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for symbol in stock_list:
                future = executor.submit(
                    download_one,
                    symbol, args.start, args.end, args.period,
                    dtype, opendata_processor, progress, args.force,
                )
                futures[future] = symbol

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    status = result.split(':')[0]
                    
                    # 每完成一个任务都输出进度
                    if status == "cached":
                        logger.info(f"[{progress.completed} / {progress.total}] {symbol} - 缓存命中")
                    elif status == "ok":
                        logger.info(f"[{progress.completed} / {progress.total}] {symbol} - 下载成功")
                    elif status == "empty":
                        logger.warning(f"[{progress.completed} / {progress.total}] {symbol} - 数据为空")
                    elif status == "err":
                        logger.error(f"[{progress.completed} / {progress.total}] {symbol} - 下载失败: {result.split(':', 2)[-1]}")
                except Exception as e:
                    logger.error(f"[{progress.completed} / {progress.total}] {symbol} - 异常: {e}")

                # 每5个任务输出一次总体进度报告
                if progress.completed % 5 == 0 or progress.completed == progress.total:
                    logger.info(progress.report())

        # 下载完成后统一保存索引（避免多线程文件冲突）
        logger.info("正在保存索引文件...")
        cache_manager.index_manager.save_index()
        logger.info("索引文件保存完成")

        logger.info(f"{'='*60}")
        logger.info(f"{dtype} 行情数据下载完成: {progress.report()}")
        logger.info(f"{'='*60}")

    logger.info("全部下载任务完成!")


if __name__ == '__main__':
    main()
