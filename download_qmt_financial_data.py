import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd

from core.cache import cache_manager
from core.data.index_constituent import IndexConstituentManager
from core.data.opendata import OpenDataProcessor
from core.data.qmt import QMTDataProcessor


def setup_logger(log_to_file: bool = False, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger('download_qmt_financial_data')
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
            log_file = f"{log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_download_financial_data.log"

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


logger = logging.getLogger('download_qmt_financial_data')


class FinancialDownloadProgress:
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


def _needs_financial_download(symbol: str, tables: List[str],
                               start_time: str, end_time: str,
                               report_type: str) -> bool:
    idx = cache_manager.index_manager
    req_years = cache_manager._parse_years_from_range(
        start_time if start_time else '20000101',
        end_time if end_time else datetime.now().strftime('%Y%m%d')
    )
    if not req_years:
        return True

    current_year = datetime.now().year
    current_month = datetime.now().month

    for table in tables:
        table_suffix = f"{table}_{report_type}"
        available = idx.get_available_financial_years(symbol, table_suffix)
        if not available:
            namespace = 'QMTDataProcessor_Financial'
            available = cache_manager.disk_cache.list_yearly_files(
                namespace, symbol, table_suffix
            )
        checked = set(idx.get_checked_financial_years(symbol, table_suffix))

        cached_years = set(available)
        missing = set(req_years) - cached_years - checked

        # 过滤掉当前年份（当年财报可能尚未完全公布）
        # 以及未来年份（尚无数据）
        truly_missing = set()
        for y in missing:
            if y > current_year:
                continue
            if y == current_year and current_month <= 8:
                # 当年且在8月前，年报尚未全部披露完毕，跳过
                continue
            truly_missing.add(y)

        if truly_missing:
            logger.debug(
                f"[{symbol}] {table_suffix} 需要下载: "
                f"缺失年份={sorted(truly_missing)}, "
                f"已缓存={sorted(cached_years)}, "
                f"已检查={sorted(checked)}"
            )
            return True

    logger.debug(f"[{symbol}] 所有报表已缓存，跳过下载")
    return False


def resolve_stock_list(qmt_processor: Optional[QMTDataProcessor],
                        opendata_processor: OpenDataProcessor,
                        pool: Optional[str],
                        stocks: Optional[str]) -> List[str]:
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

        logger.info(f"获取板块 '{pool}' 当前成分股...")
        stock_list = []
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


def download_financial_batch(qmt_processor: QMTDataProcessor,
                              stock_list: List[str],
                              tables: List[str],
                              start_time: str,
                              end_time: str,
                              report_type: str,
                              force: bool) -> FinancialDownloadProgress:
    progress = FinancialDownloadProgress(len(stock_list))

    for i, symbol in enumerate(stock_list, 1):
        try:
            if not force and not _needs_financial_download(symbol, tables, start_time, end_time, report_type):
                progress.record_cached()
                logger.info(f"[ {i} / {progress.total} ] {symbol} - 缓存命中")
                if i % 5 == 0 or i == progress.total:
                    logger.info(progress.report())
                continue

            logger.debug(f"[{symbol}] 开始下载财务数据...")
            data = qmt_processor.xtdata.get_financial_data(
                [symbol], tables,
                start_time=start_time, end_time=end_time,
                report_type=report_type,
            )

            if data and symbol in data and data[symbol]:
                stock_data = data[symbol]
                has_valid_data = False
                any_new_written = False
                tables_info = []

                for table, df in stock_data.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        has_valid_data = True
                        df = qmt_processor._normalize_qmt_financial_df(df, report_type)
                        table_suffix = f"{table}_{report_type}"

                        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                            written = cache_manager.disk_cache.put_yearly_from_df(
                                'QMTDataProcessor_Financial', symbol, table_suffix, df,
                                skip_existing=True,
                            )
                            for y in written:
                                cache_manager.index_manager.update_financial_index(symbol, table_suffix, y)

                            if written:
                                any_new_written = True
                                years_str = ', '.join([str(y) for y in sorted(written)[:5]])
                                if len(written) > 5:
                                    years_str += f' 等{len(written)}个年份'
                                tables_info.append(f"{table}({years_str})")
                            else:
                                tables_info.append(f"{table}(已存在)")
                        else:
                            time_suffix = f"_{start_time}_{end_time}" if start_time or end_time else ""
                            cache_key = f"{symbol}{time_suffix}_{table}_{report_type}"
                            cache_manager.disk_cache.put('QMTDataProcessor_Financial', cache_key, df, 'parquet')
                            any_new_written = True
                            tables_info.append(f"{table}({len(df)}行)")
                    else:
                        tables_info.append(f"{table}(空)")

                if has_valid_data:
                    if any_new_written:
                        progress.record_downloaded()
                        detail = ', '.join(tables_info[:4])
                        if len(tables_info) > 4:
                            detail += f' 等{len(tables_info)}表'
                        logger.info(f"[ {i} / {progress.total} ] {symbol} - 下载成功: {detail}")
                    else:
                        progress.record_cached()
                        logger.info(f"[ {i} / {progress.total} ] {symbol} - 缓存命中（无新数据）")
                else:
                    progress.record_skipped()
                    logger.warning(f"[ {i} / {progress.total} ] {symbol} - 数据为空")
            else:
                progress.record_skipped()
                logger.warning(f"[ {i} / {progress.total} ] {symbol} - 无数据")

        except Exception as e:
            progress.record_failed()
            logger.error(f"[ {i} / {progress.total} ] {symbol} - 下载失败: {e}")

        if i % 5 == 0 or i == progress.total:
            logger.info(progress.report())

    return progress


def main():
    parser = argparse.ArgumentParser(
        description='批量下载QMT财务数据（支持跳过已下载缓存）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载沪深A股全部财务数据
  python download_qmt_financial_data.py --pool 沪深A股 --start 20200101 --end 20260428

  # 下载沪深300财务数据（仅资产负债表和利润表）
  python download_qmt_financial_data.py --pool 沪深300 --start 20200101 --end 20260428 --tables Balance,Income

  # 指定股票代码下载
  python download_qmt_financial_data.py --stocks 000001.SZ,600000.SH --start 20200101 --end 20260428

  # 强制重新下载（忽略已有缓存）
  python download_qmt_financial_data.py --pool 沪深A股 --start 20200101 --end 20260428 --force

  # 启用详细日志 + 写入日志文件
  python download_qmt_financial_data.py --pool 沪深A股 --start 20200101 --end 20260428 --verbose --log

  # 使用按报告期筛选
  python download_qmt_financial_data.py --pool 沪深A股 --start 20200101 --end 20260428 --report-type report_time
        """,
    )
    parser.add_argument('--pool', type=str, default=None,
                        help='股票池板块名称: 沪深300, 中证500, 中证1000, 上证50, 沪深A股')
    parser.add_argument('--stocks', type=str, default=None,
                        help='手动指定股票代码，逗号分隔，如 000001.SZ,600000.SH')
    parser.add_argument('--start', type=str, required=True,
                        help='数据起始日期，如 20200101')
    parser.add_argument('--end', type=str, required=True,
                        help='数据结束日期，如 20260428')
    parser.add_argument('--tables', type=str, default=None,
                        help='财务报表列表，逗号分隔，如 Balance,Income。默认下载全部报表')
    parser.add_argument('--report-type', type=str, default='announce_time',
                        choices=['announce_time', 'report_time'],
                        help='报表筛选方式: announce_time=按披露日期(默认), report_time=按报告期')
    parser.add_argument('--force', action='store_true', default=False,
                        help='强制重新下载，忽略已有缓存')
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

    if args.cache_dir:
        cache_manager.configure(cache_dir=args.cache_dir)

    # 解析财务报表列表
    all_tables = QMTDataProcessor.FINANCIAL_TABLES
    if args.tables:
        tables = [t.strip() for t in args.tables.split(',')]
        invalid = set(tables) - set(all_tables)
        if invalid:
            logger.error(f"无效的报表名称: {invalid}，可用报表: {all_tables}")
            sys.exit(1)
    else:
        tables = list(all_tables)

    # 初始化 QMT 处理器
    qmt_processor = None
    try:
        qmt_processor = QMTDataProcessor(fallback_to_simulated=False)
        logger.info("QMT 处理器初始化成功")
    except Exception as e:
        logger.error(f"QMT 处理器初始化失败: {e}，财务数据下载需要 QMT")
        sys.exit(1)

    if not qmt_processor.xtdata:
        logger.error("xtquant 未安装，无法下载财务数据")
        sys.exit(1)

    # 初始化 OpenData 处理器（仅用于获取股票列表）
    opendata_processor = OpenDataProcessor(fallback_to_simulated=False)

    # 获取股票列表
    stock_list = resolve_stock_list(
        qmt_processor, opendata_processor, args.pool, args.stocks
    )
    if not stock_list:
        logger.error("未获取到任何股票，退出")
        sys.exit(1)

    logger.info(f"{'='*60}")
    logger.info(f"开始下载财务数据: {len(stock_list)} 只股票")
    logger.info(f"  日期范围: {args.start} ~ {args.end}")
    logger.info(f"  报表: {', '.join(tables)}")
    logger.info(f"  筛选方式: {args.report_type}")
    logger.info(f"  强制重新下载: {'是' if args.force else '否'}")
    logger.info(f"{'='*60}")

    # 先调用 QMT 的 download_financial_data2 批量下载数据到本地
    logger.info("步骤1: 从 QMT 服务器批量下载财务数据到本地...")
    try:
        qmt_processor.xtdata.download_financial_data2(
            stock_list, tables,
            start_time=args.start, end_time=args.end,
        )
        logger.info("QMT 服务器数据下载完成，等待数据同步...")
        time.sleep(2)
    except Exception as e:
        logger.warning(f"QMT 批量下载失败: {e}，将逐只获取")

    # 逐只获取数据并保存到缓存
    logger.info("步骤2: 逐只获取数据并保存到缓存...")
    progress = download_financial_batch(
        qmt_processor, stock_list, tables,
        args.start, args.end, args.report_type, args.force,
    )

    # 下载完成后统一保存索引
    logger.info("正在保存索引文件...")
    cache_manager.index_manager.save_index()
    logger.info("索引文件保存完成")

    logger.info(f"{'='*60}")
    logger.info(f"财务数据下载完成: {progress.report()}")
    logger.info(f"{'='*60}")
    logger.info("全部下载任务完成!")


if __name__ == '__main__':
    main()
