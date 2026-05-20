import argparse
import itertools
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from xtquant import xtdata
except ImportError:
    xtdata = None
    print("错误: xtquant 未安装，请先安装 MiniQMT 客户端并配置 xtquant")

from core.cache import cache_manager


SPINNER_CHARS = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'


class DownloadSpinner:
    def __init__(self, idx: int, total: int, symbol: str):
        self.idx = idx
        self.total = total
        self.symbol = symbol
        self._stop = threading.Event()
        self._start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for ch in itertools.cycle(SPINNER_CHARS):
            if self._stop.is_set():
                break
            elapsed = time.time() - self._start
            msg = f"\r  [{self.idx}/{self.total}] {ch} get_data({self.symbol}) {elapsed:.1f}s"
            sys.stderr.write(msg)
            sys.stderr.flush()
            self._stop.wait(0.2)
        clear_len = 80
        sys.stderr.write('\r' + ' ' * clear_len + '\r')
        sys.stderr.flush()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)


def setup_logger(log_to_file: bool = False, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger('download_qmt_data')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    if log_to_file:
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_download_qmt_data.log"
        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"日志将同时写入文件: {log_file}")

    return logger


logger = logging.getLogger('download_qmt_data')


def _delete_cache(symbol: str, period: str, data_type: str):
    """删除指定标的的缓存，触发重新下载"""
    if data_type in ('adjusted', 'all'):
        namespace = 'QMTDataProcessor'
        years = cache_manager.disk_cache.list_yearly_files(namespace, symbol, period)
        for year in years:
            cache_manager.disk_cache.delete_yearly(namespace, symbol, year, period)
        cache_manager.index_manager.remove_market_index(symbol, period)

    if data_type in ('raw', 'all'):
        namespace = 'QMTDataProcessor_Raw'
        years = cache_manager.disk_cache.list_yearly_files(namespace, symbol, period)
        for year in years:
            cache_manager.disk_cache.delete_yearly(namespace, symbol, year, period)
        cache_manager.index_manager.remove_market_raw_index(symbol, period)

    cache_manager.mem_cache.clear()


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1073741824:
        return f"{size_bytes / 1073741824:.1f} GB"
    if size_bytes >= 1048576:
        return f"{size_bytes / 1048576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def show_cache_info(symbols: List[str]):
    """显示缓存信息（通过 cache_manager API）"""
    cache_dir = cache_manager.disk_cache.cache_dir / 'QMTData'
    logger.info(f"\n{'='*60}")
    logger.info(f"缓存信息 (目录: {cache_dir})")
    logger.info(f"{'='*60}")

    for symbol in symbols:
        # 后复权
        years = cache_manager.index_manager.get_available_market_years(symbol, '1d')
        if not years:
            years = cache_manager.disk_cache.list_yearly_files('QMTDataProcessor', symbol, '1d')
        if years:
            year_dir = cache_manager.disk_cache._get_yearly_dir('QMTDataProcessor', symbol)
            total_size = sum(f.stat().st_size for f in year_dir.glob('*.parquet')) if year_dir.exists() else 0
            size_str = _format_size(total_size)
            logger.info(f"  {symbol} 后复权: {len(years)} 个年份 "
                        f"({min(years)}~{max(years)}), 总大小={size_str}")
        else:
            logger.info(f"  {symbol} 后复权: 无缓存")

        # 不复权
        raw_years = cache_manager.index_manager.get_available_market_raw_years(symbol, '1d')
        if not raw_years:
            raw_years = cache_manager.disk_cache.list_yearly_files('QMTDataProcessor_Raw', symbol, '1d')
        if raw_years:
            year_dir = cache_manager.disk_cache._get_yearly_dir('QMTDataProcessor_Raw', symbol)
            total_size = sum(f.stat().st_size for f in year_dir.glob('*.parquet')) if year_dir.exists() else 0
            size_str = _format_size(total_size)
            logger.info(f"  {symbol} 不复权: {len(raw_years)} 个年份 "
                        f"({min(raw_years)}~{max(raw_years)}), 总大小={size_str}")
        else:
            logger.info(f"  {symbol} 不复权: 无缓存")

    logger.info(f"{'='*60}")


def load_historical_constituents(index_code: str) -> List[str]:
    csv_path = Path(__file__).parent / '.cache' / 'JQData' / 'index_constituent' / f'{index_code}.csv'
    if not csv_path.exists():
        return []

    df = pd.read_csv(str(csv_path))
    if 'codes' not in df.columns:
        return []

    all_codes = set()
    for codes_str in df['codes']:
        try:
            codes = eval(str(codes_str))
            if isinstance(codes, (list, tuple)):
                all_codes.update(codes)
        except Exception:
            pass

    return sorted(all_codes)


def resolve_stock_list(stock_arg: Optional[str], pool: Optional[str],
                        index: Optional[str] = None) -> List[str]:
    if stock_arg:
        stock_list = []
        for s in stock_arg.split(','):
            s = s.strip()
            if '.' not in s:
                s = f"{s}.SH" if s.startswith(('6', '9')) else f"{s}.SZ"
            stock_list.append(s)
        logger.info(f"使用手动指定股票列表: {len(stock_list)} 只")
        return stock_list

    if index:
        historical = load_historical_constituents(index)
        if historical:
            logger.info(f"从 JQData 历史成分股获取到 {len(historical)} 只 ({index})")
            return historical

        if xtdata:
            logger.info(f"JQData 无 {index} 成分股CSV，尝试从 QMT 获取...")
            stock_list = xtdata.get_stock_list_in_sector(index)
            if stock_list:
                logger.info(f"从 QMT 获取到 {len(stock_list)} 只 {index} 股票")
                return stock_list

        raise RuntimeError(f"无法获取指数 '{index}' 的成分股")

    if pool:
        if not xtdata:
            raise RuntimeError("xtquant 未安装，无法获取板块成分股")
        logger.info(f"正在获取 {pool} 成分股列表...")
        stock_list = xtdata.get_stock_list_in_sector(pool)
        if not stock_list:
            logger.info(f"本地无缓存，正在下载板块数据...")
            try:
                xtdata.download_sector_data()
            except Exception:
                pass
            stock_list = xtdata.get_stock_list_in_sector(pool)
        if stock_list:
            logger.info(f"获取到 {len(stock_list)} 只 {pool} 股票")
            return stock_list
        raise RuntimeError(f"无法获取板块 '{pool}' 的成分股")

    raise ValueError("请指定 --stocks, --index 或 --pool")


def main():
    parser = argparse.ArgumentParser(
        description='使用 QMT 下载行情数据，通过 smart_cache 缓存到本地 Parquet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            '使用示例:',
            '  # 测试单只股票',
            '  python download_qmt_data.py --stocks 000001.SZ --type all',
            '',
            '  # 从JQData历史成分股CSV下载沪深300全部日线(含退市股)',
            '  python download_qmt_data.py --index 000300.SH --type all',
            '',
            '  # 下载沪深300当前成分股日线',
            '  python download_qmt_data.py --pool 沪深300 --type all',
            '',
            '  # 下载全市场日线数据',
            '  python download_qmt_data.py --pool 沪深A股 --type all',
            '',
            '  # 强制重新下载（删除已有缓存后重新获取）',
            '  python download_qmt_data.py --stocks 000001.SZ --type all --force',
            '',
            '  # 查看缓存信息',
            '  python download_qmt_data.py --stocks 000001.SZ --info',
        ]),
    )
    parser.add_argument('--stocks', type=str, default=None,
                        help='手动指定股票代码，逗号分隔，如 000001.SZ,600000.SH')
    parser.add_argument('--index', type=str, default=None,
                        help='指数代码，从JQData历史成分股CSV读取(含退市股): 000300.SH, 000905.SH 等')
    parser.add_argument('--pool', type=str, default=None,
                        help='股票池板块名称(仅当前成分股): 沪深300, 中证500, 沪深A股 等')
    parser.add_argument('--period', type=str, default='1d',
                        choices=['1d', '1w', '1mon'],
                        help='数据周期 (默认: 1d)')
    parser.add_argument('--type', type=str, default='all',
                        choices=['adjusted', 'raw', 'all'],
                        help='数据类型: adjusted=后复权 raw=不复权 all=两者都下载 (默认: all)')
    parser.add_argument('--start', type=str, default='',
                        help='数据起始日期，如 20200101 (留空则下载全部可用历史)')
    parser.add_argument('--end', type=str, default='',
                        help='数据结束日期，如 20261231 (留空则下载到最新)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='强制重新下载，删除已有缓存后重新获取')
    parser.add_argument('--info', action='store_true', default=False,
                        help='仅查看缓存信息，不下载')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='启用详细日志输出')
    parser.add_argument('--log', action='store_true', default=False,
                        help='将日志同时写入文件')

    args = parser.parse_args()

    setup_logger(log_to_file=args.log, verbose=args.verbose)

    if xtdata is None:
        logger.error("xtquant 未安装，无法下载数据。请先安装 MiniQMT 客户端。")
        sys.exit(1)

    stock_list = None
    if args.stocks:
        stock_list = []
        for s in args.stocks.split(','):
            s = s.strip()
            if '.' not in s:
                s = f"{s}.SH" if s.startswith(('6', '9')) else f"{s}.SZ"
            stock_list.append(s)
    elif args.index:
        stock_list = resolve_stock_list(None, None, args.index)
    elif args.pool:
        stock_list = resolve_stock_list(None, args.pool)
    elif args.info:
        logger.error("--info 需要配合 --stocks, --index 或 --pool 使用")
        sys.exit(1)
    else:
        logger.error("请指定 --stocks, --index 或 --pool")
        parser.print_help()
        sys.exit(1)

    if args.info:
        show_cache_info(stock_list)
        return

    # 创建 QMTDataProcessor（不使用 OpenData 补充）
    from core.data.qmt import QMTDataProcessor
    processor = QMTDataProcessor(use_opendata=False)

    total = len(stock_list)
    data_types = []
    if args.type in ('adjusted', 'all'):
        data_types.append('adjusted')
    if args.type in ('raw', 'all'):
        data_types.append('raw')

    start_date = args.start if args.start else '19900101'
    end_date = args.end if args.end else '20991231'

    cache_dir = cache_manager.disk_cache.cache_dir / 'QMTData'
    logger.info(f"开始下载: {total} 只股票, 周期={args.period}, 类型={args.type}")
    logger.info(f"缓存目录: {cache_dir}")
    if args.force:
        logger.info("强制模式: 将先删除已有缓存再重新下载")

    stats = {'success': 0, 'cached': 0, 'empty': 0, 'failed': 0}
    start_ts = time.time()

    for i, symbol in enumerate(stock_list, 1):
        spinner = DownloadSpinner(i, total, symbol)

        # 强制模式：先删除缓存
        if args.force:
            _delete_cache(symbol, args.period, args.type)

        stock_has_error = False
        stock_has_data = False

        try:
            for data_type in data_types:
                try:
                    if data_type == 'adjusted':
                        df = processor.get_data(symbol, start_date, end_date, args.period)
                    else:
                        df = processor.get_raw_data(symbol, start_date, end_date, args.period)

                    if df is not None and not df.empty:
                        stock_has_data = True

                except RuntimeError as e:
                    if '无数据' in str(e) or 'QMT' in str(e):
                        logger.debug(f"{symbol} {data_type}: {e}")
                    else:
                        raise
                except Exception as e:
                    logger.warning(f"{symbol} {data_type} 获取失败: {e}")
                    stock_has_error = True

        except Exception as e:
            spinner.stop()
            stats['failed'] += 1
            logger.error(f"[{i}/{total}] ✗ {symbol}: {e}")
            continue

        spinner.stop()

        if stock_has_error:
            stats['failed'] += 1
            tag = '✗'
        elif not stock_has_data:
            stats['empty'] += 1
            tag = '○'
        else:
            stats['success'] += 1
            tag = '✓'

        done = stats['success'] + stats['cached'] + stats['empty'] + stats['failed']
        logger.info(f"[{i}/{total}] {tag} {symbol}  |  "
                    f"✓{stats['success']} △{stats['cached']} ○{stats['empty']} ✗{stats['failed']}")

    elapsed = time.time() - start_ts
    logger.info(f"\n{'='*60}")
    logger.info(f"下载完成: ✓{stats['success']} △{stats['cached']} "
                f"○{stats['empty']} ✗{stats['failed']}, 耗时={elapsed:.1f}秒")
    logger.info(f"{'='*60}")

    show_cache_info(stock_list[:5])


if __name__ == '__main__':
    main()
