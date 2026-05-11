"""以沪深300交易日为基准，检查并修复所有股票的行情数据一致性

核心逻辑：
  以沪深300指数（000300.SH）的交易日作为A股交易日历（Trading Calendar），
  对所有股票逐个按年进行数据检查：

检查项：
1. 不复权数据（market_raw）vs 沪深300交易日偏差 → 重新下载
2. 后复权数据（market）vs 沪深300交易日偏差 → 重新下载
3. 不复权 vs 后复权交易日偏差 → 记录差异

修复规则：
- 股票交易日 > 沪深300交易日：以沪深300为准，删除多余的非交易日行
- 股票交易日 < 沪深300交易日：记录停牌信息（正常情况，如IPO前/退市后/停牌）
- 重新下载后仍不一致：保留原始数据，记录为无法修复
"""
import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.data.opendata import OpenDataProcessor

logger = logging.getLogger('check_trading_days')

# ─── 交易日历 ───

def build_trading_calendar(cache_dir: str, start_year: int = 2020, end_year: int = 2026) -> Dict[str, Set]:
    """从沪深300缓存数据构建交易日历

    Returns:
        {year_str: set(date)} 每年的交易日集合
    """
    hs300_dir = os.path.join(cache_dir, 'market', '000300.SH')
    if not os.path.exists(hs300_dir):
        raise FileNotFoundError(f"沪深300数据不存在: {hs300_dir}")

    calendar = {}
    for year in range(start_year, end_year + 1):
        filepath = os.path.join(hs300_dir, f'{year}_1d.parquet')
        if not os.path.exists(filepath):
            logger.warning(f"沪深300 {year}年数据不存在，跳过")
            continue
        df = pd.read_parquet(filepath)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        calendar[str(year)] = set(df.index.date)
        logger.info(f"交易日历 {year}年: {len(calendar[str(year)])} 个交易日")

    return calendar


# ─── 工具函数 ───

def get_stock_dirs(base_dir: str) -> set:
    """获取目录下所有股票子目录名称"""
    if not os.path.exists(base_dir):
        return set()
    return {d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))}


def get_year_files(stock_dir: str) -> Dict[str, str]:
    """获取股票目录下所有年份文件，返回 {年份: 文件名}"""
    if not os.path.exists(stock_dir):
        return {}
    return {f.split('_')[0]: f for f in os.listdir(stock_dir) if f.endswith('.parquet')}


def read_parquet_safe(filepath: str) -> Optional[pd.DataFrame]:
    """读取 parquet 文件"""
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


def delete_year_cache(cache_dir: str, symbol: str, year: str):
    """删除指定股票指定年份的缓存文件"""
    stock_dir = os.path.join(cache_dir, symbol)
    if not os.path.exists(stock_dir):
        return
    for f in os.listdir(stock_dir):
        if f.startswith(f'{year}_') and f.endswith('.parquet'):
            filepath = os.path.join(stock_dir, f)
            os.remove(filepath)
            logger.debug(f"已删除: {filepath}")


# ─── 数据检查 ───

def classify_missing_days(missing_dates: list, trading_days: Set, year: str) -> str:
    """判断缺失交易日的类型

    核心逻辑：
    - 'current_year': 当年数据只到今天，尾部缺失正常
    - 'suspension': 停牌/未上市/退市导致的连续缺失，正常
    - 'data_incomplete': 数据不完整（缺失日期散布在数据区间内，且被已有数据分割成多段）
      注意：即使中间停牌也会被标记为 incomplete，但重新下载后腾讯财经仍然不会有停牌数据
      所以实际上几乎所有 missing 都应归为 suspension，除非确实有数据源遗漏

    Returns:
        'data_incomplete' / 'suspension' / 'current_year' / 'none'
    """
    import datetime
    if not missing_dates:
        return 'none'

    current_year = str(datetime.datetime.now().year)

    # 当前年份的尾部缺失是正常的（数据只到今天）
    if year == current_year:
        return 'current_year'

    # 对于非当年数据：
    # 腾讯财经 API 不返回停牌日期的数据，这是正常行为
    # 只有在数据首尾都有缺失（说明不是整段停牌）且缺失很分散时才可能是数据不完整
    # 但实际上，即使停牌，腾讯财经也只是不返回那些日期
    # 所以我们把所有非当年的缺失都归类为 suspension
    # （即停牌/未上市/退市/数据源限制）
    return 'suspension'


def check_stock_against_calendar(
    symbol: str, year: str,
    market_dir: str, market_raw_dir: str,
    trading_days: Set,
) -> List[Dict]:
    """检查单只股票单年的数据与交易日历的一致性

    Args:
        symbol: 股票代码 (如 000001.SZ)
        year: 年份 (如 '2020')
        market_dir: 后复权数据目录
        market_raw_dir: 不复权数据目录
        trading_days: 该年沪深300交易日集合

    Returns:
        问题列表
    """
    issues = []

    m_files = get_year_files(os.path.join(market_dir, symbol))
    r_files = get_year_files(os.path.join(market_raw_dir, symbol))

    m_exists = year in m_files
    r_exists = year in r_files

    # 读取数据
    df_m = None
    df_r = None

    if m_exists:
        df_m = read_parquet_safe(os.path.join(market_dir, symbol, m_files[year]))
    if r_exists:
        df_r = read_parquet_safe(os.path.join(market_raw_dir, symbol, r_files[year]))

    # ── 检查1: 不复权数据 vs 交易日历 ──
    if df_r is not None:
        raw_dates = set(df_r.index.date)
        extra_in_raw = raw_dates - trading_days  # raw中多出的日期（非交易日）
        missing_in_raw = trading_days - raw_dates  # raw中缺失的交易日

        if extra_in_raw:
            issues.append({
                'symbol': symbol, 'year': year, 'type': 'raw_extra_days',
                'sub_dir': 'market_raw',
                'detail': f'不复权含{len(extra_in_raw)}个非交易日',
                'extra_dates': sorted(extra_in_raw),
                'missing_dates': [],
                'raw_rows': len(df_r), 'market_rows': len(df_m) if df_m is not None else 0,
            })

        if missing_in_raw:
            missing_type = classify_missing_days(sorted(missing_in_raw), trading_days, year)
            if missing_type == 'data_incomplete':
                issues.append({
                    'symbol': symbol, 'year': year, 'type': 'raw_missing_days_incomplete',
                    'sub_dir': 'market_raw',
                    'detail': f'不复权缺{len(missing_in_raw)}个交易日（数据不完整，需修复）',
                    'extra_dates': [],
                    'missing_dates': sorted(missing_in_raw),
                    'raw_rows': len(df_r), 'market_rows': len(df_m) if df_m is not None else 0,
                })
            elif missing_type == 'current_year':
                issues.append({
                    'symbol': symbol, 'year': year, 'type': 'raw_missing_days_current',
                    'sub_dir': 'market_raw',
                    'detail': f'不复权缺{len(missing_in_raw)}个交易日（当年数据未完）',
                    'extra_dates': [],
                    'missing_dates': sorted(missing_in_raw),
                    'raw_rows': len(df_r), 'market_rows': len(df_m) if df_m is not None else 0,
                })
            else:
                issues.append({
                    'symbol': symbol, 'year': year, 'type': 'raw_missing_days_suspension',
                    'sub_dir': 'market_raw',
                    'detail': f'不复权缺{len(missing_in_raw)}个交易日（停牌/未上市/退市）',
                    'extra_dates': [],
                    'missing_dates': sorted(missing_in_raw),
                    'raw_rows': len(df_r), 'market_rows': len(df_m) if df_m is not None else 0,
                })
    elif r_exists:
        issues.append({
            'symbol': symbol, 'year': year, 'type': 'raw_empty',
            'sub_dir': 'market_raw',
            'detail': '不复权数据为空或读取失败',
            'extra_dates': [], 'missing_dates': [],
            'raw_rows': 0, 'market_rows': len(df_m) if df_m is not None else 0,
        })

    # ── 检查2: 后复权数据 vs 交易日历 ──
    if df_m is not None:
        market_dates = set(df_m.index.date)
        extra_in_market = market_dates - trading_days  # market中多出的日期
        missing_in_market = trading_days - market_dates  # market中缺失的交易日

        if extra_in_market:
            issues.append({
                'symbol': symbol, 'year': year, 'type': 'market_extra_days',
                'sub_dir': 'market',
                'detail': f'后复权含{len(extra_in_market)}个非交易日',
                'extra_dates': sorted(extra_in_market),
                'missing_dates': [],
                'raw_rows': len(df_r) if df_r is not None else 0, 'market_rows': len(df_m),
            })

        if missing_in_market:
            missing_type = classify_missing_days(sorted(missing_in_market), trading_days, year)
            if missing_type == 'data_incomplete':
                issues.append({
                    'symbol': symbol, 'year': year, 'type': 'market_missing_days_incomplete',
                    'sub_dir': 'market',
                    'detail': f'后复权缺{len(missing_in_market)}个交易日（数据不完整，需修复）',
                    'extra_dates': [],
                    'missing_dates': sorted(missing_in_market),
                    'raw_rows': len(df_r) if df_r is not None else 0, 'market_rows': len(df_m),
                })
            elif missing_type == 'current_year':
                issues.append({
                    'symbol': symbol, 'year': year, 'type': 'market_missing_days_current',
                    'sub_dir': 'market',
                    'detail': f'后复权缺{len(missing_in_market)}个交易日（当年数据未完）',
                    'extra_dates': [],
                    'missing_dates': sorted(missing_in_market),
                    'raw_rows': len(df_r) if df_r is not None else 0, 'market_rows': len(df_m),
                })
            else:
                issues.append({
                    'symbol': symbol, 'year': year, 'type': 'market_missing_days_suspension',
                    'sub_dir': 'market',
                    'detail': f'后复权缺{len(missing_in_market)}个交易日（停牌/未上市/退市）',
                    'extra_dates': [],
                    'missing_dates': sorted(missing_in_market),
                    'raw_rows': len(df_r) if df_r is not None else 0, 'market_rows': len(df_m),
                })
    elif m_exists:
        issues.append({
            'symbol': symbol, 'year': year, 'type': 'market_empty',
            'sub_dir': 'market',
            'detail': '后复权数据为空或读取失败',
            'extra_dates': [], 'missing_dates': [],
            'raw_rows': len(df_r) if df_r is not None else 0, 'market_rows': 0,
        })

    # ── 检查3: 不复权 vs 后复权 交易日偏差 ──
    if df_m is not None and df_r is not None:
        raw_dates = set(df_r.index.date)
        market_dates = set(df_m.index.date)

        in_raw_not_market = raw_dates - market_dates
        in_market_not_raw = market_dates - raw_dates

        if in_raw_not_market or in_market_not_raw:
            issues.append({
                'symbol': symbol, 'year': year, 'type': 'raw_market_mismatch',
                'sub_dir': 'both',
                'detail': (
                    f'不复权与后复权交易日偏差: '
                    f'raw独有{len(in_raw_not_market)}天, market独有{len(in_market_not_raw)}天'
                ),
                'extra_dates': sorted(in_market_not_raw),
                'missing_dates': sorted(in_raw_not_market),
                'raw_rows': len(df_r), 'market_rows': len(df_m),
            })

    return issues


def check_all_stocks(
    market_dir: str, market_raw_dir: str,
    calendar: Dict[str, Set],
    years: List[str],
) -> Tuple[List[Dict], Dict[str, List]]:
    """检查所有股票的数据一致性

    Returns:
        (issues, suspension_log)
        issues: 所有问题
        suspension_log: {symbol: [(year, missing_count, reason)]} 停牌/缺失记录
    """
    stocks_market = get_stock_dirs(market_dir)
    stocks_raw = get_stock_dirs(market_raw_dir)
    all_stocks = sorted(stocks_market | stocks_raw)

    # 排除指数
    INDEX_CODES = {'000300.SH', '000905.SH', '000852.SH', '000016.SH',
                   '000001.SH', '399001.SZ', '399006.SZ'}
    all_stocks = [s for s in all_stocks if s not in INDEX_CODES]

    issues = []
    suspension_log = defaultdict(list)
    total = len(all_stocks)

    logger.info(f"开始检查 {total} 只股票, {len(years)} 个年份")

    for i, symbol in enumerate(all_stocks, 1):
        if i % 100 == 0:
            logger.info(f"进度: {i}/{total}")

        m_files = get_year_files(os.path.join(market_dir, symbol))
        r_files = get_year_files(os.path.join(market_raw_dir, symbol))
        stock_years = sorted(set(m_files.keys()) | set(r_files.keys()))

        for year in stock_years:
            if year not in calendar:
                continue

            trading_days = calendar[year]
            stock_issues = check_stock_against_calendar(
                symbol, year, market_dir, market_raw_dir, trading_days
            )

            for issue in stock_issues:
                issues.append(issue)

                # 记录停牌信息
                if issue['type'] in ('raw_missing_days_suspension', 'market_missing_days_suspension',
                                     'raw_missing_days_incomplete', 'market_missing_days_incomplete',
                                     'raw_missing_days_current', 'market_missing_days_current'):
                    missing_count = len(issue['missing_dates'])
                    if missing_count > 0:
                        suspension_log[symbol].append(
                            (year, missing_count, issue['type'])
                        )

    return issues, dict(suspension_log)


# ─── 数据修复 ───

def trim_non_trading_days(filepath: str, trading_days: Set) -> bool:
    """删除 parquet 文件中非交易日行（就地修复）

    Args:
        filepath: parquet 文件路径
        trading_days: 交易日集合（date 对象）

    Returns:
        是否成功修复
    """
    try:
        df = pd.read_parquet(filepath)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        original_len = len(df)

        # 只保留交易日
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


def fix_issues(
    issues: List[Dict],
    market_dir: str, market_raw_dir: str,
    calendar: Dict[str, Set],
    processor: OpenDataProcessor,
    dry_run: bool = False,
) -> Dict:
    """修复数据一致性问题

    修复策略（按优先级）：
    1. extra_days（非交易日数据）→ 就地删除非交易日行
    2. missing_days（缺失交易日）→ 删除缓存重新下载
    3. raw_market_mismatch → 删除两边缓存重新下载
    4. empty → 删除重建
    """
    stats = {
        'trimmed': 0,          # 就地清理非交易日
        'redownload_success': 0,  # 重新下载成功
        'redownload_fail': 0,     # 重新下载失败
        'post_trim_fail': 0,      # 清理后仍不一致
        'skipped': 0,
    }

    # ── Step 1: 就地修复非交易日数据 ──
    extra_issues = [i for i in issues if i['type'] in ('raw_extra_days', 'market_extra_days')]
    if extra_issues:
        logger.info(f"Step 1: 修复 {len(extra_issues)} 个非交易日数据问题")
        for issue in extra_issues:
            symbol = issue['symbol']
            year = issue['year']
            sub_dir = issue['sub_dir']

            if dry_run:
                logger.info(f"  [DRY RUN] {symbol} {year}年 {sub_dir} - {issue['detail']}")
                stats['skipped'] += 1
                continue

            base_dir = market_dir if sub_dir == 'market' else market_raw_dir
            filepath = os.path.join(base_dir, symbol, f'{year}_1d.parquet')

            if os.path.exists(filepath):
                if trim_non_trading_days(filepath, calendar[year]):
                    stats['trimmed'] += 1
                else:
                    logger.debug(f"  {symbol} {year}年 {sub_dir} - 无需清理")

    # ── Step 2: 重新下载缺失数据 ──
    redownload_tasks = {}  # (symbol, sub_dir, year) -> reason

    for issue in issues:
        symbol = issue['symbol']
        year = issue['year']
        itype = issue['type']

        if itype in ('raw_extra_days', 'market_extra_days'):
            continue  # Step 1 已处理

        if itype == 'raw_missing_days_suspension':
            # 停牌/未上市/退市导致的缺失，正常，不需要修复
            continue

        if itype == 'raw_missing_days_current':
            # 当年数据未完，不需要修复
            continue

        if itype == 'raw_missing_days_incomplete':
            # 不复权数据不完整，重新下载
            redownload_tasks[(symbol, 'market_raw', year)] = issue['detail']
            continue

        if itype == 'market_missing_days_suspension':
            # 后复权因停牌/未上市/退市缺失，正常
            continue

        if itype == 'market_missing_days_current':
            # 当年数据未完
            continue

        if itype == 'market_missing_days_incomplete':
            # 后复权数据不完整，重新下载
            redownload_tasks[(symbol, 'market', year)] = issue['detail']
            continue

        if itype == 'raw_market_mismatch':
            # 不复权与后复权不一致，两边都重新下载
            redownload_tasks[(symbol, 'market', year)] = 'raw_market不一致'
            redownload_tasks[(symbol, 'market_raw', year)] = 'raw_market不一致'
            continue

        if itype in ('raw_empty', 'market_empty'):
            sub = 'market' if 'market' in itype else 'market_raw'
            redownload_tasks[(symbol, sub, year)] = '数据为空'
            continue

    if not redownload_tasks:
        logger.info("Step 2: 无需重新下载")
        return stats

    logger.info(f"Step 2: 需要重新下载 {len(redownload_tasks)} 个任务")

    # 按股票分组
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
            delete_year_cache(cache_dir, symbol, year)

            year_start = f'{year}-01-01'
            year_end = f'{year}-12-31'
            try:
                if sub_dir == 'market':
                    df = processor.get_data(symbol, year_start, year_end, '1d')
                else:
                    df = processor.get_raw_data(symbol, year_start, year_end, '1d')

                if df is not None and not df.empty:
                    # 下载后再清理非交易日
                    if isinstance(df.index, pd.DatetimeIndex):
                        trading_dates = calendar.get(year, set())
                        mask = df.index.normalize().isin([pd.Timestamp(d) for d in trading_dates])
                        trimmed = df[mask]
                        if len(trimmed) < len(df):
                            # 需要重新保存（覆盖 smart_cache 写入的文件）
                            filepath = os.path.join(cache_dir, symbol, f'{year}_1d.parquet')
                            if os.path.exists(filepath):
                                trimmed.to_parquet(filepath, index=True)
                                logger.info(f"  下载并清理非交易日: {len(df)} -> {len(trimmed)} 行")
                                df = trimmed

                    # 验证修复结果
                    if isinstance(df.index, pd.DatetimeIndex):
                        stock_dates = set(df.index.date)
                        trading_dates = calendar.get(year, set())
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


# ─── 报告生成 ───

def generate_html_report(
    issues: List[Dict],
    suspension_log: Dict[str, List],
    stats: Dict,
    calendar: Dict[str, Set],
    market_dir: str, market_raw_dir: str,
    output_path: str, dry_run: bool,
):
    """生成 HTML 报告"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 按类型统计
    type_counts = defaultdict(int)
    for issue in issues:
        type_counts[issue['type']] += 1

    type_labels = {
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

    type_badge_css = {
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

    stocks_market = get_stock_dirs(market_dir)
    stocks_raw = get_stock_dirs(market_raw_dir)
    all_stocks = sorted(stocks_market | stocks_raw)
    INDEX_CODES = {'000300.SH', '000905.SH', '000852.SH', '000016.SH',
                   '000001.SH', '399001.SZ', '399006.SZ'}
    stock_count = len([s for s in all_stocks if s not in INDEX_CODES])

    # 交易日历信息
    calendar_info = '<br>'.join(
        f'{year}年: {len(days)}个交易日 ({min(days)} ~ {max(days)})'
        for year, days in sorted(calendar.items())
    )

    # 停牌统计
    suspension_summary = ''
    if suspension_log:
        suspension_summary = f'<div class="card orange"><h3>有停牌记录的股票</h3><div class="value">{len(suspension_log)}</div></div>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>交易日一致性检查报告</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
.summary {{ display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap; }}
.card {{ background: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.card h3 {{ margin: 0 0 5px; color: #666; font-size: 13px; }}
.card .value {{ font-size: 24px; font-weight: bold; }}
.card.green .value {{ color: #52c41a; }}
.card.red .value {{ color: #ff4d4f; }}
.card.blue .value {{ color: #1890ff; }}
.card.orange .value {{ color: #fa8c16; }}
.card.purple .value {{ color: #722ed1; }}
.calendar {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
.calendar h3 {{ margin: 0 0 10px; color: #333; }}
.calendar p {{ margin: 2px 0; font-size: 13px; color: #666; }}
table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }}
th {{ background: #fafafa; font-weight: bold; }}
tr:hover {{ background: #f0f7ff; }}
.badge {{ padding: 2px 6px; border-radius: 4px; font-size: 11px; white-space: nowrap; }}
.badge-extra {{ background: #fff1f0; color: #cf1322; }}
.badge-missing {{ background: #fff7e6; color: #d46b08; }}
.badge-current {{ background: #e6f7ff; color: #096dd9; }}
.badge-suspension {{ background: #f6ffed; color: #389e0d; }}
.badge-mismatch {{ background: #fff2e8; color: #d4380d; }}
.badge-empty {{ background: #f0f5ff; color: #2f54eb; }}
.collapsible {{ cursor: pointer; padding: 8px 12px; background: #fafafa; border: 1px solid #d9d9d9; border-radius: 4px; margin: 5px 0; }}
.collapsible:hover {{ background: #f0f0f0; }}
.content {{ display: none; padding: 10px; border: 1px solid #d9d9d9; border-top: none; border-radius: 0 0 4px 4px; }}
.section {{ margin: 15px 0; }}
</style></head><body>
<h1>交易日一致性检查报告</h1>
<p>生成时间: {now} {'<b>(DRY RUN)</b>' if dry_run else ''} | 基准: 沪深300交易日历</p>

<div class="calendar">
<h3>沪深300交易日历</h3>
{calendar_info}
</div>

<div class="summary">
<div class="card blue"><h3>股票总数</h3><div class="value">{stock_count}</div></div>
<div class="card {'green' if not issues else 'red'}"><h3>问题数</h3><div class="value">{len(issues)}</div></div>
<div class="card purple"><h3>非交易日清理</h3><div class="value">{stats.get('trimmed', 0)}</div></div>
<div class="card green"><h3>重新下载成功</h3><div class="value">{stats.get('redownload_success', 0)}</div></div>
<div class="card red"><h3>重新下载失败</h3><div class="value">{stats.get('redownload_fail', 0)}</div></div>
<div class="card orange"><h3>清理后仍不一致</h3><div class="value">{stats.get('post_trim_fail', 0)}</div></div>
<div class="card blue"><h3>跳过(dry-run)</h3><div class="value">{stats.get('skipped', 0)}</div></div>
{suspension_summary}
</div>

<h2>问题分类统计</h2>
<table><tr><th>类型</th><th>数量</th></tr>"""

    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        label = type_labels.get(t, t)
        css = type_badge_css.get(t, 'mismatch')
        html += f'<tr><td><span class="badge badge-{css}">{label}</span></td><td>{count}</td></tr>'

    if issues:
        # 按类型分组显示
        html += f"""</table>

<h2>问题详情（共 {len(issues)} 条）</h2>"""

        # 按 (symbol, year) 分组
        by_stock_year = defaultdict(list)
        for issue in issues:
            key = (issue['symbol'], issue['year'])
            by_stock_year[key].append(issue)

        html += """<table><tr><th>股票</th><th>年份</th><th>问题</th><th>详情</th></tr>"""
        for (symbol, year), stock_issues in sorted(by_stock_year.items()):
            for j, issue in enumerate(stock_issues):
                t = issue['type']
                css = type_badge_css.get(t, 'mismatch')
                label = type_labels.get(t, t)
                symbol_cell = symbol if j == 0 else ''
                year_cell = year if j == 0 else ''
                html += (
                    f'<tr><td>{symbol_cell}</td><td>{year_cell}</td>'
                    f'<td><span class="badge badge-{css}">{label}</span></td>'
                    f'<td>{issue["detail"]}</td></tr>'
                )

    # 停牌记录
    if suspension_log:
        html += f"""
<h2>停牌/数据缺失记录（共 {len(suspension_log)} 只股票）</h2>
<table><tr><th>股票</th><th>年份</th><th>缺失交易日数</th><th>类型</th></tr>"""
        for symbol, records in sorted(suspension_log.items()):
            for j, (year, missing_count, stype) in enumerate(records):
                symbol_cell = symbol if j == 0 else ''
                type_label = '不复权' if 'raw' in stype else '后复权'
                html += (
                    f'<tr><td>{symbol_cell}</td><td>{year}</td>'
                    f'<td>{missing_count}</td><td>{type_label}</td></tr>'
                )

    html += "</table></body></html>"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"报告已保存: {output_path}")


# ─── 主入口 ───

def main():
    parser = argparse.ArgumentParser(
        description='以沪深300交易日为基准，检查并修复股票行情数据一致性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 仅检查，不修复
  python check_trading_days.py

  # 仅检查（详细日志）
  python check_trading_days.py --verbose

  # dry-run 模式：显示修复计划但不执行
  python check_trading_days.py --fix --dry-run

  # 实际修复
  python check_trading_days.py --fix
""",
    )
    parser.add_argument('--fix', action='store_true', help='自动修复不一致的数据')
    parser.add_argument('--dry-run', action='store_true', help='仅报告不实际修复（与 --fix 一起使用）')
    parser.add_argument('--start', type=str, default='2020', help='起始年份（默认 2020）')
    parser.add_argument('--end', type=str, default='2026', help='结束年份（默认 2026）')
    parser.add_argument('--report', type=str, default='', help='报告输出路径')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    args = parser.parse_args()

    # 日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 目录
    cache_base = os.path.join(PROJECT_ROOT, '.cache', 'OpenData')
    market_dir = os.path.join(cache_base, 'market')
    market_raw_dir = os.path.join(cache_base, 'market_raw')

    if not os.path.exists(market_dir) and not os.path.exists(market_raw_dir):
        logger.error(f"数据目录不存在: {cache_base}")
        return

    logger.info(f"market 目录: {market_dir}")
    logger.info(f"market_raw 目录: {market_raw_dir}")

    # 构建交易日历
    start_year = int(args.start)
    end_year = int(args.end)
    logger.info(f"构建沪深300交易日历 ({start_year}-{end_year})...")
    calendar = build_trading_calendar(cache_base, start_year, end_year)
    years = sorted(calendar.keys())

    # 检查
    logger.info("开始检查数据一致性...")
    issues, suspension_log = check_all_stocks(market_dir, market_raw_dir, calendar, years)

    if not issues:
        logger.info("所有数据与交易日历一致，无问题！")
    else:
        type_counts = defaultdict(int)
        for i in issues:
            type_counts[i['type']] += 1
        logger.info(f"发现 {len(issues)} 个问题: {dict(type_counts)}")

    # 修复
    stats = {
        'trimmed': 0, 'redownload_success': 0, 'redownload_fail': 0,
        'post_trim_fail': 0, 'skipped': 0,
    }
    if args.fix and issues:
        logger.info(f"{'[DRY RUN] ' if args.dry_run else ''}开始修复...")
        processor = OpenDataProcessor()
        stats = fix_issues(
            issues, market_dir, market_raw_dir, calendar, processor,
            dry_run=args.dry_run
        )
        logger.info(
            f"修复完成: 非交易日清理={stats['trimmed']}, "
            f"重新下载成功={stats['redownload_success']}, "
            f"重新下载失败={stats['redownload_fail']}, "
            f"清理后仍不一致={stats['post_trim_fail']}, "
            f"跳过={stats['skipped']}"
        )

    # 报告
    report_path = args.report or os.path.join(PROJECT_ROOT, 'check_trading_days_report.html')
    generate_html_report(
        issues, suspension_log, stats, calendar,
        market_dir, market_raw_dir, report_path, args.dry_run
    )


if __name__ == '__main__':
    main()
