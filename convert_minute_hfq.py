"""
分钟级后复权数据转换模块

核心原理：
  QMT后复权价格与不复权价格是线性关系：hfq = bfq × a + b
  - a 在非除权区间恒定，送股/转增时跳变
  - b 在除权除息日跳变，非除权日不变
  - volume/amount 等非价格字段无需调整

数据合并策略：
  QMT分钟数据（直接获取）优先，聚宽CSV转换数据补足历史缺失部分。
  - QMT有数据的区间：直接使用QMT的后复权和不复权数据
  - QMT无数据的区间：用聚宽不复权数据 + 线性变换生成后复权数据

用法（CLI）：
  # 仅使用QMT数据转换
  python convert_minute_hfq.py --stocks 000001.SZ

  # 使用聚宽CSV补足历史（自动查找CSV）
  python convert_minute_hfq.py --stocks 000001.SZ --jq

  # 指定聚宽CSV路径
  python convert_minute_hfq.py --stocks 000001.SZ --jq --jq-dir /path/to/csv

  # 强制覆盖已有缓存
  python convert_minute_hfq.py --stocks 000001.SZ --jq --force

  # 查看缓存信息
  python convert_minute_hfq.py --stocks 000001.SZ --info

用法（Python API）：
  from convert_minute_hfq import convert_minute_data
  convert_minute_data('000001.SZ', jq=True)
"""

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from xtquant import xtdata
except ImportError:
    xtdata = None

from core.cache import cache_manager


logger = logging.getLogger('convert_minute_hfq')

JQ_DEFAULT_DIR = os.path.join(os.getcwd(), '.cache', 'JQData', 'market_raw')


def _parse_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    try:
        df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S')
    except (ValueError, TypeError):
        try:
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
        except (ValueError, TypeError):
            pass
    return df


def _symbol_to_jq_code(symbol: str) -> str:
    return symbol.replace('.', '_')


def _find_jq_csv(symbol: str, jq_dir: str) -> Optional[str]:
    jq_code = _symbol_to_jq_code(symbol)
    jq_path = Path(jq_dir)
    if not jq_path.exists():
        return None
    for f in jq_path.iterdir():
        if f.is_file() and f.name.startswith(jq_code) and f.name.endswith('.csv'):
            return str(f)
    return None


def fit_daily_linear_params(daily_hfq: pd.DataFrame, daily_bfq: pd.DataFrame) -> pd.DataFrame:
    params = []
    for dt in daily_bfq.index:
        bfq_prices = []
        hfq_prices = []
        for col in ['open', 'high', 'low', 'close']:
            if col in daily_bfq.columns and col in daily_hfq.columns:
                bfq_prices.append(daily_bfq.loc[dt, col])
                hfq_prices.append(daily_hfq.loc[dt, col])

        x = np.array(bfq_prices)
        y = np.array(hfq_prices)
        n = len(x)
        if n < 2:
            a = y[0] / x[0] if x[0] != 0 else 0
            b = 0.0
        else:
            sum_x, sum_y = np.sum(x), np.sum(y)
            sum_xy, sum_x2 = np.sum(x * y), np.sum(x ** 2)
            denom = n * sum_x2 - sum_x ** 2
            if abs(denom) < 1e-10:
                a = y[0] / x[0] if x[0] != 0 else 0
                b = 0.0
            else:
                a = (n * sum_xy - sum_x * sum_y) / denom
                b = (sum_y - a * sum_x) / n

        params.append({'date': dt, 'a': a, 'b': b})

    params_df = pd.DataFrame(params).set_index('date')
    return params_df


def fetch_qmt_daily(symbol: str, start_date: str, end_date: str):
    if xtdata is None:
        raise RuntimeError("xtquant 未安装")

    xtdata.enable_hello = False
    start_time = start_date.replace('-', '')
    end_time = end_date.replace('-', '')

    xtdata.download_history_data(stock_code=symbol, period='1d', start_time='19900101', end_time='', incrementally=False)

    daily_hfq = xtdata.get_market_data_ex([], [symbol], period='1d', start_time=start_time, end_time=end_time, count=-1, dividend_type='back')[symbol]
    daily_bfq = xtdata.get_market_data_ex([], [symbol], period='1d', start_time=start_time, end_time=end_time, count=-1, dividend_type='none')[symbol]

    daily_hfq = _parse_index(daily_hfq)
    daily_bfq = _parse_index(daily_bfq)

    return daily_hfq, daily_bfq


def fetch_qmt_minute(symbol: str, start_date: str, end_date: str):
    if xtdata is None:
        raise RuntimeError("xtquant 未安装")

    xtdata.enable_hello = False
    start_time = start_date.replace('-', '')
    end_time = end_date.replace('-', '')

    xtdata.download_history_data(stock_code=symbol, period='1m', start_time=start_time, end_time='', incrementally=False)

    min_bfq = xtdata.get_market_data_ex([], [symbol], period='1m', start_time=start_time, end_time=end_time, count=-1, dividend_type='none')[symbol]
    min_hfq = xtdata.get_market_data_ex([], [symbol], period='1m', start_time=start_time, end_time=end_time, count=-1, dividend_type='back')[symbol]

    min_bfq = _parse_index(min_bfq)
    min_hfq = _parse_index(min_hfq)

    return min_bfq, min_hfq


def read_jq_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    if 'code' in df.columns:
        df = df.drop(columns=['code'])
    if 'money' in df.columns:
        df = df.rename(columns={'money': 'amount'})
    if 'volume' in df.columns:
        df['volume'] = (df['volume'] / 100).astype(np.int64)
    return df


def apply_linear_transform(min_bfq: pd.DataFrame, params_df: pd.DataFrame) -> pd.DataFrame:
    min_bfq = min_bfq.copy()
    min_bfq['trade_date'] = min_bfq.index.normalize()

    params_indexed = params_df.copy()
    params_indexed.index = params_indexed.index.normalize()

    a_map = params_indexed['a'].to_dict()
    b_map = params_indexed['b'].to_dict()

    min_bfq['a'] = min_bfq['trade_date'].map(a_map)
    min_bfq['b'] = min_bfq['trade_date'].map(b_map)

    missing = min_bfq['a'].isna()
    if missing.any():
        logger.warning(f"  {missing.sum()} 根分钟K线缺少日线参数（可能是非交易日），已跳过")
        min_bfq = min_bfq[~missing].copy()

    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in min_bfq.columns:
            min_bfq[col] = min_bfq[col] * min_bfq['a'] + min_bfq['b']

    if 'preClose' in min_bfq.columns:
        min_bfq['preClose'] = min_bfq['preClose'] * min_bfq['a'] + min_bfq['b']

    min_bfq = min_bfq.drop(columns=['trade_date', 'a', 'b'], errors='ignore')
    return min_bfq


def merge_minute_data(qmt_bfq: pd.DataFrame, qmt_hfq: pd.DataFrame,
                      jq_bfq: Optional[pd.DataFrame], jq_hfq: Optional[pd.DataFrame]
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """合并QMT和聚宽分钟数据

    QMT优先，聚宽补足QMT缺失的历史部分
    """
    merged_bfq_parts = [qmt_bfq]
    merged_hfq_parts = [qmt_hfq]

    if jq_bfq is not None and not jq_bfq.empty:
        jq_only_bfq = jq_bfq[~jq_bfq.index.isin(qmt_bfq.index)]
        if not jq_only_bfq.empty:
            merged_bfq_parts.insert(0, jq_only_bfq)
            logger.info(f"  聚宽补足不复权: {len(jq_only_bfq)} 条 "
                        f"({jq_only_bfq.index[0].date()} ~ {jq_only_bfq.index[-1].date()})")

    if jq_hfq is not None and not jq_hfq.empty:
        jq_only_hfq = jq_hfq[~jq_hfq.index.isin(qmt_hfq.index)]
        if not jq_only_hfq.empty:
            merged_hfq_parts.insert(0, jq_only_hfq)
            logger.info(f"  聚宽补足后复权: {len(jq_only_hfq)} 条 "
                        f"({jq_only_hfq.index[0].date()} ~ {jq_only_hfq.index[-1].date()})")

    merged_bfq = pd.concat(merged_bfq_parts).sort_index()
    merged_bfq = merged_bfq[~merged_bfq.index.duplicated(keep='last')]

    merged_hfq = pd.concat(merged_hfq_parts).sort_index()
    merged_hfq = merged_hfq[~merged_hfq.index.duplicated(keep='last')]

    return merged_bfq, merged_hfq


def save_to_cache(symbol: str, df: pd.DataFrame, namespace: str, period: str,
                  force: bool = False) -> List[int]:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return []

    written = cache_manager.disk_cache.put_yearly_from_df(
        namespace=namespace,
        symbol=symbol,
        suffix=period,
        df=df,
        format_type='parquet',
        skip_existing=not force,
    )
    return written


def show_cache_info(symbols: List[str]):
    cache_dir = cache_manager.disk_cache.cache_dir / 'QMTData'
    logger.info(f"\n{'='*70}")
    logger.info(f"分钟级缓存信息 (目录: {cache_dir})")
    logger.info(f"{'='*70}")

    for symbol in symbols:
        for label, namespace in [('后复权', 'QMTDataProcessor'), ('不复权', 'QMTDataProcessor_Raw')]:
            years = cache_manager.disk_cache.list_yearly_files(namespace, symbol, '1m')
            if years:
                year_dir = cache_manager.disk_cache._get_yearly_dir(namespace, symbol)
                total_size = sum(f.stat().st_size for f in year_dir.glob('*.parquet')) if year_dir.exists() else 0
                size_str = _format_size(total_size)
                logger.info(f"  {symbol} 分钟{label}: {len(years)} 个年份 "
                            f"({min(years)}~{max(years)}), 总大小={size_str}")
            else:
                logger.info(f"  {symbol} 分钟{label}: 无缓存")

    logger.info(f"{'='*70}")


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1073741824:
        return f"{size_bytes / 1073741824:.1f} GB"
    if size_bytes >= 1048576:
        return f"{size_bytes / 1048576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def validate_merged_data(symbol: str,
                         qmt_min_bfq: pd.DataFrame, qmt_min_hfq: pd.DataFrame,
                         jq_min_bfq: Optional[pd.DataFrame], jq_min_hfq: Optional[pd.DataFrame],
                         merged_bfq: pd.DataFrame, merged_hfq: pd.DataFrame,
                         strict: bool = False) -> Dict:
    """验证合并后的数据质量

    检验项：
    1. 重叠区间：QMT直接后复权 vs 聚宽线性变换后复权 的价格偏差
    2. 边界连续性：QMT/聚宽数据接缝处价格跳变检查
    3. 价格合理性：后复权 >= 不复权 & 无负价格
    4. 数据完整性：合并后无重复索引、无异常空值

    Args:
        symbol: 股票代码
        qmt_min_bfq: QMT不复权分钟数据
        qmt_min_hfq: QMT后复权分钟数据
        jq_min_bfq: 聚宽不复权分钟数据
        jq_min_hfq: 聚宽转换后后复权数据
        merged_bfq: 合并后不复权数据
        merged_hfq: 合并后后复权数据
        strict: 严格模式，检验失败时抛出异常

    Returns:
        验证结果字典，包含 passed, warnings, details
    """
    validation = {'passed': True, 'warnings': [], 'details': {}}

    known_qmt_bug_stocks = {
        '600076.SH': 'QMT后复权算法在2024年存在严重bug，315个close为负，建议排除此股票',
        '600734.SH': 'QMT后复权算法在2020-2021年存在严重bug，23416个close为负，建议排除此股票',
        '600052.SH': 'QMT后复权在2021-02-01~02-18期间存在轻微异常(ratio~0.63)，影响较小',
        '600744.SH': 'QMT后复权在2020-02-04当天存在异常(1个open为负)，影响极小',
    }

    def _warn(msg: str):
        logger.warning(f"  [验证] {msg}")
        validation['warnings'].append(msg)

    def _fail(msg: str):
        validation['passed'] = False
        _warn(msg)

    logger.info(f"数据验证 {symbol}...")

    # ---- 1. 重叠区间对比 ----
    if (jq_min_hfq is not None and not jq_min_hfq.empty
            and not qmt_min_hfq.empty):
        overlap_idx = qmt_min_hfq.index[qmt_min_hfq.index.isin(jq_min_hfq.index)]
        if len(overlap_idx) > 0:
            qmt_overlap = qmt_min_hfq.loc[overlap_idx]
            jq_overlap = jq_min_hfq.loc[overlap_idx]

            overlap_stats = {}
            for col in ['open', 'high', 'low', 'close']:
                if col not in qmt_overlap.columns or col not in jq_overlap.columns:
                    continue

                qmt_vals = qmt_overlap[col].values.astype(float)
                jq_vals = jq_overlap[col].values.astype(float)

                nonzero = (qmt_vals != 0) & (jq_vals != 0)
                if nonzero.sum() == 0:
                    continue

                diff_pct = np.abs(qmt_vals[nonzero] - jq_vals[nonzero]) / qmt_vals[nonzero] * 100
                exact_match = np.sum(diff_pct < 0.001)
                close_match_1pct = np.sum(diff_pct < 1.0)
                large_diff_count = int(np.sum(diff_pct >= 1.0))
                max_diff = float(diff_pct.max())
                mean_diff = float(diff_pct.mean())

                overlap_stats[col] = {
                    'total': int(nonzero.sum()),
                    'exact_match': int(exact_match),
                    'exact_match_pct': round(exact_match / nonzero.sum() * 100, 2),
                    'close_match_1pct': int(close_match_1pct),
                    'close_match_1pct_pct': round(close_match_1pct / nonzero.sum() * 100, 2),
                    'large_diff_count': large_diff_count,
                    'max_diff_pct': round(max_diff, 4),
                    'mean_diff_pct': round(mean_diff, 6),
                }

                logger.info(f"  重叠验证 {col}: 精确匹配={exact_match}/{nonzero.sum()}"
                            f"({exact_match / nonzero.sum() * 100:.2f}%), "
                            f"偏差<1%={close_match_1pct}/{nonzero.sum()}, "
                            f"最大偏差={max_diff:.4f}%, 平均偏差={mean_diff:.6f}%")

                if col == 'close' and max_diff > 1.0:
                    _fail(f"close 最大偏差 {max_diff:.4f}% > 1%, 线性变换可能存在问题")
                elif col == 'close' and mean_diff > 0.01:
                    _warn(f"close 平均偏差 {mean_diff:.6f}% > 0.01%")

                if large_diff_count > 0 and col == 'open':
                    _warn(f"open 有 {large_diff_count} 个bar偏差>=1% (除权日开盘价跳变, 属正常现象)")

            validation['details']['overlap'] = overlap_stats
        else:
            logger.info(f"  无重叠区间，跳过重叠验证")
    else:
        logger.info(f"  缺少聚宽或QMT后复权数据，跳过重叠验证")

    # ---- 2. 边界连续性检查 ----
    if (jq_min_hfq is not None and not jq_min_hfq.empty
            and not qmt_min_hfq.empty):
        qmt_start = qmt_min_hfq.index[0]
        jq_before_qmt = jq_min_hfq[jq_min_hfq.index < qmt_start]

        if not jq_before_qmt.empty and not qmt_min_hfq.empty:
            jq_last_bar = jq_before_qmt.iloc[-1]
            qmt_first_bar = qmt_min_hfq.iloc[0]

            boundary_stats = {}
            for col in ['open', 'high', 'low', 'close']:
                if col not in jq_last_bar.index or col not in qmt_first_bar.index:
                    continue
                jq_val = float(jq_last_bar[col])
                qmt_val = float(qmt_first_bar[col])
                if jq_val != 0:
                    jump_pct = abs(qmt_val - jq_val) / jq_val * 100
                    boundary_stats[col] = {
                        'jq_last': round(jq_val, 4),
                        'qmt_first': round(qmt_val, 4),
                        'jump_pct': round(jump_pct, 4),
                    }
                    logger.info(f"  边界 {col}: 聚宽最后={jq_val:.4f}, QMT首个={qmt_val:.4f}, "
                                f"跳变={jump_pct:.4f}%")

                    if col == 'close' and jump_pct > 5.0:
                        _fail(f"边界close跳变 {jump_pct:.4f}% > 5%, 数据接缝可能有问题")
                    elif col == 'close' and jump_pct > 1.0:
                        _warn(f"边界close跳变 {jump_pct:.4f}% > 1%, 请关注")

            validation['details']['boundary'] = boundary_stats

            time_gap = qmt_start - jq_before_qmt.index[-1]
            logger.info(f"  边界时间间隔: {time_gap}")

    # ---- 3. 价格合理性：后复权 >= 不复权 & 无负价格 ----
    if not merged_bfq.empty and not merged_hfq.empty:
        common_idx = merged_bfq.index[merged_bfq.index.isin(merged_hfq.index)]
        if len(common_idx) > 0:
            bfq_common = merged_bfq.loc[common_idx]
            hfq_common = merged_hfq.loc[common_idx]

            negative_count = 0
            for col in ['close']:
                if col not in bfq_common.columns or col not in hfq_common.columns:
                    continue
                bfq_vals = bfq_common[col].values.astype(float)
                hfq_vals = hfq_common[col].values.astype(float)
                valid = (bfq_vals > 0) & (hfq_vals > 0)
                if valid.sum() == 0:
                    continue
                ratio = hfq_vals[valid] / bfq_vals[valid]
                negative_count = int(np.sum(ratio < 0.99))

            if negative_count > 0:
                if symbol in known_qmt_bug_stocks:
                    _warn(f"后复权 < 不复权 的bar数: {negative_count} "
                          f"[已知QMT数据问题] {known_qmt_bug_stocks[symbol]}")
                else:
                    _warn(f"后复权 < 不复权 的bar数: {negative_count} "
                          f"(QMT日线后复权数据本身可能存在异常, 非转换问题)")
            else:
                logger.info(f"  价格合理性: 通过 (后复权 >= 不复权)")

            validation['details']['price_sanity'] = {'negative_ratio_count': negative_count}

    # ---- 3b. 负价格检测 ----
    if not merged_hfq.empty:
        neg_price_info = {}
        for col in ['open', 'high', 'low', 'close']:
            if col not in merged_hfq.columns:
                continue
            neg_count = int((merged_hfq[col].astype(float) < 0).sum())
            if neg_count > 0:
                neg_price_info[col] = neg_count
        if neg_price_info:
            if symbol in known_qmt_bug_stocks:
                _warn(f"后复权存在负价格: {neg_price_info} "
                      f"[已知QMT数据问题] {known_qmt_bug_stocks[symbol]}")
            else:
                _fail(f"后复权存在负价格: {neg_price_info} (非已知QMT数据问题，需排查)")
        validation['details']['negative_prices'] = neg_price_info

    # ---- 4. 数据完整性 ----
    integrity = {}
    if not merged_hfq.empty:
        dup_count = int(merged_hfq.index.duplicated().sum())
        integrity['duplicate_index'] = dup_count
        if dup_count > 0:
            _fail(f"合并后存在 {dup_count} 个重复索引")

        nan_cols = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in merged_hfq.columns:
                nan_count = int(merged_hfq[col].isna().sum())
                if nan_count > 0:
                    nan_cols[col] = nan_count
        if nan_cols:
            _warn(f"后复权数据存在空值: {nan_cols}")
        integrity['nan_in_hfq'] = nan_cols

    if not merged_bfq.empty:
        dup_count_bfq = int(merged_bfq.index.duplicated().sum())
        integrity['duplicate_index_bfq'] = dup_count_bfq
        if dup_count_bfq > 0:
            _fail(f"合并不复权存在 {dup_count_bfq} 个重复索引")

    logger.info(f"  数据完整性: 索引重复={integrity.get('duplicate_index', 'N/A')}, "
                f"空值={integrity.get('nan_in_hfq', {})}")

    validation['details']['integrity'] = integrity

    # ---- 总结 ----
    if validation['passed']:
        logger.info(f"  验证通过 ✓")
    else:
        logger.warning(f"  验证未通过 ✗ ({len(validation['warnings'])} 个问题)")
        if strict:
            raise ValueError(f"{symbol} 数据验证失败: {'; '.join(validation['warnings'])}")

    return validation


def convert_minute_data(symbol: str, start_date: str = '19900101',
                        end_date: str = '20991231', force: bool = False,
                        jq: bool = False, jq_dir: str = JQ_DEFAULT_DIR) -> Dict:
    """转换单只股票的分钟级数据

    合并策略：
    1. 从QMT获取分钟不复权+后复权数据（QMT优先）
    2. 如果启用--jq，读取聚宽CSV不复权数据，用线性变换生成后复权
    3. 合并：QMT区间用QMT数据，QMT缺失区间用聚宽转换数据补足
    4. 保存到缓存
    """
    result = {'symbol': symbol, 'status': 'ok'}

    logger.info(f"获取 {symbol} QMT日线数据...")
    try:
        daily_hfq, daily_bfq = fetch_qmt_daily(symbol, start_date, end_date)
    except Exception as e:
        logger.error(f"  获取QMT日线数据失败: {e}")
        result['status'] = 'fetch_error'
        return result

    if daily_hfq.empty or daily_bfq.empty:
        logger.warning(f"  {symbol} QMT日线数据为空")
        result['status'] = 'no_daily_data'
        return result

    logger.info(f"  QMT日线: {len(daily_hfq)} 条")

    logger.info(f"拟合线性变换参数...")
    params_df = fit_daily_linear_params(daily_hfq, daily_bfq)
    logger.info(f"  参数拟合: {len(params_df)} 天")

    logger.info(f"获取 {symbol} QMT分钟数据...")
    try:
        qmt_min_bfq, qmt_min_hfq = fetch_qmt_minute(symbol, start_date, end_date)
    except Exception as e:
        logger.error(f"  获取QMT分钟数据失败: {e}")
        result['status'] = 'fetch_minute_error'
        return result

    if qmt_min_bfq.empty:
        logger.warning(f"  {symbol} QMT分钟数据为空")
        qmt_min_bfq = pd.DataFrame()
        qmt_min_hfq = pd.DataFrame()
    else:
        logger.info(f"  QMT分钟: {len(qmt_min_bfq)} 条 "
                    f"({qmt_min_bfq.index[0].date()} ~ {qmt_min_bfq.index[-1].date()})")

    jq_min_bfq = None
    jq_min_hfq = None

    if jq:
        csv_path = _find_jq_csv(symbol, jq_dir)
        if csv_path:
            logger.info(f"  读取聚宽CSV: {csv_path}")
            try:
                jq_min_bfq = read_jq_csv(csv_path)
                logger.info(f"  聚宽不复权: {len(jq_min_bfq)} 条 "
                            f"({jq_min_bfq.index[0].date()} ~ {jq_min_bfq.index[-1].date()})")

                logger.info(f"  转换聚宽后复权数据...")
                jq_min_hfq = apply_linear_transform(jq_min_bfq, params_df)
                logger.info(f"  聚宽后复权: {len(jq_min_hfq)} 条")
            except Exception as e:
                logger.warning(f"  读取聚宽CSV失败: {e}")
                jq_min_bfq = None
                jq_min_hfq = None
        else:
            logger.info(f"  未找到聚宽CSV (目录: {jq_dir})")

    logger.info(f"合并数据 (QMT优先，聚宽补足)...")
    merged_bfq, merged_hfq = merge_minute_data(qmt_min_bfq, qmt_min_hfq, jq_min_bfq, jq_min_hfq)
    logger.info(f"  合并不复权: {len(merged_bfq)} 条")
    logger.info(f"  合并后复权: {len(merged_hfq)} 条")

    if not merged_bfq.empty:
        logger.info(f"  数据范围: {merged_bfq.index[0].date()} ~ {merged_bfq.index[-1].date()}")

    validation = validate_merged_data(
        symbol, qmt_min_bfq, qmt_min_hfq, jq_min_bfq, jq_min_hfq,
        merged_bfq, merged_hfq, strict=False,
    )
    result['validation'] = validation

    logger.info(f"保存不复权数据到缓存...")
    raw_years = save_to_cache(symbol, merged_bfq, 'QMTDataProcessor_Raw', '1m', force=force)
    logger.info(f"  不复权写入年份: {raw_years}")

    logger.info(f"保存后复权数据到缓存...")
    hfq_years = save_to_cache(symbol, merged_hfq, 'QMTDataProcessor', '1m', force=force)
    logger.info(f"  后复权写入年份: {hfq_years}")

    qmt_count = len(qmt_min_bfq)
    jq_count = len(merged_bfq) - qmt_count

    result['daily_count'] = len(daily_hfq)
    result['minute_count'] = len(merged_bfq)
    result['qmt_count'] = qmt_count
    result['jq_count'] = jq_count
    result['raw_years'] = raw_years
    result['hfq_years'] = hfq_years

    return result


def main():
    parser = argparse.ArgumentParser(
        description='分钟级后复权数据转换工具（支持QMT+聚宽合并）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            '使用示例:',
            '  # 仅使用QMT数据',
            '  python convert_minute_hfq.py --stocks 000001.SZ',
            '',
            '  # 使用聚宽CSV补足历史',
            '  python convert_minute_hfq.py --stocks 000001.SZ --jq',
            '',
            '  # 指定聚宽CSV目录',
            '  python convert_minute_hfq.py --stocks 000001.SZ --jq --jq-dir /path/to/csv',
            '',
            '  # 强制覆盖已有缓存',
            '  python convert_minute_hfq.py --stocks 000001.SZ --jq --force',
            '',
            '  # 查看缓存信息',
            '  python convert_minute_hfq.py --stocks 000001.SZ --info',
        ]),
    )
    parser.add_argument('--stocks', type=str, required=True,
                        help='股票代码，逗号分隔，如 000001.SZ,600519.SH')
    parser.add_argument('--start', type=str, default='19900101',
                        help='起始日期 (默认: 19900101)')
    parser.add_argument('--end', type=str, default='20991231',
                        help='结束日期 (默认: 20991231)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='强制覆盖已有缓存')
    parser.add_argument('--jq', action='store_true', default=False,
                        help='启用聚宽CSV补足历史数据')
    parser.add_argument('--jq-dir', type=str, default=JQ_DEFAULT_DIR,
                        help=f'聚宽CSV目录 (默认: {JQ_DEFAULT_DIR})')
    parser.add_argument('--info', action='store_true', default=False,
                        help='仅查看缓存信息，不转换')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='详细日志')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    if xtdata is None:
        logger.error("xtquant 未安装，无法获取数据。请先安装 MiniQMT 客户端。")
        sys.exit(1)

    stock_list = []
    for s in args.stocks.split(','):
        s = s.strip()
        if '.' not in s:
            s = f"{s}.SH" if s.startswith(('6', '9')) else f"{s}.SZ"
        stock_list.append(s)

    if args.info:
        show_cache_info(stock_list)
        return

    start_date = args.start if args.start else '19900101'
    end_date = args.end if args.end else '20991231'

    logger.info(f"开始转换: {len(stock_list)} 只股票, 日期范围={start_date}~{end_date}")
    if args.jq:
        logger.info(f"聚宽补足: 启用 (目录: {args.jq_dir})")
    if args.force:
        logger.info("强制模式: 将覆盖已有缓存")

    stats = {'ok': 0, 'error': 0}
    start_ts = time.time()

    for i, symbol in enumerate(stock_list, 1):
        logger.info(f"\n[{i}/{len(stock_list)}] {symbol}")
        result = convert_minute_data(
            symbol, start_date, end_date,
            force=args.force, jq=args.jq, jq_dir=args.jq_dir,
        )

        if result['status'] == 'ok':
            stats['ok'] += 1
            qmt_c = result.get('qmt_count', 0)
            jq_c = result.get('jq_count', 0)
            logger.info(f"  完成: {result.get('minute_count', 0)} 条K线 "
                        f"(QMT={qmt_c}, 聚宽补足={jq_c}), "
                        f"不复权年份={result.get('raw_years', [])}, "
                        f"后复权年份={result.get('hfq_years', [])}")
        else:
            stats['error'] += 1
            logger.warning(f"  跳过: {result['status']}")

    elapsed = time.time() - start_ts
    logger.info(f"\n{'='*60}")
    logger.info(f"转换完成: 成功={stats['ok']}, 失败={stats['error']}, 耗时={elapsed:.1f}秒")
    logger.info(f"{'='*60}")

    show_cache_info(stock_list)


if __name__ == '__main__':
    main()
