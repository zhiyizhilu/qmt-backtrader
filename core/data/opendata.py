import pandas as pd
import numpy as np
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any

from core.cache import smart_cache, cache_manager
from core.data.base import DataProcessor, _merged_dict_to_parquet, _parquet_to_merged_dict


def _convert_symbol_to_qmt(symbol: str) -> str:
    """将各种格式的股票代码转换为QMT格式 (如 000001.SZ)"""
    if '.' in symbol and not symbol.startswith(('sz.', 'sh.', 'SZ.', 'SH.')):
        return symbol
    if symbol.lower().startswith(('sz.', 'sh.')):
        parts = symbol.split('.')
        return f"{parts[1]}.{parts[0].upper()}"
    code = symbol.replace('.', '')
    return f"{code}.SH" if code.startswith(('6', '9')) else f"{code}.SZ"


def _convert_symbol_to_opendata(symbol: str) -> str:
    """将QMT格式股票代码转换为OpenData行情格式 (纯数字)"""
    return symbol.split('.')[0]


def _convert_symbol_to_opendata_financial(symbol: str) -> str:
    """将QMT格式股票代码转换为OpenData财务数据格式 (如 SZ000001)"""
    parts = symbol.split('.')
    if len(parts) == 2:
        return f"{parts[1].upper()}{parts[0]}"
    return f"SH{symbol}" if symbol.startswith(('6', '9')) else f"SZ{symbol}"


class OpenDataProcessor(DataProcessor):
    """OpenData数据处理器

    基于腾讯财经数据源，支持获取A股历史行情数据，数据范围远超QMT的一年限制。
    股票代码统一使用QMT格式 (000001.SZ)，内部自动转换。
    """

    INDEX_CODE_MAP = {
        '000300.SH': '000300',
        '000905.SH': '000905',
        '000852.SH': '000852',
        '000016.SH': '000016',
    }

    # 中证指数代码 -> 新浪指数代码（深证代码体系）
    _SINA_INDEX_CODE_MAP = {
        '000300': '399300',  # 沪深300
        '000905': '399905',  # 中证500
        '000852': '399852',  # 中证1000
        '000016': '000016',  # 上证50
    }

    PERIOD_MAP = {
        '1d': 'daily', 'day': 'daily', 'daily': 'daily',
        '1w': 'weekly', 'week': 'weekly', 'weekly': 'weekly',
        '1M': 'monthly', 'month': 'monthly', 'monthly': 'monthly',
    }

    FINANCIAL_TABLES = ['Balance', 'Income', 'CashFlow', 'Pershareindex']

    def __init__(self, fallback_to_simulated: bool = True):
        try:
            import akshare as ak
            self._ak = ak
        except ImportError:
            self._ak = None
            logging.getLogger(__name__).warning("akshare not installed, pip install akshare")
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    @smart_cache(cache_type='market', incremental=True)
    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取行情数据（仅使用腾讯财经）"""
        if not self._ak:
            if self._fallback_to_simulated:
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("akshare 未安装，请 pip install akshare")

        # 使用腾讯财经获取数据
        df = self._get_data_from_tx(symbol, start_date, end_date)
        if df is not None and not df.empty:
            return self._process_akshare_data(df, symbol, start_date, end_date)

        # 腾讯财经失败
        if self._fallback_to_simulated:
            return self._generate_simulated_data(start_date, end_date, symbol)
        raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

    def _get_data_from_tx(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从腾讯财经获取数据"""
        try:
            # 转换代码格式为腾讯格式: 000001.SZ -> sz000001
            if '.' in symbol:
                code, suffix = symbol.split('.')
                tx_symbol = f"{suffix.lower()}{code}"
            else:
                tx_symbol = symbol

            # 使用腾讯财经的历史数据接口（后复权，避免前复权随时间变化的问题）
            df = self._ak.stock_zh_a_hist_tx(symbol=tx_symbol, start_date=start_date, end_date=end_date, adjust='hfq')
            if df is not None and not df.empty:
                return df
        except Exception as e:
            self.logger.debug(f"腾讯财经接口失败: {e}")
        return None

    def _process_akshare_data(self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """处理AKShare返回的数据"""
        col_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
                    '最低': 'low', '成交量': 'volume', '成交额': 'amount',
                    '振幅': 'amplitude', '涨跌幅': 'pct_change', '涨跌额': 'change', '换手率': 'turnover',
                    # 新浪财经的列名
                    'day': 'date', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low',
                    'volume': 'volume', 'outstanding_share': 'outstanding_share', 'turnover': 'turnover',
                    # 腾讯财经的列名
                    'Date': 'date', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low',
                    'Volume': 'volume', 'Amount': 'amount', 'Amplitude': 'amplitude',
                    'PctChange': 'pct_change', 'Change': 'change', 'Turnover': 'turnover'}
        df = df.rename(columns=col_map)

        # 处理日期列
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
            df = df.set_index('datetime')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 日期过滤
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        if df.empty:
            raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

        # 数据类型转换
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 保留需要的列
        # 注意：腾讯返回的是 amount（成交额），不是 volume（成交量）
        # 如果没有 volume 但有 amount，使用 amount 作为替代
        keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        if 'volume' not in df.columns and 'amount' in df.columns:
            # 腾讯数据使用 amount 作为成交量（单位：手）
            df['volume'] = df['amount']
            keep_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[keep_cols]

        time.sleep(0.5)  # 减少延迟
        return self.preprocess_data(df)

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        if not self._ak:
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

        # 构造缓存key
        cache_key = f"stock_list_{sector}"
        namespace = 'OpenDataProcessor_Sector'

        # 尝试从缓存读取
        cached = cache_manager.disk_cache.get(namespace, cache_key, 'pickle')
        if cached is not None:
            self.logger.info(f"从缓存加载成分股列表: {sector}, 共 {len(cached)} 只")
            return cached

        try:
            result = self._fetch_stock_list(sector)

            if not result:
                result = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

            # 保存到缓存
            cache_manager.disk_cache.put(namespace, cache_key, result, 'pickle')
            self.logger.info(f"成分股列表已缓存: {sector}, 共 {len(result)} 只")
            return result
        except Exception as e:
            self.logger.warning(f"AKShare获取板块成分股失败: {e}")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

    @staticmethod
    def _get_adjustment_dates() -> List[pd.Timestamp]:
        """生成沪深300/中证系列指数的成分股调整日期列表

        中证指数每年6月和12月的第二个星期五调整成分股，
        此处使用近似日期（每月15日），足够用于历史成分股还原。
        """
        dates = []
        for year in range(2005, 2030):
            for month in [6, 12]:
                dates.append(pd.Timestamp(f'{year}-{month}-15'))
        return dates

    @staticmethod
    def _find_prev_adjustment(date: pd.Timestamp,
                               adj_dates: List[pd.Timestamp]) -> pd.Timestamp:
        """找到指定日期之前最近的调整日期"""
        prev = adj_dates[0]
        for d in adj_dates:
            if d <= date:
                prev = d
            else:
                break
        return prev

    def _fetch_index_constituent_changes(self, index_code: str) -> pd.DataFrame:
        """获取指数成分股变动记录（含纳入日期和剔除日期）

        数据来源:
        1. akshare index_stock_cons: 当前成分股的所有纳入记录
           - 同一股票可能出现多次（被移除后重新纳入）
           - 通过多次纳入记录推断剔除日期
        2. 新浪财经历史成分股页面: 早期（2005-2009）已剔除成分股的精确日期

        对于 index_stock_cons 中的重复纳入记录，推断剔除日期为
        下一次纳入前的最近调整日（中证指数每年6月和12月中旬调整成分股）。

        结果按指数代码缓存，避免重复网络请求。

        Args:
            index_code: 中证指数代码，如 '000300'

        Returns:
            DataFrame with columns: [stock_code, in_date, out_date]
            stock_code: 6位纯数字代码
            in_date: 纳入日期 (datetime)
            out_date: 剔除日期 (datetime, NaT表示仍在指数中)
        """
        namespace = 'OpenDataProcessor_Sector'
        cache_key = f"index_changes_{index_code}"
        cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            self.logger.info(f"从缓存加载指数变动数据: {index_code}, {len(cached)} 条记录")
            return cached

        sina_code = self._SINA_INDEX_CODE_MAP.get(index_code, index_code)
        adj_dates = self._get_adjustment_dates()
        all_records = []

        # ── Step 1: 当前成分股的所有纳入记录 ──
        # index_stock_cons 可能返回同一股票的多条记录（移除后重新纳入）
        stock_entries = {}  # code -> [in_date1, in_date2, ...]
        try:
            df_current = self._ak.index_stock_cons(symbol=sina_code)
            if df_current is not None and not df_current.empty:
                code_col = next((c for c in ['品种代码', '股票代码', 'code']
                                 if c in df_current.columns), None)
                in_date_col = next((c for c in ['纳入日期', '日期']
                                    if c in df_current.columns), None)
                if code_col and in_date_col:
                    for _, row in df_current.iterrows():
                        raw_code = str(row[code_col]).strip()
                        if not raw_code or raw_code == 'nan':
                            continue
                        code = raw_code.zfill(6)
                        in_date = pd.to_datetime(row[in_date_col], errors='coerce')
                        if pd.isna(in_date):
                            continue
                        stock_entries.setdefault(code, []).append(in_date)

                    # 对每只股票，按纳入日期排序，推断剔除日期
                    for code, in_dates in stock_entries.items():
                        in_dates.sort()
                        for i, in_d in enumerate(in_dates):
                            if i < len(in_dates) - 1:
                                # 非最后一次纳入：剔除日期约为下一次纳入前的调整日
                                next_in = in_dates[i + 1]
                                out_d = self._find_prev_adjustment(next_in, adj_dates)
                            else:
                                # 最后一次纳入：仍在指数中
                                out_d = pd.NaT
                            all_records.append({
                                'stock_code': code,
                                'in_date': in_d,
                                'out_date': out_d,
                            })
                    self.logger.info(
                        f"当前成分股: {len(stock_entries)} 只唯一股票, "
                        f"{len(df_current)} 条纳入记录"
                    )
                else:
                    self.logger.warning(f"index_stock_cons 列不包含纳入日期: {list(df_current.columns)}")
        except Exception as e:
            self.logger.warning(f"获取当前成分股变动数据失败: {e}", exc_info=True)

        # ── Step 2: 新浪历史成分股（早期2005-2009的精确剔除记录） ──
        try:
            import requests
            from io import StringIO

            base_url = (f"https://vip.stock.finance.sina.com.cn/corp/view/"
                        f"vII_HistoryComponent.php?indexid={sina_code}")
            headers = {
                'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                               'AppleWebKit/537.36 (KHTML, like Gecko) '
                               'Chrome/120.0.0.0 Safari/537.36')
            }

            page = 1
            total_history = 0
            while page <= 20:
                url = f"{base_url}&p={page}"
                try:
                    resp = requests.get(url, headers=headers, timeout=15)
                    resp.encoding = 'gb2312'
                except Exception:
                    break

                if resp.status_code != 200 or len(resp.text) < 500:
                    break

                try:
                    dfs = pd.read_html(StringIO(resp.text), header=0)
                except Exception:
                    break

                # 新浪页面表格列名为"历史成分"系列，第一行才是真实列名
                page_data = None
                for df in dfs:
                    if len(df) > 3 and any(str(c).startswith('历史成分') for c in df.columns):
                        first_row = df.iloc[0]
                        real_cols = [str(v).strip() for v in first_row.values]
                        df = df.iloc[1:].copy()
                        df.columns = real_cols[:len(df.columns)]
                        df = df.reset_index(drop=True)
                        df = df.dropna(subset=[real_cols[0]], how='all')
                        page_data = df
                        break

                if page_data is None or page_data.empty:
                    break

                code_col = next((c for c in ['品种代码', '股票代码', 'code']
                                 if c in page_data.columns), None)
                in_col = next((c for c in ['纳入日期'] if c in page_data.columns), None)
                out_col = next((c for c in ['剔除日期', '退出日期']
                                if c in page_data.columns), None)

                if code_col:
                    for _, row in page_data.iterrows():
                        code = str(row[code_col]).strip().zfill(6)
                        if code == 'nan' or len(code) < 4:
                            continue
                        in_date = pd.to_datetime(row[in_col], errors='coerce') if in_col else pd.NaT
                        out_date = pd.to_datetime(row[out_col], errors='coerce') if out_col else pd.NaT
                        all_records.append({
                            'stock_code': code,
                            'in_date': in_date,
                            'out_date': out_date,
                        })
                    total_history += len(page_data)

                # 总页数信息
                total_pages = 1
                for df in dfs:
                    for col in df.columns:
                        page_info = str(col)
                        if '共' in page_info and '页' in page_info:
                            import re
                            m = re.search(r'共(\d+)页', page_info)
                            if m:
                                total_pages = int(m.group(1))
                                break

                if page >= total_pages:
                    break
                page += 1
                time.sleep(0.3)

            self.logger.info(f"从新浪获取历史变动: {page} 页, {total_history} 条记录")
        except ImportError:
            self.logger.warning("requests 未安装，无法获取新浪历史成分股变动数据")
        except Exception as e:
            self.logger.warning(f"从新浪获取历史成分股变动失败: {e}")

        if not all_records:
            return pd.DataFrame(columns=['stock_code', 'in_date', 'out_date'])

        df_changes = pd.DataFrame(all_records)
        cache_manager.disk_cache.put(namespace, cache_key, df_changes, 'parquet')
        self.logger.info(f"指数变动数据已缓存: {index_code}, {len(df_changes)} 条记录")
        return df_changes

    def get_historical_stock_list(self, sector: str = '沪深A股',
                                   date: Optional[str] = None) -> List[str]:
        """获取指定日期的历史成分股列表

        基于新浪财经的历史成分股变动数据（纳入日期 + 剔除日期）还原
        任意日期的指数成分股，确保回测时成分股与回测时期一致。

        支持的指数: 沪深300、中证500、中证1000、上证50

        数据按日期缓存到磁盘，变动记录按指数代码单独缓存。

        Args:
            sector: 板块名称，如 '沪深300', '中证500', '中证1000', '上证50'
            date: 目标日期，格式 'YYYY-MM-DD'，默认为当前日期

        Returns:
            股票代码列表（QMT格式，如 ['000001.SZ', '600000.SH']）
        """
        if not self._ak:
            return self.get_stock_list(sector)

        if date is None:
            date = pd.Timestamp.now().strftime('%Y-%m-%d')

        # 沪深A股只有当前成分股，没有历史版本
        if sector in ('沪深A股', 'A股', '全部A股'):
            return self.get_stock_list(sector)

        # 构造按日期的缓存key
        date_compact = date.replace('-', '')
        cache_key = f"hist_stock_list_{sector}_{date_compact}"
        namespace = 'OpenDataProcessor_Sector'

        # 尝试从缓存读取
        cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            if 'stock_code' in cached.columns:
                stock_list = cached['stock_code'].astype(str).tolist()
                self.logger.info(f"从缓存加载历史成分股: {sector}, date={date}, 共 {len(stock_list)} 只")
                return stock_list

        result = []
        sector_map = {'沪深300': '000300', '中证500': '000905', '中证1000': '000852', '上证50': '000016'}
        index_code = sector_map.get(sector)

        if index_code:
            # 获取成分股变动记录
            df_changes = self._fetch_index_constituent_changes(index_code)

            if not df_changes.empty:
                target_date = pd.Timestamp(date)
                # 筛选: 纳入日期 <= 目标日期 AND (剔除日期为空 OR 剔除日期 > 目标日期)
                mask = (
                    (df_changes['in_date'] <= target_date) &
                    (df_changes['out_date'].isna() | (df_changes['out_date'] > target_date))
                )
                hist_codes = df_changes.loc[mask, 'stock_code'].unique().tolist()
                result = [_convert_symbol_to_qmt(c) for c in hist_codes]
                self.logger.info(
                    f"还原 {sector} 历史成分股: date={date}, "
                    f"变动记录={len(df_changes)}, 还原={len(result)} 只"
                )

        if not result:
            # 回退到当前成分股（不缓存，避免错误数据污染缓存）
            self.logger.warning(f"历史成分股还原失败，使用当前成分股: {sector}")
            result = self.get_stock_list(sector)
            return result

        # 缓存到磁盘（仅缓存成功还原的历史成分股）
        if result:
            df_cache = pd.DataFrame({'stock_code': result})
            cache_manager.disk_cache.put(namespace, cache_key, df_cache, 'parquet')
            self.logger.info(f"历史成分股已缓存: {sector}, date={date}, 共 {len(result)} 只")

        return result

    def _fetch_stock_list(self, sector: str = '沪深A股') -> List[str]:
        """从AKShare获取成分股列表（内部方法）"""
        sector_map = {'沪深300': '000300', '中证500': '000905', '中证1000': '000852', '上证50': '000016'}
        index_code = sector_map.get(sector)
        result = []
        if index_code:
            df = self._ak.index_stock_cons_csindex(symbol=index_code)
            if df is not None and not df.empty:
                code_col = next((c for c in ['品种代码', '股票代码', 'code', '成分券代码'] if c in df.columns), None)
                if code_col:
                    result = [_convert_symbol_to_qmt(c) for c in df[code_col].astype(str).tolist()]
        if sector in ('沪深A股', 'A股', '全部A股'):
            df = self._ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                code_col = next((c for c in ['代码', '股票代码', 'code'] if c in df.columns), None)
                if code_col:
                    result = [_convert_symbol_to_qmt(c) for c in df[code_col].astype(str).tolist()]
        return result

    def download_financial_data(self, stock_list: List[str],
                                  table_list: Optional[List[str]] = None,
                                  start_time: str = '', end_time: str = '') -> None:
        pass  # AKShare的财务数据是实时获取的，不需要预下载

    def get_financial_data(self, stock_list: List[str],
                           table_list: Optional[List[str]] = None,
                           start_time: str = '', end_time: str = '',
                           report_type: str = 'announce_time') -> Dict[str, Any]:
        if not self._ak:
            return {}
        tables = table_list or self.FINANCIAL_TABLES
        namespace = 'OpenDataProcessor_Financial'

        result = {}
        cache_hits = 0
        download_hits = 0
        total = len(stock_list)
        phase_start = time.time()

        for i, symbol in enumerate(stock_list, 1):
            ak_code = _convert_symbol_to_opendata_financial(symbol)
            symbol_data = {}
            need_download_tables = []

            for table in tables:
                table_suffix = f"{table}_{report_type}"

                available_years = cache_manager.index_manager.get_available_financial_years(symbol, table_suffix)
                if not available_years:
                    available_years = cache_manager.disk_cache.list_yearly_files(namespace, symbol, table_suffix)

                if available_years:
                    df = cache_manager.disk_cache.get_yearly_range(namespace, symbol, sorted(available_years), table_suffix)
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        df = self._convert_akshare_columns(df, table)
                        df = self._set_datetime_index(df)
                        df = self._filter_cache_by_request(df, start_time, end_time)
                        if not df.empty:
                            symbol_data[table] = df
                            continue

                cache_key = f"{symbol}_{table}"
                cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
                if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                    cached_with_index = self._set_datetime_index(cached.copy())
                    if not cached_with_index.empty and self._is_cache_sufficient(cached_with_index, start_time, end_time):
                        filtered_cache = self._filter_cache_by_request(cached_with_index, start_time, end_time)
                        cached = self._convert_akshare_columns(filtered_cache, table)
                        cached = self._set_datetime_index(cached)
                        symbol_data[table] = cached

                        if isinstance(cached_with_index.index, pd.DatetimeIndex) and not cached_with_index.empty:
                            written = cache_manager.disk_cache.put_yearly_from_df(namespace, symbol, table_suffix, cached_with_index)
                            for y in written:
                                cache_manager.index_manager.update_financial_index(symbol, table_suffix, y)
                            cache_manager.index_manager.save_index()
                        continue

                need_download_tables.append(table)

            if not need_download_tables and symbol_data:
                result[symbol] = symbol_data
                cache_hits += 1
                continue

            if need_download_tables:
                for table in need_download_tables:
                    try:
                        raw_df = self._get_akshare_financial_data_raw(ak_code, table, start_time, end_time)
                        if raw_df is not None and not raw_df.empty:
                            table_suffix = f"{table}_{report_type}"

                            raw_with_index = self._set_datetime_index(raw_df.copy())
                            if isinstance(raw_with_index.index, pd.DatetimeIndex) and not raw_with_index.empty:
                                written = cache_manager.disk_cache.put_yearly_from_df(namespace, symbol, table_suffix, raw_with_index)
                                for y in written:
                                    cache_manager.index_manager.update_financial_index(symbol, table_suffix, y)
                                cache_manager.index_manager.save_index()

                            df = self._convert_akshare_columns(raw_df, table)
                            df = self._set_datetime_index(df)
                            symbol_data[table] = df
                            download_hits += 1
                    except Exception as e:
                        self.logger.warning(f"下载 {symbol} {table} 失败: {e}")

            if symbol_data:
                result[symbol] = symbol_data

            if i % 20 == 0 or i == total:
                self.logger.info(
                    f"[ {i} / {total} ] 进度: {cache_hits} 缓存命中, {download_hits} 已下载"
                )

        phase_elapsed = time.time() - phase_start
        self.logger.info(
            f"OpenData财务数据获取完成: 缓存命中 {cache_hits}, 新下载 {download_hits}, 耗时 {phase_elapsed:.1f}秒"
        )
        return result

    def _get_akshare_financial_data_raw(self, ak_code: str, table: str, start_time: str = '', end_time: str = '') -> Optional[pd.DataFrame]:
        """获取原始财务数据（不进行列名映射，用于缓存）"""
        try:
            df = None
            if table == 'Balance':
                df = self._ak.stock_balance_sheet_by_report_em(symbol=ak_code)
            elif table == 'Income':
                df = self._ak.stock_profit_sheet_by_report_em(symbol=ak_code)
            elif table == 'CashFlow':
                df = self._ak.stock_cash_flow_sheet_by_report_em(symbol=ak_code)
            elif table == 'Pershareindex':
                df = self._get_pershareindex_data(ak_code, start_time, end_time)
            else:
                return None
            if df is None or df.empty:
                return None
            # 不调用 _convert_akshare_columns，返回原始数据
            return df
        except Exception as e:
            self.logger.debug(f"AKShare获取 {ak_code} {table} 原始数据失败: {e}")
            return None

    def _set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """设置DataFrame的日期索引

        关键：使用公告日期作为索引，而不是报告期。
        因为回测时必须避免 look-ahead bias（前瞻性偏差），
        只有等公告日期之后，财报数据才真正可用。
        """
        if df is None or df.empty:
            return df
        index_set = False
        # 优先使用公告日期列（避免未来数据）
        # 支持中文和英文列名
        for date_col in ['公告日期', '最新公告日期', 'NOTICE_DATE']:
            if date_col in df.columns:
                try:
                    dt_col = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col])
                    if not df.empty:
                        df.index = dt_col.loc[df.index]
                        df = df.sort_index()
                        index_set = True
                        break
                except Exception:
                    pass
        # 如果没有公告日期，使用报告期（不推荐，但为了兼容性保留）
        if not index_set:
            for report_col in ['报告期', 'REPORT_DATE']:
                if report_col in df.columns:
                    try:
                        dt_col = pd.to_datetime(df[report_col], errors='coerce')
                        df = df.dropna(subset=[report_col])
                        if not df.empty:
                            df.index = dt_col.loc[df.index]
                            df = df.sort_index()
                            index_set = True
                            break
                    except Exception:
                        pass
        return df

    def _is_cache_sufficient(self, cached_df: pd.DataFrame, start_time: str, end_time: str) -> bool:
        """检查缓存数据的时间范围是否满足需求"""
        if cached_df.empty:
            return False
        try:
            # 获取缓存数据的时间范围
            cache_start = cached_df.index.min()
            cache_end = cached_df.index.max()
            
            # 解析请求的时间范围（不使用往前推一年的逻辑）
            if start_time and len(start_time) >= 10:
                req_start = pd.to_datetime(start_time[:10])
            else:
                req_start = cache_start
            
            if end_time and len(end_time) >= 10:
                req_end = pd.to_datetime(end_time[:10])
            else:
                req_end = pd.Timestamp.now()
            
            # 对于财务数据（公告日期），判断缓存是否满足需求：
            # 1. 缓存的起始日期是否覆盖请求起始日期（允许1年误差，因为财报是季度数据，且回测可能只需要近期数据）
            # 2. 缓存的结束日期是否覆盖请求结束日期（允许30天误差）
            # 3. 或者缓存的最新日期已经接近今天（60天内），则认为没有更新的财报了
            
            today = pd.Timestamp.now().normalize()
            cache_is_recent = (today - cache_end).days <= 60
            
            # 检查缓存是否覆盖请求的起始日期（允许1年的误差）
            # 因为财务数据是季度的，且回测时可能只需要覆盖回测期间的数据
            cache_covers_start = cache_start <= req_start
            cache_covers_start_approx = (cache_start - req_start).days <= 365  # 1年缓冲期
            
            # 检查缓存是否覆盖请求的结束日期（允许30天的误差）
            cache_covers_end = cache_end >= req_end
            cache_covers_end_approx = (req_end - cache_end).days <= 30  # 30天缓冲期
            
            # 特殊情况1：如果缓存的起始日期在请求起始日期之后，但差距不超过2年，
            # 且缓存覆盖了请求的结束日期（或接近），则认为缓存足够
            # 这可能是因为股票是新上市的，或者缓存只包含近期数据
            cache_is_new_stock = (cache_start - req_start).days <= 730 and (cache_covers_end or cache_covers_end_approx)
            
            # 特殊情况2：新股处理 - 如果缓存的起始日期在请求的结束日期之后，
            # 这意味着在请求的时间范围内，这只股票还没有财务数据（可能还未上市）
            # 这种情况下，缓存是"足够"的，因为不需要下载不存在的历史数据
            cache_starts_after_request_end = cache_start > req_end
            
            # 缓存足够条件：
            # 1. 缓存覆盖起始日期（或接近）
            # 2. 缓存覆盖结束日期（或接近，或缓存较新）
            # 3. 或者是新上市股票（缓存起始日期在请求起始日期后2年内，且覆盖结束日期）
            # 4. 或者是未上市股票（缓存起始日期在请求结束日期之后）
            covers_start = cache_covers_start or cache_covers_start_approx
            covers_end = cache_covers_end or cache_covers_end_approx or cache_is_recent
            
            return (covers_start and covers_end) or cache_is_new_stock or cache_starts_after_request_end
        except Exception:
            return False
    
    def _get_cache_date_ranges(self, cached_df: pd.DataFrame, start_time: str, end_time: str) -> dict:
        """计算缓存数据与请求范围的差异，返回需要增量下载的范围
        
        Returns:
            dict: {
                'cache_start': 缓存开始日期,
                'cache_end': 缓存结束日期,
                'requested_start': 请求开始日期（考虑往前推一年）,
                'requested_end': 请求结束日期,
                'need_earlier': 是否需要下载更早的数据,
                'need_later': 是否需要下载更新的数据
            }
        """
        from datetime import datetime
        
        # 确保 cached_df 有 DatetimeIndex
        cached_df = self._set_datetime_index(cached_df.copy())
        if cached_df.empty:
            return {
                'cache_start': '',
                'cache_end': '',
                'requested_start': '',
                'requested_end': '',
                'need_earlier': False,
                'need_later': True
            }
        
        # 获取缓存数据的时间范围
        cache_start = cached_df.index.min()
        cache_end = cached_df.index.max()
        
        # 解析请求的时间范围（考虑往前推一年的逻辑）
        if start_time and len(start_time) >= 10:
            req_start = pd.to_datetime(start_time[:10])
            # 往前推一年，因为 _get_pershareindex_data 会这样做
            req_start = req_start - pd.Timedelta(days=365)
        else:
            req_start = cache_start
        
        if end_time and len(end_time) >= 10:
            req_end = pd.to_datetime(end_time[:10])
        else:
            req_end = pd.Timestamp.now()
        
        # 计算是否需要下载更早或更新的数据（允许1个月的误差，因为财报是季度数据）
        need_earlier = req_start < (cache_start - pd.Timedelta(days=30))
        need_later = req_end > (cache_end + pd.Timedelta(days=30))
        
        return {
            'cache_start': cache_start.strftime('%Y-%m-%d') if isinstance(cache_start, pd.Timestamp) else str(cache_start),
            'cache_end': cache_end.strftime('%Y-%m-%d') if isinstance(cache_end, pd.Timestamp) else str(cache_end),
            'requested_start': req_start.strftime('%Y-%m-%d') if isinstance(req_start, pd.Timestamp) else str(req_start),
            'requested_end': req_end.strftime('%Y-%m-%d') if isinstance(req_end, pd.Timestamp) else str(req_end),
            'need_earlier': need_earlier,
            'need_later': need_later
        }
    
    def _filter_cache_by_request(self, cached_df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
        """根据请求的时间范围过滤缓存数据，只保留实际需要的数据
        
        财务数据策略：最多往前看一年（用于计算同比增长等），再往前没有意义
        
        特殊情况：如果缓存数据的起始日期在请求结束日期之后，说明在请求时间范围内
        这只股票还没有财务数据（可能还未上市），返回空DataFrame
        """
        if cached_df.empty:
            return cached_df
        
        try:
            # 解析请求的时间范围
            if start_time and len(start_time) >= 10:
                req_start = pd.to_datetime(start_time[:10])
                # 往前推一年，因为财务分析需要历史数据做对比
                filter_start = req_start - pd.Timedelta(days=365)
            else:
                filter_start = cached_df.index.min()
            
            if end_time and len(end_time) >= 10:
                filter_end = pd.to_datetime(end_time[:10])
            else:
                filter_end = cached_df.index.max()
            
            # 特殊情况：如果缓存数据的起始日期在请求结束日期之后，
            # 说明在请求时间范围内这只股票还没有财务数据，返回空DataFrame
            cache_start = cached_df.index.min()
            if cache_start > filter_end:
                self.logger.debug(f"缓存起始日期({cache_start.date()})在请求结束日期({filter_end.date()})之后，"
                                  f"股票在请求期间尚未有财务数据")
                return pd.DataFrame(columns=cached_df.columns)
            
            # 过滤数据：只保留 filter_start ~ filter_end 范围内的数据
            mask = (cached_df.index >= filter_start) & (cached_df.index <= filter_end)
            filtered = cached_df.loc[mask].copy()
            
            if len(filtered) < len(cached_df):
                self.logger.debug(f"缓存数据过滤: {len(cached_df)} 条 -> {len(filtered)} 条 "
                                  f"({filter_start.date()} ~ {filter_end.date()})")
            
            return filtered
        except Exception:
            # 如果过滤失败，返回原始数据
            return cached_df

    def _merge_financial_data(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """合并现有缓存数据和新下载的数据，去除重复"""
        try:
            # 合并数据
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            # 根据报告期去重（如果存在报告期列）
            if '报告期' in merged.columns:
                merged = merged.drop_duplicates(subset=['报告期'], keep='last')
            elif '公告日期' in merged.columns:
                merged = merged.drop_duplicates(subset=['公告日期'], keep='last')
            else:
                # 如果没有明确的日期列，尝试根据所有列去重
                merged = merged.drop_duplicates(keep='last')
            return merged
        except Exception:
            # 如果合并失败，返回新数据
            return new_df

    def _get_pershareindex_data(self, ak_code: str, start_time: str = '', end_time: str = '') -> Optional[pd.DataFrame]:
        import re
        from datetime import datetime, timedelta
        code = re.sub(r'^[A-Za-z]+', '', ak_code)
        now = datetime.now()
        
        # 解析时间范围
        start_year = now.year - 4  # 默认获取最近4年的数据
        end_year = now.year
        
        if start_time and len(start_time) >= 4:
            try:
                # 如果传入了开始时间，往前推一年获取财务数据，确保回测开始时有足够历史数据
                start_dt = datetime.strptime(start_time[:10], '%Y-%m-%d')
                start_dt = start_dt - timedelta(days=365)
                start_year = start_dt.year
            except ValueError:
                try:
                    start_year = int(start_time[:4]) - 1  # 往前推一年
                except ValueError:
                    pass
        if end_time and len(end_time) >= 4:
            try:
                end_year = int(end_time[:4])
            except ValueError:
                pass
        
        # 确保年份范围合理
        start_year = max(start_year, now.year - 10)  # 最多获取10年历史数据
        end_year = min(end_year, now.year)
        
        dates = []
        for y in range(end_year, start_year - 1, -1):  # 从最新年份开始
            for period in ['1231', '0930', '0630', '0331']:
                date_str = f"{y}{period}"
                if date_str <= now.strftime('%Y%m%d'):
                    dates.append(date_str)
        
        all_dfs = []
        for date in dates:
            try:
                batch_df = self._ak.stock_yjbb_em(date=date)
                if batch_df is not None and not batch_df.empty:
                    code_col = next((c for c in ['股票代码', '代码', 'symbol', 'code'] if c in batch_df.columns), None)
                    if code_col:
                        filtered = batch_df[batch_df[code_col].astype(str) == code]
                        if not filtered.empty:
                            # 添加报告期列（如果不存在），用于正确设置日期索引
                            # date 格式为 "20241231"，转换为日期
                            filtered = filtered.copy()
                            if '报告期' not in filtered.columns:
                                report_date = datetime.strptime(date, '%Y%m%d')
                                filtered['报告期'] = report_date
                            all_dfs.append(filtered)
                time.sleep(0.3)  # 减少延迟，加快获取速度
            except Exception:
                continue
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        return None

    def _convert_akshare_columns(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        column_map = {
            '资产总计': 'total_assets', '负债合计': 'total_liabilities',
            '所有者权益合计': 'total_equity', '流动资产合计': 'total_current_assets',
            '非流动资产合计': 'total_noncurrent_assets', '流动负债合计': 'total_current_liabilities',
            '非流动负债合计': 'total_noncurrent_liabilities', '货币资金': 'cash_and_equivalents',
            '应收账款': 'accounts_receivable', '存货': 'inventory',
            '营业总收入': 'total_operate_income', '营业收入': 'operate_income',
            '营业总成本': 'total_operate_cost', '营业成本': 'operate_cost',
            '营业利润': 'operate_profit', '利润总额': 'total_profit',
            '净利润': 'net_profit', '归属于母公司股东的净利润': 'net_profit_parent',
            '基本每股收益': 'eps_basic', '稀释每股收益': 'eps_diluted',
            '经营活动产生的现金流量净额': 'net_operate_cash_flow',
            '投资活动产生的现金流量净额': 'net_invest_cash_flow',
            '筹资活动产生的现金流量净额': 'net_finance_cash_flow',
            '现金及现金等价物净增加额': 'net_cash_increase',
            '每股收益': 's_fa_eps_basic', '每股净资产': 's_fa_bps', '每股经营现金流量': 's_fa_ocfps',
            '每股资本公积金': 's_fa_surpluscapitalps', '每股未分配利润': 's_fa_undistributedps',
            '净资产收益率': 'du_return_on_equity', '净资产收益率-加权': 'equity_roe',
            '净资产收益率-摊薄': 'net_roe', '销售毛利率': 'gross_profit_margin',
            # stock_yjbb_em 接口返回的ROE字段名
            'roe': 'du_return_on_equity',
            '营业收入同比增长率': 'inc_revenue_rate', '净利润同比增长率': 'inc_net_profit_rate',
            # stock_yjbb_em 返回的列名（带连字符）
            '净利润-同比增长': 'inc_net_profit_rate',
            '营业总收入-同比增长': 'inc_revenue_rate',
            # 其他可能的列名格式
            'NETPROFIT_YOY': 'inc_net_profit_rate',
            'OPERATE_INCOME_YOY': 'inc_revenue_rate',
        }
        return df.rename(columns=column_map)

    def get_industry_mapping(self, level: int = 1,
                               stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        """获取行业分类映射

        优先使用申万行业分类（从CSIndex获取，稳定性高），
        如果失败则尝试东方财富行业板块接口。

        Args:
            level: 行业级别，1=一级行业，2=二级行业，3=三级行业
            stock_pool: 股票池，为空则返回所有股票

        Returns:
            { stock_code: industry_name, ... } 映射字典
        """
        if not self._ak:
            return {}

        # 构造缓存key
        pool_hash = hashlib.md5(','.join(sorted(stock_pool)).encode()).hexdigest()[:8] if stock_pool else 'all'
        cache_key = f"industry_mapping_level{level}_{pool_hash}"
        namespace = 'OpenDataProcessor_Industry'

        # 尝试从缓存读取
        cached = cache_manager.disk_cache.get(namespace, cache_key, 'pickle')
        if cached is not None:
            self.logger.info(f"从缓存加载行业映射数据: level={level}, 共 {len(cached)} 只股票")
            return cached

        # 尝试申万行业分类（主方法，稳定性高）
        try:
            industry_map = self._get_industry_mapping_sw_all(level, stock_pool)
            if industry_map:
                cache_manager.disk_cache.put(namespace, cache_key, industry_map, 'pickle')
                self.logger.info(f"行业映射数据已缓存(申万): level={level}, 共 {len(industry_map)} 只股票")
                return industry_map
        except Exception as e:
            self.logger.debug(f"申万行业分类获取失败: {e}")

        # 备选：东方财富行业板块
        try:
            self.logger.info("尝试东方财富行业板块获取...")
            industry_map = self._get_industry_mapping_by_em_sector(stock_pool)
            if industry_map:
                cache_manager.disk_cache.put(namespace, cache_key, industry_map, 'pickle')
                self.logger.info(f"行业映射数据已缓存(东财): level={level}, 共 {len(industry_map)} 只股票")
                return industry_map
        except Exception as e:
            self.logger.debug(f"东方财富行业板块获取失败: {e}")

        self.logger.warning("所有方法获取行业数据均失败，返回空数据")
        return {}

    def _get_industry_mapping_sw_all(self, level: int = 1,
                                      stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        """使用申万行业分类获取所有股票的行业映射

        申万行业分类数据从CSIndex获取，不受东方财富连接问题影响。
        """
        industry_map = {}
        pool_set = set(stock_pool) if stock_pool else None

        # 申万一级行业指数代码列表（部分主要行业）
        sw_index_codes = {
            '000819': '有色金属', '000820': '采掘', '000821': '化工',
            '000822': '钢铁', '000823': '建筑材料', '000824': '建筑装饰',
            '000825': '电气设备', '000826': '机械设备', '000827': '国防军工',
            '000828': '汽车', '000829': '家用电器', '000830': '纺织服装',
            '000831': '轻工制造', '000832': '商业贸易', '000833': '农林牧渔',
            '000834': '食品饮料', '000835': '休闲服务', '000836': '医药生物',
            '000837': '公用事业', '000838': '交通运输', '000839': '房地产',
            '000840': '电子', '000841': '计算机', '000842': '传媒',
            '000843': '通信', '000844': '银行', '000845': '非银金融',
            '000846': '综合', '000847': '建筑材料', '000848': '建筑装饰',
        }

        self.logger.info(f"开始获取申万行业分类数据，共 {len(sw_index_codes)} 个行业...")

        for idx, (index_code, industry_name) in enumerate(sw_index_codes.items(), 1):
            try:
                # 获取指数成分股
                df = self._ak.index_stock_cons_weight_csindex(symbol=index_code)
                if df is not None and not df.empty:
                    code_col = next((c for c in ['成分券代码', '成分券代码', '股票代码', 'code'] if c in df.columns), None)

                    if code_col:
                        count = 0
                        for _, row in df.iterrows():
                            code = str(row.get(code_col, '')).strip()
                            if code:
                                qmt_code = _convert_symbol_to_qmt(code)
                                if pool_set is None or qmt_code in pool_set:
                                    industry_map[qmt_code] = industry_name
                                    count += 1

                        self.logger.debug(f"  [{idx}/{len(sw_index_codes)}] {industry_name}: {count}只股票")

                # 添加延迟避免请求过快
                if idx < len(sw_index_codes):
                    time.sleep(0.2)

            except Exception as e:
                self.logger.debug(f"获取申万行业 {industry_name} 失败: {e}")
                continue

        if industry_map:
            self.logger.info(f"申万行业分类获取完成: {len(industry_map)} 只股票")
        else:
            self.logger.warning("申万行业分类未获取到任何数据")

        return industry_map

    def _get_industry_mapping_by_em_sector(self, stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        """通过东方财富行业板块获取行业映射（带重试）"""
        def _retry_request(func, max_retries=3, delay=2):
            for attempt in range(max_retries):
                try:
                    return func()
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.debug(f"请求失败，{delay}秒后重试 ({attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay * (attempt + 1))
                    else:
                        raise
            return None

        df = _retry_request(lambda: self._ak.stock_board_industry_name_em())
        if df is None or df.empty:
            return {}

        industry_map = {}
        total_industries = len(df)

        for idx, (_, row) in enumerate(df.iterrows(), 1):
            industry_name = row.get('板块名称', '')
            if not industry_name:
                continue

            try:
                stocks_df = _retry_request(
                    lambda: self._ak.stock_board_industry_cons_em(symbol=industry_name),
                    max_retries=2,
                    delay=1
                )

                if stocks_df is not None and not stocks_df.empty:
                    code_col = next((c for c in ['代码', '股票代码', 'code'] if c in stocks_df.columns), None)
                    if code_col:
                        for code in stocks_df[code_col].astype(str).tolist():
                            qmt_code = _convert_symbol_to_qmt(code)
                            if stock_pool is None or qmt_code in stock_pool:
                                industry_map[qmt_code] = industry_name

                if idx < total_industries:
                    time.sleep(0.3)

            except Exception as e:
                self.logger.debug(f"获取行业 '{industry_name}' 成分股失败: {e}")
                continue

        return industry_map

    def get_historical_industry_mapping(self,
                                         stock_list: List[str],
                                         date: Optional[str] = None,
                                         classification: str = '申银万国行业分类标准') -> Dict[str, str]:
        """获取指定日期的历史行业分类映射

        基于 stock_industry_change_cninfo 接口，可以获取个股历史上的行业变更记录，
        从而还原任意时点的行业分类。

        Args:
            stock_list: 股票代码列表（QMT格式，如 ['000001.SZ', '600000.SH']）
            date: 目标日期，格式 'YYYY-MM-DD'，默认为当前日期
            classification: 行业分类标准，默认'申银万国行业分类标准'，
                          可选：'申银万国行业分类标准', '中证行业分类标准', '巨潮行业分类标准'

        Returns:
            { stock_code: industry_name, ... } 映射字典

        Examples:
            >>> processor = OpenDataProcessor()
            >>> # 获取2023年1月1日的行业分类
            >>> mapping = processor.get_historical_industry_mapping(
            ...     ['000001.SZ', '600519.SH'],
            ...     date='2023-01-01'
            ... )
            >>> print(mapping)
            {'000001.SZ': '银行', '600519.SH': '食品饮料'}
        """
        if not self._ak:
            return {}

        if date is None:
            date = pd.Timestamp.now().strftime('%Y-%m-%d')

        target_date = pd.Timestamp(date)

        # 构造缓存key
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:8]
        cache_key = f"hist_industry_{classification}_{date}_{stocks_hash}"
        namespace = 'OpenDataProcessor_Industry'

        # 尝试从缓存读取
        cached = cache_manager.disk_cache.get(namespace, cache_key, 'pickle')
        if cached is not None:
            self.logger.info(f"从缓存加载历史行业数据: date={date}, 共 {len(cached)} 只股票")
            return cached

        industry_map = {}
        total = len(stock_list)

        self.logger.info(f"开始获取历史行业数据: date={date}, classification={classification}, 共 {total} 只股票")

        for i, symbol in enumerate(stock_list, 1):
            try:
                # 转换股票代码格式（去掉.SZ/.SH后缀）
                ak_code = _convert_symbol_to_opendata(symbol)

                # 获取行业变更历史
                df = self._ak.stock_industry_change_cninfo(symbol=ak_code)

                if df is not None and not df.empty:
                    # 检查是否有变更日期列（某些次新股可能没有）
                    if '变更日期' not in df.columns:
                        self.logger.debug(f"{symbol} 返回数据缺少'变更日期'列，跳过")
                        continue

                    # 转换日期列
                    df['变更日期'] = pd.to_datetime(df['变更日期'])

                    # 筛选指定分类标准
                    mask = df['分类标准'] == classification
                    df_filtered = df[mask] if mask.any() else df

                    # 找到目标日期前最新的行业
                    mask_date = df_filtered['变更日期'] <= target_date
                    if mask_date.any():
                        latest = df_filtered[mask_date].iloc[-1]

                        # 根据分类标准选择合适的行业级别
                        if classification == '申银万国行业分类标准':
                            # 申万：使用行业次类（一级行业）或行业中类（二级行业）
                            industry = latest.get('行业次类') or latest.get('行业大类') or latest.get('行业门类')
                        elif classification == '中证行业分类标准':
                            # 中证：使用行业次类
                            industry = latest.get('行业次类') or latest.get('行业大类')
                        else:
                            # 其他：使用行业中类或行业大类
                            industry = latest.get('行业中类') or latest.get('行业大类') or latest.get('行业门类')

                        if industry and pd.notna(industry):
                            industry_map[symbol] = industry

                if (i % 10 == 0 or i == total) and total > 1:
                    self.logger.info(f"  进度: [{i}/{total}] {symbol} -> {industry_map.get(symbol, 'N/A')}")

                # 添加延迟避免请求过快
                if i < total:
                    time.sleep(0.3)

            except Exception as e:
                self.logger.debug(f"获取 {symbol} 历史行业数据失败: {e}")
                continue

        if industry_map:
            # 保存到缓存
            cache_manager.disk_cache.put(namespace, cache_key, industry_map, 'pickle')
            self.logger.info(f"历史行业数据已缓存: date={date}, 共 {len(industry_map)} 只股票")
        else:
            self.logger.warning(f"未获取到任何历史行业数据: date={date}")

        return industry_map

    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        if not self._ak:
            return {}
        namespace = 'OpenDataProcessor_Financial'
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"dividend_merged_{stocks_hash}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='dividend')
            self.logger.info(f"从合并缓存(精确)加载分红数据: {len(merged_cached)} 只股票")
            return merged_cached
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.get_namespace_dir(namespace)
        if ns_dir.exists():
            for cache_file in ns_dir.glob('dividend_merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='dividend')
                                result_subset = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if len(result_subset) == len(stock_list):
                                    return result_subset
                except Exception:
                    continue
        result = {}
        total = len(stock_list)
        for i, symbol in enumerate(stock_list, 1):
            cache_key = f"{symbol}_dividend"
            cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
            if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                result[symbol] = cached
                continue
            try:
                ak_code = _convert_symbol_to_opendata(symbol)
                df = self._ak.stock_dividend_cninfo(symbol=ak_code)
                if df is not None and not df.empty:
                    df = df.rename(columns={'分红年度': 'dividend_year', '派息日': 'dividend_date',
                                             '每股派息': 'dividend_per_share', '股权登记日': 'record_date',
                                             '除权除息日': 'ex_dividend_date'})
                    result[symbol] = df
                    cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
            except Exception:
                pass
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='dividend')
            if merged_df is not None:
                cache_manager.disk_cache.put(namespace, merged_cache_key, merged_df, 'parquet')
        return result

    def _generate_simulated_data(self, start_date: str, end_date: str, symbol: str = None) -> pd.DataFrame:
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        seed = hash(symbol) % 10000 if symbol else 42
        rng = np.random.default_rng(seed)
        base_price = 10.0 + rng.random() * 5.0
        prices = []
        for i in range(len(date_range)):
            base_price *= (1 + rng.normal(0, 0.02))
            prices.append(base_price)
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices], 'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices], 'close': prices,
            'volume': rng.integers(10000, 100000, len(date_range)),
        }, index=date_range)
        return self.preprocess_data(df)
