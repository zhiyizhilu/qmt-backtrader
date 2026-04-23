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


def _convert_symbol_to_akshare(symbol: str) -> str:
    """将QMT格式股票代码转换为akshare行情格式 (纯数字)"""
    return symbol.split('.')[0]


def _convert_symbol_to_akshare_financial(symbol: str) -> str:
    """将QMT格式股票代码转换为akshare财务数据格式 (如 SZ000001)"""
    parts = symbol.split('.')
    if len(parts) == 2:
        return f"{parts[1].upper()}{parts[0]}"
    return f"SH{symbol}" if symbol.startswith(('6', '9')) else f"SZ{symbol}"


def _convert_symbol_to_baostock(symbol: str) -> str:
    """将QMT格式股票代码转换为baostock格式 (如 sz.000001)"""
    parts = symbol.split('.')
    if len(parts) == 2:
        return f"{parts[1].lower()}.{parts[0]}"
    return f"sh.{symbol}" if symbol.startswith(('6', '9')) else f"sz.{symbol}"


class AKShareDataProcessor(DataProcessor):
    """AKShare数据处理器

    基于东方财富数据源，支持获取A股历史行情数据，数据范围远超QMT的一年限制。
    股票代码统一使用QMT格式 (000001.SZ)，内部自动转换。
    """

    INDEX_CODE_MAP = {
        '000300.SH': '000300',
        '000905.SH': '000905',
        '000852.SH': '000852',
        '000016.SH': '000016',
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
        if not self._ak:
            if self._fallback_to_simulated:
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("akshare 未安装，请 pip install akshare")
        try:
            ak_code = _convert_symbol_to_akshare(symbol)
            ak_period = self.PERIOD_MAP.get(period, 'daily')
            start_dt = start_date.replace('-', '')
            end_dt = end_date.replace('-', '')
            max_retries = 3
            df = None
            for retry in range(max_retries):
                try:
                    df = self._ak.stock_zh_a_hist(symbol=ak_code, period=ak_period,
                                                   start_date=start_dt, end_date=end_dt, adjust="qfq")
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(2 * (retry + 1))
                    else:
                        raise
            if df is None or df.empty:
                if self._fallback_to_simulated:
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")
            col_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
                        '最低': 'low', '成交量': 'volume', '成交额': 'amount',
                        '振幅': 'amplitude', '涨跌幅': 'pct_change', '涨跌额': 'change', '换手率': 'turnover'}
            df = df.rename(columns=col_map)
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                df = df.set_index('datetime')
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if df.empty:
                if self._fallback_to_simulated:
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[keep_cols]
            time.sleep(1.0)
            return self.preprocess_data(df)
        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError(f"AKShare获取数据失败: {e}")

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        if not self._ak:
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        try:
            sector_map = {'沪深300': '000300', '中证500': '000905', '中证1000': '000852', '上证50': '000016'}
            index_code = sector_map.get(sector)
            if index_code:
                df = self._ak.index_stock_cons_csindex(symbol=index_code)
                if df is not None and not df.empty:
                    code_col = next((c for c in ['品种代码', '股票代码', 'code', '成分券代码'] if c in df.columns), None)
                    if code_col:
                        return [_convert_symbol_to_qmt(c) for c in df[code_col].astype(str).tolist()]
            if sector in ('沪深A股', 'A股', '全部A股'):
                df = self._ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    code_col = next((c for c in ['代码', '股票代码', 'code'] if c in df.columns), None)
                    if code_col:
                        return [_convert_symbol_to_qmt(c) for c in df[code_col].astype(str).tolist()]
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        except Exception as e:
            self.logger.warning(f"AKShare获取板块成分股失败: {e}")
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

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
        namespace = 'AKShareDataProcessor_Financial'
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"merged_{stocks_hash}_{start_time}_{end_time}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='financial')
            self.logger.info(f"从合并缓存(精确)加载财务数据: {len(merged_cached)} 只股票")
            return merged_cached
        request_set = set(sorted_stocks)
        ns_dir = cache_manager.disk_cache.get_namespace_dir(namespace)
        if ns_dir.exists():
            for cache_file in ns_dir.glob('merged_*.parquet'):
                if cache_file.name == f"{merged_cache_key}.parquet":
                    continue
                try:
                    candidate = pd.read_parquet(cache_file)
                    if candidate is not None and isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        if '_stock_code' in candidate.columns:
                            cached_stocks = set(candidate['_stock_code'].unique())
                            if request_set.issubset(cached_stocks):
                                candidate_dict = _parquet_to_merged_dict(candidate, mode='financial')
                                result_subset = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if len(result_subset) == len(stock_list):
                                    self.logger.info(f"从合并缓存(子集)加载财务数据: 缓存{len(cached_stocks)}只, 请求{len(stock_list)}只")
                                    return result_subset
                except Exception:
                    continue
        result = {}
        cache_hits = 0
        total = len(stock_list)
        for i, symbol in enumerate(stock_list, 1):
            ak_code = _convert_symbol_to_akshare_financial(symbol)
            symbol_data = {}
            all_cached = True
            for table in tables:
                cache_key = f"{symbol}_{table}"
                cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
                if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                    symbol_data[table] = cached
                else:
                    all_cached = False
            if all_cached and symbol_data:
                result[symbol] = symbol_data
                cache_hits += 1
                continue
            downloaded_tables, failed_tables = [], []
            for table in tables:
                if table in symbol_data:
                    continue
                try:
                    df = self._get_akshare_financial_data(ak_code, table)
                    if df is not None and not df.empty:
                        symbol_data[table] = df
                        cache_key = f"{symbol}_{table}"
                        cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
                        downloaded_tables.append(table)
                    else:
                        failed_tables.append(table)
                except Exception as e:
                    failed_tables.append(table)
            if symbol_data:
                result[symbol] = symbol_data
            if downloaded_tables and not failed_tables:
                self.logger.info(f"[ {i} / {total} ] {symbol} 财务数据已下载 ({len(downloaded_tables)}表)")
            elif downloaded_tables:
                self.logger.info(f"[ {i} / {total} ] {symbol} 部分下载 ({len(downloaded_tables)}/{len(tables)}表)")
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='financial')
            if merged_df is not None:
                cache_manager.disk_cache.put(namespace, merged_cache_key, merged_df, 'parquet')
        return result

    def _get_akshare_financial_data(self, ak_code: str, table: str) -> Optional[pd.DataFrame]:
        try:
            df = None
            if table == 'Balance':
                df = self._ak.stock_balance_sheet_by_report_em(symbol=ak_code)
            elif table == 'Income':
                df = self._ak.stock_profit_sheet_by_report_em(symbol=ak_code)
            elif table == 'CashFlow':
                df = self._ak.stock_cash_flow_sheet_by_report_em(symbol=ak_code)
            elif table == 'Pershareindex':
                df = self._get_pershareindex_data(ak_code)
            else:
                return None
            if df is None or df.empty:
                return None
            df = self._convert_akshare_columns(df, table)
            index_set = False
            if '公告日期' in df.columns:
                try:
                    dt_col = pd.to_datetime(df['公告日期'], errors='coerce')
                    df = df.dropna(subset=['公告日期'])
                    if not df.empty:
                        df.index = dt_col.loc[df.index]
                        df = df.sort_index()
                        index_set = True
                except Exception:
                    pass
            if not index_set and '报告期' in df.columns:
                try:
                    dt_col = pd.to_datetime(df['报告期'], errors='coerce')
                    df = df.dropna(subset=['报告期'])
                    if not df.empty:
                        df.index = dt_col.loc[df.index]
                        df = df.sort_index()
                except Exception:
                    pass
            return df
        except Exception as e:
            self.logger.debug(f"AKShare获取 {ak_code} {table} 数据失败: {e}")
            return None

    def _get_pershareindex_data(self, ak_code: str) -> Optional[pd.DataFrame]:
        import re
        from datetime import datetime
        code = re.sub(r'^[A-Za-z]+', '', ak_code)
        now = datetime.now()
        dates = []
        for year_offset in range(2):
            y = now.year - year_offset
            for period in ['1231', '0930', '0630', '0331']:
                date_str = f"{y}{period}"
                if date_str <= now.strftime('%Y%m%d'):
                    dates.append(date_str)
        all_dfs = []
        for date in dates[:4]:
            try:
                batch_df = self._ak.stock_yjbb_em(date=date)
                if batch_df is not None and not batch_df.empty:
                    code_col = next((c for c in ['股票代码', '代码', 'symbol', 'code'] if c in batch_df.columns), None)
                    if code_col:
                        filtered = batch_df[batch_df[code_col].astype(str) == code]
                        if not filtered.empty:
                            all_dfs.append(filtered)
                time.sleep(0.5)
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
            '每股收益': 'eps', '每股净资产': 'bps', '每股经营现金流量': 'ocfps',
            '每股资本公积金': 'capital_reserve_ps', '每股未分配利润': 'undistributed_profit_ps',
            '净资产收益率': 'roe', '净资产收益率-加权': 'roe_weighted',
            '净资产收益率-摊薄': 'roe_diluted', '销售毛利率': 'gross_profit_margin',
            '营业收入同比增长率': 'inc_operate_income_rate', '净利润同比增长率': 'inc_net_profit_rate',
        }
        return df.rename(columns=column_map)

    def get_industry_mapping(self, level: int = 1,
                               stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        if not self._ak:
            return {}
        try:
            df = self._ak.stock_board_industry_name_em()
            if df is None or df.empty:
                return {}
            industry_map = {}
            for _, row in df.iterrows():
                industry_name = row.get('板块名称', '')
                if not industry_name:
                    continue
                try:
                    stocks_df = self._ak.stock_board_industry_cons_em(symbol=industry_name)
                    if stocks_df is not None and not stocks_df.empty:
                        code_col = next((c for c in ['代码', '股票代码', 'code'] if c in stocks_df.columns), None)
                        if code_col:
                            for code in stocks_df[code_col].astype(str).tolist():
                                qmt_code = _convert_symbol_to_qmt(code)
                                if stock_pool is None or qmt_code in stock_pool:
                                    industry_map[qmt_code] = industry_name
                except Exception:
                    pass
            return industry_map
        except Exception as e:
            self.logger.warning(f"AKShare获取行业数据失败: {e}")
            return {}

    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        if not self._ak:
            return {}
        namespace = 'AKShareDataProcessor_Financial'
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
                ak_code = _convert_symbol_to_akshare(symbol)
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
