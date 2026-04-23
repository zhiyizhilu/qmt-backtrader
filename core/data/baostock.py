import pandas as pd
import numpy as np
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any

from core.cache import smart_cache, cache_manager
from core.data.base import DataProcessor, _merged_dict_to_parquet, _parquet_to_merged_dict
from core.data.akshare import _convert_symbol_to_qmt, _convert_symbol_to_baostock


class BaoStockDataProcessor(DataProcessor):
    """BaoStock数据处理器

    免费开源证券数据平台，支持1990年至今的A股历史数据。
    股票代码统一使用QMT格式 (000001.SZ)，内部自动转换。
    """

    PERIOD_MAP = {
        '1d': 'd', 'day': 'd', 'daily': 'd',
        '1w': 'w', 'week': 'w', 'weekly': 'w',
        '1M': 'm', 'month': 'm', 'monthly': 'm',
        '5m': '5', '5min': '5', '15m': '15', '15min': '15',
        '30m': '30', '30min': '30', '60m': '60', '60min': '60',
    }

    def __init__(self, fallback_to_simulated: bool = True):
        self._bs = None
        self._logged_in = False
        try:
            import baostock as bs
            self._bs = bs
        except ImportError:
            logging.getLogger(__name__).warning("baostock not installed, pip install baostock")
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def _ensure_login(self):
        if not self._bs:
            return False
        if not self._logged_in:
            return self._do_login()
        return True

    def _do_login(self):
        try:
            try:
                self._bs.logout()
            except Exception:
                pass
            lg = self._bs.login()
            if lg.error_code == '0':
                self._logged_in = True
                return True
            else:
                self.logger.error(f"baostock登录失败: {lg.error_msg}")
                self._logged_in = False
                return False
        except Exception as e:
            self.logger.error(f"baostock登录异常: {e}")
            self._logged_in = False
            return False

    def _reconnect(self):
        self._logged_in = False
        return self._do_login()

    def __del__(self):
        if self._bs and self._logged_in:
            try:
                self._bs.logout()
                self._logged_in = False
            except Exception:
                pass

    @smart_cache(cache_type='market', incremental=True)
    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        if not self._bs:
            if self._fallback_to_simulated:
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("baostock 未安装，请 pip install baostock")
        if not self._ensure_login():
            if self._fallback_to_simulated:
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("baostock 登录失败")
        try:
            bs_code = _convert_symbol_to_baostock(symbol)
            bs_freq = self.PERIOD_MAP.get(period, 'd')
            if bs_freq in ('d', 'w', 'm'):
                fields = "date,open,high,low,close,volume,amount,turn,pctChg"
            else:
                fields = "date,time,open,high,low,close,volume,amount"
            max_retries = 2
            rs = None
            for retry in range(max_retries + 1):
                rs = self._bs.query_history_k_data_plus(
                    code=bs_code, fields=fields, start_date=start_date,
                    end_date=end_date, frequency=bs_freq, adjustflag="2")
                if rs.error_code == '0':
                    break
                if '未登录' in str(rs.error_msg) or 'login' in str(rs.error_msg).lower():
                    if retry < max_retries:
                        time.sleep(1)
                        if self._reconnect():
                            continue
                break
            if rs.error_code != '0':
                if self._fallback_to_simulated:
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise RuntimeError(f"BaoStock查询失败: {rs.error_msg}")
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                if self._fallback_to_simulated:
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")
            df = pd.DataFrame(data_list, columns=rs.fields)
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'date' in df.columns:
                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(
                        df['date'] + ' ' + df['time'].str[:6].apply(
                            lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}" if len(x) >= 6 else x),
                        errors='coerce')
                else:
                    df['datetime'] = pd.to_datetime(df['date'])
                df = df.set_index('datetime')
            df = df.dropna(subset=['close'])
            if df.empty:
                if self._fallback_to_simulated:
                    return self._generate_simulated_data(start_date, end_date, symbol)
                raise ValueError(f"{symbol} 数据清洗后为空")
            keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[keep_cols]
            time.sleep(0.1)
            return self.preprocess_data(df)
        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            if self._fallback_to_simulated:
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError(f"BaoStock获取数据失败: {e}")

    def get_stock_list(self, sector: str = '沪深A股') -> List[str]:
        if not self._bs:
            return self._get_stock_list_from_cache(sector) or ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        if not self._ensure_login():
            return self._get_stock_list_from_cache(sector) or ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        try:
            sector_map = {'沪深300': 'sh.000300', '上证50': 'sh.000016',
                          '中证500': 'sh.000905', '中证1000': 'sh.000852'}
            bs_index_code = sector_map.get(sector)
            if bs_index_code:
                rs = None
                if sector == '沪深300':
                    rs = self._bs.query_hs300_stocks()
                elif sector == '上证50':
                    rs = self._bs.query_sz50_stocks()
                elif sector == '中证500':
                    rs = self._bs.query_zz500_stocks()
                if rs and '未登录' in str(rs.error_msg):
                    if self._reconnect():
                        if sector == '沪深300':
                            rs = self._bs.query_hs300_stocks()
                        elif sector == '上证50':
                            rs = self._bs.query_sz50_stocks()
                        elif sector == '中证500':
                            rs = self._bs.query_zz500_stocks()
                if rs and rs.error_code == '0':
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        code_col = 'code' if 'code' in df.columns else rs.fields[0]
                        return [_convert_symbol_to_qmt(c) for c in df[code_col].tolist()]
            if sector in ('沪深A股', 'A股', '全部A股'):
                rs = self._bs.query_stock_basic()
                if rs and rs.error_code == '0':
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        code_col = 'code' if 'code' in df.columns else rs.fields[0]
                        return [_convert_symbol_to_qmt(c) for c in df[code_col].tolist()]
            return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        except Exception as e:
            self.logger.warning(f"BaoStock获取板块成分股失败: {e}")
            return self._get_stock_list_from_cache(sector) or ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

    def _get_stock_list_from_cache(self, sector: str = '沪深A股') -> Optional[List[str]]:
        namespace = 'BaoStockDataProcessor_Financial'
        for try_len in [300, 500, 50, 800]:
            for mode in ['dividend', 'financial']:
                merged_cache_key = f"{mode}_merged_{try_len}"
                merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
                if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
                    try:
                        merged_dict = _parquet_to_merged_dict(merged_cached, mode=mode)
                        if merged_dict:
                            return list(merged_dict.keys())
                    except Exception:
                        continue
        import glob, os
        cache_dir = os.path.join('.cache', namespace)
        if os.path.exists(cache_dir):
            stocks = set()
            for f in glob.glob(os.path.join(cache_dir, '*_dividend.parquet')):
                stock = os.path.basename(f).replace('_dividend.parquet', '')
                stocks.add(stock)
            if stocks:
                return sorted(stocks)
        return None

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

    def download_financial_data(self, stock_list: List[str],
                                table_list: Optional[List[str]] = None,
                                start_time: str = '', end_time: str = '') -> None:
        pass  # BaoStock财务数据为实时查询，无需预下载

    def _get_financial_data_single(self, stock: str, yq_list: List[tuple]) -> Optional[pd.DataFrame]:
        bs_code = _convert_symbol_to_baostock(stock)
        return self._fetch_baostock_financial_data(bs_code, yq_list)

    def get_financial_data(self, stock_list: List[str],
                           table_list: Optional[List[str]] = None,
                           start_time: str = '', end_time: str = '',
                           report_type: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        namespace = 'BaoStockDataProcessor_Financial'
        can_query = self._bs is not None and self._ensure_login()
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"merged_{stocks_hash}_{start_time}_{end_time}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='financial')
            filtered = {k: v for k, v in merged_cached.items() if k in stock_list}
            if filtered:
                if len(filtered) >= len(stock_list):
                    return filtered
                for stock in stock_list:
                    if stock not in filtered:
                        filtered[stock] = {}
                return filtered
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
                                filtered = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if filtered:
                                    if len(filtered) >= len(stock_list):
                                        return filtered
                                    for stock in stock_list:
                                        if stock not in filtered:
                                            filtered[stock] = {}
                                    return filtered
                except Exception:
                    continue
        yq_list = self._get_year_quarter_range(start_time, end_time)
        if not yq_list:
            return {stock: {} for stock in stock_list}
        if not can_query:
            result = {}
            yq_key = f"{yq_list[0][0]}Q{yq_list[0][1]}_{yq_list[-1][0]}Q{yq_list[-1][1]}" if yq_list else "all"
            for stock in stock_list:
                cache_key = f"{stock}_{yq_key}"
                cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
                if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                    result[stock] = {'Pershareindex': cached}
            return result
        result = {}
        cache_hits = 0
        yq_key = f"{yq_list[0][0]}Q{yq_list[0][1]}_{yq_list[-1][0]}Q{yq_list[-1][1]}" if yq_list else "all"
        total = len(stock_list)
        for i, stock in enumerate(stock_list, 1):
            cache_key = f"{stock}_{yq_key}"
            cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
            if cached is not None and not (isinstance(cached, pd.DataFrame) and cached.empty):
                result[stock] = {'Pershareindex': cached}
                cache_hits += 1
            else:
                df = self._get_financial_data_single(stock, yq_list)
                if df is not None and not df.empty:
                    result[stock] = {'Pershareindex': df}
                    cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='financial')
            if merged_df is not None:
                save_cache_key = f"merged_{stocks_hash}_{start_time}_{end_time}"
                cache_manager.disk_cache.put(namespace, save_cache_key, merged_df, 'parquet')
        return result

    def _get_year_quarter_range(self, start_time: str, end_time: str) -> List[tuple]:
        try:
            if start_time:
                start_dt = pd.to_datetime(start_time, errors='coerce')
            else:
                start_dt = pd.Timestamp.now() - pd.DateOffset(years=5)
            if end_time:
                end_dt = pd.to_datetime(end_time, errors='coerce')
            else:
                end_dt = pd.Timestamp.now()
        except Exception:
            start_dt = pd.Timestamp.now() - pd.DateOffset(years=5)
            end_dt = pd.Timestamp.now()
        if pd.isna(start_dt):
            start_dt = pd.Timestamp.now() - pd.DateOffset(years=5)
        if pd.isna(end_dt):
            end_dt = pd.Timestamp.now()
        start_year, start_quarter = start_dt.year, (start_dt.month - 1) // 3 + 1
        end_year, end_quarter = end_dt.year, (end_dt.month - 1) // 3 + 1
        yq_list = []
        for y in range(start_year, end_year + 1):
            for q in range(1, 5):
                if (y == start_year and q < start_quarter) or (y == end_year and q > end_quarter):
                    continue
                yq_list.append((y, q))
        return yq_list

    def _fetch_baostock_financial_data(self, bs_code: str, yq_list: List[tuple]) -> Optional[pd.DataFrame]:
        records_by_date: Dict[str, Dict[str, Any]] = {}
        query_funcs = [
            (self._bs.query_dupont_data, 'dupont'),
            (self._bs.query_growth_data, 'growth'),
            (self._bs.query_profit_data, 'profit'),
            (self._bs.query_cash_flow_data, 'cashflow'),
        ]
        for year, quarter in yq_list:
            for query_func, source in query_funcs:
                try:
                    rs = query_func(bs_code, year=year, quarter=quarter)
                    if rs.error_code != '0':
                        continue
                    while rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        stat_date = row.get('statDate')
                        if not stat_date:
                            continue
                        if stat_date not in records_by_date:
                            records_by_date[stat_date] = {}
                        records_by_date[stat_date].update(row)
                except Exception:
                    continue
        if not records_by_date:
            return None
        df = pd.DataFrame.from_dict(records_by_date, orient='index')
        column_map = {'dupontROE': 'du_return_on_equity', 'YOYNI': 'inc_net_profit_rate'}
        df = df.rename(columns=column_map)
        for col in df.columns:
            if col not in ('code', 'pubDate', 'statDate'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'NCIOperAct' in df.columns and 'totalShare' in df.columns:
            df['s_fa_ocfps'] = df['NCIOperAct'] / df['totalShare']
        date_col = 'pubDate' if 'pubDate' in df.columns else 'statDate'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            df = df.sort_index()
        else:
            return None
        return df

    def get_industry_mapping(self, level: int = 1,
                             stock_pool: Optional[List[str]] = None) -> Dict[str, str]:
        if not self._bs or not self._ensure_login():
            return {}
        try:
            rs = self._bs.query_stock_industry()
            if rs.error_code != '0':
                return {}
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return {}
            df = pd.DataFrame(data_list, columns=rs.fields)
            mapping = {}
            code_col = 'code' if 'code' in df.columns else rs.fields[0]
            industry_col = 'industry' if 'industry' in df.columns else None
            if not industry_col:
                return {}
            pool_set = set(stock_pool) if stock_pool else None
            for _, row in df.iterrows():
                bs_code = row.get(code_col)
                industry = row.get(industry_col)
                if not bs_code or not industry:
                    continue
                qmt_code = _convert_symbol_to_qmt(bs_code)
                if pool_set is not None and qmt_code not in pool_set:
                    continue
                mapping[qmt_code] = str(industry)
            return mapping
        except Exception as e:
            self.logger.warning(f"BaoStock获取行业映射失败: {e}")
            return {}

    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        namespace = 'BaoStockDataProcessor_Financial'
        can_query = self._bs is not None and self._ensure_login()
        sorted_stocks = sorted(stock_list)
        stocks_hash = hashlib.md5(','.join(sorted_stocks).encode()).hexdigest()[:12]
        merged_cache_key = f"dividend_merged_{stocks_hash}"
        merged_cached = cache_manager.disk_cache.get(namespace, merged_cache_key, 'parquet')
        if merged_cached is not None and isinstance(merged_cached, pd.DataFrame) and not merged_cached.empty:
            merged_cached = _parquet_to_merged_dict(merged_cached, mode='dividend')
            filtered = {k: v for k, v in merged_cached.items() if k in stock_list}
            if filtered:
                if len(filtered) >= len(stock_list):
                    return filtered
                if not can_query:
                    return filtered
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
                                filtered = {s: candidate_dict[s] for s in stock_list if s in candidate_dict}
                                if filtered:
                                    if len(filtered) >= len(stock_list):
                                        return filtered
                                    if not can_query:
                                        return filtered
                                    break
                except Exception:
                    continue
        result = {}
        current_year = pd.Timestamp.now().year
        total = len(stock_list)
        if not can_query:
            for stock in stock_list:
                cached = cache_manager.disk_cache.get(namespace, f"{stock}_dividend", 'parquet')
                if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                    result[stock] = cached
            return result
        for i, stock in enumerate(stock_list, 1):
            cache_key = f"{stock}_dividend"
            cached = cache_manager.disk_cache.get(namespace, cache_key, 'parquet')
            if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
                result[stock] = cached
                continue
            bs_code = _convert_symbol_to_baostock(stock)
            rs_list = []
            for year in range(current_year - 9, current_year + 1):
                try:
                    rs = self._bs.query_dividend_data(code=bs_code, year=str(year), yearType='report')
                    if rs.error_code == '0':
                        while rs.next():
                            rs_list.append(rs.get_row_data())
                except Exception:
                    continue
            if not rs_list:
                continue
            try:
                df = pd.DataFrame(rs_list, columns=rs.fields)
                if 'dividCashPsBeforeTax' in df.columns:
                    df['interest'] = pd.to_numeric(df['dividCashPsBeforeTax'], errors='coerce')
                date_col = 'dividOperateDate' if 'dividOperateDate' in df.columns else 'dividPayDate'
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col])
                    df = df.set_index(date_col)
                    df = df.sort_index()
                result[stock] = df
                cache_manager.disk_cache.put(namespace, cache_key, df, 'parquet')
            except Exception:
                pass
        if result:
            merged_df = _merged_dict_to_parquet(result, mode='dividend')
            if merged_df is not None:
                save_cache_key = f"dividend_merged_{stocks_hash}"
                cache_manager.disk_cache.put(namespace, save_cache_key, merged_df, 'parquet')
        return result
