import os
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


class OpenDataProcessor(DataProcessor):
    """OpenData数据处理器

    基于腾讯财经数据源，支持获取A股历史行情数据，数据范围远超QMT的一年限制。
    股票代码统一使用QMT格式 (000001.SZ)，内部自动转换。

    注意：财务数据由 QMTDataProcessor 统一管理，OpenData 不再提供财务数据功能。
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

    def __init__(self, fallback_to_simulated: bool = True):
        try:
            import akshare as ak
            self._ak = ak
        except ImportError:
            self._ak = None
            logging.getLogger(__name__).warning("akshare not installed, pip install akshare")
        self._fallback_to_simulated = fallback_to_simulated
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self._raw_fetcher = OpenDataProcessor_Raw(self)

    @smart_cache(cache_type='market', incremental=True)
    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取行情数据（个股用腾讯财经，指数用akshare指数接口）"""
        if not self._ak:
            if self._fallback_to_simulated:
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("akshare 未安装，请 pip install akshare")

        # 判断是否为指数
        if symbol in self.INDEX_CODE_MAP or symbol.replace('.', '') in self.INDEX_CODE_MAP.values():
            df = self._get_index_data(symbol, start_date, end_date, period)
            if df is not None and not df.empty:
                return df
            if self._fallback_to_simulated:
                return self._generate_simulated_data(start_date, end_date, symbol)
            raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有数据")

        # 使用腾讯财经获取个股数据
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

    def _get_index_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """获取指数历史行情数据（优先腾讯财经，备用东方财富/新浪）"""
        if '.' in symbol:
            code, suffix = symbol.split('.')
            tx_symbol = f"{suffix.lower()}{code}"
            em_symbol = f"{suffix.lower()}{code}"
            sina_symbol = f"{suffix.lower()}{code}"
        else:
            tx_symbol = symbol
            em_symbol = symbol
            sina_symbol = symbol

        try:
            df = self._ak.stock_zh_index_daily_tx(symbol=tx_symbol)
            if df is not None and not df.empty:
                return self._process_akshare_data(df, symbol, start_date, end_date)
        except Exception as e:
            self.logger.debug(f"腾讯财经指数接口失败: {e}")

        try:
            em_start = start_date.replace('-', '')
            em_end = end_date.replace('-', '')
            df = self._ak.stock_zh_index_daily_em(symbol=em_symbol, start_date=em_start, end_date=em_end)
            if df is not None and not df.empty:
                return self._process_akshare_data(df, symbol, start_date, end_date)
        except Exception as e:
            self.logger.debug(f"东方财富指数接口失败: {e}")

        try:
            df = self._ak.stock_zh_index_daily(symbol=sina_symbol)
            if df is not None and not df.empty:
                return self._process_akshare_data(df, symbol, start_date, end_date)
        except Exception as e:
            self.logger.debug(f"新浪指数接口失败: {e}")

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

        try:
            result = self._fetch_stock_list(sector)

            if not result:
                result = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

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

        try:
            industry_map = self._get_industry_mapping_sw_all(level, stock_pool)
            if industry_map:
                return industry_map
        except Exception as e:
            self.logger.debug(f"申万行业分类获取失败: {e}")

        try:
            self.logger.info("尝试东方财富行业板块获取...")
            industry_map = self._get_industry_mapping_by_em_sector(stock_pool)
            if industry_map:
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
            self.logger.info(f"历史行业数据获取完成: date={date}, 共 {len(industry_map)} 只股票")
        else:
            self.logger.warning(f"未获取到任何历史行业数据: date={date}")

        return industry_map

    def get_dividend_data(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        if not self._ak:
            return {}
        namespace = 'OpenDataProcessor_Dividend'
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
        """生成模拟数据（fallback，仅当真实数据完全不可用时使用）"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        seed = hash(symbol) % 10000 if symbol else 42
        rng = np.random.default_rng(seed)

        # 根据symbol类型设置合理的基准价格
        if symbol and ('300' in symbol or '000016' in symbol):
            base_price = 4000.0  # 沪深300/上证50 约4000点
        elif symbol and ('905' in symbol or '852' in symbol):
            base_price = 6000.0  # 中证500/1000 约6000点
        else:
            base_price = 10.0 + rng.random() * 5.0

        prices = []
        for i in range(len(date_range)):
            base_price *= (1 + rng.normal(0, 0.015))
            prices.append(base_price)
        df = pd.DataFrame({
            'open': [p * 0.999 for p in prices], 'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices], 'close': prices,
            'volume': rng.integers(10000, 100000, len(date_range)),
        }, index=date_range)
        return self.preprocess_data(df)

    def get_raw_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取不复权行情数据，用于股息率等需要实际价格的计算

        与 get_data() 使用后复权数据不同，此方法获取不复权数据，
        独立缓存于 market_raw 命名空间，与后复权缓存互不干扰。
        """
        return self._raw_fetcher.get_data(symbol, start_date, end_date, period, **kwargs)


class OpenDataProcessor_Raw:
    """不复权行情数据获取器（OpenData 数据源）

    类名 OpenDataProcessor_Raw 会被 smart_cache 装饰器用作命名空间，
    映射到 .cache/OpenData/market_raw/ 目录，与后复权缓存完全隔离。
    """

    def __init__(self, processor: OpenDataProcessor):
        self._processor = processor

    @smart_cache(cache_type='market', incremental=True)
    def get_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d", **kwargs) -> pd.DataFrame:
        """获取不复权行情数据"""
        if not self._processor._ak:
            if self._processor._fallback_to_simulated:
                return self._processor._generate_simulated_data(start_date, end_date, symbol)
            raise RuntimeError("akshare 未安装，请 pip install akshare")

        df = self._get_raw_data_from_tx(symbol, start_date, end_date)
        if df is not None and not df.empty:
            return self._processor._process_akshare_data(df, symbol, start_date, end_date)

        if self._processor._fallback_to_simulated:
            return self._processor._generate_simulated_data(start_date, end_date, symbol)
        raise ValueError(f"{symbol} 在 {start_date} 到 {end_date} 期间没有不复权数据")

    def _get_raw_data_from_tx(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从腾讯财经获取不复权数据"""
        try:
            if '.' in symbol:
                code, suffix = symbol.split('.')
                tx_symbol = f"{suffix.lower()}{code}"
            else:
                tx_symbol = symbol

            df = self._processor._ak.stock_zh_a_hist_tx(symbol=tx_symbol, start_date=start_date, end_date=end_date, adjust='')
            if df is not None and not df.empty:
                return df
        except Exception as e:
            self._processor.logger.debug(f"腾讯财经不复权接口失败: {e}")
        return None


_QVIX_FUNCS = {
    '510500.SH': 'index_option_500etf_qvix',
    '510050.SH': 'index_option_50etf_qvix',
    '510300.SH': 'index_option_300etf_qvix',
    '510880.SH': 'index_option_50etf_qvix',
    '000852.XSHG': 'index_option_1000index_qvix',
}

_QVIX_CACHE_MAX_AGE = 4 * 3600


def _get_qvix_cache_dir() -> str:
    base = os.environ.get('QMT_CACHE_DIR', os.path.join(os.getcwd(), '.cache'))
    cache_dir = os.path.join(base, 'OpenData', 'vix')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_qvix_cache_path(symbol: str) -> str:
    return os.path.join(_get_qvix_cache_dir(), f'QVIX_{symbol}.parquet')


def _is_qvix_cache_valid(cache_path: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    mtime = os.path.getmtime(cache_path)
    return (time.time() - mtime) < _QVIX_CACHE_MAX_AGE


def _load_qvix_from_cache(symbol: str) -> Optional[pd.DataFrame]:
    cache_path = _get_qvix_cache_path(symbol)
    if not _is_qvix_cache_valid(cache_path):
        return None
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        logging.getLogger(__name__).info(f"QVIX缓存命中: {cache_path}, 数据{len(df)}行")
        return df
    except Exception as e:
        logging.getLogger(__name__).warning(f"QVIX缓存读取失败: {cache_path}, {e}")
        return None


def _save_qvix_to_cache(symbol: str, df: pd.DataFrame):
    cache_path = _get_qvix_cache_path(symbol)
    try:
        save_df = df.copy()
        save_df['date'] = save_df['date'].astype(str)
        save_df.to_parquet(cache_path, index=False)
        logging.getLogger(__name__).info(f"QVIX数据已缓存: {cache_path}, {len(df)}行")
    except Exception as e:
        logging.getLogger(__name__).warning(f"QVIX缓存写入失败: {cache_path}, {e}")


def get_qvix_data(symbol: str = '510500.SH') -> Optional[pd.DataFrame]:
    """获取 QVIX 隐含波动率数据

    数据来源：期权论坛 (http://1.optbbs.com/s/vix.shtml)，通过 akshare 接口获取。
    QVIX 是基于期权价格计算的隐含波动率指数，收盘后产生。

    缓存策略：
    - 缓存路径: .cache/OpenData/vix/QVIX_{symbol}.parquet
    - 缓存有效期: 4小时
    - Parquet 格式: 列 [date, vix], date 为日期字符串, vix 为 float64

    支持的标的：
    - 510500.SH → 中证500ETF 期权 QVIX
    - 510050.SH → 上证50ETF 期权 QVIX
    - 510300.SH → 沪深300ETF 期权 QVIX
    - 510880.SH → 上证50ETF 期权 QVIX
    - 000852.XSHG → 中证1000指数期权 QVIX

    Args:
        symbol: 标的代码

    Returns:
        包含 date 和 vix 列的 DataFrame，获取失败返回 None
    """
    cached = _load_qvix_from_cache(symbol)
    if cached is not None:
        return cached

    try:
        import akshare as ak
    except ImportError:
        logging.getLogger(__name__).error("akshare 未安装，无法获取 QVIX 数据")
        return None

    func_name = _QVIX_FUNCS.get(symbol)
    if func_name is None:
        func_name = 'index_option_500etf_qvix'

    try:
        func = getattr(ak, func_name)
        qvix_df = func()

        qvix_df = qvix_df.copy()
        qvix_df['date'] = pd.to_datetime(qvix_df['date'], errors='coerce')
        qvix_df['close'] = pd.to_numeric(qvix_df['close'], errors='coerce')
        qvix_df = qvix_df.dropna(subset=['date', 'close'])
        qvix_df = qvix_df.sort_values('date').reset_index(drop=True)

        if qvix_df.empty:
            logging.getLogger(__name__).warning(f"akshare QVIX 数据为空: {func_name}")
            return None

        result = pd.DataFrame({
            'date': qvix_df['date'],
            'vix': qvix_df['close']
        })

        _save_qvix_to_cache(symbol, result)

        logging.getLogger(__name__).info(
            f"QVIX 数据获取成功: {func_name}, "
            f"有效数据 {len(result)} 行, "
            f"日期范围 {result['date'].min().strftime('%Y-%m-%d')} ~ "
            f"{result['date'].max().strftime('%Y-%m-%d')}"
        )
        return result

    except Exception as e:
        logging.getLogger(__name__).error(f"QVIX 数据获取失败: {func_name}, 错误: {e}")
        return None
