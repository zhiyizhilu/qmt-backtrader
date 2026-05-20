import re
import json
import threading
import logging
from typing import Dict, Optional, List
from pathlib import Path
import pandas as pd


class CacheIndexManager:
    """缓存索引管理器 - 维护行情/财报数据的年份索引

    索引结构:
        market_index: { symbol: { period: { years: [2020, 2021, ...], last_update: "..." } } }
        financial_index: { symbol: { table: { years: [2020, 2021, ...], last_update: "..." } } }
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._market_index: Dict[str, Dict[str, Dict]] = {}
        self._market_raw_index: Dict[str, Dict[str, Dict]] = {}
        self._financial_index: Dict[str, Dict[str, Dict]] = {}
        self._dividend_checked: Dict[str, str] = {}
        self._financial_nodata: Dict[str, Dict[str, str]] = {}
        self._market_raw_nodata: Dict[str, str] = {}
        self._delisted_stocks: Dict[str, str] = {}
        self._suspended_ranges: Dict[str, List[List[str]]] = {}
        self._dirty_flags = {
            'market': False,
            'market_raw': False,
            'financial': False,
            'dividend': False,
            'financial_nodata': False,
            'market_raw_nodata': False,
            'delisted': False,
            'suspended': False,
        }
        self.load_index()

    @property
    def market_index_path(self) -> Path:
        return self.cache_dir / 'index' / 'market_index.json'

    @property
    def market_raw_index_path(self) -> Path:
        return self.cache_dir / 'index' / 'market_raw_index.json'

    @property
    def financial_index_path(self) -> Path:
        return self.cache_dir / 'index' / 'financial_index.json'

    @property
    def dividend_checked_path(self) -> Path:
        return self.cache_dir / 'index' / 'dividend_checked.json'

    @property
    def financial_nodata_path(self) -> Path:
        return self.cache_dir / 'index' / 'financial_nodata.json'

    @property
    def market_raw_nodata_path(self) -> Path:
        return self.cache_dir / 'index' / 'market_raw_nodata.json'

    @property
    def delisted_stocks_path(self) -> Path:
        return self.cache_dir / 'index' / 'delisted_stocks.json'

    @property
    def suspended_ranges_path(self) -> Path:
        return self.cache_dir / 'index' / 'suspended_ranges.json'

    def load_index(self) -> None:
        for path, attr in [
            (self.market_index_path, '_market_index'),
            (self.market_raw_index_path, '_market_raw_index'),
            (self.financial_index_path, '_financial_index'),
            (self.dividend_checked_path, '_dividend_checked'),
            (self.financial_nodata_path, '_financial_nodata'),
            (self.market_raw_nodata_path, '_market_raw_nodata'),
            (self.delisted_stocks_path, '_delisted_stocks'),
            (self.suspended_ranges_path, '_suspended_ranges'),
        ]:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        setattr(self, attr, json.load(f))
                except Exception as e:
                    self.logger.warning(f"加载索引文件失败: {path}, 错误: {e}")
                    setattr(self, attr, {} if attr != '_dividend_checked' else {})

    def save_index(self) -> None:
        if not any(self._dirty_flags.values()):
            return
        with self.lock:
            files_to_save = []
            if self._dirty_flags['market']:
                files_to_save.append((self.market_index_path, self._market_index))
            if self._dirty_flags['market_raw']:
                files_to_save.append((self.market_raw_index_path, self._market_raw_index))
            if self._dirty_flags['financial']:
                files_to_save.append((self.financial_index_path, self._financial_index))
            if self._dirty_flags['dividend']:
                files_to_save.append((self.dividend_checked_path, self._dividend_checked))
            if self._dirty_flags['financial_nodata']:
                files_to_save.append((self.financial_nodata_path, self._financial_nodata))
            if self._dirty_flags['market_raw_nodata']:
                files_to_save.append((self.market_raw_nodata_path, self._market_raw_nodata))
            if self._dirty_flags['delisted']:
                files_to_save.append((self.delisted_stocks_path, self._delisted_stocks))
            if self._dirty_flags['suspended']:
                files_to_save.append((self.suspended_ranges_path, self._suspended_ranges))

            for path, data in files_to_save:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path = path.with_suffix('.json.tmp')
                        with open(tmp_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        tmp_path.replace(path)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            import time as _time
                            _time.sleep(0.1 * (attempt + 1))
                        else:
                            self.logger.warning(f"保存索引文件失败: {path}, 错误: {e}")

            for key in self._dirty_flags:
                self._dirty_flags[key] = False

    def update_market_index(self, symbol: str, period: str, year: int) -> None:
        with self.lock:
            if symbol not in self._market_index:
                self._market_index[symbol] = {}
            if period not in self._market_index[symbol]:
                self._market_index[symbol][period] = {'years': [], 'last_update': ''}
            years = self._market_index[symbol][period]['years']
            if year not in years:
                years.append(year)
                years.sort()
            self._market_index[symbol][period]['last_update'] = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            self._dirty_flags['market'] = True

    def get_available_market_years(self, symbol: str, period: str) -> List[int]:
        with self.lock:
            entry = self._market_index.get(symbol, {}).get(period, {})
            return list(entry.get('years', []))

    def update_checked_market_years(self, symbol: str, period: str, years: List[int]) -> None:
        with self.lock:
            if symbol not in self._market_index:
                self._market_index[symbol] = {}
            if period not in self._market_index[symbol]:
                self._market_index[symbol][period] = {'years': [], 'checked_years': {}, 'last_update': ''}
            if 'checked_years' not in self._market_index[symbol][period]:
                self._market_index[symbol][period]['checked_years'] = {}
            now = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            for y in years:
                self._market_index[symbol][period]['checked_years'][str(y)] = now
            self._market_index[symbol][period]['last_update'] = now
            self._dirty_flags['market'] = True

    def remove_market_index(self, symbol: str, period: str) -> None:
        with self.lock:
            if symbol in self._market_index and period in self._market_index[symbol]:
                del self._market_index[symbol][period]
                if not self._market_index[symbol]:
                    del self._market_index[symbol]
                self._dirty_flags['market'] = True

    def update_earliest_data_date(self, symbol: str, period: str, date: str) -> None:
        """记录股票的最早可用数据日期（通常是上市日期）

        一旦记录，缓存覆盖检查将永久跳过该日期之前的年份，
        不再因 checked_years 过期而重复尝试获取不存在的数据。
        只在已知日期比现有记录更早时更新。
        """
        with self.lock:
            if symbol not in self._market_index:
                self._market_index[symbol] = {}
            if period not in self._market_index[symbol]:
                self._market_index[symbol][period] = {'years': [], 'last_update': ''}
            existing = self._market_index[symbol][period].get('earliest_data_date')
            if existing and existing <= date:
                return  # 已有更早的记录，不更新
            self._market_index[symbol][period]['earliest_data_date'] = date
            self._dirty_flags['market'] = True

    def get_earliest_data_date(self, symbol: str, period: str) -> Optional[str]:
        """获取股票的最早可用数据日期"""
        with self.lock:
            entry = self._market_index.get(symbol, {}).get(period, {})
            return entry.get('earliest_data_date')

    def update_latest_data_date(self, symbol: str, period: str, date: str) -> None:
        """记录股票的最晚可用数据日期（通常是退市日期）

        一旦记录，缓存覆盖检查将永久跳过该日期之后的年份，
        不再因 checked_years 过期而重复尝试获取不存在的数据。
        只在已知日期比现有记录更晚时更新。
        """
        with self.lock:
            if symbol not in self._market_index:
                self._market_index[symbol] = {}
            if period not in self._market_index[symbol]:
                self._market_index[symbol][period] = {'years': [], 'last_update': ''}
            existing = self._market_index[symbol][period].get('latest_data_date')
            if existing and existing >= date:
                return  # 已有更晚的记录，不更新
            self._market_index[symbol][period]['latest_data_date'] = date
            self._dirty_flags['market'] = True

    def get_latest_data_date(self, symbol: str, period: str) -> Optional[str]:
        """获取股票的最晚可用数据日期"""
        with self.lock:
            entry = self._market_index.get(symbol, {}).get(period, {})
            return entry.get('latest_data_date')

    def get_checked_market_years(self, symbol: str, period: str, max_age_days: int = 30) -> List[int]:
        with self.lock:
            entry = self._market_index.get(symbol, {}).get(period, {})
            checked = entry.get('checked_years', {})
            if not checked:
                return []
            now = pd.Timestamp.now()
            current_year = now.year
            valid_years = []
            for year_str, check_time in checked.items():
                try:
                    year = int(year_str)
                    if year < current_year:
                        valid_years.append(year)
                    else:
                        check_dt = pd.Timestamp(check_time)
                        if (now - check_dt).days < max_age_days:
                            valid_years.append(year)
                except Exception:
                    continue
            return valid_years

    def update_market_raw_index(self, symbol: str, period: str, year: int) -> None:
        with self.lock:
            if symbol not in self._market_raw_index:
                self._market_raw_index[symbol] = {}
            if period not in self._market_raw_index[symbol]:
                self._market_raw_index[symbol][period] = {'years': [], 'last_update': ''}
            years = self._market_raw_index[symbol][period]['years']
            if year not in years:
                years.append(year)
                years.sort()
            self._market_raw_index[symbol][period]['last_update'] = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            self._dirty_flags['market_raw'] = True

    def get_available_market_raw_years(self, symbol: str, period: str) -> List[int]:
        with self.lock:
            entry = self._market_raw_index.get(symbol, {}).get(period, {})
            return list(entry.get('years', []))

    def update_earliest_raw_data_date(self, symbol: str, period: str, date: str) -> None:
        """记录不复权数据的最早可用数据日期（通常是上市日期）"""
        with self.lock:
            if symbol not in self._market_raw_index:
                self._market_raw_index[symbol] = {}
            if period not in self._market_raw_index[symbol]:
                self._market_raw_index[symbol][period] = {'years': [], 'last_update': ''}
            existing = self._market_raw_index[symbol][period].get('earliest_data_date')
            if existing and existing <= date:
                return
            self._market_raw_index[symbol][period]['earliest_data_date'] = date
            self._dirty_flags['market_raw'] = True

    def get_earliest_raw_data_date(self, symbol: str, period: str) -> Optional[str]:
        """获取不复权数据的最早可用数据日期"""
        with self.lock:
            entry = self._market_raw_index.get(symbol, {}).get(period, {})
            return entry.get('earliest_data_date')

    def update_latest_raw_data_date(self, symbol: str, period: str, date: str) -> None:
        """记录不复权数据的最晚可用数据日期（通常是退市日期）"""
        with self.lock:
            if symbol not in self._market_raw_index:
                self._market_raw_index[symbol] = {}
            if period not in self._market_raw_index[symbol]:
                self._market_raw_index[symbol][period] = {'years': [], 'last_update': ''}
            existing = self._market_raw_index[symbol][period].get('latest_data_date')
            if existing and existing >= date:
                return
            self._market_raw_index[symbol][period]['latest_data_date'] = date
            self._dirty_flags['market_raw'] = True

    def get_latest_raw_data_date(self, symbol: str, period: str) -> Optional[str]:
        """获取不复权数据的最晚可用数据日期"""
        with self.lock:
            entry = self._market_raw_index.get(symbol, {}).get(period, {})
            return entry.get('latest_data_date')

    def update_checked_market_raw_years(self, symbol: str, period: str, years: List[int]) -> None:
        with self.lock:
            if symbol not in self._market_raw_index:
                self._market_raw_index[symbol] = {}
            if period not in self._market_raw_index[symbol]:
                self._market_raw_index[symbol][period] = {'years': [], 'checked_years': {}, 'last_update': ''}
            if 'checked_years' not in self._market_raw_index[symbol][period]:
                self._market_raw_index[symbol][period]['checked_years'] = {}
            now = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            for y in years:
                self._market_raw_index[symbol][period]['checked_years'][str(y)] = now
            self._market_raw_index[symbol][period]['last_update'] = now
            self._dirty_flags['market_raw'] = True

    def get_checked_market_raw_years(self, symbol: str, period: str, max_age_days: int = 30) -> List[int]:
        with self.lock:
            entry = self._market_raw_index.get(symbol, {}).get(period, {})
            checked = entry.get('checked_years', {})
            if not checked:
                return []
            now = pd.Timestamp.now()
            current_year = now.year
            valid_years = []
            for year_str, check_time in checked.items():
                try:
                    year = int(year_str)
                    if year < current_year:
                        valid_years.append(year)
                    else:
                        check_dt = pd.Timestamp(check_time)
                        if (now - check_dt).days < max_age_days:
                            valid_years.append(year)
                except Exception:
                    continue
            return valid_years

    def update_financial_index(self, symbol: str, table: str, year: int) -> None:
        with self.lock:
            if symbol not in self._financial_index:
                self._financial_index[symbol] = {}
            if table not in self._financial_index[symbol]:
                self._financial_index[symbol][table] = {'years': [], 'last_update': ''}
            years = self._financial_index[symbol][table]['years']
            if year not in years:
                years.append(year)
                years.sort()
            self._financial_index[symbol][table]['last_update'] = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            self._dirty_flags['financial'] = True

    def get_available_financial_years(self, symbol: str, table: str) -> List[int]:
        with self.lock:
            entry = self._financial_index.get(symbol, {}).get(table, {})
            return list(entry.get('years', []))

    def update_checked_financial_years(self, symbol: str, table: str, years: List[int]) -> None:
        with self.lock:
            if symbol not in self._financial_index:
                self._financial_index[symbol] = {}
            if table not in self._financial_index[symbol]:
                self._financial_index[symbol][table] = {'years': [], 'checked_years': {}, 'last_update': ''}
            if 'checked_years' not in self._financial_index[symbol][table]:
                self._financial_index[symbol][table]['checked_years'] = {}
            now = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            for y in years:
                self._financial_index[symbol][table]['checked_years'][str(y)] = now
            self._financial_index[symbol][table]['last_update'] = now
            self._dirty_flags['financial'] = True

    def get_checked_financial_years(self, symbol: str, table: str, max_age_days: int = 7) -> List[int]:
        with self.lock:
            entry = self._financial_index.get(symbol, {}).get(table, {})
            checked = entry.get('checked_years', {})
            if not checked:
                return []
            now = pd.Timestamp.now()
            valid_years = []
            for year_str, check_time in checked.items():
                try:
                    check_dt = pd.Timestamp(check_time)
                    if (now - check_dt).days < max_age_days:
                        valid_years.append(int(year_str))
                except Exception:
                    continue
            return valid_years

    def update_checked_dividend_stocks(self, stocks: List[str]) -> None:
        with self.lock:
            now = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            for stock in stocks:
                self._dividend_checked[stock] = now
            self._dirty_flags['dividend'] = True

    def mark_financial_nodata(self, symbol: str, table: str) -> None:
        with self.lock:
            if symbol not in self._financial_nodata:
                self._financial_nodata[symbol] = {}
            self._financial_nodata[symbol][table] = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            self._dirty_flags['financial_nodata'] = True

    def is_financial_nodata(self, symbol: str, table: str, max_age_days: int = 7) -> bool:
        with self.lock:
            entry = self._financial_nodata.get(symbol, {})
            check_time = entry.get(table)
            if not check_time:
                return False
            try:
                check_dt = pd.Timestamp(check_time)
                return (pd.Timestamp.now() - check_dt).days < max_age_days
            except Exception:
                return False

    def mark_market_raw_nodata(self, symbol: str, period: str = '1d') -> None:
        with self.lock:
            key = f"{symbol}_{period}"
            self._market_raw_nodata[key] = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S')
            self._dirty_flags['market_raw_nodata'] = True

    def is_market_raw_nodata(self, symbol: str, period: str = '1d', max_age_days: int = 7) -> bool:
        with self.lock:
            key = f"{symbol}_{period}"
            check_time = self._market_raw_nodata.get(key)
            if not check_time:
                return False
            try:
                check_dt = pd.Timestamp(check_time)
                return (pd.Timestamp.now() - check_dt).days < max_age_days
            except Exception:
                return False

    def mark_delisted(self, symbol: str, delist_date: str = '') -> None:
        """标记股票为已退市（永久标记，不过期）

        Args:
            symbol: 股票代码
            delist_date: 退市日期，格式 'YYYY-MM-DD'，未知则传空字符串
        """
        with self.lock:
            self._delisted_stocks[symbol] = delist_date or 'unknown'
            self._dirty_flags['delisted'] = True

    def is_delisted(self, symbol: str) -> bool:
        """检查股票是否已退市（永久标记，永不过期）"""
        with self.lock:
            return symbol in self._delisted_stocks

    def get_delist_date(self, symbol: str) -> Optional[str]:
        """获取退市日期"""
        with self.lock:
            date = self._delisted_stocks.get(symbol)
            if date and date != 'unknown':
                return date
            return None

    def mark_suspended(self, symbol: str, ranges: List[List[str]]) -> None:
        with self.lock:
            self._suspended_ranges[symbol] = ranges
            self._dirty_flags['suspended'] = True

    def get_suspended_ranges(self, symbol: str) -> List[List[str]]:
        with self.lock:
            return self._suspended_ranges.get(symbol, [])

    def is_suspended_on(self, symbol: str, date_str: str) -> bool:
        with self.lock:
            ranges = self._suspended_ranges.get(symbol, [])
            for start, end in ranges:
                if start <= date_str <= end:
                    return True
            return False

    def get_checked_dividend_stocks(self, max_age_days: int = 30) -> set:
        with self.lock:
            if not self._dividend_checked:
                return set()
            now = pd.Timestamp.now()
            valid = set()
            for stock, check_time in self._dividend_checked.items():
                try:
                    check_dt = pd.Timestamp(check_time)
                    if (now - check_dt).days < max_age_days:
                        valid.add(stock)
                except Exception:
                    continue
            return valid

    def rebuild_index_from_disk(self, disk_cache: 'DiskCache') -> None:
        """从磁盘文件重建索引（用于索引损坏或首次迁移时）"""
        self.logger.info("开始从磁盘重建缓存索引...")

        for namespace, index_attr, key_parts_count in [
            ('QMTDataProcessor', '_market_index', None),
            ('OpenDataProcessor', '_market_index', None),
            ('QMTDataProcessor_Raw', '_market_raw_index', None),
            ('OpenDataProcessor_Raw', '_market_raw_index', None),
            ('QMTDataProcessor_Financial', '_financial_index', None),
            ('OpenDataProcessor_Financial', '_financial_index', None),
        ('OpenDataProcessor_Dividend', '_financial_index', None),
        ]:
            ns_dir = disk_cache.get_namespace_dir(namespace)
            if not ns_dir.exists():
                continue

            is_market = 'market' in str(ns_dir) and 'market_raw' not in str(ns_dir)
            is_market_raw = 'market_raw' in str(ns_dir)
            is_financial = 'financial' in str(ns_dir)

            for f in ns_dir.rglob('*.parquet'):
                if not f.is_file():
                    continue
                try:
                    self._index_file(f, is_market, is_financial, namespace, is_market_raw)
                except Exception as e:
                    self.logger.debug(f"索引文件跳过: {f}, 错误: {e}")

        self._dirty_flags['market'] = True
        self._dirty_flags['market_raw'] = True
        self._dirty_flags['financial'] = True
        self.save_index()
        self.logger.info("缓存索引重建完成")

    def _index_file(self, filepath: Path, is_market: bool, is_financial: bool, namespace: str, is_market_raw: bool = False) -> None:
        """根据文件路径模式更新索引"""
        name = filepath.stem

        if is_market_raw:
            year_match = re.match(r'^(\d{4})_(\w+)$', name)
            if year_match:
                year = int(year_match.group(1))
                period = year_match.group(2)
                symbol = filepath.parent.name
                self.update_market_raw_index(symbol, period, year)
                return

        if is_market:
            year_match = re.match(r'^(\d{4})_(\w+)$', name)
            if year_match:
                year = int(year_match.group(1))
                period = year_match.group(2)
                symbol = filepath.parent.name
                self.update_market_index(symbol, period, year)
                return

        if is_financial:
            year_match = re.match(r'^(\d{4})_(.+)$', name)
            if year_match:
                year = int(year_match.group(1))
                table = year_match.group(2)
                symbol = filepath.parent.name
                self.update_financial_index(symbol, table, year)
                return
