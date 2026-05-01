import os
import re
import json
import time
import pickle
import hashlib
import threading
import logging
import inspect
from typing import Dict, Any, Optional, Callable, Tuple, Union, List, Set
from collections import OrderedDict
from functools import wraps
from pathlib import Path
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logging.warning("pyarrow 未安装，行情数据缓存将退级使用 pickle，建议 pip install pyarrow 提升性能")


class MemCache:
    """基于 OrderedDict 的线程安全 LRU 内存缓存"""
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()


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
        self._financial_index: Dict[str, Dict[str, Dict]] = {}
        self._dividend_checked: Dict[str, str] = {}
        self._dirty = False
        self.load_index()

    @property
    def market_index_path(self) -> Path:
        return self.cache_dir / 'index' / 'market_index.json'

    @property
    def financial_index_path(self) -> Path:
        return self.cache_dir / 'index' / 'financial_index.json'

    @property
    def dividend_checked_path(self) -> Path:
        return self.cache_dir / 'index' / 'dividend_checked.json'

    def load_index(self) -> None:
        for path, attr in [
            (self.market_index_path, '_market_index'),
            (self.financial_index_path, '_financial_index'),
            (self.dividend_checked_path, '_dividend_checked'),
        ]:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        setattr(self, attr, json.load(f))
                except Exception as e:
                    self.logger.warning(f"加载索引文件失败: {path}, 错误: {e}")
                    setattr(self, attr, {} if attr != '_dividend_checked' else {})

    def save_index(self) -> None:
        if not self._dirty:
            return
        with self.lock:
            for path, data in [
                (self.market_index_path, self._market_index),
                (self.financial_index_path, self._financial_index),
                (self.dividend_checked_path, self._dividend_checked),
            ]:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = path.with_suffix('.json.tmp')
                    with open(tmp_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    tmp_path.replace(path)
                except Exception as e:
                    self.logger.warning(f"保存索引文件失败: {path}, 错误: {e}")
            self._dirty = False

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
            self._dirty = True

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
            self._dirty = True

    def get_checked_market_years(self, symbol: str, period: str, max_age_days: int = 30) -> List[int]:
        with self.lock:
            entry = self._market_index.get(symbol, {}).get(period, {})
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
            self._dirty = True

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
            self._dirty = True

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
            self._dirty = True

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
            ('QMTDataProcessor_Financial', '_financial_index', None),
            ('OpenDataProcessor_Financial', '_financial_index', None),
        ]:
            ns_dir = disk_cache.get_namespace_dir(namespace)
            if not ns_dir.exists():
                continue

            is_market = 'market' in str(ns_dir)
            is_financial = 'financial' in str(ns_dir)

            for f in ns_dir.rglob('*.parquet'):
                if not f.is_file():
                    continue
                try:
                    self._index_file(f, is_market, is_financial, namespace)
                except Exception as e:
                    self.logger.debug(f"索引文件跳过: {f}, 错误: {e}")

        self._dirty = True
        self.save_index()
        self.logger.info("缓存索引重建完成")

    def _index_file(self, filepath: Path, is_market: bool, is_financial: bool, namespace: str) -> None:
        """根据文件路径模式更新索引"""
        name = filepath.stem

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


class DiskCache:
    """支持 Parquet 和 Pickle 的磁盘持久化缓存

    V2: 支持按年份分片存储
        行情数据: .cache/QMTData/market/{symbol}/{year}_{period}.parquet
        财报数据: .cache/QMTData/financial/{symbol}/{year}_{table}.parquet
    """
    NAMESPACE_MAP = {
        'QMTDataProcessor': 'market',
        'QMTDataProcessor_Financial': 'financial',
        'QMTDataProcessor_Industry': 'industry',
        'QMTDataProcessor_Sector': 'sector',
        'OpenDataProcessor': 'market',
        'OpenDataProcessor_Financial': 'financial',
        'OpenDataProcessor_Industry': 'industry',
        'OpenDataProcessor_Sector': 'sector',
    }

    BASE_DIR_MAP = {
        'QMTDataProcessor': 'QMTData',
        'QMTDataProcessor_Financial': 'QMTData',
        'QMTDataProcessor_Industry': 'QMTData',
        'QMTDataProcessor_Sector': 'QMTData',
        'OpenDataProcessor': 'OpenData',
        'OpenDataProcessor_Financial': 'OpenData',
        'OpenDataProcessor_Industry': 'OpenData',
        'OpenDataProcessor_Sector': 'OpenData',
    }

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_dir()

    def _ensure_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_namespace(self, namespace: str) -> str:
        return self.NAMESPACE_MAP.get(namespace, namespace)

    def _get_base_dir(self, namespace: str) -> Path:
        base_name = self.BASE_DIR_MAP.get(namespace)
        if base_name:
            return self.cache_dir / base_name
        return self.cache_dir

    def get_namespace_dir(self, namespace: str) -> Path:
        """返回映射后的 namespace 目录路径"""
        ns_dir = self._get_base_dir(namespace) / self._resolve_namespace(namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir

    def _get_file_path(self, namespace: str, key: str, format_type: str) -> Path:
        ns_dir = self._get_base_dir(namespace) / self._resolve_namespace(namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        ext = '.parquet' if format_type == 'parquet' else '.pkl'
        illegal_chars = '<>:"/\\|?*'
        safe_key = "".join([c for c in key if c not in illegal_chars]).rstrip()
        return ns_dir / f"{safe_key}{ext}"

    def _get_yearly_dir(self, namespace: str, symbol: str) -> Path:
        """获取按年份分片存储的子目录: .cache/QMTData/market/{symbol}/"""
        ns_dir = self._get_base_dir(namespace) / self._resolve_namespace(namespace) / symbol
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir

    def _get_yearly_file_path(self, namespace: str, symbol: str,
                               year: int, suffix: str, format_type: str = 'parquet') -> Path:
        """获取按年份分片的文件路径

        行情: .cache/QMTData/market/000001.SZ/2025_1d.parquet
        财报: .cache/QMTData/financial/000001.SZ/2023_Balance.parquet
        """
        year_dir = self._get_yearly_dir(namespace, symbol)
        ext = '.parquet' if format_type == 'parquet' else '.pkl'
        return year_dir / f"{year}_{suffix}{ext}"

    def get(self, namespace: str, key: str, format_type: str) -> Optional[Any]:
        file_path = self._get_file_path(namespace, key, format_type)
        with self.lock:
            if not file_path.exists():
                return None
            try:
                if format_type == 'parquet' and HAS_PYARROW:
                    return pd.read_parquet(file_path)
                else:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"读取磁盘缓存失败 (文件可能损坏)，自动删除: {file_path}, 错误: {e}")
                try:
                    file_path.unlink(missing_ok=True)
                except:
                    pass
                return None

    def get_yearly(self, namespace: str, symbol: str, year: int,
                    suffix: str, format_type: str = 'parquet') -> Optional[pd.DataFrame]:
        """按年份获取分片数据

        Args:
            namespace: 命名空间
            symbol: 股票代码
            year: 年份
            suffix: 行情为period(如1d)，财报为table名(如Balance)
            format_type: 文件格式
        """
        file_path = self._get_yearly_file_path(namespace, symbol, year, suffix, format_type)
        with self.lock:
            if not file_path.exists():
                return None
            try:
                if format_type == 'parquet' and HAS_PYARROW:
                    return pd.read_parquet(file_path)
                else:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"读取年份缓存失败: {file_path}, 错误: {e}")
                try:
                    file_path.unlink(missing_ok=True)
                except:
                    pass
                return None

    def get_yearly_range(self, namespace: str, symbol: str, years: List[int],
                          suffix: str, format_type: str = 'parquet') -> Optional[pd.DataFrame]:
        """获取多个年份的分片数据并合并

        Args:
            namespace: 命名空间
            symbol: 股票代码
            years: 年份列表
            suffix: 行情为period(如1d)，财报为table名(如Balance)
            format_type: 文件格式
        """
        frames = []
        for year in sorted(years):
            df = self.get_yearly(namespace, symbol, year, suffix, format_type)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]

        merged = pd.concat(frames)
        if isinstance(merged.index, pd.DatetimeIndex):
            merged = merged[~merged.index.duplicated(keep='last')]
            merged.sort_index(inplace=True)
        return merged

    def put_yearly(self, namespace: str, symbol: str, year: int,
                    suffix: str, value: pd.DataFrame, format_type: str = 'parquet') -> bool:
        """按年份存储分片数据

        自动按年份切割DataFrame，只保留该年份的数据。
        """
        if value is None or not isinstance(value, pd.DataFrame) or value.empty:
            return False

        year_df = value.copy()
        if isinstance(year_df.index, pd.DatetimeIndex):
            year_df = year_df[year_df.index.year == year]
            if year_df.empty:
                return False

        file_path = self._get_yearly_file_path(namespace, symbol, year, suffix, format_type)
        with self.lock:
            try:
                temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
                if format_type == 'parquet' and HAS_PYARROW:
                    save_df = year_df.copy()
                    cols = list(save_df.columns)
                    seen = {}
                    new_cols = []
                    for c in cols:
                        c_str = str(c)
                        if c_str in seen:
                            seen[c_str] += 1
                            new_cols.append(f"{c_str}_{seen[c_str]}")
                        else:
                            seen[c_str] = 0
                            new_cols.append(c_str)
                    save_df.columns = new_cols
                    save_df.to_parquet(temp_path, engine='pyarrow', compression='snappy')
                else:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(year_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                temp_path.replace(file_path)
                return True
            except Exception as e:
                self.logger.error(f"写入年份缓存失败: {file_path}, 错误: {e}")
                if 'temp_path' in locals() and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                return False

    def put_yearly_from_df(self, namespace: str, symbol: str,
                            suffix: str, df: pd.DataFrame,
                            format_type: str = 'parquet',
                            only_years: Optional[List[int]] = None,
                            skip_existing: bool = False) -> List[int]:
        """将一个DataFrame按年份拆分并存储

        Args:
            only_years: 如果指定，只保存这些年份的数据，跳过其他年份。
                        为 None 时保存所有年份（默认行为）。
            skip_existing: 如果为 True，跳过磁盘上已存在的年份文件，避免重复写入。

        Returns:
            成功写入的年份列表（不含跳过的年份）
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []

        if not isinstance(df.index, pd.DatetimeIndex):
            return []

        written_years = []
        for year in sorted(df.index.year.unique()):
            if only_years is not None and year not in only_years:
                continue
            if skip_existing:
                file_path = self._get_yearly_file_path(namespace, symbol, year, suffix, format_type)
                if file_path.exists():
                    continue
            if self.put_yearly(namespace, symbol, year, suffix, df, format_type):
                written_years.append(year)

        return written_years

    def list_yearly_files(self, namespace: str, symbol: str,
                           suffix: str, format_type: str = 'parquet') -> List[int]:
        """列出某股票某后缀的所有可用年份

        Returns:
            年份列表，升序排列
        """
        year_dir = self._get_yearly_dir(namespace, symbol)
        if not year_dir.exists():
            return []

        ext = '.parquet' if format_type == 'parquet' else '.pkl'
        years = []
        with self.lock:
            for f in year_dir.iterdir():
                if not f.is_file() or f.suffix != ext:
                    continue
                m = re.match(r'^(\d{4})_' + re.escape(suffix) + r'$', f.stem)
                if m:
                    years.append(int(m.group(1)))
        years.sort()
        return years

    def find_by_prefix(self, namespace: str, prefix: str, format_type: str) -> Optional[Tuple[str, Any]]:
        """按前缀查找缓存文件，返回 (完整key, 数据) 或 None"""
        ns_dir = self._get_base_dir(namespace) / self._resolve_namespace(namespace)
        if not ns_dir.exists():
            return None
        ext = '.parquet' if format_type == 'parquet' else '.pkl'
        illegal_chars = '<>:"/\\|?*'
        safe_prefix = "".join([c for c in prefix if c not in illegal_chars]).rstrip()
        with self.lock:
            for f in ns_dir.iterdir():
                if f.is_file() and f.suffix == ext and f.stem.startswith(safe_prefix):
                    key = f.stem
                    try:
                        if format_type == 'parquet' and HAS_PYARROW:
                            data = pd.read_parquet(f)
                        else:
                            with open(f, 'rb') as fh:
                                data = pickle.load(fh)
                        return (key, data)
                    except Exception as e:
                        self.logger.warning(f"读取磁盘缓存失败: {f}, 错误: {e}")
                        try:
                            f.unlink(missing_ok=True)
                        except:
                            pass
            return None

    def find_by_pattern(self, namespace: str, pattern: str,
                        format_type: str = 'parquet') -> List[Tuple[str, Any]]:
        """按 glob 模式查找缓存文件，返回 [(完整key, 数据), ...]"""
        ns_dir = self._get_base_dir(namespace) / self._resolve_namespace(namespace)
        if not ns_dir.exists():
            return []
        ext = '.parquet' if format_type == 'parquet' else '.pkl'
        illegal_chars = '<>:"/\\|'
        safe_pattern = "".join([c for c in pattern if c not in illegal_chars]).rstrip()
        results = []
        with self.lock:
            for f in ns_dir.glob(f"{safe_pattern}{ext}"):
                if not f.is_file():
                    continue
                key = f.stem
                try:
                    if format_type == 'parquet' and HAS_PYARROW:
                        data = pd.read_parquet(f)
                    else:
                        with open(f, 'rb') as fh:
                            data = pickle.load(fh)
                    if data is not None:
                        results.append((key, data))
                except Exception as e:
                    self.logger.warning(f"读取磁盘缓存失败: {f}, 错误: {e}")
                    try:
                        f.unlink(missing_ok=True)
                    except:
                        pass
        results.sort(key=lambda x: x[0])
        return results

    def delete(self, namespace: str, key: str, format_type: str) -> bool:
        """删除指定的缓存文件"""
        file_path = self._get_file_path(namespace, key, format_type)
        with self.lock:
            if file_path.exists():
                try:
                    file_path.unlink()
                    return True
                except Exception as e:
                    self.logger.warning(f"删除缓存文件失败: {file_path}, 错误: {e}")
                    return False
            return False

    def delete_yearly(self, namespace: str, symbol: str, year: int,
                       suffix: str, format_type: str = 'parquet') -> bool:
        """删除指定年份的分片缓存"""
        file_path = self._get_yearly_file_path(namespace, symbol, year, suffix, format_type)
        with self.lock:
            if file_path.exists():
                try:
                    file_path.unlink()
                    return True
                except Exception as e:
                    self.logger.warning(f"删除年份缓存失败: {file_path}, 错误: {e}")
                    return False
            return False

    def put(self, namespace: str, key: str, value: Any, format_type: str) -> bool:
        if value is None:
            return False
        file_path = self._get_file_path(namespace, key, format_type)
        with self.lock:
            try:
                temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
                if format_type == 'parquet' and HAS_PYARROW and isinstance(value, pd.DataFrame):
                    save_df = value.copy()
                    cols = list(save_df.columns)
                    seen = {}
                    new_cols = []
                    for c in cols:
                        c_str = str(c)
                        if c_str in seen:
                            seen[c_str] += 1
                            new_cols.append(f"{c_str}_{seen[c_str]}")
                        else:
                            seen[c_str] = 0
                            new_cols.append(c_str)
                    save_df.columns = new_cols
                    save_df.to_parquet(temp_path, engine='pyarrow', compression='snappy')
                else:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                temp_path.replace(file_path)
                return True
            except Exception as e:
                self.logger.error(f"写入磁盘缓存失败: {file_path}, 错误: {e}")
                if 'temp_path' in locals() and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                return False

    def migrate_old_market_cache(self, namespace: str = 'QMTDataProcessor',
                                   dry_run: bool = False) -> Dict[str, Any]:
        """将旧格式行情缓存迁移为按年份分片格式

        旧格式: .cache/QMTData/market/000001.SZ_1d_20200101_20251231.parquet
        新格式: .cache/QMTData/market/000001.SZ/2020_1d.parquet, 2021_1d.parquet, ...

        Args:
            namespace: 命名空间
            dry_run: True=只统计不实际迁移

        Returns:
            迁移统计信息
        """
        ns_dir = self.get_namespace_dir(namespace)
        if not ns_dir.exists():
            return {'status': 'no_cache_dir', 'migrated': 0}

        stats = {'migrated': 0, 'skipped': 0, 'failed': 0, 'files': []}
        pattern = re.compile(r'^(.+?)_(\w+)_(\d{8})_(\d{8})$')

        for cache_file in list(ns_dir.glob('*.parquet')):
            if not cache_file.is_file():
                continue
            m = pattern.match(cache_file.stem)
            if not m:
                stats['skipped'] += 1
                continue

            symbol = m.group(1)
            period = m.group(2)

            try:
                df = pd.read_parquet(cache_file)
                if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                    stats['skipped'] += 1
                    continue

                if not isinstance(df.index, pd.DatetimeIndex):
                    stats['skipped'] += 1
                    continue

                if dry_run:
                    years = sorted(df.index.year.unique())
                    stats['files'].append({
                        'old': str(cache_file.name),
                        'symbol': symbol,
                        'period': period,
                        'years': years,
                        'rows': len(df)
                    })
                    stats['migrated'] += 1
                else:
                    written = self.put_yearly_from_df(namespace, symbol, period, df)
                    if written:
                        cache_file.unlink()
                        stats['migrated'] += 1
                        stats['files'].append({
                            'old': str(cache_file.name),
                            'symbol': symbol,
                            'years': written
                        })
                    else:
                        stats['failed'] += 1
            except Exception as e:
                self.logger.warning(f"迁移行情缓存失败: {cache_file.name}, 错误: {e}")
                stats['failed'] += 1

        return stats

    def migrate_old_financial_cache(self, namespace: str = 'QMTDataProcessor_Financial',
                                      dry_run: bool = False) -> Dict[str, Any]:
        """将旧格式财报缓存迁移为按年份分片格式

        旧格式: .cache/QMTData/financial/000001.SZ_20200101_20251231_Balance_announce_time.parquet
        新格式: .cache/QMTData/financial/000001.SZ/2023_Balance_announce_time.parquet, ...

        Args:
            namespace: 命名空间
            dry_run: True=只统计不实际迁移

        Returns:
            迁移统计信息
        """
        ns_dir = self.get_namespace_dir(namespace)
        if not ns_dir.exists():
            return {'status': 'no_cache_dir', 'migrated': 0}

        stats = {'migrated': 0, 'skipped': 0, 'failed': 0, 'files': []}

        for cache_file in list(ns_dir.glob('*.parquet')):
            if not cache_file.is_file():
                continue
            if cache_file.stem.startswith('merged_'):
                stats['skipped'] += 1
                continue

            name = cache_file.stem
            parts = name.split('_')
            if len(parts) < 3:
                stats['skipped'] += 1
                continue

            try:
                df = pd.read_parquet(cache_file)
                if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                    stats['skipped'] += 1
                    continue

                if not isinstance(df.index, pd.DatetimeIndex):
                    stats['skipped'] += 1
                    continue

                symbol = parts[0]
                table_suffix = '_'.join(parts[-2:]) if len(parts) >= 4 else parts[-1]

                if dry_run:
                    years = sorted(df.index.year.unique())
                    stats['files'].append({
                        'old': str(cache_file.name),
                        'symbol': symbol,
                        'table_suffix': table_suffix,
                        'years': years,
                        'rows': len(df)
                    })
                    stats['migrated'] += 1
                else:
                    written = self.put_yearly_from_df(namespace, symbol, table_suffix, df)
                    if written:
                        cache_file.unlink()
                        stats['migrated'] += 1
                        stats['files'].append({
                            'old': str(cache_file.name),
                            'symbol': symbol,
                            'years': written
                        })
                    else:
                        stats['failed'] += 1
            except Exception as e:
                self.logger.warning(f"迁移财报缓存失败: {cache_file.name}, 错误: {e}")
                stats['failed'] += 1

        for cache_file in list(ns_dir.glob('merged_*.parquet')):
            if not cache_file.is_file():
                continue
            if dry_run:
                stats['files'].append({
                    'old': str(cache_file.name),
                    'note': '合并缓存，需单独处理'
                })
                stats['skipped'] += 1
            else:
                try:
                    cache_file.unlink()
                    stats['skipped'] += 1
                except:
                    pass

        return stats


class SmartCacheManager:
    """智能缓存核心调度器（支持内存+磁盘+增量合并+按年份分片）

    V2 核心变化:
    - 行情数据按年份分片存储，一个股票一年一个文件
    - 财报数据按年份分片存储，一个股票一年一个文件
    - 通过 CacheIndexManager 维护索引，快速判断缓存是否可用
    - 保持向后兼容，旧格式文件仍可读取
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SmartCacheManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.base_cache_dir = os.environ.get('QMT_CACHE_DIR', os.path.join(os.getcwd(), '.cache'))
            self.mem_cache = MemCache(capacity=int(os.environ.get('QMT_MEM_CACHE_LIMIT', 500)))
            self.disk_cache = DiskCache(self.base_cache_dir)
            self.index_manager = CacheIndexManager(Path(self.base_cache_dir))
            self.stats = {
                'mem_hits': 0,
                'disk_hits': 0,
                'misses': 0,
                'incremental_merges': 0,
                'yearly_hits': 0,
                'total_load_time_ms': 0.0
            }
            self.logger = logging.getLogger(self.__class__.__name__)
            self.initialized = True

    def configure(self, cache_dir: str, mem_limit: int = 500):
        self.base_cache_dir = cache_dir
        self.disk_cache = DiskCache(self.base_cache_dir)
        self.index_manager = CacheIndexManager(Path(self.base_cache_dir))
        self.mem_cache.capacity = mem_limit
        self.logger.info(f"缓存系统配置更新: 目录={cache_dir}, 内存限制={mem_limit}")

    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        generic_funcs = {'get_data', 'get_financial_data', 'get_dividend_data'}
        key_parts = [] if func_name in generic_funcs else [func_name]
        skip_keys = {'start_date', 'end_date', 'start_time', 'end_time'}
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')

        def _format_arg(v):
            if isinstance(v, (list, tuple)):
                if len(v) > 3:
                    s = str(sorted([str(x) for x in v]))
                    h = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
                    return f"list_{len(v)}_{h}"
                return str(v)
            if isinstance(v, dict):
                s = str(sorted(v.items()))
                h = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
                return f"dict_{len(v)}_{h}"
            return str(v)

        for arg in args:
            if isinstance(arg, (pd.DataFrame, pd.Series)):
                continue
            if not isinstance(arg, (str, int, float, bool, list, tuple, dict, type(None))):
                continue
            if isinstance(arg, str) and date_pattern.match(arg):
                continue
            key_parts.append(_format_arg(arg))

        for k, v in sorted(kwargs.items()):
            if k not in skip_keys and not isinstance(v, (pd.DataFrame, pd.Series)):
                key_parts.append(f"{k}={_format_arg(v)}")

        return "_".join(key_parts)

    def _get_param_value(self, func: Callable, args: tuple, kwargs: dict, param_name: str, default: Any = '') -> Any:
        if param_name in kwargs:
            return kwargs[param_name]
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if param_name in params:
                idx = params.index(param_name)
                if idx < len(args):
                    return args[idx]
        except (ValueError, TypeError):
            pass
        return default

    def _build_incremental_args(self, func: Callable, args: tuple, kwargs: dict,
                                 start_date_override: Optional[str] = None,
                                 end_date_override: Optional[str] = None) -> Tuple[tuple, dict]:
        new_kwargs = dict(kwargs)
        if start_date_override is not None:
            new_kwargs['start_date'] = start_date_override
        if end_date_override is not None:
            new_kwargs['end_date'] = end_date_override

        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
        except (ValueError, TypeError) as e:
            self.logger.warning(f"[增量参数] 无法获取函数签名: {e}, func={getattr(func, '__name__', func)}, 回退简单处理")
            return (), new_kwargs

        override_params = set()
        if start_date_override is not None:
            override_params.add('start_date')
        if end_date_override is not None:
            override_params.add('end_date')

        min_override_idx = None
        for i, name in enumerate(params):
            if name in override_params:
                min_override_idx = i
                break

        if min_override_idx is None:
            return args, new_kwargs

        if min_override_idx >= len(args):
            return args, new_kwargs

        new_args = args[:min_override_idx]
        for i in range(min_override_idx, len(args)):
            param_name = params[i] if i < len(params) else None
            if param_name and param_name not in new_kwargs:
                new_kwargs[param_name] = args[i]

        return new_args, new_kwargs

    def _merge_incremental_data(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        if old_df is None or old_df.empty:
            return new_df
        if new_df is None or new_df.empty:
            return old_df
        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)
        if old_cols != new_cols:
            common_cols = list(old_cols & new_cols)
            if common_cols:
                old_df = old_df[common_cols]
                new_df = new_df[common_cols]
            else:
                self.logger.warning(f"增量数据与缓存数据列完全不同，放弃合并: 旧={list(old_cols)}, 新={list(new_cols)}")
                return old_df
        merged = pd.concat([old_df, new_df])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged.sort_index(inplace=True)
        return merged

    def _parse_years_from_range(self, start_date: str, end_date: str) -> List[int]:
        """从日期范围解析出涉及的年份列表"""
        try:
            start_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year
            return list(range(start_year, end_year + 1))
        except Exception:
            return []

    def execute_with_cache(self, namespace: str, cache_type: str, incremental: bool,
                           func: Callable, args: tuple, kwargs: dict) -> Any:
        """缓存执行入口 - 优先使用按年份分片的新逻辑"""
        if cache_type == 'market' and incremental:
            return self._execute_market_cache_v2(namespace, func, args, kwargs)

        start_time = time.time()
        format_type = 'parquet' if cache_type == 'market' and incremental else 'pkl'
        base_key = self._get_cache_key(func.__name__, *args, **kwargs)
        full_key = f"{namespace}_{base_key}"

        if not incremental:
            time_params = []
            for tk in ['start_date', 'end_date', 'start_time', 'end_time']:
                if tk in kwargs:
                    time_params.append(f"{tk}={kwargs[tk]}")
            if time_params:
                base_key = f"{base_key}_" + "_".join(time_params)
                full_key = f"{namespace}_{base_key}"

        mem_data = self.mem_cache.get(full_key)
        if mem_data is not None:
            self.stats['mem_hits'] += 1
            self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
            if incremental and isinstance(mem_data, pd.DataFrame) and not mem_data.empty:
                req_start = self._get_param_value(func, args, kwargs, 'start_date', '')
                req_end = self._get_param_value(func, args, kwargs, 'end_date', '')
                mem_end = mem_data.index.max().strftime('%Y-%m-%d') if isinstance(mem_data.index, pd.DatetimeIndex) else str(mem_data.index.max())
                if req_end and mem_end >= req_end:
                    return self._filter_by_date(mem_data, req_start, req_end)
            else:
                return mem_data

        disk_data = None
        disk_key = base_key
        if incremental and cache_type == 'market':
            found = self.disk_cache.find_by_prefix(namespace, base_key, format_type)
            if found is not None:
                disk_key, disk_data = found
        else:
            disk_data = self.disk_cache.get(namespace, base_key, format_type)

        if disk_data is not None:
            self.mem_cache.put(full_key, disk_data)
            self.stats['disk_hits'] += 1

            if incremental and isinstance(disk_data, pd.DataFrame) and not disk_data.empty:
                req_start = self._get_param_value(func, args, kwargs, 'start_date', '')
                req_end = self._get_param_value(func, args, kwargs, 'end_date', '')
                disk_start = disk_data.index.min().strftime('%Y-%m-%d') if isinstance(disk_data.index, pd.DatetimeIndex) else str(disk_data.index.min())
                disk_end = disk_data.index.max().strftime('%Y-%m-%d') if isinstance(disk_data.index, pd.DatetimeIndex) else str(disk_data.index.max())

                if req_start and req_end and disk_start <= req_start and disk_end >= req_end:
                    self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                    return self._filter_by_date(disk_data, req_start, req_end)

                self.logger.info(f"[{namespace}] 触发智能增量更新: {base_key}, 本地范围 {disk_start}~{disk_end}, 需求范围 {req_start}~{req_end}")
                self.stats['incremental_merges'] += 1

                if pd.Timestamp(disk_start) > pd.Timestamp(req_start):
                    self.logger.debug(f"[{namespace}] 向前扩展: 从 {req_start} 到 {disk_start}")
                    inc_args, inc_kwargs = self._build_incremental_args(
                        func, args, kwargs,
                        start_date_override=req_start,
                        end_date_override=(pd.Timestamp(disk_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    )
                else:
                    next_day = (pd.Timestamp(disk_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    self.logger.debug(f"[{namespace}] 向后扩展: 从 {next_day} 到 {req_end}")
                    inc_args, inc_kwargs = self._build_incremental_args(
                        func, args, kwargs, start_date_override=next_day
                    )

                try:
                    new_data = func(*inc_args, **inc_kwargs)
                    merged_data = self._merge_incremental_data(disk_data, new_data)

                    new_disk_key = self._build_disk_key_with_dates(base_key, merged_data)
                    self.disk_cache.delete(namespace, disk_key, format_type)
                    self.disk_cache.put(namespace, new_disk_key, merged_data, format_type)
                    self.mem_cache.put(full_key, merged_data)

                    self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                    return self._filter_by_date(merged_data, req_start, req_end)
                except Exception as e:
                    self.logger.warning(f"增量更新失败，退回全量下载: {e}")
            elif not incremental or not (isinstance(disk_data, pd.DataFrame) and not disk_data.empty):
                self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                return disk_data

        self.stats['misses'] += 1
        self.logger.info(f"[{namespace}] 缓存未命中，全量获取: {base_key}")

        try:
            result = func(*args, **kwargs)
            if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                if incremental and cache_type == 'market' and isinstance(result, pd.DataFrame):
                    disk_write_key = self._build_disk_key_with_dates(base_key, result)
                else:
                    disk_write_key = base_key
                self.disk_cache.put(namespace, disk_write_key, result, format_type)
                self.mem_cache.put(full_key, result)

            self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
            return result
        except Exception as e:
            self.logger.error(f"[{namespace}] 获取数据失败: {e}")
            raise

    def _execute_market_cache_v2(self, namespace: str, func: Callable,
                                  args: tuple, kwargs: dict) -> Any:
        """行情数据按年份分片缓存逻辑

        核心流程:
        1. 从请求参数中提取 symbol, period, start_date, end_date
        2. 计算请求涉及的年份列表
        3. 查索引/磁盘，判断哪些年份已有缓存
        4. 对缺失年份调用原始函数获取数据
        5. 按年份写入磁盘，更新索引
        6. 合并所有年份的数据并返回
        """
        start_time = time.time()
        symbol = self._get_param_value(func, args, kwargs, 'symbol', '')
        period = self._get_param_value(func, args, kwargs, 'period', '1d')
        req_start = self._get_param_value(func, args, kwargs, 'start_date', '')
        req_end = self._get_param_value(func, args, kwargs, 'end_date', '')

        if not symbol or not req_start or not req_end:
            return func(*args, **kwargs)

        mem_key = f"{namespace}_market_{symbol}_{period}_{req_start}_{req_end}"
        mem_data = self.mem_cache.get(mem_key)
        if mem_data is not None:
            self.stats['mem_hits'] += 1
            self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
            return mem_data

        req_years = self._parse_years_from_range(req_start, req_end)
        if not req_years:
            return func(*args, **kwargs)

        available_years = self.index_manager.get_available_market_years(symbol, period)
        if not available_years:
            available_years = self.disk_cache.list_yearly_files(namespace, symbol, period)

        cached_years = set(available_years) & set(req_years)
        missing_years = set(req_years) - cached_years

        checked_years = set(self.index_manager.get_checked_market_years(symbol, period))
        truly_missing = missing_years - checked_years

        cached_frames = []
        if cached_years:
            df = self.disk_cache.get_yearly_range(namespace, symbol, sorted(cached_years), period)
            if df is not None and not df.empty:
                cached_frames.append(df)
                self.stats['yearly_hits'] += 1

        if not truly_missing:
            merged = self.disk_cache.get_yearly_range(namespace, symbol, sorted(cached_years), period)
            if merged is not None and not merged.empty:
                if isinstance(merged.index, pd.DatetimeIndex):
                    disk_start = merged.index.min().strftime('%Y-%m-%d')
                    disk_end = merged.index.max().strftime('%Y-%m-%d')
                    if disk_end >= req_end:
                        result = self._filter_by_date(merged, req_start, req_end)
                        self.mem_cache.put(mem_key, result)
                        self.stats['disk_hits'] += 1
                        self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                        return result
                    last_cached_year = merged.index.max().year
                    truly_missing = set(y for y in req_years if y > last_cached_year)
                    truly_missing -= checked_years
                    if not truly_missing:
                        truly_missing = {max(req_years)} - checked_years
                        if not truly_missing:
                            result = self._filter_by_date(merged, req_start, req_end)
                            self.mem_cache.put(mem_key, result)
                            self.stats['disk_hits'] += 1
                            self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                            return result
                    self.logger.info(
                        f"[{namespace}] {symbol} 缓存未覆盖结束日期: "
                        f"缓存截止={disk_end}, 请求截止={req_end}, 增量获取年份={sorted(truly_missing)}"
                    )
                else:
                    truly_missing = set(req_years)
                    cached_frames = []
                    self.logger.info(f"[{namespace}] {symbol} 缓存索引非时间类型，重新获取: {sorted(truly_missing)}")
            else:
                truly_missing = set(req_years)
                cached_frames = []
                self.logger.info(f"[{namespace}] {symbol} 缓存数据为空，重新获取: {sorted(truly_missing)}")

        if truly_missing:
            self.stats['incremental_merges'] += 1
            actual_cached = sorted(set(req_years) - truly_missing)
            self.logger.info(
                f"[{namespace}] {symbol} 行情增量更新: "
                f"已有年份={actual_cached}, 缺失年份={sorted(truly_missing)}"
            )

            min_missing = min(truly_missing)
            max_missing = max(truly_missing)

            inc_start = req_start if min_missing == min(req_years) else f"{min_missing}-01-01"
            inc_end = req_end if max_missing == max(req_years) else f"{max_missing}-12-31"

            inc_args, inc_kwargs = self._build_incremental_args(
                func, args, kwargs,
                start_date_override=inc_start,
                end_date_override=inc_end
            )

            try:
                new_data = func(*inc_args, **inc_kwargs)
                if new_data is not None and isinstance(new_data, pd.DataFrame) and not new_data.empty:
                    written = self.disk_cache.put_yearly_from_df(namespace, symbol, period, new_data)
                    for y in written:
                        self.index_manager.update_market_index(symbol, period, y)
                    self.index_manager.save_index()
                    cached_frames.append(new_data)
                    fetched_years_with_data = set(new_data.index.year.unique()) if isinstance(new_data.index, pd.DatetimeIndex) else set()
                    no_data_years = truly_missing - fetched_years_with_data
                else:
                    no_data_years = truly_missing
                if no_data_years:
                    self.index_manager.update_checked_market_years(symbol, period, sorted(no_data_years))
                    self.index_manager.save_index()
            except Exception as e:
                self.logger.warning(f"[{namespace}] {symbol} 增量获取失败: {e}")

        if not cached_frames:
            self.stats['misses'] += 1
            self.logger.info(f"[{namespace}] {symbol} 缓存未命中，全量获取")
            try:
                result = func(*args, **kwargs)
                if result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                    written = self.disk_cache.put_yearly_from_df(namespace, symbol, period, result)
                    for y in written:
                        self.index_manager.update_market_index(symbol, period, y)
                    self.index_manager.save_index()
                    self.mem_cache.put(mem_key, result)
                self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                return result
            except Exception as e:
                self.logger.error(f"[{namespace}] 获取数据失败: {e}")
                raise

        if len(cached_frames) == 1:
            merged = cached_frames[0]
        else:
            merged = self._merge_incremental_data(cached_frames[0], cached_frames[1])
            for i in range(2, len(cached_frames)):
                merged = self._merge_incremental_data(merged, cached_frames[i])

        result = self._filter_by_date(merged, req_start, req_end)
        self.mem_cache.put(mem_key, result)
        self.stats['disk_hits'] += 1
        self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
        return result

    def _build_disk_key_with_dates(self, base_key: str, df: pd.DataFrame) -> str:
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return base_key
        start = df.index.min().strftime('%Y%m%d')
        end = df.index.max().strftime('%Y%m%d')
        return f"{base_key}_{start}_{end}"

    def _filter_by_date(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        filtered = df
        if start_date:
            filtered = filtered[filtered.index >= start_date]
        if end_date:
            filtered = filtered[filtered.index <= end_date]
        return filtered

    def get_stats_report(self) -> str:
        total_requests = self.stats['mem_hits'] + self.stats['disk_hits'] + self.stats['misses']
        hit_rate = (self.stats['mem_hits'] + self.stats['disk_hits']) / total_requests if total_requests > 0 else 0
        avg_time = self.stats['total_load_time_ms'] / total_requests if total_requests > 0 else 0
        report = (
            f"=== 智能缓存系统性能报告 ===\n"
            f"总请求数: {total_requests}\n"
            f"内存命中: {self.stats['mem_hits']} 次\n"
            f"磁盘命中: {self.stats['disk_hits']} 次\n"
            f"年份命中: {self.stats['yearly_hits']} 次\n"
            f"未命中数: {self.stats['misses']} 次\n"
            f"增量合并: {self.stats['incremental_merges']} 次\n"
            f"综合命中率: {hit_rate:.2%}\n"
            f"平均加载耗时: {avg_time:.2f} ms/次\n"
            f"============================"
        )
        return report


cache_manager = SmartCacheManager()


def smart_cache(cache_type: str = 'market', incremental: bool = False):
    """智能数据缓存装饰器

    Args:
        cache_type: 'market' (行情数据, 优先使用Parquet) 或 'financial' (财务数据, 使用Pickle)
        incremental: 是否开启智能增量合并（仅针对带有 start_date, end_date 参数且返回 DataFrame 的函数有效）

    使用示例:
        @smart_cache(cache_type='market', incremental=True)
        def get_data(self, symbol, start_date, end_date):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            namespace = self.__class__.__name__
            if cache_type == 'financial':
                namespace = f"{namespace}_Financial"
            return cache_manager.execute_with_cache(
                namespace=namespace,
                cache_type=cache_type,
                incremental=incremental,
                func=func,
                args=(self,) + args,
                kwargs=kwargs
            )
        return wrapper
    return decorator
