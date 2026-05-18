import pickle
import re
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logging.warning("pyarrow 未安装，行情数据缓存将退级使用 pickle，建议 pip install pyarrow 提升性能")


_SAFE_PICKLE_MODULES = {
    'pandas', 'pandas.core', 'pandas.core.frame', 'pandas.core.series',
    'pandas.core.indexes', 'pandas.core.indexes.base', 'pandas.core.indexes.datetimes',
    'pandas.core.indexes.range', 'pandas.core.indexes.multi',
    'numpy', 'numpy.core', 'numpy.core.multiarray', 'numpy.core.numeric',
    'numpy.dtype', 'numpy.ndarray',
    'collections', 'collections.OrderedDict',
    'datetime', 'datetime datetime', 'datetime date',
    'builtins', '__builtin__',
}


class _RestrictedUnpickler(pickle.Unpickler):
    """限制 pickle 反序列化的类白名单，防止恶意代码执行"""

    def find_class(self, module, name):
        if module in _SAFE_PICKLE_MODULES or module.startswith('pandas.') or module.startswith('numpy.'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Forbidden class: {module}.{name} — "
            f"pickle cache only allows pandas/numpy/datetime types. "
            f"If this is legitimate, add '{module}' to _SAFE_PICKLE_MODULES."
        )


def _safe_pickle_load(file_obj):
    """安全地加载 pickle 数据，限制可反序列化的类型"""
    return _RestrictedUnpickler(file_obj).load()


class DiskCache:
    """支持 Parquet 和 Pickle 的磁盘持久化缓存

    V2: 支持按年份分片存储
        行情数据: .cache/QMTData/market/{symbol}/{year}_{period}.parquet
        财报数据: .cache/QMTData/financial/{symbol}/{year}_{table}.parquet
    """
    NAMESPACE_MAP = {
        'QMTDataProcessor': 'market',
        'QMTDataProcessor_Financial': 'financial',
        'QMTDataProcessor_Raw': 'market_raw',
        'OpenDataProcessor': 'market',
        'OpenDataProcessor_Financial': 'financial',
        'OpenDataProcessor_Dividend': 'dividend',
        'OpenDataProcessor_Industry': 'industry',
        'OpenDataProcessor_Sector': 'sector',
        'OpenDataProcessor_Raw': 'market_raw',
    }

    BASE_DIR_MAP = {
        'QMTDataProcessor': 'QMTData',
        'QMTDataProcessor_Financial': 'QMTData',
        'QMTDataProcessor_Raw': 'QMTData',
        'OpenDataProcessor': 'OpenData',
        'OpenDataProcessor_Financial': 'OpenData',
        'OpenDataProcessor_Dividend': 'OpenData',
        'OpenDataProcessor_Industry': 'OpenData',
        'OpenDataProcessor_Sector': 'OpenData',
        'OpenDataProcessor_Raw': 'OpenData',
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

    def _try_read_cache_file(self, file_path: Path, format_type: str) -> Optional[Any]:
        """尝试读取缓存文件，支持 parquet → pickle 回退

        处理 .parquet 扩展名文件实际包含 pickle 数据的情况
        （之前在无 pyarrow 环境下写入的缓存）。
        """
        try:
            if format_type == 'parquet' and HAS_PYARROW:
                return pd.read_parquet(file_path)
            else:
                with open(file_path, 'rb') as f:
                    return _safe_pickle_load(f)
        except Exception as first_err:
            if format_type == 'parquet' and HAS_PYARROW:
                try:
                    with open(file_path, 'rb') as f:
                        data = _safe_pickle_load(f)
                    self.logger.info(f"缓存文件pickle回退读取成功，将重写为parquet: {file_path}")
                    try:
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            self._rewrite_pickle_as_parquet(file_path, data)
                    except Exception as rewrite_err:
                        self.logger.debug(f"重写缓存文件跳过: {rewrite_err}")
                    return data
                except Exception as pickle_err:
                    self.logger.warning(f"读取磁盘缓存失败 (parquet和pickle均失败): {file_path}, 错误: {first_err} / {pickle_err}")
                    try:
                        file_path.unlink(missing_ok=True)
                    except:
                        pass
                    return None
            else:
                self.logger.warning(f"读取磁盘缓存失败 (文件可能损坏)，自动删除: {file_path}, 错误: {first_err}")
                try:
                    file_path.unlink(missing_ok=True)
                except:
                    pass
                return None

    def _rewrite_pickle_as_parquet(self, file_path: Path, data: pd.DataFrame) -> bool:
        """将pickle格式的缓存文件重写为parquet格式"""
        if not HAS_PYARROW:
            return False
        try:
            save_df = data.copy()
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
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
            save_df.to_parquet(temp_path, engine='pyarrow', compression='snappy')
            temp_path.replace(file_path)
            return True
        except Exception as e:
            self.logger.debug(f"重写parquet失败: {file_path}, {e}")
            return False

    def get(self, namespace: str, key: str, format_type: str) -> Optional[Any]:
        file_path = self._get_file_path(namespace, key, format_type)
        if not file_path.exists():
            return None
        return self._try_read_cache_file(file_path, format_type)

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
        if not file_path.exists():
            return None
        return self._try_read_cache_file(file_path, format_type)

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
                    data = self._try_read_cache_file(f, format_type)
                    if data is not None:
                        return (key, data)
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
                data = self._try_read_cache_file(f, format_type)
                if data is not None:
                    results.append((key, data))
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
