import os
import re
import time
import pickle
import hashlib
import threading
import logging
import inspect
from typing import Dict, Any, Optional, Callable, Tuple, Union
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


class DiskCache:
    """支持 Parquet 和 Pickle 的磁盘持久化缓存"""
    NAMESPACE_MAP = {
        'QMTDataProcessor': 'market',
        'QMTDataProcessor_Financial': 'financial',
        'QMTDataProcessor_Industry': 'industry',
        'QMTDataProcessor_Sector': 'sector',
    }

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.qmt_data_dir = self.cache_dir / 'QMTData'
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_dir()

    def _ensure_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_namespace(self, namespace: str) -> str:
        return self.NAMESPACE_MAP.get(namespace, namespace)

    def _is_qmt_namespace(self, namespace: str) -> bool:
        return namespace in self.NAMESPACE_MAP

    def _get_base_dir(self, namespace: str) -> Path:
        return self.qmt_data_dir if self._is_qmt_namespace(namespace) else self.cache_dir

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

    def put(self, namespace: str, key: str, value: Any, format_type: str) -> bool:
        if value is None:
            return False
        file_path = self._get_file_path(namespace, key, format_type)
        with self.lock:
            try:
                temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
                if format_type == 'parquet' and HAS_PYARROW and isinstance(value, pd.DataFrame):
                    save_df = value.copy()
                    save_df.columns = [str(c) for c in save_df.columns]
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


class SmartCacheManager:
    """智能缓存核心调度器（支持内存+磁盘+增量合并）"""
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
            self.stats = {
                'mem_hits': 0,
                'disk_hits': 0,
                'misses': 0,
                'incremental_merges': 0,
                'total_load_time_ms': 0.0
            }
            self.logger = logging.getLogger(self.__class__.__name__)
            self.initialized = True

    def configure(self, cache_dir: str, mem_limit: int = 500):
        self.base_cache_dir = cache_dir
        self.disk_cache = DiskCache(self.base_cache_dir)
        self.mem_cache.capacity = mem_limit
        self.logger.info(f"缓存系统配置更新: 目录={cache_dir}, 内存限制={mem_limit}")

    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        import re
        import hashlib
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

    def _build_disk_key_with_dates(self, base_key: str, df: pd.DataFrame) -> str:
        """为行情数据构建带日期范围的磁盘缓存 key"""
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return base_key
        start = df.index.min().strftime('%Y%m%d')
        end = df.index.max().strftime('%Y%m%d')
        return f"{base_key}_{start}_{end}"

    def _get_param_value(self, func: Callable, args: tuple, kwargs: dict, param_name: str, default: Any = '') -> Any:
        """从函数调用的位置参数和关键字参数中获取指定参数的值"""
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
                                 start_date_override: Optional[str] = None) -> Tuple[tuple, dict]:
        """构建增量更新调用的参数，正确处理位置参数和关键字参数的冲突
        
        当 start_date 作为位置参数传入 args 时，需要将其移到 kwargs 中
        以避免 "multiple values for argument" 错误。
        """
        new_kwargs = dict(kwargs)
        if start_date_override is not None:
            new_kwargs['start_date'] = start_date_override
        
        # 检查 start_date 是否在 args 位置参数中
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
        except (ValueError, TypeError) as e:
            self.logger.warning(f"[增量参数] 无法获取函数签名: {e}, func={getattr(func, '__name__', func)}, 回退简单处理")
            # 回退：无法解析签名，强制把所有 args 转为 kwargs 不可行
            # 但可以尝试删除 args 中与 kwargs 重复的参数
            # 最安全做法：直接只传 kwargs
            return (), new_kwargs
        
        self.logger.debug(f"[增量参数] 函数签名参数: {params}, args长度={len(args)}, kwargs keys={list(kwargs.keys())}")
        
        # 找到 start_date 在函数参数列表中的位置
        start_date_idx = None
        for i, name in enumerate(params):
            if name == 'start_date':
                start_date_idx = i
                break
        
        if start_date_idx is None:
            self.logger.debug(f"[增量参数] start_date 不在函数签名中，无需调整位置参数")
            return args, new_kwargs
        
        if start_date_idx >= len(args):
            self.logger.debug(f"[增量参数] start_date 索引 {start_date_idx} >= args长度 {len(args)}，无需调整")
            return args, new_kwargs
        
        # start_date 在 args 中，需要将其及之后的参数移到 kwargs
        new_args = args[:start_date_idx]
        for i in range(start_date_idx, len(args)):
            param_name = params[i] if i < len(params) else None
            if param_name and param_name not in new_kwargs:
                new_kwargs[param_name] = args[i]
        
        self.logger.debug(f"[增量参数] 调整后: new_args长度={len(new_args)}, new_kwargs keys={list(new_kwargs.keys())}")
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

    def execute_with_cache(self, namespace: str, cache_type: str, incremental: bool,
                           func: Callable, args: tuple, kwargs: dict) -> Any:
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

        # 查磁盘缓存：行情数据使用前缀匹配（文件名含日期范围）
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
                disk_end = disk_data.index.max().strftime('%Y-%m-%d') if isinstance(disk_data.index, pd.DatetimeIndex) else str(disk_data.index.max())

                if req_end and disk_end >= req_end:
                    self.stats['total_load_time_ms'] += (time.time() - start_time) * 1000
                    return self._filter_by_date(disk_data, req_start, req_end)

                self.logger.info(f"[{namespace}] 触发智能增量更新: {base_key}, 本地截止 {disk_end}, 需求截止 {req_end}")
                self.stats['incremental_merges'] += 1

                next_day = (pd.Timestamp(disk_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                inc_args, inc_kwargs = self._build_incremental_args(
                    func, args, kwargs, start_date_override=next_day
                )

                try:
                    new_data = func(*inc_args, **inc_kwargs)
                    merged_data = self._merge_incremental_data(disk_data, new_data)

                    # 写入磁盘时使用带日期范围的 key，并删除旧文件
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
                # 行情数据写入磁盘时使用带日期范围的 key
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
