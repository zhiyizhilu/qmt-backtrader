import os
import sys
import tempfile
import shutil
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cache import (
    MemCache, DiskCache, CacheIndexManager,
    SmartCacheManager, smart_cache, HAS_PYARROW
)


class TestMemCache:
    def test_put_and_get(self):
        mc = MemCache()
        mc.put("key1", "value1")
        assert mc.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        mc = MemCache()
        assert mc.get("nonexistent") is None

    def test_clear(self):
        mc = MemCache()
        mc.put("key1", "value1")
        mc.put("key2", "value2")
        mc.clear()
        assert mc.get("key1") is None
        assert mc.get("key2") is None

    def test_lru_eviction(self):
        mc = MemCache(capacity=3)
        mc.put("a", 1)
        mc.put("b", 2)
        mc.put("c", 3)
        mc.put("d", 4)
        assert mc.get("a") is None
        assert mc.get("b") == 2
        assert mc.get("c") == 3
        assert mc.get("d") == 4

    def test_lru_access_reorder(self):
        mc = MemCache(capacity=3)
        mc.put("a", 1)
        mc.put("b", 2)
        mc.put("c", 3)
        mc.get("a")
        mc.put("d", 4)
        assert mc.get("a") == 1
        assert mc.get("b") is None
        assert mc.get("c") == 3
        assert mc.get("d") == 4

    def test_put_overwrite(self):
        mc = MemCache()
        mc.put("key1", "old")
        mc.put("key1", "new")
        assert mc.get("key1") == "new"

    def test_capacity_one(self):
        mc = MemCache(capacity=1)
        mc.put("a", 1)
        mc.put("b", 2)
        assert mc.get("a") is None
        assert mc.get("b") == 2


class TestDiskCache:
    def setup_method(self):
        self.cache_dir = tempfile.mkdtemp()
        self.disk = DiskCache(self.cache_dir)

    def teardown_method(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_put_and_get_dataframe(self):
        df = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2)
        )
        fmt = "parquet" if HAS_PYARROW else "pkl"
        assert self.disk.put("test_ns", "test_key", df, fmt) is True
        result = self.disk.get("test_ns", "test_key", fmt)
        assert result is not None
        assert len(result) == 2
        assert list(result["close"]) == [1.0, 2.0]

    def test_put_and_get_pickle(self):
        data = {"key": "value", "number": 42}
        assert self.disk.put("test_ns", "test_key", data, "pkl") is True
        result = self.disk.get("test_ns", "test_key", "pkl")
        assert result == data

    def test_get_missing_returns_none(self):
        assert self.disk.get("test_ns", "nonexistent", "parquet") is None

    def test_put_none_returns_false(self):
        assert self.disk.put("test_ns", "key", None, "pkl") is False

    def test_put_yearly_and_get_yearly(self):
        df = pd.DataFrame(
            {"close": [10.0, 20.0, 30.0]},
            index=pd.date_range("2024-01-01", periods=3)
        )
        fmt = "parquet" if HAS_PYARROW else "pkl"
        assert self.disk.put_yearly("QMTDataProcessor", "000001.SZ", 2024, "1d", df, fmt) is True
        result = self.disk.get_yearly("QMTDataProcessor", "000001.SZ", 2024, "1d", fmt)
        assert result is not None
        assert len(result) == 3
        assert list(result["close"]) == [10.0, 20.0, 30.0]

    def test_put_yearly_wrong_year_returns_false(self):
        df = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2)
        )
        assert self.disk.put_yearly("QMTDataProcessor", "000001.SZ", 2023, "1d", df, "pkl") is False

    def test_get_yearly_missing_returns_none(self):
        assert self.disk.get_yearly("QMTDataProcessor", "000001.SZ", 2020, "1d", "parquet") is None

    def test_put_yearly_empty_df_returns_false(self):
        df = pd.DataFrame()
        assert self.disk.put_yearly("QMTDataProcessor", "000001.SZ", 2024, "1d", df, "pkl") is False

    def test_put_yearly_none_returns_false(self):
        assert self.disk.put_yearly("QMTDataProcessor", "000001.SZ", 2024, "1d", None, "pkl") is False

    def test_put_yearly_multi_year_isolation(self):
        df_2023 = pd.DataFrame(
            {"close": [5.0, 6.0]},
            index=pd.date_range("2023-06-01", periods=2)
        )
        df_2024 = pd.DataFrame(
            {"close": [7.0, 8.0]},
            index=pd.date_range("2024-06-01", periods=2)
        )
        fmt = "parquet" if HAS_PYARROW else "pkl"
        self.disk.put_yearly("QMTDataProcessor", "000001.SZ", 2023, "1d", df_2023, fmt)
        self.disk.put_yearly("QMTDataProcessor", "000001.SZ", 2024, "1d", df_2024, fmt)
        r2023 = self.disk.get_yearly("QMTDataProcessor", "000001.SZ", 2023, "1d", fmt)
        r2024 = self.disk.get_yearly("QMTDataProcessor", "000001.SZ", 2024, "1d", fmt)
        assert r2023 is not None and len(r2023) == 2
        assert r2024 is not None and len(r2024) == 2

    def test_delete(self):
        data = {"x": 1}
        self.disk.put("test_ns", "del_key", data, "pkl")
        assert self.disk.get("test_ns", "del_key", "pkl") is not None
        assert self.disk.delete("test_ns", "del_key", "pkl") is True
        assert self.disk.get("test_ns", "del_key", "pkl") is None


class TestCacheIndexManager:
    def setup_method(self):
        self.cache_dir = Path(tempfile.mkdtemp())
        self.idx = CacheIndexManager(self.cache_dir)

    def teardown_method(self):
        shutil.rmtree(str(self.cache_dir), ignore_errors=True)

    def test_update_and_get_market_index(self):
        self.idx.update_market_index("000001.SZ", "1d", 2024)
        years = self.idx.get_available_market_years("000001.SZ", "1d")
        assert 2024 in years

    def test_get_market_index_missing(self):
        years = self.idx.get_available_market_years("000001.SZ", "1d")
        assert years == []

    def test_update_and_get_financial_index(self):
        self.idx.update_financial_index("000001.SZ", "Balance", 2023)
        years = self.idx.get_available_financial_years("000001.SZ", "Balance")
        assert 2023 in years

    def test_save_and_load_index(self):
        self.idx.update_market_index("000001.SZ", "1d", 2024)
        self.idx.update_financial_index("000001.SZ", "Balance", 2023)
        self.idx.save_index()
        new_idx = CacheIndexManager(self.cache_dir)
        assert 2024 in new_idx.get_available_market_years("000001.SZ", "1d")
        assert 2023 in new_idx.get_available_financial_years("000001.SZ", "Balance")

    def test_update_market_index_no_duplicate(self):
        self.idx.update_market_index("000001.SZ", "1d", 2024)
        self.idx.update_market_index("000001.SZ", "1d", 2024)
        years = self.idx.get_available_market_years("000001.SZ", "1d")
        assert years.count(2024) == 1

    def test_update_market_index_multiple_years(self):
        self.idx.update_market_index("000001.SZ", "1d", 2022)
        self.idx.update_market_index("000001.SZ", "1d", 2023)
        self.idx.update_market_index("000001.SZ", "1d", 2024)
        years = self.idx.get_available_market_years("000001.SZ", "1d")
        assert years == [2022, 2023, 2024]

    def test_remove_market_index(self):
        self.idx.update_market_index("000001.SZ", "1d", 2024)
        self.idx.remove_market_index("000001.SZ", "1d")
        years = self.idx.get_available_market_years("000001.SZ", "1d")
        assert years == []

    def test_delisted_stock(self):
        self.idx.mark_delisted("000001.SZ", "2024-06-01")
        assert self.idx.is_delisted("000001.SZ") is True
        assert self.idx.get_delist_date("000001.SZ") == "2024-06-01"
        assert self.idx.is_delisted("000002.SZ") is False
        assert self.idx.get_delist_date("000002.SZ") is None

    def test_delisted_stock_unknown_date(self):
        self.idx.mark_delisted("000003.SZ")
        assert self.idx.is_delisted("000003.SZ") is True
        assert self.idx.get_delist_date("000003.SZ") is None

    def test_financial_nodata(self):
        self.idx.mark_financial_nodata("000001.SZ", "Balance")
        assert self.idx.is_financial_nodata("000001.SZ", "Balance") is True
        assert self.idx.is_financial_nodata("000001.SZ", "Income") is False

    def test_suspended_ranges(self):
        self.idx.mark_suspended("000001.SZ", [["2024-01-01", "2024-01-15"]])
        assert self.idx.is_suspended_on("000001.SZ", "2024-01-10") is True
        assert self.idx.is_suspended_on("000001.SZ", "2024-01-01") is True
        assert self.idx.is_suspended_on("000001.SZ", "2024-01-15") is True
        assert self.idx.is_suspended_on("000001.SZ", "2024-02-01") is False

    def test_market_raw_index(self):
        self.idx.update_market_raw_index("000001.SZ", "1d", 2024)
        years = self.idx.get_available_market_raw_years("000001.SZ", "1d")
        assert 2024 in years

    def test_checked_market_years(self):
        self.idx.update_checked_market_years("000001.SZ", "1d", [2023, 2024])
        checked = self.idx.get_checked_market_years("000001.SZ", "1d")
        assert 2023 in checked
        assert 2024 in checked


class TestSmartCacheManager:
    def setup_method(self):
        SmartCacheManager._instance = None
        self.cache_dir = tempfile.mkdtemp()
        self.mgr = SmartCacheManager()
        self.mgr.configure(self.cache_dir, mem_limit=500)

    def teardown_method(self):
        SmartCacheManager._instance = None
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_execute_with_cache_miss(self):
        call_count = 0

        def fetch_data(symbol, start_date, end_date):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame(
                {"close": [1.0, 2.0]},
                index=pd.date_range(start_date, periods=2)
            )

        result = self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )
        assert result is not None
        assert len(result) == 2
        assert call_count == 1
        assert self.mgr.stats['misses'] == 1

    def test_execute_with_cache_mem_hit(self):
        call_count = 0

        def fetch_data(symbol, start_date, end_date):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame(
                {"close": [1.0, 2.0]},
                index=pd.date_range(start_date, periods=2)
            )

        self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )

        result = self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )
        assert result is not None
        assert call_count == 1
        assert self.mgr.stats['mem_hits'] == 1

    def test_execute_with_cache_disk_hit(self):
        call_count = 0

        def fetch_data(symbol, start_date, end_date):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame(
                {"close": [1.0, 2.0]},
                index=pd.date_range(start_date, periods=2)
            )

        self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )

        self.mgr.mem_cache.clear()

        result = self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )
        assert result is not None
        assert call_count == 1
        assert self.mgr.stats['disk_hits'] == 1

    def test_execute_with_cache_miss_then_disk_then_mem(self):
        call_count = 0

        def fetch_data(symbol, start_date, end_date):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame(
                {"close": [1.0, 2.0]},
                index=pd.date_range(start_date, periods=2)
            )

        self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )

        self.mgr.mem_cache.clear()

        self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )
        assert self.mgr.stats['disk_hits'] == 1

        result = self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )
        assert result is not None
        assert self.mgr.stats['mem_hits'] == 1
        assert call_count == 1

    def test_execute_with_cache_different_args_separate_cache(self):
        call_count = 0

        def fetch_data(symbol, start_date, end_date):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame(
                {"close": [float(call_count)]},
                index=pd.date_range(start_date, periods=1)
            )

        r1 = self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000001.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )
        r2 = self.mgr.execute_with_cache(
            namespace="TestProcessor",
            cache_type="financial",
            incremental=False,
            func=fetch_data,
            args=("000002.SZ", "2024-01-01", "2024-12-31"),
            kwargs={}
        )
        assert call_count == 2
        assert r1["close"].iloc[0] != r2["close"].iloc[0]


class TestSmartCacheDecorator:
    def setup_method(self):
        from core.cache import cache_manager
        self.cache_dir = tempfile.mkdtemp()
        cache_manager.configure(self.cache_dir)
        cache_manager.mem_cache.clear()
        cache_manager.stats = {
            'mem_hits': 0, 'disk_hits': 0, 'misses': 0,
            'incremental_merges': 0, 'yearly_hits': 0, 'total_load_time_ms': 0.0
        }

    def teardown_method(self):
        from core.cache import cache_manager
        cache_manager.configure(os.path.join(os.getcwd(), '.cache'))
        cache_manager.mem_cache.clear()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_decorator_caches_result(self):
        from core.cache import cache_manager

        call_count = 0

        class MockProcessor:
            @smart_cache(cache_type='financial', incremental=False)
            def get_data(self, symbol, start_date, end_date):
                nonlocal call_count
                call_count += 1
                return pd.DataFrame(
                    {"close": [1.0, 2.0]},
                    index=pd.date_range(start_date, periods=2)
                )

        proc = MockProcessor()
        result1 = proc.get_data("000001.SZ", "2024-01-01", "2024-12-31")
        assert result1 is not None
        assert call_count == 1
        assert cache_manager.stats['misses'] == 1

        result2 = proc.get_data("000001.SZ", "2024-01-01", "2024-12-31")
        assert result2 is not None
        assert call_count == 1
        assert cache_manager.stats['mem_hits'] == 1

    def test_decorator_financial_namespace(self):
        from core.cache import cache_manager

        class MockProcessor:
            @smart_cache(cache_type='financial', incremental=False)
            def get_data(self, symbol, start_date, end_date):
                return pd.DataFrame(
                    {"close": [1.0]},
                    index=pd.date_range(start_date, periods=1)
                )

        proc = MockProcessor()
        proc.get_data("000001.SZ", "2024-01-01", "2024-12-31")
        assert cache_manager.stats['misses'] == 1
