from __future__ import annotations
import datetime
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Type
from core.data import DataProcessor, QMTDataProcessor, OpenDataProcessor, create_data_processor
from core.data.futu import FutuServiceError
from core.strategy_logic import StrategyLogic
from core.stock_selection import StockSelectionStrategy
from core.financial_data import FinancialDataCache, FinancialDataAdapter
from core.analyzer import PerformanceAnalyzer
from core.stock_lifecycle import get_lifecycle_manager
from api.base_api import BaseAPI
from engine import BacktestEngine, SimulatedBroker


class BacktestAPI(BaseAPI):
    """回测API - 通过自研回测引擎驱动策略

    策略逻辑与执行环境解耦：
    - 策略类继承StrategyLogic，仅定义交易逻辑
    - BacktestAPI负责创建引擎，将StrategyLogic桥接到回测引擎
    - 数据访问通过EngineDataAdapter
    - 交易执行通过EngineExecutor
    - 回测结果通过PerformanceAnalyzer构建BacktestingResult

    使用方式：
        # 简洁模式 — 策略自带 backtest_config，add_strategy 自动加载数据
        api = BacktestAPI()
        api.configure(**backtest_config, period='1d')
        api.add_strategy(strategy_class, **default_kwargs)
        api.run()
        api.show_report()

        # 高级模式 — 手动控制数据加载
        api = BacktestAPI()
        api.set_cash(200000)
        api.add_data('000001.SZ', '2025-01-01', '2026-04-17', period='1d')
        api.add_strategy(strategy_class, **default_kwargs)
        api.run()

        # 指定富途数据源
        api = BacktestAPI(data_source='futu')
        api.configure(**backtest_config, period='1d')
        api.add_strategy(strategy_class, **default_kwargs)
        api.run()
    """

    def __init__(self, proxy: str = '', data_source: str = 'qmt'):
        super().__init__()
        self._engine = BacktestEngine()
        self._broker = SimulatedBroker()
        self._engine.set_broker(self._broker)
        self._data_source = data_source
        self.data_processor = create_data_processor(
            fallback_to_simulated=False, proxy=proxy, data_source=data_source
        )
        self._market_data_processor = create_data_processor(
            fallback_to_simulated=False, proxy=proxy, data_source=data_source
        )
        self._financial_data_processor = create_data_processor(
            fallback_to_simulated=False, proxy=proxy, use_opendata=False
        )
        self._opendata_processor = OpenDataProcessor(fallback_to_simulated=False)
        self._symbols: List[str] = []
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._strategy_logic_class: Optional[Type[StrategyLogic]] = None
        self._strategy_kwargs: Dict[str, Any] = {}
        self._initial_cash: float = 0.0
        self._analyzer = PerformanceAnalyzer()
        self._backtest_result = None
        self._period: str = '1d'
        self._trade_start_date: Optional[str] = None
        self._benchmark: str = '000300.SH'
        self._data_start_date: Optional[str] = None
        self._data_end_date: Optional[str] = None
        self._financial_adapter: Optional[FinancialDataAdapter] = None
        self._stock_pool: Optional[List[str]] = None
        self._compare_symbols: List[str] = []
        self._ai_mode: bool = False
        self._no_record: bool = False
        self._strategy_name: str = ''
        self._log_file: str = ''
        self._backtest_config: Dict[str, Any] = {}
        self._lifecycle_manager = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def configure(self, cash: float = 200000, commission: float = 0.0001,
                  start_date: Optional[str] = None, end_date: Optional[str] = None,
                  data_lookback_days: int = 40, benchmark: str = '000300.SH',
                  period: str = '1d', trade_start_date: Optional[str] = None,
                  slippage: float = 0.0, compare_symbols: Optional[List[str]] = None,
                  **kwargs):
        self.set_cash(cash)
        self.set_commission(commission)
        self.set_slippage(slippage)
        self.set_benchmark(benchmark)
        self._period = period
        self._compare_symbols = compare_symbols or []

        if start_date and end_date:
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')

            data_start_dt = start_dt - datetime.timedelta(days=data_lookback_days)
            data_start_date = data_start_dt.strftime('%Y-%m-%d')
            self.set_data_range(data_start_date, end_date, period=period)
            trade_start = trade_start_date or start_date
            self.set_trade_start_date(trade_start)
            self.logger.info(
                f'[configure] 数据范围={data_start_date}~{end_date}, '
                f'交易起始日={trade_start}, 前移天数={data_lookback_days}'
            )

    def set_data_range(self, start_date: str, end_date: str, period: str = '1d'):
        self._data_start_date = start_date
        self._data_end_date = end_date
        self._period = period

    def add_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d",
                 skip_if_late_start: bool = False):
        self._period = period
        self._data_start_date = start_date
        self._data_end_date = end_date

        data = None

        try:
            data = self._market_data_processor.get_data(symbol, start_date, end_date, period)
        except FutuServiceError:
            raise
        except Exception as e:
            self.logger.warning(f'[add_data] {symbol}: 行情数据获取失败({e})')

        if data is not None and not data.empty:
            self._engine.add_data(symbol, data)
            self._symbols.append(symbol)
            self._data_cache[symbol] = data
        else:
            self.logger.warning(f'[add_data] {symbol}: 数据为空! start={start_date}, end={end_date}, period={period}')

    def load_financial_data(self, stock_list: Optional[List[str]] = None,
                            table_list: Optional[List[str]] = None,
                            sector: Optional[str] = None,
                            start_time: str = '', end_time: str = '',
                            report_type: str = 'announce_time') -> FinancialDataAdapter:
        import time as _time
        overall_start = _time.time()

        if not start_time and self._data_start_date:
            start_time = self._data_start_date
        if not end_time and self._data_end_date:
            end_time = self._data_end_date

        if stock_list is None and sector:
            if start_time and end_time:
                self.logger.info(f"[阶段1/6] 获取板块 '{sector}' 回测期间全部历史成分股 "
                                 f"({start_time} ~ {end_time})...")
                stock_list = self._financial_data_processor.get_all_historical_stocks_in_range(
                    sector, start_date=start_time, end_date=end_time
                )
                self.logger.info(f"[阶段1/6] 板块 '{sector}' 回测期间共涉及 {len(stock_list)} 只历史成分股")
            else:
                hist_date = start_time if start_time else end_time
                if hist_date:
                    self.logger.info(f"[阶段1/6] 获取板块 '{sector}' 历史成分股 (基准日期: {hist_date})...")
                    stock_list = self._financial_data_processor.get_historical_stock_list(sector, date=hist_date)
                    self.logger.info(f"[阶段1/6] 板块 '{sector}' 获取到 {len(stock_list)} 只历史成分股")
                else:
                    self.logger.info(f"[阶段1/6] 获取板块 '{sector}' 成分股...")
                    stock_list = self._financial_data_processor.get_stock_list(sector)
                    self.logger.info(f"[阶段1/6] 板块 '{sector}' 获取到 {len(stock_list)} 只股票")
        else:
            self.logger.info(f"[阶段1/6] 使用指定股票列表: {len(stock_list or [])} 只")

        if not stock_list:
            raise ValueError("必须指定 stock_list 或 sector")

        try:
            if self._lifecycle_manager is None:
                self._lifecycle_manager = get_lifecycle_manager(
                    xtdata=self._financial_data_processor.xtdata if hasattr(self._financial_data_processor, 'xtdata') else None
                )
            self.logger.info(f"[阶段1.5/6] 更新股票生命周期数据: {len(stock_list)} 只股票...")
            self._lifecycle_manager.batch_update(stock_list)
            delisted_count = sum(1 for s in stock_list if self._lifecycle_manager.is_delisted(s))
            late_list_count = sum(
                1 for s in stock_list
                if self._lifecycle_manager.is_listed_after(s, self._data_start_date or start_time or '')
            )
            self.logger.info(
                f"[阶段1.5/6] 生命周期数据更新完成: "
                f"{delisted_count} 只退市, {late_list_count} 只晚上市"
            )

            from core.cache import cache_manager
            for s in stock_list:
                if self._lifecycle_manager.is_delisted(s) and not cache_manager.index_manager.is_delisted(s):
                    delist_date = self._lifecycle_manager.get_delist_date(s) or ''
                    cache_manager.index_manager.mark_delisted(s, delist_date)
            cache_manager.index_manager.save_index()
        except Exception as e:
            self.logger.warning(f"[阶段1.5/6] 生命周期数据更新失败: {e}，将按原有逻辑运行")

        tables = table_list or ['Balance', 'Income', 'CashFlow', 'Capital',
                                'HolderNum', 'Top10Holder', 'Top10FlowHolder', 'Pershareindex']

        from core.cache import cache_manager
        namespace = f"{self._financial_data_processor.__class__.__name__}_Financial"
        time_suffix = f"_{start_time}_{end_time}" if start_time or end_time else ""
        ns_dir = cache_manager.disk_cache.get_namespace_dir(namespace)

        req_years = []
        if start_time and end_time:
            try:
                req_years = list(range(pd.Timestamp(start_time).year, pd.Timestamp(end_time).year + 1))
            except Exception:
                pass

        cached_stocks = set()
        uncached_stocks = []

        for stock in stock_list:
            all_tables_cached = True
            for table in tables:
                table_suffix = f"{table}_{report_type}"

                if req_years:
                    available_years = cache_manager.index_manager.get_available_financial_years(stock, table_suffix)
                    if not available_years:
                        available_years = cache_manager.disk_cache.list_yearly_files(namespace, stock, table_suffix)
                    checked_years = cache_manager.index_manager.get_checked_financial_years(stock, table_suffix)
                    missing_years = set(req_years) - set(available_years) - set(checked_years)
                    if missing_years:
                        all_tables_cached = False
                        break
                else:
                    available_years = cache_manager.index_manager.get_available_financial_years(stock, table_suffix)
                    if not available_years:
                        available_years = cache_manager.disk_cache.list_yearly_files(namespace, stock, table_suffix)
                    if not available_years:
                        cache_key = f"{stock}{time_suffix}_{table}_{report_type}"
                        file_path = ns_dir / f"{cache_key}.parquet"
                        if not file_path.exists():
                            pkl_path = ns_dir / f"{cache_key}.pkl"
                            if not pkl_path.exists():
                                all_tables_cached = False
                                break
            if all_tables_cached:
                cached_stocks.add(stock)
            else:
                uncached_stocks.append(stock)

        if uncached_stocks:
            self.logger.info(
                f"[阶段2/6] 下载缺失的财务数据: "
                f"{len(uncached_stocks)} 只股票缺少缓存 (已有 {len(cached_stocks)} 只缓存)"
            )
            self._financial_data_processor.download_financial_data(
                uncached_stocks, table_list, start_time, end_time
            )
            self._financial_data_processor.get_financial_data(
                uncached_stocks, table_list, start_time, end_time, report_type
            )
        else:
            self.logger.info(f"[阶段2/6] 全部 {len(stock_list)} 只股票已有磁盘缓存，跳过下载")

        cache = FinancialDataCache(
            data_processor=self._financial_data_processor,
            report_type=report_type,
            start_time=start_time,
            end_time=end_time,
        )
        self._financial_adapter = FinancialDataAdapter(cache)
        self._stock_pool = stock_list

        if sector:
            from core.data.index_constituent import IndexConstituentManager
            sector_benchmark = IndexConstituentManager.SECTOR_TO_INDEX.get(sector)
            if sector_benchmark:
                self._benchmark = sector_benchmark
                self.logger.info(f"[基准] 根据 sector='{sector}' 自动设置基准为 {sector_benchmark}")

            if hasattr(self._financial_data_processor, '_index_constituent_mgr') \
                    and self._financial_data_processor._index_constituent_mgr:
                self._financial_adapter.set_index_constituent_mgr(
                    self._financial_data_processor._index_constituent_mgr, sector
                )
                self.logger.info(f"[阶段3/6] 已设置指数成分股动态查询: sector='{sector}'")

            if hasattr(self._financial_data_processor, '_industry_constituent_mgr') \
                    and self._financial_data_processor._industry_constituent_mgr:
                self._financial_adapter.set_industry_constituent_mgr(
                    self._financial_data_processor._industry_constituent_mgr
                )
                self.logger.info(f"[阶段3/6] 已设置行业分类动态查询")

        self.logger.info(
            f"[阶段3/6] 财务数据适配器创建完成(按需加载模式): {len(stock_list)} 只股票待查询"
        )

        self.logger.info(f"[阶段4/6] 获取申万行业分类映射...")
        try:
            hist_date = start_time if start_time else end_time
            if not hist_date:
                hist_date = pd.Timestamp.now().strftime('%Y-%m-%d')

            self.logger.info(f"[阶段4/6] 使用历史行业数据 (基准日期: {hist_date})...")
            industry_mapping = self._financial_data_processor.get_historical_industry_mapping(
                stock_list=stock_list,
                date=hist_date,
                classification='申银万国行业分类标准'
            )

            if industry_mapping:
                self._financial_adapter.set_industry_mapping(industry_mapping)
                self.logger.info(f"[阶段4/6] 行业分类映射已加载: {len(industry_mapping)} 只股票, {len(set(industry_mapping.values()))} 个行业")
            else:
                self.logger.warning(f"[阶段4/6] 行业分类映射为空")
        except Exception as e:
            self.logger.warning(f"[阶段4/6] 行业分类映射加载失败: {e}，策略将无法使用行业筛选")

        self.logger.info(f"[阶段5/6] 获取分红数据，共 {len(stock_list)} 只股票...")
        try:
            dividend_data = self._financial_data_processor.get_dividend_data(stock_list)
            if dividend_data:
                self._financial_adapter.set_dividend_data(dividend_data)
                self.logger.info(f"[阶段5/6] 分红数据已加载: {len(dividend_data)} 只股票")
            else:
                self.logger.warning(f"[阶段5/6] 分红数据为空，策略将无法计算股息率")
        except Exception as e:
            self.logger.warning(f"[阶段5/6] 分红数据加载失败: {e}，策略将无法计算股息率")

        overall_elapsed = _time.time() - overall_start
        self.logger.info(
            f"财务数据加载全部完成: {len(stock_list)} 只股票(按需加载), "
            f"总耗时 {overall_elapsed:.1f}秒"
        )

        return self._financial_adapter

    def load_stock_pool(self, sector: str = '沪深A股',
                        stock_list: Optional[List[str]] = None) -> List[str]:
        if stock_list:
            self._stock_pool = stock_list
        else:
            start_date = self._data_start_date
            end_date = self._data_end_date
            if start_date and end_date:
                self.logger.info(f"获取板块 '{sector}' 回测期间全部历史成分股 "
                                 f"({start_date} ~ {end_date})...")
                self._stock_pool = self.data_processor.get_all_historical_stocks_in_range(
                    sector, start_date=start_date, end_date=end_date
                )
                self.logger.info(f"板块 '{sector}' 回测期间共涉及 {len(self._stock_pool)} 只历史成分股")
            else:
                self._stock_pool = self.data_processor.get_stock_list(sector)
                self.logger.info(f"板块 '{sector}' 获取到 {len(self._stock_pool)} 只股票")

        self.logger.info(f"股票池加载完成: {len(self._stock_pool)} 只股票")
        return self._stock_pool

    def add_stock_selection_strategy(self, strategy_class: Type[StockSelectionStrategy],
                                     stock_pool: Optional[List[str]] = None,
                                     **kwargs):
        if not self._data_start_date or not self._data_end_date:
            today = datetime.date.today()
            self._data_end_date = today.strftime('%Y-%m-%d')
            self._data_start_date = (today - datetime.timedelta(days=400)).strftime('%Y-%m-%d')
            self.logger.info(f"数据范围未设置，使用默认值: {self._data_start_date} ~ {self._data_end_date}")

        pool = stock_pool or self._stock_pool
        if not pool:
            raise ValueError("请先调用 load_stock_pool 或 load_financial_data 设置股票池")

        if not self._financial_adapter:
            self.logger.warning("财务数据适配器未创建，策略将无法访问财报数据。"
                                "建议先调用 load_financial_data()")

        kwargs['stock_pool'] = pool

        self._strategy_logic_class = strategy_class
        self._strategy_kwargs = kwargs

        loaded_count, failed_count, skipped_count = self._load_pool_market_data(pool)

        self._preload_auxiliary_data(pool)

    def _load_pool_market_data(self, pool: List[str]):
        import time as _time
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor, as_completed
        load_start = _time.time()

        loaded_count = 0
        failed_count = 0
        skipped_count = 0
        total = len(pool)

        self.logger.info(f"开始加载股票池行情数据: {total} 只股票, 周期={self._period}")

        if self._lifecycle_manager is None:
            try:
                self._lifecycle_manager = get_lifecycle_manager(
                    xtdata=self.data_processor.xtdata if hasattr(self.data_processor, 'xtdata') else None
                )
            except Exception:
                pass

        symbols_to_load = [s for s in pool if s not in self._symbols]
        skipped_count = total - len(symbols_to_load)

        def _fetch_market_data(symbol):
            try:
                lifecycle_mgr = self._lifecycle_manager
                if lifecycle_mgr:
                    effective_start, effective_end = lifecycle_mgr.get_effective_date_range(
                        symbol, self._data_start_date, self._data_end_date
                    )
                else:
                    effective_start, effective_end = self._data_start_date, self._data_end_date

                if effective_start is None or effective_end is None:
                    self.logger.debug(f"跳过 {symbol}: 在回测区间内无有效数据(退市或未上市)")
                    return (symbol, None, None)

                data = None
                try:
                    data = self._market_data_processor.get_data(symbol, effective_start, effective_end, self._period)
                except FutuServiceError:
                    raise
                except Exception:
                    pass
                return (symbol, data, None)
            except FutuServiceError:
                raise
            except Exception as e:
                return (symbol, None, e)

        if symbols_to_load:
            max_workers = min(8, len(symbols_to_load))
            data_results = {}
            fetch_errors = {}

            if max_workers <= 1 or len(symbols_to_load) <= 10:
                for i, symbol in enumerate(symbols_to_load, 1):
                    sym, data, err = _fetch_market_data(symbol)
                    if err:
                        fetch_errors[sym] = err
                    else:
                        data_results[sym] = data
                    if i % 100 == 0 or i == len(symbols_to_load):
                        self.logger.info(f"[ {i} / {len(symbols_to_load)} ] 行情数据获取进度")
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_fetch_market_data, s): s for s in symbols_to_load}
                    done_count = 0
                    for future in as_completed(futures):
                        sym, data, err = future.result()
                        if err:
                            fetch_errors[sym] = err
                        else:
                            data_results[sym] = data
                        done_count += 1
                        if done_count % 100 == 0 or done_count == len(symbols_to_load):
                            self.logger.info(f"[ {done_count} / {len(symbols_to_load)} ] 行情数据获取进度")

            for symbol in symbols_to_load:
                if symbol in fetch_errors:
                    failed_count += 1
                    if failed_count <= 3:
                        self.logger.debug(f"加载 {symbol} 行情数据失败: {fetch_errors[symbol]}")
                    continue

                data = data_results.get(symbol)
                if data is None or (hasattr(data, 'empty') and data.empty):
                    failed_count += 1
                    continue

                try:
                    self._engine.add_data(symbol, data)
                    self._symbols.append(symbol)
                    self._data_cache[symbol] = data
                    loaded_count += 1
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        self.logger.debug(f"注册 {symbol} 数据源失败: {e}")

        load_elapsed = _time.time() - load_start
        self.logger.info(
            f"股票池行情数据加载完成: 成功 {loaded_count}, 失败 {failed_count}, "
            f"已存在 {skipped_count}, 耗时 {load_elapsed:.1f}秒"
        )
        return loaded_count, failed_count, skipped_count

    def _preload_auxiliary_data(self, pool: List[str]):
        import time as _time
        from concurrent.futures import ThreadPoolExecutor

        if self._financial_adapter and self._financial_adapter.cache:
            preload_start = _time.time()
            strategy_tables = ['Pershareindex']
            self.logger.info(
                f"预加载财务数据到内存: {len(pool)} 只股票, 表={strategy_tables}"
            )

            raw_download_future = None
            raw_executor = None
            if self._market_data_processor:
                raw_executor = ThreadPoolExecutor(max_workers=4)

                def _download_raw_data():
                    raw_start = _time.time()
                    raw_loaded = 0
                    raw_skipped = 0
                    raw_nodata = 0

                    from core.cache import cache_manager as _cm
                    idx = _cm.index_manager

                    symbols_to_fetch = []
                    for symbol in pool:
                        if idx.is_market_raw_nodata(symbol, self._period):
                            raw_nodata += 1
                            continue
                        symbols_to_fetch.append(symbol)

                    if raw_nodata > 0:
                        self.logger.info(
                            f"不复权行情预下载: 跳过 {raw_nodata} 只无数据股票(黑名单), "
                            f"剩余 {len(symbols_to_fetch)} 只待下载"
                        )

                    def _fetch_one(sym):
                        try:
                            df = self._market_data_processor.get_raw_data(
                                sym, self._data_start_date, self._data_end_date, self._period,
                            )
                            if df is not None and not df.empty:
                                return sym, True, False
                            else:
                                return sym, False, True
                        except Exception:
                            return sym, False, True

                    completed = 0
                    total = len(symbols_to_fetch)
                    if total <= 10:
                        for sym in symbols_to_fetch:
                            _, loaded, skipped = _fetch_one(sym)
                            if loaded:
                                raw_loaded += 1
                            if skipped:
                                raw_skipped += 1
                                idx.mark_market_raw_nodata(sym, self._period)
                            completed += 1
                    else:
                        from concurrent.futures import as_completed as _as_completed
                        futures = {raw_executor.submit(_fetch_one, s): s for s in symbols_to_fetch}
                        for future in _as_completed(futures):
                            sym, loaded, skipped = future.result()
                            if loaded:
                                raw_loaded += 1
                            if skipped:
                                raw_skipped += 1
                                idx.mark_market_raw_nodata(sym, self._period)
                            completed += 1
                            if completed % 100 == 0 or completed == total:
                                self.logger.info(f"[ {completed} / {total} ] 不复权行情预下载进度")

                    try:
                        idx.save_index()
                    except Exception:
                        pass

                    raw_elapsed = _time.time() - raw_start
                    self.logger.info(
                        f"不复权行情数据预下载完成: {raw_loaded} 已缓存, "
                        f"{raw_skipped} 跳过, {raw_nodata} 黑名单跳过, 耗时 {raw_elapsed:.1f}秒"
                    )
                    return raw_loaded, raw_skipped

                raw_download_future = raw_executor.submit(_download_raw_data)
                self.logger.info(f"预下载不复权行情数据: {len(pool)} 只股票 (与财务数据预加载并行)")

            self._financial_adapter.cache.preload_stocks(pool, strategy_tables)
            preload_elapsed = _time.time() - preload_start
            self.logger.info(
                f"财务数据预加载完成: {len(pool)} 只股票, 耗时 {preload_elapsed:.1f}秒"
            )

            if raw_download_future is not None and raw_executor is not None:
                try:
                    raw_download_future.result()
                except Exception as e:
                    self.logger.warning(f"不复权行情预下载异常: {e}")
                finally:
                    raw_executor.shutdown(wait=False)

        elif self._market_data_processor:
            self._preload_raw_market_data(pool)

    def _preload_raw_market_data(self, pool: List[str]):
        import time as _time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        raw_start = _time.time()
        raw_loaded = 0
        raw_skipped = 0
        raw_nodata = 0
        self.logger.info(f"预下载不复权行情数据: {len(pool)} 只股票")

        from core.cache import cache_manager as _cm
        idx = _cm.index_manager

        symbols_to_fetch = []
        for symbol in pool:
            if idx.is_market_raw_nodata(symbol, self._period):
                raw_nodata += 1
                continue
            symbols_to_fetch.append(symbol)

        if raw_nodata > 0:
            self.logger.info(
                f"不复权行情预下载: 跳过 {raw_nodata} 只无数据股票(黑名单), "
                f"剩余 {len(symbols_to_fetch)} 只待下载"
            )

        def _fetch_one(sym):
            try:
                df = self._market_data_processor.get_raw_data(
                    sym, self._data_start_date, self._data_end_date, self._period,
                )
                if df is not None and not df.empty:
                    return sym, True, False
                else:
                    return sym, False, True
            except Exception:
                return sym, False, True

        completed = 0
        total = len(symbols_to_fetch)
        if total <= 10:
            for sym in symbols_to_fetch:
                _, loaded, skipped = _fetch_one(sym)
                if loaded:
                    raw_loaded += 1
                if skipped:
                    raw_skipped += 1
                    idx.mark_market_raw_nodata(sym, self._period)
                completed += 1
        else:
            max_workers = min(4, total)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fetch_one, s): s for s in symbols_to_fetch}
                for future in as_completed(futures):
                    sym, loaded, skipped = future.result()
                    if loaded:
                        raw_loaded += 1
                    if skipped:
                        raw_skipped += 1
                        idx.mark_market_raw_nodata(sym, self._period)
                    completed += 1
                    if completed % 100 == 0 or completed == total:
                        self.logger.info(f"[ {completed} / {total} ] 不复权行情预下载进度")

        try:
            idx.save_index()
        except Exception:
            pass

        raw_elapsed = _time.time() - raw_start
        self.logger.info(
            f"不复权行情数据预下载完成: {raw_loaded} 已缓存, "
            f"{raw_skipped} 跳过, {raw_nodata} 黑名单跳过, 耗时 {raw_elapsed:.1f}秒"
        )

    def add_strategy(self, strategy_logic_class: Type[StrategyLogic], **kwargs):
        self._strategy_logic_class = strategy_logic_class
        self._strategy_kwargs = kwargs

        temp_logic = strategy_logic_class(**kwargs)
        symbols_needed = temp_logic.get_symbols()

        if not self._data_start_date or not self._data_end_date:
            today = datetime.date.today()
            self._data_end_date = today.strftime('%Y-%m-%d')
            self._data_start_date = (today - datetime.timedelta(days=400)).strftime('%Y-%m-%d')
            self.logger.info(f"数据范围未设置，使用默认值: {self._data_start_date} ~ {self._data_end_date}")

        for symbol in symbols_needed:
            if symbol not in self._symbols:
                self.add_data(symbol, self._data_start_date, self._data_end_date, self._period)

    def set_cash(self, cash: float):
        self._initial_cash = cash
        self._broker.setcash(cash)
        self._broker.set_checksubmit(False)
        self._broker.set_coc(True)

    def set_trade_start_date(self, trade_start_date: str):
        self._trade_start_date = trade_start_date

    def set_benchmark(self, benchmark: str):
        self._benchmark = benchmark

    def set_commission(self, commission: float):
        self._broker.setcommission(commission)

    def set_slippage(self, slippage: float):
        if slippage > 0:
            self._broker.set_slippage_perc(slippage)

    def add_analyzer(self, analyzer_class, **kwargs):
        pass

    def _ensure_default_analyzers(self):
        pass

    def run(self):
        if self._initial_cash == 0.0:
            self.set_cash(200000)

        if self._strategy_logic_class is None:
            self.logger.warning('[run] 未设置策略，无法运行回测')
            return None

        self.logger.info(
            f'[run] 回测启动: symbols={self._symbols}, period={self._period}, '
            f'数据范围={self._data_start_date}~{self._data_end_date}, '
            f'交易起始日={self._trade_start_date}, '
            f'初始资金={self._initial_cash}, '
            f'策略={self._strategy_logic_class.__name__}, '
            f'数据源数量={len(self._symbols)}'
        )

        strategy_logic = self._strategy_logic_class(**self._strategy_kwargs)

        if self._financial_adapter:
            strategy_logic.set_financial_data_adapter(self._financial_adapter)

        if hasattr(self, '_market_data_processor') and self._market_data_processor:
            strategy_logic.set_data_processor(self._market_data_processor)

        if self._data_start_date:
            strategy_logic._data_start_date = self._data_start_date
        if self._data_end_date:
            strategy_logic._data_end_date = self._data_end_date

        self._engine.set_strategy(strategy_logic)
        self._engine.set_trade_start_date(self._trade_start_date)
        self._engine.set_period(self._period)

        self._load_trading_calendar()

        engine_result = self._engine.run()

        if engine_result:
            strategy_params = self._build_strategy_params()
            self._backtest_result = self._analyzer.build_result_from_engine(
                engine_result,
                strategy_params=strategy_params,
                show_kline=True,
                trade_start_date=self._trade_start_date,
                build_close_prices=not self._ai_mode,
            )

            self._fetch_benchmark_data()
            self._fetch_compare_data()
            self._log_summary(engine_result)

            if not self._no_record:
                self._auto_record()

        return engine_result

    def _log_summary(self, engine_result):
        try:
            initial = engine_result.initial_cash
            final_value = engine_result.final_value
            total_return = (final_value - initial) / initial * 100 if initial > 0 else 0.0

            self.logger.info(f"初始资金: {initial}")
            self.logger.info(f"最终资金: {final_value:.2f}")
            self.logger.info(f"总收益率: {total_return:.2f}%")

            if self._backtest_result and self._backtest_result.df is not None:
                sharpe = self._backtest_result.sharpe_ratio()
                max_dd = self._backtest_result.max_drawdown()
                self.logger.info(f"夏普比率: {sharpe:.2f}")
                self.logger.info(f"最大回撤: {max_dd * 100:.2f}%")

            trade_count = len(engine_result.trade_records)
            self.logger.info(f"交易次数: {trade_count}")
        except Exception as e:
            self.logger.warning(f"回测摘要输出失败: {e}")

    def _load_trading_calendar(self):
        if not self._benchmark or not self._data_start_date or not self._data_end_date:
            return
        try:
            benchmark_data = None
            try:
                benchmark_data = self._market_data_processor.get_data(
                    self._benchmark,
                    self._data_start_date,
                    self._data_end_date,
                    "1d",
                )
            except Exception as e:
                self.logger.warning(f"交易日历基准数据获取失败: {self._benchmark}, {e}")

            if benchmark_data is None or benchmark_data.empty:
                self.logger.warning(f"交易日历基准数据为空: {self._benchmark}，将使用默认判断逻辑")
                return

            if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                benchmark_data.index = pd.to_datetime(benchmark_data.index)

            self._benchmark_df = benchmark_data

            trading_dates = set()
            for ts in benchmark_data.index:
                trading_dates.add(pd.Timestamp(ts).date())

            self._engine.set_trading_dates(trading_dates)
            self.logger.info(
                f"交易日历已加载: 基准={self._benchmark}, "
                f"交易日数量={len(trading_dates)}, "
                f"范围={min(trading_dates)}~{max(trading_dates)}"
            )
        except Exception as e:
            self.logger.warning(f"交易日历加载失败: {e}，将使用默认判断逻辑")

    def _fetch_benchmark_data(self):
        if self._backtest_result is None or not self._benchmark:
            return
        if not self._data_start_date or not self._data_end_date:
            return
        try:
            benchmark_data = getattr(self, '_benchmark_df', None)
            if benchmark_data is None or benchmark_data.empty:
                benchmark_data = None
                try:
                    benchmark_data = self._market_data_processor.get_data(
                        self._benchmark,
                        self._data_start_date,
                        self._data_end_date,
                        "1d",
                    )
                    if benchmark_data is not None and not benchmark_data.empty:
                        self.logger.info(f"基准数据获取成功: {self._benchmark}, {len(benchmark_data)}条")
                except Exception as e:
                    self.logger.warning(f"基准数据获取失败: {self._benchmark}, {e}")

            if benchmark_data is None or benchmark_data.empty:
                self.logger.warning(f"基准数据获取失败: {self._benchmark}")

            if benchmark_data is not None and not benchmark_data.empty:
                if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                    benchmark_data.index = pd.to_datetime(benchmark_data.index)
                keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in benchmark_data.columns]
                benchmark_data = benchmark_data[keep_cols].copy()
                self._backtest_result.benchmark_df = benchmark_data
                self._backtest_result.benchmark_symbol = self._benchmark
            else:
                self.logger.warning(f"基准数据获取失败: {self._benchmark}, 所有数据源均无数据")
        except Exception as e:
            self.logger.warning(f"基准数据获取异常: {self._benchmark}, {e}")

    def _fetch_compare_data(self):
        if not self._data_start_date or not self._data_end_date:
            return
        if self._backtest_result is None:
            return

        compare_symbols = list(self._compare_symbols) if self._compare_symbols else []

        if not compare_symbols:
            return

        compare_data = {}
        for symbol in compare_symbols:
            try:
                df = self._market_data_processor.get_data(
                    symbol,
                    self._data_start_date,
                    self._data_end_date,
                    "1d",
                )
                if df is not None and not df.empty:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                    compare_data[symbol] = df[keep_cols].copy()
                    first_dt = df.index[0]
                    last_dt = df.index[-1]
                    self.logger.info(
                        f"对比数据获取成功: {symbol}, {len(df)}条, "
                        f"日期范围: {first_dt.strftime('%Y-%m-%d')} ~ {last_dt.strftime('%Y-%m-%d')}"
                    )
                else:
                    self.logger.warning(f"对比数据为空: {symbol}")
            except Exception as e:
                self.logger.warning(f"对比数据获取失败: {symbol}, {e}")

        self._backtest_result.compare_data = compare_data

    def get_result(self):
        return self._backtest_result

    def set_ai_mode(self, enabled: bool):
        self._ai_mode = enabled
        try:
            from utils.report import set_ai_mode as set_report_ai_mode
            set_report_ai_mode(enabled)
        except ImportError:
            pass
        self.logger.info(f"AI自动运行模式: {'开启' if enabled else '关闭'}")

    def is_ai_mode(self) -> bool:
        return self._ai_mode

    def set_no_record(self, no_record: bool):
        self._no_record = no_record

    def set_strategy_name(self, name: str):
        self._strategy_name = name

    def set_log_file(self, log_file: str):
        self._log_file = log_file

    def set_backtest_config(self, config: Dict[str, Any]):
        self._backtest_config = config

    def _auto_record(self):
        try:
            from utils.backtest_recorder import BacktestRecorder
            recorder = BacktestRecorder()
            strategy_name = self._strategy_name or (
                self._strategy_logic_class.__name__ if self._strategy_logic_class else 'unknown'
            )
            config = dict(self._backtest_config)
            config.setdefault('initial_cash', self._initial_cash)
            config.setdefault('period', self._period)
            config.setdefault('benchmark', self._benchmark)
            if self._data_start_date:
                config.setdefault('data_start_date', self._data_start_date)
            if self._data_end_date:
                config.setdefault('data_end_date', self._data_end_date)
            if self._trade_start_date:
                config.setdefault('trade_start_date', self._trade_start_date)
            run_id = recorder.record(self._backtest_result, strategy_name, config, log_file=self._log_file)
            self.logger.info(f'[run] 回测结果已自动记录: {run_id}')
        except Exception as e:
            self.logger.warning(f'[run] 回测结果自动记录失败: {e}')

    def show_report(self):
        if self._backtest_result is None:
            return
        if self._ai_mode:
            self.logger.info("AI自动运行模式下跳过报告绘图")
            return
        try:
            from utils.report import generate_report
            generate_report(self._backtest_result)
        except Exception as e:
            self.logger.warning(f"报告显示失败({type(e).__name__}: {e})，回测结果已记录到文件")

    def plot(self, **kwargs):
        self.logger.info("自研引擎不支持 plot()，请使用 show_report() 查看结果")

    def _build_strategy_params(self) -> Dict[str, Any]:
        params = {}
        if self._symbols:
            symbol = self._symbols[0]
            parts = symbol.split(".")
            params["instrument_id"] = parts[0] if len(parts) > 0 else symbol
            params["exchange"] = parts[1] if len(parts) > 1 else ""
        params["kline_style"] = self._period
        params.update(self._strategy_kwargs)
        return params
