from __future__ import annotations
import backtrader as bt
import datetime
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Type
from tqdm import tqdm
from core.data import DataProcessor, QMTDataProcessor, create_data_processor
from core.executor import BacktestExecutor
from core.data_adapter import BacktraderDataAdapter
from core.strategy_logic import StrategyLogic
from core.stock_selection import StockSelectionStrategy
from core.financial_data import FinancialDataCache, FinancialDataAdapter
from core.strategy import BaseStrategy
from core.analyzer import PerformanceAnalyzer
from api.base_api import BaseAPI

_ADAPTER_CLASS_CACHE: Dict[str, Type] = {}


def _make_adapter_class(
    strategy_logic_class: Type[StrategyLogic],
    strategy_kwargs: Dict[str, Any],
    symbols: List[str],
    period: str,
    financial_adapter: Optional[FinancialDataAdapter] = None,
) -> Type[BacktestStrategyAdapter]:
    cache_key = f"{strategy_logic_class.__module__}.{strategy_logic_class.__qualname__}_{period}_{'_'.join(symbols)}"
    if cache_key in _ADAPTER_CLASS_CACHE:
        return _ADAPTER_CLASS_CACHE[cache_key]

    params_tuple = tuple(strategy_logic_class.params) if hasattr(strategy_logic_class, 'params') else ()

    class _CachedParamsAdapter(BacktestStrategyAdapter):
        params = params_tuple

        def __init__(self, **bt_kwargs):
            super().__init__(
                strategy_logic_class=strategy_logic_class,
                strategy_kwargs=strategy_kwargs,
                symbols=symbols,
                period=period,
                financial_adapter=financial_adapter,
                **bt_kwargs,
            )

    _CachedParamsAdapter.__qualname__ = f'_CachedParamsAdapter_{strategy_logic_class.__name__}'
    _ADAPTER_CLASS_CACHE[cache_key] = _CachedParamsAdapter
    return _CachedParamsAdapter


class BacktestStrategyAdapter(BaseStrategy):
    """回测策略适配器 - 将StrategyLogic桥接到backtrader框架

    独立于BacktestAPI的顶层类，通过构造函数接收所有依赖，
    避免闭包捕获外部变量带来的隐式耦合。
    """

    def __init__(self, strategy_logic_class: Type[StrategyLogic],
                 strategy_kwargs: Dict[str, Any],
                 symbols: List[str],
                 period: str = '1d',
                 financial_adapter: Optional[FinancialDataAdapter] = None,
                 **bt_kwargs):
        super().__init__()
        self._symbols = symbols

        logic = strategy_logic_class(**strategy_kwargs)

        executor = BacktestExecutor(self)
        adapter = BacktraderDataAdapter(period=period)

        logging.getLogger(__name__).debug(
            f'[BacktestStrategyAdapter] 初始化: symbols={symbols}, period={period}, '
            f'datas数量={len(self.datas) if hasattr(self, "datas") else 0}'
        )

        for i, sym in enumerate(symbols):
            if i < len(self.datas):
                adapter.register_data(sym, self.datas[i])
                executor.register_data(sym, self.datas[i])
                logging.getLogger(__name__).debug(
                    f'[BacktestStrategyAdapter] 映射: {sym} -> datas[{i}], '
                    f'name={self.datas[i]._name if hasattr(self.datas[i], "_name") else "N/A"}'
                )
            else:
                logging.getLogger(__name__).warning(
                    f'[BacktestStrategyAdapter] 无法映射: {sym}, datas数量不足 '
                    f'(需要索引{i}, 只有{len(self.datas)}个数据源)'
                )

        logic.set_data_adapter(adapter)
        logic.executor = executor

        if financial_adapter:
            logic.set_financial_data_adapter(financial_adapter)

        self.set_strategy_logic(logic)


class BacktestAPI(BaseAPI):
    """回测API - 通过backtrader引擎驱动策略

    策略逻辑与执行环境解耦：
    - 策略类继承StrategyLogic，仅定义交易逻辑
    - BacktestAPI负责创建适配层，将StrategyLogic桥接到backtrader
    - 数据访问通过BacktraderDataAdapter
    - 交易执行通过BacktestExecutor
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
    """

    def __init__(self, data_source: str = 'qmt'):
        super().__init__()
        self.cerebro = bt.Cerebro()
        self.data_processor = create_data_processor(data_source, fallback_to_simulated=True)
        self._data_source = data_source
        # 财务数据处理器根据数据源选择
        self._financial_data_processor = create_data_processor(data_source, fallback_to_simulated=True)
        self._symbols: List[str] = []
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
        self._custom_analyzers_added = False
        self._financial_adapter: Optional[FinancialDataAdapter] = None
        self._stock_pool: Optional[List[str]] = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    # QMT 数据源最多只能获取最近一年的数据
    QMT_MAX_DATA_DAYS = 365

    def configure(self, cash: float = 200000, commission: float = 0.0001,
                  start_date: Optional[str] = None, end_date: Optional[str] = None,
                  data_lookback_days: int = 40, benchmark: str = '000300.SH',
                  period: str = '1d', trade_start_date: Optional[str] = None, **kwargs):
        """一次性配置回测参数

        将 start_date 自动前移 data_lookback_days 天作为数据加载起始日，
        start_date 本身作为交易起始日（trade_start_date）。

        当数据源为 QMT 时，自动将回测起始日限制在最近一年内，
        以确保 QMT 数据完整（QMT 只能获取最近约一年的行情数据）。

        Args:
            cash: 初始资金
            commission: 手续费率
            start_date: 交易起始日，如 '2025-07-10'
            end_date: 回测结束日，如 '2026-04-17'
            data_lookback_days: 数据前移天数，用于指标预热
            benchmark: 基准标的
            period: 数据周期
            trade_start_date: 交易起始日，默认等于 start_date
        """
        self.set_cash(cash)
        self.set_commission(commission)
        self.set_benchmark(benchmark)
        self._period = period

        if start_date and end_date:
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

            # QMT 数据源：自动限制回测起始日，确保数据完整
            if self._data_source == 'qmt':
                min_start_dt = end_dt - datetime.timedelta(days=self.QMT_MAX_DATA_DAYS)
                if start_dt < min_start_dt:
                    old_start = start_date
                    start_date = min_start_dt.strftime('%Y-%m-%d')
                    self.logger.warning(
                        f"数据源 '{self._data_source}' 最多支持最近 {self.QMT_MAX_DATA_DAYS} 天数据，"
                        f"回测起始日已从 {old_start} 调整为 {start_date}"
                    )
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
        """设置数据加载范围

        与 add_data 不同，此方法仅记录范围参数，不立即加载数据。
        后续 add_strategy 会根据此范围自动加载策略所需的标的数据。

        Args:
            start_date: 数据起始日
            end_date: 数据结束日
            period: 数据周期
        """
        self._data_start_date = start_date
        self._data_end_date = end_date
        self._period = period

    def add_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d"):
        self._period = period
        self._data_start_date = start_date
        self._data_end_date = end_date
        data = self.data_processor.get_data(symbol, start_date, end_date, period)

        if not data.empty:
            bt_data = bt.feeds.PandasData(
                dataname=data,
                datetime='datetime' if 'datetime' in data.columns else None,
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest='openinterest' if 'openinterest' in data.columns else -1
            )

            self.cerebro.adddata(bt_data, name=symbol)
            self._symbols.append(symbol)
            self.logger.debug(
                f'[add_data] {symbol}: 加载成功, {len(data)}行, '
                f'日期范围={data.index[0]}~{data.index[-1]}, '
                f'列={list(data.columns)}'
            )
        else:
            self.logger.warning(f'[add_data] {symbol}: 数据为空! start={start_date}, end={end_date}, period={period}')

    def load_financial_data(self, stock_list: Optional[List[str]] = None,
                            table_list: Optional[List[str]] = None,
                            sector: Optional[str] = None,
                            start_time: str = '', end_time: str = '',
                            report_type: str = 'announce_time') -> FinancialDataAdapter:
        """加载财务数据并创建财务数据适配器

        Args:
            stock_list: 股票代码列表，与 sector 二选一
            table_list: 财务报表列表，为空则加载全部
            sector: 板块名称，如 '沪深300'，与 stock_list 二选一
            start_time: 财务数据起始时间
            end_time: 财务数据结束时间
            report_type: 报表筛选方式
                'announce_time' - 按披露日期（回测推荐，避免未来数据）
                'report_time' - 按报告期

        Returns:
            FinancialDataAdapter 实例

        示例:
            # 方式1：指定股票列表
            adapter = api.load_financial_data(
                stock_list=['000001.SZ', '600000.SH'],
                table_list=['Balance', 'Income', 'Pershareindex']
            )

            # 方式2：指定板块
            adapter = api.load_financial_data(sector='沪深300')
        """
        import time as _time
        overall_start = _time.time()

        # ── 阶段1：获取股票池 ──
        if stock_list is None and sector:
            self.logger.info(f"[阶段1/5] 获取板块 '{sector}' 成分股...")
            stock_list = self._financial_data_processor.get_stock_list(sector)
            self.logger.info(f"[阶段1/5] 板块 '{sector}' 获取到 {len(stock_list)} 只股票")
        else:
            self.logger.info(f"[阶段1/5] 使用指定股票列表: {len(stock_list or [])} 只")

        if not stock_list:
            raise ValueError("必须指定 stock_list 或 sector")

        # ── 阶段2：下载财务数据到本地 ──
        self.logger.info(f"[阶段2/5] 下载财务数据到本地缓存，共 {len(stock_list)} 只股票...")
        self._financial_data_processor.download_financial_data(
            stock_list, table_list, start_time, end_time
        )

        # ── 阶段3：读取并缓存财务数据 ──
        self.logger.info(f"[阶段3/5] 读取财务数据，共 {len(stock_list)} 只股票...")
        raw_data = self._financial_data_processor.get_financial_data(
            stock_list, table_list, start_time, end_time, report_type
        )

        cache = FinancialDataCache(raw_data)
        self._financial_adapter = FinancialDataAdapter(cache)
        self._stock_pool = stock_list

        loaded_stocks = len(cache.get_stocks())
        self.logger.info(
            f"[阶段3/5] 财务数据适配器创建完成: {loaded_stocks}/{len(stock_list)} 只股票有数据"
        )

        # ── 阶段4：获取行业映射 ──
        self.logger.info(f"[阶段4/5] 获取申万行业分类映射...")
        try:
            industry_mapping = self._financial_data_processor.get_industry_mapping(
                level=1, stock_pool=stock_list
            )
            if industry_mapping:
                self._financial_adapter.set_industry_mapping(industry_mapping)
                self.logger.info(f"[阶段4/5] 行业分类映射已加载: {len(industry_mapping)} 只股票, {len(set(industry_mapping.values()))} 个行业")
            else:
                self.logger.warning(f"[阶段4/5] 行业分类映射为空")
        except Exception as e:
            self.logger.warning(f"[阶段4/5] 行业分类映射加载失败: {e}，策略将无法使用行业筛选")

        # ── 阶段5：获取分红数据 ──
        self.logger.info(f"[阶段5/5] 获取分红数据，共 {len(stock_list)} 只股票...")
        try:
            dividend_data = self._financial_data_processor.get_dividend_data(stock_list)
            if dividend_data:
                self._financial_adapter.set_dividend_data(dividend_data)
                self.logger.info(f"[阶段5/5] 分红数据已加载: {len(dividend_data)} 只股票")
            else:
                self.logger.warning(f"[阶段5/5] 分红数据为空，策略将无法计算股息率")
        except Exception as e:
            self.logger.warning(f"[阶段5/5] 分红数据加载失败: {e}，策略将无法计算股息率")

        overall_elapsed = _time.time() - overall_start
        self.logger.info(
            f"财务数据加载全部完成: {loaded_stocks}/{len(stock_list)} 只股票有数据, "
            f"总耗时 {overall_elapsed:.1f}秒"
        )

        return self._financial_adapter

    def load_stock_pool(self, sector: str = '沪深A股',
                        stock_list: Optional[List[str]] = None) -> List[str]:
        """加载股票池

        Args:
            sector: 板块名称
            stock_list: 自定义股票列表，优先于 sector

        Returns:
            股票代码列表
        """
        if stock_list:
            self._stock_pool = stock_list
        else:
            self._stock_pool = self.data_processor.get_stock_list(sector)

        self.logger.info(f"股票池加载完成: {len(self._stock_pool)} 只股票")
        return self._stock_pool

    def add_stock_selection_strategy(self, strategy_class: Type[StockSelectionStrategy],
                                     stock_pool: Optional[List[str]] = None,
                                     **kwargs):
        """添加选股策略 - 支持多标的调仓回测

        选股策略会自动加载股票池中所有标的的行情数据，
        并注入财务数据适配器。

        Args:
            strategy_class: 选股策略类（须继承 StockSelectionStrategy）
            stock_pool: 股票池，为空则使用 load_stock_pool 加载的池子
            **kwargs: 策略参数

        使用方式:
            api = BacktestAPI()
            api.set_cash(1000000)
            api.configure(start_date='2024-01-01', end_date='2026-04-17')

            # 加载财务数据
            api.load_financial_data(sector='沪深300')

            # 添加选股策略
            api.add_stock_selection_strategy(MyFundamentalStrategy, max_stocks=10)
            api.run()
        """
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

        import time as _time
        load_start = _time.time()

        loaded_count = 0
        failed_count = 0
        skipped_count = 0
        total = len(pool)

        self.logger.info(f"开始加载股票池行情数据: {total} 只股票, 周期={self._period}")

        for i, symbol in enumerate(pool, 1):
            if symbol not in self._symbols:
                try:
                    self.add_data(symbol, self._data_start_date, self._data_end_date, self._period)
                    loaded_count += 1
                    if i % 20 == 0 or i == total:
                        elapsed = _time.time() - load_start
                        self.logger.info(
                            f"[ {i} / {total} ] 行情加载进度: "
                            f"{loaded_count} 已加载, {failed_count} 失败, "
                            f"{skipped_count} 已存在, 耗时 {elapsed:.1f}秒"
                        )
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 5:
                        self.logger.debug(f"加载 {symbol} 行情数据失败: {e}")
            else:
                skipped_count += 1

        load_elapsed = _time.time() - load_start
        self.logger.info(
            f"股票池行情数据加载完成: 成功 {loaded_count}, 失败 {failed_count}, "
            f"已存在 {skipped_count}, 耗时 {load_elapsed:.1f}秒"
        )

        self._ensure_default_analyzers()

        strategy_logic_class_ref = strategy_class
        strategy_kwargs = dict(kwargs)
        symbols = list(self._symbols)
        period = self._period
        trade_start_date = self._trade_start_date
        financial_adapter = self._financial_adapter

        adapter_cls = _make_adapter_class(
            strategy_logic_class_ref, strategy_kwargs, symbols, period, financial_adapter
        )
        self.cerebro.addstrategy(adapter_cls, trade_start_date=trade_start_date)

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

        self._ensure_default_analyzers()

        strategy_logic_class_ref = strategy_logic_class
        strategy_kwargs = dict(kwargs)
        symbols = list(self._symbols)
        period = self._period
        trade_start_date = self._trade_start_date
        financial_adapter = self._financial_adapter

        adapter_cls = _make_adapter_class(
            strategy_logic_class_ref, strategy_kwargs, symbols, period, financial_adapter
        )
        self.cerebro.addstrategy(adapter_cls, trade_start_date=trade_start_date)

    def set_cash(self, cash: float):
        self._initial_cash = cash
        self.cerebro.broker.setcash(cash)

    def set_trade_start_date(self, trade_start_date: str):
        self._trade_start_date = trade_start_date

    def set_benchmark(self, benchmark: str):
        self._benchmark = benchmark

    def set_commission(self, commission: float):
        self.cerebro.broker.setcommission(commission=commission)

    def add_analyzer(self, analyzer_class, **kwargs):
        self._custom_analyzers_added = True
        self.cerebro.addanalyzer(analyzer_class, **kwargs)

    def _ensure_default_analyzers(self):
        if self._custom_analyzers_added:
            return
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')

    def run(self):
        if self._initial_cash == 0.0:
            self.set_cash(200000)

        self.logger.info(
            f'[run] 回测启动: symbols={self._symbols}, period={self._period}, '
            f'数据范围={self._data_start_date}~{self._data_end_date}, '
            f'交易起始日={self._trade_start_date}, '
            f'初始资金={self._initial_cash}, '
            f'策略={self._strategy_logic_class.__name__ if self._strategy_logic_class else "N/A"}, '
            f'数据源数量={len(self.cerebro.datas)}'
        )

        # 检查数据源
        for i, data in enumerate(self.cerebro.datas):
            name = data._name if hasattr(data, '_name') else f'data[{i}]'
            self.logger.debug(f'[run] cerebro.datas[{i}]: {name}')

        results = self.cerebro.run()
        self.cerebro.runstrats = [results]

        if results:
            strategy = results[0]
            self._analyzer.set_context(
                self.cerebro, strategy, self._initial_cash
            )
            strategy_params = self._build_strategy_params()
            self._backtest_result = self._analyzer.build_result(
                strategy_params=strategy_params,
                show_kline=True,
            )

            self._fetch_benchmark_data()
            self._log_summary(strategy)

        return results

    def _log_summary(self, strategy):
        try:
            sharpe = strategy.analyzers.sharpe.get_analysis()
            drawdown = strategy.analyzers.drawdown.get_analysis()
            returns = strategy.analyzers.returns.get_analysis()
            transactions = strategy.analyzers.transactions.get_analysis()

            final_value = self.cerebro.broker.getvalue()
            total_return = returns['rtot'] * 100

            self.logger.info(f"初始资金: {self._initial_cash}")
            self.logger.info(f"最终资金: {final_value:.2f}")
            self.logger.info(f"总收益率: {total_return:.2f}%")

            sharpe_ratio = sharpe.get('sharperatio')
            if sharpe_ratio is not None:
                self.logger.info(f"夏普比率: {sharpe_ratio:.2f}")
            else:
                self.logger.info("夏普比率: N/A")

            self.logger.info(f"最大回撤: {drawdown['max']['drawdown']:.2f}%")
            self.logger.info(f"交易次数: {len(transactions)}")
        except Exception as e:
            self.logger.warning(f"回测摘要输出失败: {e}")

    def _fetch_benchmark_data(self):
        if self._backtest_result is None or not self._benchmark:
            return
        if not self._data_start_date or not self._data_end_date:
            return
        try:
            benchmark_data = self.data_processor.get_data(
                self._benchmark,
                self._data_start_date,
                self._data_end_date,
                self._period,
            )
            if benchmark_data is not None and not benchmark_data.empty:
                if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                    benchmark_data.index = pd.to_datetime(benchmark_data.index)
                benchmark_data = benchmark_data[["close"]].copy()
                benchmark_data.columns = ["close"]
                self._backtest_result.benchmark_df = benchmark_data
                self._backtest_result.benchmark_symbol = self._benchmark
        except Exception:
            pass

    def get_result(self):
        return self._backtest_result

    def show_report(self):
        if self._backtest_result is None:
            return
        from utils.report import generate_report
        generate_report(self._backtest_result)

    def plot(self, **kwargs):
        if not hasattr(self.cerebro, '_exactbars'):
            self.cerebro._exactbars = 0
        self.cerebro.plot(**kwargs)

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
