import datetime as dt_module
import logging
from typing import Dict, List, Optional, Any, Type

import pandas as pd

from core.strategy_logic import StrategyLogic, BarData, OrderInfo
from engine.data_feed import ArrayDataFeed, LazyDataFeed
from engine.broker import SimulatedBroker, Order
from engine.timeline import Timeline
from engine.adapter import EngineDataAdapter, EngineExecutor
from engine.result import EngineResult


class BacktestEngine:
    """自研回测引擎 - 替代 Backtrader Cerebro

    核心优化：
    1. 预加载 numpy 数组替代 Backtrader lines 对象
    2. 预计算统一时间轴替代运行时逐 bar 对齐
    3. 简化订单执行（COC 立即成交）替代 Backtrader 状态机

    支持日线和分钟线两种模式：
    - 日线模式：每个全局索引对应一个交易日，每天执行一次 on_bar
    - 分钟线模式：每个全局索引对应一根分钟K线，每分钟执行一次 on_bar
    """

    DAILY_PERIODS = {'1d', 'day', 'daily'}

    def __init__(self):
        self._data_feeds: Dict[str, ArrayDataFeed] = {}
        self._data_feed_list: List[ArrayDataFeed] = []
        self._lazy_feeds: Dict[str, LazyDataFeed] = {}
        self._lazy_mode: bool = False
        self._broker: Optional[SimulatedBroker] = None
        self._timeline: Optional[Timeline] = None
        self._strategy_logic: Optional[StrategyLogic] = None
        self._data_adapter: Optional[EngineDataAdapter] = None
        self._executor: Optional[EngineExecutor] = None
        self._trade_start_date: Optional[str] = None
        self._trading_dates: Optional[set] = None
        self._period: str = '1d'
        self._is_daily: bool = True
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def add_data(self, symbol: str, dataframe: pd.DataFrame):
        feed = ArrayDataFeed(symbol, dataframe)
        self._data_feeds[symbol] = feed
        self._data_feed_list.append(feed)

    def add_lazy_feed(self, symbol: str, data_processor, period: str = '1d',
                       start_date: str = '', end_date: str = ''):
        """注册按需加载的数据源（不立即加载数据）"""
        self._lazy_feeds[symbol] = LazyDataFeed(
            symbol, data_processor, period, start_date, end_date
        )

    def set_lazy_mode(self, lazy_mode: bool):
        """启用/禁用按需加载模式"""
        self._lazy_mode = lazy_mode

    def set_broker(self, broker: SimulatedBroker):
        self._broker = broker

    def set_strategy(self, strategy_logic: StrategyLogic):
        self._strategy_logic = strategy_logic

    def set_trade_start_date(self, trade_start_date: str):
        self._trade_start_date = trade_start_date

    def set_trading_dates(self, trading_dates: set):
        self._trading_dates = trading_dates

    def set_period(self, period: str):
        self._period = period
        self._is_daily = period in self.DAILY_PERIODS

    def _prepare(self):
        if self._broker is None:
            self._broker = SimulatedBroker()

        self._timeline = Timeline(self._data_feed_list, period=self._period,
                                   lazy_mode=self._lazy_mode)

        self._data_adapter = EngineDataAdapter(
            self._data_feeds, self._timeline, period=self._period,
            lazy_feeds=self._lazy_feeds
        )

        self._executor = EngineExecutor(
            self._broker, self._data_adapter, self._data_feeds,
            lazy_feeds=self._lazy_feeds
        )

        if self._strategy_logic:
            self._strategy_logic.set_data_adapter(self._data_adapter)
            self._strategy_logic.executor = self._executor
            self._executor.set_strategy_logic(self._strategy_logic)

    def run(self) -> EngineResult:
        self._prepare()

        num_bars = self._timeline.get_num_days()
        if num_bars == 0:
            self.logger.warning('[run] 无数据')
            return EngineResult(
                broker=self._broker,
                strategy_logic=self._strategy_logic,
                initial_cash=self._broker.startingcash,
            )

        self.logger.info(
            f'[run] 回测启动: 数据源数量={len(self._data_feeds)}, '
            f'bar数量={num_bars}, 周期={self._period}'
        )

        if self._strategy_logic and num_bars > 0:
            self._strategy_logic._backtest_start_date = self._timeline.get_date(0)
            self._strategy_logic._backtest_end_date = self._timeline.get_date(num_bars - 1)

        equity_history: List[tuple] = []
        trade_records: List[Dict] = []
        last_equity_date = None
        current_bar = 0

        for global_idx in range(num_bars):
            current_date = self._timeline.get_date(global_idx)
            if current_date is None:
                continue

            self._data_adapter.update(global_idx)

            if self._strategy_logic:
                self._strategy_logic.update_data()

            current_indices = self._data_adapter._current_local_indices
            current_date_str = current_date.isoformat() if current_date else None
            value = self._broker.getvalue(
                self._data_feeds, current_indices,
                lazy_feeds=self._lazy_feeds, current_date=current_date_str
            )

            if self._trading_dates is not None:
                is_trading_day = current_date in self._trading_dates
            else:
                first_symbol = next(iter(self._data_feeds), None)
                first_local_idx = current_indices.get(first_symbol, -1) if first_symbol else -1
                is_trading_day = first_local_idx >= 0

            if is_trading_day:
                if last_equity_date != current_date:
                    equity_history.append((current_date, value))
                    last_equity_date = current_date
                else:
                    if equity_history:
                        equity_history[-1] = (current_date, value)

            if self._trade_start_date:
                current_date_str = current_date.isoformat()
                if current_date_str < self._trade_start_date:
                    continue

            if self._strategy_logic and is_trading_day:
                self._executor.snapshot_cash()
                bar_symbol = None
                bar_local_idx = -1
                for sym, idx in current_indices.items():
                    if idx >= 0 and not self._data_feeds[sym].is_nan(idx):
                        bar_symbol = sym
                        bar_local_idx = idx
                        break
                if bar_symbol and bar_local_idx >= 0:
                    feed = self._data_feeds[bar_symbol]
                    bar = feed.get_bar(bar_local_idx)
                    self._strategy_logic.on_bar(bar)

            current_bar += 1
            if current_bar % max(1, num_bars // 100) == 0:
                if self._is_daily:
                    print(f"回测日期: {current_date}")
                else:
                    current_dt = self._timeline.get_datetime(global_idx)
                    if current_dt:
                        print(f"回测时间: {current_dt.strftime('%Y-%m-%d %H:%M')}")

        self._data_adapter.finalize_daily_bars()

        if self._strategy_logic and hasattr(self._strategy_logic, 'on_backtest_end'):
            has_position = False
            for symbol, feed in self._data_feeds.items():
                pos = self._broker.get_position(symbol)
                if pos.size != 0:
                    has_position = True
            if has_position:
                self._strategy_logic.on_backtest_end()

        for trade in self._broker.get_trades():
            trade_records.append({
                'datetime': trade.datetime,
                'symbol': trade.symbol,
                'direction': trade.direction,
                'pnl': trade.pnlcomm,
                'pnl_no_commission': trade.pnl,
                'price': trade.price,
                'size': trade.size,
                'commission': trade.commission,
                'is_long': trade.direction == 'buy',
            })

        final_indices = self._data_adapter._current_local_indices
        final_date_str = None
        if self._timeline and final_indices:
            last_global_idx = self._data_adapter._current_global_idx
            last_date = self._timeline.get_date(last_global_idx)
            if last_date:
                final_date_str = last_date.isoformat()
        final_value = self._broker.getvalue(
            self._data_feeds, final_indices,
            lazy_feeds=self._lazy_feeds, current_date=final_date_str
        )

        result = EngineResult(
            equity_history=equity_history,
            trade_records=trade_records,
            orders=self._broker.get_orders(),
            data_feeds=self._data_feeds,
            broker=self._broker,
            strategy_logic=self._strategy_logic,
            initial_cash=self._broker.startingcash,
            final_value=final_value,
        )

        self.logger.info(
            f'[run] 回测完成: 初始资金={result.initial_cash:.2f}, '
            f'最终资金={final_value:.2f}, '
            f'交易记录={len(trade_records)}条'
        )

        return result
