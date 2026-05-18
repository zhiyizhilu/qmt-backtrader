import copy
from typing import Dict, List, Optional, Any
import datetime as dt_module
import logging
from collections import deque
import numpy as np
from core.executor import StrategyExecutor
from core.data_adapter import MarketDataAdapter


class SimpleParams:
    """简单参数容器 - 兼容backtrader的params属性访问接口"""

    def __init__(self, params_tuple=()):
        if isinstance(params_tuple, dict):
            for key, value in params_tuple.items():
                setattr(self, key, value)
        elif isinstance(params_tuple, (tuple, list)):
            for item in params_tuple:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    setattr(self, item[0], item[1])

    def clone(self) -> 'SimpleParams':
        new_params = SimpleParams()
        new_params.__dict__ = copy.deepcopy(self.__dict__)
        return new_params

    def __repr__(self):
        return f'SimpleParams({self.__dict__})'


class OrderInfo:
    """订单信息 - 统一不同环境的订单数据结构"""

    STATUS_SUBMITTED = 'submitted'
    STATUS_ACCEPTED = 'accepted'
    STATUS_PARTIAL = 'partial'
    STATUS_COMPLETED = 'completed'
    STATUS_CANCELED = 'canceled'
    STATUS_REJECTED = 'rejected'
    STATUS_MARGIN = 'margin'

    __slots__ = (
        'order_id', 'symbol', 'direction', 'price', 'volume', 'status',
        'executed_volume', 'executed_price', 'commission', 'datetime',
    )

    def __init__(self, order_id: str = '', symbol: str = '', direction: str = '',
                 price: float = 0.0, volume: int = 0, status: str = '',
                 executed_volume: int = 0, executed_price: float = 0.0,
                 commission: float = 0.0, datetime=None, **kwargs):
        self.order_id = order_id
        self.symbol = symbol
        self.direction = direction
        self.price = price
        self.volume = volume
        self.status = status
        self.executed_volume = executed_volume
        self.executed_price = executed_price
        self.commission = commission
        self.datetime = datetime
        if kwargs:
            import logging
            logging.getLogger(__name__).warning(
                f"OrderInfo 忽略未知字段: {list(kwargs.keys())}. "
                f"请将新字段添加到 OrderInfo.__slots__ 中"
            )

    @property
    def is_active(self) -> bool:
        return self.status in (self.STATUS_SUBMITTED, self.STATUS_ACCEPTED, self.STATUS_PARTIAL)

    @property
    def is_completed(self) -> bool:
        return self.status == self.STATUS_COMPLETED

    @property
    def is_buy(self) -> bool:
        return self.direction == 'buy'

    @property
    def is_sell(self) -> bool:
        return self.direction == 'sell'

    def __repr__(self):
        return (f'OrderInfo(id={self.order_id}, {self.direction} {self.symbol} '
                f'x{self.volume}@{self.price}, status={self.status})')


class TradeInfo:
    """成交信息 - 统一不同环境的成交数据结构"""

    __slots__ = (
        'trade_id', 'order_id', 'symbol', 'direction', 'price', 'volume',
        'commission', 'pnl',
    )

    def __init__(self, trade_id: str = '', order_id: str = '', symbol: str = '',
                 direction: str = '', price: float = 0.0, volume: int = 0,
                 commission: float = 0.0, pnl: float = 0.0, **kwargs):
        self.trade_id = trade_id
        self.order_id = order_id
        self.symbol = symbol
        self.direction = direction
        self.price = price
        self.volume = volume
        self.commission = commission
        self.pnl = pnl
        if kwargs:
            import logging
            logging.getLogger(__name__).warning(
                f"TradeInfo 忽略未知字段: {list(kwargs.keys())}. "
                f"请将新字段添加到 TradeInfo.__slots__ 中"
            )

    @property
    def is_buy(self) -> bool:
        return self.direction == 'buy'

    @property
    def is_sell(self) -> bool:
        return self.direction == 'sell'

    def __repr__(self):
        return (f'TradeInfo(id={self.trade_id}, {self.direction} {self.symbol} '
                f'x{self.volume}@{self.price})')


class BarData:
    """K线数据 - 统一的bar数据结构"""

    def __init__(self, symbol: str = '', open: float = 0.0, high: float = 0.0,
                 low: float = 0.0, close: float = 0.0, volume: float = 0.0,
                 datetime=None, **kwargs):
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.datetime = datetime
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return (f'BarData({self.symbol}, {self.datetime}, '
                f'O={self.open} H={self.high} L={self.low} C={self.close} V={self.volume})')


class TickData:
    """逐笔数据 - 统一的tick数据结构"""

    def __init__(self, symbol: str = '', last_price: float = 0.0, volume: float = 0.0,
                 bid_price: float = 0.0, ask_price: float = 0.0,
                 bid_volume: float = 0.0, ask_volume: float = 0.0,
                 datetime=None, **kwargs):
        self.symbol = symbol
        self.last_price = last_price
        self.volume = volume
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_volume = bid_volume
        self.ask_volume = ask_volume
        self.datetime = datetime
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f'TickData({self.symbol}, price={self.last_price}, vol={self.volume})'


class StrategyLogic:
    """策略逻辑基类 - 不依赖任何执行框架，可在回测/模拟/实盘环境下运行

    策略类应继承此类，根据需要重写以下事件回调：
    - on_bar():   K线数据到达时触发（日线/分钟线策略的核心入口）
    - on_tick():  逐笔数据到达时触发（高频策略的核心入口）
    - on_order(): 委托状态变化时触发
    - on_trade(): 成交回报时触发

    通过数据适配器访问行情数据，通过执行器执行交易，
    实现策略逻辑与执行环境的完全解耦。
    """

    params = (
        ('t_plus_1', True),
    )

    def __init__(self, executor: Optional[StrategyExecutor] = None, **kwargs):
        if isinstance(self.params, tuple):
            self.params = SimpleParams(self.params)
        elif isinstance(self.params, SimpleParams):
            self.params = self.params.clone()
        for key, value in kwargs.items():
            setattr(self.params, key, value)

        self.executor = executor
        self._data_adapter: Optional[MarketDataAdapter] = None
        self._financial_data_adapter = None
        self._data_processor = None
        self._orders: Dict[str, OrderInfo] = {}
        self._risk_controller: Optional['RiskController'] = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        self._today_buys: Dict[str, int] = {}
        self._current_trade_date = None
        self._t_plus_1_overrides: Dict[str, bool] = {}
        self._unadjusted_price_cache: Dict[str, float] = {}
        self._unadjusted_price_cache_date: Optional[dt_module.date] = None
        self._unadjusted_price_df_cache: Dict[str, Any] = {}
        self._backtest_start_date: Optional[dt_module.date] = None
        self._backtest_end_date: Optional[dt_module.date] = None
        self._data_start_date: Optional[str] = None
        self._data_end_date: Optional[str] = None

    def set_data_adapter(self, adapter: MarketDataAdapter) -> None:
        """设置数据适配器"""
        self._data_adapter = adapter

    def set_financial_data_adapter(self, adapter) -> None:
        """设置财务数据适配器

        Args:
            adapter: FinancialDataAdapter 实例
        """
        self._financial_data_adapter = adapter

    def set_data_processor(self, processor) -> None:
        """设置数据处理器（用于获取不复权行情等数据）

        Args:
            processor: DataProcessor 实例（如 QMTDataProcessor）
        """
        self._data_processor = processor

    @property
    def financial_adapter(self):
        """获取财务数据适配器"""
        return self._financial_data_adapter

    def update_data(self) -> None:
        """通知数据适配器更新当前bar的数据

        由适配层（如BaseStrategy）在每个bar到达时调用，
        策略代码不应直接调用此方法。
        """
        if self._data_adapter and hasattr(self._data_adapter, 'update_from_backtrader'):
            self._data_adapter.update_from_backtrader()

        if self._financial_data_adapter:
            current_date = self.get_current_date()
            if current_date:
                self._financial_data_adapter.set_current_date(current_date)

    def get_orders(self) -> Dict[str, 'OrderInfo']:
        """获取所有订单记录的副本

        Returns:
            订单字典的浅拷贝，key为order_id，value为OrderInfo
        """
        return dict(self._orders)

    # ================================================================
    # 事件回调 - 框架调用，策略可重写
    # ================================================================

    def on_bar(self, bar: BarData):
        """K线数据到达时触发

        这是日线/分钟线策略的核心入口。
        默认实现调用 generate_signals()，保持向后兼容。
        子类应重写此方法或 generate_signals() 来定义交易逻辑。

        Args:
            bar: 当前K线数据
        """
        self.generate_signals()

    def on_tick(self, tick: TickData):
        """逐笔数据到达时触发

        这是高频策略的核心入口。
        默认实现为空，子类按需重写。

        Args:
            tick: 当前逐笔数据
        """
        pass

    def on_order(self, order: OrderInfo):
        self._orders[order.order_id] = order
        self.log(f'委托回调: {order}')

        if order.is_completed and order.is_buy and order.executed_volume > 0:
            self._update_today_buys(order.symbol, order.executed_volume)

    def on_trade(self, trade: TradeInfo):
        """成交回报时触发

        当订单成交时，框架会调用此方法通知策略。

        默认实现仅记录日志，子类可重写以实现自定义的成交处理逻辑。

        Args:
            trade: 成交信息
        """
        self.log(f'成交回调: {trade}')

    def generate_signals(self):
        """生成交易信号 - 旧版接口，保持向后兼容

        推荐使用 on_bar() 替代此方法。
        如果子类没有重写 on_bar()，默认的 on_bar() 会调用此方法。
        """
        pass

    # ================================================================
    # 交易操作 - 策略主动调用
    # ================================================================

    def buy(self, symbol: str, price: float, volume: int):
        self.log(f'买入委托: {symbol}, 价格: {price:.2f}, 数量: {volume}')
        if not self._check_risk_before_buy(symbol, price, volume):
            self.log(f'风控拒绝买入: {symbol}', level='warning')
            return None
        if self.is_suspended(symbol):
            self.log(f'停牌拒绝买入: {symbol}', level='warning')
            return None
        if self.is_limit_up(symbol):
            self.log(f'涨停拒绝买入: {symbol}', level='warning')
            return None
        if self.executor:
            return self.executor.execute_buy(symbol, price, volume)
        self.log(f'[WARNING] 买入失败: executor未设置!', level='error')
        return None

    def sell(self, symbol: str, price: float, volume: int):
        self.log(f'卖出委托: {symbol}, 价格: {price:.2f}, 数量: {volume}')

        sellable = self.get_sellable_volume(symbol)
        if sellable <= 0:
            if self.is_t_plus_1(symbol):
                self.log(f'T+1拒绝卖出: {symbol}, 当天买入不可卖出', level='warning')
            else:
                self.log(f'拒绝卖出: {symbol}, 无持仓', level='warning')
            return None
        if volume > sellable:
            self.log(f'T+1调整卖出数量: {symbol}, 请求{volume}股, 可卖{sellable}股', level='warning')
            volume = sellable

        if not self._check_risk_before_sell(symbol, price, volume):
            self.log(f'风控拒绝卖出: {symbol}', level='warning')
            return None
        if self.is_suspended(symbol):
            self.log(f'停牌拒绝卖出: {symbol}', level='warning')
            return None
        if self.is_limit_down(symbol):
            self.log(f'跌停拒绝卖出: {symbol}', level='warning')
            return None
        if self.executor:
            return self.executor.execute_sell(symbol, price, volume)
        self.log(f'[WARNING] 卖出失败: executor未设置!', level='error')
        return None

    def cancel(self, order_id: str):
        """撤单操作 - 通过执行器路由到对应环境

        Args:
            order_id: 订单ID

        Returns:
            是否撤单成功
        """
        self.log(f'撤单委托: {order_id}')
        if self.executor and hasattr(self.executor, 'cancel_order'):
            return self.executor.cancel_order(order_id)
        return False

    # ================================================================
    # 查询操作 - 策略主动调用
    # ================================================================

    def get_position(self, symbol: str = None):
        """获取持仓"""
        if self.executor:
            return self.executor.get_position(symbol)
        return None

    def get_position_size(self, symbol: str) -> int:
        """获取指定标的的持仓数量"""
        if self.executor and hasattr(self.executor, 'get_position_size'):
            return self.executor.get_position_size(symbol)
        return 0

    def get_sellable_volume(self, symbol: str) -> int:
        """获取指定标的的可卖出数量

        T+1品种（默认）：当天买入不可卖出，可卖 = 总持仓 - 当天买入量
        T+0品种（可转债、部分基金等）：可卖 = 总持仓

        Returns:
            可卖出数量
        """
        if not self.is_t_plus_1(symbol):
            return self.get_position_size(symbol)

        self._check_trade_date_rollover()
        total = self.get_position_size(symbol)
        today_bought = self._today_buys.get(symbol, 0)
        return max(0, total - today_bought)

    def is_t_plus_1(self, symbol: str) -> bool:
        """判断指定标的是否适用T+1规则

        优先级：逐标的覆盖 > 全局参数 > 自动推断

        Args:
            symbol: 标的代码

        Returns:
            True表示T+1（当天买入不可卖出），False表示T+0
        """
        if symbol in self._t_plus_1_overrides:
            return self._t_plus_1_overrides[symbol]
        return getattr(self.params, 't_plus_1', True)

    def set_t_plus_1(self, symbol: str, t_plus_1: bool):
        """设置指定标的的T+1/T+0规则

        用于覆盖全局默认设置，例如：
        - 可转债设为T+0: set_t_plus_1('123456.SZ', False)
        - T+0基金设为T+0: set_t_plus_1('510050.SH', False)

        Args:
            symbol: 标的代码
            t_plus_1: True为T+1（当天买入不可卖出），False为T+0
        """
        self._t_plus_1_overrides[symbol] = t_plus_1

    def _check_trade_date_rollover(self):
        """检查是否跨日，跨日则清零当天买入记录"""
        current_date = self.get_current_date()
        if current_date is not None:
            date_key = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
        else:
            return
        if date_key != self._current_trade_date:
            self._current_trade_date = date_key
            self._today_buys.clear()

    def _update_today_buys(self, symbol: str, volume: int):
        """记录当天买入数量"""
        self._check_trade_date_rollover()
        self._today_buys[symbol] = self._today_buys.get(symbol, 0) + volume

    def get_cash(self) -> float:
        """获取可用现金"""
        if self.executor and hasattr(self.executor, 'get_cash'):
            return self.executor.get_cash()
        return 0.0

    def get_account(self):
        """获取账户信息"""
        if self.executor:
            return self.executor.get_account()
        return None

    # ================================================================
    # 数据访问 - 通过数据适配器
    # ================================================================

    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取指定标的的当前价格"""
        if self._data_adapter:
            return self._data_adapter.get_current_price(symbol)
        return None

    def get_unadjusted_price(self, symbol: str) -> Optional[float]:
        """获取指定标的的不复权（实际）价格

        用于股息率等需要真实市场价格的计算。
        回测模式下通过 data_processor 获取不复权行情数据，
        实盘模式下与 get_current_price 一致（实盘价格本身就是实际价格）。

        优化策略：
        1. 同一天同一股票的结果会被缓存（日期级缓存）
        2. 首次查询某股票时加载整个回测期间的DataFrame并缓存，后续日期直接查找

        Returns:
            不复权价格，无法获取时返回 None
        """
        current_date = self.get_current_date()

        if current_date != self._unadjusted_price_cache_date:
            self._unadjusted_price_cache.clear()
            self._unadjusted_price_cache_date = current_date

        if symbol in self._unadjusted_price_cache:
            return self._unadjusted_price_cache[symbol]

        result = self._get_unadjusted_price_from_df_cache(symbol, current_date)

        if result is not None and result > 0:
            self._unadjusted_price_cache[symbol] = result

        return result

    def _get_unadjusted_price_from_df_cache(self, symbol: str, current_date) -> Optional[float]:
        import pandas as pd

        cached_df = self._unadjusted_price_df_cache.get(symbol)
        if cached_df is not None and not cached_df.empty:
            if isinstance(cached_df.index, pd.DatetimeIndex):
                ts = pd.Timestamp(current_date)
                pos = cached_df.index.searchsorted(ts, side='right') - 1
                if pos >= 0:
                    price = float(cached_df.iloc[pos]['close'])
                    if price > 0:
                        return price
            return self.get_current_price(symbol)

        if self._data_processor is not None and hasattr(self._data_processor, 'get_raw_data'):
            if current_date is None:
                return self.get_current_price(symbol)

            start_str = self._data_start_date if self._data_start_date else (self._backtest_start_date.strftime('%Y-%m-%d') if self._backtest_start_date else current_date.strftime('%Y-%m-%d'))
            end_str = self._data_end_date if self._data_end_date else (self._backtest_end_date.strftime('%Y-%m-%d') if self._backtest_end_date else current_date.strftime('%Y-%m-%d'))
            try:
                raw_df = self._data_processor.get_raw_data(
                    symbol, start_str, end_str, '1d', skip_current_year_refresh=True
                )
                if raw_df is not None and not raw_df.empty:
                    if isinstance(raw_df.index, pd.DatetimeIndex) and 'close' in raw_df.columns:
                        self._unadjusted_price_df_cache[symbol] = raw_df
                    ts = pd.Timestamp(current_date)
                    if isinstance(raw_df.index, pd.DatetimeIndex):
                        pos = raw_df.index.searchsorted(ts, side='right') - 1
                        if pos >= 0 and 'close' in raw_df.columns:
                            price = float(raw_df.iloc[pos]['close'])
                            if price > 0:
                                return price
                    if 'close' in raw_df.columns:
                        price = float(raw_df['close'].iloc[-1])
                        if price > 0:
                            return price
            except Exception as e:
                from core.data.futu import FutuServiceError
                if isinstance(e, FutuServiceError):
                    raise
                self.logger.debug(f"获取不复权价格失败 {symbol}: {e}")

        return self.get_current_price(symbol)

    def get_close_prices(self, symbol: str, period: int = None) -> List[float]:
        """获取指定标的的收盘价序列"""
        if self._data_adapter:
            return self._data_adapter.get_close_prices(symbol, period)
        return []

    def get_ohlcv_data(self, symbol: str, period: int = None) -> List[Dict[str, float]]:
        """获取指定标的的OHLCV数据序列

        Args:
            symbol: 标的代码
            period: 获取的周期数，None表示全部

        Returns:
            [{'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}, ...]
        """
        if self._data_adapter and hasattr(self._data_adapter, 'get_ohlcv_data'):
            return self._data_adapter.get_ohlcv_data(symbol, period)
        return []

    def get_current_date(self) -> Optional[Any]:
        """获取当前日期"""
        if self._data_adapter:
            return self._data_adapter.get_current_date()
        return None

    def get_current_datetime(self) -> Optional[Any]:
        """获取当前完整的日期时间（含时分秒）"""
        if self._data_adapter and hasattr(self._data_adapter, 'get_current_datetime'):
            return self._data_adapter.get_current_datetime()
        return None

    def get_symbols(self) -> List[str]:
        """获取策略需要的标的列表"""
        symbol = getattr(self.params, 'symbol', None)
        if symbol:
            return [symbol]
        if self._data_adapter:
            return self._data_adapter.get_symbols()
        return []

    def is_suspended(self, symbol: str) -> bool:
        if self._data_adapter and hasattr(self._data_adapter, 'is_suspended'):
            return self._data_adapter.is_suspended(symbol)
        return False

    def is_limit_up(self, symbol: str) -> bool:
        if self._data_adapter and hasattr(self._data_adapter, 'is_limit_up'):
            return self._data_adapter.is_limit_up(symbol)
        return False

    def is_limit_down(self, symbol: str) -> bool:
        if self._data_adapter and hasattr(self._data_adapter, 'is_limit_down'):
            return self._data_adapter.is_limit_down(symbol)
        return False

    def get_lookback_days(self) -> int:
        """获取需要的历史数据天数"""
        return getattr(self.params, 'lookback_period', 30) + 10

    # ================================================================
    # 财报数据访问 - 通过财务数据适配器
    # ================================================================

    def get_financial_field(self, stock_code: str, table_name: str,
                            field: str, date=None) -> Optional[Any]:
        """获取指定股票的最新已披露财务字段值

        Args:
            stock_code: 股票代码，如 '000001.SZ'
            table_name: 报表名称，如 'Balance', 'Income', 'Pershareindex'
            field: 字段名，如 'total_assets', 'eps_diluted'
            date: 查询日期，默认使用当前回测日期

        Returns:
            字段值，无数据返回None

        示例:
            eps = self.get_financial_field('000001.SZ', 'Pershareindex', 'eps_diluted')
            revenue = self.get_financial_field('000001.SZ', 'Income', 'total_operate_income')
        """
        if self._financial_data_adapter:
            query_date = date
            if query_date is None:
                current = self.get_current_date()
                if current is not None:
                    query_date = current if isinstance(current, dt_module.date) else None
            return self._financial_data_adapter.get_financial_field(
                stock_code, table_name, field, query_date
            )
        return None

    def get_financial_fields(self, stock_code: str, table_name: str,
                             fields: List[str], date=None) -> Dict[str, Any]:
        """获取指定股票的最新已披露财务多个字段值

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            fields: 字段名列表
            date: 查询日期

        Returns:
            { field1: value1, field2: value2, ... }
        """
        if self._financial_data_adapter:
            query_date = date
            if query_date is None:
                current = self.get_current_date()
                if current is not None:
                    query_date = current if isinstance(current, dt_module.date) else None
            return self._financial_data_adapter.get_financial_fields(
                stock_code, table_name, fields, query_date
            )
        return {f: None for f in fields}

    def get_financial_fields_batch(self, stock_list: List[str], table_name: str,
                                   fields: List[str], date=None) -> Dict[str, Dict[str, Any]]:
        """批量获取多只股票的最新已披露财务字段值

        Args:
            stock_list: 股票代码列表
            table_name: 报表名称
            fields: 字段名列表
            date: 查询日期，默认使用当前回测日期

        Returns:
            { stock_code: { field1: value1, field2: value2, ... }, ... }
        """
        if self._financial_data_adapter:
            query_date = date
            if query_date is None:
                current = self.get_current_date()
                if current is not None:
                    query_date = current if isinstance(current, dt_module.date) else None
            return self._financial_data_adapter.get_financial_fields_batch(
                stock_list, table_name, fields, query_date
            )
        return {stock: {f: None for f in fields} for stock in stock_list}

    def get_financial_history(self, stock_code: str, table_name: str,
                              field: str, count: int = 4, date=None) -> List[Any]:
        """获取指定股票最近N期的财务字段值

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            field: 字段名
            count: 期数，默认4期
            date: 查询日期

        Returns:
            字段值列表，按时间升序
        """
        if self._financial_data_adapter:
            query_date = date
            if query_date is None:
                current = self.get_current_date()
                if current is not None:
                    query_date = current if isinstance(current, dt_module.date) else None
            return self._financial_data_adapter.get_financial_history(
                stock_code, table_name, field, count, query_date
            )
        return []

    def screen_stocks(self, condition, stock_pool: Optional[List[str]] = None) -> List[str]:
        """基于财务条件筛选股票

        Args:
            condition: 筛选条件函数，参数为股票代码，返回bool
            stock_pool: 股票池，默认使用缓存中的全部股票

        Returns:
            满足条件的股票代码列表

        示例:
            # 筛选EPS > 0.5的股票
            selected = self.screen_stocks(
                lambda s: (self.get_financial_field(s, 'Pershareindex', 'eps_diluted') or 0) > 0.5
            )
        """
        if self._financial_data_adapter:
            return self._financial_data_adapter.screen_stocks(condition, stock_pool)
        return []

    def rank_stocks(self, score_func, stock_pool: Optional[List[str]] = None,
                    ascending: bool = False, top_n: Optional[int] = None) -> List[tuple]:
        """基于财务指标对股票排序

        Args:
            score_func: 评分函数，参数为股票代码，返回数值（None表示排除）
            stock_pool: 股票池
            ascending: 是否升序
            top_n: 返回前N名

        Returns:
            [(stock_code, score), ...] 排序后的列表

        示例:
            # 按营收增长率排序，取前10
            ranked = self.rank_stocks(
                lambda s: self._financial_data_adapter.compute_growth_rate(
                    s, 'Income', 'total_operate_income'
                ),
                top_n=10
            )
        """
        if self._financial_data_adapter:
            return self._financial_data_adapter.rank_stocks(score_func, stock_pool, ascending, top_n)
        return []

    def compute_growth_rate(self, stock_code: str, table_name: str,
                            field: str, periods: int = 1, date=None) -> Optional[float]:
        """计算财务字段的同比增长率

        Args:
            stock_code: 股票代码
            table_name: 报表名称
            field: 字段名
            periods: 增长期数
            date: 查询日期

        Returns:
            增长率，如 0.15 表示增长15%
        """
        if self._financial_data_adapter:
            query_date = date
            if query_date is None:
                current = self.get_current_date()
                if current is not None:
                    query_date = current if isinstance(current, dt_module.date) else None
            return self._financial_data_adapter.compute_growth_rate(
                stock_code, table_name, field, periods, query_date
            )
        return None

    def get_industry(self, stock_code: str) -> Optional[str]:
        """获取指定股票的行业分类

        Args:
            stock_code: 股票代码

        Returns:
            行业名称，无数据返回None
        """
        if self._financial_data_adapter:
            return self._financial_data_adapter.get_industry(stock_code)
        return None

    def get_industry_mapping(self) -> Dict[str, str]:
        """获取完整的行业分类映射"""
        if self._financial_data_adapter:
            return self._financial_data_adapter.get_industry_mapping()
        return {}

    def get_latest_dvps(self, stock_code: str) -> Optional[float]:
        """获取最近一次每股派息金额"""
        if self._financial_data_adapter:
            return self._financial_data_adapter.get_latest_dvps(stock_code)
        return None

    def get_dvps_history(self, stock_code: str, count: int = 3) -> List[float]:
        """获取最近N次每股派息金额"""
        if self._financial_data_adapter:
            return self._financial_data_adapter.get_dvps_history(stock_code, count)
        return []

    def get_dividend_yield(self, stock_code: str, use_avg: bool = True) -> Optional[float]:
        """计算股息率

        使用不复权（实际）价格作为分母，避免后复权价格导致股息率偏低。
        长期高分红股票的后复权价格远高于实际价格，若用后复权价格计算
        会系统性地低估其股息率，扭曲选股排名。

        Args:
            stock_code: 股票代码
            use_avg: 是否使用近N年平均派息，否则用最近一次

        Returns:
            股息率 (0~1之间)，无数据返回None
        """
        if use_avg:
            dvps_list = self.get_dvps_history(stock_code, count=3)
            if not dvps_list:
                return None
            dvps = sum(dvps_list) / len(dvps_list)
        else:
            dvps = self.get_latest_dvps(stock_code)
            if dvps is None:
                return None

        price = self.get_unadjusted_price(stock_code)
        if price is None or price <= 0:
            return None

        return dvps / price

    # ================================================================
    # 通用工具
    # ================================================================

    def log(self, txt, dt=None, level='debug'):
        """日志记录

        Args:
            txt: 日志内容
            dt: 日期时间，可选
            level: 日志级别，'debug'/'info'/'warning'，默认 'debug'
        """
        if dt:
            log_text = f'{dt.isoformat()}, {txt}'
        else:
            log_text = txt
        log_fn = getattr(self.logger, level, self.logger.debug)
        log_fn(log_text)

    # ================================================================
    # 向后兼容别名
    # ================================================================

    def execute_buy(self, symbol: str, price: float, volume: int):
        """买入操作 - 向后兼容别名，推荐使用 buy()"""
        return self.buy(symbol, price, volume)

    def execute_sell(self, symbol: str, price: float, volume: int):
        """卖出操作 - 向后兼容别名，推荐使用 sell()"""
        return self.sell(symbol, price, volume)

    def set_risk_controller(self, controller: 'RiskController') -> None:
        self._risk_controller = controller

    def _check_risk_before_buy(self, symbol: str, price: float, volume: int) -> bool:
        if self._risk_controller is None:
            return True
        return self._risk_controller.check_buy(self, symbol, price, volume)

    def _check_risk_before_sell(self, symbol: str, price: float, volume: int) -> bool:
        if self._risk_controller is None:
            return True
        return self._risk_controller.check_sell(self, symbol, price, volume)


class RiskController:
    """风控控制器 - 提供止损、仓位限制、最大回撤控制等风控机制

    使用方式：
        risk = RiskController(
            max_position_ratio=0.3,
            max_drawdown_limit=0.15,
            stop_loss_ratio=0.08,
        )
        strategy.set_risk_controller(risk)
    """

    def __init__(
        self,
        max_position_ratio: float = 1.0,
        max_drawdown_limit: float = 0.2,
        stop_loss_ratio: float = 0.1,
        max_single_order_ratio: float = 0.3,
        peak_value: float = 0.0,
        max_industry_ratio: float = 1.0,
        max_var_limit: float = 0.0,
        max_volume_ratio: float = 0.0,
        var_window: int = 252,
        var_confidence: float = 0.95,
    ):
        self.max_position_ratio = max_position_ratio
        self.max_drawdown_limit = max_drawdown_limit
        self.stop_loss_ratio = stop_loss_ratio
        self.max_single_order_ratio = max_single_order_ratio
        self.max_industry_ratio = max_industry_ratio
        self.max_var_limit = max_var_limit
        self.max_volume_ratio = max_volume_ratio
        self.var_window = var_window
        self.var_confidence = var_confidence
        self._peak_value = peak_value
        self._entry_prices: Dict[str, float] = {}
        self._triggered = False
        self._daily_returns: deque = deque(maxlen=var_window)
        self._prev_portfolio_value: Optional[float] = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def check_buy(self, strategy: StrategyLogic, symbol: str, price: float, volume: int) -> bool:
        if self._triggered:
            self.logger.warning(f'风控已触发，禁止买入: {symbol}')
            return False

        if self._check_drawdown(strategy):
            self._triggered = True
            self.logger.warning(f'最大回撤超限，禁止买入: {symbol}')
            return False

        cash = strategy.get_cash()
        order_value = price * volume
        account_value = self._get_account_value(strategy)

        if account_value > 0 and order_value / account_value > self.max_single_order_ratio:
            self.logger.warning(
                f'单笔买入占比超限: {order_value / account_value:.2%} > {self.max_single_order_ratio:.2%}, {symbol}'
            )
            return False

        current_pos_value = self._get_position_value(strategy)
        if account_value > 0 and (current_pos_value + order_value) / account_value > self.max_position_ratio:
            self.logger.warning(
                f'总仓位占比超限: {(current_pos_value + order_value) / account_value:.2%} > {self.max_position_ratio:.2%}, {symbol}'
            )
            return False

        if self.max_industry_ratio < 1.0:
            if not self._check_industry_concentration(strategy, symbol, order_value, account_value):
                return False

        if self.max_var_limit > 0:
            self._update_daily_return(strategy)
            if not self._check_var(strategy, order_value, account_value):
                return False

        if self.max_volume_ratio > 0:
            if not self._check_liquidity(strategy, symbol, volume):
                return False

        self._entry_prices[symbol] = price
        return True

    def check_sell(self, strategy: StrategyLogic, symbol: str, price: float, volume: int) -> bool:
        if self._triggered:
            pos_size = strategy.get_position_size(symbol)
            if pos_size > 0:
                return True
            return False

        entry_price = self._entry_prices.get(symbol)
        if entry_price and entry_price > 0 and price < entry_price:
            loss_ratio = (entry_price - price) / entry_price
            if loss_ratio >= self.stop_loss_ratio:
                self.logger.warning(
                    f'止损触发: {symbol}, 入场价: {entry_price:.2f}, 当前价: {price:.2f}, 亏损: {loss_ratio:.2%}'
                )
                return True

        return True

    def _check_drawdown(self, strategy: StrategyLogic) -> bool:
        account_value = self._get_account_value(strategy)
        if account_value <= 0:
            return False

        if self._peak_value <= 0:
            self._peak_value = account_value
        elif account_value > self._peak_value:
            self._peak_value = account_value

        if self._peak_value > 0:
            drawdown = (self._peak_value - account_value) / self._peak_value
            if drawdown >= self.max_drawdown_limit:
                self.logger.warning(
                    f'最大回撤触发: {drawdown:.2%} >= {self.max_drawdown_limit:.2%}'
                )
                return True

        return False

    def _get_account_value(self, strategy: StrategyLogic) -> float:
        cash = strategy.get_cash()
        pos_value = self._get_position_value(strategy)
        return cash + pos_value

    def _get_position_value(self, strategy: StrategyLogic) -> float:
        total = 0.0
        symbols = strategy.get_symbols()
        for symbol in symbols:
            pos_size = strategy.get_position_size(symbol)
            if pos_size > 0:
                price = strategy.get_current_price(symbol)
                if price:
                    total += pos_size * price
        return total

    def _check_industry_concentration(self, strategy: StrategyLogic, symbol: str,
                                       order_value: float, account_value: float) -> bool:
        industry = strategy.get_industry(symbol)
        if industry is None:
            return True
        industry_value = 0.0
        symbols = strategy.get_symbols()
        for s in symbols:
            pos_size = strategy.get_position_size(s)
            if pos_size > 0:
                s_industry = strategy.get_industry(s)
                if s_industry == industry:
                    price = strategy.get_current_price(s)
                    if price:
                        industry_value += pos_size * price
        industry_value += order_value
        if account_value > 0 and industry_value / account_value > self.max_industry_ratio:
            self.logger.warning(
                f'行业集中度超限: {industry} 占比 {industry_value / account_value:.2%} > {self.max_industry_ratio:.2%}, {symbol}'
            )
            return False
        return True

    def _update_daily_return(self, strategy: StrategyLogic) -> None:
        current_value = self._get_account_value(strategy)
        if self._prev_portfolio_value is not None and self._prev_portfolio_value > 0:
            daily_ret = (current_value - self._prev_portfolio_value) / self._prev_portfolio_value
            self._daily_returns.append(daily_ret)
        self._prev_portfolio_value = current_value

    def _check_var(self, strategy: StrategyLogic, order_value: float, account_value: float) -> bool:
        if len(self._daily_returns) < 20:
            return True
        returns_arr = np.array(self._daily_returns)
        sorted_returns = np.sort(returns_arr)
        index = int((1 - self.var_confidence) * len(sorted_returns))
        var_value = abs(sorted_returns[index])
        projected_var = var_value * (account_value + order_value)
        if projected_var / account_value > self.max_var_limit:
            self.logger.warning(
                f'VaR超限: 预估VaR {projected_var / account_value:.2%} > 限制 {self.max_var_limit:.2%}'
            )
            return False
        return True

    def _check_liquidity(self, strategy: StrategyLogic, symbol: str, volume: int) -> bool:
        ohlcv_data = strategy.get_ohlcv_data(symbol, period=20)
        if not ohlcv_data:
            return True
        volumes = [bar.get('volume', 0) for bar in ohlcv_data if bar.get('volume', 0) > 0]
        if not volumes:
            return True
        avg_volume = sum(volumes) / len(volumes)
        if avg_volume > 0 and volume / avg_volume > self.max_volume_ratio:
            self.logger.warning(
                f'流动性风控超限: 买入量 {volume} / 日均成交量 {avg_volume:.0f} = {volume / avg_volume:.2%} > {self.max_volume_ratio:.2%}, {symbol}'
            )
            return False
        return True

    def reset(self) -> None:
        self._triggered = False
        self._peak_value = 0.0
        self._entry_prices.clear()
        self._daily_returns.clear()
        self._prev_portfolio_value = None
