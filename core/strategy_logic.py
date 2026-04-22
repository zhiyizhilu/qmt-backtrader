import copy
from typing import Dict, List, Optional, Any
import datetime as dt_module
import logging
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

    def __init__(self, order_id: str = '', symbol: str = '', direction: str = '',
                 price: float = 0.0, volume: int = 0, status: str = '',
                 executed_volume: int = 0, executed_price: float = 0.0,
                 commission: float = 0.0, **kwargs):
        self.order_id = order_id
        self.symbol = symbol
        self.direction = direction
        self.price = price
        self.volume = volume
        self.status = status
        self.executed_volume = executed_volume
        self.executed_price = executed_price
        self.commission = commission
        for key, value in kwargs.items():
            setattr(self, key, value)

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
        for key, value in kwargs.items():
            setattr(self, key, value)

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

    params = ()

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
        self._orders: Dict[str, OrderInfo] = {}
        self._risk_controller: Optional['RiskController'] = None
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def set_data_adapter(self, adapter: MarketDataAdapter) -> None:
        """设置数据适配器"""
        self._data_adapter = adapter

    def set_financial_data_adapter(self, adapter) -> None:
        """设置财务数据适配器

        Args:
            adapter: FinancialDataAdapter 实例
        """
        self._financial_data_adapter = adapter

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
        """买入操作 - 通过执行器路由到对应环境

        Args:
            symbol: 标的代码
            price: 买入价格
            volume: 买入数量

        Returns:
            订单ID或订单对象
        """
        self.log(f'买入委托: {symbol}, 价格: {price:.2f}, 数量: {volume}')
        if not self._check_risk_before_buy(symbol, price, volume):
            self.log(f'风控拒绝买入: {symbol}')
            return None
        if self.executor:
            return self.executor.execute_buy(symbol, price, volume)
        self.log(f'[WARNING] 买入失败: executor未设置!')
        return None

    def sell(self, symbol: str, price: float, volume: int):
        """卖出操作 - 通过执行器路由到对应环境

        Args:
            symbol: 标的代码
            price: 卖出价格
            volume: 卖出数量

        Returns:
            订单ID或订单对象
        """
        self.log(f'卖出委托: {symbol}, 价格: {price:.2f}, 数量: {volume}')
        if not self._check_risk_before_sell(symbol, price, volume):
            self.log(f'风控拒绝卖出: {symbol}')
            return None
        if self.executor:
            return self.executor.execute_sell(symbol, price, volume)
        self.log(f'[WARNING] 卖出失败: executor未设置!')
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

    def get_close_prices(self, symbol: str, period: int = None) -> List[float]:
        """获取指定标的的收盘价序列"""
        if self._data_adapter:
            return self._data_adapter.get_close_prices(symbol, period)
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

        price = self.get_current_price(stock_code)
        if price is None or price <= 0:
            return None

        return dvps / price

    # ================================================================
    # 通用工具
    # ================================================================

    def log(self, txt, dt=None):
        """日志记录"""
        if dt:
            log_text = f'{dt.isoformat()}, {txt}'
        else:
            log_text = txt
        self.logger.info(log_text)

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
    ):
        self.max_position_ratio = max_position_ratio
        self.max_drawdown_limit = max_drawdown_limit
        self.stop_loss_ratio = stop_loss_ratio
        self.max_single_order_ratio = max_single_order_ratio
        self._peak_value = peak_value
        self._entry_prices: Dict[str, float] = {}
        self._triggered = False
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

        if account_value > self._peak_value:
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

    def reset(self) -> None:
        self._triggered = False
        self._peak_value = 0.0
        self._entry_prices.clear()
