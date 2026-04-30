import datetime as dt_module
from abc import abstractmethod
from typing import Dict, List, Optional, Any
from core.strategy_logic import StrategyLogic, BarData, OrderInfo


class StockSelectionStrategy(StrategyLogic):
    """选股策略基类 - 支持定期选股+组合调仓模式

    继承此类的策略只需实现 select_stocks() 方法，
    框架会在每个调仓日自动执行选股和调仓。

    核心机制：
    1. is_rebalance_day() 判断当前是否为调仓日
    2. select_stocks() 返回目标持仓列表
    3. rebalance_to() 自动计算买卖并执行调仓

    调仓模式：
    - 回测模式：同步执行，卖出和买入在同一 bar 内完成
      （backtrader 会在下一根K线统一处理，先卖后买，资金自动衔接）
    - 实盘/模拟盘模式：两阶段异步执行
      阶段1：下单卖出，等待成交确认
      阶段2：所有卖出确认后，根据实际可用资金下单买入
      避免卖出未成交时资金不足导致买入失败

    使用方式：
        class MyStrategy(StockSelectionStrategy):
            params = (
                ('rebalance_freq', 'monthly'),
                ('max_stocks', 10),
            )

            def select_stocks(self):
                return self.screen_stocks(
                    lambda s: (self.get_financial_field(s, 'Pershareindex', 'eps_diluted') or 0) > 0.5,
                    top_n=self.params.max_stocks
                )
    """

    params = (
        ('rebalance_freq', 'monthly'),
        ('max_stocks', 10),
        ('position_ratio', 0.95),
        ('stock_pool', None),
    )

    REBALANCE_FREQ_WEEKLY = 'weekly'
    REBALANCE_FREQ_BIWEEKLY = 'biweekly'
    REBALANCE_FREQ_MONTHLY = 'monthly'
    REBALANCE_FREQ_QUARTERLY = 'quarterly'

    PHASE_IDLE = 0
    PHASE_SELLING = 1
    PHASE_BUYING = 2

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._current_holdings: Dict[str, int] = {}
        self._last_rebalance_date: Optional[dt_module.date] = None
        self._rebalance_count: int = 0
        self._selected_stocks: List[str] = []
        self._rebalance_phase: int = self.PHASE_IDLE
        self._pending_sell_symbols: set = set()
        self._rebalance_target_stocks: List[str] = []
        self._per_stock_target: float = 0.0

    def _is_live_trading(self) -> bool:
        from core.executor import QMTExecutor
        return isinstance(self.executor, QMTExecutor)

    def on_bar(self, bar: BarData):
        """K线数据到达 - 执行选股调仓逻辑"""
        current_date = self.get_current_date()
        if current_date is None:
            return

        if self._rebalance_phase != self.PHASE_IDLE:
            return

        if self.is_rebalance_day(current_date):
            self.log(f'[选股] 调仓日: {current_date}')
            self._execute_rebalance(current_date)

    def on_order(self, order: OrderInfo):
        """委托状态回调 - 实盘两阶段调仓的核心驱动

        在实盘模式下，当卖出委托完成/失败后，检查是否所有卖出都已确认，
        若是则进入买入阶段。
        """
        super().on_order(order)

        if self._rebalance_phase != self.PHASE_SELLING:
            return

        if order.symbol not in self._pending_sell_symbols:
            return

        if order.is_active:
            return

        self._pending_sell_symbols.discard(order.symbol)

        if order.is_completed:
            self.log(f'再平衡卖出确认: {order.symbol}, 成交量: {order.executed_volume}')
        else:
            self.log(f'再平衡卖出未成交: {order.symbol}, 状态: {order.status}')

        if not self._pending_sell_symbols:
            self.log('所有卖出订单已确认，进入买入阶段')
            self._rebalance_buy_phase(self._rebalance_target_stocks, self._per_stock_target)

    def is_rebalance_day(self, current_date: dt_module.date) -> bool:
        """判断当前日期是否为调仓日"""
        freq = getattr(self.params, 'rebalance_freq', self.REBALANCE_FREQ_MONTHLY)

        if self._last_rebalance_date is None:
            return True

        if freq == self.REBALANCE_FREQ_WEEKLY:
            delta = (current_date - self._last_rebalance_date).days
            return delta >= 5
        elif freq == self.REBALANCE_FREQ_BIWEEKLY:
            delta = (current_date - self._last_rebalance_date).days
            return delta >= 10
        elif freq == self.REBALANCE_FREQ_MONTHLY:
            if current_date.month != self._last_rebalance_date.month:
                return True
            return False
        elif freq == self.REBALANCE_FREQ_QUARTERLY:
            if current_date.month != self._last_rebalance_date.month:
                quarter_current = (current_date.month - 1) // 3
                quarter_last = (self._last_rebalance_date.month - 1) // 3
                return quarter_current != quarter_last
            return False

        return False

    @abstractmethod
    def select_stocks(self) -> List[str]:
        """选股逻辑 - 子类必须实现

        Returns:
            目标持仓的股票代码列表
        """
        pass

    def _execute_rebalance(self, current_date: dt_module.date):
        """执行调仓操作"""
        target_stocks = self.select_stocks()

        if not target_stocks:
            self.log(f'调仓日 {current_date}: 选股结果为空，清仓')
            self._sell_all()
            return

        max_stocks = getattr(self.params, 'max_stocks', 10)
        target_stocks = target_stocks[:max_stocks]

        self._selected_stocks = target_stocks
        self._last_rebalance_date = current_date
        self._rebalance_count += 1

        self.log(f'调仓 #{self._rebalance_count} @ {current_date}: 选中 {len(target_stocks)} 只股票')

        self.rebalance_to(target_stocks)

    def rebalance_to(self, target_stocks: List[str]):
        """调仓到目标持仓 - 等权重部分再平衡

        与全仓清仓再买入不同，此方法只调整偏差：
        - 不在目标列表的股票：全部清仓
        - 超出目标市值的持仓：卖出超出部分（减仓）
        - 低于目标市值的持仓：买入不足部分（补仓）
        - 新入选的股票：等权建仓

        回测模式：同步执行，卖出和买入在同一 bar 内完成
          （需配合 broker.set_checksubmit(False)，让卖出回款可用于买入）
        实盘模式：两阶段异步执行
          阶段1：下单卖出，通过 on_order 回调等待成交确认
          阶段2：所有卖出确认后，根据实际可用资金下单买入

        Args:
            target_stocks: 目标持仓股票列表
        """
        current_symbols = set(self._current_holdings.keys())
        target_symbols = set(target_stocks)

        cash = self.get_cash()
        position_value = 0.0
        for symbol, volume in self._current_holdings.items():
            price = self.get_current_price(symbol)
            if price and price > 0:
                position_value += price * volume

        total_assets = cash + position_value
        position_ratio = getattr(self.params, 'position_ratio', 0.95)
        investable = total_assets * position_ratio
        per_stock_target = investable / len(target_stocks) if target_stocks else 0

        self._per_stock_target = per_stock_target

        self.log(
            f'调仓计算: 总资产={total_assets:.0f}, '
            f'可投资={investable:.0f}, '
            f'每只目标={per_stock_target:.0f}'
        )

        full_sell_symbols = current_symbols - target_symbols
        hold_symbols = current_symbols & target_symbols

        has_sells = False
        pending_sell_symbols = set()

        for symbol in full_sell_symbols:
            pos_size = self._current_holdings.get(symbol, 0)
            if pos_size > 0:
                price = self.get_current_price(symbol)
                if price and price > 0:
                    self.sell(symbol, price, pos_size)
                    sell_value = price * pos_size
                    self.log(
                        f'调仓卖出(清仓): {symbol}, '
                        f'数量: {pos_size}, 价格: {price:.2f}, 市值: {sell_value:.0f}'
                    )
                    has_sells = True
                    pending_sell_symbols.add(symbol)
            del self._current_holdings[symbol]

        for symbol in hold_symbols:
            pos_size = self._current_holdings.get(symbol, 0)
            if pos_size > 0:
                price = self.get_current_price(symbol)
                if price and price > 0:
                    current_value = price * pos_size
                    if current_value > per_stock_target * 1.01:
                        excess_value = current_value - per_stock_target
                        sell_volume = int(excess_value / price / 100) * 100
                        if sell_volume >= 100:
                            self.sell(symbol, price, sell_volume)
                            self._current_holdings[symbol] = pos_size - sell_volume
                            self.log(
                                f'调仓卖出(减仓): {symbol}, '
                                f'卖出: {sell_volume}, 剩余: {pos_size - sell_volume}, '
                                f'价格: {price:.2f}'
                            )
                            has_sells = True
                            pending_sell_symbols.add(symbol)
                        else:
                            self.log(
                                f'调仓跳过(超出不足1手): {symbol}, 超出: {excess_value:.0f}'
                            )

        if self._is_live_trading() and has_sells:
            self._rebalance_phase = self.PHASE_SELLING
            self._pending_sell_symbols = pending_sell_symbols
            self._rebalance_target_stocks = list(target_stocks)
            self.log(f'再平衡阶段1: 等待 {len(pending_sell_symbols)} 只股票卖出确认')
        else:
            self._rebalance_buy_phase(target_stocks, per_stock_target)

    def _rebalance_buy_phase(self, target_stocks: List[str], per_stock_target: float = None):
        """调仓买入阶段 - 补仓至目标市值

        对于已持有但低于目标市值的股票，买入不足部分；
        对于新入选的股票，按目标市值等权建仓。

        Args:
            target_stocks: 目标持仓股票列表
            per_stock_target: 每只股票的目标市值（由 rebalance_to 预计算）
        """
        if self._is_live_trading():
            self._sync_holdings_from_executor()

        if per_stock_target is None:
            per_stock_target = getattr(self, '_per_stock_target', None)

        tradeable_symbols = []
        skipped_symbols = []
        for symbol in target_stocks:
            price = self.get_current_price(symbol)
            if price is not None and price > 0:
                tradeable_symbols.append(symbol)
            else:
                skipped_symbols.append(symbol)

        if skipped_symbols:
            self.log(f'调仓跳过无数据股票: {skipped_symbols}')

        if not tradeable_symbols:
            self._rebalance_phase = self.PHASE_IDLE
            return

        if per_stock_target is None:
            cash = self.get_cash()
            position_value = 0.0
            for symbol, volume in self._current_holdings.items():
                price = self.get_current_price(symbol)
                if price and price > 0:
                    position_value += price * volume
            total_assets = cash + position_value
            position_ratio = getattr(self.params, 'position_ratio', 0.95)
            investable = total_assets * position_ratio
            per_stock_target = investable / len(tradeable_symbols)
        elif len(tradeable_symbols) < len(target_stocks):
            per_stock_target = per_stock_target * len(target_stocks) / len(tradeable_symbols)

        for symbol in tradeable_symbols:
            current_volume = self._current_holdings.get(symbol, 0)
            price = self.get_current_price(symbol)
            if price and price > 0:
                current_value = current_volume * price
                deficit = per_stock_target - current_value
                if deficit > per_stock_target * 0.01:
                    buy_volume = int(deficit / price / 100) * 100
                    if buy_volume >= 100:
                        self.buy(symbol, price, buy_volume)
                        self._current_holdings[symbol] = current_volume + buy_volume
                        self.log(
                            f'调仓买入: {symbol}, '
                            f'数量: {buy_volume}, 价格: {price:.2f}'
                        )

        self._rebalance_phase = self.PHASE_IDLE

    def _sync_holdings_from_executor(self):
        """从执行器同步真实持仓到 _current_holdings

        实盘/模拟盘中，订单可能部分成交或未成交，
        内部持仓记录可能与实际不一致，需要从券商端同步。
        """
        synced = {}
        for symbol in list(self._current_holdings.keys()):
            real_size = self.get_position_size(symbol)
            if real_size > 0:
                synced[symbol] = real_size
                if real_size != self._current_holdings.get(symbol, 0):
                    self.log(f'持仓同步: {symbol} 内部={self._current_holdings.get(symbol, 0)}, 实际={real_size}')
        self._current_holdings = synced

    def _sell_all(self):
        """卖出所有持仓"""
        for symbol, pos_size in list(self._current_holdings.items()):
            if pos_size > 0:
                price = self.get_current_price(symbol)
                if price and price > 0:
                    self.sell(symbol, price, pos_size)
        self._current_holdings.clear()

    def get_current_holdings(self) -> Dict[str, int]:
        """获取当前持仓"""
        return dict(self._current_holdings)

    def get_selected_stocks(self) -> List[str]:
        """获取最近一次选股结果"""
        return list(self._selected_stocks)

    def get_rebalance_count(self) -> int:
        """获取调仓次数"""
        return self._rebalance_count

    def get_stock_pool(self) -> List[str]:
        """获取股票池

        优先使用参数中指定的股票池，否则从财务数据缓存获取
        """
        pool = getattr(self.params, 'stock_pool', None)
        if pool:
            return pool

        if self._financial_data_adapter:
            return self._financial_data_adapter.cache.get_stocks()

        return self.get_symbols()
