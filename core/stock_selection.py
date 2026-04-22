import datetime as dt_module
from abc import abstractmethod
from typing import Dict, List, Optional, Any
from core.strategy_logic import StrategyLogic, BarData


class StockSelectionStrategy(StrategyLogic):
    """选股策略基类 - 支持定期选股+组合调仓模式

    继承此类的策略只需实现 select_stocks() 方法，
    框架会在每个调仓日自动执行选股和调仓。

    核心机制：
    1. is_rebalance_day() 判断当前是否为调仓日
    2. select_stocks() 返回目标持仓列表
    3. rebalance_to() 自动计算买卖并执行调仓

    使用方式：
        class MyStrategy(StockSelectionStrategy):
            params = (
                ('rebalance_freq', 'monthly'),
                ('max_stocks', 10),
            )

            def select_stocks(self):
                # 基于财务条件筛选
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

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._current_holdings: Dict[str, int] = {}
        self._last_rebalance_date: Optional[dt_module.date] = None
        self._rebalance_count: int = 0
        self._selected_stocks: List[str] = []

    def on_bar(self, bar: BarData):
        """K线数据到达 - 执行选股调仓逻辑"""
        current_date = self.get_current_date()
        if current_date is None:
            return

        if self.is_rebalance_day(current_date):
            self.log(f'[选股] 调仓日: {current_date}')
            self._execute_rebalance(current_date)

    def is_rebalance_day(self, current_date: dt_module.date) -> bool:
        """判断当前日期是否为调仓日

        Args:
            current_date: 当前日期

        Returns:
            是否为调仓日
        """
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
        """调仓到目标持仓

        自动计算需要卖出和买入的标的，等权重分配资金。
        跳过当前无数据（价格为None）的标的，避免对NaN数据执行交易。

        Args:
            target_stocks: 目标持仓股票列表
        """
        current_symbols = set(self._current_holdings.keys())
        target_symbols = set(target_stocks)

        sell_symbols = current_symbols - target_symbols
        buy_symbols = target_symbols - current_symbols
        hold_symbols = current_symbols & target_symbols

        for symbol in sell_symbols:
            pos_size = self._current_holdings.get(symbol, 0)
            if pos_size > 0:
                price = self.get_current_price(symbol)
                if price and price > 0:
                    self.sell(symbol, price, pos_size)
                    self.log(f'调仓卖出: {symbol}, 数量: {pos_size}, 价格: {price:.2f}')
            del self._current_holdings[symbol]

        for symbol in hold_symbols:
            pos_size = self._current_holdings.get(symbol, 0)
            if pos_size > 0:
                del self._current_holdings[symbol]

        cash = self.get_cash()
        position_ratio = getattr(self.params, 'position_ratio', 0.95)
        available_cash = cash * position_ratio

        # 过滤掉当前无数据的股票（价格为None表示该日期尚无实际行情）
        tradeable_symbols = []
        skipped_symbols = []
        for symbol in buy_symbols | hold_symbols:
            price = self.get_current_price(symbol)
            if price is not None and price > 0:
                tradeable_symbols.append(symbol)
            else:
                skipped_symbols.append(symbol)

        if skipped_symbols:
            self.log(f'调仓跳过无数据股票: {skipped_symbols}')

        if not tradeable_symbols:
            return

        per_stock_cash = available_cash / len(tradeable_symbols)

        for symbol in tradeable_symbols:
            price = self.get_current_price(symbol)
            if price and price > 0:
                volume = int(per_stock_cash / price / 100) * 100
                if volume >= 100:
                    self.buy(symbol, price, volume)
                    self._current_holdings[symbol] = volume
                    self.log(f'调仓买入: {symbol}, 数量: {volume}, 价格: {price:.2f}')

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
