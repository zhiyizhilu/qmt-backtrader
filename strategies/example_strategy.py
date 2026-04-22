from core.strategy_logic import StrategyLogic, BarData, OrderInfo, TradeInfo
from strategies import register_strategy
import numpy as np


@register_strategy('double_ma', default_kwargs={'fast_period': 5, 'slow_period': 20, 'symbol': '000001.SZ'},
                   backtest_config={'cash': 200000, 'commission': 0.0001,
                                    'start_date': '2025-07-10', 'end_date': '2026-04-17'})
class DoubleMAStrategy(StrategyLogic):
    """双均线策略 - 仅定义交易逻辑，通过统一接口访问数据和执行交易

    事件回调：
    - on_bar():  K线到达时执行均线交叉判断
    - on_order(): 委托状态变化时的处理
    """

    params = (
        ('fast_period', 5),
        ('slow_period', 20),
        ('symbol', '159915.SZ'),
        ('position_ratio', 0.9),
    )

    def __init__(self, executor=None, **kwargs):
        """初始化策略"""
        super().__init__(executor, **kwargs)

        self.fast_ma = None
        self.slow_ma = None
        self.crossover = None

    def on_bar(self, bar: BarData):
        """K线数据到达时触发 - 执行均线交叉判断"""
        symbol = self.params.symbol
        close_prices = self.get_close_prices(symbol)

        if len(close_prices) < self.params.slow_period:
            if len(close_prices) <= 3:  # 只在前几个bar输出
                self.log(f'[DEBUG] {symbol} 收盘价不足: 有{len(close_prices)}条, 需要{self.params.slow_period}条')
            return

        fast_ma, slow_ma, crossover = self._calculate_indicators(close_prices)

        if fast_ma is None:
            return

        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.crossover = crossover

        pos_size = self.get_position_size(symbol)

        if crossover == 1 and pos_size == 0:
            current_price = self.get_current_price(symbol)
            if current_price and current_price > 0:
                cash = self.get_cash()
                buy_volume = int(cash * self.params.position_ratio / current_price / 100) * 100
                if buy_volume >= 100:
                    self.log(f'买入信号: {symbol}, 价格: {current_price:.2f}, 数量: {buy_volume}')
                    self.buy(symbol, current_price, buy_volume)
                else:
                    self.log(f'[DEBUG] 买入量不足100股: 现金={cash:.2f}, 价格={current_price:.2f}, 计算量={buy_volume}')
            else:
                self.log(f'[DEBUG] 无法获取价格: {symbol}, price={current_price}')
        elif crossover == -1 and pos_size > 0:
            current_price = self.get_current_price(symbol)
            if current_price:
                self.log(f'卖出信号: {symbol}, 价格: {current_price:.2f}')
                self.sell(symbol, current_price, pos_size)

    def on_order(self, order: OrderInfo):
        super().on_order(order)
        if order.is_completed:
            direction = '买入' if order.is_buy else '卖出'
            self.log(f'{direction}成交: {order.symbol}, 价格: {order.executed_price:.2f}, 数量: {order.executed_volume}')
        elif order.status in (order.STATUS_CANCELED, order.STATUS_REJECTED, order.STATUS_MARGIN):
            self.log(f'委托异常: {order.symbol}, 状态: {order.status}')

    def _calculate_indicators(self, close_prices):
        """计算指标"""
        if len(close_prices) >= self.params.slow_period:
            fast_ma = np.mean(close_prices[-self.params.fast_period:])
            slow_ma = np.mean(close_prices[-self.params.slow_period:])

            if self.fast_ma is not None and self.slow_ma is not None:
                prev_fast_ma = self.fast_ma
                prev_slow_ma = self.slow_ma

                if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                    crossover = 1
                elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                    crossover = -1
                else:
                    crossover = 0
            else:
                crossover = 0

            return fast_ma, slow_ma, crossover
        return None, None, None
