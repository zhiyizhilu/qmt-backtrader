from core.strategy_logic import StrategyLogic, BarData, OrderInfo, TradeInfo
from strategies import register_strategy
from strategies.config import ETF_CODES
import numpy as np
import datetime


@register_strategy('etf_rotation',
                   backtest_config={'cash': 200000, 'commission': 0.0005,
                                    'start_date': '2025-07-10', 'end_date': '2026-04-17'})
class ETFRotationStrategy(StrategyLogic):
    """ETF轮动策略 - 仅定义交易逻辑，通过统一接口访问数据和执行交易

    策略逻辑与执行环境完全解耦：
    - 数据访问通过数据适配器（BacktraderDataAdapter / LiveDataAdapter）
    - 交易执行通过执行器（BacktestExecutor / QMTExecutor）
    - 同一份代码在回测、模拟盘、实盘环境下运行

    事件回调：
    - on_bar():  K线到达时执行轮动逻辑
    - on_order(): 委托状态变化时的处理
    - on_trade(): 成交回报时的处理
    """

    params = (
        ('lookback_period', 21),
    )

    def __init__(self, executor=None, **kwargs):
        """初始化策略"""
        super().__init__(executor, **kwargs)

        self.etf_data = {}
        for name, symbol in ETF_CODES.items():
            self.etf_data[symbol] = {'name': name}
        self.current_holding = None

    def get_symbols(self):
        """获取策略需要的标的列表"""
        return list(ETF_CODES.values())

    def on_bar(self, bar: BarData):
        """K线数据到达时触发 - 执行轮动逻辑"""
        current_date = self.get_current_date()
        current_date_str = current_date.strftime('%Y-%m-%d') if current_date else 'N/A'

        # 时间过滤逻辑：
        # - 日线回测: bar.datetime 是 datetime 对象但时间为 00:00:00，默认执行
        # - 分钟线回测: bar.datetime 包含完整时间，只在14:50执行
        # - 实盘/模拟盘: bar.datetime 包含完整时间，只在14:50执行
        bar_datetime = getattr(bar, 'datetime', None)
        if bar_datetime and isinstance(bar_datetime, datetime.datetime):
            hour = bar_datetime.hour
            minute = bar_datetime.minute
            if hour != 0 or minute != 0:
                # 有具体时间信息（分钟线/实盘），只在14:50执行
                if hour != 14 or minute != 50:
                    return

        returns = self._calculate_returns()

        if not returns:
            return

        # 添加14:50时间信息
        current_datetime = f"{current_date_str} 14:50:00"
        sorted_returns = sorted(returns.items(), key=lambda x: x[1]['rate'], reverse=True)

        self.log(f'[{current_datetime}] 各ETF {self.params.lookback_period}日涨幅:')
        for symbol, ret_data in sorted_returns:
            ret = ret_data['rate']
            start_price = ret_data['start_price']
            end_price = ret_data['end_price']
            self.log(f'  {self.etf_data[symbol]["name"]}: {ret:.2%} (起始价: {start_price:.2f}, 结束价: {end_price:.2f})')

        top_symbol, top_ret_data = sorted_returns[0]
        top_return = top_ret_data['rate']

        if top_return > 0:
            self._handle_buy_signal(current_date_str, top_symbol, top_ret_data)
        else:
            self._handle_sell_signal(current_date_str, top_ret_data)

        self._log_portfolio_status(current_date_str)

    def on_order(self, order: OrderInfo):
        """委托状态变化时触发"""
        super().on_order(order)  # 调用父类方法存储订单
        if order.is_completed:
            if order.is_buy:
                self.log(f'买入成交: {order.symbol}, 价格: {order.executed_price:.3f}, 数量: {order.executed_volume}, 手续费: {order.commission:.2f}')
            else:
                self.log(f'卖出成交: {order.symbol}, 价格: {order.executed_price:.3f}, 数量: {order.executed_volume}, 手续费: {order.commission:.2f}')
        elif order.status in (order.STATUS_CANCELED, order.STATUS_REJECTED, order.STATUS_MARGIN):
            self.log(f'委托异常: {order.symbol}, 状态: {order.status}')

    def on_trade(self, trade: TradeInfo):
        """成交回报时触发"""
        self.log(f'成交回报: {trade.symbol}, 方向: {trade.direction}, 价格: {trade.price:.2f}, 数量: {trade.volume}, 盈亏: {trade.pnl:.2f}')

    def _calculate_returns(self):
        """计算各ETF的收益率"""
        returns = {}
        for symbol in self.etf_data:
            close_prices = self.get_close_prices(symbol)
            if len(close_prices) >= self.params.lookback_period:
                start_price = close_prices[-self.params.lookback_period]
                end_price = close_prices[-1]
                return_rate = (end_price - start_price) / start_price
                returns[symbol] = {
                    'rate': return_rate,
                    'start_price': start_price,
                    'end_price': end_price,
                }
        return returns

    def _handle_buy_signal(self, current_date, top_symbol, top_ret_data):
        """处理买入信号 - 涨幅最高的ETF为正时买入"""
        current_datetime = f"{current_date} 14:50:00"
        top_return = top_ret_data['rate']
        start_price = top_ret_data['start_price']
        end_price = top_ret_data['end_price']
        self.log(f'[{current_datetime}] 产生买入信号: {self.etf_data[top_symbol]["name"]} 涨幅最高: {top_return:.2%} (起始价: {start_price:.3f}, 结束价: {end_price:.3f})')

        if self.current_holding == top_symbol:
            return

        # 计算卖出持仓后的预期可用现金（保守估算，使用99%的安全边际）
        current_cash = self.get_cash()
        expected_cash = current_cash

        for name, symbol in ETF_CODES.items():
            if symbol != top_symbol:
                pos_size = self.get_position_size(symbol)
                if pos_size > 0:
                    sell_price = self.get_current_price(symbol)
                    if sell_price:
                        sell_value = sell_price * pos_size
                        commission = self.get_commission(self.etf_data[symbol]['name'], sell_value, pos_size)
                        net_proceeds = sell_value - commission
                        expected_cash += net_proceeds * 0.99

        # 先卖出所有非目标ETF的持仓
        for name, symbol in ETF_CODES.items():
            if symbol != top_symbol:
                pos_size = self.get_position_size(symbol)
                if pos_size > 0:
                    self._sell_position(symbol, current_date)

        # 使用保守估算的预期现金计算买入数量
        buy_price = self.get_current_price(top_symbol)

        if buy_price <= 0 or expected_cash <= 0:
            self.log(f'[{current_datetime}] 现金不足，无法买入 {self.etf_data[top_symbol]["name"]}')
            return

        buy_size = self._calculate_buy_size(top_symbol, buy_price, expected_cash)

        if buy_size > 0:
            buy_value = buy_price * buy_size
            commission = self.get_commission(self.etf_data[top_symbol]['name'], buy_value, buy_size)
            total_cost = buy_value + commission
            self.log(f'[{current_datetime}] 买入标的: {self.etf_data[top_symbol]["name"]} ({top_symbol}), 价格: {buy_price:.3f}, 数量: {buy_size}, 收益率: {top_return:.2%}')
            self.log(f'[{current_datetime}] 交易费用: 金额: {buy_value:.2f}, 手续费: {commission:.2f}, 总成本: {total_cost:.2f}')
            self.log(f'[{current_datetime}] 预期可用现金: {expected_cash:.2f}, 当前现金: {current_cash:.2f}')
            self.buy(top_symbol, buy_price, buy_size)
            self.current_holding = top_symbol
        else:
            self.log(f'[{current_datetime}] 现金不足，无法买入 {self.etf_data[top_symbol]["name"]}')

        self.log(f'[{current_datetime}] 持仓更新: 持有 {self.etf_data[top_symbol]["name"]}')

    def _handle_sell_signal(self, current_date, top_ret_data):
        """处理卖出信号 - 所有ETF涨幅为负时空仓"""
        # 添加14:50时间信息
        current_datetime = f"{current_date} 14:50:00"
        start_price = top_ret_data['start_price']
        end_price = top_ret_data['end_price']
        self.log(f'[{current_datetime}] 产生卖出信号: 所有ETF涨幅为负 (起始价: {start_price:.3f}, 结束价: {end_price:.3f})')

        if self.current_holding:
            self._sell_position(self.current_holding, current_date)
            self.current_holding = None
            self.log(f'[{current_datetime}] 持仓更新: 空仓')

    def _sell_position(self, symbol, current_date):
        """卖出指定标的的持仓"""
        # 添加14:50时间信息
        current_datetime = f"{current_date} 14:50:00"
        sell_price = self.get_current_price(symbol)
        pos_size = self.get_position_size(symbol)
        if pos_size > 0:
            sell_value = sell_price * pos_size
            commission = self.get_commission(self.etf_data[symbol]['name'], sell_value, pos_size)
            net_proceeds = sell_value - commission
            self.log(f'[{current_datetime}] 卖出标的: {self.etf_data[symbol]["name"]} ({symbol}), 价格: {sell_price:.3f}, 数量: {pos_size}')
            self.log(f'[{current_datetime}] 交易费用: 金额: {sell_value:.2f}, 手续费: {commission:.2f}, 净收入: {net_proceeds:.2f}')
            self.sell(symbol, sell_price, pos_size)

    def _calculate_buy_size(self, symbol, price, cash):
        """计算可买入数量（考虑手续费）"""
        if price <= 0 or cash <= 0:
            return 0
        estimated_buy_value = cash * 0.9995
        buy_size = int(estimated_buy_value / price)
        buy_value = price * buy_size
        commission = self.get_commission(self.etf_data[symbol]['name'], buy_value, buy_size)
        total_cost = buy_value + commission

        while total_cost > cash and buy_size > 0:
            buy_size -= 1
            buy_value = price * buy_size
            commission = self.get_commission(self.etf_data[symbol]['name'], buy_value, buy_size)
            total_cost = buy_value + commission

        return buy_size

    def _log_portfolio_status(self, current_date):
        """输出持仓和资产状况"""
        # 添加14:50时间信息
        current_datetime = f"{current_date} 14:50:00"
        cash = self.get_cash()
        total_position_value = 0
        position_symbol = "无"
        position_size = 0

        for name, symbol in ETF_CODES.items():
            pos_size = self.get_position_size(symbol)
            if pos_size > 0:
                pos_price = self.get_current_price(symbol)
                if pos_price:
                    pos_value = pos_size * pos_price
                    total_position_value += pos_value
                    self.log(f'[{current_datetime}] {symbol} 持仓: {pos_size}, 价格: {pos_price:.3f}, 市值: {pos_value:.2f}')
                    if symbol == self.current_holding:
                        position_symbol = name
                        position_size = pos_size

        total_value = cash + total_position_value
        self.log(f'[{current_datetime}] 资产状况: 现金: {cash:.2f}, 持仓市值: {total_position_value:.2f}, 总资产: {total_value:.2f}, 持仓: {position_symbol}, 份额: {position_size}')

    def get_commission(self, etf_name, value, size):
        """计算手续费（假设万0.5的手续费，最低5元）"""
        commission_rate = 0.0005
        min_commission = 5
        calculated_commission = value * commission_rate
        return max(calculated_commission, min_commission)

    def on_backtest_end(self):
        """回测结束时调用 - 处理未卖出的持仓"""
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log(f'[{current_datetime}] 回测结束，处理未卖出持仓...')
        
        # 检查所有ETF的持仓
        total_position_value = 0
        for name, symbol in ETF_CODES.items():
            pos_size = self.get_position_size(symbol)
            if pos_size > 0:
                pos_price = self.get_current_price(symbol)
                if pos_price:
                    pos_value = pos_size * pos_price
                    total_position_value += pos_value
                    self.log(f'[{current_datetime}] 未卖出持仓: {name} ({symbol}), 数量: {pos_size}, 价格: {pos_price:.3f}, 市值: {pos_value:.2f}')
        
        cash = self.get_cash()
        total_value = cash + total_position_value
        self.log(f'[{current_datetime}] 回测结束时总资产: {total_value:.2f} (现金: {cash:.2f}, 持仓市值: {total_position_value:.2f})')
        
        # 如果有未卖出的持仓，可以在这里执行卖出操作
        # 但为了保持回测的一致性，我们只记录而不执行实际卖出
        # 最终的回测结果会自动包含这些持仓的价值
