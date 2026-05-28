import datetime
from core.strategy_logic import StrategyLogic, BarData, OrderInfo, TradeInfo
from strategies import register_strategy
from .config import BANK_STOCKS


@register_strategy('bank_rotation',
                   backtest_config={'cash': 200000, 'commission': 0.0002,
                                    'slippage': 0.0007,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d',
                                    'compare_symbols': ['601398.SH', '601288.SH', '601939.SH', '601988.SH']})
class BankRotationStrategy(StrategyLogic):
    """银行轮动策略 - 四大银行（工、农、建、中）动量轮动

    复现聚宽囚徒的银行轮动策略：
    1. 计算四只银行股的涨跌幅比率（当前价格 / 上一交易日收盘价）
    2. 若空仓：当最大比率与最小比率的差值超过阈值时，买入比率最小的银行股
    3. 若持仓：当持仓股比率与最小比率的差值超过阈值时，换仓至比率最小的银行股
    4. 核心思想：买入当日相对跌幅最大的银行股，利用银行股之间的均值回归效应

    支持日线和分钟线两种模式：
    - 日线模式(period='1d')：比率 = 当日收盘价 / 上一交易日收盘价，每日检查一次
    - 分钟线模式(period='1m')：比率 = 当前分钟收盘价 / 上一交易日收盘价，每分钟检查一次

    原策略使用分钟线数据，每分钟检查比率差，一旦超过阈值立即交易。
    原策略年化收益约77%（2015-2016年回测），无止损机制。
    """

    params = (
        ('spread_threshold', 0.004),  # 比率差阈值，低于此值不交易
        ('switch_threshold', 0.004),  # 换仓阈值：基于单边价差优化，综合评分最优(样本内1.05+样本外0.69)
        ('max_volatility', None),  # 波动率过滤：标的日波动率超过此值时暂停交易（None=禁用）
        ('min_holding_bars', None),  # 最小持仓K线数：换仓后至少持有N根K线再允许下次换仓（None=禁用）
        ('no_trade_start', None),  # 开盘不交易时段起始时间，如'09:30'（None=禁用）
        ('no_trade_end', None),  # 开盘不交易时段结束时间，如'09:45'（None=禁用）
        ('no_trade_close_start', None),  # 收盘不交易时段起始时间，如'14:55'（None=禁用）
        ('ma_period', None),  # 趋势过滤MA周期：价格低于MA时空仓（None=禁用）
        ('adaptive_threshold', None),  # 自适应阈值系数（None=禁用）
        ('confirm_bars', None),  # 比率确认K线数：比率差需持续N根K线才触发交易（None=禁用）
        ('max_daily_trades', None),  # 每日最大换仓次数（None=禁用）
        ('daily_drawdown_limit', None),  # 当日最大回撤限制：超过此比例空仓（None=禁用）
        ('extra_stocks', None),  # 扩展标的字典，如{'交通银行': '601328.SH'}（None=禁用）
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self.bank_names = {}
        for name, symbol in BANK_STOCKS.items():
            self.bank_names[symbol] = name
        # 扩展标的
        if self.params.extra_stocks:
            for name, symbol in self.params.extra_stocks.items():
                self.bank_names[symbol] = name
        self.current_holding = None
        self._last_trade_date = None
        # 优化参数的内部状态
        self._last_buy_bar_count = 0
        self._bar_count = 0
        self._daily_trade_count = 0
        self._last_trade_date_str = ''
        self._daily_start_value = None
        self._confirm_start_bar = None
        self._last_confirm_signal = None
        self._spread_history = []

    def get_symbols(self):
        symbols = list(BANK_STOCKS.values())
        if self.params.extra_stocks:
            symbols.extend(list(self.params.extra_stocks.values()))
        return symbols

    def _get_all_symbols(self):
        """获取所有标的列表（包括扩展标的）"""
        symbols = list(BANK_STOCKS.values())
        if self.params.extra_stocks:
            symbols.extend(list(self.params.extra_stocks.values()))
        return symbols

    def on_bar(self, bar: BarData):
        self._bar_count += 1

        # 重置每日计数器
        current_date = self.get_current_date()
        date_str = current_date.strftime('%Y-%m-%d') if current_date else 'N/A'
        if date_str != self._last_trade_date_str:
            self._daily_trade_count = 0
            self._last_trade_date_str = date_str
            self._daily_start_value = None

        # 当日起始总值
        if self._daily_start_value is None:
            self._daily_start_value = self._get_total_value()

        # 分钟线模式下，添加时间信息
        current_dt = self.get_current_datetime()
        if current_dt and hasattr(current_dt, 'strftime'):
            time_str = current_dt.strftime('%H:%M')
        else:
            time_str = ''

        # 优化：开盘/收盘不交易时段过滤
        if self.params.no_trade_start and self.params.no_trade_end:
            if time_str and self.params.no_trade_start <= time_str < self.params.no_trade_end:
                return
        if self.params.no_trade_close_start:
            if time_str and time_str >= self.params.no_trade_close_start:
                return

        # 优化：当日回撤限制
        if self.params.daily_drawdown_limit and self._daily_start_value and self._daily_start_value > 0:
            current_value = self._get_total_value()
            if current_value < self._daily_start_value * (1 - self.params.daily_drawdown_limit):
                holding_symbol = self._find_holding()
                if holding_symbol:
                    sellable = self.get_sellable_volume(holding_symbol)
                    if sellable > 0:
                        self.log(f'[{date_str} {time_str}] 当日回撤超限: {(1 - current_value/self._daily_start_value)*100:.2f}% > {self.params.daily_drawdown_limit*100:.1f}%, 清仓')
                        self._sell_position(holding_symbol)
                return

        # 计算各银行股的涨跌幅比率 (当前价格 / 昨日收盘)
        all_symbols = self._get_all_symbols()
        ratios = self._calculate_ratios(all_symbols)
        if not ratios:
            return

        # 优化：波动率过滤 - 移除波动率过高的标的
        if self.params.max_volatility:
            filtered = {}
            for symbol, ratio in ratios.items():
                daily_closes = self.get_close_prices_for_days(symbol, 20)
                if len(daily_closes) >= 5:
                    returns = [(daily_closes[i] - daily_closes[i-1]) / daily_closes[i-1]
                               for i in range(1, len(daily_closes)) if daily_closes[i-1] > 0]
                    if returns:
                        import statistics
                        vol = statistics.stdev(returns) if len(returns) >= 2 else 0
                        if vol <= self.params.max_volatility:
                            filtered[symbol] = ratio
                else:
                    filtered[symbol] = ratio
            ratios = filtered
            if not ratios:
                return

        # 优化：趋势过滤 - 价格低于MA的标的不参与
        if self.params.ma_period:
            filtered = {}
            for symbol, ratio in ratios.items():
                daily_closes = self.get_close_prices_for_days(symbol, self.params.ma_period)
                if len(daily_closes) >= self.params.ma_period:
                    ma = sum(daily_closes[-self.params.ma_period:]) / self.params.ma_period
                    curr_price = self.get_current_price(symbol)
                    if curr_price and curr_price >= ma:
                        filtered[symbol] = ratio
                else:
                    filtered[symbol] = ratio
            ratios = filtered
            if not ratios:
                return

        max_ratio = max(ratios.values())
        min_ratio = min(ratios.values())
        min_symbol = min(ratios, key=ratios.get)
        spread = max_ratio - min_ratio

        # 优化：自适应阈值
        effective_threshold = self.params.spread_threshold
        if self.params.adaptive_threshold is not None:
            self._spread_history.append(spread)
            if len(self._spread_history) > 240:
                self._spread_history = self._spread_history[-240:]
            if len(self._spread_history) >= 20:
                import statistics
                spread_std = statistics.stdev(self._spread_history)
                effective_threshold = self.params.spread_threshold * (1 + self.params.adaptive_threshold * spread_std / max(self.params.spread_threshold, 1e-9))

        # 优化：比率确认 - 比率差需持续N根K线
        confirm_signal = (min_symbol, spread > effective_threshold)
        if self.params.confirm_bars is not None:
            if confirm_signal != self._last_confirm_signal:
                self._confirm_start_bar = self._bar_count
                self._last_confirm_signal = confirm_signal
            if self._confirm_start_bar and (self._bar_count - self._confirm_start_bar) < self.params.confirm_bars:
                return

        # 查找当前持仓
        holding_symbol = self._find_holding()

        if holding_symbol is None:
            # 空仓：比率差超过阈值时买入比率最小的
            if spread > effective_threshold:
                min_name = self.bank_names.get(min_symbol, min_symbol)
                sorted_ratios = sorted(ratios.items(), key=lambda x: x[1])
                ratio_detail = ', '.join(f'{self.bank_names.get(s, s)}:{r:.4f}' for s, r in sorted_ratios)
                self.log(f'[{date_str} {time_str}] 空仓买入: 比率差{spread:.4f} > 阈值{effective_threshold:.4f}, '
                         f'买入{min_name}({min_symbol}), 比率: {min_ratio:.4f} [{ratio_detail}]')
                self._buy_full_position(min_symbol)
                self._last_buy_bar_count = self._bar_count
        else:
            # 优化：最小持仓K线数
            if self.params.min_holding_bars is not None:
                if (self._bar_count - self._last_buy_bar_count) < self.params.min_holding_bars:
                    return

            # 优化：每日最大换仓次数
            if self.params.max_daily_trades is not None:
                if self._daily_trade_count >= self.params.max_daily_trades:
                    return

            # 持仓：持仓股比率与最小比率差超过阈值时换仓
            holding_ratio = ratios.get(holding_symbol, 0)
            holding_name = self.bank_names.get(holding_symbol, holding_symbol)
            min_name = self.bank_names.get(min_symbol, min_symbol)

            # 优化：换仓阈值
            switch_diff = holding_ratio - min_ratio
            effective_switch_threshold = effective_threshold
            if self.params.switch_threshold is not None:
                effective_switch_threshold = self.params.switch_threshold

            if holding_symbol != min_symbol and switch_diff > effective_switch_threshold:
                sellable = self.get_sellable_volume(holding_symbol)
                if sellable <= 0:
                    return

                sorted_ratios = sorted(ratios.items(), key=lambda x: x[1])
                ratio_detail = ', '.join(f'{self.bank_names.get(s, s)}:{r:.4f}' for s, r in sorted_ratios)
                self.log(f'[{date_str} {time_str}] 换仓: {holding_name}比率{holding_ratio:.4f} - '
                         f'{min_name}比率{min_ratio:.4f} = {switch_diff:.4f} > 阈值{effective_switch_threshold:.4f} [{ratio_detail}]')
                total_value = self._get_total_value()
                self._sell_position(holding_symbol)
                self._buy_with_value(min_symbol, total_value)
                self._last_buy_bar_count = self._bar_count
                self._daily_trade_count += 1

    def on_order(self, order: OrderInfo):
        super().on_order(order)
        if order.is_completed:
            name = self.bank_names.get(order.symbol, order.symbol)
            if order.is_buy:
                self.log(f'买入成交: {name}({order.symbol}), 价格: {order.executed_price:.3f}, '
                         f'数量: {order.executed_volume}, 手续费: {order.commission:.2f}')
            else:
                self.log(f'卖出成交: {name}({order.symbol}), 价格: {order.executed_price:.3f}, '
                         f'数量: {order.executed_volume}, 手续费: {order.commission:.2f}')

    def on_trade(self, trade: TradeInfo):
        super().on_trade(trade)

    def _calculate_ratios(self, symbols=None):
        """计算各银行股 当前价格/上一交易日收盘价 的比率"""
        if symbols is None:
            symbols = self._get_all_symbols()
        ratios = {}
        for symbol in symbols:
            daily_closes = self.get_close_prices(symbol)
            if not daily_closes:
                continue

            if len(daily_closes) >= 2:
                prev_close = daily_closes[-2]
            else:
                continue

            if not prev_close or prev_close <= 0:
                continue

            curr_price = self.get_current_price(symbol)
            if curr_price and curr_price > 0:
                ratios[symbol] = curr_price / prev_close
        return ratios

    def _find_holding(self):
        """查找当前持仓的银行股"""
        for symbol in self._get_all_symbols():
            pos_size = self.get_position_size(symbol)
            if pos_size > 0:
                return symbol
        return None

    def _get_total_value(self):
        """计算组合总值（现金 + 持仓市值）"""
        cash = self.get_cash()
        position_value = 0
        for symbol in self._get_all_symbols():
            pos_size = self.get_position_size(symbol)
            if pos_size > 0:
                price = self.get_current_price(symbol)
                if price and price > 0:
                    position_value += pos_size * price
        return cash + position_value

    def _buy_full_position(self, symbol):
        """全仓买入指定银行股（空仓时调用，使用当前现金）"""
        total_value = self._get_total_value()
        self._buy_with_value(symbol, total_value)

    def _buy_with_value(self, symbol, total_value):
        """按指定总值全仓买入指定银行股"""
        price = self.get_current_price(symbol)
        if price is None or price <= 0:
            self.log(f'买入失败: {symbol} 无法获取价格', level='warning')
            return

        max_shares = int(total_value * 0.999 / price)
        buy_volume = (max_shares // 100) * 100

        if buy_volume > 0:
            self.buy(symbol, price, buy_volume)
            self.current_holding = symbol
        else:
            self.log(f'买入失败: 资金不足买入1手 {symbol}, 价格: {price:.2f}, 可用: {total_value:.2f}', level='warning')

    def _sell_position(self, symbol):
        """卖出指定银行股的全部持仓"""
        pos_size = self.get_sellable_volume(symbol)
        if pos_size <= 0:
            reason = 'T+1当天买入不可卖出' if self.is_t_plus_1(symbol) else '无持仓'
            self.log(f'卖出跳过: {symbol} {reason}', level='warning')
            return

        price = self.get_current_price(symbol)
        if price is None or price <= 0:
            self.log(f'卖出失败: {symbol} 无法获取价格', level='warning')
            return

        self.sell(symbol, price, pos_size)
        if self.current_holding == symbol:
            self.current_holding = None
