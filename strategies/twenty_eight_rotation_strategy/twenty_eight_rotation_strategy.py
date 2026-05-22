import datetime as dt_module
from typing import Dict, List, Optional
from core.stock_selection import StockSelectionStrategy
from strategies import register_strategy


@register_strategy('twenty_eight_rotation',
                   default_kwargs={'max_stocks': 3},
                   backtest_config={'cash': 1000000, 'commission': 0.0003,
                                    'start_date': '2020-04-28', 'end_date': '2026-04-28',
                                    'period': '1d', 'pool': '中证全指'})
class TwentyEightRotationStrategy(StockSelectionStrategy):
    """二八轮动小市值策略 - 基于沪深300和中证500指数20日涨幅择时，小市值选股

    来源: 聚宽社区 https://www.joinquant.com/view/community/detail/41a3aea42bc3b8cefccf257916d1395d

    策略逻辑：
    1. 每日检查沪深300和中证500指数的20日涨幅
    2. 二八止损（必需）：两个指数20日涨幅都为负，清仓避险
    3. 大盘止损（可选）：130日高低价2倍判断 + 三只黑鸦判断
    4. 个股止损（可选）：持仓后最高价回撤超阈值则平仓
    5. 选股：按市值升序排列，选取前N只小市值股票

    调仓规则：
    - 每N个交易日调仓（默认5日），等权重持仓
    - 择时信号为负时清仓
    """

    params = (
        ('rebalance_freq', 'weekly'),
        ('rebalance_interval', 5),
        ('max_stocks', 3),
        ('position_ratio', 0.95),
        ('stock_pool', None),
        ('momentum_period', 20),
        ('index_hs300', '510300.SH'),
        ('index_zz500', '510500.SH'),
        ('market_benchmark', '000300.SH'),
        ('filter_chinext', True),
        ('filter_limit_down', True),
        ('max_market_cap', None),
        ('enable_market_stoploss', True),
        ('market_price_period', 130),
        ('market_price_ratio', 2.0),
        ('enable_three_crows', True),
        ('three_crows_period', 4),
        ('enable_stock_stoploss', True),
        ('stock_stoploss_threshold', 0.15),
    )

    def __init__(self, executor=None, **kwargs):
        super().__init__(executor, **kwargs)
        self._market_timing_positive = True
        self._market_stopped = False
        self._skip_next_day = False
        self._stock_highest_prices: Dict[str, float] = {}

    def on_bar(self, bar):
        current_date = self.get_current_date()
        if current_date is None:
            return

        bar_datetime = getattr(bar, 'datetime', None)
        if bar_datetime and isinstance(bar_datetime, dt_module.datetime):
            hour = bar_datetime.hour
            minute = bar_datetime.minute
            if hour != 0 or minute != 0:
                trade_hour = getattr(self.params, 'trade_hour', 14)
                trade_minute = getattr(self.params, 'trade_minute', 50)
                if hour != trade_hour or minute != trade_minute:
                    return

        if self._rebalance_phase != self.PHASE_IDLE:
            return

        if self._skip_next_day:
            self.log(f'[大盘止损] 跳过交易日: {current_date}', level='info')
            self._skip_next_day = False
            self._market_stopped = False
            return

        market_positive = self._check_market_timing()

        if not market_positive:
            if self._current_holdings:
                self.log(f'[二八止损] 两指数{self.params.momentum_period}日涨幅均为负，清仓避险', level='info')
                self._sell_all()
                self._market_timing_positive = False
                self._last_rebalance_date = current_date
            return

        self._market_timing_positive = True

        if self.params.enable_market_stoploss:
            if self._check_market_price_stoploss():
                if self._current_holdings:
                    self.log(f'[大盘止损] 大盘{self.params.market_price_period}日最高价超过最低价{self.params.market_price_ratio}倍，清仓', level='info')
                    self._sell_all()
                    self._market_stopped = True
                    self._last_rebalance_date = current_date
                return

        if self.params.enable_three_crows:
            if self._check_three_black_crows():
                if self._current_holdings:
                    self.log(f'[大盘止损] 大盘三只黑鸦形态，清仓并停止交易', level='info')
                    self._sell_all()
                    self._market_stopped = True
                    self._skip_next_day = True
                    self._last_rebalance_date = current_date
                return

        if self.params.enable_stock_stoploss:
            self._check_stock_stoploss()

        self._update_stock_highest_prices()

        if self.is_rebalance_day(current_date):
            self.log(f'[选股] 调仓日: {current_date}', level='info')
            self._execute_rebalance(current_date)

    def is_rebalance_day(self, current_date: dt_module.date) -> bool:
        interval = getattr(self.params, 'rebalance_interval', 5)
        if interval and interval > 0:
            if self._last_rebalance_date is None:
                return True
            delta = (current_date - self._last_rebalance_date).days
            return delta >= interval
        return super().is_rebalance_day(current_date)

    def select_stocks(self) -> List[str]:
        if not self._market_timing_positive:
            return []

        pool = self.get_stock_pool()
        self.log(f'股票池: {len(pool)} 只')

        filtered = self._filter_tradeable(pool)
        if not filtered:
            self.log('过滤后无股票')
            return []

        market_caps = self._calc_market_caps(filtered)
        if not market_caps:
            self.log('无法计算市值')
            return []

        if self.params.max_market_cap is not None:
            cap_limit = self.params.max_market_cap * 1e8
            before = len(market_caps)
            market_caps = {s: v for s, v in market_caps.items() if v <= cap_limit}
            self.log(f'市值上限过滤: {before} -> {len(market_caps)} 只 (上限{self.params.max_market_cap}亿)')

        if not market_caps:
            self.log('过滤后无股票')
            return []

        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1])
        max_stocks = self.params.max_stocks
        selected = [stock for stock, _ in sorted_stocks[:max_stocks]]

        self.log(f'按市值排序选股: {len(pool)} -> {len(filtered)} -> {len(selected)} 只')
        for stock, mc in sorted_stocks[:max_stocks]:
            self.log(f'  {stock} | 市值: {mc / 1e8:.2f}亿')

        return selected

    def _check_market_timing(self) -> bool:
        hs300_symbol = self.params.index_hs300
        zz500_symbol = self.params.index_zz500
        period = self.params.momentum_period

        hs300_ret = self.get_return_over_days(hs300_symbol, period)
        zz500_ret = self.get_return_over_days(zz500_symbol, period)

        hs300_rate = hs300_ret.get('rate', 0) if hs300_ret else None
        zz500_rate = zz500_ret.get('rate', 0) if zz500_ret else None

        if hs300_rate is not None:
            self.log(f'沪深300 {period}日涨幅: {hs300_rate:.2%}')
        else:
            self.log(f'沪深300数据不可用')

        if zz500_rate is not None:
            self.log(f'中证500 {period}日涨幅: {zz500_rate:.2%}')
        else:
            self.log(f'中证500数据不可用')

        if hs300_rate is None and zz500_rate is None:
            return True

        if hs300_rate is not None and zz500_rate is not None:
            return hs300_rate > 0 or zz500_rate > 0

        if hs300_rate is not None:
            return hs300_rate > 0

        return zz500_rate > 0

    def _check_market_price_stoploss(self) -> bool:
        benchmark = self.params.market_benchmark
        period = self.params.market_price_period
        ratio_threshold = self.params.market_price_ratio

        ohlcv = self.get_ohlcv_data(benchmark, period)
        if not ohlcv or len(ohlcv) < period:
            return False

        recent = ohlcv[-period:]
        highest = max(d['high'] for d in recent)
        lowest = min(d['low'] for d in recent)

        if lowest > 0 and highest / lowest >= ratio_threshold:
            self.log(f'[大盘价格止损] {period}日最高={highest:.2f}, 最低={lowest:.2f}, '
                     f'比值={highest / lowest:.2f} >= {ratio_threshold}', level='info')
            return True

        return False

    def _check_three_black_crows(self) -> bool:
        benchmark = self.params.market_benchmark
        period = self.params.three_crows_period + 1

        ohlcv = self.get_ohlcv_data(benchmark, period)
        if not ohlcv or len(ohlcv) < period:
            return False

        recent = ohlcv[-period:]
        crows_count = 0
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            curr = recent[i]
            open_price = curr.get('open', 0)
            close_price = curr.get('close', 0)
            prev_close = prev.get('close', 0)
            prev_open = prev.get('open', 0)

            if open_price <= 0 or close_price <= 0:
                crows_count = 0
                continue

            is_black = close_price < open_price
            gaps_down = open_price < prev_close
            closes_lower = close_price < prev_close

            body = abs(close_price - open_price)
            prev_body = abs(prev_close - prev_open)
            body_not_tiny = body > prev_body * 0.3 if prev_body > 0 else body > 0

            if is_black and closes_lower and body_not_tiny:
                crows_count += 1
            else:
                crows_count = 0

        if crows_count >= 3:
            self.log(f'[三只黑鸦] 大盘连续{crows_count}日黑鸦形态', level='info')
            return True

        return False

    def _check_stock_stoploss(self):
        threshold = self.params.stock_stoploss_threshold
        stocks_to_sell = []

        for symbol in list(self._current_holdings.keys()):
            pos_size = self.get_position_size(symbol)
            if pos_size <= 0:
                continue

            highest = self._stock_highest_prices.get(symbol)
            if highest is None or highest <= 0:
                continue

            current_price = self.get_current_price(symbol)
            if current_price is None or current_price <= 0:
                continue

            drawdown = (highest - current_price) / highest
            if drawdown >= threshold:
                self.log(f'[个股止损] {symbol} 从最高价{highest:.2f}回撤{drawdown:.2%} >= {threshold:.0%}，'
                         f'当前价{current_price:.2f}', level='info')
                stocks_to_sell.append(symbol)

        for symbol in stocks_to_sell:
            sellable = self.get_sellable_volume(symbol)
            if sellable > 0:
                price = self.get_current_price(symbol)
                if price and price > 0 and not self.is_suspended(symbol) and not self.is_limit_down(symbol):
                    self.sell(symbol, price, sellable)
                    if symbol in self._current_holdings:
                        remaining = self._current_holdings[symbol] - sellable
                        if remaining <= 0:
                            del self._current_holdings[symbol]
                            if symbol in self._stock_highest_prices:
                                del self._stock_highest_prices[symbol]
                        else:
                            self._current_holdings[symbol] = remaining

    def _update_stock_highest_prices(self):
        for symbol in list(self._current_holdings.keys()):
            pos_size = self.get_position_size(symbol)
            if pos_size <= 0:
                if symbol in self._stock_highest_prices:
                    del self._stock_highest_prices[symbol]
                continue

            current_price = self.get_current_price(symbol)
            if current_price and current_price > 0:
                prev_highest = self._stock_highest_prices.get(symbol, 0)
                self._stock_highest_prices[symbol] = max(prev_highest, current_price)

    def _filter_tradeable(self, pool: List[str]) -> List[str]:
        result = []
        for stock in pool:
            if self.is_suspended(stock):
                continue
            if self.is_limit_up(stock):
                continue
            if self.params.filter_limit_down and self.is_limit_down(stock):
                continue
            if self.params.filter_chinext and stock.startswith('300'):
                continue
            result.append(stock)
        self.log(f'过滤(停牌/涨停/跌停/创业板): {len(pool)} -> {len(result)} 只')
        return result

    def _calc_market_caps(self, stocks: List[str]) -> Dict[str, float]:
        result = {}
        no_price = 0
        no_cap_data = 0

        for stock in stocks:
            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                price = self.get_current_price(stock)
            if price is None or price <= 0:
                no_price += 1
                continue

            total_equity = self.get_financial_field(stock, 'Balance', 'total_equity')
            bps = self.get_financial_field(stock, 'Pershareindex', 's_fa_bps')

            if total_equity and total_equity > 0 and bps and bps > 0:
                total_shares = total_equity / bps
                market_cap = total_shares * price
                result[stock] = market_cap
            elif total_equity and total_equity > 0:
                result[stock] = total_equity
            else:
                no_cap_data += 1

        if no_price > 0 or no_cap_data > 0:
            self.log(f'市值计算: {len(stocks)} 只 -> {len(result)} 只有有效数据 '
                     f'(无价格={no_price}, 无股本数据={no_cap_data})')

        if not result and no_cap_data > 0:
            self.log(f'财务数据不可用，回退为价格排序选股', level='info')
            return self._rank_by_price_fallback(stocks)

        return result

    def _rank_by_price_fallback(self, stocks: List[str]) -> Dict[str, float]:
        result = {}
        for stock in stocks:
            price = self.get_unadjusted_price(stock)
            if price is None or price <= 0:
                price = self.get_current_price(stock)
            if price and price > 0:
                result[stock] = price
        return result
